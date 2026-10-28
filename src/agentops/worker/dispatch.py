"""Durable outbox dispatcher with a deterministic publisher seam."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Protocol
from uuid import UUID, uuid4

from agentops.persistence import Database, OutboxClaim, OutboxClaimError, OutboxRepository


class PublisherError(RuntimeError):
    """A broker rejected a message before it was acknowledged."""


class DispatchMessageError(ValueError):
    """A broker payload is missing required command identity fields."""


@dataclass(frozen=True)
class DispatchMessage:
    """Small broker payload; business truth remains in the outbox row."""

    event_id: UUID
    event_key: str
    event_type: str
    aggregate_type: str
    aggregate_id: str
    payload: dict[str, Any]
    attempt: int

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> DispatchMessage:
        if not isinstance(payload, Mapping):
            raise DispatchMessageError("command payload must be an object")
        required = (
            "event_id",
            "event_key",
            "event_type",
            "aggregate_type",
            "aggregate_id",
            "payload",
            "attempt",
        )
        missing = [name for name in required if name not in payload]
        if missing:
            raise DispatchMessageError("command payload missing: " + ", ".join(missing))
        try:
            event_id = UUID(str(payload["event_id"]))
        except (TypeError, ValueError) as exc:
            raise DispatchMessageError("command payload event_id is invalid") from exc
        event_key = payload["event_key"]
        event_type = payload["event_type"]
        aggregate_type = payload["aggregate_type"]
        aggregate_id = payload["aggregate_id"]
        body = payload["payload"]
        attempt = payload["attempt"]
        if not all(
            isinstance(value, str) and value
            for value in (event_key, event_type, aggregate_type, aggregate_id)
        ):
            raise DispatchMessageError("command payload identity fields are invalid")
        if (
            not isinstance(body, dict)
            or not isinstance(attempt, int)
            or isinstance(attempt, bool)
            or attempt < 1
        ):
            raise DispatchMessageError("command payload body or attempt is invalid")
        return cls(event_id, event_key, event_type, aggregate_type, aggregate_id, body, attempt)

    @classmethod
    def from_claim(cls, claim: OutboxClaim) -> DispatchMessage:
        return cls(
            event_id=claim.event_id,
            event_key=claim.event_key,
            event_type=claim.event_type,
            aggregate_type=claim.aggregate_type,
            aggregate_id=claim.aggregate_id,
            payload=claim.payload,
            attempt=claim.dispatch_attempts,
        )

    def as_payload(self) -> dict[str, Any]:
        return {
            "event_id": str(self.event_id),
            "event_key": self.event_key,
            "event_type": self.event_type,
            "aggregate_type": self.aggregate_type,
            "aggregate_id": self.aggregate_id,
            "payload": self.payload,
            "attempt": self.attempt,
        }


class Publisher(Protocol):
    def publish(self, message: DispatchMessage) -> None:
        """Publish one message; raise PublisherError on a failed acknowledgement."""


class FakePublisher:
    """Deterministic publisher used by offline tests and local demos."""

    def __init__(self, *, fail_event_keys: Iterable[str] = ()) -> None:
        self.fail_event_keys = set(fail_event_keys)
        self.calls: list[str] = []
        self.messages: list[DispatchMessage] = []
        self._delivered_keys: set[str] = set()

    def publish(self, message: DispatchMessage) -> None:
        self.calls.append(message.event_key)
        if message.event_key in self.fail_event_keys:
            raise PublisherError(f"publisher rejected {message.event_key}")
        if message.event_key in self._delivered_keys:
            return
        self._delivered_keys.add(message.event_key)
        self.messages.append(message)


class CeleryPublisher:
    """Publish the canonical command payload through Celery's broker protocol."""

    def __init__(self, celery_app: Any, *, task_name: str, queue_name: str) -> None:
        if not task_name.strip():
            raise ValueError("task_name must not be empty")
        if not queue_name.strip():
            raise ValueError("queue_name must not be empty")
        self.celery_app = celery_app
        self.task_name = task_name
        self.queue_name = queue_name

    def publish(self, message: DispatchMessage) -> None:
        try:
            self.celery_app.send_task(
                self.task_name,
                args=[message.as_payload()],
                queue=self.queue_name,
                task_id=str(message.event_id),
                retry=False,
                serializer="json",
            )
        except Exception as exc:
            raise PublisherError("celery publish failed") from exc


class RedisPublisher:
    """Optional Redis list publisher; the dependency is loaded only when used."""

    def __init__(self, redis_url: str, *, queue_name: str = "agentops.commands") -> None:
        try:
            import redis
        except ImportError as exc:  # pragma: no cover - exercised when optional extra is absent
            raise RuntimeError("redis package is required for RedisPublisher") from exc
        self.client = redis.Redis.from_url(redis_url)
        self.queue_name = queue_name

    def publish(self, message: DispatchMessage) -> None:
        try:
            self.client.rpush(self.queue_name, json.dumps(message.as_payload(), sort_keys=True))
        except Exception as exc:
            raise PublisherError("redis publish failed") from exc


@dataclass(frozen=True)
class DispatchReport:
    claimed: int
    published: int
    failed: int
    exhausted: int
    event_ids: tuple[UUID, ...]


class OutboxDispatcher:
    """Publish due outbox rows without holding a DB transaction over broker I/O."""

    def __init__(
        self,
        database: Database,
        publisher: Publisher,
        *,
        owner_id: str | None = None,
        lease_seconds: float = 30.0,
        max_attempts: int = 5,
        recovery_delay_seconds: float = 60.0,
        base_retry_delay_seconds: float = 1.0,
        max_retry_delay_seconds: float = 60.0,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        if max_attempts < 1:
            raise ValueError("max_attempts must be positive")
        if recovery_delay_seconds <= 0:
            raise ValueError("recovery_delay_seconds must be positive")
        if base_retry_delay_seconds < 0 or max_retry_delay_seconds < 0:
            raise ValueError("retry delays cannot be negative")
        self.database = database
        self.publisher = publisher
        self.owner_id = owner_id or f"dispatcher-{uuid4()}"
        self.lease_seconds = lease_seconds
        self.max_attempts = max_attempts
        self.recovery_delay_seconds = recovery_delay_seconds
        self.base_retry_delay_seconds = base_retry_delay_seconds
        self.max_retry_delay_seconds = max_retry_delay_seconds
        self.clock = clock or (lambda: datetime.now(UTC))

    def recover_unprocessed_once(self, *, limit: int = 100) -> DispatchReport:
        """Republish published commands that have no durable processing acknowledgement."""
        now = self.clock()
        with self.database.session() as session:
            claims = OutboxRepository(session).claim_unprocessed(
                owner_id=self.owner_id,
                now=now,
                lease_seconds=self.lease_seconds,
                limit=limit,
                max_attempts=self.max_attempts,
            )
        published = failed = exhausted = 0
        event_ids: list[UUID] = []
        for claim in claims:
            message = DispatchMessage.from_claim(claim)
            event_ids.append(message.event_id)
            try:
                self.publisher.publish(message)
            except Exception as exc:
                delay = min(
                    self.base_retry_delay_seconds * (2 ** max(0, claim.dispatch_attempts - 1)),
                    self.max_retry_delay_seconds,
                )
                try:
                    with self.database.session() as session:
                        row = OutboxRepository(session).mark_republish_failed(
                            message.event_id,
                            owner_id=self.owner_id,
                            dispatch_attempt=message.attempt,
                            now=self.clock(),
                            error_code="RepublishError",
                            error_message=str(exc),
                            retry_delay_seconds=delay,
                            max_attempts=self.max_attempts,
                        )
                except OutboxClaimError:
                    failed += 1
                    continue
                if row.last_error_code == "RetryExhausted":
                    exhausted += 1
                else:
                    failed += 1
                continue
            with self.database.session() as session:
                try:
                    OutboxRepository(session).mark_published(
                        message.event_id,
                        owner_id=self.owner_id,
                        dispatch_attempt=message.attempt,
                        now=self.clock(),
                        recovery_delay_seconds=self.recovery_delay_seconds,
                    )
                except OutboxClaimError:
                    failed += 1
                    continue
            published += 1
        return DispatchReport(
            claimed=len(claims),
            published=published,
            failed=failed,
            exhausted=exhausted,
            event_ids=tuple(event_ids),
        )

    def dispatch_once(self, *, limit: int = 100) -> DispatchReport:
        now = self.clock()
        with self.database.session() as session:
            claims = OutboxRepository(session).claim_due(
                owner_id=self.owner_id,
                now=now,
                lease_seconds=self.lease_seconds,
                limit=limit,
                max_attempts=self.max_attempts,
            )
        published = failed = exhausted = 0
        event_ids: list[UUID] = []
        for claim in claims:
            message = DispatchMessage.from_claim(claim)
            event_ids.append(message.event_id)
            try:
                self.publisher.publish(message)
            except Exception as exc:
                delay = min(
                    self.base_retry_delay_seconds * (2 ** max(0, claim.dispatch_attempts - 1)),
                    self.max_retry_delay_seconds,
                )
                try:
                    with self.database.session() as session:
                        row = OutboxRepository(session).mark_failed(
                            message.event_id,
                            owner_id=self.owner_id,
                            dispatch_attempt=message.attempt,
                            now=self.clock(),
                            error_code="PublisherError",
                            error_message=str(exc),
                            retry_delay_seconds=delay,
                            max_attempts=self.max_attempts,
                        )
                except OutboxClaimError:
                    failed += 1
                    continue
                if row.last_error_code == "RetryExhausted":
                    exhausted += 1
                else:
                    failed += 1
                continue
            with self.database.session() as session:
                try:
                    OutboxRepository(session).mark_published(
                        message.event_id,
                        owner_id=self.owner_id,
                        dispatch_attempt=message.attempt,
                        now=self.clock(),
                        recovery_delay_seconds=self.recovery_delay_seconds,
                    )
                except OutboxClaimError:
                    # A lost DB claim is recoverable; leave the event unpublished.
                    failed += 1
                    continue
            published += 1
        return DispatchReport(
            claimed=len(claims),
            published=published,
            failed=failed,
            exhausted=exhausted,
            event_ids=tuple(event_ids),
        )
