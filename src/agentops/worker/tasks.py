"""Durable command worker contract and Celery task registration."""

from __future__ import annotations

import json
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from agentops.persistence import (
    AgentRunRow,
    CommandClaimError,
    CommandClaimState,
    CommandOutcomeRepository,
    Database,
    EntityNotFound,
    OutboxRepository,
)
from agentops.worker.dispatch import DispatchMessage, DispatchMessageError
from agentops.worker.transport import DEFAULT_COMMAND_QUEUE_NAME, DEFAULT_COMMAND_TASK_NAME

_NON_COMMAND_EVENT_TYPES = frozenset({"incident.created", "incident.reopened"})


@dataclass(frozen=True)
class WorkerResult:
    event_key: str
    processed: bool
    reason: str
    attempt: int = 0
    error_code: str | None = None


@dataclass
class CommandDeduplicator:
    """Process each logical event key once per worker instance."""

    _processed: set[str] = field(default_factory=set)

    def seen(self, event_key: str) -> bool:
        return event_key in self._processed

    def mark(self, event_key: str) -> None:
        self._processed.add(event_key)


class CommandWorker:
    """Process one complete command with durable claim and acknowledgement."""

    def __init__(
        self,
        executor: Callable[[DispatchMessage], Any],
        *,
        deduplicator: CommandDeduplicator | None = None,
        database: Database | None = None,
        owner_id: str | None = None,
        lease_seconds: float = 60.0,
        max_attempts: int = 5,
        base_retry_delay_seconds: float = 1.0,
        max_retry_delay_seconds: float = 60.0,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        if lease_seconds <= 0:
            raise ValueError("lease_seconds must be positive")
        if max_attempts < 1:
            raise ValueError("max_attempts must be positive")
        if base_retry_delay_seconds < 0 or max_retry_delay_seconds < 0:
            raise ValueError("retry delays cannot be negative")
        self.executor = executor
        self.deduplicator = deduplicator or CommandDeduplicator()
        self.database = database
        self.owner_id = owner_id or f"worker-{uuid4()}"
        self.lease_seconds = lease_seconds
        self.max_attempts = max_attempts
        self.base_retry_delay_seconds = base_retry_delay_seconds
        self.max_retry_delay_seconds = max_retry_delay_seconds
        self.clock = clock or (lambda: datetime.now(UTC))

    def handle(self, message: DispatchMessage) -> WorkerResult:
        if self.database is None:
            return self._handle_in_memory(message)
        try:
            with self.database.session() as session:
                claim_result = OutboxRepository(session).claim_command(
                    message.event_id,
                    event_key=message.event_key,
                    owner_id=self.owner_id,
                    now=self.clock(),
                    lease_seconds=self.lease_seconds,
                    max_attempts=self.max_attempts,
                )
        except EntityNotFound as exc:
            raise DispatchMessageError("command is missing from durable outbox") from exc
        if claim_result.state is not CommandClaimState.CLAIMED:
            reason = {
                CommandClaimState.PROCESSED: "duplicate",
                CommandClaimState.IN_PROGRESS: "in_progress",
                CommandClaimState.RETRY_WAIT: "retry_wait",
                CommandClaimState.EXHAUSTED: "processing_retry_exhausted",
            }[claim_result.state]
            return WorkerResult(
                message.event_key,
                processed=False,
                reason=reason,
                attempt=claim_result.processing_attempts,
                error_code=(
                    "ProcessingRetryExhausted"
                    if claim_result.state is CommandClaimState.EXHAUSTED
                    else None
                ),
            )

        attempt = claim_result.processing_attempts
        if claim_result.claim is None:
            raise RuntimeError("claimed command has no durable payload")
        canonical_message = DispatchMessage.from_claim(claim_result.claim)
        try:
            if canonical_message.event_type in _NON_COMMAND_EVENT_TYPES:
                execution_result = {
                    "ignored": True,
                    "reason": "non-command audit event",
                }
            else:
                execution_result = self.executor(canonical_message)
        except Exception as exc:
            delay = min(
                self.base_retry_delay_seconds * (2 ** max(0, attempt - 1)),
                self.max_retry_delay_seconds,
            )
            try:
                with self.database.session() as session:
                    row = OutboxRepository(session).mark_command_failed(
                        message.event_id,
                        owner_id=self.owner_id,
                        processing_attempt=attempt,
                        now=self.clock(),
                        error_code="CommandExecutorError",
                        error_message=str(exc),
                        retry_delay_seconds=delay,
                        max_attempts=self.max_attempts,
                    )
                    outcome_run_id = _durable_run_id(session, canonical_message)
                    exhausted = row.processing_error_code == "ProcessingRetryExhausted"
                    CommandOutcomeRepository(session).record_failure(
                        event_id=canonical_message.event_id,
                        event_key=canonical_message.event_key,
                        event_type=canonical_message.event_type,
                        aggregate_type=canonical_message.aggregate_type,
                        aggregate_id=canonical_message.aggregate_id,
                        run_id=outcome_run_id,
                        attempt=attempt,
                        error_code=row.processing_error_code or "CommandExecutorError",
                        error_message=row.processing_error_message or str(exc),
                        exhausted=exhausted,
                        now=self.clock(),
                    )
            except CommandClaimError:
                return WorkerResult(
                    message.event_key,
                    processed=False,
                    reason="claim_lost",
                    attempt=attempt,
                    error_code="CommandClaimError",
                )
            return WorkerResult(
                message.event_key,
                processed=False,
                reason="processing_retry_exhausted" if exhausted else "retry",
                attempt=attempt,
                error_code=row.processing_error_code,
            )

        try:
            with self.database.session() as session:
                OutboxRepository(session).mark_command_processed(
                    message.event_id,
                    owner_id=self.owner_id,
                    processing_attempt=attempt,
                    now=self.clock(),
                )
                CommandOutcomeRepository(session).record_success(
                    event_id=canonical_message.event_id,
                    event_key=canonical_message.event_key,
                    event_type=canonical_message.event_type,
                    aggregate_type=canonical_message.aggregate_type,
                    aggregate_id=canonical_message.aggregate_id,
                    run_id=_durable_run_id(session, canonical_message),
                    attempt=attempt,
                    result=_result_projection(execution_result),
                    now=self.clock(),
                )
        except CommandClaimError:
            return WorkerResult(
                message.event_key,
                processed=False,
                reason="claim_lost",
                attempt=attempt,
                error_code="CommandClaimError",
            )
        return WorkerResult(message.event_key, processed=True, reason="processed", attempt=attempt)

    def _handle_in_memory(self, message: DispatchMessage) -> WorkerResult:
        if self.deduplicator.seen(message.event_key):
            return WorkerResult(message.event_key, processed=False, reason="duplicate")
        if message.event_type not in _NON_COMMAND_EVENT_TYPES:
            self.executor(message)
            reason = "processed"
        else:
            reason = "ignored"
        self.deduplicator.mark(message.event_key)
        return WorkerResult(message.event_key, processed=True, reason=reason, attempt=1)


def _durable_run_id(session: Any, message: DispatchMessage) -> str | None:
    """Keep optional run FK valid even for generic or malformed command fixtures."""

    value = message.payload.get("run_id") if isinstance(message.payload, dict) else None
    if not isinstance(value, str) or not value.strip():
        return None
    row = session.get(AgentRunRow, value)
    return value if row is not None else None


def _result_projection(value: Any) -> dict[str, Any] | None:
    """Store a small JSON-safe result summary without leaking arbitrary objects."""

    if value is None:
        return None
    run = getattr(value, "run", None)
    state = getattr(value, "state", None)
    if run is not None and isinstance(state, dict):
        status = getattr(run.status, "value", run.status)
        return {
            "run_id": str(run.id),
            "run_status": str(status),
            "final_outcome": state.get("final_outcome"),
            "last_error_code": state.get("last_error_code"),
        }
    candidate: Any = value
    if isinstance(value, (str, int, float, bool, list, dict)):
        candidate = deepcopy(value)
    else:
        return {"result_type": type(value).__name__}
    try:
        json.dumps(candidate)
    except (TypeError, ValueError, OverflowError):
        return {"result_type": type(value).__name__}
    return {"value": candidate}


class RuntimeCommandError(ValueError):
    """A command cannot be bound to a run-scoped runtime."""


class RuntimeCommandHandler:
    """Protocol-like base for one complete run-scoped command boundary."""

    def start(self, message: DispatchMessage) -> Any:
        raise NotImplementedError

    def resume_approval(self, message: DispatchMessage) -> Any:
        raise NotImplementedError

    def cancel(self, message: DispatchMessage) -> Any:
        raise NotImplementedError


class RuntimeCommandExecutor:
    """Route durable command messages to an injected run-scoped runtime factory."""

    def __init__(self, runtime_factory: Callable[[DispatchMessage], RuntimeCommandHandler]) -> None:
        self.runtime_factory = runtime_factory

    def __call__(self, message: DispatchMessage) -> Any:
        if message.event_type not in {
            "run.created",
            "approval.decided",
            "run.approval_wakeup",
            "run.cancel_requested",
        }:
            raise RuntimeCommandError(f"unsupported command event type: {message.event_type}")
        if not isinstance(message.payload, dict):
            raise RuntimeCommandError("command payload must be an object")
        run_id = message.payload.get("run_id")
        if not isinstance(run_id, str) or not run_id.strip():
            raise RuntimeCommandError("command payload requires a non-empty run_id")
        handler = self.runtime_factory(message)
        if message.event_type == "run.created":
            return handler.start(message)
        if message.event_type in {"approval.decided", "run.approval_wakeup"}:
            return handler.resume_approval(message)
        return handler.cancel(message)


def register_celery_task(
    celery_app: Any,
    worker_factory: Callable[[], CommandWorker],
    *,
    task_name: str = DEFAULT_COMMAND_TASK_NAME,
    queue_name: str = DEFAULT_COMMAND_QUEUE_NAME,
) -> Any:
    """Register one Celery task; command processing remains in the durable worker."""

    @celery_app.task(name=task_name, queue=queue_name)
    def execute_command(payload: dict[str, Any]) -> dict[str, Any]:
        message = DispatchMessage.from_payload(payload)
        result = worker_factory().handle(message)
        return {
            "event_key": result.event_key,
            "processed": result.processed,
            "reason": result.reason,
            "attempt": result.attempt,
            "error_code": result.error_code,
        }

    return execute_command
