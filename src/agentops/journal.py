"""Durable, fenced execution observations and their existing audit projection."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from typing import Any, Literal
from uuid import UUID, uuid4, uuid5

from sqlalchemy import func
from sqlalchemy.orm import Session

from agentops.journal_contract import NAMES, safe_facts, safe_projection
from agentops.persistence import (
    ActionRepository,
    AgentRunRepository,
    ApprovalRepository,
    AuditEventRow,
    ObservationRepository,
    PersistenceError,
    RunJournalRow,
    SimulationReceiptRow,
)

JournalKind = Literal["node", "tool", "llm", "run", "hypothesis"]
JournalPhase = Literal["started", "completed", "failed", "interrupted"]


class JournalConflict(PersistenceError):
    """An immutable journal event was replayed with contradictory facts."""


class RunJournalRepository:
    """Callers hold their existing execution fence; this class grants no authority.

    The run row lock serializes sequence/attempt allocation, including SQLite's
    explicit write-lock fallback. Events and audit projection commit together.
    Duplicate event identity returns the original record, never a new timestamp.
    """

    def __init__(self, session: Session):
        self.session = session

    def start(
        self,
        *,
        run_id: UUID,
        kind: JournalKind,
        name: str,
        logical_step: int,
        step_id: UUID | None = None,
        attempt_id: UUID | None = None,
        facts: Mapping[str, Any] | None = None,
        action_id: UUID | None = None,
        approval_id: UUID | None = None,
    ) -> RunJournalRow:
        if (
            kind not in {"node", "tool", "llm", "run", "hypothesis"}
            or not 0 <= logical_step <= 1_000_000
        ):
            raise PersistenceError("invalid journal event category")
        run = AgentRunRepository(self.session).require_for_update(run_id)
        attempt_id = attempt_id or uuid4()
        if kind == "node":
            step_id = attempt_id
        elif step_id is not None:
            parent = (
                self.session.query(RunJournalRow)
                .filter_by(
                    run_id=str(run_id), attempt_id=str(step_id), kind="node", phase="started"
                )
                .one_or_none()
            )
            if parent is None:
                raise PersistenceError("journal parent does not belong to this run")
        name = name if name in NAMES else "unknown"
        safe = safe_facts(facts or {})
        existing = self.session.get(RunJournalRow, str(uuid5(attempt_id, "started")))
        if existing is not None:
            if (
                existing.run_id,
                existing.kind,
                existing.name,
                existing.logical_step,
                existing.step_id,
                existing.facts_json,
            ) != (str(run_id), kind, name, logical_step, str(step_id) if step_id else None, safe):
                raise JournalConflict("journal start identity has conflicting facts")
            return existing
        number = (
            self.session.query(func.max(RunJournalRow.attempt_number))
            .filter_by(run_id=str(run_id), kind=kind, name=name, logical_step=logical_step)
            .scalar()
            or 0
        )
        return self._append(
            run=run,
            attempt_id=attempt_id,
            step_id=step_id,
            kind=kind,
            name=name,
            logical_step=logical_step,
            number=number + 1,
            phase="started",
            duration_ms=None,
            facts=safe,
            links=self._links(run_id, action_id, approval_id),
        )

    def finish(
        self,
        start: RunJournalRow,
        *,
        phase: JournalPhase,
        duration_ms: float,
        facts: Mapping[str, Any] | None = None,
        action_id: UUID | None = None,
        approval_id: UUID | None = None,
    ) -> RunJournalRow:
        if phase not in {"completed", "failed", "interrupted"}:
            raise PersistenceError("invalid journal terminal phase")
        stored = self.session.get(RunJournalRow, start.id)
        if stored is None or stored.phase != "started":
            raise PersistenceError("journal outcome requires a stored start event")
        start = stored
        run = AgentRunRepository(self.session).require_for_update(start.run_id)
        facts = safe_facts({**start.facts_json, **(facts or {})})
        existing = (
            self.session.query(RunJournalRow)
            .filter(RunJournalRow.attempt_id == start.attempt_id, RunJournalRow.phase != "started")
            .one_or_none()
        )
        if existing is not None:
            if existing.phase != phase or existing.facts_json != facts:
                raise JournalConflict("journal outcome identity has conflicting facts")
            return existing
        return self._append(
            run=run,
            attempt_id=UUID(start.attempt_id),
            step_id=UUID(start.step_id) if start.step_id else None,
            kind=start.kind,
            name=start.name,
            logical_step=start.logical_step,
            number=start.attempt_number,
            phase=phase,
            duration_ms=max(0.0, duration_ms),
            facts=facts,
            links=self._links(UUID(start.run_id), action_id, approval_id),
        )

    def record_state(self, run) -> None:
        # Idempotent state observation; no new state transitions or commands.
        started = self.start(
            run_id=run.id,
            kind="run",
            name="lifecycle",
            logical_step=0,
            attempt_id=uuid5(run.id, f"journal-status:{run.status.value}"),
            facts={"run_status": run.status.value},
        )
        self.finish(started, phase="completed", duration_ms=0)

    def _links(self, run_id: UUID, action_id: UUID | None, approval_id: UUID | None) -> dict:
        links: dict[str, Any] = {}
        if action_id is not None:
            action = ActionRepository(self.session).require(action_id)
            if action.run_id != run_id:
                raise PersistenceError("journal action does not belong to this run")
            links["action_id"] = str(action.id)
            receipts = (
                self.session.query(SimulationReceiptRow)
                .filter_by(
                    idempotency_key=action.idempotency_key,
                )
                .all()
            )
            # Receipt scope additionally follows the run's environment binding.
            run = AgentRunRepository(self.session).require(run_id)
            links["receipt_ids"] = [
                r.id for r in receipts if r.environment_id == str(run.environment_id)
            ][:10]
        if approval_id is not None:
            approval = ApprovalRepository(self.session).require(approval_id)
            if approval.run_id != run_id:
                raise PersistenceError("journal approval does not belong to this run")
            links["approval_id"] = str(approval.id)
        links["observation_ids"] = [
            r.id for r in ObservationRepository(self.session).list_for_run(run_id)[-50:]
        ]
        return safe_projection({"links": links})["links"]

    def _append(
        self,
        *,
        run,
        attempt_id,
        step_id,
        kind,
        name,
        logical_step,
        number,
        phase,
        duration_ms,
        facts,
        links,
    ) -> RunJournalRow:
        last = (
            self.session.query(RunJournalRow)
            .filter_by(run_id=str(run.id))
            .order_by(RunJournalRow.sequence.desc())
            .first()
        )
        recorded = datetime.now(UTC)
        if last is not None:
            previous = (
                last.occurred_at.replace(tzinfo=UTC)
                if last.occurred_at.tzinfo is None
                else last.occurred_at
            )
            # Deterministic ordering in the existing timestamp-sorted timeline,
            # even during a backward wall-clock adjustment. Durations are monotonic.
            recorded = max(recorded, previous + timedelta(microseconds=1))
        row = RunJournalRow(
            id=str(uuid5(attempt_id, phase)),
            run_id=str(run.id),
            sequence=last.sequence + 1 if last else 1,
            attempt_id=str(attempt_id),
            step_id=str(step_id) if step_id else None,
            attempt_number=number,
            logical_step=logical_step,
            kind=kind,
            name=name,
            phase=phase,
            occurred_at=recorded,
            duration_ms=duration_ms,
            model_configuration_id=run.model_configuration_id,
            facts_json=facts,
            links_json=links,
        )
        self.session.add(row)
        self.session.flush()
        projection = safe_projection(
            {
                "journal_version": 1,
                "journal_id": row.id,
                "attempt_id": row.attempt_id,
                "sequence": row.sequence,
                "step_id": row.step_id,
                "attempt": row.attempt_number,
                "kind": row.kind,
                "name": row.name,
                "phase": row.phase,
                "duration_ms": row.duration_ms,
                "model_configuration_id": row.model_configuration_id,
                "facts": facts,
                "links": links,
            }
        )
        self.session.add(
            AuditEventRow(
                id=row.id,
                event_key=f"journal:{row.id}",
                occurred_at=recorded,
                actor_type="agent_runtime",
                actor_id="agent-runtime",
                event_type=f"journal.{kind}.{phase}",
                entity_type="agent_run",
                entity_id=str(run.id),
                run_id=str(run.id),
                payload_json=projection,
            )
        )
        self.session.flush()
        return row
