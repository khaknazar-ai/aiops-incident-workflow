from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

from agentops.domain import (
    AgentRunStatus,
    ApprovalRequest,
    InvariantViolation,
    stable_parameter_hash,
)
from agentops.observability import TelemetrySink, emit_safely
from agentops.observability.audit import durable_payload
from agentops.persistence import (
    ActionRepository,
    AgentRunRepository,
    ApprovalDecisionConflictError,
    ApprovalDecisionRepository,
    ApprovalDecisionRow,
    ApprovalRepository,
    AuditEventRow,
    Database,
    OutboxRow,
)


@dataclass(frozen=True)
class ApprovalDecisionResult:
    """Committed approval decision returned to the orchestration layer."""

    decision_id: UUID
    approval: ApprovalRequest
    replayed: bool


class ApprovalRunCancelledError(InvariantViolation):
    """A new approval decision cannot be admitted after run cancellation."""


class ApprovalDecisionService:
    """Commit a human decision and its durable evidence in one transaction."""

    _ALLOWED_ROLES = {"approver", "admin"}

    def __init__(self, database: Database, telemetry: TelemetrySink | None = None) -> None:
        self.database = database
        self.telemetry = telemetry

    def decide(
        self,
        approval_id: UUID | str,
        *,
        actor_id: str,
        actor_role: str,
        approved: bool,
        approved_parameters: dict[str, Any] | None = None,
        reason: str | None = None,
    ) -> ApprovalDecisionResult:
        if not actor_id:
            raise InvariantViolation("decision actor_id is required")
        if actor_role not in self._ALLOWED_ROLES:
            raise InvariantViolation("decision actor_role is not authorized")
        if not isinstance(approved, bool):
            raise InvariantViolation("approved must be a strict boolean")
        if approved_parameters is not None and not isinstance(approved_parameters, dict):
            raise InvariantViolation("approved_parameters must be an object")
        if not approved and approved_parameters is not None:
            raise InvariantViolation("rejected decisions cannot include approved_parameters")

        fingerprint = stable_parameter_hash(
            {
                "approved": approved,
                "approved_parameters": approved_parameters,
                "actor_id": actor_id,
                "actor_role": actor_role,
                "reason": reason,
            }
        )
        with self.database.session() as session:
            approval_repo = ApprovalRepository(session)
            approval = approval_repo.require(approval_id)
            # Serialize new decisions with cancellation on the authoritative
            # run row. Historical identical decisions remain replayable.
            run = AgentRunRepository(session).require_for_update(approval.run_id)
            decision_repo = ApprovalDecisionRepository(session)
            existing = decision_repo.get_for_approval(approval.id)
            if existing is not None:
                if existing.decision_fingerprint != fingerprint:
                    raise ApprovalDecisionConflictError(
                        f"approval {approval.id} already has a different committed decision"
                    )
                return ApprovalDecisionResult(
                    decision_id=UUID(existing.id),
                    approval=approval_repo.repair_decision_projection(approval.id),
                    replayed=True,
                )

            action = ActionRepository(session).require(approval.action_id)
            if run.cancel_requested_at is not None or run.status is AgentRunStatus.CANCELLED:
                raise ApprovalRunCancelledError(
                    "cancelled run cannot admit a new approval decision"
                )
            approval.decide(
                approved=approved,
                decided_by=actor_id,
                approved_parameters=approved_parameters,
                reason=reason,
            )
            approval_repo.save(approval)
            now = approval.decided_at or datetime.now(UTC)
            decision_id = uuid4()
            decision_revision = 1
            event_key = f"approval:{approval.id}:decision:{decision_revision}"
            payload = {
                "approval_id": str(approval.id),
                "action_id": str(action.id),
                "run_id": str(approval.run_id),
                "environment_id": str(run.environment_id) if run.environment_id else None,
                "status": approval.status.value,
                "approved": approved,
                "approved_parameters": deepcopy(approved_parameters),
                "decided_by": actor_id,
                "actor_role": actor_role,
                "decision_reason": approval.decision_reason,
                "approved_parameter_hash": approval.approved_parameter_hash,
                "decision_revision": decision_revision,
            }
            decision_row = ApprovalDecisionRow(
                id=str(decision_id),
                approval_id=str(approval.id),
                action_id=str(action.id),
                run_id=str(approval.run_id),
                status=approval.status.value,
                decided_by=actor_id,
                actor_role=actor_role,
                approved_parameters_json=deepcopy(approval.approved_parameters),
                approved_parameter_hash=approval.approved_parameter_hash,
                decision_reason=approval.decision_reason,
                decision_fingerprint=fingerprint,
                decision_revision=decision_revision,
                decided_at=now,
                created_at=now,
            )
            session.add(decision_row)
            session.add(
                AuditEventRow(
                    id=str(uuid4()),
                    event_key=event_key,
                    occurred_at=now,
                    actor_type="user",
                    actor_id=actor_id,
                    event_type="approval.decided",
                    entity_type="approval_request",
                    entity_id=str(approval.id),
                    run_id=str(approval.run_id),
                    payload_json=durable_payload(payload),
                )
            )
            emit_safely(self.telemetry, event_type="approval.decided", payload=payload)
            session.add(
                OutboxRow(
                    id=str(uuid4()),
                    event_key=event_key,
                    event_type="approval.decided",
                    aggregate_type="approval_request",
                    aggregate_id=str(approval.id),
                    payload_json=payload,
                    created_at=now,
                )
            )
            wakeup_event_key = f"approval:{approval.id}:wakeup:{decision_revision}"
            wakeup_payload = {
                **payload,
                "wakeup_for_event_key": event_key,
            }
            session.add(
                OutboxRow(
                    id=str(uuid4()),
                    event_key=wakeup_event_key,
                    event_type="run.approval_wakeup",
                    aggregate_type="agent_run",
                    aggregate_id=str(approval.run_id),
                    payload_json=wakeup_payload,
                    created_at=now,
                )
            )
            session.flush()
            return ApprovalDecisionResult(
                decision_id=decision_id,
                approval=approval,
                replayed=False,
            )

    def get_for_approval(self, approval_id: UUID | str) -> ApprovalDecisionRow | None:
        with self.database.session() as session:
            row = ApprovalDecisionRepository(session).get_for_approval(approval_id)
            if row is None:
                return None
            session.expunge(row)
            return row
