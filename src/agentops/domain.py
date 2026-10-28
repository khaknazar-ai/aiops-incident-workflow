"""Explicit domain entities and state transitions for incident operations."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from datetime import UTC, datetime
from enum import Enum, StrEnum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator

from agentops.model_identity import ModelConfigurationIdentity

INCIDENT_TRIAGE_WORKFLOW_VERSION = "0.1.0"
INCIDENT_TRIAGE_STATE_SCHEMA_VERSION = 2


def utc_now() -> datetime:
    """Return a timezone-aware UTC timestamp."""

    return datetime.now(UTC)


def stable_parameter_hash(parameters: Mapping[str, Any]) -> str:
    """Hash JSON-compatible parameters in a deterministic way."""

    try:
        encoded = json.dumps(
            parameters,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise InvariantViolation("parameters must be strict JSON values") from exc
    return hashlib.sha256(encoded).hexdigest()


class DomainError(ValueError):
    """Base error for invalid domain operations."""


class InvalidTransition(DomainError):
    """Raised when a state machine transition is not allowed."""


class InvariantViolation(DomainError):
    """Raised when an entity invariant would be violated."""


class CancellationRequested(RuntimeError):
    """Raised when a new workflow step or effect is fenced by cancellation."""


class Severity(StrEnum):
    SEV1 = "SEV1"
    SEV2 = "SEV2"
    SEV3 = "SEV3"
    SEV4 = "SEV4"


class IncidentStatus(StrEnum):
    OPEN = "OPEN"
    INVESTIGATING = "INVESTIGATING"
    MITIGATION_PROPOSED = "MITIGATION_PROPOSED"
    MITIGATING = "MITIGATING"
    VERIFYING = "VERIFYING"
    RESOLVED = "RESOLVED"
    ESCALATED = "ESCALATED"
    CANCELLED = "CANCELLED"


class AgentRunStatus(StrEnum):
    CREATED = "CREATED"
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    WAITING_FOR_APPROVAL = "WAITING_FOR_APPROVAL"
    WAITING_FOR_RETRY = "WAITING_FOR_RETRY"
    BLOCKED = "BLOCKED"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    TIMED_OUT = "TIMED_OUT"
    CANCELLED = "CANCELLED"


class ActionStatus(StrEnum):
    PROPOSED = "PROPOSED"
    VALIDATING = "VALIDATING"
    DENIED = "DENIED"
    REJECTED = "REJECTED"
    AUTHORIZED = "AUTHORIZED"
    WAITING_FOR_APPROVAL = "WAITING_FOR_APPROVAL"
    READY = "READY"
    EXECUTING = "EXECUTING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    EXPIRED = "EXPIRED"


class ApprovalStatus(StrEnum):
    PENDING = "PENDING"
    APPROVED = "APPROVED"
    APPROVED_WITH_CHANGES = "APPROVED_WITH_CHANGES"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"
    CANCELLED = "CANCELLED"


class RiskLevel(int, Enum):
    R0 = 0
    R1 = 1
    R2 = 2
    R3 = 3


class ActorType(StrEnum):
    USER = "user"
    AGENT = "agent"
    SYSTEM = "system"
    WORKER = "worker"


_INCIDENT_TRANSITIONS: dict[IncidentStatus, set[IncidentStatus]] = {
    IncidentStatus.OPEN: {IncidentStatus.INVESTIGATING},
    IncidentStatus.INVESTIGATING: {
        IncidentStatus.MITIGATION_PROPOSED,
        IncidentStatus.ESCALATED,
        IncidentStatus.CANCELLED,
    },
    IncidentStatus.MITIGATION_PROPOSED: {
        IncidentStatus.MITIGATING,
        IncidentStatus.INVESTIGATING,
        IncidentStatus.ESCALATED,
    },
    IncidentStatus.MITIGATING: {IncidentStatus.VERIFYING, IncidentStatus.ESCALATED},
    IncidentStatus.VERIFYING: {
        IncidentStatus.RESOLVED,
        IncidentStatus.INVESTIGATING,
        IncidentStatus.ESCALATED,
    },
    IncidentStatus.RESOLVED: {IncidentStatus.OPEN},
    IncidentStatus.ESCALATED: set(),
    IncidentStatus.CANCELLED: set(),
}


_RUN_TRANSITIONS: dict[AgentRunStatus, set[AgentRunStatus]] = {
    AgentRunStatus.CREATED: {AgentRunStatus.QUEUED},
    AgentRunStatus.QUEUED: {AgentRunStatus.RUNNING, AgentRunStatus.CANCELLED},
    AgentRunStatus.RUNNING: {
        AgentRunStatus.WAITING_FOR_APPROVAL,
        AgentRunStatus.WAITING_FOR_RETRY,
        AgentRunStatus.BLOCKED,
        AgentRunStatus.SUCCEEDED,
        AgentRunStatus.FAILED,
        AgentRunStatus.TIMED_OUT,
        AgentRunStatus.CANCELLED,
    },
    AgentRunStatus.WAITING_FOR_APPROVAL: {
        AgentRunStatus.RUNNING,
        AgentRunStatus.BLOCKED,
        AgentRunStatus.CANCELLED,
    },
    AgentRunStatus.WAITING_FOR_RETRY: {
        AgentRunStatus.RUNNING,
        AgentRunStatus.FAILED,
        AgentRunStatus.CANCELLED,
    },
    AgentRunStatus.BLOCKED: set(),
    AgentRunStatus.SUCCEEDED: set(),
    AgentRunStatus.FAILED: set(),
    AgentRunStatus.TIMED_OUT: set(),
    AgentRunStatus.CANCELLED: set(),
}


class Incident(BaseModel):
    """An operational incident with explicit, validated lifecycle transitions."""

    model_config = ConfigDict(validate_assignment=True)

    id: UUID = Field(default_factory=uuid4)
    title: str = Field(min_length=1)
    description: str = Field(min_length=1)
    service_name: str = Field(min_length=1)
    severity: Severity
    status: IncidentStatus = IncidentStatus.OPEN
    detected_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime | None = None
    resolved_at: datetime | None = None
    created_by: str = Field(min_length=1)
    active_run_id: UUID | None = None
    version: int = Field(default=1, ge=1)

    def transition_to(
        self,
        target: IncidentStatus,
        *,
        verification_succeeded: bool = False,
    ) -> None:
        """Apply one allowed transition and enforce the resolved invariant."""

        if target not in _INCIDENT_TRANSITIONS[self.status]:
            raise InvalidTransition(
                f"incident: {self.status.value} -> {target.value} is not allowed"
            )
        if target is IncidentStatus.RESOLVED and not verification_succeeded:
            raise InvariantViolation("an incident requires successful verification before RESOLVED")
        self.status = target
        self.version += 1
        if target is IncidentStatus.RESOLVED:
            self.resolved_at = utc_now()
        elif target is IncidentStatus.OPEN:
            self.resolved_at = None


class AgentRun(BaseModel):
    """One bounded workflow execution mapped to one durable thread."""

    model_config = ConfigDict(validate_assignment=True)

    id: UUID = Field(default_factory=uuid4)
    incident_id: UUID
    environment_id: UUID | None = None
    status: AgentRunStatus = AgentRunStatus.CREATED
    workflow_name: str = "incident_triage"
    workflow_version: str = INCIDENT_TRIAGE_WORKFLOW_VERSION
    state_schema_version: int = Field(default=INCIDENT_TRIAGE_STATE_SCHEMA_VERSION, ge=1)
    policy_version: str = "0.1.0"
    tool_registry_version: str = "0.1.0"
    model_configuration: ModelConfigurationIdentity | None = None
    thread_id: str = Field(default_factory=lambda: str(uuid4()))
    started_at: datetime | None = None
    finished_at: datetime | None = None
    cancel_requested_at: datetime | None = None
    step_count: int = Field(default=0, ge=0)
    llm_call_count: int = Field(default=0, ge=0)
    tool_call_count: int = Field(default=0, ge=0)
    last_error_code: str | None = None
    last_error_message: str | None = None

    @property
    def model_configuration_id(self) -> str | None:
        return self.model_configuration.configuration_id if self.model_configuration else None

    def transition_to(self, target: AgentRunStatus) -> None:
        """Apply one allowed run transition."""

        if target not in _RUN_TRANSITIONS[self.status]:
            raise InvalidTransition(f"run: {self.status.value} -> {target.value} is not allowed")
        self.status = target
        if target is AgentRunStatus.RUNNING and self.started_at is None:
            self.started_at = utc_now()
        if target in {
            AgentRunStatus.SUCCEEDED,
            AgentRunStatus.FAILED,
            AgentRunStatus.TIMED_OUT,
            AgentRunStatus.CANCELLED,
        }:
            self.finished_at = utc_now()

    def request_cancel(self, *, requested_at: datetime | None = None) -> bool:
        """Record one cancellation intent without pretending active work stopped."""

        if self.status in {
            AgentRunStatus.BLOCKED,
            AgentRunStatus.SUCCEEDED,
            AgentRunStatus.FAILED,
            AgentRunStatus.TIMED_OUT,
            AgentRunStatus.CANCELLED,
        }:
            raise InvalidTransition(f"terminal run cannot be cancelled: {self.status.value}")
        if self.cancel_requested_at is not None:
            return False
        self.cancel_requested_at = requested_at or utc_now()
        return True


class ApprovalRequest(BaseModel):
    """Approval bound to an exact action parameter hash."""

    model_config = ConfigDict(validate_assignment=True)

    id: UUID = Field(default_factory=uuid4)
    action_id: UUID
    run_id: UUID
    tool_name: str = ""
    status: ApprovalStatus = ApprovalStatus.PENDING
    requested_at: datetime = Field(default_factory=utc_now)
    expires_at: datetime | None = None
    decided_at: datetime | None = None
    decided_by: str | None = None
    original_parameters: dict[str, Any]
    original_parameter_hash: str = ""
    approved_parameters: dict[str, Any] | None = None
    approved_parameter_hash: str | None = None
    decision_reason: str | None = None

    @model_validator(mode="after")
    def validate_original_hash(self) -> ApprovalRequest:
        expected = stable_parameter_hash(self.original_parameters)
        if not self.original_parameter_hash:
            object.__setattr__(self, "original_parameter_hash", expected)
        elif self.original_parameter_hash != expected:
            raise InvariantViolation("original approval parameter hash does not match parameters")
        return self

    def decide(
        self,
        *,
        approved: bool,
        decided_by: str,
        approved_parameters: dict[str, Any] | None = None,
        reason: str | None = None,
    ) -> None:
        """Record an approval or rejection atomically in the in-memory model."""

        if self.status is not ApprovalStatus.PENDING:
            raise InvariantViolation("only a pending approval can receive a decision")
        if self.expires_at is not None and self.expires_at <= utc_now():
            self.status = ApprovalStatus.EXPIRED
            raise InvariantViolation("approval has expired")
        self.decided_at = utc_now()
        self.decided_by = decided_by
        self.decision_reason = reason
        if not approved:
            self.status = ApprovalStatus.REJECTED
            return
        self.approved_parameters = (
            dict(self.original_parameters)
            if approved_parameters is None
            else dict(approved_parameters)
        )
        self.approved_parameter_hash = stable_parameter_hash(self.approved_parameters)
        self.status = (
            ApprovalStatus.APPROVED
            if self.approved_parameters == self.original_parameters
            else ApprovalStatus.APPROVED_WITH_CHANGES
        )

    def permits(
        self,
        current_parameters: Mapping[str, Any],
        *,
        action_id: UUID | str | None = None,
        run_id: UUID | str | None = None,
        tool_name: str | None = None,
    ) -> bool:
        """Return whether the exact scoped parameters are approved and unexpired."""

        if self.expires_at is not None and self.expires_at <= utc_now():
            object.__setattr__(self, "status", ApprovalStatus.EXPIRED)
            return False
        if self.status not in {
            ApprovalStatus.APPROVED,
            ApprovalStatus.APPROVED_WITH_CHANGES,
        }:
            return False
        if action_id is not None and self.action_id != UUID(str(action_id)):
            return False
        if run_id is not None and self.run_id != UUID(str(run_id)):
            return False
        if tool_name is not None and self.tool_name and self.tool_name != tool_name:
            return False
        return self.approved_parameter_hash == stable_parameter_hash(current_parameters)


class Action(BaseModel):
    """A proposed side effect whose policy and approval state are explicit."""

    model_config = ConfigDict(validate_assignment=True)

    id: UUID = Field(default_factory=uuid4)
    run_id: UUID
    action_type: str = Field(min_length=1)
    tool_name: str = Field(min_length=1)
    parameters: dict[str, Any]
    parameter_hash: str = ""
    risk_level: RiskLevel
    status: ActionStatus = ActionStatus.PROPOSED
    requires_approval: bool = False
    idempotency_key: str = ""
    created_at: datetime = Field(default_factory=utc_now)
    executed_at: datetime | None = None
    completed_at: datetime | None = None
    execution_result_id: UUID | None = None

    @model_validator(mode="after")
    def initialize_policy_fields(self) -> Action:
        expected_hash = stable_parameter_hash(self.parameters)
        if not self.parameter_hash:
            object.__setattr__(self, "parameter_hash", expected_hash)
        elif self.parameter_hash != expected_hash:
            raise InvariantViolation("action parameter hash does not match parameters")
        object.__setattr__(
            self,
            "requires_approval",
            self.risk_level in {RiskLevel.R1, RiskLevel.R2},
        )
        if not self.idempotency_key:
            object.__setattr__(self, "idempotency_key", str(self.id))
        return self

    def mark_ready(self, approval: ApprovalRequest | None = None) -> None:
        """Move an action to READY only when policy and approval allow execution."""

        if self.risk_level is RiskLevel.R3:
            object.__setattr__(self, "status", ActionStatus.DENIED)
            raise InvariantViolation("R3 actions are forbidden")
        if self.status in {ActionStatus.DENIED, ActionStatus.REJECTED}:
            raise InvariantViolation(f"action is already {self.status.value}")
        if self.requires_approval and (
            approval is None
            or not approval.permits(
                self.parameters,
                action_id=self.id,
                run_id=self.run_id,
                tool_name=self.tool_name,
            )
        ):
            object.__setattr__(self, "status", ActionStatus.WAITING_FOR_APPROVAL)
            raise InvariantViolation("valid approval is required before an action is READY")
        self.status = ActionStatus.READY


class AuditEvent(BaseModel):
    """Append-only logical audit event payload."""

    id: UUID = Field(default_factory=uuid4)
    occurred_at: datetime = Field(default_factory=utc_now)
    actor_type: ActorType
    actor_id: str
    event_type: str
    entity_type: str
    entity_id: UUID | None = None
    run_id: UUID | None = None
    payload: dict[str, Any] = Field(default_factory=dict)
