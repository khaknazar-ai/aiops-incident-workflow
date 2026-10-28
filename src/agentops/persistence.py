"""SQLAlchemy persistence boundary for the core incident domain.

The models use portable SQL types so the same metadata works with the local
SQLite smoke database and PostgreSQL in Docker. Domain objects remain separate
from ORM rows; orchestration code should depend on repositories, not tables.
"""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
from copy import deepcopy
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from threading import RLock, local
from typing import Any
from uuid import UUID, uuid4, uuid5

from pydantic import ValidationError
from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    UniqueConstraint,
    and_,
    case,
    create_engine,
    event,
    exists,
    or_,
    update,
)
from sqlalchemy.dialects.postgresql import insert as postgres_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.engine import Engine
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker
from sqlalchemy.orm.exc import StaleDataError
from sqlalchemy.pool import StaticPool

from agentops.domain import (
    Action,
    ActionStatus,
    AgentRun,
    AgentRunStatus,
    ApprovalRequest,
    ApprovalStatus,
    Incident,
    IncidentStatus,
    RiskLevel,
    Severity,
    stable_parameter_hash,
    utc_now,
)
from agentops.journal_contract import observed_facts
from agentops.llm import HypothesisClaimSnapshot, HypothesisSnapshot, HypothesisUpdate


class PersistenceError(RuntimeError):
    """Base error for persistence failures."""


class EntityNotFound(PersistenceError):
    """Raised when a repository is asked for a missing entity."""


class StaleWriteError(PersistenceError):
    """Raised when a repository save would overwrite a newer entity version."""


class ExecutionConflictError(PersistenceError):
    """Raised when an execution key is bound to a different tool or fingerprint."""


class ExecutionInProgressError(PersistenceError):
    """Raised when another worker currently owns an execution claim."""


class ApprovalDecisionConflictError(PersistenceError):
    """Raised when a committed approval is replayed with different input."""


class RunLeaseHeldError(PersistenceError):
    """Raised when a live lease belongs to another worker."""


class RunLeaseFencedError(PersistenceError):
    """Raised when a worker lease is stale, expired, or has the wrong generation."""


class EnvironmentReservationHeldError(PersistenceError):
    """Raised when an environment is reserved by another run."""


class EnvironmentReservationFencedError(PersistenceError):
    """Raised when a run no longer owns the environment reservation."""


class OutboxClaimError(PersistenceError):
    """Raised when a dispatcher loses ownership of an outbox event."""


class CommandClaimError(PersistenceError):
    """Raised when a worker loses ownership of a command processing claim."""


class CommandOutcomeConflictError(PersistenceError):
    """Raised when an event is replayed with conflicting outcome identity."""


class ExecutionStatus(StrEnum):
    CLAIMED = "CLAIMED"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    UNKNOWN = "UNKNOWN"


class Base(DeclarativeBase):
    """Declarative base shared by all domain tables."""


class BrowserSessionRevocationRow(Base):
    """Digest-only tombstone; raw browser credentials are never persisted."""

    __tablename__ = "browser_session_revocations"
    __table_args__ = (
        CheckConstraint("length(token_hash) = 64", name="ck_browser_revocations_hash_length"),
        CheckConstraint("expires_at > revoked_at", name="ck_browser_revocations_valid_expiry"),
    )
    token_hash: Mapped[str] = mapped_column(String(64), primary_key=True)
    expires_at: Mapped[int] = mapped_column(BigInteger, nullable=False, index=True)
    revoked_at: Mapped[int] = mapped_column(BigInteger, nullable=False)


class BrowserSessionRevocationRepository:
    """Shared replay denial, independent of API-key authentication."""

    def __init__(self, session: Session) -> None:
        self.session = session

    def is_revoked(self, token_hash: str) -> bool:
        return self.session.get(BrowserSessionRevocationRow, token_hash) is not None

    def prune_expired(self, now: int) -> int:
        # A five-minute grace tolerates small clock skew; instances still need
        # synchronized clocks. Arbitrary rollback/skew is not supported.
        rows = (
            self.session.query(BrowserSessionRevocationRow.token_hash)
            .filter(BrowserSessionRevocationRow.expires_at <= now - 300)
            .order_by(BrowserSessionRevocationRow.token_hash)
            .limit(200)
            .all()
        )
        if not rows:
            return 0
        return (
            self.session.query(BrowserSessionRevocationRow)
            .filter(BrowserSessionRevocationRow.token_hash.in_([row[0] for row in rows]))
            .delete(synchronize_session=False)
        )

    def revoke(self, token_hash: str, *, expires_at: int, now: int) -> None:
        if len(token_hash) != 64 or any(char not in "0123456789abcdef" for char in token_hash):
            raise ValueError("invalid session digest")
        if expires_at <= now:
            return
        dialect = self.session.get_bind().dialect.name
        if dialect not in {"sqlite", "postgresql"}:
            raise ValueError("browser session revocation requires SQLite or PostgreSQL")
        insert = sqlite_insert if dialect == "sqlite" else postgres_insert
        statement = (
            insert(BrowserSessionRevocationRow)
            .values(
                token_hash=token_hash,
                expires_at=expires_at,
                revoked_at=now,
            )
            .on_conflict_do_nothing(index_elements=["token_hash"])
        )
        self.session.execute(statement)


class IncidentRow(Base):
    __tablename__ = "incidents"
    __table_args__ = (CheckConstraint("version >= 1", name="ck_incidents_version_positive"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    title: Mapped[str] = mapped_column(String(300), nullable=False)
    description: Mapped[str] = mapped_column(String(4000), nullable=False)
    service_name: Mapped[str] = mapped_column(String(200), nullable=False, index=True)
    severity: Mapped[str] = mapped_column(String(10), nullable=False)
    status: Mapped[str] = mapped_column(String(40), nullable=False, index=True)
    detected_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    updated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_by: Mapped[str] = mapped_column(String(200), nullable=False)
    active_run_id: Mapped[str | None] = mapped_column(String(36))
    version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)

    __mapper_args__ = {"version_id_col": version, "version_id_generator": False}

    @classmethod
    def from_domain(cls, incident: Incident) -> IncidentRow:
        return cls(
            id=str(incident.id),
            title=incident.title,
            description=incident.description,
            service_name=incident.service_name,
            severity=incident.severity.value,
            status=incident.status.value,
            detected_at=incident.detected_at,
            updated_at=incident.updated_at,
            resolved_at=incident.resolved_at,
            created_by=incident.created_by,
            active_run_id=str(incident.active_run_id) if incident.active_run_id else None,
            version=incident.version,
        )

    def update_from_domain(self, incident: Incident) -> None:
        replacement = self.from_domain(incident)
        for name in (
            "title",
            "description",
            "service_name",
            "severity",
            "status",
            "detected_at",
            "resolved_at",
            "created_by",
            "active_run_id",
            "version",
        ):
            setattr(self, name, getattr(replacement, name))

    def to_domain(self) -> Incident:
        return Incident(
            id=UUID(self.id),
            title=self.title,
            description=self.description,
            service_name=self.service_name,
            severity=Severity(self.severity),
            status=IncidentStatus(self.status),
            detected_at=_as_utc(self.detected_at),
            updated_at=_as_utc(self.updated_at),
            resolved_at=_as_utc(self.resolved_at),
            created_by=self.created_by,
            active_run_id=UUID(self.active_run_id) if self.active_run_id else None,
            version=self.version,
        )


class AgentRunRow(Base):
    __tablename__ = "agent_runs"
    __table_args__ = (
        CheckConstraint("step_count >= 0", name="ck_agent_runs_step_count_nonnegative"),
        CheckConstraint("llm_call_count >= 0", name="ck_agent_runs_llm_count_nonnegative"),
        CheckConstraint("tool_call_count >= 0", name="ck_agent_runs_tool_count_nonnegative"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    incident_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("incidents.id"), nullable=False, index=True
    )
    environment_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("simulation_environments.id"), nullable=True, index=True
    )
    status: Mapped[str] = mapped_column(String(40), nullable=False, index=True)
    workflow_name: Mapped[str] = mapped_column(String(200), nullable=False)
    workflow_version: Mapped[str] = mapped_column(String(50), nullable=False)
    state_schema_version: Mapped[int] = mapped_column(Integer, nullable=False)
    policy_version: Mapped[str] = mapped_column(String(50), nullable=False)
    tool_registry_version: Mapped[str] = mapped_column(String(50), nullable=False)
    model_configuration_json: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    thread_id: Mapped[str] = mapped_column(String(200), nullable=False, unique=True)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    cancel_requested_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), index=True
    )
    step_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    llm_call_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    tool_call_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    last_error_code: Mapped[str | None] = mapped_column(String(100))
    last_error_message: Mapped[str | None] = mapped_column(String(2000))

    @classmethod
    def from_domain(cls, run: AgentRun) -> AgentRunRow:
        return cls(
            id=str(run.id),
            incident_id=str(run.incident_id),
            environment_id=str(run.environment_id) if run.environment_id is not None else None,
            status=run.status.value,
            workflow_name=run.workflow_name,
            workflow_version=run.workflow_version,
            state_schema_version=run.state_schema_version,
            policy_version=run.policy_version,
            tool_registry_version=run.tool_registry_version,
            model_configuration_json=(
                run.model_configuration.model_dump(mode="json") if run.model_configuration else None
            ),
            thread_id=run.thread_id,
            started_at=run.started_at,
            finished_at=run.finished_at,
            cancel_requested_at=run.cancel_requested_at,
            step_count=run.step_count,
            llm_call_count=run.llm_call_count,
            tool_call_count=run.tool_call_count,
            last_error_code=run.last_error_code,
            last_error_message=run.last_error_message,
        )

    def update_from_domain(self, run: AgentRun) -> None:
        replacement = self.from_domain(run)
        for name in (
            "incident_id",
            "environment_id",
            "status",
            "workflow_name",
            "workflow_version",
            "state_schema_version",
            "policy_version",
            "tool_registry_version",
            "model_configuration_json",
            "thread_id",
            "started_at",
            "finished_at",
            "step_count",
            "llm_call_count",
            "tool_call_count",
            "last_error_code",
            "last_error_message",
        ):
            setattr(self, name, getattr(replacement, name))

    def to_domain(self) -> AgentRun:
        return AgentRun(
            id=UUID(self.id),
            incident_id=UUID(self.incident_id),
            environment_id=UUID(self.environment_id) if self.environment_id else None,
            status=AgentRunStatus(self.status),
            workflow_name=self.workflow_name,
            workflow_version=self.workflow_version,
            state_schema_version=self.state_schema_version,
            policy_version=self.policy_version,
            tool_registry_version=self.tool_registry_version,
            model_configuration=self.model_configuration_json,
            thread_id=self.thread_id,
            started_at=_as_utc(self.started_at),
            finished_at=_as_utc(self.finished_at),
            cancel_requested_at=_as_utc(self.cancel_requested_at),
            step_count=self.step_count,
            llm_call_count=self.llm_call_count,
            tool_call_count=self.tool_call_count,
            last_error_code=self.last_error_code,
            last_error_message=self.last_error_message,
        )


class RunBudgetRow(Base):
    """Durable limits and usage for one workflow run."""

    __tablename__ = "run_budgets"
    __table_args__ = (
        CheckConstraint("max_steps >= 1", name="ck_run_budgets_max_steps_positive"),
        CheckConstraint("max_llm_calls >= 1", name="ck_run_budgets_max_llm_positive"),
        CheckConstraint("max_tool_calls >= 1", name="ck_run_budgets_max_tool_positive"),
        CheckConstraint(
            "max_active_seconds > 0",
            name="ck_run_budgets_max_active_seconds_positive",
        ),
        CheckConstraint("steps_used >= 0", name="ck_run_budgets_steps_nonnegative"),
        CheckConstraint("llm_calls_used >= 0", name="ck_run_budgets_llm_calls_nonnegative"),
        CheckConstraint("tool_calls_used >= 0", name="ck_run_budgets_tool_calls_nonnegative"),
        CheckConstraint(
            "active_elapsed_seconds >= 0",
            name="ck_run_budgets_active_elapsed_nonnegative",
        ),
    )

    run_id: Mapped[str] = mapped_column(String(36), ForeignKey("agent_runs.id"), primary_key=True)
    max_steps: Mapped[int] = mapped_column(Integer, nullable=False)
    max_llm_calls: Mapped[int] = mapped_column(Integer, nullable=False)
    max_tool_calls: Mapped[int] = mapped_column(Integer, nullable=False)
    max_active_seconds: Mapped[float] = mapped_column(Float, nullable=False)
    steps_used: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    llm_calls_used: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    tool_calls_used: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    active_elapsed_seconds: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    active_started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class RunLeaseRow(Base):
    """Current fenced ownership record for one run."""

    __tablename__ = "run_leases"
    __table_args__ = (CheckConstraint("generation >= 1", name="ck_run_leases_generation_positive"),)

    run_id: Mapped[str] = mapped_column(String(36), ForeignKey("agent_runs.id"), primary_key=True)
    owner_id: Mapped[str] = mapped_column(String(200), nullable=False, index=True)
    generation: Mapped[int] = mapped_column(Integer, nullable=False)
    acquired_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    renewed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    expires_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, index=True
    )


class EnvironmentReservationRow(Base):
    """Current durable reservation for one simulator environment."""

    __tablename__ = "environment_reservations"
    __table_args__ = (
        CheckConstraint(
            "generation >= 1",
            name="ck_environment_reservations_generation_positive",
        ),
        UniqueConstraint("run_id", name="uq_environment_reservations_run"),
        Index("ix_environment_reservations_run_id", "run_id"),
        Index("ix_environment_reservations_released_at", "released_at"),
    )

    environment_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("simulation_environments.id"),
        primary_key=True,
    )
    run_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("agent_runs.id"),
        nullable=False,
    )
    generation: Mapped[int] = mapped_column(Integer, nullable=False)
    acquired_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    released_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class ActionRow(Base):
    __tablename__ = "actions"
    __table_args__ = (
        CheckConstraint("risk_level BETWEEN 0 AND 3", name="ck_actions_risk_level"),
        CheckConstraint("requires_approval IN (false, true)", name="ck_actions_requires_approval"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    run_id: Mapped[str] = mapped_column(String(36), ForeignKey("agent_runs.id"), nullable=False)
    action_type: Mapped[str] = mapped_column(String(100), nullable=False)
    tool_name: Mapped[str] = mapped_column(String(200), nullable=False)
    parameters_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    parameter_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    risk_level: Mapped[int] = mapped_column(Integer, nullable=False)
    status: Mapped[str] = mapped_column(String(40), nullable=False, index=True)
    requires_approval: Mapped[bool] = mapped_column(nullable=False, default=False)
    idempotency_key: Mapped[str] = mapped_column(String(200), nullable=False, unique=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    executed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    execution_result_id: Mapped[str | None] = mapped_column(String(36))

    @classmethod
    def from_domain(cls, action: Action) -> ActionRow:
        return cls(
            id=str(action.id),
            run_id=str(action.run_id),
            action_type=action.action_type,
            tool_name=action.tool_name,
            parameters_json=deepcopy(action.parameters),
            parameter_hash=action.parameter_hash,
            risk_level=action.risk_level.value,
            status=action.status.value,
            requires_approval=action.requires_approval,
            idempotency_key=action.idempotency_key,
            created_at=action.created_at,
            executed_at=action.executed_at,
            completed_at=action.completed_at,
            execution_result_id=str(action.execution_result_id)
            if action.execution_result_id
            else None,
        )

    def update_from_domain(self, action: Action) -> None:
        replacement = self.from_domain(action)
        for name in (
            "run_id",
            "action_type",
            "tool_name",
            "parameters_json",
            "parameter_hash",
            "risk_level",
            "status",
            "requires_approval",
            "idempotency_key",
            "created_at",
            "executed_at",
            "completed_at",
            "execution_result_id",
        ):
            setattr(self, name, getattr(replacement, name))

    def to_domain(self) -> Action:
        return Action(
            id=UUID(self.id),
            run_id=UUID(self.run_id),
            action_type=self.action_type,
            tool_name=self.tool_name,
            parameters=deepcopy(self.parameters_json),
            parameter_hash=self.parameter_hash,
            risk_level=RiskLevel(self.risk_level),
            status=ActionStatus(self.status),
            requires_approval=self.requires_approval,
            idempotency_key=self.idempotency_key,
            created_at=_as_utc(self.created_at),
            executed_at=_as_utc(self.executed_at),
            completed_at=_as_utc(self.completed_at),
            execution_result_id=UUID(self.execution_result_id)
            if self.execution_result_id
            else None,
        )


class ApprovalRequestRow(Base):
    __tablename__ = "approval_requests"
    __table_args__ = (
        CheckConstraint(
            "approved_parameter_hash IS NULL OR length(approved_parameter_hash) = 64",
            name="ck_approvals_approved_hash",
        ),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    action_id: Mapped[str] = mapped_column(String(36), ForeignKey("actions.id"), nullable=False)
    run_id: Mapped[str] = mapped_column(String(36), ForeignKey("agent_runs.id"), nullable=False)
    status: Mapped[str] = mapped_column(String(40), nullable=False, index=True)
    requested_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    decided_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    decided_by: Mapped[str | None] = mapped_column(String(200))
    original_parameters_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    original_parameter_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    approved_parameters_json: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    approved_parameter_hash: Mapped[str | None] = mapped_column(String(64))
    decision_reason: Mapped[str | None] = mapped_column(String(2000))

    @classmethod
    def from_domain(cls, approval: ApprovalRequest) -> ApprovalRequestRow:
        return cls(
            id=str(approval.id),
            action_id=str(approval.action_id),
            run_id=str(approval.run_id),
            status=approval.status.value,
            requested_at=approval.requested_at,
            expires_at=approval.expires_at,
            decided_at=approval.decided_at,
            decided_by=approval.decided_by,
            original_parameters_json=deepcopy(approval.original_parameters),
            original_parameter_hash=approval.original_parameter_hash,
            approved_parameters_json=deepcopy(approval.approved_parameters)
            if approval.approved_parameters is not None
            else None,
            approved_parameter_hash=approval.approved_parameter_hash,
            decision_reason=approval.decision_reason,
        )

    def update_from_domain(self, approval: ApprovalRequest) -> None:
        replacement = self.from_domain(approval)
        for name in (
            "action_id",
            "run_id",
            "status",
            "requested_at",
            "expires_at",
            "decided_at",
            "decided_by",
            "original_parameters_json",
            "original_parameter_hash",
            "approved_parameters_json",
            "approved_parameter_hash",
            "decision_reason",
        ):
            setattr(self, name, getattr(replacement, name))

    def to_domain(self, *, tool_name: str) -> ApprovalRequest:
        return ApprovalRequest(
            id=UUID(self.id),
            action_id=UUID(self.action_id),
            run_id=UUID(self.run_id),
            tool_name=tool_name,
            status=ApprovalStatus(self.status),
            requested_at=_as_utc(self.requested_at),
            expires_at=_as_utc(self.expires_at),
            decided_at=_as_utc(self.decided_at),
            decided_by=self.decided_by,
            original_parameters=deepcopy(self.original_parameters_json),
            original_parameter_hash=self.original_parameter_hash,
            approved_parameters=deepcopy(self.approved_parameters_json)
            if self.approved_parameters_json is not None
            else None,
            approved_parameter_hash=self.approved_parameter_hash,
            decision_reason=self.decision_reason,
        )


class ApprovalDecisionRow(Base):
    __tablename__ = "approval_decisions"
    __table_args__ = (
        CheckConstraint("decision_revision >= 1", name="ck_approval_decisions_revision_positive"),
        UniqueConstraint("approval_id", name="uq_approval_decisions_approval"),
        UniqueConstraint(
            "approval_id",
            "decision_revision",
            name="uq_approval_decisions_approval_revision",
        ),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    approval_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("approval_requests.id"), nullable=False, index=True
    )
    action_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("actions.id"), nullable=False, index=True
    )
    run_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("agent_runs.id"), nullable=False, index=True
    )
    status: Mapped[str] = mapped_column(String(40), nullable=False)
    decided_by: Mapped[str] = mapped_column(String(200), nullable=False)
    actor_role: Mapped[str] = mapped_column(String(40), nullable=False)
    approved_parameters_json: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    approved_parameter_hash: Mapped[str | None] = mapped_column(String(64))
    decision_reason: Mapped[str | None] = mapped_column(String(2000))
    decision_fingerprint: Mapped[str] = mapped_column(String(64), nullable=False)
    decision_revision: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    decided_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class AuditEventRow(Base):
    __tablename__ = "audit_events"
    __table_args__ = (UniqueConstraint("event_key", name="uq_audit_events_event_key"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    event_key: Mapped[str] = mapped_column(String(300), nullable=False)
    occurred_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    actor_type: Mapped[str] = mapped_column(String(40), nullable=False)
    actor_id: Mapped[str] = mapped_column(String(200), nullable=False)
    event_type: Mapped[str] = mapped_column(String(100), nullable=False)
    entity_type: Mapped[str] = mapped_column(String(100), nullable=False)
    entity_id: Mapped[str | None] = mapped_column(String(36))
    run_id: Mapped[str | None] = mapped_column(String(36))
    payload_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)


class RunJournalRow(Base):
    """Append-only observations of execution; never an execution authority."""

    __tablename__ = "run_journal"
    __table_args__ = (
        UniqueConstraint("run_id", "sequence", name="uq_run_journal_sequence"),
        UniqueConstraint("attempt_id", "phase", name="uq_run_journal_attempt_phase"),
        CheckConstraint("sequence >= 1 AND attempt_number >= 1", name="ck_run_journal_positive"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    run_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("agent_runs.id"), nullable=False, index=True
    )
    sequence: Mapped[int] = mapped_column(Integer, nullable=False)
    attempt_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True)
    step_id: Mapped[str | None] = mapped_column(String(36))
    attempt_number: Mapped[int] = mapped_column(Integer, nullable=False)
    logical_step: Mapped[int] = mapped_column(Integer, nullable=False)
    kind: Mapped[str] = mapped_column(String(16), nullable=False)
    name: Mapped[str] = mapped_column(String(64), nullable=False)
    phase: Mapped[str] = mapped_column(String(16), nullable=False)
    occurred_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    duration_ms: Mapped[float | None] = mapped_column(Float)
    model_configuration_id: Mapped[str | None] = mapped_column(String(64))
    facts_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    links_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)


class HypothesisDecisionRow(Base):
    """One committed structured model decision; replay freezes its exact snapshot."""

    __tablename__ = "hypothesis_decisions"
    __table_args__ = (
        UniqueConstraint("run_id", "logical_step", name="uq_hypothesis_decision_step"),
    )
    id: Mapped[str] = mapped_column(String(36), ForeignKey("run_journal.id"), primary_key=True)
    run_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("agent_runs.id"), nullable=False, index=True
    )
    logical_step: Mapped[int] = mapped_column(Integer, nullable=False)
    request_fingerprint: Mapped[str] = mapped_column(String(64), nullable=False)
    snapshot_version_ids: Mapped[list[str]] = mapped_column(JSON, nullable=False)
    changed_version_ids: Mapped[list[str]] = mapped_column(JSON, nullable=False)
    needs_more_evidence: Mapped[bool] = mapped_column(Boolean, nullable=False)
    diagnosis_json: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class HypothesisVersionRow(Base):
    """Immutable individual hypothesis version; no raw model text is stored."""

    __tablename__ = "hypothesis_versions"
    __table_args__ = (
        UniqueConstraint("hypothesis_id", "version", name="uq_hypothesis_version"),
        CheckConstraint("version >= 1", name="ck_hypothesis_version_positive"),
    )
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    hypothesis_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True)
    run_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("agent_runs.id"), nullable=False, index=True
    )
    version: Mapped[int] = mapped_column(Integer, nullable=False)
    decision_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("hypothesis_decisions.id"), nullable=False
    )
    snapshot_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class OutboxRow(Base):
    __tablename__ = "outbox_events"
    __table_args__ = (UniqueConstraint("event_key", name="uq_outbox_events_event_key"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    event_key: Mapped[str] = mapped_column(String(300), nullable=False)
    event_type: Mapped[str] = mapped_column(String(100), nullable=False)
    aggregate_type: Mapped[str] = mapped_column(String(100), nullable=False)
    aggregate_id: Mapped[str] = mapped_column(String(36), nullable=False)
    payload_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    published_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    available_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), index=True)
    dispatch_attempts: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    claimed_by: Mapped[str | None] = mapped_column(String(200), index=True)
    claimed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    last_error_code: Mapped[str | None] = mapped_column(String(100))
    last_error_message: Mapped[str | None] = mapped_column(String(2000))

    processing_available_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), index=True
    )
    processing_attempts: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    processing_owner: Mapped[str | None] = mapped_column(String(200), index=True)
    processing_started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    processing_expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    processed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    processing_error_code: Mapped[str | None] = mapped_column(String(100))
    processing_error_message: Mapped[str | None] = mapped_column(String(2000))


class CommandOutcomeRow(Base):
    """Durable, one-row projection of the latest outcome for one command event."""

    __tablename__ = "command_outcomes"
    __table_args__ = (
        CheckConstraint("attempt >= 1", name="ck_command_outcomes_attempt_positive"),
        UniqueConstraint("event_id", name="uq_command_outcomes_event_id"),
        Index("ix_command_outcomes_run_id", "run_id"),
        Index("ix_command_outcomes_status", "status"),
        Index("ix_command_outcomes_updated_at", "updated_at"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    event_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("outbox_events.id"), nullable=False
    )
    event_key: Mapped[str] = mapped_column(String(300), nullable=False)
    event_type: Mapped[str] = mapped_column(String(100), nullable=False)
    aggregate_type: Mapped[str] = mapped_column(String(100), nullable=False)
    aggregate_id: Mapped[str] = mapped_column(String(36), nullable=False)
    run_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("agent_runs.id"), nullable=True
    )
    status: Mapped[str] = mapped_column(String(40), nullable=False)
    attempt: Mapped[int] = mapped_column(Integer, nullable=False)
    result_json: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    error_code: Mapped[str | None] = mapped_column(String(100))
    error_message: Mapped[str | None] = mapped_column(String(2000))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class ApiRequestIdempotencyRow(Base):
    """Durable response binding for an authenticated HTTP request key."""

    __tablename__ = "api_request_idempotency"
    __table_args__ = (
        UniqueConstraint(
            "actor_id",
            "operation",
            "target_id",
            "idempotency_key",
            name="uq_api_request_idempotency_scope",
        ),
        Index("ix_api_request_idempotency_created_at", "created_at"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    actor_id: Mapped[str] = mapped_column(String(200), nullable=False)
    operation: Mapped[str] = mapped_column(String(100), nullable=False)
    target_id: Mapped[str] = mapped_column(String(200), nullable=False)
    idempotency_key: Mapped[str] = mapped_column(String(200), nullable=False)
    request_fingerprint: Mapped[str] = mapped_column(String(64), nullable=False)
    resource_type: Mapped[str] = mapped_column(String(100), nullable=False)
    resource_id: Mapped[str] = mapped_column(String(36), nullable=False)
    response_status: Mapped[int] = mapped_column(Integer, nullable=False)
    response_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class EvaluationJobRow(Base):
    """Durable admission and outcome for a bounded offline evaluation request.

    The nullable unique slot admits at most one active/quarantined executor in
    the database. Expiry never transfers the slot to another executor.
    """

    __tablename__ = "evaluation_jobs"
    __table_args__ = (
        UniqueConstraint("actor_id", "idempotency_key", name="uq_evaluation_jobs_request"),
        UniqueConstraint("active_slot", name="uq_evaluation_jobs_active_slot"),
        CheckConstraint("active_slot IS NULL OR active_slot = 1", name="ck_evaluation_jobs_slot"),
        CheckConstraint("seed >= 0", name="ck_evaluation_jobs_seed"),
        CheckConstraint(
            "status IN ('RUNNING', 'SUCCEEDED', 'FAILED', 'QUARANTINED')",
            name="ck_evaluation_jobs_status",
        ),
        CheckConstraint(
            "(status IN ('RUNNING', 'QUARANTINED') "
            "AND active_slot IS NOT NULL AND active_slot = 1) "
            "OR (status IN ('SUCCEEDED', 'FAILED') AND active_slot IS NULL)",
            name="ck_evaluation_jobs_status_slot",
        ),
    )
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    actor_id: Mapped[str] = mapped_column(String(200), nullable=False)
    idempotency_key: Mapped[str] = mapped_column(String(200), nullable=False)
    request_fingerprint: Mapped[str] = mapped_column(String(64), nullable=False)
    mode: Mapped[str] = mapped_column(String(40), nullable=False)
    seed: Mapped[int] = mapped_column(Integer, nullable=False)
    owner_token: Mapped[str] = mapped_column(String(36), nullable=False)
    status: Mapped[str] = mapped_column(String(20), nullable=False)
    active_slot: Mapped[int | None] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    error_code: Mapped[str | None] = mapped_column(String(100))
    evaluation_run_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("evaluation_runs.id")
    )


class EvaluationRunRow(Base):
    """Immutable persisted output from one bounded offline evaluation run."""

    __tablename__ = "evaluation_runs"
    __table_args__ = (
        CheckConstraint("seed >= 0", name="ck_evaluation_runs_seed_nonnegative"),
        CheckConstraint("denominator >= 0", name="ck_evaluation_runs_denominator_nonnegative"),
        CheckConstraint("pass_count >= 0", name="ck_evaluation_runs_pass_count_nonnegative"),
        CheckConstraint("fail_count >= 0", name="ck_evaluation_runs_fail_count_nonnegative"),
        CheckConstraint(
            "unavailable_count >= 0",
            name="ck_evaluation_runs_unavailable_count_nonnegative",
        ),
        CheckConstraint(
            "pass_count + fail_count + unavailable_count = denominator",
            name="ck_evaluation_runs_counts_match_denominator",
        ),
        Index("ix_evaluation_runs_created_at", "created_at"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    mode: Mapped[str] = mapped_column(String(40), nullable=False)
    seed: Mapped[int] = mapped_column(Integer, nullable=False)
    manifest_version: Mapped[str] = mapped_column(String(100), nullable=False)
    manifest_digest: Mapped[str] = mapped_column(String(64), nullable=False)
    source_revision: Mapped[str] = mapped_column(String(80), nullable=False)
    cohort: Mapped[str] = mapped_column(String(100), nullable=False)
    denominator: Mapped[int] = mapped_column(Integer, nullable=False)
    pass_count: Mapped[int] = mapped_column(Integer, nullable=False)
    fail_count: Mapped[int] = mapped_column(Integer, nullable=False)
    unavailable_count: Mapped[int] = mapped_column(Integer, nullable=False)
    created_by: Mapped[str] = mapped_column(String(200), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    report_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)


@dataclass(frozen=True)
class OutboxClaim:
    event_id: UUID
    event_key: str
    event_type: str
    aggregate_type: str
    aggregate_id: str
    payload: dict[str, Any]
    dispatch_attempts: int


class CommandClaimState(StrEnum):
    CLAIMED = "CLAIMED"
    PROCESSED = "PROCESSED"
    IN_PROGRESS = "IN_PROGRESS"
    RETRY_WAIT = "RETRY_WAIT"
    EXHAUSTED = "EXHAUSTED"


class CommandOutcomeStatus(StrEnum):
    """Durable projection status for one command event."""

    RETRYING = "RETRYING"
    SUCCEEDED = "SUCCEEDED"
    EXHAUSTED = "EXHAUSTED"


@dataclass(frozen=True)
class CommandClaimResult:
    state: CommandClaimState
    claim: OutboxClaim | None = None
    processing_attempts: int = 0


class CheckpointRow(Base):
    __tablename__ = "run_checkpoints"
    __table_args__ = (
        CheckConstraint(
            "state_schema_version >= 1", name="ck_run_checkpoints_schema_version_positive"
        ),
        UniqueConstraint("run_id", name="uq_run_checkpoints_run"),
        UniqueConstraint("thread_id", name="uq_run_checkpoints_thread"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    run_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("agent_runs.id"), nullable=False, index=True
    )
    thread_id: Mapped[str] = mapped_column(String(200), nullable=False)
    checkpoint_id: Mapped[str] = mapped_column(String(300), nullable=False)
    checkpoint_kind: Mapped[str] = mapped_column(String(40), nullable=False)
    workflow_name: Mapped[str] = mapped_column(String(200), nullable=False)
    workflow_version: Mapped[str] = mapped_column(String(50), nullable=False)
    state_schema_version: Mapped[int] = mapped_column(Integer, nullable=False)
    policy_version: Mapped[str] = mapped_column(String(50), nullable=False)
    tool_registry_version: Mapped[str] = mapped_column(String(50), nullable=False)
    phase: Mapped[str] = mapped_column(String(80), nullable=False)
    action_id: Mapped[str | None] = mapped_column(String(36), ForeignKey("actions.id"))
    approval_id: Mapped[str | None] = mapped_column(String(36), ForeignKey("approval_requests.id"))
    decision_id: Mapped[str | None] = mapped_column(String(36), ForeignKey("approval_decisions.id"))
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class ObservationRow(Base):
    __tablename__ = "run_observations"
    __table_args__ = (
        CheckConstraint("sequence >= 0", name="ck_run_observations_sequence_nonnegative"),
        UniqueConstraint("run_id", "sequence", name="uq_run_observations_run_sequence"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    run_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("agent_runs.id"), nullable=False, index=True
    )
    sequence: Mapped[int] = mapped_column(Integer, nullable=False)
    environment_id: Mapped[str] = mapped_column(String(36), nullable=False)
    environment_version: Mapped[int] = mapped_column(Integer, nullable=False)
    source_tool: Mapped[str] = mapped_column(String(200), nullable=False)
    tool_call_id: Mapped[str | None] = mapped_column(String(36), nullable=True, index=True)
    payload_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    observed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class ExecutionRow(Base):
    __tablename__ = "executions"
    __table_args__ = (
        CheckConstraint("attempt_count >= 1", name="ck_executions_attempt_count_positive"),
        UniqueConstraint("execution_key", name="uq_executions_execution_key"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    execution_key: Mapped[str] = mapped_column(String(300), nullable=False)
    tool_name: Mapped[str] = mapped_column(String(200), nullable=False)
    parameter_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    result_json: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    attempt_count: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    last_error_code: Mapped[str | None] = mapped_column(String(100))
    last_error_message: Mapped[str | None] = mapped_column(String(2000))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


@dataclass(frozen=True)
class ExecutionClaim:
    row: ExecutionRow
    replayed: bool


class SimulationEnvironmentRow(Base):
    __tablename__ = "simulation_environments"
    __table_args__ = (
        CheckConstraint("generation >= 1", name="ck_simulation_environments_generation_positive"),
        CheckConstraint("version >= 1", name="ck_simulation_environments_version_positive"),
    )
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    scenario: Mapped[str] = mapped_column(String(100), nullable=False)
    generation: Mapped[int] = mapped_column(Integer, nullable=False)
    version: Mapped[int] = mapped_column(Integer, nullable=False)
    state_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    __mapper_args__ = {"version_id_col": version, "version_id_generator": False}


class SimulationReceiptRow(Base):
    __tablename__ = "simulation_receipts"
    __table_args__ = (
        CheckConstraint("generation >= 1", name="ck_simulation_receipts_generation_positive"),
        UniqueConstraint(
            "environment_id",
            "generation",
            "idempotency_key",
            name="uq_simulation_receipts_environment_generation_key",
        ),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    environment_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("simulation_environments.id"), nullable=False, index=True
    )
    generation: Mapped[int] = mapped_column(Integer, nullable=False)
    idempotency_key: Mapped[str] = mapped_column(String(200), nullable=False)
    tool_name: Mapped[str] = mapped_column(String(200), nullable=False)
    parameter_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    result_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    environment_version: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class ExecutionLedgerRepository:
    """Claim and finalize one durable side-effect execution key."""

    def __init__(self, session: Session) -> None:
        self.session = session

    def get(self, execution_key: str) -> ExecutionRow | None:
        return self.session.query(ExecutionRow).filter_by(execution_key=execution_key).one_or_none()

    def claim(
        self,
        *,
        execution_key: str,
        tool_name: str,
        parameter_hash: str,
    ) -> ExecutionClaim:
        row = self.get(execution_key)
        if row is None:
            now = datetime.now(UTC)
            row = ExecutionRow(
                id=str(uuid4()),
                execution_key=execution_key,
                tool_name=tool_name,
                parameter_hash=parameter_hash,
                status=ExecutionStatus.CLAIMED.value,
                result_json=None,
                attempt_count=1,
                created_at=now,
                updated_at=now,
            )
            try:
                with self.session.begin_nested():
                    self.session.add(row)
                    self.session.flush()
            except IntegrityError as exc:
                raise ExecutionInProgressError("execution was concurrently claimed") from exc
            return ExecutionClaim(row=row, replayed=False)
        if row.tool_name != tool_name or row.parameter_hash != parameter_hash:
            raise ExecutionConflictError(
                f"execution key is bound to {row.tool_name} and its original parameters"
            )
        status = ExecutionStatus(row.status)
        if status is ExecutionStatus.SUCCEEDED:
            return ExecutionClaim(row=row, replayed=True)
        if status is ExecutionStatus.UNKNOWN:
            return ExecutionClaim(row=row, replayed=True)
        if status is ExecutionStatus.CLAIMED:
            raise ExecutionInProgressError(f"execution is already claimed: {execution_key}")
        changed = self.session.execute(
            update(ExecutionRow)
            .where(
                ExecutionRow.id == row.id,
                ExecutionRow.status == ExecutionStatus.FAILED.value,
                ExecutionRow.attempt_count == row.attempt_count,
            )
            .values(
                status=ExecutionStatus.CLAIMED.value,
                attempt_count=ExecutionRow.attempt_count + 1,
                last_error_code=None,
                last_error_message=None,
                updated_at=datetime.now(UTC),
            )
            .execution_options(synchronize_session=False)
        )
        self.session.refresh(row)
        if changed.rowcount == 0:
            if row.status in {ExecutionStatus.SUCCEEDED.value, ExecutionStatus.UNKNOWN.value}:
                return ExecutionClaim(row=row, replayed=True)
            raise ExecutionInProgressError("execution claim changed concurrently")
        return ExecutionClaim(row=row, replayed=False)

    def complete(self, execution_id: UUID | str, result: dict[str, Any]) -> ExecutionRow:
        return self._transition(
            execution_id,
            (ExecutionStatus.CLAIMED,),
            ExecutionStatus.SUCCEEDED,
            result_json=deepcopy(result),
        )

    def recover_confirmed(self, execution_id: UUID | str, result: dict[str, Any]) -> ExecutionRow:
        """Called only after trusted backend receipt and current-owner verification."""
        return self._transition(
            execution_id,
            (ExecutionStatus.CLAIMED, ExecutionStatus.UNKNOWN),
            ExecutionStatus.SUCCEEDED,
            result_json=deepcopy(result),
            last_error_code=None,
            last_error_message=None,
        )

    def reconcile(self, execution_id: UUID | str, result: dict[str, Any]) -> ExecutionRow:
        return self._transition(
            execution_id,
            (ExecutionStatus.UNKNOWN,),
            ExecutionStatus.SUCCEEDED,
            result_json=deepcopy(result),
            last_error_code=None,
            last_error_message=None,
        )

    def fail(
        self,
        execution_id: UUID | str,
        *,
        error_code: str,
        error_message: str,
    ) -> ExecutionRow:
        return self._transition(
            execution_id,
            (ExecutionStatus.CLAIMED,),
            ExecutionStatus.FAILED,
            last_error_code=error_code,
            last_error_message=error_message,
        )

    def mark_unknown(
        self,
        execution_id: UUID | str,
        *,
        error_code: str,
        error_message: str,
    ) -> ExecutionRow:
        return self._transition(
            execution_id,
            (ExecutionStatus.CLAIMED,),
            ExecutionStatus.UNKNOWN,
            last_error_code=error_code,
            last_error_message=error_message,
        )

    def _transition(
        self,
        execution_id: UUID | str,
        allowed: tuple[ExecutionStatus, ...],
        status: ExecutionStatus,
        **values: Any,
    ) -> ExecutionRow:
        changed = self.session.execute(
            update(ExecutionRow)
            .where(
                ExecutionRow.id == _key(execution_id),
                ExecutionRow.status.in_([item.value for item in allowed]),
            )
            .values(status=status.value, updated_at=datetime.now(UTC), **values)
            .execution_options(synchronize_session=False)
        )
        row = self.session.get(ExecutionRow, _key(execution_id))
        if row is None:
            raise EntityNotFound(f"execution not found: {execution_id}")
        self.session.refresh(row)
        if changed.rowcount == 0:
            if row.status == ExecutionStatus.SUCCEEDED.value and (
                status is not ExecutionStatus.SUCCEEDED
                or row.result_json == values.get("result_json")
            ):
                # A late failure report cannot erase a confirmed success.
                return row
            raise ExecutionConflictError("execution transition conflicts with its durable state")
        return row


class ObservationRepository:
    """Append-only, replay-idempotent observations for one agent run."""

    def __init__(self, session: Session) -> None:
        self.session = session

    def get(self, run_id: UUID | str, sequence: int) -> ObservationRow | None:
        return (
            self.session.query(ObservationRow)
            .filter_by(run_id=_key(run_id), sequence=sequence)
            .one_or_none()
        )

    def list_for_run(self, run_id: UUID | str) -> list[ObservationRow]:
        return (
            self.session.query(ObservationRow)
            .filter_by(run_id=_key(run_id))
            .order_by(ObservationRow.sequence.asc())
            .all()
        )

    def save(
        self,
        *,
        run_id: UUID | str,
        sequence: int,
        environment_id: UUID | str,
        environment_version: int,
        source_tool: str,
        payload: dict[str, Any],
        tool_call_id: UUID | str | None = None,
    ) -> ObservationRow:
        if tool_call_id is not None:
            linked = (
                self.session.query(RunJournalRow)
                .filter_by(
                    run_id=_key(run_id),
                    attempt_id=_key(tool_call_id),
                    kind="tool",
                    phase="started",
                    name=source_tool,
                )
                .one_or_none()
            )
            if linked is None:
                raise PersistenceError("observation tool attempt does not belong to this run")
        existing = self.get(run_id, sequence)
        if existing is not None:
            if (
                existing.environment_id != _key(environment_id)
                or existing.source_tool != source_tool
                or existing.payload_json != payload
            ):
                raise StaleWriteError(
                    f"observation {run_id}/{sequence} conflicts with the stored payload"
                )
            return existing
        row = ObservationRow(
            id=str(uuid4()),
            run_id=_key(run_id),
            sequence=sequence,
            environment_id=_key(environment_id),
            environment_version=environment_version,
            source_tool=source_tool,
            tool_call_id=str(tool_call_id) if tool_call_id else None,
            payload_json=deepcopy(payload),
            observed_at=datetime.now(UTC),
        )
        self.session.add(row)
        self.session.flush()
        return row


class HypothesisLineageError(PersistenceError):
    """A typed update cannot be bound to this run's observed history."""


class HypothesisLineageRepository:
    def __init__(self, session: Session):
        self.session = session

    def observations(self, run_id) -> list[dict[str, Any]]:
        return [
            {"observation_id": row.id, "tool_name": row.source_tool, "payload": row.payload_json}
            for row in ObservationRepository(self.session).list_for_run(run_id)
        ]

    def versions(self, run_id, hypothesis_id=None) -> list[HypothesisSnapshot]:
        query = self.session.query(HypothesisVersionRow).filter_by(run_id=str(run_id))
        if hypothesis_id is not None:
            query = query.filter_by(hypothesis_id=str(hypothesis_id))
        rows = query.order_by(
            HypothesisVersionRow.hypothesis_id, HypothesisVersionRow.version
        ).all()
        return [self.snapshot(row) for row in rows]

    @staticmethod
    def snapshot(row: HypothesisVersionRow) -> HypothesisSnapshot:
        try:
            item = HypothesisSnapshot.model_validate(row.snapshot_json)
        except ValidationError:
            raise HypothesisLineageError("stored hypothesis snapshot is invalid") from None
        if (
            str(item.version_id),
            str(item.id),
            str(item.run_id),
            item.version,
            str(item.decision_id),
        ) != (row.id, row.hypothesis_id, row.run_id, row.version, row.decision_id):
            raise HypothesisLineageError("stored hypothesis snapshot ownership is invalid")
        return item

    def heads(self, run_id) -> list[HypothesisSnapshot]:
        heads = {}
        for item in self.versions(run_id):
            if item.id not in heads or item.version > heads[item.id].version:
                heads[item.id] = item
        return sorted(heads.values(), key=lambda item: str(item.id))

    def decision(self, run_id, logical_step) -> HypothesisDecisionRow | None:
        return (
            self.session.query(HypothesisDecisionRow)
            .filter_by(run_id=str(run_id), logical_step=logical_step)
            .one_or_none()
        )

    def replay(self, decision: HypothesisDecisionRow) -> dict[str, Any]:
        def read(ids):
            values = []
            for identity in ids:
                row = self.session.get(HypothesisVersionRow, identity)
                if row is None or row.run_id != decision.run_id:
                    raise HypothesisLineageError("hypothesis snapshot references invalid history")
                values.append(self.snapshot(row).model_dump(mode="json"))
            return values

        return {
            "hypotheses": read(decision.snapshot_version_ids),
            "hypothesis_changes": read(decision.changed_version_ids),
            "needs_more_evidence": decision.needs_more_evidence,
            "diagnosis": decision.diagnosis_json,
        }

    @staticmethod
    def plan(
        *,
        run_id: UUID,
        logical_step: int,
        decision_id: UUID,
        updates: HypothesisUpdate,
        previous: list[HypothesisSnapshot],
        observations: list[dict],
        now: datetime,
    ):
        """Pure bounded update validation shared by SQLite and offline in-memory mode."""
        heads = {item.id: item for item in previous}
        facts = {item.id: item for item in observed_facts(observations)}
        known_obs = {UUID(item["observation_id"]) for item in observations}
        by_tool = {item["tool_name"]: UUID(item["observation_id"]) for item in observations}
        changed = []
        seen = set()
        for index, proposal in enumerate(updates.hypotheses):
            old = heads.get(proposal.id) if proposal.id is not None else None
            if proposal.id is not None and old is None:
                raise HypothesisLineageError("hypothesis does not belong to this run")
            if old is not None and (
                proposal.version != old.version or proposal.previous_decision_id != old.decision_id
            ):
                raise HypothesisLineageError("hypothesis prior version or decision is stale")
            if old is None and (
                proposal.version is not None or proposal.previous_decision_id is not None
            ):
                raise HypothesisLineageError("new hypothesis cannot reference prior history")
            if old is not None and old.status == "retracted" and proposal.status != "retracted":
                raise HypothesisLineageError("retracted hypothesis cannot be silently reactivated")
            if old is None and proposal.status == "retracted":
                raise HypothesisLineageError("only an existing hypothesis can be retracted")
            identity = old.id if old else uuid5(run_id, f"hypothesis:{logical_step}:{index}")
            if identity in seen:
                raise HypothesisLineageError("one decision cannot update a hypothesis twice")
            seen.add(identity)
            if any(ref not in by_tool for ref in proposal.evidence_refs):
                raise HypothesisLineageError("hypothesis references unknown evidence")
            support = sorted(
                set(
                    proposal.supporting_observation_ids
                    or [by_tool[ref] for ref in proposal.evidence_refs]
                ),
                key=str,
            )
            against = sorted(set(proposal.contradicting_observation_ids), key=str)
            if not set(support + against) <= known_obs or set(support) & set(against):
                raise HypothesisLineageError(
                    "hypothesis evidence has invalid ownership or polarity"
                )
            if (proposal.status == "supported" and not support) or (
                proposal.status == "contradicted" and not against
            ):
                raise HypothesisLineageError("hypothesis status requires matching evidence")
            if proposal.claim is None:
                if old is None:
                    raise HypothesisLineageError(
                        "new hypothesis requires an observed structured claim"
                    )
                claim = old.claim
            else:
                cause = facts.get(proposal.claim.cause_fact_id)
                effect = facts.get(proposal.claim.effect_fact_id)
                if cause is None or effect is None:
                    raise HypothesisLineageError(
                        "hypothesis claim references an unknown observed fact"
                    )
                if cause.id == effect.id:
                    raise HypothesisLineageError("hypothesis cannot explain a fact with itself")
                if not {cause.observation_id, effect.observation_id} <= set(support + against):
                    raise HypothesisLineageError(
                        "hypothesis claim facts require explicit evidence links"
                    )
                claim = HypothesisClaimSnapshot(
                    cause=cause, effect=effect, relation=proposal.claim.relation
                )
            # Retained claims still require their original observed evidence.
            if not {claim.cause.observation_id, claim.effect.observation_id} <= set(
                support + against
            ):
                raise HypothesisLineageError("hypothesis evidence cannot silently drop its claim")
            number = old.version + 1 if old else 1
            snapshot = HypothesisSnapshot(
                id=identity,
                run_id=run_id,
                version_id=uuid5(identity, f"version:{number}"),
                version=number,
                category=proposal.category,
                confidence=proposal.confidence,
                status=proposal.status,
                supporting_observation_ids=support,
                contradicting_observation_ids=against,
                decision_id=decision_id,
                previous_version_id=old.version_id if old else None,
                created_at=old.created_at if old else now,
                updated_at=now,
                claim=claim,
            )
            heads[identity] = snapshot
            changed.append(snapshot)
        if len(heads) > 10:
            raise HypothesisLineageError("run hypothesis limit is exhausted")
        diagnosis = None
        if updates.diagnosis is not None:
            if any(ref not in by_tool for ref in updates.diagnosis.evidence_refs):
                raise HypothesisLineageError("diagnosis references unknown evidence")
            selected = next(
                (item for item in changed if item.category == updates.diagnosis.category), None
            )
            selected = selected or next(
                (item for item in heads.values() if item.category == updates.diagnosis.category),
                None,
            )
            if selected is None:
                raise HypothesisLineageError(
                    "diagnosis requires a matching observed hypothesis claim"
                )
            diagnosis = {
                "category": updates.diagnosis.category,
                "summary": selected.summary,
                "evidence_refs": list(updates.diagnosis.evidence_refs),
            }
        return sorted(heads.values(), key=lambda item: str(item.id)), changed, diagnosis

    def validate_output(self, run_id: UUID, logical_step: int, updates: HypothesisUpdate) -> None:
        self.plan(
            run_id=run_id,
            logical_step=logical_step,
            decision_id=uuid5(run_id, "validation-only"),
            updates=updates,
            previous=self.heads(run_id),
            observations=self.observations(run_id),
            now=datetime.now(UTC),
        )

    def apply(
        self, *, run_id: UUID, logical_step: int, decision_id: UUID, updates: HypothesisUpdate
    ):
        # Runtime renews its lease in this same transaction before calling apply.
        AgentRunRepository(self.session).require_for_update(run_id)
        decision_event = self.session.get(RunJournalRow, str(decision_id))
        if decision_event is None or (
            decision_event.run_id,
            decision_event.kind,
            decision_event.name,
            decision_event.phase,
        ) != (str(run_id), "llm", "hypothesis_update", "completed"):
            raise HypothesisLineageError("model decision does not belong to this run")
        if decision_event.logical_step != logical_step:
            raise HypothesisLineageError("model decision belongs to another logical step")
        fingerprint = stable_parameter_hash(
            {
                "hypotheses": [
                    item.model_dump(mode="json", exclude={"summary"}) for item in updates.hypotheses
                ],
                "needs_more_evidence": updates.needs_more_evidence,
                "diagnosis": updates.diagnosis.model_dump(mode="json", exclude={"summary"})
                if updates.diagnosis
                else None,
            }
        )
        existing = self.decision(run_id, logical_step)
        if existing is not None:
            if existing.id != str(decision_id) or existing.request_fingerprint != fingerprint:
                raise HypothesisLineageError(
                    "hypothesis decision conflicts with its committed update"
                )
            return self.replay(existing)
        now = datetime.now(UTC)
        snapshots, changed, diagnosis = self.plan(
            run_id=run_id,
            logical_step=logical_step,
            decision_id=decision_id,
            updates=updates,
            previous=self.heads(run_id),
            observations=self.observations(run_id),
            now=now,
        )
        decision = HypothesisDecisionRow(
            id=str(decision_id),
            run_id=str(run_id),
            logical_step=logical_step,
            request_fingerprint=fingerprint,
            snapshot_version_ids=[str(item.version_id) for item in snapshots],
            changed_version_ids=[str(item.version_id) for item in changed],
            needs_more_evidence=updates.needs_more_evidence,
            diagnosis_json=diagnosis,
            created_at=now,
        )
        self.session.add(decision)
        self.session.flush()
        for item in changed:
            self.session.add(
                HypothesisVersionRow(
                    id=str(item.version_id),
                    hypothesis_id=str(item.id),
                    run_id=str(run_id),
                    version=item.version,
                    decision_id=str(decision_id),
                    snapshot_json=item.model_dump(mode="json"),
                    created_at=now,
                )
            )
        self.session.flush()
        # Local import avoids a module cycle; the existing writer is reused,
        # with versions and their journal projection in this one transaction.
        from agentops.journal import RunJournalRepository

        journal = RunJournalRepository(self.session)
        entry = journal.start(
            run_id=run_id,
            kind="hypothesis",
            name="hypothesis_update",
            logical_step=logical_step,
            attempt_id=uuid5(run_id, f"hypothesis-lineage:{logical_step}"),
            facts={"hypothesis_changes": [item.model_dump(mode="json") for item in changed]},
        )
        journal.finish(entry, phase="completed", duration_ms=0)
        return self.replay(decision)


class EvaluationJobRepository:
    """Conditional admission finalization; no automatic replay of unknown work."""

    def __init__(self, session: Session) -> None:
        self.session = session

    def get_request(self, actor_id: str, key: str) -> EvaluationJobRow | None:
        return (
            self.session.query(EvaluationJobRow)
            .filter_by(actor_id=actor_id, idempotency_key=key)
            .one_or_none()
        )

    def get(self, job_id: UUID | str) -> EvaluationJobRow | None:
        return self.session.get(EvaluationJobRow, _key(job_id))

    @staticmethod
    def is_capacity_conflict(error: IntegrityError) -> bool:
        """Classify the actual admission collision, even after slot release."""
        diagnostic = getattr(error.orig, "diag", None)
        if getattr(diagnostic, "constraint_name", None) == "uq_evaluation_jobs_active_slot":
            return True
        return str(error.orig) == "UNIQUE constraint failed: evaluation_jobs.active_slot"

    def claim(
        self,
        *,
        actor_id: str,
        key: str,
        fingerprint: str,
        mode: str,
        seed: int,
        lifetime_seconds: float = 600.0,
    ) -> EvaluationJobRow:
        if lifetime_seconds <= 0 or lifetime_seconds > 600:
            raise ValueError("evaluation admission deadline must be within 600 seconds")
        now = datetime.now(UTC)
        row = EvaluationJobRow(
            id=str(uuid4()),
            actor_id=actor_id,
            idempotency_key=key,
            request_fingerprint=fingerprint,
            mode=mode,
            seed=seed,
            owner_token=str(uuid4()),
            status="RUNNING",
            active_slot=1,
            created_at=now,
            expires_at=now + timedelta(seconds=lifetime_seconds),
        )
        self.session.add(row)
        self.session.flush()
        return row

    def quarantine_expired(self, job_id: UUID | str) -> None:
        self.session.execute(
            update(EvaluationJobRow)
            .where(
                EvaluationJobRow.id == _key(job_id),
                EvaluationJobRow.status == "RUNNING",
                EvaluationJobRow.expires_at <= datetime.now(UTC),
            )
            .values(status="QUARANTINED", error_code="EvaluationInterrupted")
            .execution_options(synchronize_session=False)
        )
        self.session.flush()
        self.session.expire_all()

    def finish(
        self,
        job_id: UUID | str,
        *,
        owner_token: str,
        succeeded: bool,
        error_code: str | None = None,
    ) -> bool:
        now = datetime.now(UTC)
        conditions = [
            EvaluationJobRow.id == _key(job_id),
            EvaluationJobRow.owner_token == owner_token,
        ]
        if succeeded:
            conditions.extend(
                (EvaluationJobRow.status == "RUNNING", EvaluationJobRow.expires_at > now)
            )
        else:
            conditions.append(EvaluationJobRow.status.in_(("RUNNING", "QUARANTINED")))
        result = self.session.execute(
            update(EvaluationJobRow)
            .where(*conditions)
            .values(
                status="SUCCEEDED" if succeeded else "FAILED",
                active_slot=None,
                finished_at=now,
                error_code=error_code,
                evaluation_run_id=_key(job_id) if succeeded else None,
            )
            .execution_options(synchronize_session=False)
        )
        self.session.flush()
        return result.rowcount == 1


class SimulationEnvironmentRepository:
    """Repository for opaque simulator snapshots and their mutation receipts."""

    def __init__(self, session: Session) -> None:
        self.session = session

    def get(self, environment_id: UUID | str) -> SimulationEnvironmentRow | None:
        return self.session.get(SimulationEnvironmentRow, _key(environment_id))

    def require(self, environment_id: UUID | str) -> SimulationEnvironmentRow:
        row = self.get(environment_id)
        if row is None:
            raise EntityNotFound(f"simulation environment not found: {environment_id}")
        return row

    def require_for_update(self, environment_id: UUID | str) -> SimulationEnvironmentRow:
        """Load one environment while serializing concurrent durable mutations."""

        row = (
            self.session.query(SimulationEnvironmentRow)
            .filter(SimulationEnvironmentRow.id == _key(environment_id))
            .with_for_update()
            .one_or_none()
        )
        if row is None:
            raise EntityNotFound(f"simulation environment not found: {environment_id}")
        # SQLite ignores FOR UPDATE. A no-op UPDATE acquires its write lock so
        # a concurrent cancellation commit linearizes before or after this mutation.
        if self.session.bind is not None and self.session.bind.dialect.name == "sqlite":
            self.session.execute(
                update(SimulationEnvironmentRow)
                .where(SimulationEnvironmentRow.id == _key(environment_id))
                .values(updated_at=SimulationEnvironmentRow.updated_at)
            )
        return row

    def create(
        self,
        *,
        environment_id: UUID | str,
        scenario: str,
        generation: int,
        version: int,
        state: dict[str, Any],
    ) -> SimulationEnvironmentRow:
        now = datetime.now(UTC)
        row = SimulationEnvironmentRow(
            id=_key(environment_id),
            scenario=scenario,
            generation=generation,
            version=version,
            state_json=deepcopy(state),
            created_at=now,
            updated_at=now,
        )
        self.session.add(row)
        self.session.flush()
        return row

    def get_receipt(
        self,
        *,
        environment_id: UUID | str,
        generation: int,
        idempotency_key: str,
    ) -> SimulationReceiptRow | None:
        return (
            self.session.query(SimulationReceiptRow)
            .filter_by(
                environment_id=_key(environment_id),
                generation=generation,
                idempotency_key=idempotency_key,
            )
            .one_or_none()
        )

    def save_receipt(
        self,
        *,
        environment_id: UUID | str,
        generation: int,
        idempotency_key: str,
        tool_name: str,
        parameter_hash: str,
        result: dict[str, Any],
        environment_version: int,
    ) -> SimulationReceiptRow:
        receipt = SimulationReceiptRow(
            id=str(uuid4()),
            environment_id=_key(environment_id),
            generation=generation,
            idempotency_key=idempotency_key,
            tool_name=tool_name,
            parameter_hash=parameter_hash,
            result_json=deepcopy(result),
            environment_version=environment_version,
            created_at=datetime.now(UTC),
        )
        self.session.add(receipt)
        self.session.flush()
        return receipt


def _enable_sqlite_foreign_keys(dbapi_connection: Any, _connection_record: Any) -> None:
    cursor = dbapi_connection.cursor()
    try:
        cursor.execute("PRAGMA foreign_keys=ON")
    finally:
        cursor.close()


class Database:
    """Create engines and transaction-scoped sessions for repositories."""

    def __init__(
        self, url: str = "sqlite+pysqlite:///./data/agentops.db", *, echo: bool = False
    ) -> None:
        engine_kwargs: dict[str, Any] = {"echo": echo, "future": True}
        if url.endswith(":memory:"):
            engine_kwargs["connect_args"] = {"check_same_thread": False}
            engine_kwargs["poolclass"] = StaticPool
        self.engine: Engine = create_engine(url, **engine_kwargs)
        # StaticPool shares one SQLite connection across graph/checkpoint threads.
        # Separate Sessions on that connection must not overlap transactions.
        self._session_lock = RLock() if isinstance(self.engine.pool, StaticPool) else None
        self._session_scope = local()
        if self.engine.dialect.name == "sqlite":
            event.listen(self.engine, "connect", _enable_sqlite_foreign_keys)
        self.session_factory = sessionmaker(
            bind=self.engine,
            autoflush=True,
            expire_on_commit=False,
            class_=Session,
        )

    def create_schema(self) -> None:
        """Create all tables for local smoke tests and development."""

        Base.metadata.create_all(self.engine)

    @contextmanager
    def session(self):
        """Yield a committed transaction, rolling back on any exception."""

        with self._session_lock if self._session_lock is not None else nullcontext():
            current = getattr(self._session_scope, "session", None)
            if self._session_lock is not None and current is not None:
                # Nested work on a shared connection belongs to the outer
                # transaction. Only its owner may commit or close the Session.
                try:
                    yield current
                except BaseException:
                    self._session_scope.rollback_only = True
                    raise
                return
            session = self.session_factory()
            if self._session_lock is not None:
                self._session_scope.session = session
                self._session_scope.rollback_only = False
            try:
                yield session
                if self._session_lock is not None and self._session_scope.rollback_only:
                    raise PersistenceError("nested session failed; outer transaction cannot commit")
                session.commit()
            except BaseException:
                session.rollback()
                raise
            finally:
                session.close()
                if self._session_lock is not None:
                    self._session_scope.session = None
                    self._session_scope.rollback_only = False


def _as_utc(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    return value if value.tzinfo is not None else value.replace(tzinfo=UTC)


def _key(value: UUID | str) -> str:
    return str(value)


class IncidentRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def get(self, incident_id: UUID | str) -> Incident | None:
        row = self.session.get(IncidentRow, _key(incident_id))
        return row.to_domain() if row else None

    def require(self, incident_id: UUID | str) -> Incident:
        incident = self.get(incident_id)
        if incident is None:
            raise EntityNotFound(f"incident not found: {incident_id}")
        return incident

    def save(self, incident: Incident) -> Incident:
        row = self.session.get(IncidentRow, _key(incident.id))
        if row is None:
            incident.updated_at = utc_now()
            self.session.add(IncidentRow.from_domain(incident))
        else:
            if row.version > incident.version:
                raise StaleWriteError(
                    f"incident {incident.id} is stale: stored version {row.version}, "
                    f"received version {incident.version}"
                )
            replacement = IncidentRow.from_domain(incident)
            changed = False
            for name in (
                "title",
                "description",
                "service_name",
                "severity",
                "status",
                "detected_at",
                "resolved_at",
                "created_by",
                "active_run_id",
            ):
                old_value, new_value = getattr(row, name), getattr(replacement, name)
                if isinstance(old_value, datetime):
                    old_value = _as_utc(old_value)
                if isinstance(new_value, datetime):
                    new_value = _as_utc(new_value)
                if old_value != new_value:
                    changed = True
                    break
            row.update_from_domain(incident)
            if changed:
                row.updated_at = utc_now()
            # A stale domain copy must never overwrite this server-owned clock.
            incident.updated_at = _as_utc(row.updated_at)
        try:
            self.session.flush()
        except StaleDataError as exc:
            raise StaleWriteError(f"incident {incident.id} changed during save") from exc
        return incident


class AgentRunRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def get(self, run_id: UUID | str) -> AgentRun | None:
        row = self.session.get(AgentRunRow, _key(run_id))
        return row.to_domain() if row else None

    def require(self, run_id: UUID | str) -> AgentRun:
        run = self.get(run_id)
        if run is None:
            raise EntityNotFound(f"agent run not found: {run_id}")
        return run

    def get_for_update(self, run_id: UUID | str) -> AgentRun | None:
        """Load one run with a database row lock where the backend supports it."""

        row = (
            self.session.query(AgentRunRow)
            .filter(AgentRunRow.id == _key(run_id))
            .populate_existing()
            .with_for_update()
            .one_or_none()
        )
        if (
            row is not None
            and self.session.bind is not None
            and self.session.bind.dialect.name == "sqlite"
        ):
            self.session.execute(
                update(AgentRunRow)
                .where(AgentRunRow.id == _key(run_id))
                .values(cancel_requested_at=AgentRunRow.cancel_requested_at)
                .execution_options(synchronize_session=False)
            )
            self.session.refresh(row)
        return row.to_domain() if row else None

    def require_for_update(self, run_id: UUID | str) -> AgentRun:
        run = self.get_for_update(run_id)
        if run is None:
            raise EntityNotFound(f"agent run not found: {run_id}")
        return run

    def save(self, run: AgentRun) -> AgentRun:
        row = self.session.get(AgentRunRow, _key(run.id))
        status_changed = row is None or row.status != run.status.value
        if row is None:
            self.session.add(AgentRunRow.from_domain(run))
        else:
            # A stale progress object cannot clear or replace established provenance.
            if row.model_configuration_json is not None:
                stored = row.to_domain().model_configuration
                if run.model_configuration is None:
                    run.model_configuration = stored
                elif run.model_configuration != stored:
                    raise StaleWriteError("model configuration binding is immutable")
            # Cancellation is control-owned and monotonic, never cleared by a
            # stale progress object. SQL serializes competing intent writers.
            if run.cancel_requested_at is not None:
                self.session.execute(
                    update(AgentRunRow)
                    .where(
                        AgentRunRow.id == _key(run.id), AgentRunRow.cancel_requested_at.is_(None)
                    )
                    .values(cancel_requested_at=run.cancel_requested_at)
                )
            row.update_from_domain(run)
        self.session.flush()
        if status_changed:
            # Only the incident's linked current run contributes to its activity
            # time. Progress counters, polling and old runs do not change it.
            self.session.execute(
                update(IncidentRow)
                .where(
                    IncidentRow.id == _key(run.incident_id),
                    IncidentRow.active_run_id == _key(run.id),
                )
                .values(updated_at=utc_now())
                .execution_options(synchronize_session="fetch")
            )
        return run


class RunBudgetRepository:
    """Persist immutable run limits and monotonic usage snapshots."""

    def __init__(self, session: Session) -> None:
        self.session = session

    def get_for_run(self, run_id: UUID | str) -> RunBudgetRow | None:
        return self.session.get(RunBudgetRow, _key(run_id))

    def require(self, run_id: UUID | str) -> RunBudgetRow:
        row = self.get_for_run(run_id)
        if row is None:
            raise EntityNotFound(f"run budget not found: {run_id}")
        return row

    def ensure(
        self,
        *,
        run_id: UUID | str,
        max_steps: int,
        max_llm_calls: int,
        max_tool_calls: int,
        max_active_seconds: float,
    ) -> RunBudgetRow:
        row = self.get_for_run(run_id)
        if row is None:
            now = datetime.now(UTC)
            row = RunBudgetRow(
                run_id=_key(run_id),
                max_steps=max_steps,
                max_llm_calls=max_llm_calls,
                max_tool_calls=max_tool_calls,
                max_active_seconds=max_active_seconds,
                created_at=now,
                updated_at=now,
            )
            self.session.add(row)
            self.session.flush()
            return row
        limits = (row.max_steps, row.max_llm_calls, row.max_tool_calls, row.max_active_seconds)
        requested = (max_steps, max_llm_calls, max_tool_calls, max_active_seconds)
        if limits != requested:
            raise PersistenceError("run budget limits do not match the persisted run")
        return row

    def reserve_llm_attempt(self, run_id: UUID | str) -> int:
        # Atomically reserve one provider-boundary attempt before I/O.
        key = _key(run_id)
        now = datetime.now(UTC)
        result = self.session.execute(
            update(RunBudgetRow)
            .where(
                RunBudgetRow.run_id == key,
                RunBudgetRow.llm_calls_used < RunBudgetRow.max_llm_calls,
            )
            .values(
                llm_calls_used=RunBudgetRow.llm_calls_used + 1,
                updated_at=now,
            )
        )
        if result.rowcount != 1:
            row = self.get_for_run(run_id)
            if row is None:
                raise EntityNotFound(f"run budget not found: {run_id}")
            raise PersistenceError("LLM attempt budget exhausted")
        row = self.require(run_id)
        return row.llm_calls_used

    def reserve_tool_attempt(
        self,
        run_id: UUID | str,
        *,
        minimum_used: int = 0,
        required_remaining: int = 1,
    ) -> int:
        """Charge one attempt before I/O, preserving room for required verification."""
        if type(minimum_used) is not int or minimum_used < 0:
            raise ValueError("minimum_used must be a nonnegative integer")
        if type(required_remaining) is not int or required_remaining < 1:
            raise ValueError("required_remaining must be a positive integer")
        used = case(
            (RunBudgetRow.tool_calls_used < minimum_used, minimum_used),
            else_=RunBudgetRow.tool_calls_used,
        )
        result = self.session.execute(
            update(RunBudgetRow)
            .where(
                RunBudgetRow.run_id == _key(run_id),
                used + required_remaining <= RunBudgetRow.max_tool_calls,
            )
            .values(tool_calls_used=used + 1, updated_at=datetime.now(UTC))
        )
        if result.rowcount != 1:
            self.require(run_id)
            raise PersistenceError("tool attempt budget exhausted")
        return self.require(run_id).tool_calls_used

    def update_snapshot(
        self,
        run_id: UUID | str,
        *,
        steps_used: int,
        llm_calls_used: int,
        tool_calls_used: int,
        active_elapsed_seconds: float,
        active_started_at: datetime | None,
    ) -> RunBudgetRow:
        row = self.require(run_id)
        # Checkpoint state can be stale after a crash; usage is monotonic and
        # must never overwrite a durable reservation with a smaller value.
        row.steps_used = max(row.steps_used, steps_used)
        row.llm_calls_used = max(row.llm_calls_used, llm_calls_used)
        row.tool_calls_used = max(row.tool_calls_used, tool_calls_used)
        row.active_elapsed_seconds = max(row.active_elapsed_seconds, active_elapsed_seconds)
        row.active_started_at = active_started_at
        row.updated_at = datetime.now(UTC)
        self.session.flush()
        return row

    def recover_active_segment(self, run_id: UUID | str, *, now: datetime) -> RunBudgetRow:
        row = self.require(run_id)
        if row.active_started_at is not None:
            started_at = _as_utc(row.active_started_at)
            if started_at is None:
                raise PersistenceError("run budget active start time is invalid")
            row.active_elapsed_seconds += max(0.0, (now - started_at).total_seconds())
            row.active_started_at = None
            row.updated_at = now
            self.session.flush()
        return row


class EnvironmentReservationRepository:
    """Claim and release one environment for one run."""

    def __init__(self, session: Session) -> None:
        self.session = session

    def get_for_environment(
        self,
        environment_id: UUID | str,
    ) -> EnvironmentReservationRow | None:
        return self.session.get(EnvironmentReservationRow, _key(environment_id))

    def get_for_run(self, run_id: UUID | str) -> EnvironmentReservationRow | None:
        return (
            self.session.query(EnvironmentReservationRow)
            .filter_by(run_id=_key(run_id))
            .one_or_none()
        )

    def require_for_environment(
        self,
        environment_id: UUID | str,
    ) -> EnvironmentReservationRow:
        row = self.get_for_environment(environment_id)
        if row is None:
            raise EntityNotFound(f"environment reservation not found: {environment_id}")
        return row

    @staticmethod
    def _validate_generation(generation: int) -> None:
        if generation < 1:
            raise PersistenceError("environment generation must be positive")

    def acquire(
        self,
        environment_id: UUID | str,
        run_id: UUID | str,
        *,
        generation: int,
        now: datetime,
    ) -> EnvironmentReservationRow:
        self._validate_generation(generation)
        environment_key = _key(environment_id)
        run_key = _key(run_id)
        row = (
            self.session.query(EnvironmentReservationRow)
            .filter_by(environment_id=environment_key)
            .with_for_update()
            .one_or_none()
        )
        existing_for_run = (
            self.session.query(EnvironmentReservationRow)
            .filter_by(run_id=run_key)
            .with_for_update()
            .one_or_none()
        )
        if existing_for_run is not None and existing_for_run.environment_id != environment_key:
            raise EnvironmentReservationHeldError(
                f"run {run_id} already reserves another environment"
            )
        if row is None:
            row = EnvironmentReservationRow(
                environment_id=environment_key,
                run_id=run_key,
                generation=generation,
                acquired_at=now,
                released_at=None,
            )
            self.session.add(row)
            self.session.flush()
            return row
        if row.released_at is None:
            if row.run_id != run_key:
                raise EnvironmentReservationHeldError(
                    f"environment {environment_id} is reserved by another run"
                )
            if row.generation != generation:
                raise EnvironmentReservationFencedError(
                    f"environment {environment_id} reservation generation is stale"
                )
            return row
        if row.run_id != run_key and existing_for_run is not None:
            raise EnvironmentReservationHeldError(
                f"run {run_id} already reserves another environment"
            )
        row.run_id = run_key
        row.generation = generation
        row.acquired_at = now
        row.released_at = None
        self.session.flush()
        return row

    def assert_owner(
        self,
        environment_id: UUID | str,
        run_id: UUID | str,
        *,
        generation: int,
    ) -> EnvironmentReservationRow:
        row = self.require_for_environment(environment_id)
        if (
            row.released_at is not None
            or row.run_id != _key(run_id)
            or row.generation != generation
        ):
            raise EnvironmentReservationFencedError(
                f"environment {environment_id} reservation is stale or owned by another run"
            )
        return row

    def release(
        self,
        environment_id: UUID | str,
        run_id: UUID | str,
        *,
        generation: int,
        now: datetime,
    ) -> EnvironmentReservationRow:
        row = (
            self.session.query(EnvironmentReservationRow)
            .filter_by(environment_id=_key(environment_id))
            .with_for_update()
            .one_or_none()
        )
        if row is None:
            raise EntityNotFound(f"environment reservation not found: {environment_id}")
        if row.released_at is not None:
            if row.run_id == _key(run_id) and row.generation == generation:
                return row
            raise EnvironmentReservationFencedError(
                f"environment {environment_id} reservation is already released"
            )
        self.assert_owner(environment_id, run_id, generation=generation)
        row.released_at = now
        self.session.flush()
        return row


class RunLeaseRepository:
    """Acquire, renew, and fence one worker lease per run."""

    def __init__(self, session: Session) -> None:
        self.session = session

    def get_for_run(self, run_id: UUID | str) -> RunLeaseRow | None:
        return self.session.get(RunLeaseRow, _key(run_id))

    def require(self, run_id: UUID | str) -> RunLeaseRow:
        row = self.get_for_run(run_id)
        if row is None:
            raise EntityNotFound(f"run lease not found: {run_id}")
        return row

    @staticmethod
    def _expiry(now: datetime, ttl_seconds: float) -> datetime:
        if ttl_seconds <= 0:
            raise PersistenceError("lease ttl must be positive")
        return now + timedelta(seconds=ttl_seconds)

    def acquire(
        self,
        run_id: UUID | str,
        *,
        owner_id: str,
        now: datetime,
        ttl_seconds: float,
    ) -> RunLeaseRow:
        if not owner_id:
            raise PersistenceError("lease owner_id is required")
        expires_at = self._expiry(now, ttl_seconds)
        row = (
            self.session.query(RunLeaseRow)
            .filter_by(run_id=_key(run_id))
            .with_for_update()
            .one_or_none()
        )
        if row is None:
            row = RunLeaseRow(
                run_id=_key(run_id),
                owner_id=owner_id,
                generation=1,
                acquired_at=now,
                renewed_at=now,
                expires_at=expires_at,
            )
            try:
                with self.session.begin_nested():
                    self.session.add(row)
                    self.session.flush()
            except IntegrityError as exc:
                raise RunLeaseHeldError(f"run {run_id} lease was acquired concurrently") from exc
            return row
        row_expires_at = _as_utc(row.expires_at)
        if row_expires_at is None:
            raise PersistenceError("run lease expiry is invalid")
        if row.owner_id != owner_id and row_expires_at > now:
            raise RunLeaseHeldError(f"run {run_id} is leased by another live worker")
        if row.owner_id == owner_id and row_expires_at > now:
            return self.renew(
                run_id,
                owner_id=owner_id,
                generation=row.generation,
                now=now,
                ttl_seconds=ttl_seconds,
            )
        result = self.session.execute(
            update(RunLeaseRow)
            .where(
                RunLeaseRow.run_id == _key(run_id),
                RunLeaseRow.owner_id == row.owner_id,
                RunLeaseRow.generation == row.generation,
                RunLeaseRow.expires_at == row.expires_at,
            )
            .values(
                owner_id=owner_id,
                generation=RunLeaseRow.generation + 1,
                acquired_at=now,
                renewed_at=now,
                expires_at=expires_at,
            )
            .execution_options(synchronize_session="fetch")
        )
        if result.rowcount != 1:
            raise RunLeaseHeldError(f"run {run_id} lease changed during takeover")
        return row

    def assert_owner(
        self,
        run_id: UUID | str,
        *,
        owner_id: str,
        generation: int,
        now: datetime,
    ) -> RunLeaseRow:
        row = self.require(run_id)
        expires_at = _as_utc(row.expires_at)
        if expires_at is None:
            raise PersistenceError("run lease expiry is invalid")
        if row.owner_id != owner_id or row.generation != generation or expires_at <= now:
            raise RunLeaseFencedError(f"run {run_id} lease is stale or owned by another worker")
        return row

    def renew(
        self,
        run_id: UUID | str,
        *,
        owner_id: str,
        generation: int,
        now: datetime,
        ttl_seconds: float,
    ) -> RunLeaseRow:
        # Renewal is fenced by owner and generation, but does not require the
        # previous expiry to still be in the future. This lets the same live
        # worker recover from a slow database/provider operation. A takeover
        # always changes owner or generation, so an old worker cannot revive a
        # lease after another worker has acquired it.
        result = self.session.execute(
            update(RunLeaseRow)
            .where(
                RunLeaseRow.run_id == _key(run_id),
                RunLeaseRow.owner_id == owner_id,
                RunLeaseRow.generation == generation,
            )
            .values(renewed_at=now, expires_at=self._expiry(now, ttl_seconds))
            .execution_options(synchronize_session="fetch")
        )
        if result.rowcount != 1:
            raise RunLeaseFencedError(f"run {run_id} lease is stale or owned by another worker")
        return self.require(run_id)

    def release(
        self,
        run_id: UUID | str,
        *,
        owner_id: str,
        generation: int,
        now: datetime,
    ) -> RunLeaseRow:
        result = self.session.execute(
            update(RunLeaseRow)
            .where(
                RunLeaseRow.run_id == _key(run_id),
                RunLeaseRow.owner_id == owner_id,
                RunLeaseRow.generation == generation,
                RunLeaseRow.expires_at > now,
            )
            .values(renewed_at=now, expires_at=now)
            .execution_options(synchronize_session="fetch")
        )
        if result.rowcount != 1:
            raise RunLeaseFencedError(f"run {run_id} lease is stale or owned by another worker")
        return self.require(run_id)


class ActionRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def get(self, action_id: UUID | str) -> Action | None:
        row = self.session.get(ActionRow, _key(action_id))
        return row.to_domain() if row else None

    def list_for_run(self, run_id: UUID | str) -> list[Action]:
        rows = self.session.query(ActionRow).filter_by(run_id=_key(run_id)).limit(2).all()
        return [row.to_domain() for row in rows]

    def get_by_idempotency_key(self, idempotency_key: str) -> Action | None:
        row = self.session.query(ActionRow).filter_by(idempotency_key=idempotency_key).one_or_none()
        return row.to_domain() if row else None

    def require(self, action_id: UUID | str) -> Action:
        action = self.get(action_id)
        if action is None:
            raise EntityNotFound(f"action not found: {action_id}")
        return action

    def save(self, action: Action) -> Action:
        row = self.session.get(ActionRow, _key(action.id))
        if row is None:
            self.session.add(ActionRow.from_domain(action))
        else:
            row.update_from_domain(action)
        self.session.flush()
        return action


class ApprovalRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def _to_domain(self, row: ApprovalRequestRow) -> ApprovalRequest:
        action = self.session.get(ActionRow, row.action_id)
        if action is None:
            raise EntityNotFound(f"action not found: {row.action_id}")
        approval = row.to_domain(tool_name=action.tool_name)
        decision = ApprovalDecisionRepository(self.session).get_for_approval(row.id)
        if decision is None:
            return approval
        if decision.action_id != row.action_id or decision.run_id != row.run_id:
            raise ApprovalDecisionConflictError("immutable decision scope conflicts with approval")
        values = approval.model_dump()
        values.update(
            status=decision.status,
            decided_at=_as_utc(decision.decided_at),
            decided_by=decision.decided_by,
            approved_parameters=deepcopy(decision.approved_parameters_json),
            approved_parameter_hash=decision.approved_parameter_hash,
            decision_reason=decision.decision_reason,
        )
        return ApprovalRequest.model_validate(values)

    def repair_decision_projection(self, approval_id: UUID | str) -> ApprovalRequest:
        """Repair only mutable decision fields from their immutable authority."""
        self.require(approval_id)
        decision = ApprovalDecisionRepository(self.session).get_for_approval(approval_id)
        if decision is None:
            raise ApprovalDecisionConflictError("no immutable decision exists to repair")
        self.session.execute(
            update(ApprovalRequestRow)
            .where(
                ApprovalRequestRow.id == _key(approval_id),
                ApprovalRequestRow.action_id == decision.action_id,
                ApprovalRequestRow.run_id == decision.run_id,
            )
            .values(
                status=decision.status,
                decided_at=decision.decided_at,
                decided_by=decision.decided_by,
                approved_parameters_json=deepcopy(decision.approved_parameters_json),
                approved_parameter_hash=decision.approved_parameter_hash,
                decision_reason=decision.decision_reason,
            )
            .execution_options(synchronize_session=False)
        )
        self.session.expire_all()
        return self.require(approval_id)

    def get(self, approval_id: UUID | str) -> ApprovalRequest | None:
        row = self.session.get(ApprovalRequestRow, _key(approval_id))
        return self._to_domain(row) if row else None

    def list_for_run(self, run_id: UUID | str) -> list[ApprovalRequest]:
        rows = self.session.query(ApprovalRequestRow).filter_by(run_id=_key(run_id)).limit(2).all()
        return [self._to_domain(row) for row in rows]

    def get_for_action(self, action_id: UUID | str) -> ApprovalRequest | None:
        row = (
            self.session.query(ApprovalRequestRow)
            .filter_by(action_id=_key(action_id))
            .order_by(ApprovalRequestRow.requested_at.desc())
            .first()
        )
        return self._to_domain(row) if row else None

    def get_for_run(self, run_id: UUID | str) -> ApprovalRequest | None:
        row = (
            self.session.query(ApprovalRequestRow)
            .filter_by(run_id=_key(run_id))
            .order_by(ApprovalRequestRow.requested_at.desc())
            .first()
        )
        return self._to_domain(row) if row else None

    def require(self, approval_id: UUID | str) -> ApprovalRequest:
        approval = self.get(approval_id)
        if approval is None:
            raise EntityNotFound(f"approval request not found: {approval_id}")
        return approval

    def save(self, approval: ApprovalRequest) -> ApprovalRequest:
        row = self.session.get(ApprovalRequestRow, _key(approval.id))
        if row is None:
            self.session.add(ApprovalRequestRow.from_domain(approval))
        else:
            if (
                row.action_id != _key(approval.action_id)
                or row.run_id != _key(approval.run_id)
                or row.original_parameter_hash != approval.original_parameter_hash
            ):
                raise ApprovalDecisionConflictError("approval scope is immutable")
            replacement = ApprovalRequestRow.from_domain(approval)
            values = {
                column.name: getattr(replacement, column.name)
                for column in ApprovalRequestRow.__table__.columns
                if column.name != "id"
            }
            changed = self.session.execute(
                update(ApprovalRequestRow)
                .where(
                    ApprovalRequestRow.id == _key(approval.id),
                    ApprovalRequestRow.status == ApprovalStatus.PENDING.value,
                    ~exists().where(ApprovalDecisionRow.approval_id == ApprovalRequestRow.id),
                )
                .values(**values)
                .execution_options(synchronize_session=False)
            )
            self.session.refresh(row)
            if changed.rowcount == 0:
                stored = self._to_domain(row)
                if approval.status is ApprovalStatus.PENDING:
                    return stored
                if stored.model_dump() != approval.model_dump():
                    raise ApprovalDecisionConflictError(
                        "approval already has a different committed decision"
                    )
                return stored
        self.session.flush()
        return approval


class ApprovalDecisionRepository:
    """Read committed immutable approval decisions for recovery and replay."""

    def __init__(self, session: Session) -> None:
        self.session = session

    def get_for_approval(self, approval_id: UUID | str) -> ApprovalDecisionRow | None:
        return (
            self.session.query(ApprovalDecisionRow)
            .filter_by(approval_id=_key(approval_id))
            .one_or_none()
        )

    def require(self, decision_id: UUID | str) -> ApprovalDecisionRow:
        row = self.session.get(ApprovalDecisionRow, _key(decision_id))
        if row is None:
            raise EntityNotFound(f"approval decision not found: {decision_id}")
        return row


class OutboxRepository:
    """Claim, publish, retry, and recover durable outbox events."""

    def __init__(self, session: Session) -> None:
        self.session = session

    def _locked_row(self, event_id: UUID | str) -> OutboxRow | None:
        return (
            self.session.query(OutboxRow)
            .filter_by(id=_key(event_id))
            .populate_existing()
            .with_for_update()
            .one_or_none()
        )

    def claim_due(
        self,
        *,
        owner_id: str,
        now: datetime,
        lease_seconds: float = 30.0,
        limit: int = 100,
        max_attempts: int = 5,
    ) -> list[OutboxClaim]:
        if not owner_id:
            raise PersistenceError("outbox owner_id is required")
        if lease_seconds <= 0:
            raise PersistenceError("outbox lease_seconds must be positive")
        if limit < 1:
            raise PersistenceError("outbox limit must be positive")
        if max_attempts < 1:
            raise PersistenceError("outbox max_attempts must be positive")
        now = _as_utc(now) or datetime.now(UTC)
        stale_at = now - timedelta(seconds=lease_seconds)
        rows = (
            self.session.query(OutboxRow)
            .filter(
                OutboxRow.published_at.is_(None),
                OutboxRow.dispatch_attempts < max_attempts,
                or_(OutboxRow.available_at.is_(None), OutboxRow.available_at <= now),
                or_(OutboxRow.claimed_at.is_(None), OutboxRow.claimed_at <= stale_at),
            )
            .order_by(OutboxRow.created_at.asc(), OutboxRow.id.asc())
            .limit(limit)
            .populate_existing()
            .with_for_update(skip_locked=True)
            .all()
        )
        claims: list[OutboxClaim] = []
        for row in rows:
            row.claimed_by = owner_id
            row.claimed_at = now
            row.dispatch_attempts += 1
            claims.append(
                OutboxClaim(
                    event_id=UUID(row.id),
                    event_key=row.event_key,
                    event_type=row.event_type,
                    aggregate_type=row.aggregate_type,
                    aggregate_id=row.aggregate_id,
                    payload=deepcopy(row.payload_json),
                    dispatch_attempts=row.dispatch_attempts,
                )
            )
        self.session.flush()
        return claims

    def claim_unprocessed(
        self,
        *,
        owner_id: str,
        now: datetime,
        lease_seconds: float = 30.0,
        limit: int = 100,
        max_attempts: int = 5,
    ) -> list[OutboxClaim]:
        if not owner_id:
            raise PersistenceError("outbox owner_id is required")
        if lease_seconds <= 0:
            raise PersistenceError("outbox lease_seconds must be positive")
        if limit < 1:
            raise PersistenceError("outbox limit must be positive")
        if max_attempts < 1:
            raise PersistenceError("outbox max_attempts must be positive")
        now = _as_utc(now) or datetime.now(UTC)
        stale_at = now - timedelta(seconds=lease_seconds)
        rows = (
            self.session.query(OutboxRow)
            .filter(
                OutboxRow.published_at.is_not(None),
                OutboxRow.processed_at.is_(None),
                OutboxRow.dispatch_attempts < max_attempts,
                or_(OutboxRow.available_at.is_(None), OutboxRow.available_at <= now),
                or_(OutboxRow.claimed_at.is_(None), OutboxRow.claimed_at <= stale_at),
                or_(
                    and_(
                        OutboxRow.processing_owner.is_(None),
                        OutboxRow.processing_started_at.is_(None),
                        OutboxRow.processing_expires_at.is_(None),
                    ),
                    OutboxRow.processing_expires_at <= now,
                ),
                or_(
                    OutboxRow.processing_available_at.is_(None),
                    OutboxRow.processing_available_at <= now,
                ),
            )
            .order_by(OutboxRow.created_at.asc(), OutboxRow.id.asc())
            .limit(limit)
            .populate_existing()
            .with_for_update(skip_locked=True)
            .all()
        )
        claims: list[OutboxClaim] = []
        for row in rows:
            row.claimed_by = owner_id
            row.claimed_at = now
            row.dispatch_attempts += 1
            claims.append(
                OutboxClaim(
                    event_id=UUID(row.id),
                    event_key=row.event_key,
                    event_type=row.event_type,
                    aggregate_type=row.aggregate_type,
                    aggregate_id=row.aggregate_id,
                    payload=deepcopy(row.payload_json),
                    dispatch_attempts=row.dispatch_attempts,
                )
            )
        self.session.flush()
        return claims

    def mark_published(
        self,
        event_id: UUID | str,
        *,
        owner_id: str,
        dispatch_attempt: int,
        now: datetime,
        recovery_delay_seconds: float = 60.0,
    ) -> OutboxRow:
        if recovery_delay_seconds <= 0:
            raise PersistenceError("outbox recovery delay must be positive")
        now = _as_utc(now) or datetime.now(UTC)
        row = self._locked_row(event_id)
        if row is None:
            raise EntityNotFound(f"outbox event not found: {event_id}")
        if row.dispatch_attempts != dispatch_attempt:
            raise OutboxClaimError("dispatch attempt has been replaced")
        if row.published_at is not None:
            if row.claimed_by == owner_id:
                row.claimed_by = None
                row.claimed_at = None
                row.available_at = now + timedelta(seconds=recovery_delay_seconds)
                self.session.flush()
            return row
        if row.claimed_by != owner_id:
            raise OutboxClaimError(f"outbox event {event_id} is not owned by {owner_id}")
        row.published_at = now
        row.claimed_by = None
        row.claimed_at = None
        row.available_at = now + timedelta(seconds=recovery_delay_seconds)
        self.session.flush()
        return row

    def mark_failed(
        self,
        event_id: UUID | str,
        *,
        owner_id: str,
        dispatch_attempt: int,
        now: datetime,
        error_code: str,
        error_message: str,
        retry_delay_seconds: float,
        max_attempts: int = 5,
    ) -> OutboxRow:
        now = _as_utc(now) or datetime.now(UTC)
        row = self._locked_row(event_id)
        if row is None:
            raise EntityNotFound(f"outbox event not found: {event_id}")
        if row.dispatch_attempts != dispatch_attempt:
            raise OutboxClaimError("dispatch attempt has been replaced")
        if row.published_at is not None:
            return row
        if row.claimed_by != owner_id:
            raise OutboxClaimError(f"outbox event {event_id} is not owned by {owner_id}")
        row.last_error_code = error_code[:100]
        row.last_error_message = error_message[:2000]
        row.claimed_by = None
        row.claimed_at = None
        if row.dispatch_attempts >= max_attempts:
            row.available_at = None
            row.last_error_code = "RetryExhausted"
        else:
            row.available_at = now + timedelta(seconds=max(0.0, retry_delay_seconds))
        self.session.flush()
        return row

    def claim_command(
        self,
        event_id: UUID | str,
        *,
        event_key: str,
        owner_id: str,
        now: datetime,
        lease_seconds: float = 60.0,
        max_attempts: int = 5,
    ) -> CommandClaimResult:
        if not owner_id:
            raise PersistenceError("command owner_id is required")
        if not event_key:
            raise PersistenceError("command event_key is required")
        if lease_seconds <= 0:
            raise PersistenceError("command lease_seconds must be positive")
        if max_attempts < 1:
            raise PersistenceError("command max_attempts must be positive")
        now = _as_utc(now) or datetime.now(UTC)
        row = self._locked_row(event_id)
        if row is None:
            raise EntityNotFound(f"outbox event not found: {event_id}")
        if row.event_key != event_key:
            raise CommandClaimError("command event key does not match outbox event")
        attempts = row.processing_attempts or 0
        if row.processed_at is not None:
            return CommandClaimResult(CommandClaimState.PROCESSED, processing_attempts=attempts)
        if attempts >= max_attempts:
            return CommandClaimResult(CommandClaimState.EXHAUSTED, processing_attempts=attempts)
        available_at = _as_utc(row.processing_available_at)
        if available_at is not None and available_at > now:
            return CommandClaimResult(CommandClaimState.RETRY_WAIT, processing_attempts=attempts)
        started_at = _as_utc(row.processing_started_at)
        expires_at = _as_utc(row.processing_expires_at)
        if (
            started_at is not None or row.processing_owner is not None or expires_at is not None
        ) and (expires_at is None or expires_at > now):
            return CommandClaimResult(CommandClaimState.IN_PROGRESS, processing_attempts=attempts)
        row.processing_owner = owner_id
        row.processing_started_at = now
        row.processing_expires_at = now + timedelta(seconds=lease_seconds)
        row.processing_attempts = attempts + 1
        row.processing_available_at = None
        self.session.flush()
        claim = OutboxClaim(
            event_id=UUID(row.id),
            event_key=row.event_key,
            event_type=row.event_type,
            aggregate_type=row.aggregate_type,
            aggregate_id=row.aggregate_id,
            payload=deepcopy(row.payload_json),
            dispatch_attempts=row.dispatch_attempts or 0,
        )
        return CommandClaimResult(
            CommandClaimState.CLAIMED,
            claim=claim,
            processing_attempts=row.processing_attempts,
        )

    def mark_command_processed(
        self,
        event_id: UUID | str,
        *,
        owner_id: str,
        processing_attempt: int,
        now: datetime,
    ) -> OutboxRow:
        now = _as_utc(now) or datetime.now(UTC)
        row = self._locked_row(event_id)
        if row is None:
            raise EntityNotFound(f"outbox event not found: {event_id}")
        if row.processing_attempts != processing_attempt:
            raise CommandClaimError("command attempt has been replaced")
        if row.processed_at is not None:
            return row
        if row.processing_owner != owner_id:
            raise CommandClaimError(f"command {event_id} is not owned by {owner_id}")
        row.processed_at = now
        row.processing_owner = None
        row.processing_started_at = None
        row.processing_expires_at = None
        row.processing_available_at = None
        row.processing_error_code = None
        row.processing_error_message = None
        self.session.flush()
        return row

    def mark_command_failed(
        self,
        event_id: UUID | str,
        *,
        owner_id: str,
        processing_attempt: int,
        now: datetime,
        error_code: str,
        error_message: str,
        retry_delay_seconds: float,
        max_attempts: int = 5,
    ) -> OutboxRow:
        now = _as_utc(now) or datetime.now(UTC)
        row = self._locked_row(event_id)
        if row is None:
            raise EntityNotFound(f"outbox event not found: {event_id}")
        if row.processing_attempts != processing_attempt:
            raise CommandClaimError("command attempt has been replaced")
        if row.processed_at is not None:
            return row
        if row.processing_owner != owner_id:
            raise CommandClaimError(f"command {event_id} is not owned by {owner_id}")
        row.processing_error_code = error_code[:100]
        row.processing_error_message = error_message[:2000]
        row.processing_owner = None
        row.processing_started_at = None
        row.processing_expires_at = None
        if (row.processing_attempts or 0) >= max_attempts:
            row.processing_available_at = None
            row.processing_error_code = "ProcessingRetryExhausted"
        else:
            row.processing_available_at = now + timedelta(seconds=max(0.0, retry_delay_seconds))
        self.session.flush()
        return row

    def mark_republish_failed(
        self,
        event_id: UUID | str,
        *,
        owner_id: str,
        dispatch_attempt: int,
        now: datetime,
        error_code: str,
        error_message: str,
        retry_delay_seconds: float,
        max_attempts: int = 5,
    ) -> OutboxRow:
        now = _as_utc(now) or datetime.now(UTC)
        row = self._locked_row(event_id)
        if row is None:
            raise EntityNotFound(f"outbox event not found: {event_id}")
        if row.dispatch_attempts != dispatch_attempt:
            raise OutboxClaimError("dispatch attempt has been replaced")
        if row.processed_at is not None:
            return row
        if row.published_at is None:
            raise PersistenceError("republish failure requires a published outbox event")
        if row.claimed_by != owner_id:
            raise OutboxClaimError(f"outbox event {event_id} is not owned by {owner_id}")
        row.last_error_code = error_code[:100]
        row.last_error_message = error_message[:2000]
        row.claimed_by = None
        row.claimed_at = None
        if (row.dispatch_attempts or 0) >= max_attempts:
            row.available_at = None
            row.last_error_code = "RetryExhausted"
        else:
            row.available_at = now + timedelta(seconds=max(0.0, retry_delay_seconds))
        self.session.flush()
        return row


class CommandOutcomeRepository:
    """Persist one replay-safe outcome projection for each durable command."""

    def __init__(self, session: Session) -> None:
        self.session = session

    def get_for_event(self, event_id: UUID | str) -> CommandOutcomeRow | None:
        return (
            self.session.query(CommandOutcomeRow).filter_by(event_id=_key(event_id)).one_or_none()
        )

    def list_for_run(self, run_id: UUID | str, *, limit: int = 200) -> list[CommandOutcomeRow]:
        """Return a deterministic, bounded read projection for one run."""
        if limit < 1:
            raise ValueError("command outcome limit must be positive")
        return (
            self.session.query(CommandOutcomeRow)
            .filter_by(run_id=_key(run_id))
            .order_by(CommandOutcomeRow.created_at.asc(), CommandOutcomeRow.id.asc())
            .limit(limit)
            .all()
        )

    def _record(
        self,
        *,
        event_id: UUID | str,
        event_key: str,
        event_type: str,
        aggregate_type: str,
        aggregate_id: UUID | str,
        run_id: UUID | str | None,
        status: CommandOutcomeStatus,
        attempt: int,
        result: dict[str, Any] | None,
        error_code: str | None,
        error_message: str | None,
        now: datetime | None,
    ) -> CommandOutcomeRow:
        if not event_key:
            raise PersistenceError("command outcome event_key is required")
        if not event_type:
            raise PersistenceError("command outcome event_type is required")
        if not aggregate_type:
            raise PersistenceError("command outcome aggregate_type is required")
        if attempt < 1:
            raise PersistenceError("command outcome attempt must be positive")
        now = _as_utc(now) or datetime.now(UTC)
        event_id_key = _key(event_id)
        aggregate_id_key = _key(aggregate_id)
        run_id_key = _key(run_id) if run_id is not None else None
        result_copy = deepcopy(result)
        error_code_value = error_code[:100] if error_code else None
        error_message_value = error_message[:2000] if error_message else None
        existing = self.get_for_event(event_id)
        if existing is None:
            row = CommandOutcomeRow(
                id=str(uuid4()),
                event_id=event_id_key,
                event_key=event_key,
                event_type=event_type,
                aggregate_type=aggregate_type,
                aggregate_id=aggregate_id_key,
                run_id=run_id_key,
                status=status.value,
                attempt=attempt,
                result_json=result_copy,
                error_code=error_code_value,
                error_message=error_message_value,
                created_at=now,
                updated_at=now,
            )
            self.session.add(row)
            self.session.flush()
            return row

        if (
            existing.event_key != event_key
            or existing.event_type != event_type
            or existing.aggregate_type != aggregate_type
            or existing.aggregate_id != aggregate_id_key
            or existing.run_id != run_id_key
        ):
            raise CommandOutcomeConflictError(
                f"command outcome identity conflicts for event {event_id_key}"
            )
        try:
            current_status = CommandOutcomeStatus(existing.status)
        except ValueError as exc:
            raise CommandOutcomeConflictError(
                f"command outcome has unknown status for event {event_id_key}"
            ) from exc
        same_values = (
            existing.status == status.value
            and existing.attempt == attempt
            and existing.result_json == result_copy
            and existing.error_code == error_code_value
            and existing.error_message == error_message_value
        )
        if same_values:
            return existing
        if current_status in {
            CommandOutcomeStatus.SUCCEEDED,
            CommandOutcomeStatus.EXHAUSTED,
        }:
            raise CommandOutcomeConflictError(
                f"terminal command outcome cannot change for event {event_id_key}"
            )
        if attempt < existing.attempt:
            return existing
        existing.status = status.value
        existing.attempt = attempt
        existing.result_json = result_copy
        existing.error_code = error_code_value
        existing.error_message = error_message_value
        existing.updated_at = now
        self.session.flush()
        return existing

    def record_success(
        self,
        *,
        event_id: UUID | str,
        event_key: str,
        event_type: str,
        aggregate_type: str,
        aggregate_id: UUID | str,
        run_id: UUID | str | None,
        attempt: int,
        result: dict[str, Any] | None = None,
        now: datetime | None = None,
    ) -> CommandOutcomeRow:
        return self._record(
            event_id=event_id,
            event_key=event_key,
            event_type=event_type,
            aggregate_type=aggregate_type,
            aggregate_id=aggregate_id,
            run_id=run_id,
            status=CommandOutcomeStatus.SUCCEEDED,
            attempt=attempt,
            result=result,
            error_code=None,
            error_message=None,
            now=now,
        )

    def record_failure(
        self,
        *,
        event_id: UUID | str,
        event_key: str,
        event_type: str,
        aggregate_type: str,
        aggregate_id: UUID | str,
        run_id: UUID | str | None,
        attempt: int,
        error_code: str,
        error_message: str,
        exhausted: bool,
        now: datetime | None = None,
    ) -> CommandOutcomeRow:
        return self._record(
            event_id=event_id,
            event_key=event_key,
            event_type=event_type,
            aggregate_type=aggregate_type,
            aggregate_id=aggregate_id,
            run_id=run_id,
            status=(CommandOutcomeStatus.EXHAUSTED if exhausted else CommandOutcomeStatus.RETRYING),
            attempt=attempt,
            result=None,
            error_code=error_code,
            error_message=error_message,
            now=now,
        )


class CheckpointRepository:
    """Persist one compatibility record for the latest checkpoint of a run."""

    def __init__(self, session: Session) -> None:
        self.session = session

    def get_for_run(self, run_id: UUID | str) -> CheckpointRow | None:
        return self.session.query(CheckpointRow).filter_by(run_id=_key(run_id)).one_or_none()

    def save(
        self,
        *,
        run_id: UUID | str,
        thread_id: str,
        checkpoint_id: str,
        checkpoint_kind: str,
        workflow_name: str,
        workflow_version: str,
        state_schema_version: int,
        policy_version: str,
        tool_registry_version: str,
        phase: str,
        action_id: UUID | str | None,
        approval_id: UUID | str | None,
        decision_id: UUID | str | None,
        metadata: dict[str, Any],
    ) -> CheckpointRow:
        if not checkpoint_id:
            raise PersistenceError("checkpoint_id is required")
        now = datetime.now(UTC)
        row = self.get_for_run(run_id)
        values = {
            "run_id": _key(run_id),
            "thread_id": thread_id,
            "checkpoint_id": checkpoint_id,
            "checkpoint_kind": checkpoint_kind,
            "workflow_name": workflow_name,
            "workflow_version": workflow_version,
            "state_schema_version": state_schema_version,
            "policy_version": policy_version,
            "tool_registry_version": tool_registry_version,
            "phase": phase,
            "action_id": _key(action_id) if action_id is not None else None,
            "approval_id": _key(approval_id) if approval_id is not None else None,
            "decision_id": _key(decision_id) if decision_id is not None else None,
            "metadata_json": deepcopy(metadata),
            "updated_at": now,
        }
        if row is None:
            row = CheckpointRow(id=str(uuid4()), created_at=now, **values)
            self.session.add(row)
        else:
            for key, value in values.items():
                setattr(row, key, value)
        self.session.flush()
        return row


class AuditRepository:
    """Read-only projections over the append-only audit event table."""

    def __init__(self, session: Session) -> None:
        self.session = session

    def list_for_run(self, run_id: UUID | str, *, limit: int = 200) -> list[AuditEventRow]:
        if limit < 1 or limit > 500:
            raise ValueError("audit limit must be between 1 and 500")
        return (
            self.session.query(AuditEventRow)
            .filter_by(run_id=_key(run_id))
            .order_by(AuditEventRow.occurred_at.asc(), AuditEventRow.id.asc())
            .limit(limit)
            .all()
        )

    def list_recent(self, *, limit: int = 100) -> list[AuditEventRow]:
        if limit < 1 or limit > 500:
            raise ValueError("audit limit must be between 1 and 500")
        return (
            self.session.query(AuditEventRow)
            .order_by(AuditEventRow.occurred_at.desc(), AuditEventRow.id.desc())
            .limit(limit)
            .all()
        )


class EvaluationRunRepository:
    """Store and retrieve immutable evaluation reports."""

    def __init__(self, session: Session) -> None:
        self.session = session

    def get(self, evaluation_id: UUID | str) -> EvaluationRunRow | None:
        return self.session.get(EvaluationRunRow, _key(evaluation_id))

    def require(self, evaluation_id: UUID | str) -> EvaluationRunRow:
        row = self.get(evaluation_id)
        if row is None:
            raise EntityNotFound(f"evaluation run not found: {evaluation_id}")
        return row

    def list_recent(self, *, limit: int = 50) -> list[EvaluationRunRow]:
        if limit < 1 or limit > 200:
            raise ValueError("evaluation limit must be between 1 and 200")
        return (
            self.session.query(EvaluationRunRow)
            .order_by(EvaluationRunRow.created_at.desc(), EvaluationRunRow.id.desc())
            .limit(limit)
            .all()
        )

    def save(self, row: EvaluationRunRow) -> EvaluationRunRow:
        if self.get(row.id) is not None:
            raise PersistenceError(f"evaluation run is immutable: {row.id}")
        self.session.add(row)
        self.session.flush()
        return row
