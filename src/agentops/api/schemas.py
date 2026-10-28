"""Stable public schemas for the control API."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator

from agentops.domain import AgentRunStatus, ApprovalStatus, IncidentStatus, RiskLevel, Severity
from agentops.persistence import CommandOutcomeStatus
from agentops.simulator import ScenarioName


class DemoIncidentCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_key: UUID
    scenario: ScenarioName
    title: str = Field(default="", max_length=300)

    @field_validator("title")
    @classmethod
    def normalize_title(cls, value: str) -> str:
        if any(ord(char) < 32 or ord(char) == 127 for char in value):
            raise ValueError("title cannot contain control characters")
        return value.strip()


class IncidentCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str = Field(min_length=1, max_length=300)
    description: str = Field(min_length=1, max_length=4000)
    service_name: str = Field(min_length=1, max_length=200)
    severity: Severity


class IncidentRead(BaseModel):
    id: UUID
    title: str
    description: str
    service_name: str
    severity: Severity
    status: IncidentStatus
    detected_at: datetime
    updated_at: datetime | None = None
    resolved_at: datetime | None
    created_by: str
    active_run_id: UUID | None
    version: int


class RunCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    workflow_name: str = Field(default="incident_triage", min_length=1, max_length=200)
    workflow_version: str = Field(default="0.1.0", min_length=1, max_length=50)
    environment_id: UUID | None = None
    scenario: str | None = Field(default=None, max_length=100)
    risk_level: RiskLevel = RiskLevel.R1
    parameters: dict[str, Any] = Field(default_factory=dict)

    @field_validator("risk_level", mode="before")
    @classmethod
    def parse_risk_level(cls, value: Any) -> Any:
        if isinstance(value, str) and value.upper().startswith("R"):
            return int(value[1:])
        return value


class RunRead(BaseModel):
    id: UUID
    incident_id: UUID
    environment_id: UUID | None
    status: AgentRunStatus
    workflow_name: str
    workflow_version: str
    state_schema_version: int
    policy_version: str
    tool_registry_version: str
    thread_id: str
    started_at: datetime | None
    finished_at: datetime | None
    cancel_requested_at: datetime | None
    step_count: int
    llm_call_count: int
    tool_call_count: int
    last_error_code: str | None
    last_error_message: str | None


class ApprovalRead(BaseModel):
    id: UUID
    action_id: UUID
    run_id: UUID
    tool_name: str
    status: ApprovalStatus
    requested_at: datetime
    expires_at: datetime | None
    decided_at: datetime | None
    decided_by: str | None
    original_parameters: dict[str, Any]
    original_parameter_hash: str
    approved_parameters: dict[str, Any] | None
    approved_parameter_hash: str | None
    decision_reason: str | None


class ApprovalDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    approved_parameters: dict[str, Any] | None = None
    reason: str | None = Field(default=None, max_length=2000)


class TimelineItem(BaseModel):
    event_type: str
    entity_type: str
    entity_id: UUID | None
    actor_id: str
    occurred_at: datetime
    payload: dict[str, Any]


class CommandOutcomeRead(BaseModel):
    event_id: UUID
    event_key: str
    event_type: str
    aggregate_type: str
    aggregate_id: str
    run_id: UUID | None
    status: CommandOutcomeStatus
    attempt: int
    result: dict[str, Any] | None
    error_code: str | None
    error_message: str | None
    created_at: datetime
    updated_at: datetime


class RunStateRead(BaseModel):
    run_id: UUID
    status: AgentRunStatus
    phase: str | None
    action_id: UUID | None
    approval_id: UUID | None
    last_error_code: str | None
    last_error_message: str | None


EvaluationMode = Literal[
    "fake",
    "rule",
    "graph",
    "durable",
    "ablation",
    "adversarial",
    "variants",
    "ollama",
]


class EvaluationCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: EvaluationMode = "graph"
    seed: int = Field(default=20260920, ge=0, le=2_147_483_647)


class EvaluationRead(BaseModel):
    id: UUID
    mode: EvaluationMode
    seed: int
    manifest_version: str
    manifest_digest: str
    source_revision: str
    cohort: str
    denominator: int
    pass_count: int
    fail_count: int
    unavailable_count: int
    created_by: str
    created_at: datetime


class EvaluationJobRead(BaseModel):
    id: UUID
    mode: EvaluationMode
    seed: int
    status: Literal["RUNNING", "SUCCEEDED", "FAILED", "QUARANTINED"]
    created_by: str
    created_at: datetime
    error_code: str | None
    evaluation_run_id: UUID | None


class ApiError(BaseModel):
    code: str
    message: str
    request_id: str
