"""Compact LangGraph state for incident triage."""

from __future__ import annotations

from typing import Annotated, Any, Literal, TypedDict
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, ValidationError

from agentops.domain import INCIDENT_TRIAGE_WORKFLOW_VERSION, AgentRun
from agentops.llm import HypothesisSnapshot

# These are semantic readers, not source hashes or an execution registry.
# v1 is the lineage-era state, accepted only after the same shape/provenance
# checks as v2. Earlier unstructured hypotheses are explicitly unsupported.
SUPPORTED_STATE_SCHEMAS = frozenset({1, 2})
TRIAGE_NODES = (
    "load_incident",
    "triage",
    "create_plan",
    "select_next_diagnostic_step",
    "execute_read_tool",
    "record_observation",
    "update_hypotheses",
    "propose_remediation",
    "policy_gate",
    "human_approval_interrupt",
    "execute_safe_action",
    "verify_environment",
    "review",
)
READ_TOOLS = frozenset(
    {
        "get_service_status",
        "query_metrics",
        "search_logs",
        "get_deployment_history",
        "get_service_config",
    }
)
OBSERVATION_TOOLS = READ_TOOLS | {
    "restart_service",
    "restart_worker",
    "rollback_deployment",
    "verify_service",
}
Nonnegative = Annotated[int, Field(ge=0)]
Positive = Annotated[int, Field(ge=1)]


def run_contract_reason(run: AgentRun) -> str | None:
    if run.workflow_name != "incident_triage":
        return "unsupported_workflow"
    if run.workflow_version != INCIDENT_TRIAGE_WORKFLOW_VERSION:
        return "unsupported_workflow_version"
    if run.state_schema_version not in SUPPORTED_STATE_SCHEMAS:
        return "unsupported_state_schema"
    if run.policy_version != "0.1.0" or run.tool_registry_version != "0.1.0":
        return "unsupported_policy_or_tools"
    return None


class _CheckpointShape(BaseModel):
    # Unknown additive presentation fields are harmless. Execution fields below
    # are strict; validation failures are never rendered/logged verbatim.
    model_config = ConfigDict(strict=True, extra="ignore")

    run_id: str
    incident_id: str
    environment_id: str
    workflow_version: str
    model_configuration_id: str
    phase: Literal[
        "load",
        "triage",
        "plan",
        "diagnostics",
        "execute_read",
        "record_observation",
        "hypotheses",
        "remediation",
        "policy",
        "approval",
        "execute_safe_action",
        "verify",
        "review",
        "escalated",
        "cancelled",
        "resolved",
    ]
    title: str
    description: str
    service_name: str
    severity: Literal["SEV1", "SEV2", "SEV3", "SEV4"]
    diagnostic_plan: list[str] = Field(default_factory=list, max_length=8)
    current_plan_index: Nonnegative
    observations: list[dict[str, Any]] = Field(max_length=200)
    hypotheses: list[dict[str, Any]] = Field(max_length=10)
    step_count: Nonnegative
    llm_call_count: Nonnegative
    tool_call_count: Nonnegative
    retry_count: Nonnegative
    policy_denial_count: Nonnegative
    max_steps: Positive
    max_llm_calls: Positive
    max_tool_calls: Positive
    max_active_seconds: Annotated[FiniteFloat, Field(gt=0)]
    needs_more_evidence: bool | None = None
    selected_tool_name: str | None = None
    selected_tool_arguments: dict[str, Any] | None = None
    proposed_action: dict[str, Any] | None = None
    diagnosis: dict[str, Any] | None = None
    final_outcome: (
        Literal[
            "resolved",
            "escalated",
            "approval_required",
            "tool_error",
            "step_limit_exceeded",
            "budget_exceeded",
            "cancelled",
        ]
        | None
    ) = None
    last_error_code: str | None = None
    last_error_message: str | None = None


def checkpoint_shape_reason(run: AgentRun, values: Any, next_nodes: Any) -> str | None:
    """Return a finite public reason, never raw checkpoint/validation text."""
    if not isinstance(values, dict):
        return "invalid_state_shape"
    try:
        state = _CheckpointShape.model_validate(values)
    except (ValidationError, TypeError, ValueError):
        return "invalid_state_shape"
    if (
        state.run_id != str(run.id)
        or state.incident_id != str(run.incident_id)
        or state.environment_id != str(run.environment_id)
        or state.workflow_version != run.workflow_version
    ):
        return "state_identity_mismatch"
    if run.state_schema_version == 2 and (
        values.get("state_schema_version") != 2
        or values.get("workflow_name") != "incident_triage"
        or "hypothesis_ids" not in values
    ):
        return "state_contract_header_mismatch"
    if (
        type(values.get("state_schema_version", run.state_schema_version)) is not int
        or values.get("state_schema_version", run.state_schema_version) != run.state_schema_version
        or values.get("workflow_name", run.workflow_name) != run.workflow_name
    ):
        return "state_contract_header_mismatch"
    if (
        not isinstance(next_nodes, tuple)
        or len(next_nodes) > 1
        or any(node not in TRIAGE_NODES for node in next_nodes)
    ):
        return "unsupported_continuation"
    if (
        not set(state.diagnostic_plan) <= READ_TOOLS
        or state.current_plan_index > len(state.diagnostic_plan)
        or state.selected_tool_name is not None
        and state.selected_tool_name not in READ_TOOLS
    ):
        return "invalid_diagnostic_state"
    for item in state.observations:
        if (
            not isinstance(item.get("tool_name"), str)
            or item["tool_name"] not in OBSERVATION_TOOLS
            or not isinstance(item.get("payload"), dict)
        ):
            return "invalid_observation_shape"
    try:
        hypotheses = [HypothesisSnapshot.model_validate(item) for item in state.hypotheses]
    except (ValidationError, TypeError, ValueError):
        return "unsupported_hypothesis_shape"
    ids = [str(item.id) for item in hypotheses]
    if len(set(ids)) != len(ids) or values.get("hypothesis_ids", ids) != ids:
        return "invalid_hypothesis_identity"
    if any(
        item.run_id != run.id or raw.get("summary") != item.summary
        for item, raw in zip(hypotheses, state.hypotheses, strict=True)
    ):
        return "invalid_hypothesis_identity"
    if state.diagnosis is not None and not any(
        item.category == state.diagnosis.get("category")
        and item.summary == state.diagnosis.get("summary")
        for item in hypotheses
    ):
        return "invalid_diagnosis_shape"
    proposal = state.proposed_action
    if proposal is not None:
        if (
            not isinstance(proposal.get("tool_name"), (str, type(None)))
            or not isinstance(proposal.get("arguments"), dict)
            or type(proposal.get("risk_level")) is not int
            or proposal["risk_level"] not in range(4)
        ):
            return "invalid_proposal_shape"
        for key in ("action_id", "approval_request_id"):
            if key in proposal:
                try:
                    UUID(proposal[key])
                except (TypeError, ValueError, AttributeError):
                    return "invalid_proposal_shape"
    if (
        state.step_count > state.max_steps
        or state.llm_call_count > state.max_llm_calls
        or state.tool_call_count > state.max_tool_calls
    ):
        return "invalid_budget_state"
    return None


class IncidentAgentState(TypedDict, total=False):
    incident_id: str
    environment_id: str
    run_id: str
    workflow_version: str
    workflow_name: str
    state_schema_version: int
    model_configuration_id: str | None
    phase: str
    title: str
    description: str
    service_name: str
    severity: str
    diagnostic_plan: list[str]
    current_plan_index: int
    observations: list[dict[str, Any]]
    hypotheses: list[dict[str, Any]]
    hypothesis_ids: list[str]
    diagnosis: dict[str, Any] | None
    needs_more_evidence: bool
    selected_tool_name: str | None
    selected_tool_arguments: dict[str, Any] | None
    proposed_action: dict[str, Any] | None
    final_outcome: str | None
    retry_count: int
    step_count: int
    max_steps: int
    max_llm_calls: int
    max_tool_calls: int
    max_active_seconds: float
    llm_call_count: int
    tool_call_count: int
    policy_denial_count: int
    last_error_code: str | None
    last_error_message: str | None
