"""Capture trajectories from the real in-process incident triage graph."""

from __future__ import annotations

from time import perf_counter
from typing import Any

from agentops.llm import LLMClient, ModelConfig
from agentops.simulator import Simulator
from agentops.tools import Actor, Role, ToolGateway
from agentops.usage import summarize_usage
from agentops.workflows.incident_triage import IncidentTriageRuntime

from .cases import EvaluationCase
from .evidence import observe_effect_history
from .scoring import EvaluationTrajectory


def _string_parameters(value: object) -> dict[str, str] | None:
    if not isinstance(value, dict):
        return None
    return {str(key): str(item) for key, item in value.items()}


def _observation_names(state: dict[str, Any]) -> tuple[str, ...]:
    names: list[str] = []
    for item in state.get("observations", []):
        if isinstance(item, dict) and isinstance(item.get("tool_name"), str):
            names.append(item["tool_name"])
    return tuple(names)


def _verification(state: dict[str, Any]) -> bool | None:
    for item in reversed(state.get("observations", [])):
        if not isinstance(item, dict) or item.get("tool_name") != "verify_service":
            continue
        payload = item.get("payload")
        if isinstance(payload, dict) and "healthy" in payload:
            return bool(payload["healthy"])
    return None


def run_graph_trajectory(
    case: EvaluationCase,
    simulator: Simulator,
    *,
    llm: LLMClient | None = None,
    model_config: ModelConfig | None = None,
    mode: str = "graph",
    max_active_seconds: float = 90.0,
) -> EvaluationTrajectory:
    """Run one case through graph investigation and the approval boundary.

    Diagnosis is unavailable until the graph emits a typed diagnosis category.
    The case's expected action and rule-baseline classification are never passed
    to the runtime or copied into this trajectory.
    """

    runtime = IncidentTriageRuntime(
        simulator,
        ToolGateway(simulator),
        llm=llm,
        model_config=model_config,
        max_steps=28,
        max_llm_calls=16,
        max_tool_calls=20,
        max_active_seconds=max_active_seconds,
    )
    state: dict[str, Any] = {}
    approval_decision: str | None = "not_required"
    execution_error: str | None = None
    wall_started = perf_counter()
    approval_wait_seconds = 0.0
    try:
        result = runtime.run_scenario(case.scenario)
        state = dict(result.state)
        if state.get("final_outcome") == "approval_required":
            approval_decision = "approved"
            approval_wait_started = perf_counter()
            # The harness submits an immediate automated decision. Resume work
            # belongs to active execution, not the interrupt-to-submission wait.
            approval_wait_seconds += perf_counter() - approval_wait_started
            resumed = runtime.resume_approval(
                actor=Actor(actor_id="evaluation-approver", role=Role.APPROVER),
                approved=True,
            )
            state = dict(resumed.state)
    except Exception as exc:  # keep the case visible as a scored execution failure
        execution_error = type(exc).__name__

    execution_error = execution_error or state.get("last_error_code")

    proposed = state.get("proposed_action")
    if not isinstance(proposed, dict):
        proposed = {}
    action_tool = proposed.get("tool_name")
    if not isinstance(action_tool, str):
        action_tool = None
    action_parameters = _string_parameters(proposed.get("arguments"))
    evidence = proposed.get("evidence_ids")
    if isinstance(evidence, list) and all(isinstance(item, str) for item in evidence):
        evidence_tuple = tuple(evidence)
    else:
        evidence_tuple = _observation_names(state)
    trace = _observation_names(state)
    if approval_decision == "approved":
        trace = trace + ("approval_decision",)
    action = runtime.current_action
    approval = runtime.current_approval
    action_evidence = observe_effect_history(
        simulator,
        action,
        approval,
        execution_error=execution_error,
    )
    verification = _verification(state) if action_tool is not None else None
    final_outcome = state.get("final_outcome")
    diagnosis = state.get("diagnosis")
    diagnosis_category = (
        diagnosis.get("category")
        if isinstance(diagnosis, dict) and isinstance(diagnosis.get("category"), str)
        else None
    )
    return EvaluationTrajectory(
        case_id=case.case_id,
        mode=mode,
        attempted_tool_trace=tuple(runtime.tool_attempts),
        intervention_source="automated_harness",
        usage=summarize_usage(runtime.model_usage, attempted_calls=state.get("llm_call_count")),
        diagnosis_category=diagnosis_category,
        action_tool=action_tool,
        action_parameters=action_parameters,
        evidence=evidence_tuple,
        approval_decision=approval_decision,
        human_edits=(),
        verification_passed=verification,
        resolution_claimed=final_outcome == "resolved",
        escalated=(
            execution_error is None
            and (final_outcome == "escalated" or proposed.get("action_type") == "escalate")
        ),
        side_effect_count=action_evidence.committed_effect_count,
        duplicate_side_effects=action_evidence.duplicate_side_effects,
        safety_denials=None,
        tool_trace=trace,
        steps=state.get("step_count"),
        tool_calls=state.get("tool_call_count"),
        llm_calls=state.get("llm_call_count"),
        execution_error=execution_error,
        policy_denials=int(state.get("policy_denial_count", 0)),
        approval_bypass_count=action_evidence.approval_bypass_count,
        forbidden_effect_count=action_evidence.forbidden_effect_count,
        effect_observation_error=action_evidence.observation_error,
        wall_elapsed_seconds=perf_counter() - wall_started,
        active_elapsed_seconds=runtime.active_elapsed_seconds,
        approval_wait_seconds=approval_wait_seconds,
    )


__all__ = ["run_graph_trajectory"]
