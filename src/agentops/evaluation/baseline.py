"""Independent rule baseline over agent-visible simulator observations."""

from __future__ import annotations

from dataclasses import replace
from typing import Any
from uuid import NAMESPACE_URL, uuid5

from agentops.domain import Action, ApprovalRequest
from agentops.simulator import Simulator
from agentops.tools import Actor, Role, ToolGateway
from agentops.usage import summarize_usage

from .cases import EvaluationCase
from .evidence import observe_effect_history
from .scoring import EvaluationTrajectory

_METRIC_NAMES = (
    "p95_latency_ms",
    "error_rate",
    "pool_waiters",
    "queue_depth",
    "processing_rate",
    "dns_errors",
    "fraud_api_5xx",
)


def _classify(
    *,
    status: dict[str, Any],
    metrics: dict[str, Any],
    logs: list[str],
    deployments: list[dict[str, str]],
    config: dict[str, str],
) -> tuple[str, str | None, dict[str, str] | None, bool]:
    lowered = " ".join(logs).casefold()
    if len(deployments) > 1 and "deployment" in lowered and metrics.get("p95_latency_ms", 0) > 300:
        stable = next(
            (item["version"] for item in deployments if item.get("status") == "stable"),
            None,
        )
        return (
            "bad_deployment",
            "rollback_deployment",
            {"service": status["service"], "target_version": stable} if stable else None,
            False,
        )
    if metrics.get("pool_waiters", 0) > 0 or "connection pool" in lowered:
        return (
            "connection_pool_exhaustion",
            "restart_service",
            {"service": status["service"]},
            False,
        )
    dependencies = status.get("dependencies", {})
    if dependencies.get("fraud-api") in {"unavailable", "unreachable"}:
        return "external_dependency_outage", None, None, True
    if status.get("queue_depth", 0) > 0 or status.get("worker_running") is False:
        return (
            "queue_consumer_failure",
            "restart_worker",
            {"service": status["service"]},
            False,
        )
    if (
        metrics.get("dns_errors", 0) > 0
        or "hostname" in lowered
        or "configuration validation" in lowered
        or "inventory_api_host" in config
    ):
        return "invalid_configuration", None, None, True
    return "unknown", None, None, True


def run_rule_baseline(
    case: EvaluationCase,
    simulator: Simulator,
    *,
    mode: str = "rule",
    include_deployment_history: bool = True,
    include_service_config: bool = True,
) -> EvaluationTrajectory:
    """Run a deterministic baseline without reading evaluator-only truth.

    ``include_deployment_history=False`` enables the observation ablation. It
    removes one visible observation from the baseline but leaves policy,
    approval, verification, and safety checks unchanged.
    """

    simulator.load(case.scenario)
    context = simulator.incident_context()
    service = context["service_name"]
    status = simulator.get_service_status(service)
    metrics_payload = simulator.query_metrics(service, list(_METRIC_NAMES), 15)
    logs_payload = simulator.search_logs(service, "e", 100)
    deployments_payload = (
        simulator.get_deployment_history(service)
        if include_deployment_history
        else {"deployments": []}
    )
    config_payload = (
        simulator.get_service_config(service) if include_service_config else {"values": {}}
    )
    diagnosis, action, parameters, escalate = _classify(
        status=status,
        metrics=metrics_payload["metrics"],
        logs=logs_payload["entries"],
        deployments=deployments_payload["deployments"],
        config=config_payload["values"],
    )
    evidence = ["get_service_status", "query_metrics", "search_logs"]
    if include_deployment_history:
        evidence.append("get_deployment_history")
    if include_service_config:
        evidence.append("get_service_config")
    measured = observe_effect_history(simulator, None, None, execution_error=None)
    return EvaluationTrajectory(
        case_id=case.case_id,
        mode=mode,
        attempted_tool_trace=tuple(evidence),
        intervention_source="automated_harness",
        usage=summarize_usage([]),
        diagnosis_category=diagnosis,
        action_tool=action,
        action_parameters=parameters,
        evidence=tuple(evidence),
        approval_decision=None,
        human_edits=(),
        verification_passed=None,
        resolution_claimed=False,
        escalated=escalate,
        side_effect_count=measured.committed_effect_count,
        duplicate_side_effects=measured.duplicate_side_effects,
        safety_denials=0,
        tool_trace=tuple(evidence),
        steps=len(evidence),
        tool_calls=len(evidence),
        llm_calls=0,
        policy_denials=0,
        approval_bypass_count=measured.approval_bypass_count,
        forbidden_effect_count=measured.forbidden_effect_count,
        effect_observation_error=measured.observation_error,
    )


def execute_approved_proposal(
    trajectory: EvaluationTrajectory,
    simulator: Simulator,
) -> EvaluationTrajectory:
    """Apply a fixed approve-all harness without consulting evaluator truth."""

    if trajectory.action_tool is None or trajectory.action_parameters is None:
        return trajectory
    action_id = trajectory.case_id + "-action"
    approval = ApprovalRequest(
        action_id=uuid5(NAMESPACE_URL, action_id),
        run_id=uuid5(NAMESPACE_URL, trajectory.case_id),
        tool_name=trajectory.action_tool,
        original_parameters=trajectory.action_parameters,
    )
    approval.decide(
        approved=True,
        decided_by="evaluation-harness",
        approved_parameters=trajectory.action_parameters,
    )
    gateway = ToolGateway(simulator)
    actor = Actor(actor_id="evaluation-operator", role=Role.OPERATOR)
    idempotency_key = f"eval-{trajectory.case_id}"
    definition = next(item for item in gateway.definitions() if item.name == trajectory.action_tool)
    action = Action(
        id=approval.action_id,
        run_id=approval.run_id,
        action_type=trajectory.action_tool,
        tool_name=trajectory.action_tool,
        parameters=trajectory.action_parameters,
        risk_level=definition.risk_level,
        idempotency_key=idempotency_key,
    )
    attempted_tools: list[str] = []

    def call(tool_name: str, parameters: dict[str, str], **kwargs):
        attempted_tools.append(tool_name)
        return gateway.call(tool_name, parameters, actor, **kwargs)

    try:
        call(
            trajectory.action_tool,
            trajectory.action_parameters,
            approval=approval,
            idempotency_key=idempotency_key,
        )
        call(
            trajectory.action_tool,
            trajectory.action_parameters,
            approval=approval,
            idempotency_key=idempotency_key,
        )
        verify = call(
            "verify_service",
            {"service": trajectory.action_parameters["service"]},
        )
        healthy = bool(getattr(verify.payload, "healthy", False))
        measured = observe_effect_history(simulator, action, approval, execution_error=None)
        return replace(
            trajectory,
            approval_decision="approved",
            verification_passed=healthy,
            resolution_claimed=healthy,
            side_effect_count=measured.committed_effect_count,
            duplicate_side_effects=measured.duplicate_side_effects,
            approval_bypass_count=measured.approval_bypass_count,
            forbidden_effect_count=measured.forbidden_effect_count,
            effect_observation_error=measured.observation_error,
            tool_trace=trajectory.tool_trace + ("approval_decision", *attempted_tools),
            attempted_tool_trace=(trajectory.attempted_tool_trace or ()) + tuple(attempted_tools),
            tool_calls=(trajectory.tool_calls or 0) + len(attempted_tools),
            steps=(trajectory.steps or 0) + 1,
        )
    except Exception as exc:
        measured = observe_effect_history(
            simulator, action, approval, execution_error=type(exc).__name__
        )
        return replace(
            trajectory,
            approval_decision="approved",
            execution_error=type(exc).__name__,
            side_effect_count=measured.committed_effect_count,
            duplicate_side_effects=measured.duplicate_side_effects,
            safety_denials=None,
            approval_bypass_count=measured.approval_bypass_count,
            forbidden_effect_count=measured.forbidden_effect_count,
            effect_observation_error=measured.observation_error,
            tool_trace=trajectory.tool_trace + ("approval_decision", *attempted_tools),
            attempted_tool_trace=(trajectory.attempted_tool_trace or ()) + tuple(attempted_tools),
            tool_calls=(trajectory.tool_calls or 0) + len(attempted_tools),
        )


__all__ = ["execute_approved_proposal", "run_rule_baseline"]
