"""Deterministic local demonstration of the incident triage workflow."""

from __future__ import annotations

from typing import Any, Literal

from agentops.simulator import ScenarioName, Simulator
from agentops.tools import Actor, Role, ToolGateway
from agentops.workflows.incident_triage import IncidentTriageRuntime

DemoDecision = Literal["approve", "reject"]


def run_demo(
    *,
    scenario: ScenarioName | str = ScenarioName.BAD_DEPLOYMENT,
    decision: DemoDecision = "approve",
) -> dict[str, Any]:
    """Run one fresh, in-process demo and return observable evidence.

    This deliberately uses the deterministic local runtime. It is a smoke/demo
    path, not a claim that the HTTP, Redis, or worker deployment is available.
    """

    scenario_name = scenario if isinstance(scenario, ScenarioName) else ScenarioName(scenario)
    if decision not in {"approve", "reject"}:
        raise ValueError("decision must be approve or reject")

    simulator = Simulator()
    runtime = IncidentTriageRuntime(simulator, ToolGateway(simulator))
    pending = runtime.run_scenario(scenario_name)
    pending_payload = pending.state.get("interrupt_payload")
    pending_run_status = pending.run.status.value
    decision_applied = pending.state.get("final_outcome") == "approval_required"
    final = pending
    if decision_applied:
        final = runtime.resume_approval(
            actor=Actor(actor_id="demo-approver", role=Role.APPROVER),
            approved=decision == "approve",
            reason=("local demo approval" if decision == "approve" else "local demo rejection"),
        )

    service_status = simulator.get_service_status(final.incident.service_name)
    evidence_tools = [
        item.get("tool_name")
        for item in pending.state.get("observations", [])
        if isinstance(item, dict) and isinstance(item.get("tool_name"), str)
    ]
    rollback_log_count = len(
        simulator.search_logs(final.incident.service_name, "rolled back", 20).get("entries", [])
    )
    approval_summary: dict[str, Any] | None = None
    if isinstance(pending_payload, dict):
        parameters = pending_payload.get("parameters")
        approval_summary = {
            "tool_name": pending_payload.get("tool_name"),
            "risk_level": pending_payload.get("risk_level"),
            "target_version": (
                parameters.get("target_version") if isinstance(parameters, dict) else None
            ),
            "decision": decision if decision_applied else "not-applied",
        }

    return {
        "demo": "local-direct-runtime",
        "scenario": scenario_name.value,
        "decision": decision,
        "decision_applied": decision_applied,
        "pending_run_status": pending_run_status,
        "run_status": final.run.status.value,
        "incident_status": final.incident.status.value,
        "outcome": final.state.get("final_outcome"),
        "last_error_code": final.state.get("last_error_code"),
        "run_id": str(final.run.id),
        "incident_id": str(final.incident.id),
        "evidence_tools": evidence_tools,
        "approval": approval_summary,
        "service": service_status,
        "rollback_log_count": rollback_log_count,
    }
