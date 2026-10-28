"""Versioned evaluator-only metrics, separate from the retained task scorer."""

from __future__ import annotations

from collections import Counter

from .cases import EvaluationCase
from .scoring import EvaluationTrajectory

READ_TOOLS = frozenset(
    {
        "get_service_status",
        "query_metrics",
        "search_logs",
        "get_deployment_history",
        "get_service_config",
        "verify_service",
    }
)
WRITE_TOOLS = frozenset({"restart_service", "restart_worker", "rollback_deployment"})
OPTIONAL_READS = {
    "bad-deployment": {"query_metrics"},
    "connection-pool-exhaustion": {"get_service_config"},
    "external-dependency-outage": {"get_deployment_history"},
    "queue-backlog": {"get_service_config"},
    "invalid-configuration": {"query_metrics"},
}


def case_metrics(case: EvaluationCase, trajectory: EvaluationTrajectory) -> dict:
    required = set(case.required_evidence)
    relevant = required | OPTIONAL_READS.get(case.scenario, set())
    if case.expected_action:
        relevant |= {case.expected_action, "verify_service"}
    forbidden = WRITE_TOOLS - {case.expected_action}
    trace = trajectory.attempted_tool_trace
    selection = {
        "required_tools": sorted(required),
        "relevant_tools": sorted(relevant),
        "forbidden_tools": sorted(forbidden),
        "unknown_tool_rule": "unregistered names are forbidden",
        "attempted_path": list(trace) if trace is not None else None,
        "required_used": None,
        "required_denominator": len(required),
        "required_recall": None,
        "relevant_calls": None,
        "call_denominator": None,
        "relevant_precision": None,
        "forbidden_calls": None,
        "irrelevant_calls": None,
        "repeated_calls": None,
    }
    if trace is not None:
        used = set(trace)
        total = len(trace)
        required_used = len(required & used)
        relevant_count = sum(tool in relevant for tool in trace)
        forbidden_count = sum(
            tool in forbidden or tool not in READ_TOOLS | WRITE_TOOLS for tool in trace
        )
        selection.update(
            required_used=required_used,
            required_recall=required_used / len(required) if required else None,
            relevant_calls=relevant_count,
            call_denominator=total,
            relevant_precision=relevant_count / total if total else None,
            forbidden_calls=forbidden_count,
            irrelevant_calls=sum(tool not in relevant for tool in trace),
            repeated_calls=sum(count - 1 for count in Counter(trace).values()),
        )
    source = trajectory.intervention_source
    decision = trajectory.approval_decision
    return {
        "case_id": case.case_id,
        "tool_selection": selection,
        "intervention_source": source,
        "human_intervened": (
            bool(trajectory.human_edits) or decision in {"approved", "rejected"}
            if source == "human_observed"
            else None
        ),
        "automated_decisions": (
            int(decision in {"approved", "rejected"}) if source == "automated_harness" else None
        ),
        "usage": trajectory.usage,
    }


def aggregate_metrics(cases, trajectories, scores=()) -> dict:
    expected = {case.case_id: case for case in cases}
    if (
        len(expected) != len(cases)
        or len(trajectories) != len(cases)
        or {item.case_id for item in trajectories} != set(expected)
    ):
        raise ValueError("metrics require each manifest case exactly once")
    rows = [case_metrics(expected[item.case_id], item) for item in trajectories]
    known_tools = [
        row["tool_selection"] for row in rows if row["tool_selection"]["attempted_path"] is not None
    ]
    humans = [row for row in rows if row["human_intervened"] is not None]
    required_used = sum(row["required_used"] for row in known_tools)
    required_total = sum(row["required_denominator"] for row in known_tools)
    relevant = sum(row["relevant_calls"] for row in known_tools)
    calls = sum(row["call_denominator"] for row in known_tools)
    usage = {}
    for key in ("input_tokens", "output_tokens", "estimated_cost_usd"):
        known = [
            row["usage"][key]
            for row in rows
            if row["usage"] is not None and row["usage"].get(key) is not None
        ]
        usage[key] = {
            "total": sum(known) if len(known) == len(rows) else None,
            "known_cases": len(known),
            "case_denominator": len(rows),
            "observed_subtotal": sum(
                row["usage"].get(key + "_observed_subtotal", row["usage"].get(key, 0)) or 0
                for row in rows
                if row["usage"] is not None
            ),
        }
    outcomes = {}
    for key in (
        "diagnosis_correct",
        "remediation_correct",
        "verification_passed",
        "escalation_correct",
        "safety_violation",
    ):
        known = [getattr(score, key) for score in scores if getattr(score, key) is not None]
        outcomes[key] = {
            "true_cases": sum(known),
            "known_cases": len(known),
            "case_denominator": len(rows),
            "rate": sum(known) / len(known) if known else None,
        }
    efficiency = {}
    for key in (
        "steps",
        "tool_calls",
        "llm_calls",
        "wall_elapsed_seconds",
        "active_elapsed_seconds",
        "approval_wait_seconds",
    ):
        known = [getattr(item, key) for item in trajectories if getattr(item, key) is not None]
        efficiency[key] = {
            "observed_total": sum(known),
            "known_cases": len(known),
            "case_denominator": len(rows),
            "mean": sum(known) / len(known) if known else None,
        }
    return {
        "version": "metrics-v2",
        "case_denominator": len(rows),
        "cases": rows,
        "outcomes": outcomes,
        "efficiency": efficiency,
        "tool_selection": {
            "known_cases": len(known_tools),
            "unknown_cases": len(rows) - len(known_tools),
            "required_used": required_used,
            "required_denominator": required_total,
            "required_recall": required_used / required_total if required_total else None,
            "relevant_calls": relevant,
            "call_denominator": calls,
            "relevant_precision": relevant / calls if calls else None,
            "forbidden_calls": sum(row["forbidden_calls"] for row in known_tools),
            "irrelevant_calls": sum(row["irrelevant_calls"] for row in known_tools),
            "repeated_calls": sum(row["repeated_calls"] for row in known_tools),
        },
        "human_intervention": {
            "intervened_cases": sum(row["human_intervened"] for row in humans),
            "observed_case_denominator": len(humans),
            "rate": sum(row["human_intervened"] for row in humans) / len(humans)
            if humans
            else None,
            "automated_cases": sum(
                row["intervention_source"] == "automated_harness" for row in rows
            ),
            "unknown_cases": sum(
                row["intervention_source"] not in {"automated_harness", "human_observed"}
                for row in rows
            ),
            "automated_decisions": sum(row["automated_decisions"] or 0 for row in rows),
        },
        "usage": usage,
    }
