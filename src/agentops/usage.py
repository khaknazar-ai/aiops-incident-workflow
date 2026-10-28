"""Numeric-only usage observations; missing metadata never becomes free usage."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from typing import Any


def usage_facts(raw: Mapping[str, Any]) -> dict[str, Any]:
    facts: dict[str, Any] = {"usage_version": 1}
    if not isinstance(raw, Mapping):
        return facts
    for key in ("input_tokens", "output_tokens"):
        value = raw.get(key)
        if type(value) is int and 0 <= value <= 10**15:
            facts[key] = value
    cost = raw.get("estimated_cost_usd")
    if (
        type(cost) in {int, float}
        and 0 <= cost <= 10**9
        and math.isfinite(cost)
        and raw.get("cost_source") == "provider_reported"
    ):
        facts.update(estimated_cost_usd=float(cost), cost_source="provider_reported")
    return facts


def summarize_usage(
    samples: Iterable[Mapping[str, Any]], *, attempted_calls: int | None = None
) -> dict[str, Any]:
    observations = [usage_facts(item) for item in samples]
    count = max(len(observations), attempted_calls or 0)
    result: dict[str, Any] = {"version": 1, "attempted_calls": count}
    for key in ("input_tokens", "output_tokens", "estimated_cost_usd"):
        values = [item[key] for item in observations if key in item]
        known = sum(values)
        result[key] = known if len(values) == count else None
        result[key + "_observed_subtotal"] = known
        result[key + "_known_calls"] = len(values)
    result["cost_source"] = (
        "provider_reported" if any("estimated_cost_usd" in item for item in observations) else None
    )
    return result


def read_run_usage(session, run_id) -> dict[str, Any]:
    from agentops.persistence import AgentRunRepository, RunBudgetRepository, RunJournalRow

    run = AgentRunRepository(session).require(run_id)
    budget = RunBudgetRepository(session).get_for_run(run_id)
    rows = session.query(RunJournalRow).filter_by(run_id=str(run_id), kind="llm").all()
    attempts: dict[str, dict] = {}
    for row in rows:
        if row.phase == "started":
            attempts.setdefault(row.attempt_id, {})
        else:
            attempts[row.attempt_id] = row.facts_json
    count = max(run.llm_call_count, budget.llm_calls_used if budget else 0)
    return summarize_usage(attempts.values(), attempted_calls=count)


def read_run_metrics(session, run_id) -> dict[str, Any]:
    from agentops.persistence import AgentRunRepository, RunBudgetRepository, RunJournalRow

    run = AgentRunRepository(session).require(run_id)
    budget = RunBudgetRepository(session).get_for_run(run_id)
    rows = session.query(RunJournalRow).filter_by(run_id=str(run_id)).all()
    execution = {}
    for kind in ("node", "tool", "llm"):
        starts = {
            row.attempt_id: row for row in rows if row.kind == kind and row.phase == "started"
        }
        ends = {row.attempt_id: row for row in rows if row.kind == kind and row.phase != "started"}
        execution[kind] = {
            "started": len(starts),
            "completed": sum(row.phase == "completed" for row in ends.values()),
            "failed": sum(row.phase == "failed" for row in ends.values()),
            "interrupted": sum(row.phase == "interrupted" for row in ends.values()),
            "unfinished": len(starts.keys() - ends.keys()),
            "additional_attempts": sum(row.attempt_number > 1 for row in starts.values()),
            "observed_duration_ms": sum(row.duration_ms or 0 for row in ends.values()),
        }
    return {
        "version": 1,
        "run_id": str(run.id),
        "status": run.status.value,
        "steps": max(run.step_count, budget.steps_used if budget else 0),
        "tool_calls": max(run.tool_call_count, budget.tool_calls_used if budget else 0),
        "llm_calls": max(run.llm_call_count, budget.llm_calls_used if budget else 0),
        "active_elapsed_seconds": budget.active_elapsed_seconds if budget else None,
        "execution": execution,
        "usage": read_run_usage(session, run_id),
        "journal_coverage": "observed_attempts_only; legacy/unmatched attempts remain unknown",
    }
