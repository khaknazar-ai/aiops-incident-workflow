"""Exports of recorded reports; rendering never reruns or rescores a cohort."""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path

from .runner import EvaluationRun


def _csv_value(value):
    if isinstance(value, (dict, list, tuple)):
        value = json.dumps(value, sort_keys=True)
    if isinstance(value, str) and value.lstrip().startswith(("=", "+", "-", "@")):
        return "'" + value
    return value


def _md(value):
    if value is None:
        return "Unknown"
    return (
        str(value).replace("|", "&#124;").replace("\n", " ").replace("\r", " ").replace("<", "&lt;")
    )


def render_report(payload: dict, format: str) -> str:
    """Export saved v1/v2 without synthesizing missing historical metrics."""
    if format == "json":
        return json.dumps(payload, indent=2, sort_keys=True) + "\n"
    scores = payload.get("scores", [])
    trajectories = {item["case_id"]: item for item in payload.get("trajectories", [])}
    metrics = payload.get("metrics")
    metric_cases = {item["case_id"]: item for item in metrics["cases"]} if metrics else {}
    if format == "csv":
        rows = []
        for score in scores:
            identity = score["case_id"]
            row = {**trajectories.get(identity, {}), **score}
            measured = metric_cases.get(identity, {})
            row.update(
                report_version=payload.get("report_version", 1),
                metrics_version=metrics.get("version") if metrics else None,
                tool_selection=measured.get("tool_selection"),
                intervention_source=measured.get("intervention_source", "unknown"),
                human_intervened=measured.get("human_intervened"),
                automated_decisions=measured.get("automated_decisions"),
                usage=measured.get("usage"),
            )
            rows.append(row)
        output = io.StringIO(newline="")
        fields = sorted({key for row in rows for key in row}) or ["case_id", "status"]
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: _csv_value(value) for key, value in row.items()} for row in rows)
        return output.getvalue()
    if format != "md":
        raise ValueError("report format must be json, csv or md")
    lines = [
        "# AgentOps evaluation report",
        "",
        f"- report version: {payload.get('report_version', 1)}",
        f"- mode: {_md(payload.get('mode'))}",
        f"- cohort: {_md(payload.get('cohort'))}",
        f"- manifest digest: {_md(payload.get('manifest_digest'))}",
        f"- source revision: {_md(payload.get('source_revision'))}",
        f"- denominator: {payload.get('denominator')}",
        f"- pass: {sum(item.get('status') == 'pass' for item in scores)}",
        f"- fail: {sum(item.get('status') == 'fail' for item in scores)}",
        f"- unavailable: {sum(item.get('status') == 'unavailable' for item in scores)}",
    ]
    model = payload.get("model_provenance")
    if isinstance(model, dict):
        lines += [
            "",
            "## Model configuration",
            "",
            f"- provider: {_md(model.get('provider'))}",
            f"- protocol: {_md(model.get('protocol'))}",
            f"- model: {_md(model.get('model'))}",
            f"- model digest: {_md(model.get('model_digest'))}",
            f"- configuration identity: {_md(model.get('configuration_id'))}",
            f"- retries: {_md(model.get('max_retries'))}",
        ]
    lines += [
        "",
        "| Case | Status | Diagnosis | Remediation | Verification | "
        "Safety violation | Failures | Missing |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for score in scores:
        lines.append(
            "| "
            + " | ".join(
                _md(score.get(key))
                for key in (
                    "case_id",
                    "status",
                    "diagnosis_correct",
                    "remediation_correct",
                    "verification_passed",
                    "safety_violation",
                    "failure_reasons",
                    "missing_fields",
                )
            )
            + " |"
        )
    lines += [
        "",
        "## Runtime measurements",
        "",
        "| Case | Steps | Tool calls | LLM calls | Wall s | Active s | Wait s |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for trajectory in trajectories.values():
        lines.append(
            "| "
            + " | ".join(
                _md(trajectory.get(key))
                for key in (
                    "case_id",
                    "steps",
                    "tool_calls",
                    "llm_calls",
                    "wall_elapsed_seconds",
                    "active_elapsed_seconds",
                    "approval_wait_seconds",
                )
            )
            + " |"
        )
    lines += ["", "## Measured coverage", ""]
    lines += [
        "| Case | Effects | Duplicates | Policy denials | Approval bypasses | Forbidden effects |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for trajectory in trajectories.values():
        lines.append(
            "| "
            + " | ".join(
                _md(trajectory.get(key))
                for key in (
                    "case_id",
                    "side_effect_count",
                    "duplicate_side_effects",
                    "policy_denials",
                    "approval_bypass_count",
                    "forbidden_effect_count",
                )
            )
            + " |"
        )
    if metrics:
        tools = metrics["tool_selection"]
        human = metrics["human_intervention"]
        lines += [
            f"Metrics contract: {metrics['version']}.",
            f"Required tools used: {tools['required_used']}/{tools['required_denominator']}; "
            f"tool trace coverage: {tools['known_cases']}/{metrics['case_denominator']} cases.",
            f"Relevant calls: {tools['relevant_calls']}/{tools['call_denominator']}; "
            f"forbidden: {tools['forbidden_calls']}; repeated: {tools['repeated_calls']}.",
            f"Human intervention rate: {_md(human['rate'])}; "
            f"observed human denominator: {human['observed_case_denominator']}; "
            f"automated cases: {human['automated_cases']}; "
            f"unknown cases: {human['unknown_cases']}.",
            "",
            "| Usage | Total | Known cases / denominator | Observed subtotal |",
            "| --- | --- | --- | --- |",
        ]
        for name, value in metrics["usage"].items():
            lines.append(
                f"| {name} | {_md(value['total'])} | {value['known_cases']}/"
                f"{value['case_denominator']} | {value['observed_subtotal']} |"
            )
        for item in metrics["cases"]:
            lines += [
                "",
                f"### {_md(item['case_id'])}",
                "Attempted tool path: " + _md(item["tool_selection"]["attempted_path"]),
                "Tool sets and denominators: "
                + _md(json.dumps(item["tool_selection"], sort_keys=True)),
            ]
        lines += [
            "",
            "### Aggregate outcomes and efficiency",
            "",
            "| Measurement | True / observed total | Known cases / denominator | Rate / mean |",
            "| --- | --- | --- | --- |",
        ]
        for category, total, rate in (
            ("outcomes", "true_cases", "rate"),
            ("efficiency", "observed_total", "mean"),
        ):
            for name, value in metrics[category].items():
                lines.append(
                    f"| {name} | {value[total]} | {value['known_cases']}/"
                    f"{value['case_denominator']} | {_md(value[rate])} |"
                )
    else:
        lines.append("Unknown: this historical report has no versioned measurement contract.")
    lines += ["", "## Recorded failures", ""]
    failed = [score for score in scores if score.get("status") != "pass"]
    if not failed:
        lines.append("No failed or unavailable cases recorded in this cohort.")
    for score in failed:
        actual = trajectories.get(score["case_id"], {})
        lines.append(
            f"- {_md(score['case_id'])}: {_md(score.get('status'))}; "
            f"reasons={_md(score.get('failure_reasons'))}; "
            f"missing={_md(score.get('missing_fields'))}; "
            f"execution error={_md(actual.get('execution_error'))}."
        )
    lines += [
        "",
        "This is a saved offline deterministic cohort, not an external LLM benchmark.",
        "Automated approvals do not measure human intervention. Missing usage/cost is unknown; "
        "observed subtotals are not complete totals. No prices are inferred.",
        "Timings are observations, not benchmark guarantees. Failures and unavailable outcomes "
        "retain the recorded denominator; exporting does not rescore them.",
    ]
    return "\n".join(lines) + "\n"


def write_reports(run: EvaluationRun, output: str | Path) -> tuple[Path, Path, Path]:
    directory = Path(output)
    directory.mkdir(parents=True, exist_ok=True)
    payload = run.as_dict()
    paths = tuple(directory / ("evaluation." + format) for format in ("json", "csv", "md"))
    for path, format in zip(paths, ("json", "csv", "md"), strict=True):
        path.write_text(render_report(payload, format), encoding="utf-8", newline="")
    return paths


__all__ = ["render_report", "write_reports"]
