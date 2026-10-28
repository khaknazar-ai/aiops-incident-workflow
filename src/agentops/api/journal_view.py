"""Compact rendering of journal records in the existing run timeline."""

import html
import json

from agentops.api.dashboard import badge, timestamp
from agentops.journal_contract import safe_projection
from agentops.observability.redaction import redact_payload

LABELS = {
    "load_incident": "Incident loaded",
    "triage": "Triage",
    "create_plan": "Investigation plan",
    "select_next_diagnostic_step": "Next diagnostic step",
    "execute_read_tool": "Read evidence",
    "record_observation": "Evidence saved",
    "update_hypotheses": "Hypotheses updated",
    "propose_remediation": "Remediation proposed",
    "policy_gate": "Policy check",
    "human_approval_interrupt": "Human review",
    "execute_safe_action": "Remediation execution",
    "verify_environment": "Environment verification",
    "review": "Review",
    "get_service_status": "Service status",
    "query_metrics": "Metrics",
    "search_logs": "Log search",
    "get_deployment_history": "Deployment history",
    "get_service_config": "Service configuration",
    "restart_service": "Restart service",
    "restart_worker": "Restart worker",
    "rollback_deployment": "Roll back deployment",
    "verify_service": "Verify service",
    "diagnostic_plan": "Investigation plan",
    "tool_selection": "Tool selection",
    "hypothesis_update": "Hypothesis update",
    "remediation_proposal": "Remediation proposal",
}


def audit_payload(row):
    return (
        safe_projection(row.payload_json)
        if row.event_type.startswith("journal.")
        else redact_payload(row.payload_json)
    )


def timeline_entry(row) -> str:
    item = audit_payload(row)
    if not row.event_type.startswith("journal."):
        return (
            "<li>" + timestamp(row.occurred_at) + f"<h2>{html.escape(row.event_type)}</h2>"
            "<details><summary>Event details</summary>"
            f"<pre>{html.escape(json.dumps(item, sort_keys=True))}</pre></details></li>"
        )
    name, phase, facts = item["name"], item["phase"], item["facts"]
    title = LABELS.get(name, "Execution attempt")
    if item["kind"] == "llm":
        title = f"Model · {title}"
    if item["kind"] == "run":
        title = "Run " + str(facts.get("run_status", "recorded")).replace("_", " ").lower()
    if item["kind"] == "hypothesis":
        title = "Hypothesis changes"
    status = phase.upper()
    if item["kind"] == "run":
        status = facts.get("run_status", "RECORDED")
    if phase == "started":
        summary = "Attempt started. Its outcome is recorded separately when available."
    elif phase == "interrupted":
        summary = (
            "Paused for human review."
            if name == "human_approval_interrupt"
            else "Attempt interrupted."
        )
    elif "verification_passed" in facts:
        summary = (
            "Verification passed." if facts["verification_passed"] else "Verification did not pass."
        )
    elif facts.get("outcome"):
        summary = "Outcome: " + facts["outcome"].replace("_", " ") + "."
    elif facts.get("error_code"):
        summary = "Stopped safely · " + facts["error_code"]
    else:
        summary = ""
    duration = item.get("duration_ms")
    timing = f" · {duration:.0f} ms" if isinstance(duration, (int, float)) else ""
    attempt_line = (
        f'<span class="muted">Attempt {item.get("attempt", 1)}{timing}</span>'
        if item["kind"] != "run"
        else ""
    )
    score_note = (
        '<p class="muted">Scores are model-reported, not calibrated probabilities.</p>'
        if facts.get("hypothesis_scores")
        else ""
    )
    changes = facts.get("hypothesis_changes", [])
    hypothesis_lines = ""
    for hypothesis in changes:
        target = f"/dashboard/runs/{hypothesis['run_id']}/hypotheses/{hypothesis['id']}"
        hypothesis_lines += (
            f'<p style="overflow-wrap:anywhere">{html.escape(hypothesis["summary"])} '
            f"{badge(hypothesis['status'].upper())}<br>"
            f'<span class="muted">Model score {hypothesis["confidence"]:.2f} (uncalibrated)</span> '
            f'<a href="{target}">History · v{hypothesis["version"]}</a></p>'
        )
    identity = getattr(row, "id", item.get("journal_id"))
    opening = f'<li id="event-{html.escape(str(identity), quote=True)}">' if identity else "<li>"
    return (
        opening + timestamp(row.occurred_at) + f"<h2>{html.escape(title)}</h2>"
        f"<p>{badge(status)} {attempt_line}</p>"
        + (f'<p class="muted">{html.escape(summary)}</p>' if summary else "")
        + hypothesis_lines
        + "<details><summary>Technical details</summary>"
        + score_note
        + f"<pre>{html.escape(json.dumps(item, sort_keys=True))}</pre></details></li>"
    )
