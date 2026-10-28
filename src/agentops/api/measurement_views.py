"""Small read-only measurement views; detailed evidence stays collapsed."""

from __future__ import annotations

import html


def _value(value):
    return "Unknown" if value is None else html.escape(str(value))


def _usage_table(usage, denominator_key):
    rows = []
    for key, label in (
        ("input_tokens", "Input tokens"),
        ("output_tokens", "Output tokens"),
        ("estimated_cost_usd", "Estimated USD"),
    ):
        item = usage[key]
        rows.append(
            f"<tr><td>{label}</td><td>{_value(item['total'])}</td>"
            f"<td>{item['known']}/{item['denominator']}</td>"
            f"<td>{_value(item['subtotal'])}</td></tr>"
        )
    return (
        '<div class="table-panel"><table><thead><tr><th>Usage</th><th>Total</th>'
        f"<th>{denominator_key}</th><th>Observed subtotal</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table></div>"
    )


def usage_details(metrics):
    usage = metrics["usage"]
    table = _usage_table(
        {
            key: {
                "total": usage[key],
                "known": usage[key + "_known_calls"],
                "denominator": usage["attempted_calls"],
                "subtotal": usage[key + "_observed_subtotal"],
            }
            for key in ("input_tokens", "output_tokens", "estimated_cost_usd")
        },
        "Known calls",
    )
    execution = "".join(
        f"<li>{kind.title()}: {value['started']} attempts, "
        f"{value['failed']} failed, {value['unfinished']} unfinished; "
        f"{value['observed_duration_ms']:.1f} ms observed.</li>"
        for kind, value in metrics["execution"].items()
    )
    return (
        '<details class="panel"><summary>Usage and execution</summary>'
        f"<p>Model attempts: {usage['attempted_calls']} · "
        f"Input tokens: {_value(usage['input_tokens'])} · "
        f"Output tokens: {_value(usage['output_tokens'])} · "
        f"Estimated USD: {_value(usage['estimated_cost_usd'])}</p>"
        '<p class="muted">Missing provider metadata stays unknown. '
        "Subtotals cover observed metadata only; cost requires a declared source.</p>"
        + table
        + "<ul>"
        + execution
        + "</ul></details>"
    )


def report_details(report):
    metrics = report.get("metrics")
    if not isinstance(metrics, dict):
        return '<p class="muted">This older report has no versioned measurements.</p>'
    tools = metrics["tool_selection"]
    humans = metrics["human_intervention"]
    usage_table = _usage_table(
        {
            key: {
                "total": item["total"],
                "known": item["known_cases"],
                "denominator": item["case_denominator"],
                "subtotal": item["observed_subtotal"],
            }
            for key, item in metrics["usage"].items()
        },
        "Known cases",
    )
    cases = []
    for item in metrics["cases"]:
        selection = item["tool_selection"]
        path = selection["attempted_path"]
        path_text = " → ".join(path) if path is not None else "Unknown"
        cases.append(
            "<details><summary>" + html.escape(item["case_id"]) + "</summary>"
            f"<p>Required tools: {_value(selection['required_used'])}/"
            f"{selection['required_denominator']}; "
            f"forbidden attempts: {_value(selection['forbidden_calls'])}; "
            f"repeated calls: {_value(selection['repeated_calls'])}.</p>"
            '<p class="muted" style="overflow-wrap:anywhere">' + html.escape(path_text) + "</p>"
            "<p>Required: " + html.escape(", ".join(selection["required_tools"])) + "</p>"
            "<p>Relevant: " + html.escape(", ".join(selection["relevant_tools"])) + "</p>"
            "<p>Forbidden: "
            + html.escape(", ".join(selection["forbidden_tools"]))
            + "</p></details>"
        )
    return (
        '<section class="panel"><h2>Measurements</h2>'
        f"<p>Required tools: {tools['required_used']}/{tools['required_denominator']} · "
        f"Forbidden calls: {tools['forbidden_calls']} · "
        f"Human intervention rate: {_value(humans['rate'])}</p>"
        f'<p class="muted">Tool coverage: {tools["known_cases"]}/'
        f"{metrics['case_denominator']} cases. "
        f"Automated decisions: {humans['automated_decisions']}; "
        f"observed human cases: {humans['observed_case_denominator']}. "
        "Automated approvals do not measure human intervention.</p>"
        "<details><summary>Usage, denominators and tool paths</summary>"
        + usage_table
        + '<p class="muted">Unknown totals are not zero. Subtotals cover known metadata only. '
        "No prices are inferred.</p>" + "".join(cases) + "</details></section>"
    )
