"""Presentation helpers for the local operator dashboard; no state mutations."""

from __future__ import annotations

import html
from datetime import UTC, datetime

from agentops.api.auth import ApiPrincipal, BrowserSession
from agentops.api.dashboard_queries import FILTERS, PAGE_SIZES, BrowseKind, BrowsePage

STYLES = """
:root {color-scheme:light;--ink:#242b33;--muted:#59636f;--line:#dfe3e8;
  --paper:#fafbfc;--card:#fff;--accent:#315b8a;--green:#21654a;--amber:#825515}
* {box-sizing:border-box}
body {margin:0;background:var(--paper);color:var(--ink);
  font:15px/1.6 system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
a {color:#245d76;text-underline-offset:3px;overflow-wrap:anywhere}
a:hover {color:var(--accent)}
:focus-visible {outline:3px solid var(--accent);outline-offset:4px}
.skip-link {position:absolute;left:16px;top:-80px;background:white;padding:12px;z-index:5}
.skip-link:focus {top:12px}
.topbar {background:var(--card);border-bottom:1px solid var(--line)}
.topbar-inner {max-width:1200px;margin:auto;padding:12px 32px;display:flex;
  align-items:center;justify-content:space-between;gap:16px;flex-wrap:wrap}
.brand {display:inline-flex;gap:10px;align-items:center;color:var(--ink);
  text-decoration:none;font-size:19px;font-weight:700;letter-spacing:-.4px}
.brand:hover {color:var(--accent)}
.brand-mark {display:grid;place-items:center;width:34px;height:34px;
  border:1px solid var(--line);border-radius:5px;color:var(--muted);font:600 18px system-ui}
.environment {font-size:12px;color:var(--muted)}
.account {display:flex;align-items:center;gap:14px;flex-wrap:wrap;max-width:100%;
  font-size:12px;color:var(--muted);min-width:0}
.account .identity {overflow-wrap:anywhere;max-width:100%}
.account form {margin:0}
.account button {border-color:var(--line);min-height:44px;background:white;color:var(--ink)}
.account button:hover {background:#f1f3f5}
.workspace {max-width:1200px;margin:auto;padding:0 32px 40px;min-width:0}
.nav {display:flex;gap:6px 28px;flex-wrap:wrap;border-bottom:1px solid var(--line);
  padding:0;margin-bottom:24px}
.nav a {padding:16px 0 13px;text-decoration:none;color:var(--muted);
  border-bottom:3px solid transparent;font-weight:600;font-size:14px}
.nav a[aria-current=page] {color:var(--ink);border-color:var(--accent)}
.page-head {display:flex;align-items:flex-start;justify-content:space-between;
  gap:20px;flex-wrap:wrap;margin:28px 0 20px}
.eyebrow {color:var(--accent);font-size:11px;text-transform:uppercase;
  letter-spacing:1.8px;font-weight:750;margin:0 0 7px}
h1 {font-size:28px;line-height:1.25;letter-spacing:-.6px;
  margin:0 0 10px;font-weight:650}
h2 {font-size:18px;line-height:1.4;margin:0;overflow-wrap:anywhere}
p {margin:0 0 16px}
.subtitle,.muted {color:var(--muted)}
.subtitle {max-width:760px;margin:0;font-size:14px}
.role {font-size:12px;color:var(--muted);padding:5px 0;overflow-wrap:anywhere;max-width:100%}
.stats {display:flex;flex-wrap:wrap;gap:8px 24px;margin:18px 0}
.stat {display:flex;align-items:baseline;gap:7px}
.stat strong {font-size:15px;font-weight:650}
.stat span {font-size:12px;color:var(--muted)}
.panel,.approval-card {border:1px solid var(--line);border-radius:6px;background:var(--card)}
.panel {padding:24px;margin-bottom:18px;min-width:0}
.panel-heading {display:flex;align-items:center;justify-content:space-between;
  gap:16px;flex-wrap:wrap;margin-bottom:16px}
.count {font-size:12px;color:var(--muted)}
.table-panel {padding:0;overflow:hidden}
table {width:100%;border-collapse:collapse;table-layout:fixed;font-size:13px}
th {text-align:left;font-size:12px;
  background:#f5f6f8;color:var(--muted);font-weight:600}
th,td {padding:16px 18px;border-bottom:1px solid var(--line);vertical-align:top;
  overflow-wrap:anywhere;min-width:0}
tbody tr:last-child td {border-bottom:0}
tbody tr:hover {background:#fafbfc}
td strong {display:block;font-size:14px;line-height:1.5;margin-bottom:5px}
td code {display:block;color:var(--muted);font-size:10px}
code,pre {font-family:ui-monospace,SFMono-Regular,Consolas,monospace;
  overflow-wrap:anywhere;word-break:normal}
code {font-size:.82em}
pre {white-space:pre-wrap;background:#f6f7f9;border:1px solid var(--line);
  border-radius:4px;padding:14px;margin:10px 0 0;font-size:12px;line-height:1.7;
  max-width:100%;min-width:0}
.badge {display:inline-block;border:1px solid #d5dfda;border-radius:6px;
  padding:3px 8px;font-size:10px;font-weight:750;letter-spacing:.4px;
  line-height:1.5;max-width:100%;overflow-wrap:anywhere;background:#edf1ef;color:#41565e}
.badge-success {background:#e7f3eb;border-color:#bad9c6;color:#20583c}
.badge-warning {background:#fcf1dc;border-color:#e8cfa0;color:#744a12}
.badge-danger {background:#fbeae6;border-color:#ecc3b9;color:#96351e}
.badges {display:flex;gap:6px;flex-wrap:wrap;align-items:center}
.approval-list {display:grid;gap:20px}
.approval-card {padding:24px;display:grid;grid-template-columns:repeat(2,minmax(0,1fr));
  gap:18px 24px;min-width:0}
.approval-card>header,.approval-card>.meta,.approval-card>.parameters,
.approval-card>details,.approval-card>.notice {grid-column:1/-1;min-width:0}
.approval-card>header {display:flex;justify-content:space-between;gap:14px;flex-wrap:wrap}
.action-name {margin:4px 0 0;color:var(--muted);font-size:13px}
.feedback {margin:0 0 18px;padding:10px 14px;border:1px solid #bad9c6;
  border-radius:4px;background:#f1f8f3;color:#20583c;font-size:13px}
.feedback a {margin-left:8px}
.page-action {margin:0 0 16px;font-size:14px}
.create-panel {max-width:640px}
.create-panel label:not(:first-child) {margin-top:18px}
.create-panel .muted {font-size:12px;margin:8px 0 20px}
.meta {display:flex;gap:12px 28px;flex-wrap:wrap;font-size:12px;color:var(--muted)}
.meta>span {min-width:0;overflow-wrap:anywhere}
.field-title {font-size:13px;font-weight:600;color:var(--muted);margin:0 0 8px}
details {min-width:0;border-top:1px solid var(--line);padding-top:14px}
summary {cursor:pointer;font-weight:650;font-size:13px;overflow-wrap:anywhere}
summary .count {margin-left:8px;font-weight:400}
details[open]>summary {margin-bottom:14px}
label {display:block;font-size:12px;font-weight:650;margin-bottom:6px}
input,select {display:block;width:100%;min-width:0;border:1px solid #b8c8c4;
  padding:11px 12px;border-radius:4px;background:white;color:var(--ink);font:inherit}
input[type=hidden] {display:none}
button {border:1px solid #172d35;background:#172d35;color:#fff;font:600 13px/1.5 system-ui;
  padding:11px 18px;border-radius:4px;cursor:pointer;min-height:44px;max-width:100%;
  white-space:normal;overflow-wrap:anywhere}
button:hover {background:#294b54}
button.secondary {background:white;color:#96351e;border-color:#d7b2a8}
button.secondary:hover {background:#fbeae6}
.decision {margin:0;padding-top:16px;border-top:1px solid var(--line);min-width:0}
.decision button {margin-top:12px;width:100%}
.timeline {list-style:none;padding:0;margin:0}
.timeline>li {position:relative;padding:0 0 24px 26px;border-left:1px solid #cedbd3;min-width:0}
.timeline>li:before {content:"";width:9px;height:9px;position:absolute;left:-5px;top:7px;
  border-radius:50%;background:#517f69;box-shadow:0 0 0 4px var(--card)}
.timeline time {display:block;color:var(--muted);font-size:11px;margin-bottom:4px}
.timeline details {border:0;padding-top:6px}
.empty {text-align:center;padding:44px 24px;color:var(--muted)}
.empty strong {display:block;color:var(--ink);font-size:17px;margin-bottom:5px}
.notice {font-size:13px;padding:12px 16px;border-left:3px solid #a3b4c7;
  background:#f0f4f8;margin-bottom:20px;color:#34475a;overflow-wrap:anywhere}
.footer {margin-top:32px;padding-top:18px;border-top:1px solid var(--line);
  font-size:11px;color:var(--muted);display:flex;justify-content:space-between;
  gap:12px;flex-wrap:wrap}
.login-layout {max-width:400px;margin:64px auto}
.login-panel {padding:32px}
.login-panel button {width:100%;margin-top:20px}
.login-panel h1 {margin-bottom:12px}
.login-panel .muted {font-size:13px}
.filters {display:flex;flex-wrap:wrap;gap:12px;align-items:flex-end;margin:22px 0}
.filter-field {flex:1 1 150px;min-width:0}.filter-field.search {flex:2 1 260px}
.filter-options {flex-basis:100%;border:0;padding:0}
.filter-options>summary {color:var(--muted);min-height:44px;padding:10px 0}
.filter-grid {display:flex;flex-wrap:wrap;gap:14px;padding:16px;
  background:var(--card);border:1px solid var(--line);border-radius:4px}
.filter-options[open]>summary {margin-bottom:0}
.filter-grid .filter-help {margin-top:4px}
.record-id {border:0;padding:0;margin-top:4px}
.record-id>summary {font-size:11px;color:var(--muted);font-weight:400}
.record-id[open]>summary {margin-bottom:4px}
.filter-actions {display:flex;gap:16px;align-items:center;flex-wrap:wrap}
.filter-help {flex-basis:100%;margin:0;font-size:12px;color:var(--muted)}
.pagination {display:flex;justify-content:space-between;align-items:center;gap:14px;
  flex-wrap:wrap;margin:24px 0;font-size:13px}
.pagination-controls {display:flex;gap:16px;align-items:center;flex-wrap:wrap}
.pagination a,.pagination .disabled {display:inline-block;padding:9px 12px;
  border:1px solid var(--line);border-radius:4px;background:white;min-height:44px}
.pagination .disabled {color:var(--muted);background:transparent}
@media(max-width:760px) {
  .workspace {padding:0 18px 30px}.topbar-inner {padding:17px 18px}
  .nav {gap:0 20px;margin-bottom:24px}.nav a {font-size:12px;padding:14px 0 11px}
  .page-head {margin-top:24px}.role {align-self:flex-start}
  .stats {gap:6px 18px}.panel {padding:18px}.table-panel {padding:0}
  table,tbody,tr,td {display:block;width:100%}
  thead {position:absolute;width:1px;height:1px;clip-path:inset(50%);overflow:hidden}
  tbody tr {padding:14px 18px;border-bottom:1px solid var(--line)}
  tbody tr:last-child {border-bottom:0}td {border:0;padding:6px 0}
  td:before {content:attr(data-label);display:block;color:var(--muted);
    font-size:11px;margin-bottom:3px}
  td code {font-size:11px}.approval-card {grid-template-columns:1fr;padding:18px;gap:16px}
  .login-layout {margin:36px auto}.login-panel {padding:24px}
}
@media(prefers-reduced-motion:reduce) {* {scroll-behavior:auto}}
"""


def badge(value: str) -> str:
    tone = (
        "success"
        if value in {"SUCCEEDED", "RESOLVED", "APPROVED", "pass"}
        else "danger"
        if value in {"FAILED", "REJECTED", "TIMED_OUT", "fail", "SEV1"}
        else "warning"
        if value in {"PENDING", "WAITING_FOR_APPROVAL", "ESCALATED", "BLOCKED", "SEV2"}
        else "neutral"
    )
    return f'<span class="badge badge-{tone}">{html.escape(value)}</span>'


def timestamp(value: datetime | None) -> str:
    if value is None:
        return '<span class="muted">Not recorded</span>'
    utc = (value if value.tzinfo else value.replace(tzinfo=UTC)).astimezone(UTC)
    return f'<time datetime="{utc.isoformat()}">{utc:%d %b %Y · %H:%M} UTC</time>'


def stats(items: list[tuple[str, int]]) -> str:
    return (
        '<section class="stats" aria-label="Displayed records">'
        + "".join(
            f'<div class="stat"><strong>{value}</strong><span>{html.escape(label)}</span></div>'
            for label, value in items
        )
        + "</section>"
    )


def empty(title: str, description: str) -> str:
    return (
        f'<div class="empty"><strong>{html.escape(title)}</strong>'
        f"<span>{html.escape(description)}</span></div>"
    )


def browse_controls(kind: BrowseKind, page: BrowsePage, path: str) -> str:
    params = page.params
    search = (
        '<div class="filter-field search"><label for="browse-q">Search</label>'
        f'<input id="browse-q" type="search" name="q" maxlength="160" '
        f'value="{html.escape(params.q, quote=True)}" aria-describedby="browse-help"></div>'
    )
    fields = ""
    for spec in FILTERS[kind]:
        value = dict(params.filters).get(spec.name, "")
        label = f'<label for="browse-{spec.name}">{html.escape(spec.label)}</label>'
        if spec.choices:
            options = '<option value="">All</option>' + "".join(
                f'<option value="{html.escape(choice, quote=True)}"'
                f"{' selected' if choice == value else ''}>{html.escape(choice)}</option>"
                for choice in spec.choices
            )
            control = f'<select id="browse-{spec.name}" name="{spec.name}">{options}</select>'
        else:
            control = (
                f'<input id="browse-{spec.name}" name="{spec.name}" '
                f'maxlength="{spec.max_length}" value="{html.escape(value, quote=True)}">'
            )
        fields += f'<div class="filter-field">{label}{control}</div>'
    sizes = "".join(
        f'<option value="{size}"{" selected" if size == params.page_size else ""}>{size}</option>'
        for size in PAGE_SIZES
    )
    hints = {
        "incidents": "Search incident IDs, titles and services.",
        "approvals": "Search approval IDs, run IDs and tool names.",
        "evaluations": "Search report IDs and modes.",
        "audit": "Search event names, actor IDs and record IDs. Payloads are not searched.",
        "timeline": "Search event names. Oldest records appear first.",
    }
    expanded = bool(params.filters) or params.page_size != (200 if kind == "timeline" else 25)
    filter_label = f"Filters ({len(params.filters)} active)" if params.filters else "Filters"
    return (
        f'<form class="filters" method="get" action="{html.escape(path, quote=True)}" '
        'aria-label="Search and filters">'
        + search
        + '<div class="filter-actions"><button type="submit">Apply filters</button>'
        f'<a href="{html.escape(path, quote=True)}">Reset filters</a></div>'
        f'<p id="browse-help" class="filter-help">{hints[kind]}</p>'
        f'<details class="filter-options"{" open" if expanded else ""}>'
        f'<summary>{filter_label}</summary><div class="filter-grid">'
        + fields
        + '<div class="filter-field"><label for="browse-size">Rows per page</label>'
        f'<select id="browse-size" name="page_size">{sizes}</select></div>'
        '<p class="filter-help">Search treats % and _ literally. ASCII search ignores case; '
        "other case matching follows the database. Text filters marked exact must match fully."
        "</p></div></details></form>"
    )


def pagination(page: BrowsePage, path: str) -> str:
    params = page.params
    first = (params.page - 1) * params.page_size + 1 if page.total else 0
    last = min(params.page * params.page_size, page.total)
    previous = (
        f'<a href="{html.escape(params.url(path, page=params.page - 1), quote=True)}">Previous</a>'
        if params.page > 1
        else '<span class="disabled">Previous</span>'
    )
    following = (
        f'<a href="{html.escape(params.url(path, page=params.page + 1), quote=True)}">Next</a>'
        if params.page < page.pages
        else '<span class="disabled">Next</span>'
    )
    return (
        '<nav class="pagination" aria-label="Pagination">'
        f"<span>{page.total} matching {'record' if page.total == 1 else 'records'} "
        f"· Showing {first}–{last}</span>"
        f'<div class="pagination-controls">{previous}'
        f'<span aria-current="page">Page {params.page} of {page.pages}</span>'
        f"{following}</div></nav>"
    )


def render_page(
    title: str,
    body: str,
    *,
    principal: ApiPrincipal | None = None,
    active: str = "",
    subtitle: str = "",
    browser_session: BrowserSession | None = None,
) -> str:
    """Render only trusted HTML fragments; dynamic text is escaped by callers."""
    brand_target = "/dashboard" if principal is not None else "/dashboard/login"
    navigation = ""
    heading = ""
    account = ""
    if principal is not None:
        if browser_session is not None:
            account = (
                '<div class="account"><span class="identity">Signed in as '
                f'{html.escape(principal.actor_id)}</span><a href="/dashboard/session">Session</a>'
                '<form method="post" action="/dashboard/logout">'
                f'<input type="hidden" name="csrf_token" value="'
                f'{html.escape(browser_session.csrf_token, quote=True)}">'
                '<button type="submit">Sign out</button></form></div>'
            )
        else:
            account = '<span class="account">API key access</span>'
        destinations = [
            ("/dashboard", "Incidents", "incidents:read"),
            ("/dashboard/approvals", "Approvals", "approvals:read"),
            ("/dashboard/evaluations", "Evaluations", "evaluations:read"),
            ("/dashboard/audit", "Audit", "audit:read"),
        ]
        links = []
        for href, label, scope in destinations:
            if principal.allows(scope):
                current = ' aria-current="page"' if href == active else ""
                links.append(f'<a href="{href}"{current}>{label}</a>')
        navigation = '<nav class="nav" aria-label="Main navigation">' + "".join(links) + "</nav>"
        display_title = title.removeprefix("AgentOps ").capitalize()
        heading = (
            '<header class="page-head"><div>'
            f'<h1>{html.escape(display_title)}</h1><p class="subtitle">{html.escape(subtitle)}</p>'
            f'</div><span class="role">{html.escape(principal.role.capitalize())} access</span>'
            "</header>"
        )
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        '<link rel="icon" href="data:image/svg+xml,%3Csvg '
        "xmlns=%22http://www.w3.org/2000/svg%22 viewBox=%220 0 32 32%22%3E"
        "%3Crect width=%2232%22 height=%2232%22 rx=%228%22 fill=%22%231d554c%22/%3E"
        "%3Ctext x=%2216%22 y=%2223%22 text-anchor=%22middle%22 "
        "font-family=%22sans-serif%22 font-size=%2223%22 "
        'fill=%22white%22%3EA%3C/text%3E%3C/svg%3E">'
        f"<title>{html.escape(title)}</title><style>{STYLES}</style></head><body>"
        '<a class="skip-link" href="#main-content">Skip to content</a>'
        '<header class="topbar"><div class="topbar-inner">'
        f'<a class="brand" href="{brand_target}">'
        '<span class="brand-mark" aria-hidden="true">A</span>'
        'AgentOps</a><span class="environment">Local simulator</span>'
        f"{account}</div></header>"
        f'<div class="workspace">{navigation}<main id="main-content" tabindex="-1">'
        f'{heading}{body}</main><footer class="footer"><span>Agent Operations Platform</span>'
        "<span>Local simulator · actions require policy authorization</span></footer></div>"
        "</body></html>"
    )
