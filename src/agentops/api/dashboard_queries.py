"""Bounded, read-only SQL projections for the operator dashboard."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal, get_args
from urllib.parse import urlencode, urlsplit
from uuid import UUID

from sqlalchemy import func, or_
from sqlalchemy.orm import Session
from starlette.datastructures import QueryParams

from agentops.api.schemas import EvaluationMode
from agentops.domain import ActorType, ApprovalStatus, IncidentStatus, Severity
from agentops.persistence import (
    ActionRow,
    ApprovalRequestRow,
    AuditEventRow,
    EvaluationRunRow,
    IncidentRow,
    RunJournalRow,
)

BrowseKind = Literal["incidents", "approvals", "evaluations", "audit", "timeline"]
BrowseRow = IncidentRow | ApprovalRequestRow | EvaluationRunRow | AuditEventRow
PAGE_SIZES = (25, 50, 100, 200)


@dataclass(frozen=True)
class FilterSpec:
    name: str
    label: str
    choices: tuple[str, ...] = ()
    max_length: int = 200


FILTERS: dict[BrowseKind, tuple[FilterSpec, ...]] = {
    "incidents": (
        FilterSpec("status", "Status", tuple(IncidentStatus)),
        FilterSpec("severity", "Severity", tuple(Severity)),
        FilterSpec("service", "Service (exact)"),
    ),
    "approvals": (
        FilterSpec("status", "Status", tuple(ApprovalStatus)),
        FilterSpec("risk", "Risk", ("0", "1", "2", "3")),
        FilterSpec("tool", "Tool (exact)"),
    ),
    "evaluations": (FilterSpec("mode", "Mode", get_args(EvaluationMode)),),
    "audit": (
        FilterSpec("event_type", "Event (exact)", max_length=100),
        FilterSpec("actor_type", "Actor type", tuple(ActorType)),
        FilterSpec("run_id", "Run ID (exact)", max_length=36),
    ),
    "timeline": (FilterSpec("event_type", "Event (exact)", max_length=100),),
}


class BrowseError(ValueError):
    """Invalid navigation input, never an authorization or storage error."""


@dataclass(frozen=True)
class BrowseParams:
    q: str = ""
    page: int = 1
    page_size: int = 25
    filters: tuple[tuple[str, str], ...] = ()

    @property
    def filtered(self) -> bool:
        return bool(self.q or self.filters)

    def url(self, path: str, *, page: int | None = None) -> str:
        values = {"q": self.q, **dict(self.filters)}
        values = {key: value for key, value in values.items() if value}
        values.update(page=str(self.page if page is None else page), page_size=str(self.page_size))
        return path + "?" + urlencode(values)


def parse_browse(query: QueryParams, kind: BrowseKind) -> BrowseParams:
    """Reject ambiguity and unknowns rather than silently dropping user filters."""
    if len(str(query)) > 4096:
        raise BrowseError("The filter query is too long. Reset filters and try again.")
    allowed = {"q", "page", "page_size", *(spec.name for spec in FILTERS[kind])}
    values: dict[str, str] = {}
    for key, value in query.multi_items():
        if key not in allowed or key in values:
            raise BrowseError("Use each supported filter once. Reset filters and try again.")
        if any(ord(char) < 32 or ord(char) == 127 for char in value):
            raise BrowseError("Filters cannot contain control characters.")
        values[key] = value.strip()
    q = values.get("q", "")
    if len(q) > 160:
        raise BrowseError("Search must be at most 160 characters.")
    for key, default in (("page", 1), ("page_size", 200 if kind == "timeline" else 25)):
        raw = values.get(key, str(default))
        if not raw.isascii() or not raw.isdecimal() or len(raw) > 7:
            raise BrowseError("Page and rows per page must be valid positive integers.")
        values[key] = raw
    page, size = int(values["page"]), int(values["page_size"])
    if not 1 <= page <= 1_000_000 or size not in PAGE_SIZES:
        raise BrowseError("Page must be 1–1000000; rows per page must be 25, 50, 100 or 200.")
    filters = []
    for spec in FILTERS[kind]:
        value = values.get(spec.name, "")
        if len(value) > spec.max_length or (value and spec.choices and value not in spec.choices):
            raise BrowseError(f"Choose a valid {spec.label.lower()} filter.")
        if value and spec.name == "run_id":
            try:
                value = str(UUID(value))
            except ValueError as exc:
                raise BrowseError("Run ID must be a valid UUID.") from exc
        if value:
            filters.append((spec.name, value))
    return BrowseParams(q=q, page=page, page_size=size, filters=tuple(filters))


@dataclass(frozen=True)
class BrowsePage:
    rows: list[BrowseRow]
    total: int
    pages: int
    params: BrowseParams


def read_page(
    session: Session, kind: BrowseKind, params: BrowseParams, *, run_id: UUID | None = None
) -> BrowsePage:
    """Filter/count across the complete table before ordering and limiting rows."""
    if kind == "incidents":
        model, date = IncidentRow, IncidentRow.detected_at
        search = (IncidentRow.id, IncidentRow.title, IncidentRow.service_name)
        fields = {
            "status": IncidentRow.status,
            "severity": IncidentRow.severity,
            "service": IncidentRow.service_name,
        }
    elif kind == "approvals":
        model, date = ApprovalRequestRow, ApprovalRequestRow.requested_at
        search = (ApprovalRequestRow.id, ApprovalRequestRow.run_id, ActionRow.tool_name)
        fields = {
            "status": ApprovalRequestRow.status,
            "risk": ActionRow.risk_level,
            "tool": ActionRow.tool_name,
        }
    elif kind == "evaluations":
        model, date = EvaluationRunRow, EvaluationRunRow.created_at
        search = (EvaluationRunRow.id, EvaluationRunRow.mode)
        fields = {"mode": EvaluationRunRow.mode}
    else:
        model, date = AuditEventRow, AuditEventRow.occurred_at
        search = (
            AuditEventRow.id,
            AuditEventRow.event_type,
            AuditEventRow.actor_id,
            AuditEventRow.entity_id,
            AuditEventRow.run_id,
        )
        if kind == "timeline":
            search = (AuditEventRow.event_type,)
        fields = {
            "event_type": AuditEventRow.event_type,
            "actor_type": AuditEventRow.actor_type,
            "run_id": AuditEventRow.run_id,
        }
    query = session.query(model)
    if kind == "approvals":
        query = query.join(ActionRow, ActionRow.id == ApprovalRequestRow.action_id)
    if kind == "timeline":
        if run_id is None:
            raise ValueError("timeline requires an authoritative run ID")
        query = query.filter(AuditEventRow.run_id == str(run_id))
        if not dict(params.filters).get("event_type"):
            # Compact default: completed attempts replace their own start card.
            # A start without an outcome stays visible as an unfinished attempt.
            terminal = (
                session.query(RunJournalRow.id)
                .filter(
                    RunJournalRow.run_id == AuditEventRow.run_id,
                    RunJournalRow.attempt_id
                    == AuditEventRow.payload_json["attempt_id"].as_string(),
                    RunJournalRow.phase != "started",
                )
                .exists()
            )
            query = query.filter(
                or_(
                    ~AuditEventRow.event_type.in_(
                        [
                            "journal.node.started",
                            "journal.tool.started",
                            "journal.llm.started",
                            "journal.run.started",
                            "journal.hypothesis.started",
                        ]
                    ),
                    ~terminal,
                )
            )
    if params.q:
        # Escape before SQL folding; both operands must use the same engine's
        # case semantics. Python lower() disagrees with SQLite for Cyrillic text.
        escaped = params.q.replace("/", "//").replace("%", "/%").replace("_", "/_")
        pattern = func.lower(f"%{escaped}%")
        query = query.filter(or_(*(func.lower(col).like(pattern, escape="/") for col in search)))
    for key, value in params.filters:
        query = query.filter(fields[key] == (int(value) if key == "risk" else value))
    total = query.count()
    pages = max(1, (total + params.page_size - 1) // params.page_size)
    effective = replace(params, page=min(params.page, pages))
    ordering = (
        (date.asc(), model.id.asc()) if kind == "timeline" else (date.desc(), model.id.desc())
    )
    rows = (
        query.order_by(*ordering)
        .offset((effective.page - 1) * params.page_size)
        .limit(params.page_size)
        .all()
    )
    return BrowsePage(rows=rows, total=total, pages=pages, params=effective)


def approval_return_target(value: str | None) -> str:
    """Reconstruct only this fixed local route; never redirect to untrusted URLs."""
    if value is None:
        return "/dashboard/approvals"
    if len(value) > 4096 or any(ord(char) < 32 for char in value):
        raise BrowseError("Invalid approval return context.")
    try:
        target = urlsplit(value)
    except ValueError as exc:
        raise BrowseError("Invalid approval return context.") from exc
    if target.scheme or target.netloc or target.fragment or target.path != "/dashboard/approvals":
        raise BrowseError("Approval return context must point to this approval list.")
    return parse_browse(QueryParams(target.query), "approvals").url("/dashboard/approvals")
