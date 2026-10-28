"""Small authenticated FastAPI control plane.

The API records durable commands and returns stable projections.  It deliberately
does not construct or invoke the workflow graph; a later dispatcher/worker owns
that responsibility.
"""

from __future__ import annotations

import hashlib
import html
import json
import os
import time
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any, Literal
from urllib.parse import parse_qs, urlencode, urlsplit
from uuid import UUID, uuid4

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response
from pydantic import ValidationError
from sqlalchemy import and_, or_, text
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from starlette.concurrency import run_in_threadpool

from agentops.api.auth import (
    ApiKeyAuthenticator,
    ApiPrincipal,
    BrowserSession,
    BrowserSessionManager,
)
from agentops.api.dashboard import (
    badge,
    browse_controls,
    empty,
    pagination,
    render_page,
    stats,
    timestamp,
)
from agentops.api.dashboard_queries import (
    BrowseError,
    approval_return_target,
    parse_browse,
    read_page,
)
from agentops.api.journal_view import audit_payload, timeline_entry
from agentops.api.measurement_views import report_details, usage_details
from agentops.api.schemas import (
    ApprovalDecision,
    ApprovalRead,
    CommandOutcomeRead,
    DemoIncidentCreate,
    EvaluationCreate,
    EvaluationJobRead,
    EvaluationRead,
    IncidentCreate,
    IncidentRead,
    RunCreate,
    RunRead,
    RunStateRead,
    TimelineItem,
)
from agentops.approvals import ApprovalDecisionService, ApprovalRunCancelledError
from agentops.database_config import database_url_from_environment
from agentops.domain import (
    INCIDENT_TRIAGE_WORKFLOW_VERSION,
    AgentRun,
    AgentRunStatus,
    ApprovalStatus,
    Incident,
    IncidentStatus,
    RiskLevel,
    Severity,
    stable_parameter_hash,
)
from agentops.evaluation import run_evaluation
from agentops.evaluation.reports import render_report
from agentops.journal_contract import NAMES, observed_facts
from agentops.observability import MetricsRegistry, MetricsTelemetry, TelemetrySink, emit_safely
from agentops.observability.audit import durable_payload
from agentops.observability.redaction import redact_payload
from agentops.persistence import (
    ActionRow,
    AgentRunRepository,
    AgentRunRow,
    ApiRequestIdempotencyRow,
    ApprovalRepository,
    ApprovalRequestRow,
    AuditEventRow,
    AuditRepository,
    BrowserSessionRevocationRepository,
    CheckpointRow,
    CommandOutcomeRepository,
    Database,
    EntityNotFound,
    EvaluationJobRepository,
    EvaluationJobRow,
    EvaluationRunRepository,
    EvaluationRunRow,
    HypothesisLineageRepository,
    IncidentRepository,
    IncidentRow,
    ObservationRow,
    OutboxRow,
    RunJournalRow,
    SimulationEnvironmentRepository,
)
from agentops.simulator import ScenarioName, Simulator
from agentops.usage import read_run_metrics

# FastAPI dependency declarations intentionally call Depends in defaults.
# ruff: noqa: B008


def _error(status: int, code: str, message: str) -> None:
    raise HTTPException(status_code=status, detail={"code": code, "message": message})


def _json_model(model: Any) -> dict[str, Any]:
    return model.model_dump(mode="json")


def _incident_read(incident: Incident) -> IncidentRead:
    return IncidentRead.model_validate(incident.model_dump())


def _run_read(run: AgentRun) -> RunRead:
    return RunRead.model_validate(run.model_dump())


def _evaluation_read(row: EvaluationRunRow) -> EvaluationRead:
    return EvaluationRead(
        id=UUID(row.id),
        mode=row.mode,
        seed=row.seed,
        manifest_version=row.manifest_version,
        manifest_digest=row.manifest_digest,
        source_revision=row.source_revision,
        cohort=row.cohort,
        denominator=row.denominator,
        pass_count=row.pass_count,
        fail_count=row.fail_count,
        unavailable_count=row.unavailable_count,
        created_by=row.created_by,
        created_at=(
            row.created_at
            if row.created_at.tzinfo is not None
            else row.created_at.replace(tzinfo=UTC)
        ),
    )


def _evaluation_job_read(row: EvaluationJobRow) -> EvaluationJobRead:
    return EvaluationJobRead(
        id=UUID(row.id),
        mode=row.mode,
        seed=row.seed,
        status=row.status,
        created_by=row.actor_id,
        created_at=row.created_at if row.created_at.tzinfo else row.created_at.replace(tzinfo=UTC),
        error_code=row.error_code,
        evaluation_run_id=UUID(row.evaluation_run_id) if row.evaluation_run_id else None,
    )


def _evaluation_job_response(session: Any, row: EvaluationJobRow, fingerprint: str) -> JSONResponse:
    if row.request_fingerprint != fingerprint:
        _error(409, "IdempotencyConflict", "request key is bound to a different request")
    if row.status == "SUCCEEDED":
        saved = EvaluationRunRepository(session).require(row.id)
        return JSONResponse(
            status_code=201,
            content=_json_model(_evaluation_read(saved)),
            headers={"X-Idempotent-Replay": "true"},
        )
    if row.status in {"FAILED", "QUARANTINED"}:
        return JSONResponse(
            status_code=503,
            content=_json_model(_evaluation_job_read(row)),
            headers={"X-Idempotent-Replay": "true"},
        )
    return JSONResponse(
        status_code=202,
        content=_json_model(_evaluation_job_read(row)),
        headers={
            "X-Idempotent-Replay": "true",
            "Retry-After": "1",
            "Location": f"/api/v1/evaluation-jobs/{row.id}",
        },
    )


def _html_json(value: Any) -> str:
    return html.escape(json.dumps(redact_payload(value), sort_keys=True), quote=True)


def _audit_and_outbox(
    session: Any,
    *,
    event_key: str,
    event_type: str,
    aggregate_type: str,
    aggregate_id: UUID | str,
    actor_id: str,
    payload: dict[str, Any],
    occurred_at: datetime | None = None,
    telemetry: TelemetrySink | None = None,
) -> None:
    now = occurred_at or datetime.now(UTC)
    session.add(
        AuditEventRow(
            id=str(uuid4()),
            event_key=event_key,
            occurred_at=now,
            actor_type="user",
            actor_id=actor_id,
            event_type=event_type,
            entity_type=aggregate_type,
            entity_id=str(aggregate_id),
            run_id=str(payload["run_id"]) if payload.get("run_id") else None,
            payload_json=durable_payload(payload),
        )
    )
    emit_safely(telemetry, event_type=event_type, payload=payload)
    session.add(
        OutboxRow(
            id=str(uuid4()),
            event_key=event_key,
            event_type=event_type,
            aggregate_type=aggregate_type,
            aggregate_id=str(aggregate_id),
            payload_json=payload,
            created_at=now,
        )
    )


def _request_key(request: Request) -> str:
    value = request.headers.get("Idempotency-Key")
    if value is None or not value.strip():
        return f"auto-{uuid4()}"
    if len(value) > 200:
        _error(422, "InvalidIdempotencyKey", "Idempotency-Key is too long")
    return value.strip()


def _stored_response(
    session: Any,
    *,
    actor_id: str,
    operation: str,
    target_id: str,
    key: str,
    fingerprint: str,
) -> ApiRequestIdempotencyRow | None:
    row = (
        session.query(ApiRequestIdempotencyRow)
        .filter_by(
            actor_id=actor_id,
            operation=operation,
            target_id=target_id,
            idempotency_key=key,
        )
        .one_or_none()
    )
    if row is not None and row.request_fingerprint != fingerprint:
        _error(409, "IdempotencyConflict", "request key is bound to a different request")
    return row


def _save_response_binding(
    session: Any,
    *,
    actor_id: str,
    operation: str,
    target_id: str,
    key: str,
    fingerprint: str,
    resource_type: str,
    resource_id: UUID | str,
    status: int,
    body: dict[str, Any],
) -> None:
    session.add(
        ApiRequestIdempotencyRow(
            id=str(uuid4()),
            actor_id=actor_id,
            operation=operation,
            target_id=target_id,
            idempotency_key=key,
            request_fingerprint=fingerprint,
            resource_type=resource_type,
            resource_id=str(resource_id),
            response_status=status,
            response_json=body,
            created_at=datetime.now(UTC),
        )
    )


def _request_api_key(request: Request) -> str | None:
    api_key = request.headers.get("X-API-Key")
    if not api_key:
        authorization = request.headers.get("Authorization", "")
        if authorization.lower().startswith("bearer "):
            api_key = authorization[7:].strip()
    return api_key


def get_principal(request: Request) -> ApiPrincipal:
    api_key = _request_api_key(request)
    principal = request.app.state.authenticator.authenticate(api_key)
    if principal is None:
        _error(401, "Unauthenticated", "a valid API key is required")
    return principal


def _record_scope_denial(request: Request, principal: ApiPrincipal, scope: str) -> None:
    # Record identified approval attempts without request bodies, keys or cookies.
    if scope != "approvals:decide":
        return
    try:
        approval_id = str(UUID(str(request.path_params.get("approval_id", ""))))
    except ValueError:
        return
    with _database(request).session() as session:
        approval = session.get(ApprovalRequestRow, approval_id)
        session.add(
            AuditEventRow(
                id=str(uuid4()),
                event_key=f"authorization-denied:{uuid4()}",
                occurred_at=datetime.now(UTC),
                actor_type="user",
                actor_id=principal.actor_id,
                event_type="authorization.denied",
                entity_type="approval",
                entity_id=approval_id,
                run_id=approval.run_id if approval is not None else None,
                payload_json=durable_payload(
                    {
                        "schema_version": 1,
                        "role": principal.role,
                        "required_scope": scope,
                        "method": request.method,
                        "action_id": approval.action_id if approval is not None else None,
                    }
                ),
            )
        )


def require_scope(scope: str) -> Callable[..., ApiPrincipal]:
    def dependency(
        request: Request, principal: ApiPrincipal = Depends(get_principal)
    ) -> ApiPrincipal:
        if not principal.allows(scope):
            _record_scope_denial(request, principal, scope)
            _error(403, "Forbidden", f"scope {scope} is required")
        return principal

    return dependency


def get_browser_principal(request: Request) -> ApiPrincipal:
    principal = request.app.state.authenticator.authenticate(_request_api_key(request))
    if principal is not None:
        return principal
    session = _active_browser_session(request)
    if session is None:
        _error(401, "Unauthenticated", "a valid browser session or API key is required")
    request.state.browser_session = session
    return session.principal


def _session_digest(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _active_browser_session(request: Request) -> BrowserSession | None:
    token = request.cookies.get(BrowserSessionManager.cookie_name)
    verified = request.app.state.browser_sessions.authenticate(token)
    if verified is None:
        return None
    try:
        with _database(request).session() as session:
            revoked = BrowserSessionRevocationRepository(session).is_revoked(_session_digest(token))
    except SQLAlchemyError:
        _error(503, "BrowserSessionUnavailable", "browser session storage is unavailable")
    return None if revoked else verified


def require_browser_scope(scope: str) -> Callable[..., ApiPrincipal]:
    def dependency(principal: ApiPrincipal = Depends(get_browser_principal)) -> ApiPrincipal:
        if not principal.allows(scope):
            _error(403, "Forbidden", f"scope {scope} is required")
        return principal

    return dependency


def require_browser_session_scope(scope: str) -> Callable[..., ApiPrincipal]:
    def dependency(request: Request) -> ApiPrincipal:
        session = _active_browser_session(request)
        if session is None:
            _error(401, "Unauthenticated", "a valid browser session is required")
        if not session.principal.allows(scope):
            _record_scope_denial(request, session.principal, scope)
            _error(403, "Forbidden", f"scope {scope} is required")
        request.state.browser_session = session
        return session.principal

    return dependency


async def _form_fields(request: Request, *, reject_duplicates: bool = False) -> dict[str, str]:
    content_type = request.headers.get("Content-Type", "").split(";", 1)[0].strip().lower()
    if content_type != "application/x-www-form-urlencoded":
        _error(415, "UnsupportedMediaType", "form content type is required")
    body = await request.body()
    if len(body) > 16_384:
        _error(413, "RequestTooLarge", "form body is too large")
    try:
        parsed = parse_qs(
            body.decode("utf-8"),
            keep_blank_values=True,
            strict_parsing=True,
            max_num_fields=20,
        )
    except (UnicodeDecodeError, ValueError):
        _error(422, "InvalidForm", "form body is invalid")
    if reject_duplicates and any(len(values) != 1 for values in parsed.values()):
        _error(422, "InvalidForm", "form fields must occur once")
    return {key: values[-1] for key, values in parsed.items() if values}


def _check_login_origin(request: Request) -> None:
    if request.headers.get("Sec-Fetch-Site", "").lower() == "cross-site":
        _error(403, "CsrfRejected", "cross-site login is forbidden")
    origin = request.headers.get("Origin")
    if origin is None:
        _error(403, "CsrfRejected", "browser login requires an Origin header")
    try:
        source, target = urlsplit(origin), urlsplit(str(request.url))
        source_port = source.port or (443 if source.scheme == "https" else 80)
        target_port = target.port or (443 if target.scheme == "https" else 80)
        valid = (
            source.scheme in {"http", "https"}
            and source.hostname is not None
            and not source.username
            and not source.password
            and not source.query
            and not source.fragment
            and source.path in {"", "/"}
            and (source.scheme, source.hostname, source_port)
            == (target.scheme, target.hostname, target_port)
            and origin.rstrip("/") in request.app.state.browser_origins
        )
    except ValueError:
        valid = False
    if not valid:
        _error(403, "CsrfRejected", "login origin does not match the application")


def _database(request: Request) -> Database:
    database = request.app.state.database
    if database is None:
        _error(503, "DatabaseUnavailable", "database is not configured")
    return database


def _dashboard_page(request: Request, title: str, body: str, **kwargs: Any) -> str:
    return render_page(
        title, body, browser_session=getattr(request.state, "browser_session", None), **kwargs
    )


_APPROVAL_NOTICE_COOKIE = "agentops_approval_notice"
_DEMO_SCOPES = ("incidents:read", "incidents:write", "runs:read", "runs:start")


def _approval_feedback(session: Any, request: Request, principal: ApiPrincipal) -> str:
    """A short-lived cookie is only a pointer to the actor's saved decision."""
    if getattr(request.state, "browser_session", None) is None:
        return ""
    try:
        approval_id = str(UUID(request.cookies.get(_APPROVAL_NOTICE_COOKIE, "")))
    except ValueError:
        return ""
    row = session.get(ApprovalRequestRow, approval_id)
    if row is None or row.decided_by != principal.actor_id or row.decided_at is None:
        return ""
    decided_at = (
        row.decided_at.replace(tzinfo=UTC) if row.decided_at.tzinfo is None else row.decided_at
    )
    if not 0 <= (datetime.now(UTC) - decided_at).total_seconds() <= 120:
        return ""
    messages = {
        "APPROVED": "Request approved. Check the run for its outcome.",
        "REJECTED": "Request rejected. The proposed action was not authorized.",
    }
    message = messages.get(row.status)
    if message is None:
        return ""
    link = (
        f' <a href="/dashboard/runs/{html.escape(row.run_id, quote=True)}">View run</a>'
        if principal.allows("runs:read")
        else ""
    )
    return f'<p class="feedback" role="status">{message}{link}</p>'


def _revoke_current_browser_cookie(request: Request) -> None:
    token = request.cookies.get(BrowserSessionManager.cookie_name)
    now = int(time.time())
    verified = request.app.state.browser_sessions.authenticate(token, now=now)
    try:
        with _database(request).session() as session:
            repository = BrowserSessionRevocationRepository(session)
            repository.prune_expired(now)
            if verified is not None:
                repository.revoke(_session_digest(token), expires_at=verified.expires_at, now=now)
    except SQLAlchemyError:
        _error(503, "BrowserSessionUnavailable", "browser session storage is unavailable")


def create_app(
    database: Database | None = None,
    *,
    api_keys: dict[str, Any] | None = None,
    auto_create_schema: bool = False,
    telemetry: TelemetrySink | None = None,
    session_secret: str | None = None,
    session_cookie_secure: bool | None = None,
    browser_origins: tuple[str, ...] | None = None,
) -> FastAPI:
    """Build an API app; schema creation is opt-in for local tests only."""

    app = FastAPI(title="AgentOps Control API", version="0.1.0")
    app.state.database = database or Database(database_url_from_environment())
    app.state.browser_origins = frozenset(
        browser_origins
        if browser_origins is not None
        else os.getenv(
            "AGENTOPS_BROWSER_ORIGINS", "http://localhost:8000,http://127.0.0.1:8000"
        ).split(",")
    )
    app.state.authenticator = (
        ApiKeyAuthenticator(api_keys)
        if api_keys is not None
        else ApiKeyAuthenticator.from_environment()
    )
    app.state.browser_sessions = (
        BrowserSessionManager(session_secret)
        if session_secret is not None
        else BrowserSessionManager.from_environment()
    )
    app.state.session_cookie_secure = (
        session_cookie_secure
        if session_cookie_secure is not None
        else os.getenv("AGENTOPS_SESSION_COOKIE_SECURE", "false").lower()
        in {"1", "true", "yes", "on"}
    )
    app.state.metrics = (
        telemetry.metrics if isinstance(telemetry, MetricsTelemetry) else MetricsRegistry()
    )
    app.state.telemetry = telemetry or MetricsTelemetry(app.state.metrics)
    if auto_create_schema and app.state.database is not None:
        app.state.database.create_schema()

    @app.middleware("http")
    async def request_id_middleware(request: Request, call_next: Any) -> Any:
        request_id = request.headers.get("X-Request-ID") or str(uuid4())
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        if request.url.path == "/dashboard" or request.url.path.startswith("/dashboard/"):
            response.headers["Cache-Control"] = "no-store"
        return response

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException) -> Response:
        if (
            exc.status_code == 401
            and request.method == "GET"
            and (request.url.path == "/dashboard" or request.url.path.startswith("/dashboard/"))
        ):
            return HTMLResponse(
                render_page(
                    "AgentOps sign in required",
                    '<section class="panel"><h1>Sign in required</h1>'
                    '<p role="alert">Your session is missing, expired or no longer valid. '
                    "Sign in again to continue.</p>"
                    '<a href="/dashboard/login">Sign in</a></section>',
                ),
                status_code=401,
            )
        detail = (
            exc.detail
            if isinstance(exc.detail, dict)
            else {"code": "HttpError", "message": str(exc.detail)}
        )
        body = {
            "error": {
                "code": detail.get("code", "HttpError"),
                "message": detail.get("message", "request failed"),
                "request_id": getattr(request.state, "request_id", ""),
            }
        }
        return JSONResponse(status_code=exc.status_code, content=body)

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content={
                "error": {
                    "code": "InvalidRequest",
                    "message": "request validation failed",
                    "request_id": getattr(request.state, "request_id", ""),
                    "details": jsonable_encoder(exc.errors()),
                }
            },
        )

    @app.exception_handler(BrowseError)
    async def browse_exception_handler(request: Request, exc: BrowseError) -> HTMLResponse:
        path = request.url.path
        if path.startswith("/dashboard/approvals/"):
            path = "/dashboard/approvals"
        body = (
            '<section class="panel"><h1>Check your filters</h1>'
            f'<p role="alert">{html.escape(str(exc))}</p>'
            f'<a href="{html.escape(path, quote=True)}">Reset filters</a></section>'
        )
        return HTMLResponse(
            _dashboard_page(
                request,
                "AgentOps filters",
                body,
                principal=getattr(request.state, "dashboard_principal", None),
            ),
            status_code=422,
        )

    @app.get("/health")
    @app.get("/healthz")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/ready")
    @app.get("/readyz")
    async def ready(request: Request) -> dict[str, str]:
        database = request.app.state.database
        if database is None:
            _error(503, "DatabaseUnavailable", "database is not configured")
        try:
            with database.engine.connect() as connection:
                connection.execute(text("SELECT 1 FROM incidents LIMIT 1"))
        except Exception:
            _error(503, "DatabaseUnavailable", "database is unavailable")
        return {"status": "ready"}

    @app.post("/api/v1/incidents", response_model=IncidentRead, status_code=201)
    async def create_incident(
        payload: IncidentCreate,
        request: Request,
        principal: ApiPrincipal = Depends(require_scope("incidents:write")),
    ) -> JSONResponse:
        database = _database(request)
        key = _request_key(request)
        fingerprint = stable_parameter_hash(payload.model_dump(mode="json"))
        body: dict[str, Any]
        with database.session() as session:
            existing = _stored_response(
                session,
                actor_id=principal.actor_id,
                operation="create_incident",
                target_id="collection",
                key=key,
                fingerprint=fingerprint,
            )
            if existing is not None:
                return JSONResponse(
                    status_code=existing.response_status,
                    content=existing.response_json,
                    headers={"X-Idempotent-Replay": "true"},
                )
            incident = Incident(created_by=principal.actor_id, **payload.model_dump())
            IncidentRepository(session).save(incident)
            body = _json_model(_incident_read(incident))
            event_payload = {"incident_id": str(incident.id)}
            _audit_and_outbox(
                session,
                telemetry=request.app.state.telemetry,
                event_key=f"incident:{incident.id}:created",
                event_type="incident.created",
                aggregate_type="incident",
                aggregate_id=incident.id,
                actor_id=principal.actor_id,
                payload=event_payload,
            )
            _save_response_binding(
                session,
                actor_id=principal.actor_id,
                operation="create_incident",
                target_id="collection",
                key=key,
                fingerprint=fingerprint,
                resource_type="incident",
                resource_id=incident.id,
                status=201,
                body=body,
            )
        return JSONResponse(status_code=201, content=body)

    @app.get("/api/v1/incidents", response_model=list[IncidentRead])
    async def list_incidents(
        request: Request,
        principal: ApiPrincipal = Depends(require_scope("incidents:read")),
        limit: int = Query(50, ge=1, le=200),
    ) -> list[IncidentRead]:
        database = _database(request)
        with database.session() as session:
            rows = (
                session.query(IncidentRow)
                .order_by(IncidentRow.detected_at.desc())
                .limit(limit)
                .all()
            )
            return [_incident_read(row.to_domain()) for row in rows]

    @app.get("/api/v1/incidents/{incident_id}", response_model=IncidentRead)
    async def get_incident(
        incident_id: UUID,
        request: Request,
        principal: ApiPrincipal = Depends(require_scope("incidents:read")),
    ) -> IncidentRead:
        database = _database(request)
        with database.session() as session:
            incident = IncidentRepository(session).get(incident_id)
            if incident is None:
                _error(404, "NotFound", "incident not found")
            return _incident_read(incident)

    @app.post("/api/v1/incidents/{incident_id}/reopen", response_model=IncidentRead)
    async def reopen_incident(
        incident_id: UUID,
        request: Request,
        principal: ApiPrincipal = Depends(require_scope("incidents:write")),
    ) -> IncidentRead:
        database = _database(request)
        with database.session() as session:
            incident = IncidentRepository(session).get(incident_id)
            if incident is None:
                _error(404, "NotFound", "incident not found")
            try:
                incident.transition_to(IncidentStatus.OPEN)
            except Exception as exc:
                _error(409, "InvalidTransition", str(exc))
            IncidentRepository(session).save(incident)
            _audit_and_outbox(
                session,
                telemetry=request.app.state.telemetry,
                event_key=f"incident:{incident.id}:reopened:{incident.version}",
                event_type="incident.reopened",
                aggregate_type="incident",
                aggregate_id=incident.id,
                actor_id=principal.actor_id,
                payload={"incident_id": str(incident.id)},
            )
            return _incident_read(incident)

    @app.post("/api/v1/incidents/{incident_id}/runs", response_model=RunRead, status_code=202)
    async def create_run(
        incident_id: UUID,
        request: Request,
        payload: RunCreate | None = None,
        principal: ApiPrincipal = Depends(require_scope("runs:start")),
    ) -> JSONResponse:
        payload = payload or RunCreate()
        if payload.risk_level is RiskLevel.R3:
            _error(403, "R3Forbidden", "R3 actions cannot be started through the control API")
        database = _database(request)
        key = _request_key(request)
        fingerprint = stable_parameter_hash(payload.model_dump(mode="json"))
        with database.session() as session:
            incident = IncidentRepository(session).get(incident_id)
            if incident is None:
                _error(404, "NotFound", "incident not found")
            existing = _stored_response(
                session,
                actor_id=principal.actor_id,
                operation="create_run",
                target_id=str(incident_id),
                key=key,
                fingerprint=fingerprint,
            )
            if existing is not None:
                return JSONResponse(
                    status_code=existing.response_status,
                    content=existing.response_json,
                    headers={"X-Idempotent-Replay": "true"},
                )
            if incident.active_run_id is not None:
                active = AgentRunRepository(session).get(incident.active_run_id)
                if active is not None and active.status not in {
                    AgentRunStatus.SUCCEEDED,
                    AgentRunStatus.FAILED,
                    AgentRunStatus.TIMED_OUT,
                    AgentRunStatus.CANCELLED,
                    AgentRunStatus.BLOCKED,
                }:
                    _error(409, "RunAlreadyActive", "incident already has an active run")
            if (
                payload.workflow_name != "incident_triage"
                or payload.workflow_version != INCIDENT_TRIAGE_WORKFLOW_VERSION
            ):
                _error(422, "UnsupportedRuntimeContract", "requested workflow is not supported")
            run = AgentRun(
                incident_id=incident.id,
                environment_id=payload.environment_id,
                workflow_name=payload.workflow_name,
                workflow_version=payload.workflow_version,
            )
            run.transition_to(AgentRunStatus.QUEUED)
            incident.active_run_id = run.id
            incident.version += 1
            IncidentRepository(session).save(incident)
            AgentRunRepository(session).save(run)
            body = _json_model(_run_read(run))
            event_payload = {
                "run_id": str(run.id),
                "incident_id": str(incident.id),
                "environment_id": str(run.environment_id) if run.environment_id else None,
                "configuration": payload.model_dump(mode="json"),
            }
            _audit_and_outbox(
                session,
                telemetry=request.app.state.telemetry,
                event_key=f"run:{run.id}:created",
                event_type="run.created",
                aggregate_type="agent_run",
                aggregate_id=run.id,
                actor_id=principal.actor_id,
                payload=event_payload,
            )
            _save_response_binding(
                session,
                actor_id=principal.actor_id,
                operation="create_run",
                target_id=str(incident_id),
                key=key,
                fingerprint=fingerprint,
                resource_type="agent_run",
                resource_id=run.id,
                status=202,
                body=body,
            )
        return JSONResponse(status_code=202, content=body)

    @app.get("/api/v1/runs/{run_id}", response_model=RunRead)
    async def get_run(
        run_id: UUID,
        request: Request,
        principal: ApiPrincipal = Depends(require_scope("runs:read")),
    ) -> RunRead:
        database = _database(request)
        with database.session() as session:
            run = AgentRunRepository(session).get(run_id)
            if run is None:
                _error(404, "NotFound", "run not found")
            return _run_read(run)

    @app.get("/api/v1/runs/{run_id}/metrics")
    async def run_metrics(
        run_id: UUID,
        request: Request,
        principal: ApiPrincipal = Depends(require_scope("runs:read")),
    ) -> dict[str, Any]:
        with _database(request).session() as session:
            if AgentRunRepository(session).get(run_id) is None:
                _error(404, "NotFound", "run not found")
            return read_run_metrics(session, run_id)

    @app.get(
        "/api/v1/runs/{run_id}/command-outcomes",
        response_model=list[CommandOutcomeRead],
    )
    async def run_command_outcomes(
        run_id: UUID,
        request: Request,
        principal: ApiPrincipal = Depends(require_scope("runs:read")),
        limit: int = Query(200, ge=1, le=500),
    ) -> list[CommandOutcomeRead]:
        database = _database(request)
        with database.session() as session:
            if AgentRunRepository(session).get(run_id) is None:
                _error(404, "NotFound", "run not found")
            rows = CommandOutcomeRepository(session).list_for_run(run_id, limit=limit)
            return [
                CommandOutcomeRead(
                    event_id=UUID(row.event_id),
                    event_key=row.event_key,
                    event_type=row.event_type,
                    aggregate_type=row.aggregate_type,
                    aggregate_id=row.aggregate_id,
                    run_id=UUID(row.run_id) if row.run_id else None,
                    status=row.status,
                    attempt=row.attempt,
                    result=row.result_json,
                    error_code=row.error_code,
                    error_message=row.error_message,
                    created_at=row.created_at,
                    updated_at=row.updated_at,
                )
                for row in rows
            ]

    @app.get("/api/v1/runs/{run_id}/timeline", response_model=list[TimelineItem])
    async def run_timeline(
        run_id: UUID,
        request: Request,
        principal: ApiPrincipal = Depends(require_scope("runs:read")),
        limit: int = Query(200, ge=1, le=500),
    ) -> list[TimelineItem]:
        database = _database(request)
        with database.session() as session:
            if session.get(AgentRunRow, str(run_id)) is None:
                _error(404, "NotFound", "run not found")
            rows = AuditRepository(session).list_for_run(run_id, limit=limit)
            return [
                TimelineItem(
                    event_type=row.event_type,
                    entity_type=row.entity_type,
                    entity_id=UUID(row.entity_id) if row.entity_id else None,
                    actor_id=row.actor_id,
                    occurred_at=row.occurred_at,
                    payload=audit_payload(row),
                )
                for row in rows
            ]

    @app.get("/api/v1/runs/{run_id}/state", response_model=RunStateRead)
    async def run_state(
        run_id: UUID,
        request: Request,
        principal: ApiPrincipal = Depends(require_scope("runs:read")),
    ) -> RunStateRead:
        database = _database(request)
        with database.session() as session:
            run = AgentRunRepository(session).get(run_id)
            if run is None:
                _error(404, "NotFound", "run not found")
            checkpoint = session.query(CheckpointRow).filter_by(run_id=str(run_id)).one_or_none()
            approval_id = (
                UUID(checkpoint.approval_id) if checkpoint and checkpoint.approval_id else None
            )
            action_id = UUID(checkpoint.action_id) if checkpoint and checkpoint.action_id else None
            return RunStateRead(
                run_id=run.id,
                status=run.status,
                phase=checkpoint.phase if checkpoint else None,
                action_id=action_id,
                approval_id=approval_id,
                last_error_code=run.last_error_code,
                last_error_message=run.last_error_message,
            )

    @app.get("/api/v1/runs/{run_id}/hypotheses")
    async def run_hypotheses(
        run_id: UUID,
        request: Request,
        principal: ApiPrincipal = Depends(require_scope("runs:read")),
        hypothesis_id: UUID | None = None,
        history: bool = False,
        offset: int = Query(0, ge=0, le=10000),
        limit: int = Query(50, ge=1, le=100),
    ) -> list[dict[str, Any]]:
        with _database(request).session() as session:
            if AgentRunRepository(session).get(run_id) is None:
                _error(404, "NotFound", "run not found")
            repo = HypothesisLineageRepository(session)
            items = repo.versions(run_id, hypothesis_id) if history else repo.heads(run_id)
            if hypothesis_id is not None:
                items = [item for item in items if item.id == hypothesis_id]
                if not items:
                    _error(404, "NotFound", "hypothesis not found in this run")
            return [item.model_dump(mode="json") for item in items[offset : offset + limit]]

    @app.get("/dashboard/runs/{run_id}/hypotheses/{hypothesis_id}", response_class=HTMLResponse)
    async def dashboard_hypothesis(
        run_id: UUID,
        hypothesis_id: UUID,
        request: Request,
        principal: ApiPrincipal = Depends(require_browser_scope("runs:read")),
        page: int = Query(1, ge=1, le=10000),
    ) -> HTMLResponse:
        request.state.dashboard_principal = principal
        with _database(request).session() as session:
            versions = sorted(
                HypothesisLineageRepository(session).versions(run_id, hypothesis_id),
                key=lambda item: item.version,
            )
            if not versions:
                _error(404, "NotFound", "hypothesis not found in this run")
        size = 25
        pages = max(1, (len(versions) + size - 1) // size)
        page = min(page, pages)
        entries = []
        for item in versions[(page - 1) * size : page * size]:

            def evidence_links(ids):
                return (
                    " ".join(
                        f'<a href="/dashboard/runs/{run_id}/evidence/{identity}">'
                        f"Observation {index + 1}</a>"
                        for index, identity in enumerate(ids)
                    )
                    or "None recorded"
                )

            entries.append(
                "<li>"
                + timestamp(item.updated_at)
                + f"<h2>Version {item.version} · {badge(item.status.upper())}</h2>"
                + f'<p style="overflow-wrap:anywhere">{html.escape(item.summary)}</p>'
                + f'<p class="muted">Model score {item.confidence:.2f} '
                + "— uncalibrated; not a probability.</p>"
                + "<details><summary>Evidence and decision</summary>"
                + f"<p>Supporting: {evidence_links(item.supporting_observation_ids)}</p>"
                + f"<p>Contradicting: {evidence_links(item.contradicting_observation_ids)}</p>"
                + f'<p><a href="/dashboard/runs/{run_id}/journal/{item.decision_id}">'
                + "Structured model decision</a></p>"
                + f'<p class="muted">Hypothesis <code>{item.id}</code> '
                + f"· version <code>{item.version_id}</code></p>"
                + "</details></li>"
            )
        path = f"/dashboard/runs/{run_id}/hypotheses/{hypothesis_id}"
        nav = f'<a href="{path}?page={page - 1}">Previous</a> ' if page > 1 else ""
        nav += f"Page {page} of {pages}"
        nav += f' <a href="{path}?page={page + 1}">Next</a>' if page < pages else ""
        return HTMLResponse(
            _dashboard_page(
                request,
                "Hypothesis history",
                f'<p><a href="/dashboard/runs/{run_id}">Back to run</a></p>'
                + '<section class="panel"><ol class="timeline">'
                + "".join(entries)
                + "</ol>"
                + f'<nav aria-label="Hypothesis pagination">{nav}</nav></section>',
                principal=principal,
                active="/dashboard",
                subtitle="Recorded versions and evidence polarity.",
            )
        )

    @app.get("/dashboard/runs/{run_id}/evidence/{observation_id}", response_class=HTMLResponse)
    async def dashboard_observation(
        run_id: UUID,
        observation_id: UUID,
        request: Request,
        principal: ApiPrincipal = Depends(require_browser_scope("runs:read")),
    ) -> HTMLResponse:
        request.state.dashboard_principal = principal
        with _database(request).session() as session:
            row = session.get(ObservationRow, str(observation_id))
            if row is None or row.run_id != str(run_id):
                _error(404, "NotFound", "observation not found in this run")
            facts = observed_facts([{"observation_id": row.id, "payload": row.payload_json}])
            source = row.source_tool if row.source_tool in NAMES else "unknown"
            attempt = (
                session.query(RunJournalRow)
                .filter_by(
                    run_id=str(run_id), attempt_id=row.tool_call_id, kind="tool", phase="completed"
                )
                .one_or_none()
                if row.tool_call_id
                else None
            )
        body = (
            f'<p><a href="/dashboard/runs/{run_id}">Back to run</a></p>'
            + '<section class="panel"><h2>'
            + html.escape(source.replace("_", " "))
            + "</h2>"
            + timestamp(row.observed_at)
            + "".join(
                f'<p style="overflow-wrap:anywhere">{html.escape(fact.label)}: '
                f"<strong>{html.escape(str(fact.value))}</strong></p>"
                for fact in facts
            )
            + (
                '<p class="muted">No supported scalar preview is available '
                "for this observation.</p>"
                if not facts
                else ""
            )
            + (
                f'<p><a href="/dashboard/runs/{run_id}/journal/{attempt.id}">'
                "Recorded tool attempt</a></p>"
                if attempt
                else '<p class="muted">No individual tool attempt was recorded.</p>'
            )
            + "<details><summary>Observation identity</summary>"
            + f"<code>{observation_id}</code></details></section>"
        )
        return HTMLResponse(
            _dashboard_page(
                request,
                "Observed evidence",
                body,
                principal=principal,
                active="/dashboard",
                subtitle="Safe typed facts; raw tool text is not displayed.",
            )
        )

    @app.get("/dashboard/runs/{run_id}/journal/{event_id}")
    async def dashboard_journal_reference(
        run_id: UUID,
        event_id: UUID,
        request: Request,
        principal: ApiPrincipal = Depends(require_browser_scope("runs:read")),
    ) -> RedirectResponse:
        with _database(request).session() as session:
            event = session.get(RunJournalRow, str(event_id))
            if (
                event is None
                or event.run_id != str(run_id)
                or event.phase != "completed"
                or event.kind not in {"tool", "llm", "hypothesis"}
            ):
                _error(404, "NotFound", "recorded decision or attempt not found in this run")
            event_type = f"journal.{event.kind}.{event.phase}"
            preceding = (
                session.query(AuditEventRow)
                .filter(
                    AuditEventRow.run_id == str(run_id),
                    AuditEventRow.event_type == event_type,
                    or_(
                        AuditEventRow.occurred_at < event.occurred_at,
                        and_(
                            AuditEventRow.occurred_at == event.occurred_at,
                            AuditEventRow.id < event.id,
                        ),
                    ),
                )
                .count()
            )
        query = urlencode(
            {"event_type": event_type, "page_size": 200, "page": preceding // 200 + 1}
        )
        return RedirectResponse(
            f"/dashboard/runs/{run_id}?{query}#event-{event_id}", status_code=303
        )

    @app.post("/api/v1/runs/{run_id}/cancel", response_model=RunRead, status_code=202)
    async def cancel_run(
        run_id: UUID,
        request: Request,
        principal: ApiPrincipal = Depends(require_scope("runs:cancel")),
    ) -> JSONResponse:
        database = _database(request)
        with database.session() as session:
            run = AgentRunRepository(session).get_for_update(run_id)
            if run is None:
                _error(404, "NotFound", "run not found")
            if run.status is not AgentRunStatus.CANCELLED:
                if run.status in {
                    AgentRunStatus.SUCCEEDED,
                    AgentRunStatus.FAILED,
                    AgentRunStatus.TIMED_OUT,
                    AgentRunStatus.BLOCKED,
                }:
                    _error(409, "RunTerminal", "terminal run cannot be cancelled")
                requested_at = datetime.now(UTC)
                created = run.request_cancel(requested_at=requested_at)
                if run.status in {
                    AgentRunStatus.QUEUED,
                    AgentRunStatus.WAITING_FOR_APPROVAL,
                    AgentRunStatus.WAITING_FOR_RETRY,
                }:
                    run.transition_to(AgentRunStatus.CANCELLED)
                AgentRunRepository(session).save(run)
                if created:
                    _audit_and_outbox(
                        session,
                        telemetry=request.app.state.telemetry,
                        event_key=f"run:{run.id}:cancelled",
                        event_type="run.cancel_requested",
                        aggregate_type="agent_run",
                        aggregate_id=run.id,
                        actor_id=principal.actor_id,
                        payload={
                            "run_id": str(run.id),
                            "environment_id": str(run.environment_id)
                            if run.environment_id
                            else None,
                            "cancel_requested_at": run.cancel_requested_at.isoformat()
                            if run.cancel_requested_at
                            else None,
                        },
                    )
            body = _json_model(_run_read(run))
        return JSONResponse(status_code=202, content=body)

    @app.post(
        "/api/v1/evaluations",
        response_model=EvaluationRead,
        status_code=201,
        responses={202: {"model": EvaluationJobRead}, 503: {"model": EvaluationJobRead}},
    )
    async def create_evaluation(
        payload: EvaluationCreate,
        request: Request,
        principal: ApiPrincipal = Depends(require_scope("evaluations:run")),
    ) -> JSONResponse:
        database = _database(request)
        key = _request_key(request)
        fingerprint = stable_parameter_hash(payload.model_dump(mode="json"))
        try:
            with database.session() as session:
                # Older completed response bindings remain valid after migration.
                existing = _stored_response(
                    session,
                    actor_id=principal.actor_id,
                    operation="create_evaluation",
                    target_id="collection",
                    key=key,
                    fingerprint=fingerprint,
                )
                if existing is not None:
                    return JSONResponse(
                        status_code=existing.response_status,
                        content=existing.response_json,
                        headers={"X-Idempotent-Replay": "true"},
                    )
                repository = EvaluationJobRepository(session)
                job = repository.get_request(principal.actor_id, key)
                if job is not None:
                    repository.quarantine_expired(job.id)
                    job = repository.get(job.id)
                    return _evaluation_job_response(session, job, fingerprint)
                job = repository.claim(
                    actor_id=principal.actor_id,
                    key=key,
                    fingerprint=fingerprint,
                    mode=payload.mode,
                    seed=payload.seed,
                )
        except IntegrityError as exc:
            with database.session() as session:
                repository = EvaluationJobRepository(session)
                job = repository.get_request(principal.actor_id, key)
                if job is not None:
                    repository.quarantine_expired(job.id)
                    job = repository.get(job.id)
                    return _evaluation_job_response(session, job, fingerprint)
                if not repository.is_capacity_conflict(exc):
                    raise
                return JSONResponse(
                    status_code=429,
                    content={
                        "error": {
                            "code": "EvaluationCapacityExceeded",
                            "message": "one evaluation is already active or quarantined",
                            "request_id": request.state.request_id,
                        }
                    },
                    headers={"Retry-After": "1"},
                )

        # The durable slot is committed before expensive work enters the threadpool.
        try:
            completed = await run_in_threadpool(
                run_evaluation, mode=payload.mode, seed=payload.seed
            )
            report = completed.as_dict()
            counts = {
                status: sum(score.status == status for score in completed.scores)
                for status in ("pass", "fail", "unavailable")
            }
            if sum(counts.values()) != completed.denominator:
                raise ValueError("evaluation produced an unknown score status")
            row = EvaluationRunRow(
                id=job.id,
                mode=payload.mode,
                seed=payload.seed,
                manifest_version=completed.manifest.version,
                manifest_digest=str(report["manifest_digest"]),
                source_revision=str(report["source_revision"]),
                cohort=str(report["cohort"]),
                denominator=completed.denominator,
                pass_count=counts["pass"],
                fail_count=counts["fail"],
                unavailable_count=counts["unavailable"],
                created_by=principal.actor_id,
                created_at=datetime.now(UTC),
                report_json=report,
            )
            body = _json_model(_evaluation_read(row))
            with database.session() as session:
                EvaluationRunRepository(session).save(row)
                if not EvaluationJobRepository(session).finish(
                    job.id, owner_token=job.owner_token, succeeded=True
                ):
                    raise RuntimeError("evaluation admission expired or was quarantined")
                _save_response_binding(
                    session,
                    actor_id=principal.actor_id,
                    operation="create_evaluation",
                    target_id="collection",
                    key=key,
                    fingerprint=fingerprint,
                    resource_type="evaluation_run",
                    resource_id=job.id,
                    status=201,
                    body=body,
                )
        except Exception as exc:
            # This branch runs only after the executor has returned/raised. It
            # can release its own slot; no exception detail is published.
            with database.session() as session:
                repository = EvaluationJobRepository(session)
                repository.finish(
                    job.id,
                    owner_token=job.owner_token,
                    succeeded=False,
                    error_code=type(exc).__name__,
                )
                failed = repository.get(job.id)
                return _evaluation_job_response(session, failed, fingerprint)
        return JSONResponse(status_code=201, content=body)

    @app.get("/api/v1/evaluation-jobs/{job_id}", response_model=EvaluationJobRead)
    async def get_evaluation_job(
        job_id: UUID,
        request: Request,
        principal: ApiPrincipal = Depends(require_scope("evaluations:read")),
    ) -> EvaluationJobRead:
        with _database(request).session() as session:
            repository = EvaluationJobRepository(session)
            repository.quarantine_expired(job_id)
            job = repository.get(job_id)
            if job is None:
                _error(404, "NotFound", "evaluation job not found")
            return _evaluation_job_read(job)

    @app.get("/api/v1/evaluations", response_model=list[EvaluationRead])
    async def list_evaluations(
        request: Request,
        principal: ApiPrincipal = Depends(require_scope("evaluations:read")),
        limit: int = Query(50, ge=1, le=200),
    ) -> list[EvaluationRead]:
        database = _database(request)
        with database.session() as session:
            return [
                _evaluation_read(row)
                for row in EvaluationRunRepository(session).list_recent(limit=limit)
            ]

    @app.get("/api/v1/evaluations/{evaluation_id}", response_model=EvaluationRead)
    async def get_evaluation(
        evaluation_id: UUID,
        request: Request,
        principal: ApiPrincipal = Depends(require_scope("evaluations:read")),
    ) -> EvaluationRead:
        database = _database(request)
        with database.session() as session:
            row = EvaluationRunRepository(session).get(evaluation_id)
            if row is None:
                _error(404, "NotFound", "evaluation run not found")
            return _evaluation_read(row)

    @app.get("/api/v1/evaluations/{evaluation_id}/report")
    async def get_evaluation_report(
        evaluation_id: UUID,
        request: Request,
        principal: ApiPrincipal = Depends(require_scope("evaluations:read")),
    ) -> dict[str, Any]:
        database = _database(request)
        with database.session() as session:
            row = EvaluationRunRepository(session).get(evaluation_id)
            if row is None:
                _error(404, "NotFound", "evaluation run not found")
            return row.report_json

    def _report_export(request, evaluation_id, format):
        with _database(request).session() as session:
            row = EvaluationRunRepository(session).get(evaluation_id)
            if row is None:
                _error(404, "NotFound", "evaluation run not found")
            content = render_report(row.report_json, format)
        return Response(
            content,
            media_type={"json": "application/json", "csv": "text/csv", "md": "text/markdown"}[
                format
            ],
            headers={
                "Content-Disposition": f'attachment; filename="evaluation-{evaluation_id}.{format}"'
            },
        )

    @app.get("/api/v1/evaluations/{evaluation_id}/report/export/{format}")
    async def export_evaluation_report(
        evaluation_id: UUID,
        format: Literal["json", "csv", "md"],
        request: Request,
        principal: ApiPrincipal = Depends(require_scope("evaluations:read")),
    ) -> Response:
        return _report_export(request, evaluation_id, format)

    @app.get("/dashboard/evaluations/{evaluation_id}/export/{format}")
    async def browser_export_evaluation_report(
        evaluation_id: UUID,
        format: Literal["json", "csv", "md"],
        request: Request,
        principal: ApiPrincipal = Depends(require_browser_scope("evaluations:read")),
    ) -> Response:
        return _report_export(request, evaluation_id, format)

    def _approval_read(session: Any, row: ApprovalRequestRow) -> ApprovalRead:
        action = session.get(ActionRow, row.action_id)
        if action is None:
            _error(409, "InvalidApproval", "approval action is missing")
        return ApprovalRead.model_validate(ApprovalRepository(session).require(row.id).model_dump())

    @app.get("/api/v1/approvals", response_model=list[ApprovalRead])
    async def list_approvals(
        request: Request,
        principal: ApiPrincipal = Depends(require_scope("approvals:read")),
        status: str | None = None,
        limit: int = Query(50, ge=1, le=200),
    ) -> list[ApprovalRead]:
        database = _database(request)
        normalized_status: ApprovalStatus | None = None
        if status is not None:
            try:
                normalized_status = ApprovalStatus(status.upper())
            except ValueError:
                _error(422, "InvalidStatus", "status must be a valid approval status")
        with database.session() as session:
            query = session.query(ApprovalRequestRow).order_by(
                ApprovalRequestRow.requested_at.desc()
            )
            if normalized_status is not None:
                query = query.filter_by(status=normalized_status.value)
            return [_approval_read(session, row) for row in query.limit(limit).all()]

    @app.get("/api/v1/approvals/{approval_id}", response_model=ApprovalRead)
    async def get_approval(
        approval_id: UUID,
        request: Request,
        principal: ApiPrincipal = Depends(require_scope("approvals:read")),
    ) -> ApprovalRead:
        database = _database(request)
        with database.session() as session:
            row = session.get(ApprovalRequestRow, str(approval_id))
            if row is None:
                _error(404, "NotFound", "approval not found")
            return _approval_read(session, row)

    def decide_approval(
        approval_id: UUID,
        request: Request,
        principal: ApiPrincipal,
        payload: ApprovalDecision,
        approved: bool,
    ) -> JSONResponse:
        database = _database(request)
        try:
            with database.session() as session:
                row = session.get(ApprovalRequestRow, str(approval_id))
                if row is None:
                    _error(404, "NotFound", "approval not found")
                action = session.get(ActionRow, row.action_id)
                if action is None:
                    _error(409, "InvalidApproval", "approval action is missing")
                if action.risk_level == RiskLevel.R3.value:
                    _error(
                        403, "R3Forbidden", "R3 actions cannot be approved through the control API"
                    )
            result = ApprovalDecisionService(
                database, telemetry=request.app.state.telemetry
            ).decide(
                approval_id,
                actor_id=principal.actor_id,
                actor_role=principal.role,
                approved=approved,
                approved_parameters=payload.approved_parameters,
                reason=payload.reason,
            )
        except ApprovalRunCancelledError:
            _error(409, "RunCancelled", "cancelled run cannot admit a new approval decision")
        except EntityNotFound:
            _error(404, "NotFound", "approval not found")
        except Exception as exc:
            message = str(exc)
            if "expired" in message.lower():
                _error(410, "ApprovalExpired", "approval has expired")
            if "different committed decision" in message:
                _error(409, "ApprovalDecisionConflict", message)
            if "authorized" in message.lower() or "pending approval" in message.lower():
                _error(409, "InvalidApprovalDecision", message)
            raise
        with database.session() as session:
            body = _json_model(
                _approval_read(session, session.get(ApprovalRequestRow, str(approval_id)))
            )
        return JSONResponse(
            status_code=200,
            content=body,
            headers={"X-Idempotent-Replay": "true"} if result.replayed else None,
        )

    @app.post("/api/v1/approvals/{approval_id}/approve", response_model=ApprovalRead)
    async def approve(
        approval_id: UUID,
        request: Request,
        payload: ApprovalDecision | None = None,
        principal: ApiPrincipal = Depends(require_scope("approvals:decide")),
    ) -> JSONResponse:
        return decide_approval(approval_id, request, principal, payload or ApprovalDecision(), True)

    @app.post("/api/v1/approvals/{approval_id}/reject", response_model=ApprovalRead)
    async def reject(
        approval_id: UUID,
        request: Request,
        payload: ApprovalDecision | None = None,
        principal: ApiPrincipal = Depends(require_scope("approvals:decide")),
    ) -> JSONResponse:
        payload = payload or ApprovalDecision()
        if payload.approved_parameters is not None:
            _error(
                422,
                "InvalidApprovalDecision",
                "rejected decisions cannot include approved_parameters",
            )
        return decide_approval(approval_id, request, principal, payload, False)

    @app.get("/api/v1/audit-events")
    async def audit_events(
        request: Request,
        principal: ApiPrincipal = Depends(require_scope("audit:read")),
        limit: int = Query(100, ge=1, le=500),
    ) -> list[dict[str, Any]]:
        database = _database(request)
        with database.session() as session:
            rows = AuditRepository(session).list_recent(limit=limit)
            return [
                {
                    "event_type": row.event_type,
                    "entity_type": row.entity_type,
                    "entity_id": row.entity_id,
                    "actor_id": row.actor_id,
                    "occurred_at": row.occurred_at,
                    "payload": audit_payload(row),
                }
                for row in rows
            ]

    @app.get("/metrics", response_class=Response)
    @app.get("/api/v1/metrics", response_class=Response)
    async def metrics_endpoint(
        request: Request,
        principal: ApiPrincipal = Depends(require_scope("metrics:read")),
    ) -> Response:
        """Expose only the in-process, bounded metrics projection."""

        return Response(
            content=request.app.state.metrics.render_prometheus(),
            media_type="text/plain; version=0.0.4",
        )

    def _login_form(error: str = "") -> str:
        feedback = f'<p class="notice" role="alert">{html.escape(error)}</p>' if error else ""
        body = (
            '<div class="login-layout">'
            '<section class="panel login-panel" aria-labelledby="sign-in-title">'
            '<h1 id="sign-in-title">Sign in</h1>'
            '<p class="muted" id="key-help">Use the API key configured for your role.</p>'
            f'{feedback}<form method="post" action="/dashboard/login">'
            '<label>API key <input type="password" name="api_key" '
            'autocomplete="current-password" aria-describedby="key-help" required></label>'
            '<button type="submit">Sign in</button></form></section></div>'
        )
        return render_page("AgentOps sign in", body)

    @app.get("/dashboard/login", response_class=HTMLResponse)
    async def dashboard_login_page() -> HTMLResponse:
        return HTMLResponse(
            _login_form(),
            headers={"Cache-Control": "no-store"},
        )

    @app.post("/dashboard/login")
    async def dashboard_login(request: Request) -> Response:
        _check_login_origin(request)
        manager = request.app.state.browser_sessions
        if not manager.configured:
            _error(503, "BrowserSessionUnavailable", "browser session signing is not configured")
        fields = await _form_fields(request)
        api_key = fields.get("api_key")
        if api_key is None or len(api_key) > 1024:
            _error(401, "Unauthenticated", "a valid API key is required")
        principal = request.app.state.authenticator.authenticate(api_key)
        if principal is None:
            return HTMLResponse(
                _login_form("Sign-in failed. Check your API key and try again."),
                status_code=401,
                headers={"Cache-Control": "no-store"},
            )
        _revoke_current_browser_cookie(request)
        token, _csrf_token = manager.issue(principal)
        target = "/dashboard" if principal.allows("incidents:read") else "/dashboard/session"
        response = RedirectResponse(target, status_code=303)
        response.set_cookie(
            BrowserSessionManager.cookie_name,
            token,
            max_age=manager.ttl_seconds,
            httponly=True,
            secure=request.app.state.session_cookie_secure,
            samesite="strict",
            path="/",
        )
        response.headers["Cache-Control"] = "no-store"
        return response

    @app.get("/dashboard/session", response_class=HTMLResponse)
    async def dashboard_session(request: Request) -> HTMLResponse:
        current = _active_browser_session(request)
        if current is None:
            _error(401, "Unauthenticated", "a valid browser session is required")
        request.state.browser_session = current
        principal = current.principal
        scopes = (
            "".join(
                f"<li><code>{html.escape(scope)}</code></li>" for scope in sorted(principal.scopes)
            )
            or "<li>No business scopes</li>"
        )
        body = (
            '<section class="panel"><div class="meta">'
            f"<span>Signed in as <strong>{html.escape(principal.actor_id)}</strong></span>"
            f"<span>Role <strong>{html.escape(principal.role)}</strong></span>"
            f"<span>Issued {timestamp(datetime.fromtimestamp(current.issued_at, UTC))}</span>"
            f"<span>Expires {timestamp(datetime.fromtimestamp(current.expires_at, UTC))}</span>"
            '</div><p class="notice">This session has a fixed expiry. Activity does not renew it. '
            "Sign out ends this session; other browser sessions and API keys remain active.</p>"
            "<h2>Session access</h2><ul>" + scopes + "</ul>"
            '<a href="/dashboard/login">Sign in again</a>'
            '<p class="muted">Re-enter your API key to replace this browser session. '
            "A failed sign-in keeps the current session.</p></section>"
        )
        return HTMLResponse(
            _dashboard_page(
                request,
                "AgentOps current session",
                body,
                principal=principal,
                subtitle="Review your identity, access and session expiry.",
            )
        )

    @app.post("/dashboard/logout")
    async def dashboard_logout(request: Request) -> Response:
        _check_login_origin(request)
        fields = await _form_fields(request, reject_duplicates=True)
        token = request.cookies.get(BrowserSessionManager.cookie_name)
        verified = request.app.state.browser_sessions.authenticate(token)
        if verified is not None:
            if not request.app.state.browser_sessions.csrf_matches(
                verified, fields.get("csrf_token")
            ):
                _error(403, "CsrfRejected", "CSRF token is missing or invalid")
            _revoke_current_browser_cookie(request)
        response = RedirectResponse("/dashboard/login", status_code=303)
        response.delete_cookie(
            BrowserSessionManager.cookie_name,
            path="/",
            httponly=True,
            secure=request.app.state.session_cookie_secure,
            samesite="strict",
        )
        return response

    def _demo_form(
        request: Request,
        principal: ApiPrincipal,
        *,
        fields: dict[str, str] | None = None,
        error: str = "",
    ) -> HTMLResponse:
        values = fields or {}
        request_key = values.get("request_key", str(uuid4()))
        try:
            request_key = str(UUID(request_key))
        except ValueError:
            request_key = str(uuid4())
        csrf = html.escape(request.state.browser_session.csrf_token, quote=True)
        options = "".join(
            f'<option value="{scenario.value}"'
            f"{' selected' if scenario.value == values.get('scenario') else ''}>"
            f"{html.escape(scenario.value.replace('-', ' ').capitalize())}</option>"
            for scenario in ScenarioName
        )
        feedback = f'<p class="notice" role="alert">{html.escape(error)}</p>' if error else ""
        body = (
            '<section class="panel create-panel">'
            "<p>A new isolated simulator will be created. Existing incidents and "
            "environments are kept.</p>"
            f'{feedback}<form method="post" action="/dashboard/incidents/new">'
            f'<input type="hidden" name="csrf_token" value="{csrf}">'
            f'<input type="hidden" name="request_key" value="{request_key}">'
            '<label for="demo-scenario">Scenario</label>'
            f'<select id="demo-scenario" name="scenario">{options}</select>'
            '<label for="demo-title">Title (optional)</label>'
            '<input id="demo-title" name="title" maxlength="300" '
            f'value="{html.escape(values.get("title", ""), quote=True)}" '
            'aria-describedby="demo-title-help">'
            '<p id="demo-title-help" class="muted">Leave empty to use the scenario title. '
            "Service, description and SEV2 severity follow the demo setup.</p>"
            '<div class="filter-actions"><button type="submit">Create and investigate</button>'
            '<a href="/dashboard">Cancel</a></div></form></section>'
        )
        return HTMLResponse(
            _dashboard_page(
                request,
                "AgentOps new demo incident",
                body,
                principal=principal,
                active="/dashboard",
                subtitle="Start a local simulator investigation.",
            ),
            status_code=422 if error else 200,
        )

    def _require_demo_access(principal: ApiPrincipal) -> None:
        if not all(principal.allows(scope) for scope in _DEMO_SCOPES):
            _error(403, "Forbidden", "incident creation and run access are required")

    @app.get("/dashboard/incidents/new", response_class=HTMLResponse)
    async def dashboard_new_incident(
        request: Request,
        principal: ApiPrincipal = Depends(require_browser_session_scope("incidents:write")),
    ) -> HTMLResponse:
        _require_demo_access(principal)
        return _demo_form(request, principal)

    @app.post("/dashboard/incidents/new")
    async def dashboard_create_demo(
        request: Request,
        principal: ApiPrincipal = Depends(require_browser_session_scope("incidents:write")),
    ) -> Response:
        _require_demo_access(principal)
        _check_login_origin(request)
        fields = await _form_fields(request, reject_duplicates=True)
        if not request.app.state.browser_sessions.csrf_matches(
            request.state.browser_session, fields.get("csrf_token")
        ):
            _error(403, "CsrfRejected", "CSRF token is missing or invalid")
        try:
            payload = DemoIncidentCreate.model_validate(
                {key: value for key, value in fields.items() if key != "csrf_token"}
            )
        except ValidationError:
            return _demo_form(
                request,
                principal,
                fields=fields,
                error="Choose a listed scenario and a title of at most 300 characters. "
                "Reload the form if its request key is invalid.",
            )
        database = _database(request)
        fingerprint = stable_parameter_hash(payload.model_dump(mode="json"))
        binding = dict(
            actor_id=principal.actor_id,
            operation="create_demo_incident",
            target_id="collection",
            key=str(payload.request_key),
            fingerprint=fingerprint,
        )
        try:
            with database.session() as session:
                existing = _stored_response(session, **binding)
                if existing is not None:
                    return RedirectResponse(
                        f"/dashboard/runs/{existing.resource_id}",
                        status_code=303,
                    )
                simulator = Simulator()
                simulator.load(payload.scenario)
                identity = simulator.environment_identity
                context = simulator.incident_context()
                SimulationEnvironmentRepository(session).create(
                    environment_id=identity.environment_id,
                    scenario=payload.scenario.value,
                    generation=identity.generation,
                    version=identity.version,
                    state=simulator.snapshot(),
                )
                incident = Incident(
                    title=payload.title or context["title"],
                    description=context["description"],
                    service_name=context["service_name"],
                    severity=Severity.SEV2,
                    created_by=principal.actor_id,
                )
                IncidentRepository(session).save(incident)
                run = AgentRun(incident_id=incident.id, environment_id=identity.environment_id)
                run.transition_to(AgentRunStatus.QUEUED)
                AgentRunRepository(session).save(run)
                incident.active_run_id = run.id
                incident.version += 1
                IncidentRepository(session).save(incident)
                for event_type, aggregate_type, aggregate_id, event_payload in (
                    (
                        "incident.created",
                        "incident",
                        incident.id,
                        {"incident_id": str(incident.id)},
                    ),
                    (
                        "run.created",
                        "agent_run",
                        run.id,
                        {
                            "incident_id": str(incident.id),
                            "run_id": str(run.id),
                            "environment_id": str(identity.environment_id),
                            "configuration": RunCreate(
                                environment_id=identity.environment_id,
                                scenario=payload.scenario.value,
                            ).model_dump(mode="json"),
                        },
                    ),
                ):
                    _audit_and_outbox(
                        session,
                        event_key=f"{event_type.split('.')[0]}:{aggregate_id}:created",
                        event_type=event_type,
                        aggregate_type=aggregate_type,
                        aggregate_id=aggregate_id,
                        actor_id=principal.actor_id,
                        payload=event_payload,
                        telemetry=request.app.state.telemetry,
                    )
                _save_response_binding(
                    session,
                    **binding,
                    resource_type="agent_run",
                    resource_id=run.id,
                    status=202,
                    body=_json_model(_run_read(run)),
                )
                run_id = str(run.id)
        except IntegrityError:
            # A concurrent duplicate rolls back the whole new environment/run.
            with database.session() as session:
                existing = _stored_response(session, **binding)
                if existing is None:
                    _error(409, "CreationConflict", "Creation conflicted. Retry the same form.")
                run_id = existing.resource_id
        return RedirectResponse(f"/dashboard/runs/{run_id}", status_code=303)

    @app.get("/dashboard", response_class=HTMLResponse)
    async def dashboard_home(
        request: Request,
        principal: ApiPrincipal = Depends(require_browser_scope("incidents:read")),
    ) -> HTMLResponse:
        database = _database(request)
        request.state.dashboard_principal = principal
        params = parse_browse(request.query_params, "incidents")
        with database.session() as session:
            page = read_page(session, "incidents", params)
            rows = page.rows
            items: list[str] = []
            for row in rows:
                incident = row.to_domain()
                run_target = html.escape(str(incident.active_run_id), quote=True)
                latest = (
                    f'<a href="/dashboard/runs/{run_target}">latest run</a>'
                    if incident.active_run_id is not None and principal.allows("runs:read")
                    else "latest run unavailable"
                )
                items.append(
                    '<tr><td data-label="Incident">'
                    f"<strong>{html.escape(incident.title)}</strong>"
                    '<details class="record-id"><summary>Incident ID</summary>'
                    f"<code>{html.escape(str(incident.id))}</code></details></td>"
                    f'<td data-label="Service">{html.escape(incident.service_name)}</td>'
                    '<td data-label="Status"><div class="badges">'
                    f"{badge(incident.status.value)}{badge(incident.severity.value)}</div></td>"
                    f'<td data-label="Updated">{timestamp(incident.updated_at)}</td>'
                    f'<td data-label="Run">{latest}</td></tr>'
                )
        summary = stats(
            [
                ("Displayed incidents", len(rows)),
                ("Resolved in this list", sum(row.status == "RESOLVED" for row in rows)),
                ("Escalated in this list", sum(row.status == "ESCALATED" for row in rows)),
            ]
        )
        listing = (
            '<div class="panel table-panel"><table><thead><tr><th>Incident</th>'
            "<th>Service</th><th>Status / severity</th><th>Updated</th><th>Run</th></tr></thead>"
            f"<tbody>{''.join(items)}</tbody></table></div>"
            if items
            else '<section class="panel">'
            + empty(
                "No matching records" if params.filtered else "No incidents yet",
                "Adjust or reset filters."
                if params.filtered
                else "New incidents will appear here after they are created.",
            )
            + "</section>"
        )
        body = (
            (
                '<p class="page-action"><a href="/dashboard/incidents/new">'
                "New demo incident</a></p>"
                if getattr(request.state, "browser_session", None) is not None
                and all(principal.allows(scope) for scope in _DEMO_SCOPES)
                else ""
            )
            + browse_controls("incidents", page, "/dashboard")
            + summary
            + listing
            + pagination(page, "/dashboard")
        )
        return HTMLResponse(
            _dashboard_page(
                request,
                "AgentOps incidents",
                body,
                principal=principal,
                active="/dashboard",
                subtitle="Current incidents and their latest runs.",
            )
        )

    @app.get("/dashboard/runs/{run_id}", response_class=HTMLResponse)
    async def dashboard_run(
        run_id: UUID,
        request: Request,
        principal: ApiPrincipal = Depends(require_browser_scope("runs:read")),
    ) -> HTMLResponse:
        database = _database(request)
        request.state.dashboard_principal = principal
        params = parse_browse(request.query_params, "timeline")
        path = f"/dashboard/runs/{run_id}"
        with database.session() as session:
            run = AgentRunRepository(session).get(run_id)
            if run is None:
                _error(404, "NotFound", "run not found")
            page = read_page(session, "timeline", params, run_id=run_id)
            measurements = read_run_metrics(session, run_id)
            approval_link = ""
            resume_notice = ""
            if run.status is AgentRunStatus.WAITING_FOR_APPROVAL:
                recorded_approval = ApprovalRepository(session).get_for_run(run.id)
                if (
                    recorded_approval is not None
                    and recorded_approval.status is ApprovalStatus.APPROVED
                ):
                    resume_notice = (
                        '<p class="muted">Approval recorded. Resume has not started.</p>'
                    )
                elif (
                    recorded_approval is not None
                    and recorded_approval.status is ApprovalStatus.REJECTED
                ):
                    resume_notice = (
                        '<p class="muted">Rejection recorded. Waiting for worker to finalize.</p>'
                    )
            if (
                run.status is AgentRunStatus.WAITING_FOR_APPROVAL
                and principal.allows("approvals:read")
                and session.query(ApprovalRequestRow.id)
                .filter_by(run_id=str(run.id), status=ApprovalStatus.PENDING.value)
                .first()
                is not None
            ):
                approval_link = (
                    f'<a href="/dashboard/approvals?q={run.id}&amp;status=PENDING">'
                    "Review approval</a>"
                )
            rows = page.rows
            events = [timeline_entry(row) for row in rows]
        body = (
            browse_controls("timeline", page, path)
            + '<section class="panel"><div class="panel-heading"><div>'
            '<p class="field-title">Run identity</p>'
            f'<code>{html.escape(str(run.id))}</code></div><div class="badges">'
            f"{badge(run.status.value)}{approval_link}</div></div>"
            + resume_notice
            + (
                '<p class="muted">Investigation queued or running. '
                "Reload this page to see updates.</p>"
                if run.status in {AgentRunStatus.QUEUED, AgentRunStatus.RUNNING}
                else ""
            )
            + '<p class="muted">Recorded attempts, decisions and outcomes. '
            "Completed attempts replace their start entries; unfinished attempts remain visible. "
            "Older runs may have audit events only. Expand technical details when needed.</p>"
            + (
                f'<ol class="timeline">{"".join(events)}</ol>'
                if events
                else empty(
                    "No matching records" if params.filtered else "No events recorded",
                    "Adjust or reset filters."
                    if params.filtered
                    else "The run has not published any timeline events yet.",
                )
            )
            + "</section>"
            + pagination(page, path)
            + usage_details(measurements)
        )
        return HTMLResponse(
            _dashboard_page(
                request,
                "AgentOps run timeline",
                body,
                principal=principal,
                active="/dashboard",
                subtitle="Inspect the recorded sequence from investigation to verification.",
            )
        )

    @app.get("/dashboard/evaluations", response_class=HTMLResponse)
    async def dashboard_evaluations(
        request: Request,
        principal: ApiPrincipal = Depends(require_browser_scope("evaluations:read")),
    ) -> HTMLResponse:
        database = _database(request)
        request.state.dashboard_principal = principal
        params = parse_browse(request.query_params, "evaluations")
        with database.session() as session:
            page = read_page(session, "evaluations", params)
            rows = page.rows
            entries = [
                '<tr><td data-label="Report">'
                f"<strong>{html.escape(row.mode)}</strong>"
                f'<a href="/dashboard/evaluations/{html.escape(row.id, quote=True)}">'
                f"<code>{html.escape(row.id)}</code></a></td>"
                f'<td data-label="Created">{timestamp(row.created_at)}</td>'
                f'<td data-label="Seed">{row.seed}</td>'
                f'<td data-label="Pass">{row.pass_count}</td>'
                f'<td data-label="Fail">{row.fail_count}</td>'
                f'<td data-label="Unavailable">{row.unavailable_count}</td></tr>'
                for row in rows
            ]
        body = (
            browse_controls("evaluations", page, "/dashboard/evaluations")
            + '<p class="notice">Offline reports are deterministic. Ollama reports use the '
            "configured loopback model and retain failures, unavailable cases, and observed usage. "
            "They are not external LLM benchmarks.</p>"
            '<div class="panel-heading"><h2>Saved reports</h2>'
            '<span class="count">Newest reports first</span></div>'
            + (
                '<section class="panel table-panel"><table><thead><tr><th>Report</th>'
                "<th>Created</th><th>Seed</th><th>Pass</th><th>Fail</th><th>Unavailable</th>"
                f"</tr></thead><tbody>{''.join(entries)}</tbody></table></section>"
                if entries
                else '<section class="panel">'
                + empty(
                    "No matching records" if params.filtered else "No saved evaluations",
                    "Adjust or reset filters."
                    if params.filtered
                    else "Completed evaluation reports will appear here.",
                )
                + "</section>"
            )
            + pagination(page, "/dashboard/evaluations")
        )
        return HTMLResponse(
            _dashboard_page(
                request,
                "AgentOps evaluations",
                body,
                principal=principal,
                active="/dashboard/evaluations",
                subtitle="Compare recorded case outcomes, failures and missing evidence.",
            )
        )

    @app.get("/dashboard/evaluations/{evaluation_id}", response_class=HTMLResponse)
    async def dashboard_evaluation(
        evaluation_id: UUID,
        request: Request,
        principal: ApiPrincipal = Depends(require_browser_scope("evaluations:read")),
    ) -> HTMLResponse:
        database = _database(request)
        with database.session() as session:
            row = EvaluationRunRepository(session).get(evaluation_id)
            if row is None:
                _error(404, "NotFound", "evaluation run not found")
            scores = row.report_json.get("scores", [])
            entries = [
                "<tr>"
                f'<td data-label="Case">{html.escape(str(score.get("case_id", "")))}</td>'
                f'<td data-label="Status">{badge(str(score.get("status", "unavailable")))}</td>'
                '<td data-label="Failures">'
                f"<pre>{_html_json(score.get('failure_reasons', []))}</pre></td>"
                '<td data-label="Missing">'
                f"<pre>{_html_json(score.get('missing_fields', []))}</pre></td>"
                "</tr>"
                for score in scores
                if isinstance(score, dict)
            ]
        body = (
            '<p class="notice">This report is a saved deterministic offline evaluation, '
            "not an external LLM benchmark.</p>"
            + stats(
                [
                    ("Pass", row.pass_count),
                    ("Fail", row.fail_count),
                    ("Unavailable", row.unavailable_count),
                ]
            )
            + '<section class="panel"><div class="meta">'
            f"<span>Report <code>{html.escape(row.id)}</code></span>"
            f"<span>Mode {html.escape(row.mode)} · seed={row.seed}</span>"
            f"<span>manifest={html.escape(row.manifest_version)}</span>"
            f"<span>denominator={row.denominator}</span></div></section>"
            '<div class="panel-heading"><h2>Case outcomes</h2></div>'
            '<section class="panel table-panel"><table><thead><tr><th>Case</th>'
            "<th>Status</th><th>Failures</th><th>Missing evidence</th></tr></thead>"
            f"<tbody>{''.join(entries)}</tbody></table></section>"
            + report_details(row.report_json)
            + '<p class="muted">Download: '
            + " · ".join(
                f'<a href="/dashboard/evaluations/{evaluation_id}/export/{format}">{label}</a>'
                for format, label in (("json", "JSON"), ("csv", "CSV"), ("md", "Markdown"))
            )
            + "</p>"
        )
        return HTMLResponse(
            _dashboard_page(
                request,
                "AgentOps evaluation report",
                body,
                principal=principal,
                active="/dashboard/evaluations",
                subtitle="Inspect every saved case in its recorded denominator.",
            )
        )

    @app.get("/dashboard/approvals", response_class=HTMLResponse)
    async def dashboard_approvals(
        request: Request,
        principal: ApiPrincipal = Depends(require_browser_scope("approvals:read")),
    ) -> HTMLResponse:
        database = _database(request)
        request.state.dashboard_principal = principal
        params = parse_browse(request.query_params, "approvals")
        with database.session() as session:
            page = read_page(session, "approvals", params)
            rows = page.rows
            feedback = _approval_feedback(session, request, principal)
            incident_titles = (
                dict(
                    session.query(AgentRunRow.id, IncidentRow.title)
                    .join(IncidentRow, IncidentRow.id == AgentRunRow.incident_id)
                    .filter(AgentRunRow.id.in_([row.run_id for row in rows]))
                    .all()
                )
                if principal.allows("incidents:read") and rows
                else {}
            )
            entries = []
            pending_count = 0
            for row in rows:
                approval = _approval_read(session, row)
                action = session.get(ActionRow, row.action_id)
                evidence = list(
                    reversed(
                        session.query(ObservationRow)
                        .filter_by(run_id=str(approval.run_id))
                        .order_by(ObservationRow.sequence.desc())
                        .limit(8)
                        .all()
                    )
                )
                evidence_summary = [
                    {
                        "source_tool": item.source_tool,
                        "observed_at": item.observed_at.isoformat(),
                        "payload": item.payload_json,
                    }
                    for item in evidence
                ]
                browser_session = getattr(request.state, "browser_session", None)
                risk_label = str(action.risk_level if action else "unknown")
                controls = ""
                pending_count += approval.status is ApprovalStatus.PENDING
                if (
                    approval.status is ApprovalStatus.PENDING
                    and principal.allows("approvals:decide")
                    and browser_session is not None
                ):
                    csrf = html.escape(browser_session.csrf_token, quote=True)
                    approval_target = html.escape(str(approval.id), quote=True)
                    return_to = html.escape(page.params.url("/dashboard/approvals"), quote=True)
                    controls = (
                        f'<form class="decision" method="post" action="/dashboard/approvals/'
                        f'{approval_target}/approve">'
                        f'<input type="hidden" name="csrf_token" value="{csrf}">'
                        f'<input type="hidden" name="return_to" value="{return_to}">'
                        '<label>Reason (optional) <input name="reason" maxlength="2000"></label>'
                        '<button type="submit">Approve exact parameters</button></form>'
                        f'<form class="decision" method="post" action="/dashboard/approvals/'
                        f'{approval_target}/reject">'
                        f'<input type="hidden" name="csrf_token" value="{csrf}">'
                        f'<input type="hidden" name="return_to" value="{return_to}">'
                        '<label>Reason (optional) <input name="reason" maxlength="2000"></label>'
                        '<button class="secondary" type="submit">Reject</button></form>'
                    )
                elif approval.status is ApprovalStatus.PENDING:
                    controls = (
                        '<p class="notice">Review access only. A signed-in approver '
                        "must authorize or reject this request.</p>"
                    )
                run_reference = html.escape(str(approval.run_id))
                run_link = (
                    f'<a href="/dashboard/runs/{run_reference}"><code>{run_reference}</code></a>'
                    if principal.allows("runs:read")
                    else f"<code>{run_reference}</code>"
                )
                incident_title = incident_titles.get(str(approval.run_id))
                action_heading = (
                    f"<h2>{html.escape(incident_title)}</h2>"
                    f'<p class="action-name">{html.escape(approval.tool_name)}</p>'
                    if incident_title is not None
                    else f"<h2>{html.escape(approval.tool_name)}</h2>"
                )
                entries.append(
                    '<article class="approval-card"><header><div>'
                    f'{action_heading}</div><div class="badges">'
                    f"{badge(approval.status.value)}{badge('risk=' + risk_label)}</div></header>"
                    '<div class="meta">'
                    f"<span>Requested {timestamp(approval.requested_at)}</span>"
                    "<span>Expires "
                    f"{timestamp(approval.expires_at) if approval.expires_at else 'No expiry'}"
                    "</span>"
                    '</div><details><summary>Request details</summary><div class="meta">'
                    f"<span>Approval <code>{html.escape(str(approval.id))}</code></span>"
                    f"<span>Run {run_link}</span></div></details>"
                    '<section class="parameters">'
                    '<h3 class="field-title">Exact parameters</h3>'
                    f"<pre>{_html_json(approval.original_parameters)}</pre></section>"
                    "<details><summary><span>Evidence</span>"
                    f'<span class="count">{len(evidence)} observations · latest 8</span></summary>'
                    f"<pre>{_html_json(evidence_summary)}</pre></details>{controls}</article>"
                )
        body = (
            feedback
            + browse_controls("approvals", page, "/dashboard/approvals")
            + stats(
                [
                    ("Displayed requests", len(entries)),
                    ("Pending in this list", pending_count),
                    ("Not pending in this list", len(entries) - pending_count),
                ]
            )
            + '<p class="notice">Review the risk, exact parameters and evidence before deciding. '
            "Approval authorizes only the displayed parameters. Counts reflect this page.</p>"
            '<section class="approval-list" aria-label="Approval requests">'
            + (
                "".join(entries)
                if entries
                else empty(
                    "No matching records" if params.filtered else "No approval requests",
                    "Adjust or reset filters."
                    if params.filtered
                    else "Requests will appear here when an action needs review.",
                )
            )
            + "</section>"
            + pagination(page, "/dashboard/approvals")
        )
        response = HTMLResponse(
            _dashboard_page(
                request,
                "AgentOps approvals",
                body,
                principal=principal,
                active="/dashboard/approvals",
                subtitle="Actions waiting for review and previous decisions.",
            )
        )
        if _APPROVAL_NOTICE_COOKIE in request.cookies:
            response.delete_cookie(
                _APPROVAL_NOTICE_COOKIE,
                path="/dashboard/approvals",
                secure=request.app.state.session_cookie_secure,
                httponly=True,
                samesite="strict",
            )
        return response

    async def _dashboard_approval_decision(
        approval_id: UUID,
        request: Request,
        principal: ApiPrincipal,
        *,
        approved: bool,
    ) -> RedirectResponse:
        fields = await _form_fields(request, reject_duplicates=True)
        browser_session = request.state.browser_session
        if not request.app.state.browser_sessions.csrf_matches(
            browser_session, fields.get("csrf_token")
        ):
            _error(403, "CsrfRejected", "CSRF token is missing or invalid")
        reason = fields.get("reason") or None
        if reason is not None and len(reason) > 2000:
            _error(422, "InvalidApprovalDecision", "reason is too long")
        request.state.dashboard_principal = principal
        return_to = approval_return_target(fields.get("return_to"))
        decide_approval(
            approval_id,
            request,
            principal,
            ApprovalDecision(reason=reason),
            approved,
        )
        response = RedirectResponse(return_to, status_code=303)
        response.set_cookie(
            _APPROVAL_NOTICE_COOKIE,
            str(approval_id),
            max_age=60,
            path="/dashboard/approvals",
            secure=request.app.state.session_cookie_secure,
            httponly=True,
            samesite="strict",
        )
        return response

    @app.post("/dashboard/approvals/{approval_id}/approve")
    async def dashboard_approve(
        approval_id: UUID,
        request: Request,
        principal: ApiPrincipal = Depends(require_browser_session_scope("approvals:decide")),
    ) -> RedirectResponse:
        return await _dashboard_approval_decision(approval_id, request, principal, approved=True)

    @app.post("/dashboard/approvals/{approval_id}/reject")
    async def dashboard_reject(
        approval_id: UUID,
        request: Request,
        principal: ApiPrincipal = Depends(require_browser_session_scope("approvals:decide")),
    ) -> RedirectResponse:
        return await _dashboard_approval_decision(approval_id, request, principal, approved=False)

    @app.get("/dashboard/audit", response_class=HTMLResponse)
    async def dashboard_audit(
        request: Request,
        principal: ApiPrincipal = Depends(require_browser_scope("audit:read")),
    ) -> HTMLResponse:
        database = _database(request)
        request.state.dashboard_principal = principal
        params = parse_browse(request.query_params, "audit")
        with database.session() as session:
            page = read_page(session, "audit", params)
            rows = page.rows
            entries = [
                "<li>" + timestamp(row.occurred_at) + f"<h2>{html.escape(row.event_type)}</h2>"
                f'<p class="count">Actor: {html.escape(row.actor_id or "System")}</p>'
                "<details><summary>Event details</summary>"
                f"<pre>{_html_json(row.payload_json)}</pre>"
                "</details></li>"
                for row in rows
            ]
        return HTMLResponse(
            _dashboard_page(
                request,
                "AgentOps audit timeline",
                browse_controls("audit", page, "/dashboard/audit")
                + '<section class="panel"><div class="panel-heading"><h2>Recorded activity</h2>'
                '<span class="count">Newest events first</span></div>'
                + (
                    f'<ol class="timeline">{"".join(entries)}</ol>'
                    if entries
                    else empty(
                        "No matching records" if params.filtered else "No recorded activity",
                        "Adjust or reset filters."
                        if params.filtered
                        else "Audit events will appear as operations occur.",
                    )
                )
                + "</section>"
                + pagination(page, "/dashboard/audit"),
                principal=principal,
                active="/dashboard/audit",
                subtitle="Inspect who acted, when it happened and the recorded event payload.",
            )
        )

    return app


app = create_app()

__all__ = ["app", "create_app"]
