"""Allowlisted journal facts. Raw prompts, arguments, results and errors stay out."""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Mapping
from itertools import islice
from typing import Any, Literal
from uuid import UUID, uuid5

from pydantic import BaseModel, ConfigDict, Field, StrictBool, ValidationError

from agentops.llm import DIAGNOSTIC_LOG_SIGNAL_VALUES, HypothesisSnapshot, ObservedFact
from agentops.usage import usage_facts

_LOG_SIGNAL_PATTERNS = (
    (
        "configuration_validation_failed",
        re.compile(
            r"\b(?:configuration|config)\b.{0,60}\b(?:validation|validated?)\b"
            r".{0,40}\b(?:fail\w*|invalid|error)\b|"
            r"\b(?:configuration|config)\s+(?:is\s+)?(?:invalid|malformed|rejected)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "name_resolution_failed",
        re.compile(
            r"\b(?:dns|resolver|hostname|host name|domain name|name resolution)\b"
            r".{0,60}\b(?:cannot|could not|failed|unable|unresolved|not resolved|"
            r"nxdomain|timeout|error)\b|"
            r"\b(?:cannot|could not|unable to|failed to)\s+(?:resolve|look up)\b|"
            r"\bcannot be resolved\b",
            re.IGNORECASE,
        ),
    ),
    (
        "upstream_http_5xx",
        re.compile(
            r"\b(?:upstream|external dependency|remote service)\b.{0,40}"
            r"\b(?:5xx|50[0-9]|http\s*5[0-9]{2})\b|\bhttp\s*5[0-9]{2}\b",
            re.IGNORECASE,
        ),
    ),
)


def diagnostic_log_signals(entries: object) -> list[str]:
    """Return bounded categorical signals from log snippets, never raw text."""

    if not isinstance(entries, list):
        return []
    text = " ".join(entry[:1000] for entry in entries[:20] if isinstance(entry, str)).casefold()
    return [
        name
        for name, pattern in _LOG_SIGNAL_PATTERNS
        if name in DIAGNOSTIC_LOG_SIGNAL_VALUES and pattern.search(text)
    ]


def observed_facts(observations: list[dict]) -> list[ObservedFact]:
    """Typed projection of existing tool fields, never raw log/config text.

    Fact identities are server-owned observation/path identities. This is a
    privacy boundary for tool output fields, not a taxonomy of possible causes.
    Bounded known log patterns become categorical signals; raw log text never
    becomes a fact value. The original evidence reference remains available.
    """
    facts = []
    dependency_facts = []
    metric_labels = {
        "p95_latency_ms": "p95 latency (ms)",
        "error_rate": "Error rate",
        "request_rate": "Request rate",
        "pool_active": "Pool active",
        "pool_max": "Pool maximum",
        "pool_waiters": "Pool waiters",
        "fraud_api_5xx": "Upstream 5xx",
        "queue_depth": "Queue depth",
        "incoming_rate": "Incoming rate",
        "processing_rate": "Processing rate",
        "oldest_message_age_seconds": "Oldest message age (s)",
        "dns_errors": "DNS errors",
    }
    for observation in observations:
        try:
            identity = UUID(str(observation["observation_id"]))
        except (KeyError, TypeError, ValueError):
            continue
        payload = observation.get("payload")
        if not isinstance(payload, dict):
            continue
        candidates = []
        for key, label in (
            ("health", "Service health"),
            ("version", "Deployment version"),
            ("worker_running", "Worker running"),
            ("queue_depth", "Queue depth"),
        ):
            if key in payload:
                candidates.append((key, label, payload[key]))
        metrics = payload.get("metrics")
        if isinstance(metrics, dict):
            candidates += [
                ("metrics." + key, label, metrics[key])
                for key, label in metric_labels.items()
                if key in metrics
            ]
        deployments = payload.get("deployments")
        for index, record in enumerate(deployments[:20] if isinstance(deployments, list) else []):
            if isinstance(record, dict):
                for key, label in (
                    ("version", "Deployment version"),
                    ("status", "Deployment status"),
                ):
                    if key in record:
                        candidates.append((f"deployments.{index}.{key}", label, record[key]))
        config = payload.get("values", payload.get("config"))
        if isinstance(config, dict):
            for key, label in (
                ("db_pool_max", "Database pool maximum"),
                ("fraud_api_timeout_ms", "Upstream timeout (ms)"),
            ):
                try:
                    candidates.append(("config." + key, label, float(config[key])))
                except (KeyError, TypeError, ValueError, OverflowError):
                    pass
            values = payload.get("values")
            target = values.get("inventory_api_host") if isinstance(values, dict) else None
            if isinstance(target, str) and 0 < len(target) <= 1000:
                # Stable identity of this known field without exposing its raw
                # hostname. A fingerprint is pseudonymous, not anonymization.
                candidates.append(
                    (
                        "values.inventory_api_host",
                        "Inventory API target fingerprint",
                        "sha256:" + hashlib.sha256(target.encode("utf-8")).hexdigest(),
                    )
                )
        if observation.get("tool_name") == "search_logs":
            candidates.extend(
                (f"logs.{signal}", "Diagnostic signal", signal)
                for signal in diagnostic_log_signals(payload.get("entries"))
            )
        for path, label, value in candidates:
            try:
                facts.append(
                    ObservedFact(
                        id=uuid5(identity, path), observation_id=identity, label=label, value=value
                    )
                )
            except ValidationError:
                continue
        dependencies = payload.get("dependencies")
        if observation.get("tool_name") == "get_service_status" and isinstance(dependencies, dict):
            for name, value in islice(dependencies.items(), 20):
                if not isinstance(name, str) or not 0 < len(name) <= 1000:
                    continue
                try:
                    dependency_facts.append(
                        ObservedFact(
                            id=uuid5(identity, "dependencies." + name),
                            observation_id=identity,
                            label="Dependency health",
                            value=value,
                        )
                    )
                except ValidationError:
                    continue
    # Extend the catalog without reinterpreting its existing positional refs.
    # Names participate in opaque identity only, never in persisted summary text.
    return (facts + dependency_facts)[:200]


def safe_hypothesis(raw) -> dict | None:
    if not isinstance(raw, Mapping):
        return None
    try:
        # Summary is derived by the typed claim, not copied from this mapping.
        data = {
            key: raw[key]
            for key in HypothesisSnapshot.model_fields
            if key in raw and key != "summary"
        }
        return HypothesisSnapshot.model_validate(data).model_dump(mode="json")
    except (ValidationError, TypeError, ValueError):
        return None


NAMES = frozenset(
    {
        "load_incident",
        "triage",
        "create_plan",
        "select_next_diagnostic_step",
        "execute_read_tool",
        "record_observation",
        "update_hypotheses",
        "propose_remediation",
        "policy_gate",
        "human_approval_interrupt",
        "execute_safe_action",
        "verify_environment",
        "review",
        "tool_selection",
        "hypothesis_update",
        "diagnostic_plan",
        "remediation_proposal",
        "lifecycle",
        "get_service_status",
        "query_metrics",
        "search_logs",
        "get_deployment_history",
        "get_service_config",
        "restart_service",
        "restart_worker",
        "rollback_deployment",
        "verify_service",
        "unknown",
    }
)
ERRORS = frozenset(
    {
        "LLMTimeout",
        "LLMUnavailable",
        "LLMRateLimited",
        "LLMAuthentication",
        "LLMConfiguration",
        "LLMRefusal",
        "LLMIncomplete",
        "LLMOutputInvalid",
        "LLMOutputTooLarge",
        "LLMProviderContract",
        "ToolError",
        "ToolNotFound",
        "ToolValidationError",
        "PermissionDenied",
        "ForbiddenAction",
        "ApprovalRequired",
        "IdempotencyKeyRequired",
        "IdempotencyConflict",
        "ToolExecutionError",
        "RunLeaseFencedError",
        "CancellationRequested",
        "WorkflowBudgetExceeded",
        "WorkflowError",
        "tool_error",
        "ApprovalRejected",
        "ApprovalExpired",
        "VerificationFailed",
        "StepBudgetExceeded",
        "ToolBudgetExceeded",
        "LLMBudgetExceeded",
        "TimeBudgetExceeded",
        "RuntimeInterrupted",
    }
)


class JournalFacts(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    usage_version: Literal[1] | None = None
    input_tokens: int | None = Field(default=None, ge=0, le=10**15)
    output_tokens: int | None = Field(default=None, ge=0, le=10**15)
    estimated_cost_usd: float | None = Field(default=None, ge=0, le=10**9)
    cost_source: Literal["provider_reported"] | None = None

    error_code: str | None = None
    arguments_fingerprint: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    outcome: (
        Literal[
            "resolved",
            "escalated",
            "approval_required",
            "tool_error",
            "budget_exceeded",
            "step_limit_exceeded",
            "cancelled",
        ]
        | None
    ) = None
    run_status: (
        Literal[
            "CREATED",
            "QUEUED",
            "RUNNING",
            "WAITING_FOR_APPROVAL",
            "SUCCEEDED",
            "FAILED",
            "BLOCKED",
            "CANCELLED",
            "TIMED_OUT",
        ]
        | None
    ) = None
    approval_status: Literal["PENDING", "APPROVED", "REJECTED", "EXPIRED"] | None = None
    verification_passed: StrictBool | None = None
    diagnosis_category: (
        Literal[
            "bad_deployment",
            "connection_pool_exhaustion",
            "external_dependency_outage",
            "queue_consumer_failure",
            "invalid_configuration",
            "unknown",
        ]
        | None
    ) = None
    hypothesis_count: int | None = Field(default=None, ge=0, le=10)
    hypothesis_scores: list[float] = Field(default_factory=list, max_length=10)
    hypothesis_changes: list[HypothesisSnapshot] = Field(default_factory=list, max_length=10)


def safe_facts(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Construct from finite categories only, before any DB serialization.

    Unknown fields/values are omitted, never stringified. In particular, a model
    summary or exception message cannot sneak in through a misleading safe key.
    """
    values: dict[str, Any] = {}
    if raw.get("usage_version") == 1:
        values.update(usage_facts(raw))
    for key, allowed in (
        ("error_code", ERRORS),
        (
            "outcome",
            {
                "resolved",
                "escalated",
                "approval_required",
                "tool_error",
                "budget_exceeded",
                "step_limit_exceeded",
                "cancelled",
            },
        ),
        (
            "run_status",
            {
                "CREATED",
                "QUEUED",
                "RUNNING",
                "WAITING_FOR_APPROVAL",
                "SUCCEEDED",
                "FAILED",
                "BLOCKED",
                "CANCELLED",
                "TIMED_OUT",
            },
        ),
        ("approval_status", {"PENDING", "APPROVED", "REJECTED", "EXPIRED"}),
        (
            "diagnosis_category",
            {
                "bad_deployment",
                "connection_pool_exhaustion",
                "external_dependency_outage",
                "queue_consumer_failure",
                "invalid_configuration",
                "unknown",
            },
        ),
    ):
        value = raw.get(key)
        if isinstance(value, str) and value in allowed:
            values[key] = value
    fingerprint = raw.get("arguments_fingerprint")
    if (
        isinstance(fingerprint, str)
        and len(fingerprint) == 64
        and all(c in "0123456789abcdef" for c in fingerprint)
    ):
        values["arguments_fingerprint"] = fingerprint
    if isinstance(raw.get("verification_passed"), bool):
        values["verification_passed"] = raw["verification_passed"]
    count = raw.get("hypothesis_count")
    if type(count) is int and 0 <= count <= 10:
        values["hypothesis_count"] = count
    scores = raw.get("hypothesis_scores")
    if isinstance(scores, list):
        values["hypothesis_scores"] = [
            float(x)
            for x in scores[:10]
            if type(x) in {float, int} and math.isfinite(x) and 0 <= x <= 1
        ]
    changes = raw.get("hypothesis_changes")
    if isinstance(changes, list):
        values["hypothesis_changes"] = [
            item for raw_item in changes[:10] if (item := safe_hypothesis(raw_item)) is not None
        ]
    return JournalFacts.model_validate(values).model_dump(mode="json", exclude_none=True)


def safe_error(error: BaseException) -> str:
    code = getattr(error, "code", None)
    if isinstance(code, str) and code in ERRORS:
        return code
    name = type(error).__name__
    return name if name in ERRORS else "WorkflowError"


def safe_projection(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Same finite projection for persistence and defensive API/HTML rendering."""
    result: dict[str, Any] = {"journal_version": 1}
    for key, allowed, default in (
        ("kind", {"node", "tool", "llm", "run", "hypothesis"}, "node"),
        ("phase", {"started", "completed", "failed", "interrupted"}, "started"),
        ("name", NAMES, "unknown"),
    ):
        value = raw.get(key)
        result[key] = value if isinstance(value, str) and value in allowed else default
    for key in ("sequence", "attempt"):
        value = raw.get(key)
        if type(value) is int and 0 < value <= 10_000_000:
            result[key] = value
    for key in ("journal_id", "attempt_id", "step_id"):
        try:
            result[key] = str(UUID(raw[key]))
        except (KeyError, TypeError, ValueError, AttributeError):
            pass
    identity = raw.get("model_configuration_id")
    if (
        isinstance(identity, str)
        and len(identity) == 64
        and all(c in "0123456789abcdef" for c in identity)
    ):
        result["model_configuration_id"] = identity
    duration = raw.get("duration_ms")
    if type(duration) in {float, int} and math.isfinite(duration) and 0 <= duration <= 1e12:
        result["duration_ms"] = duration
    facts = raw.get("facts")
    result["facts"] = safe_facts(facts if isinstance(facts, Mapping) else {})
    links = raw.get("links")
    safe_links: dict[str, Any] = {}
    if isinstance(links, Mapping):
        for key in (
            "action_id",
            "approval_id",
            "observation_ids",
            "receipt_ids",
            "hypothesis_ids",
            "decision_ids",
            "version_ids",
        ):
            value = links.get(key)
            values = value[:50] if isinstance(value, list) else [value]
            valid = []
            for item in values:
                try:
                    valid.append(str(UUID(item)))
                except (TypeError, ValueError, AttributeError):
                    pass
            if valid:
                safe_links[key] = valid if isinstance(value, list) else valid[0]
    result["links"] = safe_links
    return result
