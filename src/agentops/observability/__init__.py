"""Operational telemetry seams that never become the business truth."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol

from .metrics import MetricsRegistry
from .redaction import redact_payload


class TelemetrySink(Protocol):
    def record_audit(self, event_type: str, payload: Mapping[str, Any]) -> None:
        """Record an operational observation; implementations may fail."""


class NoopTelemetry:
    def record_audit(self, event_type: str, payload: Mapping[str, Any]) -> None:
        return None


class MetricsTelemetry:
    """Default in-process sink: redacts observations and updates bounded metrics."""

    def __init__(self, metrics: MetricsRegistry | None = None, *, max_events: int = 200) -> None:
        if not 1 <= max_events <= 10_000:
            raise ValueError("max_events must be between 1 and 10000")
        self._max_events = max_events
        self.dropped_events = 0
        self.metrics = metrics or MetricsRegistry()
        self.events: list[dict[str, Any]] = []

    def record_audit(self, event_type: str, payload: Mapping[str, Any]) -> None:
        self.metrics.record_event(event_type)
        self.events.append(
            {
                "event_type": event_type,
                "payload": redact_payload(payload),
            }
        )
        if len(self.events) > self._max_events:
            del self.events[0]
            self.dropped_events += 1


def emit_safely(
    telemetry: TelemetrySink | None,
    *,
    event_type: str,
    payload: Mapping[str, Any],
) -> None:
    """Send telemetry after durable staging and isolate operational failures."""

    if telemetry is None:
        return
    try:
        telemetry.record_audit(event_type, redact_payload(payload))
    except Exception:
        # Telemetry is explicitly best-effort; the durable audit transaction is
        # committed by the caller and is the source of business truth.
        return None


__all__ = ["MetricsTelemetry", "NoopTelemetry", "TelemetrySink", "emit_safely"]
