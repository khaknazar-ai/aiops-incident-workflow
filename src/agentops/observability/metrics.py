"""Small in-process metrics registry with bounded labels.

This is an offline operational seam, not a claim that a production exporter is
configured. Labels are allow-listed and IDs are rejected so the registry cannot
silently become a high-cardinality store.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass, field
from threading import Lock
from typing import Any

_LABEL_NAME = re.compile(r"^[a-z][a-z0-9_]{0,31}$")
_METRIC_NAME = re.compile(r"^[a-z][a-z0-9_]{0,127}$")
_ID_VALUE = re.compile(
    r"^(?:[0-9a-f]{8}-[0-9a-f-]{27,}|[0-9a-f]{32,}|[A-Za-z0-9_-]{24,})$",
    re.IGNORECASE,
)
_ALLOWED_LABELS = frozenset({"decision", "event", "outcome", "reason", "status", "tool"})
_MAX_LABEL_LENGTH = 80
_BUCKETS = (0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, math.inf)


@dataclass
class _Histogram:
    count: int = 0
    total: float = 0.0
    buckets: list[int] = field(default_factory=lambda: [0] * len(_BUCKETS))


class MetricsRegistry:
    """Thread-safe counters/histograms suitable for local smoke tests."""

    def __init__(self, *, max_series: int = 256) -> None:
        if not 1 <= max_series <= 10_000:
            raise ValueError("max_series must be between 1 and 10000")
        self._max_series = max_series
        self._lock = Lock()
        self._counters: dict[tuple[str, tuple[tuple[str, str], ...]], float] = defaultdict(float)
        self._histograms: dict[tuple[str, tuple[tuple[str, str], ...]], _Histogram] = {}

    def _admit(self, key, *, histogram):
        own, other = (
            (self._histograms, self._counters) if histogram else (self._counters, self._histograms)
        )
        if any(name == key[0] for name, _ in other):
            raise ValueError("metric name already has another type")
        if key not in own and len(self._counters) + len(self._histograms) >= self._max_series:
            raise ValueError("metric series capacity exceeded")

    @staticmethod
    def _labels(
        labels: Mapping[str, Any] | None, extra: Mapping[str, Any]
    ) -> tuple[tuple[str, str], ...]:
        values: dict[str, Any] = {}
        if labels:
            values.update(labels)
        values.update(extra)
        normalized: list[tuple[str, str]] = []
        for raw_name, raw_value in values.items():
            name = str(raw_name)
            if not _LABEL_NAME.fullmatch(name) or name not in _ALLOWED_LABELS:
                raise ValueError(f"metric label is not allow-listed: {name}")
            value = str(raw_value)
            if len(value) > _MAX_LABEL_LENGTH or _ID_VALUE.fullmatch(value):
                raise ValueError(f"metric label value is unbounded: {name}")
            normalized.append((name, value))
        if len(normalized) > 4:
            raise ValueError("metrics support at most four labels")
        return tuple(sorted(normalized))

    @staticmethod
    def _name(name: str) -> str:
        if not _METRIC_NAME.fullmatch(name):
            raise ValueError("invalid metric name")
        return name

    def inc(
        self,
        name: str,
        value: float = 1,
        labels: Mapping[str, Any] | None = None,
        **extra_labels: Any,
    ) -> None:
        metric_name = self._name(name)
        if type(value) not in {int, float} or not 0 <= value <= 1e300:
            raise ValueError("counter increments must be finite and nonnegative")
        key = (metric_name, self._labels(labels, extra_labels))
        with self._lock:
            self._admit(key, histogram=False)
            if not math.isfinite(self._counters.get(key, 0) + value):
                raise ValueError("counter total overflow")
            self._counters[key] += float(value)

    increment = inc

    def observe(
        self,
        name: str,
        value: float,
        labels: Mapping[str, Any] | None = None,
        **extra_labels: Any,
    ) -> None:
        metric_name = self._name(name)
        if type(value) not in {int, float} or not 0 <= value <= 1e300:
            raise ValueError("histogram observations must be finite and nonnegative")
        key = (metric_name, self._labels(labels, extra_labels))
        with self._lock:
            self._admit(key, histogram=True)
            sample = self._histograms.setdefault(key, _Histogram())
            if not math.isfinite(sample.total + value):
                raise ValueError("histogram total overflow")
            sample.count += 1
            sample.total += float(value)
            for index, bound in enumerate(_BUCKETS):
                sample.buckets[index] += value <= bound

    def record_event(self, event_type: str) -> None:
        """Record a fixed-cardinality business event without raw entity IDs."""

        mapping = {
            "incident.created": ("agentops_incidents_total", "created"),
            "incident.reopened": ("agentops_incidents_total", "reopened"),
            "run.created": ("agentops_runs_total", "queued"),
            "run.cancel_requested": ("agentops_runs_total", "cancel_requested"),
            "approval.decided": ("agentops_approvals_total", "decided"),
        }
        metric_name, event = mapping.get(event_type, ("agentops_audit_events_total", "other"))
        self.inc(metric_name, event=event)

    @staticmethod
    def _escape_label(value: str) -> str:
        return (
            value.replace(chr(92), chr(92) * 2)
            .replace(chr(34), chr(92) + chr(34))
            .replace("\n", "\\n")
        )

    @staticmethod
    def _render_labels(labels: tuple[tuple[str, str], ...]) -> str:
        if not labels:
            return ""
        encoded = ",".join(
            f'{name}="{MetricsRegistry._escape_label(value)}"' for name, value in labels
        )
        return "{" + encoded + "}"

    def render_prometheus(self) -> str:
        with self._lock:
            counters = list(self._counters.items())
            histograms = [
                (key, value.count, value.total, list(value.buckets))
                for key, value in self._histograms.items()
            ]
        lines: list[str] = []
        typed = set()
        for (name, labels), value in sorted(counters):
            if name not in typed:
                lines.append(f"# TYPE {name} counter")
                typed.add(name)
            lines.append(f"{name}{self._render_labels(labels)} {value:g}")
        for (name, labels), count, total, buckets in sorted(histograms):
            if name not in typed:
                lines.append(f"# TYPE {name} histogram")
                typed.add(name)
            for bound, bucket in zip(_BUCKETS, buckets, strict=True):
                le = "+Inf" if math.isinf(bound) else f"{bound:g}"
                bucket_labels = tuple(sorted((*labels, ("le", le))))
                lines.append(f"{name}_bucket{self._render_labels(bucket_labels)} {bucket}")
            lines.append(f"{name}_count{self._render_labels(labels)} {count}")
            lines.append(f"{name}_sum{self._render_labels(labels)} {total:g}")
        return "\n".join(lines) + ("\n" if lines else "")


__all__ = ["MetricsRegistry"]
