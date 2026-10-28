"""Audit helpers shared by API and approval transitions."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .redaction import redact_payload


def durable_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return the bounded payload stored in append-only audit history."""

    result = redact_payload(payload)
    if not isinstance(result, dict):
        raise TypeError("audit payload must be an object")
    return result


__all__ = ["durable_payload"]
