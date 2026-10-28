"""Deterministic redaction for persisted audit and rendered evidence.

The redactor is deliberately conservative: keys that conventionally carry
credentials are replaced, while arbitrary incident text remains data and is only
bounded. It does not attempt to detect prompt injection or infer whether a value
is safe.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

REDACTED = "[REDACTED]"
TRUNCATED = "[TRUNCATED]"
_BINARY_OMITTED = "[BINARY_OMITTED]"

_SENSITIVE_KEYS = frozenset(
    {
        "accesskey",
        "apikey",
        "authorization",
        "clientsecret",
        "cookie",
        "credential",
        "password",
        "privatekey",
        "refreshtoken",
        "secret",
        "setcookie",
        "token",
    }
)


def _key_name(key: Any) -> str:
    return "".join(character for character in str(key).casefold() if character.isalnum())


def _is_sensitive_key(key: Any) -> bool:
    normalized = _key_name(key)
    if normalized in _SENSITIVE_KEYS:
        return True
    return any(
        normalized.endswith(suffix)
        for suffix in ("apikey", "authorization", "password", "privatekey", "secret", "token")
    )


def _bounded_text(value: str, max_string_length: int) -> str:
    if len(value) <= max_string_length:
        return value
    return f"{value[:max_string_length]}{TRUNCATED}"


def redact_payload(
    value: Any,
    *,
    max_depth: int = 6,
    max_items: int = 50,
    max_string_length: int = 2_000,
) -> Any:
    """Return a JSON-shaped, redacted and bounded copy of a value.

    The function never mutates its input. Depth and item bounds handle
    adversarial nested payloads, and sensitive values are replaced before any
    recursive traversal. Unknown objects are represented by a bounded string.
    """

    if max_depth < 1 or max_items < 1 or max_string_length < 1:
        raise ValueError("redaction bounds must be positive")

    def visit(item: Any, depth: int) -> Any:
        if isinstance(item, str):
            return _bounded_text(item, max_string_length)
        if item is None or isinstance(item, (bool, int, float)):
            return item
        if isinstance(item, bytes):
            return _BINARY_OMITTED
        if depth >= max_depth:
            return TRUNCATED
        if isinstance(item, Mapping):
            result: dict[str, Any] = {}
            for index, (key, child) in enumerate(item.items()):
                if index >= max_items:
                    result["[truncated_items]"] = TRUNCATED
                    break
                normalized_key = _bounded_text(str(key), 200)
                result[normalized_key] = (
                    REDACTED if _is_sensitive_key(key) else visit(child, depth + 1)
                )
            return result
        if isinstance(item, (list, tuple, set, frozenset)):
            result_list = [visit(child, depth + 1) for child in list(item)[:max_items]]
            if len(item) > max_items:
                result_list.append(TRUNCATED)
            return result_list
        return _bounded_text(repr(item), max_string_length)

    return visit(value, 0)


def redact_text(value: str, *, max_string_length: int = 2_000) -> str:
    """Bound untrusted display text without interpreting it as markup."""

    return _bounded_text(value, max_string_length)


__all__ = ["REDACTED", "TRUNCATED", "redact_payload", "redact_text"]
