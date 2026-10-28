"""Server-side API-key authentication and scope checks."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import time
from collections.abc import Mapping
from dataclasses import dataclass
from secrets import compare_digest, token_urlsafe
from typing import Any


@dataclass(frozen=True)
class ApiPrincipal:
    """Identity resolved from a server-side API-key registry."""

    actor_id: str
    role: str
    scopes: frozenset[str]

    def allows(self, scope: str) -> bool:
        return scope in self.scopes or "admin:*" in self.scopes


@dataclass(frozen=True)
class BrowserSession:
    principal: ApiPrincipal
    csrf_token: str
    expires_at: int
    issued_at: int


def _urlsafe_encode(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")


def _urlsafe_decode(value: str) -> bytes:
    padding = "=" * (-len(value) % 4)
    return base64.urlsafe_b64decode(value + padding)


class BrowserSessionManager:
    """Issue and verify bounded HMAC-signed local dashboard sessions."""

    cookie_name = "agentops_session"

    def __init__(self, secret: str | None, *, ttl_seconds: int = 8 * 60 * 60) -> None:
        # Public denylist entries are never valid signing credentials.
        if secret in {"agentops-local-session-change-me-2026"}:
            raise ValueError("the public demo signing secret is forbidden; generate a private key")
        if ttl_seconds < 60 or ttl_seconds > 24 * 60 * 60:
            raise ValueError("browser session TTL must be between 60 seconds and 24 hours")
        if secret is not None and len(secret.encode("utf-8")) < 32:
            raise ValueError("browser session secret must contain at least 32 bytes")
        self._secret = secret.encode("utf-8") if secret is not None else None
        self.ttl_seconds = ttl_seconds

    @classmethod
    def from_environment(cls) -> BrowserSessionManager:
        secret = os.getenv("AGENTOPS_SESSION_SECRET") or None
        raw_ttl = os.getenv("AGENTOPS_SESSION_TTL_SECONDS", str(8 * 60 * 60))
        try:
            ttl_seconds = int(raw_ttl)
        except ValueError as exc:
            raise ValueError("AGENTOPS_SESSION_TTL_SECONDS must be an integer") from exc
        return cls(secret, ttl_seconds=ttl_seconds)

    @property
    def configured(self) -> bool:
        return self._secret is not None

    def issue(self, principal: ApiPrincipal, *, now: int | None = None) -> tuple[str, str]:
        if self._secret is None:
            raise RuntimeError("browser session signing is not configured")
        issued_at = int(time.time()) if now is None else now
        csrf_token = token_urlsafe(32)
        payload = {
            "actor_id": principal.actor_id,
            "role": principal.role,
            "scopes": sorted(principal.scopes),
            "issued_at": issued_at,
            "expires_at": issued_at + self.ttl_seconds,
            "csrf_token": csrf_token,
        }
        encoded = _urlsafe_encode(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        )
        signature = _urlsafe_encode(
            hmac.new(self._secret, encoded.encode("ascii"), hashlib.sha256).digest()
        )
        return f"{encoded}.{signature}", csrf_token

    def authenticate(self, token: str | None, *, now: int | None = None) -> BrowserSession | None:
        if self._secret is None or not token or len(token) > 4096:
            return None
        try:
            encoded, supplied_signature = token.split(".", 1)
            expected_signature = _urlsafe_encode(
                hmac.new(self._secret, encoded.encode("ascii"), hashlib.sha256).digest()
            )
            if not compare_digest(supplied_signature, expected_signature):
                return None
            payload = json.loads(_urlsafe_decode(encoded))
            if not isinstance(payload, dict):
                return None
            expires_at = int(payload["expires_at"])
            issued_at = int(payload["issued_at"])
            current = int(time.time()) if now is None else now
            if issued_at > current + 60 or current >= expires_at:
                return None
            if expires_at - issued_at != self.ttl_seconds:
                return None
            csrf_token = payload["csrf_token"]
            if not isinstance(csrf_token, str) or len(csrf_token) < 32:
                return None
            principal = principal_from_value(
                {
                    "actor_id": payload["actor_id"],
                    "role": payload["role"],
                    "scopes": payload["scopes"],
                }
            )
        except (KeyError, TypeError, ValueError, UnicodeDecodeError, json.JSONDecodeError):
            return None
        return BrowserSession(
            principal=principal, csrf_token=csrf_token, expires_at=expires_at, issued_at=issued_at
        )

    def csrf_matches(self, session: BrowserSession, supplied_token: str | None) -> bool:
        return bool(supplied_token) and compare_digest(
            session.csrf_token.encode("utf-8"), supplied_token.encode("utf-8")
        )


_ROLE_SCOPES: dict[str, frozenset[str]] = {
    "auditor": frozenset(
        {
            "incidents:read",
            "runs:read",
            "approvals:read",
            "audit:read",
            "metrics:read",
            "evaluations:read",
        }
    ),
    "viewer": frozenset(
        {
            "incidents:read",
            "runs:read",
            "approvals:read",
            "audit:read",
            "metrics:read",
            "evaluations:read",
        }
    ),
    "operator": frozenset(
        {
            "incidents:read",
            "incidents:write",
            "runs:read",
            "runs:start",
            "runs:cancel",
            "metrics:read",
            "evaluations:read",
            "evaluations:run",
        }
    ),
    "approver": frozenset(
        {
            "incidents:read",
            "runs:read",
            "approvals:read",
            "approvals:decide",
            "metrics:read",
            "evaluations:read",
        }
    ),
    "admin": frozenset(
        {
            "incidents:read",
            "incidents:write",
            "runs:read",
            "runs:start",
            "runs:cancel",
            "approvals:read",
            "approvals:decide",
            "audit:read",
            "metrics:read",
            "evaluations:read",
            "evaluations:run",
        }
    ),
}


def principal_from_value(value: Any) -> ApiPrincipal:
    """Normalize a test/config value without trusting client-supplied roles."""

    if isinstance(value, ApiPrincipal):
        return value
    if isinstance(value, Mapping):
        actor_id = str(value.get("actor_id", ""))
        role = str(value.get("role", "")).lower()
        raw_scopes = value.get("scopes")
    elif isinstance(value, str):
        parts = value.split(":", 2)
        actor_id = parts[0] if parts else ""
        role = parts[1] if len(parts) > 1 else ""
        raw_scopes = parts[2].replace("|", " ").split() if len(parts) > 2 else None
    elif isinstance(value, (tuple, list)) and len(value) >= 2:
        actor_id = str(value[0])
        role = str(value[1])
        raw_scopes = value[2] if len(value) > 2 else None
    else:
        actor_id = str(getattr(value, "actor_id", ""))
        role = str(getattr(value, "role", ""))
        raw_scopes = getattr(value, "scopes", None)
    role = role.lower()
    if not actor_id or not role:
        raise ValueError("API principal requires actor_id and role")
    if isinstance(raw_scopes, str):
        raw_scopes = [item for item in raw_scopes.replace(",", " ").split() if item]
    scopes = (
        _ROLE_SCOPES.get(role, frozenset())
        if raw_scopes is None
        else frozenset(str(item) for item in raw_scopes)
    )
    return ApiPrincipal(actor_id=actor_id, role=role, scopes=scopes)


class ApiKeyAuthenticator:
    """Resolve API keys held by the server, never by request payloads."""

    def __init__(self, api_keys: Mapping[str, Any] | None = None) -> None:
        self._keys = {key: principal_from_value(value) for key, value in (api_keys or {}).items()}

    @classmethod
    def from_environment(cls) -> ApiKeyAuthenticator:
        """Parse ``key=actor:role[:scope,scope]`` entries from configuration."""

        raw = os.getenv("AGENTOPS_API_KEYS", "")
        parsed: dict[str, dict[str, Any]] = {}
        for item in raw.split(","):
            item = item.strip()
            if not item or "=" not in item:
                continue
            key, identity = item.split("=", 1)
            parts = identity.split(":", 2)
            if len(parts) < 2:
                continue
            value: dict[str, Any] = {"actor_id": parts[0], "role": parts[1]}
            if len(parts) == 3:
                value["scopes"] = parts[2].split("|") if parts[2] else []
            parsed[key] = value
        return cls(parsed)

    def authenticate(self, api_key: str | None) -> ApiPrincipal | None:
        """Resolve a request key without an early-exit dictionary comparison.

        Keys are configuration-held secrets, so comparison stays server-side and
        uses ``compare_digest`` for every configured candidate without returning
        early.  A matching principal is retained while unknown keys remain
        unauthenticated.
        """

        if not api_key:
            return None
        candidate = api_key.encode("utf-8")
        matched: ApiPrincipal | None = None
        for registered_key, principal in self._keys.items():
            if isinstance(registered_key, str) and compare_digest(
                registered_key.encode("utf-8"), candidate
            ):
                matched = principal
        return matched
