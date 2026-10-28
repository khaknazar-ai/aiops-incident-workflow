"""Redacted provenance and semantic compatibility for a run's model boundary.

Identity v1 binds provider/protocol, adapter type, model, endpoint route,
temperature and output validation limit. Credentials and URL userinfo/query/
fragment are never persisted or hashed. Timeout/retry are recorded provenance,
not semantic compatibility inputs: the durable workflow budget remains separate.
"""

from __future__ import annotations

import hashlib
import json
from typing import Literal
from urllib.parse import parse_qsl, urlsplit

from pydantic import BaseModel, ConfigDict, Field

from agentops.llm import (
    FakeLLMClient,
    LLMClient,
    LLMConfigurationError,
    ModelConfig,
    OpenAIResponsesClient,
)
from agentops.ollama import OllamaClient

Digest = str


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


class ModelConfigurationIdentity(BaseModel):
    """Allowlisted immutable descriptor; arbitrary raw config cannot enter storage."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    contract: Literal["model-configuration-v1"] = "model-configuration-v1"
    provider: Literal["fake", "openai_responses", "custom", "ollama"]
    protocol: Literal["fake", "responses", "structured-client", "ollama_chat"]
    adapter_sha256: Digest = Field(pattern=r"^[0-9a-f]{64}$")
    model_sha256: Digest = Field(pattern=r"^[0-9a-f]{64}$")
    endpoint_sha256: Digest | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    temperature: float = Field(ge=0.0, le=2.0)
    max_output_chars: int = Field(ge=1, le=1_000_000)
    timeout_seconds: float = Field(gt=0.0)
    max_retries: int = Field(ge=0, le=2)

    @property
    def configuration_id(self) -> str:
        semantic = self.model_dump(exclude={"timeout_seconds", "max_retries"})
        return _digest(json.dumps(semantic, sort_keys=True, separators=(",", ":")))


def model_configuration_identity(
    client: LLMClient, config: ModelConfig
) -> ModelConfigurationIdentity:
    """Inspect only declared adapter/config fields, never keys or client reprs.

    Custom adapters are distinguished by their Python type. Changes to hidden
    instance behaviour or a provider's model alias cannot be detected by v1;
    callers must version the adapter/model when that contract changes.
    """

    if isinstance(client, FakeLLMClient):
        provider, protocol = "fake", "fake"
        endpoint = None  # Fake mode performs no endpoint request.
    elif isinstance(client, OpenAIResponsesClient):
        provider, protocol = "openai_responses", "responses"
        endpoint = config.base_url or client.base_url
    elif isinstance(client, OllamaClient):
        provider, protocol = "ollama", "ollama_chat"
        endpoint = config.base_url or client.base_url
    else:
        provider, protocol = "custom", "structured-client"
        endpoint = config.base_url
    endpoint_digest = None
    if endpoint:
        try:
            url = urlsplit(endpoint)
            if url.scheme.lower() not in {"http", "https"} or not url.hostname:
                raise ValueError
            # Unknown query fields can change routing/model semantics. Do not
            # silently ignore them or hash an opaque value that may be a secret.
            credential_fields = {
                "key",
                "api_key",
                "api-key",
                "token",
                "access_token",
                "signature",
                "sig",
                "auth",
                "authorization",
            }
            if any(
                key.casefold() not in credential_fields
                for key, _ in parse_qsl(url.query, keep_blank_values=True)
            ):
                raise ValueError
            port = url.port
            if port == {"http": 80, "https": 443}[url.scheme.lower()]:
                port = None
            # No raw endpoint is retained, including in validation errors.
            route = [url.scheme.lower(), url.hostname.lower(), port, url.path.rstrip("/")]
            endpoint_digest = _digest(json.dumps(route, separators=(",", ":")))
        except (ValueError, TypeError):
            raise LLMConfigurationError("model identity endpoint is invalid") from None
    adapter_type = type(client)
    adapter_identity = f"{adapter_type.__module__}.{adapter_type.__qualname__}"
    if isinstance(client, OllamaClient):
        # Existing v1 identities remain byte-identical. The new adapter additionally
        # binds its installed digest, context/output budgets and thinking mode.
        adapter_identity += ":" + client.semantic_identity
    return ModelConfigurationIdentity(
        provider=provider,
        protocol=protocol,
        adapter_sha256=_digest(adapter_identity),
        model_sha256=_digest(config.model),
        endpoint_sha256=endpoint_digest,
        temperature=config.temperature,
        max_output_chars=config.max_output_chars,
        timeout_seconds=config.timeout_seconds,
        max_retries=config.max_retries,
    )
