"""Explicit provider settings and construction for the LLM boundary."""

from __future__ import annotations

import os

from pydantic import BaseModel, Field, SecretStr, ValidationError

from agentops.llm import (
    FakeLLMClient,
    LLMClient,
    LLMConfigurationError,
    ModelConfig,
    OpenAIResponsesClient,
    ResponsesTransport,
)
from agentops.ollama import OllamaClient, OllamaTransport


class LLMSettings(BaseModel):
    """Configuration that never silently changes a live provider into a fake one."""

    provider: str = Field(min_length=1)
    protocol: str = Field(default="responses", min_length=1)
    model: str = Field(default="fake-deterministic", min_length=1)
    api_key: SecretStr | None = None
    base_url: str | None = None
    timeout_seconds: float = Field(default=10.0, gt=0.0)
    max_output_chars: int = Field(default=20_000, ge=1, le=1_000_000)
    max_retries: int = Field(default=0, ge=0, le=2)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    model_digest: str | None = None
    num_ctx: int = Field(default=16_384, ge=4096, le=32768)
    num_predict: int = Field(default=2048, ge=128, le=4096)
    think: bool = False

    @classmethod
    def from_environment(cls) -> LLMSettings:
        provider = os.getenv("AGENTOPS_LLM_PROVIDER")
        if not provider or not provider.strip():
            raise LLMConfigurationError(
                "AGENTOPS_LLM_PROVIDER must explicitly select fake, openai_responses or ollama"
            )
        values: dict[str, object] = {
            "provider": provider.strip(),
            "protocol": os.getenv("AGENTOPS_LLM_PROTOCOL", "responses"),
            "model": os.getenv("AGENTOPS_LLM_MODEL", "fake-deterministic"),
            "base_url": os.getenv("AGENTOPS_LLM_BASE_URL"),
            "model_digest": os.getenv("AGENTOPS_LLM_MODEL_DIGEST"),
        }
        api_key = (
            os.getenv("AGENTOPS_LLM_API_KEY") if provider.strip().casefold() != "ollama" else None
        )
        if api_key:
            values["api_key"] = api_key
        for field, env_name in (
            ("timeout_seconds", "AGENTOPS_LLM_TIMEOUT_SECONDS"),
            ("max_output_chars", "AGENTOPS_LLM_MAX_OUTPUT_CHARS"),
            ("max_retries", "AGENTOPS_LLM_MAX_RETRIES"),
            ("temperature", "AGENTOPS_LLM_TEMPERATURE"),
            ("num_ctx", "AGENTOPS_LLM_NUM_CTX"),
            ("num_predict", "AGENTOPS_LLM_NUM_PREDICT"),
            ("think", "AGENTOPS_LLM_THINK"),
        ):
            value = os.getenv(env_name)
            if value is not None:
                values[field] = value
        try:
            settings = cls.model_validate(values)
        except ValidationError as exc:
            raise LLMConfigurationError("LLM environment settings are invalid") from exc
        return settings

    def model_config_for(self) -> ModelConfig:
        return ModelConfig(
            model=self.model,
            temperature=self.temperature,
            timeout_seconds=self.timeout_seconds,
            max_output_chars=self.max_output_chars,
            max_retries=self.max_retries,
            base_url=self.base_url,
        )


def create_llm_client(
    settings: LLMSettings,
    *,
    transport: ResponsesTransport | None = None,
    ollama_transport: OllamaTransport | None = None,
) -> LLMClient:
    provider = settings.provider.strip().casefold()
    protocol = settings.protocol.strip().casefold()
    if provider == "ollama":
        if (
            protocol != "ollama_chat"
            or settings.api_key is not None
            or settings.max_retries != 0
            or settings.timeout_seconds > 90
            or transport is not None
        ):
            raise LLMConfigurationError(
                "Ollama requires keyless ollama_chat and bounded single calls"
            )
        return OllamaClient(
            base_url=settings.base_url or "",
            model=settings.model,
            model_digest=settings.model_digest or "",
            num_ctx=settings.num_ctx,
            num_predict=settings.num_predict,
            think=settings.think,
            transport=ollama_transport,
        )
    if provider in {"fake", "offline"}:
        if protocol not in {"fake", "offline", "responses"}:
            raise LLMConfigurationError("fake provider requires fake/offline protocol")
        return FakeLLMClient()
    if provider in {"openai_responses", "openai-responses", "openai"}:
        if protocol not in {"responses", "openai_responses", "openai-responses"}:
            raise LLMConfigurationError("OpenAI provider requires the Responses protocol")
        if settings.model.strip().casefold().startswith("fake"):
            raise LLMConfigurationError("live provider requires an explicit non-fake model")
        if settings.api_key is None or not settings.api_key.get_secret_value().strip():
            raise LLMConfigurationError("OpenAI Responses provider requires AGENTOPS_LLM_API_KEY")
        return OpenAIResponsesClient(
            api_key=settings.api_key.get_secret_value(),
            base_url=settings.base_url,
            transport=transport,
        )
    raise LLMConfigurationError(f"unsupported LLM provider: {settings.provider}")


def create_llm_client_from_environment(
    *, transport: ResponsesTransport | None = None
) -> tuple[LLMClient, ModelConfig]:
    settings = LLMSettings.from_environment()
    return create_llm_client(settings, transport=transport), settings.model_config_for()
