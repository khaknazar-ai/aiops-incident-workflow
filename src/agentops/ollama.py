"""Bounded, loopback-only Ollama chat adapter; no credentials or fallback."""

from __future__ import annotations

import hashlib
import http.client
import json
import re
import socket
import time
from collections.abc import Sequence
from threading import Timer
from typing import Any, Protocol
from urllib.parse import urlsplit

from pydantic import ValidationError

from agentops.incident_instructions import incident_task_instructions, native_response_schema
from agentops.llm import (
    DiagnosticPlan,
    HypothesisUpdate,
    LLMConfigurationError,
    LLMIncompleteError,
    LLMOutputError,
    LLMProviderError,
    LLMResult,
    LLMTimeoutError,
    LLMUnavailableError,
    Message,
    ModelConfig,
    T,
    ToolDecision,
    _last_json_payload,
    _usage_count,
)
from agentops.native_hypotheses import INSTRUCTIONS, HypothesisProjection, NativeHypothesisUpdate


def local_endpoint(value: str) -> tuple[str, int]:
    """Literal loopback only: no DNS, proxy, userinfo, query or redirected route."""
    try:
        url = urlsplit(value)
        if (
            url.scheme != "http"
            or url.hostname not in {"127.0.0.1", "::1"}
            or url.username is not None
            or url.password is not None
            or url.path not in {"", "/"}
            or url.query
            or url.fragment
            or url.port is None
            or not 1 <= url.port <= 65535
        ):
            raise ValueError
        return url.hostname, url.port
    except (ValueError, TypeError):
        raise LLMConfigurationError("Ollama requires an explicit HTTP loopback origin") from None


class OllamaTransport(Protocol):
    def request(
        self, path: str, payload: dict[str, Any] | None, *, deadline: float
    ) -> dict[str, Any]: ...


class LocalOllamaTransport:
    """Direct socket HTTP; deadline shared by metadata and inference, no redirects."""

    MAX_RESPONSE_BYTES = 1_048_576

    def __init__(self, base_url: str) -> None:
        self.host, self.port = local_endpoint(base_url)

    def request(self, path, payload, *, deadline):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise LLMTimeoutError()
        connection = http.client.HTTPConnection(self.host, self.port, timeout=remaining)
        sock = None

        def expire():
            active_socket = sock or connection.sock
            if active_socket is not None:
                try:
                    active_socket.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass

        timer = Timer(remaining, expire)
        timer.daemon = True
        timer.start()
        try:
            body = None if payload is None else json.dumps(payload).encode("utf-8")
            connection.request(
                "GET" if body is None else "POST",
                path,
                body=body,
                headers={"Content-Type": "application/json"},
            )
            sock = connection.sock
            if sock is not None:
                sock.settimeout(max(0.001, deadline - time.monotonic()))
            response = connection.getresponse()
            if response.status != 200:
                raise LLMUnavailableError()  # Never echo the response body or follow Location.
            chunks = bytearray()
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise LLMTimeoutError()
                if sock is not None:
                    sock.settimeout(remaining)
                chunk = response.read1(min(65536, self.MAX_RESPONSE_BYTES + 1 - len(chunks)))
                chunks.extend(chunk)
                if len(chunks) > self.MAX_RESPONSE_BYTES:
                    raise LLMProviderError("LLMOutputTooLarge", "Ollama envelope exceeds limit")
                if not chunk:
                    break
            if time.monotonic() > deadline:
                raise LLMTimeoutError()
            result = json.loads(chunks)
            if not isinstance(result, dict):
                raise LLMOutputError()
            return result
        except TimeoutError:
            raise LLMTimeoutError() from None
        except (OSError, http.client.HTTPException):
            if time.monotonic() >= deadline:
                raise LLMTimeoutError() from None
            raise LLMUnavailableError() from None
        except (ValueError, UnicodeError):
            raise LLMOutputError() from None
        finally:
            timer.cancel()
            connection.close()


class OllamaClient:
    """One physical inference per call, with locally installed model digest pinning."""

    CONTRACT = "ollama-chat-v12-native-hypothesis-vocabulary"
    MAX_REQUEST_BYTES = 131_072

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        model_digest: str,
        num_ctx: int = 16_384,
        num_predict: int = 2048,
        think: bool = False,
        transport: OllamaTransport | None = None,
    ) -> None:
        local_endpoint(base_url)
        if not re.fullmatch(r"[A-Za-z0-9_.:/-]+", model) or "cloud" in model.casefold():
            raise LLMConfigurationError("Ollama requires a local model name")
        if not re.fullmatch(r"[0-9a-f]{64}", model_digest):
            raise LLMConfigurationError("Ollama requires the installed model SHA-256 digest")
        if not 4096 <= num_ctx <= 32768 or not 128 <= num_predict <= min(4096, num_ctx // 2):
            raise LLMConfigurationError("Ollama context/output budgets are invalid")
        if type(think) is not bool:
            raise LLMConfigurationError("Ollama thinking must be an explicit boolean")
        self.base_url, self.model, self.model_digest = base_url, model, model_digest
        self.num_ctx, self.num_predict = num_ctx, num_predict
        self.think = think
        self._transport = transport or LocalOllamaTransport(base_url)

    @property
    def semantic_identity(self) -> str:
        instructions = [
            incident_task_instructions(task)
            for task in (
                "triage",
                "diagnostic_plan",
                "tool_selection",
                "hypothesis_update",
                "remediation_proposal",
                "review",
            )
        ]
        return json.dumps(
            [
                self.CONTRACT,
                self.model_digest,
                self.num_ctx,
                self.num_predict,
                self.think,
                hashlib.sha256(json.dumps(instructions).encode()).hexdigest(),
                hashlib.sha256(INSTRUCTIONS.encode()).hexdigest(),
            ],
            separators=(",", ":"),
        )

    def generate_structured(
        self,
        *,
        task: str,
        messages: Sequence[Message],
        response_model: type[T],
        config: ModelConfig,
    ) -> LLMResult:
        if (
            not task.strip()
            or config.model != self.model
            or (
                config.base_url is not None
                and local_endpoint(config.base_url) != local_endpoint(self.base_url)
            )
            or config.timeout_seconds > 90
            or config.max_retries != 0
        ):
            raise LLMConfigurationError(
                "Ollama requires matching identity and bounded single calls"
            )
        projection = None
        context = _last_json_payload(messages)
        if task == "hypothesis_update" and response_model is HypothesisUpdate:
            projection = HypothesisProjection(context)
            schema = projection.schema()
        else:
            schema = native_response_schema(task, response_model, context)
        system_parts = [message.content for message in messages if message.role == "system"]
        if projection is not None:
            # The native schema uses f/h references and server-owned bookkeeping.
            # The domain DTO instructions require UUID/id/version fields absent
            # from this schema. INSTRUCTIONS includes the diagnosis/safety rules
            # for the native vocabulary; do not give the model two wire contracts.
            system_parts = [INSTRUCTIONS]
        else:
            system_parts.append(incident_task_instructions(task))
        system_parts += [
            "Return a single JSON object matching this schema: "
            + json.dumps(schema, separators=(",", ":")),
        ]
        schema_instruction = Message(
            role="system",
            content="\n\n".join(system_parts),
        )
        request = {
            "model": self.model,
            "messages": [schema_instruction.model_dump()]
            + (
                [{"role": "user", "content": json.dumps({"untrusted_context": projection.context})}]
                if projection is not None
                else [message.model_dump() for message in messages if message.role != "system"]
            ),
            "format": schema,
            "stream": False,
            "think": self.think,
            "options": {
                "temperature": config.temperature,
                "num_ctx": self.num_ctx,
                "num_predict": self.num_predict,
            },
        }
        if len(json.dumps(request).encode("utf-8")) > self.MAX_REQUEST_BYTES:
            raise LLMConfigurationError("Ollama request exceeds the bounded context envelope")
        deadline = time.monotonic() + config.timeout_seconds
        inference_started = False
        usage: dict[str, Any] = {}
        try:
            tags = self._transport.request("/api/tags", None, deadline=deadline)
            models = tags.get("models")
            if not isinstance(models, list):
                raise LLMOutputError()
            matches = [
                item for item in models if isinstance(item, dict) and item.get("name") == self.model
            ]
            if len(matches) != 1 or matches[0].get("digest") != self.model_digest:
                raise LLMConfigurationError(
                    "Installed Ollama model digest does not match the profile"
                )
            if matches[0].get("remote_host") or matches[0].get("remote_model"):
                raise LLMConfigurationError("Remote Ollama models are not supported")
            inference_started = True
            data = self._transport.request("/api/chat", request, deadline=deadline)
            usage = {
                "input_tokens": _usage_count(data, "prompt_eval_count"),
                "output_tokens": _usage_count(data, "eval_count"),
            }
            if data.get("model") != self.model:
                raise LLMOutputError("Ollama response model does not match")
            if data.get("done") is not True or data.get("done_reason") != "stop":
                raise LLMIncompleteError()
            message = data.get("message")
            if (
                not isinstance(message, dict)
                or message.get("role") != "assistant"
                or message.get("tool_calls")
            ):
                raise LLMOutputError()
            content = message.get("content")
            if not isinstance(content, str):
                raise LLMOutputError()
            if len(content) > config.max_output_chars:
                raise LLMProviderError("LLMOutputTooLarge", "Ollama content exceeds limit")
            try:
                output = (
                    projection.project(NativeHypothesisUpdate.model_validate_json(content))
                    if projection is not None
                    else response_model.model_validate_json(content)
                )
            except (ValidationError, ValueError):
                raise LLMOutputError() from None
            if isinstance(output, DiagnosticPlan) and len(output.steps) > 4:
                # A provider may ignore format constraints. Never spend the steps
                # reserved for approval/action/verification on an oversized plan.
                raise LLMOutputError("local diagnostic plan exceeds its reserved step budget")
            if isinstance(output, ToolDecision) and output.tool_name == "search_logs":
                if output.arguments.query != " " or output.arguments.limit != 20:
                    raise LLMOutputError("initial diagnostic log sample contract was violated")
            if isinstance(output, HypothesisUpdate) and any(
                item.claim is None for item in output.hypotheses
            ):
                raise LLMOutputError("local hypotheses require an explicit structured claim")
            return LLMResult(output=output, model=self.model, **usage)
        except LLMProviderError as exc:
            exc.attempts = int(inference_started)
            exc.usage = usage
            raise
