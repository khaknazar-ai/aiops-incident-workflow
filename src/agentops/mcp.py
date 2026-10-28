"""Small, policy-preserving MCP stdio client for read-only gateway tools."""

from __future__ import annotations

import json
import os
import queue
import subprocess
import threading
import time
from collections.abc import Mapping, Sequence
from typing import Any

from agentops.tools import Actor, GatewayResult, ToolError, ToolGateway

MCP_PROTOCOL_VERSION = "2025-06-18"
_EVALUATOR_KEYS = frozenset(
    {"root_cause", "expected_action", "ground_truth", "hidden_ground_truth"}
)


class MCPError(ToolError):
    """Base class for bounded MCP transport and protocol failures."""


class MCPUnavailableError(MCPError):
    """The configured MCP subprocess is unavailable or exited."""


class MCPTimeoutError(MCPError):
    """The MCP subprocess did not answer before its deadline."""


class MCPProtocolError(MCPError):
    """The subprocess returned a malformed JSON-RPC/MCP response."""


class MCPDiscoveryError(MCPError):
    """The remote tool discovery does not match the pinned local registry."""


class MCPToolError(MCPError):
    """The MCP server returned an isError tool result."""


class MCPOutputError(MCPError):
    """The MCP result exceeded limits or failed output validation."""


class MCPInputError(MCPError):
    """The serialized request exceeds the configured transport limit."""


def _json_size(value: Any) -> int:
    return len(json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str))


def _contains_evaluator_key(value: Any) -> bool:
    if isinstance(value, Mapping):
        return any(
            str(key) in _EVALUATOR_KEYS or _contains_evaluator_key(item)
            for key, item in value.items()
        )
    if isinstance(value, list):
        return any(_contains_evaluator_key(item) for item in value)
    return False


class MCPClient:
    """Persistent line-delimited JSON-RPC client with pinned read-tool discovery."""

    def __init__(
        self,
        command: Sequence[str],
        *,
        gateway: ToolGateway,
        timeout_seconds: float = 2.0,
        max_output_chars: int = 20_000,
        max_request_chars: int = 20_000,
        env: Mapping[str, str] | None = None,
    ) -> None:
        if not command or any(not str(item).strip() for item in command):
            raise ValueError("MCP command must be a non-empty argument list")
        if timeout_seconds <= 0:
            raise ValueError("MCP timeout_seconds must be positive")
        if max_output_chars < 1:
            raise ValueError("MCP max_output_chars must be positive")
        if max_request_chars < 1:
            raise ValueError("MCP max_request_chars must be positive")
        self.command = tuple(str(item) for item in command)
        self.gateway = gateway
        self.timeout_seconds = timeout_seconds
        self.max_output_chars = max_output_chars
        self.max_request_chars = max_request_chars
        self.env = dict(env) if env is not None else None
        self._process: subprocess.Popen[str] | None = None
        self._responses: queue.Queue[str | None] = queue.Queue(maxsize=1)
        self._reader_stop = threading.Event()
        self._reader: threading.Thread | None = None
        self._writer: threading.Thread | None = None
        self._request_id = 0
        self._discovered: dict[str, Mapping[str, Any]] = {}

    def __enter__(self) -> MCPClient:
        self.start()
        return self

    def __exit__(self, _type: Any, _value: Any, _traceback: Any) -> None:
        self.close()

    @property
    def discovered_tools(self) -> tuple[str, ...]:
        return tuple(sorted(self._discovered))

    def start(self) -> None:
        if self._process is not None:
            return
        self._responses = queue.Queue(maxsize=1)
        self._reader_stop = threading.Event()
        try:
            # The executable and arguments come from trusted operator configuration;
            # shell=False prevents interpretation of incident or model-controlled text.
            self._process = subprocess.Popen(  # noqa: S603
                list(self.command),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=None,
                text=True,
                encoding="utf-8",
                bufsize=1,
                shell=False,
                env=self.env or os.environ.copy(),
            )
        except OSError as exc:
            raise MCPUnavailableError("MCP server could not be started") from exc
        if self._process.stdout is None:
            self.close()
            raise MCPUnavailableError("MCP server stdout pipe is unavailable")
        self._reader = threading.Thread(
            target=self._read_responses,
            args=(self._process.stdout,),
            name="agentops-mcp-reader",
            daemon=True,
        )
        self._reader.start()
        try:
            initialized = self._request(
                "initialize",
                {
                    "protocolVersion": MCP_PROTOCOL_VERSION,
                    "capabilities": {},
                    "clientInfo": {"name": "agentops-local-client", "version": "0.1.0"},
                },
            )
            if not isinstance(initialized, Mapping):
                raise MCPProtocolError("MCP initialize result must be an object")
            if initialized.get("protocolVersion") != MCP_PROTOCOL_VERSION:
                raise MCPDiscoveryError("MCP protocol version is not pinned to the local contract")
            self._notify("notifications/initialized")
            discovery = self._request("tools/list", {})
            self._pin_discovery(discovery)
        except MCPError:
            self.close()
            raise

    def close(self) -> None:
        process = self._process
        self._process = None
        self._discovered = {}
        self._reader_stop.set()
        if process is not None:
            # Termination releases a peer-blocked pipe before close can take its
            # buffered writer lock. Closing stdin first can wait indefinitely.
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=1.0)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=1.0)
        writer = self._writer
        self._writer = None
        if writer is not None and writer is not threading.current_thread():
            writer.join(timeout=1.0)
        if process is not None and process.stdin is not None:
            if writer is None or not writer.is_alive():
                try:
                    process.stdin.close()
                except OSError:
                    pass
        reader = self._reader
        self._reader = None
        if reader is not None and reader is not threading.current_thread():
            reader.join(timeout=1.0)

    def call(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        actor: Actor,
    ) -> GatewayResult:
        if self._process is None:
            self.start()
        definition, request = self.gateway.validate_read_request(tool_name, arguments, actor)
        remote = self._discovered.get(tool_name)
        if remote is None:
            raise MCPDiscoveryError(f"remote MCP tool is not pinned: {tool_name}")
        remote_schema = remote.get("inputSchema")
        if not isinstance(remote_schema, Mapping):
            raise MCPDiscoveryError(f"remote MCP schema is malformed: {tool_name}")
        if _schema_fingerprint(remote_schema) != _schema_fingerprint(
            definition.input_model.model_json_schema()
        ):
            raise MCPDiscoveryError(f"remote MCP schema changed: {tool_name}")
        response = self._request(
            "tools/call",
            {"name": tool_name, "arguments": request.model_dump(mode="json")},
        )
        if not isinstance(response, Mapping):
            raise MCPProtocolError("MCP tools/call result must be an object")
        if response.get("isError") is True:
            raise MCPToolError("remote MCP tool reported an error")
        structured = response.get("structuredContent")
        if not isinstance(structured, Mapping):
            raise MCPOutputError("MCP result is missing structuredContent")
        if _contains_evaluator_key(structured):
            raise MCPOutputError("MCP output contains evaluator-only fields")
        if _json_size(structured) > self.max_output_chars:
            raise MCPOutputError("MCP structured output exceeds the configured limit")
        try:
            payload = definition.output_model.model_validate(structured)
        except Exception as exc:
            raise MCPOutputError("MCP structured output does not match the pinned schema") from exc
        return GatewayResult(tool_name=tool_name, payload=payload, transport="mcp")

    def _pin_discovery(self, result: Any) -> None:
        if not isinstance(result, Mapping) or not isinstance(result.get("tools"), list):
            raise MCPDiscoveryError("MCP tools/list result is malformed")
        pinned: dict[str, Mapping[str, Any]] = {}
        for item in result["tools"]:
            if (
                not isinstance(item, Mapping)
                or not isinstance(item.get("name"), str)
                or not isinstance(item.get("inputSchema"), Mapping)
            ):
                raise MCPDiscoveryError("MCP tools/list contains malformed tool metadata")
            name = item["name"]
            if name in pinned:
                raise MCPDiscoveryError(f"MCP tools/list duplicates {name}")
            pinned[name] = item
        self._discovered = pinned

    def _notify(self, method: str) -> None:
        deadline = time.monotonic() + self.timeout_seconds
        self._write_message(
            json.dumps({"jsonrpc": "2.0", "method": method}) + "\n", method, deadline
        )

    def _write_message(self, message: str, method: str, deadline: float) -> None:
        process = self._process
        if process is None or process.stdin is None:
            raise MCPUnavailableError("MCP server is not running")
        if len(message) > self.max_request_chars:
            raise MCPInputError("MCP request exceeds the configured limit")
        outcomes: queue.Queue[Exception | None] = queue.Queue(maxsize=1)

        def write() -> None:
            try:
                process.stdin.write(message)
                process.stdin.flush()
            except Exception as exc:
                outcomes.put(exc)
            else:
                outcomes.put(None)

        self._writer = threading.Thread(target=write, name="agentops-mcp-writer", daemon=True)
        self._writer.start()
        try:
            error = outcomes.get(timeout=max(0.0, deadline - time.monotonic()))
        except queue.Empty as exc:
            self.close()
            raise MCPTimeoutError(f"MCP request timed out: {method}") from exc
        self._writer = None
        if error is not None:
            self.close()
            raise MCPUnavailableError("MCP server write failed") from error

    def _request(self, method: str, params: Mapping[str, Any]) -> Any:
        deadline = time.monotonic() + self.timeout_seconds
        process = self._process
        if process is None or process.stdin is None:
            raise MCPUnavailableError("MCP server is not running")
        self._request_id += 1
        request_id = self._request_id
        message = (
            json.dumps(
                {"jsonrpc": "2.0", "id": request_id, "method": method, "params": params},
                ensure_ascii=False,
                separators=(",", ":"),
            )
            + "\n"
        )
        self._write_message(message, method, deadline)
        try:
            raw = self._responses.get(timeout=max(0.0, deadline - time.monotonic()))
        except queue.Empty as exc:
            self.close()
            raise MCPTimeoutError(f"MCP request timed out: {method}") from exc
        if raw is None:
            self.close()
            raise MCPUnavailableError("MCP server exited without a response")
        if len(raw) > self.max_output_chars:
            self.close()
            raise MCPOutputError("MCP response exceeds the configured limit")
        try:
            response = json.loads(raw)
        except json.JSONDecodeError as exc:
            self.close()
            raise MCPProtocolError("MCP response is not valid JSON") from exc
        if not isinstance(response, Mapping) or response.get("jsonrpc") != "2.0":
            raise MCPProtocolError("MCP response is not JSON-RPC 2.0")
        if response.get("id") != request_id:
            raise MCPProtocolError("MCP response id does not match the request")
        if "error" in response:
            raise MCPProtocolError("MCP server returned a protocol error")
        if "result" not in response:
            raise MCPProtocolError("MCP response has neither result nor error")
        return response["result"]

    def _read_responses(self, stream: Any) -> None:
        try:
            while not self._reader_stop.is_set():
                line = stream.readline(self.max_output_chars + 2)
                if line == "":
                    break
                if not self._put_response(line.rstrip("\r\n")):
                    return
        finally:
            self._put_response(None)

    def _put_response(self, value: str | None) -> bool:
        while not self._reader_stop.is_set():
            try:
                self._responses.put(value, timeout=0.1)
                return True
            except queue.Full:
                continue
        return False


def _schema_fingerprint(schema: Mapping[str, Any]) -> str:
    return json.dumps(schema, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _result_error(result: Mapping[str, Any]) -> str:
    content = result.get("content")
    if isinstance(content, list):
        messages = [
            item.get("text")
            for item in content
            if isinstance(item, Mapping) and isinstance(item.get("text"), str)
        ]
        if messages:
            return messages[0][:500]
    return "MCP tool returned an error"


__all__ = [
    "MCPClient",
    "MCPDiscoveryError",
    "MCPError",
    "MCPInputError",
    "MCPOutputError",
    "MCPProtocolError",
    "MCPTimeoutError",
    "MCPToolError",
    "MCPUnavailableError",
]
