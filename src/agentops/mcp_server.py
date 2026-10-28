"""Local read-only MCP server over line-delimited JSON-RPC stdio."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping
from typing import Any, TextIO
from uuid import UUID

from agentops.database_config import database_url_from_environment
from agentops.domain import RiskLevel
from agentops.mcp import MCP_PROTOCOL_VERSION
from agentops.persistence import Database
from agentops.simulator import PersistentSimulator
from agentops.tools import Actor, Role, ToolError, ToolGateway


class MCPServer:
    """Serve only the R0 read tools registered by the existing ToolGateway."""

    def __init__(
        self,
        gateway: ToolGateway,
        *,
        input_stream: TextIO = sys.stdin,
        output_stream: TextIO = sys.stdout,
        max_output_chars: int = 20_000,
    ) -> None:
        if max_output_chars < 1:
            raise ValueError("max_output_chars must be positive")
        self.gateway = gateway
        self.input_stream = input_stream
        self.output_stream = output_stream
        self.max_output_chars = max_output_chars

    def run(self) -> int:
        for line in self.input_stream:
            if len(line) > self.max_output_chars:
                self._write_error(None, -32600, "request exceeds the configured limit")
                continue
            try:
                request = json.loads(line)
            except json.JSONDecodeError:
                self._write_error(None, -32700, "request is not valid JSON")
                continue
            response = self.handle(request)
            if response is not None:
                self._write(response)
        return 0

    def handle(self, request: Any) -> dict[str, Any] | None:
        if not isinstance(request, Mapping):
            return self._error(None, -32600, "request must be an object")
        request_id = request.get("id")
        method = request.get("method")
        if not isinstance(method, str) or not method:
            return self._error(request_id, -32600, "method is required")
        if method == "notifications/initialized":
            return None
        if method == "initialize":
            return self._result(
                request_id,
                {
                    "protocolVersion": MCP_PROTOCOL_VERSION,
                    "capabilities": {"tools": {"listChanged": False}},
                    "serverInfo": {"name": "agentops-local-mcp", "version": "0.1.0"},
                },
            )
        if method == "tools/list":
            return self._result(request_id, {"tools": self._tool_list()})
        if method == "tools/call":
            return self._call(request_id, request.get("params"))
        return self._error(request_id, -32601, f"method not found: {method}")

    def _tool_list(self) -> list[dict[str, Any]]:
        tools: list[dict[str, Any]] = []
        for definition in self.gateway.definitions():
            if definition.side_effecting or definition.risk_level is not RiskLevel.R0:
                continue
            tools.append(
                {
                    "name": definition.name,
                    "description": definition.description,
                    "inputSchema": definition.input_model.model_json_schema(),
                    "x-agentops-outputSchema": definition.output_model.model_json_schema(),
                }
            )
        return tools

    def _call(self, request_id: Any, params: Any) -> dict[str, Any]:
        if not isinstance(params, Mapping):
            return self._error(request_id, -32602, "tools/call params must be an object")
        name = params.get("name")
        arguments = params.get("arguments", {})
        if not isinstance(name, str) or not name:
            return self._error(request_id, -32602, "tools/call name is required")
        if not isinstance(arguments, Mapping):
            return self._error(request_id, -32602, "tools/call arguments must be an object")
        try:
            definition, request = self.gateway.validate_read_request(
                name,
                dict(arguments),
                Actor(actor_id="mcp-client", role=Role.VIEWER),
            )
            gateway_result = self.gateway.call(
                definition.name,
                request.model_dump(mode="json"),
                Actor(actor_id="mcp-client", role=Role.VIEWER),
            )
            structured = gateway_result.payload.model_dump(mode="json")
            serialized = json.dumps(
                structured,
                ensure_ascii=False,
                separators=(",", ":"),
                default=str,
            )
            if len(serialized) > self.max_output_chars:
                raise ToolError("MCP structured output exceeds the configured limit")
            return self._result(
                request_id,
                {
                    "isError": False,
                    "content": [{"type": "text", "text": serialized}],
                    "structuredContent": structured,
                },
            )
        except ToolError as exc:
            return self._result(
                request_id,
                {
                    "isError": True,
                    "content": [{"type": "text", "text": str(exc)[:500]}],
                },
            )
        except Exception:
            return self._result(
                request_id,
                {
                    "isError": True,
                    "content": [{"type": "text", "text": "MCP tool execution failed"}],
                },
            )

    def _write(self, response: Mapping[str, Any]) -> None:
        encoded = json.dumps(response, ensure_ascii=False, separators=(",", ":"))
        if len(encoded) > self.max_output_chars:
            response = self._error(
                response.get("id"), -32000, "response exceeds the configured limit"
            )
            encoded = json.dumps(response, ensure_ascii=False, separators=(",", ":"))
        self.output_stream.write(encoded + "\n")
        self.output_stream.flush()

    @staticmethod
    def _result(request_id: Any, result: Mapping[str, Any]) -> dict[str, Any]:
        return {"jsonrpc": "2.0", "id": request_id, "result": dict(result)}

    @staticmethod
    def _error(request_id: Any, code: int, message: str) -> dict[str, Any]:
        return {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}

    def _write_error(self, request_id: Any, code: int, message: str) -> None:
        self._write(self._error(request_id, code, message))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AgentOps local MCP read server")
    parser.add_argument("--database-url")
    parser.add_argument("--environment-id", required=True)
    parser.add_argument("--max-output-chars", type=int, default=20_000)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    database = Database(args.database_url or database_url_from_environment())
    simulator = PersistentSimulator(database, UUID(args.environment_id))
    gateway = ToolGateway(
        simulator,
        database=database,
        execution_scope=f"mcp:{args.environment_id}",
    )
    return MCPServer(gateway, max_output_chars=args.max_output_chars).run()


if __name__ == "__main__":
    raise SystemExit(main())
