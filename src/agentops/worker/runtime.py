"""Concrete durable runtime binding for one command worker invocation."""

from __future__ import annotations

import math
import os
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

from agentops.domain import AgentRun, AgentRunStatus, utc_now
from agentops.llm import FakeLLMClient, LLMClient, ModelConfig
from agentops.mcp import MCPClient
from agentops.persistence import (
    AgentRunRepository,
    Database,
    EntityNotFound,
    EnvironmentReservationRepository,
)
from agentops.settings import LLMSettings, create_llm_client
from agentops.simulator import PersistentSimulator, ScenarioName
from agentops.tools import Actor, Role, ToolGateway
from agentops.worker.dispatch import DispatchMessage
from agentops.worker.tasks import RuntimeCommandError, RuntimeCommandHandler
from agentops.workflows.incident_triage import IncidentTriageRuntime

_MCP_CHILD_ENV_KEYS = (
    "COMSPEC",
    "HOME",
    "LANG",
    "LD_LIBRARY_PATH",
    "PATH",
    "PATHEXT",
    "PYTHONPATH",
    "SYSTEMROOT",
    "TEMP",
    "TMP",
    "USERPROFILE",
    "VIRTUAL_ENV",
    "WINDIR",
)


def _mcp_child_environment(database: Database) -> dict[str, str]:
    """Pass only runtime essentials and the one required database credential."""

    child_env = {
        key: value for key in _MCP_CHILD_ENV_KEYS if (value := os.environ.get(key)) is not None
    }
    child_env["AGENTOPS_DATABASE_URL"] = database.engine.url.render_as_string(hide_password=False)
    child_env["PYTHONIOENCODING"] = "utf-8"
    return child_env


class DurableRuntimeFactory:
    """Build one run-scoped graph runtime from durable command identity."""

    def __init__(
        self,
        database: Database,
        *,
        checkpoint_path: str | Path | None = None,
        checkpoint_backend: str | None = None,
        llm_factory: Callable[[], LLMClient] | None = None,
        model_config: ModelConfig | None = None,
        llm_settings: LLMSettings | None = None,
        owner_prefix: str = "worker-runtime",
        mcp_read_tool: str | None = None,
        max_active_seconds: float = 90.0,
    ) -> None:
        if not owner_prefix.strip():
            raise ValueError("owner_prefix must be non-empty")
        if not math.isfinite(max_active_seconds) or not 0 < max_active_seconds <= 1800:
            raise ValueError("runtime active budget must be between 0 and 1800 seconds")
        self.max_active_seconds = max_active_seconds
        if llm_factory is not None and llm_settings is not None:
            raise ValueError("llm_factory and llm_settings are mutually exclusive")
        if llm_settings is not None and model_config is not None:
            raise ValueError("model_config is derived from llm_settings")
        if mcp_read_tool is not None and mcp_read_tool != "get_service_status":
            raise ValueError("only get_service_status may be routed through local MCP")
        self.database = database
        self.checkpoint_path = checkpoint_path
        self.checkpoint_backend = checkpoint_backend
        self.model_config = model_config or ModelConfig()
        self.command_lease_seconds = 60.0
        if llm_settings is not None:
            self.llm_factory = lambda: create_llm_client(llm_settings)
            self.model_config = llm_settings.model_config_for()
            if llm_settings.provider.strip().casefold() == "ollama":
                # A broker redelivery must not admit a second command claimant
                # while a bounded local-model investigation is still active.
                self.command_lease_seconds = max(
                    60.0, max_active_seconds + llm_settings.timeout_seconds + 5.0
                )
        else:
            # Backward-compatible offline mode is explicit at the factory
            # boundary; environment-driven live mode uses from_environment().
            self.llm_factory = llm_factory or FakeLLMClient
        self.owner_prefix = owner_prefix
        self.mcp_read_tool = mcp_read_tool

    @classmethod
    def from_environment(
        cls,
        database: Database,
        *,
        checkpoint_path: str | Path | None = None,
        owner_prefix: str = "worker-runtime",
    ) -> DurableRuntimeFactory:
        settings = LLMSettings.from_environment()
        mcp_read_tool = os.getenv("AGENTOPS_MCP_READ_TOOL") or None
        return cls(
            database,
            checkpoint_path=checkpoint_path,
            checkpoint_backend=os.getenv("AGENTOPS_CHECKPOINT_BACKEND") or None,
            owner_prefix=owner_prefix,
            llm_settings=settings,
            mcp_read_tool=mcp_read_tool,
            max_active_seconds=float(os.getenv("AGENTOPS_RUN_MAX_ACTIVE_SECONDS", "90")),
        )

    def __call__(self, message: DispatchMessage) -> RuntimeCommandHandler:
        run_id = self._parse_uuid(message.payload.get("run_id"), "run_id")
        if message.event_type == "run.cancel_requested":
            self._resolve_run(run_id, message, require_environment=False)
            return _CancelRuntimeHandler(self.database, run_id)
        run, environment_id = self._resolve_run(run_id, message, require_environment=True)
        simulator = PersistentSimulator(self.database, environment_id)
        gateway = ToolGateway(
            simulator,
            database=self.database,
            execution_scope=f"run:{run.id}",
        )
        if self.mcp_read_tool is not None:
            read_tool_adapter = MCPClient(
                [
                    sys.executable,
                    "-m",
                    "agentops.mcp_server",
                    "--environment-id",
                    str(environment_id),
                ],
                gateway=gateway,
                env=_mcp_child_environment(self.database),
            )
            gateway.bind_read_transport(self.mcp_read_tool, read_tool_adapter)
        runtime = IncidentTriageRuntime(
            simulator,
            gateway,
            self.llm_factory(),
            model_config=self.model_config,
            database=self.database,
            checkpoint_path=self.checkpoint_path,
            checkpoint_backend=self.checkpoint_backend,
            lease_owner_id=f"{self.owner_prefix}:{run.id}:{uuid4()}",
            max_active_seconds=self.max_active_seconds,
        )
        return _GraphRuntimeHandler(
            runtime,
            run_id=run.id,
            environment_id=environment_id,
            scenario=self._scenario_from_message(message),
        )

    def _resolve_run(
        self,
        run_id: UUID,
        message: DispatchMessage,
        *,
        require_environment: bool,
    ) -> tuple[AgentRun, UUID | None]:
        with self.database.session() as session:
            try:
                run = AgentRunRepository(session).require(run_id)
            except EntityNotFound as exc:
                raise RuntimeCommandError("durable run does not exist") from exc
        payload_incident_id = message.payload.get("incident_id")
        if payload_incident_id is not None and str(payload_incident_id) != str(run.incident_id):
            raise RuntimeCommandError("command incident_id does not match the durable run")
        durable_environment = run.environment_id
        payload_environment = message.payload.get("environment_id")
        if payload_environment is not None:
            parsed_environment = self._parse_uuid(payload_environment, "environment_id")
            if durable_environment != parsed_environment:
                raise RuntimeCommandError("command environment_id does not match the durable run")
        if require_environment and durable_environment is None:
            raise RuntimeCommandError("run is not bound to an environment")
        return run, durable_environment

    @staticmethod
    def _parse_uuid(value: Any, field: str) -> UUID:
        try:
            return UUID(str(value))
        except (TypeError, ValueError) as exc:
            raise RuntimeCommandError(f"command {field} must be a UUID") from exc

    @staticmethod
    def _scenario_from_message(message: DispatchMessage) -> ScenarioName | None:
        if message.event_type != "run.created":
            return None
        configuration = message.payload.get("configuration")
        if configuration is None:
            return None
        if not isinstance(configuration, dict):
            raise RuntimeCommandError("run configuration must be an object")
        scenario = configuration.get("scenario")
        if scenario is None:
            return None
        try:
            return ScenarioName(str(scenario))
        except ValueError as exc:
            raise RuntimeCommandError("run configuration scenario is invalid") from exc


class _GraphRuntimeHandler(RuntimeCommandHandler):
    def __init__(
        self,
        runtime: IncidentTriageRuntime,
        *,
        run_id: UUID,
        environment_id: UUID,
        scenario: ScenarioName | None,
    ) -> None:
        self.runtime = runtime
        self.run_id = run_id
        self.environment_id = environment_id
        self.scenario = scenario

    def start(self, _message: DispatchMessage) -> Any:
        try:
            return self.runtime.run_existing(
                run_id=self.run_id,
                environment_id=self.environment_id,
                scenario=self.scenario,
            )
        finally:
            self.runtime.close()

    def resume_approval(self, message: DispatchMessage) -> Any:
        payload = message.payload
        approved = payload.get("approved")
        if not isinstance(approved, bool):
            raise RuntimeCommandError("approval command requires a strict boolean approved")
        actor_id = payload.get("decided_by")
        actor_role = payload.get("actor_role")
        if not isinstance(actor_id, str) or not actor_id.strip():
            raise RuntimeCommandError("approval command requires decided_by")
        if not isinstance(actor_role, str):
            raise RuntimeCommandError("approval command requires actor_role")
        try:
            actor = Actor(actor_id=actor_id, role=Role(actor_role))
        except ValueError as exc:
            raise RuntimeCommandError("approval command actor_role is invalid") from exc
        approved_parameters = payload.get("approved_parameters")
        if approved_parameters is not None and not isinstance(approved_parameters, dict):
            raise RuntimeCommandError("approval command approved_parameters must be an object")
        reason = payload.get("decision_reason")
        if reason is not None and not isinstance(reason, str):
            raise RuntimeCommandError("approval command decision_reason must be a string")
        try:
            return self.runtime.resume_approval(
                actor=actor,
                approved=approved,
                approved_parameters=approved_parameters,
                reason=reason,
                run_id=self.run_id,
            )
        finally:
            self.runtime.close()

    def cancel(self, _message: DispatchMessage) -> Any:
        raise RuntimeCommandError("cancel handler is not a graph runtime operation")


class _CancelRuntimeHandler(RuntimeCommandHandler):
    def __init__(self, database: Database, run_id: UUID) -> None:
        self.database = database
        self.run_id = run_id

    def start(self, _message: DispatchMessage) -> Any:
        raise RuntimeCommandError("cancel handler cannot start a run")

    def resume_approval(self, _message: DispatchMessage) -> Any:
        raise RuntimeCommandError("cancel handler cannot resume approval")

    def cancel(self, _message: DispatchMessage) -> Any:
        with self.database.session() as session:
            run = AgentRunRepository(session).require_for_update(self.run_id)
            if run.status not in {
                AgentRunStatus.CANCELLED,
                AgentRunStatus.SUCCEEDED,
                AgentRunStatus.FAILED,
                AgentRunStatus.TIMED_OUT,
                AgentRunStatus.BLOCKED,
            }:
                if run.status in {
                    AgentRunStatus.QUEUED,
                    AgentRunStatus.WAITING_FOR_APPROVAL,
                    AgentRunStatus.WAITING_FOR_RETRY,
                }:
                    if run.cancel_requested_at is None:
                        raise RuntimeCommandError(
                            "cancel command requires a committed cancellation intent"
                        )
                    run.transition_to(AgentRunStatus.CANCELLED)
                    AgentRunRepository(session).save(run)
                elif run.status is AgentRunStatus.RUNNING:
                    if run.cancel_requested_at is None:
                        raise RuntimeCommandError(
                            "running command cannot be cancelled without a committed intent"
                        )
                    # The active worker owns the next-step/effect fence. Keep the
                    # run RUNNING until that worker observes the committed intent.
                    return run
                else:
                    raise RuntimeCommandError("run cannot be cancelled safely")
            reservation = EnvironmentReservationRepository(session).get_for_run(run.id)
            if reservation is not None and reservation.released_at is None:
                EnvironmentReservationRepository(session).release(
                    reservation.environment_id,
                    run.id,
                    generation=reservation.generation,
                    now=utc_now(),
                )
            return run


__all__ = ["DurableRuntimeFactory"]
