"""Policy-enforced tool contracts and gateway for simulator operations."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Literal, Protocol

from pydantic import BaseModel, Field, ValidationError
from sqlalchemy.orm import Session

from agentops.domain import ApprovalRequest, RiskLevel, stable_parameter_hash
from agentops.persistence import (
    Database,
    ExecutionInProgressError,
    ExecutionLedgerRepository,
    ExecutionStatus,
    RunLeaseFencedError,
    SimulationEnvironmentRepository,
)
from agentops.persistence import (
    ExecutionConflictError as LedgerConflictError,
)
from agentops.simulator import PersistentSimulator, SimulationError, Simulator


class ToolError(RuntimeError):
    """Base error for tool gateway failures."""


class ToolNotFound(ToolError):
    """Requested tool is not in the allow-listed registry."""


class ToolValidationError(ToolError):
    """Tool input or output failed schema validation."""


class PermissionDenied(ToolError):
    """Actor lacks the permission required by a tool."""


class ForbiddenAction(ToolError):
    """The risk model forbids this operation even when requested by an admin."""


class ApprovalRequired(ToolError):
    """A valid, parameter-bound approval is required."""


class IdempotencyKeyRequired(ToolError):
    """Every side-effecting call must provide an idempotency key."""


class IdempotencyConflict(ToolError):
    """An idempotency key was previously used for another tool."""


class ToolExecutionError(ToolError):
    """The adapter or simulator failed while executing a validated call."""


class Role(StrEnum):
    VIEWER = "viewer"
    OPERATOR = "operator"
    APPROVER = "approver"
    ADMIN = "admin"
    AUDITOR = "auditor"


class Actor(BaseModel):
    actor_id: str = Field(min_length=1)
    role: Role


class ServiceInput(BaseModel):
    service: str = Field(min_length=1)


class MetricsInput(BaseModel):
    service: str = Field(min_length=1)
    metric_names: list[str] = Field(min_length=1, max_length=20)
    lookback_minutes: int = Field(default=15, ge=1, le=1440)


class LogSearchInput(BaseModel):
    service: str = Field(min_length=1)
    query: str = Field(
        min_length=1,
        max_length=200,
        description=(
            "Nonblank text is a literal case-insensitive substring. A whitespace-only query "
            "requests the latest bounded sample in stored order, without a keyword filter. "
            "truncated/total_matches describe omitted records; a sample is not complete history."
        ),
    )
    limit: int = Field(default=20, ge=1, le=100)


class RollbackInput(BaseModel):
    service: str = Field(min_length=1)
    target_version: str = Field(min_length=1)


class DeploymentRecord(BaseModel):
    version: str
    status: str


class ServiceStatusOutput(BaseModel):
    service: str
    health: str
    version: str
    instances: dict[str, int]
    dependencies: dict[str, str]
    worker_running: bool
    queue_depth: int
    # Missing on historical/external payloads means unknown, not unsupported.
    capabilities: list[Literal["service_process", "managed_queue_worker"]] | None = None


class MetricsOutput(BaseModel):
    service: str
    metrics: dict[str, float]
    units: dict[str, str]
    lookback_minutes: int


class LogSearchOutput(BaseModel):
    service: str
    query: str
    entries: list[str]
    total_matches: int
    truncated: bool


class DeploymentHistoryOutput(BaseModel):
    service: str
    deployments: list[DeploymentRecord]


class ServiceConfigOutput(BaseModel):
    service: str
    values: dict[str, str]


class ActionExecutionOutput(BaseModel):
    service: str
    action: str
    changed: bool
    message: str
    idempotency_key: str


class VerificationOutput(BaseModel):
    service: str
    healthy: bool
    checks: dict[str, bool]
    observed: dict[str, Any]


class GatewayResult(BaseModel):
    tool_name: str
    payload: BaseModel
    replayed: bool = False
    transport: str = "local"


class ReadToolTransport(Protocol):
    """A bounded execution adapter selected only by the gateway."""

    def call(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        actor: Actor,
    ) -> GatewayResult: ...

    def close(self) -> None: ...


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    description: str
    input_model: type[BaseModel]
    output_model: type[BaseModel]
    risk_level: RiskLevel
    required_permission: str
    timeout_seconds: float
    max_retries: int
    side_effecting: bool
    handler: Callable[[BaseModel, str | None], dict[str, Any]]
    required_capability: Literal["service_process", "managed_queue_worker"] | None = None
    reject_known_healthy_target: bool = False
    requires_healthy_dependencies: bool = False

    def observed_support(self, service: object, observations: object) -> bool | None:
        """Known resource applicability, not authorization or a live execution guarantee."""
        if not isinstance(observations, list):
            return None
        if not isinstance(service, str) or not service:
            return None
        for item in reversed(observations):
            if not isinstance(item, dict) or item.get("tool_name") != "get_service_status":
                continue
            payload = item.get("payload")
            if not isinstance(payload, dict) or payload.get("service") != service:
                continue
            checks: list[bool | None] = []
            if self.required_capability is not None:
                capabilities = payload.get("capabilities")
                # A newer unknown/malformed observation must not revive older support.
                if not isinstance(capabilities, list) or any(
                    not isinstance(value, str)
                    or value not in {"service_process", "managed_queue_worker"}
                    for value in capabilities
                ):
                    checks.append(None)
                else:
                    checks.append(self.required_capability in capabilities)
            if self.reject_known_healthy_target:
                health = payload.get("health")
                if health == "healthy":
                    checks.append(False)
                elif isinstance(health, str) and health in {"degraded", "unhealthy"}:
                    checks.append(True)
                else:
                    checks.append(None)
            if self.requires_healthy_dependencies:
                dependencies = payload.get("dependencies")
                if not isinstance(dependencies, dict) or any(
                    not isinstance(name, str)
                    or not name
                    or not isinstance(status, str)
                    or not status
                    for name, status in dependencies.items()
                ):
                    checks.append(False)
                else:
                    checks.append(
                        all(status.casefold() == "healthy" for status in dependencies.values())
                    )
            if not checks or False in checks:
                return False if False in checks else None
            return None if None in checks else True
        return None


_ROLE_PERMISSIONS: dict[Role, frozenset[str]] = {
    Role.VIEWER: frozenset({"incident:read"}),
    Role.OPERATOR: frozenset(
        {"incident:read", "action:propose", "action:execute:r1", "action:execute:r2"}
    ),
    Role.APPROVER: frozenset({"incident:read", "approval:decide"}),
    Role.ADMIN: frozenset(
        {
            "incident:read",
            "action:propose",
            "action:execute:r1",
            "action:execute:r2",
            "approval:decide",
            "policy:manage",
            "audit:read",
        }
    ),
    Role.AUDITOR: frozenset({"incident:read", "audit:read"}),
}


class ToolGateway:
    """Single execution boundary for all simulator tool calls."""

    def __init__(
        self,
        simulator: Simulator,
        *,
        database: Database | None = None,
        execution_scope: str = "default",
        fault_injector: Callable[[str], None] | None = None,
    ) -> None:
        if not execution_scope:
            raise ValueError("execution_scope must be non-empty")
        self.simulator = simulator
        self.database = database
        self.execution_scope = execution_scope
        self._fault_injector = fault_injector
        self._registry: dict[str, ToolDefinition] = {}
        self._read_transports: dict[str, ReadToolTransport] = {}
        self._idempotency_ledger: dict[str, tuple[str, str, GatewayResult]] = {}
        self._register_defaults()

    def register(self, definition: ToolDefinition) -> None:
        if definition.name in self._registry:
            raise ValueError(f"tool already registered: {definition.name}")
        self._registry[definition.name] = definition

    def definitions(self) -> tuple[ToolDefinition, ...]:
        return tuple(self._registry.values())

    def bind_read_transport(self, tool_name: str, transport: ReadToolTransport) -> None:
        definition = self._registry.get(tool_name)
        if definition is None:
            raise ValueError(f"read transport tool is not registered: {tool_name}")
        if definition.side_effecting or definition.risk_level is not RiskLevel.R0:
            raise ValueError(f"read transport tool must be R0 and read-only: {tool_name}")
        if tool_name in self._read_transports:
            raise ValueError(f"read transport already bound: {tool_name}")
        self._read_transports[tool_name] = transport

    def routed_read_tools(self) -> tuple[str, ...]:
        return tuple(sorted(self._read_transports))

    def close(self) -> None:
        transports = {id(item): item for item in self._read_transports.values()}
        self._read_transports = {}
        for transport in transports.values():
            transport.close()

    def validate_read_request(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        actor: Actor,
    ) -> tuple[ToolDefinition, BaseModel]:
        """Validate a remote read request without executing its handler."""

        try:
            definition = self._registry[tool_name]
        except KeyError as exc:
            raise ToolNotFound(tool_name) from exc
        try:
            request = definition.input_model.model_validate(arguments)
        except ValidationError as exc:
            raise ToolValidationError(str(exc)) from exc
        if definition.side_effecting or definition.risk_level is not RiskLevel.R0:
            raise PermissionDenied("MCP adapter only permits R0 read tools")
        if definition.required_permission not in _ROLE_PERMISSIONS[actor.role]:
            raise PermissionDenied(
                f"{actor.role.value} lacks {definition.required_permission} for {tool_name}"
            )
        return definition, request

    def call(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        actor: Actor,
        *,
        approval: ApprovalRequest | None = None,
        idempotency_key: str | None = None,
    ) -> GatewayResult:
        """Validate, authorize, execute, validate, and ledger one tool call."""

        try:
            definition = self._registry[tool_name]
        except KeyError as exc:
            raise ToolNotFound(tool_name) from exc

        try:
            request = definition.input_model.model_validate(arguments)
        except ValidationError as exc:
            raise ToolValidationError(str(exc)) from exc

        if definition.risk_level is RiskLevel.R3:
            raise ForbiddenAction(f"R3 tool is forbidden: {tool_name}")
        if definition.required_permission not in _ROLE_PERMISSIONS[actor.role]:
            raise PermissionDenied(
                f"{actor.role.value} lacks {definition.required_permission} for {tool_name}"
            )

        if not definition.side_effecting and tool_name in self._read_transports:
            transport = self._read_transports[tool_name]
            try:
                transported = transport.call(
                    tool_name,
                    request.model_dump(mode="json"),
                    actor,
                )
                payload = definition.output_model.model_validate(
                    transported.payload.model_dump(mode="json")
                )
            except ToolError:
                raise
            except ValidationError as exc:
                raise ToolExecutionError(
                    f"{tool_name} transport returned malformed output"
                ) from exc
            except Exception as exc:
                raise ToolExecutionError(f"{tool_name} transport failed") from exc
            return GatewayResult(
                tool_name=tool_name,
                payload=payload,
                replayed=transported.replayed,
                transport=transported.transport,
            )

        execution_id: str | None = None
        fingerprint: str | None = None
        if definition.side_effecting:
            if not idempotency_key:
                raise IdempotencyKeyRequired(f"{tool_name} requires an idempotency key")
            if definition.risk_level in {RiskLevel.R1, RiskLevel.R2}:
                if approval is None or not approval.permits(
                    request.model_dump(), tool_name=tool_name
                ):
                    raise ApprovalRequired(f"valid approval is required for {tool_name}")
            fingerprint = stable_parameter_hash(request.model_dump())
            previous = self._idempotency_ledger.get(idempotency_key)
            if previous is not None:
                previous_tool, previous_fingerprint, previous_result = previous
                if previous_tool != tool_name or previous_fingerprint != fingerprint:
                    raise IdempotencyConflict(
                        f"idempotency key is bound to {previous_tool} and its original parameters"
                    )
                return previous_result.model_copy(update={"replayed": True})
            if self.database is not None:
                execution_key = f"{self.execution_scope}:{idempotency_key}"
                try:
                    with self.database.session() as session:
                        claim = ExecutionLedgerRepository(session).claim(
                            execution_key=execution_key,
                            tool_name=tool_name,
                            parameter_hash=fingerprint,
                        )
                except LedgerConflictError as exc:
                    raise IdempotencyConflict(str(exc)) from exc
                except ExecutionInProgressError as exc:
                    raise ToolExecutionError(str(exc)) from exc
                if claim.replayed:
                    if claim.row.status != ExecutionStatus.SUCCEEDED.value:
                        raise ToolExecutionError(
                            "execution effect is unknown; reconciliation is required"
                        )
                    if not isinstance(claim.row.result_json, dict):
                        raise ToolExecutionError("completed execution has no valid result")
                    try:
                        replay_payload = definition.output_model.model_validate(
                            claim.row.result_json
                        )
                    except ValidationError as exc:
                        raise ToolExecutionError(
                            "completed execution has malformed stored output"
                        ) from exc
                    replay_result = GatewayResult(
                        tool_name=tool_name,
                        payload=replay_payload,
                        replayed=True,
                    )
                    self._idempotency_ledger[idempotency_key] = (
                        tool_name,
                        fingerprint,
                        replay_result,
                    )
                    return replay_result
                execution_id = claim.row.id

        try:
            raw_payload = definition.handler(request, idempotency_key)
            payload = definition.output_model.model_validate(raw_payload)
        except RunLeaseFencedError as exc:
            self._record_execution_failure(execution_id, "LeaseFenced", str(exc))
            raise
        except SimulationError as exc:
            self._record_execution_failure(execution_id, "SimulationError", str(exc))
            raise ToolExecutionError(f"{tool_name} failed: {exc}") from exc
        except ValidationError as exc:
            self._record_execution_unknown(execution_id, "MalformedToolOutput", str(exc))
            raise ToolExecutionError(f"{tool_name} returned malformed output: {exc}") from exc
        except Exception as exc:
            self._record_execution_unknown(execution_id, "UnknownEffect", str(exc))
            raise ToolExecutionError(f"{tool_name} failed unexpectedly: {exc}") from exc

        result = GatewayResult(tool_name=tool_name, payload=payload)
        if definition.side_effecting and self._fault_injector is not None:
            try:
                self._fault_injector("after_backend_receipt")
            except Exception as exc:
                self._record_execution_unknown(
                    execution_id,
                    "ExecutionInterruptedAfterBackendReceipt",
                    str(exc),
                )
                raise ToolExecutionError(
                    f"{tool_name} effect may have completed; execution status is UNKNOWN"
                ) from exc
        if execution_id is not None:
            try:
                with self.database.session() as session:
                    ExecutionLedgerRepository(session).complete(execution_id, payload.model_dump())
            except Exception as exc:
                self._record_execution_unknown(
                    execution_id, "ExecutionReceiptCommitFailed", str(exc)
                )
                raise ToolExecutionError(
                    f"{tool_name} effect may have completed; execution status is UNKNOWN"
                ) from exc
        if definition.side_effecting:
            self._idempotency_ledger[idempotency_key] = (
                tool_name,
                fingerprint or stable_parameter_hash(request.model_dump()),
                result,
            )
        return result

    def recover_simulator_execution(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        actor: Actor,
        *,
        approval: ApprovalRequest | None,
        idempotency_key: str,
        environment_generation: int | None,
        transaction_guard: Callable[[Session], None],
    ) -> None:
        """Recover a lost simulator ledger commit from a database-authenticated receipt.

        Only the runtime holding the current reservation and lease calls this
        path. Caller-supplied receipts remain insufficient for CLAIMED recovery.
        """
        if self.database is None or not isinstance(self.simulator, PersistentSimulator):
            raise ToolExecutionError("authoritative simulator recovery is unavailable")
        definition = self._registry[tool_name]
        request = definition.input_model.model_validate(arguments)
        if definition.risk_level is RiskLevel.R3:
            raise ForbiddenAction(f"R3 tool is forbidden: {tool_name}")
        if definition.required_permission not in _ROLE_PERMISSIONS[actor.role]:
            raise PermissionDenied("recovery actor lacks tool permission")
        if approval is None or not approval.permits(request.model_dump(), tool_name=tool_name):
            raise ApprovalRequired("valid parameter-bound approval is required for recovery")
        fingerprint = stable_parameter_hash(request.model_dump())
        with self.database.session() as session:
            ledger = ExecutionLedgerRepository(session)
            row = ledger.get(f"{self.execution_scope}:{idempotency_key}")
            if row is None or row.status not in {
                ExecutionStatus.CLAIMED.value,
                ExecutionStatus.UNKNOWN.value,
            }:
                return
            if row.tool_name != tool_name or row.parameter_hash != fingerprint:
                raise IdempotencyConflict("recovery execution binding conflicts")
            environment_id = self.simulator.environment_identity.environment_id
            environments = SimulationEnvironmentRepository(session)
            environment = environments.require_for_update(environment_id)
            transaction_guard(session)
            if environment_generation is None or environment.generation != environment_generation:
                raise ToolExecutionError(
                    "recovery environment generation does not match reservation"
                )
            receipt = environments.get_receipt(
                environment_id=environment_id,
                generation=environment_generation,
                idempotency_key=idempotency_key,
            )
            if receipt is None:
                raise ToolExecutionError(
                    "claimed execution has no authoritative backend receipt; "
                    "unsafe replay forbidden"
                )
            if receipt.tool_name != tool_name or receipt.parameter_hash != fingerprint:
                raise IdempotencyConflict("backend recovery receipt binding conflicts")
            payload = definition.output_model.model_validate(receipt.result_json)
            if payload.model_dump().get("idempotency_key") != idempotency_key:
                raise ToolExecutionError("backend recovery receipt key conflicts")
            ledger.recover_confirmed(row.id, payload.model_dump())

    def reconcile_unknown(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        actor: Actor,
        *,
        idempotency_key: str,
        backend_receipt: dict[str, Any],
    ) -> GatewayResult:
        """Finalize UNKNOWN only from an independently obtained backend receipt."""

        try:
            definition = self._registry[tool_name]
        except KeyError as exc:
            raise ToolNotFound(tool_name) from exc
        try:
            request = definition.input_model.model_validate(arguments)
        except ValidationError as exc:
            raise ToolValidationError(str(exc)) from exc
        if definition.risk_level is RiskLevel.R3:
            raise ForbiddenAction(f"R3 tool is forbidden: {tool_name}")
        if definition.required_permission not in _ROLE_PERMISSIONS[actor.role]:
            raise PermissionDenied(
                f"{actor.role.value} lacks {definition.required_permission} for {tool_name}"
            )
        if not definition.side_effecting:
            raise ToolExecutionError("only side-effecting executions can be reconciled")
        if not idempotency_key:
            raise IdempotencyKeyRequired(f"{tool_name} requires an idempotency key")
        if self.database is None:
            raise ToolExecutionError("a durable execution database is required for reconciliation")
        fingerprint = stable_parameter_hash(request.model_dump())
        execution_key = f"{self.execution_scope}:{idempotency_key}"
        try:
            payload = definition.output_model.model_validate(backend_receipt)
        except ValidationError as exc:
            raise ToolExecutionError("a valid backend receipt is required") from exc
        receipt_key = getattr(payload, "idempotency_key", None)
        if receipt_key != idempotency_key:
            raise ToolExecutionError("backend receipt idempotency key does not match")
        try:
            with self.database.session() as session:
                repository = ExecutionLedgerRepository(session)
                row = repository.get(execution_key)
                if row is None:
                    raise ToolExecutionError("execution claim was not found")
                if row.tool_name != tool_name or row.parameter_hash != fingerprint:
                    raise IdempotencyConflict(
                        f"execution key is bound to {row.tool_name} and its original parameters"
                    )
                if row.status != ExecutionStatus.UNKNOWN.value:
                    raise ToolExecutionError("execution is not UNKNOWN and cannot be reconciled")
                repository.reconcile(row.id, payload.model_dump())
        except (ToolExecutionError, IdempotencyConflict):
            raise
        except Exception as exc:
            raise ToolExecutionError("execution reconciliation failed") from exc
        result = GatewayResult(tool_name=tool_name, payload=payload, replayed=True)
        self._idempotency_ledger[idempotency_key] = (
            tool_name,
            fingerprint,
            result,
        )
        return result

    def _record_execution_failure(
        self, execution_id: str | None, error_code: str, error_message: str
    ) -> None:
        if execution_id is None or self.database is None:
            return
        with self.database.session() as session:
            ExecutionLedgerRepository(session).fail(
                execution_id,
                error_code=error_code,
                error_message=error_message,
            )

    def _record_execution_unknown(
        self, execution_id: str | None, error_code: str, error_message: str
    ) -> None:
        if execution_id is None or self.database is None:
            return
        try:
            with self.database.session() as session:
                ExecutionLedgerRepository(session).mark_unknown(
                    execution_id,
                    error_code=error_code,
                    error_message=error_message,
                )
        except Exception:
            # Preserve the original execution failure; reconciliation remains required.
            return

    def _register_defaults(self) -> None:
        read_permission = "incident:read"
        self.register(
            ToolDefinition(
                name="get_service_status",
                description=(
                    "Read current health, version, instances, dependencies, and queue state."
                ),
                input_model=ServiceInput,
                output_model=ServiceStatusOutput,
                risk_level=RiskLevel.R0,
                required_permission=read_permission,
                timeout_seconds=2.0,
                max_retries=1,
                side_effecting=False,
                handler=lambda request, _: self.simulator.get_service_status(request.service),
            )
        )
        self.register(
            ToolDefinition(
                name="verify_service",
                description="Run deterministic post-action health and dependency checks.",
                input_model=ServiceInput,
                output_model=VerificationOutput,
                risk_level=RiskLevel.R0,
                required_permission=read_permission,
                timeout_seconds=2.0,
                max_retries=0,
                side_effecting=False,
                handler=lambda request, _: self.simulator.verify_service(request.service),
            )
        )
        self.register(
            ToolDefinition(
                name="query_metrics",
                description="Read a bounded set of service metrics.",
                input_model=MetricsInput,
                output_model=MetricsOutput,
                risk_level=RiskLevel.R0,
                required_permission=read_permission,
                timeout_seconds=2.0,
                max_retries=1,
                side_effecting=False,
                handler=lambda request, _: self.simulator.query_metrics(
                    request.service, request.metric_names, request.lookback_minutes
                ),
            )
        )
        self.register(
            ToolDefinition(
                name="search_logs",
                description="Search a bounded number of log entries for a service.",
                input_model=LogSearchInput,
                output_model=LogSearchOutput,
                risk_level=RiskLevel.R0,
                required_permission=read_permission,
                timeout_seconds=2.0,
                max_retries=1,
                side_effecting=False,
                handler=lambda request, _: self.simulator.search_logs(
                    request.service, request.query, request.limit
                ),
            )
        )
        self.register(
            ToolDefinition(
                name="get_deployment_history",
                description="Read bounded deployment history for a service.",
                input_model=ServiceInput,
                output_model=DeploymentHistoryOutput,
                risk_level=RiskLevel.R0,
                required_permission=read_permission,
                timeout_seconds=2.0,
                max_retries=1,
                side_effecting=False,
                handler=lambda request, _: self.simulator.get_deployment_history(request.service),
            )
        )
        self.register(
            ToolDefinition(
                name="get_service_config",
                description="Read non-secret service configuration values.",
                input_model=ServiceInput,
                output_model=ServiceConfigOutput,
                risk_level=RiskLevel.R0,
                required_permission=read_permission,
                timeout_seconds=2.0,
                max_retries=1,
                side_effecting=False,
                handler=lambda request, _: self.simulator.get_service_config(request.service),
            )
        )
        self.register(
            ToolDefinition(
                name="restart_worker",
                description=(
                    "Restart one managed queue worker and begin bounded backlog draining. "
                    "Requires managed_queue_worker in the service status capabilities. "
                    "The legacy worker_running field alone does not establish this capability."
                ),
                input_model=ServiceInput,
                output_model=ActionExecutionOutput,
                risk_level=RiskLevel.R1,
                required_permission="action:execute:r1",
                timeout_seconds=5.0,
                max_retries=0,
                side_effecting=True,
                handler=lambda request, key: self.simulator.restart_worker(
                    request.service, key or ""
                ),
                required_capability="managed_queue_worker",
            )
        )
        self.register(
            ToolDefinition(
                name="restart_service",
                description=(
                    "Restart one service process as a limited write. "
                    "Requires service_process in the service status capabilities; "
                    "all observed dependencies must be healthy; a known healthy target is "
                    "excluded by the shared applicability check. This cannot repair an "
                    "unavailable dependency and is distinct from restarting a managed queue "
                    "worker."
                ),
                input_model=ServiceInput,
                output_model=ActionExecutionOutput,
                risk_level=RiskLevel.R1,
                required_permission="action:execute:r1",
                timeout_seconds=5.0,
                max_retries=0,
                side_effecting=True,
                handler=lambda request, key: self.simulator.restart_service(
                    request.service, key or ""
                ),
                required_capability="service_process",
                reject_known_healthy_target=True,
                requires_healthy_dependencies=True,
            )
        )
        self.register(
            ToolDefinition(
                name="rollback_deployment",
                description=(
                    "Rollback a deployment to an existing version; approval is mandatory. "
                    "A known healthy target is excluded by the shared applicability check."
                ),
                input_model=RollbackInput,
                output_model=ActionExecutionOutput,
                risk_level=RiskLevel.R2,
                required_permission="action:execute:r2",
                timeout_seconds=5.0,
                max_retries=0,
                side_effecting=True,
                handler=lambda request, key: self.simulator.rollback_deployment(
                    request.service, request.target_version, key or ""
                ),
                reject_known_healthy_target=True,
            )
        )
