"""Deterministic incident environment used by tests and local demonstrations."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session
from sqlalchemy.orm.exc import StaleDataError

from agentops.domain import stable_parameter_hash
from agentops.persistence import (
    Database,
    EntityNotFound,
    SimulationEnvironmentRepository,
    SimulationReceiptRow,
)


class SimulationError(RuntimeError):
    """Raised when a simulator operation cannot be applied."""


class SimulationConflict(SimulationError):
    """Raised when a receipt or environment version conflicts with the request."""


class StaleEnvironmentError(SimulationConflict):
    """Raised when a fenced worker attempts to mutate an older environment version."""


@dataclass(frozen=True)
class EnvironmentIdentity:
    """Opaque identity and generation for one isolated simulator environment."""

    environment_id: UUID
    generation: int
    version: int

    def __post_init__(self) -> None:
        if self.generation < 1:
            raise ValueError("environment generation must be positive")
        if self.version < 1:
            raise ValueError("environment version must be positive")


class ScenarioName(StrEnum):
    BAD_DEPLOYMENT = "bad-deployment"
    CONNECTION_POOL_EXHAUSTION = "connection-pool-exhaustion"
    EXTERNAL_DEPENDENCY_OUTAGE = "external-dependency-outage"
    QUEUE_BACKLOG = "queue-backlog"
    INVALID_CONFIGURATION = "invalid-configuration"


@dataclass
class _ServiceState:
    health: str
    version: str
    ready_instances: int
    desired_instances: int
    dependencies: dict[str, str]
    metrics: dict[str, float]
    metric_units: dict[str, str]
    logs: list[str]
    deployments: list[dict[str, str]]
    config: dict[str, str]
    queue_depth: int = 0
    incoming_rate: float = 0.0
    processing_rate: float = 0.0
    worker_running: bool = True
    queue_depth_before_last_action: int | None = None
    oldest_message_age_before_last_action: float | None = None


@dataclass
class _ScenarioSeed:
    name: ScenarioName
    title: str
    description: str
    service_name: str
    root_cause: str
    expected_action: str | None
    services: dict[str, _ServiceState]


def _has_managed_queue_worker(state: _ServiceState) -> bool:
    """Use the same structural resource contract for discovery and execution."""
    return {
        "queue_depth",
        "incoming_rate",
        "processing_rate",
        "oldest_message_age_seconds",
    }.issubset(state.metrics)


def _service(
    *,
    health: str,
    version: str,
    dependencies: dict[str, str],
    metrics: dict[str, float],
    metric_units: dict[str, str],
    logs: list[str],
    deployments: list[dict[str, str]],
    config: dict[str, str],
    ready_instances: int = 3,
    desired_instances: int = 3,
    queue_depth: int = 0,
    incoming_rate: float = 0.0,
    processing_rate: float = 0.0,
    worker_running: bool = True,
) -> _ServiceState:
    return _ServiceState(
        health=health,
        version=version,
        ready_instances=ready_instances,
        desired_instances=desired_instances,
        dependencies=dependencies,
        metrics=metrics,
        metric_units=metric_units,
        logs=logs,
        deployments=deployments,
        config=config,
        queue_depth=queue_depth,
        incoming_rate=incoming_rate,
        processing_rate=processing_rate,
        worker_running=worker_running,
    )


def _scenario_seeds() -> dict[ScenarioName, _ScenarioSeed]:
    """Build fresh seed objects so each loaded scenario is independently resettable."""

    return {
        ScenarioName.BAD_DEPLOYMENT: _ScenarioSeed(
            name=ScenarioName.BAD_DEPLOYMENT,
            title="Payment latency regression after deployment",
            description="p95 latency and error rate increased shortly after a payment deployment.",
            service_name="payment-service",
            root_cause="bad deployment v1.8.4",
            expected_action="rollback_deployment",
            services={
                "payment-service": _service(
                    health="degraded",
                    version="v1.8.4",
                    dependencies={"database": "healthy", "fraud-api": "healthy"},
                    metrics={"p95_latency_ms": 920.0, "error_rate": 0.14, "request_rate": 80.0},
                    metric_units={
                        "p95_latency_ms": "ms",
                        "error_rate": "ratio",
                        "request_rate": "requests_per_second",
                    },
                    logs=[
                        "deployment v1.8.4 completed 8 minutes ago",
                        "database timeout while processing payment",
                        "request failed with upstream timeout",
                    ],
                    deployments=[
                        {"version": "v1.8.4", "status": "current"},
                        {"version": "v1.8.3", "status": "stable"},
                    ],
                    config={"deployment_channel": "stable"},
                )
            },
        ),
        ScenarioName.CONNECTION_POOL_EXHAUSTION: _ScenarioSeed(
            name=ScenarioName.CONNECTION_POOL_EXHAUSTION,
            title="Order request timeouts",
            description="Order requests time out while all database connection slots are occupied.",
            service_name="order-service",
            root_cause="connection pool exhaustion caused by leaked connections",
            expected_action="restart_service",
            services={
                "order-service": _service(
                    health="degraded",
                    version="v3.2.1",
                    dependencies={"database": "healthy", "inventory-api": "healthy"},
                    metrics={
                        "pool_active": 100.0,
                        "pool_max": 100.0,
                        "pool_waiters": 42.0,
                        "error_rate": 0.31,
                    },
                    metric_units={
                        "pool_active": "connections",
                        "pool_max": "connections",
                        "pool_waiters": "requests",
                        "error_rate": "ratio",
                    },
                    logs=[
                        "connection pool active=100 max=100",
                        "connection leak suspected; waiting requests=42",
                        "request timed out waiting for database connection",
                    ],
                    deployments=[{"version": "v3.2.1", "status": "stable"}],
                    config={"db_pool_max": "100"},
                )
            },
        ),
        ScenarioName.EXTERNAL_DEPENDENCY_OUTAGE: _ScenarioSeed(
            name=ScenarioName.EXTERNAL_DEPENDENCY_OUTAGE,
            title="Fraud API dependency outage",
            description=(
                "The fraud provider is returning 503 responses while payment remains "
                "mostly healthy."
            ),
            service_name="payment-service",
            root_cause="external fraud-api outage",
            expected_action=None,
            services={
                "payment-service": _service(
                    health="healthy",
                    version="v1.8.3",
                    dependencies={"database": "healthy", "fraud-api": "unavailable"},
                    metrics={"p95_latency_ms": 240.0, "error_rate": 0.03, "fraud_api_5xx": 95.0},
                    metric_units={
                        "p95_latency_ms": "ms",
                        "error_rate": "ratio",
                        "fraud_api_5xx": "responses",
                    },
                    logs=[
                        "fraud-api returned HTTP 503",
                        "retry budget for fraud-api is nearly exhausted",
                    ],
                    deployments=[{"version": "v1.8.3", "status": "stable"}],
                    config={"fraud_api_timeout_ms": "500"},
                )
            },
        ),
        ScenarioName.QUEUE_BACKLOG: _ScenarioSeed(
            name=ScenarioName.QUEUE_BACKLOG,
            title="Notification queue backlog",
            description="Notification queue depth is growing because a consumer worker stopped.",
            service_name="notification-service",
            root_cause="failed notification consumer worker",
            expected_action="restart_worker",
            services={
                "notification-service": _service(
                    health="degraded",
                    version="v2.4.0",
                    dependencies={"message-queue": "healthy"},
                    metrics={
                        "queue_depth": 1000.0,
                        "incoming_rate": 10.0,
                        "processing_rate": 3.0,
                        "oldest_message_age_seconds": 420.0,
                    },
                    metric_units={
                        "queue_depth": "messages",
                        "incoming_rate": "messages_per_second",
                        "processing_rate": "messages_per_second",
                        "oldest_message_age_seconds": "seconds",
                    },
                    logs=[
                        "notification-consumer stopped unexpectedly",
                        "queue depth exceeded warning threshold",
                    ],
                    deployments=[{"version": "v2.4.0", "status": "stable"}],
                    config={"consumer_concurrency": "4"},
                    queue_depth=1000,
                    incoming_rate=10.0,
                    processing_rate=3.0,
                    worker_running=False,
                )
            },
        ),
        ScenarioName.INVALID_CONFIGURATION: _ScenarioSeed(
            name=ScenarioName.INVALID_CONFIGURATION,
            title="Inventory dependency resolution failure",
            description=(
                "Order service cannot resolve the inventory dependency after a "
                "configuration change."
            ),
            service_name="order-service",
            root_cause="stale inventory-api hostname in configuration",
            expected_action=None,
            services={
                "order-service": _service(
                    health="degraded",
                    version="v3.2.1",
                    dependencies={"database": "healthy", "inventory-api": "unreachable"},
                    metrics={"dns_errors": 80.0, "error_rate": 0.22, "p95_latency_ms": 700.0},
                    metric_units={
                        "dns_errors": "errors",
                        "error_rate": "ratio",
                        "p95_latency_ms": "ms",
                    },
                    logs=[
                        "inventory-api hostname stale-inventory.internal cannot be resolved",
                        "configuration validation failed for inventory endpoint",
                    ],
                    deployments=[{"version": "v3.2.1", "status": "stable"}],
                    config={"inventory_api_host": "stale-inventory.internal"},
                )
            },
        ),
    }


def _service_from_snapshot(raw: Any) -> _ServiceState:
    if not isinstance(raw, Mapping):
        raise SimulationError("persisted simulator service is malformed")
    try:
        deployments = raw["deployments"]
        if not isinstance(deployments, list):
            raise TypeError("deployments must be a list")
        return _ServiceState(
            health=str(raw["health"]),
            version=str(raw["version"]),
            ready_instances=int(raw["ready_instances"]),
            desired_instances=int(raw["desired_instances"]),
            dependencies={str(key): str(value) for key, value in raw["dependencies"].items()},
            metrics={str(key): float(value) for key, value in raw["metrics"].items()},
            metric_units={str(key): str(value) for key, value in raw["metric_units"].items()},
            logs=[str(entry) for entry in raw["logs"]],
            deployments=[
                {"version": str(item["version"]), "status": str(item["status"])}
                for item in deployments
            ],
            config={str(key): str(value) for key, value in raw["config"].items()},
            queue_depth=int(raw.get("queue_depth", 0)),
            incoming_rate=float(raw.get("incoming_rate", 0.0)),
            processing_rate=float(raw.get("processing_rate", 0.0)),
            worker_running=bool(raw.get("worker_running", True)),
            queue_depth_before_last_action=(
                int(raw["queue_depth_before_last_action"])
                if raw.get("queue_depth_before_last_action") is not None
                else None
            ),
            oldest_message_age_before_last_action=(
                float(raw["oldest_message_age_before_last_action"])
                if raw.get("oldest_message_age_before_last_action") is not None
                else None
            ),
        )
    except (KeyError, TypeError, ValueError, AttributeError) as exc:
        raise SimulationError("persisted simulator service is malformed") from exc


class Simulator:
    """Resettable service simulator with evaluator-only ground truth."""

    def __init__(self) -> None:
        self._seed: _ScenarioSeed | None = None
        self._services: dict[str, _ServiceState] = {}
        self._applied_actions: dict[str, dict[str, Any]] = {}
        self._environment_identity: EnvironmentIdentity | None = None

    @property
    def loaded_scenario(self) -> ScenarioName | None:
        return self._seed.name if self._seed else None

    @property
    def environment_identity(self) -> EnvironmentIdentity:
        if self._environment_identity is None:
            raise SimulationError("no environment is loaded")
        return self._environment_identity

    @staticmethod
    def available_scenarios() -> tuple[ScenarioName, ...]:
        return tuple(ScenarioName)

    def load(self, scenario: ScenarioName | str) -> None:
        """Load a scenario and reset its mutable state."""

        try:
            scenario_name = (
                scenario if isinstance(scenario, ScenarioName) else ScenarioName(scenario)
            )
        except ValueError as exc:
            raise SimulationError(f"unknown scenario: {scenario}") from exc
        seed = _scenario_seeds()[scenario_name]
        self._seed = deepcopy(seed)
        self._services = deepcopy(seed.services)
        self._applied_actions = {}
        self._environment_identity = EnvironmentIdentity(uuid4(), generation=1, version=1)

    def reset(self) -> None:
        """Reset the currently loaded scenario to its original state."""

        if self._seed is None:
            raise SimulationError("no scenario is loaded")
        identity = self.environment_identity
        self._services = deepcopy(self._seed.services)
        self._applied_actions = {}
        self._environment_identity = EnvironmentIdentity(
            uuid4(), generation=identity.generation + 1, version=1
        )

    def snapshot(self) -> dict[str, Any]:
        """Serialize mutable state without evaluator-only seed fields."""

        seed = self._require_seed()
        return {
            "scenario": seed.name.value,
            "services": {name: asdict(state) for name, state in self._services.items()},
            "applied_actions": deepcopy(self._applied_actions),
        }

    @classmethod
    def from_snapshot(
        cls,
        *,
        scenario: ScenarioName | str,
        identity: EnvironmentIdentity,
        snapshot: Mapping[str, Any],
    ) -> Simulator:
        """Reconstruct a simulator from a validated persistence snapshot."""

        try:
            scenario_name = (
                scenario if isinstance(scenario, ScenarioName) else ScenarioName(scenario)
            )
        except ValueError as exc:
            raise SimulationError(f"unknown scenario: {scenario}") from exc
        seed = _scenario_seeds()[scenario_name]
        if snapshot.get("scenario") != scenario_name.value:
            raise SimulationError("persisted simulator scenario does not match its environment")
        raw_services = snapshot.get("services")
        if not isinstance(raw_services, Mapping) or set(raw_services) != set(seed.services):
            raise SimulationError("persisted simulator services are incomplete or unexpected")
        raw_actions = snapshot.get("applied_actions", {})
        if not isinstance(raw_actions, Mapping):
            raise SimulationError("persisted simulator receipts are malformed")

        simulator = cls()
        simulator._seed = deepcopy(seed)
        simulator._services = {
            name: _service_from_snapshot(raw_state) for name, raw_state in raw_services.items()
        }
        simulator._applied_actions = deepcopy(dict(raw_actions))
        simulator._environment_identity = identity
        return simulator

    def incident_context(self) -> dict[str, str]:
        """Return incident fields visible to the agent; no evaluator ground truth."""

        seed = self._require_seed()
        return {
            "title": seed.title,
            "description": seed.description,
            "service_name": seed.service_name,
        }

    def inspect_ground_truth(self) -> dict[str, str | None]:
        """Return evaluator-only data; this method is never registered as a tool."""

        seed = self._require_seed()
        return {"root_cause": seed.root_cause, "expected_action": seed.expected_action}

    def get_service_status(self, service: str) -> dict[str, Any]:
        state = self._service(service)
        capabilities = ["service_process"]
        if _has_managed_queue_worker(state):
            capabilities.append("managed_queue_worker")
        return {
            "service": service,
            "health": state.health,
            "version": state.version,
            "instances": {"ready": state.ready_instances, "desired": state.desired_instances},
            "dependencies": dict(state.dependencies),
            "worker_running": state.worker_running,
            "queue_depth": state.queue_depth,
            "capabilities": capabilities,
        }

    def query_metrics(
        self,
        service: str,
        metric_names: list[str],
        lookback_minutes: int,
    ) -> dict[str, Any]:
        state = self._service(service)
        metrics = {name: state.metrics[name] for name in metric_names if name in state.metrics}
        units = {name: state.metric_units[name] for name in metrics}
        return {
            "service": service,
            "metrics": metrics,
            "units": units,
            "lookback_minutes": lookback_minutes,
        }

    def search_logs(self, service: str, query: str, limit: int) -> dict[str, Any]:
        state = self._service(service)
        normalized_query = query.casefold()
        initial_sample = query.isspace()
        matches = (
            list(state.logs)
            if initial_sample
            else [entry for entry in state.logs if normalized_query in entry.casefold()]
        )
        return {
            "service": service,
            "query": query,
            "entries": matches[-limit:] if initial_sample else matches[:limit],
            "total_matches": len(matches),
            "truncated": len(matches) > limit,
        }

    def get_deployment_history(self, service: str) -> dict[str, Any]:
        state = self._service(service)
        return {"service": service, "deployments": deepcopy(state.deployments)}

    def get_service_config(self, service: str) -> dict[str, Any]:
        state = self._service(service)
        return {"service": service, "values": dict(state.config)}

    def restart_service(self, service: str, idempotency_key: str) -> dict[str, Any]:
        state = self._service(service)
        if idempotency_key in self._applied_actions:
            return deepcopy(self._applied_actions[idempotency_key])
        dependencies_healthy = all(status == "healthy" for status in state.dependencies.values())
        state.health = "healthy" if dependencies_healthy else "degraded"
        state.ready_instances = state.desired_instances
        if "pool_waiters" in state.metrics:
            state.metrics["pool_waiters"] = 0.0
            state.metrics["error_rate"] = 0.02
            state.logs.append("service restarted; connection pool recovered")
        result = {
            "service": service,
            "action": "restart_service",
            "parameter_hash": stable_parameter_hash({"service": service}),
            "committed_at": datetime.now(UTC).isoformat(),
            "changed": True,
            "message": "service restarted successfully",
            "idempotency_key": idempotency_key,
        }
        self._applied_actions[idempotency_key] = deepcopy(result)
        self._advance_version()
        return result

    def restart_worker(self, service: str, idempotency_key: str) -> dict[str, Any]:
        state = self._service(service)
        if idempotency_key in self._applied_actions:
            return deepcopy(self._applied_actions[idempotency_key])
        if not _has_managed_queue_worker(state):
            raise SimulationError(f"service has no managed queue worker: {service}")

        state.queue_depth_before_last_action = state.queue_depth
        state.oldest_message_age_before_last_action = float(
            state.metrics["oldest_message_age_seconds"]
        )
        state.worker_running = True
        state.processing_rate = max(state.incoming_rate * 1.2, state.incoming_rate + 1.0)
        drain_window_seconds = 30.0
        drained_messages = max(
            1,
            int((state.processing_rate - state.incoming_rate) * drain_window_seconds),
        )
        state.queue_depth = max(0, state.queue_depth - drained_messages)
        state.metrics["queue_depth"] = float(state.queue_depth)
        state.metrics["processing_rate"] = state.processing_rate
        state.metrics["oldest_message_age_seconds"] = max(
            0.0,
            state.oldest_message_age_before_last_action - drain_window_seconds,
        )
        dependencies_healthy = all(status == "healthy" for status in state.dependencies.values())
        state.health = "healthy" if dependencies_healthy else "degraded"
        state.logs.append("notification consumer restarted and queue backlog is draining")
        result = {
            "service": service,
            "action": "restart_worker",
            "parameter_hash": stable_parameter_hash({"service": service}),
            "committed_at": datetime.now(UTC).isoformat(),
            "changed": True,
            "message": "queue worker restarted successfully",
            "idempotency_key": idempotency_key,
        }
        self._applied_actions[idempotency_key] = deepcopy(result)
        self._advance_version()
        return result

    def rollback_deployment(
        self,
        service: str,
        target_version: str,
        idempotency_key: str,
    ) -> dict[str, Any]:
        state = self._service(service)
        if idempotency_key in self._applied_actions:
            return deepcopy(self._applied_actions[idempotency_key])
        known_versions = {deployment["version"] for deployment in state.deployments}
        if target_version not in known_versions:
            raise SimulationError(f"unknown deployment version for {service}: {target_version}")
        if state.version == target_version:
            result = {
                "service": service,
                "action": "rollback_deployment",
                "parameter_hash": stable_parameter_hash(
                    {"service": service, "target_version": target_version}
                ),
                "committed_at": datetime.now(UTC).isoformat(),
                "changed": False,
                "message": f"deployment already at {target_version}; no remediation applied",
                "idempotency_key": idempotency_key,
            }
            self._applied_actions[idempotency_key] = deepcopy(result)
            return result
        state.version = target_version
        dependencies_healthy = all(status == "healthy" for status in state.dependencies.values())
        state.health = "healthy" if dependencies_healthy else "degraded"
        state.ready_instances = state.desired_instances
        if "p95_latency_ms" in state.metrics:
            state.metrics["p95_latency_ms"] = 180.0
        if "error_rate" in state.metrics:
            state.metrics["error_rate"] = 0.01
        state.logs.append(f"deployment rolled back to {target_version}")
        state.deployments = [
            {
                "version": deployment["version"],
                "status": (
                    "current" if deployment["version"] == target_version else deployment["status"]
                ),
            }
            for deployment in state.deployments
        ]
        self._advance_version()
        result = {
            "service": service,
            "action": "rollback_deployment",
            "parameter_hash": stable_parameter_hash(
                {"service": service, "target_version": target_version}
            ),
            "committed_at": datetime.now(UTC).isoformat(),
            "changed": True,
            "message": f"deployment rolled back to {target_version}",
            "idempotency_key": idempotency_key,
        }
        self._applied_actions[idempotency_key] = deepcopy(result)
        return result

    def get_mutation_receipt(self, idempotency_key: str) -> dict[str, Any] | None:
        """Return one committed in-memory effect without exposing scenario truth."""

        if not idempotency_key:
            raise SimulationError("idempotency_key is required")
        receipt = self._applied_actions.get(idempotency_key)
        return deepcopy(receipt) if receipt is not None else None

    def mutation_history(self) -> tuple[dict[str, Any], ...]:
        """Observe all committed keys in the loaded environment, including no-ops.

        This evaluator boundary is not registered as an agent tool. A fresh
        load/reset establishes the scope; replay of a key adds no new receipt.
        """
        self._require_seed()
        return tuple(deepcopy(receipt) for receipt in self._applied_actions.values())

    def verify_service(self, service: str) -> dict[str, Any]:
        state = self._service(service)
        checks: dict[str, bool] = {
            "healthy_status": state.health == "healthy",
            "instances_ready": state.ready_instances >= state.desired_instances,
            "dependencies_healthy": all(
                status == "healthy" for status in state.dependencies.values()
            ),
        }
        if "p95_latency_ms" in state.metrics:
            checks["p95_below_threshold"] = state.metrics["p95_latency_ms"] < 300
        if "error_rate" in state.metrics:
            checks["error_rate_below_threshold"] = state.metrics["error_rate"] < 0.05
        if "pool_waiters" in state.metrics:
            checks["pool_waiters_zero"] = state.metrics["pool_waiters"] == 0.0
        if "queue_depth" in state.metrics:
            checks["worker_running"] = state.worker_running
            checks["processing_rate_above_incoming"] = state.processing_rate > state.incoming_rate
            if state.queue_depth_before_last_action is not None:
                checks["queue_depth_declining"] = (
                    state.queue_depth < state.queue_depth_before_last_action
                )
            if state.oldest_message_age_before_last_action is not None:
                checks["oldest_message_age_declining"] = (
                    float(state.metrics.get("oldest_message_age_seconds", 0.0))
                    < state.oldest_message_age_before_last_action
                )
        return {
            "service": service,
            "healthy": all(checks.values()),
            "checks": checks,
            "observed": {
                "version": state.version,
                "dependencies": dict(state.dependencies),
                "metrics": dict(state.metrics),
                "queue_depth": state.queue_depth,
                "incoming_rate": state.incoming_rate,
                "processing_rate": state.processing_rate,
                "worker_running": state.worker_running,
                "queue_depth_before_last_action": state.queue_depth_before_last_action,
                "oldest_message_age_before_last_action": (
                    state.oldest_message_age_before_last_action
                ),
            },
        }

    def _advance_version(self) -> None:
        identity = self.environment_identity
        self._environment_identity = EnvironmentIdentity(
            identity.environment_id, generation=identity.generation, version=identity.version + 1
        )

    def _require_seed(self) -> _ScenarioSeed:
        if self._seed is None:
            raise SimulationError("no scenario is loaded")
        return self._seed

    def _service(self, service: str) -> _ServiceState:
        self._require_seed()
        try:
            return self._services[service]
        except KeyError as exc:
            raise SimulationError(f"unknown service: {service}") from exc


class PersistentSimulator:
    """SQLite-backed simulator adapter reconstructed by opaque environment id."""

    def __init__(
        self,
        database: Database,
        environment_id: UUID | str,
        *,
        expected_identity: EnvironmentIdentity | None = None,
        mutation_guard: Callable[[], None] | None = None,
        transaction_guard: Callable[[Session], None] | None = None,
    ) -> None:
        self.database = database
        try:
            self._environment_id = UUID(str(environment_id))
        except ValueError as exc:
            raise SimulationError("environment_id must be a UUID") from exc
        if (
            expected_identity is not None
            and expected_identity.environment_id != self._environment_id
        ):
            raise SimulationError("expected identity belongs to another environment")
        self._expected_identity = expected_identity
        self._mutation_guard = mutation_guard
        self._transaction_guard = transaction_guard

    @classmethod
    def create(cls, database: Database, scenario: ScenarioName | str) -> PersistentSimulator:
        simulator = Simulator()
        simulator.load(scenario)
        identity = simulator.environment_identity
        with database.session() as session:
            SimulationEnvironmentRepository(session).create(
                environment_id=identity.environment_id,
                scenario=simulator.loaded_scenario.value,
                generation=identity.generation,
                version=identity.version,
                state=simulator.snapshot(),
            )
        return cls(database, identity.environment_id)

    def fenced(self) -> PersistentSimulator:
        """Return a worker adapter fenced to the current environment version."""
        return type(self)(
            self.database,
            self._environment_id,
            expected_identity=self.environment_identity,
            mutation_guard=self._mutation_guard,
            transaction_guard=self._transaction_guard,
        )

    def with_mutation_guard(self, mutation_guard: Callable[[], None]) -> PersistentSimulator:
        """Return an adapter that composes an additional mutation-time guard."""

        if not callable(mutation_guard):
            raise TypeError("mutation_guard must be callable")
        previous_guard = self._mutation_guard

        def combined_guard() -> None:
            if previous_guard is not None:
                previous_guard()
            mutation_guard()

        return type(self)(
            self.database,
            self._environment_id,
            expected_identity=self._expected_identity,
            mutation_guard=combined_guard,
            transaction_guard=self._transaction_guard,
        )

    def with_transaction_guard(
        self, transaction_guard: Callable[[Session], None]
    ) -> PersistentSimulator:
        """Compose a guard that executes inside the simulator mutation transaction."""

        if not callable(transaction_guard):
            raise TypeError("transaction_guard must be callable")
        previous_guard = self._transaction_guard

        def combined_guard(session: Session) -> None:
            if previous_guard is not None:
                previous_guard(session)
            transaction_guard(session)

        return type(self)(
            self.database,
            self._environment_id,
            expected_identity=self._expected_identity,
            mutation_guard=self._mutation_guard,
            transaction_guard=combined_guard,
        )

    @property
    def environment_identity(self) -> EnvironmentIdentity:
        with self.database.session() as session:
            row = SimulationEnvironmentRepository(session).require(self._environment_id)
            return EnvironmentIdentity(UUID(row.id), row.generation, row.version)

    @property
    def loaded_scenario(self) -> ScenarioName:
        with self.database.session() as session:
            row = SimulationEnvironmentRepository(session).require(self._environment_id)
            try:
                return ScenarioName(row.scenario)
            except ValueError as exc:
                raise SimulationError("persisted simulator scenario is invalid") from exc

    def incident_context(self) -> dict[str, str]:
        return self._read(lambda simulator: simulator.incident_context())

    def inspect_ground_truth(self) -> dict[str, str | None]:
        return self._read(lambda simulator: simulator.inspect_ground_truth())

    def get_service_status(self, service: str) -> dict[str, Any]:
        return self._read(lambda simulator: simulator.get_service_status(service))

    def query_metrics(
        self, service: str, metric_names: list[str], lookback_minutes: int
    ) -> dict[str, Any]:
        return self._read(
            lambda simulator: simulator.query_metrics(service, metric_names, lookback_minutes)
        )

    def search_logs(self, service: str, query: str, limit: int) -> dict[str, Any]:
        return self._read(lambda simulator: simulator.search_logs(service, query, limit))

    def get_deployment_history(self, service: str) -> dict[str, Any]:
        return self._read(lambda simulator: simulator.get_deployment_history(service))

    def get_service_config(self, service: str) -> dict[str, Any]:
        return self._read(lambda simulator: simulator.get_service_config(service))

    def verify_service(self, service: str) -> dict[str, Any]:
        return self._read(lambda simulator: simulator.verify_service(service))

    def restart_service(self, service: str, idempotency_key: str) -> dict[str, Any]:
        return self._mutate(
            "restart_service",
            {"service": service},
            idempotency_key,
            lambda simulator: simulator.restart_service(service, idempotency_key),
        )

    def restart_worker(self, service: str, idempotency_key: str) -> dict[str, Any]:
        return self._mutate(
            "restart_worker",
            {"service": service},
            idempotency_key,
            lambda simulator: simulator.restart_worker(service, idempotency_key),
        )

    def rollback_deployment(
        self, service: str, target_version: str, idempotency_key: str
    ) -> dict[str, Any]:
        return self._mutate(
            "rollback_deployment",
            {"service": service, "target_version": target_version},
            idempotency_key,
            lambda simulator: simulator.rollback_deployment(
                service, target_version, idempotency_key
            ),
        )

    def get_mutation_receipt(self, idempotency_key: str) -> dict[str, Any] | None:
        """Read a durable receipt for this environment without applying a mutation."""

        if not idempotency_key:
            raise SimulationError("idempotency_key is required")
        with self.database.session() as session:
            repository = SimulationEnvironmentRepository(session)
            environment = repository.require(self._environment_id)
            receipt = repository.get_receipt(
                environment_id=self._environment_id,
                generation=environment.generation,
                idempotency_key=idempotency_key,
            )
            return deepcopy(receipt.result_json) if receipt is not None else None

    def mutation_history(self) -> tuple[dict[str, Any], ...]:
        """Observe complete backend state, not just gateway-selected receipts.

        Nested backend mutations may be in state without separate durable
        receipt rows. Keep them visible and validate the durable rows against
        that independently reconstructed history in the same transaction.
        """
        with self.database.session() as session:
            environment = SimulationEnvironmentRepository(session).require(self._environment_id)
            raw = environment.state_json.get("applied_actions")
            if not isinstance(raw, dict):
                raise SimulationError("backend mutation history is malformed")
            history = deepcopy(raw)
            receipts = (
                session.query(SimulationReceiptRow)
                .filter_by(
                    environment_id=str(self._environment_id), generation=environment.generation
                )
                .all()
            )
            for receipt in receipts:
                value = history.get(receipt.idempotency_key)
                if (
                    not isinstance(value, dict)
                    or value != receipt.result_json
                    or value.get("action") != receipt.tool_name
                    or value.get("parameter_hash") != receipt.parameter_hash
                ):
                    raise SimulationError("backend history and durable receipt disagree")
            return tuple(history.values())

    def reset(self) -> PersistentSimulator:
        simulator = self._load_simulator()
        self._assert_fence(simulator.environment_identity)
        simulator.reset()
        identity = simulator.environment_identity
        with self.database.session() as session:
            SimulationEnvironmentRepository(session).create(
                environment_id=identity.environment_id,
                scenario=simulator.loaded_scenario.value,
                generation=identity.generation,
                version=identity.version,
                state=simulator.snapshot(),
            )
        return PersistentSimulator(
            self.database,
            identity.environment_id,
            mutation_guard=self._mutation_guard,
            transaction_guard=self._transaction_guard,
        )

    def _load_simulator(self) -> Simulator:
        with self.database.session() as session:
            row = SimulationEnvironmentRepository(session).require(self._environment_id)
            identity = EnvironmentIdentity(UUID(row.id), row.generation, row.version)
            return Simulator.from_snapshot(
                scenario=row.scenario, identity=identity, snapshot=deepcopy(row.state_json)
            )

    def _read(self, reader: Callable[[Simulator], Any]) -> Any:
        return reader(self._load_simulator())

    def _assert_fence(self, current_identity: EnvironmentIdentity) -> None:
        expected = self._expected_identity
        if expected is None:
            return
        if (
            current_identity.environment_id != expected.environment_id
            or current_identity.generation != expected.generation
            or current_identity.version != expected.version
        ):
            raise StaleEnvironmentError(
                "environment fence is stale: expected "
                f"{expected.generation}/{expected.version}, got "
                f"{current_identity.generation}/{current_identity.version}"
            )

    def _mutate(
        self,
        tool_name: str,
        parameters: Mapping[str, Any],
        idempotency_key: str,
        operation: Callable[[Simulator], dict[str, Any]],
    ) -> dict[str, Any]:
        fingerprint = stable_parameter_hash(parameters)
        committed_identity: EnvironmentIdentity | None = None
        committed_result: dict[str, Any] | None = None
        try:
            with self.database.session() as session:
                repository = SimulationEnvironmentRepository(session)
                row = repository.require_for_update(self._environment_id)
                receipt = repository.get_receipt(
                    environment_id=self._environment_id,
                    generation=row.generation,
                    idempotency_key=idempotency_key,
                )
                # A durable receipt remains replayable even if a later action advanced the fence.
                if receipt is not None:
                    if receipt.tool_name != tool_name or receipt.parameter_hash != fingerprint:
                        raise SimulationConflict(
                            f"idempotency key is bound to {receipt.tool_name} "
                            "and its original parameters"
                        )
                    committed_result = deepcopy(receipt.result_json)
                else:
                    identity = EnvironmentIdentity(UUID(row.id), row.generation, row.version)
                    if self._mutation_guard is not None:
                        self._mutation_guard()
                    if self._transaction_guard is not None:
                        self._transaction_guard(session)
                    self._assert_fence(identity)
                    simulator = Simulator.from_snapshot(
                        scenario=row.scenario, identity=identity, snapshot=deepcopy(row.state_json)
                    )
                    result = operation(simulator)
                    row.state_json = simulator.snapshot()
                    row.version = simulator.environment_identity.version
                    row.updated_at = datetime.now(UTC)
                    key = result.get("idempotency_key")
                    if not isinstance(key, str) or not key:
                        raise SimulationError("mutation result did not contain an idempotency key")
                    repository.save_receipt(
                        environment_id=self._environment_id,
                        generation=row.generation,
                        idempotency_key=key,
                        tool_name=tool_name,
                        parameter_hash=fingerprint,
                        result=result,
                        environment_version=row.version,
                    )
                    committed_identity = simulator.environment_identity
                    committed_result = deepcopy(result)
        except EntityNotFound:
            raise
        except SimulationError:
            raise
        except (StaleDataError, IntegrityError) as exc:
            raise SimulationConflict("simulator environment changed during mutation") from exc
        if committed_identity is not None:
            self._expected_identity = committed_identity
        if committed_result is None:
            raise SimulationError("mutation completed without a result")
        return committed_result
