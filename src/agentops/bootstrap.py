"""Explicit operator bootstrap operations for disposable demo environments."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any
from uuid import UUID

from agentops.database_config import database_url_from_environment
from agentops.persistence import Database
from agentops.simulator import PersistentSimulator, ScenarioName


@dataclass(frozen=True)
class SeededEnvironment:
    """Public metadata for one newly persisted simulator environment."""

    environment_id: UUID
    scenario: str
    generation: int
    version: int

    def as_dict(self) -> dict[str, str | int]:
        payload = asdict(self)
        payload["environment_id"] = str(self.environment_id)
        return payload


@dataclass(frozen=True)
class EnvironmentInspection:
    """Public, agent-visible state for one persisted simulator environment."""

    environment_id: UUID
    scenario: str
    generation: int
    version: int
    incident: dict[str, str]
    service: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["environment_id"] = str(self.environment_id)
        return payload


def seed_environment(
    scenario: ScenarioName | str,
    *,
    database_url: str | None = None,
) -> SeededEnvironment:
    """Persist one isolated scenario after migrations have created the schema."""

    scenario_name = scenario if isinstance(scenario, ScenarioName) else ScenarioName(scenario)
    database = Database(database_url or database_url_from_environment())
    try:
        simulator = PersistentSimulator.create(database, scenario_name)
        identity = simulator.environment_identity
        return SeededEnvironment(
            environment_id=identity.environment_id,
            scenario=scenario_name.value,
            generation=identity.generation,
            version=identity.version,
        )
    finally:
        database.engine.dispose()


def inspect_environment(
    environment_id: UUID | str,
    *,
    database_url: str | None = None,
) -> EnvironmentInspection:
    """Read public simulator state without exposing evaluator-only ground truth."""

    database = Database(database_url or database_url_from_environment())
    try:
        simulator = PersistentSimulator(database, environment_id)
        identity = simulator.environment_identity
        incident = simulator.incident_context()
        service = simulator.get_service_status(incident["service_name"])
        return EnvironmentInspection(
            environment_id=identity.environment_id,
            scenario=simulator.loaded_scenario.value,
            generation=identity.generation,
            version=identity.version,
            incident=incident,
            service=service,
        )
    finally:
        database.engine.dispose()


__all__ = [
    "EnvironmentInspection",
    "SeededEnvironment",
    "inspect_environment",
    "seed_environment",
]
