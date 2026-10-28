"""Shared production command transport settings."""

from __future__ import annotations

import os
from dataclasses import dataclass

DEFAULT_COMMAND_TASK_NAME = "agentops.execute_command"
DEFAULT_COMMAND_QUEUE_NAME = "agentops.commands"
DEFAULT_BROKER_URL = "redis://redis:6379/0"


@dataclass(frozen=True)
class CommandTransportSettings:
    broker_url: str
    task_name: str
    queue_name: str
    publish_timeout_seconds: float

    @classmethod
    def from_environment(cls) -> CommandTransportSettings:
        broker_url = os.getenv("AGENTOPS_REDIS_URL", DEFAULT_BROKER_URL).strip()
        task_name = os.getenv("AGENTOPS_CELERY_TASK_NAME", DEFAULT_COMMAND_TASK_NAME).strip()
        queue_name = os.getenv("AGENTOPS_REDIS_QUEUE", DEFAULT_COMMAND_QUEUE_NAME).strip()
        publish_timeout_seconds = float(os.getenv("AGENTOPS_BROKER_PUBLISH_TIMEOUT_SECONDS", "5"))
        if not broker_url:
            raise ValueError("AGENTOPS_REDIS_URL must not be empty")
        if not task_name:
            raise ValueError("AGENTOPS_CELERY_TASK_NAME must not be empty")
        if not queue_name:
            raise ValueError("AGENTOPS_REDIS_QUEUE must not be empty")
        if publish_timeout_seconds <= 0:
            raise ValueError("AGENTOPS_BROKER_PUBLISH_TIMEOUT_SECONDS must be positive")
        return cls(
            broker_url=broker_url,
            task_name=task_name,
            queue_name=queue_name,
            publish_timeout_seconds=publish_timeout_seconds,
        )


__all__ = [
    "CommandTransportSettings",
    "DEFAULT_BROKER_URL",
    "DEFAULT_COMMAND_QUEUE_NAME",
    "DEFAULT_COMMAND_TASK_NAME",
]
