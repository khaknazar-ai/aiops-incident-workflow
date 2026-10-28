"""Celery worker entrypoint for the durable command boundary."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from agentops.database_config import database_url_from_environment
from agentops.persistence import Database
from agentops.worker.celery_app import create_celery_app
from agentops.worker.runtime import DurableRuntimeFactory
from agentops.worker.tasks import CommandWorker, RuntimeCommandExecutor, register_celery_task
from agentops.worker.transport import CommandTransportSettings


def build_celery_app() -> Any:
    """Build the broker-facing app without importing Celery in the core package."""

    database = Database(database_url_from_environment())
    checkpoint_path = os.getenv("AGENTOPS_CHECKPOINT_PATH")
    transport = CommandTransportSettings.from_environment()

    def worker_factory() -> CommandWorker:
        runtime_factory = DurableRuntimeFactory.from_environment(
            database,
            checkpoint_path=Path(checkpoint_path) if checkpoint_path else None,
            owner_prefix=os.getenv("AGENTOPS_WORKER_OWNER_PREFIX", "celery-worker"),
        )
        return CommandWorker(
            RuntimeCommandExecutor(runtime_factory),
            database=database,
            owner_id=os.getenv("AGENTOPS_WORKER_OWNER_ID"),
            lease_seconds=runtime_factory.command_lease_seconds,
        )

    app = create_celery_app(
        broker_url=transport.broker_url,
        task_name=transport.task_name,
        queue_name=transport.queue_name,
        publish_timeout_seconds=transport.publish_timeout_seconds,
    )
    register_celery_task(
        app,
        worker_factory,
        task_name=transport.task_name,
        queue_name=transport.queue_name,
    )
    return app


celery_app = build_celery_app()
