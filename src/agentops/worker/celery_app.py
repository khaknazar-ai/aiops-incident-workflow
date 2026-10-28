"""Optional Celery application factory for the broker adapter."""

from __future__ import annotations

import os
from typing import Any

from agentops.worker.transport import (
    DEFAULT_COMMAND_QUEUE_NAME,
    DEFAULT_COMMAND_TASK_NAME,
)


def create_celery_app(
    *,
    broker_url: str | None = None,
    result_backend: str | None = None,
    task_name: str = DEFAULT_COMMAND_TASK_NAME,
    queue_name: str = DEFAULT_COMMAND_QUEUE_NAME,
    publish_timeout_seconds: float = 5.0,
) -> Any:
    """Create a Celery app lazily; importing the core package stays offline-safe."""

    if not task_name.strip():
        raise ValueError("task_name must not be empty")
    if not queue_name.strip():
        raise ValueError("queue_name must not be empty")
    if publish_timeout_seconds <= 0:
        raise ValueError("publish_timeout_seconds must be positive")

    try:
        from celery import Celery
    except ImportError as exc:  # pragma: no cover - optional dependency path
        raise RuntimeError("celery package is required for the worker") from exc
    app = Celery(
        "agentops",
        broker=broker_url or os.getenv("AGENTOPS_REDIS_URL", "redis://localhost:6379/0"),
        backend=result_backend,
    )
    app.conf.update(
        task_acks_late=True,
        task_reject_on_worker_lost=True,
        task_track_started=True,
        task_default_queue=queue_name,
        task_routes={task_name: {"queue": queue_name}},
        task_serializer="json",
        accept_content=["json"],
        result_serializer="json",
        task_publish_retry=False,
        broker_connection_timeout=publish_timeout_seconds,
        broker_transport_options={
            "socket_connect_timeout": publish_timeout_seconds,
            "socket_timeout": publish_timeout_seconds,
        },
    )
    return app
