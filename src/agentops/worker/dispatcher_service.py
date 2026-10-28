"""Long-running Redis outbox dispatcher entrypoint."""

from __future__ import annotations

import math
import os
import signal
from collections.abc import Callable
from threading import Event

from agentops.database_config import database_url_from_environment
from agentops.persistence import Database
from agentops.worker.celery_app import create_celery_app
from agentops.worker.dispatch import CeleryPublisher, DispatchReport, OutboxDispatcher
from agentops.worker.transport import CommandTransportSettings


def build_dispatcher() -> OutboxDispatcher:
    database = Database(database_url_from_environment())
    transport = CommandTransportSettings.from_environment()
    celery_app = create_celery_app(
        broker_url=transport.broker_url,
        task_name=transport.task_name,
        queue_name=transport.queue_name,
        publish_timeout_seconds=transport.publish_timeout_seconds,
    )
    return OutboxDispatcher(
        database,
        CeleryPublisher(
            celery_app,
            task_name=transport.task_name,
            queue_name=transport.queue_name,
        ),
        owner_id=os.getenv("AGENTOPS_DISPATCHER_OWNER_ID"),
        lease_seconds=float(os.getenv("AGENTOPS_DISPATCHER_LEASE_SECONDS", "30")),
        max_attempts=int(os.getenv("AGENTOPS_DISPATCHER_MAX_ATTEMPTS", "5")),
        recovery_delay_seconds=float(os.getenv("AGENTOPS_DISPATCHER_RECOVERY_DELAY_SECONDS", "60")),
    )


def _dispatch_batch(operation: Callable[..., DispatchReport], shutdown: Event) -> None:
    # Claim only the next publication; shutdown must not strand an entire batch
    # or consume retry attempts for commands that were never sent to the broker.
    for _ in range(100):
        if shutdown.is_set():
            break
        if operation(limit=1).claimed == 0:
            break


def main() -> None:
    interval = float(os.getenv("AGENTOPS_DISPATCHER_INTERVAL_SECONDS", "1"))
    if not math.isfinite(interval) or interval <= 0:
        raise ValueError("AGENTOPS_DISPATCHER_INTERVAL_SECONDS must be finite and positive")
    shutdown = Event()

    def request_shutdown(signum: int, frame: object) -> None:
        shutdown.set()

    previous_handlers = {}
    dispatcher = None
    try:
        for stop_signal in (signal.SIGTERM, signal.SIGINT):
            previous_handlers[stop_signal] = signal.signal(stop_signal, request_shutdown)
        dispatcher = build_dispatcher()
        while not shutdown.is_set():
            _dispatch_batch(dispatcher.dispatch_once, shutdown)
            _dispatch_batch(dispatcher.recover_unprocessed_once, shutdown)
            if not shutdown.is_set():
                shutdown.wait(interval)
    finally:
        try:
            if dispatcher is not None:
                dispatcher.database.engine.dispose()
        finally:
            for stop_signal, previous in previous_handlers.items():
                signal.signal(stop_signal, previous)


if __name__ == "__main__":
    main()
