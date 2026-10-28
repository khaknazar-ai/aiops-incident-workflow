"""Explicitly migrated PostgreSQL LangGraph storage, separate from domain tables."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from contextlib import contextmanager
from typing import Any

import psycopg
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import ChannelVersions, Checkpoint, CheckpointMetadata
from psycopg.pq import TransactionStatus
from psycopg.rows import dict_row
from sqlalchemy.engine import URL, make_url
from sqlalchemy.exc import ArgumentError
from sqlalchemy.orm import Session

from agentops.checkpoint import (
    CheckpointCompatibilityError,
    CheckpointWriteGuard,
    FencedCheckpointSaver,
)

CHECKPOINT_SCHEMA = "agentops_checkpoints"
_MIGRATION_LOCK = 746219032


def _connect(url: str | URL):
    try:
        parsed = make_url(url) if isinstance(url, str) else url
    except (ArgumentError, ValueError):
        raise CheckpointCompatibilityError("PostgreSQL checkpoint URL is invalid") from None
    if parsed.drivername not in {"postgresql", "postgresql+psycopg"}:
        raise CheckpointCompatibilityError("PostgreSQL checkpoints require a PostgreSQL database")
    try:
        return psycopg.connect(
            parsed.set(drivername="postgresql").render_as_string(hide_password=False),
            autocommit=True,
            prepare_threshold=0,
            row_factory=dict_row,
            connect_timeout=5,
            options=(
                f"-c search_path={CHECKPOINT_SCHEMA} "
                "-c statement_timeout=10000 -c lock_timeout=5000"
            ),
        )
    except psycopg.Error:
        # Connection exceptions may contain credentials or private endpoints.
        raise CheckpointCompatibilityError("PostgreSQL checkpoint connection unavailable") from None


def _saver_type():
    # SQLite-only installs do not need to import the PostgreSQL adapter.
    from langgraph.checkpoint.postgres import PostgresSaver

    return PostgresSaver


def _validate_schema(connection, saver_type, *, allow_missing: bool = False) -> None:
    found = connection.execute(
        "SELECT to_regclass(%s) AS name", (f"{CHECKPOINT_SCHEMA}.checkpoint_migrations",)
    ).fetchone()
    if found["name"] is None:
        if allow_missing:
            return
        raise CheckpointCompatibilityError(
            "PostgreSQL checkpoint migrations must run before worker"
        )
    versions = [
        row["v"] for row in connection.execute("SELECT v FROM checkpoint_migrations ORDER BY v")
    ]
    expected = list(range(len(saver_type.MIGRATIONS)))
    if versions != expected[: len(versions)] or (not allow_missing and versions != expected):
        raise CheckpointCompatibilityError("PostgreSQL checkpoint schema version is incompatible")


def migrate_postgres_checkpoints(url: str | URL) -> int:
    """Apply the pinned saver's versioned migrations only when explicitly requested."""

    saver_type = _saver_type()
    with _connect(url) as connection:
        acquired = connection.execute(
            "SELECT pg_try_advisory_lock(%s) AS acquired", (_MIGRATION_LOCK,)
        ).fetchone()["acquired"]
        if not acquired:
            raise CheckpointCompatibilityError(
                "PostgreSQL checkpoint migration already in progress"
            )
        try:
            connection.execute(f"CREATE SCHEMA IF NOT EXISTS {CHECKPOINT_SCHEMA}")
            _validate_schema(connection, saver_type, allow_missing=True)
            saver_type(connection).setup()
            _validate_schema(connection, saver_type)
            return len(saver_type.MIGRATIONS) - 1
        finally:
            connection.execute("SELECT pg_advisory_unlock(%s)", (_MIGRATION_LOCK,))


class _TransactionalPostgresSaver(FencedCheckpointSaver):
    """Commit every mutation in the transaction that holds the domain lease lock."""

    @contextmanager
    def _writer(self, config: RunnableConfig):
        with self.write_guard(config) as session:
            if not isinstance(session, Session) or session.get_bind().dialect.name != "postgresql":
                raise CheckpointCompatibilityError(
                    "PostgreSQL checkpoint writes require a lease transaction"
                )
            connection = session.connection().connection.driver_connection
            if (
                not isinstance(connection, psycopg.Connection)
                or connection.autocommit
                or connection.info.transaction_status != TransactionStatus.INTRANS
            ):
                raise CheckpointCompatibilityError(
                    "PostgreSQL checkpoint lease transaction is unavailable"
                )
            previous_path = connection.execute("SHOW search_path").fetchone()[0]
            connection.execute("SELECT set_config('search_path', %s, true)", (CHECKPOINT_SCHEMA,))
            connection.execute("SET LOCAL statement_timeout = '10s'")
            connection.execute("SET LOCAL lock_timeout = '5s'")
            try:
                # The pinned saver does not commit an already-open transaction.
                # The domain Session commits the lease renewal and payload together.
                yield type(self.delegate)(connection, serde=self.serde)
            finally:
                if connection.info.transaction_status == TransactionStatus.INTRANS:
                    connection.execute(
                        "SELECT set_config('search_path', %s, true)", (previous_path,)
                    )

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        with self._writer(config) as saver:
            return saver.put(config, checkpoint, metadata, new_versions)

    def put_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        with self._writer(config) as saver:
            saver.put_writes(config, writes, task_id, task_path)

    def delete_thread(self, thread_id: str) -> None:
        with self._writer({"configurable": {"thread_id": thread_id}}) as saver:
            saver.delete_thread(thread_id)


class PostgresCheckpointStore:
    """Open already migrated storage; hold the domain fence through each saver write."""

    def __init__(self, url: str | URL, *, write_guard: CheckpointWriteGuard) -> None:
        saver_type = _saver_type()
        self.connection = _connect(url)
        try:
            _validate_schema(self.connection, saver_type)
            self.saver = _TransactionalPostgresSaver(saver_type(self.connection), write_guard)
        except BaseException:
            self.connection.close()
            raise

    def close(self) -> None:
        self.connection.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("migrate",))
    parser.parse_args(argv)
    from agentops.database_config import database_url_from_environment

    try:
        version = migrate_postgres_checkpoints(database_url_from_environment())
    except (CheckpointCompatibilityError, psycopg.Error, ValueError):
        parser.exit(
            1, "PostgreSQL checkpoint migration failed; check configuration/schema privately.\n"
        )
    print(f"PostgreSQL checkpoint schema ready: {CHECKPOINT_SCHEMA}, version {version}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
