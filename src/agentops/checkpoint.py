"""Durable LangGraph checkpoint adapters for local recovery tests."""

from __future__ import annotations

import sqlite3
from collections.abc import Callable, Iterator, Sequence
from contextlib import AbstractContextManager
from pathlib import Path
from typing import TYPE_CHECKING, Any

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    BaseCheckpointSaver,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
)
from langgraph.checkpoint.sqlite import SqliteSaver

if TYPE_CHECKING:
    from sqlalchemy.orm import Session


class CheckpointCompatibilityError(RuntimeError):
    """Raised when a persisted run cannot be safely resumed from its checkpoint."""


CheckpointWriteGuard = Callable[[RunnableConfig], AbstractContextManager["Session | None"]]


class FencedCheckpointSaver(BaseCheckpointSaver):
    """Hold the domain lease fence until each synchronous checkpoint write finishes."""

    def __init__(self, saver: BaseCheckpointSaver, write_guard: CheckpointWriteGuard) -> None:
        super().__init__(serde=saver.serde)
        self.delegate = saver
        self.write_guard = write_guard

    @property
    def config_specs(self) -> list:
        return self.delegate.config_specs

    def get_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        return self.delegate.get_tuple(config)

    def list(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> Iterator[CheckpointTuple]:
        yield from self.delegate.list(config, filter=filter, before=before, limit=limit)

    def get_next_version(self, current: Any, channel: Any) -> Any:
        return self.delegate.get_next_version(current, channel)

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        with self.write_guard(config):
            return self.delegate.put(config, checkpoint, metadata, new_versions)

    def put_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        with self.write_guard(config):
            self.delegate.put_writes(config, writes, task_id, task_path)

    def delete_thread(self, thread_id: str) -> None:
        with self.write_guard({"configurable": {"thread_id": thread_id}}):
            self.delegate.delete_thread(thread_id)


class SqliteCheckpointStore:
    """Persist LangGraph checkpoints in a local SQLite database file."""

    def __init__(
        self, path: str | Path, *, write_guard: CheckpointWriteGuard | None = None
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path, check_same_thread=False)
        saver = SqliteSaver(self.connection)
        saver.setup()
        self.saver = FencedCheckpointSaver(saver, write_guard) if write_guard is not None else saver

    def close(self) -> None:
        """Close the underlying connection after all graph calls finish."""

        self.connection.close()
