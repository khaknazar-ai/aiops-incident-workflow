"""Validated database URL construction for local services and migrations."""

from __future__ import annotations

import os
from collections.abc import Mapping

from sqlalchemy import URL

DEFAULT_DATABASE_URL = "sqlite+pysqlite:///./data/agentops.db"


def database_url_from_environment(
    environment: Mapping[str, str] | None = None,
) -> str:
    """Return an explicit URL or safely encode configured connection components."""

    values = os.environ if environment is None else environment
    explicit = values.get("AGENTOPS_DATABASE_URL")
    if explicit:
        return explicit
    password = values.get("AGENTOPS_DATABASE_PASSWORD")
    if password is None:
        return DEFAULT_DATABASE_URL
    host = values.get("AGENTOPS_DATABASE_HOST", "postgres").strip()
    database = values.get("AGENTOPS_DATABASE_NAME", "agentops").strip()
    username = values.get("AGENTOPS_DATABASE_USER", "agentops").strip()
    driver = values.get("AGENTOPS_DATABASE_DRIVER", "postgresql+psycopg").strip()
    if not host or not database or not username or not driver:
        raise ValueError("database connection components must be non-empty")
    raw_port = values.get("AGENTOPS_DATABASE_PORT", "5432")
    try:
        port = int(raw_port)
    except ValueError as exc:
        raise ValueError("AGENTOPS_DATABASE_PORT must be an integer") from exc
    if not 1 <= port <= 65535:
        raise ValueError("AGENTOPS_DATABASE_PORT must be between 1 and 65535")
    return URL.create(
        drivername=driver,
        username=username,
        password=password,
        host=host,
        port=port,
        database=database,
    ).render_as_string(hide_password=False)


__all__ = ["DEFAULT_DATABASE_URL", "database_url_from_environment"]
