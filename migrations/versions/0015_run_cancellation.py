"""Persist cancellation intent for durable workflow runs."""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0015_run_cancellation"
down_revision = "0014_command_outcomes"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "agent_runs",
        sa.Column("cancel_requested_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_agent_runs_cancel_requested_at", "agent_runs", ["cancel_requested_at"])


def downgrade() -> None:
    op.drop_index("ix_agent_runs_cancel_requested_at", table_name="agent_runs")
    op.drop_column("agent_runs", "cancel_requested_at")
