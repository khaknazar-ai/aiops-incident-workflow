"""Persist one replay-safe outcome projection for each command event."""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0014_command_outcomes"
down_revision = "0013_run_environment_binding"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "command_outcomes",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("event_id", sa.String(length=36), nullable=False),
        sa.Column("event_key", sa.String(length=300), nullable=False),
        sa.Column("event_type", sa.String(length=100), nullable=False),
        sa.Column("aggregate_type", sa.String(length=100), nullable=False),
        sa.Column("aggregate_id", sa.String(length=36), nullable=False),
        sa.Column("run_id", sa.String(length=36), nullable=True),
        sa.Column("status", sa.String(length=40), nullable=False),
        sa.Column("attempt", sa.Integer(), nullable=False),
        sa.Column("result_json", sa.JSON(), nullable=True),
        sa.Column("error_code", sa.String(length=100), nullable=True),
        sa.Column("error_message", sa.String(length=2000), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint("attempt >= 1", name="ck_command_outcomes_attempt_positive"),
        sa.ForeignKeyConstraint(["event_id"], ["outbox_events.id"]),
        sa.ForeignKeyConstraint(["run_id"], ["agent_runs.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("event_id", name="uq_command_outcomes_event_id"),
    )
    op.create_index("ix_command_outcomes_run_id", "command_outcomes", ["run_id"])
    op.create_index("ix_command_outcomes_status", "command_outcomes", ["status"])
    op.create_index("ix_command_outcomes_updated_at", "command_outcomes", ["updated_at"])


def downgrade() -> None:
    op.drop_index("ix_command_outcomes_updated_at", table_name="command_outcomes")
    op.drop_index("ix_command_outcomes_status", table_name="command_outcomes")
    op.drop_index("ix_command_outcomes_run_id", table_name="command_outcomes")
    op.drop_table("command_outcomes")
