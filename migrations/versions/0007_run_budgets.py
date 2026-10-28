"""Persist run budget limits, usage, and active execution segments."""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0007_run_budgets"
down_revision = "0006_checkpoint_metadata"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "run_budgets",
        sa.Column("run_id", sa.String(length=36), nullable=False),
        sa.Column("max_steps", sa.Integer(), nullable=False),
        sa.Column("max_llm_calls", sa.Integer(), nullable=False),
        sa.Column("max_tool_calls", sa.Integer(), nullable=False),
        sa.Column("max_active_seconds", sa.Float(), nullable=False),
        sa.Column("steps_used", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("llm_calls_used", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("tool_calls_used", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("active_elapsed_seconds", sa.Float(), nullable=False, server_default="0"),
        sa.Column("active_started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint("max_steps >= 1", name="ck_run_budgets_max_steps_positive"),
        sa.CheckConstraint("max_llm_calls >= 1", name="ck_run_budgets_max_llm_positive"),
        sa.CheckConstraint("max_tool_calls >= 1", name="ck_run_budgets_max_tool_positive"),
        sa.CheckConstraint(
            "max_active_seconds > 0",
            name="ck_run_budgets_max_active_seconds_positive",
        ),
        sa.CheckConstraint("steps_used >= 0", name="ck_run_budgets_steps_nonnegative"),
        sa.CheckConstraint("llm_calls_used >= 0", name="ck_run_budgets_llm_calls_nonnegative"),
        sa.CheckConstraint("tool_calls_used >= 0", name="ck_run_budgets_tool_calls_nonnegative"),
        sa.CheckConstraint(
            "active_elapsed_seconds >= 0",
            name="ck_run_budgets_active_elapsed_nonnegative",
        ),
        sa.ForeignKeyConstraint(["run_id"], ["agent_runs.id"]),
        sa.PrimaryKeyConstraint("run_id"),
    )


def downgrade() -> None:
    op.drop_table("run_budgets")
