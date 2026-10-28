"""Persist application execution claims and receipts."""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0004_execution_ledger"
down_revision = "0003_run_observations"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "executions",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("execution_key", sa.String(length=300), nullable=False),
        sa.Column("tool_name", sa.String(length=200), nullable=False),
        sa.Column("parameter_hash", sa.String(length=64), nullable=False),
        sa.Column("status", sa.String(length=20), nullable=False),
        sa.Column("result_json", sa.JSON(), nullable=True),
        sa.Column("attempt_count", sa.Integer(), nullable=False),
        sa.Column("last_error_code", sa.String(length=100), nullable=True),
        sa.Column("last_error_message", sa.String(length=2000), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint("attempt_count >= 1", name="ck_executions_attempt_count_positive"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("execution_key", name="uq_executions_execution_key"),
    )
    op.create_index("ix_executions_status", "executions", ["status"])


def downgrade() -> None:
    op.drop_index("ix_executions_status", table_name="executions")
    op.drop_table("executions")
