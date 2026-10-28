"""Add an execution journal and nullable observation attempt linkage; no backfill."""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0021_run_journal"
down_revision = "0020_run_model_configuration"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "run_journal",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("run_id", sa.String(36), sa.ForeignKey("agent_runs.id"), nullable=False),
        sa.Column("sequence", sa.Integer(), nullable=False),
        sa.Column("attempt_id", sa.String(36), nullable=False),
        sa.Column("step_id", sa.String(36)),
        sa.Column("attempt_number", sa.Integer(), nullable=False),
        sa.Column("logical_step", sa.Integer(), nullable=False),
        sa.Column("kind", sa.String(16), nullable=False),
        sa.Column("name", sa.String(64), nullable=False),
        sa.Column("phase", sa.String(16), nullable=False),
        sa.Column("occurred_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("duration_ms", sa.Float()),
        sa.Column("model_configuration_id", sa.String(64)),
        sa.Column("facts_json", sa.JSON(), nullable=False),
        sa.Column("links_json", sa.JSON(), nullable=False),
        sa.UniqueConstraint("run_id", "sequence", name="uq_run_journal_sequence"),
        sa.UniqueConstraint("attempt_id", "phase", name="uq_run_journal_attempt_phase"),
        sa.CheckConstraint("sequence >= 1 AND attempt_number >= 1", name="ck_run_journal_positive"),
    )
    op.create_index("ix_run_journal_run_id", "run_journal", ["run_id"])
    op.create_index("ix_run_journal_attempt_id", "run_journal", ["attempt_id"])
    op.add_column("run_observations", sa.Column("tool_call_id", sa.String(36), nullable=True))
    op.create_index("ix_run_observations_tool_call_id", "run_observations", ["tool_call_id"])


def downgrade() -> None:
    op.drop_index("ix_run_observations_tool_call_id", table_name="run_observations")
    op.drop_column("run_observations", "tool_call_id")
    op.drop_table("run_journal")
