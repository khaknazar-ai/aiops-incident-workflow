"""Persist run observations for replayable incident evidence."""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0003_run_observations"
down_revision = "0002_simulation_storage"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "run_observations",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("run_id", sa.String(length=36), nullable=False),
        sa.Column("sequence", sa.Integer(), nullable=False),
        sa.Column("environment_id", sa.String(length=36), nullable=False),
        sa.Column("environment_version", sa.Integer(), nullable=False),
        sa.Column("source_tool", sa.String(length=200), nullable=False),
        sa.Column("payload_json", sa.JSON(), nullable=False),
        sa.Column("observed_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint("sequence >= 0", name="ck_run_observations_sequence_nonnegative"),
        sa.ForeignKeyConstraint(["run_id"], ["agent_runs.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("run_id", "sequence", name="uq_run_observations_run_sequence"),
    )
    op.create_index("ix_run_observations_run_id", "run_observations", ["run_id"])


def downgrade() -> None:
    op.drop_index("ix_run_observations_run_id", table_name="run_observations")
    op.drop_table("run_observations")
