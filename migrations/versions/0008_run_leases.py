"""Persist fenced run leases for worker ownership."""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0008_run_leases"
down_revision = "0007_run_budgets"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "run_leases",
        sa.Column("run_id", sa.String(length=36), nullable=False),
        sa.Column("owner_id", sa.String(length=200), nullable=False),
        sa.Column("generation", sa.Integer(), nullable=False),
        sa.Column("acquired_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("renewed_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint("generation >= 1", name="ck_run_leases_generation_positive"),
        sa.ForeignKeyConstraint(["run_id"], ["agent_runs.id"]),
        sa.PrimaryKeyConstraint("run_id"),
    )
    op.create_index("ix_run_leases_owner_id", "run_leases", ["owner_id"])
    op.create_index("ix_run_leases_expires_at", "run_leases", ["expires_at"])


def downgrade() -> None:
    op.drop_index("ix_run_leases_expires_at", table_name="run_leases")
    op.drop_index("ix_run_leases_owner_id", table_name="run_leases")
    op.drop_table("run_leases")
