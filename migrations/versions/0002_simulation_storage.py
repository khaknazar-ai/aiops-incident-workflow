"""Persist simulator environments and mutation receipts."""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0002_simulation_storage"
down_revision = "0001_initial"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "simulation_environments",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("scenario", sa.String(length=100), nullable=False),
        sa.Column("generation", sa.Integer(), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("state_json", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "generation >= 1", name="ck_simulation_environments_generation_positive"
        ),
        sa.CheckConstraint("version >= 1", name="ck_simulation_environments_version_positive"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_table(
        "simulation_receipts",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("environment_id", sa.String(length=36), nullable=False),
        sa.Column("generation", sa.Integer(), nullable=False),
        sa.Column("idempotency_key", sa.String(length=200), nullable=False),
        sa.Column("tool_name", sa.String(length=200), nullable=False),
        sa.Column("parameter_hash", sa.String(length=64), nullable=False),
        sa.Column("result_json", sa.JSON(), nullable=False),
        sa.Column("environment_version", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint("generation >= 1", name="ck_simulation_receipts_generation_positive"),
        sa.ForeignKeyConstraint(["environment_id"], ["simulation_environments.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "environment_id",
            "generation",
            "idempotency_key",
            name="uq_simulation_receipts_environment_generation_key",
        ),
    )
    op.create_index(
        "ix_simulation_receipts_environment_id",
        "simulation_receipts",
        ["environment_id"],
    )


def downgrade() -> None:
    op.drop_index("ix_simulation_receipts_environment_id", table_name="simulation_receipts")
    op.drop_table("simulation_receipts")
    op.drop_table("simulation_environments")
