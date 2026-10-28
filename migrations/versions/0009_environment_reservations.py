"""Persist one durable environment reservation per active run."""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0009_environment_reservations"
down_revision = "0008_run_leases"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "environment_reservations",
        sa.Column("environment_id", sa.String(length=36), nullable=False),
        sa.Column("run_id", sa.String(length=36), nullable=False),
        sa.Column("generation", sa.Integer(), nullable=False),
        sa.Column("acquired_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("released_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint(
            "generation >= 1",
            name="ck_environment_reservations_generation_positive",
        ),
        sa.ForeignKeyConstraint(["environment_id"], ["simulation_environments.id"]),
        sa.ForeignKeyConstraint(["run_id"], ["agent_runs.id"]),
        sa.PrimaryKeyConstraint("environment_id"),
        sa.UniqueConstraint("run_id", name="uq_environment_reservations_run"),
    )
    op.create_index(
        "ix_environment_reservations_run_id",
        "environment_reservations",
        ["run_id"],
    )
    op.create_index(
        "ix_environment_reservations_released_at",
        "environment_reservations",
        ["released_at"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_environment_reservations_released_at",
        table_name="environment_reservations",
    )
    op.drop_index(
        "ix_environment_reservations_run_id",
        table_name="environment_reservations",
    )
    op.drop_table("environment_reservations")
