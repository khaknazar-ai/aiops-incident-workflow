"""Persist the simulator environment bound to each durable run."""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0013_run_environment_binding"
down_revision = "0012_command_processing_state"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("agent_runs") as batch:
        batch.add_column(
            sa.Column(
                "environment_id",
                sa.String(length=36),
                nullable=True,
            )
        )
        batch.create_foreign_key(
            "fk_agent_runs_environment_id_simulation_environments",
            "simulation_environments",
            ["environment_id"],
            ["id"],
        )
        batch.create_index("ix_agent_runs_environment_id", ["environment_id"])


def downgrade() -> None:
    with op.batch_alter_table("agent_runs") as batch:
        batch.drop_index("ix_agent_runs_environment_id")
        batch.drop_constraint(
            "fk_agent_runs_environment_id_simulation_environments",
            type_="foreignkey",
        )
        batch.drop_column("environment_id")
