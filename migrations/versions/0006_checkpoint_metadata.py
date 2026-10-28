"""Persist run checkpoint compatibility metadata for safe recovery."""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0006_checkpoint_metadata"
down_revision = "0005_approval_transactions"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "run_checkpoints",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("run_id", sa.String(length=36), nullable=False),
        sa.Column("thread_id", sa.String(length=200), nullable=False),
        sa.Column("checkpoint_id", sa.String(length=300), nullable=False),
        sa.Column("checkpoint_kind", sa.String(length=40), nullable=False),
        sa.Column("workflow_name", sa.String(length=200), nullable=False),
        sa.Column("workflow_version", sa.String(length=50), nullable=False),
        sa.Column("state_schema_version", sa.Integer(), nullable=False),
        sa.Column("policy_version", sa.String(length=50), nullable=False),
        sa.Column("tool_registry_version", sa.String(length=50), nullable=False),
        sa.Column("phase", sa.String(length=80), nullable=False),
        sa.Column("action_id", sa.String(length=36), nullable=True),
        sa.Column("approval_id", sa.String(length=36), nullable=True),
        sa.Column("decision_id", sa.String(length=36), nullable=True),
        sa.Column("metadata_json", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "state_schema_version >= 1",
            name="ck_run_checkpoints_schema_version_positive",
        ),
        sa.ForeignKeyConstraint(["run_id"], ["agent_runs.id"]),
        sa.ForeignKeyConstraint(["action_id"], ["actions.id"]),
        sa.ForeignKeyConstraint(["approval_id"], ["approval_requests.id"]),
        sa.ForeignKeyConstraint(["decision_id"], ["approval_decisions.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("run_id", name="uq_run_checkpoints_run"),
        sa.UniqueConstraint("thread_id", name="uq_run_checkpoints_thread"),
    )
    op.create_index("ix_run_checkpoints_run_id", "run_checkpoints", ["run_id"])


def downgrade() -> None:
    op.drop_index("ix_run_checkpoints_run_id", table_name="run_checkpoints")
    op.drop_table("run_checkpoints")
