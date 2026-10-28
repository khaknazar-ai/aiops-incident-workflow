"""Immutable individual hypotheses linked to existing structured decisions."""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0022_hypothesis_lineage"
down_revision = "0021_run_journal"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "hypothesis_decisions",
        sa.Column("id", sa.String(36), sa.ForeignKey("run_journal.id"), primary_key=True),
        sa.Column("run_id", sa.String(36), sa.ForeignKey("agent_runs.id"), nullable=False),
        sa.Column("logical_step", sa.Integer(), nullable=False),
        sa.Column("request_fingerprint", sa.String(64), nullable=False),
        sa.Column("snapshot_version_ids", sa.JSON(), nullable=False),
        sa.Column("changed_version_ids", sa.JSON(), nullable=False),
        sa.Column("needs_more_evidence", sa.Boolean(), nullable=False),
        sa.Column("diagnosis_json", sa.JSON()),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("run_id", "logical_step", name="uq_hypothesis_decision_step"),
    )
    op.create_index("ix_hypothesis_decisions_run_id", "hypothesis_decisions", ["run_id"])
    op.create_table(
        "hypothesis_versions",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("hypothesis_id", sa.String(36), nullable=False),
        sa.Column("run_id", sa.String(36), sa.ForeignKey("agent_runs.id"), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column(
            "decision_id", sa.String(36), sa.ForeignKey("hypothesis_decisions.id"), nullable=False
        ),
        sa.Column("snapshot_json", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("hypothesis_id", "version", name="uq_hypothesis_version"),
        sa.CheckConstraint("version >= 1", name="ck_hypothesis_version_positive"),
    )
    op.create_index("ix_hypothesis_versions_run_id", "hypothesis_versions", ["run_id"])
    op.create_index(
        "ix_hypothesis_versions_hypothesis_id", "hypothesis_versions", ["hypothesis_id"]
    )


def downgrade() -> None:
    op.drop_table("hypothesis_versions")
    op.drop_table("hypothesis_decisions")
