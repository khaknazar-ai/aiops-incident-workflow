"""Persist immutable offline evaluation reports."""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0016_evaluation_runs"
down_revision = "0015_run_cancellation"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "evaluation_runs",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("mode", sa.String(length=40), nullable=False),
        sa.Column("seed", sa.Integer(), nullable=False),
        sa.Column("manifest_version", sa.String(length=100), nullable=False),
        sa.Column("manifest_digest", sa.String(length=64), nullable=False),
        sa.Column("source_revision", sa.String(length=80), nullable=False),
        sa.Column("cohort", sa.String(length=100), nullable=False),
        sa.Column("denominator", sa.Integer(), nullable=False),
        sa.Column("pass_count", sa.Integer(), nullable=False),
        sa.Column("fail_count", sa.Integer(), nullable=False),
        sa.Column("unavailable_count", sa.Integer(), nullable=False),
        sa.Column("created_by", sa.String(length=200), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("report_json", sa.JSON(), nullable=False),
        sa.CheckConstraint("seed >= 0", name="ck_evaluation_runs_seed_nonnegative"),
        sa.CheckConstraint("denominator >= 0", name="ck_evaluation_runs_denominator_nonnegative"),
        sa.CheckConstraint("pass_count >= 0", name="ck_evaluation_runs_pass_count_nonnegative"),
        sa.CheckConstraint("fail_count >= 0", name="ck_evaluation_runs_fail_count_nonnegative"),
        sa.CheckConstraint(
            "unavailable_count >= 0",
            name="ck_evaluation_runs_unavailable_count_nonnegative",
        ),
        sa.CheckConstraint(
            "pass_count + fail_count + unavailable_count = denominator",
            name="ck_evaluation_runs_counts_match_denominator",
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_evaluation_runs_created_at",
        "evaluation_runs",
        ["created_at"],
    )


def downgrade() -> None:
    op.drop_index("ix_evaluation_runs_created_at", table_name="evaluation_runs")
    op.drop_table("evaluation_runs")
