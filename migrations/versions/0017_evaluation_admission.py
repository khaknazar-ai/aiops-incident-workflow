"""Claim and bound offline evaluation work before execution."""

import sqlalchemy as sa
from alembic import op

revision = "0017_evaluation_admission"
down_revision = "0016_evaluation_runs"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "evaluation_jobs",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("actor_id", sa.String(200), nullable=False),
        sa.Column("idempotency_key", sa.String(200), nullable=False),
        sa.Column("request_fingerprint", sa.String(64), nullable=False),
        sa.Column("mode", sa.String(40), nullable=False),
        sa.Column("seed", sa.Integer(), nullable=False),
        sa.Column("owner_token", sa.String(36), nullable=False),
        sa.Column("status", sa.String(20), nullable=False),
        sa.Column("active_slot", sa.Integer()),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("finished_at", sa.DateTime(timezone=True)),
        sa.Column("error_code", sa.String(100)),
        sa.Column("evaluation_run_id", sa.String(36), sa.ForeignKey("evaluation_runs.id")),
        sa.UniqueConstraint("actor_id", "idempotency_key", name="uq_evaluation_jobs_request"),
        sa.UniqueConstraint("active_slot", name="uq_evaluation_jobs_active_slot"),
        sa.CheckConstraint(
            "active_slot IS NULL OR active_slot = 1", name="ck_evaluation_jobs_slot"
        ),
        sa.CheckConstraint("seed >= 0", name="ck_evaluation_jobs_seed"),
        sa.CheckConstraint(
            "status IN ('RUNNING', 'SUCCEEDED', 'FAILED', 'QUARANTINED')",
            name="ck_evaluation_jobs_status",
        ),
        sa.CheckConstraint(
            "(status IN ('RUNNING', 'QUARANTINED') "
            "AND active_slot IS NOT NULL AND active_slot = 1) "
            "OR (status IN ('SUCCEEDED', 'FAILED') AND active_slot IS NULL)",
            name="ck_evaluation_jobs_status_slot",
        ),
    )


def downgrade() -> None:
    op.drop_table("evaluation_jobs")
