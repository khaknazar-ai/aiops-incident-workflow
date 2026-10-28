"""Persist immutable approval decisions, audit events, and outbox intents."""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0005_approval_transactions"
down_revision = "0004_execution_ledger"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "approval_decisions",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("approval_id", sa.String(length=36), nullable=False),
        sa.Column("action_id", sa.String(length=36), nullable=False),
        sa.Column("run_id", sa.String(length=36), nullable=False),
        sa.Column("status", sa.String(length=40), nullable=False),
        sa.Column("decided_by", sa.String(length=200), nullable=False),
        sa.Column("actor_role", sa.String(length=40), nullable=False),
        sa.Column("approved_parameters_json", sa.JSON(), nullable=True),
        sa.Column("approved_parameter_hash", sa.String(length=64), nullable=True),
        sa.Column("decision_reason", sa.String(length=2000), nullable=True),
        sa.Column("decision_fingerprint", sa.String(length=64), nullable=False),
        sa.Column("decision_revision", sa.Integer(), nullable=False),
        sa.Column("decided_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "decision_revision >= 1", name="ck_approval_decisions_revision_positive"
        ),
        sa.ForeignKeyConstraint(["approval_id"], ["approval_requests.id"]),
        sa.ForeignKeyConstraint(["action_id"], ["actions.id"]),
        sa.ForeignKeyConstraint(["run_id"], ["agent_runs.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("approval_id", name="uq_approval_decisions_approval"),
        sa.UniqueConstraint(
            "approval_id",
            "decision_revision",
            name="uq_approval_decisions_approval_revision",
        ),
    )
    op.create_index("ix_approval_decisions_approval_id", "approval_decisions", ["approval_id"])
    op.create_index("ix_approval_decisions_action_id", "approval_decisions", ["action_id"])
    op.create_index("ix_approval_decisions_run_id", "approval_decisions", ["run_id"])

    op.create_table(
        "audit_events",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("event_key", sa.String(length=300), nullable=False),
        sa.Column("occurred_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("actor_type", sa.String(length=40), nullable=False),
        sa.Column("actor_id", sa.String(length=200), nullable=False),
        sa.Column("event_type", sa.String(length=100), nullable=False),
        sa.Column("entity_type", sa.String(length=100), nullable=False),
        sa.Column("entity_id", sa.String(length=36), nullable=True),
        sa.Column("run_id", sa.String(length=36), nullable=True),
        sa.Column("payload_json", sa.JSON(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("event_key", name="uq_audit_events_event_key"),
    )
    op.create_table(
        "outbox_events",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("event_key", sa.String(length=300), nullable=False),
        sa.Column("event_type", sa.String(length=100), nullable=False),
        sa.Column("aggregate_type", sa.String(length=100), nullable=False),
        sa.Column("aggregate_id", sa.String(length=36), nullable=False),
        sa.Column("payload_json", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("published_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("event_key", name="uq_outbox_events_event_key"),
    )


def downgrade() -> None:
    op.drop_table("outbox_events")
    op.drop_table("audit_events")
    op.drop_index("ix_approval_decisions_run_id", table_name="approval_decisions")
    op.drop_index("ix_approval_decisions_action_id", table_name="approval_decisions")
    op.drop_index("ix_approval_decisions_approval_id", table_name="approval_decisions")
    op.drop_table("approval_decisions")
