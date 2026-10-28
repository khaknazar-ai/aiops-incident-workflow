"""Persist scoped HTTP idempotency responses."""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0010_api_request_idempotency"
down_revision = "0009_environment_reservations"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "api_request_idempotency",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("actor_id", sa.String(length=200), nullable=False),
        sa.Column("operation", sa.String(length=100), nullable=False),
        sa.Column("target_id", sa.String(length=200), nullable=False),
        sa.Column("idempotency_key", sa.String(length=200), nullable=False),
        sa.Column("request_fingerprint", sa.String(length=64), nullable=False),
        sa.Column("resource_type", sa.String(length=100), nullable=False),
        sa.Column("resource_id", sa.String(length=36), nullable=False),
        sa.Column("response_status", sa.Integer(), nullable=False),
        sa.Column("response_json", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "actor_id",
            "operation",
            "target_id",
            "idempotency_key",
            name="uq_api_request_idempotency_scope",
        ),
    )
    op.create_index(
        "ix_api_request_idempotency_created_at",
        "api_request_idempotency",
        ["created_at"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_api_request_idempotency_created_at",
        table_name="api_request_idempotency",
    )
    op.drop_table("api_request_idempotency")
