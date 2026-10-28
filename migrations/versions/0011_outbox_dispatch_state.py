"""Persist bounded outbox dispatch claims and retry state."""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0011_outbox_dispatch_state"
down_revision = "0010_api_request_idempotency"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "outbox_events",
        sa.Column("available_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column(
        "outbox_events",
        sa.Column("dispatch_attempts", sa.Integer(), nullable=False, server_default="0"),
    )
    op.add_column(
        "outbox_events",
        sa.Column("claimed_by", sa.String(length=200), nullable=True),
    )
    op.add_column(
        "outbox_events",
        sa.Column("claimed_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column(
        "outbox_events",
        sa.Column("last_error_code", sa.String(length=100), nullable=True),
    )
    op.add_column(
        "outbox_events",
        sa.Column("last_error_message", sa.String(length=2000), nullable=True),
    )
    op.create_index("ix_outbox_events_available_at", "outbox_events", ["available_at"])
    op.create_index("ix_outbox_events_claimed_by", "outbox_events", ["claimed_by"])


def downgrade() -> None:
    op.drop_index("ix_outbox_events_claimed_by", table_name="outbox_events")
    op.drop_index("ix_outbox_events_available_at", table_name="outbox_events")
    op.drop_column("outbox_events", "last_error_message")
    op.drop_column("outbox_events", "last_error_code")
    op.drop_column("outbox_events", "claimed_at")
    op.drop_column("outbox_events", "claimed_by")
    op.drop_column("outbox_events", "dispatch_attempts")
    op.drop_column("outbox_events", "available_at")
