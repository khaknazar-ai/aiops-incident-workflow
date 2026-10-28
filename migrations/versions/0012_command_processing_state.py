"""Persist command processing claims and acknowledgements."""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0012_command_processing_state"
down_revision = "0011_outbox_dispatch_state"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "outbox_events",
        sa.Column("processing_available_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column(
        "outbox_events",
        sa.Column("processing_attempts", sa.Integer(), nullable=False, server_default="0"),
    )
    op.add_column(
        "outbox_events",
        sa.Column("processing_owner", sa.String(length=200), nullable=True),
    )
    op.add_column(
        "outbox_events",
        sa.Column("processing_started_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column(
        "outbox_events",
        sa.Column("processed_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column(
        "outbox_events",
        sa.Column("processing_error_code", sa.String(length=100), nullable=True),
    )
    op.add_column(
        "outbox_events",
        sa.Column("processing_error_message", sa.String(length=2000), nullable=True),
    )
    op.create_index(
        "ix_outbox_events_processing_available_at",
        "outbox_events",
        ["processing_available_at"],
    )
    op.create_index(
        "ix_outbox_events_processing_owner",
        "outbox_events",
        ["processing_owner"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_outbox_events_processing_owner",
        table_name="outbox_events",
    )
    op.drop_index(
        "ix_outbox_events_processing_available_at",
        table_name="outbox_events",
    )
    op.drop_column("outbox_events", "processing_error_message")
    op.drop_column("outbox_events", "processing_error_code")
    op.drop_column("outbox_events", "processed_at")
    op.drop_column("outbox_events", "processing_started_at")
    op.drop_column("outbox_events", "processing_owner")
    op.drop_column("outbox_events", "processing_attempts")
    op.drop_column("outbox_events", "processing_available_at")
