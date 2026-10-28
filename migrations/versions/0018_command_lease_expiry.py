"""Persist exact command lease expiry without guessing active legacy deadlines."""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0018_command_lease_expiry"
down_revision = "0017_evaluation_admission"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "outbox_events",
        sa.Column("processing_expires_at", sa.DateTime(timezone=True), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("outbox_events", "processing_expires_at")
