"""Persist digest-only browser logout across application instances."""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0019_browser_session_revocations"
down_revision = "0018_command_lease_expiry"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "browser_session_revocations",
        sa.Column("token_hash", sa.String(64), primary_key=True),
        sa.Column("expires_at", sa.BigInteger(), nullable=False),
        sa.Column("revoked_at", sa.BigInteger(), nullable=False),
        sa.CheckConstraint("length(token_hash) = 64", name="ck_browser_revocations_hash_length"),
        sa.CheckConstraint("expires_at > revoked_at", name="ck_browser_revocations_valid_expiry"),
    )
    op.create_index(
        "ix_browser_session_revocations_expires_at", "browser_session_revocations", ["expires_at"]
    )


def downgrade() -> None:
    op.drop_index(
        "ix_browser_session_revocations_expires_at", table_name="browser_session_revocations"
    )
    op.drop_table("browser_session_revocations")
