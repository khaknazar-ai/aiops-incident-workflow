"""Persist redacted model provenance without guessing identities for old runs."""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0020_run_model_configuration"
down_revision = "0019_browser_session_revocations"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("agent_runs", sa.Column("model_configuration_json", sa.JSON(), nullable=True))


def downgrade() -> None:
    op.drop_column("agent_runs", "model_configuration_json")
