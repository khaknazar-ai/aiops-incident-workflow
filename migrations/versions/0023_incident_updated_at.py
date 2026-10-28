"""Record incident activity timestamps without inventing legacy history."""

import sqlalchemy as sa
from alembic import op

revision = "0023_incident_updated_at"
down_revision = "0022_hypothesis_lineage"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Historical last-change time cannot be inferred from detection time.
    op.add_column("incidents", sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True))


def downgrade() -> None:
    op.drop_column("incidents", "updated_at")
