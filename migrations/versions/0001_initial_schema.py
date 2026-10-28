"""Create the initial incident domain schema."""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0001_initial"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "incidents",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("title", sa.String(length=300), nullable=False),
        sa.Column("description", sa.String(length=4000), nullable=False),
        sa.Column("service_name", sa.String(length=200), nullable=False),
        sa.Column("severity", sa.String(length=10), nullable=False),
        sa.Column("status", sa.String(length=40), nullable=False),
        sa.Column("detected_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("resolved_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_by", sa.String(length=200), nullable=False),
        sa.Column("active_run_id", sa.String(length=36), nullable=True),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.CheckConstraint("version >= 1", name="ck_incidents_version_positive"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_incidents_service_name", "incidents", ["service_name"])
    op.create_index("ix_incidents_status", "incidents", ["status"])

    op.create_table(
        "agent_runs",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("incident_id", sa.String(length=36), nullable=False),
        sa.Column("status", sa.String(length=40), nullable=False),
        sa.Column("workflow_name", sa.String(length=200), nullable=False),
        sa.Column("workflow_version", sa.String(length=50), nullable=False),
        sa.Column("state_schema_version", sa.Integer(), nullable=False),
        sa.Column("policy_version", sa.String(length=50), nullable=False),
        sa.Column("tool_registry_version", sa.String(length=50), nullable=False),
        sa.Column("thread_id", sa.String(length=200), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("step_count", sa.Integer(), nullable=False),
        sa.Column("llm_call_count", sa.Integer(), nullable=False),
        sa.Column("tool_call_count", sa.Integer(), nullable=False),
        sa.Column("last_error_code", sa.String(length=100), nullable=True),
        sa.Column("last_error_message", sa.String(length=2000), nullable=True),
        sa.CheckConstraint("step_count >= 0", name="ck_agent_runs_step_count_nonnegative"),
        sa.CheckConstraint("llm_call_count >= 0", name="ck_agent_runs_llm_count_nonnegative"),
        sa.CheckConstraint("tool_call_count >= 0", name="ck_agent_runs_tool_count_nonnegative"),
        sa.ForeignKeyConstraint(["incident_id"], ["incidents.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("thread_id"),
    )
    op.create_index("ix_agent_runs_incident_id", "agent_runs", ["incident_id"])
    op.create_index("ix_agent_runs_status", "agent_runs", ["status"])

    op.create_table(
        "actions",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("run_id", sa.String(length=36), nullable=False),
        sa.Column("action_type", sa.String(length=100), nullable=False),
        sa.Column("tool_name", sa.String(length=200), nullable=False),
        sa.Column("parameters_json", sa.JSON(), nullable=False),
        sa.Column("parameter_hash", sa.String(length=64), nullable=False),
        sa.Column("risk_level", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(length=40), nullable=False),
        sa.Column("requires_approval", sa.Boolean(), nullable=False),
        sa.Column("idempotency_key", sa.String(length=200), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("executed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("execution_result_id", sa.String(length=36), nullable=True),
        sa.CheckConstraint("risk_level BETWEEN 0 AND 3", name="ck_actions_risk_level"),
        sa.CheckConstraint(
            "requires_approval IN (false, true)", name="ck_actions_requires_approval"
        ),
        sa.ForeignKeyConstraint(["run_id"], ["agent_runs.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("idempotency_key"),
    )
    op.create_index("ix_actions_status", "actions", ["status"])

    op.create_table(
        "approval_requests",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("action_id", sa.String(length=36), nullable=False),
        sa.Column("run_id", sa.String(length=36), nullable=False),
        sa.Column("status", sa.String(length=40), nullable=False),
        sa.Column("requested_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("decided_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("decided_by", sa.String(length=200), nullable=True),
        sa.Column("original_parameters_json", sa.JSON(), nullable=False),
        sa.Column("original_parameter_hash", sa.String(length=64), nullable=False),
        sa.Column("approved_parameters_json", sa.JSON(), nullable=True),
        sa.Column("approved_parameter_hash", sa.String(length=64), nullable=True),
        sa.Column("decision_reason", sa.String(length=2000), nullable=True),
        sa.CheckConstraint(
            "approved_parameter_hash IS NULL OR length(approved_parameter_hash) = 64",
            name="ck_approvals_approved_hash",
        ),
        sa.ForeignKeyConstraint(["action_id"], ["actions.id"]),
        sa.ForeignKeyConstraint(["run_id"], ["agent_runs.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_approval_requests_status", "approval_requests", ["status"])


def downgrade() -> None:
    op.drop_index("ix_approval_requests_status", table_name="approval_requests")
    op.drop_table("approval_requests")
    op.drop_index("ix_actions_status", table_name="actions")
    op.drop_table("actions")
    op.drop_index("ix_agent_runs_status", table_name="agent_runs")
    op.drop_index("ix_agent_runs_incident_id", table_name="agent_runs")
    op.drop_table("agent_runs")
    op.drop_index("ix_incidents_status", table_name="incidents")
    op.drop_index("ix_incidents_service_name", table_name="incidents")
    op.drop_table("incidents")
