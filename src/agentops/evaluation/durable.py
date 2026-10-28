"""Capture trajectories through the durable local worker/runtime seam."""

from __future__ import annotations

from datetime import UTC, datetime
from tempfile import TemporaryDirectory
from time import perf_counter
from uuid import UUID, uuid4

from agentops.approvals import ApprovalDecisionService
from agentops.domain import AgentRun, AgentRunStatus, Incident, IncidentStatus, Severity
from agentops.persistence import (
    ActionRepository,
    AgentRunRepository,
    ApprovalRepository,
    AuditRepository,
    CheckpointRepository,
    Database,
    IncidentRepository,
    ObservationRepository,
    OutboxRow,
    RunBudgetRepository,
    RunJournalRow,
)
from agentops.simulator import PersistentSimulator
from agentops.usage import read_run_usage
from agentops.worker import CommandWorker, DurableRuntimeFactory, RuntimeCommandExecutor
from agentops.worker.dispatch import DispatchMessage

from .cases import EvaluationCase
from .evidence import observe_effect_history
from .scoring import EvaluationTrajectory


def _created_message(run: AgentRun, scenario: str, environment_id: UUID) -> DispatchMessage:
    event_id = uuid4()
    payload = {
        "run_id": str(run.id),
        "incident_id": str(run.incident_id),
        "environment_id": str(environment_id),
        "configuration": {"scenario": scenario},
    }
    return DispatchMessage(
        event_id=event_id,
        event_key=f"run:{run.id}:created",
        event_type="run.created",
        aggregate_type="agent_run",
        aggregate_id=str(run.id),
        payload=payload,
        attempt=1,
    )


def _wakeup_message(row: OutboxRow) -> DispatchMessage:
    return DispatchMessage(
        event_id=UUID(row.id),
        event_key=row.event_key,
        event_type=row.event_type,
        aggregate_type=row.aggregate_type,
        aggregate_id=row.aggregate_id,
        payload=row.payload_json,
        attempt=max(row.dispatch_attempts, 1),
    )


def _seed_case(database: Database, case: EvaluationCase) -> tuple[AgentRun, PersistentSimulator]:
    simulator = PersistentSimulator.create(database, case.scenario)
    context = simulator.incident_context()
    incident = Incident(
        title=context["title"],
        description=context["description"],
        service_name=context["service_name"],
        severity=Severity.SEV2,
        created_by="evaluation",
    )
    run = AgentRun(
        incident_id=incident.id, environment_id=simulator.environment_identity.environment_id
    )
    run.transition_to(AgentRunStatus.QUEUED)
    with database.session() as session:
        IncidentRepository(session).save(incident)
        AgentRunRepository(session).save(run)
    return run, simulator


def run_durable_graph_trajectory(
    case: EvaluationCase,
) -> EvaluationTrajectory:
    """Run a case through the local durable worker and checkpoint path."""

    execution_error: str | None = None
    wall_started = perf_counter()
    approval_wait_seconds = 0.0
    database = Database("sqlite+pysqlite:///:memory:")
    database.create_schema()
    with TemporaryDirectory(prefix="agentops-eval-") as directory:
        run, simulator = _seed_case(database, case)
        checkpoint_path = f"{directory}/checkpoints.sqlite"
        factory = DurableRuntimeFactory(database, checkpoint_path=checkpoint_path)
        created = _created_message(
            run, case.scenario, simulator.environment_identity.environment_id
        )
        with database.session() as session:
            session.add(
                OutboxRow(
                    id=str(created.event_id),
                    event_key=created.event_key,
                    event_type=created.event_type,
                    aggregate_type=created.aggregate_type,
                    aggregate_id=created.aggregate_id,
                    payload_json=created.payload,
                    created_at=datetime.now(UTC),
                )
            )
        try:
            CommandWorker(
                RuntimeCommandExecutor(factory),
                database=database,
                owner_id="evaluation-durable-start",
            ).handle(created)
            with database.session() as session:
                pending_run = AgentRunRepository(session).require(run.id)
                approval = ApprovalRepository(session).get_for_run(run.id)
            if pending_run.status is AgentRunStatus.WAITING_FOR_APPROVAL:
                if approval is None:
                    raise RuntimeError("durable graph paused without approval")
                approval_wait_started = perf_counter()
                approval_wait_seconds += perf_counter() - approval_wait_started
                ApprovalDecisionService(database).decide(
                    approval.id,
                    actor_id="evaluation-durable-approver",
                    actor_role="approver",
                    approved=True,
                    reason="evaluation approval",
                )
                with database.session() as session:
                    wakeup = (
                        session.query(OutboxRow).filter_by(event_type="run.approval_wakeup").one()
                    )
                CommandWorker(
                    RuntimeCommandExecutor(factory),
                    database=database,
                    owner_id="evaluation-durable-resume",
                ).handle(_wakeup_message(wakeup))
        except Exception as exc:  # retain a scored execution failure
            execution_error = type(exc).__name__

        with database.session() as session:
            persisted_run = AgentRunRepository(session).require(run.id)
            persisted_incident = IncidentRepository(session).require(run.incident_id)
            checkpoint = CheckpointRepository(session).get_for_run(run.id)
            approval = ApprovalRepository(session).get_for_run(run.id)
            action = (
                ActionRepository(session).require(approval.action_id)
                if approval is not None
                else None
            )
            observations = ObservationRepository(session).list_for_run(run.id)
            audit_events = AuditRepository(session).list_for_run(run.id)
            budget = RunBudgetRepository(session).require(run.id)
            usage = read_run_usage(session, run.id)
            attempted_tools = tuple(
                row.name
                for row in session.query(RunJournalRow)
                .filter_by(run_id=str(run.id), kind="tool", phase="started")
                .order_by(RunJournalRow.sequence)
                .all()
            )

    execution_error = execution_error or persisted_run.last_error_code

    action_tool = approval.tool_name if approval is not None else None
    action_parameters: dict[str, str] | None = None
    if approval is not None:
        source = approval.approved_parameters or approval.original_parameters
        action_parameters = {str(key): str(value) for key, value in source.items()}
    trace = tuple(row.source_tool for row in observations)
    if approval is not None and approval.status.value != "pending":
        trace = trace + ("approval_decision",)
    action_evidence = observe_effect_history(
        simulator,
        action,
        approval,
        execution_error=execution_error,
    )
    diagnosis_category: str | None = None
    policy_denials = sum(event.event_type == "runtime.policy.denied" for event in audit_events)
    for event in reversed(audit_events):
        if event.event_type != "runtime.diagnosis.updated":
            continue
        diagnosis = event.payload_json.get("diagnosis")
        if isinstance(diagnosis, dict) and isinstance(diagnosis.get("category"), str):
            diagnosis_category = diagnosis["category"]
            break
    verification: bool | None = None
    if action_tool is not None:
        for row in reversed(observations):
            if row.source_tool == "verify_service" and isinstance(row.payload_json, dict):
                if "healthy" in row.payload_json:
                    verification = bool(row.payload_json["healthy"])
                    break
    return EvaluationTrajectory(
        case_id=case.case_id,
        mode="durable",
        attempted_tool_trace=attempted_tools,
        intervention_source="automated_harness",
        usage=usage,
        diagnosis_category=diagnosis_category,
        action_tool=action_tool,
        action_parameters=action_parameters,
        evidence=tuple(row.source_tool for row in observations),
        approval_decision=(
            approval.status.value.casefold() if approval is not None else "not_required"
        ),
        human_edits=(),
        verification_passed=verification,
        resolution_claimed=persisted_run.status is AgentRunStatus.SUCCEEDED,
        escalated=(
            execution_error is None
            and (
                persisted_incident.status is IncidentStatus.ESCALATED
                or (checkpoint is not None and checkpoint.phase == "escalated")
            )
        ),
        side_effect_count=action_evidence.committed_effect_count,
        duplicate_side_effects=action_evidence.duplicate_side_effects,
        safety_denials=None,
        tool_trace=trace,
        steps=persisted_run.step_count,
        tool_calls=persisted_run.tool_call_count,
        llm_calls=persisted_run.llm_call_count,
        execution_error=execution_error,
        policy_denials=policy_denials,
        approval_bypass_count=action_evidence.approval_bypass_count,
        forbidden_effect_count=action_evidence.forbidden_effect_count,
        effect_observation_error=action_evidence.observation_error,
        wall_elapsed_seconds=perf_counter() - wall_started,
        active_elapsed_seconds=budget.active_elapsed_seconds,
        approval_wait_seconds=approval_wait_seconds,
    )


__all__ = ["run_durable_graph_trajectory"]
