"""LangGraph orchestration for a bounded, deterministic-first triage loop."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from time import monotonic, sleep
from typing import Any, Literal
from uuid import UUID, uuid4, uuid5

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.errors import GraphInterrupt
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from pydantic import BaseModel, ValidationError
from sqlalchemy.orm import Session

from agentops.approvals import ApprovalDecisionService
from agentops.checkpoint import (
    CheckpointCompatibilityError,
    FencedCheckpointSaver,
    SqliteCheckpointStore,
)
from agentops.domain import (
    Action,
    ActionStatus,
    AgentRun,
    AgentRunStatus,
    ApprovalRequest,
    ApprovalStatus,
    CancellationRequested,
    Incident,
    IncidentStatus,
    RiskLevel,
    Severity,
    stable_parameter_hash,
    utc_now,
)
from agentops.journal import RunJournalRepository
from agentops.journal_contract import observed_facts, safe_error
from agentops.llm import (
    DiagnosticPlan,
    FakeLLMClient,
    HypothesisSnapshot,
    HypothesisUpdate,
    LLMClient,
    LLMOutputError,
    LLMProviderError,
    Message,
    ModelConfig,
    RemediationProposal,
    ReviewDecision,
    ToolDecision,
    TriageDecision,
    build_prompt_messages,
)
from agentops.model_identity import model_configuration_identity
from agentops.observability.audit import durable_payload
from agentops.observability.redaction import redact_payload
from agentops.persistence import (
    ActionRepository,
    AgentRunRepository,
    ApprovalDecisionRepository,
    ApprovalRepository,
    AuditEventRow,
    CheckpointRepository,
    Database,
    EntityNotFound,
    EnvironmentReservationFencedError,
    EnvironmentReservationHeldError,
    EnvironmentReservationRepository,
    HypothesisLineageError,
    HypothesisLineageRepository,
    HypothesisVersionRow,
    IncidentRepository,
    ObservationRepository,
    PersistenceError,
    RunBudgetRepository,
    RunLeaseFencedError,
    RunLeaseRepository,
)
from agentops.simulator import (
    PersistentSimulator,
    ScenarioName,
    SimulationError,
    Simulator,
)
from agentops.tools import (
    Actor,
    ApprovalRequired,
    ForbiddenAction,
    PermissionDenied,
    Role,
    ToolError,
    ToolGateway,
)
from agentops.workflows.incident_triage.state import (
    TRIAGE_NODES,
    IncidentAgentState,
    checkpoint_shape_reason,
    run_contract_reason,
)


def _observed_stable_rollback_targets(observations: object, service: object) -> set[str]:
    if not isinstance(observations, list) or not isinstance(service, str) or not service:
        return set()

    latest: dict[str, dict[str, Any]] = {}
    for item in reversed(observations):
        if not isinstance(item, dict):
            continue
        tool_name = item.get("tool_name")
        payload = item.get("payload")
        if (
            isinstance(tool_name, str)
            and tool_name in {"get_service_status", "get_deployment_history"}
            and tool_name not in latest
            and isinstance(payload, dict)
            and payload.get("service") == service
        ):
            latest[tool_name] = payload

    status = latest.get("get_service_status")
    history = latest.get("get_deployment_history")
    if not status or not history:
        return set()
    current_version = status.get("version")
    deployments = history.get("deployments")
    if (
        not isinstance(current_version, str)
        or not current_version
        or not isinstance(deployments, list)
    ):
        return set()

    return {
        deployment["version"]
        for deployment in deployments
        if isinstance(deployment, dict)
        and isinstance(deployment.get("version"), str)
        and deployment["version"]
        and deployment["version"] != current_version
        and deployment.get("status") == "stable"
    }


class WorkflowResult(BaseModel):
    incident: Incident
    run: AgentRun
    state: dict[str, Any]


@dataclass
class JournalSpan:
    attempt_id: UUID | None = None
    facts: dict[str, Any] = field(default_factory=dict)


class WorkflowBudgetExceeded(RuntimeError):
    """Raised before an expensive attempt when its run budget is exhausted."""

    def __init__(self, message: str, *, code: str) -> None:
        super().__init__(message)
        self.code = code


class IncidentTriageRuntime:
    """Compile and run one bounded investigation graph against a simulator."""

    def __init__(
        self,
        simulator: Simulator | PersistentSimulator,
        gateway: ToolGateway,
        llm: LLMClient | None = None,
        *,
        model_config: ModelConfig | None = None,
        max_steps: int = 28,
        max_llm_calls: int = 16,
        max_tool_calls: int = 20,
        max_active_seconds: float = 90.0,
        clock: Callable[[], float] | None = None,
        fault_injector: Callable[[str], None] | None = None,
        database: Database | None = None,
        checkpoint_path: str | Path | None = None,
        checkpoint_backend: str | None = None,
        auto_create_schema: bool = True,
        lease_owner_id: str | None = None,
        lease_ttl_seconds: float = 30.0,
        lease_clock: Callable[[], datetime] | None = None,
        retry_sleep: Callable[[float], None] | None = None,
    ) -> None:
        if max_steps < 1:
            raise ValueError("max_steps must be positive")
        if max_llm_calls < 1:
            raise ValueError("max_llm_calls must be positive")
        if max_tool_calls < 1:
            raise ValueError("max_tool_calls must be positive")
        if max_active_seconds <= 0:
            raise ValueError("max_active_seconds must be positive")
        if lease_ttl_seconds <= 0:
            raise ValueError("lease_ttl_seconds must be positive")
        if lease_owner_id is not None and not lease_owner_id:
            raise ValueError("lease_owner_id must be non-empty")
        self.simulator = simulator
        self.gateway = gateway
        self.llm = llm or FakeLLMClient()
        self.model_config = (model_config or ModelConfig()).model_copy(deep=True)
        self.max_steps = max_steps
        self.max_llm_calls = max_llm_calls
        self.max_tool_calls = max_tool_calls
        self.max_active_seconds = max_active_seconds
        self._clock = clock or monotonic
        self._fault_injector = fault_injector
        self._active_started_at: float | None = None
        self._active_started_wall: datetime | None = None
        self._active_elapsed_seconds = 0.0
        self.database = database
        self._lease_owner_id = lease_owner_id or f"runtime-{uuid4()}"
        self._lease_ttl_seconds = lease_ttl_seconds
        self._lease_clock = lease_clock or utc_now
        self._retry_sleep = retry_sleep or sleep
        self._lease_generation: int | None = None
        self._lease_active = False
        self._lease_guard_bound = False
        self._reservation_environment_id: UUID | None = None
        self._reservation_generation: int | None = None
        self._reservation_active = False
        self._journal_step: ContextVar[UUID | None] = ContextVar("journal_step", default=None)
        self._last_tool_attempt_id: str | None = None
        self._last_model_decision_id: UUID | None = None
        self.model_usage: list[dict[str, Any]] = []
        self.tool_attempts: list[str] = []
        if checkpoint_backend not in {None, "sqlite", "postgres", "memory"}:
            raise CheckpointCompatibilityError("unsupported checkpoint backend")
        if checkpoint_backend == "postgres" and (database is None or checkpoint_path is not None):
            raise CheckpointCompatibilityError(
                "PostgreSQL checkpoints require a database and no SQLite checkpoint path"
            )
        if checkpoint_backend == "postgres" and database.engine.dialect.name != "postgresql":
            raise CheckpointCompatibilityError(
                "PostgreSQL checkpoints require a PostgreSQL database"
            )
        if checkpoint_backend == "sqlite" and checkpoint_path is None:
            raise CheckpointCompatibilityError("SQLite checkpoints require an explicit path")
        if checkpoint_backend == "memory" and checkpoint_path is not None:
            raise CheckpointCompatibilityError("memory checkpoints do not accept a SQLite path")
        if checkpoint_path is not None:
            self._validate_checkpoint_path(checkpoint_path)
        if self.database is not None and auto_create_schema:
            self.database.create_schema()
        self._checkpoint_backend = checkpoint_backend
        self._checkpoint_store = (
            SqliteCheckpointStore(
                checkpoint_path,
                write_guard=self._checkpoint_write_guard if self.database is not None else None,
            )
            if checkpoint_path is not None
            else None
        )
        if checkpoint_backend == "postgres":
            from agentops.checkpoint_postgres import PostgresCheckpointStore

            self._checkpoint_store = PostgresCheckpointStore(
                database.engine.url, write_guard=self._checkpoint_write_guard
            )
        self._incident: Incident | None = None
        self._run: AgentRun | None = None
        self._action: Action | None = None
        self._approval: ApprovalRequest | None = None
        self.pending_approval_payload: dict[str, Any] | None = None
        self._llm_attempts = 0
        self._durable_llm_calls = 0
        self._tool_attempts = 0
        self.last_graph: Any | None = None
        self.last_config: dict[str, Any] | None = None

    def compile(self, *, checkpointer: Any | None = None):
        builder = StateGraph(IncidentAgentState)
        for name in TRIAGE_NODES:
            builder.add_node(name, self._journal_node(name, getattr(self, f"_{name}")))
        builder.add_edge(START, "load_incident")
        builder.add_edge("load_incident", "triage")
        builder.add_conditional_edges(
            "triage",
            self._route_after_triage,
            {"create_plan": "create_plan", "end": END},
        )
        builder.add_edge("create_plan", "select_next_diagnostic_step")
        builder.add_edge("select_next_diagnostic_step", "execute_read_tool")
        builder.add_edge("execute_read_tool", "record_observation")
        builder.add_edge("record_observation", "update_hypotheses")
        builder.add_conditional_edges(
            "update_hypotheses",
            self._route_after_hypotheses,
            {
                "select_next_diagnostic_step": "select_next_diagnostic_step",
                "propose_remediation": "propose_remediation",
                "end": END,
            },
        )
        builder.add_edge("propose_remediation", "policy_gate")
        builder.add_conditional_edges(
            "policy_gate",
            self._route_after_policy,
            {
                "human_approval_interrupt": "human_approval_interrupt",
                "execute_safe_action": "execute_safe_action",
                "verify_environment": "verify_environment",
                "end": END,
            },
        )
        builder.add_conditional_edges(
            "human_approval_interrupt",
            self._route_after_approval,
            {
                "execute_safe_action": "execute_safe_action",
                "end": END,
            },
        )
        builder.add_conditional_edges(
            "execute_safe_action",
            self._route_after_execution,
            {"verify_environment": "verify_environment", "end": END},
        )
        builder.add_edge("verify_environment", "review")
        builder.add_edge("review", END)
        selected_checkpointer = checkpointer
        if selected_checkpointer is None and self._checkpoint_store is not None:
            selected_checkpointer = self._checkpoint_store.saver
        if selected_checkpointer is None:
            selected_checkpointer = InMemorySaver()
        own_saver = self._checkpoint_store.saver if self._checkpoint_store is not None else None
        if self._checkpoint_backend == "postgres" and selected_checkpointer is not own_saver:
            raise CheckpointCompatibilityError("PostgreSQL checkpoint saver cannot be overridden")
        if self.database is not None and selected_checkpointer is not own_saver:
            # A foreign guard is not evidence of this runtime's ownership.
            # Reject it rather than silently discarding its policy or nesting
            # domain transactions that can deadlock.
            if isinstance(selected_checkpointer, FencedCheckpointSaver):
                raise CheckpointCompatibilityError("foreign fenced checkpoint saver is unsupported")
            if isinstance(selected_checkpointer, SqliteSaver):
                for _, _, path in selected_checkpointer.conn.execute("PRAGMA database_list"):
                    if path:
                        self._validate_checkpoint_path(path)
            selected_checkpointer = FencedCheckpointSaver(
                selected_checkpointer, self._checkpoint_write_guard
            )
        return builder.compile(checkpointer=selected_checkpointer)

    @contextmanager
    def _journal_attempt(
        self, kind: str, name: str, state: dict, *, arguments=None
    ) -> Iterator[JournalSpan]:
        span = JournalSpan()
        if self.database is None or self._run is None:
            yield span
            return
        if arguments is not None:
            span.facts["arguments_fingerprint"] = stable_parameter_hash(redact_payload(arguments))
        with self.database.session() as session:
            self._renew_lease(session)
            start = RunJournalRepository(session).start(
                run_id=self._run.id,
                kind=kind,
                name=name,
                logical_step=int(state.get("step_count", 0)) + 1,
                step_id=self._journal_step.get(),
                facts=span.facts,
                action_id=self._action.id if self._action else None,
                approval_id=self._approval.id if self._approval else None,
            )
        span.attempt_id = UUID(start.attempt_id)
        token = self._journal_step.set(span.attempt_id) if kind == "node" else None
        started_at = monotonic()
        phase = "completed"
        try:
            yield span
        except RunLeaseFencedError:
            # A stale owner cannot even append an outcome observation.
            phase = "failed"
            span.facts["error_code"] = "RunLeaseFencedError"
            raise
        except BaseException as exc:
            phase = (
                "interrupted"
                if isinstance(exc, GraphInterrupt) or not isinstance(exc, Exception)
                else "failed"
            )
            span.facts["error_code"] = (
                "RuntimeInterrupted" if phase == "interrupted" else safe_error(exc)
            )
            # LangGraph can persist task exceptions in checkpoint writes. Clean
            # provider/transport error text before it reaches that boundary too.
            if isinstance(exc, LLMProviderError):
                raise LLMProviderError(
                    safe_error(exc), exc.public_message, attempts=exc.attempts
                ) from None
            if isinstance(exc, TimeoutError):
                raise TimeoutError("Workflow boundary timed out") from None
            if isinstance(exc, ConnectionError):
                raise ConnectionError("Workflow boundary is unavailable") from None
            raise
        finally:
            if token is not None:
                self._journal_step.reset(token)
            if phase == "completed" and span.facts.get("error_code"):
                phase = "failed"
            # Renewing the same existing fence also rejects takeover during I/O.
            with self.database.session() as session:
                self._renew_lease(session)
                RunJournalRepository(session).finish(
                    start,
                    phase=phase,
                    duration_ms=(monotonic() - started_at) * 1000,
                    facts=span.facts,
                    action_id=self._action.id if self._action else None,
                    approval_id=self._approval.id if self._approval else None,
                )

    def _journal_node(self, name: str, operation: Callable):
        def invoke(state: IncidentAgentState) -> dict[str, Any]:
            with self._journal_attempt("node", name, state) as span:
                result = operation(state)
                span.facts.update(
                    {
                        "outcome": result.get("final_outcome"),
                        "error_code": result.get("last_error_code"),
                        "diagnosis_category": (result.get("diagnosis") or {}).get("category"),
                        "approval_status": self._approval.status.value if self._approval else None,
                    }
                )
                hypotheses = result.get("hypotheses")
                if isinstance(hypotheses, list):
                    span.facts.update(
                        hypothesis_count=len(hypotheses),
                        hypothesis_scores=[
                            h.get("confidence") for h in hypotheses if isinstance(h, dict)
                        ],
                    )
                if result.get("last_error_code"):
                    span.facts["outcome"] = (
                        "tool_error" if not result.get("final_outcome") else result["final_outcome"]
                    )
                return result

        return invoke

    def _call_tool(self, state, tool_name, arguments, actor, **kwargs):
        self.tool_attempts.append(tool_name)
        with self._journal_attempt("tool", tool_name, state, arguments=arguments) as span:
            result = self.gateway.call(tool_name, arguments, actor, **kwargs)
            self._last_tool_attempt_id = str(span.attempt_id) if span.attempt_id else None
            if tool_name == "verify_service":
                payload = result.payload.model_dump()
                span.facts["verification_passed"] = (
                    payload.get("healthy") is True
                    and bool(payload.get("checks"))
                    and all(x is True for x in payload["checks"].values())
                )
            return result

    def _call_model(self, task, messages, response_model, config, state):
        from agentops.usage import usage_facts

        self._last_model_decision_id = None
        with self._journal_attempt("llm", task, state) as span:
            # An unanswered/failed attempt remains unknown, never zero tokens.
            sample: dict[str, Any] = {"usage_version": 1}
            self.model_usage.append(sample)
            try:
                result = self.llm.generate_structured(
                    task=task, messages=messages, response_model=response_model, config=config
                )
            except ValidationError:
                raise LLMOutputError("structured provider output failed validation") from None
            except LLMProviderError as exc:
                sample.update(usage_facts(getattr(exc, "usage", {})))
                span.facts.update(sample)
                raise
            sample.update(
                usage_facts(
                    {
                        "input_tokens": getattr(result, "input_tokens", None),
                        "output_tokens": getattr(result, "output_tokens", None),
                        "estimated_cost_usd": getattr(result, "estimated_cost_usd", None),
                        "cost_source": getattr(result, "cost_source", None),
                    }
                )
            )
            # Keep billed response metadata even if semantic validation fails.
            span.facts.update(sample)
            if getattr(result, "attempts", 1) != 1:
                raise LLMProviderError(
                    "LLMProviderContract",
                    "provider adapter performed hidden retries outside the workflow budget",
                )
            try:
                output_size = len(result.output.model_dump_json())
            except AttributeError as exc:
                raise LLMProviderError(
                    "LLMOutputInvalid", "provider output is not structured"
                ) from exc
            if output_size > self.model_config.max_output_chars:
                raise LLMProviderError(
                    "LLMOutputTooLarge", "structured output exceeds the configured character limit"
                )
            try:
                self._validate_semantics(task, result.output, state)
            except LLMProviderError as exc:
                # Semantic rejection follows a completed response and its usage stays charged.
                exc.attempts = result.attempts
                raise
            if span.attempt_id is not None:
                self._last_model_decision_id = uuid5(span.attempt_id, "completed")
            return result

    def _validate_checkpoint_path(self, checkpoint_path: str | Path) -> None:
        if self.database is None or self.database.engine.dialect.name != "sqlite":
            return
        domain_path = self.database.engine.url.database
        if domain_path and domain_path != ":memory:":
            domain_file, checkpoint_file = Path(domain_path), Path(checkpoint_path)
            same_file = domain_file.resolve() == checkpoint_file.resolve()
            if domain_file.exists() and checkpoint_file.exists():
                same_file = same_file or domain_file.samefile(checkpoint_file)
            if same_file:
                raise CheckpointCompatibilityError(
                    "domain data and checkpoints require separate SQLite files"
                )

    @contextmanager
    def _checkpoint_write_guard(self, config: RunnableConfig) -> Iterator[Session | None]:
        if self.database is None:
            yield
            return
        if (
            self._run is None
            or config.get("configurable", {}).get("thread_id") != self._run.thread_id
        ):
            raise CheckpointCompatibilityError("checkpoint write is bound to another run thread")
        with self.database.session() as session:
            # PostgreSQL writes use this very transaction, so losing the lease
            # connection cannot leave a separately committing checkpoint writer.
            # SQLite keeps its existing separate-store guard semantics.
            self._renew_lease(session)
            yield session

    def _acquire_reservation(self) -> None:
        if self.database is None or not isinstance(self.simulator, PersistentSimulator):
            return
        if self._run is None:
            raise RuntimeError("run context is incomplete")
        identity = self.simulator.environment_identity
        if self._reservation_active:
            if (
                self._reservation_environment_id != identity.environment_id
                or self._reservation_generation != identity.generation
            ):
                raise EnvironmentReservationHeldError(
                    "runtime already owns a different environment reservation"
                )
            self._assert_reservation()
            return
        with self.database.session() as session:
            reservation = EnvironmentReservationRepository(session).acquire(
                identity.environment_id,
                self._run.id,
                generation=identity.generation,
                now=self._lease_clock(),
            )
        self._reservation_environment_id = identity.environment_id
        self._reservation_generation = reservation.generation
        self._reservation_active = True

    def _assert_reservation(self, session: Session | None = None) -> None:
        if self.database is None or not isinstance(self.simulator, PersistentSimulator):
            return
        if (
            self._run is None
            or self._reservation_environment_id is None
            or self._reservation_generation is None
            or not self._reservation_active
        ):
            raise EnvironmentReservationFencedError(
                "runtime does not hold an active environment reservation"
            )
        if session is None:
            with self.database.session() as owned_session:
                EnvironmentReservationRepository(owned_session).assert_owner(
                    self._reservation_environment_id,
                    self._run.id,
                    generation=self._reservation_generation,
                )
        else:
            EnvironmentReservationRepository(session).assert_owner(
                self._reservation_environment_id,
                self._run.id,
                generation=self._reservation_generation,
            )

    def _release_reservation(self) -> None:
        if (
            self.database is None
            or self._run is None
            or self._reservation_environment_id is None
            or not self._reservation_active
        ):
            return
        try:
            if self._reservation_generation is not None:
                with self.database.session() as session:
                    EnvironmentReservationRepository(session).release(
                        self._reservation_environment_id,
                        self._run.id,
                        generation=self._reservation_generation,
                        now=self._lease_clock(),
                    )
        except EnvironmentReservationFencedError:
            # A different run must never have its reservation released by a stale worker.
            pass
        finally:
            self._reservation_active = False

    def _maybe_release_reservation(self, result: WorkflowResult) -> None:
        if result.run.status in {
            AgentRunStatus.SUCCEEDED,
            AgentRunStatus.FAILED,
            AgentRunStatus.TIMED_OUT,
            AgentRunStatus.CANCELLED,
        }:
            self._assert_lease()
            self._release_reservation()

    def _acquire_lease(self) -> None:
        if self.database is None or self._run is None:
            return
        with self.database.session() as session:
            lease = RunLeaseRepository(session).acquire(
                self._run.id,
                owner_id=self._lease_owner_id,
                now=self._lease_clock(),
                ttl_seconds=self._lease_ttl_seconds,
            )
        self._lease_generation = lease.generation
        self._lease_active = True
        if isinstance(self.simulator, PersistentSimulator) and not self._lease_guard_bound:
            self.simulator = self.simulator.with_mutation_guard(
                self._assert_mutation_allowed
            ).with_transaction_guard(self._assert_mutation_transaction)
            self.gateway.simulator = self.simulator
            self._lease_guard_bound = True

    def _assert_lease(self) -> None:
        if self.database is None:
            return
        if self._run is None or self._lease_generation is None or not self._lease_active:
            raise RunLeaseFencedError("runtime does not hold an active run lease")
        self._assert_reservation()
        with self.database.session() as session:
            RunLeaseRepository(session).assert_owner(
                self._run.id,
                owner_id=self._lease_owner_id,
                generation=self._lease_generation,
                now=self._lease_clock(),
            )

    def _assert_cancellation_clear(self, session: Session | None = None) -> None:
        if self.database is None or self._run is None:
            return
        if session is None:
            with self.database.session() as owned_session:
                durable = AgentRunRepository(owned_session).require(self._run.id)
        else:
            durable = AgentRunRepository(session).require_for_update(self._run.id)
        if durable.cancel_requested_at is not None or durable.status is AgentRunStatus.CANCELLED:
            # Discard uncommitted terminal transitions before producing the
            # cancellation result; committed domain state is authoritative.
            object.__setattr__(self._run, "status", durable.status)
            self._run.cancel_requested_at = durable.cancel_requested_at
            if session is not None:
                self._incident = IncidentRepository(session).require(durable.incident_id)
            else:
                with self.database.session() as owned_session:
                    self._incident = IncidentRepository(owned_session).require(durable.incident_id)
            raise CancellationRequested(f"run {self._run.id} cancellation was requested")

    def _assert_mutation_transaction(self, session: Session) -> None:
        """Fence ownership and cancellation inside the backend commit transaction."""

        self._renew_lease(session)
        self._assert_cancellation_clear(session)

    def _assert_mutation_allowed(self) -> None:
        self._assert_lease()
        self._assert_cancellation_clear()

    def _renew_lease(self, session: Session | None = None) -> None:
        if self.database is None:
            return
        if self._run is None or self._lease_generation is None or not self._lease_active:
            raise RunLeaseFencedError("runtime does not hold an active run lease")
        # Reservation ownership remains a separate fence. The lease repository
        # renews by owner/generation even after the previous expiry, allowing a
        # slow but still-current worker to refresh itself before the next write.
        self._assert_reservation(session)
        if session is None:
            with self.database.session() as owned_session:
                RunLeaseRepository(owned_session).renew(
                    self._run.id,
                    owner_id=self._lease_owner_id,
                    generation=self._lease_generation,
                    now=self._lease_clock(),
                    ttl_seconds=self._lease_ttl_seconds,
                )
        else:
            RunLeaseRepository(session).renew(
                self._run.id,
                owner_id=self._lease_owner_id,
                generation=self._lease_generation,
                now=self._lease_clock(),
                ttl_seconds=self._lease_ttl_seconds,
            )

    def _release_lease(self) -> None:
        if self.database is None or self._run is None or not self._lease_active:
            return
        try:
            if self._lease_generation is not None:
                with self.database.session() as session:
                    RunLeaseRepository(session).release(
                        self._run.id,
                        owner_id=self._lease_owner_id,
                        generation=self._lease_generation,
                        now=self._lease_clock(),
                    )
        except RunLeaseFencedError:
            # A takeover already fenced this runtime; never overwrite the new owner.
            pass
        finally:
            self._lease_active = False

    def _recover_active_budget(self) -> None:
        if self.database is None or self._run is None:
            return
        with self.database.session() as session:
            # Keep the lease refresh and recovered budget write in one
            # transaction so a takeover cannot land between the fence and write.
            self._renew_lease(session)
            budget = RunBudgetRepository(session).recover_active_segment(
                self._run.id,
                now=self._lease_clock(),
            )
            self._active_elapsed_seconds = budget.active_elapsed_seconds
            self._tool_attempts = max(self._tool_attempts, budget.tool_calls_used)
            self._active_started_wall = None

    def close(self) -> None:
        """Release a durable checkpoint connection owned by this runtime."""

        self._release_lease()
        self.gateway.close()
        if self._checkpoint_store is not None:
            self._checkpoint_store.close()
            self._checkpoint_store = None

    @property
    def current_action(self) -> Action | None:
        """Expose the current persisted action for read-only evidence projection."""

        return self._action

    @property
    def current_approval(self) -> ApprovalRequest | None:
        """Expose the current approval for read-only evidence projection."""

        return self._approval

    @property
    def active_elapsed_seconds(self) -> float:
        """Return measured workflow-active time, excluding approval wait."""

        return self._active_elapsed_seconds

    def run_scenario(self, scenario: ScenarioName | str) -> WorkflowResult:
        self.model_usage.clear()
        self.tool_attempts.clear()
        scenario_name = scenario if isinstance(scenario, ScenarioName) else ScenarioName(scenario)
        if isinstance(self.simulator, PersistentSimulator):
            if self.simulator.loaded_scenario is not scenario_name:
                raise SimulationError(
                    f"persisted environment contains {self.simulator.loaded_scenario.value}, "
                    f"not requested {scenario_name.value}"
                )
        else:
            self.simulator.load(scenario_name)
        context = self.simulator.incident_context()
        self._incident = Incident(
            title=context["title"],
            description=context["description"],
            service_name=context["service_name"],
            severity=Severity.SEV2,
            created_by="demo-user",
        )
        self._run = AgentRun(
            incident_id=self._incident.id,
            model_configuration=model_configuration_identity(self.llm, self.model_config),
            environment_id=(
                self.simulator.environment_identity.environment_id
                if isinstance(self.simulator, PersistentSimulator)
                else None
            ),
        )
        self._action = None
        self._approval = None
        self.pending_approval_payload = None
        self._llm_attempts = 0
        self._durable_llm_calls = 0
        self._tool_attempts = 0
        self._active_elapsed_seconds = 0.0
        self._active_started_at = None
        self._active_started_wall = None
        self._run.transition_to(AgentRunStatus.QUEUED)
        self._persist_entities(check_lease=False)
        initial = self._build_initial_state(context)
        config = {
            "configurable": {"thread_id": self._run.thread_id},
            "recursion_limit": self.max_steps + 10,
        }
        self._acquire_reservation()
        self._acquire_lease()
        try:
            result = self._invoke_new_run(initial, config)
            self._maybe_release_reservation(result)
            return result
        finally:
            self._release_lease()

    def _build_initial_state(self, context: dict[str, str]) -> IncidentAgentState:
        if self._incident is None or self._run is None:
            raise RuntimeError("workflow context is incomplete")
        return {
            "incident_id": str(self._incident.id),
            "environment_id": str(self.simulator.environment_identity.environment_id),
            "run_id": str(self._run.id),
            "workflow_version": self._run.workflow_version,
            "workflow_name": self._run.workflow_name,
            "state_schema_version": self._run.state_schema_version,
            "model_configuration_id": self._run.model_configuration_id,
            "phase": "load",
            "title": self._incident.title,
            "description": self._incident.description,
            "service_name": self._incident.service_name,
            "severity": self._incident.severity.value,
            "current_plan_index": 0,
            "observations": [],
            "hypotheses": [],
            "hypothesis_ids": [],
            "diagnosis": None,
            "retry_count": 0,
            "step_count": 0,
            "max_steps": self.max_steps,
            "max_llm_calls": self.max_llm_calls,
            "max_tool_calls": self.max_tool_calls,
            "max_active_seconds": self.max_active_seconds,
            "llm_call_count": 0,
            "tool_call_count": 0,
            "policy_denial_count": 0,
            "last_error_code": None,
            "last_error_message": None,
        }

    def run_existing(
        self,
        *,
        run_id: UUID | str,
        environment_id: UUID | str,
        scenario: ScenarioName | str | None = None,
    ) -> WorkflowResult:
        """Start a queued durable run using its persisted incident and environment."""

        if self.database is None:
            raise RuntimeError("database is required for a durable run")
        with self.database.session() as session:
            self._run = AgentRunRepository(session).require(run_id)
            self._incident = IncidentRepository(session).require(self._run.incident_id)
        if self._run.environment_id is None or self._run.environment_id != UUID(
            str(environment_id)
        ):
            raise RuntimeError("run is not bound to the requested environment")
        if self._run.status in {
            AgentRunStatus.WAITING_FOR_APPROVAL,
            AgentRunStatus.SUCCEEDED,
            AgentRunStatus.FAILED,
            AgentRunStatus.BLOCKED,
            AgentRunStatus.CANCELLED,
            AgentRunStatus.TIMED_OUT,
        }:
            # Runtime completion and command acknowledgement are separate commits.
            # Redelivery returns the committed outcome without touching an environment
            # that may already belong to the next run.
            return WorkflowResult(
                incident=self._incident,
                run=self._run,
                state={
                    "phase": "durable_replay",
                    "replayed": True,
                    "final_outcome": "approval_required"
                    if self._run.status is AgentRunStatus.WAITING_FOR_APPROVAL
                    else self._run.status.value.lower(),
                    "step_count": self._run.step_count,
                    "llm_call_count": self._run.llm_call_count,
                    "tool_call_count": self._run.tool_call_count,
                    "last_error_code": self._run.last_error_code,
                    "last_error_message": self._run.last_error_message,
                },
            )
        if not isinstance(self.simulator, PersistentSimulator):
            self.simulator = PersistentSimulator(self.database, environment_id)
            self.gateway.simulator = self.simulator
            self._lease_guard_bound = False
        elif self.simulator.environment_identity.environment_id != UUID(str(environment_id)):
            raise RuntimeError("runtime simulator is bound to another environment")
        with self.database.session() as session:
            self._run = AgentRunRepository(session).require(run_id)
            self._incident = IncidentRepository(session).require(self._run.incident_id)
        starting_status = self._run.status
        if starting_status not in {AgentRunStatus.QUEUED, AgentRunStatus.RUNNING}:
            raise RuntimeError(f"run {self._run.id} is not startable")
        if starting_status is AgentRunStatus.RUNNING and self._checkpoint_store is None:
            raise CheckpointCompatibilityError(
                "a durable checkpoint store is required to recover a RUNNING run"
            )
        bound_environment = self._run.environment_id
        if bound_environment is None or bound_environment != UUID(str(environment_id)):
            raise RuntimeError("run is not bound to the requested environment")
        loaded_scenario = self.simulator.loaded_scenario
        if scenario is not None:
            requested_scenario = (
                scenario if isinstance(scenario, ScenarioName) else ScenarioName(scenario)
            )
            if loaded_scenario is not requested_scenario:
                raise SimulationError(
                    f"persisted environment contains {loaded_scenario.value}, "
                    f"not requested {requested_scenario.value}"
                )
        context = self.simulator.incident_context()
        self._action = None
        self._approval = None
        self.pending_approval_payload = None
        self._llm_attempts = 0
        self._durable_llm_calls = 0
        self._tool_attempts = 0
        self._active_elapsed_seconds = 0.0
        self._active_started_at = None
        self._active_started_wall = None
        initial = self._build_initial_state(context)
        config = {
            "configurable": {"thread_id": self._run.thread_id},
            "recursion_limit": self.max_steps + 10,
        }
        self._acquire_reservation()
        self._acquire_lease()
        try:
            if self._run.cancel_requested_at is not None:
                result = self._cancellation_result(initial)
                self._maybe_release_reservation(result)
                return result
            result = self._invoke_new_run(
                initial,
                config,
                recover=starting_status is AgentRunStatus.RUNNING,
            )
            self._maybe_release_reservation(result)
            return result
        finally:
            self._release_lease()

    def _invoke_new_run(
        self,
        initial: IncidentAgentState,
        config: dict[str, Any],
        *,
        recover: bool = False,
    ) -> WorkflowResult:
        if self._run is None:
            raise RuntimeError("workflow run is not initialized")
        self._ensure_runtime_contract()
        self._ensure_model_configuration(allow_bind=not recover)
        # run_existing builds initial state before the lease-protected binding.
        initial["model_configuration_id"] = self._run.model_configuration_id
        if self._run.status is AgentRunStatus.QUEUED:
            self._run.transition_to(AgentRunStatus.RUNNING)
            self._persist_entities()
        elif self._run.status is not AgentRunStatus.RUNNING:
            raise RuntimeError(f"run {self._run.id} cannot execute from {self._run.status.value}")
        self.last_graph = self.compile()
        self.last_config = config
        graph_input: IncidentAgentState | None = initial
        if recover:
            self._recover_active_budget()
            snapshot = self._checkpoint_snapshot()
            self._validate_checkpoint_compatibility(snapshot, require_metadata=False)
            self._llm_attempts = max(
                snapshot.values["llm_call_count"],
                self._durable_llm_calls,
            )
            graph_input = None
        self._start_active_segment()
        try:
            raw_state = self.last_graph.invoke(
                graph_input, config, durability="sync" if self.database is not None else None
            )
        except RunLeaseFencedError:
            raise
        except CancellationRequested as exc:
            self._pause_active_segment()
            return self._cancellation_result(initial, exc)
        except WorkflowBudgetExceeded as exc:
            self._pause_active_segment()
            return self._workflow_error_result(initial, exc.code, exc)
        except LLMProviderError as exc:
            self._pause_active_segment()
            return self._workflow_error_result(initial, exc.code, exc)
        except ValidationError as exc:
            self._pause_active_segment()
            return self._workflow_error_result(initial, "LLMValidationError", exc)
        except Exception as exc:
            self._pause_active_segment()
            budget_error = self._find_budget_error(exc)
            if budget_error is not None:
                return self._workflow_error_result(initial, budget_error.code, budget_error)
            provider_error = self._find_provider_error(exc)
            if provider_error is not None:
                return self._workflow_error_result(initial, provider_error.code, provider_error)
            return self._workflow_error_result(initial, "WorkflowError", exc)
        self._pause_active_segment()
        state = self._clean_state(raw_state)
        outcome = state.get("final_outcome")
        if "__interrupt__" in raw_state:
            state["final_outcome"] = "approval_required"
            self._run.transition_to(AgentRunStatus.WAITING_FOR_APPROVAL)
        elif outcome in {"escalated", "step_limit_exceeded", "tool_error", "budget_exceeded"}:
            self._run.transition_to(AgentRunStatus.BLOCKED)
        else:
            self._run.transition_to(AgentRunStatus.SUCCEEDED)
        self._run.step_count = state.get("step_count", 0)
        self._run.llm_call_count = state.get("llm_call_count", 0)
        self._run.tool_call_count = state.get("tool_call_count", 0)
        self._run.last_error_code = state.get("last_error_code")
        self._run.last_error_message = state.get("last_error_message")
        try:
            self._persist_entities()
        except CancellationRequested as exc:
            return self._cancellation_result(state, exc)
        self._persist_checkpoint_metadata(state)
        return WorkflowResult(incident=self._incident, run=self._run, state=state)

    def resume_approval(
        self,
        *,
        actor: Actor,
        approved: bool,
        approved_parameters: dict[str, Any] | None = None,
        reason: str | None = None,
        run_id: UUID | str | None = None,
    ) -> WorkflowResult:
        """Resume the paused thread after a role-checked human decision."""

        if actor.role not in {Role.APPROVER, Role.ADMIN}:
            raise PermissionDenied("only approver or admin may decide an approval")
        if run_id is not None:
            requested_run_id = UUID(str(run_id))
            if self._run is None or self._run.id != requested_run_id:
                if self.database is None:
                    raise RuntimeError("run_id does not match the pending approval context")
                self._hydrate_pending_context(requested_run_id)
        if self._approval is None:
            raise RuntimeError("no pending approval to resume")
        if self._run is not None and self._run.status is AgentRunStatus.CANCELLED:
            return self._cancellation_result(
                self._build_initial_state(self.simulator.incident_context()),
                check_lease=False,
            )
        self._acquire_reservation()
        self._acquire_lease()
        try:
            try:
                self._assert_cancellation_clear()
            except CancellationRequested as exc:
                result = self._cancellation_result(
                    self._build_initial_state(self.simulator.incident_context()),
                    exc,
                )
                self._maybe_release_reservation(result)
                return result
            self._recover_active_budget()
            result = self._resume_approval_after_lease(
                actor=actor,
                approved=approved,
                approved_parameters=approved_parameters,
                reason=reason,
            )
            self._maybe_release_reservation(result)
            return result
        finally:
            self._release_lease()

    def _resume_approval_after_lease(
        self,
        *,
        actor: Actor,
        approved: bool,
        approved_parameters: dict[str, Any] | None,
        reason: str | None,
    ) -> WorkflowResult:
        self._ensure_runtime_contract()
        self._ensure_model_configuration(allow_bind=False)
        if self.last_graph is None:
            if self._run is None:
                raise RuntimeError("workflow context is incomplete")
            self.last_graph = self.compile()
            self.last_config = {
                "configurable": {"thread_id": self._run.thread_id},
                "recursion_limit": self.max_steps + 10,
            }
        if self.last_config is None:
            raise RuntimeError("workflow checkpoint configuration is missing")
        self._validate_checkpoint_compatibility()
        recovered = self._recover_terminal_checkpoint(
            actor=actor,
            approved=approved,
            approved_parameters=approved_parameters,
            reason=reason,
        )
        if recovered is not None:
            return recovered
        decision_id: UUID | None = None
        if self.database is not None:
            # Loading/validating a checkpoint can outlast the lease. Renew the
            # exact generation before decision admission; a takeover still fences
            # us, while elapsed read time alone does not strand a current owner.
            self._renew_lease()
            committed = ApprovalDecisionService(self.database).decide(
                self._approval.id,
                actor_id=actor.actor_id,
                actor_role=actor.role.value,
                approved=approved,
                approved_parameters=approved_parameters,
                reason=reason,
            )
            self._approval = committed.approval
            decision_id = committed.decision_id
            decision = {"decision_id": str(decision_id)}
            self._persist_checkpoint_metadata(
                self.last_graph.get_state(self.last_config).values,
                decision_id=decision_id,
                checkpoint_kind="approval_decision_committed",
            )
            self._inject_fault("after_approval_commit")
        else:
            if self._approval.status is not ApprovalStatus.PENDING:
                raise RuntimeError("approval has already been decided")
            decision = {
                "approved": approved,
                "approved_parameters": approved_parameters,
                "decided_by": actor.actor_id,
                "actor_role": actor.role.value,
                "reason": reason,
            }
        resume_context = dict(self.last_graph.get_state(self.last_config).values)
        checkpoint_llm_calls = int(resume_context.get("llm_call_count", 0))
        self._llm_attempts = max(checkpoint_llm_calls, self._durable_llm_calls)
        resume_context["llm_call_count"] = self._llm_attempts
        if (
            approved
            and self._run is not None
            and self._run.status is AgentRunStatus.WAITING_FOR_APPROVAL
        ):
            # The owning worker has validated checkpoint/configuration and the
            # exact human decision. API enqueue alone never performs this step.
            self._run.transition_to(AgentRunStatus.RUNNING)
            self._persist_entities()
            self._record_runtime_evidence(
                event_type="runtime.resumed",
                event_key_suffix=f"resume-admitted:{decision_id or self._approval.id}",
                payload={"schema_version": 1, "run_status": AgentRunStatus.RUNNING.value},
            )
        self._start_active_segment()
        try:
            raw_state = self.last_graph.invoke(
                Command(resume=decision),
                self.last_config,
                durability="sync" if self.database is not None else None,
            )
        except RunLeaseFencedError:
            raise
        except CancellationRequested as exc:
            self._pause_active_segment()
            return self._cancellation_result(resume_context, exc)
        except WorkflowBudgetExceeded as exc:
            self._pause_active_segment()
            return self._workflow_error_result(resume_context, exc.code, exc)
        except LLMProviderError as exc:
            self._pause_active_segment()
            return self._workflow_error_result(resume_context, exc.code, exc)
        except ValidationError as exc:
            self._pause_active_segment()
            return self._workflow_error_result(
                resume_context,
                "LLMValidationError",
                exc,
            )
        except Exception as exc:
            self._pause_active_segment()
            budget_error = self._find_budget_error(exc)
            if budget_error is not None:
                return self._workflow_error_result(
                    resume_context,
                    budget_error.code,
                    budget_error,
                )
            provider_error = self._find_provider_error(exc)
            if provider_error is not None:
                return self._workflow_error_result(
                    resume_context,
                    provider_error.code,
                    provider_error,
                )
            return self._workflow_error_result(resume_context, "WorkflowError", exc)
        self._pause_active_segment()
        state = self._clean_state(raw_state)
        self._persist_checkpoint_metadata(
            state,
            decision_id=decision_id,
            checkpoint_kind="workflow_resumed",
        )
        self._inject_fault("after_graph_checkpoint")
        return self._finalize_resume_state(
            state,
            decision_id=decision_id,
            checkpoint_kind="workflow_resumed",
        )

    def _recover_terminal_checkpoint(
        self,
        *,
        actor: Actor,
        approved: bool,
        approved_parameters: dict[str, Any] | None,
        reason: str | None,
    ) -> WorkflowResult | None:
        if self.database is None or self._checkpoint_store is None:
            return None
        snapshot = self._checkpoint_snapshot()
        if snapshot.values.get("final_outcome") != "resolved":
            return None
        if self._approval is None or self._run is None:
            raise CheckpointCompatibilityError("terminal checkpoint is missing approval context")
        with self.database.session() as session:
            metadata = CheckpointRepository(session).get_for_run(self._run.id)
            if metadata is None or metadata.decision_id is None:
                raise CheckpointCompatibilityError(
                    "terminal checkpoint has no committed approval decision"
                )
            decision_id = metadata.decision_id
        self._renew_lease()
        committed = ApprovalDecisionService(self.database).decide(
            self._approval.id,
            actor_id=actor.actor_id,
            actor_role=actor.role.value,
            approved=approved,
            approved_parameters=approved_parameters,
            reason=reason,
        )
        if str(committed.decision_id) != decision_id:
            raise CheckpointCompatibilityError("terminal checkpoint decision lineage changed")
        self._approval = committed.approval
        state = self._clean_state(dict(snapshot.values))
        return self._finalize_resume_state(
            state,
            decision_id=committed.decision_id,
            checkpoint_kind="workflow_recovered",
        )

    def _finalize_resume_state(
        self,
        state: dict[str, Any],
        *,
        decision_id: UUID | None,
        checkpoint_kind: str,
    ) -> WorkflowResult:
        outcome = state.get("final_outcome")
        if self._run is None or self._incident is None:
            raise RuntimeError("workflow context is incomplete")
        if outcome == "resolved":
            if self._run.status is AgentRunStatus.WAITING_FOR_APPROVAL:
                self._run.transition_to(AgentRunStatus.RUNNING)
                self._run.transition_to(AgentRunStatus.SUCCEEDED)
            elif self._run.status is AgentRunStatus.RUNNING:
                self._run.transition_to(AgentRunStatus.SUCCEEDED)
            elif self._run.status is not AgentRunStatus.SUCCEEDED:
                raise RuntimeError("resolved checkpoint cannot finalize the current run state")
        elif outcome in {"escalated", "tool_error", "step_limit_exceeded", "budget_exceeded"}:
            if self._run.status in {
                AgentRunStatus.WAITING_FOR_APPROVAL,
                AgentRunStatus.RUNNING,
            }:
                self._run.transition_to(AgentRunStatus.BLOCKED)
        self._run.step_count = state.get("step_count", 0)
        self._run.llm_call_count = state.get("llm_call_count", 0)
        self._run.tool_call_count = state.get("tool_call_count", 0)
        self._run.last_error_code = state.get("last_error_code")
        self._run.last_error_message = state.get("last_error_message")
        try:
            self._persist_entities()
        except CancellationRequested as exc:
            return self._cancellation_result(state, exc)
        self._persist_checkpoint_metadata(
            state,
            decision_id=decision_id,
            checkpoint_kind=checkpoint_kind,
        )
        return WorkflowResult(incident=self._incident, run=self._run, state=state)

    def _hydrate_pending_context(self, run_id: UUID | str) -> None:
        if self.database is None:
            raise RuntimeError("database is required to resume a persisted run")
        with self.database.session() as session:
            self._run = AgentRunRepository(session).require(run_id)
            self._incident = IncidentRepository(session).require(self._run.incident_id)
            metadata = CheckpointRepository(session).get_for_run(self._run.id)
            approval = (
                ApprovalRepository(session).require(metadata.approval_id)
                if metadata is not None and metadata.approval_id is not None
                else ApprovalRepository(session).get_for_run(self._run.id)
            )
            if approval is None:
                raise RuntimeError("no approval is associated with the persisted run")
            self._approval = approval
            self._action = ActionRepository(session).require(approval.action_id)
            if approval.run_id != self._run.id or self._action.run_id != self._run.id:
                raise CheckpointCompatibilityError("approval context is bound to another run")
            if metadata is not None and metadata.action_id != str(self._action.id):
                raise CheckpointCompatibilityError("approval action does not match checkpoint")
            try:
                budget = RunBudgetRepository(session).require(self._run.id)
            except EntityNotFound as exc:
                raise CheckpointCompatibilityError("run budget metadata is missing") from exc
            if (
                budget.max_steps != self.max_steps
                or budget.max_llm_calls != self.max_llm_calls
                or budget.max_tool_calls != self.max_tool_calls
                or budget.max_active_seconds != self.max_active_seconds
            ):
                raise CheckpointCompatibilityError(
                    "run budget limits do not match the persisted run"
                )
            budget = RunBudgetRepository(session).require(self._run.id)
            self._durable_llm_calls = budget.llm_calls_used
            self._tool_attempts = budget.tool_calls_used
            self._llm_attempts = max(self._llm_attempts, budget.llm_calls_used)
            self._active_elapsed_seconds = budget.active_elapsed_seconds
            self._active_started_wall = budget.active_started_at
        self.last_graph = None
        self.last_config = None

    def _restore_action_context(self, state: dict[str, Any]) -> None:
        """Recover exact checkpoint entities; never select an unrelated latest approval."""
        proposal = state.get("proposed_action") or {}
        action_id = proposal.get("action_id")
        approval_id = proposal.get("approval_request_id")
        if action_id is None and approval_id is None:
            return
        if self.database is None or self._run is None or not action_id or not approval_id:
            raise CheckpointCompatibilityError("checkpoint approval context is incomplete")
        with self.database.session() as session:
            action = ActionRepository(session).require(action_id)
            approval = ApprovalRepository(session).require(approval_id)
        if (
            action.run_id != self._run.id
            or approval.run_id != self._run.id
            or approval.action_id != action.id
            or approval.tool_name != action.tool_name
            or action.tool_name != proposal.get("tool_name")
            or proposal.get("parameter_hash")
            not in {
                approval.original_parameter_hash,
                approval.approved_parameter_hash,
            }
            or action.parameter_hash != proposal.get("parameter_hash")
            or stable_parameter_hash(proposal.get("arguments", {})) != action.parameter_hash
        ):
            raise CheckpointCompatibilityError("checkpoint approval scope does not match the run")
        self._action, self._approval = action, approval

    def _load_incident(self, state: IncidentAgentState) -> dict[str, Any]:
        step = self._advance(state)
        if step is None:
            return self._limit_update()
        if self._incident is None:
            raise RuntimeError("workflow incident is not initialized")
        if self._incident.status.value == "OPEN":
            self._incident.transition_to(IncidentStatus.INVESTIGATING)
            # Commit domain progress before LangGraph can checkpoint this node.
            # Recovery must not skip the only transition out of OPEN.
            self._persist_entities()
        return {"step_count": step, "phase": "triage"}

    def _triage(self, state: IncidentAgentState) -> dict[str, Any]:
        step = self._advance(state)
        if step is None:
            return self._limit_update()
        result = self._generate(
            "triage",
            build_prompt_messages("triage", {"description": state["description"]}),
            TriageDecision,
            state,
        )
        update = {
            "step_count": step,
            "phase": "plan",
            "severity": result.output.severity_assessment,
            "llm_call_count": state.get("llm_call_count", 0),
        }
        if result.output.needs_immediate_escalation:
            if self._incident is not None and self._incident.status is IncidentStatus.INVESTIGATING:
                self._incident.transition_to(IncidentStatus.ESCALATED)
            self._persist_entities()
            update.update({"phase": "escalated", "final_outcome": "escalated"})
        return update

    def _create_plan(self, state: IncidentAgentState) -> dict[str, Any]:
        step = self._advance(state)
        if step is None:
            return self._limit_update()
        result = self._generate(
            "diagnostic_plan",
            build_prompt_messages(
                "diagnostic_plan",
                {
                    "service": state["service_name"],
                    "description": state["description"],
                },
            ),
            DiagnosticPlan,
            state,
        )
        return {
            "step_count": step,
            "phase": "diagnostics",
            "diagnostic_plan": result.output.steps,
            "llm_call_count": state.get("llm_call_count", 0),
        }

    def _select_next_diagnostic_step(self, state: IncidentAgentState) -> dict[str, Any]:
        step_count = self._advance(state)
        if step_count is None:
            return self._limit_update()
        index = state.get("current_plan_index", 0)
        plan = state.get("diagnostic_plan", [])
        if index >= len(plan):
            return {"step_count": step_count, "phase": "remediation"}
        result = self._generate(
            "tool_selection",
            build_prompt_messages(
                "tool_selection",
                {
                    "service": state["service_name"],
                    "description": state["description"],
                    "step": plan[index],
                    "observations": state.get("observations", []),
                },
            ),
            ToolDecision,
            state,
        )
        return {
            "step_count": step_count,
            "phase": "execute_read",
            "current_plan_index": index + 1,
            "selected_tool_name": result.output.tool_name,
            "selected_tool_arguments": result.output.arguments.model_dump(exclude_none=True),
            "llm_call_count": state.get("llm_call_count", 0),
        }

    def _execute_read_tool(self, state: IncidentAgentState) -> dict[str, Any]:
        step = self._advance(state)
        if step is None:
            return self._limit_update()
        tool_name = state.get("selected_tool_name")
        arguments = state.get("selected_tool_arguments") or {}
        if not tool_name:
            return self._error_update("invalid_tool_decision", "model returned no diagnostic tool")
        attempt_count = self._reserve_tool_attempt(state)
        try:
            result = self._call_tool(
                state,
                tool_name,
                arguments,
                Actor(actor_id="agent", role=Role.VIEWER),
            )
        except ToolError as exc:
            return self._error_update(
                "tool_error",
                str(exc),
                tool_call_count=attempt_count,
            )
        observations = state.get("observations", []) + [
            {
                "tool_name": tool_name,
                "payload": result.payload.model_dump(),
                "tool_call_id": self._last_tool_attempt_id,
                "transport": result.transport,
            }
        ]
        self._persist_observations(
            {"environment_id": state.get("environment_id"), "observations": observations}
        )
        return {
            "step_count": step,
            "phase": "record_observation",
            "observations": observations,
            "tool_call_count": attempt_count,
        }

    def _record_observation(self, state: IncidentAgentState) -> dict[str, Any]:
        step = self._advance(state)
        if step is None:
            return self._limit_update()
        self._persist_observations(state)
        return {"step_count": step, "phase": "hypotheses"}

    def _hypothesis_context(self, state):
        if self.database is not None:
            with self.database.session() as session:
                repo = HypothesisLineageRepository(session)
                return repo.heads(self._run.id), repo.observations(self._run.id)
        observations = [
            {**item, "observation_id": str(uuid5(self._run.id, f"observation:{index}"))}
            for index, item in enumerate(state.get("observations", []))
        ]
        return [
            HypothesisSnapshot.model_validate(item) for item in state.get("hypotheses", [])
        ], observations

    def _rejected_hypothesis_update(
        self,
        state: IncidentAgentState,
        step: int,
        previous: list[HypothesisSnapshot],
        error: LLMProviderError,
    ) -> dict[str, Any]:
        """Discard one answered invalid model decision without authorizing an action."""
        remaining_reads = max(
            0,
            len(state.get("diagnostic_plan", [])) - state.get("current_plan_index", 0),
        )
        snapshots = [item.model_dump(mode="json") for item in previous]
        update: dict[str, Any] = {
            "step_count": step,
            "phase": "diagnostics" if remaining_reads else "escalated",
            "hypotheses": snapshots,
            "hypothesis_ids": [item["id"] for item in snapshots],
            "needs_more_evidence": bool(remaining_reads),
            "llm_call_count": max(
                state.get("llm_call_count", 0), self._llm_attempts, self._durable_llm_calls
            ),
        }
        if not remaining_reads:
            update.update(
                self._terminal_output_rejection(
                    state,
                    step,
                    error,
                    "Hypothesis output was invalid after all planned reads; "
                    "the run escalated without remediation.",
                )
            )
        return update

    def _terminal_output_rejection(
        self,
        state: IncidentAgentState,
        step: int,
        error: LLMProviderError,
        message: str,
    ) -> dict[str, Any]:
        """End safely after a completed but unusable model response."""
        return {
            "step_count": step,
            "phase": "escalated",
            "final_outcome": "escalated",
            "last_error_code": error.code,
            "last_error_message": message,
            "llm_call_count": max(
                state.get("llm_call_count", 0), self._llm_attempts, self._durable_llm_calls
            ),
        }

    def _update_hypotheses(self, state: IncidentAgentState) -> dict[str, Any]:
        step = self._advance(state)
        if step is None:
            return self._limit_update()
        saved = None
        if self.database is not None:
            with self.database.session() as session:
                repo = HypothesisLineageRepository(session)
                committed = repo.decision(self._run.id, step)
                if committed is not None:
                    saved = repo.replay(committed)
        if saved is None:
            previous, evidence = self._hypothesis_context(state)
            messages = build_prompt_messages(
                "hypothesis_update",
                {
                    "service": state["service_name"],
                    "description": state["description"],
                    "observation_count": len(evidence),
                    "plan_index": state.get("current_plan_index", 0),
                    "plan_size": len(state.get("diagnostic_plan", [])),
                    "observations": evidence,
                    "prior_hypotheses": [item.model_dump(mode="json") for item in previous],
                    "fact_catalog": [
                        item.model_dump(mode="json") for item in observed_facts(evidence)
                    ],
                },
            )
            try:
                result = self._generate("hypothesis_update", messages, HypothesisUpdate, state)
            except LLMProviderError as exc:
                # Journal propagation normalizes provider subclasses to this base
                # type. Recover only an invalid output after one completed call;
                # transport, metadata, and configuration failures remain terminal.
                if exc.code != "LLMOutputInvalid" or exc.attempts != 1:
                    raise
                return self._rejected_hypothesis_update(state, step, previous, exc)
            if self.database is not None:
                with self.database.session() as session:
                    self._renew_lease(session)
                    try:
                        saved = HypothesisLineageRepository(session).apply(
                            run_id=self._run.id,
                            logical_step=step,
                            decision_id=self._last_model_decision_id,
                            updates=result.output,
                        )
                    except HypothesisLineageError:
                        raise LLMOutputError("hypothesis lineage update is invalid") from None
                self._inject_fault("after_hypothesis_commit")
            else:
                snapshots, changes, diagnosis = HypothesisLineageRepository.plan(
                    run_id=self._run.id,
                    logical_step=step,
                    decision_id=uuid5(self._run.id, f"offline-decision:{step}"),
                    updates=result.output,
                    previous=previous,
                    observations=evidence,
                    now=utc_now(),
                )
                saved = {
                    "hypotheses": [item.model_dump(mode="json") for item in snapshots],
                    "needs_more_evidence": result.output.needs_more_evidence,
                    "diagnosis": diagnosis,
                }
        update: dict[str, Any] = {
            "step_count": step,
            "phase": "diagnostics",
            "hypotheses": saved["hypotheses"],
            "hypothesis_ids": [item["id"] for item in saved["hypotheses"]],
            "needs_more_evidence": saved["needs_more_evidence"],
            "llm_call_count": max(
                state.get("llm_call_count", 0), self._llm_attempts, self._durable_llm_calls
            ),
        }
        if saved["diagnosis"] is not None:
            diagnosis = saved["diagnosis"]
            update["diagnosis"] = diagnosis
            self._record_runtime_evidence(
                event_type="runtime.diagnosis.updated",
                event_key_suffix=f"diagnosis:{step}",
                payload={"schema_version": 1, "diagnosis": diagnosis},
            )
        return update

    def _propose_remediation(self, state: IncidentAgentState) -> dict[str, Any]:
        step = self._advance(state)
        if step is None:
            return self._limit_update()
        messages = build_prompt_messages(
            "remediation_proposal",
            {
                "service": state["service_name"],
                "observations": state.get("observations", []),
                "hypotheses": state.get("hypotheses", []),
            },
        )
        try:
            result = self._generate(
                "remediation_proposal",
                messages,
                RemediationProposal,
                state,
            )
        except LLMOutputError as exc:
            if exc.attempts != 1:
                raise
            return self._terminal_output_rejection(
                state,
                step,
                exc,
                "Remediation output was invalid; the run escalated without an action.",
            )
        proposal = result.output.model_dump(mode="json")
        proposal["arguments"] = result.output.arguments.model_dump(exclude_none=True)
        return {
            "step_count": step,
            "phase": "policy",
            "proposed_action": proposal,
            "llm_call_count": state.get("llm_call_count", 0),
        }

    def _policy_gate(self, state: IncidentAgentState) -> dict[str, Any]:
        step = self._advance(state)
        if step is None:
            return self._limit_update()
        proposal = state.get("proposed_action") or {}
        tool_name = proposal.get("tool_name")
        if not tool_name:
            return {"step_count": step, "phase": "escalated", "final_outcome": "escalated"}
        definition = next(
            (item for item in self.gateway.definitions() if item.name == tool_name),
            None,
        )
        reason = None
        if definition is None or definition.risk_level.value >= 3:
            reason = "unknown_tool" if definition is None else "forbidden_r3"
        elif (
            definition.observed_support(
                (proposal.get("arguments") or {}).get("service"), state.get("observations", [])
            )
            is False
        ):
            reason = "observed_action_inapplicable"
        if reason is not None:
            self._record_runtime_evidence(
                event_type="runtime.policy.denied",
                event_key_suffix=f"policy-denial:{step}",
                payload={
                    "schema_version": 1,
                    "tool_name": str(tool_name),
                    "reason": reason,
                },
            )
            return {
                "step_count": step,
                "phase": "escalated",
                "final_outcome": "escalated",
                "policy_denial_count": state.get("policy_denial_count", 0) + 1,
            }
        if definition.side_effecting and definition.risk_level.value >= 1:
            if self._run is None:
                raise RuntimeError("workflow run is not initialized")
            candidate_action = Action(
                id=uuid5(self._run.id, f"incident_triage:policy:{step}:action"),
                run_id=self._run.id,
                action_type=str(proposal.get("action_type", tool_name)),
                tool_name=tool_name,
                parameters=dict(proposal.get("arguments") or {}),
                risk_level=RiskLevel(definition.risk_level.value),
            )
            candidate_approval = ApprovalRequest(
                id=uuid5(self._run.id, f"incident_triage:policy:{step}:approval"),
                action_id=candidate_action.id,
                run_id=self._run.id,
                tool_name=candidate_action.tool_name,
                original_parameters=candidate_action.parameters,
            )
            if self.database is not None:
                with self.database.session() as session:
                    stored_action = ActionRepository(session).get(candidate_action.id)
                    stored_approval = ApprovalRepository(session).get(candidate_approval.id)
                    if stored_action is None and stored_approval is None:
                        # Older runs published random IDs before the checkpoint.
                        # Adopt one exact proposal; ambiguity blocks recovery.
                        actions = ActionRepository(session).list_for_run(self._run.id)
                        approvals = ApprovalRepository(session).list_for_run(self._run.id)
                        if actions or approvals:
                            if len(actions) != 1 or len(approvals) != 1:
                                raise CheckpointCompatibilityError(
                                    "published policy proposal is ambiguous"
                                )
                            stored_action, stored_approval = actions[0], approvals[0]
                if stored_action is not None or stored_approval is not None:
                    if (
                        stored_action is None
                        or stored_approval is None
                        or stored_action.run_id != self._run.id
                        or stored_approval.action_id != stored_action.id
                        or stored_approval.original_parameter_hash
                        != candidate_action.parameter_hash
                        or stored_action.tool_name != candidate_action.tool_name
                    ):
                        raise CheckpointCompatibilityError(
                            "published policy proposal conflicts with replay"
                        )
                    candidate_action, candidate_approval = stored_action, stored_approval
            self._action, self._approval = candidate_action, candidate_approval
            if self._incident is not None and self._incident.status is IncidentStatus.INVESTIGATING:
                self._incident.transition_to(IncidentStatus.MITIGATION_PROPOSED)
            self._persist_entities()
            return {
                "step_count": step,
                "phase": "approval",
                "proposed_action": {
                    **proposal,
                    "action_id": str(self._action.id),
                    "approval_request_id": str(self._approval.id),
                    "parameter_hash": self._action.parameter_hash,
                },
            }
        return {"step_count": step, "phase": "execute_safe_action"}

    def _human_approval_interrupt(self, state: IncidentAgentState) -> dict[str, Any]:
        step = self._advance(state)
        if step is None:
            return self._limit_update()
        if self._action is None or self._approval is None:
            return self._error_update("ApprovalStateMissing", "approval state was not created")
        payload = {
            "approval_request_id": str(self._approval.id),
            "action_id": str(self._action.id),
            "tool_name": self._action.tool_name,
            "parameters": self._action.parameters,
            "risk_level": f"R{self._action.risk_level.value}",
            "evidence_summary": [
                item.get("tool_name") for item in state.get("observations", [])[-8:]
            ],
            "reason": (state.get("proposed_action") or {}).get("justification", ""),
        }
        self.pending_approval_payload = payload
        decision = interrupt(payload)
        if not isinstance(decision, dict):
            return self._error_update("InvalidApprovalDecision", "resume payload must be an object")
        committed_decision = decision.get("decision_id")
        if committed_decision is not None:
            if self.database is None or self._run is None:
                return self._error_update(
                    "ApprovalStateMissing", "a committed decision requires durable storage"
                )
            try:
                with self.database.session() as session:
                    stored_decision = ApprovalDecisionRepository(session).require(
                        committed_decision
                    )
                    stored_approval = ApprovalRepository(session).require(self._approval.id)
                    if (
                        stored_decision.approval_id != str(self._approval.id)
                        or stored_decision.action_id != str(self._action.id)
                        or stored_decision.run_id != str(self._run.id)
                    ):
                        return self._error_update(
                            "ApprovalScopeMismatch",
                            "committed decision is bound to another action or run",
                        )
                    self._approval = stored_approval
                    decision = {
                        "approved": stored_decision.status
                        in {
                            ApprovalStatus.APPROVED.value,
                            ApprovalStatus.APPROVED_WITH_CHANGES.value,
                        },
                        "approved_parameters": stored_decision.approved_parameters_json,
                        "decided_by": stored_decision.decided_by,
                        "actor_role": stored_decision.actor_role,
                        "reason": stored_decision.decision_reason,
                    }
            except (RuntimeError, ValueError) as exc:
                return self._error_update("InvalidApprovalDecision", str(exc))
        if decision.get("actor_role") not in {Role.APPROVER.value, Role.ADMIN.value}:
            return self._error_update("UnauthorizedApproval", "approval role is not authorized")
        if not bool(decision.get("approved")):
            if committed_decision is None:
                self._approval.decide(
                    approved=False,
                    decided_by=str(decision.get("decided_by", "unknown")),
                    reason=decision.get("reason"),
                )
            object.__setattr__(self._action, "status", ActionStatus.REJECTED)
            if (
                self._incident is not None
                and self._incident.status is IncidentStatus.MITIGATION_PROPOSED
            ):
                self._incident.transition_to(IncidentStatus.ESCALATED)
            self._persist_entities()
            return {
                "step_count": step,
                "phase": "escalated",
                "final_outcome": "escalated",
                "last_error_code": "ApprovalRejected",
                "last_error_message": decision.get("reason") or "approval rejected",
            }
        edited = decision.get("approved_parameters")
        if edited is not None and not isinstance(edited, dict):
            return self._error_update(
                "InvalidApprovalDecision", "approved_parameters must be an object"
            )
        if committed_decision is None:
            self._approval.decide(
                approved=True,
                decided_by=str(decision.get("decided_by", "unknown")),
                approved_parameters=edited,
                reason=decision.get("reason"),
            )
        approved_values = (
            self._approval.approved_parameters
            if self._approval.approved_parameters is not None
            else self._action.parameters
        )
        object.__setattr__(self._action, "parameters", dict(approved_values))
        object.__setattr__(self._action, "parameter_hash", stable_parameter_hash(approved_values))
        self._action.mark_ready(self._approval)
        self._persist_entities()
        return {
            "step_count": step,
            "phase": "execute_safe_action",
            "proposed_action": {
                **(state.get("proposed_action") or {}),
                "arguments": dict(approved_values),
                "parameter_hash": self._action.parameter_hash,
            },
        }

    def _execute_safe_action(self, state: IncidentAgentState) -> dict[str, Any]:
        step = self._advance(state)
        if step is None:
            return self._limit_update()
        proposal = state.get("proposed_action") or {}
        attempt_count = self._reserve_tool_attempt(state, required_remaining=3)
        try:
            if self._action is not None:
                object.__setattr__(self._action, "status", ActionStatus.EXECUTING)
                if (
                    self._incident is not None
                    and self._incident.status is IncidentStatus.MITIGATION_PROPOSED
                ):
                    self._incident.transition_to(IncidentStatus.MITIGATING)
                self._persist_entities()
            self._assert_mutation_allowed()
            if self._action is not None and isinstance(self.simulator, PersistentSimulator):
                self.gateway.recover_simulator_execution(
                    str(proposal["tool_name"]),
                    dict(proposal.get("arguments") or {}),
                    Actor(actor_id="agent", role=Role.OPERATOR),
                    approval=self._approval,
                    idempotency_key=self._action.idempotency_key,
                    environment_generation=self._reservation_generation,
                    transaction_guard=self._assert_mutation_transaction,
                )
            result = self._call_tool(
                state,
                str(proposal["tool_name"]),
                dict(proposal.get("arguments") or {}),
                Actor(actor_id="agent", role=Role.OPERATOR),
                approval=self._approval,
                idempotency_key=self._action.idempotency_key if self._action else str(uuid4()),
            )
            # A short lease can expire while a side effect runs; renew before
            # persisting its outcome so the durable action record is fenced.
            self._renew_lease()
        except ToolError as exc:
            denial_count = state.get("policy_denial_count", 0)
            if isinstance(exc, (ApprovalRequired, ForbiddenAction, PermissionDenied)):
                denial_count += 1
                self._record_runtime_evidence(
                    event_type="runtime.policy.denied",
                    event_key_suffix=f"policy-denial:{step}",
                    payload={
                        "schema_version": 1,
                        "tool_name": str(proposal.get("tool_name", "unknown")),
                        "reason": type(exc).__name__,
                    },
                )
            if self._action is not None:
                object.__setattr__(self._action, "status", ActionStatus.FAILED)
                self._persist_entities()
            return self._error_update(
                "tool_error",
                str(exc),
                tool_call_count=attempt_count,
                policy_denial_count=denial_count,
            )
        if self._action is not None:
            object.__setattr__(self._action, "status", ActionStatus.SUCCEEDED)
            self._action.executed_at = utc_now()
            self._action.completed_at = utc_now()
            self._persist_entities()
        observations = state.get("observations", []) + [
            {
                "tool_name": str(proposal["tool_name"]),
                "payload": result.payload.model_dump(),
                "tool_call_id": self._last_tool_attempt_id,
            }
        ]
        self._persist_observations(
            {"environment_id": state.get("environment_id"), "observations": observations}
        )
        return {
            "step_count": step,
            "phase": "verify",
            "observations": observations,
            "tool_call_count": attempt_count,
        }

    def _verify_environment(self, state: IncidentAgentState) -> dict[str, Any]:
        step = self._advance(state)
        if step is None:
            return self._limit_update()
        attempt_count = self._reserve_tool_attempt(state, required_remaining=2)
        try:
            status_result = self._call_tool(
                state,
                "get_service_status",
                {"service": state["service_name"]},
                Actor(actor_id="agent", role=Role.VIEWER),
            )
        except ToolError as exc:
            return self._error_update(
                "tool_error",
                str(exc),
                tool_call_count=attempt_count,
            )
        status_call_id = self._last_tool_attempt_id
        attempt_count = self._reserve_tool_attempt(state)
        try:
            verification_result = self._call_tool(
                state,
                "verify_service",
                {"service": state["service_name"]},
                Actor(actor_id="agent", role=Role.VIEWER),
            )
        except ToolError as exc:
            return self._error_update(
                "tool_error",
                str(exc),
                tool_call_count=attempt_count,
            )
        if self._incident is not None and self._incident.status is IncidentStatus.MITIGATING:
            self._incident.transition_to(IncidentStatus.VERIFYING)
        self._persist_entities()
        observations = state.get("observations", []) + [
            {
                "tool_name": "get_service_status",
                "payload": status_result.payload.model_dump(),
                "tool_call_id": status_call_id,
            },
            {
                "tool_name": "verify_service",
                "payload": verification_result.payload.model_dump(),
                "tool_call_id": self._last_tool_attempt_id,
            },
        ]
        self._persist_observations(
            {"environment_id": state.get("environment_id"), "observations": observations}
        )
        return {
            "step_count": step,
            "phase": "review",
            "observations": observations,
            "tool_call_count": attempt_count,
        }

    def _review(self, state: IncidentAgentState) -> dict[str, Any]:
        step = self._advance(state)
        if step is None:
            return self._limit_update()
        verification = next(
            (
                item["payload"]
                for item in reversed(state.get("observations", []))
                if item.get("tool_name") == "verify_service"
            ),
            {},
        )
        result = self._generate(
            "review",
            build_prompt_messages("review", verification),
            ReviewDecision,
            state,
        )
        self._assert_cancellation_clear()
        checks = verification.get("checks")
        verification_succeeded = (
            verification.get("healthy") is True
            and isinstance(checks, dict)
            and bool(checks)
            and all(value is True for value in checks.values())
        )
        outcome = (
            "resolved"
            if verification_succeeded and result.output.outcome == "resolved"
            else "escalated"
        )
        if outcome == "resolved" and self._incident is not None:
            self._incident.transition_to(IncidentStatus.RESOLVED, verification_succeeded=True)
        elif self._incident is not None and self._incident.status is IncidentStatus.VERIFYING:
            self._incident.transition_to(IncidentStatus.ESCALATED)
        self._persist_entities()
        return {
            "step_count": step,
            "phase": outcome,
            "final_outcome": outcome,
            "llm_call_count": state.get("llm_call_count", 0),
        }

    @staticmethod
    def _route_after_triage(
        state: IncidentAgentState,
    ) -> Literal["create_plan", "end"]:
        return "end" if state.get("final_outcome") else "create_plan"

    def _route_after_hypotheses(
        self, state: IncidentAgentState
    ) -> Literal["select_next_diagnostic_step", "propose_remediation", "end"]:
        if state.get("final_outcome"):
            return "end"
        if state.get("needs_more_evidence") is False:
            return "propose_remediation"
        if state.get("current_plan_index", 0) < len(state.get("diagnostic_plan", [])):
            return "select_next_diagnostic_step"
        return "propose_remediation"

    def _route_after_policy(
        self, state: IncidentAgentState
    ) -> Literal["human_approval_interrupt", "execute_safe_action", "verify_environment", "end"]:
        if state.get("final_outcome"):
            return "end"
        if state.get("phase") == "approval":
            return "human_approval_interrupt"
        if state.get("phase") == "execute_safe_action":
            return "execute_safe_action"
        return "verify_environment"

    @staticmethod
    def _route_after_execution(
        state: IncidentAgentState,
    ) -> Literal["verify_environment", "end"]:
        return "end" if state.get("final_outcome") else "verify_environment"

    @staticmethod
    def _route_after_approval(
        state: IncidentAgentState,
    ) -> Literal["execute_safe_action", "end"]:
        return "end" if state.get("final_outcome") else "execute_safe_action"

    def _generate(
        self,
        task: str,
        messages: list[Message],
        response_model: type[Any],
        state: IncidentAgentState,
    ) -> Any:
        retryable_codes = {"LLMTimeout", "LLMUnavailable", "LLMRateLimited"}
        last_error: LLMProviderError | None = None
        for retry_index in range(self.model_config.max_retries + 1):
            self._assert_cancellation_clear()
            self._renew_lease()
            if self._active_budget_exceeded():
                raise WorkflowBudgetExceeded(
                    "active workflow time budget is exhausted",
                    code="TimeBudgetExceeded",
                )
            self._reserve_llm_attempt(state)
            # The workflow is the sole retry owner. The adapter/SDK must issue
            # exactly one provider-boundary request for each reservation.
            call_config = self.model_config.model_copy(update={"max_retries": 0})
            try:
                result = self._call_model(
                    task=task,
                    messages=messages,
                    response_model=response_model,
                    config=call_config,
                    state=state,
                )
            except LLMProviderError as exc:
                last_error = exc
                if exc.code not in retryable_codes or retry_index >= self.model_config.max_retries:
                    raise
                self._retry_sleep(min(0.5 * (2**retry_index), 2.0))
                continue
            except TimeoutError as exc:
                last_error = LLMProviderError("LLMTimeout", "LLM provider timed out")
                if retry_index >= self.model_config.max_retries:
                    raise last_error from exc
                self._retry_sleep(min(0.5 * (2**retry_index), 2.0))
                continue
            except ConnectionError as exc:
                last_error = LLMProviderError("LLMUnavailable", "LLM provider is unavailable")
                if retry_index >= self.model_config.max_retries:
                    raise last_error from exc
                self._retry_sleep(min(0.5 * (2**retry_index), 2.0))
                continue
            return result
        raise last_error or LLMProviderError("LLMUnavailable", "LLM provider call did not complete")

    def _reserve_tool_attempt(
        self, state: IncidentAgentState, *, required_remaining: int = 1
    ) -> int:
        self._renew_lease()
        self._assert_cancellation_clear()
        if self._active_budget_exceeded():
            raise WorkflowBudgetExceeded(
                "active workflow time budget is exhausted", code="TimeBudgetExceeded"
            )
        current = max(int(state.get("tool_call_count", 0)), self._tool_attempts)
        if self.database is None:
            if current + required_remaining > int(state.get("max_tool_calls", self.max_tool_calls)):
                raise WorkflowBudgetExceeded(
                    "no tool attempt remains with the required verification allowance",
                    code="ToolBudgetExceeded",
                )
            used = current + 1
        else:
            if self._run is None:
                raise RuntimeError("workflow context is incomplete")
            with self.database.session() as session:
                self._renew_lease(session)
                self._assert_cancellation_clear(session)
                repository = RunBudgetRepository(session)
                # Read the durable counter even when admission fails, so terminal
                # state cannot report a smaller count from an old checkpoint.
                self._tool_attempts = max(current, repository.require(self._run.id).tool_calls_used)
                try:
                    used = repository.reserve_tool_attempt(
                        self._run.id,
                        minimum_used=current,
                        required_remaining=required_remaining,
                    )
                except PersistenceError as exc:
                    raise WorkflowBudgetExceeded(
                        "no tool attempt remains with the required verification allowance",
                        code="ToolBudgetExceeded",
                    ) from exc
        self._tool_attempts = used
        state["tool_call_count"] = used
        return used

    def _reserve_llm_attempt(self, state: IncidentAgentState) -> None:
        current = int(state.get("llm_call_count", 0))
        if self.database is None:
            if current >= int(state.get("max_llm_calls", self.max_llm_calls)):
                raise WorkflowBudgetExceeded(
                    "no LLM attempt remains within the configured run budget",
                    code="LLMBudgetExceeded",
                )
            used = current + 1
        else:
            if self._run is None:
                raise RuntimeError("workflow context is incomplete")
            with self.database.session() as session:
                self._renew_lease(session)
                repository = RunBudgetRepository(session)
                repository.ensure(
                    run_id=self._run.id,
                    max_steps=self.max_steps,
                    max_llm_calls=self.max_llm_calls,
                    max_tool_calls=self.max_tool_calls,
                    max_active_seconds=self.max_active_seconds,
                )
                try:
                    used = repository.reserve_llm_attempt(self._run.id)
                except PersistenceError as exc:
                    raise WorkflowBudgetExceeded(
                        "no LLM attempt remains within the configured run budget",
                        code="LLMBudgetExceeded",
                    ) from exc
        state["llm_call_count"] = max(current, used, self._durable_llm_calls)
        self._llm_attempts = max(self._llm_attempts, used)
        self._durable_llm_calls = max(self._durable_llm_calls, used)

    def _validate_semantics(self, task: str, output: Any, state: IncidentAgentState) -> None:
        service_name = state.get("service_name")
        if task == "triage" and output.severity_assessment not in {"SEV1", "SEV2", "SEV3", "SEV4"}:
            raise LLMOutputError("triage severity assessment is not an allowed value")
        if task == "tool_selection":
            arguments = output.arguments.model_dump(exclude_none=True)
            if arguments.get("service") not in {None, service_name}:
                raise LLMOutputError("tool selection service is outside the incident scope")
        if task == "hypothesis_update":
            try:
                if self.database is not None:
                    with self.database.session() as session:
                        HypothesisLineageRepository(session).validate_output(
                            self._run.id, state.get("step_count", 0) + 1, output
                        )
                else:
                    previous, evidence = self._hypothesis_context(state)
                    HypothesisLineageRepository.plan(
                        run_id=self._run.id,
                        logical_step=state.get("step_count", 0) + 1,
                        decision_id=uuid5(self._run.id, "validation-only"),
                        updates=output,
                        previous=previous,
                        observations=evidence,
                        now=utc_now(),
                    )
            except (HypothesisLineageError, ValidationError):
                raise LLMOutputError(
                    "hypothesis references or structured claim are invalid"
                ) from None
        if task == "hypothesis_update" and output.diagnosis is not None:
            visible_evidence = {
                str(item.get("tool_name"))
                for item in state.get("observations", [])
                if item.get("tool_name")
            }
            unknown_refs = set(output.diagnosis.evidence_refs) - visible_evidence
            if unknown_refs:
                raise LLMOutputError("diagnosis references evidence outside this run")
        if task == "remediation_proposal":
            arguments = output.arguments.model_dump(exclude_none=True)
            if arguments.get("service") not in {None, service_name}:
                raise LLMOutputError("remediation service is outside the incident scope")
            visible_evidence = {
                str(item.get("tool_name"))
                for item in state.get("observations", [])
                if item.get("tool_name")
            }
            unknown_refs = set(output.evidence_ids) - visible_evidence
            if unknown_refs:
                raise LLMOutputError("remediation references evidence outside this run")
            if output.tool_name is None:
                if output.action_type != "escalate" or arguments:
                    raise LLMOutputError("escalation proposal must not carry a write tool")
                return
            if not output.evidence_ids:
                raise LLMOutputError("remediation write requires observed evidence references")
            if set(output.evidence_ids) != visible_evidence or len(output.evidence_ids) != len(
                visible_evidence
            ):
                raise LLMOutputError("write proposal must cite every observed tool exactly once")
            if output.tool_name == "rollback_deployment":
                rollback_targets = _observed_stable_rollback_targets(
                    state.get("observations", []), service_name
                )
                if arguments.get("target_version") not in rollback_targets:
                    raise LLMOutputError(
                        "rollback target must be an observed stable version other than current"
                    )
                if not {"get_service_status", "get_deployment_history"}.issubset(
                    set(output.evidence_ids)
                ):
                    raise LLMOutputError(
                        "rollback proposal must cite current status and deployment history"
                    )
            definition = next(
                (item for item in self.gateway.definitions() if item.name == output.tool_name),
                None,
            )
            if definition is None or not definition.side_effecting:
                raise LLMOutputError("remediation tool is not an allow-listed side effect")
            if output.action_type != output.tool_name:
                raise LLMOutputError("remediation action_type does not match tool_name")

    def _advance(self, state: IncidentAgentState) -> int | None:
        current = state.get("step_count", 0)
        self._renew_lease()
        self._assert_cancellation_clear()
        if current >= state.get("max_steps", 20):
            return None
        if self._active_budget_exceeded():
            raise WorkflowBudgetExceeded(
                "active workflow time budget is exhausted",
                code="TimeBudgetExceeded",
            )
        return current + 1

    def _start_active_segment(self) -> None:
        if self._active_started_at is None:
            self._active_started_at = self._clock()
            self._active_started_wall = utc_now()
            self._persist_budget_snapshot()
            # Snapshot persistence can consume most of a deliberately short
            # lease; start graph work from a freshly renewed fence.
            self._renew_lease()

    def _pause_active_segment(self) -> None:
        if self._active_started_at is None:
            return
        self._active_elapsed_seconds += max(0.0, self._clock() - self._active_started_at)
        self._active_started_at = None
        self._active_started_wall = None
        self._persist_budget_snapshot()

    def _active_budget_exceeded(self) -> bool:
        elapsed = self._active_elapsed_seconds
        if self._active_started_at is not None:
            elapsed += max(0.0, self._clock() - self._active_started_at)
        return elapsed >= self.max_active_seconds

    @staticmethod
    def _find_provider_error(error: BaseException) -> LLMProviderError | None:
        seen: set[int] = set()
        current: BaseException | None = error
        while current is not None and id(current) not in seen:
            seen.add(id(current))
            if isinstance(current, LLMProviderError):
                return current
            current = current.__cause__ or current.__context__
        return None

    @staticmethod
    def _find_budget_error(error: BaseException) -> WorkflowBudgetExceeded | None:
        seen: set[int] = set()
        current: BaseException | None = error
        while current is not None and id(current) not in seen:
            seen.add(id(current))
            if isinstance(current, WorkflowBudgetExceeded):
                return current
            current = current.__cause__ or current.__context__
        return None

    @staticmethod
    def _budget_update(step: int, code: str, message: str) -> dict[str, Any]:
        return {
            "step_count": step,
            "phase": "escalated",
            "final_outcome": "budget_exceeded",
            "last_error_code": code,
            "last_error_message": message,
        }

    @staticmethod
    def _limit_update() -> dict[str, Any]:
        return {
            "phase": "escalated",
            "final_outcome": "step_limit_exceeded",
            "last_error_code": "StepLimitExceeded",
            "last_error_message": "maximum workflow step count reached",
        }

    def _cancellation_result(
        self,
        context: IncidentAgentState | dict[str, Any],
        error: CancellationRequested | None = None,
        *,
        check_lease: bool = True,
    ) -> WorkflowResult:
        if self._run is None or self._incident is None:
            raise RuntimeError("workflow context is incomplete")
        state = dict(context)
        state.update(
            {
                "phase": "cancelled",
                "final_outcome": "cancelled",
                "last_error_code": "RunCancelled",
                "last_error_message": str(error)
                if error is not None
                else "run cancellation was requested",
                "tool_call_count": max(
                    int(context.get("tool_call_count", 0)),
                    self._run.tool_call_count,
                    self._tool_attempts,
                ),
            }
        )
        if self._run.status in {
            AgentRunStatus.RUNNING,
            AgentRunStatus.WAITING_FOR_APPROVAL,
            AgentRunStatus.WAITING_FOR_RETRY,
            AgentRunStatus.QUEUED,
        }:
            self._run.transition_to(AgentRunStatus.CANCELLED)
        self._run.last_error_code = "RunCancelled"
        self._run.last_error_message = state["last_error_message"]
        self._run.tool_call_count = state["tool_call_count"]
        self._persist_entities(check_lease=check_lease)
        return WorkflowResult(incident=self._incident, run=self._run, state=state)

    def _workflow_error_result(
        self,
        initial: IncidentAgentState,
        code: str,
        error: Exception,
    ) -> WorkflowResult:
        state = dict(initial)
        message = error.public_message if isinstance(error, LLMProviderError) else str(error)[:500]
        state.update(
            {
                "phase": "escalated",
                "final_outcome": "budget_exceeded"
                if code.endswith("BudgetExceeded")
                else "tool_error",
                "last_error_code": code,
                "last_error_message": message,
                "llm_call_count": max(
                    state.get("llm_call_count", 0),
                    self._llm_attempts,
                    self._durable_llm_calls,
                ),
                "tool_call_count": max(state.get("tool_call_count", 0), self._tool_attempts),
            }
        )
        if self._incident is not None and self._incident.status in {
            IncidentStatus.INVESTIGATING,
            IncidentStatus.MITIGATION_PROPOSED,
            IncidentStatus.MITIGATING,
            IncidentStatus.VERIFYING,
        }:
            self._incident.transition_to(IncidentStatus.ESCALATED)
        if self._run is not None and self._run.status in {
            AgentRunStatus.RUNNING,
            AgentRunStatus.WAITING_FOR_APPROVAL,
        }:
            self._run.transition_to(AgentRunStatus.BLOCKED)
            self._run.step_count = state.get("step_count", 0)
            self._run.llm_call_count = state.get("llm_call_count", 0)
            self._run.tool_call_count = state.get("tool_call_count", 0)
            self._run.last_error_code = code
            self._run.last_error_message = state["last_error_message"]
        try:
            self._persist_entities()
        except CancellationRequested as exc:
            return self._cancellation_result(state, exc)
        if self._incident is None or self._run is None:
            raise RuntimeError("workflow terminal state is missing persisted entities")
        return WorkflowResult(incident=self._incident, run=self._run, state=state)

    @staticmethod
    def _error_update(
        code: str,
        message: str,
        **updates: Any,
    ) -> dict[str, Any]:
        return {
            "phase": "escalated",
            "final_outcome": "tool_error",
            "last_error_code": code,
            "last_error_message": "Workflow step could not complete; inspect its error code",
            **updates,
        }

    def _record_runtime_evidence(
        self,
        *,
        event_type: str,
        event_key_suffix: str,
        payload: dict[str, Any],
    ) -> None:
        if self.database is None or self._run is None:
            return
        event_key = f"run:{self._run.id}:{event_key_suffix}"
        bounded_payload = durable_payload(payload)
        with self.database.session() as session:
            self._renew_lease(session)
            existing = session.query(AuditEventRow).filter_by(event_key=event_key).one_or_none()
            if existing is not None:
                if existing.event_type != event_type or existing.payload_json != bounded_payload:
                    raise RuntimeError("runtime evidence key is bound to different content")
                return
            session.add(
                AuditEventRow(
                    id=str(uuid4()),
                    event_key=event_key,
                    occurred_at=utc_now(),
                    actor_type="agent_runtime",
                    actor_id=self._lease_owner_id,
                    event_type=event_type,
                    entity_type="agent_run",
                    entity_id=str(self._run.id),
                    run_id=str(self._run.id),
                    payload_json=bounded_payload,
                )
            )

    def _persist_observations(self, state: IncidentAgentState | dict[str, Any]) -> None:
        if self.database is None or self._run is None:
            return
        observations = state.get("observations", [])
        if not observations:
            return
        environment_id = state.get("environment_id")
        if not isinstance(environment_id, str) or not environment_id:
            raise RuntimeError("observation persistence requires an environment_id")
        identity = self.simulator.environment_identity
        with self.database.session() as session:
            self._renew_lease(session)
            repository = ObservationRepository(session)
            for sequence, observation in enumerate(observations):
                source_tool = observation.get("tool_name")
                payload = observation.get("payload")
                if not isinstance(source_tool, str) or not isinstance(payload, dict):
                    raise RuntimeError("observation persistence requires tool and payload")
                repository.save(
                    run_id=self._run.id,
                    sequence=sequence,
                    environment_id=environment_id,
                    environment_version=identity.version,
                    source_tool=source_tool,
                    payload=payload,
                    tool_call_id=observation.get("tool_call_id"),
                )

    def _persist_entities(self, *, check_lease: bool = True) -> None:
        if self.database is None or self._incident is None or self._run is None:
            return
        with self.database.session() as session:
            if check_lease:
                self._renew_lease(session)
                if self._run.status is not AgentRunStatus.CANCELLED:
                    self._assert_cancellation_clear(session)
            IncidentRepository(session).save(self._incident)
            AgentRunRepository(session).save(self._run)
            self._incident.updated_at = (
                IncidentRepository(session).require(self._incident.id).updated_at
            )
            RunJournalRepository(session).record_state(self._run)
            if self._action is not None:
                ActionRepository(session).save(self._action)
            if self._approval is not None:
                self._approval = ApprovalRepository(session).save(self._approval)
            RunBudgetRepository(session).ensure(
                run_id=self._run.id,
                max_steps=self.max_steps,
                max_llm_calls=self.max_llm_calls,
                max_tool_calls=self.max_tool_calls,
                max_active_seconds=self.max_active_seconds,
            )

    def _persist_budget_snapshot(self, state: dict[str, Any] | None = None) -> None:
        if self.database is None or self._run is None:
            return
        values = state or {}
        with self.database.session() as session:
            self._renew_lease(session)
            repository = RunBudgetRepository(session)
            repository.ensure(
                run_id=self._run.id,
                max_steps=self.max_steps,
                max_llm_calls=self.max_llm_calls,
                max_tool_calls=self.max_tool_calls,
                max_active_seconds=self.max_active_seconds,
            )
            repository.update_snapshot(
                self._run.id,
                steps_used=int(values.get("step_count", self._run.step_count)),
                llm_calls_used=max(
                    int(values.get("llm_call_count", self._run.llm_call_count)),
                    self._llm_attempts,
                    self._durable_llm_calls,
                ),
                tool_calls_used=max(
                    int(values.get("tool_call_count", self._run.tool_call_count)),
                    self._tool_attempts,
                ),
                active_elapsed_seconds=self._active_elapsed_seconds,
                active_started_at=self._active_started_wall,
            )

    def _inject_fault(self, point: str) -> None:
        if self._fault_injector is not None:
            self._fault_injector(point)

    def _checkpoint_snapshot(self) -> Any:
        if self._checkpoint_store is None or self.last_graph is None or self.last_config is None:
            raise CheckpointCompatibilityError("durable checkpoint configuration is missing")
        try:
            snapshot = self.last_graph.get_state(self.last_config)
            checkpoint_id = snapshot.config.get("configurable", {}).get("checkpoint_id")
        except Exception:
            self._reject_runtime_contract("checkpoint_unreadable")
        if not snapshot.values or not checkpoint_id:
            self._reject_runtime_contract("checkpoint_missing")
        return snapshot

    def _reject_runtime_contract(self, reason: str) -> None:
        self._record_runtime_evidence(
            event_type="runtime.contract.rejected",
            event_key_suffix=f"runtime-contract-rejected:{reason}",
            payload={"code": "RuntimeContractIncompatible", "reason": reason},
        )
        raise CheckpointCompatibilityError(f"runtime contract is incompatible: {reason}") from None

    def _ensure_runtime_contract(self) -> None:
        if self._run is None:
            raise CheckpointCompatibilityError("run context is missing")
        reason = run_contract_reason(self._run)
        if reason is not None:
            self._reject_runtime_contract(reason)

    def _validate_runtime_state(self, snapshot: Any) -> None:
        self._ensure_runtime_contract()
        configurable = snapshot.config.get("configurable", {})
        if (
            configurable.get("thread_id") != self._run.thread_id
            or configurable.get("checkpoint_ns", "") != ""
        ):
            self._reject_runtime_contract("checkpoint_thread_mismatch")
        values = snapshot.values
        reason = checkpoint_shape_reason(self._run, values, snapshot.next)
        if reason is not None:
            self._reject_runtime_contract(reason)
        expected_limits = (
            self.max_steps,
            self.max_llm_calls,
            self.max_tool_calls,
            self.max_active_seconds,
        )
        actual_limits = tuple(
            values[key]
            for key in (
                "max_steps",
                "max_llm_calls",
                "max_tool_calls",
                "max_active_seconds",
            )
        )
        if actual_limits != expected_limits:
            self._reject_runtime_contract("budget_limits_mismatch")
        reason = None
        with self.database.session() as session:
            observations = ObservationRepository(session).list_for_run(self._run.id)
            if len(values["observations"]) > len(observations):
                reason = "observation_lineage_mismatch"
            for index, item in enumerate(values["observations"]):
                if reason:
                    break
                durable = observations[index]
                if (
                    durable.environment_id != values["environment_id"]
                    or durable.source_tool != item["tool_name"]
                    or durable.payload_json != item["payload"]
                    or durable.tool_call_id != item.get("tool_call_id")
                ):
                    reason = "observation_lineage_mismatch"
            repo = HypothesisLineageRepository(session)
            for raw in values["hypotheses"]:
                item = HypothesisSnapshot.model_validate(raw)
                row = session.get(HypothesisVersionRow, str(item.version_id))
                try:
                    stored = repo.snapshot(row) if row is not None else None
                except HypothesisLineageError:
                    stored = None
                if stored is None or stored.model_dump(mode="json") != item.model_dump(mode="json"):
                    reason = "hypothesis_lineage_mismatch"
                    break
        if reason is not None:
            self._reject_runtime_contract(reason)
        try:
            self._restore_action_context(values)
        except (CheckpointCompatibilityError, EntityNotFound, ValueError, TypeError):
            self._reject_runtime_contract("proposal_scope_mismatch")
        proposal = values.get("proposed_action") or {}
        definition = next(
            (item for item in self.gateway.definitions() if item.name == proposal.get("tool_name")),
            None,
        )
        if "human_approval_interrupt" in snapshot.next or (
            "execute_safe_action" in snapshot.next
            and definition is not None
            and definition.side_effecting
        ):
            if not proposal.get("action_id") or not proposal.get("approval_request_id"):
                self._reject_runtime_contract("proposal_scope_mismatch")

    def _reject_model_configuration(self, reason: str) -> None:
        current = model_configuration_identity(self.llm, self.model_config)
        self._record_runtime_evidence(
            event_type="runtime.model_configuration.rejected",
            event_key_suffix=f"model-config-rejected:{reason}:{current.configuration_id}",
            payload={"code": "ModelConfigurationIncompatible", "reason": reason},
        )
        raise CheckpointCompatibilityError(f"model configuration is incompatible: {reason}")

    def _ensure_model_configuration(self, *, allow_bind: bool) -> None:
        """Bind only a never-started run; reject unproven legacy provenance.

        The existing lease fence and row lock serialize admission. Operational
        timeout/retry changes keep the originally recorded descriptor intact.
        A reopened incident gets a new run; its old run's identity is immutable.
        """
        if self._run is None:
            raise CheckpointCompatibilityError("run context is missing")
        current = model_configuration_identity(self.llm, self.model_config)
        if self.database is not None:
            with self.database.session() as session:
                self._renew_lease(session)
                repository = AgentRunRepository(session)
                durable = repository.require_for_update(self._run.id)
                if (
                    durable.model_configuration is None
                    and allow_bind
                    and durable.status in {AgentRunStatus.CREATED, AgentRunStatus.QUEUED}
                    and durable.started_at is None
                    and not (
                        durable.step_count or durable.llm_call_count or durable.tool_call_count
                    )
                    and CheckpointRepository(session).get_for_run(durable.id) is None
                ):
                    durable.model_configuration = current
                    repository.save(durable)
                self._run.model_configuration = durable.model_configuration
        elif self._run.model_configuration is None and allow_bind:
            self._run.model_configuration = current
        if self._run.model_configuration is None:
            self._reject_model_configuration("identity_missing")
        if self._run.model_configuration_id != current.configuration_id:
            self._reject_model_configuration("semantic_configuration_changed")

    def _assert_checkpoint_model_configuration(self, values: dict[str, Any]) -> None:
        if (
            self._run is None
            or values.get("model_configuration_id") != self._run.model_configuration_id
        ):
            self._reject_model_configuration("checkpoint_identity_mismatch")

    def _validate_checkpoint_compatibility(
        self, snapshot: Any = None, *, require_metadata: bool = True
    ) -> None:
        if self.database is None or self._checkpoint_store is None:
            return
        if self._run is None:
            raise CheckpointCompatibilityError("run context is missing")
        snapshot = snapshot or self._checkpoint_snapshot()
        self._validate_runtime_state(snapshot)
        self._assert_checkpoint_model_configuration(snapshot.values)
        with self.database.session() as session:
            metadata = CheckpointRepository(session).get_for_run(self._run.id)
            if metadata is None:
                if not require_metadata:
                    # A running process can stop before its first domain
                    # checkpoint publication; actual state/provenance still
                    # passed the same runtime reader above.
                    return
                raise CheckpointCompatibilityError(
                    f"checkpoint metadata is missing for run {self._run.id}"
                )
            if (
                metadata.metadata_json.get("model_configuration_id")
                != self._run.model_configuration_id
            ):
                self._reject_model_configuration("checkpoint_identity_mismatch")
            configurable = snapshot.config.get("configurable", {})
            checkpoint_id = configurable.get("checkpoint_id")
            values = snapshot.values
            expected = {
                "thread_id": self._run.thread_id,
                "checkpoint_id": checkpoint_id,
                "workflow_name": self._run.workflow_name,
                "workflow_version": self._run.workflow_version,
                "state_schema_version": self._run.state_schema_version,
                "policy_version": self._run.policy_version,
                "tool_registry_version": self._run.tool_registry_version,
                "model_configuration_id": self._run.model_configuration_id,
                "run_id": str(self._run.id),
            }
            actual = {
                "thread_id": metadata.thread_id,
                "checkpoint_id": metadata.checkpoint_id,
                "workflow_name": metadata.workflow_name,
                "workflow_version": metadata.workflow_version,
                "state_schema_version": metadata.state_schema_version,
                "policy_version": metadata.policy_version,
                "tool_registry_version": metadata.tool_registry_version,
                "model_configuration_id": metadata.metadata_json.get("model_configuration_id"),
                "run_id": values.get("run_id"),
            }
            if expected != actual:
                # Graph checkpoints can advance before domain metadata when the
                # process disappears mid-resume. Accept only a proven descendant
                # of the recorded checkpoint with the same committed decision.
                expected_without_id = {
                    key: value for key, value in expected.items() if key != "checkpoint_id"
                }
                actual_without_id = {
                    key: value for key, value in actual.items() if key != "checkpoint_id"
                }
                ancestor = snapshot
                matched = False
                if expected_without_id == actual_without_id and metadata.decision_id is not None:
                    decision = ApprovalDecisionRepository(session).require(metadata.decision_id)
                    if (
                        decision.run_id == str(self._run.id)
                        and decision.approval_id == metadata.approval_id
                    ):
                        for _ in range(self.max_steps + 20):
                            parent = ancestor.parent_config
                            if not parent:
                                break
                            ancestor = self.last_graph.get_state(parent)
                            if (
                                ancestor.config.get("configurable", {}).get("checkpoint_id")
                                == metadata.checkpoint_id
                            ):
                                matched = True
                                break
                if not matched:
                    raise CheckpointCompatibilityError(
                        "checkpoint metadata does not match the persisted run lineage"
                    )
            if isinstance(self.simulator, PersistentSimulator):
                environment_id = str(self.simulator.environment_identity.environment_id)
                if values.get("environment_id") != environment_id:
                    raise CheckpointCompatibilityError(
                        "checkpoint environment does not match the runtime binding"
                    )
            if self._action is not None and metadata.action_id != str(self._action.id):
                raise CheckpointCompatibilityError("checkpoint action scope does not match the run")
            if self._approval is not None and metadata.approval_id != str(self._approval.id):
                raise CheckpointCompatibilityError(
                    "checkpoint approval scope does not match the run"
                )

    def _persist_checkpoint_metadata(
        self,
        state: dict[str, Any],
        *,
        decision_id: UUID | None = None,
        checkpoint_kind: str | None = None,
    ) -> None:
        if self.database is None or self._checkpoint_store is None:
            return
        if self._run is None:
            raise CheckpointCompatibilityError("run context is missing")
        self._persist_budget_snapshot(state)
        snapshot = self._checkpoint_snapshot()
        configurable = snapshot.config.get("configurable", {})
        phase = str(state.get("phase") or snapshot.values.get("phase") or "unknown")
        kind = checkpoint_kind or (
            "approval_interrupt"
            if state.get("final_outcome") == "approval_required"
            else "workflow"
        )
        with self.database.session() as session:
            # Snapshot loading is independent I/O. Refresh the fence again and
            # commit it with metadata so takeover and metadata cannot interleave.
            self._renew_lease(session)
            CheckpointRepository(session).save(
                run_id=self._run.id,
                thread_id=self._run.thread_id,
                checkpoint_id=str(configurable["checkpoint_id"]),
                checkpoint_kind=kind,
                workflow_name=self._run.workflow_name,
                workflow_version=self._run.workflow_version,
                state_schema_version=self._run.state_schema_version,
                policy_version=self._run.policy_version,
                tool_registry_version=self._run.tool_registry_version,
                phase=phase,
                action_id=self._action.id if self._action is not None else None,
                approval_id=self._approval.id if self._approval is not None else None,
                decision_id=decision_id,
                metadata={
                    **dict(snapshot.metadata or {}),
                    "model_configuration_id": self._run.model_configuration_id,
                },
            )

    def _clean_state(self, raw_state: dict[str, Any]) -> dict[str, Any]:
        state = {key: value for key, value in raw_state.items() if key != "__interrupt__"}
        state["tool_call_count"] = max(state.get("tool_call_count", 0), self._tool_attempts)
        interrupts = raw_state.get("__interrupt__")
        if interrupts:
            interrupt_value = getattr(interrupts[0], "value", self.pending_approval_payload)
            state["interrupt_payload"] = interrupt_value
        return state
