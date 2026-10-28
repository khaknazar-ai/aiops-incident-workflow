"""Independent evaluator package."""

from .adversarial import PROMPT_INJECTION_LOG, AdversarialSimulatorView, run_adversarial_case
from .baseline import execute_approved_proposal, run_rule_baseline
from .cases import (
    EvaluationCase,
    EvaluationManifest,
    adversarial_cases,
    build_manifest,
    canonical_cases,
    manifest_digest,
    variant_cases,
)
from .durable import run_durable_graph_trajectory
from .graph import run_graph_trajectory
from .provenance import working_tree_revision
from .reports import write_reports
from .runner import EvaluationRun, run_evaluation
from .scoring import CaseScore, EvaluationTrajectory, score_case, score_cases

__all__ = [
    "AdversarialSimulatorView",
    "CaseScore",
    "EvaluationCase",
    "EvaluationManifest",
    "EvaluationRun",
    "EvaluationTrajectory",
    "PROMPT_INJECTION_LOG",
    "adversarial_cases",
    "build_manifest",
    "execute_approved_proposal",
    "canonical_cases",
    "manifest_digest",
    "run_adversarial_case",
    "run_durable_graph_trajectory",
    "run_evaluation",
    "run_graph_trajectory",
    "run_rule_baseline",
    "score_case",
    "score_cases",
    "variant_cases",
    "write_reports",
    "working_tree_revision",
]
