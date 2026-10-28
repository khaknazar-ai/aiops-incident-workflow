"""Offline and explicit local-model evaluation cohorts."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Literal

from agentops.llm import LLMConfigurationError
from agentops.model_identity import model_configuration_identity
from agentops.settings import LLMSettings, create_llm_client
from agentops.simulator import Simulator

from .adversarial import run_adversarial_case
from .baseline import execute_approved_proposal, run_rule_baseline
from .cases import EvaluationManifest, build_manifest
from .durable import run_durable_graph_trajectory
from .graph import run_graph_trajectory
from .metrics import aggregate_metrics
from .provenance import SOURCE_REVISION_SCOPE, working_tree_revision
from .scoring import CaseScore, EvaluationTrajectory, score_cases
from .variants import EvaluationVariantSimulator


@dataclass(frozen=True)
class EvaluationRun:
    manifest: EvaluationManifest
    mode: str
    trajectories: tuple[EvaluationTrajectory, ...]
    scores: tuple[CaseScore, ...]
    source_revision: str
    model_provenance: dict[str, object] | None = None

    @property
    def denominator(self) -> int:
        return len(self.scores)

    def as_dict(self) -> dict[str, object]:
        cohort = {
            "fake": "offline_fake",
            "rule": "offline_rule",
            "graph": "offline_graph_fake",
            "durable": "offline_durable_graph_fake",
            "ablation": "offline_ablation_without_deployment_history",
            "adversarial": "offline_adversarial_rule",
            "variants": "offline_graph_fake_variants",
            "ollama": "live_local_ollama",
        }[self.mode]
        report = {
            "report_version": 2,
            "metrics": aggregate_metrics(self.manifest.cases, self.trajectories, self.scores),
            "manifest": self.manifest.as_dict(),
            "manifest_digest": self.manifest.digest(),
            "source_revision": self.source_revision,
            "source_revision_kind": "working-tree-content-sha256",
            "source_revision_scope": SOURCE_REVISION_SCOPE,
            "mode": self.mode,
            "cohort": cohort,
            "external_llm_benchmark": False,
            "denominator": self.denominator,
            "trajectories": [item.as_dict() for item in self.trajectories],
            "scores": [item.as_dict() for item in self.scores],
        }
        if self.model_provenance is not None:
            report["model_provenance"] = dict(self.model_provenance)
        return report


def run_evaluation(
    *,
    mode: Literal[
        "fake",
        "rule",
        "graph",
        "durable",
        "ablation",
        "adversarial",
        "variants",
        "ollama",
    ] = "fake",
    seed: int = 20260920,
) -> EvaluationRun:
    if mode not in {
        "fake",
        "rule",
        "graph",
        "durable",
        "ablation",
        "adversarial",
        "variants",
        "ollama",
    }:
        raise ValueError(
            "evaluator supports fake, rule, graph, durable, ablation, adversarial, variants "
            "or ollama mode"
        )
    source_revision = working_tree_revision()
    split = (
        "adversarial"
        if mode == "adversarial"
        else "variants"
        if mode == "variants"
        else "canonical"
    )
    manifest = build_manifest(seed, split=split)
    llm_client = None
    model_config = None
    model_provenance = None
    max_active_seconds = 90.0
    if mode == "ollama":
        settings = LLMSettings.from_environment()
        if settings.provider.strip().casefold() != "ollama":
            raise LLMConfigurationError("Ollama evaluation requires the local Ollama provider")
        try:
            max_active_seconds = float(os.getenv("AGENTOPS_RUN_MAX_ACTIVE_SECONDS", "90"))
        except ValueError:
            raise LLMConfigurationError("Ollama evaluation active budget is invalid") from None
        if not math.isfinite(max_active_seconds) or not 0 < max_active_seconds <= 1800:
            raise LLMConfigurationError("Ollama evaluation active budget is invalid")
        llm_client = create_llm_client(settings)
        model_config = settings.model_config_for()
        identity = model_configuration_identity(llm_client, model_config)
        model_provenance = {
            "provider": "ollama",
            "protocol": "ollama_chat",
            "model": settings.model,
            "model_digest": settings.model_digest,
            "configuration_id": identity.configuration_id,
            "temperature": settings.temperature,
            "num_ctx": settings.num_ctx,
            "num_predict": settings.num_predict,
            "think": settings.think,
            "max_active_seconds": max_active_seconds,
            "timeout_seconds": settings.timeout_seconds,
            "max_retries": settings.max_retries,
        }
    trajectories_list = []
    for case in manifest.cases:
        simulator = Simulator()
        if mode == "adversarial":
            trajectory = run_adversarial_case(case)
        elif mode == "variants":
            trajectory = run_graph_trajectory(case, EvaluationVariantSimulator(case.case_id))
        elif mode == "durable":
            trajectory = run_durable_graph_trajectory(case)
        elif mode == "graph":
            trajectory = run_graph_trajectory(case, simulator)
        elif mode == "ollama":
            trajectory = run_graph_trajectory(
                case,
                simulator,
                llm=llm_client,
                model_config=model_config,
                mode="ollama",
                max_active_seconds=max_active_seconds,
            )
        else:
            trajectory = run_rule_baseline(
                case,
                simulator,
                mode=("ablation_without_deployment_history" if mode == "ablation" else mode),
                include_deployment_history=mode != "ablation",
            )
            trajectory = execute_approved_proposal(trajectory, simulator)
        trajectories_list.append(trajectory)
    trajectories = tuple(trajectories_list)
    scores = score_cases(manifest.cases, trajectories)
    return EvaluationRun(
        manifest=manifest,
        mode=mode,
        trajectories=trajectories,
        scores=scores,
        source_revision=source_revision,
        model_provenance=model_provenance,
    )


__all__ = ["EvaluationRun", "run_evaluation"]
