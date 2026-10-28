"""Frozen visible-input perturbations for the offline variant cohort."""

from __future__ import annotations

from typing import Any

from agentops.simulator import Simulator


class EvaluationVariantSimulator(Simulator):
    def __init__(self, variant_id: str) -> None:
        super().__init__()
        self.variant_id = variant_id

    def incident_context(self) -> dict[str, str]:
        context = super().incident_context()
        if self.variant_id in {
            "development-description-paraphrase",
            "held-out-description-log-order",
        }:
            context["description"] = (
                "Customer impact is unchanged. Investigate the observed service signals: "
                + context["description"]
            )
        return context

    def search_logs(self, service: str, query: str, limit: int) -> dict[str, Any]:
        payload = super().search_logs(service, query, limit)
        if self.variant_id in {
            "development-log-order",
            "held-out-description-log-order",
        }:
            payload["entries"] = list(reversed(payload["entries"]))
        return payload


__all__ = ["EvaluationVariantSimulator"]
