"""Public incident task contracts; no scenario labels or expected answers."""

import json
from uuid import UUID


def incident_task_instructions(task: str) -> str:
    rules = {
        "triage": (
            "Assess urgency from the description; do not infer a diagnosis without observations."
            " severity_assessment must be SEV1, SEV2, SEV3 or SEV4."
            " needs_immediate_escalation means stop this investigation before any diagnostic read, "
            "not notify an on-call person. Set it only when the description establishes that "
            "bounded read-only investigation itself is unsafe or outside this service scope. "
            "High business impact alone does not prevent read-only diagnosis."
        ),
        "diagnostic_plan": (
            "Choose at most four distinct diagnostic reads from the tool catalog; reserve "
            "the remaining workflow steps for approval, execution, verification and review. "
            "Each step is exactly one tool name. Collect independent evidence to distinguish "
            "deployment, resource, dependency, worker and configuration problems. When the "
            "description mentions a configuration change or inability to connect to or resolve "
            "a dependency, prioritize service configuration and a broad log sample; use "
            "deployment history only when rollout evidence is relevant. Do not classify a "
            "dependency outage from health status alone. "
            "Do not include remediation or post-action verification in a diagnostic plan."
        ),
        "tool_selection": (
            "Select the one read tool for the supplied step. Copy the supplied service exactly. "
            "Use only that tool's input fields; omit other optional fields. "
            "query_metrics requires a nonempty metric_names list; search_logs requires query. "
            "Available metric names: p95_latency_ms, error_rate, request_rate, pool_active, "
            "pool_max, pool_waiters, queue_depth, incoming_rate, processing_rate, "
            "oldest_message_age_seconds, fraud_api_5xx, dns_errors. "
            "Use the incident description and observations collected so far to choose "
            "relevant read parameters; absent observations are unknown. "
            "For search_logs request a broad bounded sample (query is one space, limit 20), "
            "rather than inventing a keyword that may silently omit the incident evidence."
            " This returns the latest bounded records in stored order. Inspect truncated and "
            "total_matches: omitted history is unknown, and no match is not proof of no fault."
        ),
        "hypothesis_update": (
            "Create or update a concise structured hypothesis only when the fact catalog supports "
            "a relationship between distinct facts. A hypothesis is tentative: may_explain means "
            "a possible explanation, associated_with means correlation without proving causality. "
            "Do not require a proven root cause before recording a supported tentative relation. "
            "Never invent IDs. It is valid to emit no "
            "hypotheses when evidence is insufficient. New hypotheses use null id/version/"
            "previous_decision_id; existing ones copy those fields from the supplied snapshot "
            "(previous_decision_id copies decision_id). Include the observations owning both claim "
            "facts in supporting_observation_ids. For diagnosis.evidence_refs use observed tool "
            "names. A non-null diagnosis requires a matching category in an existing or newly "
            "returned hypothesis with a structured claim. Without that claim, diagnosis must "
            "be null, even for category unknown. Continue the remaining plan when evidence is "
            "insufficient; do not mistake "
            "an unhealthy service or its current version alone for a proven cause. "
            "recommended_next_focus is a short label, not an instruction to execute anything."
            " Reassess category, confidence and claim against all current observations each time; "
            "preserve hypothesis identity, not an outdated judgment. A tentative explanation "
            "does not need certainty, but must have supporting observations and no ignored "
            "contradiction. Distinguish affected local components from unavailable dependencies. "
            "An unreachable dependency or DNS error is a symptom, not proof of a remote outage. "
            "Compare observed endpoint configuration with resolution and validation logs. "
            "Name-resolution or configuration-validation failures weigh against a remote-outage "
            "diagnosis; require evidence that the configured endpoint is valid and requests "
            "reach the remote service before classifying a remote outage. If that distinction "
            "is unresolved, keep the diagnosis unknown and continue available reads."
            " Every returned hypothesis must include its structured claim; copy a retained "
            "claim explicitly when unchanged. If no meaningful claim is supported, return "
            "an empty hypotheses list rather than a claim-less placeholder."
        ),
        "remediation_proposal": (
            "Propose only a catalog action supported by the observations and hypotheses. "
            "A rollback target must be an observed stable deployment version other than the "
            "currently running version. "
            "A bounded temporary mitigation may be proposed when supported by observed "
            "conditions and the action contract, even if permanent correction is outside "
            "the available tools. In justification distinguish the expected immediate benefit "
            "from unresolved cause, recurrence and operational risks. Do not assume guarantees "
            "absent from the contract, or treat approval as proof that an action is safe. "
            "Use restart_service only when observations support an independent local-process "
            "fault; restarting a service does not repair an invalid dependency endpoint or a "
            "remote dependency outage. Do not treat an unreachable dependency alone as a "
            "reason to restart the affected service. If no catalog action can correct the "
            "observed configuration problem, escalate. "
            "If evidence is insufficient or no safe catalog action can correct the cause "
            "or mitigate its observed impact, "
            "use action_type=escalate, tool_name=null, risk_level=3. "
            "For escalation arguments must be empty. A write proposal must list every distinct "
            "observed tool name in evidence_ids exactly once, including observations that "
            "contradict or leave uncertainty. This is the complete source set considered, not a "
            "claim that every source supports the action; explain in justification which sources "
            "support or contradict it. If evidence is incomplete, contradictory, or fails to "
            "substantiate the target or action, escalate. "
            "Otherwise use action_type equal to tool_name and the catalog risk level. "
            "Copy the service; omit idempotency_key (the runtime owns it). "
            "A proposal is not execution; the server must authorize and verify it."
        ),
        "review": (
            "outcome is resolved only if healthy is true and every supplied verification check "
            "is true; otherwise outcome is escalated. Never claim success without verification."
        ),
    }
    text = rules.get(task, "")
    if task in {"diagnostic_plan", "tool_selection", "remediation_proposal"}:
        # Registry metadata only: no simulator scenario is loaded or inspected.
        from agentops.simulator import Simulator
        from agentops.tools import ToolGateway

        tools = ToolGateway(Simulator()).definitions()
        catalog = [
            {
                "name": item.name,
                "description": item.description,
                "risk_level": item.risk_level.value,
                "required_capability": item.required_capability,
                "reject_known_healthy_target": item.reject_known_healthy_target,
                "input_schema": item.input_model.model_json_schema(),
            }
            for item in tools
            if item.side_effecting == (task == "remediation_proposal")
            and item.name != "verify_service"
        ]
        text += " Public tool catalog: " + json.dumps(catalog, separators=(",", ":"))
    return text


def native_response_schema(task: str, response_model: type, context: dict | None = None) -> dict:
    """Expose the real read-tool argument requirements, not the loose union DTO alone."""
    from agentops.llm import RemediationProposal, ToolDecision

    is_proposal = task == "remediation_proposal" and response_model is RemediationProposal
    schema = response_model.model_json_schema()
    _remove_generation_defaults(schema)
    if task == "diagnostic_plan":
        schema["properties"]["steps"]["maxItems"] = 4
    context = context or {}
    observed_tools = sorted(
        {
            item["tool_name"]
            for item in context.get("observations", [])
            if isinstance(item, dict) and isinstance(item.get("tool_name"), str)
        }
    )
    rollback_targets = _observed_stable_rollback_targets(context) if is_proposal else []
    if task in {"hypothesis_update", "remediation_proposal"} and observed_tools:
        if task == "hypothesis_update":
            schema["$defs"]["DiagnosisAssessment"]["properties"]["evidence_refs"]["items"] = {
                "type": "string",
                "enum": observed_tools,
            }
        else:
            schema["properties"]["evidence_ids"]["items"] = {
                "type": "string",
                "enum": observed_tools,
            }
    if task == "hypothesis_update":
        # Require an explicit answer, including []/null when evidence is insufficient.
        schema["required"] = list(schema["properties"])
        hypothesis = schema["$defs"]["Hypothesis"]
        hypothesis["properties"]["claim"] = {"$ref": "#/$defs/HypothesisClaim"}
        hypothesis["required"] = list(
            dict.fromkeys(hypothesis["required"] + ["claim", "category", "status"])
        )
        facts = context.get("fact_catalog", [])
        fact_ids, observation_ids = set(), set()
        for item in facts:
            if not isinstance(item, dict):
                continue
            try:
                fact_ids.add(str(UUID(str(item["id"]))))
                observation_ids.add(str(UUID(str(item["observation_id"]))))
            except (KeyError, ValueError, TypeError):
                continue
        if fact_ids:
            for key in ("cause_fact_id", "effect_fact_id"):
                schema["$defs"]["HypothesisClaim"]["properties"][key]["enum"] = sorted(fact_ids)
        if observation_ids:
            for key in ("supporting_observation_ids", "contradicting_observation_ids"):
                schema["$defs"]["Hypothesis"]["properties"][key]["items"]["enum"] = sorted(
                    observation_ids
                )
    if not is_proposal and (task != "tool_selection" or response_model is not ToolDecision):
        return schema
    from agentops.simulator import Simulator
    from agentops.tools import ToolGateway

    variants = []
    for tool in ToolGateway(Simulator()).definitions():
        if tool.side_effecting != is_proposal or tool.name == "verify_service":
            continue
        if is_proposal and not observed_tools:
            continue
        if is_proposal and tool.name == "rollback_deployment" and not rollback_targets:
            continue
        if (
            is_proposal
            and tool.observed_support(context.get("service"), context.get("observations")) is False
        ):
            continue
        arguments = tool.input_model.model_json_schema()
        _remove_generation_defaults(arguments)
        arguments["additionalProperties"] = False
        if not is_proposal and tool.name == "search_logs":
            # An initial read has no observed keyword to search for. Preserve a
            # bounded evidence sample instead of silently treating no match as no fault.
            arguments["properties"]["query"]["const"] = " "
            arguments["properties"]["limit"]["const"] = 20
            arguments["required"] = list(arguments["properties"])
        properties = {
            "tool_name": {"type": "string", "const": tool.name},
            "arguments": arguments,
        }
        if is_proposal:
            if tool.name == "rollback_deployment":
                arguments["properties"]["target_version"]["enum"] = rollback_targets
            properties.update(
                {
                    "action_type": {"type": "string", "const": tool.name},
                    "risk_level": {"type": "integer", "const": tool.risk_level.value},
                    "justification": schema["properties"]["justification"],
                    "evidence_ids": schema["properties"]["evidence_ids"]
                    | {
                        "minItems": len(observed_tools),
                        "maxItems": len(observed_tools),
                        "uniqueItems": True,
                    },
                }
            )
        else:
            properties["purpose"] = {"type": "string"}
        variants.append(
            {
                "type": "object",
                "properties": properties,
                "required": list(properties),
                "additionalProperties": False,
            }
        )
    if is_proposal:
        properties = {
            "action_type": {"type": "string", "const": "escalate"},
            "tool_name": {"type": "null"},
            "arguments": {"type": "object", "properties": {}, "additionalProperties": False},
            "risk_level": {"type": "integer", "const": 3},
            "justification": schema["properties"]["justification"],
            "evidence_ids": schema["properties"]["evidence_ids"],
        }
        variants.append(
            {
                "type": "object",
                "properties": properties,
                "required": list(properties),
                "additionalProperties": False,
            }
        )
    return {"title": schema["title"], "oneOf": variants}


def _observed_stable_rollback_targets(context: dict) -> list[str]:
    service = context.get("service")
    observations = context.get("observations")
    if not isinstance(service, str) or not service or not isinstance(observations, list):
        return []

    latest: dict[str, dict] = {}
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
        return []
    current_version = status.get("version")
    deployments = history.get("deployments")
    if (
        not isinstance(current_version, str)
        or not current_version
        or not isinstance(deployments, list)
    ):
        return []

    return sorted(
        {
            deployment["version"]
            for deployment in deployments
            if isinstance(deployment, dict)
            and isinstance(deployment.get("version"), str)
            and deployment["version"]
            and deployment["version"] != current_version
            and deployment.get("status") == "stable"
        }
    )


def _remove_generation_defaults(value: object) -> None:
    """DTO defaults are compatibility rules, not suggested model judgments."""
    if isinstance(value, dict):
        value.pop("default", None)
        for child in value.values():
            _remove_generation_defaults(child)
    elif isinstance(value, list):
        for child in value:
            _remove_generation_defaults(child)
