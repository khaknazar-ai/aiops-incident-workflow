"""Local command line entrypoints for reproducible offline operations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from uuid import UUID

from agentops.bootstrap import inspect_environment, seed_environment
from agentops.demo import run_demo
from agentops.evaluation import run_evaluation, write_reports
from agentops.simulator import ScenarioName


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="agentops")
    commands = parser.add_subparsers(dest="command", required=True)
    evaluate = commands.add_parser(
        "evaluate", help="run a frozen offline evaluation or the local Ollama cohort"
    )
    evaluate.add_argument(
        "--mode",
        choices=(
            "fake",
            "rule",
            "graph",
            "durable",
            "ablation",
            "adversarial",
            "variants",
            "ollama",
        ),
        default="fake",
    )
    evaluate.add_argument("--seed", type=int, default=20260920)
    evaluate.add_argument("--output", type=Path, required=True)
    demo = commands.add_parser("demo", help="run the deterministic local workflow demo")
    demo.add_argument(
        "--scenario",
        choices=tuple(scenario.value for scenario in ScenarioName),
        default=ScenarioName.BAD_DEPLOYMENT.value,
    )
    demo.add_argument("--decision", choices=("approve", "reject"), required=True)
    seed = commands.add_parser(
        "seed",
        help="persist one simulator environment in the configured database",
    )
    seed.add_argument(
        "--scenario",
        choices=tuple(scenario.value for scenario in ScenarioName),
        required=True,
    )
    inspect = commands.add_parser(
        "inspect-environment",
        help="read public state for one persisted simulator environment",
    )
    inspect.add_argument("--environment-id", type=UUID, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "evaluate":
        run = run_evaluation(mode=args.mode, seed=args.seed)
        paths = write_reports(run, args.output)
        for path in paths:
            print(path)
        return 0
    if args.command == "demo":
        print(
            json.dumps(
                run_demo(scenario=args.scenario, decision=args.decision), indent=2, sort_keys=True
            )
        )
        return 0
    if args.command == "seed":
        print(json.dumps(seed_environment(args.scenario).as_dict(), indent=2, sort_keys=True))
        return 0
    if args.command == "inspect-environment":
        print(
            json.dumps(inspect_environment(args.environment_id).as_dict(), indent=2, sort_keys=True)
        )
        return 0
    raise RuntimeError(f"unsupported command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
