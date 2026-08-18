#!/usr/bin/env python3
# Copyright (c) 2026 RPent Contributors
"""Run the recoverable rollout-evolution state machine until a terminal result.

Preparation remains adapter-owned: this entrypoint consumes the immutable
manifest and tool catalog emitted by an environment adapter. Once prepared,
one invocation resumes Rollout -> Cluster -> Diagnose -> Proposal -> Shadow
Replay -> Same-seed -> Held-out -> Reject/Promote, including child generations
when ``--max-generations`` is greater than one.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from rpent.evolution.jsonio import read_json  # noqa: E402
from rpent.evolution.models import CampaignPhase, CampaignManifest  # noqa: E402
from rpent.evolution.store import CampaignStore  # noqa: E402
from rpent.evolution.supervisor import EvolutionSupervisor  # noqa: E402


def _workers(value: str) -> tuple[str, ...]:
    workers = tuple(part.strip() for part in value.split(",") if part.strip())
    if not workers:
        raise ValueError("at least one worker host is required")
    return workers


def _print(value: dict[str, Any]) -> None:
    print(json.dumps(value, ensure_ascii=False, sort_keys=True, default=str), flush=True)


def _worker_processes(
    template: list[str] | None, *, queue_root: Path, workers: tuple[str, ...]
) -> list[subprocess.Popen[Any]]:
    if not template:
        return []
    processes: list[subprocess.Popen[Any]] = []
    for host in workers:
        values = {"queue_root": str(queue_root), "host": host}
        command = [str(part).format_map(values) for part in template]
        processes.append(subprocess.Popen(command, cwd=str(REPOSITORY_ROOT)))
    return processes


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--queue-root", required=True, type=Path)
    parser.add_argument("--tool-catalog", required=True, type=Path)
    parser.add_argument("--workers", required=True)
    parser.add_argument("--model")
    parser.add_argument("--poll-s", type=float, default=5.0)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--max-generations", type=int, default=1)
    parser.add_argument(
        "--worker-command",
        nargs=argparse.REMAINDER,
        help=(
            "optional final command template using {queue_root} and {host}; "
            "all remaining arguments belong to the worker command"
        ),
    )
    parser.add_argument("--once", action="store_true", help="execute one bounded step")
    return parser


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.poll_s < 0 or args.max_steps < 0 or args.max_generations < 1:
        parser.error("poll and maximum values must be non-negative/positive")
    if args.worker_command == []:
        parser.error("--worker-command requires a command and must be the final option")
    return args


def main() -> int:
    args = _parse_args()

    workers = _workers(args.workers)
    root = args.root.resolve()
    queue_root = args.queue_root.resolve()
    manifest = CampaignManifest.from_dict(read_json(args.manifest))
    store = CampaignStore(root)
    store.initialize(manifest)
    tool_catalog = read_json(args.tool_catalog)
    processes = _worker_processes(
        args.worker_command, queue_root=queue_root, workers=workers
    )
    active_generation = manifest.generation
    steps = 0
    try:
        while True:
            if args.max_steps and steps >= args.max_steps:
                return 3
            store = CampaignStore(root)
            phase = CampaignPhase(store.state()["phase"])
            if phase == CampaignPhase.COMPLETE:
                _print({"action": "complete", "root": str(root), "generation": active_generation})
                if active_generation >= manifest.generation + args.max_generations - 1:
                    return 0
                continuation = root / "analysis" / "generation-continuation.json"
                if not continuation.is_file():
                    return 0
                handoff = read_json(continuation)
                child_root = handoff.get("child_campaign_root")
                if not isinstance(child_root, str) or not child_root:
                    raise ValueError("generation continuation has no child root")
                root = Path(child_root).resolve()
                active_generation += 1
                continue

            supervisor = EvolutionSupervisor(
                campaign_root=root,
                queue_root=queue_root,
                worker_hosts=workers,
                tool_catalog=tool_catalog,
                model=args.model,
            )
            report = supervisor.step()
            steps += 1
            _print(
                {
                    "step": steps,
                    "generation": active_generation,
                    "root": str(root),
                    "phase_before": phase.value,
                    "report": report,
                    "phase_after": CampaignStore(root).state()["phase"],
                }
            )
            if args.once:
                return 0
            if report.get("action", "").startswith("waiting_") and args.poll_s:
                time.sleep(args.poll_s)
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()


if __name__ == "__main__":
    raise SystemExit(main())
