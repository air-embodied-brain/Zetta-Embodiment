#!/usr/bin/env python3
# Copyright (c) 2026 RPent Contributors
"""Run or inspect an append-only paired candidate gate scheduler."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from rpent.evolution.gate_runner import (  # noqa: E402
    CandidateGateRunner,
    PairedGateRunner,
)


def _workers(value: str) -> tuple[str, ...]:
    result = tuple(part.strip() for part in value.split(",") if part.strip())
    if not result:
        raise ValueError("at least one worker host is required")
    return result


def _print(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, default=str))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Schedule paired parent/candidate gate arms"
    )
    parser.add_argument(
        "command", choices=("run", "status", "authorize-infra-recovery")
    )
    parser.add_argument("--root", required=True)
    parser.add_argument("--queue-root", required=True)
    parser.add_argument("--workers", required=True, help="comma-separated host IDs")
    parser.add_argument("--candidate-sha")
    parser.add_argument("--additional-attempts", type=int)
    parser.add_argument("--reason")
    parser.add_argument(
        "--logical-ids",
        help="optional comma-separated exhausted logical arms; defaults to all blocked arms",
    )
    parser.add_argument(
        "--gate-kind",
        choices=("same_seed", "heldout_20", "heldout"),
        default="same_seed",
        help=(
            "paired gate kind; heldout_20 uses the fixed 20-seed contract and "
            "heldout uses the preregistered 10-to-50 contract"
        ),
    )
    args = parser.parse_args()
    runner_type = (
        CandidateGateRunner if args.gate_kind == "same_seed" else PairedGateRunner
    )
    runner_kwargs = {
        "campaign_root": args.root,
        "queue_root": args.queue_root,
        "worker_hosts": _workers(args.workers),
        "candidate_sha256": args.candidate_sha,
    }
    if args.gate_kind != "same_seed":
        runner_kwargs["gate_kind"] = args.gate_kind
    runner = runner_type(**runner_kwargs)
    if args.command == "run":
        result = runner.run_once()
    elif args.command == "status":
        result = runner.status()
    else:
        if args.additional_attempts is None or args.reason is None:
            parser.error(
                "authorize-infra-recovery requires --additional-attempts and --reason"
            )
        logical_ids = (
            tuple(
                part.strip()
                for part in args.logical_ids.split(",")
                if part.strip()
            )
            if args.logical_ids
            else None
        )
        result = runner.authorize_infrastructure_recovery(
            additional_attempts=args.additional_attempts,
            reason=args.reason,
            logical_ids=logical_ids,
        )
    _print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
