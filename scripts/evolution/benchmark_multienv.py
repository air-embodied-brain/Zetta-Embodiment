# Copyright (c) 2026 RPent Contributors
"""Benchmark many persistent environment slots on one machine."""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any

# Keep direct ``python scripts/evolution/benchmark_multienv.py`` execution
# independent of whether the project was installed into the active venv.
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from rpent.evolution.capacity import (  # noqa: E402
    DEFAULT_API_CONCURRENCY,
    DEFAULT_SLOT_LADDER,
    CapacityConfig,
    CapacityRules,
    FakeEnvironmentWorker,
    MultiEnvironmentCapacityBenchmark,
    command_worker_factory,
    write_secret_free_report,
)


def _slot_ladder(value: str) -> tuple[int, ...]:
    try:
        slots = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "slots must be comma-separated integers"
        ) from exc
    if not slots:
        raise argparse.ArgumentTypeError("slots cannot be empty")
    return slots


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--worker-mode",
        choices=("fake", "command"),
        default="fake",
    )
    parser.add_argument(
        "--command-template",
        help=(
            "Persistent JSONL worker command. Placeholders: {slot}, {slots}, "
            "{max_api}. The command is never written to the report."
        ),
    )
    parser.add_argument(
        "--worker-log-root",
        type=Path,
        default=None,
        help="Private per-slot stderr directory; never persisted in the public report.",
    )
    parser.add_argument(
        "--progress-log",
        type=Path,
        default=None,
        help="Append-only secret-free JSONL phase and micro-batch progress log.",
    )
    parser.add_argument(
        "--slots",
        type=_slot_ladder,
        default=DEFAULT_SLOT_LADDER,
    )
    parser.add_argument("--episodes-per-slot", type=int, default=2)
    parser.add_argument("--steps-per-episode", type=int, default=8)
    parser.add_argument("--warmup-steps", type=int, default=1)
    parser.add_argument("--operation-timeout-s", type=float, default=120.0)
    parser.add_argument("--sample-interval-s", type=float, default=0.25)
    parser.add_argument(
        "--max-active-environment-operations",
        type=int,
        default=None,
        help=(
            "Maximum resident environments allowed to execute reset/step at once. "
            "Residency is unchanged; excess work waits in scheduler micro-batches."
        ),
    )
    parser.add_argument(
        "--operation-resource-shards",
        type=int,
        default=None,
        help=(
            "Round-robin resident slots across this many independent resource "
            "shards and refill a shard as soon as its prior operation completes."
        ),
    )
    parser.add_argument(
        "--max-active-operations-per-shard",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--max-api-concurrency",
        type=int,
        default=DEFAULT_API_CONCURRENCY,
    )
    parser.add_argument("--api-limited-steps", action="store_true")

    parser.add_argument("--max-infra-invalid-rate", type=float, default=0.02)
    parser.add_argument("--max-reset-p95-s", type=float, default=60.0)
    parser.add_argument("--max-step-p95-s", type=float, default=5.0)
    parser.add_argument("--max-vla-queue-p95-s", type=float, default=2.0)
    parser.add_argument("--max-gpu-memory-fraction", type=float, default=0.92)
    parser.add_argument("--max-cpu-percent", type=float, default=95.0)
    parser.add_argument("--max-rss-mib", type=float)
    parser.add_argument(
        "--min-valid-throughput-per-hour",
        type=float,
        default=25.0,
    )
    parser.add_argument("--continue-after-failure", action="store_true")

    parser.add_argument("--fake-reset-delay-s", type=float, default=0.001)
    parser.add_argument("--fake-step-delay-s", type=float, default=0.001)
    parser.add_argument("--fake-vla-queue-s", type=float, default=0.0)
    parser.add_argument("--fake-fail-at-slots", type=int)
    parser.add_argument("--fake-fail-every-operations", type=int, default=0)

    # Hidden mode implements the same persistent JSONL contract used by real
    # workers. It lets CI exercise subprocess lifecycle and timeout behavior.
    parser.add_argument(
        "--serve-fake-worker", action="store_true", help=argparse.SUPPRESS
    )
    return parser


def _serve_fake_worker(args: argparse.Namespace) -> int:
    worker = FakeEnvironmentWorker(
        slot_index=0,
        total_slots=1,
        reset_delay_s=args.fake_reset_delay_s,
        step_delay_s=args.fake_step_delay_s,
        vla_queue_s=args.fake_vla_queue_s,
        fail_every_operations=args.fake_fail_every_operations,
    )
    for line in sys.stdin:
        try:
            request = json.loads(line)
            operation = request.get("op")
        except (json.JSONDecodeError, AttributeError):
            response = {
                "ok": False,
                "valid": False,
                "infra_invalid": True,
                "failure_class": "worker_protocol",
            }
        else:
            if operation == "health":
                response = {
                    "ok": True,
                    "valid": True,
                    "infra_invalid": False,
                    "failure_class": "none",
                }
            elif operation == "reset":
                result = worker.reset(120.0)
                response = {
                    "ok": result.ok,
                    "valid": result.valid,
                    "infra_invalid": result.infra_invalid,
                    "vla_queue_s": result.vla_queue_s,
                    "failure_class": result.failure_class,
                }
            elif operation == "step":
                result = worker.step(120.0)
                response = {
                    "ok": result.ok,
                    "valid": result.valid,
                    "infra_invalid": result.infra_invalid,
                    "vla_queue_s": result.vla_queue_s,
                    "failure_class": result.failure_class,
                }
            elif operation == "close":
                response = {
                    "ok": True,
                    "valid": True,
                    "infra_invalid": False,
                    "failure_class": "none",
                }
                print(json.dumps(response, separators=(",", ":")), flush=True)
                return 0
            else:
                response = {
                    "ok": False,
                    "valid": False,
                    "infra_invalid": True,
                    "failure_class": "unknown_operation",
                }
        print(json.dumps(response, separators=(",", ":")), flush=True)
    return 0


def _build_config(args: argparse.Namespace) -> CapacityConfig:
    rules = CapacityRules(
        maximum_infra_invalid_rate=args.max_infra_invalid_rate,
        maximum_reset_p95_s=args.max_reset_p95_s,
        maximum_step_p95_s=args.max_step_p95_s,
        maximum_vla_queue_p95_s=args.max_vla_queue_p95_s,
        maximum_gpu_memory_fraction=args.max_gpu_memory_fraction,
        maximum_cpu_percent=args.max_cpu_percent,
        maximum_rss_mib=args.max_rss_mib,
        minimum_valid_throughput_per_hour=args.min_valid_throughput_per_hour,
        stop_after_failed_level=not args.continue_after_failure,
    )
    return CapacityConfig(
        slot_ladder=args.slots,
        episodes_per_slot=args.episodes_per_slot,
        steps_per_episode=args.steps_per_episode,
        warmup_steps=args.warmup_steps,
        operation_timeout_s=args.operation_timeout_s,
        sample_interval_s=args.sample_interval_s,
        maximum_active_environment_operations=(args.max_active_environment_operations),
        operation_resource_shards=args.operation_resource_shards,
        maximum_active_operations_per_shard=(args.max_active_operations_per_shard),
        maximum_api_concurrency=args.max_api_concurrency,
        api_limited_steps=args.api_limited_steps,
        rules=rules,
    )


def main() -> int:
    args = _build_parser().parse_args()
    if args.serve_fake_worker:
        return _serve_fake_worker(args)
    if args.output is None:
        raise SystemExit("--output is required")

    config = _build_config(args)
    if args.worker_mode == "command":
        if not args.command_template:
            raise SystemExit("--command-template is required in command mode")
        factory = command_worker_factory(
            args.command_template,
            config,
            diagnostic_root=args.worker_log_root,
        )
    else:

        def factory(slot: int, slots: int) -> FakeEnvironmentWorker:
            return FakeEnvironmentWorker(
                slot_index=slot,
                total_slots=slots,
                reset_delay_s=args.fake_reset_delay_s,
                step_delay_s=args.fake_step_delay_s,
                vla_queue_s=args.fake_vla_queue_s,
                fail_at_or_above_slots=args.fake_fail_at_slots,
                fail_every_operations=args.fake_fail_every_operations,
            )

    progress_lock = threading.Lock()

    def progress(event: dict[str, Any]) -> None:
        line = json.dumps(event, ensure_ascii=False, sort_keys=True)
        print(line, file=sys.stderr, flush=True)
        if args.progress_log is None:
            return
        args.progress_log.parent.mkdir(parents=True, exist_ok=True)
        with progress_lock, args.progress_log.open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    started = time.perf_counter()
    report = MultiEnvironmentCapacityBenchmark(
        config,
        factory,
        worker_mode=args.worker_mode,
        progress_callback=progress,
    ).run()
    report["total_wall_time_s"] = time.perf_counter() - started
    write_secret_free_report(args.output, report)
    summary = {
        "recommended_slots": report["recommended_slots"],
        "tested_maximum_slots": report["tested_maximum_slots"],
        "completed_ladder": report["completed_ladder"],
        "report": str(args.output),
    }
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0 if report["recommended_slots"] > 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
