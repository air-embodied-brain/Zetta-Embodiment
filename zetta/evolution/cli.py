# Copyright (c) 2026 Zetta Contributors
"""Command line interface for immutable rollout evolution campaigns."""

from __future__ import annotations

import argparse
import json
import signal
import threading
from typing import Any

from zetta.evolution.campaign import (
    analyze_failures,
    enqueue_missing_rollouts,
    ingest_queue_results,
    load_episode_records,
)
from zetta.evolution.gating import (
    evaluate_fixed_heldout_20,
    evaluate_paired_gate,
    evaluate_two_stage_heldout,
)
from zetta.evolution.jsonio import read_json
from zetta.evolution.lifecycle import (
    authorize_provisional_hypothesis,
    authorize_same_seed_threshold_override,
    authorize_shadow_falsification,
    promote_and_complete,
    record_gate_and_advance,
    reject_shadow_candidate,
    run_diagnosis_stage,
    run_proposal_stage,
)
from zetta.evolution.models import CampaignManifest
from zetta.evolution.queue import SharedHostQueue, run_worker, run_worker_pool
from zetta.evolution.store import CampaignStore
from zetta.evolution.supervisor import EvolutionSupervisor, promote_and_spawn_generation


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Zetta VLA rollout evolution")
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run", help="initialize and enqueue missing rollout jobs")
    run.add_argument("--manifest", required=True)
    run.add_argument("--root", required=True)
    run.add_argument("--queue-root", required=True)
    run.add_argument("--workers", required=True, help="comma-separated worker host IDs")

    resume = commands.add_parser(
        "resume", help="ingest completed jobs and re-enqueue missing jobs"
    )
    resume.add_argument("--root", required=True)
    resume.add_argument("--queue-root", required=True)
    resume.add_argument("--workers", required=True)

    analyze = commands.add_parser("analyze", help="cluster valid failed trajectories")
    analyze.add_argument("--root", required=True)

    diagnose = commands.add_parser(
        "diagnose", help="run the audited offline Diagnoser (Stage1 schema)"
    )
    diagnose.add_argument("--root", required=True)
    diagnose.add_argument("--tool-catalog", required=True)
    diagnose.add_argument("--model")

    propose = commands.add_parser(
        "propose", help="run the audited offline Evolver (Stage2 schema)"
    )
    propose.add_argument("--root", required=True)
    propose.add_argument("--tool-catalog", required=True)
    propose.add_argument("--model")

    provisional = commands.add_parser(
        "authorize-provisional",
        help="auditably test one leading hypothesis from a terminal diagnosis",
    )
    provisional.add_argument("--root", required=True)
    provisional.add_argument("--minimum-same-seed-successes", type=int, default=1)
    provisional.add_argument("--skip-regression", action="store_true")
    provisional.add_argument("--deadline", required=True)

    threshold_override = commands.add_parser(
        "authorize-same-seed-threshold-override",
        help="append a candidate-bound reduction to an active same-seed gate",
    )
    threshold_override.add_argument("--root", required=True)
    threshold_override.add_argument(
        "--minimum-same-seed-successes", type=int, required=True
    )
    threshold_override.add_argument("--skip-regression", action="store_true")
    threshold_override.add_argument("--reason", required=True)
    threshold_override.add_argument("--deadline", required=True)
    threshold_override.add_argument("--author", required=True)

    shadow = commands.add_parser(
        "authorize-shadow-falsification",
        help="authorize one candidate-specific timeboxed shadow-FP live test",
    )
    shadow.add_argument("--root", required=True)
    shadow.add_argument("--candidate-output", required=True)
    shadow.add_argument("--max-false-positive-rate", type=float, required=True)
    shadow.add_argument("--deadline", required=True)
    shadow.add_argument("--reason", required=True)

    reject_shadow = commands.add_parser(
        "reject-shadow-candidate",
        help="append-only reject one unregistered candidate with failed shadow preflight",
    )
    reject_shadow.add_argument("--root", required=True)
    reject_shadow.add_argument("--candidate-output", required=True)
    reject_shadow.add_argument("--reason", required=True)

    gate = commands.add_parser("gate", help="evaluate a paired candidate gate")
    gate.add_argument(
        "--kind",
        required=True,
        choices=[
            "same_seed",
            "regression",
            "heldout_10",
            "heldout_20",
            "heldout_50",
        ],
    )
    gate.add_argument("--candidate-sha", required=True)
    gate.add_argument("--parent-sha", default=None)
    gate.add_argument("--candidate-results", required=True)
    gate.add_argument("--parent-results", required=True)
    gate.add_argument(
        "--seeds", required=True, help="comma-separated preregistered seed order"
    )
    gate.add_argument("--root", help="record the decision and advance the campaign")

    promote = commands.add_parser("promote", help="atomically promote and complete")
    promote.add_argument("--root", required=True)

    status = commands.add_parser("status", help="print campaign and queue status")
    status.add_argument("--root", required=True)
    status.add_argument("--queue-root")

    optimize = commands.add_parser(
        "optimize-step", help="execute one idempotent bounded optimization step"
    )
    optimize.add_argument("--root", required=True)
    optimize.add_argument("--queue-root", required=True)
    optimize.add_argument("--workers", required=True)
    optimize.add_argument("--tool-catalog", required=True)
    optimize.add_argument("--model")
    optimize.add_argument("--next-root")
    optimize.add_argument("--next-queue-root")
    optimize.add_argument("--next-master-seed", type=int)

    continuation = commands.add_parser(
        "continue-generation",
        help="promote a fully gated candidate and spawn a preregistered child generation",
    )
    continuation.add_argument("--root", required=True)
    continuation.add_argument("--next-manifest")
    continuation.add_argument("--next-root", required=True)
    continuation.add_argument("--next-queue-root", required=True)
    continuation.add_argument("--workers", required=True)
    continuation.add_argument("--next-master-seed", type=int)

    worker = commands.add_parser(
        "worker", help="run one persistent shared-queue worker"
    )
    worker.add_argument("--queue-root", required=True)
    worker.add_argument("--host", required=True)
    worker.add_argument("--poll-s", type=float, default=2.0)
    worker.add_argument("--once", action="store_true")
    worker.add_argument("--concurrency", type=int, default=1)
    worker.add_argument("--slot-broker-root")
    worker.add_argument("--environment-ready-manifest")
    worker.add_argument("--maximum-active-environment-slots", type=int)

    recover = commands.add_parser(
        "recover-abandoned",
        help="append-only close abandoned queue claims without replaying side effects",
    )
    recover.add_argument("--queue-root", required=True)
    recover.add_argument("--host", required=True)
    recover.add_argument("--stale-after-s", type=float, required=True)
    return parser


def _workers(value: str) -> tuple[str, ...]:
    result = tuple(part.strip() for part in value.split(",") if part.strip())
    if not result:
        raise ValueError("at least one worker is required")
    return result


def _print(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, default=str))


def main() -> int:
    args = _parser().parse_args()
    if args.command == "run":
        manifest = CampaignManifest.from_dict(read_json(args.manifest))
        CampaignStore(args.root).initialize(manifest)
        _print(
            enqueue_missing_rollouts(
                campaign_root=args.root,
                queue_root=args.queue_root,
                worker_hosts=_workers(args.workers),
            )
        )
        return 0
    if args.command == "resume":
        ingestion = ingest_queue_results(
            campaign_root=args.root, queue_root=args.queue_root
        )
        enqueue = enqueue_missing_rollouts(
            campaign_root=args.root,
            queue_root=args.queue_root,
            worker_hosts=_workers(args.workers),
        )
        _print({"ingestion": ingestion, "enqueue": enqueue})
        return 0
    if args.command == "analyze":
        _print(analyze_failures(args.root))
        return 0
    if args.command == "diagnose":
        _print(
            run_diagnosis_stage(
                campaign_root=args.root,
                tool_catalog=read_json(args.tool_catalog),
                model=args.model,
            )
        )
        return 0
    if args.command == "propose":
        _print(
            run_proposal_stage(
                campaign_root=args.root,
                tool_catalog=read_json(args.tool_catalog),
                model=args.model,
            )
        )
        return 0
    if args.command == "authorize-provisional":
        _print(
            authorize_provisional_hypothesis(
                campaign_root=args.root,
                minimum_same_seed_successes=args.minimum_same_seed_successes,
                skip_regression=args.skip_regression,
                deadline=args.deadline,
            )
        )
        return 0
    if args.command == "authorize-same-seed-threshold-override":
        _print(
            authorize_same_seed_threshold_override(
                campaign_root=args.root,
                minimum_same_seed_successes=args.minimum_same_seed_successes,
                skip_regression=args.skip_regression,
                reason=args.reason,
                deadline=args.deadline,
                author=args.author,
            )
        )
        return 0
    if args.command == "authorize-shadow-falsification":
        _print(
            authorize_shadow_falsification(
                campaign_root=args.root,
                candidate_output=args.candidate_output,
                max_false_positive_rate=args.max_false_positive_rate,
                deadline=args.deadline,
                reason=args.reason,
            )
        )
        return 0
    if args.command == "reject-shadow-candidate":
        _print(
            reject_shadow_candidate(
                campaign_root=args.root,
                candidate_output=args.candidate_output,
                reason=args.reason,
            )
        )
        return 0
    if args.command == "gate":
        seeds = tuple(int(value) for value in args.seeds.split(",") if value.strip())
        candidate = load_episode_records(args.candidate_results)
        parent = load_episode_records(args.parent_results)
        if args.kind == "heldout_20":
            decision = evaluate_fixed_heldout_20(
                candidate_sha256=args.candidate_sha,
                parent_sha256=args.parent_sha,
                candidate_records=candidate,
                parent_records=parent,
                preregistered_seeds=seeds,
            )
        elif args.kind.startswith("heldout"):
            decision = evaluate_two_stage_heldout(
                candidate_sha256=args.candidate_sha,
                parent_sha256=args.parent_sha,
                candidate_records=candidate,
                parent_records=parent,
                preregistered_seeds=seeds,
                stage=1 if args.kind == "heldout_10" else 2,
            )
        else:
            decision = evaluate_paired_gate(
                kind=args.kind,
                candidate_sha256=args.candidate_sha,
                parent_sha256=args.parent_sha,
                candidate_records=candidate,
                parent_records=parent,
                expected_seeds=seeds,
            )
        if args.root:
            state = record_gate_and_advance(campaign_root=args.root, decision=decision)
            _print({"decision": decision.as_dict(), "state": state})
        else:
            _print(decision.as_dict())
        return 0 if decision.passed else 2
    if args.command == "promote":
        _print(promote_and_complete(campaign_root=args.root))
        return 0
    if args.command == "status":
        report = CampaignStore(args.root).status()
        if args.queue_root:
            report["queue"] = SharedHostQueue(args.queue_root).counts()
        _print(report)
        return 0
    if args.command == "optimize-step":
        _print(
            EvolutionSupervisor(
                campaign_root=args.root,
                queue_root=args.queue_root,
                worker_hosts=_workers(args.workers),
                tool_catalog=read_json(args.tool_catalog),
                model=args.model,
            ).step()
        )
        return 0
    if args.command == "continue-generation":
        _print(
            promote_and_spawn_generation(
                campaign_root=args.root,
                next_manifest=(
                    CampaignManifest.from_dict(read_json(args.next_manifest))
                    if args.next_manifest
                    else None
                ),
                next_campaign_root=args.next_root,
                next_queue_root=args.next_queue_root,
                worker_hosts=_workers(args.workers),
                next_master_seed=args.next_master_seed,
            )
        )
        return 0
    if args.command == "worker":
        if args.once and args.concurrency != 1:
            raise ValueError("--once requires --concurrency 1")
        if args.concurrency > 1:
            stop = threading.Event()

            def request_stop(_signum: int, _frame: Any) -> None:
                stop.set()

            previous_term = signal.signal(signal.SIGTERM, request_stop)
            previous_int = signal.signal(signal.SIGINT, request_stop)
            try:
                return run_worker_pool(
                    queue_root=args.queue_root,
                    host=args.host,
                    concurrency=args.concurrency,
                    poll_s=args.poll_s,
                    stop_event=stop,
                    environment_slot_broker_root=args.slot_broker_root,
                    environment_ready_manifest=args.environment_ready_manifest,
                    maximum_active_environment_slots=(
                        args.maximum_active_environment_slots
                    ),
                )
            finally:
                signal.signal(signal.SIGTERM, previous_term)
                signal.signal(signal.SIGINT, previous_int)
        return run_worker(
            queue_root=args.queue_root,
            host=args.host,
            poll_s=args.poll_s,
            once=args.once,
            environment_slot_broker_root=args.slot_broker_root,
            environment_ready_manifest=args.environment_ready_manifest,
            maximum_active_environment_slots=args.maximum_active_environment_slots,
        )
    if args.command == "recover-abandoned":
        queue = SharedHostQueue(args.queue_root)
        recovered = queue.recover_abandoned(
            args.host,
            stale_after_s=args.stale_after_s,
        )
        _print(
            {
                "host": args.host,
                "recovered": recovered,
                "queue": queue.counts(),
            }
        )
        return 0
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
