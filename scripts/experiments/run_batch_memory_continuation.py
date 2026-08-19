#!/usr/bin/env python3
"""Run or continue a same-snapshot eight-seed batch memory experiment."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import sys
import uuid
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from zetta.memory.task_memory import (
    TaskMemoryStore,
    canonical_sha256,
    validate_task_memory,
)
from scripts.experiments import run_memory_loop as batch_runner
from scripts.experiments import token_budget
from scripts.experiments.run_paired_episode import _git, _sha256_file, _sha256_tree

SCHEMA_VERSION = 2


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _validate_manifest(manifest: dict[str, Any]) -> None:
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {SCHEMA_VERSION}")
    if manifest.get("model") != "gpt-5.6-terra":
        raise ValueError("batch continuation requires gpt-5.6-terra")
    if manifest.get("reasoning_effort") != "medium":
        raise ValueError("batch continuation requires medium reasoning")
    seeds = [int(seed) for seed in manifest.get("seeds", [])]
    if len(seeds) != 8 or len(set(seeds)) != 8:
        raise ValueError("batch continuation requires eight unique seeds")
    start_round = int(manifest.get("start_round", -1))
    if start_round < 0:
        raise ValueError("start_round must be non-negative")
    if start_round == 0:
        for key in ("initial_memory_path", "initial_memory_sha256"):
            if key not in manifest:
                raise ValueError(f"fresh batch manifest requires {key}")
    if int(manifest.get("max_rounds", 0)) <= int(manifest["start_round"]):
        raise ValueError("max_rounds must exceed start_round")
    if not 1 <= int(manifest.get("global_reservation_cap", 0)) <= 5000:
        raise ValueError("global_reservation_cap must be in [1, 5000]")
    required = (
        "protocol_id",
        "repo",
        "commit",
        "python",
        "output_root",
        "memory_root",
        "ledger_path",
        "token_budget_path",
        "task_key",
        "suite",
        "task",
        "common",
        "frozen_resources",
    )
    for key in required:
        if key not in manifest:
            raise ValueError(f"manifest field is required: {key}")


def _digest(manifest: dict[str, Any]) -> str:
    return canonical_sha256(manifest)


def freeze_manifest(path: Path, manifest: dict[str, Any]) -> None:
    _validate_manifest(manifest)
    _atomic_json(path, manifest)
    path.with_suffix(path.suffix + ".sha256").write_text(
        _digest(manifest) + "\n", encoding="ascii"
    )


def _load_frozen_manifest(path: Path) -> dict[str, Any]:
    manifest = _load_json(path)
    _validate_manifest(manifest)
    digest_path = path.with_suffix(path.suffix + ".sha256")
    if not digest_path.is_file():
        raise RuntimeError("manifest is not frozen")
    if digest_path.read_text(encoding="ascii").strip() != _digest(manifest):
        raise RuntimeError("manifest changed after freeze")
    return manifest


def build_manifest(args: argparse.Namespace) -> dict[str, Any]:
    repo = args.repo.expanduser().resolve()
    generic = repo / "resources" / "libero" / "memory"
    generic_sha, generic_count = _sha256_tree(generic)
    initial_memory = None
    if args.initial_memory is not None:
        initial_memory = _load_json(args.initial_memory.expanduser().resolve())
        validate_task_memory(
            initial_memory,
            expected_task_key=f"{args.suite}/task-{int(args.task)}",
            expected_revision=0,
        )
    if int(args.start_round) == 0 and initial_memory is None:
        raise ValueError("--initial-memory is required when --start-round is zero")
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": args.protocol_id,
        "created_at": batch_runner._utc_now(),
        "repo": str(repo),
        "commit": _git(repo, "rev-parse", "HEAD"),
        "python": str(args.python.expanduser().absolute()),
        "output_root": str(args.output_root.expanduser().resolve()),
        "memory_root": str(args.memory_root.expanduser().resolve()),
        "ledger_path": str(args.ledger.expanduser().resolve()),
        "token_budget_path": str(args.token_budget.expanduser().resolve()),
        "model": "gpt-5.6-terra",
        "reasoning_effort": "medium",
        "suite": args.suite,
        "task": int(args.task),
        "task_key": f"{args.suite}/task-{int(args.task)}",
        "mode": "fresh" if int(args.start_round) == 0 else "resume_existing",
        "seeds": [int(seed) for seed in args.seeds],
        "start_round": int(args.start_round),
        "max_rounds": int(args.max_rounds),
        "max_attempts_per_seed": int(args.max_attempts_per_seed),
        "concurrency": int(args.concurrency),
        "gpus": [int(gpu) for gpu in args.gpus],
        "global_reservation_cap": int(args.global_reservation_cap),
        "common": {
            "max_turns": int(args.max_turns),
            "max_episode_steps": int(args.max_episode_steps),
            "planner_timeout_s": int(args.planner_timeout_s),
            "episode_timeout_s": int(args.episode_timeout_s),
            "disable_sam3": True,
            "hf_hub_offline": True,
            "libero_type": "pro",
        },
        "frozen_resources": {
            "generic_memory": {
                "path": str(generic),
                "sha256": generic_sha,
                "file_count": generic_count,
            },
            "batch_runner_sha256": _sha256_file(
                repo / "scripts" / "experiments" / "run_memory_loop.py"
            ),
            "continuation_runner_sha256": _sha256_file(Path(__file__).resolve()),
        },
        "invalid_attempt_policy": {
            "success_metric": "excluded",
            "memory_update": False,
            "budget": "actual API tokens only, never a flat episode charge",
        },
        "stop_rule": {
            "baseline": "round_0 success_rate",
            "minimum_post_baseline_rounds": 3,
            "condition": (
                "at least two post-baseline complete rounds strictly exceed baseline "
                "and pooled post-baseline rate exceeds baseline"
            ),
        },
    }
    if initial_memory is not None:
        manifest["initial_memory_path"] = str(
            args.initial_memory.expanduser().resolve()
        )
        manifest["initial_memory_sha256"] = canonical_sha256(initial_memory)
    return manifest


def _store(manifest: dict[str, Any]) -> TaskMemoryStore:
    store = TaskMemoryStore(
        manifest["memory_root"],
        task_key=manifest["task_key"],
        suite=manifest["suite"],
        task=int(manifest["task"]),
    )
    initial_path = manifest.get("initial_memory_path")
    if initial_path is None:
        store.initialize()
        return store
    initial = _load_json(Path(initial_path))
    if canonical_sha256(initial) != manifest["initial_memory_sha256"]:
        raise RuntimeError("initial memory changed after manifest freeze")
    validate_task_memory(
        initial,
        expected_task_key=manifest["task_key"],
        expected_revision=0,
    )
    if not store.current_path.exists():
        batch_runner._atomic_json(store.versions_dir / "rev-0000.json", initial)
        batch_runner._atomic_json(store.current_path, initial)
    current = store.load()
    if int(current["revision"]) == 0 and canonical_sha256(current) != canonical_sha256(
        initial
    ):
        raise RuntimeError("batch revision zero differs from the frozen initial memory")
    return store


def analyze(manifest: dict[str, Any], store: TaskMemoryStore) -> dict[str, Any]:
    seeds = [int(seed) for seed in manifest["seeds"]]
    rounds = []
    for round_index in range(int(manifest["max_rounds"])):
        attempts = batch_runner._existing_results(manifest, round_index)
        formal = batch_runner._formal_round_results(attempts, seeds)
        successes = sum(bool(row["success"]) for row in formal)
        rounds.append(
            {
                "round": round_index,
                "complete": len(formal) == len(seeds),
                "valid_seeds": len(formal),
                "attempts": len(attempts),
                "successes": successes,
                "success_rate": successes / len(formal) if formal else None,
                "snapshot_revision": (
                    int(formal[0]["memory_snapshot_revision"]) if formal else None
                ),
                "episode_ids": [row["episode_id"] for row in formal],
            }
        )
    complete = [row for row in rounds if row["complete"]]
    baseline = complete[0]["success_rate"] if complete else None
    post = complete[1:]
    improved = [row for row in post if row["success_rate"] > baseline]
    pooled_successes = sum(int(row["successes"]) for row in post)
    pooled_trials = 8 * len(post)
    pooled_rate = pooled_successes / pooled_trials if pooled_trials else None
    confirmed = bool(
        baseline is not None
        and len(post) >= 3
        and len(improved) >= 2
        and pooled_rate is not None
        and pooled_rate > baseline
    )
    current = store.load()
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": manifest["protocol_id"],
        "generated_at": batch_runner._utc_now(),
        "task_key": manifest["task_key"],
        "rounds": rounds,
        "complete_rounds": len(complete),
        "baseline_rate": baseline,
        "post_baseline_complete_rounds": len(post),
        "improved_rounds": [int(row["round"]) for row in improved],
        "pooled_post_baseline_rate": pooled_rate,
        "confirmation_observed": confirmed,
        "current_memory_revision": int(current["revision"]),
        "current_memory_sha256": canonical_sha256(current),
        "token_budget": token_budget.measure(Path(manifest["token_budget_path"])),
    }


def _write_analysis(manifest: dict[str, Any], store: TaskMemoryStore) -> dict[str, Any]:
    report = analyze(manifest, store)
    _atomic_json(Path(manifest["output_root"]) / "analysis-continuation.json", report)
    return report


def _write_infrastructure_halt(
    manifest: dict[str, Any],
    store: TaskMemoryStore,
    *,
    round_index: int,
    attempt_index: int,
    invalid_episode_ids: list[str],
) -> dict[str, Any]:
    report = analyze(manifest, store)
    report["runtime_halt"] = {
        "reason": "all_new_rollouts_in_wave_infrastructure_invalid",
        "round": int(round_index),
        "attempt_index": int(attempt_index),
        "episode_ids": invalid_episode_ids,
        "halted_at": batch_runner._utc_now(),
    }
    _atomic_json(Path(manifest["output_root"]) / "analysis-continuation.json", report)
    return report


def _episode_job(
    *,
    manifest: dict[str, Any],
    digest: str,
    store: TaskMemoryStore,
    round_index: int,
    seed: int,
    attempt_index: int,
    gpu: int,
    snapshot_path: Path,
    snapshot: dict[str, Any],
    guard_before: dict[str, Any],
) -> dict[str, Any]:
    episode_id = f"r{round_index:02d}-s{seed:02d}-a{attempt_index:02d}"
    result_path = (
        Path(manifest["output_root"])
        / f"round-{round_index:02d}"
        / episode_id
        / "result.json"
    )
    reused_result = result_path.is_file()
    lease_id = f"{manifest['protocol_id']}:{episode_id}"
    lease = token_budget.acquire(
        Path(manifest["token_budget_path"]),
        lease_id=lease_id,
        arm="batch",
    )
    if lease is None:
        return {"budget_denied": True, "episode_id": episode_id, "seed": seed}
    result = batch_runner._run_one_episode(
        manifest={
            **manifest,
            "budget_cap": int(manifest["global_reservation_cap"]),
        },
        manifest_digest=digest,
        round_index=round_index,
        seed=seed,
        attempt_index=attempt_index,
        gpu=gpu,
        snapshot_path=snapshot_path,
        snapshot_sha256=canonical_sha256(snapshot),
        snapshot_revision=int(snapshot["revision"]),
        guard_before=guard_before,
    )
    if not result.get("valid"):
        token_budget.release(
            Path(manifest["token_budget_path"]),
            lease_id=lease_id,
            outcome="infrastructure_invalid",
        )
    return {**result, "_runtime_reused_result": reused_result}


def run(manifest_path: Path) -> dict[str, Any]:
    manifest = _load_frozen_manifest(manifest_path)
    repo = Path(manifest["repo"])
    if _git(repo, "rev-parse", "HEAD") != manifest["commit"]:
        raise RuntimeError("repository commit does not match frozen manifest")
    if _git(repo, "status", "--porcelain"):
        raise RuntimeError("repository must be clean")
    generic = manifest["frozen_resources"]["generic_memory"]
    generic_sha, generic_count = _sha256_tree(Path(generic["path"]))
    if generic_sha != generic["sha256"] or generic_count != int(generic["file_count"]):
        raise RuntimeError("generic memory changed after manifest freeze")
    store = _store(manifest)
    output_root = Path(manifest["output_root"])
    digest = _digest(manifest)
    seeds = [int(seed) for seed in manifest["seeds"]]
    gpus = [int(gpu) for gpu in manifest["gpus"]]

    for round_index in range(int(manifest["start_round"]), int(manifest["max_rounds"])):
        report = _write_analysis(manifest, store)
        if report["confirmation_observed"]:
            return report
        round_dir = output_root / f"round-{round_index:02d}"
        round_dir.mkdir(parents=True, exist_ok=True)
        snapshot_path = round_dir / "memory_snapshot.json"
        snapshot = (
            _load_json(snapshot_path)
            if snapshot_path.exists()
            else store.snapshot(snapshot_path)
        )
        existing = batch_runner._existing_results(manifest, round_index)
        if not existing and canonical_sha256(snapshot) != canonical_sha256(store.load()):
            raise RuntimeError("fresh round snapshot does not match canonical memory")

        # Resume a completed valid rollout whose updater was interrupted.
        for result in sorted(
            (row for row in existing if row.get("valid")),
            key=lambda row: (int(row["attempt_index"]), int(row["seed"])),
        ):
            marker = output_root / "memory-updates" / result["episode_id"] / "accepted.json"
            if not marker.exists():
                batch_runner._update_memory_after_episode(
                    manifest=manifest,
                    store=store,
                    result=result,
                )
            token_budget.release(
                Path(manifest["token_budget_path"]),
                lease_id=f"{manifest['protocol_id']}:{result['episode_id']}",
                outcome="valid_success" if result["success"] else "valid_failure",
            )

        budget_denied = False
        for attempt_index in range(int(manifest["max_attempts_per_seed"])):
            existing = batch_runner._existing_results(manifest, round_index)
            formal = batch_runner._formal_round_results(existing, seeds)
            done = {int(row["seed"]) for row in formal}
            pending = [seed for seed in seeds if seed not in done]
            if not pending:
                break
            guard_before = batch_runner._memory_guard_snapshot(manifest, store)
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=int(manifest["concurrency"])
            ) as pool:
                jobs = [
                    pool.submit(
                        _episode_job,
                        manifest=manifest,
                        digest=digest,
                        store=store,
                        round_index=round_index,
                        seed=seed,
                        attempt_index=attempt_index,
                        gpu=gpus[offset % len(gpus)],
                        snapshot_path=snapshot_path,
                        snapshot=snapshot,
                        guard_before=guard_before,
                    )
                    for offset, seed in enumerate(pending)
                ]
                completed = [job.result() for job in jobs]
            if any(row.get("budget_denied") for row in completed):
                budget_denied = True
            new_results = [
                row
                for row in completed
                if not row.get("budget_denied")
                and not row.get("_runtime_reused_result")
            ]
            guard_after = batch_runner._memory_guard_snapshot(manifest, store)
            for key in (
                "task_memory_sha256",
                "task_memory_revision",
                "generic_memory_sha256",
                "generic_memory_file_count",
                "repo_status",
            ):
                if guard_after.get(key) != guard_before.get(key):
                    raise RuntimeError(f"memory changed during batch rollout wave: {key}")
            if new_results and all(not row.get("valid") for row in new_results):
                return _write_infrastructure_halt(
                    manifest,
                    store,
                    round_index=round_index,
                    attempt_index=attempt_index,
                    invalid_episode_ids=[str(row["episode_id"]) for row in new_results],
                )
            for result in sorted(
                (row for row in completed if row.get("valid")),
                key=lambda row: int(row["seed"]),
            ):
                persisted_result = {
                    key: value
                    for key, value in result.items()
                    if not key.startswith("_runtime_")
                }
                marker = (
                    output_root
                    / "memory-updates"
                    / result["episode_id"]
                    / "accepted.json"
                )
                if not marker.exists():
                    batch_runner._update_memory_after_episode(
                        manifest=manifest,
                        store=store,
                        result=persisted_result,
                    )
                token_budget.release(
                    Path(manifest["token_budget_path"]),
                    lease_id=f"{manifest['protocol_id']}:{result['episode_id']}",
                    outcome="valid_success" if result["success"] else "valid_failure",
                )
            if budget_denied:
                return _write_analysis(manifest, store)

        formal = batch_runner._formal_round_results(
            batch_runner._existing_results(manifest, round_index), seeds
        )
        if len(formal) != len(seeds):
            raise RuntimeError(
                f"round {round_index} has only {len(formal)}/{len(seeds)} valid seeds"
            )
        report = _write_analysis(manifest, store)
        if report["confirmation_observed"]:
            return report
    return _write_analysis(manifest, store)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    init = sub.add_parser("init")
    init.add_argument("--manifest", type=Path, required=True)
    init.add_argument("--protocol-id", required=True)
    init.add_argument("--repo", type=Path, required=True)
    init.add_argument("--python", type=Path, required=True)
    init.add_argument("--output-root", type=Path, required=True)
    init.add_argument("--memory-root", type=Path, required=True)
    init.add_argument("--ledger", type=Path, required=True)
    init.add_argument("--token-budget", type=Path, required=True)
    init.add_argument("--initial-memory", type=Path)
    init.add_argument("--suite", default="libero_10")
    init.add_argument("--task", type=int, default=3)
    init.add_argument("--seeds", type=int, nargs=8, default=list(range(1, 9)))
    init.add_argument("--start-round", type=int, default=0)
    init.add_argument("--max-rounds", type=int, default=64)
    init.add_argument("--max-attempts-per-seed", type=int, default=20)
    init.add_argument("--concurrency", type=int, default=2)
    init.add_argument("--gpus", type=int, nargs="+", default=[2, 6])
    init.add_argument("--global-reservation-cap", type=int, default=2000)
    init.add_argument("--max-turns", type=int, default=100)
    init.add_argument("--max-episode-steps", type=int, default=10000)
    init.add_argument("--planner-timeout-s", type=int, default=1800)
    init.add_argument("--episode-timeout-s", type=int, default=3600)
    run_parser = sub.add_parser("run")
    run_parser.add_argument("--manifest", type=Path, required=True)
    analyze_parser = sub.add_parser("analyze")
    analyze_parser.add_argument("--manifest", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.command == "init":
        manifest = build_manifest(args)
        freeze_manifest(args.manifest, manifest)
        report: dict[str, Any] = {
            "manifest": str(args.manifest),
            "sha256": _digest(manifest),
        }
    else:
        manifest = _load_frozen_manifest(args.manifest)
        store = _store(manifest)
        report = run(args.manifest) if args.command == "run" else _write_analysis(manifest, store)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
