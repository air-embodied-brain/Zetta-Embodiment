#!/usr/bin/env python3
"""Run strictly sequential, post-episode task-memory experiments.

Unlike ``run_memory_loop.py`` (whose seeds in a batch share one snapshot), this
runner makes every valid rollout a dependency boundary:

    snapshot(n) -> rollout(n) -> memory update(n) -> snapshot(n + 1)

Infrastructure-invalid attempts consume reservation budget but do not update
memory or enter the success metric. Two arms are supported: cycling through a
configurable seed set and repeatedly evaluating one fixed seed.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
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
from scripts.experiments.run_paired_episode import _git, _sha256_file

SCHEMA_VERSION = 2
LEGACY_WINDOW_SIZE = 8
ARMS = {"cycling", "fixed"}
MAX_CONSECUTIVE_INFRASTRUCTURE_INVALID = 3


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _window_size(manifest: dict[str, Any]) -> int:
    """Return the reporting block size, preserving schema-v2 compatibility."""

    return int(manifest.get("window_size", LEGACY_WINDOW_SIZE))


def _validate_manifest(manifest: dict[str, Any]) -> None:
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {SCHEMA_VERSION}")
    if manifest.get("model") != "gpt-5.6-terra":
        raise ValueError("sequential memory protocol requires gpt-5.6-terra")
    if manifest.get("reasoning_effort") != "medium":
        raise ValueError("sequential memory protocol requires medium reasoning")
    if manifest.get("arm") not in ARMS:
        raise ValueError(f"arm must be one of {sorted(ARMS)}")
    seed_cycle = [int(seed) for seed in manifest.get("seed_cycle", [])]
    window_size = _window_size(manifest)
    if not 1 <= window_size <= 100:
        raise ValueError("window_size must be in [1, 100]")
    if manifest["arm"] == "cycling":
        if len(seed_cycle) != window_size or len(set(seed_cycle)) != window_size:
            raise ValueError("cycling arm requires window_size unique seeds")
    elif seed_cycle != [int(manifest.get("fixed_seed", -1))]:
        raise ValueError("fixed arm seed_cycle must contain only fixed_seed")
    if "early_stop_on_metric" in manifest and not isinstance(
        manifest["early_stop_on_metric"], bool
    ):
        raise ValueError("early_stop_on_metric must be a boolean")
    global_cap = int(manifest.get("global_reservation_cap", 0))
    arm_cap = int(manifest.get("arm_reservation_cap", 0))
    max_valid = int(manifest.get("max_valid_episodes", 0))
    if not 1 <= global_cap <= 5000:
        raise ValueError("global_reservation_cap must be in [1, 5000]")
    if not 1 <= arm_cap <= global_cap:
        raise ValueError("arm_reservation_cap must be in [1, global_reservation_cap]")
    if not 1 <= max_valid <= arm_cap:
        raise ValueError("max_valid_episodes must be in [1, arm_reservation_cap]")
    if int(manifest.get("ceiling_confirmation_blocks", 0)) < 2:
        raise ValueError("ceiling_confirmation_blocks must be at least two")
    if int(manifest.get("round_offset", -1)) < 0:
        raise ValueError("round_offset must be non-negative")
    required = (
        "protocol_id",
        "repo",
        "python",
        "output_root",
        "memory_root",
        "ledger_path",
        "commit",
        "suite",
        "task_key",
        "initial_memory_path",
        "initial_memory_sha256",
        "token_budget_path",
        "frozen_resources",
        "common",
    )
    for key in required:
        if key not in manifest:
            raise ValueError(f"manifest field is required: {key}")
    common = manifest["common"]
    if not isinstance(common, dict):
        raise ValueError("common must be an object")
    if common.get("tool_profile", "all") not in {"all", "pi05_only"}:
        raise ValueError("common.tool_profile must be all or pi05_only")
    if common.get("disable_sam3") and common.get("sam3_endpoint"):
        raise ValueError("disable_sam3 and sam3_endpoint are mutually exclusive")


def _manifest_digest(manifest: dict[str, Any]) -> str:
    return canonical_sha256(manifest)


def freeze_manifest(path: Path, manifest: dict[str, Any]) -> None:
    _validate_manifest(manifest)
    batch_runner._atomic_json(path, manifest)
    path.with_suffix(path.suffix + ".sha256").write_text(
        _manifest_digest(manifest) + "\n", encoding="ascii"
    )


def _load_frozen_manifest(path: Path) -> dict[str, Any]:
    manifest = _load_json(path)
    _validate_manifest(manifest)
    digest_path = path.with_suffix(path.suffix + ".sha256")
    if not digest_path.is_file():
        raise RuntimeError("manifest is not frozen")
    if digest_path.read_text(encoding="ascii").strip() != _manifest_digest(manifest):
        raise RuntimeError("manifest changed after freeze")
    return manifest


def build_manifest(args: argparse.Namespace) -> dict[str, Any]:
    repo = args.repo.expanduser().resolve()
    initial_memory = _load_json(args.initial_memory.expanduser().resolve())
    validate_task_memory(initial_memory, expected_task_key=f"{args.suite}/task-{args.task}")
    if int(initial_memory["revision"]) != 0:
        raise ValueError("both sequential arms must start from revision zero")
    seed_cycle = list(args.seeds) if args.arm == "cycling" else [args.fixed_seed]
    window_size = (
        int(args.window_size)
        if args.window_size is not None
        else len(seed_cycle) if args.arm == "cycling" else LEGACY_WINDOW_SIZE
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": args.protocol_id,
        "created_at": batch_runner._utc_now(),
        "arm": args.arm,
        "seed_cycle": [int(seed) for seed in seed_cycle],
        "window_size": window_size,
        "fixed_seed": int(args.fixed_seed),
        "early_stop_on_metric": not bool(args.run_to_max_valid),
        "global_reservation_cap": int(args.global_reservation_cap),
        "arm_reservation_cap": int(args.arm_reservation_cap),
        "max_valid_episodes": int(args.max_valid_episodes),
        "ceiling_confirmation_blocks": int(args.ceiling_confirmation_blocks),
        "round_offset": int(args.round_offset),
        "gpu": int(args.gpu),
        "suite": args.suite,
        "task": int(args.task),
        "task_key": f"{args.suite}/task-{int(args.task)}",
        "repo": str(repo),
        "commit": _git(repo, "rev-parse", "HEAD"),
        "python": str(args.python.expanduser().absolute()),
        "output_root": str(args.output_root.expanduser().resolve()),
        "memory_root": str(args.memory_root.expanduser().resolve()),
        "ledger_path": str(args.ledger.expanduser().resolve()),
        "initial_memory_path": str(args.initial_memory.expanduser().resolve()),
        "initial_memory_sha256": canonical_sha256(initial_memory),
        "token_budget_path": str(args.token_budget.expanduser().resolve()),
        "model": "gpt-5.6-terra",
        "reasoning_effort": "medium",
        "common": {
            "max_turns": int(args.max_turns),
            "max_episode_steps": int(args.max_episode_steps),
            "planner_timeout_s": int(args.planner_timeout_s),
            "episode_timeout_s": int(args.episode_timeout_s),
            "base_url": args.base_url,
            "use_local_adapter_key_placeholder": bool(
                args.use_local_adapter_key_placeholder
            ),
            "disable_sam3": args.sam3_endpoint is None,
            "sam3_endpoint": args.sam3_endpoint,
            "vla_endpoint": args.vla_endpoint,
            "contact_graspnet_endpoint": args.contact_graspnet_endpoint,
            "graspgen_endpoint": args.graspgen_endpoint,
            "tool_profile": args.tool_profile,
            "require_camera_health": True,
            "hf_hub_offline": True,
            "libero_type": "pro",
        },
        "frozen_resources": {
            "generic_memory": batch_runner._generic_memory_descriptor(
                repo, disabled=bool(args.disable_generic_memory)
            ),
            "batch_runner_sha256": _sha256_file(
                repo / "scripts" / "experiments" / "run_memory_loop.py"
            ),
            "sequential_runner_sha256": _sha256_file(Path(__file__).resolve()),
        },
        "stop_rule": {
            "metric": f"non_overlapping_{window_size}_episode_success_rate",
            "baseline": f"first_complete_{window_size}-valid-episode block",
            "condition": "later_complete_block_rate > baseline_rate",
            "also_report": f"rolling_last_{window_size}_after_every_valid_episode",
            "early_stop_on_metric": not bool(args.run_to_max_valid),
        },
    }


def _initialize_store(manifest: dict[str, Any]) -> TaskMemoryStore:
    store = TaskMemoryStore(
        manifest["memory_root"],
        task_key=manifest["task_key"],
        suite=manifest["suite"],
        task=int(manifest["task"]),
    )
    initial = _load_json(Path(manifest["initial_memory_path"]))
    if canonical_sha256(initial) != manifest["initial_memory_sha256"]:
        raise RuntimeError("initial memory changed after manifest freeze")
    validate_task_memory(initial, expected_task_key=manifest["task_key"], expected_revision=0)
    if not store.current_path.exists():
        batch_runner._atomic_json(store.versions_dir / "rev-0000.json", initial)
        batch_runner._atomic_json(store.current_path, initial)
    store.load()
    return store


def _seed_for_step(manifest: dict[str, Any], step_index: int) -> int:
    seeds = [int(seed) for seed in manifest["seed_cycle"]]
    return seeds[step_index % len(seeds)]


def _round_for_step(manifest: dict[str, Any], step_index: int) -> int:
    return int(manifest["round_offset"]) + step_index


def _results_for_step(manifest: dict[str, Any], step_index: int) -> list[dict[str, Any]]:
    round_index = _round_for_step(manifest, step_index)
    root = Path(manifest["output_root"]) / f"round-{round_index:02d}"
    return sorted(
        (_load_json(path) for path in root.glob("r*/result.json")),
        key=lambda row: int(row.get("attempt_index", 0)),
    )


def _step_summary_path(manifest: dict[str, Any], step_index: int) -> Path:
    return (
        Path(manifest["output_root"])
        / "steps"
        / f"step-{step_index:03d}"
        / "summary.json"
    )


def _completed_steps(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    root = Path(manifest["output_root"]) / "steps"
    values = [_load_json(path) for path in root.glob("step-*/summary.json")]
    return sorted(values, key=lambda row: int(row["step_index"]))


def _ledger_counts(manifest: dict[str, Any]) -> tuple[int, int]:
    total = 0
    arm = 0
    path = Path(manifest["ledger_path"])
    if not path.is_file():
        return total, arm
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("event") != "reserved":
            continue
        total += 1
        arm += row.get("protocol_id") == manifest["protocol_id"]
    return total, arm


def analyze(manifest: dict[str, Any], store: TaskMemoryStore) -> dict[str, Any]:
    steps = _completed_steps(manifest)
    window_size = _window_size(manifest)
    rolling = []
    for end in range(window_size - 1, len(steps)):
        window = steps[end - window_size + 1 : end + 1]
        successes = sum(bool(row["success"]) for row in window)
        rolling.append(
            {
                "end_step": end,
                "episode_ids": [row["episode_id"] for row in window],
                "successes": successes,
                "success_rate": successes / window_size,
            }
        )
    blocks = []
    for start in range(0, len(steps), window_size):
        window = steps[start : start + window_size]
        if len(window) != window_size:
            break
        successes = sum(bool(row["success"]) for row in window)
        blocks.append(
            {
                "block": start // window_size,
                "start_step": start,
                "end_step": start + window_size - 1,
                "seeds": [int(row["seed"]) for row in window],
                "successes": successes,
                "success_rate": successes / window_size,
                "episode_ids": [row["episode_id"] for row in window],
            }
        )
    baseline_rate = blocks[0]["success_rate"] if blocks else None
    improved_block = None
    if baseline_rate is not None:
        improved_block = next(
            (row["block"] for row in blocks[1:] if row["success_rate"] > baseline_rate),
            None,
        )
    rolling_improved_at = None
    if baseline_rate is not None:
        rolling_improved_at = next(
            (
                row["end_step"]
                for row in rolling
                if row["end_step"] >= (2 * window_size - 1)
                and row["success_rate"] > baseline_rate
            ),
            None,
        )
    total_reserved, arm_reserved = _ledger_counts(manifest)
    current = store.load()
    attempts = sum(len(_results_for_step(manifest, index)) for index in range(len(steps) + 1))
    budget = token_budget.measure(Path(manifest["token_budget_path"]))
    terminal_reason = None
    early_stop = bool(manifest.get("early_stop_on_metric", True))
    if early_stop and improved_block is not None:
        terminal_reason = "strict_block_improvement"
    elif early_stop and baseline_rate == 1.0 and len(blocks) >= int(
        manifest["ceiling_confirmation_blocks"]
    ):
        terminal_reason = "ceiling_confirmed"
    elif len(steps) >= int(manifest["max_valid_episodes"]):
        terminal_reason = "max_valid_episodes"
    report = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": manifest["protocol_id"],
        "generated_at": batch_runner._utc_now(),
        "arm": manifest["arm"],
        "task_key": manifest["task_key"],
        "seed_cycle": manifest["seed_cycle"],
        "window_size": window_size,
        "early_stop_on_metric": early_stop,
        "valid_episodes": len(steps),
        "attempt_results": attempts,
        "successes": sum(bool(row["success"]) for row in steps),
        "blocks": blocks,
        "rolling_last_n": rolling,
        "baseline_rate": baseline_rate,
        "improved_block": improved_block,
        "rolling_improved_at": rolling_improved_at,
        "improvement_observed": improved_block is not None,
        "terminal_reason": terminal_reason,
        "global_reservations": total_reserved,
        "arm_reservations": arm_reserved,
        "global_reservation_cap": int(manifest["global_reservation_cap"]),
        "arm_reservation_cap": int(manifest["arm_reservation_cap"]),
        "token_budget": budget,
        "current_memory_revision": int(current["revision"]),
        "current_memory_sha256": canonical_sha256(current),
    }
    report[f"rolling_last_{window_size}"] = rolling
    return report


def _write_analysis(manifest: dict[str, Any], store: TaskMemoryStore) -> dict[str, Any]:
    report = analyze(manifest, store)
    batch_runner._atomic_json(Path(manifest["output_root"]) / "analysis.json", report)
    return report


def _write_infrastructure_halt(
    manifest: dict[str, Any],
    store: TaskMemoryStore,
    *,
    step_index: int,
    consecutive_invalid: int,
) -> dict[str, Any]:
    report = analyze(manifest, store)
    report["runtime_halt"] = {
        "reason": "consecutive_infrastructure_invalid",
        "step_index": int(step_index),
        "consecutive_invalid_in_invocation": int(consecutive_invalid),
        "threshold": MAX_CONSECUTIVE_INFRASTRUCTURE_INVALID,
        "halted_at": batch_runner._utc_now(),
    }
    batch_runner._atomic_json(Path(manifest["output_root"]) / "analysis.json", report)
    return report


def _record_step(
    manifest: dict[str, Any],
    store: TaskMemoryStore,
    *,
    step_index: int,
    result: dict[str, Any],
    update: dict[str, Any],
) -> dict[str, Any]:
    summary = {
        "step_index": step_index,
        "round_index": _round_for_step(manifest, step_index),
        "seed": int(result["seed"]),
        "episode_id": result["episode_id"],
        "attempt_index": int(result["attempt_index"]),
        "valid": True,
        "success": bool(result["success"]),
        "memory_snapshot_revision": int(result["memory_snapshot_revision"]),
        "memory_snapshot_sha256": result["memory_snapshot_sha256"],
        "memory_update_revision": int(update["revision"]),
        "memory_update_sha256": update["memory_sha256"],
        "completed_at": batch_runner._utc_now(),
    }
    batch_runner._atomic_json(_step_summary_path(manifest, step_index), summary)
    expected_revision = step_index + 1
    if int(store.load()["revision"]) != expected_revision:
        raise RuntimeError(
            f"memory revision invariant failed: expected {expected_revision}, "
            f"got {store.load()['revision']}"
        )
    return summary


def run(manifest_path: Path) -> dict[str, Any]:
    manifest = _load_frozen_manifest(manifest_path)
    repo = Path(manifest["repo"])
    if _git(repo, "rev-parse", "HEAD") != manifest["commit"]:
        raise RuntimeError("repository commit does not match frozen manifest")
    if _git(repo, "status", "--porcelain"):
        raise RuntimeError("repository must be clean")
    batch_runner._verify_generic_memory(manifest)
    store = _initialize_store(manifest)
    output_root = Path(manifest["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    digest = _manifest_digest(manifest)

    completed = _completed_steps(manifest)
    consecutive_infrastructure_invalid = 0
    for expected, row in enumerate(completed):
        if int(row["step_index"]) != expected:
            raise RuntimeError("completed step sequence has a gap")
    current_revision = int(store.load()["revision"])
    if current_revision == len(completed) + 1:
        # Recovery point: the updater committed its append-only revision but the
        # process stopped before writing the derived step summary.
        step_index = len(completed)
        valid = next(
            (row for row in _results_for_step(manifest, step_index) if row.get("valid")),
            None,
        )
        if valid is None:
            raise RuntimeError("memory advanced without a valid rollout result")
        marker = Path(manifest["output_root"]) / "memory-updates" / valid[
            "episode_id"
        ] / "accepted.json"
        if not marker.is_file():
            raise RuntimeError("memory advanced without an accepted updater marker")
        _record_step(
            manifest,
            store,
            step_index=step_index,
            result=valid,
            update=_load_json(marker),
        )
        completed = _completed_steps(manifest)
        current_revision = int(store.load()["revision"])
    if current_revision != len(completed):
        raise RuntimeError("canonical memory revision does not match completed steps")

    for step_index in range(len(completed), int(manifest["max_valid_episodes"])):
        report = _write_analysis(manifest, store)
        if report["terminal_reason"] is not None:
            return report
        total_reserved, arm_reserved = _ledger_counts(manifest)
        if total_reserved >= int(manifest["global_reservation_cap"]):
            break
        if arm_reserved >= int(manifest["arm_reservation_cap"]):
            break

        round_index = _round_for_step(manifest, step_index)
        seed = _seed_for_step(manifest, step_index)
        round_dir = output_root / f"round-{round_index:02d}"
        snapshot_path = round_dir / "memory_snapshot.json"
        snapshot = (
            _load_json(snapshot_path)
            if snapshot_path.exists()
            else store.snapshot(snapshot_path)
        )
        snapshot_sha = canonical_sha256(snapshot)
        if snapshot_sha != canonical_sha256(store.load()):
            raise RuntimeError("step snapshot does not match current canonical memory")
        if int(snapshot["revision"]) != step_index:
            raise RuntimeError("step snapshot revision does not equal step index")

        attempts = _results_for_step(manifest, step_index)
        valid = next((row for row in attempts if row.get("valid")), None)
        while valid is None:
            total_reserved, arm_reserved = _ledger_counts(manifest)
            if total_reserved >= int(manifest["global_reservation_cap"]):
                return _write_analysis(manifest, store)
            if arm_reserved >= int(manifest["arm_reservation_cap"]):
                return _write_analysis(manifest, store)
            next_attempt = 1 + max(
                (int(row.get("attempt_index", -1)) for row in attempts), default=-1
            )
            episode_id = f"r{round_index:02d}-s{seed:02d}-a{next_attempt:02d}"
            lease_id = f"{manifest['protocol_id']}:{episode_id}"
            lease = token_budget.acquire(
                Path(manifest["token_budget_path"]),
                lease_id=lease_id,
                arm=str(manifest["arm"]),
            )
            if lease is None:
                return _write_analysis(manifest, store)
            guard_before = batch_runner._memory_guard_snapshot(manifest, store)
            result = batch_runner._run_one_episode(
                manifest={
                    **manifest,
                    "budget_cap": int(manifest["global_reservation_cap"]),
                },
                manifest_digest=digest,
                round_index=round_index,
                seed=seed,
                attempt_index=next_attempt,
                gpu=int(manifest["gpu"]),
                snapshot_path=snapshot_path,
                snapshot_sha256=snapshot_sha,
                snapshot_revision=int(snapshot["revision"]),
                guard_before=guard_before,
            )
            attempts.append(result)
            if result.get("valid"):
                valid = result
            else:
                token_budget.release(
                    Path(manifest["token_budget_path"]),
                    lease_id=lease_id,
                    outcome="infrastructure_invalid",
                )
                consecutive_infrastructure_invalid += 1
                if (
                    consecutive_infrastructure_invalid
                    >= MAX_CONSECUTIVE_INFRASTRUCTURE_INVALID
                ):
                    return _write_infrastructure_halt(
                        manifest,
                        store,
                        step_index=step_index,
                        consecutive_invalid=consecutive_infrastructure_invalid,
                    )

        marker = output_root / "memory-updates" / valid["episode_id"] / "accepted.json"
        update = (
            _load_json(marker)
            if marker.exists()
            else batch_runner._update_memory_after_episode(
                manifest=manifest,
                store=store,
                result=valid,
            )
        )
        _record_step(
            manifest,
            store,
            step_index=step_index,
            result=valid,
            update=update,
        )
        token_budget.release(
            Path(manifest["token_budget_path"]),
            lease_id=f"{manifest['protocol_id']}:{valid['episode_id']}",
            outcome="valid_success" if valid["success"] else "valid_failure",
        )
        consecutive_infrastructure_invalid = 0

    return _write_analysis(manifest, store)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    init = sub.add_parser("init")
    init.add_argument("--manifest", type=Path, required=True)
    init.add_argument("--protocol-id", required=True)
    init.add_argument("--arm", choices=sorted(ARMS), required=True)
    init.add_argument("--repo", type=Path, required=True)
    init.add_argument("--python", type=Path, required=True)
    init.add_argument("--output-root", type=Path, required=True)
    init.add_argument("--memory-root", type=Path, required=True)
    init.add_argument("--ledger", type=Path, required=True)
    init.add_argument("--initial-memory", type=Path, required=True)
    init.add_argument("--suite", default="libero_10")
    init.add_argument("--task", type=int, default=3)
    init.add_argument("--seeds", type=int, nargs="+", default=list(range(1, 9)))
    init.add_argument(
        "--window-size",
        type=int,
        help="reporting block size; defaults to seed count for cycling or 8 for fixed",
    )
    init.add_argument("--fixed-seed", type=int, default=1)
    init.add_argument("--global-reservation-cap", type=int, default=2000)
    init.add_argument("--arm-reservation-cap", type=int, required=True)
    init.add_argument("--max-valid-episodes", type=int, required=True)
    init.add_argument(
        "--run-to-max-valid",
        action="store_true",
        help="record improvements but do not stop before max_valid_episodes",
    )
    init.add_argument("--ceiling-confirmation-blocks", type=int, default=3)
    init.add_argument("--round-offset", type=int, required=True)
    init.add_argument("--gpu", type=int, required=True)
    init.add_argument("--token-budget", type=Path, required=True)
    init.add_argument("--max-turns", type=int, default=100)
    init.add_argument("--max-episode-steps", type=int, default=10000)
    init.add_argument("--planner-timeout-s", type=int, default=1800)
    init.add_argument("--episode-timeout-s", type=int, default=3600)
    init.add_argument("--base-url")
    init.add_argument("--use-local-adapter-key-placeholder", action="store_true")
    init.add_argument("--sam3-endpoint")
    init.add_argument("--vla-endpoint")
    init.add_argument("--contact-graspnet-endpoint")
    init.add_argument("--graspgen-endpoint")
    init.add_argument(
        "--tool-profile",
        choices=("all", "pi05_only"),
        default="all",
    )
    init.add_argument(
        "--disable-generic-memory",
        action="store_true",
        help="exclude legacy static task memory from this frozen protocol",
    )
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
        print(
            json.dumps(
                {"manifest": str(args.manifest), "sha256": _manifest_digest(manifest)},
                indent=2,
            )
        )
        return 0
    manifest = _load_frozen_manifest(args.manifest)
    store = _initialize_store(manifest)
    report = run(args.manifest) if args.command == "run" else _write_analysis(manifest, store)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
