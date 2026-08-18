#!/usr/bin/env python3
"""Recoverable pure-Pi0.5 screening over LIBERO task suites.

This runner intentionally has no planner or RPent memory dependency.  Each
environment episode receives the authoritative LIBERO task language directly
as the Pi0.5 prompt and replans every five actions, matching the public openpi
LIBERO evaluation protocol.
"""
from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback
from typing import Any

import numpy as np

from robots.libero.env_client import LiberoEnvClient
from robots.libero.tools import LiberoPrimitives
from rpent.utils.daemon import ProcessDaemon, pick_free_port
from rpent.utils.http_rpc import HttpRpcClient
from rpent.utils.rpc import wait_for_ready
from rpent.utils.sam3_client import UnavailableSam3Client
from rpent.utils.vla_client import VLAClient


SUITES = (
    ("libero_10", "standard"),
    ("libero_10_task", "pro"),
    ("libero_10_swap", "pro"),
)
DEFAULT_POLICY_STEPS = 520
DEFAULT_WAIT_STEPS = 10
REPLAN_STEPS = 5
DUMMY_ACTION = np.asarray([0.0] * 6 + [-1.0], dtype=np.float32)


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    tmp.write_bytes(_json_bytes(value))
    os.replace(tmp, path)


def _append_jsonl(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        stream.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
        fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_manifest(path: Path, excluded_tasks: set[tuple[str, int]] | None = None) -> dict[str, Any]:
    excluded_tasks = set(excluded_tasks or ())
    jobs: list[dict[str, Any]] = []
    ordinal = 0
    for suite, libero_type in SUITES:
        for task in range(10):
            if (suite, task) in excluded_tasks:
                continue
            for seed in range(10):
                jobs.append(
                    {
                        "ordinal": ordinal,
                        "job_id": f"{suite}-task{task:02d}-seed{seed:02d}",
                        "suite": suite,
                        "libero_type": libero_type,
                        "task": task,
                        "initial_state_id": seed,
                    }
                )
                ordinal += 1
    manifest = {
        "schema": "pure-pi05-libero-screen/v1",
        "protocol": {
            "planner": None,
            "memory": None,
            "policy": "pi0.5",
            "success": "libero_terminated",
            "replan_steps": REPLAN_STEPS,
            "stabilization_steps": DEFAULT_WAIT_STEPS,
            "policy_steps": DEFAULT_POLICY_STEPS,
            "valid_episode_cap": 300,
        },
        "suite_order": [suite for suite, _ in SUITES],
        "excluded_tasks": [
            {"suite": suite, "task": task, "reason": "no fixed initial states in audited installed benchmark"}
            for suite, task in sorted(excluded_tasks)
        ],
        "jobs": jobs,
    }
    _atomic_json(path, manifest)
    return manifest


def _load_manifest(path: Path) -> tuple[dict[str, Any], str]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    jobs = manifest.get("jobs")
    if not isinstance(jobs, list) or not 1 <= len(jobs) <= 300:
        raise ValueError("manifest must contain between 1 and 300 preregistered jobs")
    ordinals = [int(job["ordinal"]) for job in jobs]
    if ordinals != list(range(len(jobs))):
        raise ValueError("manifest ordinals must be contiguous from zero")
    return manifest, _sha256(path)


def _episode_dir(root: Path, job: dict[str, Any]) -> Path:
    return root / "episodes" / str(job["suite"]) / f"task-{int(job['task']):02d}" / f"seed-{int(job['initial_state_id']):02d}"


def _next_attempt_dir(episode_dir: Path) -> Path:
    existing = sorted(episode_dir.glob("attempt-*"))
    attempt = 1 + max((int(path.name.split("-")[-1]) for path in existing), default=0)
    path = episode_dir / f"attempt-{attempt:03d}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def _normalize_endpoint(endpoint: str) -> str:
    endpoint = endpoint.strip().rstrip("/")
    return endpoint if "://" in endpoint else f"http://{endpoint}"


def run_episode(
    *,
    job: dict[str, Any],
    artifact_root: Path,
    vla_endpoint: str,
    gpu: int,
    manifest_sha256: str,
    policy_steps: int,
    wait_steps: int,
) -> dict[str, Any]:
    episode_dir = _episode_dir(artifact_root, job)
    result_path = episode_dir / "result.json"
    if result_path.exists():
        existing = json.loads(result_path.read_text(encoding="utf-8"))
        if existing.get("valid") is True:
            return existing

    attempt_dir = _next_attempt_dir(episode_dir)
    started = time.time()
    event_base = {
        "job_id": job["job_id"],
        "ordinal": int(job["ordinal"]),
        "suite": job["suite"],
        "task": int(job["task"]),
        "initial_state_id": int(job["initial_state_id"]),
        "attempt": int(attempt_dir.name.split("-")[-1]),
        "worker_pid": os.getpid(),
        "gpu": gpu,
        "manifest_sha256": manifest_sha256,
    }
    _append_jsonl(artifact_root / "ledger.jsonl", {**event_base, "event": "attempt_started", "time": started})

    env_daemon: ProcessDaemon | None = None
    primitives: LiberoPrimitives | None = None
    chunks: list[dict[str, Any]] = []
    try:
        vla_rpc = HttpRpcClient(_normalize_endpoint(vla_endpoint))
        model = VLAClient(vla_rpc)
        health = model.healthz(timeout_s=10.0)

        port = pick_free_port()
        max_episode_steps = int(policy_steps) + int(wait_steps)
        env_vars = os.environ.copy()
        env_vars.update(
            {
                "LIBERO_TYPE": str(job["libero_type"]),
                "MUJOCO_GL": "egl",
                "PYOPENGL_PLATFORM": "egl",
                "ROBOT_PLATFORM": "LIBERO",
            }
        )
        env_vars.pop("CUDA_VISIBLE_DEVICES", None)
        env_daemon = ProcessDaemon(
            name=f"env-{job['job_id']}",
            cmd=[
                sys.executable,
                str(Path(__file__).resolve().parents[2] / "robots" / "libero" / "env_server.py"),
                "--suite", str(job["suite"]),
                "--task", str(int(job["task"])),
                "--seed", str(int(job["initial_state_id"])),
                "--max-episode-steps", str(max_episode_steps),
                "--transport", "http",
                "--host", "127.0.0.1",
                "--port", str(port),
                "--parent-watch",
                "--cuda-device", str(gpu),
            ],
            env=env_vars,
            log_path=str(attempt_dir / "env_server.log"),
            cwd=str(Path(__file__).resolve().parents[2]),
        )
        env_daemon.start()
        env_rpc = HttpRpcClient(f"http://127.0.0.1:{port}")
        wait_for_ready(env_rpc, daemon=env_daemon, timeout_s=300.0)
        expected_meta = {
            "suite": str(job["suite"]),
            "task": int(job["task"]),
            "seed": int(job["initial_state_id"]),
            "max_episode_steps": max_episode_steps,
        }
        env = LiberoEnvClient(env_rpc, expected_meta=expected_meta, return_all_frames=True)
        primitives = LiberoPrimitives(
            env,
            model,
            UnavailableSam3Client("pure Pi0.5 screening has no analytic tools"),
        )
        obs, _ = env.reset()
        primitives.set_obs(obs)
        task_language = str(env.get_task_language() or obs.get("task_descriptions") or "").strip()
        if not task_language:
            raise RuntimeError("authoritative LIBERO task language is empty")

        primitives.start_recording()
        primitives.record_frame(obs)
        for _ in range(int(wait_steps)):
            if env.episode_terminated or env.episode_truncated:
                break
            primitives._step_env(DUMMY_ACTION)

        planned_chunks = int(math.ceil(int(policy_steps) / REPLAN_STEPS))
        for chunk_index in range(planned_chunks):
            if env.episode_terminated or env.episode_truncated:
                break
            before = time.time()
            primitives._vlm_chunk(
                task_language,
                mode="eval",
                actions_per_chunk=REPLAN_STEPS,
            )
            row = {
                "chunk": chunk_index + 1,
                "elapsed_s": time.time() - before,
                "libero_terminated": bool(env.episode_terminated),
                "episode_truncated": bool(env.episode_truncated),
                **primitives._last_vla_diagnostics,
            }
            chunks.append(row)
            _append_jsonl(attempt_dir / "chunks.jsonl", row)

        video = primitives.stop_recording_and_save(str(attempt_dir / "episode.mp4"), fps=10)
        success = bool(env.episode_terminated)
        result = {
            **event_base,
            "valid": True,
            "success": success,
            "libero_terminated": success,
            "episode_truncated": bool(env.episode_truncated),
            "task_language": task_language,
            "wait_steps": int(wait_steps),
            "policy_steps_limit": int(policy_steps),
            "chunks_used": len(chunks),
            "actions_requested": sum(
                int(row.get("executed_horizon", REPLAN_STEPS)) for row in chunks
            ),
            "elapsed_s": time.time() - started,
            "vla_health": health,
            "video": video,
            "attempt_dir": str(attempt_dir),
        }
        _atomic_json(attempt_dir / "attempt-result.json", result)
        _atomic_json(result_path, result)
        _append_jsonl(artifact_root / "ledger.jsonl", {**result, "event": "valid_episode"})
        return result
    except Exception as exc:
        if primitives is not None:
            with contextlib.suppress(Exception):
                primitives.stop_recording_and_save(str(attempt_dir / "partial.mp4"), fps=10)
        failure = {
            **event_base,
            "valid": False,
            "success": False,
            "infrastructure_error": type(exc).__name__,
            "error": str(exc),
            "elapsed_s": time.time() - started,
            "attempt_dir": str(attempt_dir),
        }
        (attempt_dir / "traceback.txt").write_text(traceback.format_exc(), encoding="utf-8")
        _atomic_json(attempt_dir / "attempt-result.json", failure)
        _append_jsonl(artifact_root / "ledger.jsonl", {**failure, "event": "invalid_attempt"})
        return failure
    finally:
        if env_daemon is not None:
            with contextlib.suppress(Exception):
                env_daemon.stop()


def analyze(manifest: dict[str, Any], artifact_root: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    by_task: list[dict[str, Any]] = []
    for job in manifest["jobs"]:
        result_path = _episode_dir(artifact_root, job) / "result.json"
        if result_path.exists():
            result = json.loads(result_path.read_text(encoding="utf-8"))
            if result.get("valid") is True:
                rows.append(result)
    for suite, _ in SUITES:
        for task in range(10):
            selected = [row for row in rows if row["suite"] == suite and int(row["task"]) == task]
            successes = sum(bool(row["success"]) for row in selected)
            by_task.append(
                {
                    "suite": suite,
                    "task": task,
                    "valid": len(selected),
                    "successes": successes,
                    "success_rate": successes / len(selected) if selected else None,
                    "complete": len(selected) == 10,
                    "selected_hard": len(selected) == 10 and 0 < successes <= 5,
                }
            )
    summary = {
        "valid_episodes": len(rows),
        "successes": sum(bool(row["success"]) for row in rows),
        "complete_tasks": sum(bool(row["complete"]) for row in by_task),
        "selected_tasks": [row for row in by_task if row["selected_hard"]],
        "tasks": by_task,
    }
    _atomic_json(artifact_root / "analysis.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--build-manifest", action="store_true")
    parser.add_argument(
        "--exclude-task",
        action="append",
        default=[],
        metavar="SUITE:TASK",
        help="Record and omit a task that has no executable fixed initial states",
    )
    parser.add_argument("--analyze", action="store_true")
    parser.add_argument("--vla-endpoint")
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--worker-index", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--only-job-id")
    parser.add_argument("--max-consecutive-invalid", type=int, default=3)
    parser.add_argument("--policy-steps", type=int, default=DEFAULT_POLICY_STEPS)
    parser.add_argument("--wait-steps", type=int, default=DEFAULT_WAIT_STEPS)
    args = parser.parse_args()

    args.artifact_root.mkdir(parents=True, exist_ok=True)
    if args.build_manifest:
        excluded: set[tuple[str, int]] = set()
        for item in args.exclude_task:
            suite, separator, raw_task = item.rpartition(":")
            if not separator or not suite:
                raise SystemExit(f"invalid --exclude-task {item!r}; expected SUITE:TASK")
            excluded.add((suite, int(raw_task)))
        build_manifest(args.manifest, excluded)
    manifest, manifest_sha256 = _load_manifest(args.manifest)
    if args.analyze:
        print(json.dumps(analyze(manifest, args.artifact_root), ensure_ascii=False, indent=2))
        return
    if args.vla_endpoint is None or args.gpu is None:
        raise SystemExit("--vla-endpoint and --gpu are required when running episodes")
    if not 0 <= args.worker_index < args.workers:
        raise SystemExit("worker-index must be in [0, workers)")

    pid_path = args.artifact_root / f"worker-{args.worker_index:02d}.pid"
    _atomic_json(
        pid_path,
        {
            "pid": os.getpid(),
            "worker_index": args.worker_index,
            "workers": args.workers,
            "gpu": args.gpu,
            "vla_endpoint": args.vla_endpoint,
            "manifest_sha256": manifest_sha256,
        },
    )
    consecutive_invalid = 0
    try:
        for job in manifest["jobs"]:
            if args.only_job_id is not None and job["job_id"] != args.only_job_id:
                continue
            if args.only_job_id is None and int(job["ordinal"]) % args.workers != args.worker_index:
                continue
            while True:
                result = run_episode(
                    job=job,
                    artifact_root=args.artifact_root,
                    vla_endpoint=args.vla_endpoint,
                    gpu=args.gpu,
                    manifest_sha256=manifest_sha256,
                    policy_steps=args.policy_steps,
                    wait_steps=args.wait_steps,
                )
                if result.get("valid") is True:
                    consecutive_invalid = 0
                    analyze(manifest, args.artifact_root)
                    break
                consecutive_invalid += 1
                if consecutive_invalid >= args.max_consecutive_invalid:
                    raise SystemExit(
                        f"circuit breaker: {consecutive_invalid} consecutive infrastructure-invalid attempts"
                    )
                # Retry this exact job before advancing; invalid attempts never
                # consume a logical seed or alter the preregistered order.
    finally:
        with contextlib.suppress(FileNotFoundError):
            pid_path.unlink()
        analyze(manifest, args.artifact_root)


if __name__ == "__main__":
    main()
