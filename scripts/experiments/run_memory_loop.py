#!/usr/bin/env python3
"""Run resumable episode-frozen, task-scoped memory-update experiments.

Every rollout in one seed batch reads the exact same immutable memory snapshot.
After a rollout has ended, a separate terra-medium VLM call rewrites the
canonical task parameter block.  The next batch snapshots the latest revision.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import signal
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:  # pragma: no cover - production experiments run on Linux/H100.
    import fcntl
except ImportError:  # Windows unit-test import compatibility.
    fcntl = None  # type: ignore[assignment]

# Direct script execution puts only ``scripts/experiments`` on sys.path.  Bind
# imports to this checkout rather than an older installed Zetta package.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from zetta.memory.task_memory import (
    TaskMemoryStore,
    canonical_json,
    canonical_sha256,
    validate_parameter_block,
)
from zetta.tools.contracts import READ_ONLY_CONTRACT
from zetta.tools.toolkit import Toolkit
from scripts.experiments.run_paired_episode import (
    _episode_command,
    _git,
    _load_last_json_object,
    _sha256_file,
    _sha256_tree,
    _summarize,
)

SCHEMA_VERSION = 1
MAX_ALLOWED_EPISODES = 200
DEFAULT_SEEDS = tuple(range(1, 9))


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _generic_memory_descriptor(repo: Path, *, disabled: bool) -> dict[str, Any]:
    """Freeze the optional legacy static-memory tree.

    Clean-prompt experiments deliberately disable this resource so historical
    task recipes cannot leak into a revision-zero task-memory study.  Existing
    manifests did not carry ``enabled`` and remain enabled when they have a
    path.
    """

    if disabled:
        return {
            "enabled": False,
            "path": None,
            "sha256": None,
            "file_count": 0,
        }
    resource_memory = repo / "resources" / "libero" / "memory"
    resource_sha, resource_count = _sha256_tree(resource_memory)
    return {
        "enabled": True,
        "path": str(resource_memory),
        "sha256": resource_sha,
        "file_count": resource_count,
    }


def _generic_memory_state(manifest: dict[str, Any]) -> tuple[str | None, int]:
    """Return the live generic-memory hash/count or the disabled sentinel."""

    generic = manifest["frozen_resources"]["generic_memory"]
    enabled = bool(generic.get("enabled", bool(generic.get("path"))))
    if not enabled:
        if generic.get("path") is not None:
            raise RuntimeError("disabled generic memory must not define a path")
        return None, 0
    path = generic.get("path")
    if not isinstance(path, str) or not path:
        raise RuntimeError("enabled generic memory requires a path")
    return _sha256_tree(Path(path))


def _verify_generic_memory(manifest: dict[str, Any]) -> None:
    generic = manifest["frozen_resources"]["generic_memory"]
    actual_sha, actual_count = _generic_memory_state(manifest)
    if actual_sha != generic.get("sha256") or actual_count != int(
        generic.get("file_count", -1)
    ):
        raise RuntimeError("generic memory changed after manifest freeze")


def _tool_endpoint_environment(common: dict[str, Any]) -> dict[str, str]:
    """Materialize manifest-frozen HTTP tool endpoints for a rollout child."""

    result: dict[str, str] = {}
    for manifest_key, environment_key in (
        ("contact_graspnet_endpoint", "CONTACT_GRASPNET_URL"),
        ("graspgen_endpoint", "GRASPGEN_URL"),
    ):
        endpoint = common.get(manifest_key)
        if endpoint:
            result[environment_key] = str(endpoint)
    return result


def _planner_environment(common: dict[str, Any]) -> dict[str, str]:
    """Freeze provider routing without persisting an external credential."""

    result: dict[str, str] = {}
    base_url = common.get("base_url")
    if base_url:
        result["CODEX_BASE_URL"] = str(base_url)
    if common.get("use_local_adapter_key_placeholder"):
        result["CODEX_API_KEY"] = "zetta-local-adapter"
    return result


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _manifest_digest(manifest: dict[str, Any]) -> str:
    return canonical_sha256(manifest)


def _validate_manifest(manifest: dict[str, Any]) -> None:
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"manifest schema_version must be {SCHEMA_VERSION}")
    if manifest.get("model") != "gpt-5.6-terra":
        raise ValueError("memory protocol requires gpt-5.6-terra")
    if manifest.get("reasoning_effort") != "medium":
        raise ValueError("memory protocol requires reasoning_effort=medium")
    cap = int(manifest.get("budget_cap", 0))
    if not 1 <= cap <= MAX_ALLOWED_EPISODES:
        raise ValueError(f"budget_cap must be in [1,{MAX_ALLOWED_EPISODES}]")
    seeds = manifest.get("seeds")
    if not isinstance(seeds, list) or len(seeds) != 8:
        raise ValueError("seeds must contain exactly eight values")
    if len(set(int(seed) for seed in seeds)) != 8:
        raise ValueError("seeds must be unique")
    if int(manifest.get("max_rounds", 0)) * len(seeds) > cap:
        raise ValueError("max_rounds * batch size exceeds budget_cap")
    if not isinstance(manifest.get("protocol_id"), str) or not manifest[
        "protocol_id"
    ].strip():
        raise ValueError("protocol_id is required")
    for key in (
        "repo",
        "python",
        "output_root",
        "memory_root",
        "ledger_path",
        "commit",
        "suite",
        "task_key",
    ):
        if key not in manifest:
            raise ValueError(f"manifest field is required: {key}")


def build_manifest(args: argparse.Namespace) -> dict[str, Any]:
    repo = args.repo.expanduser().resolve()
    commit = _git(repo, "rev-parse", "HEAD")
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": args.protocol_id,
        "created_at": _utc_now(),
        "budget_cap": int(args.budget_cap),
        "max_rounds": int(args.max_rounds),
        "max_attempts_per_seed": int(args.max_attempts_per_seed),
        "concurrency": int(args.concurrency),
        "seeds": [int(seed) for seed in args.seeds],
        "suite": args.suite,
        "task": int(args.task),
        "task_key": f"{args.suite}/task-{int(args.task)}",
        "repo": str(repo),
        "commit": commit,
        # Preserve the virtualenv symlink rather than resolving to base Python.
        "python": str(args.python.expanduser().absolute()),
        "output_root": str(args.output_root.expanduser().resolve()),
        "memory_root": str(args.memory_root.expanduser().resolve()),
        "ledger_path": str(args.ledger.expanduser().resolve()),
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
            "disable_sam3": True,
            "contact_graspnet_endpoint": args.contact_graspnet_endpoint,
            "graspgen_endpoint": args.graspgen_endpoint,
            "hf_hub_offline": True,
            "libero_type": "pro",
        },
        "frozen_resources": {
            "generic_memory": _generic_memory_descriptor(
                repo, disabled=bool(args.disable_generic_memory)
            ),
            "runner_sha256": _sha256_file(Path(__file__).resolve()),
        },
        "stop_rule": {
            "metric": "eight_seed_batch_success_rate",
            "baseline": "first_complete_valid_batch",
            "condition": "later_complete_valid_batch_rate > baseline_rate",
        },
    }


def freeze_manifest(path: Path, manifest: dict[str, Any]) -> None:
    _validate_manifest(manifest)
    _atomic_json(path, manifest)
    path.with_suffix(path.suffix + ".sha256").write_text(
        _manifest_digest(manifest) + "\n", encoding="ascii"
    )


def _load_frozen_manifest(path: Path) -> dict[str, Any]:
    manifest = _load_json(path)
    _validate_manifest(manifest)
    digest_path = path.with_suffix(path.suffix + ".sha256")
    if not digest_path.is_file():
        raise RuntimeError("manifest is not frozen")
    actual = _manifest_digest(manifest)
    if digest_path.read_text(encoding="ascii").strip() != actual:
        raise RuntimeError("manifest changed after freeze")
    return manifest


def _append_ledger(path: Path, row: dict[str, Any], *, reserve: bool) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as handle:
        if fcntl is not None:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        handle.seek(0)
        rows = []
        for line in handle:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        reservations = [item for item in rows if item.get("event") == "reserved"]
        if reserve:
            cap = int(row["budget_cap"])
            if len(reservations) >= cap:
                raise RuntimeError(f"episode budget exhausted: {len(reservations)}/{cap}")
            identity = (row.get("protocol_id"), row.get("episode_id"))
            if any(
                (item.get("protocol_id"), item.get("episode_id")) == identity
                for item in reservations
            ):
                raise RuntimeError(
                    "episode id already reserved for protocol: "
                    f"{row.get('protocol_id')}:{row.get('episode_id')}"
                )
        handle.seek(0, os.SEEK_END)
        handle.write(canonical_json(row) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
        if fcntl is not None:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        return len(reservations) + (1 if reserve else 0)


def _memory_guard_snapshot(manifest: dict[str, Any], store: TaskMemoryStore) -> dict[str, Any]:
    current = store.load()
    generic_sha, generic_count = _generic_memory_state(manifest)
    return {
        "task_memory_sha256": canonical_sha256(current),
        "task_memory_revision": int(current["revision"]),
        "generic_memory_sha256": generic_sha,
        "generic_memory_file_count": generic_count,
        "repo_status": _git(Path(manifest["repo"]), "status", "--porcelain"),
    }


def _run_one_episode(
    *,
    manifest: dict[str, Any],
    manifest_digest: str,
    round_index: int,
    seed: int,
    attempt_index: int,
    gpu: int,
    snapshot_path: Path,
    snapshot_sha256: str,
    snapshot_revision: int,
    guard_before: dict[str, Any],
) -> dict[str, Any]:
    episode_id = f"r{round_index:02d}-s{seed:02d}-a{attempt_index:02d}"
    output_dir = Path(manifest["output_root"]) / f"round-{round_index:02d}" / episode_id
    existing = output_dir / "result.json"
    if existing.is_file():
        return _load_json(existing)
    episode = {
        "id": episode_id,
        "phase": f"memory-round-{round_index}",
        "variant": "episode-memory",
        "suite": manifest["suite"],
        "task": int(manifest["task"]),
        "seed": seed,
        "cuda_device": gpu,
        "extra_args": [
            "--libero-type",
            str(manifest["common"]["libero_type"]),
            "--task-memory-snapshot",
            str(snapshot_path),
            "--episode-memory-frozen",
        ],
    }
    common = {
        **manifest["common"],
        "model": manifest["model"],
        "reasoning_effort": manifest["reasoning_effort"],
        "protocol_id": manifest["protocol_id"],
    }
    ledger_path = Path(manifest["ledger_path"])
    if output_dir.exists():
        # A previous orchestrator may have been interrupted after reservation.
        # Preserve the partial artifacts and close the attempt as infrastructure-
        # invalid instead of overwriting it or spending a duplicate episode ID.
        ledger_rows = []
        if ledger_path.is_file():
            for line in ledger_path.read_text(encoding="utf-8").splitlines():
                try:
                    ledger_rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        reservation = next(
            (
                row
                for row in reversed(ledger_rows)
                if row.get("event") == "reserved"
                and row.get("protocol_id") == manifest["protocol_id"]
                and row.get("episode_id") == episode_id
            ),
            None,
        )
        if reservation is None:
            raise FileExistsError(
                f"incomplete output has no matching reservation: {output_dir}"
            )
        attempt_id = str(reservation["attempt_id"])
        result = _summarize(
            episode=episode,
            common=common,
            output_dir=output_dir,
            commit=manifest["commit"],
            manifest_digest=manifest_digest,
            attempt_id=attempt_id,
            exit_code=143,
            timed_out=False,
            elapsed_s=0.0,
        )
        result.update(
            {
                "round": round_index,
                "attempt_index": attempt_index,
                "memory_snapshot_path": str(snapshot_path),
                "memory_snapshot_sha256": snapshot_sha256,
                "memory_snapshot_revision": int(snapshot_revision),
                "memory_guard": {
                    "before": guard_before,
                    "after": None,
                    "planner_sandbox": None,
                    "write_scope": "output_dir_only",
                    "injected": None,
                    "reasons": ["orchestrator_interrupted_before_resume"],
                },
                "recovery": {
                    "classification": "interrupted_infrastructure_attempt",
                    "recovered_at": _utc_now(),
                },
            }
        )
        result["valid"] = False
        result["invalid_reasons"] = [
            *result.get("invalid_reasons", []),
            "orchestrator_interrupted_before_resume",
        ]
        _atomic_json(existing, result)
        if not any(
            row.get("event") == "completed" and row.get("episode_id") == episode_id
            for row in ledger_rows
        ):
            _append_ledger(
                ledger_path,
                {
                    "event": "completed",
                    "timestamp": _utc_now(),
                    "protocol_id": manifest["protocol_id"],
                    "episode_id": episode_id,
                    "attempt_id": attempt_id,
                    "round": round_index,
                    "seed": seed,
                    "success": False,
                    "valid": False,
                    "result_sha256": _sha256_file(existing),
                    "recovered_interruption": True,
                },
                reserve=False,
            )
        return result

    attempt_id = uuid.uuid4().hex
    used = _append_ledger(
        ledger_path,
        {
            "event": "reserved",
            "timestamp": _utc_now(),
            "protocol_id": manifest["protocol_id"],
            "episode_id": episode_id,
            "attempt_id": attempt_id,
            "round": round_index,
            "seed": seed,
            "memory_snapshot_sha256": snapshot_sha256,
            "budget_cap": int(manifest["budget_cap"]),
        },
        reserve=True,
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    command = _episode_command(
        episode,
        common,
        str(Path(manifest["python"]).expanduser().absolute()),
        output_dir,
    )
    env = os.environ.copy()
    repo = Path(manifest["repo"])
    env["PYTHONPATH"] = str(repo) + os.pathsep + env.get("PYTHONPATH", "")
    env["CODEX_MODEL"] = manifest["model"]
    env["CODEX_REASONING_EFFORT"] = manifest["reasoning_effort"]
    # Physical MCP calls are side effects and Codex rejects them in read-only
    # mode. Immutability is instead enforced by not exposing the mutable store,
    # output-only MCP writes, and pre/post hash guards over every memory surface.
    env["ZETTA_CODEX_SANDBOX"] = "full-access"
    env["ZETTA_EPISODE_MEMORY_FROZEN"] = "1"
    env["HF_HUB_OFFLINE"] = "1"
    env.update(_planner_environment(common))
    env.update(_tool_endpoint_environment(common))
    timeout_s = int(common["episode_timeout_s"])
    started = time.time()
    exit_code = 1
    timed_out = False
    runner_log = output_dir / "runner.log"
    with runner_log.open("w", encoding="utf-8") as log:
        log.write(
            canonical_json(
                {
                    "command": command,
                    "budget_used": used,
                    "memory_guard_before": guard_before,
                }
            )
            + "\n"
        )
        log.flush()
        process = subprocess.Popen(
            command,
            cwd=repo,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        try:
            exit_code = process.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(process.pid, signal.SIGTERM)
            try:
                exit_code = process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                exit_code = process.wait()

    result = _summarize(
        episode=episode,
        common=common,
        output_dir=output_dir,
        commit=manifest["commit"],
        manifest_digest=manifest_digest,
        attempt_id=attempt_id,
        exit_code=exit_code,
        timed_out=timed_out,
        elapsed_s=time.time() - started,
    )
    transcripts = sorted(output_dir.glob("transcript_*.json"))
    transcript = _load_last_json_object(transcripts[-1]) if transcripts else {}
    stats = transcript.get("stats") if isinstance(transcript.get("stats"), dict) else {}
    injected = transcript.get("episode_memory")
    store = TaskMemoryStore(
        manifest["memory_root"],
        task_key=manifest["task_key"],
        suite=manifest["suite"],
        task=int(manifest["task"]),
    )
    guard_after = _memory_guard_snapshot(manifest, store)
    guard_reasons: list[str] = []
    if not isinstance(injected, dict) or injected.get("sha256") != snapshot_sha256:
        guard_reasons.append("injected_snapshot_mismatch")
    if stats.get("sandbox") != "full-access":
        guard_reasons.append(f"planner_sandbox={stats.get('sandbox')!r}")
    for key in (
        "task_memory_sha256",
        "task_memory_revision",
        "generic_memory_sha256",
        "generic_memory_file_count",
    ):
        if guard_after.get(key) != guard_before.get(key):
            guard_reasons.append(f"memory_guard_changed={key}")
    try:
        snapshot_after = canonical_sha256(_load_json(snapshot_path))
    except Exception as exc:
        snapshot_after = None
        guard_reasons.append(f"snapshot_unreadable={type(exc).__name__}")
    if snapshot_after != snapshot_sha256:
        guard_reasons.append("snapshot_changed_during_episode")
    if guard_after.get("repo_status"):
        guard_reasons.append("repository_changed_during_episode")
    result.update(
        {
            "round": round_index,
            "attempt_index": attempt_index,
            "memory_snapshot_path": str(snapshot_path),
            "memory_snapshot_sha256": snapshot_sha256,
            "memory_snapshot_revision": int(snapshot_revision),
            "memory_guard": {
                "before": guard_before,
                "after": guard_after,
                "snapshot_after_sha256": snapshot_after,
                "planner_sandbox": stats.get("sandbox"),
                "write_scope": "output_dir_only",
                "injected": injected,
                "reasons": guard_reasons,
            },
        }
    )
    if guard_reasons:
        result["valid"] = False
        result["invalid_reasons"] = [*result.get("invalid_reasons", []), *guard_reasons]
    _atomic_json(existing, result)
    _append_ledger(
        Path(manifest["ledger_path"]),
        {
            "event": "completed",
            "timestamp": _utc_now(),
            "protocol_id": manifest["protocol_id"],
            "episode_id": episode_id,
            "attempt_id": attempt_id,
            "round": round_index,
            "seed": seed,
            "success": bool(result["success"]),
            "valid": bool(result["valid"]),
            "result_sha256": _sha256_file(existing),
        },
        reserve=False,
    )
    return result


class _MemoryUpdateToolkit(Toolkit):
    """Single submission tool; the model cannot write the canonical store."""

    def __init__(self) -> None:
        super().__init__()
        self._tools.clear()
        self._contracts.clear()
        self.add_tool(
            "finish",
            {
                "name": "finish",
                "description": (
                    "Submit the complete rewritten task parameter block. This is the "
                    "only permitted completion action."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "status": {"type": "string", "enum": ["success"]},
                        "summary": {"type": "string"},
                        "memory_json": {
                            "type": "string",
                            "description": "Strict JSON encoding of the rewritten parameter block.",
                        },
                    },
                    "required": ["status", "summary", "memory_json"],
                },
            },
            lambda status, summary, memory_json: {
                "_finish": True,
                "status": status,
                "summary": summary,
                "memory_json": memory_json,
            },
            contract=READ_ONLY_CONTRACT,
        )


def _strip_scene_specific(value: Any) -> Any:
    """Keep tool policy evidence while removing per-seed coordinates and paths."""
    forbidden = {
        "xyz",
        "world_xyz",
        "target_xyz",
        "eef_pos",
        "final_eef_pos",
        "path",
        "image_path",
        "image_cam_path",
        "image_wrist_path",
    }
    if isinstance(value, dict):
        return {
            key: _strip_scene_specific(child)
            for key, child in value.items()
            if str(key).lower() not in forbidden
        }
    if isinstance(value, list):
        return [_strip_scene_specific(child) for child in value]
    return value


def _update_evidence(result: dict[str, Any]) -> dict[str, Any]:
    output_dir = Path(result["states_path"]).parent
    try:
        states = json.loads((output_dir / "states.json").read_text(encoding="utf-8"))
    except Exception:
        states = []
    actions = []
    for row in states if isinstance(states, list) else []:
        if not isinstance(row, dict) or not isinstance(row.get("command"), dict):
            continue
        actions.append(
            {
                "step": row.get("step_idx"),
                "command": _strip_scene_specific(row.get("command")),
                "result": _strip_scene_specific(row.get("result")),
                "libero_terminated": bool(row.get("libero_terminated")),
                "episode_truncated": bool(row.get("episode_truncated")),
            }
        )
    latest_agent = sorted((output_dir / "images_cam_hi").glob("*.png"))[-2:]
    latest_wrist = sorted((output_dir / "images_wrist_hi").glob("*.png"))[-2:]
    return {
        "episode_id": result["episode_id"],
        "round": result["round"],
        "seed": result["seed"],
        "valid": bool(result["valid"]),
        "success": bool(result["success"]),
        "invalid_reasons": result.get("invalid_reasons", []),
        "elapsed_s": result.get("elapsed_s"),
        "physical_actions": actions,
        "planner_finish": result.get("planner", {}).get("finish"),
        "visual_evidence": {
            "latest_agentview": [str(path) for path in latest_agent],
            "latest_wrist": [str(path) for path in latest_wrist],
            "videos": result.get("videos", {}),
        },
        "result_sha256": _sha256_file(output_dir / "result.json"),
        "memory_snapshot_sha256": result["memory_snapshot_sha256"],
    }


def _memory_update_prompt(previous: dict[str, Any], evidence: dict[str, Any]) -> str:
    schema_example = {
        "strategy_summary": "concise seed-invariant strategy",
        "decision_rules": ["observation-conditioned rule"],
        "tool_parameters": [
            {
                "tool": "pi0_pick",
                "when": "perceptual precondition",
                "parameters": {"max_chunks": 8},
                "why": "evidence-backed reason",
                "confidence": 0.5,
            }
        ],
        "prompt_ladder": ["short object-centric VLA prompt"],
        "failure_modes": [
            {
                "symptom": "observable symptom",
                "likely_cause": "hypothesis",
                "response": "safe next-episode response",
                "evidence_ids": [evidence["episode_id"]],
            }
        ],
        "success_checks": ["observable check"],
        "open_questions": ["uncertainty needing more rollouts"],
    }
    return f"""You are the POST-EPISODE task-memory updater for Zetta. The robot
episode has fully ended; this is the only legal update boundary. Rewrite the
whole parameter block for the same task using the prior block and this rollout.

Rules:
1. Generalize across seeds. Never store absolute xyz, pixels, artifact paths,
   BDDL/ground-truth poses, or facts that identify one seed's layout.
2. Treat failed and invalid episodes differently. Infrastructure-invalid
   evidence must not change the strategy except to preserve uncertainty.
3. Keep only observation-conditioned decision rules and reusable tool/VLA
   prompts or scalar parameters. Memory is a compact parameter block, not a log.
4. Do not claim causality from one rollout. Calibrate confidence and retain
   contradictory evidence as an open question.
5. You may inspect the listed final images in read-only mode if useful. Do not
   write files or run experiments.
6. Never override the static Zetta prompt or a tool contract. In particular,
   pi0_pick success is only a hint that requires perceptual confirmation, and
   task success is authoritative only when libero_terminated is true.
7. Call the finish tool exactly once. memory_json must encode exactly the seven
   keys shown in the schema example; no markdown and no extra keys.

PRIOR PARAMETER BLOCK:
{json.dumps(previous['parameter_block'], ensure_ascii=False, indent=2)}

ROLLOUT EVIDENCE:
{json.dumps(evidence, ensure_ascii=False, indent=2)}

REQUIRED SHAPE EXAMPLE (replace all content):
{json.dumps(schema_example, ensure_ascii=False, indent=2)}
"""


def _decode_json_submission(text: str) -> dict[str, Any]:
    """Decode a strict JSON object, tolerating only one outer Markdown fence."""
    candidate = text.strip()
    if candidate.startswith("```") and candidate.endswith("```"):
        lines = candidate.splitlines()
        if len(lines) < 3:
            raise ValueError("empty fenced JSON submission")
        candidate = "\n".join(lines[1:-1]).strip()
    value = json.loads(candidate)
    if not isinstance(value, dict):
        raise ValueError("memory submission must decode to an object")
    return value


def _update_memory_after_episode(
    *,
    manifest: dict[str, Any],
    store: TaskMemoryStore,
    result: dict[str, Any],
) -> dict[str, Any]:
    # Keep the experiment analysis/build path importable on machines that do not
    # have the H100 Codex SDK runtime installed.
    from zetta.planner.codex import CodexPlanner

    evidence = _update_evidence(result)
    update_root = (
        Path(manifest["output_root"])
        / "memory-updates"
        / result["episode_id"]
    )
    previous = store.load()
    validation_errors: list[str] = []
    transport_errors: list[str] = []
    existing_attempts = sorted(update_root.glob("attempt-*"))
    first_attempt = len(existing_attempts) + 1
    max_model_calls = 5
    for update_attempt in range(first_attempt, first_attempt + max_model_calls):
        attempt_dir = update_root / f"attempt-{update_attempt:02d}"
        attempt_dir.mkdir(parents=True, exist_ok=False)
        prompt = _memory_update_prompt(previous, evidence)
        if validation_errors:
            prompt += (
                "\nThe previous submission was rejected by the external validator: "
                + validation_errors[-1]
                + "\nReturn a corrected full block.\n"
            )
        old_sandbox = os.environ.get("ZETTA_CODEX_SANDBOX")
        os.environ["ZETTA_CODEX_SANDBOX"] = "read-only"
        try:
            planner = CodexPlanner(
                output_dir=str(attempt_dir),
                repo_root=manifest["repo"],
                timeout_s=420,
                output_path=attempt_dir / "codex_memory_update.txt",
                model=manifest["model"],
            )
            planner_result = planner.solve(
                system_prompt=prompt,
                user_message="Rewrite and submit the task parameter block now.",
                toolkit=_MemoryUpdateToolkit(),
                max_turns=3,
            )
        finally:
            if old_sandbox is None:
                os.environ.pop("ZETTA_CODEX_SANDBOX", None)
            else:
                os.environ["ZETTA_CODEX_SANDBOX"] = old_sandbox
        finish = planner_result.finish_result or {}
        if planner_result.error:
            rendered_error = str(planner_result.error)
            transport_markers = (
                "stream disconnected",
                "error sending request",
                "reconnecting",
                "temporary failure in name resolution",
            )
            if any(marker in rendered_error.lower() for marker in transport_markers):
                transport_errors.append(rendered_error)
                _atomic_json(
                    attempt_dir / "update_result.json",
                    {
                        "accepted": False,
                        "classification": "transport_error",
                        "error": rendered_error,
                        "stats": planner_result.stats,
                    },
                )
                if len(transport_errors) >= 3:
                    break
                time.sleep(min(15 * len(transport_errors), 30))
                continue
        submission_mode = "finish_tool"
        try:
            if planner_result.error:
                raise RuntimeError(planner_result.error)
            if finish.get("status") == "success" and finish.get("memory_json"):
                proposal = _decode_json_submission(str(finish["memory_json"]))
            else:
                last_message_path = Path(
                    str(planner_result.stats.get("last_message_path", ""))
                )
                if not last_message_path.is_file():
                    raise ValueError(f"updater did not submit success: {finish}")
                proposal = _decode_json_submission(
                    last_message_path.read_text(encoding="utf-8", errors="replace")
                )
                submission_mode = "final_response_json"
            parameters = validate_parameter_block(proposal)
        except Exception as exc:
            validation_errors.append(f"{type(exc).__name__}: {exc}")
            _atomic_json(
                attempt_dir / "update_result.json",
                {
                    "accepted": False,
                    "error": validation_errors[-1],
                    "stats": planner_result.stats,
                },
            )
            if len(validation_errors) >= 2:
                break
            continue

        updater = {
            "backend": planner_result.stats.get("backend"),
            "model": planner_result.stats.get("model"),
            "reasoning_effort": planner_result.stats.get("reasoning_effort"),
            "sandbox": planner_result.stats.get("sandbox"),
            "usage": {
                key: planner_result.stats.get(key)
                for key in (
                    "total_input_tokens",
                    "total_cached_input_tokens",
                    "total_output_tokens",
                    "total_reasoning_output_tokens",
                    "tool_calls",
                )
            },
            "artifact_dir": str(attempt_dir),
            "proposal_sha256": canonical_sha256(parameters),
            "attempt": update_attempt,
            "submission_mode": submission_mode,
        }
        episode_ref = {
            key: evidence[key]
            for key in (
                "episode_id",
                "round",
                "seed",
                "valid",
                "success",
                "result_sha256",
                "memory_snapshot_sha256",
            )
        }
        updated = store.update(
            parameter_block=parameters,
            episode=episode_ref,
            updater=updater,
        )
        payload = {
            "accepted": True,
            "previous_revision": previous["revision"],
            "revision": updated["revision"],
            "memory_sha256": canonical_sha256(updated),
            "updater": updater,
        }
        _atomic_json(attempt_dir / "update_result.json", payload)
        _atomic_json(update_root / "accepted.json", payload)
        return payload
    raise RuntimeError(
        f"memory updater failed validation for {result['episode_id']}: "
        + " | ".join([*validation_errors, *transport_errors])
    )


def _existing_results(manifest: dict[str, Any], round_index: int) -> list[dict[str, Any]]:
    root = Path(manifest["output_root"]) / f"round-{round_index:02d}"
    return sorted(
        (_load_json(path) for path in root.glob("r*/result.json")),
        key=lambda row: (int(row["seed"]), int(row.get("attempt_index", 0))),
    )


def _formal_round_results(
    results: list[dict[str, Any]], seeds: list[int]
) -> list[dict[str, Any]]:
    selected = []
    for seed in seeds:
        candidates = [row for row in results if int(row["seed"]) == int(seed)]
        valid = next((row for row in candidates if row.get("valid")), None)
        if valid is not None:
            selected.append(valid)
    return selected


def analyze(manifest: dict[str, Any], store: TaskMemoryStore) -> dict[str, Any]:
    seeds = [int(seed) for seed in manifest["seeds"]]
    rounds = []
    chronological: list[dict[str, Any]] = []
    for round_index in range(int(manifest["max_rounds"])):
        attempts = _existing_results(manifest, round_index)
        formal = _formal_round_results(attempts, seeds)
        chronological.extend(formal)
        complete = len(formal) == len(seeds)
        success_count = sum(bool(row["success"]) for row in formal)
        rounds.append(
            {
                "round": round_index,
                "snapshot_revision": (
                    int(formal[0]["memory_snapshot_revision"]) if formal else None
                ),
                "attempts": len(attempts),
                "valid_seeds": len(formal),
                "complete": complete,
                "successes": success_count,
                "success_rate": success_count / len(formal) if formal else None,
                "episode_ids": [row["episode_id"] for row in formal],
            }
        )
    complete_rounds = [row for row in rounds if row["complete"]]
    baseline_rate = complete_rounds[0]["success_rate"] if complete_rounds else None
    improved_round = None
    if baseline_rate is not None:
        improved_round = next(
            (
                row["round"]
                for row in complete_rounds[1:]
                if row["success_rate"] > baseline_rate
            ),
            None,
        )
    rolling = []
    for index in range(len(chronological)):
        window = chronological[max(0, index - 7) : index + 1]
        if len(window) == 8:
            rolling.append(
                {
                    "episode_index": index + 1,
                    "last_episode_id": chronological[index]["episode_id"],
                    "successes": sum(bool(row["success"]) for row in window),
                    "success_rate": sum(bool(row["success"]) for row in window) / 8,
                }
            )
    ledger = Path(manifest["ledger_path"])
    reserved = 0
    if ledger.is_file():
        for line in ledger.read_text(encoding="utf-8").splitlines():
            try:
                reserved += json.loads(line).get("event") == "reserved"
            except json.JSONDecodeError:
                continue
    current = store.load()
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": manifest["protocol_id"],
        "generated_at": _utc_now(),
        "task_key": manifest["task_key"],
        "seeds": seeds,
        "rounds": rounds,
        "complete_rounds": len(complete_rounds),
        "baseline_rate": baseline_rate,
        "improved_round": improved_round,
        "improvement_observed": improved_round is not None,
        "rolling_last_8": rolling,
        "episodes_reserved": reserved,
        "episode_budget": int(manifest["budget_cap"]),
        "episodes_remaining": int(manifest["budget_cap"]) - reserved,
        "current_memory_revision": current["revision"],
        "current_memory_sha256": canonical_sha256(current),
    }


def run(manifest_path: Path) -> dict[str, Any]:
    manifest = _load_frozen_manifest(manifest_path)
    repo = Path(manifest["repo"])
    if _git(repo, "rev-parse", "HEAD") != manifest["commit"]:
        raise RuntimeError("repository commit does not match frozen manifest")
    if _git(repo, "status", "--porcelain"):
        raise RuntimeError("repository must be clean before running experiments")
    _verify_generic_memory(manifest)
    store = TaskMemoryStore(
        manifest["memory_root"],
        task_key=manifest["task_key"],
        suite=manifest["suite"],
        task=int(manifest["task"]),
    )
    store.initialize()
    output_root = Path(manifest["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    digest = _manifest_digest(manifest)
    seeds = [int(seed) for seed in manifest["seeds"]]

    for round_index in range(int(manifest["max_rounds"])):
        analysis = analyze(manifest, store)
        if analysis["improvement_observed"]:
            break
        round_dir = output_root / f"round-{round_index:02d}"
        round_dir.mkdir(parents=True, exist_ok=True)
        snapshot_path = round_dir / "memory_snapshot.json"
        if snapshot_path.exists():
            snapshot = _load_json(snapshot_path)
        else:
            snapshot = store.snapshot(snapshot_path)
        snapshot_sha = canonical_sha256(snapshot)
        round_guard = _memory_guard_snapshot(manifest, store)
        if round_guard["task_memory_sha256"] != snapshot_sha:
            # A resumed incomplete batch must continue to use its original frozen
            # snapshot even if completed rollouts already updated canonical memory.
            if not _existing_results(manifest, round_index):
                raise RuntimeError("fresh round snapshot does not match canonical memory")

        for attempt_index in range(int(manifest["max_attempts_per_seed"])):
            existing = _existing_results(manifest, round_index)
            formal = _formal_round_results(existing, seeds)
            done_seeds = {int(row["seed"]) for row in formal}
            pending = [seed for seed in seeds if seed not in done_seeds]
            if not pending:
                break
            wave_guard_before = _memory_guard_snapshot(manifest, store)
            jobs = []
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=int(manifest["concurrency"])
            ) as pool:
                for offset, seed in enumerate(pending):
                    jobs.append(
                        pool.submit(
                            _run_one_episode,
                            manifest=manifest,
                            manifest_digest=digest,
                            round_index=round_index,
                            seed=seed,
                            attempt_index=attempt_index,
                            gpu=offset % 8,
                            snapshot_path=snapshot_path,
                            snapshot_sha256=snapshot_sha,
                            snapshot_revision=int(snapshot["revision"]),
                            guard_before=wave_guard_before,
                        )
                    )
                completed = [job.result() for job in jobs]

            # Every rollout is now over before its updater can mutate memory.
            guard_after_rollouts = _memory_guard_snapshot(manifest, store)
            for key in (
                "task_memory_sha256",
                "task_memory_revision",
                "generic_memory_sha256",
                "generic_memory_file_count",
                "repo_status",
            ):
                if guard_after_rollouts.get(key) != wave_guard_before.get(key):
                    raise RuntimeError(f"memory guard changed during rollout wave: {key}")
            for result in sorted(completed, key=lambda row: int(row["seed"])):
                update_marker = (
                    output_root / "memory-updates" / result["episode_id"] / "accepted.json"
                )
                if not update_marker.exists():
                    _update_memory_after_episode(
                        manifest=manifest,
                        store=store,
                        result=result,
                    )

            guard_after = _memory_guard_snapshot(manifest, store)
            if guard_after["generic_memory_sha256"] != wave_guard_before[
                "generic_memory_sha256"
            ]:
                raise RuntimeError("generic memory changed during a rollout wave")
            if guard_after["repo_status"]:
                raise RuntimeError("repository changed during a rollout wave")

        formal = _formal_round_results(_existing_results(manifest, round_index), seeds)
        if len(formal) != len(seeds):
            raise RuntimeError(
                f"round {round_index} has only {len(formal)}/{len(seeds)} valid seeds "
                "after infrastructure retries"
            )
        report = analyze(manifest, store)
        _atomic_json(output_root / "analysis.json", report)
        if report["improvement_observed"]:
            break

    report = analyze(manifest, store)
    _atomic_json(output_root / "analysis.json", report)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    init = sub.add_parser("init")
    init.add_argument("--manifest", type=Path, required=True)
    init.add_argument("--protocol-id", default="zetta-episode-memory-loop-v1")
    init.add_argument("--repo", type=Path, required=True)
    init.add_argument("--python", type=Path, required=True)
    init.add_argument("--output-root", type=Path, required=True)
    init.add_argument("--memory-root", type=Path, required=True)
    init.add_argument("--ledger", type=Path, required=True)
    init.add_argument("--suite", default="libero_10")
    init.add_argument("--task", type=int, default=3)
    init.add_argument("--seeds", type=int, nargs=8, default=list(DEFAULT_SEEDS))
    init.add_argument("--budget-cap", type=int, default=MAX_ALLOWED_EPISODES)
    init.add_argument("--max-rounds", type=int, default=25)
    init.add_argument("--max-attempts-per-seed", type=int, default=3)
    init.add_argument("--concurrency", type=int, default=2)
    init.add_argument("--max-turns", type=int, default=100)
    init.add_argument("--max-episode-steps", type=int, default=10000)
    init.add_argument("--planner-timeout-s", type=int, default=1800)
    init.add_argument("--episode-timeout-s", type=int, default=3600)
    init.add_argument("--base-url")
    init.add_argument("--use-local-adapter-key-placeholder", action="store_true")
    init.add_argument("--contact-graspnet-endpoint")
    init.add_argument("--graspgen-endpoint")
    init.add_argument(
        "--disable-generic-memory",
        action="store_true",
        help="freeze legacy resources/libero/memory as explicitly disabled",
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
        print(json.dumps({"manifest": str(args.manifest), "sha256": _manifest_digest(manifest)}, indent=2))
        return 0
    manifest = _load_frozen_manifest(args.manifest)
    store = TaskMemoryStore(
        manifest["memory_root"],
        task_key=manifest["task_key"],
        suite=manifest["suite"],
        task=int(manifest["task"]),
    )
    store.initialize()
    report = run(args.manifest) if args.command == "run" else analyze(manifest, store)
    if args.command == "analyze":
        _atomic_json(Path(manifest["output_root"]) / "analysis.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
