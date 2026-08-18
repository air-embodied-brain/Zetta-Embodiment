#!/usr/bin/env python3
"""Run the frozen pure-VLA Loop 1 batch for PickPlaceToasterToCounter.

The runner is deliberately narrower than the general RPent campaign runner:
it has no candidate bundle, Role1, tool runtime, critic, recovery, planner, CAP,
or fallback path. Infrastructure-invalid attempts are append-only and retried
on the same seed until one valid, readable rollout is available.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import statistics
import subprocess
import sys
import time
import traceback
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


TASK = "PickPlaceToasterToCounter"
SPLIT = "target"
SEEDS = tuple(range(100, 150))
GENERATION = 0
FROZEN_EVALUATION_HORIZON = 1000
CAMERA_KEYS = (
    "video.robot0_agentview_left",
    "video.robot0_agentview_right",
    "video.robot0_eye_in_hand",
)
PRIMARY_CAMERA = "video.robot0_agentview_left"
EVALUATION_PREFIX = "privileged.pick_place."
MILESTONES = (
    {
        "id": "initial_object_in_toaster",
        "description": "The toasted item contacts a toaster slot at reset.",
    },
    {
        "id": "object_grasped",
        "description": "The official RoboCasa grasp predicate becomes true.",
    },
    {
        "id": "object_exited_toaster",
        "description": "After reset, the object no longer contacts a toaster slot.",
    },
    {
        "id": "object_contacted_plate",
        "description": "The toasted item contacts the target plate.",
    },
    {
        "id": "released_far_on_plate",
        "description": "The toasted item is on the plate and the gripper is over 0.25 m away.",
    },
    {
        "id": "authoritative_task_success",
        "description": "The environment emits its sparse success reward >= 1.0.",
    },
)
REQUIRED_EVALUATION_KEYS = (
    "privileged.pick_place.object_position_world",
    "privileged.pick_place.object_quaternion_wxyz",
    "privileged.pick_place.object_position_relative_to_toaster",
    "privileged.pick_place.object_position_relative_to_plate",
    "privileged.pick_place.object_position_relative_to_counter",
    "privileged.pick_place.object_toaster_contact",
    "privileged.pick_place.object_toaster_slot_contact",
    "privileged.pick_place.object_plate_contact",
    "privileged.pick_place.object_on_plate",
    "privileged.pick_place.object_any_counter_contact",
    "privileged.pick_place.gripper_object_distance",
    "privileged.pick_place.gripper_object_far",
    "privileged.pick_place.gripper_object_contact",
    "privileged.pick_place.object_grasped",
    "privileged.pick_place.object_contact_pairs",
    "privileged.pick_place.robot_contact_pairs",
    "privileged.pick_place.robot_non_target_contact",
    "privileged.pick_place.success_predicate",
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")


def _sha256_value(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_exclusive(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n"
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    payload = json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n"
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _append_jsonl(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, default=str) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            rows.append(value)
    return rows


def _http_json(base_url: str, path: str, timeout_s: float = 30.0) -> dict[str, Any]:
    request = urllib.request.Request(
        base_url.rstrip("/") + path,
        headers={"Accept": "application/json"},
        method="GET",
    )
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        value = json.loads(response.read().decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{base_url}{path} returned a non-object")
    return value


def _git(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return completed.stdout.strip()


def _source_hashes(repo: Path, robocasa_root: Path) -> dict[str, str]:
    paths = {
        "run_rollout": repo / "robots/robocasa/run_rollout.py",
        "env_server": repo / "robots/robocasa/env_server.py",
        "privileged_state": repo / "robots/robocasa/privileged_state.py",
        "groot_client": repo / "robots/robocasa/groot_client.py",
        "action_contract": repo / "robots/robocasa/action_contract.py",
        "batch_runner": repo / "scripts/evolution/run_pure_vla_loop1.py",
        "task_source": robocasa_root
        / "robocasa/environments/kitchen/atomic/kitchen_pick_place.py",
        "object_utils": robocasa_root / "robocasa/utils/object_utils.py",
    }
    return {name: _sha256_file(path) for name, path in paths.items()}


def _config_hash(config: dict[str, Any]) -> str:
    payload = dict(config)
    payload.pop("config_sha256", None)
    return _sha256_value(payload)


def _new_frozen_config(args: argparse.Namespace) -> dict[str, Any]:
    repo = Path(args.repo_root).resolve()
    robocasa_root = Path(args.robocasa_root).resolve()
    dirty = _git(repo, "status", "--porcelain")
    if dirty:
        raise RuntimeError("refusing to freeze a campaign from a dirty worktree")
    env_health = _http_json(args.env_endpoint, "/health")
    protocol = env_health.get("write_protocol", {})
    if not isinstance(protocol, dict) or protocol.get("phase") != "FREE":
        raise RuntimeError("RoboCasa slot must be FREE before freezing the batch")
    env_schema = _http_json(args.env_endpoint, "/schema")
    vla_health = _http_json(args.vla_endpoint, "/health")
    vla_schema = _http_json(args.vla_endpoint, "/schema")
    prompt_template = "Place the toasted item on a plate."
    config: dict[str, Any] = {
        "schema_version": 1,
        "created_at": _now(),
        "loop": "Loop 1: Pure-VLA Rollout Collection",
        "task": TASK,
        "split": SPLIT,
        "seeds": list(SEEDS),
        "one_valid_rollout_per_seed": True,
        "policy_rng_by_seed": {str(seed): seed for seed in SEEDS},
        "execution_mode": {
            "controller": "frozen_vla_only",
            "role1_planner": "none",
            "agent": False,
            "planner": False,
            "tool_runtime": "none_pure_vla",
            "runtime_critic": False,
            "recovery": False,
            "cap": False,
            "manual_fallback": False,
            "candidate_bundle": None,
            "critic_rules": [],
            "interrupt_on_proposal": False,
        },
        "policy": {
            "endpoint": args.vla_endpoint,
            "health": vla_health,
            "schema": vla_schema,
            "schema_sha256": _sha256_value(vla_schema),
            "inference_seed_algorithm": "sha256(f'{policy_rng}:{chunk_index}')[:4] & 0x7fffffff",
            "timeout_s": args.vla_timeout_s,
        },
        "prompt": {
            "source": "reset.observation.state.annotation.human.task_description",
            "override": None,
            "template": prompt_template,
            "template_sha256": hashlib.sha256(prompt_template.encode()).hexdigest(),
        },
        "simulator": {
            "endpoint": args.env_endpoint,
            "health": {
                "status": env_health.get("status"),
                "persistent": env_health.get("persistent"),
                "renderer": env_health.get("renderer"),
                "gpu_visible": env_health.get("gpu_visible"),
                "egl_device": env_health.get("egl_device"),
            },
            "schema": env_schema,
            "schema_sha256": _sha256_value(env_schema),
            "camera_size": args.camera_size,
            "max_steps": args.sim_max_steps,
            "action_scale": {
                "end_effector_position": 1.0,
                "end_effector_rotation": 1.0,
                "base_xy": 1.0,
                "base_yaw": 1.0,
                "torso": 1.0,
            },
            "rpc_timeout_s": args.rpc_timeout_s,
        },
        "rollout": {
            "max_actions": args.max_actions,
            "actions_per_chunk": args.actions_per_chunk,
            "maximum_infrastructure_attempts_per_seed": args.max_infrastructure_attempts,
            "episode_timeout_s": args.episode_timeout_s,
            "retry_policy": "preserve invalid attempt and rerun the same seed",
        },
        "milestones": list(MILESTONES),
        "final_progress_definition": "longest completed milestone prefix divided by six",
        "generation": GENERATION,
        "versions": {
            "repo_root": str(repo),
            "git_branch": _git(repo, "branch", "--show-current"),
            "git_commit": _git(repo, "rev-parse", "HEAD"),
            "source_sha256": _source_hashes(repo, robocasa_root),
        },
        "artifact_contract": {
            "preserve_successes": True,
            "preserve_failures": True,
            "preserve_infrastructure_invalid_attempts": True,
            "required_camera_videos": list(CAMERA_KEYS),
            "required_evaluation_keys": list(REQUIRED_EVALUATION_KEYS),
            "keyframe_source": PRIMARY_CAMERA,
        },
    }
    config["config_sha256"] = _config_hash(config)
    return config


def _load_or_create_config(args: argparse.Namespace, root: Path) -> dict[str, Any]:
    path = root / "frozen_batch_config.json"
    if path.exists():
        config = _read_json(path)
        if config.get("config_sha256") != _config_hash(config):
            raise RuntimeError("frozen batch configuration hash is invalid")
        current_commit = _git(Path(args.repo_root), "rev-parse", "HEAD")
        if config.get("versions", {}).get("git_commit") != current_commit:
            raise RuntimeError("repository commit differs from the frozen batch")
        if config.get("task") != TASK or tuple(config.get("seeds", ())) != SEEDS:
            raise RuntimeError("existing batch does not cover the required task/seeds")
        return config
    root.mkdir(parents=True, exist_ok=False)
    config = _new_frozen_config(args)
    _write_json_exclusive(path, config)
    return config


def _next_attempt_dir(seed_root: Path) -> tuple[int, Path]:
    seed_root.mkdir(parents=True, exist_ok=True)
    indexes: list[int] = []
    for path in seed_root.glob("attempt_*"):
        try:
            indexes.append(int(path.name.split("_", 1)[1]))
        except (IndexError, ValueError):
            continue
    index = max(indexes, default=0) + 1
    path = seed_root / f"attempt_{index:04d}"
    path.mkdir(exist_ok=False)
    return index, path


def _existing_canonical(seed_root: Path) -> Path | None:
    pointer = seed_root / "canonical_attempt.json"
    if not pointer.is_file():
        return None
    value = _read_json(pointer)
    path = Path(str(value.get("attempt_dir", "")))
    return path if path.is_dir() else None


def _run_command(
    args: argparse.Namespace,
    config: dict[str, Any],
    seed: int,
    attempt_index: int,
    attempt_dir: Path,
) -> tuple[int, list[str], str | None]:
    repo = Path(args.repo_root).resolve()
    result_path = attempt_dir / "episode_record.json"
    command = [
        args.python,
        str(repo / "robots/robocasa/run_rollout.py"),
        "--env-endpoint",
        args.env_endpoint,
        "--vla-endpoint",
        args.vla_endpoint,
        "--task",
        TASK,
        "--split",
        SPLIT,
        "--seed",
        str(seed),
        "--policy-rng",
        str(seed),
        "--logical-id",
        f"loop1-{TASK}-seed-{seed:05d}",
        "--attempt-index",
        str(attempt_index - 1),
        "--generation",
        str(GENERATION),
        "--bundle",
        "none",
        "--bundle-sha256",
        "none",
        "--role1-planner",
        "none",
        "--max-actions",
        str(config["rollout"]["max_actions"]),
        "--actions-per-chunk",
        str(config["rollout"]["actions_per_chunk"]),
        "--rpc-timeout-s",
        str(config["simulator"]["rpc_timeout_s"]),
        "--vla-timeout-s",
        str(config["policy"]["timeout_s"]),
        "--output-dir",
        str(attempt_dir),
        "--result-file",
        str(result_path),
    ]
    _write_json_exclusive(
        attempt_dir / "attempt_command.json",
        {
            "started_at": _now(),
            "seed": seed,
            "attempt_index": attempt_index,
            "config_sha256": config["config_sha256"],
            "argv": command,
        },
    )
    environment = dict(os.environ)
    python_path = [str(repo)]
    existing = environment.get("PYTHONPATH")
    if existing:
        python_path.append(existing)
    environment["PYTHONPATH"] = os.pathsep.join(python_path)
    timeout_error = None
    with (attempt_dir / "runner_stdout.log").open("x", encoding="utf-8") as stdout:
        with (attempt_dir / "runner_stderr.log").open("x", encoding="utf-8") as stderr:
            try:
                completed = subprocess.run(
                    command,
                    cwd=repo,
                    env=environment,
                    stdout=stdout,
                    stderr=stderr,
                    timeout=float(config["rollout"]["episode_timeout_s"]),
                    check=False,
                )
                return_code = int(completed.returncode)
            except subprocess.TimeoutExpired as exc:
                return_code = 124
                timeout_error = f"TimeoutExpired: {exc}"
    return return_code, command, timeout_error


def _ffprobe(path: Path) -> dict[str, Any]:
    completed = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name,width,height,nb_frames,duration",
            "-show_entries",
            "format=duration,size",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=60,
    )
    value = json.loads(completed.stdout)
    if not isinstance(value, dict) or not value.get("streams"):
        raise ValueError(f"ffprobe found no video stream in {path}")
    return value


def _extract_keyframe(video: Path, frame_index: int, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    expression = f"select=eq(n\\,{max(0, int(frame_index))})"
    completed = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-i",
            str(video),
            "-vf",
            expression,
            "-frames:v",
            "1",
            str(output),
        ],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=60,
    )
    if completed.returncode != 0 or not output.is_file() or output.stat().st_size == 0:
        raise RuntimeError(
            f"keyframe extraction failed for frame {frame_index}: {completed.stderr[-400:]}"
        )


def _state_rows(attempt_dir: Path) -> list[dict[str, Any]]:
    rows = _read_jsonl(attempt_dir / "trajectory/states.jsonl")
    if not rows or int(rows[0].get("step_index", -1)) != 0:
        raise ValueError("state trajectory must start with reset step 0")
    return rows


def _action_rows(attempt_dir: Path) -> list[dict[str, Any]]:
    return _read_jsonl(attempt_dir / "trajectory/actions.jsonl")


def _object_relative_state(state: dict[str, Any]) -> dict[str, Any]:
    return {
        key.removeprefix(EVALUATION_PREFIX): value
        for key, value in sorted(state.items())
        if key.startswith(EVALUATION_PREFIX)
    }


def _milestone_steps(
    states: list[dict[str, Any]], success: bool
) -> tuple[dict[str, int | None], list[str], str | None, float]:
    initial_state = states[0].get("state", {})
    if not isinstance(initial_state, dict):
        raise ValueError("reset state is not an object")
    initial_in_toaster = bool(
        initial_state.get(
            "privileged.pick_place.object_toaster_slot_contact", False
        )
    )
    steps: dict[str, int | None] = {
        milestone["id"]: None for milestone in MILESTONES
    }
    if initial_in_toaster:
        steps["initial_object_in_toaster"] = 0
    for row in states:
        step = int(row.get("step_index", 0))
        state = row.get("state", {})
        if not isinstance(state, dict):
            continue
        if steps["object_grasped"] is None and bool(
            state.get("privileged.pick_place.object_grasped", False)
        ):
            steps["object_grasped"] = step
        if (
            initial_in_toaster
            and step > 0
            and steps["object_exited_toaster"] is None
            and not bool(
                state.get("privileged.pick_place.object_toaster_slot_contact", True)
            )
        ):
            steps["object_exited_toaster"] = step
        plate_contact = bool(
            state.get("privileged.pick_place.object_plate_contact", False)
        )
        if steps["object_contacted_plate"] is None and plate_contact:
            steps["object_contacted_plate"] = step
        if (
            steps["released_far_on_plate"] is None
            and bool(state.get("privileged.pick_place.object_on_plate", False))
            and bool(state.get("privileged.pick_place.gripper_object_far", False))
        ):
            steps["released_far_on_plate"] = step
    if success:
        steps["authoritative_task_success"] = int(states[-1].get("step_index", 0))
    completed = [milestone["id"] for milestone in MILESTONES if steps[milestone["id"]] is not None]
    first_missing = next(
        (milestone["id"] for milestone in MILESTONES if steps[milestone["id"]] is None),
        None,
    )
    prefix_count = 0
    for milestone in MILESTONES:
        if steps[milestone["id"]] is None:
            break
        prefix_count += 1
    progress = prefix_count / len(MILESTONES)
    return steps, completed, first_missing, progress


def _termination_reason(record: dict[str, Any], states: list[dict[str, Any]]) -> str:
    if record.get("success") is True:
        return "authoritative_task_success"
    final = states[-1]
    if bool(final.get("truncated")):
        return "simulator_max_steps"
    if bool(final.get("terminated")):
        return "simulator_terminated_without_success"
    actions = int(record.get("artifact_index", {}).get("actions_executed", 0))
    return "max_actions" if actions > 0 else "no_action_executed"


def _audit_purity(record: dict[str, Any], attempt_dir: Path) -> dict[str, Any]:
    artifact_index = record.get("artifact_index", {})
    tool_runtime = artifact_index.get("tool_runtime", {})
    tool_events = attempt_dir / "tool_events.jsonl"
    chunks = _read_jsonl(attempt_dir / "trajectory/chunks.jsonl")
    violations: list[str] = []
    if record.get("bundle_sha256") is not None:
        violations.append("candidate_bundle_present")
    if tool_runtime.get("backend") != "none_pure_vla":
        violations.append("tool_runtime_enabled")
    if int(artifact_index.get("role1_decisions", -1)) != 0:
        violations.append("role1_decisions_present")
    if tool_events.stat().st_size != 0:
        violations.append("tool_events_present")
    if (attempt_dir / "role1").exists():
        violations.append("role1_artifacts_present")
    for chunk in chunks:
        environment = chunk.get("environment", {})
        if environment.get("critic_proposals"):
            violations.append("critic_proposal_present")
        for step in environment.get("steps", ()):
            if step.get("proposal_rule_ids"):
                violations.append("critic_rule_trigger_present")
    return {
        "pure_vla": not violations,
        "violations": sorted(set(violations)),
        "tool_events_size": tool_events.stat().st_size,
        "tool_runtime_backend": tool_runtime.get("backend"),
        "role1_decisions": artifact_index.get("role1_decisions"),
    }


def _postprocess_valid_attempt(
    config: dict[str, Any], seed: int, attempt_dir: Path
) -> dict[str, Any]:
    record = _read_json(attempt_dir / "episode_record.json")
    if record.get("status") != "valid" or not isinstance(record.get("success"), bool):
        raise ValueError("episode record is not a valid rollout")
    if int(record.get("seed", -1)) != seed:
        raise ValueError("episode record seed mismatch")
    states = _state_rows(attempt_dir)
    actions = _action_rows(attempt_dir)
    if len(actions) != len(states) - 1:
        raise ValueError("action and post-reset state counts disagree")
    for row in states:
        state = row.get("state")
        if not isinstance(state, dict):
            raise ValueError("state trajectory contains a non-object state")
        missing = [key for key in REQUIRED_EVALUATION_KEYS if key not in state]
        if missing:
            raise ValueError(
                f"evaluation state keys missing at step {row.get('step_index')}: {missing}"
            )

    purity = _audit_purity(record, attempt_dir)
    if not purity["pure_vla"]:
        raise ValueError(f"pure-VLA audit failed: {purity['violations']}")
    videos = record.get("artifact_index", {}).get("videos", {})
    video_probe: dict[str, Any] = {}
    for camera in CAMERA_KEYS:
        path = Path(str(videos.get(camera, "")))
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"missing rollout video for {camera}")
        video_probe[camera] = _ffprobe(path)

    success = bool(record["success"])
    milestone_steps, completed, first_missing, progress = _milestone_steps(states, success)
    primary_video = Path(str(videos[PRIMARY_CAMERA]))
    keyframes: dict[str, str] = {}
    frame_requests: dict[str, int] = {"initial": 0}
    for milestone, step in milestone_steps.items():
        if step is not None:
            frame_requests[milestone] = int(step)
    frame_requests["final"] = int(states[-1].get("step_index", 0))
    for name, frame in frame_requests.items():
        output = attempt_dir / "keyframes" / f"{name}-step-{frame:04d}.jpg"
        _extract_keyframe(primary_video, frame, output)
        keyframes[name] = str(output)

    object_stream = [
        {
            "step_index": int(row.get("step_index", 0)),
            "object_relative_state": _object_relative_state(row["state"]),
        }
        for row in states
    ]
    _write_json_exclusive(
        attempt_dir / "trajectory/object_relative_state.json",
        {
            "schema_version": 1,
            "task": TASK,
            "seed": seed,
            "states": object_stream,
        },
    )
    final = states[-1]
    final_reward = float(final.get("reward", 0.0) or 0.0)
    contact_summary = {
        "object_grasped_ever": any(
            bool(row["state"].get("privileged.pick_place.object_grasped", False))
            for row in states
        ),
        "gripper_object_contact_ever": any(
            bool(row["state"].get("privileged.pick_place.gripper_object_contact", False))
            for row in states
        ),
        "object_toaster_slot_contact_ever": any(
            bool(
                row["state"].get(
                    "privileged.pick_place.object_toaster_slot_contact", False
                )
            )
            for row in states
        ),
        "object_plate_contact_ever": any(
            bool(
                row["state"].get(
                    "privileged.pick_place.object_plate_contact", False
                )
            )
            for row in states
        ),
        "object_on_plate_ever": any(
            bool(
                row["state"].get(
                    "privileged.pick_place.object_on_plate", False
                )
            )
            for row in states
        ),
        "robot_non_target_contact_ever": any(
            bool(row["state"].get("privileged.pick_place.robot_non_target_contact", False))
            for row in states
        ),
        "object_contact_pair_samples": sum(
            len(row["state"].get("privileged.pick_place.object_contact_pairs", ()))
            for row in states
        ),
        "robot_contact_pair_samples": sum(
            len(row["state"].get("privileged.pick_place.robot_contact_pairs", ()))
            for row in states
        ),
    }
    instruction = str(states[0]["state"].get("annotation.human.task_description", ""))
    result = {
        "schema_version": 1,
        "valid": True,
        "seed": seed,
        "attempt_dir": str(attempt_dir),
        "attempt_index": int(record.get("attempt_index", 0)) + 1,
        "episode_id": record.get("episode_id"),
        "config_sha256": config["config_sha256"],
        "policy_version": config["policy"],
        "prompt": {
            "instruction": instruction,
            "instruction_sha256": hashlib.sha256(instruction.encode()).hexdigest(),
            "template_sha256": config["prompt"]["template_sha256"],
        },
        "simulator_version": {
            "git_commit": config["versions"]["git_commit"],
            "source_sha256": config["versions"]["source_sha256"],
            "schema_sha256": config["simulator"]["schema_sha256"],
        },
        "initial_state": states[0]["state"],
        "success": success,
        "reward": final_reward,
        "final_task_progress": progress,
        "completed_milestones": completed,
        "first_missing_milestone": first_missing,
        "milestone_first_steps": milestone_steps,
        "final_state": final["state"],
        "contact_collision_grasp_summary": contact_summary,
        "termination_reason": _termination_reason(record, states),
        "actions_executed": len(actions),
        "vla_chunks": int(record.get("artifact_index", {}).get("vla_chunks", 0)),
        "elapsed_s": float(record.get("elapsed_s", 0.0)),
        "purity_audit": purity,
        "errors": {
            "inference": [],
            "simulator": [],
            "infrastructure": [],
        },
        "artifacts": {
            "episode_record": str(attempt_dir / "episode_record.json"),
            "states": str(attempt_dir / "trajectory/states.jsonl"),
            "actions": str(attempt_dir / "trajectory/actions.jsonl"),
            "chunks": str(attempt_dir / "trajectory/chunks.jsonl"),
            "object_relative_state": str(
                attempt_dir / "trajectory/object_relative_state.json"
            ),
            "tool_events": str(attempt_dir / "tool_events.jsonl"),
            "videos": videos,
            "video_probe": video_probe,
            "keyframes": keyframes,
        },
    }
    _write_json_exclusive(attempt_dir / "rollout_result.json", result)
    return result


def _invalid_category(reason: str) -> str:
    normalized = reason.lower()
    if "gr00t" in normalized or "vla" in normalized or "model" in normalized:
        return "model_service_or_inference"
    if "robocasa" in normalized or "simulator" in normalized or "mujoco" in normalized:
        return "simulator"
    if "log" in normalized or "artifact" in normalized or "video" in normalized:
        return "logging_or_artifact"
    return "infrastructure"


def _mark_invalid(
    seed: int,
    attempt_index: int,
    attempt_dir: Path,
    return_code: int,
    reason: str,
) -> dict[str, Any]:
    value = {
        "schema_version": 1,
        "status": "infra_invalid",
        "seed": seed,
        "attempt_index": attempt_index,
        "recorded_at": _now(),
        "return_code": return_code,
        "category": _invalid_category(reason),
        "reason": reason,
        "preserved_attempt_dir": str(attempt_dir),
        "artifact_paths": [str(path) for path in sorted(attempt_dir.rglob("*")) if path.is_file()],
    }
    _write_json_exclusive(attempt_dir / "infrastructure_invalid.json", value)
    return value


def _attempt_invalid_reason(attempt_dir: Path, return_code: int, timeout: str | None) -> str:
    if timeout:
        return timeout
    infrastructure = attempt_dir / "infrastructure_error.json"
    if infrastructure.is_file():
        try:
            value = _read_json(infrastructure)
            return str(value.get("error", value))
        except Exception as exc:
            return f"unreadable infrastructure_error.json: {exc}"
    return f"run_rollout exited {return_code} without a valid episode record"


def _run_seed(
    args: argparse.Namespace,
    config: dict[str, Any],
    root: Path,
    seed: int,
) -> dict[str, Any]:
    seed_root = root / "seeds" / f"seed_{seed:05d}"
    existing = _existing_canonical(seed_root)
    if existing is not None:
        return _read_json(existing / "rollout_result.json")
    for _ in range(int(config["rollout"]["maximum_infrastructure_attempts_per_seed"])):
        attempt_index, attempt_dir = _next_attempt_dir(seed_root)
        _append_jsonl(
            root / "campaign_events.jsonl",
            {
                "type": "attempt_started",
                "at": _now(),
                "seed": seed,
                "attempt_index": attempt_index,
                "attempt_dir": str(attempt_dir),
            },
        )
        return_code, _, timeout = _run_command(
            args, config, seed, attempt_index, attempt_dir
        )
        result_path = attempt_dir / "episode_record.json"
        if return_code == 0 and result_path.is_file():
            try:
                result = _postprocess_valid_attempt(config, seed, attempt_dir)
            except Exception as exc:
                reason = f"logging/artifact validation failed: {type(exc).__name__}: {exc}"
                invalid = _mark_invalid(
                    seed, attempt_index, attempt_dir, return_code, reason
                )
                _append_jsonl(root / "campaign_events.jsonl", invalid)
                continue
            _write_json_exclusive(
                seed_root / "canonical_attempt.json",
                {
                    "seed": seed,
                    "attempt_index": attempt_index,
                    "attempt_dir": str(attempt_dir),
                    "rollout_result": str(attempt_dir / "rollout_result.json"),
                    "selected_at": _now(),
                },
            )
            _append_jsonl(
                root / "campaign_events.jsonl",
                {
                    "type": "valid_rollout_saved",
                    "at": _now(),
                    "seed": seed,
                    "attempt_index": attempt_index,
                    "success": result["success"],
                    "attempt_dir": str(attempt_dir),
                },
            )
            return result
        reason = _attempt_invalid_reason(attempt_dir, return_code, timeout)
        invalid = _mark_invalid(seed, attempt_index, attempt_dir, return_code, reason)
        _append_jsonl(root / "campaign_events.jsonl", invalid)
        health = _http_json(args.env_endpoint, "/health")
        protocol = health.get("write_protocol", {})
        if not isinstance(protocol, dict) or protocol.get("phase") != "FREE":
            raise RuntimeError(
                f"seed {seed} attempt {attempt_index} was invalid and the "
                "environment slot is not FREE"
            )
    raise RuntimeError(
        f"seed {seed} exceeded the frozen infrastructure retry limit without a valid rollout"
    )


def _attempt_paths(seed_root: Path) -> list[str]:
    return [str(path) for path in sorted(seed_root.glob("attempt_*")) if path.is_dir()]


def _local_window(
    result: dict[str, Any], milestone: str, radius: int = 2
) -> list[dict[str, Any]]:
    step = result["milestone_first_steps"].get(milestone)
    if step is None:
        return []
    attempt = Path(result["attempt_dir"])
    states = {int(row["step_index"]): row for row in _state_rows(attempt)}
    actions = {int(row["step_index"]): row for row in _action_rows(attempt)}
    window: list[dict[str, Any]] = []
    for index in range(max(0, int(step) - radius), int(step) + radius + 1):
        state_row = states.get(index)
        if state_row is None:
            continue
        window.append(
            {
                "step_index": index,
                "object_relative_state": _object_relative_state(state_row["state"]),
                "action": actions.get(index, {}).get("action"),
                "action_sha256": actions.get(index, {}).get("action_sha256"),
            }
        )
    return window


def _aggregate(root: Path, config: dict[str, Any], results: list[dict[str, Any]]) -> None:
    ordered = sorted(results, key=lambda item: int(item["seed"]))
    table: list[dict[str, Any]] = []
    artifact_manifest: list[dict[str, Any]] = []
    successful_references: list[dict[str, Any]] = []
    failed_manifest: list[dict[str, Any]] = []
    invalid_attempt_manifest: list[dict[str, Any]] = []
    for result in ordered:
        seed = int(result["seed"])
        seed_root = root / "seeds" / f"seed_{seed:05d}"
        invalid_attempts = []
        for attempt in sorted(seed_root.glob("attempt_*")):
            invalid_path = attempt / "infrastructure_invalid.json"
            if invalid_path.is_file():
                invalid = _read_json(invalid_path)
                invalid_attempts.append(invalid)
                invalid_attempt_manifest.append(invalid)
        row = {
            "seed": seed,
            "success": bool(result["success"]),
            "reward": float(result["reward"]),
            "final_task_progress": float(result["final_task_progress"]),
            "completed_milestones": result["completed_milestones"],
            "first_missing_milestone": result["first_missing_milestone"],
            "termination_reason": result["termination_reason"],
            "actions_executed": int(result["actions_executed"]),
            "elapsed_s": float(result["elapsed_s"]),
            "canonical_attempt_dir": result["attempt_dir"],
            "all_attempt_dirs": _attempt_paths(seed_root),
            "infrastructure_invalid_attempt_count": len(invalid_attempts),
            "rollout_result": str(Path(result["attempt_dir"]) / "rollout_result.json"),
        }
        table.append(row)
        artifact_manifest.append(
            {
                "seed": seed,
                "success": result["success"],
                "canonical_attempt_dir": result["attempt_dir"],
                "all_attempt_dirs": row["all_attempt_dirs"],
                "artifacts": result["artifacts"],
            }
        )
        if result["success"]:
            for milestone in result["completed_milestones"]:
                step = result["milestone_first_steps"].get(milestone)
                if step is None:
                    continue
                states = _state_rows(Path(result["attempt_dir"]))
                state_by_step = {int(item["step_index"]): item for item in states}
                successful_references.append(
                    {
                        "task": TASK,
                        "milestone": milestone,
                        "seed": seed,
                        "step_index": step,
                        "policy_version": {
                            "checkpoint": config["policy"]["health"].get("checkpoint"),
                            "schema_sha256": config["policy"]["schema_sha256"],
                            "git_commit": config["versions"]["git_commit"],
                        },
                        "object_relative_state": _object_relative_state(
                            state_by_step[int(step)]["state"]
                        ),
                        "local_trajectory_window": _local_window(result, milestone),
                        "artifact_paths": result["artifacts"],
                    }
                )
        else:
            failed_manifest.append(
                {
                    "seed": seed,
                    "success": False,
                    "reward": result["reward"],
                    "final_task_progress": result["final_task_progress"],
                    "completed_milestones": result["completed_milestones"],
                    "first_missing_milestone": result["first_missing_milestone"],
                    "termination_reason": result["termination_reason"],
                    "canonical_attempt_dir": result["attempt_dir"],
                    "artifact_paths": result["artifacts"],
                    "preserved_infrastructure_invalid_attempts": invalid_attempts,
                }
            )

    success_count = sum(1 for item in table if item["success"])
    rewards = [float(item["reward"]) for item in table]
    progress = [float(item["final_task_progress"]) for item in table]
    milestone_stats = {}
    for milestone in MILESTONES:
        milestone_id = milestone["id"]
        count = sum(
            1 for item in table if milestone_id in item["completed_milestones"]
        )
        milestone_stats[milestone_id] = {
            "count": count,
            "rate": count / len(table),
        }
    aggregate = {
        "schema_version": 1,
        "generated_at": _now(),
        "config_sha256": config["config_sha256"],
        "valid_rollouts": len(table),
        "expected_rollouts": len(SEEDS),
        "success_count": success_count,
        "success_rate": success_count / len(table),
        "reward": {
            "mean": statistics.fmean(rewards),
            "median": statistics.median(rewards),
            "min": min(rewards),
            "max": max(rewards),
        },
        "final_task_progress": {
            "mean": statistics.fmean(progress),
            "median": statistics.median(progress),
            "min": min(progress),
            "max": max(progress),
        },
        "milestone_completion": milestone_stats,
    }
    _write_json_atomic(root / "per_seed_results.json", {"results": table})
    with (root / "per_seed_results.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        fieldnames = [
            "seed",
            "success",
            "reward",
            "final_task_progress",
            "completed_milestones",
            "first_missing_milestone",
            "termination_reason",
            "actions_executed",
            "elapsed_s",
            "canonical_attempt_dir",
            "all_attempt_dirs",
            "infrastructure_invalid_attempt_count",
            "rollout_result",
        ]
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in table:
            encoded = dict(row)
            encoded["completed_milestones"] = json.dumps(
                row["completed_milestones"], ensure_ascii=False
            )
            encoded["all_attempt_dirs"] = json.dumps(
                row["all_attempt_dirs"], ensure_ascii=False
            )
            writer.writerow(encoded)
    _write_json_atomic(root / "aggregate_report.json", aggregate)
    _write_json_atomic(
        root / "artifact_manifest.json", {"rollouts": artifact_manifest}
    )
    _write_json_atomic(
        root / "successful_trajectory_reference_index.json",
        {"references": successful_references},
    )
    _write_json_atomic(
        root / "failed_seed_manifest.json",
        {"failed_seeds": failed_manifest, "loop2_stage": "Loop 2 Stage 1"},
    )
    _write_json_atomic(
        root / "infrastructure_invalid_attempt_manifest.json",
        {"invalid_attempts": invalid_attempt_manifest},
    )


def _final_readability_audit(root: Path, results: list[dict[str, Any]]) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    errors: list[str] = []
    for result in sorted(results, key=lambda item: int(item["seed"])):
        seed = int(result["seed"])
        attempt = Path(result["attempt_dir"])
        item_errors: list[str] = []
        try:
            _read_json(attempt / "episode_record.json")
            _read_json(attempt / "rollout_result.json")
            _read_json(attempt / "trajectory/object_relative_state.json")
            _read_jsonl(attempt / "trajectory/states.jsonl")
            _read_jsonl(attempt / "trajectory/actions.jsonl")
            _read_jsonl(attempt / "trajectory/chunks.jsonl")
            if (attempt / "tool_events.jsonl").stat().st_size != 0:
                item_errors.append("tool_events_not_empty")
            for path in result["artifacts"]["videos"].values():
                _ffprobe(Path(path))
            for path in result["artifacts"]["keyframes"].values():
                image = Path(path)
                if not image.is_file() or image.stat().st_size == 0:
                    item_errors.append(f"unreadable_keyframe:{image}")
        except Exception as exc:
            item_errors.append(f"{type(exc).__name__}: {exc}")
        entries.append(
            {
                "seed": seed,
                "attempt_dir": str(attempt),
                "readable": not item_errors,
                "errors": item_errors,
            }
        )
        errors.extend(f"seed {seed}: {error}" for error in item_errors)
    return {
        "schema_version": 1,
        "audited_at": _now(),
        "all_readable": not errors,
        "valid_rollout_count": len(results),
        "entries": entries,
        "errors": errors,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--robocasa-root", required=True)
    parser.add_argument("--env-endpoint", required=True)
    parser.add_argument("--vla-endpoint", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--camera-size", type=int, default=256)
    parser.add_argument(
        "--sim-max-steps", type=int, default=FROZEN_EVALUATION_HORIZON
    )
    parser.add_argument(
        "--max-actions", type=int, default=FROZEN_EVALUATION_HORIZON
    )
    parser.add_argument("--actions-per-chunk", type=int, default=16)
    parser.add_argument("--rpc-timeout-s", type=float, default=300.0)
    parser.add_argument("--vla-timeout-s", type=float, default=180.0)
    parser.add_argument("--episode-timeout-s", type=float, default=1800.0)
    parser.add_argument("--max-infrastructure-attempts", type=int, default=8)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        raise RuntimeError("ffmpeg and ffprobe are required for artifact validation")
    if (
        args.max_actions != FROZEN_EVALUATION_HORIZON
        or args.sim_max_steps != FROZEN_EVALUATION_HORIZON
    ):
        raise ValueError(
            "this frozen Loop 1 protocol requires a 1000-action horizon"
        )
    if args.actions_per_chunk != 16:
        raise ValueError("this frozen Loop 1 protocol requires 16 actions per VLA chunk")
    root = Path(args.output_root).resolve()
    config = _load_or_create_config(args, root)
    _append_jsonl(
        root / "campaign_events.jsonl",
        {
            "type": "batch_started_or_resumed",
            "at": _now(),
            "config_sha256": config["config_sha256"],
        },
    )
    results: list[dict[str, Any]] = []
    for seed in SEEDS:
        result = _run_seed(args, config, root, seed)
        results.append(result)
        _aggregate(root, config, results)
    if {int(item["seed"]) for item in results} != set(SEEDS):
        raise RuntimeError("the canonical result set does not exactly cover seeds 100-149")
    audit = _final_readability_audit(root, results)
    _write_json_atomic(root / "readability_audit.json", audit)
    if not audit["all_readable"]:
        raise RuntimeError("final artifact readability audit failed")
    completion = {
        "schema_version": 1,
        "loop1_complete": True,
        "completed_at": _now(),
        "config_sha256": config["config_sha256"],
        "valid_rollout_count": len(results),
        "seeds": list(SEEDS),
        "outputs": {
            "frozen_batch_config": str(root / "frozen_batch_config.json"),
            "per_seed_results_json": str(root / "per_seed_results.json"),
            "per_seed_results_csv": str(root / "per_seed_results.csv"),
            "aggregate_report": str(root / "aggregate_report.json"),
            "artifact_manifest": str(root / "artifact_manifest.json"),
            "successful_reference_index": str(
                root / "successful_trajectory_reference_index.json"
            ),
            "failed_seed_manifest": str(root / "failed_seed_manifest.json"),
            "infrastructure_invalid_attempt_manifest": str(
                root / "infrastructure_invalid_attempt_manifest.json"
            ),
            "readability_audit": str(root / "readability_audit.json"),
        },
    }
    _write_json_exclusive(root / "loop1_complete.json", completion)
    _append_jsonl(root / "campaign_events.jsonl", completion)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        sys.stderr.write(f"{type(exc).__name__}: {exc}\n")
        sys.stderr.write(traceback.format_exc())
        raise
