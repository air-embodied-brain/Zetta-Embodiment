"""Budget-guarded runner for paired RPent LIBERO experiments.

The manifest contains no credentials.  API keys remain in the inherited
environment.  Every attempted launch is reserved in an append-only JSONL
ledger before the simulator is started, so crashes and invalid runs still
consume the hard episode budget.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1
MAX_ALLOWED_EPISODES = 100


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _canonical_digest(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _append_locked(path: Path, row: dict[str, Any], *, reserve: bool = False) -> int:
    """Append one ledger row under an exclusive Linux file lock."""
    import fcntl

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        handle.seek(0)
        rows = []
        for line in handle:
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                rows.append(value)
        reservations = [item for item in rows if item.get("event") == "reserved"]
        if reserve:
            budget_cap = int(row["budget_cap"])
            if len(reservations) >= budget_cap:
                raise RuntimeError(
                    f"episode budget exhausted: {len(reservations)}/{budget_cap}"
                )
            duplicate = [
                item
                for item in reservations
                if item.get("episode_id") == row.get("episode_id")
            ]
            if duplicate:
                raise RuntimeError(
                    f"episode {row['episode_id']!r} was already reserved; "
                    "use a distinct manifest episode id for any retry"
                )
        handle.seek(0, os.SEEK_END)
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        return len(reservations) + (1 if reserve else 0)


def _validate_manifest(manifest: dict[str, Any]) -> None:
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"manifest schema_version must be {SCHEMA_VERSION}")
    cap = int(manifest.get("budget_cap", 0))
    if not 1 <= cap <= MAX_ALLOWED_EPISODES:
        raise ValueError(f"budget_cap must be in [1,{MAX_ALLOWED_EPISODES}]")
    if not isinstance(manifest.get("protocol_id"), str) or not manifest[
        "protocol_id"
    ].strip():
        raise ValueError("manifest requires a non-empty protocol_id")
    common = manifest.get("common")
    episodes = manifest.get("episodes")
    if not isinstance(common, dict) or not isinstance(episodes, list):
        raise ValueError("manifest requires object 'common' and array 'episodes'")
    if len(episodes) > cap:
        raise ValueError("manifest contains more episodes than budget_cap")
    ids = [row.get("id") for row in episodes if isinstance(row, dict)]
    if len(ids) != len(episodes) or any(not value for value in ids):
        raise ValueError("every episode requires a non-empty id")
    if len(set(ids)) != len(ids):
        raise ValueError("episode ids must be unique")
    if common.get("model") != "gpt-5.6-terra":
        raise ValueError("the registered protocol requires model gpt-5.6-terra")
    if common.get("reasoning_effort") != "medium":
        raise ValueError("the registered protocol requires reasoning_effort medium")


def _ledger_path(manifest_path: Path, manifest: dict[str, Any]) -> Path:
    """Resolve one global ledger shared by all manifests in this protocol."""
    raw = manifest.get("ledger_path", "episode_ledger.jsonl")
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError("ledger_path must be a non-empty path string")
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = manifest_path.parent / path
    return path.resolve()


def _git(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_tree(path: Path) -> tuple[str, int]:
    """Hash a directory as sorted ``sha256  relative/path`` records."""
    if not path.is_dir():
        raise FileNotFoundError(f"frozen resource tree is missing: {path}")
    digest = hashlib.sha256()
    files = sorted(item for item in path.rglob("*") if item.is_file())
    for item in files:
        relative = item.relative_to(path).as_posix()
        digest.update(f"{_sha256_file(item)}  {relative}\n".encode("utf-8"))
    return digest.hexdigest(), len(files)


def _verify_file_snapshot(snapshot: dict[str, Any]) -> None:
    files = snapshot.get("files", [])
    if not isinstance(files, list):
        raise ValueError("resource_snapshot.files must be an array")
    for row in files:
        if not isinstance(row, dict) or not row.get("path") or not row.get("sha256"):
            raise ValueError("each resource snapshot file requires path and sha256")
        path = Path(row["path"]).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"frozen resource is missing: {path}")
        actual = _sha256_file(path)
        if actual != row["sha256"]:
            raise RuntimeError(
                f"frozen resource hash mismatch for {path}: "
                f"expected {row['sha256']}, got {actual}"
            )
    trees = snapshot.get("trees", [])
    if not isinstance(trees, list):
        raise ValueError("resource_snapshot.trees must be an array")
    for row in trees:
        if not isinstance(row, dict) or not row.get("path") or not row.get("sha256"):
            raise ValueError("each resource snapshot tree requires path and sha256")
        path = Path(row["path"]).expanduser().resolve()
        actual, file_count = _sha256_tree(path)
        if actual != row["sha256"]:
            raise RuntimeError(
                f"frozen resource tree hash mismatch for {path}: "
                f"expected {row['sha256']}, got {actual}"
            )
        expected_count = row.get("file_count")
        if expected_count is not None and file_count != int(expected_count):
            raise RuntimeError(
                f"frozen resource tree file-count mismatch for {path}: "
                f"expected {expected_count}, got {file_count}"
            )


def _load_last_json_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    text = path.read_text(encoding="utf-8", errors="replace")
    decoder = json.JSONDecoder()
    cursor = 0
    last: dict[str, Any] = {}
    while cursor < len(text):
        while cursor < len(text) and text[cursor].isspace():
            cursor += 1
        if cursor >= len(text):
            break
        try:
            value, cursor = decoder.raw_decode(text, cursor)
        except json.JSONDecodeError:
            break
        if isinstance(value, dict):
            last = value
    return last


def _video_snapshot(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "exists": path.is_file(),
        "size_bytes": path.stat().st_size if path.is_file() else 0,
    }


def _camera_artifact_health(output_dir: Path) -> dict[str, Any]:
    """Audit saved LIBERO camera/depth artifacts without using task labels."""

    import numpy as np
    from PIL import Image

    image_dirs = {
        "agent_low": output_dir / "images_cam",
        "wrist_low": output_dir / "images_wrist",
        "agent_high": output_dir / "images_cam_hi",
        "wrist_high": output_dir / "images_wrist_hi",
    }

    def sampled(paths: list[Path], limit: int = 12) -> list[Path]:
        if len(paths) <= limit:
            return paths
        indices = sorted({round(i * (len(paths) - 1) / (limit - 1)) for i in range(limit)})
        return [paths[index] for index in indices]

    issues: list[str] = []
    images: dict[str, list[dict[str, Any]]] = {}
    digests: dict[str, dict[str, str]] = {}
    for label, directory in image_dirs.items():
        paths = sampled(sorted(directory.glob("*.png"))) if directory.is_dir() else []
        images[label] = []
        digests[label] = {}
        for path in paths:
            try:
                array = np.asarray(Image.open(path).convert("RGB"))
                digest = hashlib.sha256(array.tobytes()).hexdigest()
                suffix = path.stem.rsplit("_", 1)[-1]
                row = {
                    "path": str(path),
                    "mean": float(array.mean()),
                    "std": float(array.std()),
                    "sha256": digest,
                }
                images[label].append(row)
                digests[label][suffix] = digest
                if row["mean"] < 1.0 or row["std"] <= 0.0:
                    issues.append(f"camera_black_or_constant={label}:{path.name}")
            except Exception as exc:
                issues.append(f"camera_unreadable={label}:{path.name}:{type(exc).__name__}")

    for suffix in sorted(set(digests["agent_high"]) & set(digests["wrist_high"])):
        if digests["agent_high"][suffix] == digests["wrist_high"][suffix]:
            issues.append(f"camera_views_identical=high:{suffix}")

    arrays: list[dict[str, Any]] = []
    for directory_name in ("depths", "depths_wrist", "world", "world_wrist", "world_hi", "world_wrist_hi"):
        directory = output_dir / directory_name
        paths = sampled(sorted(directory.glob("*.npy")), limit=4) if directory.is_dir() else []
        for path in paths:
            try:
                value = np.load(path, mmap_mode="r")
                finite = bool(np.isfinite(value).all())
                arrays.append({"path": str(path), "finite": finite})
                if not finite:
                    issues.append(f"camera_array_nonfinite={directory_name}:{path.name}")
            except Exception as exc:
                issues.append(f"camera_array_unreadable={directory_name}:{path.name}:{type(exc).__name__}")

    expected = {"agent_low", "wrist_low", "agent_high", "wrist_high"}
    present = {name for name, rows in images.items() if rows}
    return {
        "checked": bool(present),
        "complete_views": present == expected,
        "images": images,
        "arrays": arrays,
        "issues": issues,
        "healthy": present == expected and not issues,
    }


def _planner_stream_health(output_dir: Path) -> dict[str, Any]:
    """Classify planner-budget exhaustion separately from transport failure."""
    paths = sorted(output_dir.glob("*.stream.jsonl"))
    transport_markers = (
        "stream disconnected",
        "response_stream_disconnected",
        "error sending request",
        "reconnecting",
    )
    transport_errors = 0
    fatal_errors = 0
    budget_exhausted = False
    malformed_rows = 0
    for path in paths:
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                malformed_rows += 1
                continue
            if row.get("type") == "timeout":
                budget_exhausted = True
            method = str(row.get("method", "")).lower()
            if method not in {"error", "fatal"}:
                continue
            rendered = json.dumps(row.get("payload"), ensure_ascii=False).lower()
            if any(marker in rendered for marker in transport_markers):
                transport_errors += 1
            else:
                fatal_errors += 1
    return {
        "paths": [str(path) for path in paths],
        "transport_error_count": transport_errors,
        "fatal_error_count": fatal_errors,
        "budget_exhausted": budget_exhausted,
        "malformed_row_count": malformed_rows,
    }


def _summarize(
    *,
    episode: dict[str, Any],
    common: dict[str, Any],
    output_dir: Path,
    commit: str,
    manifest_digest: str,
    attempt_id: str,
    exit_code: int,
    timed_out: bool,
    elapsed_s: float,
) -> dict[str, Any]:
    states_path = output_dir / "states.json"
    try:
        states = json.loads(states_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        states = []
    if not isinstance(states, list):
        states = []
    scored = [row for row in states if isinstance(row, dict)]
    success = any(bool(row.get("libero_terminated")) for row in scored)
    truncated = any(bool(row.get("episode_truncated")) for row in scored)
    physical_actions = [
        row.get("command", {}).get("action")
        for row in scored
        if isinstance(row.get("command"), dict)
        and row.get("command", {}).get("action")
    ]
    transcripts = sorted(output_dir.glob("transcript_*.json"))
    transcript = _load_last_json_object(transcripts[-1]) if transcripts else {}
    stats = transcript.get("stats") if isinstance(transcript.get("stats"), dict) else {}
    stream_health = _planner_stream_health(output_dir)
    camera_health = _camera_artifact_health(output_dir)
    videos = {
        name: _video_snapshot(output_dir / filename)
        for name, filename in (
            ("agentview", "episode.mp4"),
            ("wrist", "episode_wrist.mp4"),
            ("multiview", "episode_multiview.mp4"),
        )
    }
    invalid: list[str] = []
    if exit_code != 0:
        invalid.append(f"runner_exit_code={exit_code}")
    if timed_out:
        invalid.append("orchestrator_timeout")
    if not scored:
        invalid.append("state_trace_missing")
    if not transcript:
        invalid.append("transcript_missing")
    if stats.get("model") != common["model"]:
        invalid.append(f"model_mismatch={stats.get('model')!r}")
    if stats.get("reasoning_effort") != common["reasoning_effort"]:
        invalid.append(
            f"reasoning_effort_mismatch={stats.get('reasoning_effort')!r}"
        )
    if not transcript.get("finish"):
        if stream_health["transport_error_count"]:
            invalid.append("planner_transport_error")
        elif stream_health["fatal_error_count"]:
            invalid.append("planner_fatal_error")
        elif not stream_health["budget_exhausted"]:
            invalid.append("planner_finish_missing")
    for name, snapshot in videos.items():
        if not snapshot["exists"] or snapshot["size_bytes"] <= 0:
            invalid.append(f"{name}_video_missing")
    if common.get("require_camera_health", False):
        if not camera_health["checked"] or not camera_health["complete_views"]:
            invalid.append("camera_artifacts_missing")
        invalid.extend(camera_health["issues"])
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": common["protocol_id"],
        "attempt_id": attempt_id,
        "episode_id": episode["id"],
        "pair_id": episode.get("pair_id"),
        "phase": episode.get("phase"),
        "variant": episode.get("variant"),
        "suite": episode["suite"],
        "task": int(episode["task"]),
        "seed": int(episode["seed"]),
        "commit": commit,
        "manifest_sha256": manifest_digest,
        "runner_sha256": _sha256_file(Path(__file__).resolve()),
        "tool_manifest_sha256": episode.get("tool_manifest_sha256"),
        "resource_snapshot": episode.get("resource_snapshot", {}),
        "planner": {
            "backend": stats.get("backend"),
            "model": stats.get("model"),
            "reasoning_effort": stats.get("reasoning_effort"),
            "provider": stats.get("provider"),
            "finish": transcript.get("finish"),
            "stream_health": stream_health,
            "usage": {
                key: stats.get(key)
                for key in (
                    "total_input_tokens",
                    "total_cached_input_tokens",
                    "total_output_tokens",
                    "total_reasoning_output_tokens",
                    "tool_calls",
                )
            },
        },
        "success": success,
        "valid": not invalid,
        "invalid_reasons": invalid,
        "episode_truncated": truncated,
        "physical_actions": physical_actions,
        "camera_health": camera_health,
        "videos": videos,
        "states_path": str(states_path),
        "runner_exit_code": exit_code,
        "timed_out": timed_out,
        "elapsed_s": round(elapsed_s, 3),
        "completed_at": _utc_now(),
    }


def _episode_command(
    episode: dict[str, Any], common: dict[str, Any], python: str, output_dir: Path
) -> list[str]:
    command = [
        python,
        "-m",
        "rpent.cli.main",
        "--env",
        "libero",
        "--planner",
        "codex",
        "--model",
        str(common["model"]),
        "--suite",
        str(episode["suite"]),
        "--task",
        str(int(episode["task"])),
        "--seed",
        str(int(episode["seed"])),
        "--max-turns",
        str(int(common.get("max_turns", 100))),
        "--max-episode-steps",
        str(int(common.get("max_episode_steps", 10000))),
        "--planner-timeout-s",
        str(int(common.get("planner_timeout_s", 1800))),
        "--output-dir",
        str(output_dir),
    ]
    if common.get("disable_sam3", False):
        command.append("--disable-sam3")
    elif common.get("sam3_endpoint"):
        command.extend(["--sam3-endpoint", str(common["sam3_endpoint"])])
    if common.get("vla_endpoint"):
        command.extend(["--vla-endpoint", str(common["vla_endpoint"])])
    if common.get("tool_profile"):
        command.extend(["--tool-profile", str(common["tool_profile"])])
    if episode.get("cuda_device") is not None:
        command.extend(["--cuda-device", str(int(episode["cuda_device"]))])
    if episode.get("vla_endpoint") and not common.get("vla_endpoint"):
        command.extend(["--vla-endpoint", str(episode["vla_endpoint"])])
    extra = episode.get("extra_args", [])
    if not isinstance(extra, list) or not all(isinstance(item, str) for item in extra):
        raise ValueError("episode extra_args must be an array of strings")
    command.extend(extra)
    return command


def _frozen_runtime_environment(common: dict[str, Any]) -> dict[str, str]:
    """Return non-secret provider/tool routing frozen by the manifest."""

    result: dict[str, str] = {}
    if common.get("base_url"):
        result["CODEX_BASE_URL"] = str(common["base_url"])
    if common.get("use_local_adapter_key_placeholder"):
        result["CODEX_API_KEY"] = "rpent-local-adapter"
    for manifest_key, environment_key in (
        ("contact_graspnet_endpoint", "CONTACT_GRASPNET_URL"),
        ("graspgen_endpoint", "GRASPGEN_URL"),
    ):
        if common.get(manifest_key):
            result[environment_key] = str(common[manifest_key])
    return result


def freeze_manifest(path: Path) -> int:
    manifest = _load_json(path)
    _validate_manifest(manifest)
    digest = _canonical_digest(manifest)
    digest_path = path.with_suffix(path.suffix + ".sha256")
    digest_path.write_text(digest + "\n", encoding="ascii")
    print(json.dumps({"manifest": str(path), "sha256": digest}, indent=2))
    return 0


def run_episode(path: Path, episode_id: str, *, dry_run: bool = False) -> int:
    manifest = _load_json(path)
    _validate_manifest(manifest)
    digest = _canonical_digest(manifest)
    digest_path = path.with_suffix(path.suffix + ".sha256")
    if not digest_path.is_file() or digest_path.read_text().strip() != digest:
        raise RuntimeError("manifest is not frozen or changed after freeze")
    episodes = {row["id"]: row for row in manifest["episodes"]}
    if episode_id not in episodes:
        raise KeyError(f"episode id not found: {episode_id}")
    episode = episodes[episode_id]
    common = {**manifest["common"], "protocol_id": manifest["protocol_id"]}
    repo = Path(episode["repo"]).expanduser().resolve()
    commit = _git(repo, "rev-parse", "HEAD")
    expected_commit = str(episode["commit"])
    if commit != expected_commit:
        raise RuntimeError(f"commit mismatch: expected {expected_commit}, got {commit}")
    if _git(repo, "status", "--porcelain"):
        raise RuntimeError(f"repository is dirty: {repo}")
    _verify_file_snapshot(episode.get("resource_snapshot", {}))
    output_dir = Path(episode["output_dir"]).expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to reuse output directory: {output_dir}")
    # Preserve a virtualenv's ``bin/python`` symlink. Resolving it points at
    # the base interpreter and drops the venv's site-packages at launch.
    python = str(Path(manifest["python"]).expanduser().absolute())
    command = _episode_command(episode, common, python, output_dir)
    if dry_run:
        print(json.dumps({"cwd": str(repo), "command": command}, indent=2))
        return 0

    attempt_id = uuid.uuid4().hex
    ledger = _ledger_path(path, manifest)
    used = _append_locked(
        ledger,
        {
            "event": "reserved",
            "protocol_id": manifest["protocol_id"],
            "timestamp": _utc_now(),
            "attempt_id": attempt_id,
            "episode_id": episode_id,
            "pair_id": episode.get("pair_id"),
            "variant": episode.get("variant"),
            "manifest_sha256": digest,
            "budget_cap": int(manifest["budget_cap"]),
        },
        reserve=True,
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo) + os.pathsep + env.get("PYTHONPATH", "")
    env["CODEX_MODEL"] = str(common["model"])
    env["CODEX_REASONING_EFFORT"] = str(common["reasoning_effort"])
    if common.get("hf_hub_offline", False):
        env["HF_HUB_OFFLINE"] = "1"
    env.update(_frozen_runtime_environment(common))
    runner_log = output_dir / "runner.log"
    timeout_s = int(common.get("episode_timeout_s", 7200))
    started = time.time()
    exit_code = 1
    timed_out = False
    with runner_log.open("w", encoding="utf-8") as log:
        log.write(json.dumps({"command": command, "budget_used": used}) + "\n")
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
        commit=commit,
        manifest_digest=digest,
        attempt_id=attempt_id,
        exit_code=exit_code,
        timed_out=timed_out,
        elapsed_s=time.time() - started,
    )
    (output_dir / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _append_locked(
        ledger,
        {
            "event": "completed",
            "timestamp": _utc_now(),
            "attempt_id": attempt_id,
            "episode_id": episode_id,
            "valid": result["valid"],
            "success": result["success"],
            "result": str(output_dir / "result.json"),
        },
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["valid"] else 2


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    freeze = sub.add_parser("freeze")
    freeze.add_argument("--manifest", type=Path, required=True)
    run = sub.add_parser("run")
    run.add_argument("--manifest", type=Path, required=True)
    run.add_argument("--episode-id", required=True)
    run.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.command == "freeze":
        return freeze_manifest(args.manifest.resolve())
    return run_episode(
        args.manifest.resolve(), args.episode_id, dry_run=args.dry_run
    )


if __name__ == "__main__":
    raise SystemExit(main())
