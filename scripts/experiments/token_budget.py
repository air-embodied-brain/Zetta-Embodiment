#!/usr/bin/env python3
"""Shared API-token-equivalent budget accounting for memory experiments."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:  # pragma: no cover - production budget gates run on Linux.
    import fcntl
except ImportError:  # pragma: no cover - Windows unit-test compatibility.
    fcntl = None  # type: ignore[assignment]

from rpent.memory.task_memory import canonical_json, canonical_sha256

SCHEMA_VERSION = 1


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def _usage(stats: Any) -> dict[str, int]:
    if not isinstance(stats, dict):
        stats = {}
    input_tokens = int(stats.get("total_input_tokens") or 0)
    cached_tokens = int(stats.get("total_cached_input_tokens") or 0)
    output_tokens = int(stats.get("total_output_tokens") or 0)
    reasoning_tokens = int(stats.get("total_reasoning_output_tokens") or 0)
    return {
        "input_tokens": input_tokens,
        "cached_input_tokens": cached_tokens,
        "output_tokens": output_tokens,
        "reasoning_output_tokens": reasoning_tokens,
        # Cached input is already a subset of input, and reasoning is reported
        # as a detail of output by the Responses API.  Neither is added twice.
        "api_tokens": input_tokens + output_tokens,
    }


def _add_usage(left: dict[str, int], right: dict[str, int]) -> dict[str, int]:
    return {key: int(left.get(key, 0)) + int(right.get(key, 0)) for key in _usage({})}


def _stream_usage(path: Path) -> dict[str, int]:
    """Recover the last cumulative usage from an interrupted Codex stream."""
    best = _usage({})
    best_total = 0
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return best
    for line in lines:
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        payload = row.get("payload")
        usage = payload.get("token_usage") if isinstance(payload, dict) else None
        total = usage.get("total") if isinstance(usage, dict) else None
        if not isinstance(total, dict):
            continue
        candidate = _usage(
            {
                "total_input_tokens": total.get("input_tokens"),
                "total_cached_input_tokens": total.get("cached_input_tokens"),
                "total_output_tokens": total.get("output_tokens"),
                "total_reasoning_output_tokens": total.get(
                    "reasoning_output_tokens"
                ),
            }
        )
        candidate_total = int(candidate["api_tokens"])
        if candidate_total >= best_total:
            best = candidate
            best_total = candidate_total
    return best


def _planner_usage(episode_dir: Path) -> dict[str, int]:
    total = _usage({})
    found_transcript_usage = False
    for transcript in sorted(episode_dir.glob("transcript_*.json")):
        try:
            usage = _usage(_load_json(transcript).get("stats"))
            total = _add_usage(total, usage)
            found_transcript_usage |= usage["api_tokens"] > 0
        except (OSError, ValueError, json.JSONDecodeError):
            continue
    if not found_transcript_usage:
        for stream in sorted(episode_dir.glob("*.stream.jsonl")):
            total = _add_usage(total, _stream_usage(stream))
    return total


def _updater_usage(update_dir: Path) -> dict[str, int]:
    total = _usage({})
    for attempt_dir in sorted(update_dir.glob("attempt-*")):
        result_path = attempt_dir / "update_result.json"
        usage = _usage({})
        if result_path.is_file():
            try:
                value = _load_json(result_path)
                stats = (
                    value.get("updater", {}).get("usage")
                    if value.get("accepted") and isinstance(value.get("updater"), dict)
                    else value.get("stats")
                )
                usage = _usage(stats)
            except (OSError, ValueError, json.JSONDecodeError):
                pass
        if usage["api_tokens"] == 0:
            for stream in sorted(attempt_dir.glob("*.stream.jsonl")):
                usage = _add_usage(usage, _stream_usage(stream))
        total = _add_usage(total, usage)
    return total


def scan_root(name: str, root: Path) -> list[dict[str, Any]]:
    records = []
    for result_path in sorted(root.glob("round-*/r*/result.json")):
        try:
            result = _load_json(result_path)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        episode_id = str(result.get("episode_id") or result_path.parent.name)
        planner = _planner_usage(result_path.parent)
        updater = _updater_usage(root / "memory-updates" / episode_id)
        combined = _add_usage(planner, updater)
        records.append(
            {
                "arm": name,
                "episode_id": episode_id,
                "valid": bool(result.get("valid")),
                "success": bool(result.get("success")),
                "planner": planner,
                "updater": updater,
                "usage": combined,
                "result_path": str(result_path),
            }
        )
    return records


def _validate_config(config: dict[str, Any]) -> None:
    if config.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"token budget schema_version must be {SCHEMA_VERSION}")
    if float(config.get("cap_equivalents", 0)) <= 0:
        raise ValueError("cap_equivalents must be positive")
    if int(config.get("reference_tokens", 0)) <= 0:
        raise ValueError("reference_tokens must be positive")
    roots = config.get("experiment_roots")
    if not isinstance(roots, list) or not roots:
        raise ValueError("experiment_roots must be non-empty")
    if any(not isinstance(row, dict) or set(row) != {"name", "path"} for row in roots):
        raise ValueError("each experiment root requires exactly name and path")
    if float(config.get("lease_reserve_equivalents", 0)) <= 0:
        raise ValueError("lease_reserve_equivalents must be positive")


def freeze_config(path: Path, config: dict[str, Any]) -> None:
    _validate_config(config)
    _atomic_json(path, config)
    path.with_suffix(path.suffix + ".sha256").write_text(
        canonical_sha256(config) + "\n", encoding="ascii"
    )


def load_frozen_config(path: Path) -> dict[str, Any]:
    config = _load_json(path)
    _validate_config(config)
    digest_path = path.with_suffix(path.suffix + ".sha256")
    if not digest_path.is_file():
        raise RuntimeError("token budget config is not frozen")
    if digest_path.read_text(encoding="ascii").strip() != canonical_sha256(config):
        raise RuntimeError("token budget config changed after freeze")
    return config


def measure(path: Path) -> dict[str, Any]:
    config = load_frozen_config(path)
    records = []
    for row in config["experiment_roots"]:
        records.extend(scan_root(str(row["name"]), Path(row["path"])))
    reference = int(config["reference_tokens"])
    by_arm: dict[str, dict[str, Any]] = {}
    for record in records:
        arm = by_arm.setdefault(
            record["arm"],
            {
                "episodes_with_results": 0,
                "valid_results": 0,
                "invalid_results": 0,
                "successes": 0,
                "api_tokens": 0,
            },
        )
        arm["episodes_with_results"] += 1
        arm["valid_results"] += int(record["valid"])
        arm["invalid_results"] += int(not record["valid"])
        arm["successes"] += int(record["success"])
        arm["api_tokens"] += int(record["usage"]["api_tokens"])
    for value in by_arm.values():
        value["token_equivalents"] = value["api_tokens"] / reference
    total_tokens = sum(int(row["usage"]["api_tokens"]) for row in records)
    valid_tokens = sum(
        int(row["usage"]["api_tokens"]) for row in records if row["valid"]
    )
    invalid_tokens = total_tokens - valid_tokens
    return {
        "generated_at": _utc_now(),
        "cap_equivalents": float(config["cap_equivalents"]),
        "reference_tokens": reference,
        "reference_definition": config["reference_definition"],
        "api_tokens": total_tokens,
        "token_equivalents": total_tokens / reference,
        "valid_token_equivalents": valid_tokens / reference,
        "invalid_token_equivalents": invalid_tokens / reference,
        "remaining_equivalents": float(config["cap_equivalents"])
        - total_tokens / reference,
        "by_arm": by_arm,
        "records": records,
    }


def _lease_paths(config_path: Path) -> tuple[Path, Path]:
    return config_path.parent / "token-leases.jsonl", config_path.parent / "token-budget.lock"


@contextmanager
def _locked(config_path: Path) -> Iterator[None]:
    _, lock_path = _lease_paths(config_path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as lock:
        if fcntl is not None:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        yield
        if fcntl is not None:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def _active_leases(config_path: Path) -> dict[str, dict[str, Any]]:
    lease_path, _ = _lease_paths(config_path)
    active: dict[str, dict[str, Any]] = {}
    if not lease_path.is_file():
        return active
    for line in lease_path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        lease_id = str(row.get("lease_id", ""))
        if row.get("event") == "acquired":
            active[lease_id] = row
        elif row.get("event") == "released":
            active.pop(lease_id, None)
    return active


def _append_lease(config_path: Path, row: dict[str, Any]) -> None:
    lease_path, _ = _lease_paths(config_path)
    lease_path.parent.mkdir(parents=True, exist_ok=True)
    with lease_path.open("a", encoding="utf-8") as handle:
        handle.write(canonical_json(row) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def acquire(config_path: Path, *, lease_id: str, arm: str) -> dict[str, Any] | None:
    config_path = config_path.expanduser().resolve()
    with _locked(config_path):
        config = load_frozen_config(config_path)
        active = _active_leases(config_path)
        if lease_id in active:
            return active[lease_id]
        report = measure(config_path)
        reserve = float(config["lease_reserve_equivalents"])
        held = sum(float(row["reserve_equivalents"]) for row in active.values())
        if report["token_equivalents"] + held + reserve > float(
            config["cap_equivalents"]
        ):
            return None
        row = {
            "event": "acquired",
            "timestamp": _utc_now(),
            "lease_id": lease_id,
            "arm": arm,
            "reserve_equivalents": reserve,
            "observed_equivalents_before": report["token_equivalents"],
        }
        _append_lease(config_path, row)
        return row


def release(config_path: Path, *, lease_id: str, outcome: str) -> None:
    config_path = config_path.expanduser().resolve()
    with _locked(config_path):
        active = _active_leases(config_path)
        if lease_id not in active:
            return
        _append_lease(
            config_path,
            {
                "event": "released",
                "timestamp": _utc_now(),
                "lease_id": lease_id,
                "outcome": outcome,
            },
        )


def build_calibrated_config(args: argparse.Namespace) -> dict[str, Any]:
    roots = []
    for encoded in args.root:
        name, separator, path = encoded.partition("=")
        if not separator or not name or not path:
            raise ValueError("--root values must use name=/absolute/path")
        roots.append({"name": name, "path": str(Path(path).expanduser().resolve())})
    calibration_records = scan_root(args.calibration_name, args.calibration_root)
    candidates = [
        int(row["usage"]["api_tokens"])
        for row in calibration_records
        if row["valid"] and int(row["usage"]["api_tokens"]) > 0
    ]
    if len(candidates) < 8:
        raise RuntimeError("at least eight valid calibration episodes are required")
    reference = int(statistics.median(candidates))
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": _utc_now(),
        "cap_equivalents": float(args.cap_equivalents),
        "reference_tokens": reference,
        "reference_definition": {
            "metric": "planner_input_plus_output_tokens_plus_updater_input_plus_output_tokens",
            "calibration_root": str(args.calibration_root.resolve()),
            "calibration_arm": args.calibration_name,
            "statistic": "median_over_valid_completed_results",
            "candidate_count": len(candidates),
            "candidate_min": min(candidates),
            "candidate_max": max(candidates),
        },
        "lease_reserve_equivalents": float(args.lease_reserve_equivalents),
        "experiment_roots": roots,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    calibrate = sub.add_parser("calibrate")
    calibrate.add_argument("--config", type=Path, required=True)
    calibrate.add_argument("--cap-equivalents", type=float, default=500.0)
    calibrate.add_argument("--lease-reserve-equivalents", type=float, default=2.5)
    calibrate.add_argument("--calibration-name", default="batch")
    calibrate.add_argument("--calibration-root", type=Path, required=True)
    calibrate.add_argument("--root", action="append", required=True)
    report = sub.add_parser("report")
    report.add_argument("--config", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.command == "calibrate":
        config = build_calibrated_config(args)
        freeze_config(args.config, config)
        value: dict[str, Any] = config
    else:
        value = measure(args.config)
    print(json.dumps(value, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
