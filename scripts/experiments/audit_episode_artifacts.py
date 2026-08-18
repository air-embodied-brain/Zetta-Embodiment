#!/usr/bin/env python3
"""Audit episode artifacts without mutating historical rollout directories."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


KEY_FILE_GROUPS = {
    "planner_output": "codex_*.txt",
    "raw_stream": "codex_*.txt.stream.jsonl",
    "last_planner_message": "codex_*.txt.last",
    "transcript": "transcript_*.json",
    "states": "states.json",
    "agentview_video": "episode.mp4",
    "wrist_video": "episode_wrist.mp4",
    "multiview_video": "episode_multiview.mp4",
    "run_log": "run.log",
    "runner_log": "runner.log",
    "environment_log": "env_server.log",
}


def audit_arm(name: str, root: Path) -> dict[str, Any]:
    episodes: list[dict[str, Any]] = []
    for result_path in sorted(root.rglob("result.json")):
        episode = _audit_episode(name, root, result_path)
        episodes.append(episode)

    valid = [episode for episode in episodes if episode["valid"]]
    invalid = [episode for episode in episodes if not episode["valid"]]
    missing_valid = Counter(
        key for episode in valid for key in episode["missing_or_empty_key_files"]
    )
    missing_invalid = Counter(
        key for episode in invalid for key in episode["missing_or_empty_key_files"]
    )
    stream_valid = [episode for episode in valid if episode["stream_audit"]]
    visible_reasoning_episodes = sum(
        bool(episode["stream_audit"].get("visible_reasoning_chars"))
        for episode in stream_valid
    )
    return {
        "name": name,
        "root": str(root),
        "result_files": len(episodes),
        "valid": len(valid),
        "invalid": len(invalid),
        "successful_valid": sum(bool(episode["success"]) for episode in valid),
        "valid_with_complete_key_files": sum(
            not episode["missing_or_empty_key_files"] for episode in valid
        ),
        "valid_missing_key_file_counts": dict(sorted(missing_valid.items())),
        "invalid_missing_key_file_counts": dict(sorted(missing_invalid.items())),
        "valid_with_raw_stream": len(stream_valid),
        "valid_with_terminal_turn_event": sum(
            bool(episode["stream_audit"].get("terminal_event_present"))
            for episode in stream_valid
        ),
        "valid_with_reasoning_events": sum(
            bool(episode["stream_audit"].get("reasoning_events"))
            for episode in stream_valid
        ),
        "valid_with_visible_reasoning_text": visible_reasoning_episodes,
        "valid_reasoning_events_total": sum(
            int(episode["stream_audit"].get("reasoning_events", 0))
            for episode in stream_valid
        ),
        "valid_visible_reasoning_chars_total": sum(
            int(episode["stream_audit"].get("visible_reasoning_chars", 0))
            for episode in stream_valid
        ),
        "episodes": episodes,
    }


def _audit_episode(name: str, root: Path, result_path: Path) -> dict[str, Any]:
    try:
        result = json.loads(result_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {
            "arm": name,
            "path": str(result_path.parent.relative_to(root)),
            "valid": False,
            "success": False,
            "result_error": f"{type(exc).__name__}: {exc}",
            "missing_or_empty_key_files": list(KEY_FILE_GROUPS),
            "stream_audit": {},
        }
    episode_dir = result_path.parent
    groups: dict[str, list[str]] = {}
    missing: list[str] = []
    for key, pattern in KEY_FILE_GROUPS.items():
        paths = sorted(path for path in episode_dir.glob(pattern) if path.is_file())
        groups[key] = [path.name for path in paths if path.stat().st_size > 0]
        if not groups[key]:
            missing.append(key)

    streams = sorted(episode_dir.glob(KEY_FILE_GROUPS["raw_stream"]))
    stream_audit = _audit_stream(streams[0]) if streams else {}
    return {
        "arm": name,
        "path": str(episode_dir.relative_to(root)),
        "episode_id": result.get("episode_id", episode_dir.name),
        "round": result.get("round"),
        "seed": result.get("seed"),
        "attempt_index": result.get("attempt_index"),
        "valid": bool(result.get("valid")),
        "success": bool(result.get("success")),
        "timed_out": bool(result.get("timed_out")),
        "invalid_reasons": result.get("invalid_reasons", []),
        "key_files": groups,
        "missing_or_empty_key_files": missing,
        "stream_audit": stream_audit,
    }


def _audit_stream(path: Path) -> dict[str, Any]:
    methods: Counter[str] = Counter()
    item_types: Counter[str] = Counter()
    reasoning_events = 0
    visible_reasoning_chars = 0
    malformed_lines = 0
    total_lines = 0
    with path.open("rb") as stream:
        for raw_line in stream:
            total_lines += 1
            try:
                record = json.loads(raw_line.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                malformed_lines += 1
                continue
            if not isinstance(record, dict):
                malformed_lines += 1
                continue
            method = str(record.get("method") or record.get("type") or "")
            methods[method] += 1
            payload = record.get("payload")
            item = payload.get("item") if isinstance(payload, dict) else None
            item_type = str(item.get("type", "")) if isinstance(item, dict) else ""
            if item_type:
                item_types[item_type] += 1
            if item_type == "reasoning" or "reasoning" in method.lower():
                reasoning_events += 1
                visible_reasoning_chars += sum(
                    len(text) for text in _visible_reasoning(record)
                )
    return {
        "path": path.name,
        "bytes": path.stat().st_size,
        "lines": total_lines,
        "malformed_lines": malformed_lines,
        "event_counts": dict(sorted(methods.items())),
        "item_type_counts": dict(sorted(item_types.items())),
        "reasoning_events": reasoning_events,
        "visible_reasoning_chars": visible_reasoning_chars,
        "terminal_event_present": any(
            method in methods for method in ("turn/completed", "timeout", "error", "fatal")
        ),
    }


def _visible_reasoning(record: dict[str, Any]) -> list[str]:
    method = str(record.get("method") or "").lower()
    payload = record.get("payload")
    if not isinstance(payload, dict):
        return []
    output: list[str] = []
    if "reasoning" in method and isinstance(payload.get("delta"), str):
        output.append(payload["delta"])
    item = payload.get("item")
    if isinstance(item, dict) and item.get("type") == "reasoning":
        output.extend(_texts(item.get("summary")))
        output.extend(_texts(item.get("content")))
    return [text for text in output if text]


def _texts(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, list):
        return [text for item in value for text in _texts(item)]
    if isinstance(value, dict):
        return [
            text
            for key in ("text", "summary", "content", "delta")
            if key in value
            for text in _texts(value[key])
        ]
    return []


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--arm",
        action="append",
        required=True,
        metavar="NAME=PATH",
        help="Arm label and artifact root; may be repeated.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    arms = []
    for value in args.arm:
        name, separator, raw_path = value.partition("=")
        if not separator or not name or not raw_path:
            parser.error(f"invalid --arm {value!r}; expected NAME=PATH")
        arms.append(audit_arm(name, Path(raw_path)))

    report = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "definition": {
            "valid_failure": "A completed task rollout with score 0; expected to retain all key artifacts.",
            "infrastructure_invalid": "An attempt rejected by the runner; partial robot artifacts may legitimately be absent.",
            "reasoning_scope": "Only provider-emitted reasoning content/summary/delta is observable.",
        },
        "arms": arms,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(args.output), "arms": [
        {
            "name": arm["name"],
            "valid": arm["valid"],
            "invalid": arm["invalid"],
            "valid_with_complete_key_files": arm["valid_with_complete_key_files"],
            "valid_missing_key_file_counts": arm["valid_missing_key_file_counts"],
            "valid_with_terminal_turn_event": arm["valid_with_terminal_turn_event"],
            "valid_with_visible_reasoning_text": arm["valid_with_visible_reasoning_text"],
        }
        for arm in arms
    ]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
