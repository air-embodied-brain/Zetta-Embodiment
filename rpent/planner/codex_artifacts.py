"""Crash-tolerant, auditable derivatives of a Codex app-server stream.

The raw ``*.stream.jsonl`` file remains the source of truth.  This module
creates smaller indexes for reasoning, planner-message, and tool events, plus
an integrity manifest.  It never claims access to model-internal reasoning
that the API did not emit.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1


def export_codex_stream_artifacts(
    raw_stream_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Index every API-emitted event and write an integrity manifest.

    The function is intentionally usable after a timeout or process recovery:
    malformed/truncated JSONL lines are counted rather than aborting the audit.
    """
    raw_path = Path(raw_stream_path)
    planner_path = Path(output_path)
    reasoning_path = planner_path.with_suffix(planner_path.suffix + ".reasoning.jsonl")
    messages_path = planner_path.with_suffix(planner_path.suffix + ".messages.jsonl")
    tools_path = planner_path.with_suffix(planner_path.suffix + ".tools.jsonl")
    visible_path = planner_path.with_suffix(planner_path.suffix + ".reasoning.md")
    manifest_path = planner_path.with_suffix(planner_path.suffix + ".artifacts.json")

    methods: Counter[str] = Counter()
    item_types: Counter[str] = Counter()
    reasoning_records: list[dict[str, Any]] = []
    message_records: list[dict[str, Any]] = []
    tool_records: list[dict[str, Any]] = []
    visible_reasoning: list[str] = []
    malformed: list[dict[str, Any]] = []
    parsed_lines = 0
    total_lines = 0
    raw_digest = hashlib.sha256()

    if raw_path.exists():
        with raw_path.open("rb") as stream:
            for line_number, raw_line in enumerate(stream, start=1):
                total_lines += 1
                raw_digest.update(raw_line)
                try:
                    record = json.loads(raw_line.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    malformed.append(
                        {
                            "line": line_number,
                            "bytes": len(raw_line),
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
                    continue
                if not isinstance(record, dict):
                    malformed.append(
                        {
                            "line": line_number,
                            "bytes": len(raw_line),
                            "error": "decoded JSON value is not an object",
                        }
                    )
                    continue

                parsed_lines += 1
                method = str(record.get("method") or record.get("type") or "")
                methods[method] += 1
                item = _event_item(record)
                item_type = str(item.get("type", "")) if item else ""
                if item_type:
                    item_types[item_type] += 1

                if _is_reasoning_event(method, item_type):
                    reasoning_records.append(record)
                    visible_reasoning.extend(_visible_reasoning_fragments(record))
                if _is_message_event(method, item_type):
                    message_records.append(record)
                if _is_tool_event(method, item_type):
                    tool_records.append(record)

    _write_jsonl(reasoning_path, reasoning_records)
    _write_jsonl(messages_path, message_records)
    _write_jsonl(tools_path, tool_records)

    visible_text = "".join(fragment for fragment in visible_reasoning if fragment)
    visible_header = (
        "# API-emitted reasoning\n\n"
        "This file contains only reasoning text or summaries actually emitted "
        "by the provider. Hidden model chain-of-thought is not exposed by the "
        "client API and cannot be reconstructed from token counts.\n\n"
    )
    if visible_text:
        visible_path.write_text(visible_header + visible_text + "\n", encoding="utf-8")
    else:
        visible_path.write_text(
            visible_header + "No textual reasoning content was emitted.\n",
            encoding="utf-8",
        )

    terminal_methods = {
        "turn/completed",
        "error",
        "fatal",
        "timeout",
    }
    terminal_event_present = any(method in methods for method in terminal_methods)
    raw_bytes = raw_path.stat().st_size if raw_path.exists() else 0
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_of_truth": {
            "path": str(raw_path),
            "exists": raw_path.exists(),
            "bytes": raw_bytes,
            "sha256": raw_digest.hexdigest() if raw_path.exists() else None,
            "lines": total_lines,
            "parsed_lines": parsed_lines,
            "malformed_lines": malformed,
        },
        "derived_artifacts": {
            "reasoning_events": str(reasoning_path),
            "reasoning_visible": str(visible_path),
            "planner_messages": str(messages_path),
            "tool_events": str(tools_path),
        },
        "event_counts": dict(sorted(methods.items())),
        "item_type_counts": dict(sorted(item_types.items())),
        "reasoning": {
            "events_preserved": len(reasoning_records),
            "visible_fragments": len(visible_reasoning),
            "visible_text_chars": len(visible_text),
            "visible_text_available": bool(visible_text),
            "hidden_chain_of_thought_retrievable": False,
            "limitation": (
                "Only provider-emitted reasoning content, summaries, and deltas "
                "can be saved; unreturned hidden reasoning is unavailable."
            ),
        },
        "planner_message_events_preserved": len(message_records),
        "tool_events_preserved": len(tool_records),
        "completeness": {
            "raw_stream_present": raw_path.exists() and raw_bytes > 0,
            "raw_stream_parse_complete": raw_path.exists() and not malformed,
            "terminal_event_present": terminal_event_present,
            "reasoning_events_indexed": raw_path.exists() and not malformed,
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    manifest["manifest_path"] = str(manifest_path)
    return manifest


def _event_item(record: dict[str, Any]) -> dict[str, Any]:
    payload = record.get("payload")
    if not isinstance(payload, dict):
        return {}
    item = payload.get("item")
    return item if isinstance(item, dict) else {}


def _is_reasoning_event(method: str, item_type: str) -> bool:
    return item_type == "reasoning" or "reasoning" in method.lower()


def _is_message_event(method: str, item_type: str) -> bool:
    return item_type in {"agentMessage", "userMessage"} or "agentmessage" in method.lower()


def _is_tool_event(method: str, item_type: str) -> bool:
    return item_type in {
        "mcpToolCall",
        "dynamicToolCall",
        "commandExecution",
        "fileChange",
    } or "toolcall" in method.lower()


def _visible_reasoning_fragments(record: dict[str, Any]) -> list[str]:
    """Return provider-visible reasoning text without inferring hidden content."""
    method = str(record.get("method") or "").lower()
    payload = record.get("payload")
    if not isinstance(payload, dict):
        return []
    fragments: list[str] = []
    if "reasoning" in method and isinstance(payload.get("delta"), str):
        fragments.append(payload["delta"])
    item = payload.get("item")
    if isinstance(item, dict) and item.get("type") == "reasoning":
        fragments.extend(_text_fragments(item.get("summary")))
        fragments.extend(_text_fragments(item.get("content")))
    return [text for text in fragments if text]


def _text_fragments(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, list):
        result: list[str] = []
        for item in value:
            result.extend(_text_fragments(item))
        return result
    if isinstance(value, dict):
        result = []
        for key in ("text", "summary", "content", "delta"):
            if key in value:
                result.extend(_text_fragments(value[key]))
        return result
    return []


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
