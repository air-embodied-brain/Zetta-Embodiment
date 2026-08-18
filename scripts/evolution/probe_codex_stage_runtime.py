#!/usr/bin/env python3
# Copyright (c) 2026 RPent Contributors
"""Probe the real Codex planner runtime without exposing provider secrets."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import uuid
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from rpent.evolution.jsonio import atomic_write_json  # noqa: E402
from rpent.planner.base import build_planner  # noqa: E402
from rpent.tools.toolkit import Toolkit  # noqa: E402


def run_probe(
    *,
    output_root: Path,
    model: str,
    reasoning_effort: str,
    timeout_s: int,
) -> dict[str, object]:
    """Run one tool-free Codex turn and return a secret-free audit summary."""

    output_root.mkdir(parents=True, exist_ok=False)
    nonce = f"codex-stage-probe-{uuid.uuid4().hex}"
    toolkit = Toolkit()
    toolkit.retain_tools(set())
    try:
        planner = build_planner(
            "codex",
            output_dir=output_root / "planner",
            recipe_tag="stage-runtime-probe",
            env_name="robocasa",
            model=model,
            reasoning_effort=reasoning_effort,
            planner_timeout_s=timeout_s,
        )
        result = planner.solve(
            system_prompt=(
                "You are a runtime health probe. Do not use tools. Return the exact "
                "nonce from the user message and no other text."
            ),
            user_message=nonce,
            toolkit=toolkit,
            max_turns=1,
        )
    finally:
        toolkit.close()

    response = "\n".join(
        str(message.get("content", "")) for message in result.messages
    )
    stats = dict(result.stats)
    thread_id = stats.get("thread_id")
    checks = {
        "planner_error_absent": result.error is None,
        "nonce_returned": nonce in response,
        "persistent_thread_id_present": isinstance(thread_id, str)
        and bool(thread_id.strip()),
        "raw_stream_parse_complete": stats.get("raw_stream_parse_complete") is True,
        "terminal_event_present": stats.get("terminal_event_present") is True,
    }
    report: dict[str, object] = {
        "schema_version": 1,
        "model": model,
        "reasoning_effort": reasoning_effort,
        "passed": all(checks.values()),
        "checks": checks,
        "response_sha256": hashlib.sha256(response.encode("utf-8")).hexdigest(),
        "response_chars": len(response),
        "thread_id_sha256": (
            hashlib.sha256(str(thread_id).encode("utf-8")).hexdigest()
            if isinstance(thread_id, str) and thread_id
            else None
        ),
        "provider_mode": stats.get("provider"),
        "elapsed_s": stats.get("elapsed_s"),
        "reasoning_events_preserved": stats.get("reasoning_events_preserved"),
        "reasoning_visible_text_chars": stats.get("reasoning_visible_text_chars"),
        "artifact_manifest_path": stats.get("artifact_manifest_path"),
    }
    atomic_write_json(output_root / "report.json", report, overwrite=False)
    if not report["passed"]:
        error_summary = {
            "error_type": (
                str(result.error).split(":", 1)[0] if result.error else None
            ),
            "failed_checks": sorted(
                name for name, passed in checks.items() if not passed
            ),
        }
        atomic_write_json(
            output_root / "failure_summary.json", error_summary, overwrite=False
        )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--model", default="gpt-5.6-sol")
    parser.add_argument(
        "--reasoning-effort",
        choices=("low", "medium", "high", "xhigh"),
        default="high",
    )
    parser.add_argument("--timeout-s", type=int, default=600)
    args = parser.parse_args()
    report = run_probe(
        output_root=args.output_root,
        model=args.model,
        reasoning_effort=args.reasoning_effort,
        timeout_s=args.timeout_s,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
