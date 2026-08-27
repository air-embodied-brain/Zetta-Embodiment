# Copyright (c) 2026 Zetta Contributors
"""Tests for the self-contained Cosmos-Lite replay report."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "deployment"
    / "visualize_cosmos_lite_replay.py"
)


def _request(index: int, *, include_actions: bool = True) -> dict[str, object]:
    request: dict[str, object] = {
        "index": index,
        "latency_ms": 400.0 + index,
        "shape": [2, 2],
        "dtype": "float32",
        "action_sha256": "stable-hash",
        "model_version": "cosmos-lite:test:1234",
        "auxiliary_outputs": {
            "server_timing": {"infer_ms": 390.0 + index},
            "cosmos_lite_identity": {
                "model_family": "cosmos3_edge",
                "strategy": "gen_branch_w8a8",
                "manifest_sha256": "a" * 64,
            },
        },
    }
    if include_actions:
        request["actions"] = [[0.0, 1.0], [0.5, 0.25]]
    return request


def _render(tmp_path: Path, *, include_actions: bool) -> str:
    source = tmp_path / "replay.json"
    output = tmp_path / "replay.html"
    source.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "deterministic": True,
                "max_action_abs_diff": 0.0,
                "input": {"instruction": "move <safely>"},
                "requests": [
                    _request(0, include_actions=include_actions),
                    _request(1, include_actions=include_actions),
                ],
            }
        ),
        encoding="utf-8",
    )
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--input",
            str(source),
            "--output",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert str(output) in completed.stdout
    return output.read_text(encoding="utf-8")


def test_visualizer_generates_self_contained_action_and_latency_charts(
    tmp_path: Path,
) -> None:
    rendered = _render(tmp_path, include_actions=True)
    assert "Cosmos-Lite Model Replay" in rendered
    assert "全量请求延迟" in rendered
    assert "热态请求延迟" in rendered
    assert "2 步动作轨迹" in rendered
    assert "stable-hash" in rendered
    assert "move &lt;safely&gt;" in rendered
    assert rendered.count("<svg") == 4
    assert "https://" not in rendered


def test_visualizer_explains_how_to_upgrade_legacy_report(tmp_path: Path) -> None:
    rendered = _render(tmp_path, include_actions=False)
    assert "动作轨迹未写入旧版 Replay" in rendered
    assert "--include-actions" in rendered
