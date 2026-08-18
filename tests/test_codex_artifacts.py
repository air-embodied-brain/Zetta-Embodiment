from __future__ import annotations

import json

from rpent.planner.codex_artifacts import export_codex_stream_artifacts


def _write_stream(path, records) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for record in records:
            if isinstance(record, str):
                stream.write(record + "\n")
            else:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")


def test_exports_all_provider_emitted_reasoning_and_audits_stream(tmp_path):
    raw = tmp_path / "planner.txt.stream.jsonl"
    output = tmp_path / "planner.txt"
    output.write_text("rendered", encoding="utf-8")
    records = [
        {"method": "turn/started", "payload": {}},
        {
            "method": "item/completed",
            "payload": {
                "item": {
                    "type": "reasoning",
                    "summary": [{"type": "summary_text", "text": "检查场景"}],
                    "content": [{"type": "reasoning_text", "text": "可见内容"}],
                }
            },
        },
        {
            "method": "item/reasoning/summaryTextDelta",
            "payload": {"delta": "增量摘要"},
        },
        {
            "method": "item/agentMessage/delta",
            "payload": {"delta": "准备调用工具"},
        },
        {
            "method": "item/completed",
            "payload": {
                "item": {
                    "type": "mcpToolCall",
                    "tool": "mcp__rpent__view_driver_state",
                }
            },
        },
        {"method": "turn/completed", "payload": {}},
    ]
    _write_stream(raw, records)

    manifest = export_codex_stream_artifacts(raw, output)

    assert manifest["reasoning"]["events_preserved"] == 2
    assert manifest["reasoning"]["visible_text_available"] is True
    assert manifest["reasoning"]["hidden_chain_of_thought_retrievable"] is False
    assert manifest["completeness"]["raw_stream_parse_complete"] is True
    assert manifest["completeness"]["terminal_event_present"] is True
    visible = (tmp_path / "planner.txt.reasoning.md").read_text(encoding="utf-8")
    assert "检查场景" in visible
    assert "可见内容" in visible
    assert "增量摘要" in visible
    assert (
        len(
            (tmp_path / "planner.txt.messages.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        )
        == 1
    )
    assert len((tmp_path / "planner.txt.tools.jsonl").read_text().splitlines()) == 1


def test_empty_reasoning_is_preserved_without_fabricating_text(tmp_path):
    raw = tmp_path / "planner.txt.stream.jsonl"
    output = tmp_path / "planner.txt"
    _write_stream(
        raw,
        [
            {
                "method": "item/completed",
                "payload": {
                    "item": {
                        "type": "reasoning",
                        "summary": [],
                        "content": [],
                    }
                },
            },
            {"type": "timeout", "message": "deadline"},
            "{truncated",
        ],
    )

    manifest = export_codex_stream_artifacts(raw, output)

    assert manifest["reasoning"]["events_preserved"] == 1
    assert manifest["reasoning"]["visible_text_chars"] == 0
    assert manifest["reasoning"]["visible_text_available"] is False
    assert manifest["completeness"]["terminal_event_present"] is True
    assert manifest["completeness"]["raw_stream_parse_complete"] is False
    assert len(manifest["source_of_truth"]["malformed_lines"]) == 1
    visible = (tmp_path / "planner.txt.reasoning.md").read_text(encoding="utf-8")
    assert "No textual reasoning content was emitted" in visible
