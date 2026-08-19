# Copyright (c) 2026 Zetta Contributors
from __future__ import annotations

import json
from pathlib import Path

import pytest

from zetta.evolution.trajectory import (
    TrajectoryArtifacts,
    TrajectoryFormatError,
    index_episode_trajectory,
    trajectory_agent_summary,
)


def _jsonl(path: Path, rows: list[dict[str, object]]) -> Path:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    return path


def _artifacts(
    tmp_path: Path, *, rows: dict[str, list[dict[str, object]]]
) -> TrajectoryArtifacts:
    video = tmp_path / "front.mp4"
    video.write_bytes(b"video-bytes")
    return TrajectoryArtifacts(
        chunks=_jsonl(tmp_path / "chunks.jsonl", rows.get("chunks", [])),
        actions=_jsonl(tmp_path / "actions.jsonl", rows.get("actions", [])),
        states=_jsonl(tmp_path / "states.jsonl", rows.get("states", [])),
        tools=_jsonl(tmp_path / "tools.jsonl", rows.get("tools", [])),
        videos=(video,),
    )


def _result(*, success: bool = False, status: str = "valid") -> dict[str, object]:
    return {
        "episode_id": "episode-a",
        "logical_id": "wave-000-trial-00",
        "status": status,
        "success": success if status == "valid" else None,
        "max_actions": 20,
        "seed": 73,
        "policy_rng": 991,
    }


def test_valid_failure_builds_content_addressed_index_and_priority_segments(
    tmp_path: Path,
) -> None:
    artifacts = _artifacts(
        tmp_path,
        rows={
            "chunks": [
                {
                    "chunk_index": 0,
                    "environment": {
                        "executed_horizon": 5,
                        "critic_proposals": [{"rule_id": "engagement"}],
                    },
                }
            ],
            "actions": [{"step_index": index, "action": [0.0]} for index in range(5)],
            "states": [
                {"step_index": index, "progress": 0.0, "robot_x": float(index)}
                for index in range(5)
            ],
            "tools": [
                {
                    "step_index": 2,
                    "tool": "grasp_planner",
                    "ok": False,
                    "error": "bad candidate",
                }
            ],
        },
    )
    analysis = index_episode_trajectory(result=_result(), artifacts=artifacts)

    assert analysis.index is not None
    assert analysis.index.action_count == 5
    assert set(analysis.index.artifact_paths) == set(analysis.index.artifact_sha256)
    assert [segment.failure_class for segment in analysis.segments] == [
        "critic_reject",
        "tool_error",
    ]
    assert analysis.segments[0].earliest_divergence_step == 0
    assert analysis.segments[1].earliest_divergence_step == 2
    assert all(segment.embedding for segment in analysis.segments)
    assert all("@sha256:" in ref for ref in analysis.segments[0].artifact_paths)


def test_explicit_critic_reject_has_priority_over_earlier_tool_error(
    tmp_path: Path,
) -> None:
    artifacts = _artifacts(
        tmp_path,
        rows={
            "chunks": [{"executed_horizon": 10}],
            "actions": [{"step": index} for index in range(10)],
            "states": [],
            "tools": [
                {"step": 1, "tool_name": "reach", "status": "failed"},
                {"step": 7, "type": "critic_reject", "tool": "critic"},
            ],
        },
    )
    analysis = index_episode_trajectory(result=_result(), artifacts=artifacts)
    assert [segment.failure_class for segment in analysis.segments[:2]] == [
        "critic_reject",
        "tool_error",
    ]
    assert analysis.segments[0].earliest_divergence_step == 7
    assert analysis.segments[1].earliest_divergence_step == 1


def test_no_progress_window_uses_window_start_not_last_step(tmp_path: Path) -> None:
    artifacts = _artifacts(
        tmp_path,
        rows={
            "chunks": [{"executed_horizon": 12}],
            "actions": [{"step": index} for index in range(12)],
            "states": [
                {"step": index, "task_progress": 0.25 if index >= 3 else index / 10}
                for index in range(12)
            ],
            "tools": [],
        },
    )
    analysis = index_episode_trajectory(
        result=_result(), artifacts=artifacts, no_progress_window=5
    )
    stagnant = next(
        segment
        for segment in analysis.segments
        if segment.failure_class == "no_progress"
    )
    assert stagnant.earliest_divergence_step == 3
    assert stagnant.earliest_divergence_step != 11


def test_startup_plateau_without_progress_or_contact_is_not_no_progress(
    tmp_path: Path,
) -> None:
    artifacts = _artifacts(
        tmp_path,
        rows={
            "chunks": [{"executed_horizon": 8}],
            "actions": [{"step": index} for index in range(8)],
            "states": [
                {
                    "step": index,
                    "privileged.dishwasher.rack.residual_to_success": 0.42,
                    "privileged.dishwasher.rack.target_contact": False,
                }
                for index in range(8)
            ],
            "tools": [],
        },
    )
    analysis = index_episode_trajectory(result=_result(), artifacts=artifacts)
    assert [segment.failure_class for segment in analysis.segments] == [
        "horizon_incomplete"
    ]


def test_early_termination_precedes_horizon_fallback(tmp_path: Path) -> None:
    artifacts = _artifacts(
        tmp_path,
        rows={
            "chunks": [{"executed_horizon": 4}],
            "actions": [{"step": index} for index in range(4)],
            "states": [],
            "tools": [{"step": 3, "type": "role1_terminated", "terminate": True}],
        },
    )
    analysis = index_episode_trajectory(result=_result(), artifacts=artifacts)
    assert [segment.failure_class for segment in analysis.segments] == [
        "early_termination",
    ]
    assert analysis.segments[0].earliest_divergence_step == 3


def test_full_horizon_without_signal_does_not_blame_final_action(
    tmp_path: Path,
) -> None:
    artifacts = _artifacts(
        tmp_path,
        rows={
            "chunks": [{"executed_horizon": 20}],
            "actions": [{"step": index} for index in range(20)],
            "states": [],
            "tools": [],
        },
    )
    analysis = index_episode_trajectory(result=_result(), artifacts=artifacts)
    assert len(analysis.segments) == 1
    assert analysis.segments[0].failure_class == "horizon_incomplete"
    assert analysis.segments[0].earliest_divergence_step is None
    assert analysis.segments[0].start_step == 0
    assert analysis.segments[0].end_step == 20


def test_privileged_residual_stall_and_safety_event_are_structured(
    tmp_path: Path,
) -> None:
    artifacts = _artifacts(
        tmp_path,
        rows={
            "chunks": [{"executed_horizon": 8}],
            "actions": [{"step": index} for index in range(8)],
            "states": [
                {
                    "step": index,
                    "privileged.dishwasher.rack.residual_to_success": 0.42,
                    "privileged.dishwasher.rack.target_contact": True,
                }
                for index in range(8)
            ],
            "tools": [
                {"step": 6, "event": "out_of_bounds", "out_of_bounds": True}
            ],
        },
    )
    analysis = index_episode_trajectory(result=_result(), artifacts=artifacts)
    by_class = {segment.failure_class: segment for segment in analysis.segments}
    assert by_class["no_progress"].earliest_divergence_step == 0
    assert by_class["safety_event"].earliest_divergence_step == 6
    assert "horizon_incomplete" not in by_class


def test_collision_only_telemetry_does_not_create_failure_segment(
    tmp_path: Path,
) -> None:
    artifacts = _artifacts(
        tmp_path,
        rows={
            "chunks": [{"executed_horizon": 2}],
            "actions": [{"step": 0}, {"step": 1}],
            "states": [],
            "tools": [{"step": 1, "event": "collision", "collision": True}],
        },
    )
    analysis = index_episode_trajectory(result=_result(), artifacts=artifacts)
    assert Path(artifacts.tools).is_file()
    assert all(segment.failure_class != "safety_event" for segment in analysis.segments)


def test_success_has_index_and_no_failure_segments(tmp_path: Path) -> None:
    artifacts = _artifacts(
        tmp_path,
        rows={
            "chunks": [{"executed_horizon": 2}],
            "actions": [{"step": 0}, {"step": 1}],
            "states": [],
            "tools": [],
        },
    )
    analysis = index_episode_trajectory(
        result=_result(success=True), artifacts=artifacts
    )
    assert analysis.index is not None and analysis.index.success
    assert analysis.segments == ()


def test_infrastructure_invalid_is_excluded_without_reading_artifacts(
    tmp_path: Path,
) -> None:
    missing = tmp_path / "missing.jsonl"
    artifacts = TrajectoryArtifacts(missing, missing, missing, missing)
    analysis = index_episode_trajectory(
        result=_result(status="infra_invalid"), artifacts=artifacts
    )
    assert analysis.index is None
    assert analysis.segments == ()


@pytest.mark.parametrize(
    "payload",
    [b'{"step": 0}', b'{"step":\n', b"[]\n", b"\xff\n", b"\n"],
)
def test_truncated_or_damaged_jsonl_fails_closed(
    tmp_path: Path, payload: bytes
) -> None:
    artifacts = _artifacts(
        tmp_path,
        rows={"chunks": [], "actions": [], "states": [], "tools": []},
    )
    Path(artifacts.actions).write_bytes(payload)
    with pytest.raises(TrajectoryFormatError):
        index_episode_trajectory(result=_result(), artifacts=artifacts)


def test_same_input_is_idempotent(tmp_path: Path) -> None:
    artifacts = _artifacts(
        tmp_path,
        rows={
            "chunks": [{"executed_horizon": 3}],
            "actions": [{"step": index} for index in range(3)],
            "states": [{"step": 0, "progress": 0.0}],
            "tools": [],
        },
    )
    first = index_episode_trajectory(result=_result(), artifacts=artifacts)
    second = index_episode_trajectory(result=_result(), artifacts=artifacts)
    assert first == second
    assert first.index is not None and second.index is not None
    assert first.index.sha256 == second.index.sha256


def test_agent_summary_does_not_expose_seed_rng_paths_or_identifiers(
    tmp_path: Path,
) -> None:
    hidden_dir = tmp_path / "seed-73-policy-rng-991"
    hidden_dir.mkdir()
    artifacts = _artifacts(
        hidden_dir,
        rows={
            "chunks": [{"executed_horizon": 1}],
            "actions": [{"step": 0}],
            "states": [],
            "tools": [],
        },
    )
    summary = trajectory_agent_summary(
        index_episode_trajectory(result=_result(), artifacts=artifacts)
    )
    rendered = json.dumps(summary, sort_keys=True).lower()
    assert "seed" not in rendered
    assert "rng" not in rendered
    assert "episode-a" not in rendered
    assert "wave-000" not in rendered
    assert str(hidden_dir).lower() not in rendered


def test_result_authority_conflict_fails_closed(tmp_path: Path) -> None:
    artifacts = _artifacts(
        tmp_path,
        rows={"chunks": [], "actions": [], "states": [], "tools": []},
    )
    result = _result(success=False)
    result["authoritative_success"] = True
    with pytest.raises(TrajectoryFormatError, match="conflicts"):
        index_episode_trajectory(result=result, artifacts=artifacts)
