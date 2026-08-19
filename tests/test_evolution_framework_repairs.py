# Copyright (c) 2026 Zetta Contributors
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import imageio.v2 as iio
import numpy as np
import pytest

from zetta.evolution.jsonio import canonical_sha256
from zetta.evolution.models import (
    CampaignManifest,
    CandidateBundle,
    CausalDiagnosis,
    CriticRule,
    EpisodeRecord,
    FailureCluster,
    FailureSegment,
    RecoveryRule,
    RecoveryStep,
)
from zetta.evolution.shadow_replay import evaluate_shadow_replay
from zetta.evolution.stages import (
    CodexStageAgent,
    _cluster_visual_contract,
    _diagnosis_visual_contract,
    _require_atomic_parent_inheritance,
    _require_group_event_coverage,
    _require_group_visual_coverage,
)
from zetta.evolution.visual_artifacts import (
    build_episode_visual_artifacts,
    write_video_metadata,
)
from zetta.tools.toolkit import Toolkit
from scripts.evolution.audit_robocasa_success_labels import audit


def test_cluster_visual_contract_includes_overviews_and_temporal_windows() -> None:
    failure_ids = [f"artifact-{'a' * 63}{index}" for index in range(2)]
    success_id = f"artifact-{'b' * 64}"
    divergence_id = f"artifact-{'c' * 64}"
    artifacts = [
        {"content_id": content_id, "type": "image", "description": "overview"}
        for content_id in [*failure_ids, success_id]
    ] + [
        {
            "content_id": divergence_id,
            "type": "image",
            "description": "early divergence window",
        }
    ]
    relationships = [
        {
            "episode": f"episode-{index}",
            "outcome": "failure",
            "segments": [],
            "visual_evidence": [
                {
                    "content_id": content_id,
                    "type": "image",
                    "role": "episode_overview",
                },
                {
                    "content_id": divergence_id,
                    "type": "image",
                    "role": "divergence_window",
                },
            ],
        }
        for index, content_id in enumerate(failure_ids)
    ] + [
        {
            "episode": "episode-success",
            "outcome": "success",
            "segments": [],
            "visual_evidence": [
                {
                    "content_id": success_id,
                    "type": "image",
                    "role": "episode_overview",
                }
            ],
        }
    ]

    compact, failures, successes = _cluster_visual_contract(
        {"artifacts": artifacts, "relationships": relationships}
    )

    assert {row["content_id"] for row in compact["artifacts"]} == {
        *failure_ids,
        success_id,
        divergence_id,
    }
    assert failures == set(failure_ids)
    assert successes == {success_id}
    assert compact["selection_contract"] == {
        "visual_role": "episode_overview",
        "failure_overviews_required": 2,
        "success_overview_required": True,
        "temporal_detail_roles": ["event_window", "divergence_window"],
        "event_window_required_per_group_when_available": True,
    }
    assert {
        visual["role"]
        for relationship in compact["relationships"]
        for visual in relationship["visual_evidence"]
    } == {"episode_overview", "divergence_window"}


def test_cluster_visual_evidence_must_cover_each_groups_own_members() -> None:
    groups = [
        {"member_segment_ids": ["segment-a"]},
        {"member_segment_ids": ["segment-b"]},
    ]
    relationships = [
        {
            "outcome": "failure",
            "segments": [{"segment": "segment-a"}],
            "visual_evidence": [
                {"role": "episode_overview", "content_id": "overview-a"}
            ],
        },
        {
            "outcome": "failure",
            "segments": [{"segment": "segment-b"}],
            "visual_evidence": [
                {"role": "episode_overview", "content_id": "overview-b"}
            ],
        },
    ]
    _require_group_visual_coverage(
        groups=groups,
        cited_visual_ids={"overview-a", "overview-b"},
        relationships=relationships,
    )
    with pytest.raises(ValueError, match="group 1 cited no overview"):
        _require_group_visual_coverage(
            groups=groups,
            cited_visual_ids={"overview-a"},
            relationships=relationships,
        )


def test_cluster_and_diagnosis_require_dense_member_event_windows() -> None:
    groups = [{"member_segment_ids": ["segment-a"]}]
    relationships = [
        {
            "episode": "episode-a",
            "outcome": "failure",
            "segments": [{"segment": "segment-a"}],
            "visual_evidence": [
                {"role": "episode_overview", "content_id": "overview-a"},
                {"role": "event_window", "content_id": "event-a-0"},
                {"role": "event_window", "content_id": "event-a-1"},
            ],
        },
        {
            "episode": "episode-success",
            "outcome": "success",
            "segments": [],
            "visual_evidence": [
                {"role": "episode_overview", "content_id": "overview-success"}
            ],
        },
    ]
    _require_group_event_coverage(
        groups=groups,
        cited_visual_ids={"event-a-0"},
        relationships=relationships,
    )
    with pytest.raises(ValueError, match="dense event window"):
        _require_group_event_coverage(
            groups=groups,
            cited_visual_ids={"overview-a"},
            relationships=relationships,
        )

    cluster = FailureCluster(
        cluster_id="cluster-a",
        hard_key=("failure",),
        member_segment_ids=("segment-a",),
        episode_ids=("episode-a",),
        representative_segment_ids=("segment-a",),
        medoid_segment_id="segment-a",
        summary="failure",
        prevalence=1.0,
        mean_severity=1.0,
    )
    contract = _diagnosis_visual_contract(
        cluster=cluster,
        artifact_index={"relationships": relationships},
    )
    assert contract["minimum_distinct_failure_overviews"] == 1
    assert contract["minimum_distinct_event_windows"] == 2
    assert contract["success_comparator_required"] is True


def test_legacy_manifest_and_diagnosis_digests_remain_stable() -> None:
    legacy_manifest = {
        "campaign_id": "legacy",
        "environment": "robocasa",
        "task": "SlideDishwasherRack",
        "generation": 0,
        "code_commit": "a" * 40,
        "prompt_sha256": "b" * 64,
        "model": "gpt-5.6-sol",
        "tool_catalog_sha256": "c" * 64,
        "rollout_seeds": [1],
        "heldout_seeds": [2],
        "policy_rng_by_seed": {"1": 11, "2": 22},
        "parent_bundle_sha256": None,
        "expected_rollouts": 1,
        "expected_heldout": 1,
        "initial_logical_slots": 8,
        "maximum_logical_slots": 50,
        "continuous_logical_slots": 4,
        "maximum_api_concurrency": 20,
        "episode_timeout_s": 2700,
        "no_progress_timeout_s": 180,
        "target_valid_episodes_per_hour": 25.0,
        "max_infrastructure_attempts": 8,
        "reasoning_effort": "high",
        "runtime": {},
        "schema_version": 1,
    }
    manifest = CampaignManifest.from_dict(legacy_manifest)
    assert manifest.baseline_mode == "strict_pure_vla"
    assert manifest.sha256 == canonical_sha256(legacy_manifest)

    legacy_diagnosis = {
        "diagnosis_id": "diagnosis-legacy",
        "cluster_id": "cluster-legacy",
        "outcome": "stall",
        "immediate_trigger": "no progress",
        "root_cause": "candidate cause",
        "contributing_causes": [],
        "competing_hypotheses": ["tool", "policy"],
        "owner_layer": "vla",
        "affected_component": "policy",
        "earliest_divergence": "step 4",
        "supporting_evidence_ids": ["evidence"],
        "counterevidence_ids": [],
        "falsifier": "paired run",
        "distinguishing_check": "observe action",
        "required_validation": "same-seed gate",
        "confidence": 0.5,
    }
    diagnosis = CausalDiagnosis(
        **{
            **legacy_diagnosis,
            "contributing_causes": (),
            "competing_hypotheses": tuple(
                legacy_diagnosis["competing_hypotheses"]
            ),
            "supporting_evidence_ids": ("evidence",),
            "counterevidence_ids": (),
        }
    )
    assert diagnosis.sha256 == canonical_sha256(legacy_diagnosis)


def test_success_label_audit_is_read_only_and_marks_downstream_audit_only(
    tmp_path: Path,
) -> None:
    episode = tmp_path / "episode"
    episode.mkdir()
    states = episode / "states.jsonl"
    states.write_text(
        json.dumps(
            {
                "step_index": 8,
                "reward": 1,
                "terminated": False,
                "truncated": False,
                "state": {
                    "privileged.dishwasher.rack.remaining_to_success_m": 0.0
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    record = episode / "episode_record.json"
    record.write_text(
        json.dumps(
            {
                "episode_id": "episode-legacy",
                "logical_id": "logical-legacy",
                "success": False,
                "artifact_index": {"states": str(states)},
            }
        ),
        encoding="utf-8",
    )
    before = record.read_bytes(), states.read_bytes()
    report = audit(tmp_path)
    assert report["affected_count"] == 1
    assert report["affected_episodes"][0]["disposition"] == (
        "invalid_for_scoring_and_learning_audit_only"
    )
    assert "must be recomputed" in report["downstream_disposition"]
    assert before == (record.read_bytes(), states.read_bytes())


def _rule(name: str) -> CriticRule:
    return CriticRule(
        rule_id=f"critic-{name}",
        title=name,
        feature="progress",
        operator="ge",
        threshold=1.0,
        dwell_steps=1,
        cooldown_steps=0,
        activation_conditions=(),
        proposal="recover",
        evidence_ids=("evidence",),
    )


def _recovery(name: str) -> RecoveryRule:
    return RecoveryRule(
        recovery_id=f"recovery-{name}",
        title=name,
        trigger_rule_ids=(f"critic-{name}",),
        precondition="triggered",
        steps=(RecoveryStep(tool="tool", parameters={}, stop_when="done"),),
        safety_constraints=("finite action",),
        stop_condition="done",
        fallback="stop",
        evidence_ids=("evidence",),
    )


def _bundle(name: str, *, parent: str | None, rules: tuple[str, ...]) -> CandidateBundle:
    return CandidateBundle(
        candidate_id=name,
        generation=1 if parent else 0,
        parent_sha256=parent,
        diagnosis_sha256="d" * 64,
        causal_hypothesis="one cause",
        mechanism_change="one rule pair",
        validation_plan="paired gate",
        critic_rules=tuple(_rule(value) for value in rules),
        recovery_rules=tuple(_recovery(value) for value in rules),
    )


def test_candidate_effective_bundle_preserves_parent_and_adds_one_delta() -> None:
    parent = _bundle("parent", parent=None, rules=("parent",))
    candidate = _bundle(
        "candidate", parent=parent.sha256, rules=("parent", "delta")
    )
    _require_atomic_parent_inheritance(candidate, parent)
    missing_parent = _bundle("bad", parent=parent.sha256, rules=("delta",))
    with pytest.raises(ValueError, match="inherit every promoted parent rule"):
        _require_atomic_parent_inheritance(missing_parent, parent)
    too_many = _bundle(
        "bad-many", parent=parent.sha256, rules=("parent", "one", "two")
    )
    with pytest.raises(ValueError, match="exactly one critic"):
        _require_atomic_parent_inheritance(too_many, parent)


def _record(
    episode: str,
    *,
    success: bool,
    parent: str | None,
    divergence: int | None,
) -> EpisodeRecord:
    segment = (
        FailureSegment(
            segment_id=f"segment-{episode}",
            episode_id=episode,
            failure_class="stall",
            stage="execution",
            tool="vla",
            summary="stalled",
            earliest_divergence_step=divergence,
            start_step=0,
            end_step=max(1, divergence or 1),
        )
        if not success
        else None
    )
    return EpisodeRecord(
        episode_id=episode,
        logical_id=f"logical-{episode}",
        generation=0,
        seed=1,
        policy_rng=2,
        bundle_sha256=parent,
        status="valid",
        success=success,
        started_at="2026-01-01T00:00:00+00:00",
        finished_at="2026-01-01T00:00:01+00:00",
        elapsed_s=1.0,
        artifact_index={},
        failure_segment=segment,
        failure_segments=(segment,) if segment else (),
    )


def test_shadow_replay_binds_parent_schema_digest_recall_and_false_positive(
    tmp_path: Path,
) -> None:
    candidate = _bundle("candidate", parent=None, rules=("delta",))
    failed = tmp_path / "failed.jsonl"
    failed.write_text(
        '\n'.join(
            json.dumps({"step_index": step, "state": {"progress": value}})
            for step, value in ((0, 0.0), (2, 1.0), (3, 1.0))
        )
        + "\n",
        encoding="utf-8",
    )
    success = tmp_path / "success.jsonl"
    success.write_text(
        json.dumps({"step_index": 0, "state": {"progress": 0.0}}) + "\n",
        encoding="utf-8",
    )
    report = evaluate_shadow_replay(
        candidate=candidate,
        parent=None,
        target_records=((_record("failed", success=False, parent=None, divergence=3), failed),),
        success_controls=((_record("success", success=True, parent=None, divergence=None), success),),
    )
    assert report["target_recall"] == 1.0
    assert report["success_control_false_positive_rate"] == 0.0
    assert report["outcomes"][0]["first_trigger_step"] == 2
    assert report["outcomes"][0]["lead_time_steps"] == 1
    assert report["passed_detection_preflight"] is True
    assert len(report["feature_schema_sha256"]) == 64


def test_shadow_replay_rejects_late_detection_and_is_inconclusive_without_control(
    tmp_path: Path,
) -> None:
    candidate = _bundle("late-candidate", parent=None, rules=("delta",))
    late = tmp_path / "late.jsonl"
    late.write_text(
        "".join(
            json.dumps({"step_index": step, "state": {"progress": value}}) + "\n"
            for step, value in ((0, 0.0), (3, 0.0), (4, 1.0))
        ),
        encoding="utf-8",
    )
    report = evaluate_shadow_replay(
        candidate=candidate,
        parent=None,
        target_records=(
            (_record("late", success=False, parent=None, divergence=3), late),
        ),
        success_controls=(),
    )
    assert report["target_triggered_anywhere"] == 1
    assert report["target_detected"] == 0
    assert report["target_recall"] == 0.0
    assert report["preflight_conclusive"] is False
    assert report["passed_detection_preflight"] is False


def test_shadow_replay_does_not_turn_unknown_divergence_into_zero_recall(
    tmp_path: Path,
) -> None:
    candidate = _bundle("unknown-divergence", parent=None, rules=("delta",))
    states = tmp_path / "unknown.jsonl"
    states.write_text(
        "".join(
            json.dumps({"step_index": step, "state": {"progress": value}}) + "\n"
            for step, value in ((0, 0.0), (2, 1.0), (4, 1.0))
        ),
        encoding="utf-8",
    )
    report = evaluate_shadow_replay(
        candidate=candidate,
        parent=None,
        target_records=(
            (_record("unknown", success=False, parent=None, divergence=None), states),
        ),
        success_controls=(
            (_record("control", success=True, parent=None, divergence=None), states),
        ),
    )
    assert report["target_recall"] is None
    assert report["target_known_divergence_count"] == 0
    assert report["target_unknown_divergence_count"] == 1
    assert report["preflight_conclusive"] is False
    assert report["preflight_disposition"] == "inconclusive_unknown_divergence"
    assert report["passed_detection_preflight"] is False


def test_shadow_replay_allows_only_an_unavailable_feature_prefix(
    tmp_path: Path,
) -> None:
    candidate = _bundle("dynamic-feature", parent=None, rules=("delta",))
    startup_missing = tmp_path / "startup-missing.jsonl"
    startup_missing.write_text(
        "".join(
            json.dumps({"step_index": step, "state": state}) + "\n"
            for step, state in (
                (0, {"other": 0.0}),
                (1, {"progress": 0.0}),
                (2, {"progress": 1.0}),
            )
        ),
        encoding="utf-8",
    )
    report = evaluate_shadow_replay(
        candidate=candidate,
        parent=None,
        target_records=(
            (
                _record("startup-missing", success=False, parent=None, divergence=2),
                startup_missing,
            ),
        ),
        success_controls=(),
    )
    assert report["outcomes"][0]["unavailable_feature_prefix_steps"] == [0]
    assert report["outcomes"][0]["first_trigger_step"] == 2

    later_missing = tmp_path / "later-missing.jsonl"
    later_missing.write_text(
        "".join(
            json.dumps({"step_index": step, "state": state}) + "\n"
            for step, state in (
                (0, {"progress": 0.0}),
                (1, {"other": 0.0}),
            )
        ),
        encoding="utf-8",
    )
    late_report = evaluate_shadow_replay(
        candidate=candidate,
        parent=None,
        target_records=(
            (
                _record("later-missing", success=False, parent=None, divergence=2),
                later_missing,
            ),
        ),
        success_controls=(),
    )
    assert late_report["preflight_conclusive"] is False
    assert late_report["replay_unevaluable_count"] == 1
    assert late_report["outcomes"][0]["replay_evaluable"] is False


def test_shadow_replay_marks_legacy_missing_features_inconclusive(
    tmp_path: Path,
) -> None:
    """Old audit rows fail closed instead of crashing proposal stage."""
    candidate = _bundle("legacy-missing", parent=None, rules=("delta",))
    states = tmp_path / "legacy.jsonl"
    states.write_text(
        "".join(
            json.dumps({"step_index": step, "state": {"legacy_only": value}}) + "\n"
            for step, value in ((0, 0.0), (1, 1.0))
        ),
        encoding="utf-8",
    )
    report = evaluate_shadow_replay(
        candidate=candidate,
        parent=None,
        target_records=(
            (_record("legacy-target", success=False, parent=None, divergence=1), states),
        ),
        success_controls=(
            (_record("legacy-control", success=True, parent=None, divergence=None), states),
        ),
    )
    assert report["replay_unevaluable_count"] == 2
    assert report["preflight_conclusive"] is False
    assert report["passed_detection_preflight"] is False
    assert all(row["replay_evaluable"] is False for row in report["outcomes"])


def test_three_camera_visual_artifacts_are_materialized_and_aligned(
    tmp_path: Path,
) -> None:
    videos: dict[str, str] = {}
    for camera_index, camera in enumerate(("left", "right", "eye_in_hand")):
        path = tmp_path / f"{camera}.mp4"
        writer = iio.get_writer(path, fps=20, codec="libx264")
        try:
            for step in range(12):
                frame = np.full((32, 32, 3), step * 10 + camera_index, dtype=np.uint8)
                writer.append_data(frame)
        finally:
            writer.close()
        videos[f"video.robot0_{camera}"] = str(path)
    states = tmp_path / "states.jsonl"
    states.write_text(
        "".join(
            json.dumps({"step_index": step, "state": {"progress": step / 10}})
            + "\n"
            for step in range(11)
        ),
        encoding="utf-8",
    )
    report = build_episode_visual_artifacts(
        video_paths=videos,
        states_path=states,
        output_root=tmp_path / "visual",
        divergence_steps=(7,),
    )
    assert report["cameras"] == ["eye_in_hand", "left", "right"]
    assert report["divergence_steps"] == [7]
    assert set(report["artifacts"]) >= {
        "overview_contact_sheet",
        "divergence_contact_sheet_00",
        "event_contact_sheet_00",
        "divergence_clip",
        "manifest",
    }
    assert all(Path(path).is_file() for path in report["artifacts"].values())
    assert set(report["artifacts"]) == set(report["artifact_sha256"])


def test_two_camera_libero_visual_artifacts_are_materialized(tmp_path: Path) -> None:
    videos: dict[str, str] = {}
    for camera_index, camera in enumerate(("agentview", "wrist")):
        path = tmp_path / f"{camera}.mp4"
        writer = iio.get_writer(path, fps=10, codec="libx264")
        try:
            for step in range(8):
                writer.append_data(
                    np.full((24, 24, 3), step * 8 + camera_index, dtype=np.uint8)
                )
        finally:
            writer.close()
        videos[camera] = str(path)
    states = tmp_path / "states.jsonl"
    states.write_text(
        "".join(
            json.dumps({"step_index": step, "state": {"progress": step / 8}})
            + "\n"
            for step in range(8)
        ),
        encoding="utf-8",
    )

    report = build_episode_visual_artifacts(
        video_paths=videos,
        states_path=states,
        output_root=tmp_path / "visual",
        divergence_steps=(5,),
        source_fps=10,
        include_privileged_state_summary=True,
    )

    assert report["cameras"] == ["agentview", "wrist"]
    assert "multi-camera" in Path(report["artifacts"]["overview_contact_sheet"]).name
    assert all(Path(path).is_file() for path in report["artifacts"].values())
    summary = json.loads(
        Path(report["artifacts"]["privileged_state_summary"]).read_text(
            encoding="utf-8"
        )
    )
    assert summary["evidence_kind"] == "libero_privileged_critic_state_summary"
    assert summary["privacy"]["seed_and_rng"] == "excluded"
    assert all(
        "position" not in key and "target_offset" not in key
        for key in summary["field_names"]
    )


@pytest.mark.parametrize("task_id", range(10))
def test_all_goal_tasks_have_human_readable_bound_video_metadata(
    tmp_path: Path, task_id: int
) -> None:
    video_dir = tmp_path / "videos"
    agentview = video_dir / "episode_agentview.mp4"
    wrist = video_dir / "episode_agentview_wrist.mp4"
    video_dir.mkdir()
    for path in (agentview, wrist):
        path.write_bytes(b"video")
    report = write_video_metadata(
        video_dir=video_dir,
        video_paths={"agentview": str(agentview), "wrist": str(wrist)},
        visual_evidence={
            "artifacts": {
                "manifest": str(tmp_path / "visual-evidence-manifest.json")
            }
        },
        suite="libero_goal_task",
        task=f"libero_goal_task/task{task_id}",
        task_id=task_id,
        generation=1,
        logical_id="g001-rollout-004",
        attempt_index=2,
        episode_id="libero-episode-004",
        outcome="failure",
        status="valid",
        seed=11,
        policy_rng=1011,
    )
    index = json.loads(Path(report["index"]).read_text(encoding="utf-8"))
    readme = Path(report["readme"]).read_text(encoding="utf-8")
    assert index["episode"]["task"] == f"libero_goal_task/task{task_id}"
    assert index["episode"]["task_id"] == task_id
    assert [row["camera"] for row in index["videos"]] == ["agentview", "wrist"]
    assert "Logical rollout: g001-rollout-004" in readme
    assert "Attempt: 002" in readme
    assert f"Task: libero_goal_task/task{task_id} (task_id={task_id})" in readme


def test_campaign_image_is_delivered_as_visual_content_and_access_is_audited(
    tmp_path: Path,
) -> None:
    import imageio.v3 as iio3

    image = tmp_path / "overview.png"
    iio3.imwrite(image, np.full((8, 8, 3), 127, dtype=np.uint8))
    access = tmp_path / "access.jsonl"
    content_id = "artifact-" + "a" * 64
    agent = CodexStageAgent(
        output_root=tmp_path / "agent",
        artifact_reader=lambda requested: {
            "content_id": requested,
            "kind": "file",
            "path": str(image),
            "source": {
                "role": "indexed_artifact",
                "raw_key": "visual_evidence.overview_contact_sheet",
            },
        },
    )
    toolkit = Toolkit()
    agent._add_evidence_tool(toolkit, access_log=access)
    result = toolkit.execute_tool(
        "read_campaign_artifact", {"content_id": content_id}
    )
    assert any(block.get("type") == "image" for block in result.content_blocks)
    row = json.loads(access.read_text(encoding="utf-8"))
    assert row["content_id"] == content_id
    assert row["kind"] == "image"
    assert len(row["decoded_image_sha256"]) == 64
    CodexStageAgent._validate_visual_access(
        access,
        {
            "visual_evidence": [
                {
                    "content_id": content_id,
                    "access_record_id": row["access_record_id"],
                }
            ]
        },
    )
    with pytest.raises(ValueError, match="was not delivered"):
        CodexStageAgent._validate_visual_access(
            access,
            {
                "visual_evidence": [
                    {
                        "content_id": "artifact-" + "b" * 64,
                        "access_record_id": "visual-access-" + "b" * 64,
                    }
                ]
            },
        )
    with pytest.raises(ValueError, match="changed after invocation"):
        CodexStageAgent._validate_visual_access(
            access,
            {
                "visual_evidence": [
                    {
                        "content_id": content_id,
                        "access_record_id": row["access_record_id"],
                    }
                ]
            },
            expected_log_sha256="0" * 64,
        )


def test_stage1_requires_read_and_citation_of_privileged_summary(tmp_path: Path) -> None:
    access = tmp_path / "evidence-access.jsonl"
    content_id = "artifact-" + "c" * 64
    access.write_text(
        json.dumps(
            {
                "content_id": content_id,
                "kind": "structured",
                "access_record_id": "visual-access-" + "d" * 64,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    CodexStageAgent._validate_structured_access(
        access,
        required_content_ids={content_id},
    )
    with pytest.raises(ValueError, match="did not read required"):
        CodexStageAgent._validate_structured_access(
            access,
            required_content_ids={content_id, "artifact-" + "e" * 64},
        )


def test_visual_access_can_bind_immutable_upstream_stage_evidence(
    tmp_path: Path,
) -> None:
    inherited = tmp_path / "cluster-access.jsonl"
    content_id = "artifact-" + "c" * 64
    row = {
        "content_id": content_id,
        "kind": "image",
        "access_record_id": "visual-access-" + "d" * 64,
    }
    inherited.write_text(json.dumps(row) + "\n", encoding="utf-8")
    digest = hashlib.sha256(inherited.read_bytes()).hexdigest()
    current = tmp_path / "empty-current.jsonl"
    current.touch()

    CodexStageAgent._validate_visual_access(
        current,
        {
            "visual_evidence": [
                {
                    "content_id": content_id,
                    "access_record_id": row["access_record_id"],
                }
            ]
        },
        inherited_logs=((inherited, digest),),
    )


def test_campaign_artifact_read_budget_is_hard_and_audited(tmp_path: Path) -> None:
    import imageio.v3 as iio3

    image = tmp_path / "overview.png"
    iio3.imwrite(image, np.zeros((8, 8, 3), dtype=np.uint8))
    access = tmp_path / "access.jsonl"
    agent = CodexStageAgent(
        output_root=tmp_path / "agent",
        max_artifact_reads=2,
        artifact_reader=lambda requested: {
            "content_id": requested,
            "kind": "file",
            "path": str(image),
            "source": {"role": "indexed_artifact", "raw_key": "overview"},
        },
    )
    toolkit = Toolkit()
    agent._add_evidence_tool(toolkit, access_log=access)

    first = toolkit.execute_tool("read_campaign_artifact", {"content_id": "a"})
    cached = toolkit.execute_tool("read_campaign_artifact", {"content_id": "a"})
    second = toolkit.execute_tool("read_campaign_artifact", {"content_id": "b"})
    rejected = toolkit.execute_tool("read_campaign_artifact", {"content_id": "c"})

    assert first.result["artifact_read_number"] == 1
    assert cached.result["kind"] == "cached_reference"
    assert cached.result["artifact_cache_hit"] is True
    assert cached.result["artifact_read_number"] == 1
    assert second.result["artifact_read_number"] == 2
    assert second.result["artifact_read_limit"] == 2
    assert "budget exhausted" in rejected.result["error"]
    assert len(access.read_text(encoding="utf-8").splitlines()) == 2
