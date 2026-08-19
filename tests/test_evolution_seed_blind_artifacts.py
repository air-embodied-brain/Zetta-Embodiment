# Copyright (c) 2026 Zetta Contributors
from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import zetta.evolution.lifecycle as lifecycle
from zetta.evolution.campaign import analyze_failures
from zetta.evolution.jsonio import read_json
from zetta.evolution.lifecycle import (
    _agent_artifact_index,
    resolve_agent_artifact,
    run_diagnosis_stage,
)
from zetta.evolution.models import (
    CampaignManifest,
    CausalDiagnosis,
    EpisodeRecord,
    FailureSegment,
)
from zetta.evolution.stages import CodexStageAgent
from zetta.evolution.store import CampaignStore
from zetta.tools.toolkit import Toolkit


def _manifest() -> CampaignManifest:
    return CampaignManifest(
        campaign_id="seed-blind-artifacts",
        environment="robocasa",
        task="SlideDishwasherRack",
        generation=0,
        code_commit="1" * 40,
        prompt_sha256="2" * 64,
        model="test",
        tool_catalog_sha256="3" * 64,
        rollout_seeds=(314159, 161803),
        heldout_seeds=(112358,),
        policy_rng_by_seed={
            "314159": 271828,
            "161803": 141421,
            "112358": 173205,
        },
        expected_rollouts=2,
        expected_heldout=1,
        # These unit fixtures intentionally use text-only evidence to isolate
        # seed/RNG redaction. Formal multimodal behavior is covered separately.
        protocol_explicit=False,
    )


def _record(
    *,
    status: str,
    logical_id: str,
    episode_id: str,
    seed: int,
    policy_rng: int,
    artifact_index: dict[str, Any],
    attempt_index: int = 0,
    success: bool = False,
) -> EpisodeRecord:
    valid = status == "valid"
    return EpisodeRecord(
        episode_id=episode_id,
        logical_id=logical_id,
        generation=0,
        seed=seed,
        policy_rng=policy_rng,
        bundle_sha256=None,
        status=status,  # type: ignore[arg-type]
        success=success if valid else None,
        started_at="2026-01-01T00:00:00+00:00",
        finished_at="2026-01-01T00:00:01+00:00",
        elapsed_s=1.0,
        artifact_index=artifact_index,
        failure_segment=(
            FailureSegment(
                segment_id="failure-seed-314159-window",
                episode_id=episode_id,
                failure_class="stall",
                stage="handoff",
                tool="robocasa.vla.groot",
                summary=(
                    "seed=314159 diverged at C:\\private\\seed-314159\\trajectory.jsonl"
                ),
                earliest_divergence_step=2,
                start_step=1,
                end_step=3,
            )
            if valid and not success
            else None
        ),
        invalid_reason=None if valid else "transport-invalid-secret",
        attempt_index=attempt_index,
    )


def _content_id_for_path(resolver: dict[str, Any], path: Path) -> str:
    target = str(path.resolve())
    matches = [
        content_id
        for content_id, entry in resolver["entries"].items()
        if any(
            source.get("kind") == "file" and source.get("path") == target
            for source in entry["sources"]
        )
    ]
    assert len(matches) == 1
    return matches[0]


def test_agent_index_is_opaque_content_addressed_and_resolvable(tmp_path: Path) -> None:
    root = tmp_path / "campaign"
    first = root / "captures" / "seed-314159-trajectory.jsonl"
    duplicate = root / "other" / "future-seed-112358-copy.jsonl"
    different = root / "captures" / "seed-314159-different.jsonl"
    overview = root / "captures" / "private-overview.png"
    invalid_only = root / "invalid" / "seed-161803-private.log"
    for path, content in (
        (first, "same trajectory bytes\n"),
        (duplicate, "same trajectory bytes\n"),
        (different, "different trajectory bytes\n"),
        (invalid_only, "must never enter the diagnosis resolver\n"),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    overview.write_bytes(b"opaque-image-bytes")

    store = CampaignStore(root)
    store.initialize(_manifest())
    store.record_episode(
        _record(
            status="infra_invalid",
            logical_id="logical-seed-161803-invalid",
            episode_id="episode-seed-161803-invalid",
            seed=161803,
            policy_rng=141421,
            artifact_index={"seed_private_path": str(invalid_only)},
        )
    )
    primary = _record(
        status="valid",
        logical_id="logical-seed-314159-valid",
        episode_id="episode-seed-314159-valid",
        seed=314159,
        policy_rng=271828,
        artifact_index={
            "trajectory_seed_314159": str(first),
            "duplicate_filename": str(duplicate),
            "different_content": str(different),
            "display_text": "policy_rng=271828 for future_schedule=[112358]",
            "future_schedule": [112358, 271828],
            "visual_evidence": {
                "artifacts": {"overview_contact_sheet": str(overview)}
            },
        },
    )
    assert primary.failure_segment is not None
    secondary = replace(
        primary.failure_segment,
        segment_id="failure-seed-314159-tool-window",
        failure_class="tool_error",
        tool="grasp_planner",
        summary="tool failed for seed=314159",
    )
    store.record_episode(
        replace(
            primary,
            failure_segments=(primary.failure_segment, secondary),
        )
    )

    first_index = _agent_artifact_index(store)
    second_index = _agent_artifact_index(CampaignStore(root))
    assert first_index == second_index
    assert first_index["artifacts"]
    assert all(
        set(descriptor) == {"type", "summary", "hash", "content_id"}
        for descriptor in first_index["artifacts"]
    )
    assert len(first_index["relationships"]) == 1
    relationship = first_index["relationships"][0]
    assert relationship["episode"].startswith("artifact-")
    assert all(
        row["segment"].startswith("artifact-") for row in relationship["segments"]
    )
    assert relationship["visual_evidence"][0]["role"] == "episode_overview"
    assert relationship["visual_evidence"][0]["content_id"].startswith("artifact-")

    visible = json.dumps(first_index, ensure_ascii=False).lower()
    for forbidden in (
        "seed",
        "rng",
        "future_schedule",
        "314159",
        "271828",
        "112358",
        first.name.lower(),
        duplicate.name.lower(),
        "logical-seed-314159-valid",
        "episode-seed-314159-valid",
        "failure-seed-314159-window",
        "transport-invalid-secret",
    ):
        assert forbidden not in visible

    resolver_path = root / ".harness-private" / "artifact-resolver.json"
    resolver = read_json(resolver_path)
    resolver_text = resolver_path.read_text(encoding="utf-8")
    assert "logical-seed-314159-valid" in resolver_text
    assert "future_schedule" in resolver_text
    assert "transport-invalid-secret" not in resolver_text
    assert str(invalid_only.resolve()) not in resolver_text
    assert set(resolver["aliases"]["segment_id"]) == {
        "failure-seed-314159-window",
        "failure-seed-314159-tool-window",
    }

    first_id = _content_id_for_path(resolver, first)
    duplicate_id = _content_id_for_path(resolver, duplicate)
    different_id = _content_id_for_path(resolver, different)
    assert first_id == duplicate_id
    assert first_id != different_id
    assert first_id.startswith("artifact-")
    assert len(first_id) == len("artifact-") + 64

    # A fresh CampaignStore/process can recover the private mapping and verify
    # the immutable content before handing its locator to the Harness.
    resolved = resolve_agent_artifact(root, first_id)
    assert resolved["kind"] == "file"
    assert Path(resolved["path"]).read_text(encoding="utf-8") == (
        "same trajectory bytes\n"
    )


def test_agent_index_reuses_accepted_visual_digest_without_rescanning_video(
    tmp_path: Path, monkeypatch: Any
) -> None:
    root = tmp_path / "campaign"
    video = root / "captures" / "three-camera-overview.mp4"
    video.parent.mkdir(parents=True, exist_ok=True)
    video.write_bytes(b"large-video-placeholder")
    digest = hashlib.sha256(video.read_bytes()).hexdigest()

    store = CampaignStore(root)
    store.initialize(_manifest())
    store.record_episode(
        _record(
            status="valid",
            logical_id="logical-seed-314159-valid",
            episode_id="episode-seed-314159-valid",
            seed=314159,
            policy_rng=271828,
            artifact_index={
                "visual_evidence": {
                    "artifacts": {"overview_contact_sheet": str(video)},
                    "artifact_sha256": {"overview_contact_sheet": digest},
                }
            },
        )
    )

    real_file_sha256 = lifecycle.file_sha256

    def fail_on_video(path: Path) -> str:
        if Path(path).resolve() == video.resolve():
            raise AssertionError("accepted visual artifact was re-scanned")
        return real_file_sha256(path)

    monkeypatch.setattr(lifecycle, "file_sha256", fail_on_video)
    index = _agent_artifact_index(store)
    assert any(row["type"] == "video" for row in index["artifacts"])


class _CapturingAgent:
    seen: dict[str, Any] = {}

    def __init__(self, **_: object) -> None:
        pass

    def diagnose(self, **values: Any) -> CausalDiagnosis:
        type(self).seen = values
        cluster = values["cluster"]
        evidence_id = values["artifact_index"]["artifacts"][0]["content_id"]
        return CausalDiagnosis(
            diagnosis_id="opaque-diagnosis",
            cluster_id=cluster.cluster_id,
            outcome="the bounded handoff does not finish the task",
            immediate_trigger="progress remains unchanged after handoff",
            root_cause="bounded handoff stalled",
            contributing_causes=(),
            competing_hypotheses=(
                "bounded handoff stalled",
                "the selected action direction is incorrect",
            ),
            owner_layer="recovery",
            affected_component="VLA handoff recovery",
            earliest_divergence="first progress stall",
            supporting_evidence_ids=(evidence_id,),
            counterevidence_ids=(),
            falsifier="paired recovery still stalls",
            distinguishing_check="progress resumes after intervention",
            required_validation="paired same-state live recovery",
            confidence=0.8,
        )


def test_stage1_receives_content_ids_not_episode_or_segment_ids(
    tmp_path: Path, monkeypatch
) -> None:
    root = tmp_path / "campaign"
    store = CampaignStore(root)
    store.initialize(_manifest())
    store.record_episode(
        _record(
            status="valid",
            logical_id="logical-seed-314159-valid",
            episode_id="episode-seed-314159-valid",
            seed=314159,
            policy_rng=271828,
            artifact_index={"display": "seed 314159 policy_rng 271828"},
        )
    )
    store.record_episode(
        _record(
            status="valid",
            logical_id="logical-seed-161803-valid",
            episode_id="episode-seed-161803-valid",
            seed=161803,
            policy_rng=141421,
            artifact_index={"display": "successful control episode"},
            success=True,
        )
    )
    analyze_failures(root)
    monkeypatch.setattr("zetta.evolution.lifecycle.CodexStageAgent", _CapturingAgent)
    run_diagnosis_stage(campaign_root=root, tool_catalog={"tools": []})

    seen = _CapturingAgent.seen
    cluster = seen["cluster"]
    assert cluster.member_segment_ids[0].startswith("artifact-")
    assert cluster.episode_ids[0].startswith("artifact-")
    assert cluster.medoid_segment_id.startswith("artifact-")
    serialized = json.dumps(
        {
            "cluster": cluster.as_dict(),
            "artifact_index": seen["artifact_index"],
        },
        ensure_ascii=False,
    ).lower()
    assert "314159" not in serialized
    assert "271828" not in serialized
    assert "logical-seed" not in serialized
    assert "episode-seed" not in serialized
    assert "failure-seed" not in serialized
    assert "policy_rng" not in serialized


def test_stage1_evidence_tool_reads_opaque_artifact_without_metadata_leak(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "private" / "seed-314159-trajectory.json"
    artifact.parent.mkdir(parents=True)
    artifact.write_text(
        json.dumps(
            {
                "environment_seed": 314159,
                "policy_rng": 271828,
                "episode_id": "opaque-but-private-episode",
                "logical_id": "opaque-but-private-logical",
                "segment_id": "opaque-but-private-segment",
                "attempt_index": 4,
                "bundle_sha256": "b" * 64,
                "phase": "handoff",
                "message": "seed=314159 at C:\\private\\seed-314159\\trace.json",
            }
        ),
        encoding="utf-8",
    )
    agent = CodexStageAgent(
        output_root=tmp_path / "agent",
        artifact_reader=lambda content_id: {
            "content_id": content_id,
            "kind": "file",
            "path": str(artifact),
        },
    )
    toolkit = Toolkit()
    agent._add_evidence_tool(toolkit)
    result = toolkit.execute_tool(
        "read_campaign_artifact", {"content_id": "artifact-" + "a" * 64}
    ).result
    serialized = json.dumps(result, ensure_ascii=False).lower()
    assert result["value"]["phase"] == "handoff"
    assert "314159" not in serialized
    assert "271828" not in serialized
    assert "policy_rng" not in serialized
    assert "opaque-but-private" not in serialized
    assert "attempt_index" not in serialized
    assert "bundle_sha256" not in serialized
    assert "private" not in serialized
    assert "path" not in serialized


def test_agent_resolver_does_not_index_file_outside_campaign(tmp_path: Path) -> None:
    root = tmp_path / "campaign"
    outside = tmp_path / "outside-seed-314159.jsonl"
    outside.write_text("private external evidence", encoding="utf-8")
    store = CampaignStore(root)
    store.initialize(_manifest())
    store.record_episode(
        _record(
            status="valid",
            logical_id="logical-seed-314159-valid",
            episode_id="episode-seed-314159-valid",
            seed=314159,
            policy_rng=271828,
            artifact_index={"external_path": str(outside)},
        )
    )
    store.record_episode(
        _record(
            status="valid",
            logical_id="logical-seed-161803-valid",
            episode_id="episode-seed-161803-valid",
            seed=161803,
            policy_rng=141421,
            artifact_index={"display": "successful control episode"},
            success=True,
        )
    )
    index = _agent_artifact_index(store)
    resolver = read_json(root / ".harness-private" / "artifact-resolver.json")
    assert not any(
        source.get("kind") == "file" and source.get("path") == str(outside.resolve())
        for entry in resolver["entries"].values()
        for source in entry["sources"]
    )
    assert "outside-seed-314159" not in json.dumps(index).lower()


def test_diagnostic_telemetry_is_compact_and_keeps_privileged_transitions(
    tmp_path: Path,
) -> None:
    root = tmp_path / "campaign"
    failure_states = root / "artifacts" / "failure" / "states.jsonl"
    success_states = root / "artifacts" / "success" / "states.jsonl"
    failure_states.parent.mkdir(parents=True, exist_ok=True)
    success_states.parent.mkdir(parents=True, exist_ok=True)

    def write_states(path: Path, *, success: bool) -> None:
        rows = []
        for step in range(8):
            grasped = step >= 3 if not success else step >= 2
            rows.append(
                {
                    "step_index": step,
                    "state": {
                        "command.translation.x": 0.1,
                        "command.realization.stalled": step == 1,
                        "privileged.task.manipulated_object.grasped": grasped,
                        "privileged.task.stage.name": (
                            "transport" if grasped else "pregrasp"
                        ),
                        "privileged.task.success": success and step == 7,
                        "privileged.joint.robot0_joint1.normalized": 0.1,
                        "privileged.joint.wooden_cabinet_top.normalized": (
                            step / 7
                        ),
                        "policy_rng": 999,
                    },
                }
            )
        path.write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )

    write_states(failure_states, success=False)
    write_states(success_states, success=True)
    store = CampaignStore(root)
    store.initialize(_manifest())
    store.record_episode(
        _record(
            status="valid",
            logical_id="logical-seed-314159-failure",
            episode_id="episode-seed-314159-failure",
            seed=314159,
            policy_rng=271828,
            artifact_index={"states": str(failure_states)},
        )
    )
    store.record_episode(
        _record(
            status="valid",
            logical_id="logical-seed-161803-success",
            episode_id="episode-seed-161803-success",
            seed=161803,
            policy_rng=141421,
            artifact_index={"states": str(success_states)},
            success=True,
        )
    )

    index = _agent_artifact_index(store)
    traces = index["diagnostic_telemetry"]
    assert len(traces) == 2
    assert {trace["outcome"] for trace in traces} == {"failure", "success"}
    failure_trace = next(trace for trace in traces if trace["outcome"] == "failure")
    resolved = resolve_agent_artifact(root, failure_trace["content_id"])
    assert resolved["kind"] == "inline"
    value = resolved["value"]
    assert value["sampling"]["source_row_count"] == 8
    assert value["sampling"]["sampled_row_count"] < 8
    steps = set(value["sampled_step_indices"])
    assert {0, 1, 3, 7}.issubset(steps)
    assert "privileged.task.manipulated_object.grasped" in value["series"]
    assert "privileged.joint.wooden_cabinet_top.normalized" in value["series"]
    assert "privileged.joint.robot0_joint1.normalized" not in value["series"]
    assert all(
        len(series) == len(value["sampled_step_indices"])
        for series in value["series"].values()
    )
    visible = json.dumps(index, ensure_ascii=False).lower()
    assert "policy_rng" not in visible
    assert "seed-314159" not in visible
