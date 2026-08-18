# Copyright (c) 2026 RPent Contributors
from __future__ import annotations

import os
import time
from dataclasses import replace
from pathlib import Path

import pytest

from rpent.evolution.campaign import analyze_failures
from rpent.evolution.clustering import cluster_failure_segments
from rpent.evolution.critic import TemporalCritic
from rpent.evolution.gating import (
    evaluate_fixed_heldout_20,
    evaluate_paired_gate,
    evaluate_two_stage_heldout,
)
from rpent.evolution.models import (
    CampaignManifest,
    CriticPredicate,
    CriticRule,
    EpisodeRecord,
    FailureSegment,
)
from rpent.evolution.queue import RolloutJob, SharedHostQueue
from rpent.evolution.schedule import preregister_seed_schedule
from rpent.evolution.stages import blind_artifact_index
from rpent.evolution.store import CampaignStore

SHA_A = "a" * 64


def _manifest(*, expected: int = 2) -> CampaignManifest:
    rollout, heldout, policy_rng = preregister_seed_schedule(
        master_seed=17,
        task="SlideDishwasherRack",
        rollout_count=expected,
        heldout_count=expected,
        population=range(100),
    )
    return CampaignManifest(
        campaign_id="test-campaign",
        environment="robocasa",
        task="SlideDishwasherRack",
        generation=0,
        code_commit="1" * 40,
        prompt_sha256=SHA_A,
        model="groot",
        tool_catalog_sha256="b" * 64,
        rollout_seeds=rollout,
        heldout_seeds=heldout,
        policy_rng_by_seed=policy_rng,
        expected_rollouts=expected,
        expected_heldout=expected,
    )


def _episode(
    *,
    logical_id: str,
    seed: int,
    policy_rng: int,
    attempt: int,
    status: str = "valid",
    success: bool | None = False,
    bundle_sha256: str | None = None,
) -> EpisodeRecord:
    return EpisodeRecord(
        episode_id=f"episode-{logical_id}-{attempt}",
        logical_id=logical_id,
        generation=0,
        seed=seed,
        policy_rng=policy_rng,
        bundle_sha256=bundle_sha256,
        status=status,
        success=success,
        started_at="2026-01-01T00:00:00+00:00",
        finished_at="2026-01-01T00:00:01+00:00",
        elapsed_s=1.0,
        artifact_index={
            "initial_observation_identity": {
                "state_sha256": f"{seed:064x}",
                "camera_sha256": {"agentview": f"{seed + 1:064x}"},
            },
            "candidate_intervention": not logical_id.startswith("p"),
            "trajectory_index": {
                "artifact_sha256": {
                    "actions": ("a" * 64 if logical_id.startswith("p") else "b" * 64)
                }
            },
        },
        invalid_reason="transport" if status == "infra_invalid" else None,
        attempt_index=attempt,
    )


def test_manifest_defaults_to_eight_api_agents_and_allows_up_to_twenty() -> None:
    manifest = _manifest(expected=1)
    assert manifest.maximum_api_concurrency == 8
    assert replace(manifest, maximum_api_concurrency=20).maximum_api_concurrency == 20
    with pytest.raises(ValueError, match="must not exceed 20"):
        replace(manifest, maximum_api_concurrency=21)


def test_attempts_are_append_only_but_only_valid_is_scored(tmp_path: Path) -> None:
    manifest = _manifest(expected=1)
    store = CampaignStore(tmp_path / "campaign")
    store.initialize(manifest)
    seed = manifest.rollout_seeds[0]
    policy_rng = manifest.policy_rng_by_seed[str(seed)]

    invalid = _episode(
        logical_id="g0000-rollout-000",
        seed=seed,
        policy_rng=policy_rng,
        attempt=0,
        status="infra_invalid",
        success=None,
    )
    assert store.record_episode(invalid) is False
    valid = _episode(
        logical_id=invalid.logical_id,
        seed=seed,
        policy_rng=policy_rng,
        attempt=1,
    )
    assert store.record_episode(valid) is True
    assert store.record_episode(valid) is False

    status = store.status()["episodes"]
    assert status == {
        "total_attempts": 2,
        "valid": 1,
        "infra_invalid": 1,
        "successes": 0,
    }
    assert len(store.attempts.records()) == 2
    assert len(store.episodes.records()) == 1


def test_seed_schedule_is_deterministic_disjoint_and_agent_blind() -> None:
    first = preregister_seed_schedule(
        master_seed=9,
        task="SlideDishwasherRack",
        rollout_count=50,
        heldout_count=50,
    )
    second = preregister_seed_schedule(
        master_seed=9,
        task="SlideDishwasherRack",
        rollout_count=50,
        heldout_count=50,
    )
    assert first == second
    rollout, heldout, policy_rng = first
    assert not set(rollout) & set(heldout)
    assert len(policy_rng) == 100
    blinded = blind_artifact_index(
        {
            "seed": 4,
            "policy_rng": 123,
            "nested": {
                "futureSeeds": [1, 2],
                "summary": "visible",
                "free_text": "seed=314159 at C:\\private\\trace.json",
            },
        }
    )
    assert blinded == {
        "nested": {
            "summary": "visible",
            "free_text": "[redacted-sensitive-metadata] at [redacted-locator]",
        }
    }


def test_seed_schedule_keeps_fixed_heldout_out_of_rollouts() -> None:
    fixed = tuple(range(1, 21))
    rollout, heldout, policy_rng = preregister_seed_schedule(
        master_seed=9,
        task="libero_goal_swap/task0",
        rollout_count=50,
        heldout_count=20,
        heldout_seeds=fixed,
    )

    assert heldout == fixed
    assert len(rollout) == 50
    assert set(rollout).isdisjoint(fixed)
    assert set(policy_rng) == {str(seed) for seed in (*rollout, *fixed)}

    with pytest.raises(ValueError, match="expected 20"):
        preregister_seed_schedule(
            master_seed=9,
            task="libero_goal_swap/task0",
            rollout_count=50,
            heldout_count=20,
            heldout_seeds=range(1, 20),
        )


def test_failure_clustering_is_order_stable() -> None:
    segments = [
        FailureSegment(
            segment_id=f"s{index}",
            episode_id=f"e{index}",
            failure_class="critic_stall",
            stage="handoff",
            tool="groot",
            summary=summary,
            earliest_divergence_step=3,
            start_step=2,
            end_step=5,
        )
        for index, summary in enumerate(
            ("rack motion stalls", "rack movement stalls", "wrong direction")
        )
    ]
    left = cluster_failure_segments(segments, similarity_threshold=0.2)
    right = cluster_failure_segments(reversed(segments), similarity_threshold=0.2)
    assert [item.as_dict() for item in left] == [item.as_dict() for item in right]


def test_episode_roundtrip_and_analysis_preserve_all_failure_segments(
    tmp_path: Path,
) -> None:
    manifest = _manifest(expected=1)
    store = CampaignStore(tmp_path / "campaign")
    store.initialize(manifest)
    seed = manifest.rollout_seeds[0]
    segments = tuple(
        FailureSegment(
            segment_id=f"segment-{failure_class}",
            episode_id="episode-multi",
            failure_class=failure_class,
            stage="engage",
            tool=tool,
            summary=summary,
            earliest_divergence_step=step,
            start_step=max(0, step - 1),
            end_step=step + 1,
        )
        for failure_class, tool, summary, step in (
            ("critic_reject", "critic", "engagement proposal rejected", 3),
            ("tool_error", "grasp_planner", "grasp candidate failed", 7),
        )
    )
    record = EpisodeRecord(
        episode_id="episode-multi",
        logical_id="g0000-rollout-000",
        generation=0,
        seed=seed,
        policy_rng=manifest.policy_rng_by_seed[str(seed)],
        bundle_sha256=None,
        status="valid",
        success=False,
        started_at="2026-01-01T00:00:00+00:00",
        finished_at="2026-01-01T00:00:01+00:00",
        elapsed_s=1.0,
        artifact_index={"trajectory": "trajectory.jsonl"},
        failure_segment=segments[0],
        failure_segments=segments,
    )
    roundtripped = EpisodeRecord.from_dict(record.as_dict())
    assert roundtripped.failure_segment == segments[0]
    assert roundtripped.all_failure_segments == segments
    assert store.record_episode(roundtripped) is True

    report = analyze_failures(store.root)
    assert report["failures_with_segments"] == 1
    assert report["failure_segments"] == 2
    clustered_ids = {
        segment_id
        for cluster in report["clusters"]
        for segment_id in cluster["member_segment_ids"]
    }
    assert clustered_ids == {segment.segment_id for segment in segments}


def test_episode_rejects_inconsistent_or_invalid_failure_segments() -> None:
    segment = FailureSegment(
        segment_id="segment-primary",
        episode_id="episode-g0000-rollout-000-0",
        failure_class="stall",
        stage="engage",
        tool="groot",
        summary="stalled",
        earliest_divergence_step=1,
        start_step=0,
        end_step=2,
    )
    other = replace(segment, segment_id="segment-other")
    base = _episode(
        logical_id="g0000-rollout-000",
        seed=1,
        policy_rng=2,
        attempt=0,
    )
    with pytest.raises(ValueError, match="primary failure segment"):
        replace(base, failure_segment=segment, failure_segments=(other,))
    with pytest.raises(ValueError, match="infra-invalid episode"):
        replace(
            base,
            status="infra_invalid",
            success=None,
            invalid_reason="transport",
            failure_segment=segment,
            failure_segments=(segment,),
        )


def test_failure_clustering_uses_complete_link_not_similarity_chaining(
    monkeypatch,
) -> None:
    segments = [
        FailureSegment(
            segment_id=value,
            episode_id=f"episode-{value}",
            failure_class="stall",
            stage="handoff",
            tool="groot",
            summary=value,
            earliest_divergence_step=1,
            start_step=0,
            end_step=2,
        )
        for value in ("a", "b", "c")
    ]
    scores = {
        frozenset(("a", "b")): 0.9,
        frozenset(("b", "c")): 0.9,
        frozenset(("a", "c")): 0.1,
    }

    def similarity(left: FailureSegment, right: FailureSegment) -> float:
        if left.segment_id == right.segment_id:
            return 1.0
        return scores[frozenset((left.segment_id, right.segment_id))]

    monkeypatch.setattr("rpent.evolution.clustering.segment_similarity", similarity)
    clusters = cluster_failure_segments(segments, similarity_threshold=0.8)
    assert sorted(len(cluster.member_segment_ids) for cluster in clusters) == [1, 2]


def test_temporal_critic_is_proposal_only_and_stagnant_uses_one_window() -> None:
    rule = CriticRule(
        rule_id="stall",
        title="stall",
        feature="progress",
        operator="stagnant",
        threshold=0.01,
        dwell_steps=3,
        cooldown_steps=2,
        proposal="replan",
        evidence_ids=("segment-1",),
    )
    critic = TemporalCritic((rule,))
    assert critic.evaluate({"progress": 1.0}, step_index=1) == []
    assert critic.evaluate({"progress": 1.0}, step_index=2) == []
    proposal = critic.evaluate({"progress": 1.0}, step_index=3)
    assert proposal[0]["environment_write"] is False
    assert proposal[0]["rule_id"] == "stall"


def test_temporal_critic_accepts_flattened_robocasa_feature_names() -> None:
    rule = CriticRule(
        rule_id="rack-residual",
        title="rack residual remains high",
        feature="privileged.dishwasher.rack.residual_to_success",
        operator="gt",
        threshold=0.1,
        dwell_steps=1,
        cooldown_steps=0,
        proposal="inspect rack engagement",
        evidence_ids=("segment-flat-state",),
    )
    critic = TemporalCritic((rule,))
    proposal = critic.evaluate(
        {"privileged.dishwasher.rack.residual_to_success": 0.45},
        step_index=1,
    )
    assert proposal[0]["rule_id"] == "rack-residual"
    assert proposal[0]["environment_write"] is False


def test_temporal_critic_activation_guard_resets_precondition_history() -> None:
    rule = CriticRule(
        rule_id="post-contact-stall",
        title="post-contact progress stall",
        feature="progress",
        operator="stagnant",
        threshold=0.01,
        dwell_steps=3,
        cooldown_steps=0,
        proposal="recover contact",
        evidence_ids=("segment-guard",),
        activation_conditions=(
            CriticPredicate(feature="target_contact", operator="eq", threshold=True),
        ),
    )
    critic = TemporalCritic((rule,))
    for step in range(1, 5):
        assert (
            critic.evaluate({"progress": 1.0, "target_contact": False}, step_index=step)
            == []
        )
    assert (
        critic.evaluate({"progress": 1.0, "target_contact": True}, step_index=5) == []
    )
    assert (
        critic.evaluate({"progress": 1.0, "target_contact": True}, step_index=6) == []
    )
    proposal = critic.evaluate({"progress": 1.0, "target_contact": True}, step_index=7)
    assert proposal[0]["activation_conditions"][0]["observed_value"] is True


def test_two_stage_heldout_requires_exact_paired_evidence() -> None:
    seeds = tuple(range(50))
    rng = {seed: seed + 1000 for seed in seeds}
    parent = [
        _episode(logical_id=f"p{seed}", seed=seed, policy_rng=rng[seed], attempt=0)
        for seed in seeds[:10]
    ]
    candidate = [
        _episode(
            logical_id=f"c{seed}",
            seed=seed,
            policy_rng=rng[seed],
            attempt=0,
            success=seed < 6,
            bundle_sha256="c" * 64,
        )
        for seed in seeds[:10]
    ]
    decision = evaluate_two_stage_heldout(
        candidate_sha256="c" * 64,
        parent_sha256=None,
        candidate_records=candidate,
        parent_records=parent,
        preregistered_seeds=seeds,
        stage=1,
    )
    assert decision.passed
    assert decision.p_value == pytest.approx(0.015625)
    with pytest.raises(ValueError, match="exactly one valid"):
        evaluate_two_stage_heldout(
            candidate_sha256="c" * 64,
            parent_sha256=None,
            candidate_records=candidate[:-1],
            parent_records=parent,
            preregistered_seeds=seeds,
            stage=1,
        )


def test_fixed_heldout_20_is_a_distinct_paired_contract() -> None:
    seeds = tuple(range(20))
    rng = {seed: seed + 1000 for seed in seeds}
    parent = [
        _episode(
            logical_id=f"p{seed}",
            seed=seed,
            policy_rng=rng[seed],
            attempt=0,
            success=False,
            bundle_sha256="d" * 64,
        )
        for seed in seeds
    ]
    candidate = [
        _episode(
            logical_id=f"c{seed}",
            seed=seed,
            policy_rng=rng[seed],
            attempt=0,
            success=seed < 6,
            bundle_sha256="c" * 64,
        )
        for seed in seeds
    ]
    decision = evaluate_fixed_heldout_20(
        candidate_sha256="c" * 64,
        parent_sha256="d" * 64,
        candidate_records=candidate,
        parent_records=parent,
        preregistered_seeds=seeds,
    )
    assert decision.kind == "heldout_20"
    assert decision.paired_count == 20
    assert decision.passed is True
    with pytest.raises(ValueError, match="exactly 20"):
        evaluate_fixed_heldout_20(
            candidate_sha256="c" * 64,
            parent_sha256="d" * 64,
            candidate_records=candidate,
            parent_records=parent,
            preregistered_seeds=seeds[:10],
        )


def test_same_seed_gate_requires_reproduced_failure_rng_state_and_divergence() -> None:
    parent = _episode(
        logical_id="parent",
        seed=7,
        policy_rng=123,
        attempt=0,
        success=False,
        bundle_sha256="d" * 64,
    )
    candidate = _episode(
        logical_id="candidate",
        seed=7,
        policy_rng=123,
        attempt=0,
        success=True,
        bundle_sha256="c" * 64,
    )
    decision = evaluate_paired_gate(
        kind="same_seed",
        candidate_sha256="c" * 64,
        parent_sha256="d" * 64,
        candidate_records=[candidate],
        parent_records=[parent],
        expected_seeds=(7,),
    )
    assert decision.passed

    with pytest.raises(ValueError, match="policy RNG"):
        evaluate_paired_gate(
            kind="same_seed",
            candidate_sha256="c" * 64,
            parent_sha256="d" * 64,
            candidate_records=[replace(candidate, policy_rng=124)],
            parent_records=[parent],
            expected_seeds=(7,),
        )
    changed_index = {
        **candidate.artifact_index,
        "initial_observation_identity": {
            "state_sha256": "f" * 64,
            "camera_sha256": {"agentview": "e" * 64},
        },
    }
    with pytest.raises(ValueError, match="observation identity differs"):
        evaluate_paired_gate(
            kind="same_seed",
            candidate_sha256="c" * 64,
            parent_sha256="d" * 64,
            candidate_records=[replace(candidate, artifact_index=changed_index)],
            parent_records=[parent],
            expected_seeds=(7,),
        )
    camera_only_change = {
        **candidate.artifact_index,
        "initial_observation_identity": {
            **candidate.artifact_index["initial_observation_identity"],
            "camera_sha256": {"agentview": "f" * 64},
        },
    }
    camera_drift = evaluate_paired_gate(
        kind="same_seed",
        candidate_sha256="c" * 64,
        parent_sha256="d" * 64,
        candidate_records=[replace(candidate, artifact_index=camera_only_change)],
        parent_records=[parent],
        expected_seeds=(7,),
    )
    assert camera_drift.passed
    assert "camera digests differed for 1/1" in camera_drift.rationale

    unowned_change = {
        key: value
        for key, value in candidate.artifact_index.items()
        if key != "candidate_intervention"
    }
    missing_intervention = evaluate_paired_gate(
        kind="same_seed",
        candidate_sha256="c" * 64,
        parent_sha256="d" * 64,
        candidate_records=[replace(candidate, artifact_index=unowned_change)],
        parent_records=[parent],
        expected_seeds=(7,),
    )
    assert not missing_intervention.passed
    assert "never changed" in missing_intervention.rationale
    same_actions = {
        **candidate.artifact_index,
        "trajectory_index": parent.artifact_index["trajectory_index"],
    }
    no_intervention = evaluate_paired_gate(
        kind="same_seed",
        candidate_sha256="c" * 64,
        parent_sha256="d" * 64,
        candidate_records=[replace(candidate, artifact_index=same_actions)],
        parent_records=[parent],
        expected_seeds=(7,),
    )
    assert no_intervention.conclusive
    assert not no_intervention.passed
    assert "never changed" in no_intervention.rationale

    already_successful = replace(parent, success=True)
    decision = evaluate_paired_gate(
        kind="same_seed",
        candidate_sha256="c" * 64,
        parent_sha256="d" * 64,
        candidate_records=[candidate],
        parent_records=[already_successful],
        expected_seeds=(7,),
    )
    assert not decision.passed

    with pytest.raises(ValueError, match="candidate gate records"):
        evaluate_paired_gate(
            kind="same_seed",
            candidate_sha256="c" * 64,
            parent_sha256="d" * 64,
            candidate_records=[replace(candidate, bundle_sha256=None)],
            parent_records=[parent],
            expected_seeds=(7,),
        )


def test_same_seed_gate_uses_frozen_half_success_threshold_and_requires_gain() -> None:
    seeds = (1, 2, 3, 4)
    parent = [
        _episode(
            logical_id=f"parent-{seed}",
            seed=seed,
            policy_rng=100 + seed,
            attempt=0,
            success=False,
            bundle_sha256="d" * 64,
        )
        for seed in seeds
    ]
    candidate = [
        _episode(
            logical_id=f"candidate-{seed}",
            seed=seed,
            policy_rng=100 + seed,
            attempt=0,
            success=seed in {1, 2},
            bundle_sha256="c" * 64,
        )
        for seed in seeds
    ]
    decision = evaluate_paired_gate(
        kind="same_seed",
        candidate_sha256="c" * 64,
        parent_sha256="d" * 64,
        candidate_records=candidate,
        parent_records=parent,
        expected_seeds=seeds,
        same_seed_pass_rate=0.5,
    )
    assert decision.passed
    assert decision.candidate_successes == 2
    assert "2/4" in decision.rationale

    below_threshold = [replace(row, success=row.seed == 1) for row in candidate]
    assert not evaluate_paired_gate(
        kind="same_seed",
        candidate_sha256="c" * 64,
        parent_sha256="d" * 64,
        candidate_records=below_threshold,
        parent_records=parent,
        expected_seeds=seeds,
        same_seed_pass_rate=0.5,
    ).passed

    parent_already_two = [
        replace(row, success=row.seed in {1, 2}) for row in parent
    ]
    assert not evaluate_paired_gate(
        kind="same_seed",
        candidate_sha256="c" * 64,
        parent_sha256="d" * 64,
        candidate_records=candidate,
        parent_records=parent_already_two,
        expected_seeds=seeds,
        same_seed_pass_rate=0.5,
    ).passed


def test_queue_counts_do_not_double_count_envelopes_and_recovers_stale(
    tmp_path: Path,
) -> None:
    queue = SharedHostQueue(tmp_path / "queue")
    job = RolloutJob(
        job_id="job-1",
        campaign_root=str(tmp_path / "campaign"),
        logical_id="logical-1",
        attempt_index=0,
        task="SlideDishwasherRack",
        seed=1,
        policy_rng=2,
        bundle_sha256=None,
        command=("python", "-c", "print('ok')"),
        output_dir=str(tmp_path / "output"),
        result_file=str(tmp_path / "output" / "record.json"),
        heartbeat_file=str(tmp_path / "output" / "heartbeat.jsonl"),
    )
    queue.enqueue("host-a", job)
    claimed, _ = queue.claim("host-a") or (None, None)
    assert claimed is not None
    stale = time.time() - 100
    os.utime(claimed, (stale, stale))
    assert queue.recover_abandoned("host-a", stale_after_s=10) == 1
    assert queue.counts() == {
        "pending": 0,
        "running": 0,
        "completed": 0,
        "failed": 1,
    }
