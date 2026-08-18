# Copyright (c) 2026 RPent Contributors
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from robots.robocasa.slide_dishwasher_program import CONTACT_PUSH_TOOL
from rpent.evolution.campaign import analyze_failures
from rpent.evolution.jsonio import atomic_write_json, canonical_sha256, read_json
from rpent.evolution.lifecycle import (
    _agent_artifact_context,
    _authoritative_task_contract,
    _latest_stage2_context,
    _materialize_multimodal_cluster_review,
    _observed_critic_features,
    _route_inconclusive_diagnosis,
    authorize_provisional_hypothesis,
    authorize_shadow_falsification,
    effective_same_seed_pass_rate,
    materialize_cluster_targets,
    promote_and_complete,
    record_gate_and_advance,
    run_diagnosis_stage,
    run_proposal_stage,
)
from rpent.evolution.models import (
    CampaignManifest,
    CampaignPhase,
    CandidateBundle,
    CausalDiagnosis,
    CriticRule,
    EpisodeRecord,
    FailureSegment,
    GateDecision,
    RecoveryRule,
    RecoveryStep,
)
from rpent.evolution.store import CampaignStore


def _manifest() -> CampaignManifest:
    return CampaignManifest(
        campaign_id="lifecycle",
        environment="robocasa",
        task="SlideDishwasherRack",
        generation=0,
        code_commit="1" * 40,
        prompt_sha256="2" * 64,
        model="test",
        tool_catalog_sha256="3" * 64,
        rollout_seeds=(7,),
        heldout_seeds=(8,),
        policy_rng_by_seed={"7": 70, "8": 80},
        expected_rollouts=1,
        expected_heldout=1,
        protocol_explicit=False,
    )


def test_formal_libero_lifecycle_requires_authoritative_task_contract(
    tmp_path: Path,
) -> None:
    root = tmp_path / "libero-missing-language"
    store = CampaignStore(root)
    store.initialize(
        replace(
            _manifest(),
            environment="libero_pro",
            task="libero_10_task/task5",
            runtime={"suite": "libero_10_task", "task_id": 5},
        )
    )

    with pytest.raises(ValueError, match="requires an authoritative task_contract"):
        _authoritative_task_contract(store)


class _FakeAgent:
    def __init__(self, **_: object) -> None:
        pass

    def diagnose(self, **values: object) -> CausalDiagnosis:
        cluster = values["cluster"]
        return CausalDiagnosis(
            diagnosis_id="diagnosis-1",
            cluster_id=cluster.cluster_id,  # type: ignore[union-attr]
            outcome="the rack remains closed",
            immediate_trigger="contact progress becomes stagnant",
            root_cause="contact is not maintained",
            contributing_causes=(),
            competing_hypotheses=(
                "contact is not maintained",
                "the planner selected the wrong direction",
            ),
            owner_layer="recovery",
            affected_component="rack contact recovery",
            earliest_divergence="first progress stall",
            supporting_evidence_ids=("segment-1",),
            counterevidence_ids=(),
            falsifier="paired recovery still stalls",
            distinguishing_check="rack moves after bounded replan",
            required_validation="paired same-seed live recovery",
            confidence=0.8,
        )

    def propose(self, **values: object) -> CandidateBundle:
        diagnosis = values["diagnosis"]
        critic = CriticRule(
            rule_id="stall",
            title="rack stall",
            feature="privileged.dishwasher.rack.position",
            operator="stagnant",
            threshold=0.01,
            dwell_steps=3,
            cooldown_steps=2,
            proposal="request a bounded VLA recovery",
            evidence_ids=("segment-1",),
        )
        return CandidateBundle(
            candidate_id="candidate-1",
            generation=int(values["generation"]),
            parent_sha256=values["parent_sha256"],  # type: ignore[arg-type]
            diagnosis_sha256=diagnosis.sha256,  # type: ignore[union-attr]
            causal_hypothesis=diagnosis.root_cause,  # type: ignore[union-attr]
            mechanism_change="request one bounded re-engagement",
            validation_plan="paired target-seed closed-loop gate",
            critic_rules=(critic,),
            recovery_rules=(
                RecoveryRule(
                    recovery_id="recover-1",
                    title="bounded replan",
                    trigger_rule_ids=(critic.rule_id,),
                    precondition="stall proposal is current",
                    steps=(
                        RecoveryStep(
                            tool=CONTACT_PUSH_TOOL,
                            parameters={},
                            stop_when="fresh observation is available",
                        ),
                    ),
                    safety_constraints=("actor remains sole writer",),
                    stop_condition="one bounded tool result",
                    fallback="terminate safely",
                    evidence_ids=("segment-1",),
                ),
            ),
        )


class _FakeVisualClusterAgent:
    seen_environment_name: str | None = None

    def __init__(self, **values: object) -> None:
        type(self).seen_environment_name = str(values["environment_name"])

    def review_clusters(self, **values: object) -> dict[str, object]:
        clusters = values["clusters"]
        cluster = clusters[0]  # type: ignore[index]
        return {
            "dominant_group_index": 0,
            "groups": [
                {
                    "member_segment_ids": list(cluster.member_segment_ids),
                    "representative_segment_ids": [cluster.medoid_segment_id],
                    "summary": "visually grounded common failure",
                }
            ],
            "visual_evidence": [],
        }


def test_multimodal_cluster_review_binds_manifest_environment(
    tmp_path: Path, monkeypatch
) -> None:
    root = tmp_path / "visual-cluster"
    store = CampaignStore(root)
    store.initialize(_manifest())
    store.record_episode(
        EpisodeRecord(
            episode_id="episode-visual",
            logical_id="g0000-rollout-000",
            generation=0,
            seed=7,
            policy_rng=70,
            bundle_sha256=None,
            status="valid",
            success=False,
            started_at="2026-01-01T00:00:00+00:00",
            finished_at="2026-01-01T00:00:01+00:00",
            elapsed_s=1.0,
            artifact_index={"overview": "overview.png"},
            failure_segment=FailureSegment(
                segment_id="segment-visual",
                episode_id="episode-visual",
                failure_class="stall",
                stage="engage",
                tool="robocasa.vla.groot",
                summary="rack did not move",
                earliest_divergence_step=2,
                start_step=1,
                end_step=3,
            ),
        )
    )
    deterministic = analyze_failures(root)
    artifact_index, aliases = _agent_artifact_context(store)
    monkeypatch.setattr(
        "rpent.evolution.lifecycle.CodexStageAgent", _FakeVisualClusterAgent
    )

    report = _materialize_multimodal_cluster_review(
        store=store,
        deterministic_report=deterministic,
        artifact_index=artifact_index,
        aliases=aliases,
        model="test",
    )

    assert report["dominant_cluster_id"].startswith("visual-cluster-")
    assert _FakeVisualClusterAgent.seen_environment_name == "robocasa"


def test_first_candidate_resumes_hashed_diagnosis_context(tmp_path: Path) -> None:
    root = tmp_path / "hashed-diagnosis"
    store = CampaignStore(root)
    store.initialize(_manifest())
    target = "a" * 64
    store.update_state(active_cluster_target_sha256=target)
    atomic_write_json(
        root
        / "agents"
        / "diagnosis"
        / target
        / "stage1-diagnosis"
        / "context.json",
        {
            "session_id": "campaign-session",
            "provider_thread_id": "diagnosis-thread",
        },
        overwrite=False,
    )
    atomic_write_json(
        root / "agents" / "diagnosis" / "stage1-diagnosis" / "context.json",
        {
            "session_id": "stale-session",
            "provider_thread_id": "stale-thread",
        },
        overwrite=False,
    )

    context = _latest_stage2_context(store)

    assert context["session_id"] == "campaign-session"
    assert context["provider_thread_id"] == "diagnosis-thread"


def test_latest_stage2_context_skips_unregistered_proposal_directory(
    tmp_path: Path,
) -> None:
    root = tmp_path / "stage2-directory-gap"
    store = CampaignStore(root)
    store.initialize(_manifest())
    store.transition(CampaignPhase.CLUSTER)
    store.transition(CampaignPhase.DIAGNOSE)
    store.transition(CampaignPhase.PROPOSE)
    diagnosis = _FakeAgent().diagnose(
        cluster=type("Cluster", (), {"cluster_id": "cluster-1"})()
    )
    registered = _FakeAgent().propose(
        diagnosis=diagnosis,
        generation=0,
        parent_sha256=None,
    )
    store.register_candidate(registered)

    rejected = registered.as_dict()
    rejected["candidate_id"] = "contract-rejected"
    for index, output in ((0, rejected), (1, registered.as_dict())):
        stage_root = root / "agents" / f"candidate-{index:03d}" / "stage2-proposal"
        attempt_root = stage_root / "attempt-000"
        atomic_write_json(stage_root / "output.json", output)
        atomic_write_json(attempt_root / "output.json", output)
        atomic_write_json(
            attempt_root / "invocation.json",
            {
                "session_id": "campaign-session",
                "provider_thread_id": f"provider-{index}",
            },
        )
        atomic_write_json(
            stage_root / "context.json",
            {
                "session_id": "campaign-session",
                "provider_thread_id": f"provider-{index}",
                "successful_attempt": "attempt-000",
                "output_sha256": canonical_sha256(output),
            },
        )

    context = _latest_stage2_context(store)

    assert context["provider_thread_id"] == "provider-1"


def test_formal_failure_clustering_fails_closed_without_visual_evidence(
    tmp_path: Path,
) -> None:
    root = tmp_path / "formal-missing-visual"
    store = CampaignStore(root)
    store.initialize(replace(_manifest(), protocol_explicit=True))
    store.record_episode(
        EpisodeRecord(
            episode_id="episode-failure",
            logical_id="g0000-rollout-000",
            generation=0,
            seed=7,
            policy_rng=70,
            bundle_sha256=None,
            status="valid",
            success=False,
            started_at="2026-01-01T00:00:00+00:00",
            finished_at="2026-01-01T00:00:01+00:00",
            elapsed_s=1.0,
            artifact_index={},
            failure_segment=FailureSegment(
                segment_id="segment-missing-visual",
                episode_id="episode-failure",
                failure_class="horizon_incomplete",
                stage="closed_loop_execution",
                tool=None,
                summary="task did not complete",
                earliest_divergence_step=10,
                start_step=2,
                end_step=10,
            ),
        )
    )
    analyze_failures(root)

    with pytest.raises(ValueError, match="requires synchronized visual artifacts"):
        run_diagnosis_stage(campaign_root=root, tool_catalog={"tools": []})

    assert CampaignStore(root).state()["phase"] == CampaignPhase.DIAGNOSE.value


def test_generation_with_no_failures_completes_without_agent_call(
    tmp_path: Path, monkeypatch
) -> None:
    root = tmp_path / "no-failures"
    store = CampaignStore(root)
    store.initialize(replace(_manifest(), protocol_explicit=True))
    store.record_episode(
        EpisodeRecord(
            episode_id="episode-success",
            logical_id="g0000-rollout-000",
            generation=0,
            seed=7,
            policy_rng=70,
            bundle_sha256=None,
            status="valid",
            success=True,
            started_at="2026-01-01T00:00:00+00:00",
            finished_at="2026-01-01T00:00:01+00:00",
            elapsed_s=1.0,
            artifact_index={},
        )
    )
    analyze_failures(root)
    monkeypatch.setattr(
        "rpent.evolution.lifecycle.CodexStageAgent",
        lambda **_: pytest.fail("no-failure generation must not invoke an agent"),
    )

    report = run_diagnosis_stage(campaign_root=root, tool_catalog={"tools": []})

    assert report["optimization_outcome"] == "no_failures_to_optimize"
    assert CampaignStore(root).state()["phase"] == CampaignPhase.COMPLETE.value


def test_inconclusive_diagnosis_skips_to_next_cluster_without_candidate(
    tmp_path: Path,
) -> None:
    root = tmp_path / "inconclusive"
    store = CampaignStore(root)
    store.initialize(_manifest())
    store.transition(CampaignPhase.CLUSTER)
    store.transition(CampaignPhase.DIAGNOSE)
    diagnosis = replace(
        _FakeAgent().diagnose(
            cluster=type("Cluster", (), {"cluster_id": "cluster-unresolved"})()
        ),
        root_cause="Inconclusive. The sampled trajectories do not share one cause.",
        owner_layer="unknown",
        confidence=0.25,
    )
    store.register_diagnosis(diagnosis)
    targets = [
        {
            "rank": 0,
            "cluster_id": "cluster-unresolved",
            "target_sha256": "a" * 64,
        },
        {
            "rank": 1,
            "cluster_id": "cluster-actionable",
            "target_sha256": "b" * 64,
        },
    ]

    report = _route_inconclusive_diagnosis(
        store=store,
        diagnosis=diagnosis,
        targets=targets,
        target_rank=0,
    )

    state = store.state()
    assert report["candidate_created"] is False
    assert report["next_cluster_id"] == "cluster-actionable"
    assert state["phase"] == CampaignPhase.DIAGNOSE.value
    assert state["active_cluster_rank"] == 1
    assert state["active_cluster_target_sha256"] == "b" * 64
    assert not store.candidate_ledger.records()


def test_inconclusive_diagnosis_can_defer_secondary_for_provisional(
    tmp_path: Path,
) -> None:
    root = tmp_path / "deferred-inconclusive"
    manifest = replace(
        _manifest(),
        runtime={
            "evolution_policy": {"defer_inconclusive_for_provisional": True}
        },
    )
    store = CampaignStore(root)
    store.initialize(manifest)
    store.transition(CampaignPhase.CLUSTER)
    store.transition(CampaignPhase.DIAGNOSE)
    diagnosis = replace(
        _FakeAgent().diagnose(
            cluster=type("Cluster", (), {"cluster_id": "cluster-unresolved"})()
        ),
        root_cause="Inconclusive. One leading hypothesis remains falsifiable.",
        owner_layer="unknown",
        confidence=0.1,
    )
    store.register_diagnosis(diagnosis)

    report = _route_inconclusive_diagnosis(
        store=store,
        diagnosis=diagnosis,
        targets=[
            {"rank": 0, "cluster_id": "cluster-unresolved"},
            {"rank": 1, "cluster_id": "cluster-secondary"},
        ],
        target_rank=0,
    )

    assert report["provisional_hypothesis_eligible"] is True
    assert report["secondary_cluster_deferred"] is True
    assert store.state()["phase"] == CampaignPhase.COMPLETE.value
    assert (
        store.state()["optimization_outcome"]
        == "no_actionable_cluster_diagnosis"
    )


def _terminal_provisional_campaign(
    tmp_path: Path,
    *,
    minimum_confidence: float | None = 0.5,
    diagnosis_confidence: float = 0.6,
) -> tuple[CampaignStore, CausalDiagnosis]:
    root = tmp_path / "provisional"
    evolution_policy = {"same_seed_pass_rate": 1.0}
    if minimum_confidence is not None:
        evolution_policy["provisional_min_diagnosis_confidence"] = minimum_confidence
    manifest = replace(
        _manifest(),
        rollout_seeds=(7, 9),
        expected_rollouts=2,
        policy_rng_by_seed={"7": 70, "9": 90, "8": 80},
        runtime={"evolution_policy": evolution_policy},
    )
    store = CampaignStore(root)
    store.initialize(manifest)
    episode_ids = []
    for index, (seed, policy_rng) in enumerate(((7, 70), (9, 90))):
        episode_id = f"failure-{index}"
        episode_ids.append(episode_id)
        store.record_episode(
            EpisodeRecord(
                episode_id=episode_id,
                logical_id=f"g0000-rollout-{index:03d}",
                generation=0,
                seed=seed,
                policy_rng=policy_rng,
                bundle_sha256=None,
                status="valid",
                success=False,
                started_at="2026-08-08T00:00:00+00:00",
                finished_at="2026-08-08T00:00:01+00:00",
                elapsed_s=1.0,
                artifact_index={},
            )
        )
    store.transition(CampaignPhase.CLUSTER)
    cluster_id = "wrong-receptacle"
    report = {
        "schema_version": 1,
        "manifest_sha256": manifest.sha256,
        "dominant_cluster_id": cluster_id,
        "failures_with_segments": 2,
        "clusters": [
            {
                "cluster_id": cluster_id,
                "episode_ids": episode_ids,
                "member_segment_ids": ["segment-a", "segment-b"],
                "representative_segment_ids": ["segment-a"],
                "medoid_segment_id": "segment-a",
                "hard_key": ["transport", "vla", "wrong-target"],
                "summary": "retained object is transported to the wrong receptacle",
                "prevalence": 1.0,
                "mean_severity": 0.8,
            }
        ],
    }
    atomic_write_json(root / "analysis" / "failure_clusters.json", report)
    target = materialize_cluster_targets(store, report)["targets"][0]
    store.transition(CampaignPhase.DIAGNOSE)
    evidence = tuple(
        {
            "content_id": "artifact-" + str(index + 1) * 64,
            "access_record_id": "visual-access-" + str(index + 4) * 64,
            "camera_views": ["agentview", "wrist"],
            "step_or_frame": "overview",
            "claim": "the retained object moves to the wrong receptacle",
        }
        for index in range(3)
    )
    diagnosis = replace(
        _FakeAgent().diagnose(
            cluster=type("Cluster", (), {"cluster_id": cluster_id})()
        ),
        diagnosis_id="diagnosis-provisional",
        root_cause=(
            "Inconclusive at the VLA versus controller boundary; semantic "
            "misgrounding is the leading falsifiable hypothesis."
        ),
        competing_hypotheses=(
            "Leading H1: VLA selects the wrong receptacle.",
            "Alternative H2: action realization drifts from the correct target.",
        ),
        visual_evidence=evidence,
        confidence=diagnosis_confidence,
    )
    store.register_diagnosis(diagnosis)
    store.update_state(
        active_cluster_id=cluster_id,
        active_cluster_rank=0,
        active_cluster_target_sha256=target["target_sha256"],
    )
    store.transition(
        CampaignPhase.COMPLETE,
        state_updates={"optimization_outcome": "no_actionable_cluster_diagnosis"},
    )
    return store, diagnosis


def test_provisional_hypothesis_recovery_is_bound_and_timeboxed(
    tmp_path: Path,
) -> None:
    store, diagnosis = _terminal_provisional_campaign(tmp_path)

    report = authorize_provisional_hypothesis(
        campaign_root=store.root,
        minimum_same_seed_successes=1,
        skip_regression=True,
        deadline="2026-08-08T15:50:00+08:00",
    )

    state = store.state()
    assert state["phase"] == CampaignPhase.PROPOSE.value
    assert state["provisional_diagnosis_sha256"] == diagnosis.sha256
    assert effective_same_seed_pass_rate(store) == 0.5
    assert report["authorization"]["diagnosis_remains_inconclusive"] is True
    authorization = next(
        (store.root / "analysis" / "provisional-hypothesis-authorizations").glob(
            "*.json"
        )
    )
    payload = read_json(authorization)
    assert payload["minimum_same_seed_successes"] == 1
    assert payload["target_failure_count"] == 2
    assert payload["skip_regression"] is True
    assert payload["minimum_diagnosis_confidence"] == 0.5
    assert len(store.episodes.records()) == 2
    assert len(store.diagnoses.records()) == 1

    with pytest.raises(ValueError, match="complete phase"):
        authorize_provisional_hypothesis(
            campaign_root=store.root,
            minimum_same_seed_successes=1,
            deadline="2026-08-08T15:50:00+08:00",
        )


def test_manifest_can_remove_provisional_confidence_threshold(tmp_path: Path) -> None:
    store, _ = _terminal_provisional_campaign(
        tmp_path,
        minimum_confidence=0.0,
        diagnosis_confidence=0.0,
    )

    report = authorize_provisional_hypothesis(
        campaign_root=store.root,
        minimum_same_seed_successes=1,
        deadline="2026-08-08T15:50:00+08:00",
    )

    assert report["authorization"]["minimum_diagnosis_confidence"] == 0.0
    assert report["authorization"]["observed_diagnosis_confidence"] == 0.0
    assert store.state()["phase"] == CampaignPhase.PROPOSE.value


def test_historical_manifest_has_no_implicit_provisional_confidence_gate(
    tmp_path: Path,
) -> None:
    store, diagnosis = _terminal_provisional_campaign(
        tmp_path,
        minimum_confidence=None,
        diagnosis_confidence=0.46,
    )
    report = authorize_provisional_hypothesis(
        campaign_root=store.root,
        minimum_same_seed_successes=1,
        deadline="2026-08-08T15:50:00+08:00",
    )
    assert report["authorization"]["minimum_diagnosis_confidence"] == 0.0
    assert report["authorization"]["diagnosis_confidence"] == 0.46
    assert report["authorization"]["confidence_is_not_an_authorization_gate"] is True
    assert report["authorization"]["hypothesis_status"].startswith(
        "leading_experiment_target"
    )
    assert store.diagnoses.records()[-1]["confidence"] == diagnosis.confidence


def test_shadow_override_can_resume_an_immutable_rejection(
    tmp_path: Path, monkeypatch
) -> None:
    store, diagnosis = _terminal_provisional_campaign(tmp_path)
    authorize_provisional_hypothesis(
        campaign_root=store.root,
        minimum_same_seed_successes=1,
        skip_regression=True,
        deadline="2026-08-10T00:00:00+00:00",
    )
    candidate = _FakeAgent().propose(
        diagnosis=diagnosis,
        generation=0,
        parent_sha256=None,
    )
    output_path = (
        store.root
        / "agents"
        / "candidate-000"
        / "stage2-proposal"
        / "attempt-000"
        / "output.json"
    )
    atomic_write_json(output_path, candidate.as_dict())
    shadow = {
        "schema_version": 1,
        "candidate_sha256": candidate.sha256,
        "parent_bundle_sha256": None,
        "target_count": 2,
        "target_detected": 2,
        "preflight_conclusive": True,
        "success_control_count": 2,
        "success_control_false_positives": 1,
        "success_control_false_positive_rate": 0.5,
        "preflight_disposition": "rejected_success_control_false_positive_rate",
        "live_gate_admission": {
            "schema_version": 1,
            "criterion": "success_control_false_positive_rate",
            "configured_max_rate": 0.0,
            "observed_rate": 0.5,
            "success_control_count": 2,
            "success_control_false_positives": 1,
            "threshold_source": "default_zero",
            "relaxed_override": False,
            "authorization_id": None,
            "eligible_for_live_gate": False,
            "disposition": "rejected_false_positive_rate_exceeded",
        },
    }
    shadow_path = (
        store.root
        / "analysis"
        / "candidate-shadow-replay"
        / f"{candidate.sha256}.json"
    )
    atomic_write_json(shadow_path, shadow)
    atomic_write_json(
        shadow_path.with_suffix(".precommit.json"),
        {
            "schema_version": 1,
            "candidate_sha256": candidate.sha256,
            "parent_bundle_sha256": None,
            "shadow_report_sha256": canonical_sha256(shadow),
            "target_trajectory_sha256": [],
        },
    )

    authorization = authorize_shadow_falsification(
        campaign_root=store.root,
        candidate_output=output_path,
        max_false_positive_rate=0.5,
        deadline="2026-08-10T00:00:00+00:00",
        reason="target recall is complete; permit one bounded falsification replay",
    )
    assert authorization["authorization"]["candidate_sha256"] == candidate.sha256
    monkeypatch.setattr("rpent.evolution.lifecycle.CodexStageAgent", _FakeAgent)
    result = run_proposal_stage(
        campaign_root=store.root,
        tool_catalog={"tools": [{"name": CONTACT_PUSH_TOOL}]},
    )
    assert result["authorized_shadow_replay"] is True
    assert store.state()["phase"] == CampaignPhase.SAME_SEED_GATE.value


def test_timeboxed_same_seed_pass_skips_regression_without_relabeling_diagnosis(
    tmp_path: Path,
) -> None:
    store, diagnosis = _terminal_provisional_campaign(tmp_path)
    authorize_provisional_hypothesis(
        campaign_root=store.root,
        minimum_same_seed_successes=1,
        skip_regression=True,
        deadline="2026-08-08T15:50:00+08:00",
    )
    candidate = _FakeAgent().propose(
        diagnosis=diagnosis,
        generation=0,
        parent_sha256=None,
    )
    candidate_sha256 = store.register_candidate(candidate)
    store.transition(CampaignPhase.SAME_SEED_GATE)
    state = record_gate_and_advance(
        campaign_root=store.root,
        decision=GateDecision(
            decision_id="provisional-same-seed-pass",
            candidate_sha256=candidate_sha256,
            parent_sha256=None,
            kind="same_seed",
            passed=True,
            conclusive=True,
            candidate_successes=1,
            parent_successes=0,
            paired_count=2,
            candidate_wins=1,
            parent_wins=0,
            p_value=None,
            alpha=None,
            candidate_safety_events=0,
            parent_safety_events=0,
            rationale="one target failure was rescued",
        ),
    )

    assert state["phase"] == CampaignPhase.HELDOUT_GATE.value
    assert state["optimization_outcome"] == (
        "same_seed_passed_regression_skipped_timeboxed"
    )
    assert store.diagnoses.records()[-1]["root_cause"].startswith("Inconclusive")


def test_same_seed_pass_can_skip_regression_and_enter_heldout_gate(
    tmp_path: Path,
) -> None:
    store = CampaignStore(tmp_path / "campaign")
    manifest = _manifest()
    store.initialize(
        replace(
            manifest,
            runtime={
                **manifest.runtime,
                "evolution_policy": {"skip_regression_gate": True},
            },
        )
    )
    store.transition(CampaignPhase.CLUSTER)
    store.transition(CampaignPhase.DIAGNOSE)
    diagnosis = _FakeAgent().diagnose(
        cluster=type("Cluster", (), {"cluster_id": "cluster-primary"})()
    )
    store.register_diagnosis(diagnosis)
    store.transition(CampaignPhase.PROPOSE)
    candidate = _FakeAgent().propose(
        diagnosis=diagnosis,
        generation=0,
        parent_sha256=None,
    )
    candidate_sha256 = store.register_candidate(candidate)
    store.transition(CampaignPhase.SAME_SEED_GATE)

    state = record_gate_and_advance(
        campaign_root=store.root,
        decision=GateDecision(
            decision_id="same-seed-direct-heldout",
            candidate_sha256=candidate_sha256,
            parent_sha256=None,
            kind="same_seed",
            passed=True,
            conclusive=True,
            candidate_successes=22,
            parent_successes=0,
            paired_count=43,
            candidate_wins=22,
            parent_wins=0,
            p_value=None,
            alpha=None,
            candidate_safety_events=0,
            parent_safety_events=0,
            rationale="same-seed threshold met",
        ),
    )

    assert state["phase"] == CampaignPhase.HELDOUT_GATE.value
    assert state["optimization_outcome"] == "same_seed_passed_regression_skipped"


def _decision(kind: str, candidate: str) -> GateDecision:
    return GateDecision(
        decision_id=f"decision-{kind}",
        candidate_sha256=candidate,
        parent_sha256=None,
        kind=kind,  # type: ignore[arg-type]
        passed=True,
        conclusive=True,
        candidate_successes=1,
        parent_successes=0,
        paired_count=1,
        candidate_wins=1,
        parent_wins=0,
        p_value=0.01,
        alpha=0.025,
        candidate_safety_events=0,
        parent_safety_events=0,
        rationale="paired gate passed",
    )


def test_full_post_rollout_lifecycle_is_append_only_and_completes(
    tmp_path: Path, monkeypatch
) -> None:
    root = tmp_path / "campaign"
    store = CampaignStore(root)
    store.initialize(_manifest())
    states = root / "artifacts" / "episode-1" / "states.jsonl"
    states.parent.mkdir(parents=True)
    states.write_text(
        '{"state":{"privileged.dishwasher.rack.position":0.25,'
        '"nested":{"progress":0.1},"policy_rng":70}}\n',
        encoding="utf-8",
    )
    store.record_episode(
        EpisodeRecord(
            episode_id="episode-1",
            logical_id="g0000-rollout-000",
            generation=0,
            seed=7,
            policy_rng=70,
            bundle_sha256=None,
            status="valid",
            success=False,
            started_at="2026-01-01T00:00:00+00:00",
            finished_at="2026-01-01T00:00:01+00:00",
            elapsed_s=1.0,
            artifact_index={"trajectory": "trajectory.jsonl", "states": str(states)},
            failure_segment=FailureSegment(
                segment_id="segment-1",
                episode_id="episode-1",
                failure_class="stall",
                stage="engage",
                tool="robocasa.vla.groot",
                summary="rack did not move",
                earliest_divergence_step=2,
                start_step=1,
                end_step=3,
            ),
        )
    )
    analyze_failures(root)
    assert _observed_critic_features(store) == (
        "nested.progress",
        "privileged.dishwasher.rack.position",
    )
    monkeypatch.setattr("rpent.evolution.lifecycle.CodexStageAgent", _FakeAgent)
    catalog = {"tools": [{"name": CONTACT_PUSH_TOOL}]}
    run_diagnosis_stage(campaign_root=root, tool_catalog=catalog)
    proposal = run_proposal_stage(campaign_root=root, tool_catalog=catalog)
    candidate = proposal["candidate_sha256"]

    assert len(CampaignStore(root).diagnoses.records()) == 1
    assert len(CampaignStore(root).candidate_ledger.records()) == 1
    for kind in ("same_seed", "regression", "heldout_10"):
        record_gate_and_advance(campaign_root=root, decision=_decision(kind, candidate))
    promotion = promote_and_complete(campaign_root=root)
    status = CampaignStore(root).status()
    assert promotion["candidate_sha256"] == candidate
    assert status["state"]["phase"] == CampaignPhase.COMPLETE.value
    assert status["state"]["current_bundle_sha256"] == candidate


def test_five_failed_candidates_switch_to_secondary_then_stop(tmp_path: Path) -> None:
    root = tmp_path / "bounded"
    store = CampaignStore(root)
    store.initialize(_manifest())
    store.transition(CampaignPhase.CLUSTER)
    report = {
        "schema_version": 1,
        "manifest_sha256": store.manifest().sha256,
        "dominant_cluster_id": "cluster-primary",
        "failures_with_segments": 2,
        "clusters": [
            {
                "cluster_id": "cluster-primary",
                "episode_ids": ["failure-a", "failure-b"],
                "member_segment_ids": ["segment-a", "segment-b"],
                "prevalence": 1.0,
                "mean_severity": 0.9,
            },
            {
                "cluster_id": "cluster-secondary",
                "episode_ids": ["failure-c"],
                "member_segment_ids": ["segment-c"],
                "prevalence": 0.5,
                "mean_severity": 0.8,
            },
        ],
    }
    atomic_write_json(
        root / "analysis" / "failure_clusters.json", report, overwrite=False
    )

    def diagnosis(cluster_id: str, suffix: str) -> CausalDiagnosis:
        return replace(
            _FakeAgent().diagnose(
                cluster=type("Cluster", (), {"cluster_id": cluster_id})()
            ),
            diagnosis_id=f"diagnosis-{suffix}",
            cluster_id=cluster_id,
        )

    def reject_round(active: CausalDiagnosis, index: int) -> None:
        candidate = replace(
            _FakeAgent().propose(
                diagnosis=active,
                generation=0,
                parent_sha256=None,
            ),
            candidate_id=f"candidate-{active.cluster_id}-{index}",
        )
        candidate_sha = store.register_candidate(candidate)
        store.transition(CampaignPhase.SAME_SEED_GATE)
        decision = GateDecision(
            decision_id=f"reject-{active.cluster_id}-{index}",
            candidate_sha256=candidate_sha,
            parent_sha256=None,
            kind="same_seed",
            passed=False,
            conclusive=True,
            candidate_successes=0,
            parent_successes=0,
            paired_count=1,
            candidate_wins=0,
            parent_wins=0,
            p_value=None,
            alpha=None,
            candidate_safety_events=0,
            parent_safety_events=0,
            rationale="candidate did not rescue the target failure",
        )
        record_gate_and_advance(campaign_root=root, decision=decision)

    store.transition(CampaignPhase.DIAGNOSE)
    primary = diagnosis("cluster-primary", "primary")
    store.register_diagnosis(primary)
    store.transition(CampaignPhase.PROPOSE)
    for index in range(5):
        reject_round(primary, index)
        if index < 4:
            assert store.state()["phase"] == CampaignPhase.PROPOSE.value
    assert store.state()["phase"] == CampaignPhase.DIAGNOSE.value
    assert store.state()["active_cluster_rank"] == 1

    secondary = diagnosis("cluster-secondary", "secondary")
    store.register_diagnosis(secondary)
    store.transition(CampaignPhase.PROPOSE)
    for index in range(5):
        reject_round(secondary, index)
    assert store.state()["phase"] == CampaignPhase.COMPLETE.value
    assert (
        store.state()["optimization_outcome"]
        == "no_candidate_passed_primary_or_secondary"
    )
