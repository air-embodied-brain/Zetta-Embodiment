# Copyright (c) 2026 RPent Contributors
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from robots.robocasa.slide_dishwasher_program import CONTACT_PUSH_TOOL
from rpent.evolution.campaign import analyze_failures
from rpent.evolution.gating import evaluate_paired_gate
from rpent.evolution.jsonio import (
    AppendOnlyLedger,
    atomic_write_json,
    canonical_sha256,
)
from rpent.evolution.lifecycle import (
    _agent_artifact_index,
    _bounded_gate_descriptors,
    _bounded_refinement_artifact_index,
    _candidate_feature_contract,
    _frozen_same_seed_pass_rate,
    _record_candidate_feature_contract_rejection,
    _recover_unadvanced_candidate,
    _rejected_candidates_for_cluster,
    _rejected_gate_refinement_context,
    _select_causal_isolation_directive,
    _select_development_calibration_directive,
    _shadow_live_gate_admission,
    record_gate_and_advance,
    reject_registered_noop_candidate,
    run_proposal_stage,
)
from rpent.evolution.models import (
    CampaignManifest,
    CampaignPhase,
    CandidateBundle,
    CausalDiagnosis,
    CriticPredicate,
    CriticRule,
    EpisodeRecord,
    FailureSegment,
    GateDecision,
    RecoveryRule,
    RecoveryStep,
)
from rpent.evolution.stages import (
    _validate_causal_isolation_candidate,
    blind_artifact_index,
)
from rpent.evolution.store import CampaignStore


def test_gate_evidence_replay_uses_the_frozen_schema_threshold() -> None:
    assert _frozen_same_seed_pass_rate({"schema_version": 1}) == 1.0
    assert (
        _frozen_same_seed_pass_rate(
            {"schema_version": 2, "same_seed_pass_rate": 0.5}
        )
        == 0.5
    )
    with pytest.raises(ValueError, match="invalid same-seed pass rate"):
        _frozen_same_seed_pass_rate(
            {"schema_version": 2, "same_seed_pass_rate": True}
        )


def test_shadow_success_control_false_positives_fail_closed_by_default(
    tmp_path: Path,
) -> None:
    root = tmp_path / "strict-shadow-admission"
    store = CampaignStore(root)
    store.initialize(_manifest())
    admission = _shadow_live_gate_admission(
        store,
        {
            "success_control_count": 8,
            "success_control_false_positives": 8,
            "success_control_false_positive_rate": 1.0,
            # Incomplete target divergence must not mask observed false positives.
            "preflight_conclusive": False,
        },
    )
    assert admission["configured_max_rate"] == 0.0
    assert admission["threshold_source"] == "default_zero"
    assert admission["eligible_for_live_gate"] is False


def test_shadow_success_control_relaxed_override_is_explicit_and_auditable(
    tmp_path: Path,
) -> None:
    root = tmp_path / "relaxed-shadow-admission"
    manifest = replace(
        _manifest(),
        runtime={
            "evolution_policy": {
                "shadow_success_control_max_false_positive_rate": 0.25,
            }
        },
    )
    store = CampaignStore(root)
    store.initialize(manifest)
    admission = _shadow_live_gate_admission(
        store,
        {
            "success_control_count": 8,
            "success_control_false_positives": 2,
            "success_control_false_positive_rate": 0.25,
        },
    )
    assert admission["eligible_for_live_gate"] is True
    assert admission["relaxed_override"] is True
    assert admission["threshold_source"] == "campaign_explicit_relaxed_override"
    assert admission["disposition"] == "accepted_under_relaxed_override"


def test_refinement_artifact_index_keeps_only_diagnosis_and_latest_gate() -> None:
    diagnosis = _diagnosis()
    diagnosis_ids = {
        *diagnosis.supporting_evidence_ids,
        *diagnosis.counterevidence_ids,
    }
    gate_id = "artifact-" + "9" * 64
    unrelated_id = "artifact-" + "8" * 64
    index = {
        "artifacts": [
            *(
                {"content_id": value, "type": "structured_data", "hash": value}
                for value in diagnosis_ids
            ),
            {"content_id": gate_id, "type": "candidate_gate_episode", "hash": gate_id},
            {"content_id": unrelated_id, "type": "video", "hash": unrelated_id},
        ],
        "relationships": [{"source": unrelated_id, "target": gate_id}],
    }
    bounded = _bounded_refinement_artifact_index(
        artifact_index=index,
        diagnosis=diagnosis,
        refinement_context={"gate_evidence": [{"content_id": gate_id}]},
    )
    assert {row["content_id"] for row in bounded["artifacts"]} == {
        *diagnosis_ids,
        gate_id,
    }
    assert bounded["relationships"] == []
    assert bounded["selection"] == {
        "mode": "active_diagnosis_and_latest_rejected_gate",
        "source_artifact_count": len(index["artifacts"]),
        "selected_artifact_count": len(diagnosis_ids) + 1,
    }


def test_gate_descriptor_prompt_slice_is_stratified_and_bounded() -> None:
    evidence = [
        {
            "content_id": f"artifact-{index:064x}",
            "type": artifact_type,
            "summary": summary,
            "hash": f"hmac-sha256:{index:064x}",
        }
        for artifact_type, summary in (
            ("candidate_gate_episode", "candidate arm"),
            ("parent_gate_episode", "parent arm"),
            ("structured_data", "telemetry"),
        )
        for index in range(10)
    ]
    selected = _bounded_gate_descriptors(evidence, per_descriptor_kind=2)
    assert len(selected) == 6
    assert {
        (row["type"], row["summary"]) for row in selected
    } == {
        ("candidate_gate_episode", "candidate arm"),
        ("parent_gate_episode", "parent arm"),
        ("structured_data", "telemetry"),
    }


def test_gate_descriptor_prompt_slice_caps_unique_summaries_per_type() -> None:
    evidence = [
        {
            "content_id": f"artifact-{index:064x}",
            "type": "structured_data",
            "summary": f"unique summary {index}",
            "hash": f"hmac-sha256:{index:064x}",
        }
        for index in range(100)
    ]
    selected = _bounded_gate_descriptors(evidence, per_descriptor_kind=2)
    assert len(selected) == 2
    assert len({row["summary"] for row in selected}) == 2


def test_causal_isolation_prefers_proven_trigger_and_underexposed_recovery() -> None:
    best_critic = {"rule_id": "critic-proven", "evidence_ids": []}
    recovery_steps = [{"tool": "pi0_pick", "parameters": {}, "stop_when": "done"}]
    directive = _select_causal_isolation_directive(
        current={
            "successful_candidate_interventions": 0,
            "candidate_interventions": 47,
        },
        history=[
            {
                "critic_rules": [best_critic],
                "recovery_rules": [{"steps": [{"tool": "set_gripper"}]}],
                "candidate_interventions": 47,
                "successful_candidate_interventions": 22,
            },
            {
                "critic_rules": [{"rule_id": "critic-underexposed"}],
                "recovery_rules": [{"steps": recovery_steps}],
                "candidate_interventions": 6,
                "successful_candidate_interventions": 0,
            },
        ],
    )
    assert directive is not None
    assert directive["preserve_critic_rules_byte_for_byte"] == [best_critic]
    assert directive["reuse_recovery_steps_byte_for_byte"] == recovery_steps

    assert (
        _select_causal_isolation_directive(
            current={
                "critic_rules": [best_critic],
                "recovery_rules": [{"steps": recovery_steps}],
                "successful_candidate_interventions": 0,
                "candidate_interventions": 47,
            },
            history=[
                {
                    "critic_rules": [best_critic],
                    "recovery_rules": [{"steps": [{"tool": "set_gripper"}]}],
                    "candidate_interventions": 47,
                    "successful_candidate_interventions": 22,
                },
                {
                    "critic_rules": [{"rule_id": "critic-underexposed"}],
                    "recovery_rules": [{"steps": recovery_steps}],
                    "candidate_interventions": 6,
                    "successful_candidate_interventions": 0,
                },
            ],
        )
        is None
    )


def test_replicated_development_calibration_preserves_the_proven_trigger() -> None:
    critic = {"rule_id": "critic-proven", "evidence_ids": []}
    parameters = {
        "grasp_offset_xyz": [-0.00945, 0.01885, 0.12536],
        "max_steps_per_move": 70,
        "grasp_confirm_steps": 4,
    }
    evidence = [
        {
            "official_success": True,
            "status": "success",
            "tool": "privileged_pick_place",
            "tool_parameters": parameters,
        }
        for _ in range(2)
    ]
    directive = _select_development_calibration_directive(
        history=[
            {
                "critic_rules": [critic],
                "candidate_interventions": 24,
                "successful_candidate_interventions": 22,
            }
        ],
        development_evidence=evidence,
    )
    assert directive is not None
    assert directive["preserve_critic_rules_byte_for_byte"] == [critic]
    assert directive["required_recovery_tool"] == "privileged_pick_place"
    assert directive["required_recovery_parameters"] == parameters
    assert directive["development_success_support"] == 2


def test_stage2_calibration_binding_requires_all_replicated_parameters() -> None:
    candidate = _candidate(candidate_id="calibrated", diagnosis=_diagnosis())
    base_step = candidate.recovery_rules[0].steps[0]
    incomplete_step = replace(
        base_step,
        tool="privileged_pick_place",
        parameters={"grasp_confirm_steps": 4},
    )
    incomplete = replace(
        candidate,
        recovery_rules=(
            replace(candidate.recovery_rules[0], steps=(incomplete_step,)),
        ),
    )
    parameters = {
        "grasp_offset_xyz": [-0.00945, 0.01885, 0.12536],
        "max_steps_per_move": 70,
        "grasp_confirm_steps": 4,
    }
    directive = {
        "preserve_critic_rules_byte_for_byte": [
            blind_artifact_index(candidate.critic_rules[0].as_dict())
        ],
        "required_recovery_tool": "privileged_pick_place",
        "required_recovery_parameters": parameters,
    }
    with pytest.raises(ValueError, match="replicated development parameters"):
        _validate_causal_isolation_candidate(incomplete, directive)

    calibrated_step = replace(incomplete_step, parameters=parameters)
    calibrated = replace(
        candidate,
        recovery_rules=(
            replace(candidate.recovery_rules[0], steps=(calibrated_step,)),
        ),
    )
    _validate_causal_isolation_candidate(calibrated, directive)


def _manifest() -> CampaignManifest:
    return CampaignManifest(
        campaign_id="refinement",
        environment="robocasa",
        task="SlideDishwasherRack",
        generation=0,
        code_commit="1" * 40,
        prompt_sha256="2" * 64,
        model="test",
        tool_catalog_sha256="3" * 64,
        rollout_seeds=(7,),
        heldout_seeds=(8,),
        policy_rng_by_seed={"7": 271828, "8": 314159},
        expected_rollouts=1,
        expected_heldout=1,
    )


def _diagnosis() -> CausalDiagnosis:
    return CausalDiagnosis(
        diagnosis_id="diagnosis-stall",
        cluster_id="cluster-stall",
        outcome="rack remains closed",
        immediate_trigger="contact produces no progress",
        root_cause="contact direction does not advance the rack",
        contributing_causes=(),
        competing_hypotheses=(
            "contact direction does not advance the rack",
            "the proposal is interrupted too early",
        ),
        owner_layer="recovery",
        affected_component="contact recovery",
        earliest_divergence="first contact chunk",
        supporting_evidence_ids=("segment-stall",),
        counterevidence_ids=(),
        falsifier="paired intervention advances the rack",
        distinguishing_check="compare progress after contact",
        required_validation="paired live target-failure gate",
        confidence=0.8,
    )


def _candidate(*, candidate_id: str, diagnosis: CausalDiagnosis) -> CandidateBundle:
    critic = CriticRule(
        rule_id=f"critic-{candidate_id}",
        title="detect progress stall",
        feature="privileged.dishwasher.rack.position",
        operator="stagnant",
        threshold=0.01,
        dwell_steps=2,
        cooldown_steps=1,
        proposal="request one bounded contact recovery",
        evidence_ids=("segment-stall",),
    )
    return CandidateBundle(
        candidate_id=candidate_id,
        generation=0,
        parent_sha256=None,
        diagnosis_sha256=diagnosis.sha256,
        causal_hypothesis=diagnosis.root_cause,
        mechanism_change=f"bounded recovery revision {candidate_id}",
        validation_plan="paired live target-failure gate",
        critic_rules=(critic,),
        recovery_rules=(
            RecoveryRule(
                recovery_id=f"recovery-{candidate_id}",
                title="bounded contact recovery",
                trigger_rule_ids=(critic.rule_id,),
                precondition="the critic proposal is current",
                steps=(
                    RecoveryStep(
                        tool=CONTACT_PUSH_TOOL,
                        parameters={"max_actions": 8},
                        stop_when="progress changes",
                    ),
                ),
                safety_constraints=("Actor remains the only writer",),
                stop_condition="one bounded attempt completes",
                fallback="stop safely",
                evidence_ids=("segment-stall",),
            ),
        ),
    )


def _decision(
    *, kind: str, candidate: CandidateBundle, passed: bool, suffix: str
) -> GateDecision:
    return GateDecision(
        decision_id=f"gate-{kind}-{suffix}",
        candidate_sha256=candidate.sha256,
        parent_sha256=candidate.parent_sha256,
        kind=kind,
        passed=passed,
        conclusive=True,
        candidate_successes=int(passed),
        parent_successes=0,
        paired_count=1,
        candidate_wins=int(passed),
        parent_wins=int(not passed),
        p_value=None,
        alpha=None,
        candidate_safety_events=0,
        parent_safety_events=0,
        rationale="bounded gate fixture",
    )


@pytest.mark.parametrize("failed_kind", ["regression", "heldout_50"])
def test_candidate_round_bound_counts_late_gate_rejections(
    tmp_path: Path, failed_kind: str
) -> None:
    root = tmp_path / failed_kind
    manifest = replace(
        _manifest(),
        runtime={
            "evolution_policy": {
                "max_candidate_rounds_per_cluster": 1,
                "maximum_target_clusters": 1,
            }
        },
    )
    store = CampaignStore(root)
    store.initialize(manifest)
    store.record_episode(
        EpisodeRecord(
            episode_id="rollout-failure",
            logical_id="g0000-rollout-000",
            generation=0,
            seed=7,
            policy_rng=271828,
            bundle_sha256=None,
            status="valid",
            success=False,
            started_at="2026-01-01T00:00:00+00:00",
            finished_at="2026-01-01T00:00:01+00:00",
            elapsed_s=1.0,
            artifact_index={},
            failure_segment=FailureSegment(
                segment_id="segment-stall",
                episode_id="rollout-failure",
                failure_class="stall",
                stage="contact",
                tool=CONTACT_PUSH_TOOL,
                summary="contact made no progress",
                earliest_divergence_step=1,
                start_step=0,
                end_step=2,
            ),
        )
    )
    cluster_report = analyze_failures(root)
    store.transition(CampaignPhase.DIAGNOSE)
    diagnosis = replace(
        _diagnosis(), cluster_id=str(cluster_report["clusters"][0]["cluster_id"])
    )
    store.register_diagnosis(diagnosis)
    store.transition(CampaignPhase.PROPOSE)
    candidate = _candidate(candidate_id="candidate-late-reject", diagnosis=diagnosis)
    store.register_candidate(candidate)
    store.transition(CampaignPhase.SAME_SEED_GATE)

    record_gate_and_advance(
        campaign_root=root,
        decision=_decision(
            kind="same_seed", candidate=candidate, passed=True, suffix=failed_kind
        ),
    )
    if failed_kind == "heldout_50":
        record_gate_and_advance(
            campaign_root=root,
            decision=_decision(
                kind="regression", candidate=candidate, passed=True, suffix="pass"
            ),
        )
    record_gate_and_advance(
        campaign_root=root,
        decision=_decision(
            kind=failed_kind, candidate=candidate, passed=False, suffix="reject"
        ),
    )

    state = CampaignStore(root).state()
    assert state["phase"] == CampaignPhase.COMPLETE
    assert state["optimization_outcome"] == "no_candidate_passed_primary_or_secondary"
    assert state["candidate_round"] == 1


def _gate_record(
    *, logical_id: str, bundle_sha256: str | None, states: Path, action_hash: str
) -> EpisodeRecord:
    return EpisodeRecord(
        episode_id=f"episode-{logical_id}",
        logical_id=logical_id,
        generation=0,
        seed=7,
        policy_rng=271828,
        bundle_sha256=bundle_sha256,
        status="valid",
        success=False,
        started_at="2026-01-01T00:00:00+00:00",
        finished_at="2026-01-01T00:00:01+00:00",
        elapsed_s=1.0,
        artifact_index={
            "states": str(states),
            "initial_observation_identity": {"reset": "paired-state"},
            "trajectory_index": {
                "artifact_sha256": {"actions": action_hash},
            },
        },
    )


def test_candidate_feature_contract_rejects_split_reset_action_features(
    tmp_path: Path,
) -> None:
    diagnosis = _diagnosis()
    base = _candidate(candidate_id="candidate-split", diagnosis=diagnosis)
    critic = replace(
        base.critic_rules[0],
        feature="command.realization.eef_motion_m",
        activation_conditions=(
            CriticPredicate(
                feature="privileged.task.manipulated_object.grasped",
                operator="eq",
                threshold=False,
            ),
        ),
    )
    candidate = replace(base, critic_rules=(critic,))
    states = tmp_path / "split-states.jsonl"
    states.write_text(
        '{"step_index":0,"state":'
        '{"privileged.task.manipulated_object.grasped":false}}\n'
        '{"step_index":1,"state":'
        '{"command.realization.eef_motion_m":0.0}}\n',
        encoding="utf-8",
    )
    record = _gate_record(
        logical_id="split",
        bundle_sha256=None,
        states=states,
        action_hash="a" * 64,
    )

    contract = _candidate_feature_contract(
        candidate=candidate,
        parent_bundle=None,
        trajectories=((record, states),),
    )

    assert contract["eligible"] is False
    assert contract["unsupported_feature_names"] == []
    assert contract["trajectory_reports"][0]["unavailable_rule_ids"] == [
        critic.rule_id
    ]

    states.write_text(
        '{"step_index":1,"state":'
        '{"command.realization.eef_motion_m":0.0,'
        '"privileged.task.manipulated_object.grasped":false}}\n',
        encoding="utf-8",
    )
    contract = _candidate_feature_contract(
        candidate=candidate,
        parent_bundle=None,
        trajectories=((record, states),),
    )
    assert contract["eligible"] is True

    states.write_text(
        '{"step_index":0,"state":'
        '{"command.realization.eef_motion_m":0.0,'
        '"privileged.task.manipulated_object.grasped":false}}\n'
        '{"step_index":1,"state":'
        '{"command.realization.eef_motion_m":0.0}}\n',
        encoding="utf-8",
    )
    contract = _candidate_feature_contract(
        candidate=candidate,
        parent_bundle=None,
        trajectories=((record, states),),
    )
    assert contract["eligible"] is False
    assert contract["trajectory_reports"][0]["unavailable_rule_ids"] == [
        critic.rule_id
    ]


def test_feature_contract_rejection_is_append_only_and_counts_candidate_round(
    tmp_path: Path,
) -> None:
    root = tmp_path / "contract-rejection"
    store = CampaignStore(root)
    store.initialize(_manifest())
    store.transition(CampaignPhase.CLUSTER)
    store.transition(CampaignPhase.DIAGNOSE)
    diagnosis = _diagnosis()
    store.register_diagnosis(diagnosis)
    store.transition(CampaignPhase.PROPOSE)
    candidate = _candidate(candidate_id="candidate-contract", diagnosis=diagnosis)
    output = (
        root
        / "agents"
        / "candidate-000"
        / "stage2-proposal"
        / "attempt-000"
        / "output.json"
    )
    atomic_write_json(output, candidate.as_dict())
    contract = {
        "schema_version": 1,
        "eligible": False,
        "unsupported_feature_names": ["privileged.missing"],
        "trajectory_reports": [],
    }

    first = _record_candidate_feature_contract_rejection(
        store=store,
        candidate=candidate,
        candidate_output=output,
        contract=contract,
        reason="unsupported feature",
    )
    second = _record_candidate_feature_contract_rejection(
        store=store,
        candidate=candidate,
        candidate_output=output,
        contract=contract,
        reason="unsupported feature",
    )

    assert first == second
    assert first["rejection"]["preflight_disposition"] == (
        "rejected_trajectory_feature_contract"
    )
    assert _rejected_candidates_for_cluster(store, diagnosis.cluster_id) == (
        candidate.sha256,
    )
    assert store.candidate_ledger.records() == []


def test_registered_noop_candidate_is_rejected_before_live_gate(
    tmp_path: Path,
) -> None:
    root = tmp_path / "registered-noop"
    store = CampaignStore(root)
    store.initialize(_manifest())
    store.transition(CampaignPhase.CLUSTER)
    store.transition(CampaignPhase.DIAGNOSE)
    diagnosis = _diagnosis()
    store.register_diagnosis(diagnosis)
    store.transition(CampaignPhase.PROPOSE)

    reference_base = _candidate(candidate_id="candidate-reference", diagnosis=diagnosis)
    reference_step = replace(
        reference_base.recovery_rules[0].steps[0], parameters={}
    )
    reference = replace(
        reference_base,
        recovery_rules=(
            replace(
                reference_base.recovery_rules[0], steps=(reference_step,)
            ),
        ),
    )
    store.register_candidate(reference)
    store.transition(CampaignPhase.SAME_SEED_GATE)
    record_gate_and_advance(
        campaign_root=root,
        decision=_decision(
            kind="same_seed", candidate=reference, passed=False, suffix="reference"
        ),
    )

    candidate_step = replace(reference_step, parameters={"max_actions": 8})
    candidate = replace(
        reference,
        candidate_id="candidate-explicit-default",
        mechanism_change="explicitly set the existing tool default",
        recovery_rules=(
            replace(
                reference.recovery_rules[0],
                recovery_id="recovery-explicit-default",
                steps=(candidate_step,),
            ),
        ),
    )
    store.register_candidate(candidate)
    store.transition(CampaignPhase.SAME_SEED_GATE)

    result = reject_registered_noop_candidate(
        campaign_root=root,
        candidate_sha256=candidate.sha256,
        equivalent_to_candidate_sha256=reference.sha256,
        tool_parameter_defaults={CONTACT_PUSH_TOOL: {"max_actions": 8}},
        reason="explicit max_actions=8 equals the frozen tool default",
    )

    assert result["rejection"]["normalized_recovery_steps"] == [
        {"tool": CONTACT_PUSH_TOOL, "parameters": {"max_actions": 8}}
    ]
    assert result["state"]["phase"] == CampaignPhase.PROPOSE
    assert result["state"]["candidate_sha256"] is None
    assert _recover_unadvanced_candidate(CampaignStore(root)) is None
    assert set(_rejected_candidates_for_cluster(store, diagnosis.cluster_id)) == {
        reference.sha256,
        candidate.sha256,
    }
    atomic_write_json(
        root / "analysis" / "development-evidence" / "zero-offset.json",
        {
            "task": 9,
            "seed": 18156,
            "grasp_offset_xyz": [0.0, 0.0, 0.0],
            "manipulated_object": "cream_cheese_1",
            "target": "wine_rack_1_top_region",
            "official_success": True,
            "result": {
                "status": "success",
                "close": {
                    "contact_seen": True,
                    "grasp_verified": True,
                    "retention_confirmation_steps": 4,
                },
            },
        },
    )
    context = _rejected_gate_refinement_context(
        CampaignStore(root), artifact_index={"artifacts": []}
    )
    assert context is not None
    assert context["mode"] == "refine_operator_rejected_noop_candidate"
    assert context["preflight_rejection"]["rejection_id"] == result["rejection"][
        "rejection_id"
    ]
    assert context["development_evidence"] == [
        {
            "evidence_kind": "development_privileged_tool_smoke",
            "artifact_sha256": context["development_evidence"][0][
                "artifact_sha256"
            ],
            "task_id": 9,
            "grasp_offset_xyz": [0.0, 0.0, 0.0],
            "manipulated_object": "cream_cheese_1",
            "target": "wine_rack_1_top_region",
            "status": "success",
            "official_success": True,
            "contact_seen": True,
            "grasp_verified": True,
            "retention_confirmation_steps": 4,
        }
    ]


class _CapturingRefinementAgent:
    init_values: dict[str, Any] = {}
    proposal_values: dict[str, Any] = {}

    def __init__(self, **values: Any) -> None:
        type(self).init_values = values

    def propose(self, **values: Any) -> CandidateBundle:
        type(self).proposal_values = values
        return _candidate(candidate_id="candidate-001", diagnosis=values["diagnosis"])


class _UnexpectedAgent:
    def __init__(self, **_: Any) -> None:
        raise AssertionError("provider must not run while recovering candidate commit")


def test_v3_style_shadow_false_positives_are_artifacted_before_live_gate(
    tmp_path: Path, monkeypatch
) -> None:
    root = tmp_path / "campaign-v3-shadow-rejection"
    store = CampaignStore(root)
    store.initialize(_manifest())
    states = root / "source" / "states.jsonl"
    states.parent.mkdir(parents=True)
    states.write_text(
        '{"step_index":0,"state":{"privileged.dishwasher.rack.position":0.0}}\n',
        encoding="utf-8",
    )
    store.record_episode(
        EpisodeRecord(
            episode_id="rollout-failure",
            logical_id="g0000-rollout-000",
            generation=0,
            seed=7,
            policy_rng=271828,
            bundle_sha256=None,
            status="valid",
            success=False,
            started_at="2026-01-01T00:00:00+00:00",
            finished_at="2026-01-01T00:00:01+00:00",
            elapsed_s=1.0,
            artifact_index={"states": str(states)},
            failure_segment=FailureSegment(
                segment_id="segment-stall",
                episode_id="rollout-failure",
                failure_class="stall",
                stage="contact",
                tool=CONTACT_PUSH_TOOL,
                summary="contact made no progress",
                earliest_divergence_step=None,
                start_step=0,
                end_step=1,
            ),
        )
    )
    clusters = analyze_failures(root)
    store.transition(CampaignPhase.DIAGNOSE)
    diagnosis = replace(
        _diagnosis(), cluster_id=str(clusters["clusters"][0]["cluster_id"])
    )
    store.register_diagnosis(diagnosis)
    store.transition(CampaignPhase.PROPOSE)
    monkeypatch.setattr(
        "rpent.evolution.lifecycle.CodexStageAgent", _CapturingRefinementAgent
    )

    def v3_shadow(**values: Any) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "candidate_sha256": values["candidate"].sha256,
            "parent_bundle_sha256": None,
            "target_count": 7,
            "target_detected": 6,
            "target_triggered_anywhere": 6,
            "success_control_count": 8,
            "success_control_false_positives": 8,
            "success_control_false_positive_rate": 1.0,
            "preflight_conclusive": False,
            "passed_detection_preflight": False,
            "outcomes": [
                {
                    "role": "target_failure",
                    "trajectory_sha256": "4" * 64,
                }
            ],
            "report_sha256": "5" * 64,
        }

    monkeypatch.setattr("rpent.evolution.lifecycle.evaluate_shadow_replay", v3_shadow)
    expected_candidate = _candidate(candidate_id="candidate-001", diagnosis=diagnosis)

    with pytest.raises(ValueError, match="false-positive rate 1 exceeds"):
        run_proposal_stage(
            campaign_root=root,
            tool_catalog={"tools": [{"name": CONTACT_PUSH_TOOL}]},
        )

    latest = CampaignStore(root)
    assert latest.state()["phase"] == CampaignPhase.PROPOSE.value
    assert latest.candidate_ledger.records() == []
    shadow_path = (
        root
        / "analysis"
        / "candidate-shadow-replay"
        / f"{expected_candidate.sha256}.json"
    )
    shadow = json.loads(shadow_path.read_text(encoding="utf-8"))
    assert shadow["live_gate_admission"]["eligible_for_live_gate"] is False
    assert shadow["live_gate_admission"]["threshold_source"] == "default_zero"
    assert shadow["preflight_disposition"] == (
        "rejected_success_control_false_positive_rate"
    )


def test_rejected_gate_evidence_is_opaque_and_refinement_reconstructs_fresh_thread(
    tmp_path: Path, monkeypatch
) -> None:
    root = tmp_path / "campaign"
    store = CampaignStore(root)
    store.initialize(_manifest())
    rollout_states = root / "artifacts" / "rollout" / "states.jsonl"
    rollout_states.parent.mkdir(parents=True)
    rollout_states.write_text(
        '{"state":{"privileged.dishwasher.rack.position":0.0}}\n',
        encoding="utf-8",
    )
    store.record_episode(
        EpisodeRecord(
            episode_id="rollout-private-episode",
            logical_id="rollout-private-logical",
            generation=0,
            seed=7,
            policy_rng=271828,
            bundle_sha256=None,
            status="valid",
            success=False,
            started_at="2026-01-01T00:00:00+00:00",
            finished_at="2026-01-01T00:00:01+00:00",
            elapsed_s=1.0,
            artifact_index={"states": str(rollout_states)},
            failure_segment=FailureSegment(
                segment_id="segment-stall",
                episode_id="rollout-private-episode",
                failure_class="stall",
                stage="contact",
                tool=CONTACT_PUSH_TOOL,
                summary="contact made no progress",
                earliest_divergence_step=1,
                start_step=0,
                end_step=2,
            ),
        )
    )
    analyze_failures(root)
    store.transition(CampaignPhase.DIAGNOSE)
    diagnosis = _diagnosis()
    store.register_diagnosis(diagnosis)
    store.transition(CampaignPhase.PROPOSE)
    candidate = _candidate(candidate_id="candidate-000", diagnosis=diagnosis)
    candidate_sha256 = store.register_candidate(candidate)
    store.transition(CampaignPhase.SAME_SEED_GATE)

    atomic_write_json(
        root / "agents" / "diagnosis" / "stage1-diagnosis" / "context.json",
        {"session_id": "session-after-gate", "provider_thread_id": "thread-stage1"},
    )
    stage2_output = {
        key: candidate.as_dict().get(key)
        for key in (
            "candidate_id",
            "mechanism_change",
            "validation_plan",
            "critic_rules",
            "recovery_rules",
            "tool_plugin",
        )
    }
    atomic_write_json(
        root / "agents" / "candidate-000" / "stage2-proposal" / "context.json",
        {
            "session_id": "session-1",
            "provider_thread_id": "thread-stage2-latest",
            "reconstructed": False,
            "successful_attempt": "attempt-000",
            "output_sha256": canonical_sha256(stage2_output),
        },
    )
    atomic_write_json(
        root / "agents" / "candidate-000" / "stage2-proposal" / "output.json",
        stage2_output,
    )
    atomic_write_json(
        root
        / "agents"
        / "candidate-000"
        / "stage2-proposal"
        / "attempt-000"
        / "output.json",
        stage2_output,
    )
    atomic_write_json(
        root
        / "agents"
        / "candidate-000"
        / "stage2-proposal"
        / "attempt-000"
        / "invocation.json",
        {
            "session_id": "session-1",
            "provider_thread_id": "thread-stage2-latest",
        },
    )

    gate_root = root / "candidates" / candidate_sha256 / "gates" / "same_seed"
    valid = AppendOnlyLedger(gate_root / "ledgers" / "valid.jsonl", key="logical_id")
    attempts = AppendOnlyLedger(
        gate_root / "ledgers" / "attempts.jsonl", key="attempt_id"
    )
    gate_records: dict[str, EpisodeRecord] = {}
    for arm, bundle in (("parent", None), ("candidate", candidate_sha256)):
        states = gate_root / "private-seed-7" / arm / "states.jsonl"
        states.parent.mkdir(parents=True, exist_ok=True)
        states.write_text(
            '{"state":{"privileged.dishwasher.rack.position":0.0,'
            '"policy_rng":271828}}\n',
            encoding="utf-8",
        )
        record = _gate_record(
            logical_id=f"private-pair-{arm}",
            bundle_sha256=bundle,
            states=states,
            action_hash=("a" * 64 if arm == "parent" else "b" * 64),
        )
        valid.append(record.as_dict())
        attempts.append({"attempt_id": record.attempt_id, **record.as_dict()})
        atomic_write_json(
            gate_root / "episodes" / record.logical_id / "record.json",
            record.as_dict(),
        )
        gate_records[arm] = record

    atomic_write_json(
        gate_root / "plan.json",
        {
            "schema_version": 1,
            "kind": "same_seed",
            "manifest_sha256": store.manifest().sha256,
            "generation": 0,
            "candidate_sha256": candidate_sha256,
            "parent_sha256": None,
            "cluster_id": "cluster-stall",
            "representative_segment_ids": ["segment-stall"],
            "pairs": [
                {
                    "pair_index": 0,
                    "source_episode_id": "rollout-private-episode",
                    "source_segment_ids": ["segment-stall"],
                    "seed": 7,
                    "policy_rng": 271828,
                    "logical_ids": {
                        "parent": "private-pair-parent",
                        "candidate": "private-pair-candidate",
                    },
                }
            ],
        },
    )

    decision = evaluate_paired_gate(
        kind="same_seed",
        candidate_sha256=candidate_sha256,
        parent_sha256=None,
        candidate_records=[gate_records["candidate"]],
        parent_records=[gate_records["parent"]],
        expected_seeds=(7,),
    )
    assert decision.passed is False
    record_gate_and_advance(
        campaign_root=root,
        decision=decision,
    )

    shadow = {
        "schema_version": 1,
        "candidate_sha256": candidate_sha256,
        "target_count": 2,
        "target_triggered_anywhere": 1,
        "success_control_count": 3,
        "success_control_false_positives": 0,
        "outcomes": [
            {
                "role": "target_failure",
                "triggered": True,
                "first_trigger_step": 91,
            },
            {
                "role": "target_failure",
                "triggered": False,
                "first_trigger_step": None,
            },
        ],
    }
    shadow_root = root / "analysis" / "candidate-shadow-replay"
    shadow_path = shadow_root / f"{candidate_sha256}.json"
    atomic_write_json(shadow_path, shadow, overwrite=False)
    atomic_write_json(
        shadow_path.with_suffix(".precommit.json"),
        {
            "schema_version": 1,
            "candidate_sha256": candidate_sha256,
            "parent_bundle_sha256": None,
            "shadow_report_sha256": canonical_sha256(shadow),
            "target_trajectory_sha256": [],
        },
        overwrite=False,
    )

    index = _agent_artifact_index(CampaignStore(root))
    gate_types = {
        row["type"]
        for row in index["artifacts"]
        if row["type"] in {"candidate_gate_episode", "parent_gate_episode"}
    }
    assert gate_types == {"candidate_gate_episode", "parent_gate_episode"}
    visible = json.dumps(index, ensure_ascii=False).lower()
    for forbidden in (
        "271828",
        "private-seed-7",
        "private-pair-parent",
        "private-pair-candidate",
    ):
        assert forbidden not in visible

    refinement = _rejected_gate_refinement_context(
        CampaignStore(root), artifact_index=index
    )
    assert refinement is not None
    assert refinement["paired_gate_result"]["passed"] is False
    assert refinement["gate_evidence"]
    assert refinement["previous_detector_replay"] == {
        "target_count": 2,
        "target_triggered_anywhere": 1,
        "success_control_count": 3,
        "success_control_false_positives": 0,
        "first_target_trigger_step_min": 91,
        "first_target_trigger_step_max": 91,
        "interpretation": (
            "Detection-only replay. Low target coverage or late first triggers "
            "must be corrected before adding more activation guards."
        ),
    }
    serialized = json.dumps(refinement, ensure_ascii=False).lower()
    assert "271828" not in serialized
    assert "private-pair" not in serialized

    monkeypatch.setattr(
        "rpent.evolution.lifecycle.CodexStageAgent", _CapturingRefinementAgent
    )
    result = run_proposal_stage(
        campaign_root=root,
        tool_catalog={"tools": [{"name": CONTACT_PUSH_TOOL}]},
    )
    assert result["bundle"]["candidate_id"] == "candidate-001"
    assert _CapturingRefinementAgent.init_values["session_id"] is None
    assert _CapturingRefinementAgent.init_values["thread_id"] is None
    assert _CapturingRefinementAgent.init_values["reconstructed"] is True
    assert _CapturingRefinementAgent.proposal_values["refinement_context"] == refinement
    assert len(CampaignStore(root).candidate_ledger.records()) == 2

    candidate_record = gate_records["candidate"]
    canonical = gate_root / "episodes" / candidate_record.logical_id / "record.json"
    atomic_write_json(
        canonical,
        {**candidate_record.as_dict(), "elapsed_s": 2.0},
        overwrite=True,
    )
    with pytest.raises(ValueError, match="canonical record is missing or changed"):
        _agent_artifact_index(CampaignStore(root))


def test_registered_candidate_recovers_without_duplicate_provider_call(
    tmp_path: Path, monkeypatch
) -> None:
    root = tmp_path / "campaign"
    store = CampaignStore(root)
    store.initialize(_manifest())
    store.transition(CampaignPhase.CLUSTER)
    store.transition(CampaignPhase.DIAGNOSE)
    diagnosis = _diagnosis()
    store.register_diagnosis(diagnosis)
    store.transition(CampaignPhase.PROPOSE)
    candidate = _candidate(candidate_id="candidate-precommitted", diagnosis=diagnosis)
    shadow = {
        "candidate_sha256": candidate.sha256,
        "passed_detection_preflight": True,
    }
    shadow_root = root / "analysis" / "candidate-shadow-replay"
    atomic_write_json(shadow_root / f"{candidate.sha256}.json", shadow, overwrite=False)
    atomic_write_json(
        shadow_root / f"{candidate.sha256}.precommit.json",
        {
            "schema_version": 1,
            "candidate_sha256": candidate.sha256,
            "parent_bundle_sha256": None,
            "shadow_report_sha256": canonical_sha256(shadow),
            "target_trajectory_sha256": [],
        },
        overwrite=False,
    )
    candidate_sha256 = store.register_candidate(candidate)

    # Model the crash window after the append-only candidate ledger write but
    # before the recoverable state pointer was published.
    state = store.state()
    atomic_write_json(
        store.state_path,
        {**state, "candidate_sha256": None},
        overwrite=True,
    )
    monkeypatch.setattr("rpent.evolution.lifecycle.CodexStageAgent", _UnexpectedAgent)
    recovered = run_proposal_stage(
        campaign_root=root,
        tool_catalog={"tools": [{"name": CONTACT_PUSH_TOOL}]},
    )
    latest = CampaignStore(root)
    assert recovered["candidate_sha256"] == candidate_sha256
    assert recovered["recovered_registration"] is True
    assert latest.state()["phase"] == CampaignPhase.SAME_SEED_GATE.value
    assert latest.state()["candidate_sha256"] == candidate_sha256
    assert len(latest.candidate_ledger.records()) == 1


def test_registered_candidate_recovery_cannot_bypass_shadow_false_positives(
    tmp_path: Path, monkeypatch
) -> None:
    root = tmp_path / "campaign-false-positive-recovery"
    store = CampaignStore(root)
    store.initialize(_manifest())
    store.transition(CampaignPhase.CLUSTER)
    store.transition(CampaignPhase.DIAGNOSE)
    diagnosis = _diagnosis()
    store.register_diagnosis(diagnosis)
    store.transition(CampaignPhase.PROPOSE)
    candidate = _candidate(candidate_id="candidate-false-positive", diagnosis=diagnosis)
    shadow = {
        "candidate_sha256": candidate.sha256,
        "preflight_conclusive": False,
        "passed_detection_preflight": False,
        "success_control_count": 8,
        "success_control_false_positives": 8,
        "success_control_false_positive_rate": 1.0,
    }
    shadow_root = root / "analysis" / "candidate-shadow-replay"
    atomic_write_json(shadow_root / f"{candidate.sha256}.json", shadow, overwrite=False)
    atomic_write_json(
        shadow_root / f"{candidate.sha256}.precommit.json",
        {
            "schema_version": 1,
            "candidate_sha256": candidate.sha256,
            "parent_bundle_sha256": None,
            "shadow_report_sha256": canonical_sha256(shadow),
            "target_trajectory_sha256": [],
        },
        overwrite=False,
    )
    store.register_candidate(candidate)
    monkeypatch.setattr("rpent.evolution.lifecycle.CodexStageAgent", _UnexpectedAgent)

    with pytest.raises(ValueError, match="false-positive rate 1 exceeds"):
        run_proposal_stage(
            campaign_root=root,
            tool_catalog={"tools": [{"name": CONTACT_PUSH_TOOL}]},
        )

    latest = CampaignStore(root)
    assert latest.state()["phase"] == CampaignPhase.PROPOSE.value
    assert len(latest.candidate_ledger.records()) == 1
