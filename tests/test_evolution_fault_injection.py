# Copyright (c) 2026 RPent Contributors
from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pytest

from rpent.evolution.campaign import analyze_failures
from rpent.evolution.fault_injection import (
    STAGE_ORDER,
    CampaignFaultHarness,
    FaultPoint,
    InjectedCampaignCrash,
    LifecycleStage,
    create_fault_plan,
    main,
)
from rpent.evolution.jsonio import atomic_write_json, read_json
from rpent.evolution.lifecycle import promote_and_complete, record_gate_and_advance
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


def _manifest(*, parent: str | None = None) -> CampaignManifest:
    return CampaignManifest(
        campaign_id="fault-injection",
        environment="robocasa",
        task="SlideDishwasherRack",
        generation=3 if parent else 0,
        code_commit="1" * 40,
        prompt_sha256="2" * 64,
        model="test",
        tool_catalog_sha256="3" * 64,
        rollout_seeds=(17,),
        heldout_seeds=(117,),
        policy_rng_by_seed={"17": 170, "117": 1170},
        parent_bundle_sha256=parent,
        expected_rollouts=1,
        expected_heldout=1,
        protocol_explicit=False,
    )


def _episode(parent: str | None) -> EpisodeRecord:
    generation = 3 if parent else 0
    segment = FailureSegment(
        segment_id="segment-17",
        episode_id="episode-17",
        failure_class="stagnation",
        stage="terminal_push",
        tool="robocasa.slide_dishwasher.vla.contact_push",
        summary="rack progress stopped before completion",
        earliest_divergence_step=3,
        start_step=2,
        end_step=5,
    )
    return EpisodeRecord(
        episode_id="episode-17",
        logical_id=f"g{generation:04d}-rollout-000",
        generation=generation,
        seed=17,
        policy_rng=170,
        bundle_sha256=parent,
        status="valid",
        success=False,
        started_at="2026-08-07T00:00:00+00:00",
        finished_at="2026-08-07T00:00:01+00:00",
        elapsed_s=1.0,
        artifact_index={"trajectory": "episode-17/trajectory.jsonl"},
        failure_segment=segment,
        failure_segments=(segment,),
    )


def _diagnosis() -> CausalDiagnosis:
    return CausalDiagnosis(
        diagnosis_id="diagnosis-stagnation",
        cluster_id="cluster-stagnation",
        outcome="rack remained short of the terminal state",
        immediate_trigger="contact progress became stagnant",
        root_cause="the actor did not re-establish bounded contact",
        contributing_causes=("the terminal suffix lacked a recovery handoff",),
        competing_hypotheses=(
            "the actor did not re-establish bounded contact",
            "the task completion threshold was misclassified",
        ),
        owner_layer="recovery",
        affected_component="terminal rack recovery",
        earliest_divergence="first sustained progress stall",
        supporting_evidence_ids=("segment-17",),
        counterevidence_ids=(),
        falsifier="paired recovery remains stalled",
        distinguishing_check="bounded re-engagement restores rack progress",
        required_validation="paired same-seed closed-loop replay",
        confidence=0.8,
    )


def _candidate(parent: str | None, diagnosis: CausalDiagnosis) -> CandidateBundle:
    critic = CriticRule(
        rule_id="rack-stall",
        title="rack progress stall",
        feature="privileged.dishwasher.rack.position",
        operator="stagnant",
        threshold=0.01,
        dwell_steps=3,
        cooldown_steps=2,
        proposal="request bounded contact recovery",
        evidence_ids=("segment-17",),
    )
    return CandidateBundle(
        candidate_id="candidate-rack-stall",
        generation=3 if parent else 0,
        parent_sha256=parent,
        diagnosis_sha256=diagnosis.sha256,
        causal_hypothesis=diagnosis.root_cause,
        mechanism_change="add one bounded re-engagement",
        validation_plan="paired same-seed, regression, and heldout gates",
        critic_rules=(critic,),
        recovery_rules=(
            RecoveryRule(
                recovery_id="recover-rack-contact",
                title="bounded rack re-engagement",
                trigger_rule_ids=(critic.rule_id,),
                precondition="the current stall proposal is active",
                steps=(
                    RecoveryStep(
                        tool="robocasa.slide_dishwasher.vla.contact_push",
                        parameters={"max_chunks": 1},
                        stop_when="fresh progress is observed",
                    ),
                ),
                safety_constraints=("actor remains the only environment writer",),
                stop_condition="one bounded tool call completes",
                fallback="terminate safely",
                evidence_ids=("segment-17",),
            ),
        ),
    )


def _gate(kind: str, candidate: CandidateBundle) -> GateDecision:
    return GateDecision(
        decision_id=f"gate-{kind}",
        candidate_sha256=candidate.sha256,
        parent_sha256=candidate.parent_sha256,
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
        rationale="fault-injection acceptance fixture passed",
    )


def _actions(
    root: Path,
    parent: str | None,
) -> dict[LifecycleStage, Callable[[], object]]:
    store = CampaignStore(root)
    diagnosis = _diagnosis()
    candidate = _candidate(parent, diagnosis)

    def rollout() -> bool:
        return store.record_episode(_episode(parent))

    def diagnose() -> None:
        store.transition(CampaignPhase.DIAGNOSE)
        store.register_diagnosis(diagnosis)
        store.transition(CampaignPhase.PROPOSE)

    def propose() -> str:
        value = store.register_candidate(candidate)
        store.transition(CampaignPhase.SAME_SEED_GATE)
        return value

    return {
        LifecycleStage.ROLLOUT: rollout,
        LifecycleStage.CLUSTER: lambda: analyze_failures(root),
        LifecycleStage.DIAGNOSE: diagnose,
        LifecycleStage.PROPOSE: propose,
        LifecycleStage.SAME_SEED_GATE: lambda: record_gate_and_advance(
            campaign_root=root,
            decision=_gate("same_seed", candidate),
        ),
        LifecycleStage.REGRESSION_GATE: lambda: record_gate_and_advance(
            campaign_root=root,
            decision=_gate("regression", candidate),
        ),
        LifecycleStage.HELDOUT_GATE: lambda: record_gate_and_advance(
            campaign_root=root,
            decision=_gate("heldout_10", candidate),
        ),
        LifecycleStage.PROMOTE: lambda: promote_and_complete(campaign_root=root),
    }


def test_all_stage_before_and_after_write_crashes_resume_without_duplicates(
    tmp_path: Path,
) -> None:
    root = tmp_path / "campaign"
    store = CampaignStore(root)
    store.initialize(_manifest())
    create_fault_plan(
        root,
        plan_id="all-stage-boundaries",
        points={stage: tuple(FaultPoint) for stage in STAGE_ORDER},
    )
    harness = CampaignFaultHarness(root)
    actions = _actions(root, None)
    calls = dict.fromkeys(STAGE_ORDER, 0)

    for stage in STAGE_ORDER:
        action = actions[stage]

        def counted_action(
            *,
            selected: LifecycleStage = stage,
            operation: Callable[[], object] = action,
        ) -> object:
            calls[selected] += 1
            return operation()

        with pytest.raises(InjectedCampaignCrash) as before:
            harness.run_stage(stage, counted_action)
        assert before.value.point == FaultPoint.BEFORE_WRITE
        assert calls[stage] == 0

        with pytest.raises(InjectedCampaignCrash) as after:
            harness.run_stage(stage, counted_action)
        assert after.value.point == FaultPoint.AFTER_WRITE
        assert calls[stage] == 1

        recovered = harness.run_stage(stage, counted_action)
        assert recovered.recovered is True
        assert calls[stage] == 1

    report = harness.verify()
    assert report["completed_stages"] == [stage.value for stage in STAGE_ORDER]
    assert report["campaign_phase"] == CampaignPhase.COMPLETE.value
    assert report["injections"] == [
        {"stage": stage.value, "point": point.value}
        for stage in STAGE_ORDER
        for point in FaultPoint
    ]
    assert len(store.episodes.records()) == 1
    assert len(store.candidate_ledger.records()) == 1
    assert len(store.gates.records()) == 3
    assert len(store.promotions.records()) == 1
    assert len(harness.terminals.records()) == len(STAGE_ORDER)

    for stage in STAGE_ORDER:
        result = harness.run_stage(
            stage,
            lambda: pytest.fail(f"recovered stage reran: {stage.value}"),
        )
        assert result.recovered is True
    assert len(store.episodes.records()) == 1
    assert len(store.candidate_ledger.records()) == 1
    assert len(store.gates.records()) == 3
    assert len(store.promotions.records()) == 1


def test_fault_plan_is_immutable(tmp_path: Path) -> None:
    root = tmp_path / "campaign"
    CampaignStore(root).initialize(_manifest())
    first = create_fault_plan(
        root,
        plan_id="frozen-plan",
        points={LifecycleStage.ROLLOUT: (FaultPoint.BEFORE_WRITE,)},
    )
    repeated = create_fault_plan(
        root,
        plan_id="frozen-plan",
        points={LifecycleStage.ROLLOUT: (FaultPoint.BEFORE_WRITE,)},
    )
    assert repeated.sha256 == first.sha256
    with pytest.raises(FileExistsError, match="immutable artifact already differs"):
        create_fault_plan(
            root,
            plan_id="changed-plan",
            points={LifecycleStage.ROLLOUT: (FaultPoint.AFTER_WRITE,)},
        )


def test_stale_parent_fails_closed_before_operation(tmp_path: Path) -> None:
    root = tmp_path / "campaign"
    original = "a" * 64
    store = CampaignStore(root)
    store.initialize(_manifest(parent=original))
    create_fault_plan(root, plan_id="stale-parent", points={})
    state = read_json(store.state_path)
    atomic_write_json(
        store.state_path,
        {**state, "current_bundle_sha256": "b" * 64},
        overwrite=True,
    )
    called = False

    def operation() -> None:
        nonlocal called
        called = True

    with pytest.raises(ValueError, match="stale parent"):
        CampaignFaultHarness(root).run_stage(LifecycleStage.ROLLOUT, operation)
    assert called is False
    assert store.episodes.records() == []


def test_incomplete_terminal_record_fails_closed(tmp_path: Path) -> None:
    root = tmp_path / "campaign"
    CampaignStore(root).initialize(_manifest())
    plan = create_fault_plan(root, plan_id="incomplete-terminal", points={})
    path = root / "fault_injection" / "stage_terminals.jsonl"
    path.write_text(
        json.dumps(
            {
                "terminal_id": f"{plan.sha256}:ROLLOUT",
                "stage": "ROLLOUT",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="incomplete terminal record"):
        CampaignFaultHarness(root).verify()


def test_terminal_with_changed_payload_fails_closed(tmp_path: Path) -> None:
    root = tmp_path / "campaign"
    store = CampaignStore(root)
    store.initialize(_manifest())
    create_fault_plan(root, plan_id="changed-payload", points={})
    harness = CampaignFaultHarness(root)
    harness.run_stage(
        LifecycleStage.ROLLOUT, lambda: store.record_episode(_episode(None))
    )
    canonical = root / "episodes" / "g0000-rollout-000" / "record.json"
    payload = read_json(canonical)
    atomic_write_json(canonical, {**payload, "success": True}, overwrite=True)
    with pytest.raises(ValueError, match="missing or changed"):
        CampaignFaultHarness(root).verify()


def test_minimal_cli_creates_and_verifies_empty_plan(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = tmp_path / "campaign"
    CampaignStore(root).initialize(_manifest())
    assert (
        main(
            [
                "create-plan",
                "--campaign-root",
                str(root),
                "--plan-id",
                "cli-plan",
            ]
        )
        == 0
    )
    created = json.loads(capsys.readouterr().out)
    assert created["plan_id"] == "cli-plan"
    assert main(["status", "--campaign-root", str(root)]) == 0
    status = json.loads(capsys.readouterr().out)
    assert status["completed_stages"] == []
