from __future__ import annotations

from pathlib import Path

import pytest

from rpent.evolution.lifecycle import record_gate_and_advance
from rpent.evolution.models import (
    CampaignManifest,
    CampaignPhase,
    CandidateBundle,
    GateDecision,
)
from rpent.evolution.protocol import EvolutionProtocol
from rpent.evolution.schedule import preregister_seed_schedule
from rpent.evolution.store import CampaignStore


def _manifest(*, heldout_mode: str, runtime_extra: dict | None = None) -> CampaignManifest:
    rollout = (21,)
    heldout = tuple(range(1, 21))
    policy = {
        "same_seed_pass_rate": 0.5,
        "heldout_mode": heldout_mode,
        "heldout_alpha": 0.05,
        "heldout_max_rounds": 1,
        "skip_regression_gate": True,
        **(runtime_extra or {}),
    }
    return CampaignManifest(
        campaign_id=f"protocol-{heldout_mode}",
        environment="libero_pro",
        task="libero_goal_swap/task3",
        generation=0,
        code_commit="0" * 40,
        prompt_sha256="1" * 64,
        model="test-model",
        tool_catalog_sha256="2" * 64,
        rollout_seeds=rollout,
        heldout_seeds=heldout,
        policy_rng_by_seed={str(seed): seed * 101 for seed in (*rollout, *heldout)},
        expected_rollouts=1,
        expected_heldout=20,
        initial_logical_slots=1,
        continuous_logical_slots=1,
        maximum_logical_slots=1,
        maximum_api_concurrency=1,
        episode_timeout_s=20,
        no_progress_timeout_s=10,
        runtime={"evolution_policy": policy, "heldout_gate_kind": "heldout_20"},
    )


def _candidate(store: CampaignStore) -> CandidateBundle:
    for phase in (
        CampaignPhase.CLUSTER,
        CampaignPhase.DIAGNOSE,
        CampaignPhase.PROPOSE,
    ):
        store.transition(phase)
    candidate = CandidateBundle(
        candidate_id="candidate-protocol",
        generation=0,
        parent_sha256=None,
        diagnosis_sha256="3" * 64,
        causal_hypothesis="one auditable failure mechanism",
        mechanism_change="one frozen plugin change",
        validation_plan="paired same-seed and fixed held-out measurement",
        critic_rules=(),
        recovery_rules=(),
        tool_plugin={"name": "test-plugin"},
    )
    store.register_candidate(candidate)
    store.transition(CampaignPhase.SAME_SEED_GATE)
    return candidate


def _decision(
    *, candidate: CandidateBundle, kind: str, passed: bool, suffix: str
) -> GateDecision:
    return GateDecision(
        decision_id=f"gate-{suffix}",
        candidate_sha256=candidate.sha256,
        parent_sha256=None,
        kind=kind,  # type: ignore[arg-type]
        passed=passed,
        conclusive=True,
        candidate_successes=0 if not passed else 1,
        parent_successes=0,
        paired_count=20 if kind == "heldout_20" else 1,
        candidate_wins=0 if not passed else 1,
        parent_wins=0,
        p_value=0.5 if kind == "heldout_20" else None,
        alpha=0.05 if kind == "heldout_20" else None,
        candidate_safety_events=0,
        parent_safety_events=0,
        rationale="test decision",
    )


def test_protocol_isolates_fixed_seed_1_to_20() -> None:
    protocol = EvolutionProtocol()
    rollout, heldout, _ = preregister_seed_schedule(
        master_seed=7,
        task="libero_goal_swap/task3",
        rollout_count=protocol.rollout_count,
        heldout_count=len(protocol.heldout_seeds),
        heldout_seeds=protocol.heldout_seeds,
    )
    protocol.validate_partition(rollout, heldout)
    assert heldout == tuple(range(1, 21))
    assert not set(rollout) & set(heldout)


def test_protocol_rejects_rollout_heldout_overlap() -> None:
    protocol = EvolutionProtocol(rollout_count=1)
    with pytest.raises(ValueError, match="overlap"):
        protocol.validate_partition((1,), protocol.heldout_seeds)


def test_protocol_parses_tunable_gate_thresholds() -> None:
    protocol = EvolutionProtocol.from_dict(
        {
            "same_seed": {"pass_rate": 0.75, "max_rounds": 4},
            "heldout": {
                "mode": "validation",
                "alpha": 0.05,
                "min_gain": 3,
                "min_success_rate": 0.6,
                "require_significance": False,
                "max_rounds": 2,
            },
        }
    )
    assert protocol.same_seed_pass_rate == 0.75
    assert protocol.same_seed_max_rounds == 4
    assert protocol.heldout_mode == "validation"
    assert protocol.heldout_min_gain == 3
    assert protocol.heldout_min_success_rate == 0.6
    assert protocol.heldout_require_significance is False


def test_report_only_heldout_failure_still_allows_promotion(tmp_path: Path) -> None:
    store = CampaignStore(tmp_path / "test-mode")
    store.initialize(_manifest(heldout_mode="test"))
    candidate = _candidate(store)
    record_gate_and_advance(
        campaign_root=store.root,
        decision=_decision(
            candidate=candidate, kind="same_seed", passed=True, suffix="same-seed"
        ),
    )
    assert store.state()["phase"] == CampaignPhase.HELDOUT_GATE.value
    record_gate_and_advance(
        campaign_root=store.root,
        decision=_decision(
            candidate=candidate, kind="heldout_20", passed=False, suffix="heldout"
        ),
    )
    assert store.state()["phase"] == CampaignPhase.PROMOTE.value
    assert store.promote(candidate.sha256)["candidate_sha256"] == candidate.sha256


def test_validation_heldout_failure_obeys_round_limit(tmp_path: Path) -> None:
    store = CampaignStore(tmp_path / "validation-mode")
    store.initialize(_manifest(heldout_mode="validation"))
    candidate = _candidate(store)
    record_gate_and_advance(
        campaign_root=store.root,
        decision=_decision(
            candidate=candidate, kind="same_seed", passed=True, suffix="same-seed"
        ),
    )
    record_gate_and_advance(
        campaign_root=store.root,
        decision=_decision(
            candidate=candidate, kind="heldout_20", passed=False, suffix="heldout"
        ),
    )
    assert store.state()["phase"] == CampaignPhase.COMPLETE.value
    assert store.state()["optimization_outcome"] == (
        "heldout_validation_iteration_budget_exhausted"
    )


def test_same_seed_failure_obeys_explicit_round_limit(tmp_path: Path) -> None:
    store = CampaignStore(tmp_path / "same-seed-limit")
    store.initialize(
        _manifest(
            heldout_mode="test",
            runtime_extra={"same_seed_max_rounds": 1},
        )
    )
    candidate = _candidate(store)
    record_gate_and_advance(
        campaign_root=store.root,
        decision=_decision(
            candidate=candidate, kind="same_seed", passed=False, suffix="same-seed"
        ),
    )
    assert store.state()["phase"] == CampaignPhase.COMPLETE.value
    assert store.state()["optimization_outcome"] == (
        "same_seed_gate_iteration_budget_exhausted"
    )
