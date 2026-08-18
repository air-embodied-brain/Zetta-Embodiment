# Copyright (c) 2026 RPent Contributors
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from rpent.evolution.jsonio import atomic_write_json, read_json
from rpent.evolution.models import (
    CampaignManifest,
    CampaignPhase,
    CandidateBundle,
    CriticRule,
    RecoveryRule,
    RecoveryStep,
)
from rpent.evolution.store import CampaignStore
from scripts.evolution.prepare_libero_heldout_validation import (
    EVALUATION_SCOPE,
    prepare,
)


def _candidate() -> CandidateBundle:
    return CandidateBundle(
        candidate_id="heldout-candidate",
        generation=1,
        parent_sha256=None,
        diagnosis_sha256="d" * 64,
        causal_hypothesis="fixture interaction moved in the wrong direction",
        mechanism_change="apply one bounded semantic joint correction",
        validation_plan="paired heldout_20",
        critic_rules=(
            CriticRule(
                rule_id="joint-wrong-direction",
                title="joint wrong direction",
                feature="privileged.joint.fixture.position",
                operator="gt",
                threshold=0.02,
                dwell_steps=2,
                cooldown_steps=100,
                proposal="request bounded recovery",
                evidence_ids=("segment-1",),
            ),
        ),
        recovery_rules=(
            RecoveryRule(
                recovery_id="joint-lower",
                title="lower semantic joint",
                trigger_rule_ids=("joint-wrong-direction",),
                precondition="wrong direction rule is active",
                steps=(
                    RecoveryStep(
                        tool="semantic_joint_interact",
                        parameters={"direction": "lower"},
                        stop_when="joint reaches the lower endpoint",
                    ),
                ),
                safety_constraints=("bounded OSC actions only",),
                stop_condition="official termination or bounded completion",
                fallback="return control to VLA",
                evidence_ids=("segment-1",),
            ),
        ),
    )


def test_prepare_libero_heldout_only_preserves_schedule_and_blocks_promotion(
    tmp_path: Path,
) -> None:
    repo = Path(__file__).resolve().parents[1]
    rollout = (101, 102)
    heldout = tuple(range(1, 21))
    policy_rng = {str(seed): seed * 17 for seed in (*rollout, *heldout)}
    source = CampaignManifest(
        campaign_id="source",
        environment="libero_pro",
        task="libero_goal_task/task7",
        generation=0,
        code_commit="1" * 40,
        prompt_sha256="2" * 64,
        model="gpt-5.6-sol",
        tool_catalog_sha256="3" * 64,
        rollout_seeds=rollout,
        heldout_seeds=heldout,
        policy_rng_by_seed=policy_rng,
        expected_rollouts=len(rollout),
        expected_heldout=20,
        runtime={
            "heldout_gate_kind": "heldout_20",
            "same_seed_gate_rollout_command": [
                "old-python",
                "/old/robots/libero/run_evolution_rollout.py",
                "--vla-endpoint",
                "http://old",
                "--no-allow-privileged-evidence",
            ],
        },
    )
    source_path = tmp_path / "source.json"
    atomic_write_json(source_path, source.as_dict(), overwrite=False)
    candidate = _candidate()
    candidate_path = tmp_path / "candidate.json"
    atomic_write_json(candidate_path, candidate.as_dict(), overwrite=False)
    root = tmp_path / "heldout"

    report = prepare(
        SimpleNamespace(
            source_manifest=source_path,
            candidate_bundle=candidate_path,
            output_root=root,
            campaign_id="heldout-only",
            repository_root=repo,
            runtime_python=Path(__import__("sys").executable),
            code_commit="4" * 40,
            vla_endpoint="http://127.0.0.1:18811",
            environment_gpus=(6,),
            vla_gpu=7,
        )
    )

    manifest = CampaignStore(root).manifest()
    state = CampaignStore(root).state()
    preregistration = read_json(root / "heldout-only-preregistration.json")
    command = manifest.runtime["heldout_gate_rollout_command"]
    assert manifest.heldout_seeds == heldout
    assert manifest.policy_rng_by_seed == policy_rng
    assert manifest.generation == 0
    assert state["phase"] == CampaignPhase.HELDOUT_GATE
    assert state["candidate_sha256"] == report["candidate_sha256"]
    assert manifest.runtime["evaluation_scope"] == EVALUATION_SCOPE
    assert command[0] == str(Path(__import__("sys").executable))
    assert command[1] == str(
        repo / "robots" / "libero" / "run_evolution_rollout.py"
    )
    assert "--allow-privileged-evidence" in command
    assert "--no-allow-privileged-evidence" not in command
    assert command[command.index("--gpu") + 1] == "6"
    assert command[command.index("--allowed-environment-gpus") + 1] == "6"
    assert command[command.index("--vla-gpu") + 1] == "7"
    assert preregistration["runtime_device_contract"]["same_gpu_forbidden"] is True
    assert report["candidate_sha256"] != candidate.sha256
    assert preregistration["input_candidate_sha256"] == candidate.sha256
    assert preregistration["promotion_authorized"] is False
    assert preregistration["lifecycle_phases_not_claimed"] == [
        "same_seed_gate",
        "regression_gate",
    ]

    store = CampaignStore(root)
    store.transition(CampaignPhase.PROMOTE)
    with pytest.raises(ValueError, match="heldout-only"):
        store.promote(report["candidate_sha256"])


def test_prepare_libero_heldout_only_accepts_disjoint_independent_50_schedule(
    tmp_path: Path,
) -> None:
    repo = Path(__file__).resolve().parents[1]
    rollout = (101, 102)
    source_heldout = tuple(range(1, 21))
    independent = tuple(range(1001, 1051))
    source_policy = {str(seed): seed * 17 for seed in (*rollout, *source_heldout)}
    source = CampaignManifest(
        campaign_id="source-50",
        environment="libero_pro",
        task="libero_goal_task/task7",
        generation=0,
        code_commit="1" * 40,
        prompt_sha256="2" * 64,
        model="gpt-5.6-sol",
        tool_catalog_sha256="3" * 64,
        rollout_seeds=rollout,
        heldout_seeds=source_heldout,
        policy_rng_by_seed=source_policy,
        expected_rollouts=len(rollout),
        expected_heldout=20,
        runtime={
            "heldout_gate_kind": "heldout_20",
            "same_seed_gate_rollout_command": [
                "old-python",
                "/old/robots/libero/run_evolution_rollout.py",
                "--vla-endpoint",
                "http://old",
            ],
        },
    )
    source_path = tmp_path / "source-50.json"
    atomic_write_json(source_path, source.as_dict(), overwrite=False)
    candidate_path = tmp_path / "candidate-50.json"
    atomic_write_json(candidate_path, _candidate().as_dict(), overwrite=False)
    schedule_path = tmp_path / "schedule-50.json"
    independent_policy = {str(seed): seed * 31 for seed in independent}
    atomic_write_json(
        schedule_path,
        {
            "schema_version": 1,
            "schedule_id": "task7-independent-heldout50-v1",
            "heldout_seeds": list(independent),
            "policy_rng_by_seed": independent_policy,
        },
        overwrite=False,
    )

    report = prepare(
        SimpleNamespace(
            source_manifest=source_path,
            candidate_bundle=candidate_path,
            output_root=tmp_path / "heldout-50",
            campaign_id="heldout-only-50",
            repository_root=repo,
            runtime_python=Path(__import__("sys").executable),
            code_commit="4" * 40,
            vla_endpoint="http://127.0.0.1:18811",
            environment_gpus=(6,),
            vla_gpu=7,
            heldout_schedule=schedule_path,
        )
    )

    manifest = CampaignStore(tmp_path / "heldout-50").manifest()
    preregistration = read_json(
        tmp_path / "heldout-50" / "heldout-only-preregistration.json"
    )
    assert manifest.heldout_seeds == independent
    assert manifest.expected_heldout == 50
    assert manifest.runtime["heldout_gate_kind"] == "heldout"
    assert set(manifest.policy_rng_by_seed) == {
        *(str(seed) for seed in rollout),
        *(str(seed) for seed in independent),
    }
    assert preregistration["gate_kind"] == "heldout"
    assert preregistration["statistical_contract"] == (
        "two_stage_paired_mcnemar_10_to_50"
    )
    assert preregistration["stage_2_count"] == 50
    assert manifest.episode_timeout_s == 2700
    assert preregistration["episode_timeout_s"] == 2700
    assert report["state"]["phase"] == CampaignPhase.HELDOUT_GATE.value
