# Copyright (c) 2026 Zetta Contributors
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Callable

import pytest

from zetta.evolution.campaign import analyze_failures
from zetta.evolution.gate_runner import (
    CandidateGateRunner,
    PairedGateRunner,
    StaleCandidateError,
)
from zetta.evolution.gating import evaluate_paired_gate
from zetta.evolution.jsonio import atomic_write_json, read_json
from zetta.evolution.lifecycle import (
    _completed_gate_episode_rows,
    authorize_same_seed_threshold_override,
    effective_same_seed_gate_pass_rate,
)
from zetta.evolution.models import (
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
from zetta.evolution.queue import RolloutJob, SharedHostQueue
from zetta.evolution.store import CampaignStore
from zetta.evolution.supervisor import EvolutionSupervisor, promote_and_spawn_generation


def _manifest(
    seeds: tuple[int, ...],
    *,
    reuse_parent_evidence: bool = False,
    max_infrastructure_attempts: int = 3,
) -> CampaignManifest:
    heldout = 99
    policy_rng = {str(seed): seed * 101 for seed in (*seeds, heldout)}
    return CampaignManifest(
        campaign_id="candidate-gate-test",
        environment="robocasa",
        task="SlideDishwasherRack",
        generation=0,
        code_commit="1" * 40,
        prompt_sha256="2" * 64,
        model="test-vla",
        tool_catalog_sha256="3" * 64,
        rollout_seeds=seeds,
        heldout_seeds=(heldout,),
        policy_rng_by_seed=policy_rng,
        expected_rollouts=len(seeds),
        expected_heldout=1,
        max_infrastructure_attempts=max_infrastructure_attempts,
        runtime={
            "reuse_rollout_parent_evidence": reuse_parent_evidence,
            "subsequent_rollout_count": 10,
            "rollout_command": [
                "python",
                "fake_rollout.py",
                "--seed",
                "{seed}",
                "--policy-rng",
                "{policy_rng}",
                "--logical-id",
                "{logical_id}",
                "--attempt-index",
                "{attempt_index}",
                "--bundle",
                "{bundle_file}",
                "--bundle-sha256",
                "{bundle_sha256}",
                "--output-dir",
                "{output_dir}",
                "--result-file",
                "{result_file}",
            ],
            "same_seed_gate_rollout_command": [
                "python",
                "fake_rollout.py",
                "--seed",
                "{seed}",
                "--policy-rng",
                "{policy_rng}",
                "--logical-id",
                "{logical_id}",
                "--attempt-index",
                "{attempt_index}",
                "--bundle",
                "{bundle_file}",
                "--bundle-sha256",
                "{bundle_sha256}",
                "--output-dir",
                "{output_dir}",
                "--result-file",
                "{result_file}",
            ],
        },
    )


def _source_episode(seed: int, policy_rng: int, index: int) -> EpisodeRecord:
    episode_id = f"source-episode-{index}"
    segment = FailureSegment(
        segment_id=f"source-segment-{index}",
        episode_id=episode_id,
        failure_class="rack_stall",
        stage="terminal_push",
        tool="robocasa.slide_dishwasher.vla.contact_push",
        summary="dishwasher rack contact stalls before task completion",
        earliest_divergence_step=5,
        start_step=4,
        end_step=8,
    )
    return EpisodeRecord(
        episode_id=episode_id,
        logical_id=f"g0000-rollout-{index:03d}",
        generation=0,
        seed=seed,
        policy_rng=policy_rng,
        bundle_sha256=None,
        status="valid",
        success=False,
        started_at="2026-08-07T00:00:00+00:00",
        finished_at="2026-08-07T00:00:01+00:00",
        elapsed_s=1.0,
        artifact_index={
            "source": f"trajectory-{index}.jsonl",
            "initial_observation_identity": {
                "state_sha256": f"{seed:064x}",
                "camera_sha256": {"agentview": f"{seed + 1:064x}"},
            },
            "trajectory_index": {
                "artifact_sha256": {"actions": "a" * 64}
            },
        },
        failure_segment=segment,
    )


def _diagnosis(cluster_id: str) -> CausalDiagnosis:
    return CausalDiagnosis(
        diagnosis_id="diagnosis-gate-test",
        cluster_id=cluster_id,
        outcome="rack does not reach the success state",
        immediate_trigger="contact progress becomes stagnant",
        root_cause="terminal contact is not maintained",
        contributing_causes=(),
        competing_hypotheses=(
            "terminal contact is not maintained",
            "the commanded direction is incorrect",
        ),
        owner_layer="recovery",
        affected_component="dishwasher rack terminal push",
        earliest_divergence="first sustained progress stall",
        supporting_evidence_ids=("source-segment-0",),
        counterevidence_ids=(),
        falsifier="paired candidate still stalls",
        distinguishing_check="rack progress resumes after the candidate intervention",
        required_validation="paired closed-loop execution on representative seeds",
        confidence=0.8,
    )


def _candidate(diagnosis: CausalDiagnosis, *, suffix: str = "one") -> CandidateBundle:
    return CandidateBundle(
        candidate_id=f"candidate-{suffix}",
        generation=0,
        parent_sha256=None,
        diagnosis_sha256=diagnosis.sha256,
        causal_hypothesis="terminal contact is not maintained",
        mechanism_change=f"detect and recover terminal rack stalls ({suffix})",
        validation_plan="paired same-seed representative-cluster gate",
        critic_rules=(
            CriticRule(
                rule_id=f"rack-stall-{suffix}",
                title="rack progress stall",
                feature="privileged.dishwasher.rack.position",
                operator="stagnant",
                threshold=0.01,
                dwell_steps=3,
                cooldown_steps=2,
                proposal="request a bounded rack recovery",
                evidence_ids=("source-segment-0",),
            ),
        ),
        recovery_rules=(
            RecoveryRule(
                recovery_id=f"rack-recovery-{suffix}",
                title="bounded rack recovery",
                trigger_rule_ids=(f"rack-stall-{suffix}",),
                precondition="the rack-stall critic is active",
                steps=(
                    RecoveryStep(
                        tool="robocasa.slide_dishwasher.reach.approach_handle",
                        parameters={},
                        stop_when="the end effector reaches the handle region",
                    ),
                ),
                safety_constraints=("stop on collision",),
                stop_condition="the recovery proposal has been reviewed",
                fallback="return control to the VLA proposal loop",
                evidence_ids=("source-segment-0",),
            ),
        ),
    )


def test_candidate_preflight_rejects_critic_without_executable_recovery() -> None:
    candidate = _candidate(_diagnosis("cluster-001"))
    with pytest.raises(ValueError, match="executable recovery"):
        replace(candidate, recovery_rules=())


def test_heldout_20_plan_is_frozen_as_its_own_statistical_contract(
    tmp_path: Path,
) -> None:
    root, queue_root, store, candidate = _setup(tmp_path)
    rollout = store.manifest().rollout_seeds
    heldout = tuple(range(100, 120))
    formal = replace(
        store.manifest(),
        heldout_seeds=heldout,
        expected_heldout=20,
        policy_rng_by_seed={
            str(seed): seed * 101 for seed in (*rollout, *heldout)
        },
        runtime={**store.manifest().runtime, "heldout_gate_kind": "heldout_20"},
    )
    state = store.state()
    atomic_write_json(root / "manifest.json", formal.as_dict(), overwrite=True)
    atomic_write_json(
        root / "state.json",
        {**state, "manifest_sha256": formal.sha256},
        overwrite=True,
    )
    store.transition(CampaignPhase.REGRESSION_GATE)
    store.transition(CampaignPhase.HELDOUT_GATE)
    runner = PairedGateRunner(
        campaign_root=root,
        queue_root=queue_root,
        worker_hosts=("host-a",),
        gate_kind="heldout_20",
        candidate_sha256=candidate.sha256,
    )
    plan = runner.prepare()
    assert len(plan["pairs"]) == 20
    assert plan["heldout_count"] == 20
    assert plan["statistical_contract"] == "fixed_paired_mcnemar_20"


def test_same_seed_gate_adopts_frozen_loop1_parent_before_dispatch(
    tmp_path: Path,
) -> None:
    root, queue_root, store, candidate = _setup(
        tmp_path, reuse_parent_evidence=True
    )
    runner = CandidateGateRunner(
        campaign_root=root,
        queue_root=queue_root,
        worker_hosts=("host-a", "host-b"),
        candidate_sha256=candidate.sha256,
    )
    first = runner.run_once()
    assert first["parent_adopted"] == 2
    assert first["enqueue"]["enqueued"] == 2
    assert first["status"]["valid_arms"] == 2
    pending = _pending_jobs(SharedHostQueue(queue_root))
    assert len(pending) == 2
    assert {job.bundle_sha256 for job in pending} == {candidate.sha256}
    assert len(runner.parent_adoptions.records()) == 2
    resumed = runner.run_once()
    assert resumed["parent_adopted"] == 0
    assert resumed["enqueue"]["enqueued"] == 0
    assert len(runner.parent_adoptions.records()) == 2


def _setup(
    tmp_path: Path,
    *,
    seeds: tuple[int, ...] = (11, 12),
    reuse_parent_evidence: bool = False,
    max_infrastructure_attempts: int = 3,
) -> tuple[Path, Path, CampaignStore, CandidateBundle]:
    root = tmp_path / "campaign"
    queue_root = tmp_path / "queue"
    manifest = _manifest(
        seeds,
        reuse_parent_evidence=reuse_parent_evidence,
        max_infrastructure_attempts=max_infrastructure_attempts,
    )
    store = CampaignStore(root)
    store.initialize(manifest)
    for index, seed in enumerate(seeds):
        store.record_episode(
            _source_episode(seed, manifest.policy_rng_by_seed[str(seed)], index)
        )
    report = analyze_failures(root)
    cluster_id = str(report["dominant_cluster_id"])
    store.transition(CampaignPhase.DIAGNOSE)
    diagnosis = _diagnosis(cluster_id)
    store.register_diagnosis(diagnosis)
    store.transition(CampaignPhase.PROPOSE)
    candidate = _candidate(diagnosis)
    store.register_candidate(candidate)
    store.transition(CampaignPhase.SAME_SEED_GATE)
    return root, queue_root, store, candidate


def _pending_jobs(queue: SharedHostQueue) -> list[RolloutJob]:
    jobs = []
    for path in sorted((queue.root / "pending").glob("*/*.json")):
        payload = read_json(path)
        jobs.append(RolloutJob.from_dict(payload["job"]))
    return jobs


def _gate_episode(
    job: RolloutJob,
    *,
    success: bool = False,
    status: str = "valid",
    policy_rng: int | None = None,
) -> EpisodeRecord:
    candidate_arm = job.bundle_sha256 is not None
    return EpisodeRecord(
        episode_id=f"episode-{job.job_id}",
        logical_id=job.logical_id,
        generation=0,
        seed=job.seed,
        policy_rng=job.policy_rng if policy_rng is None else policy_rng,
        bundle_sha256=job.bundle_sha256,
        status=status,  # type: ignore[arg-type]
        success=success if status == "valid" else None,
        started_at="2026-08-07T01:00:00+00:00",
        finished_at="2026-08-07T01:00:01+00:00",
        elapsed_s=1.0,
        artifact_index={
            "initial_observation_identity": {
                "state_sha256": f"{job.seed:064x}",
                "camera_sha256": {"agentview": f"{job.seed + 1:064x}"},
            },
            "candidate_intervention": candidate_arm,
            "trajectory_index": {
                "artifact_sha256": {"actions": ("c" if candidate_arm else "a") * 64}
            },
        },
        invalid_reason="transport_reset" if status == "infra_invalid" else None,
        attempt_index=job.attempt_index,
    )


def _finish_pending(
    queue: SharedHostQueue,
    hosts: tuple[str, ...],
    record_for: Callable[[RolloutJob], EpisodeRecord],
) -> list[EpisodeRecord]:
    records = []
    while True:
        progressed = False
        for host in hosts:
            claimed = queue.claim(host, worker_id=f"test-{host}")
            if claimed is None:
                continue
            progressed = True
            path, job = claimed
            token = queue.claim_token(path)
            record = record_for(job)
            records.append(record)
            worker_result = {
                "job_id": job.job_id,
                "logical_id": job.logical_id,
                "attempt_index": job.attempt_index,
                "return_code": 0 if record.status == "valid" else 2,
                "watchdog_reason": None,
                "elapsed_s": record.elapsed_s,
                "result": record.as_dict(),
                "success": record.status == "valid",
                "job": job.as_dict(),
            }
            queue.finish(
                path,
                success=record.status == "valid",
                result=worker_result,
                claim_token=token,
            )
        if not progressed:
            return records


def test_schedules_all_representatives_and_completes_existing_gate(
    tmp_path: Path,
) -> None:
    root, queue_root, store, candidate = _setup(tmp_path)
    hosts = ("host-a", "host-b")
    runner = CandidateGateRunner(
        campaign_root=root,
        queue_root=queue_root,
        worker_hosts=hosts,
    )

    first = runner.run_once()
    assert first["enqueue"]["enqueued"] == 4
    jobs = _pending_jobs(SharedHostQueue(queue_root))
    assert len(jobs) == 4
    by_seed: dict[int, list[RolloutJob]] = {}
    for job in jobs:
        by_seed.setdefault(job.seed, []).append(job)
    assert set(by_seed) == {11, 12}
    for seed, pair in by_seed.items():
        assert len(pair) == 2
        assert {job.bundle_sha256 for job in pair} == {None, candidate.sha256}
        assert {job.policy_rng for job in pair} == {
            store.manifest().policy_rng_by_seed[str(seed)]
        }
        assert {(job.bundle_sha256 is not None, job.requires_api) for job in pair} == {
            (False, False),
            (True, True),
        }

    _finish_pending(
        SharedHostQueue(queue_root),
        hosts,
        lambda job: _gate_episode(
            job,
            success=job.bundle_sha256 == candidate.sha256,
        ),
    )
    resumed = CandidateGateRunner(
        campaign_root=root,
        queue_root=queue_root,
        worker_hosts=hosts,
        candidate_sha256=candidate.sha256,
    ).run_once()
    assert resumed["decision"]["passed"] is True
    assert resumed["decision"]["candidate_successes"] == 2
    assert resumed["decision"]["parent_successes"] == 0
    assert resumed["status"]["complete_pairs"] == 2
    assert resumed["status"]["valid_arms"] == 4
    assert CampaignStore(root).state()["phase"] == CampaignPhase.REGRESSION_GATE

    repeated = CandidateGateRunner(
        campaign_root=root,
        queue_root=queue_root,
        worker_hosts=hosts,
    ).run_once()
    assert repeated["ingestion"]["accepted"] == 0
    assert repeated["enqueue"]["enqueued"] == 0
    assert len(CampaignStore(root).gates.records()) == 1


def test_infra_invalid_is_retried_and_accepted_valid_is_not_replayed(
    tmp_path: Path,
) -> None:
    root, queue_root, _, candidate = _setup(tmp_path, seeds=(11,))
    hosts = ("host-a",)
    runner = CandidateGateRunner(
        campaign_root=root,
        queue_root=queue_root,
        worker_hosts=hosts,
    )
    runner.run_once()
    queue = SharedHostQueue(queue_root)
    _finish_pending(
        queue,
        hosts,
        lambda job: _gate_episode(
            job,
            status=(
                "infra_invalid" if job.bundle_sha256 == candidate.sha256 else "valid"
            ),
        ),
    )

    recovered = CandidateGateRunner(
        campaign_root=root,
        queue_root=queue_root,
        worker_hosts=hosts,
    )
    cycle = recovered.run_once()
    assert cycle["status"]["valid_arms"] == 1
    assert cycle["status"]["infra_invalid"] == 1
    pending = _pending_jobs(queue)
    assert len(pending) == 1
    assert pending[0].bundle_sha256 == candidate.sha256
    assert pending[0].attempt_index == 1
    assert not any("parent" in job.logical_id for job in pending)

    # Re-ingesting immutable terminals is a no-op and cannot duplicate a valid.
    second = recovered.run_once()
    assert second["ingestion"]["accepted"] == 0
    assert len(recovered.valid.records()) == 1
    assert len(recovered.attempts.records()) == 2
    assert len(_pending_jobs(queue)) == 1


def test_exhausted_infra_budget_requires_append_only_authorization_and_resumes_at_002(
    tmp_path: Path,
) -> None:
    root, queue_root, _, candidate = _setup(
        tmp_path,
        seeds=(11,),
        reuse_parent_evidence=True,
        max_infrastructure_attempts=2,
    )
    hosts = ("host-a",)
    queue = SharedHostQueue(queue_root)
    runner = CandidateGateRunner(
        campaign_root=root,
        queue_root=queue_root,
        worker_hosts=hosts,
    )
    first = runner.run_once()
    assert first["parent_adopted"] == 1
    assert first["enqueue"]["enqueued"] == 1

    _finish_pending(queue, hosts, lambda job: _gate_episode(job, status="infra_invalid"))
    retry = runner.run_once()
    assert retry["enqueue"]["enqueued"] == 1
    assert _pending_jobs(queue)[0].attempt_index == 1
    _finish_pending(queue, hosts, lambda job: _gate_episode(job, status="infra_invalid"))
    exhausted = runner.run_once()
    candidate_id = next(
        logical_id
        for logical_id in exhausted["status"]
        ["effective_max_infrastructure_attempts_by_logical_id"]
        if logical_id.endswith("candidate")
    )
    assert exhausted["status"]["blocked"] == [candidate_id]
    assert exhausted["status"]["base_max_infrastructure_attempts"] == 2
    assert exhausted["status"]["effective_max_infrastructure_attempts"] == 2

    authorization = runner.authorize_infrastructure_recovery(
        additional_attempts=1,
        reason="Role1 workers did not inherit the frozen provider environment",
    )
    assert authorization["candidate_sha256"] == candidate.sha256
    assert authorization["gate_kind"] == "same_seed"
    assert authorization["base_max_infrastructure_attempts"] == 2
    assert authorization["logical_ids"] == [candidate_id]
    assert [row["attempt_index"] for row in authorization["evidence"]] == [0, 1]
    assert {row["status"] for row in authorization["evidence"]} == {
        "infra_invalid"
    }

    # Reissuing the same command before any new attempt is an idempotent read.
    repeated = runner.authorize_infrastructure_recovery(
        additional_attempts=1,
        reason="Role1 workers did not inherit the frozen provider environment",
    )
    assert repeated == authorization
    assert len(runner.infra_recovery_authorizations.records()) == 1

    resumed = runner.run_once()
    pending = _pending_jobs(queue)
    assert resumed["enqueue"]["enqueued"] == 1
    assert len(pending) == 1
    assert pending[0].logical_id == candidate_id
    assert pending[0].attempt_index == 2
    assert pending[0].output_dir.endswith("attempt-002")
    assert resumed["status"]["base_max_infrastructure_attempts"] == 2
    assert resumed["status"]["effective_max_infrastructure_attempts"] == 3
    assert resumed["status"]["infrastructure_recovery_authorizations"] == [
        authorization
    ]

    _finish_pending(queue, hosts, lambda job: _gate_episode(job, success=True))
    complete = runner.run_once()
    assert complete["status"]["valid_arms"] == 2
    assert complete["decision"]["passed"] is True
    assert len(runner.valid.records()) == 2
    assert len(runner.attempts.records()) == 4  # adopted parent + candidate 000..002
    assert _pending_jobs(queue) == []
    assert runner.run_once()["ingestion"]["accepted"] == 0
    assert len(runner.infra_recovery_authorizations.records()) == 1


def test_infra_recovery_rejects_unexhausted_valid_and_drifted_authorizations(
    tmp_path: Path,
) -> None:
    root, queue_root, _, _ = _setup(
        tmp_path / "unexhausted",
        seeds=(11,),
        reuse_parent_evidence=True,
        max_infrastructure_attempts=2,
    )
    runner = CandidateGateRunner(
        campaign_root=root,
        queue_root=queue_root,
        worker_hosts=("host-a",),
    )
    runner.run_once()
    candidate_id = _pending_jobs(SharedHostQueue(queue_root))[0].logical_id
    with pytest.raises(ValueError, match="exhausted arm"):
        runner.authorize_infrastructure_recovery(
            additional_attempts=1,
            reason="premature authorization",
            logical_ids=(candidate_id,),
        )

    queue = SharedHostQueue(queue_root)
    _finish_pending(queue, ("host-a",), lambda job: _gate_episode(job, success=False))
    runner.ingest()
    with pytest.raises(ValueError, match="valid arm"):
        runner.authorize_infrastructure_recovery(
            additional_attempts=1,
            reason="policy failures are not infrastructure evidence",
            logical_ids=(candidate_id,),
        )

    drift_root, drift_queue_root, _, _ = _setup(
        tmp_path / "drift",
        seeds=(11,),
        reuse_parent_evidence=True,
        max_infrastructure_attempts=2,
    )
    drift = CandidateGateRunner(
        campaign_root=drift_root,
        queue_root=drift_queue_root,
        worker_hosts=("host-a",),
    )
    drift.run_once()
    drift_queue = SharedHostQueue(drift_queue_root)
    for _ in range(2):
        _finish_pending(
            drift_queue,
            ("host-a",),
            lambda job: _gate_episode(job, status="infra_invalid"),
        )
        drift.run_once()
    drift.authorize_infrastructure_recovery(
        additional_attempts=1,
        reason="audited provider environment inheritance failure",
    )
    frozen_plan = read_json(drift.plan_path)
    atomic_write_json(
        drift.plan_path,
        {**frozen_plan, "unauthorized_plan_drift": True},
        overwrite=True,
    )
    with pytest.raises(ValueError, match="binding changed"):
        drift.status()


def test_resume_after_gate_ledger_commit_advances_without_replaying(
    tmp_path: Path,
) -> None:
    root, queue_root, store, candidate = _setup(tmp_path, seeds=(11,))
    hosts = ("host-a",)
    runner = CandidateGateRunner(
        campaign_root=root,
        queue_root=queue_root,
        worker_hosts=hosts,
    )
    runner.run_once()
    _finish_pending(
        SharedHostQueue(queue_root),
        hosts,
        lambda job: _gate_episode(
            job,
            success=job.bundle_sha256 == candidate.sha256,
        ),
    )
    assert runner.ingest()["accepted"] == 2
    plan = runner.prepare()
    candidate_records, parent_records = runner._records_by_arm(plan)
    decision = evaluate_paired_gate(
        kind="same_seed",
        candidate_sha256=candidate.sha256,
        parent_sha256=None,
        candidate_records=candidate_records,
        parent_records=parent_records,
        expected_seeds=(11,),
    )
    # Simulate a process crash after the append-only gate commit but before state advance.
    store.record_gate(decision)
    assert store.state()["phase"] == CampaignPhase.SAME_SEED_GATE

    resumed = CandidateGateRunner(
        campaign_root=root,
        queue_root=queue_root,
        worker_hosts=hosts,
    ).run_once()
    assert resumed["decision"] == decision.as_dict()
    assert CampaignStore(root).state()["phase"] == CampaignPhase.REGRESSION_GATE
    assert len(CampaignStore(root).gates.records()) == 1
    assert len(runner.valid.records()) == 2


def test_normal_zero_score_is_valid_and_advances_as_failed_gate(tmp_path: Path) -> None:
    root, queue_root, _, _ = _setup(tmp_path, seeds=(11,))
    hosts = ("host-a",)
    runner = CandidateGateRunner(
        campaign_root=root,
        queue_root=queue_root,
        worker_hosts=hosts,
    )
    runner.run_once()
    _finish_pending(
        SharedHostQueue(queue_root),
        hosts,
        lambda job: _gate_episode(job, success=False),
    )
    result = runner.run_once()
    assert result["status"]["valid_arms"] == 2
    assert result["status"]["infra_invalid"] == 0
    assert result["decision"]["candidate_successes"] == 0
    assert result["decision"]["passed"] is False
    assert CampaignStore(root).state()["phase"] == CampaignPhase.PROPOSE


def test_same_seed_gate_rejects_early_when_success_upper_bound_is_impossible(
    tmp_path: Path,
) -> None:
    seeds = tuple(range(47))
    root, queue_root, store, candidate = _setup(
        tmp_path,
        seeds=seeds,
        reuse_parent_evidence=True,
    )
    host = "host-a"
    runner = CandidateGateRunner(
        campaign_root=root,
        queue_root=queue_root,
        worker_hosts=(host,),
    )
    started = runner.run_once()
    assert started["parent_adopted"] == 47
    assert started["enqueue"]["enqueued"] == 47

    queue = SharedHostQueue(queue_root)
    for _ in range(33):
        claimed = queue.claim(host, worker_id="test-host-a")
        assert claimed is not None
        path, job = claimed
        token = queue.claim_token(path)
        record = _gate_episode(job, success=False)
        queue.finish(
            path,
            success=True,
            result={
                "job_id": job.job_id,
                "logical_id": job.logical_id,
                "attempt_index": job.attempt_index,
                "return_code": 0,
                "watchdog_reason": None,
                "elapsed_s": record.elapsed_s,
                "result": record.as_dict(),
                "success": True,
                "job": job.as_dict(),
            },
            claim_token=token,
        )

    rejected = runner.run_once()
    decision = rejected["decision"]
    assert decision["passed"] is False
    assert decision["conclusive"] is True
    assert decision["candidate_successes"] == 0
    assert decision["paired_count"] == 47
    assert "upper bound 14/47, required 24/47" in decision["rationale"]
    assert rejected["status"]["valid_arms"] == 80
    assert rejected["enqueue"]["enqueued"] == 0
    assert queue.counts()["pending"] == 14
    assert store.state()["phase"] == CampaignPhase.PROPOSE
    assert len(store.gates.records()) == 1
    indexed_rows = _completed_gate_episode_rows(store)
    assert len(indexed_rows) == 80
    assert {role for _, _, role in indexed_rows} == {
        "parent_gate_episode",
        "candidate_gate_episode",
    }

    claimed = queue.claim(host, worker_id="test-host-a-late")
    assert claimed is not None
    path, job = claimed
    token = queue.claim_token(path)
    late_record = _gate_episode(job, success=False)
    queue.finish(
        path,
        success=True,
        result={
            "job_id": job.job_id,
            "logical_id": job.logical_id,
            "attempt_index": job.attempt_index,
            "return_code": 0,
            "watchdog_reason": None,
            "elapsed_s": late_record.elapsed_s,
            "result": late_record.as_dict(),
            "success": True,
            "job": job.as_dict(),
        },
        claim_token=token,
    )

    resumed = CandidateGateRunner(
        campaign_root=root,
        queue_root=queue_root,
        worker_hosts=(host,),
        candidate_sha256=candidate.sha256,
    ).run_once()
    assert resumed["decision"] == decision
    assert resumed["terminal_decision"] == decision
    assert resumed["status"]["valid_arms"] == 80
    assert len(store.gates.records()) == 1


def test_identical_action_pairs_record_rejection_and_resume_idempotently(
    tmp_path: Path,
) -> None:
    root, queue_root, _, _ = _setup(tmp_path, seeds=(11,))
    hosts = ("host-a",)
    runner = CandidateGateRunner(
        campaign_root=root,
        queue_root=queue_root,
        worker_hosts=hosts,
    )
    runner.run_once()

    def identical_actions(job: RolloutJob) -> EpisodeRecord:
        record = _gate_episode(job, success=False)
        trajectory = {
            "artifact_sha256": {"actions": "a" * 64},
        }
        return replace(
            record,
            artifact_index={
                **record.artifact_index,
                "trajectory_index": trajectory,
            },
        )

    _finish_pending(SharedHostQueue(queue_root), hosts, identical_actions)
    rejected = runner.run_once()
    assert rejected["decision"]["passed"] is False
    assert rejected["decision"]["conclusive"] is True
    assert "never changed" in rejected["decision"]["rationale"]
    assert CampaignStore(root).state()["phase"] == CampaignPhase.PROPOSE
    assert len(CampaignStore(root).gates.records()) == 1

    resumed = runner.run_once()
    assert resumed["ingestion"]["accepted"] == 0
    assert resumed["decision"] == rejected["decision"]
    assert len(CampaignStore(root).gates.records()) == 1


def test_pair_rng_mismatch_and_conflicting_duplicate_valid_fail_closed(
    tmp_path: Path,
) -> None:
    root, queue_root, _, candidate = _setup(tmp_path, seeds=(11,))
    hosts = ("host-a",)
    runner = CandidateGateRunner(
        campaign_root=root,
        queue_root=queue_root,
        worker_hosts=hosts,
    )
    runner.run_once()
    queue = SharedHostQueue(queue_root)
    _finish_pending(
        queue,
        hosts,
        lambda job: _gate_episode(
            job,
            success=job.bundle_sha256 == candidate.sha256,
            policy_rng=(
                job.policy_rng + 1
                if job.bundle_sha256 == candidate.sha256
                else job.policy_rng
            ),
        ),
    )
    with pytest.raises(ValueError, match="frozen job fields"):
        runner.ingest()
    assert runner.attempts.records() == []
    assert runner.valid.records() == []

    # The append-only valid ledger also rejects a different payload for one logical ID.
    job = (
        next(job for job in _pending_jobs(queue) if job.bundle_sha256 is None)
        if _pending_jobs(queue)
        else None
    )
    assert (
        job is None
    )  # all jobs are terminal; construct from the terminal evidence instead
    parent_terminal = next(
        path
        for path in queue.terminal_envelope_paths()
        if read_json(path)["job"]["bundle_sha256"] is None
    )
    parent_job = RolloutJob.from_dict(read_json(parent_terminal)["job"])
    parent = _gate_episode(parent_job)
    assert runner._record_attempt(parent) is True
    with pytest.raises(ValueError, match="conflicting payload"):
        runner._record_attempt(replace(parent, episode_id="different-valid-episode"))


def test_stale_candidate_stops_before_queue_or_attempt_mutation(tmp_path: Path) -> None:
    root, queue_root, store, candidate = _setup(tmp_path, seeds=(11,))
    runner = CandidateGateRunner(
        campaign_root=root,
        queue_root=queue_root,
        worker_hosts=("host-a",),
        candidate_sha256=candidate.sha256,
    )
    runner.prepare()
    diagnosis = _diagnosis("cluster-001")
    store.transition(CampaignPhase.PROPOSE)
    replacement = _candidate(diagnosis, suffix="replacement")
    store.register_candidate(replacement)

    with pytest.raises(StaleCandidateError, match="stale"):
        runner.run_once()
    assert SharedHostQueue(queue_root).counts() == {
        "pending": 0,
        "running": 0,
        "completed": 0,
        "failed": 0,
    }
    assert runner.attempts.records() == []
    assert runner.valid.records() == []


def _enter_regression_gate(store: CampaignStore, candidate: CandidateBundle) -> None:
    decision = GateDecision(
        decision_id="gate-test-same-seed-pass",
        candidate_sha256=candidate.sha256,
        parent_sha256=candidate.parent_sha256,
        kind="same_seed",
        passed=True,
        conclusive=True,
        candidate_successes=1,
        parent_successes=0,
        paired_count=1,
        candidate_wins=1,
        parent_wins=0,
        p_value=None,
        alpha=None,
        candidate_safety_events=0,
        parent_safety_events=0,
        rationale="fixture same-seed pass",
    )
    store.record_gate(decision)
    store.transition(CampaignPhase.REGRESSION_GATE)


def test_regression_gate_adopts_all_frozen_rollout_parent_arms(
    tmp_path: Path,
) -> None:
    root, queue_root, store, candidate = _setup(
        tmp_path, reuse_parent_evidence=True
    )
    _enter_regression_gate(store, candidate)
    runner = PairedGateRunner(
        campaign_root=root,
        queue_root=queue_root,
        worker_hosts=("host-a", "host-b"),
        gate_kind="regression",
        candidate_sha256=candidate.sha256,
    )
    first = runner.run_once()
    assert first["parent_adopted"] == 2
    assert first["enqueue"]["enqueued"] == 2
    assert first["status"]["valid_arms"] == 2
    assert {
        job.bundle_sha256 for job in _pending_jobs(SharedHostQueue(queue_root))
    } == {candidate.sha256}


def test_formal_gates_retry_append_only_expand_heldout_and_spawn_child(
    tmp_path: Path,
) -> None:
    rollout_seeds = (11, 12)
    heldout_seeds = tuple(range(100, 150))
    root, queue_root, store, candidate = _setup(tmp_path, seeds=rollout_seeds)
    original = store.manifest()
    formal = replace(
        original,
        heldout_seeds=heldout_seeds,
        expected_heldout=50,
        policy_rng_by_seed={
            str(seed): seed * 101 for seed in (*rollout_seeds, *heldout_seeds)
        },
    )
    # This fixture upgrades the still-unstarted formal schedule before any
    # formal gate plan exists; immutable production manifests are never edited.
    state = store.state()
    atomic_write_json(root / "manifest.json", formal.as_dict(), overwrite=True)
    atomic_write_json(
        root / "state.json",
        {**state, "manifest_sha256": formal.sha256},
        overwrite=True,
    )
    _enter_regression_gate(store, candidate)
    hosts = ("host-a", "host-b")
    child_root = tmp_path / "child"
    child_queue = tmp_path / "child-queue"
    supervisor = EvolutionSupervisor(
        campaign_root=root,
        queue_root=queue_root,
        worker_hosts=hosts,
        tool_catalog={},
        next_campaign_root=child_root,
        next_queue_root=child_queue,
        next_master_seed=987654,
    )

    regression = PairedGateRunner(
        campaign_root=root,
        queue_root=queue_root,
        worker_hosts=hosts,
        gate_kind="regression",
    )
    first = supervisor.step()["gate"]
    assert first["enqueue"]["enqueued"] == 4
    queue = SharedHostQueue(queue_root)
    regression_jobs = _pending_jobs(queue)
    for seed in rollout_seeds:
        pair = [job for job in regression_jobs if job.seed == seed]
        assert len(pair) == 2
        assert {job.policy_rng for job in pair} == {
            formal.policy_rng_by_seed[str(seed)]
        }
        assert {job.bundle_sha256 for job in pair} == {None, candidate.sha256}
    invalidated = False

    def regression_record(job: RolloutJob) -> EpisodeRecord:
        nonlocal invalidated
        if job.bundle_sha256 == candidate.sha256 and not invalidated:
            invalidated = True
            return _gate_episode(job, status="infra_invalid")
        return _gate_episode(job, success=True)

    _finish_pending(queue, hosts, regression_record)
    retry = supervisor.step()["gate"]
    assert retry["status"]["valid_arms"] == 3
    assert retry["status"]["infra_invalid"] == 1
    pending = _pending_jobs(queue)
    assert len(pending) == 1
    assert pending[0].attempt_index == 1
    assert pending[0].bundle_sha256 == candidate.sha256
    _finish_pending(queue, hosts, lambda job: _gate_episode(job, success=True))
    regression_done = supervisor.step()["gate"]
    assert regression_done["decision"]["kind"] == "regression"
    assert regression_done["decision"]["passed"] is True
    assert CampaignStore(root).state()["phase"] == CampaignPhase.HELDOUT_GATE
    assert len(regression.valid.records()) == 4

    heldout = PairedGateRunner(
        campaign_root=root,
        queue_root=queue_root,
        worker_hosts=hosts,
        gate_kind="heldout",
    )
    stage_one = supervisor.step()["gate"]
    assert stage_one["enqueue"]["enqueued"] == 20
    assert {job.seed for job in _pending_jobs(queue)} == set(heldout_seeds[:10])
    stage_one_ids = {job.logical_id for job in _pending_jobs(queue)}
    _finish_pending(queue, hosts, lambda job: _gate_episode(job, success=False))
    expanded_step = supervisor.step()
    assert expanded_step["action"] == "heldout_gate_expanded"
    expanded = expanded_step["gate"]
    assert expanded["decision"]["kind"] == "heldout_10"
    assert expanded["decision"]["conclusive"] is False
    assert expanded["terminal_decision"] is None
    assert expanded["enqueue"]["enqueued"] == 80
    assert CampaignStore(root).state()["phase"] == CampaignPhase.HELDOUT_GATE
    remaining = _pending_jobs(queue)
    assert len(remaining) == 80
    assert not stage_one_ids & {job.logical_id for job in remaining}
    assert {job.seed for job in remaining} == set(heldout_seeds[10:])
    _finish_pending(
        queue,
        hosts,
        lambda job: _gate_episode(job, success=job.bundle_sha256 == candidate.sha256),
    )
    heldout_done = supervisor.step()["gate"]
    assert heldout_done["decision"]["kind"] == "heldout_50"
    assert heldout_done["decision"]["passed"] is True
    assert heldout_done["status"]["valid_arms"] == 100
    assert len(heldout.attempts.records()) == 100
    assert CampaignStore(root).state()["phase"] == CampaignPhase.PROMOTE

    # Fault injection: promotion ledger was fsynced, then the process crashed
    # before updating state/current bundle or materializing the bundle copy.
    interrupted_promotion = {
        "promotion_id": f"promote-{candidate.sha256[:20]}",
        "candidate_sha256": candidate.sha256,
        "parent_sha256": candidate.parent_sha256,
        "generation": 0,
        "promoted_at": "2026-08-08T00:00:00+00:00",
        "gate_decision_ids": sorted(
            row["decision_id"] for row in CampaignStore(root).gates.records()
        ),
    }
    CampaignStore(root).promotions.append(interrupted_promotion)
    promotion_step = supervisor.step()
    assert promotion_step["action"] == "promoted_and_spawned_generation"
    continuation = promotion_step
    assert continuation["promotion"] == interrupted_promotion
    child = CampaignStore(child_root).manifest()
    assert child.generation == 1
    assert child.parent_bundle_sha256 == candidate.sha256
    assert child.active_bundle_sha256 == candidate.sha256
    assert child.runtime["rollout_requires_api"] is True
    assert child.runtime["candidate_rollout_requires_api"] is True
    assert child.expected_rollouts == 10
    assert child.heldout_seeds == heldout_seeds
    assert child.policy_rng_by_seed[str(heldout_seeds[0])] == (
        store.manifest().policy_rng_by_seed[str(heldout_seeds[0])]
    )
    assert set(child.rollout_seeds).isdisjoint({*rollout_seeds, *heldout_seeds})
    assert continuation["enqueue"]["enqueued"] == 10
    assert CampaignStore(root).state()["phase"] == CampaignPhase.COMPLETE
    assert len(CampaignStore(root).promotions.records()) == 1

    # A supervisor that remains pointed at the completed parent must follow
    # the immutable continuation and supervise the child without an external
    # process or provider-session restart.
    continued = supervisor.step()
    assert continued["action"] == "continued_child_generation"
    assert continued["child"]["action"] == "waiting_for_rollouts"
    assert continued["child"]["enqueue"]["enqueued"] == 0
    assert continued["continuation"]["child_manifest_sha256"] == child.sha256

    resumed = promote_and_spawn_generation(
        campaign_root=root,
        next_campaign_root=child_root,
        next_queue_root=child_queue,
        worker_hosts=hosts,
        next_master_seed=987654,
    )
    assert resumed["child_manifest"] == continuation["child_manifest"]
    assert resumed["enqueue"]["enqueued"] == 0
    assert len(CampaignStore(root).promotions.records()) == 1
    with pytest.raises(FileExistsError, match="immutable artifact already differs"):
        promote_and_spawn_generation(
            campaign_root=root,
            next_campaign_root=child_root,
            next_queue_root=child_queue,
            worker_hosts=hosts,
            next_master_seed=123456,
        )
    assert len(CampaignStore(root).promotions.records()) == 1


def _authorize_one_of_n(
    root: Path,
    *,
    reason: str = "baseline target has zero successes; require one attributed rescue",
) -> dict:
    return authorize_same_seed_threshold_override(
        campaign_root=root,
        minimum_same_seed_successes=1,
        skip_regression=True,
        reason=reason,
        deadline="2026-08-10T15:00:00+08:00",
        author="codex/task5",
    )


def test_same_seed_threshold_override_is_idempotent_and_one_of_n_passes(
    tmp_path: Path,
) -> None:
    seeds = (11, 12, 13, 14)
    root, queue_root, store, candidate = _setup(
        tmp_path,
        seeds=seeds,
        reuse_parent_evidence=True,
    )
    runner = CandidateGateRunner(
        campaign_root=root,
        queue_root=queue_root,
        worker_hosts=("host-a",),
    )
    started = runner.run_once()
    assert started["parent_adopted"] == len(seeds)
    plan_before = (runner.plan_path).read_bytes()
    gates_before = store.gates.path.read_bytes() if store.gates.path.exists() else b""

    first = _authorize_one_of_n(root)
    second = _authorize_one_of_n(root)
    assert second["authorization_sha256"] == first["authorization_sha256"]
    assert len(store.same_seed_threshold_authorizations.records()) == 1
    assert runner.plan_path.read_bytes() == plan_before
    assert (store.gates.path.read_bytes() if store.gates.path.exists() else b"") == gates_before
    assert not (root / "candidates" / candidate.sha256 / "gates" / "heldout").exists()

    queue = SharedHostQueue(queue_root)
    pending = _pending_jobs(queue)
    first_seed = seeds[0]
    _finish_pending(
        queue,
        ("host-a",),
        lambda job: _gate_episode(job, success=job.seed == first_seed),
    )
    result = runner.run_once()
    assert len(pending) == len(seeds)
    assert result["decision"]["passed"] is True
    assert result["decision"]["candidate_successes"] == 1
    assert result["decision"]["parent_successes"] == 0
    assert "1/4 >= 1/4" in result["decision"]["rationale"]
    assert store.state()["phase"] == CampaignPhase.HELDOUT_GATE
    assert len(store.gates.records()) == 1


def test_same_seed_threshold_override_does_not_count_infra_invalid_as_rescue(
    tmp_path: Path,
) -> None:
    root, queue_root, _, candidate = _setup(
        tmp_path,
        seeds=(11, 12, 13),
        reuse_parent_evidence=True,
    )
    runner = CandidateGateRunner(
        campaign_root=root,
        queue_root=queue_root,
        worker_hosts=("host-a",),
    )
    runner.run_once()
    _authorize_one_of_n(root)
    queue = SharedHostQueue(queue_root)
    claimed = queue.claim("host-a", worker_id="infra-invalid")
    assert claimed is not None
    path, job = claimed
    token = queue.claim_token(path)
    invalid = _gate_episode(job, status="infra_invalid")
    queue.finish(
        path,
        success=True,
        result={
            "job_id": job.job_id,
            "logical_id": job.logical_id,
            "attempt_index": job.attempt_index,
            "return_code": 0,
            "watchdog_reason": None,
            "elapsed_s": invalid.elapsed_s,
            "result": invalid.as_dict(),
            "success": True,
            "job": job.as_dict(),
        },
        claim_token=token,
    )
    partial = runner.run_once()
    assert partial["decision"] is None
    assert partial["status"]["infra_invalid"] == 1
    assert partial["status"]["valid_arms"] == 3
    assert len(
        [
            row
            for row in runner.valid.records()
            if row.get("bundle_sha256") == candidate.sha256
        ]
    ) == 0


def test_same_seed_threshold_override_detects_artifact_and_state_sha_tamper(
    tmp_path: Path,
) -> None:
    root, queue_root, store, candidate = _setup(tmp_path, seeds=(11, 12, 13))
    runner = CandidateGateRunner(
        campaign_root=root,
        queue_root=queue_root,
        worker_hosts=("host-a",),
    )
    plan = runner.prepare()
    authorized = _authorize_one_of_n(root)
    store.update_state(same_seed_threshold_authorization_sha256="0" * 64)
    with pytest.raises(ValueError, match="state binding changed"):
        effective_same_seed_gate_pass_rate(
            store,
            candidate_sha256=candidate.sha256,
            plan=plan,
        )

    store.update_state(
        same_seed_threshold_authorization_sha256=authorized[
            "authorization_sha256"
        ]
    )
    path = Path(authorized["path"])
    payload = read_json(path)
    atomic_write_json(
        path,
        {**payload, "reason": "tampered"},
        overwrite=True,
    )
    with pytest.raises(ValueError, match="identity changed|ledger binding changed"):
        effective_same_seed_gate_pass_rate(
            store,
            candidate_sha256=candidate.sha256,
            plan=plan,
        )


def test_same_seed_threshold_override_rejects_stale_plan_and_reuse(
    tmp_path: Path,
) -> None:
    root, queue_root, store, candidate = _setup(tmp_path, seeds=(11, 12, 13))
    runner = CandidateGateRunner(
        campaign_root=root,
        queue_root=queue_root,
        worker_hosts=("host-a",),
    )
    plan = runner.prepare()
    _authorize_one_of_n(root)
    with pytest.raises(ValueError, match="already has a different"):
        _authorize_one_of_n(root, reason="try to reuse authorization with new scope")

    atomic_write_json(
        runner.plan_path,
        {**plan, "same_seed_pass_rate": 0.75},
        overwrite=True,
    )
    with pytest.raises(ValueError, match="gate plan is stale"):
        effective_same_seed_gate_pass_rate(
            store,
            candidate_sha256=candidate.sha256,
            plan=read_json(runner.plan_path),
        )


def test_same_seed_threshold_override_rejects_late_heldout_start(
    tmp_path: Path,
) -> None:
    root, queue_root, _, candidate = _setup(tmp_path)
    runner = CandidateGateRunner(
        campaign_root=root,
        queue_root=queue_root,
        worker_hosts=("host-a",),
    )
    runner.prepare()
    heldout_plan = (
        root / "candidates" / candidate.sha256 / "gates" / "heldout" / "plan.json"
    )
    atomic_write_json(
        heldout_plan,
        {"candidate_sha256": candidate.sha256, "started": True},
        overwrite=False,
    )
    with pytest.raises(ValueError, match="heldout started"):
        _authorize_one_of_n(root)
