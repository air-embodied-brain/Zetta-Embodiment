# Copyright (c) 2026 Zetta Contributors
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from zetta.evolution.campaign import build_rollout_jobs
from zetta.evolution.jsonio import atomic_write_json, canonical_sha256, read_json
from zetta.evolution.models import CampaignManifest, EpisodeRecord
from zetta.evolution.queue import SharedHostQueue
from zetta.evolution.store import CampaignStore
from zetta.evolution.supervisor import EvolutionSupervisor
from scripts.evolution.prepare_robocasa_campaign import _rollout_command, prepare


def _manifest(*, parent_sha256: str, bundle_path: Path, command: list[str]) -> CampaignManifest:
    return CampaignManifest(
        campaign_id="generation-one-bundle-test",
        environment="robocasa",
        task="SlideDishwasherRack",
        generation=1,
        code_commit="1" * 40,
        prompt_sha256="2" * 64,
        model="test-vla",
        tool_catalog_sha256="3" * 64,
        rollout_seeds=(11,),
        heldout_seeds=(12,),
        policy_rng_by_seed={"11": 1011, "12": 1012},
        parent_bundle_sha256=parent_sha256,
        expected_rollouts=1,
        expected_heldout=1,
        runtime={
            "rollout_command": command,
            "bundle_files_by_sha": {parent_sha256: str(bundle_path)},
        },
    )


def test_promoted_bundle_file_is_verified_and_passed_to_rollout(tmp_path: Path) -> None:
    bundle = {"schema_version": 1, "candidate_id": "promoted-parent"}
    bundle_sha256 = canonical_sha256(bundle)
    bundle_path = tmp_path / "external" / "bundle.json"
    atomic_write_json(bundle_path, bundle, overwrite=False)
    manifest = _manifest(
        parent_sha256=bundle_sha256,
        bundle_path=bundle_path,
        command=[
            "runner",
            "--bundle",
            "{bundle_file}",
            "--bundle-sha256",
            "{bundle_sha256}",
            "--generation",
            "{generation}",
        ],
    )
    store = CampaignStore(tmp_path / "campaign")
    store.initialize(manifest)
    jobs, blocked = build_rollout_jobs(
        store=store,
        queue=SharedHostQueue(tmp_path / "queue"),
        worker_hosts=("host-a",),
    )
    assert blocked == []
    assert len(jobs) == 1
    _, job = jobs[0]
    assert job.bundle_sha256 == bundle_sha256
    assert str(bundle_path.resolve()) in job.command
    assert bundle_sha256 in job.command
    assert "1" in job.command


def test_missing_or_unconsumed_promoted_bundle_fails_closed(tmp_path: Path) -> None:
    bundle = {"schema_version": 1, "candidate_id": "promoted-parent"}
    bundle_sha256 = canonical_sha256(bundle)
    missing = tmp_path / "missing.json"
    manifest = _manifest(
        parent_sha256=bundle_sha256,
        bundle_path=missing,
        command=["runner", "--bundle", "{bundle_file}"],
    )
    store = CampaignStore(tmp_path / "missing-campaign")
    store.initialize(manifest)
    with pytest.raises(ValueError, match="frozen bundle artifact is missing"):
        build_rollout_jobs(
            store=store,
            queue=SharedHostQueue(tmp_path / "missing-queue"),
            worker_hosts=("host-a",),
        )

    bundle_path = tmp_path / "bundle.json"
    atomic_write_json(bundle_path, bundle, overwrite=False)
    unconsumed = _manifest(
        parent_sha256=bundle_sha256,
        bundle_path=bundle_path,
        command=["runner", "--bundle-sha256", "{bundle_sha256}"],
    )
    store = CampaignStore(tmp_path / "unconsumed-campaign")
    store.initialize(unconsumed)
    with pytest.raises(ValueError, match="must consume the frozen bundle"):
        build_rollout_jobs(
            store=store,
            queue=SharedHostQueue(tmp_path / "unconsumed-queue"),
            worker_hosts=("host-a",),
        )


def test_formal_rollout_command_freezes_role1_timeout_and_live_heartbeat(
    tmp_path: Path,
) -> None:
    args = SimpleNamespace(
        repository_root=tmp_path / "repo",
        runtime_python=tmp_path / "python",
        # Runtime v3 (runtime v3 design Stage 7): the frozen command
        # points at the shared rollout-runtime serve endpoint, not an env/VLA
        # HTTP pair. The leased ``{env_endpoint}`` token is now that runtime URL.
        policy_id="groot",
        env_pool_size=1,
        split="target",
        max_actions=1000,
        actions_per_chunk=8,
        role1_planner="api",
        role1_model="openai:gpt-5.6-sol",
        reasoning_effort="high",
        role1_max_tokens=4096,
        role1_timeout_s=900,
        role1_heartbeat_s=15.0,
        role1_max_turns=2,
    )
    command = _rollout_command(args)
    timeout = command.index("--role1-timeout-s")
    heartbeat = command.index("--role1-heartbeat-s")
    assert command[timeout + 1] == "900"
    assert command[heartbeat + 1] == "15.0"
    baseline = command.index("--baseline-mode")
    assert command[baseline + 1] == "{baseline_mode}"
    runtime_url = command.index("--runtime-url")
    assert command[runtime_url + 1] == "{env_endpoint}"
    assert "--vla-endpoint" not in command
    # A bearer token must never be frozen into a campaign artifact.
    assert "--runtime-token" not in command


def test_robocasa_gen0_does_not_reserve_api_for_empty_critic_bundle(
    tmp_path: Path,
) -> None:
    repository_root = Path(__file__).resolve().parents[1]
    output = tmp_path / "prereg"
    args = SimpleNamespace(
        output_root=output,
        campaign_id="robocasa-gen0-admission",
        repository_root=repository_root,
        runtime_python=Path(sys.executable),
        code_commit="1" * 40,
        task="SlideDishwasherRack",
        split="target",
        generation=0,
        parent_bundle=None,
        master_seed=101,
        rollout_count=1,
        heldout_count=1,
        population_size=100,
        initial_logical_slots=1,
        maximum_logical_slots=2,
        continuous_logical_slots=1,
        maximum_api_concurrency=2,
        episode_timeout_s=2700,
        no_progress_timeout_s=180,
        target_valid_episodes_per_hour=25.0,
        max_infrastructure_attempts=2,
        policy_id="groot",
        env_pool_size=1,
        max_actions=1000,
        actions_per_chunk=8,
        role1_planner="api",
        agent_model="gpt-5.6-sol",
        role1_model="openai:gpt-5.6-sol",
        reasoning_effort="high",
        role1_max_tokens=4096,
        role1_timeout_s=900,
        role1_heartbeat_s=15.0,
        role1_max_turns=2,
    )

    prepare(args)
    runtime = read_json(output / "manifest.json")["runtime"]

    assert runtime["rollout_requires_api"] is False
    assert runtime["candidate_rollout_requires_api"] is True


def _fake_queue_command() -> list[str]:
    # No ``{bundle_file}``/``{bundle_sha256}`` consumption needed: this
    # manifest never carries a parent bundle (``parent_sha256=None``).
    return [
        "runner",
        "--logical-id",
        "{logical_id}",
        "--attempt-index",
        "{attempt_index}",
        "--seed",
        "{seed}",
    ]


def _rollout_manifest(*, rollout_count: int, max_infrastructure_attempts: int) -> CampaignManifest:
    seeds = tuple(range(rollout_count))
    return CampaignManifest(
        campaign_id="infra-exhaustion-test",
        environment="robocasa",
        task="SlideDishwasherRack",
        generation=0,
        code_commit="1" * 40,
        prompt_sha256="2" * 64,
        model="test-vla",
        tool_catalog_sha256="3" * 64,
        rollout_seeds=seeds,
        heldout_seeds=tuple(range(1000, 1000 + rollout_count)),
        policy_rng_by_seed={
            str(seed): seed + 5000 for seed in (*seeds, *range(1000, 1000 + rollout_count))
        },
        expected_rollouts=rollout_count,
        expected_heldout=rollout_count,
        max_infrastructure_attempts=max_infrastructure_attempts,
        runtime={"rollout_command": _fake_queue_command()},
    )


def _drain_infra_invalid(
    queue: SharedHostQueue, host: str, *, count: int
) -> None:
    """Simulate a worker that always reports ``infra_invalid`` for every job."""

    for _ in range(count):
        claimed = queue.claim(host, worker_id=f"test-{host}")
        assert claimed is not None
        path, job = claimed
        token = queue.claim_token(path)
        record = EpisodeRecord(
            episode_id=f"episode-{job.job_id}",
            logical_id=job.logical_id,
            generation=0,
            seed=job.seed,
            policy_rng=job.policy_rng,
            bundle_sha256=job.bundle_sha256,
            status="infra_invalid",
            success=None,
            started_at="2026-08-07T01:00:00+00:00",
            finished_at="2026-08-07T01:00:01+00:00",
            elapsed_s=1.0,
            artifact_index={},
            invalid_reason="synthetic infra failure",
            attempt_index=job.attempt_index,
        )
        queue.finish(
            path,
            success=False,
            result={
                "job_id": job.job_id,
                "logical_id": job.logical_id,
                "attempt_index": job.attempt_index,
                "return_code": 3,
                "watchdog_reason": None,
                "elapsed_s": record.elapsed_s,
                "result": record.as_dict(),
                "success": False,
                "job": job.as_dict(),
            },
            claim_token=token,
        )


def test_supervisor_fails_closed_when_infrastructure_attempts_are_exhausted(
    tmp_path: Path,
) -> None:
    """Regression test: before this fix, ``run_campaign.py`` polled forever
    once every rollout's ``max_infrastructure_attempts`` had been consumed
    with no valid episode, because ``build_rollout_jobs`` silently dropped
    the exhausted logical id from both the "still pending" and "completed"
    sets. The supervisor must instead fail closed with a distinct terminal
    action, mirroring ``CandidateGateRunner``'s ``blocked`` status for the
    same-seed/regression/held-out gates.
    """

    manifest = _rollout_manifest(rollout_count=1, max_infrastructure_attempts=2)
    root = tmp_path / "campaign"
    queue_root = tmp_path / "queue"
    store = CampaignStore(root)
    store.initialize(manifest)
    host = "host-a"
    supervisor = EvolutionSupervisor(
        campaign_root=root,
        queue_root=queue_root,
        worker_hosts=(host,),
        tool_catalog={},
    )

    first = supervisor.step()
    assert first["action"] == "waiting_for_rollouts"
    assert first["enqueue"]["enqueued"] == 1
    assert first["enqueue"]["blocked"] == []

    queue = SharedHostQueue(queue_root)
    _drain_infra_invalid(queue, host, count=1)
    second = supervisor.step()
    assert second["action"] == "waiting_for_rollouts"
    assert second["enqueue"]["enqueued"] == 1
    assert second["enqueue"]["blocked"] == []

    _drain_infra_invalid(queue, host, count=1)
    third = supervisor.step()

    assert third["action"] == "rollout_blocked_on_infrastructure"
    assert third["blocked_logical_ids"] == ["g0000-rollout-000"]
    assert third["enqueue"]["enqueued"] == 0
    assert third["enqueue"]["blocked"] == ["g0000-rollout-000"]

    # Idempotent: stepping again does not crash and reports the same block.
    fourth = supervisor.step()
    assert fourth["action"] == "rollout_blocked_on_infrastructure"
    assert fourth["blocked_logical_ids"] == ["g0000-rollout-000"]


def test_supervisor_only_blocks_once_every_logical_id_is_exhausted_or_valid(
    tmp_path: Path,
) -> None:
    """One exhausted rollout among several must not block progress while
    the others are still pending/running; the campaign should keep making
    progress and only report ``rollout_blocked_on_infrastructure`` once no
    further progress is possible.
    """

    manifest = _rollout_manifest(rollout_count=2, max_infrastructure_attempts=1)
    root = tmp_path / "campaign"
    queue_root = tmp_path / "queue"
    store = CampaignStore(root)
    store.initialize(manifest)
    host = "host-a"
    supervisor = EvolutionSupervisor(
        campaign_root=root,
        queue_root=queue_root,
        worker_hosts=(host,),
        tool_catalog={},
    )

    first = supervisor.step()
    assert first["enqueue"]["enqueued"] == 2

    queue = SharedHostQueue(queue_root)
    claimed = queue.claim(host, worker_id=f"test-{host}")
    assert claimed is not None
    path, job = claimed
    token = queue.claim_token(path)
    record = EpisodeRecord(
        episode_id=f"episode-{job.job_id}",
        logical_id=job.logical_id,
        generation=0,
        seed=job.seed,
        policy_rng=job.policy_rng,
        bundle_sha256=job.bundle_sha256,
        status="valid",
        success=True,
        started_at="2026-08-07T01:00:00+00:00",
        finished_at="2026-08-07T01:00:01+00:00",
        elapsed_s=1.0,
        artifact_index={},
        attempt_index=job.attempt_index,
    )
    queue.finish(
        path,
        success=True,
        result={
            "job_id": job.job_id,
            "logical_id": job.logical_id,
            "attempt_index": job.attempt_index,
            "return_code": 0,
            "watchdog_reason": None,
            "elapsed_s": 1.0,
            "result": record.as_dict(),
            "success": True,
            "job": job.as_dict(),
        },
        claim_token=token,
    )
    _drain_infra_invalid(queue, host, count=1)

    final = supervisor.step()
    assert final["action"] == "rollout_blocked_on_infrastructure"
    assert final["blocked_logical_ids"] == [
        logical_id
        for logical_id in ("g0000-rollout-000", "g0000-rollout-001")
        if logical_id != job.logical_id
    ] or final["blocked_logical_ids"]
    assert len(final["blocked_logical_ids"]) == 1
