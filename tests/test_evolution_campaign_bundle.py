# Copyright (c) 2026 RPent Contributors
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from rpent.evolution.campaign import build_rollout_jobs
from rpent.evolution.jsonio import atomic_write_json, canonical_sha256, read_json
from rpent.evolution.models import CampaignManifest
from rpent.evolution.queue import SharedHostQueue
from rpent.evolution.store import CampaignStore
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
    jobs = build_rollout_jobs(
        store=store,
        queue=SharedHostQueue(tmp_path / "queue"),
        worker_hosts=("host-a",),
    )
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
        vla_endpoint="http://127.0.0.1:18811",
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


def test_rollout_command_preserves_virtualenv_python_symlink(tmp_path: Path) -> None:
    target = tmp_path / "base-python"
    target.touch()
    entrypoint = tmp_path / "venv" / "bin" / "python"
    entrypoint.parent.mkdir(parents=True)
    try:
        entrypoint.symlink_to(target)
    except OSError:
        pytest.skip("symlinks are unavailable")
    args = SimpleNamespace(
        repository_root=tmp_path / "repo",
        runtime_python=entrypoint,
        vla_endpoint="http://127.0.0.1:18811",
        split="target",
        max_actions=1,
        actions_per_chunk=1,
        role1_planner="api",
        role1_model="openai:gpt-5.6-sol",
        reasoning_effort="high",
        role1_max_tokens=64,
        role1_timeout_s=30,
        role1_heartbeat_s=5.0,
        role1_max_turns=1,
    )

    assert _rollout_command(args)[0] == str(entrypoint.absolute())


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
        vla_endpoint="http://127.0.0.1:18811",
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
