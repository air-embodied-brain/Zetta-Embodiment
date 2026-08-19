# Copyright (c) 2026 Zetta Contributors
from __future__ import annotations

import sys
from pathlib import Path

import pytest

from zetta.evolution.jsonio import atomic_write_json
from zetta.evolution.models import CampaignManifest, EpisodeRecord
from zetta.evolution.queue import SharedHostQueue
from zetta.evolution.store import CampaignStore
from scripts.evolution.run_campaign import _parse_args, main


def _required_args() -> list[str]:
    return [
        "--manifest",
        "manifest.json",
        "--root",
        "campaign",
        "--queue-root",
        "queue",
        "--tool-catalog",
        "tool-catalog.json",
        "--workers",
        "libero-gpu1",
    ]


def test_worker_command_consumes_nested_worker_options() -> None:
    args = _parse_args(
        [
            *_required_args(),
            "--worker-command",
            "python",
            "-m",
            "zetta.evolution.cli",
            "worker",
            "--queue-root",
            "{queue_root}",
            "--host",
            "{host}",
            "--poll-s",
            "2",
            "--concurrency",
            "1",
        ]
    )

    assert args.queue_root == Path("queue")
    assert args.worker_command == [
        "python",
        "-m",
        "zetta.evolution.cli",
        "worker",
        "--queue-root",
        "{queue_root}",
        "--host",
        "{host}",
        "--poll-s",
        "2",
        "--concurrency",
        "1",
    ]


def test_worker_command_must_not_be_empty() -> None:
    with pytest.raises(SystemExit):
        _parse_args([*_required_args(), "--worker-command"])


def _drain_infra_invalid(queue: SharedHostQueue, host: str, *, count: int) -> None:
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


def test_run_campaign_exits_four_when_infrastructure_attempts_are_exhausted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """End-to-end regression test for the ``run_campaign.py`` blocking bug:
    before the fix, this scenario polled ``waiting_for_rollouts`` forever
    (``--poll-s`` kept sleeping) because an exhausted logical id was neither
    "pending" nor "completed". It must now exit promptly with a distinct
    code.
    """

    manifest = CampaignManifest(
        campaign_id="run-campaign-infra-exhaustion",
        environment="robocasa",
        task="SlideDishwasherRack",
        generation=0,
        code_commit="1" * 40,
        prompt_sha256="2" * 64,
        model="test-vla",
        tool_catalog_sha256="3" * 64,
        rollout_seeds=(0,),
        heldout_seeds=(1000,),
        policy_rng_by_seed={"0": 5000, "1000": 6000},
        expected_rollouts=1,
        expected_heldout=1,
        max_infrastructure_attempts=1,
        runtime={
            "rollout_command": [
                "runner",
                "--logical-id",
                "{logical_id}",
                "--attempt-index",
                "{attempt_index}",
                "--seed",
                "{seed}",
            ]
        },
    )
    root = tmp_path / "campaign"
    queue_root = tmp_path / "queue"
    tool_catalog_path = tmp_path / "tool-catalog.json"
    manifest_path = tmp_path / "manifest.json"
    atomic_write_json(manifest_path, manifest.as_dict(), overwrite=False)
    atomic_write_json(tool_catalog_path, {}, overwrite=False)
    store = CampaignStore(root)
    store.initialize(manifest)

    host = "host-a"

    def _run_once() -> int:
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "run_campaign.py",
                "--manifest",
                str(manifest_path),
                "--root",
                str(root),
                "--queue-root",
                str(queue_root),
                "--tool-catalog",
                str(tool_catalog_path),
                "--workers",
                host,
                "--once",
            ],
        )
        return main()

    # Step 1: enqueues the sole rollout's only attempt.
    assert _run_once() == 0
    queue = SharedHostQueue(queue_root)
    _drain_infra_invalid(queue, host, count=1)

    # Step 2: ingests the infra_invalid result; every attempt for the only
    # logical id is now exhausted with nothing in flight.
    exit_code = _run_once()

    assert exit_code == 4
