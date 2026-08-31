# Copyright (c) 2026 Zetta Contributors
"""Fault-injection tests for leased rollout queue recovery."""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import pytest

import zetta.evolution.queue as queue_module
from zetta.evolution.campaign import _known_job_ids, ingest_queue_results
from zetta.evolution.cli import main as evolution_cli_main
from zetta.evolution.jsonio import atomic_write_json, read_json
from zetta.evolution.models import CampaignManifest, EpisodeRecord
from zetta.evolution.queue import RolloutJob, SharedHostQueue, run_worker
from zetta.evolution.store import CampaignStore


def _job(tmp_path: Path, *, job_id: str = "job-authoritative") -> RolloutJob:
    output = tmp_path / "output"
    return RolloutJob(
        job_id=job_id,
        campaign_root=str(tmp_path / "campaign"),
        logical_id="g0000-rollout-000",
        attempt_index=0,
        task="SlideDishwasherRack",
        seed=7,
        policy_rng=107,
        bundle_sha256=None,
        command=("unused",),
        output_dir=str(output),
        result_file=str(output / "episode_record.json"),
        heartbeat_file=str(output / "heartbeat.jsonl"),
        requires_api=False,
    )


def _claim(
    queue: SharedHostQueue, job: RolloutJob
) -> tuple[Path, str, dict[str, object]]:
    queue.enqueue("host-a", job)
    claimed_job = queue.claim("host-a", worker_id="worker-a", lease_s=60)
    assert claimed_job is not None
    claimed, decoded = claimed_job
    assert decoded == job
    envelope = read_json(claimed)
    return claimed, str(envelope["claim_token"]), envelope


def _result(job: RolloutJob, *, marker: str = "first") -> dict[str, object]:
    return {
        "job_id": job.job_id,
        "logical_id": job.logical_id,
        "attempt_index": job.attempt_index,
        "success": True,
        "marker": marker,
        "job": job.as_dict(),
    }


def _initialize_ingestion_campaign(
    root: Path,
    *,
    campaign_id: str,
    seed: int = 7,
    policy_rng: int = 107,
) -> None:
    heldout_seed = seed + 100
    CampaignStore(root).initialize(
        CampaignManifest(
            campaign_id=campaign_id,
            environment="robocasa",
            task="SlideDishwasherRack",
            generation=0,
            code_commit="1" * 40,
            prompt_sha256="2" * 64,
            model="test",
            tool_catalog_sha256="3" * 64,
            rollout_seeds=(seed,),
            heldout_seeds=(heldout_seed,),
            policy_rng_by_seed={str(seed): policy_rng, str(heldout_seed): 999},
            expected_rollouts=1,
            expected_heldout=1,
        )
    )


def _make_stale(path: Path) -> None:
    stale = time.time() - 120
    os.utime(path, (stale, stale))


def test_job_identity_and_claim_metadata_come_from_json_not_filename(
    tmp_path: Path,
) -> None:
    queue = SharedHostQueue(tmp_path / "queue")
    job = _job(tmp_path)
    pending = queue.enqueue("host-a", job)
    misleading = pending.with_name("this-stem-is-not-the-job-id.json")
    pending.rename(misleading)

    assert queue.known_job_ids() == {job.job_id}
    assert _known_job_ids(queue) == {job.job_id}
    claimed_job = queue.claim("host-a", worker_id="worker-explicit", lease_s=90)
    assert claimed_job is not None
    claimed, decoded = claimed_job
    assert decoded.job_id == job.job_id

    envelope = read_json(claimed)
    assert envelope["kind"] == "rollout_claim"
    assert envelope["job_id"] == job.job_id
    assert envelope["worker_id"] == "worker-explicit"
    assert envelope["claim_token"]
    acquired = datetime.fromisoformat(envelope["acquired_at"])
    heartbeat = datetime.fromisoformat(envelope["heartbeat"])
    expires = datetime.fromisoformat(envelope["expires_at"])
    assert acquired == heartbeat
    assert (expires - heartbeat).total_seconds() == pytest.approx(90)


def test_queue_scans_tolerate_concurrent_atomic_rename(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    queue = SharedHostQueue(tmp_path / "queue")
    first = _job(tmp_path, job_id="job-moving")
    second = _job(tmp_path, job_id="job-stable")
    moving = queue.enqueue("host-a", first)
    queue.enqueue("host-a", second)
    real_read_json = queue_module.read_json
    vanished = False

    def racing_read(path: Path) -> object:
        nonlocal vanished
        if Path(path) == moving and not vanished:
            vanished = True
            moving.unlink()
            raise FileNotFoundError(moving)
        return real_read_json(path)

    monkeypatch.setattr(queue_module, "read_json", racing_read)
    assert queue.known_job_ids() == {second.job_id}

    # terminal_envelope_paths has the same glob/open race contract.
    claimed, token, _ = _claim(queue, second)
    terminal = queue.finish(
        claimed,
        success=True,
        result=_result(second),
        claim_token=token,
    )
    vanished = False

    def terminal_race(path: Path) -> object:
        nonlocal vanished
        if Path(path) == terminal and not vanished:
            vanished = True
            terminal.unlink()
            raise FileNotFoundError(terminal)
        return real_read_json(path)

    monkeypatch.setattr(queue_module, "read_json", terminal_race)
    assert queue.terminal_envelope_paths() == []


def test_persistent_worker_retries_transient_claim_lock_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class _TransientQueue:
        calls = 0

        def __init__(self, _root: str | Path) -> None:
            pass

        def claim(self, _host: str, *, worker_id: str | None = None):
            del worker_id
            type(self).calls += 1
            if type(self).calls == 1:
                raise TimeoutError("shared claim lock is temporarily busy")
            return None

    monkeypatch.setattr(queue_module, "SharedHostQueue", _TransientQueue)
    monkeypatch.setattr(queue_module.time, "sleep", lambda _seconds: None)

    assert (
        run_worker(
            queue_root=tmp_path / "queue",
            host="host-a",
            poll_s=0.01,
            once=True,
        )
        == 0
    )
    assert _TransientQueue.calls == 2


def test_pending_claim_envelope_left_before_rename_gets_fresh_fencing_token(
    tmp_path: Path,
) -> None:
    queue = SharedHostQueue(tmp_path / "queue")
    job = _job(tmp_path)
    pending = queue.enqueue("host-a", job)
    stranded = queue._claim_envelope(
        job,
        claim_token="never-owned-token",
        worker_id="crashed-before-rename",
        acquired_at=datetime.now().astimezone(),
        lease_s=60,
    )
    atomic_write_json(pending, stranded, overwrite=True)

    claimed_job = queue.claim("host-a", worker_id="replacement-worker")
    assert claimed_job is not None
    claimed, _ = claimed_job
    envelope = read_json(claimed)
    assert envelope["claim_token"] != "never-owned-token"
    assert envelope["worker_id"] == "replacement-worker"


def test_stale_token_is_rejected_and_matching_finish_is_digest_idempotent(
    tmp_path: Path,
) -> None:
    queue = SharedHostQueue(tmp_path / "queue")
    job = _job(tmp_path)
    claimed, token, envelope = _claim(queue, job)

    rotated = dict(envelope)
    rotated["claim_token"] = "replacement-token"
    atomic_write_json(claimed, rotated, overwrite=True)
    with pytest.raises(ValueError, match="stale claim token"):
        queue.finish(
            claimed,
            success=True,
            result=_result(job),
            claim_token=token,
        )

    terminal = queue.finish(
        claimed,
        success=True,
        result=_result(job),
        claim_token="replacement-token",
    )
    assert not claimed.exists()
    assert read_json(terminal)["kind"] == "rollout_terminal"
    assert list(terminal.parent.glob("*.result.json")) == []

    repeated = queue.finish(
        claimed,
        success=True,
        result=_result(job),
        claim_token="replacement-token",
    )
    assert repeated == terminal
    with pytest.raises(ValueError, match="result digest differs"):
        queue.finish(
            claimed,
            success=True,
            result=_result(job, marker="different"),
            claim_token="replacement-token",
        )
    assert len(list(terminal.parent.glob("*.json"))) == 1


def test_expired_claim_cannot_be_renewed_or_finished_by_old_worker(
    tmp_path: Path,
) -> None:
    queue = SharedHostQueue(tmp_path / "queue")
    job = _job(tmp_path)
    queue.enqueue("host-a", job)
    claimed_job = queue.claim("host-a", worker_id="short-lived", lease_s=0.01)
    assert claimed_job is not None
    claimed, _ = claimed_job
    token = queue.claim_token(claimed)
    time.sleep(0.02)

    with pytest.raises(ValueError, match="expired claim token"):
        queue.heartbeat(claimed, claim_token=token)
    with pytest.raises(ValueError, match="expired claim token"):
        queue.finish(
            claimed,
            success=True,
            result=_result(job),
            claim_token=token,
        )
    assert queue.recover_abandoned("host-a", stale_after_s=10_000) == 1


def test_claim_after_crash_closes_once_and_fences_late_worker(
    tmp_path: Path,
) -> None:
    queue = SharedHostQueue(tmp_path / "queue")
    job = _job(tmp_path)
    claimed, token, _ = _claim(queue, job)
    _make_stale(claimed)

    assert queue.recover_abandoned("host-a", stale_after_s=30) == 1
    assert queue.recover_abandoned("host-a", stale_after_s=30) == 0
    assert queue.counts() == {
        "pending": 0,
        "running": 0,
        "completed": 0,
        "failed": 1,
    }
    terminal = next((queue.root / "failed" / "host-a").glob("*.json"))
    assert read_json(terminal)["commit_source"] == "recovery"
    with pytest.raises(ValueError, match="different authority"):
        queue.finish(
            claimed,
            success=False,
            result=read_json(terminal)["result"],
            claim_token=token,
        )


def test_recover_abandoned_cli_closes_stale_claim_idempotently(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    queue = SharedHostQueue(tmp_path / "queue")
    job = _job(tmp_path)
    claimed, _, _ = _claim(queue, job)
    _make_stale(claimed)
    command = [
        "zetta.evolution.cli",
        "recover-abandoned",
        "--queue-root",
        str(queue.root),
        "--host",
        "host-a",
        "--stale-after-s",
        "30",
    ]

    monkeypatch.setattr(sys, "argv", command)
    assert evolution_cli_main() == 0
    first = json.loads(capsys.readouterr().out)
    assert first == {
        "host": "host-a",
        "recovered": 1,
        "queue": {
            "pending": 0,
            "running": 0,
            "completed": 0,
            "failed": 1,
        },
    }

    monkeypatch.setattr(sys, "argv", command)
    assert evolution_cli_main() == 0
    second = json.loads(capsys.readouterr().out)
    assert second["recovered"] == 0
    assert second["queue"] == first["queue"]


def test_published_episode_result_is_adopted_after_worker_crash(
    tmp_path: Path,
) -> None:
    queue = SharedHostQueue(tmp_path / "queue")
    job = _job(tmp_path)
    claimed, token, _ = _claim(queue, job)
    episode = {
        "episode_id": "episode-1",
        "logical_id": job.logical_id,
        "attempt_index": job.attempt_index,
        "status": "valid",
        "success": False,
    }
    atomic_write_json(job.result_file, episode, overwrite=False)
    _make_stale(claimed)

    assert queue.recover_abandoned("host-a", stale_after_s=30) == 1
    terminal = next((queue.root / "completed" / "host-a").glob("*.json"))
    payload = read_json(terminal)
    assert payload["commit_source"] == "recovery"
    assert payload["result"]["recovery_reason"] == (
        "published_result_file_after_worker_crash"
    )
    assert payload["result"]["result"] == episode
    with pytest.raises(ValueError, match="different authority"):
        queue.finish(
            claimed,
            success=True,
            result=payload["result"],
            claim_token=token,
        )


def test_terminal_publish_before_running_cleanup_recovers_without_recommit(
    tmp_path: Path,
) -> None:
    queue = SharedHostQueue(tmp_path / "queue")
    job = _job(tmp_path)
    claimed, token, claim_envelope = _claim(queue, job)
    terminal = queue.finish(
        claimed,
        success=True,
        result=_result(job),
        claim_token=token,
    )
    terminal_before = terminal.read_bytes()

    # Fault injection: terminal publication succeeded, process died before
    # removing the running claim.
    atomic_write_json(claimed, claim_envelope, overwrite=False)
    assert queue.counts()["running"] == 1
    assert queue.recover_abandoned("host-a", stale_after_s=10_000) == 1
    assert not claimed.exists()
    assert terminal.read_bytes() == terminal_before
    assert queue.counts()["completed"] == 1


def test_legacy_result_sidecar_is_recovered_into_single_v2_terminal(
    tmp_path: Path,
) -> None:
    queue = SharedHostQueue(tmp_path / "queue")
    job = _job(tmp_path)
    claimed, _, _ = _claim(queue, job)
    legacy_dir = queue.root / "completed" / "host-a"
    legacy_dir.mkdir(parents=True, exist_ok=True)
    sidecar = legacy_dir / claimed.with_suffix(".result.json").name
    legacy_result = _result(job, marker="legacy-sidecar")
    atomic_write_json(sidecar, legacy_result, overwrite=False)
    _make_stale(claimed)

    assert queue.recover_abandoned("host-a", stale_after_s=30) == 1
    terminals = [
        path
        for path in legacy_dir.glob("*.json")
        if read_json(path).get("kind") == "rollout_terminal"
    ]
    assert len(terminals) == 1
    assert read_json(terminals[0])["result"] == read_json(sidecar)
    assert not claimed.exists()


def test_campaign_ingests_v2_terminal_once(tmp_path: Path) -> None:
    campaign_root = tmp_path / "campaign"
    manifest = CampaignManifest(
        campaign_id="queue-ingestion",
        environment="robocasa",
        task="SlideDishwasherRack",
        generation=0,
        code_commit="1" * 40,
        prompt_sha256="2" * 64,
        model="test",
        tool_catalog_sha256="3" * 64,
        rollout_seeds=(7,),
        heldout_seeds=(8,),
        policy_rng_by_seed={"7": 107, "8": 108},
        expected_rollouts=1,
        expected_heldout=1,
    )
    CampaignStore(campaign_root).initialize(manifest)
    queue = SharedHostQueue(tmp_path / "queue")
    job = _job(tmp_path)
    claimed, token, _ = _claim(queue, job)
    episode = EpisodeRecord(
        episode_id="episode-1",
        logical_id=job.logical_id,
        generation=0,
        seed=job.seed,
        policy_rng=job.policy_rng,
        bundle_sha256=None,
        status="valid",
        success=False,
        started_at="2026-08-07T00:00:00+00:00",
        finished_at="2026-08-07T00:00:01+00:00",
        elapsed_s=1.0,
        artifact_index={"trajectory": "trajectory.jsonl"},
        attempt_index=0,
    )
    worker_result = _result(job)
    worker_result["result"] = episode.as_dict()
    queue.finish(
        claimed,
        success=True,
        result=worker_result,
        claim_token=token,
    )

    assert ingest_queue_results(campaign_root=campaign_root, queue_root=queue.root) == {
        "accepted": 1,
        "infra_invalid": 0,
        "invalid_envelopes": 0,
    }
    assert ingest_queue_results(campaign_root=campaign_root, queue_root=queue.root) == {
        "accepted": 0,
        "infra_invalid": 0,
        "invalid_envelopes": 0,
    }


@pytest.mark.parametrize(
    ("foreign_seed", "foreign_policy_rng"),
    [
        pytest.param(11, 111, id="different-seed-and-rng"),
        pytest.param(7, 107, id="overlapping-logical-seed-and-rng"),
    ],
)
def test_campaign_ingest_isolates_two_campaigns_sharing_one_queue(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    foreign_seed: int,
    foreign_policy_rng: int,
) -> None:
    campaign_a = tmp_path / "campaign-a" / "campaign"
    campaign_b = tmp_path / "campaign-b" / "campaign"
    for root, campaign_id, seed, policy_rng in (
        (campaign_a, "shared-queue-a", 7, 107),
        (campaign_b, "shared-queue-b", foreign_seed, foreign_policy_rng),
    ):
        _initialize_ingestion_campaign(
            root,
            campaign_id=campaign_id,
            seed=seed,
            policy_rng=policy_rng,
        )

    queue = SharedHostQueue(tmp_path / "shared-queue")
    jobs = (
        _job(tmp_path / "campaign-a", job_id="job-campaign-a"),
        RolloutJob(
            **{
                **_job(tmp_path / "campaign-b", job_id="job-campaign-b").as_dict(),
                "seed": foreign_seed,
                "policy_rng": foreign_policy_rng,
            }
        ),
    )
    for job, episode_id in zip(jobs, ("episode-a", "episode-b"), strict=True):
        claimed, token, _ = _claim(queue, job)
        episode = EpisodeRecord(
            episode_id=episode_id,
            logical_id=job.logical_id,
            generation=0,
            seed=job.seed,
            policy_rng=job.policy_rng,
            bundle_sha256=None,
            status="valid",
            success=False,
            started_at="2026-08-07T00:00:00+00:00",
            finished_at="2026-08-07T00:00:01+00:00",
            elapsed_s=1.0,
            artifact_index={"trajectory": f"{episode_id}.jsonl"},
            attempt_index=0,
        )
        worker_result = _result(job)
        worker_result["result"] = episode.as_dict()
        queue.finish(
            claimed,
            success=True,
            result=worker_result,
            claim_token=token,
        )

    original_record_episode = CampaignStore.record_episode
    campaign_a_record_calls: list[str] = []

    def record_episode_for_bound_campaign_only(
        store: CampaignStore, record: EpisodeRecord
    ) -> bool:
        if store.root.resolve() == campaign_a.resolve():
            campaign_a_record_calls.append(record.episode_id)
        return original_record_episode(store, record)

    monkeypatch.setattr(
        CampaignStore, "record_episode", record_episode_for_bound_campaign_only
    )
    assert ingest_queue_results(campaign_root=campaign_a, queue_root=queue.root) == {
        "accepted": 1,
        "infra_invalid": 0,
        "invalid_envelopes": 0,
    }
    assert campaign_a_record_calls == ["episode-a"]
    assert [
        row["episode_id"] for row in CampaignStore(campaign_a).episodes.records()
    ] == ["episode-a"]

    assert ingest_queue_results(campaign_root=campaign_b, queue_root=queue.root) == {
        "accepted": 1,
        "infra_invalid": 0,
        "invalid_envelopes": 0,
    }
    assert [
        row["episode_id"] for row in CampaignStore(campaign_b).episodes.records()
    ] == ["episode-b"]


def test_campaign_ingest_does_not_count_foreign_infra_invalid(
    tmp_path: Path,
) -> None:
    campaign_a = tmp_path / "campaign-a" / "campaign"
    campaign_b = tmp_path / "campaign-b" / "campaign"
    for root, campaign_id in (
        (campaign_a, "shared-infra-a"),
        (campaign_b, "shared-infra-b"),
    ):
        _initialize_ingestion_campaign(root, campaign_id=campaign_id)

    queue = SharedHostQueue(tmp_path / "shared-queue")
    foreign_job = _job(tmp_path / "campaign-b", job_id="job-foreign-infra")
    claimed, token, _ = _claim(queue, foreign_job)
    worker_result = _result(foreign_job)
    worker_result.update(
        {
            "success": False,
            "result": None,
            "watchdog_reason": "episode_no_progress_timeout",
            "elapsed_s": 12.5,
        }
    )
    queue.finish(
        claimed,
        success=False,
        result=worker_result,
        claim_token=token,
    )
    atomic_write_json(
        queue.root / "completed" / "host-a" / "malformed-foreign-terminal.json",
        {
            "kind": "rollout_terminal",
            "job": foreign_job.as_dict(),
            "result": None,
        },
        overwrite=False,
    )

    assert ingest_queue_results(campaign_root=campaign_a, queue_root=queue.root) == {
        "accepted": 0,
        "infra_invalid": 0,
        "invalid_envelopes": 0,
    }
    assert CampaignStore(campaign_a).attempts.records() == []
    assert CampaignStore(campaign_a).episodes.records() == []

    assert ingest_queue_results(campaign_root=campaign_b, queue_root=queue.root) == {
        "accepted": 0,
        "infra_invalid": 1,
        "invalid_envelopes": 1,
    }
    assert len(CampaignStore(campaign_b).attempts.records()) == 1
    assert CampaignStore(campaign_b).episodes.records() == []


def test_campaign_reingests_infra_invalid_terminal_idempotently(tmp_path: Path) -> None:
    campaign_root = tmp_path / "campaign"
    manifest = CampaignManifest(
        campaign_id="queue-infra-ingestion",
        environment="robocasa",
        task="SlideDishwasherRack",
        generation=0,
        code_commit="1" * 40,
        prompt_sha256="2" * 64,
        model="test",
        tool_catalog_sha256="3" * 64,
        rollout_seeds=(7,),
        heldout_seeds=(8,),
        policy_rng_by_seed={"7": 107, "8": 108},
        expected_rollouts=1,
        expected_heldout=1,
    )
    store = CampaignStore(campaign_root)
    store.initialize(manifest)
    queue = SharedHostQueue(tmp_path / "queue")
    job = _job(tmp_path)
    claimed, token, claim = _claim(queue, job)
    worker_result = _result(job)
    worker_result.update(
        {
            "success": False,
            "result": None,
            "watchdog_reason": "episode_no_progress_timeout",
            "elapsed_s": 12.5,
        }
    )
    terminal = queue.finish(
        claimed,
        success=False,
        result=worker_result,
        claim_token=token,
    )
    terminal_payload = read_json(terminal)

    assert ingest_queue_results(campaign_root=campaign_root, queue_root=queue.root) == {
        "accepted": 0,
        "infra_invalid": 1,
        "invalid_envelopes": 0,
    }
    assert ingest_queue_results(campaign_root=campaign_root, queue_root=queue.root) == {
        "accepted": 0,
        "infra_invalid": 1,
        "invalid_envelopes": 0,
    }
    attempt = store.attempts.records()[0]
    assert attempt["started_at"] == claim["acquired_at"]
    assert attempt["finished_at"] == terminal_payload["finished_at"]


def test_campaign_accepts_only_timestamp_drift_from_legacy_infra_record(
    tmp_path: Path,
) -> None:
    campaign_root = tmp_path / "campaign"
    manifest = CampaignManifest(
        campaign_id="queue-legacy-infra-ingestion",
        environment="robocasa",
        task="SlideDishwasherRack",
        generation=0,
        code_commit="1" * 40,
        prompt_sha256="2" * 64,
        model="test",
        tool_catalog_sha256="3" * 64,
        rollout_seeds=(7,),
        heldout_seeds=(8,),
        policy_rng_by_seed={"7": 107, "8": 108},
        expected_rollouts=1,
        expected_heldout=1,
    )
    store = CampaignStore(campaign_root)
    store.initialize(manifest)
    queue = SharedHostQueue(tmp_path / "queue")
    job = _job(tmp_path)
    claimed, token, _ = _claim(queue, job)
    worker_result = _result(job)
    worker_result.update(
        {
            "success": False,
            "result": None,
            "watchdog_reason": "episode_no_progress_timeout",
            "elapsed_s": 12.5,
        }
    )
    queue.finish(
        claimed,
        success=False,
        result=worker_result,
        claim_token=token,
    )
    store.record_episode(
        EpisodeRecord(
            episode_id=f"infra-{job.job_id}",
            logical_id=job.logical_id,
            generation=0,
            seed=job.seed,
            policy_rng=job.policy_rng,
            bundle_sha256=None,
            status="infra_invalid",
            success=None,
            started_at="2026-08-06T00:00:00+00:00",
            finished_at="2026-08-06T00:00:00+00:00",
            elapsed_s=12.5,
            artifact_index={"worker_envelope": "legacy-path"},
            invalid_reason="episode_no_progress_timeout",
            attempt_index=0,
        )
    )

    assert ingest_queue_results(campaign_root=campaign_root, queue_root=queue.root) == {
        "accepted": 0,
        "infra_invalid": 1,
        "invalid_envelopes": 0,
    }

    artifact = (
        campaign_root / "attempts" / job.logical_id / "attempt-000" / "record.json"
    )
    stored = read_json(artifact)
    stored["invalid_reason"] = "different-failure"
    artifact.unlink()
    atomic_write_json(artifact, stored, overwrite=False)
    with pytest.raises(FileExistsError, match="immutable artifact already differs"):
        ingest_queue_results(campaign_root=campaign_root, queue_root=queue.root)
