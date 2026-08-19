# Copyright (c) 2026 Zetta Contributors
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from zetta.evolution.jsonio import atomic_write_json
from zetta.evolution.queue import RolloutJob, SharedHostQueue
from zetta.evolution.slot_broker import EnvironmentSlotBroker


def _manifest(path: Path, *, count: int = 3) -> Path:
    atomic_write_json(
        path,
        {
            "schema_version": 2,
            "slots": [
                {
                    "slot": index,
                    "gpu": str(index % 2),
                    "endpoint": f"http://127.0.0.1:{18800 + index}",
                    "generation": 0,
                    "status": "ready",
                }
                for index in range(count)
            ],
        },
        overwrite=False,
    )
    return path


def _free(_endpoint: str) -> dict[str, object]:
    return {"status": "healthy", "write_protocol": {"phase": "FREE"}}


def test_broker_balances_gpus_and_enforces_active_limit(tmp_path: Path) -> None:
    broker = EnvironmentSlotBroker(
        tmp_path / "broker",
        ready_manifest=_manifest(tmp_path / "ready.json", count=4),
        maximum_active_slots=2,
        health_reader=_free,
    )
    first = broker.acquire(owner="worker-a", job_id="job-a", timeout_s=1)
    second = broker.acquire(owner="worker-b", job_id="job-b", timeout_s=1)
    assert {first.slot.gpu, second.slot.gpu} == {"0", "1"}
    assert broker.snapshot()["active"] == 2
    with pytest.raises(TimeoutError):
        broker.acquire(owner="worker-c", job_id="job-c", timeout_s=0.02, poll_s=0.005)
    first.release()
    third = broker.acquire(owner="worker-c", job_id="job-c", timeout_s=1)
    assert third.slot.slot == first.slot.slot
    second.release()
    third.release()
    assert broker.snapshot()["active"] == 0


def test_expired_lease_is_reclaimed_only_when_server_is_free(tmp_path: Path) -> None:
    phases: dict[str, str] = {}

    def health(endpoint: str) -> dict[str, object]:
        return {
            "status": "healthy",
            "write_protocol": {"phase": phases.get(endpoint, "FREE")},
        }

    root = tmp_path / "broker"
    broker = EnvironmentSlotBroker(
        root,
        ready_manifest=_manifest(tmp_path / "ready.json", count=1),
        lease_s=1,
        health_reader=health,
    )
    first = broker.acquire(owner="worker-a", job_id="job-a", timeout_s=1)
    lease_path = root / "leases" / "slot-000.json"
    payload = json.loads(lease_path.read_text(encoding="utf-8"))
    payload["expires_at"] = (
        datetime.now(timezone.utc) - timedelta(seconds=5)
    ).isoformat()
    atomic_write_json(lease_path, payload, overwrite=True)

    phases[first.endpoint] = "EPISODE_ACTIVE"
    with pytest.raises(TimeoutError):
        broker.acquire(owner="worker-b", job_id="job-b", timeout_s=0.02, poll_s=0.005)
    phases[first.endpoint] = "FREE"
    replacement = broker.acquire(owner="worker-b", job_id="job-b", timeout_s=1)
    assert replacement.lease_id != first.lease_id
    with pytest.raises(RuntimeError, match="stale"):
        first.heartbeat()
    replacement.release()


def test_broker_rejects_nonlocal_environment_endpoints(tmp_path: Path) -> None:
    manifest = tmp_path / "ready.json"
    atomic_write_json(
        manifest,
        {
            "slots": [
                {
                    "slot": 0,
                    "gpu": "0",
                    "endpoint": "http://remote.invalid:18800",
                    "status": "ready",
                }
            ]
        },
    )
    broker = EnvironmentSlotBroker(
        tmp_path / "broker", ready_manifest=manifest, health_reader=_free
    )
    with pytest.raises(ValueError, match="host-local"):
        broker.acquire(owner="worker", job_id="job", timeout_s=0.1)


def test_shared_queue_round_robins_task_and_candidate_domains(tmp_path: Path) -> None:
    queue = SharedHostQueue(tmp_path / "queue")

    def job(index: int, task: str, bundle: str | None) -> RolloutJob:
        output = tmp_path / f"output-{index}"
        return RolloutJob(
            job_id=f"job-{index}",
            campaign_root=str(tmp_path / "campaign"),
            logical_id=f"logical-{index}",
            attempt_index=0,
            task=task,
            seed=index,
            policy_rng=100 + index,
            bundle_sha256=bundle,
            command=("unused",),
            output_dir=str(output),
            result_file=str(output / "result.json"),
            heartbeat_file=str(output / "heartbeat.jsonl"),
        )

    for value in (
        job(0, "A", None),
        job(1, "A", None),
        job(2, "B", "b" * 64),
        job(3, "B", "b" * 64),
    ):
        queue.enqueue("host", value)
    claimed = [queue.claim("host", worker_id=f"w-{index}") for index in range(4)]
    tasks = [item[1].task for item in claimed if item is not None]
    assert tasks == ["A", "B", "A", "B"]
