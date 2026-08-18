# Copyright (c) 2026 RPent Contributors
"""Episode queue integration tests for dynamic provider admission."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any, Callable

import pytest

import rpent.evolution.queue as queue_module
from rpent.evolution.provider_admission import ProviderAdmission
from rpent.evolution.queue import (
    RolloutJob,
    SharedHostQueue,
    run_worker,
    run_worker_pool,
)


def _job(tmp_path: Path, index: int, *, requires_api: bool = True) -> RolloutJob:
    output = tmp_path / f"output-{index}"
    return RolloutJob(
        job_id=f"job-{index}",
        campaign_root=str(tmp_path / "campaign"),
        logical_id=f"logical-{index}",
        attempt_index=0,
        task="SlideDishwasherRack",
        seed=index,
        policy_rng=index + 100,
        bundle_sha256=None,
        command=("unused",),
        output_dir=str(output),
        result_file=str(output / "result.json"),
        heartbeat_file=str(output / "heartbeat.jsonl"),
        requires_api=requires_api,
    )


def _enqueue(tmp_path: Path, *jobs: RolloutJob) -> Path:
    root = tmp_path / "queue"
    queue = SharedHostQueue(root)
    for job in jobs:
        queue.enqueue("host-a", job)
    return root


class _Executor:
    def __init__(
        self,
        run: Callable[[RolloutJob, Callable[[], None] | None], dict[str, Any]],
    ) -> None:
        self._run = run

    def run(
        self,
        job: RolloutJob,
        *,
        progress_callback: Callable[[], None] | None = None,
    ) -> dict[str, Any]:
        return self._run(job, progress_callback)


def _install_executor(
    monkeypatch: pytest.MonkeyPatch,
    run: Callable[[RolloutJob, Callable[[], None] | None], dict[str, Any]],
) -> None:
    monkeypatch.setattr(
        queue_module, "SubprocessRolloutExecutor", lambda: _Executor(run)
    )


def _success(job: RolloutJob) -> dict[str, Any]:
    return {
        "job_id": job.job_id,
        "logical_id": job.logical_id,
        "attempt_index": job.attempt_index,
        "success": True,
        "provider": {"outcome": "success", "tokens": 321, "latency_s": 1.25},
        "job": job.as_dict(),
    }


def test_requires_api_jobs_obey_dynamic_route_capacity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    queue_root = _enqueue(tmp_path, _job(tmp_path, 0), _job(tmp_path, 1))
    provider_root = tmp_path / "provider"
    lock = threading.Lock()
    active = 0
    peak = 0

    def execute(job: RolloutJob, callback: Callable[[], None] | None) -> dict[str, Any]:
        nonlocal active, peak
        if callback is not None:
            callback()
        with lock:
            active += 1
            peak = max(peak, active)
        time.sleep(0.08)
        with lock:
            active -= 1
        return _success(job)

    _install_executor(monkeypatch, execute)
    errors: list[BaseException] = []

    def worker() -> None:
        try:
            assert (
                run_worker(
                    queue_root=queue_root,
                    host="host-a",
                    once=True,
                    provider_admission_root=provider_root,
                    provider_route_id="low-cost",
                    provider_initial_limit=1,
                    provider_max_limit=1,
                )
                == 0
            )
        except BaseException as exc:  # pragma: no cover - reported below
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=4)
        assert not thread.is_alive()

    assert errors == []
    assert peak == 1
    route = ProviderAdmission(provider_root).snapshot("low-cost").routes["low-cost"]
    assert route.active_leases == 0
    assert route.successful_requests == 2
    assert route.rolling_tpm == 642


def test_worker_pool_fills_configured_concurrency_and_stops_cleanly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    jobs = tuple(_job(tmp_path, index, requires_api=False) for index in range(6))
    queue_root = _enqueue(tmp_path, *jobs)
    active = 0
    peak = 0
    lock = threading.Lock()

    def execute(job: RolloutJob, callback: Callable[[], None] | None) -> dict[str, Any]:
        nonlocal active, peak
        if callback is not None:
            callback()
        with lock:
            active += 1
            peak = max(peak, active)
        time.sleep(0.05)
        with lock:
            active -= 1
        return {"success": True, "job": job.as_dict()}

    _install_executor(monkeypatch, execute)
    stop = threading.Event()
    errors: list[BaseException] = []

    def dispatch() -> None:
        try:
            run_worker_pool(
                queue_root=queue_root,
                host="host-a",
                concurrency=3,
                poll_s=0.005,
                stop_event=stop,
            )
        except BaseException as exc:  # pragma: no cover - reported below
            errors.append(exc)

    thread = threading.Thread(target=dispatch)
    thread.start()
    deadline = time.monotonic() + 3
    queue = SharedHostQueue(queue_root)
    while queue.counts()["completed"] < len(jobs) and time.monotonic() < deadline:
        time.sleep(0.01)
    stop.set()
    thread.join(timeout=3)
    assert not thread.is_alive()
    assert errors == []
    assert queue.counts()["completed"] == len(jobs)
    assert peak == 3


def test_missing_metrics_release_lease_and_heartbeat_is_persisted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    queue_root = _enqueue(tmp_path, _job(tmp_path, 0))
    provider_root = tmp_path / "provider"

    def execute(job: RolloutJob, callback: Callable[[], None] | None) -> dict[str, Any]:
        assert callback is not None
        callback()
        time.sleep(0.02)
        callback()
        return {"success": True, "job": job.as_dict()}

    _install_executor(monkeypatch, execute)
    assert (
        run_worker(
            queue_root=queue_root,
            host="host-a",
            once=True,
            provider_admission_root=provider_root,
            provider_route_id="route-a",
            provider_heartbeat_s=0.005,
        )
        == 0
    )

    route = ProviderAdmission(provider_root).snapshot("route-a").routes["route-a"]
    assert route.active_leases == 0
    assert route.successful_requests == 0
    events = (provider_root / "provider-admission-events.jsonl").read_text(
        encoding="utf-8"
    )
    assert '"event":"lease_heartbeat"' in events
    assert '"event":"lease_released"' in events


def test_provider_failure_metrics_are_reported_and_released(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    queue_root = _enqueue(tmp_path, _job(tmp_path, 0))
    provider_root = tmp_path / "provider"

    def execute(
        job: RolloutJob, _callback: Callable[[], None] | None
    ) -> dict[str, Any]:
        return {
            "success": False,
            "result": {
                "provider_metrics": {
                    "outcome": "failure",
                    "failure": "rate_limit",
                    "tokens": 17,
                    "latency_s": 2.5,
                }
            },
            "job": job.as_dict(),
        }

    _install_executor(monkeypatch, execute)
    assert (
        run_worker(
            queue_root=queue_root,
            host="host-a",
            once=True,
            provider_admission_root=provider_root,
            provider_route_id="route-a",
        )
        == 1
    )
    route = ProviderAdmission(provider_root).snapshot("route-a").routes["route-a"]
    assert route.active_leases == 0
    assert route.failed_requests == 1
    assert route.rolling_tpm == 17


def test_executor_exception_fails_closed_releases_and_redacts_message(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    queue_root = _enqueue(tmp_path, _job(tmp_path, 0))
    provider_root = tmp_path / "provider"
    secret = "https://provider.invalid/v1 sk-secret-value"

    def execute(
        _job: RolloutJob, _callback: Callable[[], None] | None
    ) -> dict[str, Any]:
        raise RuntimeError(secret)

    _install_executor(monkeypatch, execute)
    assert (
        run_worker(
            queue_root=queue_root,
            host="host-a",
            once=True,
            provider_admission_root=provider_root,
            provider_route_id="safe-alias",
        )
        == 1
    )

    route = ProviderAdmission(provider_root).snapshot("safe-alias").routes["safe-alias"]
    assert route.active_leases == 0
    envelope = next((queue_root / "failed" / "host-a").glob("*.json"))
    terminal = json.loads(envelope.read_text(encoding="utf-8"))
    persisted = (
        envelope.read_text(encoding="utf-8")
        + "\n"
        + "\n".join(
            path.read_text(encoding="utf-8") for path in provider_root.glob("*.json*")
        )
    )
    assert secret not in persisted
    assert "sk-secret-value" not in persisted
    assert terminal["kind"] == "rollout_terminal"
    assert terminal["result"]["worker_exception"] == "RuntimeError"


def test_non_api_job_ignores_provider_configuration_and_renews_queue_claim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    queue_root = _enqueue(tmp_path, _job(tmp_path, 0, requires_api=False))
    calls: list[tuple[str, bool]] = []

    def execute(job: RolloutJob, callback: Callable[[], None] | None) -> dict[str, Any]:
        calls.append((job.job_id, callback is None))
        return {"success": True, "job": job.as_dict()}

    _install_executor(monkeypatch, execute)
    monkeypatch.setenv("RPENT_PROVIDER_ROUTE_ID", "https://must-not-be-read.invalid")
    assert run_worker(queue_root=queue_root, host="host-a", once=True) == 0
    assert calls == [("job-0", False)]
    assert not (tmp_path / "provider").exists()


def test_legacy_root_and_limit_environment_bridge_to_default_alias(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    queue_root = _enqueue(tmp_path, _job(tmp_path, 0))
    provider_root = tmp_path / "legacy-provider"

    def execute(
        job: RolloutJob, _callback: Callable[[], None] | None
    ) -> dict[str, Any]:
        return _success(job)

    _install_executor(monkeypatch, execute)
    monkeypatch.setenv("RPENT_API_ADMISSION_ROOT", str(provider_root))
    monkeypatch.setenv("RPENT_API_MAX_CONCURRENCY", "2")
    assert run_worker(queue_root=queue_root, host="host-a", once=True) == 0
    route = ProviderAdmission(provider_root).snapshot("default").routes["default"]
    assert route.initial_limit == 2
    assert route.max_limit == 2
    assert route.successful_requests == 1
