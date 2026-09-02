# Copyright (c) 2026 Zetta Contributors
from __future__ import annotations

import json
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from scripts.evolution.robocasa_capacity_worker import resolve_gpu
from zetta.evolution.capacity import (
    DEFAULT_API_CONCURRENCY,
    DEFAULT_SLOT_LADDER,
    CapacityConfig,
    CapacityRules,
    CommandEnvironmentWorker,
    ConcurrencyTracker,
    FakeEnvironmentWorker,
    MultiEnvironmentCapacityBenchmark,
    OperationResult,
    ensure_secret_free_report,
    write_secret_free_report,
)


def _permissive_rules(**changes: object) -> CapacityRules:
    values = {
        "maximum_infra_invalid_rate": 0.02,
        "maximum_reset_p95_s": 10.0,
        "maximum_step_p95_s": 10.0,
        "maximum_vla_queue_p95_s": 10.0,
        "maximum_gpu_memory_fraction": 1.0,
        "maximum_cpu_percent": 100.0,
        "minimum_valid_throughput_per_hour": 1.0,
    }
    values.update(changes)
    return CapacityRules(**values)  # type: ignore[arg-type]


def test_default_ladder_and_api_limit() -> None:
    config = CapacityConfig()
    assert config.slot_ladder == (1, 2, 4, 8, 16, 32, 50)
    assert config.slot_ladder == DEFAULT_SLOT_LADDER
    assert config.maximum_api_concurrency == DEFAULT_API_CONCURRENCY == 8


def test_real_worker_gpu_map_is_deterministic_and_fail_closed() -> None:
    assert resolve_gpu(slot=0, gpu=None, gpu_map="0,2,7") == "0"
    assert resolve_gpu(slot=4, gpu=None, gpu_map="0,2,7") == "2"
    assert resolve_gpu(slot=9, gpu="3", gpu_map=None) == "3"
    with pytest.raises(ValueError, match="mutually exclusive"):
        resolve_gpu(slot=0, gpu="0", gpu_map="0,1")
    with pytest.raises(ValueError, match="at least one"):
        resolve_gpu(slot=0, gpu=None, gpu_map=",")
    with pytest.raises(ValueError, match="non-negative integers"):
        resolve_gpu(slot=0, gpu=None, gpu_map="0,cuda:1")
    with pytest.raises(ValueError, match="required"):
        resolve_gpu(slot=0, gpu=None, gpu_map=None)


def test_fake_worker_completes_full_ladder_and_records_required_metrics() -> None:
    config = CapacityConfig(
        episodes_per_slot=1,
        steps_per_episode=1,
        warmup_steps=0,
        sample_interval_s=0.001,
        rules=_permissive_rules(),
    )
    benchmark = MultiEnvironmentCapacityBenchmark(
        config,
        lambda slot, slots: FakeEnvironmentWorker(
            slot_index=slot,
            total_slots=slots,
        ),
        worker_mode="fake",
    )

    report = benchmark.run()

    assert report["completed_ladder"] is True
    assert report["recommended_slots"] == 50
    assert [level["slots"] for level in report["levels"]] == list(DEFAULT_SLOT_LADDER)
    final = report["levels"][-1]
    assert final["valid_episodes"] == 50
    assert final["infra_invalid_episodes"] == 0
    assert final["infra_invalid_operations"] == 0
    assert final["warmup_infra_invalid_operations"] == 0
    assert final["valid_throughput_per_hour"] > 0
    assert set(final["reset_latency_s"]) == {"p50", "p95"}
    assert set(final["step_latency_s"]) == {"p50", "p95"}
    assert set(final["vla_queue_latency_s"]) == {"p50", "p95"}
    assert "gpu_memory_peak_mib" in final["system"]
    assert "gpu_memory_peak_single_device_mib" in final["system"]
    assert "gpu_memory_peak_fraction" in final["system"]
    assert "cpu_peak_percent" in final["system"]
    assert "rss_peak_mib" in final["system"]


def test_failed_level_rolls_back_to_last_passing_capacity() -> None:
    config = CapacityConfig(
        slot_ladder=(1, 2, 4, 8, 16),
        episodes_per_slot=1,
        steps_per_episode=1,
        warmup_steps=0,
        sample_interval_s=0.001,
        rules=_permissive_rules(maximum_infra_invalid_rate=0.0),
    )
    benchmark = MultiEnvironmentCapacityBenchmark(
        config,
        lambda slot, slots: FakeEnvironmentWorker(
            slot_index=slot,
            total_slots=slots,
            fail_at_or_above_slots=8,
        ),
        worker_mode="fake",
    )

    report = benchmark.run()

    assert report["recommended_slots"] == 4
    assert report["tested_maximum_slots"] == 8
    assert report["completed_ladder"] is False
    assert report["levels"][-1]["passed"] is False
    assert "infra_invalid_rate" in report["levels"][-1]["decision_reasons"]


def test_api_semaphore_limits_steps_without_limiting_environment_slots() -> None:
    tracker = ConcurrencyTracker()
    config = CapacityConfig(
        slot_ladder=(8,),
        episodes_per_slot=1,
        steps_per_episode=2,
        warmup_steps=0,
        sample_interval_s=0.001,
        maximum_api_concurrency=2,
        api_limited_steps=True,
        rules=_permissive_rules(),
    )
    benchmark = MultiEnvironmentCapacityBenchmark(
        config,
        lambda slot, slots: FakeEnvironmentWorker(
            slot_index=slot,
            total_slots=slots,
            step_delay_s=0.01,
            tracker=tracker,
        ),
        worker_mode="fake",
    )

    report = benchmark.run()

    assert report["recommended_slots"] == 8
    assert tracker.maximum_active <= 2
    assert report["levels"][0]["workers_started"] == 8


def test_environment_operations_are_microbatched_without_reducing_residency() -> None:
    tracker = ConcurrencyTracker()
    progress: list[dict[str, object]] = []
    config = CapacityConfig(
        slot_ladder=(12,),
        episodes_per_slot=1,
        steps_per_episode=1,
        warmup_steps=0,
        sample_interval_s=0.001,
        maximum_active_environment_operations=3,
        rules=_permissive_rules(),
    )
    report = MultiEnvironmentCapacityBenchmark(
        config,
        lambda slot, slots: FakeEnvironmentWorker(
            slot_index=slot,
            total_slots=slots,
            reset_delay_s=0.005,
            step_delay_s=0.005,
            tracker=tracker,
        ),
        worker_mode="fake",
        progress_callback=progress.append,
    ).run()

    level = report["levels"][0]
    assert level["workers_started"] == 12
    assert level["valid_episodes"] == 12
    assert tracker.maximum_active <= 3
    assert level["scheduler_queue_latency_s"]["p95"] > 0
    assert report["config"]["maximum_active_environment_operations"] == 3
    starts = [item for item in progress if item["event"] == "operation_batch_started"]
    completes = [
        item for item in progress if item["event"] == "operation_batch_completed"
    ]
    assert len(starts) == len(completes) == 8
    assert all(int(item["batch_size"]) <= 3 for item in starts)


def test_resource_shards_refill_without_waiting_for_global_batch_tail() -> None:
    lock = threading.Lock()
    active_by_shard = [0, 0]
    maximum_by_shard = [0, 0]
    global_active = 0
    global_maximum = 0
    delays = (0.04, 0.001, 0.001, 0.04, 0.001, 0.001)
    starts: dict[tuple[int, int], float] = {}
    finishes: dict[tuple[int, int], float] = {}

    class ShardedWorker:
        def __init__(self, slot: int) -> None:
            self.slot = slot
            self.calls = 0

        def _run(self) -> OperationResult:
            nonlocal global_active, global_maximum
            shard = self.slot % 2
            call = self.calls
            self.calls += 1
            with lock:
                starts[(self.slot, call)] = time.perf_counter()
                active_by_shard[shard] += 1
                maximum_by_shard[shard] = max(
                    maximum_by_shard[shard], active_by_shard[shard]
                )
                global_active += 1
                global_maximum = max(global_maximum, global_active)
            try:
                time.sleep(delays[self.slot])
                return OperationResult()
            finally:
                with lock:
                    finishes[(self.slot, call)] = time.perf_counter()
                    active_by_shard[shard] -= 1
                    global_active -= 1

        def reset(self, timeout_s: float) -> OperationResult:
            del timeout_s
            return self._run()

        def step(self, timeout_s: float) -> OperationResult:
            del timeout_s
            return self._run()

        def close(self) -> None:
            return None

    common = {
        "slot_ladder": (6,),
        "episodes_per_slot": 1,
        "steps_per_episode": 1,
        "warmup_steps": 0,
        "sample_interval_s": 0.001,
        "maximum_active_environment_operations": 2,
        "rules": _permissive_rules(),
    }
    config = CapacityConfig(
        **common,
        operation_resource_shards=2,
        maximum_active_operations_per_shard=1,
    )
    report = MultiEnvironmentCapacityBenchmark(
        config,
        lambda slot, slots: ShardedWorker(slot),
        worker_mode="fake",
    ).run()
    assert report["levels"][0]["valid_episodes"] == 6
    assert maximum_by_shard == [1, 1]
    assert global_maximum == 2
    # Shard 1's first fast reset must immediately refill with its next slot,
    # without waiting for shard 0's slow reset to finish.
    assert starts[(3, 0)] < finishes[(0, 0)]
    assert report["config"]["operation_resource_shards"] == 2


def test_worker_process_startup_is_parallel_and_reported() -> None:
    lock = threading.Lock()
    active = 0
    peak = 0

    def factory(slot: int, slots: int) -> FakeEnvironmentWorker:
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        time.sleep(0.02)
        with lock:
            active -= 1
        return FakeEnvironmentWorker(slot_index=slot, total_slots=slots)

    config = CapacityConfig(
        slot_ladder=(8,),
        episodes_per_slot=1,
        steps_per_episode=1,
        warmup_steps=0,
        sample_interval_s=0.001,
        worker_startup_concurrency=8,
        rules=_permissive_rules(),
    )
    level = MultiEnvironmentCapacityBenchmark(
        config,
        factory,
        worker_mode="fake",
    ).run()["levels"][0]

    assert peak > 1
    assert level["startup_wall_time_s"] < 0.12
    assert level["wall_time_s"] >= level["startup_wall_time_s"]
    assert level["steady_state_valid_throughput_per_hour"] > 0


def test_worker_start_failure_cannot_be_hidden_by_operation_denominator() -> None:
    config = CapacityConfig(
        slot_ladder=(8,),
        episodes_per_slot=1,
        steps_per_episode=20,
        warmup_steps=0,
        sample_interval_s=0.001,
        rules=_permissive_rules(maximum_infra_invalid_rate=1.0),
    )

    def factory(slot: int, slots: int) -> FakeEnvironmentWorker:
        if slot == 7:
            raise RuntimeError("synthetic startup failure")
        return FakeEnvironmentWorker(slot_index=slot, total_slots=slots)

    report = MultiEnvironmentCapacityBenchmark(
        config,
        factory,
        worker_mode="fake",
    ).run()

    level = report["levels"][0]
    assert level["workers_started"] == 7
    assert level["infra_invalid_episodes"] == 1
    assert level["passed"] is False
    assert "worker_startup" in level["decision_reasons"]


def test_command_worker_uses_persistent_jsonl_protocol() -> None:
    repository = Path(__file__).resolve().parents[1]
    helper = repository / "scripts" / "evolution" / "benchmark_multienv.py"
    command = (
        f'"{sys.executable}" "{helper}" --serve-fake-worker '
        "--fake-step-delay-s 0.001 --fake-vla-queue-s 0.25"
    )
    worker = CommandEnvironmentWorker(
        command,
        slot_index=0,
        total_slots=4,
        maximum_api_concurrency=20,
        startup_timeout_s=5.0,
    )
    try:
        assert worker.reset(5.0).ok is True
        result = worker.step(5.0)
        assert result.ok is True
        assert result.vla_queue_s == pytest.approx(0.25)
    finally:
        worker.close()


def test_command_worker_can_persist_private_stderr(tmp_path: Path) -> None:
    repository = Path(__file__).resolve().parents[1]
    helper = repository / "scripts" / "evolution" / "benchmark_multienv.py"
    diagnostic = tmp_path / "private" / "worker.stderr.log"
    command = f'"{sys.executable}" "{helper}" --serve-fake-worker'
    worker = CommandEnvironmentWorker(
        command,
        slot_index=0,
        total_slots=1,
        maximum_api_concurrency=20,
        startup_timeout_s=5.0,
        diagnostic_path=diagnostic,
    )
    worker.close()
    assert diagnostic.is_file()
    if sys.platform != "win32":
        assert diagnostic.stat().st_mode & 0o777 == 0o600


def test_report_writer_rejects_urls_and_credentials(tmp_path: Path) -> None:
    safe = {
        "schema_version": 1,
        "maximum_api_concurrency": 20,
        "levels": [{"slots": 50, "passed": True}],
    }
    path = tmp_path / "capacity.json"
    write_secret_free_report(path, safe)
    assert json.loads(path.read_text(encoding="utf-8")) == safe

    with pytest.raises(ValueError, match="secret-bearing"):
        ensure_secret_free_report({"api_key": "redacted"})
    with pytest.raises(ValueError, match="secret-like"):
        ensure_secret_free_report({"failure": "https://private.invalid/v1"})
    with pytest.raises(ValueError, match="secret-like"):
        ensure_secret_free_report({"failure": "Bearer credential-material"})


def test_cli_writes_secret_free_fake_report(tmp_path: Path) -> None:
    repository = Path(__file__).resolve().parents[1]
    script = repository / "scripts" / "evolution" / "benchmark_multienv.py"
    output = tmp_path / "report.json"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--output",
            str(output),
            "--slots",
            "1,2,4",
            "--episodes-per-slot",
            "1",
            "--steps-per-episode",
            "1",
            "--warmup-steps",
            "0",
            "--sample-interval-s",
            "0.001",
            "--min-valid-throughput-per-hour",
            "1",
            "--max-gpu-memory-fraction",
            "1",
            "--max-cpu-percent",
            "100",
        ],
        cwd=repository,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    report_text = output.read_text(encoding="utf-8")
    report = json.loads(report_text)
    assert report["recommended_slots"] == 4
    assert report["config"]["rules"]["maximum_gpu_memory_fraction"] == 1.0
    assert report["config"]["rules"]["maximum_cpu_percent"] == 100.0
    assert "command_template" not in report_text
    assert "api_key" not in report_text
