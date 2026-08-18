# Copyright (c) 2026 RPent Contributors
"""Single-host multi-environment capacity benchmark primitives.

The benchmark deliberately treats a logical environment slot as a schedulable
unit rather than as a GPU.  Fifty persistent environment processes may share a
single accelerator when memory, rendering and latency measurements permit it.
"""

from __future__ import annotations

import json
import math
import os
import queue
import re
import shlex
import subprocess
import threading
import time
import uuid
from collections import Counter, deque
from collections.abc import Callable, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol

DEFAULT_SLOT_LADDER = (1, 2, 4, 8, 16, 32, 50)
DEFAULT_API_CONCURRENCY = 8


def _percentile(values: Sequence[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _finite_nonnegative(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(number) or number < 0:
        return default
    return number


@dataclass(frozen=True, slots=True)
class CapacityRules:
    """Configurable expansion and rollback rules."""

    maximum_infra_invalid_rate: float = 0.02
    maximum_reset_p95_s: float = 60.0
    maximum_step_p95_s: float = 5.0
    maximum_vla_queue_p95_s: float = 2.0
    maximum_gpu_memory_fraction: float = 0.92
    maximum_cpu_percent: float = 95.0
    maximum_rss_mib: float | None = None
    minimum_valid_throughput_per_hour: float = 25.0
    stop_after_failed_level: bool = True

    def __post_init__(self) -> None:
        fractions = (
            self.maximum_infra_invalid_rate,
            self.maximum_gpu_memory_fraction,
        )
        if any(value < 0 or value > 1 for value in fractions):
            raise ValueError("fractional capacity rules must be between zero and one")
        positive = (
            self.maximum_reset_p95_s,
            self.maximum_step_p95_s,
            self.maximum_vla_queue_p95_s,
            self.maximum_cpu_percent,
            self.minimum_valid_throughput_per_hour,
        )
        if any(value <= 0 for value in positive):
            raise ValueError("latency, CPU and throughput rules must be positive")
        if self.maximum_rss_mib is not None and self.maximum_rss_mib <= 0:
            raise ValueError("maximum_rss_mib must be positive when supplied")


@dataclass(frozen=True, slots=True)
class CapacityConfig:
    """Immutable benchmark configuration."""

    slot_ladder: tuple[int, ...] = DEFAULT_SLOT_LADDER
    episodes_per_slot: int = 2
    steps_per_episode: int = 8
    warmup_steps: int = 1
    operation_timeout_s: float = 120.0
    sample_interval_s: float = 0.25
    worker_startup_concurrency: int = 64
    maximum_active_environment_operations: int | None = None
    operation_resource_shards: int | None = None
    maximum_active_operations_per_shard: int = 1
    maximum_api_concurrency: int = DEFAULT_API_CONCURRENCY
    api_limited_steps: bool = False
    rules: CapacityRules = field(default_factory=CapacityRules)

    def __post_init__(self) -> None:
        if not self.slot_ladder:
            raise ValueError("slot_ladder cannot be empty")
        if any(slot <= 0 for slot in self.slot_ladder):
            raise ValueError("all slot counts must be positive")
        if tuple(sorted(set(self.slot_ladder))) != self.slot_ladder:
            raise ValueError("slot_ladder must be strictly increasing")
        integers = (
            self.episodes_per_slot,
            self.steps_per_episode,
            self.worker_startup_concurrency,
            self.maximum_api_concurrency,
        )
        if any(value <= 0 for value in integers) or self.warmup_steps < 0:
            raise ValueError("iteration and concurrency counts are invalid")
        if self.operation_timeout_s <= 0 or self.sample_interval_s <= 0:
            raise ValueError("timeouts and sampling intervals must be positive")
        if (
            self.maximum_active_environment_operations is not None
            and self.maximum_active_environment_operations <= 0
        ):
            raise ValueError(
                "maximum_active_environment_operations must be positive when supplied"
            )
        if (
            self.operation_resource_shards is not None
            and self.operation_resource_shards <= 0
        ):
            raise ValueError("operation_resource_shards must be positive when supplied")
        if self.maximum_active_operations_per_shard <= 0:
            raise ValueError("maximum_active_operations_per_shard must be positive")

    def public_dict(self) -> dict[str, Any]:
        """Return only values safe to persist in the public report."""

        return {
            "slot_ladder": list(self.slot_ladder),
            "episodes_per_slot": self.episodes_per_slot,
            "steps_per_episode": self.steps_per_episode,
            "warmup_steps": self.warmup_steps,
            "operation_timeout_s": self.operation_timeout_s,
            "sample_interval_s": self.sample_interval_s,
            "worker_startup_concurrency": self.worker_startup_concurrency,
            "maximum_active_environment_operations": (
                self.maximum_active_environment_operations
            ),
            "operation_resource_shards": self.operation_resource_shards,
            "maximum_active_operations_per_shard": (
                self.maximum_active_operations_per_shard
            ),
            "maximum_api_concurrency": self.maximum_api_concurrency,
            "api_limited_steps": self.api_limited_steps,
            "rules": asdict(self.rules),
        }


@dataclass(frozen=True, slots=True)
class OperationResult:
    """A reset or step result returned by a logical environment worker."""

    ok: bool = True
    valid: bool = True
    infra_invalid: bool = False
    vla_queue_s: float = 0.0
    failure_class: str = "none"


class EnvironmentWorker(Protocol):
    """Minimal persistent environment worker contract."""

    def reset(self, timeout_s: float) -> OperationResult: ...

    def step(self, timeout_s: float) -> OperationResult: ...

    def close(self) -> None: ...


class ConcurrencyTracker:
    """Thread-safe active-operation counter used by fake workers and tests."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._active = 0
        self.maximum_active = 0

    def enter(self) -> None:
        with self._lock:
            self._active += 1
            self.maximum_active = max(self.maximum_active, self._active)

    def exit(self) -> None:
        with self._lock:
            self._active -= 1


class FakeEnvironmentWorker:
    """Dependency-free deterministic worker for CI and scheduler validation."""

    def __init__(
        self,
        *,
        slot_index: int,
        total_slots: int,
        reset_delay_s: float = 0.0,
        step_delay_s: float = 0.0,
        vla_queue_s: float = 0.0,
        fail_at_or_above_slots: int | None = None,
        fail_every_operations: int = 0,
        tracker: ConcurrencyTracker | None = None,
    ) -> None:
        self.slot_index = slot_index
        self.total_slots = total_slots
        self.reset_delay_s = reset_delay_s
        self.step_delay_s = step_delay_s
        self.vla_queue_s = vla_queue_s
        self.fail_at_or_above_slots = fail_at_or_above_slots
        self.fail_every_operations = fail_every_operations
        self.tracker = tracker
        self._operations = 0

    def _operation(self, delay_s: float, *, include_queue: bool) -> OperationResult:
        if self.tracker is not None:
            self.tracker.enter()
        try:
            if delay_s:
                time.sleep(delay_s)
            self._operations += 1
            capacity_failure = (
                self.fail_at_or_above_slots is not None
                and self.total_slots >= self.fail_at_or_above_slots
            )
            periodic_failure = (
                self.fail_every_operations > 0
                and self._operations % self.fail_every_operations == 0
            )
            if capacity_failure or periodic_failure:
                return OperationResult(
                    ok=False,
                    valid=False,
                    infra_invalid=True,
                    failure_class="fake_infrastructure",
                )
            return OperationResult(
                vla_queue_s=self.vla_queue_s if include_queue else 0.0
            )
        finally:
            if self.tracker is not None:
                self.tracker.exit()

    def reset(self, timeout_s: float) -> OperationResult:
        del timeout_s
        return self._operation(self.reset_delay_s, include_queue=False)

    def step(self, timeout_s: float) -> OperationResult:
        del timeout_s
        return self._operation(self.step_delay_s, include_queue=True)

    def close(self) -> None:
        return None


class CommandEnvironmentWorker:
    """Persistent JSONL subprocess adapter for a real environment worker.

    The command is never included in reports.  The child receives one JSON line
    per operation and must answer with one JSON object containing only ``ok``,
    ``valid``, ``infra_invalid``, ``vla_queue_s`` and ``failure_class``.
    """

    _ALLOWED_FAILURE = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")

    def __init__(
        self,
        command_template: str,
        *,
        slot_index: int,
        total_slots: int,
        maximum_api_concurrency: int,
        startup_timeout_s: float,
        diagnostic_path: Path | None = None,
    ) -> None:
        rendered = command_template.format(
            slot=slot_index,
            slots=total_slots,
            max_api=maximum_api_concurrency,
        )
        # ``posix=True`` consistently removes grouping quotes while preserving
        # backslashes inside quoted Windows paths; ``posix=False`` leaves the
        # quote characters in argv[0] and makes CreateProcess fail.
        arguments = shlex.split(rendered, posix=True)
        if not arguments:
            raise ValueError("command template rendered an empty command")
        self._diagnostic_stream = None
        if diagnostic_path is not None:
            diagnostic_path.parent.mkdir(parents=True, exist_ok=True)
            self._diagnostic_stream = diagnostic_path.open(
                "a", encoding="utf-8", buffering=1
            )
            if os.name == "posix":
                os.chmod(diagnostic_path, 0o600)
        self._process = subprocess.Popen(  # noqa: S603
            arguments,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=(
                self._diagnostic_stream
                if self._diagnostic_stream is not None
                else subprocess.DEVNULL
            ),
            text=True,
            encoding="utf-8",
            bufsize=1,
        )
        self._responses: queue.Queue[dict[str, Any] | None] = queue.Queue()
        self._reader = threading.Thread(target=self._read_responses, daemon=True)
        self._reader.start()
        self._request_lock = threading.Lock()
        ready = self._request("health", startup_timeout_s)
        if not ready.ok:
            self.close()
            raise RuntimeError("worker failed its startup health check")

    def _read_responses(self) -> None:
        assert self._process.stdout is not None
        for line in self._process.stdout:
            try:
                value = json.loads(line)
            except (json.JSONDecodeError, UnicodeError):
                self._responses.put(None)
                continue
            self._responses.put(value if isinstance(value, dict) else None)
        self._responses.put(None)

    @classmethod
    def _normalize_response(cls, response: dict[str, Any] | None) -> OperationResult:
        if response is None:
            return OperationResult(
                ok=False,
                valid=False,
                infra_invalid=True,
                failure_class="worker_protocol",
            )
        failure_class = str(response.get("failure_class", "none"))
        if not cls._ALLOWED_FAILURE.fullmatch(failure_class):
            failure_class = "worker_failure"
        return OperationResult(
            ok=bool(response.get("ok", False)),
            valid=bool(response.get("valid", response.get("ok", False))),
            infra_invalid=bool(response.get("infra_invalid", False)),
            vla_queue_s=_finite_nonnegative(response.get("vla_queue_s")),
            failure_class=failure_class,
        )

    def _request(self, operation: str, timeout_s: float) -> OperationResult:
        with self._request_lock:
            if self._process.poll() is not None:
                return OperationResult(
                    ok=False,
                    valid=False,
                    infra_invalid=True,
                    failure_class="worker_exit",
                )
            request = json.dumps(
                {"op": operation, "request_id": uuid.uuid4().hex},
                separators=(",", ":"),
            )
            try:
                assert self._process.stdin is not None
                self._process.stdin.write(request + "\n")
                self._process.stdin.flush()
                response = self._responses.get(timeout=timeout_s)
            except queue.Empty:
                # A late response must never be mistaken for the next request.
                # Retire the entire persistent worker after a timeout.
                if self._process.poll() is None:
                    self._process.terminate()
                return OperationResult(
                    ok=False,
                    valid=False,
                    infra_invalid=True,
                    failure_class="worker_timeout",
                )
            except (BrokenPipeError, OSError):
                return OperationResult(
                    ok=False,
                    valid=False,
                    infra_invalid=True,
                    failure_class="worker_timeout",
                )
            return self._normalize_response(response)

    def reset(self, timeout_s: float) -> OperationResult:
        return self._request("reset", timeout_s)

    def step(self, timeout_s: float) -> OperationResult:
        return self._request("step", timeout_s)

    def close(self) -> None:
        try:
            if self._process.poll() is None:
                try:
                    self._request("close", 5.0)
                    # The worker acknowledges ``close`` before its environment
                    # finally-block releases EGL/MuJoCo resources.  Give that
                    # cleanup a bounded grace period instead of terminating the
                    # process immediately after the acknowledgement.
                    try:
                        self._process.wait(timeout=10.0)
                    except subprocess.TimeoutExpired:
                        pass
                finally:
                    if self._process.poll() is None:
                        self._process.terminate()
                        try:
                            self._process.wait(timeout=2.0)
                        except subprocess.TimeoutExpired:
                            self._process.kill()
                            self._process.wait(timeout=2.0)
        finally:
            if self._diagnostic_stream is not None:
                self._diagnostic_stream.close()
                self._diagnostic_stream = None


@dataclass(frozen=True, slots=True)
class SystemSample:
    cpu_percent: float
    rss_mib: float
    gpu_memory_used_mib: float
    gpu_memory_total_mib: float
    gpu_memory_max_device_used_mib: float
    gpu_memory_max_device_fraction: float
    gpu_utilization_percent: float


class SystemSampler:
    """Best-effort host/process/GPU sampler with no mandatory dependencies."""

    def __init__(self, interval_s: float) -> None:
        self.interval_s = interval_s
        self.samples: list[SystemSample] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @staticmethod
    def _process_sample() -> tuple[float, float]:
        try:
            import psutil  # type: ignore[import-not-found]

            process = psutil.Process()
            rss = process.memory_info().rss
            for child in process.children(recursive=True):
                try:
                    rss += child.memory_info().rss
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            return float(psutil.cpu_percent(interval=None)), rss / (1024 * 1024)
        except (ImportError, OSError):
            try:
                import resource

                raw = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
                divisor = 1024.0 if os.name != "nt" else 1024.0 * 1024.0
                return 0.0, raw / divisor
            except (ImportError, OSError):
                return 0.0, 0.0

    @staticmethod
    def _gpu_sample() -> tuple[float, float, float, float, float]:
        command = [
            "nvidia-smi",
            "--query-gpu=memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits",
        ]
        try:
            result = subprocess.run(  # noqa: S603
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=2.0,
            )
        except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
            return 0.0, 0.0, 0.0, 0.0, 0.0
        if result.returncode != 0:
            return 0.0, 0.0, 0.0, 0.0, 0.0
        used = 0.0
        total = 0.0
        utilization: list[float] = []
        per_device_used: list[float] = []
        per_device_fraction: list[float] = []
        for line in result.stdout.splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) != 3:
                continue
            try:
                device_used = float(parts[0])
                device_total = float(parts[1])
                used += device_used
                total += device_total
                per_device_used.append(device_used)
                per_device_fraction.append(
                    device_used / device_total if device_total else 0.0
                )
                utilization.append(float(parts[2]))
            except ValueError:
                continue
        return (
            used,
            total,
            max(per_device_used, default=0.0),
            max(per_device_fraction, default=0.0),
            max(utilization, default=0.0),
        )

    def sample_once(self) -> None:
        cpu, rss = self._process_sample()
        (
            gpu_used,
            gpu_total,
            gpu_max_device_used,
            gpu_max_device_fraction,
            gpu_utilization,
        ) = self._gpu_sample()
        self.samples.append(
            SystemSample(
                cpu_percent=cpu,
                rss_mib=rss,
                gpu_memory_used_mib=gpu_used,
                gpu_memory_total_mib=gpu_total,
                gpu_memory_max_device_used_mib=gpu_max_device_used,
                gpu_memory_max_device_fraction=gpu_max_device_fraction,
                gpu_utilization_percent=gpu_utilization,
            )
        )

    def _loop(self) -> None:
        while not self._stop.wait(self.interval_s):
            self.sample_once()

    def start(self) -> None:
        self.sample_once()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self.interval_s * 2))
        self.sample_once()

    def summary(self) -> dict[str, float]:
        if not self.samples:
            return {
                "cpu_peak_percent": 0.0,
                "rss_peak_mib": 0.0,
                "gpu_memory_peak_mib": 0.0,
                "gpu_memory_total_mib": 0.0,
                "gpu_memory_peak_single_device_mib": 0.0,
                "gpu_memory_peak_fraction": 0.0,
                "gpu_utilization_peak_percent": 0.0,
            }
        gpu_total = max(sample.gpu_memory_total_mib for sample in self.samples)
        gpu_peak = max(sample.gpu_memory_used_mib for sample in self.samples)
        return {
            "cpu_peak_percent": max(sample.cpu_percent for sample in self.samples),
            "rss_peak_mib": max(sample.rss_mib for sample in self.samples),
            "gpu_memory_peak_mib": gpu_peak,
            "gpu_memory_total_mib": gpu_total,
            "gpu_memory_peak_single_device_mib": max(
                sample.gpu_memory_max_device_used_mib for sample in self.samples
            ),
            # Gate on the fullest device, not aggregate host memory. Aggregation
            # can hide an OOM-bound GPU behind seven idle H100s.
            "gpu_memory_peak_fraction": max(
                sample.gpu_memory_max_device_fraction for sample in self.samples
            ),
            "gpu_utilization_peak_percent": max(
                sample.gpu_utilization_percent for sample in self.samples
            ),
        }


WorkerFactory = Callable[[int, int], EnvironmentWorker]
ProgressCallback = Callable[[dict[str, Any]], None]


def command_worker_factory(
    command_template: str,
    config: CapacityConfig,
    *,
    diagnostic_root: Path | None = None,
) -> WorkerFactory:
    """Build a worker factory without exposing the command in public output."""

    def factory(slot_index: int, total_slots: int) -> EnvironmentWorker:
        return CommandEnvironmentWorker(
            command_template,
            slot_index=slot_index,
            total_slots=total_slots,
            maximum_api_concurrency=config.maximum_api_concurrency,
            startup_timeout_s=config.operation_timeout_s,
            diagnostic_path=(
                diagnostic_root / f"slot-{slot_index:03d}.stderr.log"
                if diagnostic_root is not None
                else None
            ),
        )

    return factory


class MultiEnvironmentCapacityBenchmark:
    """Run an increasing single-host logical-environment slot ladder."""

    def __init__(
        self,
        config: CapacityConfig,
        worker_factory: WorkerFactory,
        *,
        worker_mode: str,
        progress_callback: ProgressCallback | None = None,
    ) -> None:
        self.config = config
        self.worker_factory = worker_factory
        self.worker_mode = worker_mode
        self.progress_callback = progress_callback
        self._api_semaphore = threading.BoundedSemaphore(config.maximum_api_concurrency)
        self._scheduler_queue_latencies: list[float] = []
        self._operation_round = 0

    def _emit(self, event: str, **values: Any) -> None:
        if self.progress_callback is None:
            return
        self.progress_callback(
            {
                "event": event,
                "monotonic_s": time.monotonic(),
                **values,
            }
        )

    def _invoke(
        self,
        worker: EnvironmentWorker,
        operation: str,
    ) -> tuple[OperationResult, float]:
        started = time.perf_counter()
        try:
            if operation == "reset":
                result = worker.reset(self.config.operation_timeout_s)
            elif self.config.api_limited_steps:
                with self._api_semaphore:
                    result = worker.step(self.config.operation_timeout_s)
            else:
                result = worker.step(self.config.operation_timeout_s)
        except Exception:  # A benchmark must classify worker crashes, not crash itself.
            result = OperationResult(
                ok=False,
                valid=False,
                infra_invalid=True,
                failure_class="worker_exception",
            )
        return result, time.perf_counter() - started

    def _parallel_operations(
        self,
        executor: ThreadPoolExecutor,
        workers: Sequence[EnvironmentWorker],
        operation: str,
    ) -> list[tuple[OperationResult, float]]:
        if not workers:
            return []
        if self.config.operation_resource_shards is not None:
            return self._resource_sharded_operations(executor, workers, operation)
        active_limit = self.config.maximum_active_environment_operations or len(workers)
        # Environment residency and active simulator work are intentionally
        # separate capacity axes.  Starting all resident environments remains
        # useful, but only a bounded micro-batch is submitted at once.  This
        # avoids spending an operation timeout while waiting inside a per-GPU
        # renderer lock and makes the measured operation latency service time,
        # not scheduler backlog.
        results: list[tuple[OperationResult, float]] = []
        dispatch_started = time.perf_counter()
        self._operation_round += 1
        operation_round = self._operation_round
        for start in range(0, len(workers), active_limit):
            batch = workers[start : start + active_limit]
            queue_latency = time.perf_counter() - dispatch_started
            self._scheduler_queue_latencies.extend([queue_latency] * len(batch))
            self._emit(
                "operation_batch_started",
                operation=operation,
                operation_round=operation_round,
                batch_start=start,
                batch_size=len(batch),
                resident_workers=len(workers),
                scheduler_queue_s=queue_latency,
            )
            batch_started = time.perf_counter()
            futures = [
                executor.submit(self._invoke, worker, operation) for worker in batch
            ]
            batch_results = [future.result() for future in futures]
            results.extend(batch_results)
            self._emit(
                "operation_batch_completed",
                operation=operation,
                operation_round=operation_round,
                batch_start=start,
                batch_size=len(batch),
                wall_time_s=time.perf_counter() - batch_started,
                infra_invalid=sum(
                    result.infra_invalid or not result.ok
                    for result, _latency in batch_results
                ),
            )
        return results

    def _resource_sharded_operations(
        self,
        executor: ThreadPoolExecutor,
        workers: Sequence[EnvironmentWorker],
        operation: str,
    ) -> list[tuple[OperationResult, float]]:
        """Dispatch work continuously while respecting per-resource limits.

        Slots are assigned round-robin to resource shards.  This mirrors the
        H100 farm's immutable ``slot % gpu_count`` mapping.  A completed shard
        immediately receives its next queued operation instead of waiting for
        the slowest member of a global micro-batch.
        """

        shard_count = int(self.config.operation_resource_shards or 1)
        per_shard = self.config.maximum_active_operations_per_shard
        global_limit = self.config.maximum_active_environment_operations or len(workers)
        global_limit = min(global_limit, shard_count * per_shard, len(workers))
        shard_queues: list[deque[tuple[int, EnvironmentWorker]]] = [
            deque() for _ in range(shard_count)
        ]
        for worker_index, worker in enumerate(workers):
            shard_queues[worker_index % shard_count].append((worker_index, worker))

        ordered_results: list[tuple[OperationResult, float] | None] = [
            None for _ in workers
        ]
        shard_active = [0 for _ in range(shard_count)]
        pending: dict[Future[tuple[OperationResult, float]], tuple[int, int]] = {}
        dispatch_started = time.perf_counter()
        self._operation_round += 1
        operation_round = self._operation_round
        next_shard = 0

        def fill_available() -> None:
            nonlocal next_shard
            idle_rounds = 0
            while len(pending) < global_limit and idle_rounds < shard_count:
                shard = next_shard
                next_shard = (next_shard + 1) % shard_count
                if shard_active[shard] >= per_shard or not shard_queues[shard]:
                    idle_rounds += 1
                    continue
                idle_rounds = 0
                worker_index, worker = shard_queues[shard].popleft()
                queue_latency = time.perf_counter() - dispatch_started
                self._scheduler_queue_latencies.append(queue_latency)
                future = executor.submit(self._invoke, worker, operation)
                pending[future] = (worker_index, shard)
                shard_active[shard] += 1
                self._emit(
                    "operation_dispatched",
                    operation=operation,
                    operation_round=operation_round,
                    worker_index=worker_index,
                    resource_shard=shard,
                    active_operations=len(pending),
                    scheduler_queue_s=queue_latency,
                )

        fill_available()
        while pending:
            completed, _not_done = wait(tuple(pending), return_when=FIRST_COMPLETED)
            for future in completed:
                worker_index, shard = pending.pop(future)
                shard_active[shard] -= 1
                result = future.result()
                ordered_results[worker_index] = result
                self._emit(
                    "operation_completed",
                    operation=operation,
                    operation_round=operation_round,
                    worker_index=worker_index,
                    resource_shard=shard,
                    active_operations=len(pending),
                    infra_invalid=result[0].infra_invalid or not result[0].ok,
                )
            fill_available()

        if any(result is None for result in ordered_results):
            raise RuntimeError("resource-sharded scheduler lost an operation result")
        return [result for result in ordered_results if result is not None]

    def _evaluate_level(
        self,
        metrics: dict[str, Any],
    ) -> tuple[bool, list[str]]:
        rules = self.config.rules
        failures: list[str] = []

        comparisons = (
            (
                metrics["infra_invalid_rate"],
                rules.maximum_infra_invalid_rate,
                "infra_invalid_rate",
            ),
            (metrics["reset_latency_s"]["p95"], rules.maximum_reset_p95_s, "reset_p95"),
            (metrics["step_latency_s"]["p95"], rules.maximum_step_p95_s, "step_p95"),
            (
                metrics["vla_queue_latency_s"]["p95"],
                rules.maximum_vla_queue_p95_s,
                "vla_queue_p95",
            ),
            (
                metrics["system"]["gpu_memory_peak_fraction"],
                rules.maximum_gpu_memory_fraction,
                "gpu_memory",
            ),
            (metrics["system"]["cpu_peak_percent"], rules.maximum_cpu_percent, "cpu"),
        )
        for observed, limit, name in comparisons:
            if observed is not None and float(observed) > limit:
                failures.append(name)
        if (
            rules.maximum_rss_mib is not None
            and metrics["system"]["rss_peak_mib"] > rules.maximum_rss_mib
        ):
            failures.append("rss")
        if (
            metrics["valid_throughput_per_hour"]
            < rules.minimum_valid_throughput_per_hour
        ):
            failures.append("valid_throughput")
        if metrics["valid_episodes"] == 0:
            failures.append("zero_valid")
        if metrics["workers_started"] != metrics["slots"]:
            failures.append("worker_startup")
        if metrics["warmup_infra_invalid_operations"]:
            failures.append("warmup_infrastructure")
        return not failures, failures

    def _run_level(self, slots: int) -> dict[str, Any]:
        level_started = time.perf_counter()
        self._scheduler_queue_latencies = []
        sampler = SystemSampler(self.config.sample_interval_s)
        sampler.start()
        workers_by_slot: dict[int, EnvironmentWorker] = {}
        creation_failures = 0
        startup_started = time.perf_counter()

        def start_worker(slot_index: int) -> tuple[int, EnvironmentWorker]:
            return slot_index, self.worker_factory(slot_index, slots)

        startup_workers = min(slots, self.config.worker_startup_concurrency)
        with ThreadPoolExecutor(max_workers=startup_workers) as startup_executor:
            futures = [
                startup_executor.submit(start_worker, slot_index)
                for slot_index in range(slots)
            ]
            for future in futures:
                try:
                    slot_index, worker = future.result()
                    workers_by_slot[slot_index] = worker
                except Exception:
                    creation_failures += 1
        startup_wall_time_s = max(time.perf_counter() - startup_started, 1e-9)
        workers = [workers_by_slot[index] for index in sorted(workers_by_slot)]
        self._emit(
            "workers_started",
            requested=slots,
            started=len(workers),
            creation_failures=creation_failures,
            wall_time_s=startup_wall_time_s,
        )

        reset_latencies: list[float] = []
        step_latencies: list[float] = []
        vla_queue_latencies: list[float] = []
        failure_classes: Counter[str] = Counter()
        infra_invalid_operations = creation_failures
        infra_invalid_episodes = creation_failures * self.config.episodes_per_slot
        warmup_infra_invalid_operations = 0
        total_operations = creation_failures
        attempted_episodes = slots * self.config.episodes_per_slot
        valid_episodes = 0
        operation_started = time.perf_counter()
        teardown_wall_time_s = 0.0
        try:
            if workers:
                with ThreadPoolExecutor(max_workers=len(workers)) as executor:
                    if self.config.warmup_steps:
                        warmup_resets = self._parallel_operations(
                            executor, workers, "reset"
                        )
                        for result, _latency in warmup_resets:
                            total_operations += 1
                            if result.infra_invalid or not result.ok:
                                infra_invalid_operations += 1
                                warmup_infra_invalid_operations += 1
                                failure_classes[result.failure_class] += 1
                    for _ in range(self.config.warmup_steps):
                        warmup_results = self._parallel_operations(
                            executor, workers, "step"
                        )
                        for result, _latency in warmup_results:
                            total_operations += 1
                            if result.infra_invalid or not result.ok:
                                infra_invalid_operations += 1
                                warmup_infra_invalid_operations += 1
                                failure_classes[result.failure_class] += 1
                    episode_valid = [True] * len(workers)
                    for _ in range(self.config.episodes_per_slot):
                        episode_valid = [True] * len(workers)
                        episode_infra_invalid = [False] * len(workers)
                        reset_results = self._parallel_operations(
                            executor, workers, "reset"
                        )
                        for index, (result, latency) in enumerate(reset_results):
                            total_operations += 1
                            reset_latencies.append(latency)
                            if result.infra_invalid or not result.ok:
                                infra_invalid_operations += 1
                                episode_valid[index] = False
                                episode_infra_invalid[index] = True
                                failure_classes[result.failure_class] += 1
                        for _ in range(self.config.steps_per_episode):
                            step_results = self._parallel_operations(
                                executor, workers, "step"
                            )
                            for index, (result, latency) in enumerate(step_results):
                                total_operations += 1
                                step_latencies.append(latency)
                                vla_queue_latencies.append(result.vla_queue_s)
                                if result.infra_invalid or not result.ok:
                                    infra_invalid_operations += 1
                                    episode_valid[index] = False
                                    episode_infra_invalid[index] = True
                                    failure_classes[result.failure_class] += 1
                                if not result.valid:
                                    episode_valid[index] = False
                        valid_episodes += sum(episode_valid)
                        infra_invalid_episodes += sum(episode_infra_invalid)
        finally:
            steady_state_wall_time_s = max(
                time.perf_counter() - operation_started, 1e-9
            )
            teardown_started = time.perf_counter()

            def close_worker(worker: EnvironmentWorker) -> bool:
                try:
                    worker.close()
                    return True
                except Exception:
                    return False

            if workers:
                teardown_workers = min(
                    len(workers), self.config.worker_startup_concurrency
                )
                with ThreadPoolExecutor(max_workers=teardown_workers) as closer:
                    for closed in closer.map(close_worker, workers):
                        if not closed:
                            failure_classes["close_failure"] += 1
            teardown_wall_time_s = time.perf_counter() - teardown_started
            sampler.stop()
        wall_time_s = max(time.perf_counter() - level_started, 1e-9)

        metrics: dict[str, Any] = {
            "slots": slots,
            "workers_started": len(workers),
            "attempted_episodes": attempted_episodes,
            "valid_episodes": valid_episodes,
            "infra_invalid_episodes": infra_invalid_episodes,
            "infra_invalid_operations": infra_invalid_operations,
            "warmup_infra_invalid_operations": warmup_infra_invalid_operations,
            "infra_invalid_episode_rate": (
                infra_invalid_episodes / attempted_episodes
                if attempted_episodes
                else 1.0
            ),
            "infra_invalid_operation_rate": (
                infra_invalid_operations / total_operations if total_operations else 1.0
            ),
            "wall_time_s": wall_time_s,
            "startup_wall_time_s": startup_wall_time_s,
            "steady_state_wall_time_s": steady_state_wall_time_s,
            "teardown_wall_time_s": teardown_wall_time_s,
            "valid_throughput_per_hour": valid_episodes * 3600.0 / wall_time_s,
            "steady_state_valid_throughput_per_hour": (
                valid_episodes * 3600.0 / steady_state_wall_time_s
            ),
            "reset_latency_s": {
                "p50": _percentile(reset_latencies, 0.50),
                "p95": _percentile(reset_latencies, 0.95),
            },
            "step_latency_s": {
                "p50": _percentile(step_latencies, 0.50),
                "p95": _percentile(step_latencies, 0.95),
            },
            "vla_queue_latency_s": {
                "p50": _percentile(vla_queue_latencies, 0.50),
                "p95": _percentile(vla_queue_latencies, 0.95),
            },
            "scheduler_queue_latency_s": {
                "p50": _percentile(self._scheduler_queue_latencies, 0.50),
                "p95": _percentile(self._scheduler_queue_latencies, 0.95),
            },
            "system": sampler.summary(),
            "failure_classes": dict(sorted(failure_classes.items())),
        }
        metrics["infra_invalid_rate"] = max(
            metrics["infra_invalid_episode_rate"],
            metrics["infra_invalid_operation_rate"],
        )
        passed, reasons = self._evaluate_level(metrics)
        metrics["passed"] = passed
        metrics["decision_reasons"] = reasons or ["expand"]
        self._emit(
            "capacity_level_completed",
            slots=slots,
            valid_episodes=valid_episodes,
            infra_invalid_episodes=infra_invalid_episodes,
            passed=passed,
            wall_time_s=wall_time_s,
        )
        return metrics

    def run(self) -> dict[str, Any]:
        """Execute the ladder and return a secret-free report dictionary."""

        levels: list[dict[str, Any]] = []
        recommended_slots = 0
        for slots in self.config.slot_ladder:
            level = self._run_level(slots)
            levels.append(level)
            if level["passed"]:
                recommended_slots = slots
            elif self.config.rules.stop_after_failed_level:
                break
        report = {
            "schema_version": 1,
            "run_id": uuid.uuid4().hex,
            "worker_mode": self.worker_mode,
            "config": self.config.public_dict(),
            "levels": levels,
            "recommended_slots": recommended_slots,
            "tested_maximum_slots": levels[-1]["slots"] if levels else 0,
            "completed_ladder": len(levels) == len(self.config.slot_ladder),
        }
        ensure_secret_free_report(report)
        return report


_SECRET_KEY_PARTS = ("api_key", "authorization", "credential", "password", "secret")
_SECRET_VALUE_PATTERNS = (
    re.compile(r"https?://", re.IGNORECASE),
    re.compile(r"\b(?:sk|key)-[A-Za-z0-9_-]{8,}\b", re.IGNORECASE),
    re.compile(r"bearer\s+[A-Za-z0-9._-]+", re.IGNORECASE),
)


def ensure_secret_free_report(value: Any, path: str = "report") -> None:
    """Fail closed if a report contains a likely credential, URL or secret key."""

    if isinstance(value, dict):
        for key, child in value.items():
            lowered = str(key).lower()
            if any(part in lowered for part in _SECRET_KEY_PARTS):
                raise ValueError(f"secret-bearing field is forbidden at {path}")
            ensure_secret_free_report(child, f"{path}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            ensure_secret_free_report(child, f"{path}[{index}]")
        return
    if isinstance(value, str) and any(
        pattern.search(value) for pattern in _SECRET_VALUE_PATTERNS
    ):
        raise ValueError(f"secret-like string is forbidden at {path}")


def write_secret_free_report(path: Path, report: dict[str, Any]) -> None:
    """Atomically persist a benchmark report after a final secret audit."""

    ensure_secret_free_report(report)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)
