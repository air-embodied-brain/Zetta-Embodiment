# Copyright (c) 2026 Zetta Contributors
from __future__ import annotations

import os
import threading
import time

import pytest

from robots.robocasa.gpu_gate import GpuOperationGate, GpuOperationTimeout


@pytest.mark.skipif(os.name == "nt", reason="formal GPU gate uses POSIX flock")
def test_gpu_operation_gate_limits_threads_and_recovers(tmp_path) -> None:
    gate = GpuOperationGate(tmp_path, gpu_id="0", slots=2, timeout_s=2)
    lock = threading.Lock()
    active = 0
    peak = 0

    def work() -> None:
        nonlocal active, peak
        with gate.acquire():
            with lock:
                active += 1
                peak = max(peak, active)
            time.sleep(0.03)
            with lock:
                active -= 1

    threads = [threading.Thread(target=work) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert peak == 2

    with gate.acquire():
        pass


@pytest.mark.skipif(os.name == "nt", reason="formal GPU gate uses POSIX flock")
def test_gpu_operation_gate_times_out_fail_closed(tmp_path) -> None:
    held = GpuOperationGate(tmp_path, gpu_id="1", slots=1, timeout_s=1)
    waiting = GpuOperationGate(tmp_path, gpu_id="1", slots=1, timeout_s=0.02)
    with held.acquire(), pytest.raises(GpuOperationTimeout):
        with waiting.acquire():
            pass


@pytest.mark.skipif(os.name == "nt", reason="formal GPU gate uses POSIX flock")
def test_gpu_operation_gate_does_not_release_another_waiter_on_body_error(
    tmp_path,
) -> None:
    gate = GpuOperationGate(tmp_path, gpu_id="2", slots=1, timeout_s=2)
    entered = threading.Event()
    release = threading.Event()

    def waiting_work() -> None:
        with gate.acquire():
            entered.set()
            release.wait(timeout=1)

    with pytest.raises(RuntimeError):
        with gate.acquire():
            thread = threading.Thread(target=waiting_work)
            thread.start()
            raise RuntimeError("body failure")

    assert entered.wait(timeout=1)
    competing = GpuOperationGate(tmp_path, gpu_id="2", slots=1, timeout_s=0.02)
    with pytest.raises(GpuOperationTimeout):
        with competing.acquire():
            pass
    release.set()
    thread.join(timeout=1)
    assert not thread.is_alive()
