# Copyright (c) 2026 Zetta Contributors
from __future__ import annotations

from types import SimpleNamespace

import pytest

from zetta.utils import gpu_memory_guard


class _FakeNvml:
    def __init__(self, free_mib: int) -> None:
        self.free_bytes = free_mib * 1024 * 1024

    def nvmlDeviceGetHandleByIndex(self, index: int) -> int:
        return index

    def nvmlDeviceGetMemoryInfo(self, handle: int) -> SimpleNamespace:
        del handle
        return SimpleNamespace(free=self.free_bytes)


@pytest.fixture(autouse=True)
def _clear_reservations() -> None:
    gpu_memory_guard._RESERVED_MIB.clear()  # noqa: SLF001
    yield
    gpu_memory_guard._RESERVED_MIB.clear()  # noqa: SLF001


def test_nested_reservations_account_for_inflight_allocations(monkeypatch) -> None:
    """The second caller sees the first caller's in-flight reservation."""
    monkeypatch.setattr(gpu_memory_guard, "_nvml", lambda: _FakeNvml(3000))

    with gpu_memory_guard.reserve_gpu_memory(1000, device=0, safety_margin_mib=500):
        assert gpu_memory_guard.reservation_snapshot() == {0: 1000}
        with pytest.raises(gpu_memory_guard.GpuMemoryExhausted) as excinfo:
            with gpu_memory_guard.reserve_gpu_memory(
                2000, device=0, safety_margin_mib=500
            ):
                pass
        assert excinfo.value.available_mib == 2000

    assert gpu_memory_guard.reservation_snapshot() == {}


def test_reservation_is_released_when_native_call_fails(monkeypatch) -> None:
    """Exceptions inside the guarded operation cannot leak ledger capacity."""
    monkeypatch.setattr(gpu_memory_guard, "_nvml", lambda: _FakeNvml(4096))

    with pytest.raises(RuntimeError, match="native failure"):
        with gpu_memory_guard.reserve_gpu_memory(800, device=1, safety_margin_mib=512):
            assert gpu_memory_guard.reservation_snapshot() == {1: 800}
            raise RuntimeError("native failure")

    assert gpu_memory_guard.reservation_snapshot() == {}
