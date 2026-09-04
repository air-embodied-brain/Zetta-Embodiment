# Copyright (c) 2026 Zetta Contributors
"""Fail-fast GPU memory reservations for native environment construction.

Ray's fractional GPU resource is only a scheduler quota. MuJoCo/EGL
constructors can still block inside a native call when the physical card is
nearly full. This module provides a small process-local reservation ledger and
an NVML check immediately before those calls. NVML is optional: when it is
unavailable the guard becomes a no-op unless strict mode is requested with
``ZETTA_GPU_MEMORY_GUARD_STRICT=1``.
"""

from __future__ import annotations

import contextlib
import os
import threading
from contextlib import contextmanager
from typing import Iterator


class GpuMemoryExhausted(RuntimeError):
    """The requested native allocation should be rejected before blocking."""

    def __init__(self, *, device: int, requested_mib: int, available_mib: int) -> None:
        self.device = int(device)
        self.requested_mib = int(requested_mib)
        self.available_mib = int(available_mib)
        super().__init__(
            f"GPU {self.device} has only {self.available_mib} MiB effective free; "
            f"reservation requires {self.requested_mib} MiB"
        )


_LOCK = threading.RLock()
_RESERVED_MIB: dict[int, int] = {}


def _strict() -> bool:
    return os.environ.get("ZETTA_GPU_MEMORY_GUARD_STRICT", "0") == "1"


def _nvml():
    try:
        import pynvml  # type: ignore[import-not-found]
    except Exception:  # noqa: BLE001 - optional dependency
        return None
    try:
        pynvml.nvmlInit()
    except Exception:  # noqa: BLE001 - driver may be unavailable
        return None
    return pynvml


def _device_index(device: int | None) -> int:
    if device is not None:
        return max(0, int(device))
    raw = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",", 1)[0].strip()
    with contextlib.suppress(ValueError):
        if raw:
            return max(0, int(raw))
    return 0


@contextmanager
def reserve_gpu_memory(
    requested_mib: int | float = 0,
    *,
    device: int | None = None,
    safety_margin_mib: int | float = 1024,
) -> Iterator[None]:
    """Reserve memory and fail fast if effective NVML free memory is low.

    ``requested_mib <= 0`` disables the check. The reservation is released in
    ``finally`` even when native construction raises.
    """

    requested = max(0, int(float(requested_mib)))
    if requested <= 0:
        yield
        return
    index = _device_index(device)
    nvml = _nvml()
    if nvml is None:
        if _strict():
            raise GpuMemoryExhausted(
                device=index, requested_mib=requested, available_mib=0
            )
        yield
        return

    nvml_error = False
    with _LOCK:
        try:
            handle = nvml.nvmlDeviceGetHandleByIndex(index)
            info = nvml.nvmlDeviceGetMemoryInfo(handle)
            free_mib = int(info.free // (1024 * 1024))
        except Exception:  # noqa: BLE001 - NVML failures follow strict-mode policy
            nvml_error = True
            free_mib = 0
        if not nvml_error:
            reserved = _RESERVED_MIB.get(index, 0)
            effective = free_mib - reserved
            margin = max(0, int(float(safety_margin_mib)))
            if effective < requested + margin:
                raise GpuMemoryExhausted(
                    device=index,
                    requested_mib=requested,
                    available_mib=effective,
                )
            _RESERVED_MIB[index] = reserved + requested
    if nvml_error:
        if _strict():
            raise GpuMemoryExhausted(
                device=index, requested_mib=requested, available_mib=0
            )
        yield
        return
    try:
        yield
    finally:
        with _LOCK:
            remaining = _RESERVED_MIB.get(index, 0) - requested
            if remaining > 0:
                _RESERVED_MIB[index] = remaining
            else:
                _RESERVED_MIB.pop(index, None)


def reservation_snapshot() -> dict[int, int]:
    """Return a copy of the process-local reservation ledger."""

    with _LOCK:
        return dict(_RESERVED_MIB)


__all__ = ["GpuMemoryExhausted", "reservation_snapshot", "reserve_gpu_memory"]
