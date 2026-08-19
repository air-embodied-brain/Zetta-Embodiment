# Copyright (c) 2026 Zetta Contributors
"""Recoverable per-GPU operation gate for many persistent environments."""

from __future__ import annotations

import os
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

_LOCAL_GUARD = threading.Lock()
_LOCAL_TOKENS: dict[str, threading.Lock] = {}


class GpuOperationTimeout(TimeoutError):
    """No GPU execution token became available before the deadline."""


class GpuOperationGate:
    """A small file-lock semaphore shared by environment server processes."""

    def __init__(
        self,
        root: str | Path,
        *,
        gpu_id: str,
        slots: int,
        timeout_s: float = 180.0,
        poll_s: float = 0.005,
    ) -> None:
        if slots < 1:
            raise ValueError("GPU operation slots must be positive")
        if timeout_s <= 0 or poll_s <= 0:
            raise ValueError("GPU gate timeout and poll interval must be positive")
        if not str(gpu_id).isdigit():
            raise ValueError("gpu_id must be a non-negative integer")
        self.root = Path(root) / f"gpu-{gpu_id}"
        self.root.mkdir(parents=True, exist_ok=True)
        self.gpu_id = str(gpu_id)
        self.slots = slots
        self.timeout_s = timeout_s
        self.poll_s = poll_s
        self.paths = tuple(
            self.root / f"token-{index:03d}.lock" for index in range(slots)
        )
        for path in self.paths:
            path.touch(exist_ok=True)

    @contextmanager
    def acquire(self) -> Iterator[int]:
        try:
            import fcntl
        except ImportError as exc:  # pragma: no cover - configured only on Linux.
            raise RuntimeError("GPU operation gates require POSIX flock") from exc

        deadline = time.monotonic() + self.timeout_s
        start = (os.getpid() + threading.get_ident()) % self.slots
        while True:
            for offset in range(self.slots):
                index = (start + offset) % self.slots
                path = self.paths[index]
                key = str(path.resolve())
                with _LOCAL_GUARD:
                    local = _LOCAL_TOKENS.setdefault(key, threading.Lock())
                if not local.acquire(blocking=False):
                    continue
                descriptor = path.open("a+b")
                file_locked = False
                try:
                    try:
                        fcntl.flock(descriptor.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError:
                        continue
                    file_locked = True
                    yield index
                    return
                finally:
                    if file_locked:
                        fcntl.flock(descriptor.fileno(), fcntl.LOCK_UN)
                    descriptor.close()
                    local.release()
            if time.monotonic() >= deadline:
                raise GpuOperationTimeout(
                    f"timed out waiting for a GPU {self.gpu_id} operation token"
                )
            time.sleep(self.poll_s)


__all__ = ["GpuOperationGate", "GpuOperationTimeout"]
