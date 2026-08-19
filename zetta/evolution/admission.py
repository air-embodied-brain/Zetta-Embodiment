# Copyright (c) 2026 Zetta Contributors
"""Shared, secret-free API concurrency admission."""

from __future__ import annotations

import os
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from zetta.evolution.jsonio import atomic_write_json, directory_lock, read_json


class ApiAdmissionPool:
    """Bound live planner requests independently from GPU scheduling."""

    def __init__(self, root: str | Path, *, limit: int = 8) -> None:
        if limit < 1:
            raise ValueError("API admission limit must be positive")
        self.root = Path(root)
        self.limit = limit
        self.leases = self.root / "leases"
        self.leases.mkdir(parents=True, exist_ok=True)
        self.lock = self.root / ".admission.lock"

    def active(self) -> list[dict]:
        result = []
        for path in sorted(self.leases.glob("*.json")):
            try:
                result.append(read_json(path))
            except (FileNotFoundError, ValueError):
                continue
        return result

    @contextmanager
    def acquire(
        self,
        *,
        owner: str,
        timeout_s: float = 1800,
        poll_s: float = 0.25,
    ) -> Iterator[dict]:
        deadline = time.monotonic() + timeout_s
        lease_path: Path | None = None
        lease: dict | None = None
        while lease_path is None:
            with directory_lock(self.lock):
                active = list(self.leases.glob("*.json"))
                if len(active) < self.limit:
                    lease_id = uuid.uuid4().hex
                    lease_path = self.leases / f"{lease_id}.json"
                    lease = {
                        "lease_id": lease_id,
                        "owner": owner,
                        "pid": os.getpid(),
                        "created_at_unix": time.time(),
                    }
                    atomic_write_json(lease_path, lease, overwrite=False)
            if lease_path is None:
                if time.monotonic() >= deadline:
                    raise TimeoutError("timed out waiting for API admission")
                time.sleep(poll_s)
        try:
            yield lease or {}
        finally:
            lease_path.unlink(missing_ok=True)
