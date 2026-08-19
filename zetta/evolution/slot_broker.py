# Copyright (c) 2026 Zetta Contributors
"""Host-local leases for a persistent RoboCasa environment farm.

The farm may keep many renderer processes resident while this broker limits the
number of episodes that actively own a slot.  A lease is reclaimable only when
it is expired *and* the environment server reports ``FREE``.  This deliberately
prefers a stranded slot over allowing two actors to write one environment.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

from zetta.evolution.jsonio import (
    AppendOnlyLedger,
    atomic_write_json,
    directory_lock,
    read_json,
)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _timestamp(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat()


def _parse_timestamp(value: object) -> datetime:
    if not isinstance(value, str) or not value:
        raise ValueError("slot lease timestamp must be a non-empty string")
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        raise ValueError("slot lease timestamp must include a timezone")
    return parsed.astimezone(timezone.utc)


def _health(endpoint: str, *, timeout_s: float) -> dict[str, Any]:
    request = urllib.request.Request(endpoint.rstrip("/") + "/health", method="GET")
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("environment health response must be an object")
    return payload


@dataclass(frozen=True, slots=True)
class EnvironmentSlot:
    slot: int
    gpu: str
    endpoint: str
    generation: int


class EnvironmentSlotLease:
    def __init__(
        self,
        broker: "EnvironmentSlotBroker",
        *,
        lease_id: str,
        owner: str,
        job_id: str,
        slot: EnvironmentSlot,
    ) -> None:
        self._broker = broker
        self.lease_id = lease_id
        self.owner = owner
        self.job_id = job_id
        self.slot = slot
        self._released = False

    @property
    def endpoint(self) -> str:
        return self.slot.endpoint

    def heartbeat(self) -> None:
        if self._released:
            raise RuntimeError("cannot heartbeat a released environment slot")
        self._broker.heartbeat(self)

    def release(self, *, outcome: str = "completed") -> None:
        if self._released:
            return
        self._broker.release(self, outcome=outcome)
        self._released = True

    def __enter__(self) -> "EnvironmentSlotLease":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.release(outcome="worker_exception" if exc_type else "completed")


class EnvironmentSlotBroker:
    """Cross-process, host-local slot broker with fail-closed reclamation."""

    def __init__(
        self,
        root: str | Path,
        *,
        ready_manifest: str | Path,
        lease_s: float = 120.0,
        health_timeout_s: float = 2.0,
        maximum_active_slots: int | None = None,
        health_reader: Callable[[str], dict[str, Any]] | None = None,
    ) -> None:
        if lease_s <= 0 or health_timeout_s <= 0:
            raise ValueError("slot lease and health timeout must be positive")
        if maximum_active_slots is not None and maximum_active_slots < 1:
            raise ValueError("maximum_active_slots must be positive")
        self.root = Path(root)
        self.ready_manifest = Path(ready_manifest)
        self.lease_s = float(lease_s)
        self.health_timeout_s = float(health_timeout_s)
        self.maximum_active_slots = maximum_active_slots
        self._health_reader = health_reader or (
            lambda endpoint: _health(endpoint, timeout_s=self.health_timeout_s)
        )
        self._leases = self.root / "leases"
        self._leases.mkdir(parents=True, exist_ok=True)
        self._lock = self.root / ".slot-broker.lock"
        self._events = AppendOnlyLedger(self.root / "events.jsonl", key="event_id")

    def _slots(self) -> list[EnvironmentSlot]:
        manifest = read_json(self.ready_manifest)
        records = manifest.get("slots") if isinstance(manifest, dict) else None
        if not isinstance(records, list):
            raise ValueError("environment ready manifest requires a slots array")
        result: list[EnvironmentSlot] = []
        for record in records:
            if not isinstance(record, dict) or record.get("status") != "ready":
                continue
            endpoint = record.get("endpoint")
            if not isinstance(endpoint, str) or not endpoint.startswith(
                "http://127.0.0.1:"
            ):
                raise ValueError("environment slot endpoint must be host-local HTTP")
            result.append(
                EnvironmentSlot(
                    slot=int(record["slot"]),
                    gpu=str(record["gpu"]),
                    endpoint=endpoint.rstrip("/"),
                    generation=int(record.get("generation", 0)),
                )
            )
        if not result:
            raise RuntimeError("environment farm has no ready slots")
        return sorted(result, key=lambda item: (int(item.gpu), item.slot))

    def _lease_path(self, slot: int) -> Path:
        return self._leases / f"slot-{slot:03d}.json"

    @staticmethod
    def _is_free(health: dict[str, Any]) -> bool:
        protocol = health.get("write_protocol")
        return (
            health.get("status") == "healthy"
            and isinstance(protocol, dict)
            and protocol.get("phase") == "FREE"
        )

    def _active_leases(self) -> list[dict[str, Any]]:
        values: list[dict[str, Any]] = []
        for path in sorted(self._leases.glob("slot-*.json")):
            payload = read_json(path)
            if not isinstance(payload, dict):
                raise ValueError(f"slot lease is not an object: {path}")
            values.append(payload)
        return values

    def acquire(
        self,
        *,
        owner: str,
        job_id: str,
        timeout_s: float,
        progress_callback: Callable[[], None] | None = None,
        poll_s: float = 0.25,
    ) -> EnvironmentSlotLease:
        if not owner or not job_id:
            raise ValueError("slot owner and job_id must be non-empty")
        if timeout_s <= 0 or poll_s <= 0:
            raise ValueError("slot acquire timeout and poll interval must be positive")
        deadline = time.monotonic() + timeout_s
        while True:
            if progress_callback is not None:
                progress_callback()
            with directory_lock(self._lock):
                now = _now()
                active = self._active_leases()
                if (
                    self.maximum_active_slots is None
                    or len(active) < self.maximum_active_slots
                ):
                    counts: dict[str, int] = {}
                    for value in active:
                        gpu = str(value.get("gpu", ""))
                        counts[gpu] = counts.get(gpu, 0) + 1
                    candidates = sorted(
                        self._slots(),
                        key=lambda item: (counts.get(item.gpu, 0), item.slot),
                    )
                    for slot in candidates:
                        path = self._lease_path(slot.slot)
                        if path.exists():
                            existing = read_json(path)
                            expires = _parse_timestamp(existing.get("expires_at"))
                            if expires > now:
                                continue
                            try:
                                health = self._health_reader(slot.endpoint)
                            except (OSError, ValueError, urllib.error.URLError):
                                continue
                            if not self._is_free(health):
                                continue
                            path.unlink()
                            self._events.append(
                                {
                                    "event_id": "event-" + uuid.uuid4().hex,
                                    "event": "expired_free_lease_reclaimed",
                                    "at": _timestamp(now),
                                    "slot": slot.slot,
                                    "prior_lease_id": existing.get("lease_id"),
                                }
                            )
                        try:
                            health = self._health_reader(slot.endpoint)
                        except (OSError, ValueError, urllib.error.URLError):
                            continue
                        if not self._is_free(health):
                            continue
                        lease_id = "slot-lease-" + uuid.uuid4().hex
                        payload = {
                            "schema_version": 1,
                            "lease_id": lease_id,
                            "owner": owner,
                            "job_id": job_id,
                            "slot": slot.slot,
                            "gpu": slot.gpu,
                            "endpoint": slot.endpoint,
                            "farm_generation": slot.generation,
                            "acquired_at": _timestamp(now),
                            "heartbeat_at": _timestamp(now),
                            "expires_at": _timestamp(
                                now + timedelta(seconds=self.lease_s)
                            ),
                            "pid": os.getpid(),
                        }
                        atomic_write_json(path, payload, overwrite=False)
                        self._events.append(
                            {
                                "event_id": "event-" + uuid.uuid4().hex,
                                "event": "slot_acquired",
                                "at": _timestamp(now),
                                "lease_id": lease_id,
                                "job_id": job_id,
                                "slot": slot.slot,
                                "gpu": slot.gpu,
                            }
                        )
                        return EnvironmentSlotLease(
                            self,
                            lease_id=lease_id,
                            owner=owner,
                            job_id=job_id,
                            slot=slot,
                        )
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    "timed out acquiring a healthy free environment slot"
                )
            time.sleep(poll_s)

    def heartbeat(self, lease: EnvironmentSlotLease) -> None:
        path = self._lease_path(lease.slot.slot)
        with directory_lock(self._lock):
            if not path.exists():
                raise RuntimeError("environment slot lease no longer exists")
            payload = read_json(path)
            if payload.get("lease_id") != lease.lease_id:
                raise RuntimeError("stale environment slot lease cannot heartbeat")
            now = _now()
            payload["heartbeat_at"] = _timestamp(now)
            payload["expires_at"] = _timestamp(now + timedelta(seconds=self.lease_s))
            atomic_write_json(path, payload, overwrite=True)

    def release(self, lease: EnvironmentSlotLease, *, outcome: str) -> None:
        path = self._lease_path(lease.slot.slot)
        with directory_lock(self._lock):
            if not path.exists():
                return
            payload = read_json(path)
            if payload.get("lease_id") != lease.lease_id:
                raise RuntimeError("stale environment slot lease cannot release")
            path.unlink()
            self._events.append(
                {
                    "event_id": "event-" + uuid.uuid4().hex,
                    "event": "slot_released",
                    "at": _timestamp(_now()),
                    "lease_id": lease.lease_id,
                    "job_id": lease.job_id,
                    "slot": lease.slot.slot,
                    "gpu": lease.slot.gpu,
                    "outcome": str(outcome),
                }
            )

    def snapshot(self) -> dict[str, Any]:
        with directory_lock(self._lock):
            leases = self._active_leases()
        return {
            "ready_manifest": str(self.ready_manifest),
            "active": len(leases),
            "maximum_active_slots": self.maximum_active_slots,
            "leases": leases,
        }
