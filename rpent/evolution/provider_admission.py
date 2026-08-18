# Copyright (c) 2026 RPent Contributors
"""Recoverable, adaptive admission control for model-provider routes.

The controller deliberately knows routes only by a short, public alias.  It
does not accept provider URLs, API keys, prompts, response bodies, or exception
messages, so none of those values can accidentally reach its state or audit
ledger.  State is shared through the filesystem and all mutations are guarded
by a recoverable cross-process lock.

The public scheduling API is:

* :meth:`ProviderAdmission.register_route` to declare a route;
* :meth:`ProviderAdmission.acquire` for a fair, blocking lease;
* :meth:`ProviderLease.succeed` / :meth:`ProviderLease.fail` to feed metrics;
* :meth:`ProviderAdmission.snapshot` for routing and observability; and
* :meth:`ProviderAdmission.reap_expired` for explicit maintenance.

Limits use additive increase and multiplicative decrease (AIMD).  Provider
failures immediately reduce only the affected route and start a cooldown.
Healthy completions gradually restore capacity up to the configured maximum.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import socket
import threading
import time
import uuid
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Protocol

from rpent.evolution.jsonio import atomic_write_json, canonical_json_bytes

__all__ = [
    "ProviderAdmission",
    "ProviderAdmissionCancelled",
    "ProviderAdmissionConfig",
    "ProviderAdmissionSnapshot",
    "ProviderAdmissionTimeout",
    "ProviderFailure",
    "ProviderLease",
    "RouteSnapshot",
]


_ROUTE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,79}$")
_STATE_VERSION = 1
_LOCAL_LOCKS_GUARD = threading.Lock()
_LOCAL_LOCKS: dict[str, threading.Lock] = {}


class _Cancellation(Protocol):
    def is_set(self) -> bool: ...


class ProviderAdmissionTimeout(TimeoutError):
    """Raised when a fair admission ticket reaches its deadline."""


class ProviderAdmissionCancelled(RuntimeError):
    """Raised when the caller cancels while waiting for admission."""


class ProviderFailure(str, Enum):
    """Sanitized provider failure classes accepted by the controller."""

    RATE_LIMIT = "rate_limit"
    QUOTA = "quota"
    TRANSPORT = "transport"
    STREAM = "stream"
    OOM = "oom"
    OTHER_RETRYABLE = "other_retryable"


@dataclass(frozen=True)
class ProviderAdmissionConfig:
    """Adaptive admission policy.

    ``initial_limit`` intentionally defaults to 20.  The healthy threshold is
    per route and counts successful completions since the previous limit
    adjustment.  Latency degradation is evaluated only after
    ``min_latency_samples`` observations, avoiding a single cold request from
    causing an immediate reduction.
    """

    initial_limit: int = 20
    min_limit: int = 1
    max_limit: int = 100
    increase_step: int = 2
    healthy_successes_to_increase: int = 20
    decrease_factor: float = 0.5
    cooldown_s: float = 30.0
    rpm_window_s: float = 60.0
    metric_window_s: float = 300.0
    max_metric_samples: int = 4096
    p95_latency_limit_s: float = 120.0
    p95_degrade_ratio: float = 1.75
    min_latency_samples: int = 8
    lease_ttl_s: float = 900.0
    waiter_ttl_s: float = 3600.0
    poll_s: float = 0.05
    lock_timeout_s: float = 30.0
    lock_stale_s: float = 120.0

    def __post_init__(self) -> None:
        integer_positive = (
            "initial_limit",
            "min_limit",
            "max_limit",
            "increase_step",
            "healthy_successes_to_increase",
            "max_metric_samples",
            "min_latency_samples",
        )
        for name in integer_positive:
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if not self.min_limit <= self.initial_limit <= self.max_limit:
            raise ValueError(
                "limits must satisfy min_limit <= initial_limit <= max_limit"
            )
        if not 0.0 < self.decrease_factor < 1.0:
            raise ValueError("decrease_factor must be between zero and one")
        for name in (
            "cooldown_s",
            "rpm_window_s",
            "metric_window_s",
            "p95_latency_limit_s",
            "lease_ttl_s",
            "waiter_ttl_s",
            "poll_s",
            "lock_timeout_s",
            "lock_stale_s",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.p95_degrade_ratio <= 1.0:
            raise ValueError("p95_degrade_ratio must be greater than one")


@dataclass(frozen=True)
class RouteSnapshot:
    """Secret-free health and capacity view for one route."""

    route_id: str
    limit: int
    initial_limit: int
    max_limit: int
    rpm_limit: int | None
    tpm_limit: int | None
    reserved_tokens: int
    active_leases: int
    waiting: int
    cooldown_remaining_s: float
    rolling_rpm: int
    rolling_tpm: int
    rolling_stream_errors: int
    rolling_failures: int
    p50_latency_s: float | None
    p95_latency_s: float | None
    successful_requests: int
    failed_requests: int


@dataclass(frozen=True)
class ProviderAdmissionSnapshot:
    """Controller state suitable for campaign scheduling and metrics."""

    generated_at_unix: float
    routes: Mapping[str, RouteSnapshot]
    active_leases: int
    waiting: int


class ProviderLease:
    """A provider-capacity lease returned by :meth:`ProviderAdmission.acquire`."""

    def __init__(
        self,
        controller: ProviderAdmission,
        *,
        lease_id: str,
        route_id: str,
        owner_sha256: str,
    ) -> None:
        self._controller = controller
        self.lease_id = lease_id
        self.route_id = route_id
        self.owner_sha256 = owner_sha256
        self._released = False
        self._reported = False
        self._guard = threading.Lock()

    @property
    def released(self) -> bool:
        """Whether this local handle has released its persistent lease."""

        with self._guard:
            return self._released

    def heartbeat(self) -> None:
        """Extend the lease TTL while a long provider stream remains active."""

        with self._guard:
            if self._released:
                raise RuntimeError("cannot heartbeat a released provider lease")
            self._controller.heartbeat(self.lease_id)

    def succeed(self, *, tokens: int, latency_s: float) -> None:
        """Record one successful request and release the lease."""

        self._finish(success=True, tokens=tokens, latency_s=latency_s, failure=None)

    def fail(
        self,
        failure: ProviderFailure | str,
        *,
        tokens: int = 0,
        latency_s: float | None = None,
    ) -> None:
        """Record a sanitized retryable failure and release the lease."""

        try:
            normalized = ProviderFailure(failure)
        except ValueError as exc:
            raise ValueError("unsupported provider failure class") from exc
        self._finish(
            success=False,
            tokens=tokens,
            latency_s=latency_s,
            failure=normalized,
        )

    def release(self) -> None:
        """Release without inventing an outcome when the caller did not report one."""

        with self._guard:
            if self._released:
                return
            self._controller.release(self.lease_id, outcome_reported=self._reported)
            self._released = True

    def _finish(
        self,
        *,
        success: bool,
        tokens: int,
        latency_s: float | None,
        failure: ProviderFailure | None,
    ) -> None:
        with self._guard:
            if self._released or self._reported:
                raise RuntimeError("provider lease outcome was already finalized")
            self._controller.complete(
                self.lease_id,
                success=success,
                tokens=tokens,
                latency_s=latency_s,
                failure=failure,
            )
            self._reported = True
            self._released = True

    def __enter__(self) -> ProviderLease:
        return self

    def __exit__(self, _type: object, _value: object, _traceback: object) -> None:
        self.release()


class ProviderAdmission:
    """Cross-process, per-route adaptive provider admission controller."""

    def __init__(
        self,
        root: str | Path,
        *,
        config: ProviderAdmissionConfig | None = None,
        time_source: Callable[[], float] = time.time,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.config = config or ProviderAdmissionConfig()
        self._now = time_source
        self._sleep = sleep
        self._state_path = self.root / "provider-admission-state.json"
        self._ledger_path = self.root / "provider-admission-events.jsonl"
        self._ledger_lock = self.root / ".provider-admission-ledger.lock"
        self._state_lock = self.root / ".provider-admission-state.lock"
        if not self._state_path.exists():
            with self._locked():
                if not self._state_path.exists():
                    self._write_state(self._new_state())
        self.reap_expired()

    def register_route(
        self,
        route_id: str,
        *,
        initial_limit: int | None = None,
        max_limit: int | None = None,
        rpm_limit: int | None = None,
        tpm_limit: int | None = None,
    ) -> RouteSnapshot:
        """Register a public route alias and return its current snapshot."""

        route_id = _validate_route_id(route_id)
        initial_was_explicit = initial_limit is not None
        maximum_was_explicit = max_limit is not None
        initial = self.config.initial_limit if initial_limit is None else initial_limit
        maximum = self.config.max_limit if max_limit is None else max_limit
        if (
            not isinstance(initial, int)
            or isinstance(initial, bool)
            or not isinstance(maximum, int)
            or isinstance(maximum, bool)
            or not self.config.min_limit <= initial <= maximum <= self.config.max_limit
        ):
            raise ValueError(
                "route limits must satisfy min_limit <= initial <= max_limit <= configured max_limit"
            )
        for name, value in (("rpm_limit", rpm_limit), ("tpm_limit", tpm_limit)):
            if value is not None and (
                not isinstance(value, int) or isinstance(value, bool) or value < 1
            ):
                raise ValueError(f"{name} must be a positive integer or null")
        event: dict[str, Any] | None = None
        with self._locked():
            state = self._read_state()
            now = self._now()
            self._prune(state, now)
            route = state["routes"].get(route_id)
            if route is None:
                route = _new_route(
                    initial,
                    maximum,
                    rpm_limit=rpm_limit,
                    tpm_limit=tpm_limit,
                )
                state["routes"][route_id] = route
                event = self._event(
                    "route_registered",
                    route_id=route_id,
                    now=now,
                    limit=initial,
                )
                self._write_state(state)
            elif (
                (initial_was_explicit and route["initial_limit"] != initial)
                or (maximum_was_explicit and route["max_limit"] != maximum)
                or (rpm_limit is not None and route.get("rpm_limit") != rpm_limit)
                or (tpm_limit is not None and route.get("tpm_limit") != tpm_limit)
            ):
                raise ValueError("route is already registered with different limits")
            snapshot = self._route_snapshot(state, route_id, now)
        if event:
            self._append_event(event)
        return snapshot

    @contextmanager
    def acquire(
        self,
        route_id: str,
        *,
        owner: str,
        timeout_s: float = 1800.0,
        cancel: _Cancellation | None = None,
        poll_s: float | None = None,
        estimated_tokens: int = 0,
    ) -> Iterator[ProviderLease]:
        """Acquire a FIFO lease for ``route_id``.

        Fairness is enforced independently within each route.  Waiting tickets
        are persisted, so process crashes cannot allow new callers to jump an
        older live ticket.  Cancellation objects need only implement
        ``is_set()`` (for example :class:`threading.Event`).
        """

        route_id = _validate_route_id(route_id)
        if not isinstance(owner, str) or not owner:
            raise ValueError("owner must be a non-empty string")
        if timeout_s < 0:
            raise ValueError("timeout_s must be non-negative")
        if (
            not isinstance(estimated_tokens, int)
            or isinstance(estimated_tokens, bool)
            or estimated_tokens < 0
        ):
            raise ValueError("estimated_tokens must be a non-negative integer")
        interval = self.config.poll_s if poll_s is None else poll_s
        if interval <= 0:
            raise ValueError("poll_s must be positive")
        self.register_route(route_id)
        owner_digest = hashlib.sha256(owner.encode("utf-8")).hexdigest()
        ticket_id = uuid.uuid4().hex
        created = self._now()
        deadline = created + timeout_s
        ticket_event: dict[str, Any] | None = None
        with self._locked():
            state = self._read_state()
            self._prune(state, created)
            state["waiters"].append(
                {
                    "ticket_id": ticket_id,
                    "route_id": route_id,
                    "owner_sha256": owner_digest,
                    "created_at_unix": created,
                    "expires_at_unix": min(
                        deadline,
                        created + self.config.waiter_ttl_s,
                    ),
                }
            )
            ticket_event = self._event(
                "wait_started", route_id=route_id, now=created, ticket_id=ticket_id
            )
            self._write_state(state)
        self._append_event(ticket_event)

        lease: ProviderLease | None = None
        try:
            while lease is None:
                now = self._now()
                if cancel is not None and cancel.is_set():
                    self._remove_waiter(ticket_id, route_id, "wait_cancelled")
                    raise ProviderAdmissionCancelled("provider admission was cancelled")
                if now >= deadline:
                    self._remove_waiter(ticket_id, route_id, "wait_timed_out")
                    raise ProviderAdmissionTimeout(
                        "timed out waiting for provider admission"
                    )
                event: dict[str, Any] | None = None
                with self._locked():
                    state = self._read_state()
                    self._prune(state, now)
                    route = state["routes"][route_id]
                    same_route = [
                        row for row in state["waiters"] if row["route_id"] == route_id
                    ]
                    at_front = (
                        bool(same_route) and same_route[0]["ticket_id"] == ticket_id
                    )
                    active = _active_for_route(state, route_id)
                    recent_samples = [
                        sample
                        for sample in route["samples"]
                        if sample["at_unix"] >= now - self.config.rpm_window_s
                    ]
                    rolling_tokens = sum(
                        int(sample["tokens"]) for sample in recent_samples
                    )
                    reserved_tokens = sum(
                        int(item.get("reserved_tokens", 0))
                        for item in state["leases"].values()
                        if item["route_id"] == route_id
                    )
                    rolling_rpm = len(
                        [
                            value
                            for value in route["request_starts"]
                            if value >= now - self.config.rpm_window_s
                        ]
                    )
                    within_rpm = route.get("rpm_limit") is None or (
                        rolling_rpm < int(route["rpm_limit"])
                    )
                    within_tpm = route.get("tpm_limit") is None or (
                        rolling_tokens + reserved_tokens + estimated_tokens
                        <= int(route["tpm_limit"])
                    )
                    if (
                        at_front
                        and active < route["limit"]
                        and now >= route["cooldown_until_unix"]
                        and within_rpm
                        and within_tpm
                    ):
                        state["waiters"] = [
                            row
                            for row in state["waiters"]
                            if row["ticket_id"] != ticket_id
                        ]
                        lease_id = uuid.uuid4().hex
                        state["leases"][lease_id] = {
                            "lease_id": lease_id,
                            "route_id": route_id,
                            "owner_sha256": owner_digest,
                            "acquired_at_unix": now,
                            "expires_at_unix": now + self.config.lease_ttl_s,
                            "reserved_tokens": estimated_tokens,
                        }
                        route["request_starts"].append(now)
                        event = self._event(
                            "lease_acquired",
                            route_id=route_id,
                            now=now,
                            lease_id=lease_id,
                            limit=route["limit"],
                        )
                        self._write_state(state)
                        lease = ProviderLease(
                            self,
                            lease_id=lease_id,
                            route_id=route_id,
                            owner_sha256=owner_digest,
                        )
                if event:
                    self._append_event(event)
                if lease is None:
                    self._sleep(interval)
            try:
                yield lease
            finally:
                lease.release()
        except BaseException:
            if lease is None:
                self._remove_waiter(ticket_id, route_id, "wait_abandoned")
            raise

    def heartbeat(self, lease_id: str) -> None:
        """Renew a live lease after validating its opaque identifier."""

        _validate_opaque_id(lease_id, "lease_id")
        event: dict[str, Any]
        with self._locked():
            state = self._read_state()
            now = self._now()
            self._prune(state, now)
            lease = state["leases"].get(lease_id)
            if lease is None:
                raise KeyError("provider lease is not active")
            lease["expires_at_unix"] = now + self.config.lease_ttl_s
            event = self._event(
                "lease_heartbeat",
                route_id=lease["route_id"],
                now=now,
                lease_id=lease_id,
            )
            self._write_state(state)
        self._append_event(event)

    def complete(
        self,
        lease_id: str,
        *,
        success: bool,
        tokens: int,
        latency_s: float | None,
        failure: ProviderFailure | None = None,
    ) -> None:
        """Finalize a lease with metrics and apply the adaptive policy."""

        _validate_opaque_id(lease_id, "lease_id")
        if not isinstance(tokens, int) or isinstance(tokens, bool) or tokens < 0:
            raise ValueError("tokens must be a non-negative integer")
        if latency_s is not None and (
            not isinstance(latency_s, (int, float))
            or isinstance(latency_s, bool)
            or not math.isfinite(float(latency_s))
            or latency_s < 0
        ):
            raise ValueError("latency_s must be finite and non-negative")
        if success and failure is not None:
            raise ValueError("a successful completion cannot include a failure")
        if not success and failure is None:
            raise ValueError("a failed completion requires a sanitized failure class")

        events: list[dict[str, Any]] = []
        with self._locked():
            state = self._read_state()
            now = self._now()
            self._prune(state, now)
            lease = state["leases"].pop(lease_id, None)
            if lease is None:
                raise KeyError("provider lease is not active")
            route_id = lease["route_id"]
            route = state["routes"][route_id]
            sample = {
                "at_unix": now,
                "tokens": tokens,
                "latency_s": None if latency_s is None else float(latency_s),
                "success": success,
                "stream_error": failure is ProviderFailure.STREAM,
                "failure": None if failure is None else failure.value,
            }
            route["samples"].append(sample)
            if success:
                route["successful_requests"] += 1
                route["healthy_successes"] += 1
            else:
                route["failed_requests"] += 1
                route["healthy_successes"] = 0
            events.append(
                self._event(
                    "request_completed" if success else "request_failed",
                    route_id=route_id,
                    now=now,
                    lease_id=lease_id,
                    outcome="success" if success else failure.value,
                    tokens=tokens,
                    latency_s=sample["latency_s"],
                )
            )

            old_limit = route["limit"]
            adjustment_reason: str | None = None
            if not success:
                adjustment_reason = failure.value
                self._decrease(route, now)
            else:
                latencies = _latencies(route["samples"])
                p95 = _percentile(latencies, 0.95)
                enough = len(latencies) >= self.config.min_latency_samples
                reference = route["best_p95_latency_s"]
                degraded = (
                    enough
                    and p95 is not None
                    and (
                        p95 > self.config.p95_latency_limit_s
                        or (
                            reference is not None
                            and p95 > reference * self.config.p95_degrade_ratio
                        )
                    )
                )
                if degraded:
                    adjustment_reason = "p95_degraded"
                    self._decrease(route, now)
                elif now >= route["cooldown_until_unix"]:
                    if enough and p95 is not None:
                        if reference is None or p95 < reference:
                            route["best_p95_latency_s"] = p95
                    if (
                        route["healthy_successes"]
                        >= self.config.healthy_successes_to_increase
                        and route["limit"] < route["max_limit"]
                    ):
                        route["limit"] = min(
                            route["max_limit"],
                            route["limit"] + self.config.increase_step,
                        )
                        route["healthy_successes"] = 0
                        adjustment_reason = "healthy_increase"
            if route["limit"] != old_limit:
                events.append(
                    self._event(
                        "limit_changed",
                        route_id=route_id,
                        now=now,
                        old_limit=old_limit,
                        limit=route["limit"],
                        reason=adjustment_reason,
                        cooldown_until_unix=route["cooldown_until_unix"],
                    )
                )
            self._write_state(state)
        for event in events:
            self._append_event(event)

    def release(self, lease_id: str, *, outcome_reported: bool = False) -> bool:
        """Release a lease without fabricating provider metrics.

        Returns ``False`` when another process already completed or reaped the
        lease, making cleanup naturally idempotent.
        """

        _validate_opaque_id(lease_id, "lease_id")
        event: dict[str, Any] | None = None
        with self._locked():
            state = self._read_state()
            now = self._now()
            self._prune(state, now)
            lease = state["leases"].pop(lease_id, None)
            if lease is not None:
                event = self._event(
                    "lease_released",
                    route_id=lease["route_id"],
                    now=now,
                    lease_id=lease_id,
                    outcome_reported=bool(outcome_reported),
                )
                self._write_state(state)
        if event:
            self._append_event(event)
            return True
        return False

    def reap_expired(self) -> int:
        """Recover capacity from crashed callers and stale waiting tickets."""

        events: list[dict[str, Any]] = []
        with self._locked():
            state = self._read_state()
            now = self._now()
            before_leases = dict(state["leases"])
            before_waiters = list(state["waiters"])
            self._prune(state, now)
            for lease_id, lease in before_leases.items():
                if lease_id not in state["leases"]:
                    events.append(
                        self._event(
                            "lease_expired",
                            route_id=lease["route_id"],
                            now=now,
                            lease_id=lease_id,
                        )
                    )
            live_tickets = {row["ticket_id"] for row in state["waiters"]}
            for waiter in before_waiters:
                if waiter["ticket_id"] not in live_tickets:
                    events.append(
                        self._event(
                            "waiter_expired",
                            route_id=waiter["route_id"],
                            now=now,
                            ticket_id=waiter["ticket_id"],
                        )
                    )
            if events:
                self._write_state(state)
        for event in events:
            self._append_event(event)
        return sum(event["event"] == "lease_expired" for event in events)

    def snapshot(self, route_id: str | None = None) -> ProviderAdmissionSnapshot:
        """Return a restored, pruned snapshot without exposing caller data."""

        if route_id is not None:
            route_id = _validate_route_id(route_id)
        with self._locked():
            state = self._read_state()
            now = self._now()
            changed = self._prune(state, now)
            if changed:
                self._write_state(state)
            names = sorted(state["routes"])
            if route_id is not None:
                if route_id not in state["routes"]:
                    raise KeyError(f"unknown provider route: {route_id}")
                names = [route_id]
            routes = {name: self._route_snapshot(state, name, now) for name in names}
            return ProviderAdmissionSnapshot(
                generated_at_unix=now,
                routes=routes,
                active_leases=sum(item.active_leases for item in routes.values()),
                waiting=sum(item.waiting for item in routes.values()),
            )

    def _route_snapshot(
        self, state: dict[str, Any], route_id: str, now: float
    ) -> RouteSnapshot:
        route = state["routes"][route_id]
        samples = route["samples"]
        latencies = _latencies(samples)
        recent_starts = [
            value
            for value in route["request_starts"]
            if value >= now - self.config.rpm_window_s
        ]
        minute_samples = [
            sample
            for sample in samples
            if sample["at_unix"] >= now - self.config.rpm_window_s
        ]
        return RouteSnapshot(
            route_id=route_id,
            limit=route["limit"],
            initial_limit=route["initial_limit"],
            max_limit=route["max_limit"],
            rpm_limit=route.get("rpm_limit"),
            tpm_limit=route.get("tpm_limit"),
            reserved_tokens=sum(
                int(item.get("reserved_tokens", 0))
                for item in state["leases"].values()
                if item["route_id"] == route_id
            ),
            active_leases=_active_for_route(state, route_id),
            waiting=sum(waiter["route_id"] == route_id for waiter in state["waiters"]),
            cooldown_remaining_s=max(0.0, route["cooldown_until_unix"] - now),
            rolling_rpm=len(recent_starts),
            rolling_tpm=sum(sample["tokens"] for sample in minute_samples),
            rolling_stream_errors=sum(
                bool(sample["stream_error"]) for sample in minute_samples
            ),
            rolling_failures=sum(
                not bool(sample["success"]) for sample in minute_samples
            ),
            p50_latency_s=_percentile(latencies, 0.50),
            p95_latency_s=_percentile(latencies, 0.95),
            successful_requests=route["successful_requests"],
            failed_requests=route["failed_requests"],
        )

    def _decrease(self, route: dict[str, Any], now: float) -> None:
        route["limit"] = max(
            self.config.min_limit,
            int(math.floor(route["limit"] * self.config.decrease_factor)),
        )
        route["cooldown_until_unix"] = max(
            route["cooldown_until_unix"], now + self.config.cooldown_s
        )
        route["healthy_successes"] = 0

    def _remove_waiter(self, ticket_id: str, route_id: str, event_name: str) -> None:
        event: dict[str, Any] | None = None
        with self._locked():
            state = self._read_state()
            now = self._now()
            before = len(state["waiters"])
            state["waiters"] = [
                row for row in state["waiters"] if row["ticket_id"] != ticket_id
            ]
            if len(state["waiters"]) != before:
                event = self._event(
                    event_name,
                    route_id=route_id,
                    now=now,
                    ticket_id=ticket_id,
                )
                self._write_state(state)
        if event:
            self._append_event(event)

    def _prune(self, state: dict[str, Any], now: float) -> bool:
        changed = False
        expired_leases = [
            lease_id
            for lease_id, lease in state["leases"].items()
            if lease["expires_at_unix"] <= now
        ]
        for lease_id in expired_leases:
            del state["leases"][lease_id]
            changed = True
        live_waiters = [
            waiter for waiter in state["waiters"] if waiter["expires_at_unix"] > now
        ]
        if len(live_waiters) != len(state["waiters"]):
            state["waiters"] = live_waiters
            changed = True
        for route in state["routes"].values():
            starts = [
                value
                for value in route["request_starts"]
                if value >= now - self.config.rpm_window_s
            ]
            samples = [
                sample
                for sample in route["samples"]
                if sample["at_unix"] >= now - self.config.metric_window_s
            ][-self.config.max_metric_samples :]
            if starts != route["request_starts"]:
                route["request_starts"] = starts
                changed = True
            if samples != route["samples"]:
                route["samples"] = samples
                changed = True
        return changed

    def _new_state(self) -> dict[str, Any]:
        return {
            "version": _STATE_VERSION,
            "routes": {},
            "leases": {},
            "waiters": [],
        }

    def _read_state(self) -> dict[str, Any]:
        try:
            value = json.loads(self._state_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return self._new_state()
        if not isinstance(value, dict) or value.get("version") != _STATE_VERSION:
            raise ValueError("unsupported provider admission state snapshot")
        if not isinstance(value.get("routes"), dict):
            raise ValueError("provider admission snapshot has invalid routes")
        if not isinstance(value.get("leases"), dict):
            raise ValueError("provider admission snapshot has invalid leases")
        if not isinstance(value.get("waiters"), list):
            raise ValueError("provider admission snapshot has invalid waiters")
        for route in value["routes"].values():
            if not isinstance(route, dict):
                raise ValueError("provider admission snapshot has malformed route")
            route.setdefault("rpm_limit", None)
            route.setdefault("tpm_limit", None)
        for lease in value["leases"].values():
            if not isinstance(lease, dict):
                raise ValueError("provider admission snapshot has malformed lease")
            lease.setdefault("reserved_tokens", 0)
        return value

    def _write_state(self, state: dict[str, Any]) -> None:
        atomic_write_json(self._state_path, state, overwrite=True)

    def _event(
        self, event: str, *, route_id: str, now: float, **fields: Any
    ) -> dict[str, Any]:
        allowed = {
            "cooldown_until_unix",
            "latency_s",
            "lease_id",
            "limit",
            "old_limit",
            "outcome",
            "outcome_reported",
            "reason",
            "ticket_id",
            "tokens",
        }
        if not set(fields).issubset(allowed):
            raise ValueError("unsafe provider admission audit field")
        return {
            "event_id": uuid.uuid4().hex,
            "at_unix": now,
            "event": event,
            "route_id": route_id,
            **fields,
        }

    def _append_event(self, event: dict[str, Any]) -> None:
        payload = canonical_json_bytes(event)
        with _recoverable_directory_lock(
            self._ledger_lock,
            timeout_s=self.config.lock_timeout_s,
            stale_s=self.config.lock_stale_s,
            now=self._now,
            sleep=self._sleep,
        ):
            with self._ledger_path.open("ab") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())

    @contextmanager
    def _locked(self) -> Iterator[None]:
        with _recoverable_directory_lock(
            self._state_lock,
            timeout_s=self.config.lock_timeout_s,
            stale_s=self.config.lock_stale_s,
            now=self._now,
            sleep=self._sleep,
        ):
            yield


def _new_route(
    initial_limit: int,
    max_limit: int,
    *,
    rpm_limit: int | None = None,
    tpm_limit: int | None = None,
) -> dict[str, Any]:
    return {
        "initial_limit": initial_limit,
        "max_limit": max_limit,
        "rpm_limit": rpm_limit,
        "tpm_limit": tpm_limit,
        "limit": initial_limit,
        "cooldown_until_unix": 0.0,
        "healthy_successes": 0,
        "best_p95_latency_s": None,
        "successful_requests": 0,
        "failed_requests": 0,
        "request_starts": [],
        "samples": [],
    }


def _validate_route_id(route_id: str) -> str:
    if not isinstance(route_id, str) or not _ROUTE_ID.fullmatch(route_id):
        raise ValueError(
            "route_id must be a public alias containing only letters, digits, '.', '_', or '-'"
        )
    return route_id


def _validate_opaque_id(value: str, name: str) -> None:
    if not isinstance(value, str) or not re.fullmatch(r"[a-f0-9]{32}", value):
        raise ValueError(f"{name} is not a valid opaque identifier")


def _active_for_route(state: dict[str, Any], route_id: str) -> int:
    return sum(lease["route_id"] == route_id for lease in state["leases"].values())


def _latencies(samples: list[dict[str, Any]]) -> list[float]:
    return sorted(
        float(sample["latency_s"])
        for sample in samples
        if sample.get("latency_s") is not None
    )


def _percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    position = (len(values) - 1) * quantile
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return values[lower]
    weight = position - lower
    return values[lower] * (1.0 - weight) + values[upper] * weight


@contextmanager
def _recoverable_directory_lock(
    path: Path,
    *,
    timeout_s: float,
    stale_s: float,
    now: Callable[[], float],
    sleep: Callable[[float], None],
) -> Iterator[None]:
    """Serialize local threads before taking the cross-process lock."""

    key = str(path.resolve())
    with _LOCAL_LOCKS_GUARD:
        local_lock = _LOCAL_LOCKS.setdefault(key, threading.Lock())
    with local_lock:
        with _filesystem_directory_lock(
            path,
            timeout_s=timeout_s,
            stale_s=stale_s,
            now=now,
            sleep=sleep,
        ):
            yield


@contextmanager
def _filesystem_directory_lock(
    path: Path,
    *,
    timeout_s: float,
    stale_s: float,
    now: Callable[[], float],
    sleep: Callable[[float], None],
) -> Iterator[None]:
    """Acquire an atomic directory lock whose abandoned owner can be reaped."""

    token = uuid.uuid4().hex
    deadline = time.monotonic() + timeout_s
    owner_path = path / "owner.json"
    while True:
        try:
            path.mkdir(parents=False)
            owner = {
                "token": token,
                "pid": os.getpid(),
                "host_sha256": hashlib.sha256(
                    socket.gethostname().encode("utf-8")
                ).hexdigest(),
                "created_at_unix": now(),
            }
            atomic_write_json(owner_path, owner, overwrite=False)
            break
        except FileExistsError:
            stale = False
            try:
                owner = json.loads(owner_path.read_text(encoding="utf-8"))
                stale = now() - float(owner["created_at_unix"]) > stale_s
            except (
                FileNotFoundError,
                KeyError,
                TypeError,
                ValueError,
                json.JSONDecodeError,
            ):
                try:
                    stale = now() - path.stat().st_mtime > stale_s
                except FileNotFoundError:
                    continue
            if stale:
                quarantine = path.with_name(f".{path.name}.stale.{uuid.uuid4().hex}")
                try:
                    os.replace(path, quarantine)
                except (FileNotFoundError, OSError):
                    pass
                else:
                    shutil.rmtree(quarantine, ignore_errors=True)
                    continue
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"timed out acquiring provider admission lock: {path}"
                )
            sleep(min(0.05, max(0.001, timeout_s / 100.0)))
    try:
        yield
    finally:
        try:
            owner = json.loads(owner_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            owner = {}
        if owner.get("token") == token:
            shutil.rmtree(path, ignore_errors=True)
