# Copyright (c) 2026 Zetta Contributors
"""Tests for recoverable dynamic provider admission."""

from __future__ import annotations

import json
import threading
import time
from contextlib import ExitStack
from pathlib import Path

import pytest

from zetta.evolution.provider_admission import (
    ProviderAdmission,
    ProviderAdmissionCancelled,
    ProviderAdmissionConfig,
    ProviderAdmissionTimeout,
    ProviderFailure,
)


class FakeClock:
    """Thread-safe wall clock whose sleep advances logical time."""

    def __init__(self, now: float = 1_000.0) -> None:
        self._now = now
        self._lock = threading.Lock()

    def __call__(self) -> float:
        with self._lock:
            return self._now

    def advance(self, seconds: float) -> None:
        with self._lock:
            self._now += seconds

    def sleep(self, seconds: float) -> None:
        self.advance(seconds)


def _config(**overrides: object) -> ProviderAdmissionConfig:
    values: dict[str, object] = {
        "initial_limit": 20,
        "min_limit": 1,
        "max_limit": 40,
        "increase_step": 2,
        "healthy_successes_to_increase": 4,
        "cooldown_s": 10.0,
        "lease_ttl_s": 30.0,
        "poll_s": 0.002,
    }
    values.update(overrides)
    return ProviderAdmissionConfig(**values)  # type: ignore[arg-type]


def _route(pool: ProviderAdmission, route_id: str = "primary"):
    return pool.snapshot(route_id).routes[route_id]


def test_default_initial_limit_is_twenty_and_enforced(tmp_path: Path) -> None:
    pool = ProviderAdmission(tmp_path)
    assert pool.register_route("primary").limit == 20

    with ExitStack() as stack:
        for index in range(20):
            stack.enter_context(
                pool.acquire("primary", owner=f"worker-{index}", timeout_s=1)
            )
        snapshot = _route(pool)
        assert snapshot.active_leases == 20
        assert snapshot.rolling_rpm == 20
        with pytest.raises(ProviderAdmissionTimeout):
            with pool.acquire("primary", owner="overflow", timeout_s=0.02):
                pass
    assert _route(pool).active_leases == 0


def test_tpm_reservations_are_atomic_and_released(tmp_path: Path) -> None:
    pool = ProviderAdmission(
        tmp_path,
        config=_config(initial_limit=4, max_limit=4),
    )
    pool.register_route("primary", tpm_limit=100)
    with pool.acquire("primary", owner="first", timeout_s=1, estimated_tokens=60):
        assert _route(pool).reserved_tokens == 60
        with pytest.raises(ProviderAdmissionTimeout):
            with pool.acquire(
                "primary", owner="overflow", timeout_s=0.02, estimated_tokens=50
            ):
                pass
    assert _route(pool).reserved_tokens == 0
    with pool.acquire(
        "primary", owner="after-release", timeout_s=1, estimated_tokens=50
    ):
        assert _route(pool).reserved_tokens == 50


def test_hard_rpm_limit_waits_for_rolling_window(tmp_path: Path) -> None:
    clock = FakeClock()
    pool = ProviderAdmission(
        tmp_path,
        config=_config(initial_limit=4, max_limit=4),
        time_source=clock,
        sleep=clock.sleep,
    )
    pool.register_route("primary", rpm_limit=1)
    with pool.acquire("primary", owner="first", timeout_s=1):
        pass
    with pytest.raises(ProviderAdmissionTimeout):
        with pool.acquire("primary", owner="too-soon", timeout_s=0.02):
            pass
    clock.advance(61)
    with pool.acquire("primary", owner="next-window", timeout_s=1):
        pass


def test_thread_race_never_exceeds_route_limit(tmp_path: Path) -> None:
    pool = ProviderAdmission(
        tmp_path,
        config=_config(initial_limit=3, max_limit=3),
    )
    start = threading.Barrier(12)
    guard = threading.Lock()
    active = 0
    peak = 0
    errors: list[BaseException] = []

    def worker(index: int) -> None:
        nonlocal active, peak
        try:
            start.wait(timeout=2)
            with pool.acquire("primary", owner=f"worker-{index}", timeout_s=4):
                with guard:
                    active += 1
                    peak = max(peak, active)
                time.sleep(0.1)
                with guard:
                    active -= 1
        except BaseException as exc:  # pragma: no cover - assertion reports it
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(index,)) for index in range(12)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)
        assert not thread.is_alive()
    assert errors == []
    assert peak == 3
    assert _route(pool).active_leases == 0


def test_healthy_requests_expand_limit_stepwise(tmp_path: Path) -> None:
    clock = FakeClock()
    pool = ProviderAdmission(
        tmp_path,
        config=_config(
            initial_limit=2,
            max_limit=5,
            increase_step=1,
            healthy_successes_to_increase=2,
            min_latency_samples=2,
        ),
        time_source=clock,
        sleep=clock.sleep,
    )
    pool.register_route("primary")

    for _ in range(2):
        with pool.acquire("primary", owner="worker") as lease:
            lease.succeed(tokens=100, latency_s=1.0)
        clock.advance(0.1)
    assert _route(pool).limit == 3

    for _ in range(4):
        with pool.acquire("primary", owner="worker") as lease:
            lease.succeed(tokens=100, latency_s=1.0)
        clock.advance(0.1)
    assert _route(pool).limit == 5


def test_custom_route_limits_remain_usable_by_scheduler(tmp_path: Path) -> None:
    pool = ProviderAdmission(tmp_path, config=_config(max_limit=20))
    registered = pool.register_route("low-cost", initial_limit=3, max_limit=7)
    assert registered.initial_limit == 3
    assert registered.max_limit == 7
    with pool.acquire("low-cost", owner="campaign-worker", timeout_s=1):
        assert _route(pool, "low-cost").active_leases == 1
    assert _route(pool, "low-cost").active_leases == 0


@pytest.mark.parametrize(
    "failure",
    [
        ProviderFailure.RATE_LIMIT,
        ProviderFailure.QUOTA,
        ProviderFailure.TRANSPORT,
        ProviderFailure.STREAM,
        ProviderFailure.OOM,
    ],
)
def test_retryable_failures_multiply_backoff_and_cool(
    tmp_path: Path, failure: ProviderFailure
) -> None:
    clock = FakeClock()
    pool = ProviderAdmission(
        tmp_path,
        config=_config(initial_limit=20, decrease_factor=0.5, cooldown_s=30),
        time_source=clock,
        sleep=clock.sleep,
    )
    with pool.acquire("primary", owner="worker") as lease:
        lease.fail(failure, latency_s=2.0)

    route = _route(pool)
    assert route.limit == 10
    assert route.cooldown_remaining_s == pytest.approx(30)
    assert route.failed_requests == 1
    assert route.rolling_stream_errors == (
        1 if failure is ProviderFailure.STREAM else 0
    )


def test_p95_degradation_causes_multiplicative_backoff(tmp_path: Path) -> None:
    clock = FakeClock()
    pool = ProviderAdmission(
        tmp_path,
        config=_config(
            initial_limit=8,
            healthy_successes_to_increase=99,
            min_latency_samples=3,
            p95_latency_limit_s=2.0,
        ),
        time_source=clock,
        sleep=clock.sleep,
    )
    for latency in (1.0, 1.0, 10.0):
        with pool.acquire("primary", owner="worker") as lease:
            lease.succeed(tokens=10, latency_s=latency)
        clock.advance(0.1)

    route = _route(pool)
    assert route.p50_latency_s == pytest.approx(1.0)
    assert route.p95_latency_s == pytest.approx(9.1)
    assert route.limit == 4
    assert route.cooldown_remaining_s > 0


def test_routes_have_isolated_limits_metrics_and_cooldowns(tmp_path: Path) -> None:
    clock = FakeClock()
    pool = ProviderAdmission(
        tmp_path,
        config=_config(initial_limit=8),
        time_source=clock,
        sleep=clock.sleep,
    )
    pool.register_route("route-a")
    pool.register_route("route-b")
    with pool.acquire("route-a", owner="a") as lease:
        lease.fail(ProviderFailure.RATE_LIMIT, tokens=7, latency_s=3)
    with pool.acquire("route-b", owner="b") as lease:
        lease.succeed(tokens=111, latency_s=1)

    route_a = _route(pool, "route-a")
    route_b = _route(pool, "route-b")
    assert route_a.limit == 4
    assert route_a.cooldown_remaining_s > 0
    assert route_a.rolling_tpm == 7
    assert route_b.limit == 8
    assert route_b.cooldown_remaining_s == 0
    assert route_b.rolling_tpm == 111


def test_waiters_are_fifo_fair_within_a_route(tmp_path: Path) -> None:
    pool = ProviderAdmission(
        tmp_path,
        config=_config(initial_limit=1, max_limit=1),
    )
    holder_cm = pool.acquire("primary", owner="holder", timeout_s=1)
    holder_cm.__enter__()
    order: list[int] = []
    errors: list[BaseException] = []

    def waiter(index: int) -> None:
        try:
            with pool.acquire("primary", owner=f"waiter-{index}", timeout_s=3):
                order.append(index)
                time.sleep(0.015)
        except BaseException as exc:  # pragma: no cover - assertion reports it
            errors.append(exc)

    threads: list[threading.Thread] = []
    for index in range(3):
        thread = threading.Thread(target=waiter, args=(index,))
        thread.start()
        threads.append(thread)
        deadline = time.monotonic() + 2
        while _route(pool).waiting < index + 1 and time.monotonic() < deadline:
            time.sleep(0.002)
        assert _route(pool).waiting == index + 1

    holder_cm.__exit__(None, None, None)
    for thread in threads:
        thread.join(timeout=4)
        assert not thread.is_alive()
    assert errors == []
    assert order == [0, 1, 2]


def test_wait_can_be_cancelled_and_ticket_is_removed(tmp_path: Path) -> None:
    pool = ProviderAdmission(
        tmp_path,
        config=_config(initial_limit=1, max_limit=1),
    )
    cancel = threading.Event()
    cancel.set()
    with pytest.raises(ProviderAdmissionCancelled):
        with pool.acquire("primary", owner="cancelled", timeout_s=10, cancel=cancel):
            pass
    assert _route(pool).waiting == 0
    assert _route(pool).active_leases == 0


def test_expired_crash_lease_is_reaped_and_capacity_recovers(tmp_path: Path) -> None:
    clock = FakeClock()
    pool = ProviderAdmission(
        tmp_path,
        config=_config(initial_limit=1, max_limit=1, lease_ttl_s=5),
        time_source=clock,
        sleep=clock.sleep,
    )
    crashed_cm = pool.acquire("primary", owner="crashed", timeout_s=1)
    crashed_cm.__enter__()
    assert _route(pool).active_leases == 1

    clock.advance(6)
    assert pool.reap_expired() == 1
    assert _route(pool).active_leases == 0
    with pool.acquire("primary", owner="replacement", timeout_s=1):
        pass
    crashed_cm.__exit__(None, None, None)

    events = [
        json.loads(line)
        for line in (tmp_path / "provider-admission-events.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert sum(event["event"] == "lease_expired" for event in events) == 1


def test_state_snapshot_restores_limits_metrics_and_live_leases(tmp_path: Path) -> None:
    clock = FakeClock()
    config = _config(initial_limit=6, healthy_successes_to_increase=99)
    first = ProviderAdmission(
        tmp_path, config=config, time_source=clock, sleep=clock.sleep
    )
    with first.acquire("primary", owner="completed") as lease:
        lease.succeed(tokens=321, latency_s=2.5)
    live_cm = first.acquire("primary", owner="still-running")
    live_cm.__enter__()

    restored = ProviderAdmission(
        tmp_path, config=config, time_source=clock, sleep=clock.sleep
    )
    route = _route(restored)
    assert route.limit == 6
    assert route.active_leases == 1
    assert route.rolling_rpm == 2
    assert route.rolling_tpm == 321
    assert route.p50_latency_s == pytest.approx(2.5)
    assert route.successful_requests == 1
    live_cm.__exit__(None, None, None)


def test_metrics_roll_out_of_rpm_and_tpm_windows(tmp_path: Path) -> None:
    clock = FakeClock()
    pool = ProviderAdmission(
        tmp_path,
        config=_config(rpm_window_s=60, metric_window_s=300),
        time_source=clock,
        sleep=clock.sleep,
    )
    with pool.acquire("primary", owner="worker") as lease:
        lease.succeed(tokens=500, latency_s=4)
    assert _route(pool).rolling_rpm == 1
    assert _route(pool).rolling_tpm == 500

    clock.advance(61)
    route = _route(pool)
    assert route.rolling_rpm == 0
    assert route.rolling_tpm == 0
    assert route.p95_latency_s == pytest.approx(4)


def test_ledger_and_snapshot_never_store_owner_key_url_or_prompt(
    tmp_path: Path,
) -> None:
    pool = ProviderAdmission(tmp_path, config=_config(initial_limit=1))
    sensitive_owner = (
        "https://provider.example/v1 sk-super-secret "
        "prompt=Ignore all previous instructions"
    )
    with pool.acquire("safe-route", owner=sensitive_owner) as lease:
        lease.fail(ProviderFailure.TRANSPORT, latency_s=1)

    persisted = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (
            tmp_path / "provider-admission-state.json",
            tmp_path / "provider-admission-events.jsonl",
        )
    )
    for fragment in (
        "provider.example",
        "sk-super-secret",
        "Ignore all previous instructions",
        "prompt=",
    ):
        assert fragment not in persisted

    with pytest.raises(ValueError):
        pool.register_route("https://provider.example/v1")


def test_lease_heartbeat_extends_expiry_and_context_releases(tmp_path: Path) -> None:
    clock = FakeClock()
    pool = ProviderAdmission(
        tmp_path,
        config=_config(initial_limit=1, max_limit=1, lease_ttl_s=5),
        time_source=clock,
        sleep=clock.sleep,
    )
    with pool.acquire("primary", owner="stream") as lease:
        clock.advance(4)
        lease.heartbeat()
        clock.advance(4)
        assert pool.reap_expired() == 0
        assert _route(pool).active_leases == 1
    assert _route(pool).active_leases == 0


def test_invalid_config_metrics_and_failure_inputs_fail_closed(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        ProviderAdmissionConfig(initial_limit=0)
    pool = ProviderAdmission(tmp_path)
    cm = pool.acquire("primary", owner="worker")
    lease = cm.__enter__()
    with pytest.raises(ValueError):
        lease.succeed(tokens=-1, latency_s=1)
    with pytest.raises(ValueError):
        lease.fail("raw exception: https://secret.invalid", latency_s=1)
    cm.__exit__(None, None, None)
