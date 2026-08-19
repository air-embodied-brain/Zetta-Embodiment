# Copyright (c) 2026 Zetta Contributors
from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from zetta.evolution.provider_admission import ProviderFailure
from zetta.planner.provider_request_admission import ProviderRequestAdmission


def test_global_limit_bounds_different_routes(tmp_path: Path) -> None:
    async def scenario() -> None:
        admission = ProviderRequestAdmission(
            tmp_path, initial_limit=1, max_limit=1, acquire_timeout_s=2
        )
        first = await admission.acquire("route-a")
        second_started = asyncio.Event()

        async def acquire_second():
            second_started.set()
            return await admission.acquire("route-b")

        pending = asyncio.create_task(acquire_second())
        await second_started.wait()
        await asyncio.sleep(0.1)
        assert not pending.done()
        await first.succeed(tokens=10, latency_s=0.1)
        second = await asyncio.wait_for(pending, timeout=2)
        await second.succeed(tokens=20, latency_s=0.2)
        snapshot = admission.global_admission.snapshot(
            ProviderRequestAdmission.GLOBAL_ROUTE_ID
        )
        assert snapshot.active_leases == 0
        assert (
            snapshot.routes[
                ProviderRequestAdmission.GLOBAL_ROUTE_ID
            ].successful_requests
            == 2
        )

    asyncio.run(scenario())


def test_route_failure_does_not_cool_global_fallback_capacity(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        admission = ProviderRequestAdmission(
            tmp_path, initial_limit=1, max_limit=1, acquire_timeout_s=2
        )
        failed = await admission.acquire("route-a")
        await failed.fail(ProviderFailure.RATE_LIMIT, tokens=0, latency_s=0.1)
        backup = await asyncio.wait_for(admission.acquire("route-b"), timeout=2)
        await backup.succeed(tokens=1, latency_s=0.1)
        global_route = admission.global_admission.snapshot(
            ProviderRequestAdmission.GLOBAL_ROUTE_ID
        ).routes[ProviderRequestAdmission.GLOBAL_ROUTE_ID]
        assert global_route.cooldown_remaining_s == 0
        assert global_route.active_leases == 0

    asyncio.run(scenario())


def test_environment_starts_at_eight_and_supports_twenty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ZETTA_PROVIDER_REQUEST_ADMISSION_ROOT", str(tmp_path))
    admission = ProviderRequestAdmission.from_environment()
    assert admission is not None
    assert admission.initial_limit == 8
    assert admission.max_limit == 20

    monkeypatch.setenv("ZETTA_PROVIDER_MAX_CONCURRENCY", "21")
    with pytest.raises(ValueError, match="must not exceed 20"):
        ProviderRequestAdmission.from_environment()
