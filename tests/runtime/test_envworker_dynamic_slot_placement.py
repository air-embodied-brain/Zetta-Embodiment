"""EnvWorker dynamic slot placement behavior.

These tests cover the 2026-08-12 dynamic slot design: Gateway schedules only by
worker-level load/backoff, while EnvWorker owns warm slot reuse, cold creation,
and structured resource exhaustion.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from rollout_runtime.api.enums import ErrorCode
from rollout_runtime.api.errors import RuntimeApiError, make_error
from rollout_runtime.api.ids import SessionId
from rollout_runtime.api.messages import CreateSessionRequest
from rollout_runtime.api.result import Err, Ok, unwrap
from rollout_runtime.backends.fake.env import register_fake_env_family
from rollout_runtime.launch.local import build_local_components
from rollout_runtime.workers.env_worker import RuntimeEnvWorker
from tests.runtime.conftest import local_runtime_config


@pytest.fixture(autouse=True)
def _register_fake_family() -> None:
    register_fake_env_family()


async def test_envworker_ignores_max_sessions_and_derives_active_sessions(
    fake_env_spec: Any,
) -> None:
    """Worker-local ``max_sessions`` no longer rejects bindings."""
    worker = RuntimeEnvWorker(max_sessions=1, supported_families=("fake",))
    spec = fake_env_spec(pool_size=1, max_dynamic_pool_size=2)

    await worker.create_binding(SessionId("sess-1"), spec, lease_expiration=10.0)
    await worker.create_binding(SessionId("sess-2"), spec, lease_expiration=10.0)

    assert worker.worker_info().active_sessions == 2
    assert len(worker.sessions) == 2
    assert worker.worker_info().max_sessions == 1

    await worker.aclose()


async def test_envpool_reuses_warm_slot_before_cold_creation(fake_env_spec: Any) -> None:
    """Released slots are warm and reused before calling ``core.add_slot``."""
    worker = RuntimeEnvWorker(supported_families=("fake",))
    spec = fake_env_spec(pool_size=1, max_dynamic_pool_size=4)

    await worker.create_binding(SessionId("sess-1"), spec, lease_expiration=10.0)
    pool = worker.pools.find(spec.digest())
    assert pool is not None
    assert pool.pool_size == 1

    await worker.release_binding(SessionId("sess-1"))
    await worker.create_binding(SessionId("sess-2"), spec, lease_expiration=10.0)

    assert pool.pool_size == 1
    assert pool.active_slots == {0}
    assert pool.warm_free_slots == []

    await worker.aclose()


async def test_envpool_creates_cold_slot_when_no_warm_slot_exists(
    fake_env_spec: Any,
) -> None:
    """No warm slot means EnvPool asks the dynamic core for one new slot."""
    worker = RuntimeEnvWorker(supported_families=("fake",))
    spec = fake_env_spec(pool_size=1, max_dynamic_pool_size=4)

    await worker.create_binding(SessionId("sess-1"), spec, lease_expiration=10.0)
    await worker.create_binding(SessionId("sess-2"), spec, lease_expiration=10.0)

    pool = worker.pools.find(spec.digest())
    assert pool is not None
    assert pool.pool_size == 2
    assert pool.active_slots == {0, 1}

    await worker.aclose()


async def test_cold_create_semaphore_allows_eight_concurrent_creates(
    fake_env_spec: Any,
) -> None:
    """Per-pool cold creation is capped at eight concurrent ``add_slot`` calls."""
    worker = RuntimeEnvWorker(supported_families=("fake",))
    spec = fake_env_spec(pool_size=1, max_dynamic_pool_size=12)

    await worker.create_binding(SessionId("seed"), spec, lease_expiration=10.0)
    pool = worker.pools.find(spec.digest())
    assert pool is not None

    in_add_slot = 0
    peak = 0
    original = pool.core.add_slot

    def blocking_add_slot(seed_offset: int) -> int:
        nonlocal in_add_slot, peak
        in_add_slot += 1
        peak = max(peak, in_add_slot)
        try:
            import time

            time.sleep(0.03)
            return original(seed_offset)
        finally:
            in_add_slot -= 1

    pool.core.add_slot = blocking_add_slot

    await asyncio.gather(
        *(
            worker.create_binding(
                SessionId(f"cold-{index}"), spec, lease_expiration=10.0
            )
            for index in range(9)
        )
    )

    assert peak == 8
    assert pool.pool_size == 10

    await worker.aclose()


async def test_concurrent_cold_creates_reserve_unique_seed_offsets(
    fake_env_spec: Any,
) -> None:
    """Concurrent cold creates reserve distinct seed offsets before awaiting."""
    worker = RuntimeEnvWorker(supported_families=("fake",))
    spec = fake_env_spec(pool_size=1, max_dynamic_pool_size=6)

    await worker.create_binding(SessionId("seed"), spec, lease_expiration=10.0)
    pool = worker.pools.find(spec.digest())
    assert pool is not None

    seed_offsets: list[int] = []
    original = pool.core.add_slot

    def blocking_add_slot(seed_offset: int) -> int:
        seed_offsets.append(seed_offset)
        import time

        time.sleep(0.03)
        return original(seed_offset)

    pool.core.add_slot = blocking_add_slot

    await asyncio.gather(
        *(
            worker.create_binding(
                SessionId(f"seed-offset-{index}"), spec, lease_expiration=10.0
            )
            for index in range(5)
        )
    )

    assert sorted(seed_offsets) == [1, 2, 3, 4, 5]

    await worker.aclose()


async def test_failed_cold_create_releases_reserved_capacity(
    fake_env_spec: Any,
) -> None:
    """A failed cold create frees its reservation for a later retry."""
    worker = RuntimeEnvWorker(supported_families=("fake",))
    spec = fake_env_spec(pool_size=1, max_dynamic_pool_size=2)

    await worker.create_binding(SessionId("seed"), spec, lease_expiration=10.0)
    pool = worker.pools.find(spec.digest())
    assert pool is not None

    original = pool.core.add_slot
    failures_left = 1

    def fail_once(seed_offset: int) -> int:
        nonlocal failures_left
        if failures_left:
            failures_left -= 1
            raise MemoryError("simulated gpu oom")
        return original(seed_offset)

    pool.core.add_slot = fail_once

    with pytest.raises(RuntimeApiError) as excinfo:
        await worker.create_binding(SessionId("oom"), spec, lease_expiration=10.0)
    assert excinfo.value.info.code is ErrorCode.RESOURCE_EXHAUSTED

    await worker.create_binding(SessionId("retry"), spec, lease_expiration=10.0)

    assert pool.pool_size == 2
    assert pool.active_slots == {0, 1}

    await worker.aclose()


async def test_gateway_retries_another_worker_after_resource_exhaustion(
    transport_kind: str,
    fake_env_spec: Any,
) -> None:
    """Structured resource exhaustion rolls back reservation and retries another rank."""
    if transport_kind != "inproc":
        pytest.skip("monkeypatches in-process workers directly")
    runtime = build_local_components(
        local_runtime_config(
            transport_kind,
            env_worker={"num_ranks": 2, "max_sessions_per_rank": 1},
            admission={"max_sessions_per_application": 8},
            gateway={"heartbeat_interval_seconds": 0},
        )
    )
    await runtime.start()
    try:
        first_worker = runtime.env_workers[0]
        original = first_worker.create_binding

        async def resource_exhausted_once(*args: Any, **kwargs: Any) -> Any:
            first_worker.create_binding = original  # type: ignore[method-assign]
            raise RuntimeApiError(
                make_error(
                    ErrorCode.RESOURCE_EXHAUSTED,
                    "gpu memory exhausted while creating an env slot",
                    reason="OOM",
                    resource="gpu_memory",
                    worker_rank=0,
                )
            )

        first_worker.create_binding = resource_exhausted_once  # type: ignore[method-assign]
        spec = fake_env_spec(pool_size=1, max_dynamic_pool_size=2)
        result = (
            await runtime.gateway.create_sessions(
                [
                    CreateSessionRequest(
                        application_id="test",
                        client_session_key="retry",
                        env_spec=spec,
                        default_policy_id="fake",
                    )
                ]
            )
        )[0]

        assert isinstance(result, Ok), result
        record = runtime.gateway.sessions.get(result.value.session_id)
        assert record.worker_rank == 1
        assert runtime.gateway.workers.snapshot()[0].active_sessions == 0
        assert runtime.gateway.workers.snapshot()[0].can_create_slot is False
        assert runtime.gateway.workers.snapshot()[1].active_sessions == 1
    finally:
        await runtime.gateway.stop()
        await runtime.aclose()


async def test_gateway_returns_resource_exhausted_after_all_candidates_oom(
    transport_kind: str,
    fake_env_spec: Any,
) -> None:
    """If every candidate reports structured OOM, create_session fails clearly."""
    if transport_kind != "inproc":
        pytest.skip("monkeypatches in-process workers directly")
    runtime = build_local_components(
        local_runtime_config(
            transport_kind,
            env_worker={"num_ranks": 2, "max_sessions_per_rank": 1},
            admission={"max_sessions_per_application": 8},
            gateway={"heartbeat_interval_seconds": 0},
        )
    )
    await runtime.start()
    try:
        for worker in runtime.env_workers:
            async def resource_exhausted(
                *args: Any, _rank: int = worker.worker_rank, **kwargs: Any
            ) -> Any:
                raise RuntimeApiError(
                    make_error(
                        ErrorCode.RESOURCE_EXHAUSTED,
                        "gpu memory exhausted while creating an env slot",
                        reason="OOM",
                        resource="gpu_memory",
                        worker_rank=_rank,
                    )
                )

            worker.create_binding = resource_exhausted  # type: ignore[method-assign]

        spec = fake_env_spec(pool_size=1, max_dynamic_pool_size=2)
        result = (
            await runtime.gateway.create_sessions(
                [
                    CreateSessionRequest(
                        application_id="test",
                        client_session_key="all-oom",
                        env_spec=spec,
                        default_policy_id="fake",
                    )
                ]
            )
        )[0]

        assert isinstance(result, Err)
        assert result.error.code is ErrorCode.RESOURCE_EXHAUSTED
        assert result.error.detail["resource"] == "gpu_memory"
        assert all(
            not entry.can_create_slot
            for entry in runtime.gateway.workers.snapshot().values()
        )
        assert all(
            entry.active_sessions == 0
            for entry in runtime.gateway.workers.snapshot().values()
        )
    finally:
        await runtime.gateway.stop()
        await runtime.aclose()


async def test_gateway_release_restores_can_create_slot(
    transport_kind: str,
    fake_env_spec: Any,
) -> None:
    """Successful release marks the worker assignable again."""
    if transport_kind != "inproc":
        pytest.skip("uses in-process worker state")
    runtime = build_local_components(
        local_runtime_config(
            transport_kind,
            env_worker={"num_ranks": 1, "max_sessions_per_rank": 1},
            admission={"max_sessions_per_application": 8},
            gateway={"heartbeat_interval_seconds": 0},
        )
    )
    await runtime.start()
    try:
        spec = fake_env_spec(pool_size=1, max_dynamic_pool_size=2)
        result = (
            await runtime.gateway.create_sessions(
                [
                    CreateSessionRequest(
                        application_id="test",
                        client_session_key="release",
                        env_spec=spec,
                        default_policy_id="fake",
                    )
                ]
            )
        )[0]
        session_id = unwrap(result).session_id

        runtime.gateway.workers.mark_cannot_create_slot(0)
        closed = (await runtime.gateway.close_sessions([session_id]))[0]

        assert isinstance(closed, Ok), closed
        assert runtime.gateway.workers.snapshot()[0].can_create_slot is True
    finally:
        await runtime.gateway.stop()
        await runtime.aclose()
