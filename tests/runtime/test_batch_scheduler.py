"""``InferenceBatchScheduler`` and the version fence.

Five additional assertions are required, each corresponding to a ``test_*``:

1. Incompatible ``compat_key`` values are never batched together;
2. ``max_wait_ms`` fires the batch once the timer expires;
3. High-priority requests jump the queue;
4. A single tenant cannot fill the whole batch;
5. Versions are never mixed within one batch during a version switch.

The first four are pure unit tests (no ray, no event-loop channel needed); the fifth must
actually run the two loops of ``RuntimeRolloutWorker``, so it spins up a minimal inference
service using ``InProcInferenceChannel``.
"""

from __future__ import annotations

import asyncio
import contextlib
from typing import Any

import pytest

from rollout_runtime.api.enums import Priority
from rollout_runtime.api.ids import new_request_id
from rollout_runtime.api.internal import InferenceRequest, make_routing_token
from rollout_runtime.api.messages import Observation
from rollout_runtime.backends.fake.policy import FakePolicyConfig, FakePolicyCore
from rollout_runtime.transport.inproc import InProcInferenceChannel
from rollout_runtime.workers.batch_scheduler import (
    InferenceBatchScheduler,
    PendingRequest,
    SchedulerConfig,
)
from rollout_runtime.workers.rollout_worker import RuntimeRolloutWorker

ROUTE = make_routing_token("env", 0)
"""Response routing token shared by all test cases."""


def make_pending(
    *,
    compat_key: str = "k1",
    session_id: str = "sess-1",
    application_id: str = "app-1",
    priority: Priority = Priority.INTERACTIVE,
    deadline: float | None = None,
    enqueued_at: float = 0.0,
    step_index: int = 0,
) -> PendingRequest:
    """Build a queued request.

    Args:
        compat_key: Batch compatibility key.
        session_id: Owning session.
        application_id: Tenant.
        priority: Scheduling priority.
        deadline: Absolute timestamp.
        enqueued_at: Enqueue time.
        step_index: Observation step index (used by the fake policy to derive the action).

    Returns:
        A ``PendingRequest``.
    """
    request = InferenceRequest(
        request_id=new_request_id(),
        session_id=session_id,  # type: ignore[arg-type]
        policy_id="fake",
        observation=Observation(
            session_id=session_id,  # type: ignore[arg-type]
            episode_id=1,  # type: ignore[arg-type]
            step_index=step_index,
        ),
        routing_token=ROUTE,
        compat_key=compat_key,
        deadline=deadline,
        priority=priority,
        application_id=application_id,
    )
    return PendingRequest(request=request, enqueued_at=enqueued_at, routing_token=ROUTE)


# --------------------------------------------------------------------- Assertion 1


def test_incompatible_compat_keys_never_share_a_batch() -> None:
    """Requests with different ``compat_key`` values never end up in the same batch."""
    scheduler = InferenceBatchScheduler(
        SchedulerConfig(max_batch_size=4, max_wait_ms=0.0, max_inflight_per_session=8)
    )
    for index in range(3):
        scheduler.enqueue(make_pending(compat_key="cuda-graph", step_index=index))
    for index in range(3):
        scheduler.enqueue(make_pending(compat_key="eager", step_index=index))
    assert scheduler.queue_depth == 6

    first = scheduler.next_batch(now=1.0)
    second = scheduler.next_batch(now=1.0)
    assert first and second
    assert len({item.request.compat_key for item in first}) == 1
    assert len({item.request.compat_key for item in second}) == 1
    assert {first[0].request.compat_key, second[0].request.compat_key} == {
        "cuda-graph",
        "eager",
    }
    assert scheduler.next_batch(now=1.0) == []
    assert scheduler.inflight_total == 6


def test_batch_fires_when_max_batch_size_is_reached() -> None:
    """The batch fires immediately once ``max_batch_size`` is reached, without waiting for ``max_wait_ms``."""
    scheduler = InferenceBatchScheduler(
        SchedulerConfig(
            max_batch_size=3, max_wait_ms=1000.0, max_inflight_per_session=8
        )
    )
    for index in range(2):
        scheduler.enqueue(make_pending(enqueued_at=0.0, step_index=index))
    # Not yet full and not yet due: don't fire.
    assert scheduler.next_batch(now=0.0) == []
    assert scheduler.time_until_ready(now=0.0) == pytest.approx(1.0)

    scheduler.enqueue(make_pending(enqueued_at=0.0, step_index=2))
    assert scheduler.time_until_ready(now=0.0) == 0.0
    batch = scheduler.next_batch(now=0.0)
    assert len(batch) == 3
    # The portion beyond max_batch_size stays in the bucket.
    scheduler.enqueue(make_pending(enqueued_at=0.0, step_index=3))
    assert scheduler.queue_depth == 1


# --------------------------------------------------------------------- Assertion 2


def test_max_wait_ms_fires_a_partial_batch_on_time() -> None:
    """A partial batch still fires: once the earliest request has waited ``max_wait_ms``, it triggers."""
    scheduler = InferenceBatchScheduler(
        SchedulerConfig(max_batch_size=8, max_wait_ms=20.0, max_inflight_per_session=8)
    )
    scheduler.enqueue(make_pending(enqueued_at=100.0))
    assert scheduler.next_batch(now=100.010) == []
    assert scheduler.time_until_ready(now=100.010) == pytest.approx(0.010, abs=1e-6)
    batch = scheduler.next_batch(now=100.021)
    assert len(batch) == 1
    assert scheduler.queue_depth == 0


# --------------------------------------------------------------------- Assertion 3


def test_priority_and_deadline_decide_who_goes_first() -> None:
    """High priority jumps the queue; within the same priority, the earliest deadline goes first."""
    scheduler = InferenceBatchScheduler(
        SchedulerConfig(max_batch_size=2, max_wait_ms=0.0, max_inflight_per_session=8)
    )
    scheduler.enqueue(
        make_pending(session_id="s-bg", priority=Priority.BACKGROUND, enqueued_at=0.0)
    )
    scheduler.enqueue(
        make_pending(session_id="s-batch", priority=Priority.BATCH, enqueued_at=0.0)
    )
    scheduler.enqueue(
        make_pending(
            session_id="s-late",
            priority=Priority.INTERACTIVE,
            deadline=500.0,
            enqueued_at=0.0,
        )
    )
    scheduler.enqueue(
        make_pending(
            session_id="s-urgent",
            priority=Priority.INTERACTIVE,
            deadline=100.0,
            enqueued_at=0.0,
        )
    )

    first = scheduler.next_batch(now=1.0)
    assert [str(item.request.session_id) for item in first] == ["s-urgent", "s-late"]
    second = scheduler.next_batch(now=1.0)
    assert [str(item.request.session_id) for item in second] == ["s-batch", "s-bg"]


def test_high_priority_arrival_jumps_the_queue_across_buckets() -> None:
    """Priority also decides across buckets: a newly arrived INTERACTIVE bucket goes before an already-queued BATCH bucket."""
    scheduler = InferenceBatchScheduler(
        SchedulerConfig(max_batch_size=4, max_wait_ms=0.0, max_inflight_per_session=8)
    )
    scheduler.enqueue(
        make_pending(compat_key="slow", priority=Priority.BATCH, enqueued_at=0.0)
    )
    scheduler.enqueue(
        make_pending(compat_key="fast", priority=Priority.INTERACTIVE, enqueued_at=0.5)
    )
    batch = scheduler.next_batch(now=1.0)
    assert [item.request.compat_key for item in batch] == ["fast"]


# --------------------------------------------------------------------- Assertion 4


def test_single_tenant_cannot_fill_the_batch() -> None:
    """The per-tenant in-flight limit prevents one app from filling the batch; the request resumes once quota is returned."""
    scheduler = InferenceBatchScheduler(
        SchedulerConfig(
            max_batch_size=8,
            max_wait_ms=0.0,
            max_inflight_per_application=2,
            max_inflight_per_session=8,
        )
    )
    for index in range(4):
        scheduler.enqueue(
            make_pending(application_id="greedy", session_id="g", step_index=index)
        )
    scheduler.enqueue(make_pending(application_id="quiet", session_id="q"))

    batch = scheduler.next_batch(now=1.0)
    tenants = [item.request.application_id for item in batch]
    assert tenants.count("greedy") == 2
    assert tenants.count("quiet") == 1
    assert scheduler.queue_depth == 2

    # greedy's quota is exhausted: the remaining two must wait for the dispatched ones to complete.
    assert scheduler.next_batch(now=1.0) == []
    for item in batch:
        if item.request.application_id == "greedy":
            scheduler.complete(item)
    resumed = scheduler.next_batch(now=1.0)
    assert [item.request.application_id for item in resumed] == ["greedy", "greedy"]


def test_per_session_inflight_limit_is_enforced() -> None:
    """The per-session in-flight limit is also enforced (prevents one session from monopolizing a rank)."""
    scheduler = InferenceBatchScheduler(
        SchedulerConfig(
            max_batch_size=8,
            max_wait_ms=0.0,
            max_inflight_per_application=64,
            max_inflight_per_session=1,
        )
    )
    for index in range(3):
        scheduler.enqueue(make_pending(session_id="one", step_index=index))
    assert len(scheduler.next_batch(now=1.0)) == 1
    assert scheduler.next_batch(now=1.0) == []
    assert scheduler.queue_depth == 2


def test_queue_depth_gates_backpressure() -> None:
    """Once the level reaches ``max_queue_depth``, ``accepts_more`` turns false (backpressure upstream)."""
    scheduler = InferenceBatchScheduler(
        SchedulerConfig(max_batch_size=1, max_wait_ms=0.0, max_queue_depth=2)
    )
    assert scheduler.accepts_more() is True
    scheduler.enqueue(make_pending(step_index=0))
    scheduler.enqueue(make_pending(step_index=1))
    assert scheduler.accepts_more() is False


def test_complete_releases_quota_and_counts() -> None:
    """``complete`` returns both quota levels and increments the counters."""
    scheduler = InferenceBatchScheduler(
        SchedulerConfig(max_batch_size=1, max_wait_ms=0.0)
    )
    scheduler.enqueue(make_pending(application_id="app", session_id="s"))
    (item,) = scheduler.next_batch(now=1.0)
    assert scheduler.inflight_per_application == {"app": 1}
    assert scheduler.inflight_per_session == {"s": 1}
    scheduler.complete(item)
    assert scheduler.inflight_per_application == {}
    assert scheduler.inflight_per_session == {}
    assert scheduler.inflight_total == 0
    assert scheduler.completed_count == 1


# --------------------------------------------------------------------- Assertion 5


def test_version_fence_blocks_new_batches_until_inflight_drains() -> None:
    """No new batch is dispatched during the fence; it is released only after in-flight requests drain (at the ``next_batch`` level)."""
    scheduler = InferenceBatchScheduler(
        SchedulerConfig(max_batch_size=1, max_wait_ms=0.0, max_inflight_per_session=8)
    )
    scheduler.enqueue(make_pending(step_index=0))
    scheduler.enqueue(make_pending(step_index=1))
    (running,) = scheduler.next_batch(now=1.0)

    scheduler.begin_version_fence("v2")
    assert scheduler.fenced is True
    assert scheduler.next_batch(now=1.0) == []
    assert scheduler.time_until_ready(now=1.0) is None

    scheduler.complete(running)
    # Once in-flight requests drain, the fence no longer blocks (in the real flow, weights have already been swapped by this point).
    assert len(scheduler.next_batch(now=1.0)) == 1


async def _serve(worker: RuntimeRolloutWorker, channel: Any) -> asyncio.Task[None]:
    """Start up the worker's two loops.

    Args:
        worker: The inference worker.
        channel: The request-plane channel.

    Returns:
        The ``serve`` task.
    """
    worker.init_worker({"inference": channel})
    return asyncio.get_running_loop().create_task(worker.serve(channel, channel))


async def test_update_weights_never_mixes_versions_in_one_batch() -> None:
    """``update_weights`` switches at a batch boundary: ``model_version`` is unique within each batch.

    Approach: have the fake policy record the version seen by each batch, switch versions
    while sending requests, and finally assert that
    (1) no batch mixes versions; (2) both old and new versions were actually used;
    (3) the old version never reappears after the switch point.
    """
    channel = InProcInferenceChannel(request_queue_size=64, response_queue_size=64)
    policy = FakePolicyCore(FakePolicyConfig(delay_seconds=0.002))
    worker = RuntimeRolloutWorker(
        policy=policy,
        scheduler_config=SchedulerConfig(
            max_batch_size=4, max_wait_ms=2.0, max_inflight_per_session=8
        ),
        max_concurrent_inferences=8,
    )
    channel.register_route(ROUTE)
    batch_versions: list[str] = []
    original = policy.infer_batch

    def recording(requests: list[InferenceRequest]) -> Any:
        responses = original(requests)
        versions = {response.model_version for response in responses}
        assert len(versions) == 1, f"batch mixed versions: {versions}"
        batch_versions.append(next(iter(versions)))
        return responses

    policy.infer_batch = recording  # type: ignore[method-assign]
    task = await _serve(worker, channel)
    try:
        for index in range(6):
            await channel.put_request_nowait(make_pending(step_index=index).request)
        await asyncio.sleep(0.02)
        assert await worker.update_weights("fake-v2") is True
        for index in range(6, 12):
            await channel.put_request_nowait(make_pending(step_index=index).request)
        for _ in range(200):
            if worker.responded_count >= 12:
                break
            await asyncio.sleep(0.005)
        assert worker.responded_count == 12
    finally:
        await worker.stop()
        channel.close()
        task.cancel()
        with contextlib.suppress(BaseException):
            await task

    assert batch_versions, "no batch was executed"
    assert "fake-v1" in batch_versions
    assert "fake-v2" in batch_versions
    switch = batch_versions.index("fake-v2")
    assert set(batch_versions[:switch]) == {"fake-v1"}
    assert set(batch_versions[switch:]) == {"fake-v2"}
    assert worker.weight_update_count == 1
    assert worker.scheduler.fenced is False


async def test_execute_batches_really_coalesces() -> None:
    """Concurrent requests with the same ``compat_key`` get coalesced into one batch (``batched > batch_count``)."""
    channel = InProcInferenceChannel(request_queue_size=64, response_queue_size=64)
    policy = FakePolicyCore(FakePolicyConfig())
    worker = RuntimeRolloutWorker(
        policy=policy,
        scheduler_config=SchedulerConfig(
            max_batch_size=4, max_wait_ms=10.0, max_inflight_per_session=8
        ),
        max_concurrent_inferences=8,
    )
    channel.register_route(ROUTE)
    task = await _serve(worker, channel)
    try:
        for index in range(8):
            await channel.put_request_nowait(make_pending(step_index=index).request)
        for _ in range(200):
            if worker.responded_count >= 8:
                break
            await asyncio.sleep(0.005)
        assert worker.responded_count == 8
        assert worker.scheduler.batched_count == 8
        assert worker.scheduler.batch_count <= 4, "requests were not coalesced at all"
        assert policy.batch_calls == worker.scheduler.batch_count
    finally:
        await worker.stop()
        channel.close()
        task.cancel()
        with contextlib.suppress(BaseException):
            await task
