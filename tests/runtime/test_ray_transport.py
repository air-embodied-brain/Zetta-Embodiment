"""Ray Channel transport specifics: bounded queues, msgpack frames, timeout
does not finalize state.

Three parts, all requiring a local Ray cluster (``@pytest.mark.ray``):

1. Two hard semantics of ``RayQueue``: explicit ``maxsize`` (rlinf's
   ``maxsize=0`` is **unbounded**, which would silently defeat
   backpressure) and ``put_nowait`` raising ``asyncio.QueueFull``;
2. Channel frames go through ``api.wire``'s msgpack, with
   ``CommandEnvelope`` / ``ResultEnvelope`` round-tripping faithfully;
3. **A timing semantic**: once ``transport.command_timeout_seconds``
   elapses, only ``DEADLINE_EXCEEDED`` is returned, the operation **does
   not reach a terminal state**, and it is finalized later by the result
   flow-back ("an RPC timeout is not the same as cancellation").

Rerunning the 8 end-to-end assertions under the ray transport is not here,
but in ``pytest tests/runtime/test_e2e_fake.py --transport=ray_channel``.
"""

from __future__ import annotations

import asyncio
import contextlib
from typing import Any

import pytest

from rollout_runtime.api.enums import EnvOperation, ErrorCode, OperationState
from rollout_runtime.api.ids import new_request_id
from rollout_runtime.api.internal import CommandEnvelope, ResultEnvelope
from rollout_runtime.api.messages import PolicyRequest, ResetSpec
from rollout_runtime.api.result import Err
from rollout_runtime.launch.local import build_local_components
from tests.runtime.conftest import local_runtime_config, open_sessions

pytestmark = pytest.mark.ray

POLICY = PolicyRequest(policy_id="fake")
"""Shared inference parameters."""


@pytest.fixture(scope="module", autouse=True)
def _ray_cluster() -> Any:
    """Ensure a Ray cluster is already up before this module runs (roughly
    4-5 seconds on first call).

    Returns:
        The initialized Ray module.
    """
    from zetta.runtime.ray.bootstrap import ensure_ray_initialized

    ray = ensure_ray_initialized()
    yield ray
    ray.shutdown()


async def _wait_for(predicate: Any, timeout: float = 10.0) -> bool:
    """Poll while waiting for a condition to hold.

    Args:
        predicate: A no-argument callable.
        timeout: Upper bound in seconds.

    Returns:
        Whether it held before the timeout.
    """
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        if predicate():
            return True
        await asyncio.sleep(0.005)
    return False


async def test_ray_queue_is_bounded_and_raises_queue_full() -> None:
    """A bounded queue: raises ``asyncio.QueueFull`` when full, and the
    watermark never exceeds maxsize."""
    from rollout_runtime.transport.ray_channel import RayQueue

    queue = RayQueue.create("test-bounded", maxsize=2)
    try:
        await queue.put_nowait({"body": b"a"}, key="0")
        await queue.put_nowait({"body": b"b"}, key="0")
        assert queue.qsize("0") == 2
        with pytest.raises(asyncio.QueueFull):
            await queue.put_nowait({"body": b"c"}, key="0")
        assert queue.qsize("0") == 2
        # Queues are separated by key: a different key has its own maxsize.
        await queue.put_nowait({"body": b"d"}, key="1")
        assert queue.qsize("1") == 1
        assert (await queue.get(key="0"))["body"] == b"a"
        await queue.put_nowait({"body": b"e"}, key="0")
        assert queue.qsize("0") == 2
    finally:
        await queue.close()
        queue.shutdown()


def test_ray_queue_rejects_implicit_unbounded() -> None:
    """``maxsize=0`` is unbounded in rlinf, so it must be explicitly
    rejected (a precondition for backpressure)."""
    from rollout_runtime.transport.ray_channel import RayQueue

    with pytest.raises(ValueError, match="maxsize"):
        RayQueue.create("test-unbounded", maxsize=0)


async def test_frames_round_trip_through_msgpack() -> None:
    """Channel frame bodies are msgpack: command and result envelopes
    round-trip faithfully, and byte counts are tallied."""
    from rollout_runtime.api.wire import decode_bytes
    from rollout_runtime.transport.ray_channel import RayQueue, _frame

    queue = RayQueue.create("test-frames", maxsize=4)
    try:
        envelope = CommandEnvelope(
            request_id=new_request_id(),
            session_id="sess-frame",  # type: ignore[arg-type]
            operation=EnvOperation.RESET,
            payload={"reset_spec": ResetSpec(seed=11, task_id=3)},
        )
        await queue.put_nowait(
            _frame("command", envelope, reply_key="results"), key="0"
        )
        frame = await queue.get(key="0")
        assert frame["kind"] == "command"
        assert frame["reply_key"] == "results"
        decoded = decode_bytes(frame["body"], CommandEnvelope)
        assert decoded == envelope
        assert decoded.payload["reset_spec"].seed == 11

        result = ResultEnvelope(
            request_id=envelope.request_id,
            session_id=envelope.session_id,
            operation=envelope.operation,
            state=OperationState.SUCCEEDED,
            value={"ok": True},
            side_effect_applied=True,
        )
        await queue.put_nowait(_frame("result", result), key="results")
        back = decode_bytes((await queue.get(key="results"))["body"], ResultEnvelope)
        assert back == result
        assert queue.bytes_put > 0
        assert queue.bytes_got == queue.bytes_put
    finally:
        await queue.close()
        queue.shutdown()


async def test_command_timeout_does_not_finalize_the_operation(
    fake_env_spec: Any,
) -> None:
    """When ``command_timeout_seconds`` elapses: the caller gets
    ``DEADLINE_EXCEEDED``, but the operation remains RUNNING.

    A timeout is not equivalent to cancellation: the operation remains in
    the registry, finalized later by the worker's result flow-back, so it
    **must not** be judged FAILED, nor replayed.

    Args:
        fake_env_spec: env spec factory.
    """
    config = local_runtime_config(
        "ray_channel", transport={"command_timeout_seconds": 0.25}
    )
    runtime = build_local_components(config)
    await runtime.start()
    try:
        gateway = runtime.gateway
        # The env step is deliberately much longer than the command timeout.
        spec = fake_env_spec(episode_length=64, step_delay_seconds=1.5)
        (session_id,) = await open_sessions(runtime, spec)
        await gateway.reset([session_id], ResetSpec(seed=1))

        request_id = new_request_id()
        result = (
            await gateway.policy_step([session_id], POLICY, request_ids=[request_id])
        )[0]
        assert isinstance(result, Err)
        assert result.error.code is ErrorCode.DEADLINE_EXCEEDED
        assert result.error.side_effect_applied is False
        assert runtime.transport.command_timeout_count == 1

        # Key assertion: it was not judged terminal.
        status = await gateway.get_request_status(request_id)
        assert status.state is OperationState.RUNNING
        record = gateway.operations.find(request_id)
        assert record is not None
        assert record.is_terminal is False

        # Only after the worker finishes and the result flows back is the
        # operation finalized (and honestly carrying the side effect).
        assert await _wait_for(lambda: gateway.late_result_count >= 1)
        assert await _wait_for(
            lambda: gateway.operations.get(request_id).is_terminal, timeout=5.0
        )
        final = await gateway.get_request_status(request_id)
        assert final.state is OperationState.SUCCEEDED
        assert final.side_effect_applied is True
        assert runtime.transport.late_result_count == 1
        assert runtime.transport.orphan_result_count == 0

        # The environment was only touched once: the timeout triggered no replay.
        stats = (await gateway.extension_call([session_id], "fake", "stats", {}))[
            0
        ].value
        assert stats["chunk_calls"] == 1
    finally:
        with contextlib.suppress(BaseException):
            await runtime.gateway.stop()
        await runtime.aclose()


async def test_transport_reports_queue_full_as_backpressure() -> None:
    """When the command queue is full, the transport returns
    ``QUEUE_FULL`` directly (no queueing, no blocking the caller).

    Deliberately does not attach a worker: under the normal path, the
    EnvWorker's command loop drains quickly (each command starts its own
    task), so the queue almost never piles up, and what needs verifying
    here is precisely "what happens once it does pile up."
    """
    from rollout_runtime.transport.ray_channel import ChannelNames, RayChannelTransport

    names = ChannelNames.build(env_group_name="bp", gateway_epoch=99)
    transport = RayChannelTransport.create(
        names=names,
        worker_ranks=[0],
        command_queue_size=1,
        control_queue_size=1,
        result_queue_size=4,
        command_timeout_seconds=0.2,
    )
    try:
        await transport.start()
        envelope = CommandEnvelope(
            request_id=new_request_id(),
            session_id="sess-bp",  # type: ignore[arg-type]
            operation=EnvOperation.OBSERVE,
        )
        # After the first entry goes into the queue with nobody consuming
        # it, the queue is full (maxsize=1).
        first = asyncio.create_task(transport.send_command(0, envelope))
        assert await _wait_for(lambda: transport.command_depth(0) == 1)

        rejected = await transport.send_command(
            0,
            CommandEnvelope(
                request_id=new_request_id(),
                session_id="sess-bp",  # type: ignore[arg-type]
                operation=EnvOperation.OBSERVE,
            ),
        )
        assert rejected.error is not None
        assert rejected.error.code is ErrorCode.QUEUE_FULL
        assert rejected.error.detail["queue_capacity"] == 1
        assert transport.queue_full_count == 1
        assert transport.command_depth(0) == 1, "queue must not grow past maxsize"

        # The first will ultimately only time out (no worker returns a
        # result), but operation semantics are covered by the case above.
        outcome = await first
        assert outcome.error is not None
        assert outcome.error.code is ErrorCode.DEADLINE_EXCEEDED
        assert outcome.state is OperationState.RUNNING
    finally:
        await transport.close()
        transport.shutdown()
