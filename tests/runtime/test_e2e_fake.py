"""End-to-end gate covering 8 assertions (``tests/runtime/test_e2e_fake.py``).

Each assertion has its own ``test_assertion_N_*`` case, numbered to match the
corresponding assertion:

1. Full green lifecycle; 2. 16 concurrent sessions keep ordering; 3. out-of-order
responses stay correlated; 4. late responses/commands are rejected and counted;
5. the same ``request_id`` produces only one env step; 6. ``side_effect_applied``
across the three cancellation states; 7. backpressure yields ``QUEUE_FULL`` instead
of unbounded queuing; 8. the process does not exit after error isolation.

This file is also the **first real exercise** of previously untested code paths:
``RuntimeGateway``'s ``create_sessions`` / ``_dispatch_batch`` / ``_dispatch_one`` /
``_execute`` / ``_finish`` / ``_close_session`` / ``cancel_request`` /
``recover_expired_sessions``, ``CommandDispatcher``, and ``InProcTransport`` all run
through their real code paths here.

See ``tests/conftest.py`` for the ``--transport`` parametrization: the same batch is
re-run with ``--transport=ray_channel``.
"""

from __future__ import annotations

import asyncio
import contextlib
from typing import Any

import numpy as np
import pytest

from rollout_runtime.api.enums import (
    EnvOperation,
    ErrorCode,
    OperationState,
    SessionState,
)
from rollout_runtime.api.ids import BindingToken, EpisodeId, RequestId, new_request_id
from rollout_runtime.api.internal import ActionResponse, CommandEnvelope
from rollout_runtime.api.messages import (
    EpisodeRequest,
    PolicyRequest,
    ResetSpec,
)
from rollout_runtime.api.result import Err, Ok, unwrap
from rollout_runtime.core import payload as payload_module
from rollout_runtime.launch.local import LocalRuntime, build_local_components
from tests.runtime.conftest import local_runtime_config, open_sessions

POLICY = PolicyRequest(policy_id="fake")
"""Inference parameters shared by all test cases."""


async def _stats(runtime: LocalRuntime, session_id: Any) -> dict[str, Any]:
    """Read env-side counters through the Runtime API (``fake.stats`` extension).

    Args:
        runtime: In-process runtime.
        session_id: Target session.

    Returns:
        Structured result from ``fake.stats``.
    """
    return unwrap(
        (await runtime.gateway.extension_call([session_id], "fake", "stats", {}))[0]
    )


async def _wait_for(predicate: Any, timeout: float = 5.0) -> bool:
    """Poll until a condition holds (no fixed sleep, to avoid slow/flaky tests).

    Args:
        predicate: Zero-arg callable; returning true ends the wait.
        timeout: Upper bound in seconds.

    Returns:
        Whether the condition became true before the timeout.
    """
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        if predicate():
            return True
        await asyncio.sleep(0.001)
    return False


# --------------------------------------------------------------------- Assertion 1


async def test_assertion_1_full_lifecycle(
    local_runtime: LocalRuntime, fake_env_spec: Any
) -> None:
    """Assertion 1: ``create → reset → observe → policy_step → action_step → run_episode → close``.

    Args:
        local_runtime: In-process runtime.
        fake_env_spec: env spec factory.
    """
    gateway = local_runtime.gateway
    spec = fake_env_spec(episode_length=12)
    (session_id,) = await open_sessions(local_runtime, spec)

    status = await gateway.get_session(session_id)
    assert status.state is SessionState.READY
    # READY only means the environment resource is bound, not that reset has run.
    assert status.episode_id is None
    not_ready = (await gateway.observe([session_id]))[0]
    assert isinstance(not_ready, Err)
    assert not_ready.error.code is ErrorCode.SESSION_NOT_READY

    reset_result = unwrap((await gateway.reset([session_id], ResetSpec(seed=7)))[0])
    assert reset_result.episode_id == 1
    assert reset_result.observation is not None
    assert reset_result.observation.session_id == session_id
    assert reset_result.observation.step_index == 0
    assert reset_result.side_effect_applied is True
    image = payload_module.decode_payload(reset_result.observation.main_image)
    assert image.shape == (16, 16, 3)
    assert image.dtype == np.uint8
    assert len(reset_result.observation.state) == 8

    observation = unwrap((await gateway.observe([session_id]))[0])
    assert observation.step_index == 0
    assert observation.state == reset_result.observation.state

    step = unwrap((await gateway.policy_step([session_id], POLICY))[0])
    assert step.executed_horizon == 4
    assert step.observation is not None
    assert step.observation.step_index == 4
    assert step.side_effect_applied is True
    assert step.info["model_version"] == "fake-v1"
    assert step.per_step is not None
    assert [record.step_index for record in step.per_step] == [1, 2, 3, 4]

    actions = payload_module.encode_array(np.full((2, 7), 0.25, dtype=np.float32))
    stepped = unwrap((await gateway.action_step([session_id], [actions]))[0])
    assert stepped.executed_horizon == 2
    assert stepped.observation is not None
    assert stepped.observation.step_index == 6
    # The action actually reached the environment: fake env records the sum of the last-step action.
    assert stepped.observation.extras["last_action_checksum"] == pytest.approx(0.25 * 7)

    episode = unwrap(
        (
            await gateway.run_episode(
                [session_id],
                EpisodeRequest(max_steps=10, policy=POLICY, sink_id="mem:e2e"),
            )
        )[0]
    )
    assert episode.stop_reason == "terminated"
    assert episode.terminated is True
    assert episode.executed_horizon == 6  # 12 - 6 steps have already run
    assert episode.num_policy_steps == 2
    assert episode.total_reward == pytest.approx(1.0)
    sink = local_runtime.env_workers[0].sinks.memory("mem:e2e")
    assert len(sink) == episode.num_policy_steps

    # No further steps are allowed after termination; reset must happen first.
    terminated = (await gateway.policy_step([session_id], POLICY))[0]
    assert isinstance(terminated, Err)
    assert terminated.error.code is ErrorCode.EPISODE_TERMINATED
    second = unwrap((await gateway.reset([session_id], ResetSpec(seed=7)))[0])
    assert second.episode_id == 2

    assert isinstance((await gateway.close_sessions([session_id]))[0], Ok)
    assert (await gateway.get_session(session_id)).state is SessionState.CLOSED
    assert local_runtime.env_workers[0].sessions == {}
    gone = (await gateway.policy_step([session_id], POLICY))[0]
    assert isinstance(gone, Err)
    assert gone.error.code is ErrorCode.SESSION_NOT_READY


async def test_lifecycle_operation_seq_and_status(
    local_runtime: LocalRuntime, fake_env_spec: Any
) -> None:
    """Mutating operations get a monotonically increasing per-session ``operation_seq``; read-only operations do not.

    Args:
        local_runtime: In-process runtime.
        fake_env_spec: env spec factory.
    """
    gateway = local_runtime.gateway
    (session_id,) = await open_sessions(local_runtime, fake_env_spec())
    reset_id = new_request_id()
    reset_result = unwrap(
        (await gateway.reset([session_id], ResetSpec(seed=1), request_ids=[reset_id]))[
            0
        ]
    )
    assert reset_result.operation_seq == 1
    step = unwrap((await gateway.policy_step([session_id], POLICY))[0])
    assert step.operation_seq == 2
    observation = unwrap((await gateway.observe([session_id]))[0])
    assert observation is not None
    status = await gateway.get_session(session_id)
    assert status.next_operation_seq == 3
    assert status.active_operation is None
    assert status.worker_summary is not None
    assert status.worker_summary.worker_rank == 0
    assert status.worker_summary.step_index == 4

    operation = await gateway.get_request_status(reset_id)
    assert operation.state is OperationState.SUCCEEDED
    assert operation.operation is EnvOperation.RESET
    assert operation.side_effect_applied is True


# --------------------------------------------------------------------- Assertion 2


async def test_assertion_2_sixteen_sessions_keep_order_and_isolation(
    local_runtime: LocalRuntime, fake_env_spec: Any
) -> None:
    """Assertion 2: 16 sessions interleave operations; results are gathered in input order, and state stays isolated.

    Args:
        local_runtime: In-process runtime.
        fake_env_spec: env spec factory.
    """
    gateway = local_runtime.gateway
    # The pool does not grow, so 16 concurrent sessions require enough slots in the spec upfront.
    spec = fake_env_spec(pool_size=16, episode_length=64)
    session_ids = await open_sessions(local_runtime, spec, count=16)
    assert len(set(session_ids)) == 16

    resets = await gateway.reset(session_ids, ResetSpec(seed=5))
    assert all(isinstance(result, Ok) for result in resets)
    states = [unwrap(result).observation.state for result in resets]
    # Each slot has its own seed offset, so all 16 observations must be distinct (no cross-contamination).
    assert len({tuple(state) for state in states}) == 16

    permutations = [
        session_ids[::2] + session_ids[1::2],
        session_ids[1::2] + session_ids[::2],
        list(reversed(session_ids)),
    ]
    for order in permutations:
        # Rotate through interleaving orders to verify ordering isn't "just coincidentally in order".
        assert sorted(order) == sorted(session_ids)
        results = await gateway.policy_step(order, POLICY)
        assert [unwrap(result).session_id for result in results] == order
        assert all(unwrap(result).executed_horizon == 4 for result in results)

    # Mixing policy_step and action_step concurrently; per-session counters must stay consistent.
    actions = payload_module.encode_array(np.zeros((1, 7), dtype=np.float32))
    mixed = await asyncio.gather(
        gateway.policy_step(session_ids[:8], POLICY),
        gateway.action_step(session_ids[8:], [actions] * 8),
    )
    assert [unwrap(result).session_id for result in mixed[0]] == session_ids[:8]
    assert [unwrap(result).session_id for result in mixed[1]] == session_ids[8:]

    slots: list[int] = []
    for index, session_id in enumerate(session_ids):
        stats = await _stats(local_runtime, session_id)
        slots.append(stats["slot_index"])
        assert stats["resets"] == 1
        expected = 4 * 3 + (4 if index < 8 else 1)
        assert stats["env_steps"] == expected, session_id
        assert stats["step_index"] == expected
    # Slots and sessions are one-to-one (no shared slots), but slot index is **not required** to
    # equal creation order: binding is accepted concurrently (under ray_channel, each
    # CREATE_BINDING starts its own task, and create_binding awaits the pool-building
    # to_thread first), so which one gets a slot first depends on scheduling order.
    assert sorted(slots) == list(range(16))


# --------------------------------------------------------------------- Assertion 3


async def test_assertion_3_out_of_order_responses_keep_correlation(
    fake_env_spec: Any, transport_kind: str
) -> None:
    """Assertion 3: when inference responses return out of order, the ``request_id``-to-session correlation stays intact.

    The out-of-order sequence is **explicitly gated**, not dependent on wall-clock timing: each
    session gets an ``asyncio.Event``, and the test releases them one at a time in reverse
    submission order, so completion order is always reversed regardless of load (an earlier
    version used a 5 ms staggered delay, which could flake under coverage instrumentation).
    Starting from this file, ``max_batch_size`` is explicitly set to 1 so that "one request, one
    inference call" holds, giving the gate per-request granularity; batching semantics are
    covered separately by ``test_batch_scheduler.py``.

    This assertion verifies the ``request_id`` ↔ session ↔ env slot correlation, which is
    orthogonal to batching.

    Args:
        fake_env_spec: env spec factory.
        transport_kind: transport under test.
    """
    config = local_runtime_config(
        transport_kind, rollout_worker={"max_concurrent_inferences": 8}
    )
    config.rollout_worker.scheduler.max_batch_size = 1
    runtime = build_local_components(config)
    await runtime.start()
    try:
        gateway = runtime.gateway
        spec = fake_env_spec(pool_size=8, episode_length=64)
        session_ids = await open_sessions(runtime, spec, count=8)
        await gateway.reset(session_ids, ResetSpec(seed=2))

        gates = {str(session_id): asyncio.Event() for session_id in session_ids}
        worker = runtime.rollout_workers[0]
        original = worker.infer_batch
        checksums: dict[str, float] = {}
        completion: list[str] = []

        async def recording(requests: list[Any]) -> list[Any]:
            # max_batch_size=1, so each call receives exactly one request: wait for its own gate.
            for request in requests:
                await gates[str(request.session_id)].wait()
            responses = await original(requests)
            for request, response in zip(requests, responses, strict=True):
                block = payload_module.decode_payload(response.actions)
                checksums[str(request.session_id)] = float(block[-1].sum())
                completion.append(str(request.session_id))
            return responses

        worker.infer_batch = recording  # type: ignore[method-assign]

        pending = asyncio.create_task(gateway.policy_step(session_ids, POLICY))
        # Wait for all 8 requests to be waiting on the gate, then release in reverse order:
        # this makes completion order deterministically reversed.
        assert await _wait_for(lambda: worker.scheduler.inflight_total == 8)
        for index, session_id in enumerate(reversed(session_ids)):
            gates[str(session_id)].set()
            assert await _wait_for(lambda index=index: len(completion) == index + 1)

        results = await pending
        assert [unwrap(result).session_id for result in results] == session_ids
        assert completion == [str(sid) for sid in reversed(session_ids)]

        for session_id in session_ids:
            stats = await _stats(runtime, session_id)
            assert stats["last_action_checksum"] == pytest.approx(
                checksums[str(session_id)], abs=1e-5
            ), f"action of {session_id} landed in the wrong env slot"
    finally:
        await runtime.gateway.stop()
        await runtime.aclose()


# --------------------------------------------------------------------- Assertion 4


async def test_assertion_4_late_responses_are_rejected_and_counted(
    local_runtime: LocalRuntime, fake_env_spec: Any
) -> None:
    """Assertion 4: responses with a stale ``episode_id`` / stale ``binding_token`` are dropped and counted.

    Args:
        local_runtime: In-process runtime.
        fake_env_spec: env spec factory.
    """
    gateway = local_runtime.gateway
    worker = local_runtime.env_workers[0]
    policy = local_runtime.policies[0]
    (session_id,) = await open_sessions(local_runtime, fake_env_spec(episode_length=64))
    await gateway.reset([session_id], ResetSpec(seed=4))
    slot = worker.sessions[session_id]

    # 1) A completely unrecognized request_id: dropped and counted immediately.
    before = worker.inference.late_response_count
    assert (
        worker.inference.deliver(
            ActionResponse(
                request_id=RequestId("req-never-submitted"),
                session_id=session_id,
                binding_token=slot.binding_token,
                episode_id=slot.episode_id,
            )
        )
        is False
    )
    assert worker.inference.late_response_count == before + 1

    # 2) A genuinely pending request: forge responses with a stale episode_id / stale token,
    # which must not wake it up.
    gate = policy.hold()
    policy.entered_inference.clear()
    pending = asyncio.create_task(gateway.policy_step([session_id], POLICY))
    assert await _wait_for(lambda: bool(worker.inference.pending))
    (request_id,) = list(worker.inference.pending)

    stale_episode = ActionResponse(
        request_id=request_id,
        session_id=session_id,
        binding_token=slot.binding_token,
        episode_id=EpisodeId(int(slot.episode_id) - 1),
        actions=payload_module.encode_array(np.zeros((1, 7), dtype=np.float32)),
    )
    assert worker.inference.deliver(stale_episode) is False
    stale_token = ActionResponse(
        request_id=request_id,
        session_id=session_id,
        binding_token=BindingToken("bind-stale"),
        episode_id=slot.episode_id,
        actions=payload_module.encode_array(np.zeros((1, 7), dtype=np.float32)),
    )
    assert worker.inference.deliver(stale_token) is False
    assert worker.inference.late_response_count == before + 3
    assert not pending.done(), "stale responses must not complete the operation"

    gate.set()
    policy.release()
    result = unwrap((await pending)[0])
    assert result.executed_horizon == 4
    # The real response arrived, and the late-response count didn't increase further.
    assert worker.inference.late_response_count == before + 3


async def test_late_commands_are_rejected_and_counted(
    local_runtime: LocalRuntime, fake_env_spec: Any
) -> None:
    """Late **commands** (stale episode_id / stale token / stale operation_seq) are also rejected and counted.

    Args:
        local_runtime: In-process runtime.
        fake_env_spec: env spec factory.
    """
    gateway = local_runtime.gateway
    worker = local_runtime.env_workers[0]
    (session_id,) = await open_sessions(local_runtime, fake_env_spec(episode_length=64))
    await gateway.reset([session_id], ResetSpec(seed=1))
    await gateway.policy_step([session_id], POLICY)
    slot = worker.sessions[session_id]
    before = worker.stale_command_count

    stale_token = await local_runtime.transport.send_command(
        0,
        CommandEnvelope(
            request_id=new_request_id(),
            session_id=session_id,
            binding_token=BindingToken("bind-obsolete"),
            episode_id=slot.episode_id,
            operation=EnvOperation.OBSERVE,
        ),
    )
    assert stale_token.error is not None
    assert stale_token.error.code is ErrorCode.STALE_BINDING

    await gateway.reset([session_id], ResetSpec(seed=1))
    stale_episode = await local_runtime.transport.send_command(
        0,
        CommandEnvelope(
            request_id=new_request_id(),
            session_id=session_id,
            binding_token=slot.binding_token,
            episode_id=EpisodeId(1),
            operation=EnvOperation.OBSERVE,
        ),
    )
    assert stale_episode.error is not None
    assert stale_episode.error.code is ErrorCode.STALE_BINDING

    stale_seq = await local_runtime.transport.send_command(
        0,
        CommandEnvelope(
            request_id=new_request_id(),
            session_id=session_id,
            binding_token=slot.binding_token,
            episode_id=slot.episode_id,
            operation_seq=slot.last_operation_seq,
            operation=EnvOperation.ACTION_STEP,
            payload={
                "actions": payload_module.encode_array(
                    np.zeros((1, 7), dtype=np.float32)
                )
            },
        ),
    )
    assert stale_seq.error is not None
    assert stale_seq.error.code is ErrorCode.STALE_BINDING
    assert worker.stale_command_count == before + 3

    # The late command never touched the environment.
    stats = await _stats(local_runtime, session_id)
    assert stats["chunk_calls"] == 1


# --------------------------------------------------------------------- Assertion 5


async def test_assertion_5_same_request_id_steps_env_once(
    local_runtime: LocalRuntime, fake_env_spec: Any
) -> None:
    """Assertion 5: repeating ``policy_step`` with the same ``request_id`` produces only one env step.

    Args:
        local_runtime: In-process runtime.
        fake_env_spec: env spec factory.
    """
    gateway = local_runtime.gateway
    (session_id,) = await open_sessions(local_runtime, fake_env_spec(episode_length=64))
    await gateway.reset([session_id], ResetSpec(seed=9))
    baseline = await _stats(local_runtime, session_id)

    request_id = new_request_id()
    first = unwrap(
        (await gateway.policy_step([session_id], POLICY, request_ids=[request_id]))[0]
    )
    after_first = await _stats(local_runtime, session_id)
    assert after_first["env_steps"] == baseline["env_steps"] + 4
    assert after_first["chunk_calls"] == baseline["chunk_calls"] + 1

    second = unwrap(
        (await gateway.policy_step([session_id], POLICY, request_ids=[request_id]))[0]
    )
    after_second = await _stats(local_runtime, session_id)
    assert after_second["env_steps"] == after_first["env_steps"]
    assert after_second["chunk_calls"] == after_first["chunk_calls"]
    assert second.operation_seq == first.operation_seq
    assert second.observation.step_index == first.observation.step_index

    # Same request_id with a different request → IDEMPOTENCY_CONFLICT, and the env still isn't touched.
    conflict = (
        await gateway.reset([session_id], ResetSpec(seed=9), request_ids=[request_id])
    )[0]
    assert isinstance(conflict, Err)
    assert conflict.error.code is ErrorCode.IDEMPOTENCY_CONFLICT
    assert (await _stats(local_runtime, session_id))["resets"] == 1


async def test_client_session_key_is_idempotent(
    local_runtime: LocalRuntime, fake_env_spec: Any
) -> None:
    """The same ``client_session_key`` reuses the same session without consuming an extra env slot.

    Args:
        local_runtime: In-process runtime.
        fake_env_spec: env spec factory.
    """
    spec = fake_env_spec()
    (first,) = await open_sessions(local_runtime, spec, key_prefix="dup")
    (second,) = await open_sessions(local_runtime, spec, key_prefix="dup")
    assert first == second
    assert len(local_runtime.env_workers[0].sessions) == 1
    pool = local_runtime.env_workers[0].pools.find(spec.digest())
    assert pool is not None
    assert pool.in_use == 1


async def test_recover_expired_sessions_closes_leases(
    fake_env_spec: Any, transport_kind: str
) -> None:
    """A session with an expired lease is reclaimed by the Gateway's maintenance sweep and its env slot released.

    Args:
        fake_env_spec: env spec factory.
        transport_kind: transport under test.
    """
    runtime = build_local_components(local_runtime_config(transport_kind))
    await runtime.start()
    try:
        gateway = runtime.gateway
        spec = fake_env_spec()
        (session_id,) = await open_sessions(runtime, spec)
        record = gateway.sessions.get(session_id)
        record.lease_expiration = 0.0
        closed = await gateway.recover_expired_sessions()
        assert closed == [session_id]
        assert (await gateway.get_session(session_id)).state is SessionState.CLOSED
        assert runtime.env_workers[0].sessions == {}
    finally:
        await runtime.gateway.stop()
        await runtime.aclose()


# --------------------------------------------------------------------- Assertion 6


async def test_assertion_6a_cancel_before_dispatch(
    fake_env_spec: Any, transport_kind: str
) -> None:
    """Assertion 6 (state 1): before dispatch, the Gateway cancels immediately with ``side_effect_applied=False``.

    A slow env step holds the session's mutation lock, so the second command sits in
    "accepted but not yet dispatched".

    Args:
        fake_env_spec: env spec factory.
        transport_kind: transport under test.
    """
    runtime = build_local_components(local_runtime_config(transport_kind))
    await runtime.start()
    try:
        gateway = runtime.gateway
        spec = fake_env_spec(episode_length=64, step_delay_seconds=0.2)
        (session_id,) = await open_sessions(runtime, spec)
        await gateway.reset([session_id], ResetSpec(seed=1))
        baseline = await _stats(runtime, session_id)

        blocking = asyncio.create_task(gateway.policy_step([session_id], POLICY))
        await _wait_for(
            lambda: bool(runtime.env_workers[0].sessions[session_id].active_op)
        )

        queued_id = new_request_id()
        queued = asyncio.create_task(
            gateway.policy_step([session_id], POLICY, request_ids=[queued_id])
        )
        assert await _wait_for(lambda: gateway.operations.find(queued_id) is not None)
        record = gateway.operations.find(queued_id)
        assert record is not None
        assert record.state is OperationState.ACCEPTED

        outcome = await gateway.cancel_request(queued_id)
        assert outcome.state is OperationState.CANCELLED
        assert outcome.side_effect_applied is False
        assert "before dispatch" in outcome.message

        assert isinstance(unwrap((await blocking)[0]).observation, object)
        cancelled = (await queued)[0]
        assert isinstance(cancelled, Err)
        assert cancelled.error.code is ErrorCode.CANCELLED
        assert cancelled.error.side_effect_applied is False

        # The cancelled command never took a single step.
        after = await _stats(runtime, session_id)
        assert after["chunk_calls"] == baseline["chunk_calls"] + 1
        status = await gateway.get_request_status(queued_id)
        assert status.state is OperationState.CANCELLED
        assert status.side_effect_applied is False
    finally:
        await runtime.gateway.stop()
        await runtime.aclose()


async def test_assertion_6b_cancel_while_waiting_for_inference(
    local_runtime: LocalRuntime, fake_env_spec: Any
) -> None:
    """Assertion 6 (state 2): cancelling while waiting on inference stops the wait, drops the late action, and has no side effects.

    Args:
        local_runtime: In-process runtime.
        fake_env_spec: env spec factory.
    """
    gateway = local_runtime.gateway
    worker = local_runtime.env_workers[0]
    policy = local_runtime.policies[0]
    (session_id,) = await open_sessions(local_runtime, fake_env_spec(episode_length=64))
    await gateway.reset([session_id], ResetSpec(seed=1))
    baseline = await _stats(local_runtime, session_id)

    policy.entered_inference.clear()
    gate = policy.hold()
    request_id = new_request_id()
    pending = asyncio.create_task(
        gateway.policy_step([session_id], POLICY, request_ids=[request_id])
    )
    assert await _wait_for(lambda: policy.entered_inference.is_set())
    active = worker.active[request_id]
    assert active.stage == "inference"
    assert active.side_effect_applied is False

    outcome = await gateway.cancel_request(request_id)
    assert outcome.side_effect_applied is False
    assert "no env step was started" in outcome.message

    cancelled = (await pending)[0]
    assert isinstance(cancelled, Err)
    assert cancelled.error.code is ErrorCode.CANCELLED
    assert cancelled.error.side_effect_applied is False

    # Release the policy: by the time the late action arrives, no one is waiting for it,
    # so it's dropped and counted.
    late_before = worker.inference.late_response_count
    gate.set()
    policy.release()
    assert await _wait_for(
        lambda: worker.inference.late_response_count > late_before
    ), "the late action must be discarded and counted"

    after = await _stats(local_runtime, session_id)
    assert after["env_steps"] == baseline["env_steps"]
    assert after["chunk_calls"] == baseline["chunk_calls"]
    status = await gateway.get_request_status(request_id)
    assert status.state is OperationState.CANCELLED
    assert status.side_effect_applied is False


async def test_assertion_6c_cancel_after_env_step_started(
    fake_env_spec: Any, transport_kind: str
) -> None:
    """Assertion 6 (state 3): once an env step has started, it is not rolled back; wait for it to finish and report ``side_effect_applied`` truthfully.

    Args:
        fake_env_spec: env spec factory.
        transport_kind: transport under test.
    """
    runtime = build_local_components(local_runtime_config(transport_kind))
    await runtime.start()
    try:
        gateway = runtime.gateway
        worker = runtime.env_workers[0]
        spec = fake_env_spec(episode_length=64, step_delay_seconds=0.25)
        (session_id,) = await open_sessions(runtime, spec)
        await gateway.reset([session_id], ResetSpec(seed=1))
        baseline = await _stats(runtime, session_id)

        request_id = new_request_id()
        pending = asyncio.create_task(
            gateway.policy_step([session_id], POLICY, request_ids=[request_id])
        )
        assert await _wait_for(
            lambda: request_id in worker.active
            and worker.active[request_id].stage == "env_step"
        )

        outcome = await gateway.cancel_request(request_id)
        assert outcome.side_effect_applied is True
        assert outcome.state is OperationState.RUNNING
        assert "not rolled back" in outcome.message

        result = unwrap((await pending)[0])
        assert result.side_effect_applied is True
        assert result.executed_horizon == 4
        after = await _stats(runtime, session_id)
        assert after["env_steps"] == baseline["env_steps"] + 4

        status = await gateway.get_request_status(request_id)
        assert status.state is OperationState.SUCCEEDED
        assert status.side_effect_applied is True
    finally:
        await runtime.gateway.stop()
        await runtime.aclose()


async def test_cancel_fourth_state_outcome_unknown(
    fake_env_spec: Any, transport_kind: str
) -> None:
    """Fourth state: when the worker is unreachable, return ``OUTCOME_UNKNOWN`` without auto-replay.

    Args:
        fake_env_spec: env spec factory.
        transport_kind: transport under test.
    """
    runtime = build_local_components(local_runtime_config(transport_kind))
    await runtime.start()
    try:
        gateway = runtime.gateway
        (session_id,) = await open_sessions(runtime, fake_env_spec(episode_length=64))
        await gateway.reset([session_id], ResetSpec(seed=1))

        from rollout_runtime.api.internal import ResultEnvelope

        worker = runtime.env_workers[0]
        original = worker.handle_command

        async def losing(envelope: CommandEnvelope) -> ResultEnvelope:
            if envelope.operation is EnvOperation.POLICY_STEP:
                return ResultEnvelope(
                    request_id=envelope.request_id,
                    session_id=envelope.session_id,
                    operation=envelope.operation,
                    state=OperationState.OUTCOME_UNKNOWN,
                    side_effect_applied=True,
                )
            return await original(envelope)

        worker.handle_command = losing  # type: ignore[method-assign]
        request_id = new_request_id()
        result = (
            await gateway.policy_step([session_id], POLICY, request_ids=[request_id])
        )[0]
        assert isinstance(result, Err)
        assert result.error.code is ErrorCode.WORKER_LOST
        assert result.error.retryable is False
        status = await gateway.get_request_status(request_id)
        assert status.state is OperationState.OUTCOME_UNKNOWN
    finally:
        await runtime.gateway.stop()
        await runtime.aclose()


# --------------------------------------------------------------------- Assertion 7


async def test_assertion_7_backpressure_rejects_with_queue_full(
    fake_env_spec: Any, transport_kind: str
) -> None:
    """Assertion 7: with ``rr_infer_req`` maxsize=2 and the policy held, the third request gets ``QUEUE_FULL``.

    The queue never exceeds maxsize: backpressure means explicit rejection, not unbounded queuing.

    Args:
        fake_env_spec: env spec factory.
        transport_kind: transport under test.
    """
    config = local_runtime_config(
        transport_kind,
        transport={"infer_request_queue_size": 2},
        rollout_worker={"max_concurrent_inferences": 1},
    )
    runtime = build_local_components(config)
    await runtime.start()
    channel = runtime.channel
    policy = runtime.policies[0]
    gate = policy.hold()
    inflight: list[asyncio.Task[Any]] = []
    try:
        gateway = runtime.gateway
        spec = fake_env_spec(pool_size=4, episode_length=64)
        session_ids = await open_sessions(runtime, spec, count=4)
        await gateway.reset(session_ids, ResetSpec(seed=1))
        assert channel.request_capacity == 2

        # #1 is picked up by the inference loop and held on the gate.
        policy.entered_inference.clear()
        inflight.append(
            asyncio.create_task(gateway.policy_step([session_ids[0]], POLICY))
        )
        assert await _wait_for(lambda: policy.entered_inference.is_set())

        # #2 #3 fill up the bounded queue (the rollout rank's in-flight limit of 1 stops
        # it from picking up any more).
        for session_id in session_ids[1:3]:
            inflight.append(
                asyncio.create_task(gateway.policy_step([session_id], POLICY))
            )
        assert await _wait_for(lambda: channel.request_depth == 2)

        # #4 is rejected immediately, without queuing.
        rejected = (await gateway.policy_step([session_ids[3]], POLICY))[0]
        assert isinstance(rejected, Err)
        assert rejected.error.code is ErrorCode.QUEUE_FULL
        assert rejected.error.retryable is True
        assert rejected.error.side_effect_applied is False
        assert channel.request_depth == 2, "queue must not grow past maxsize"
        assert channel.requests_rejected == 1
        assert runtime.env_workers[0].inference.queue_full_count == 1

        # The rejected command never took a single step.
        assert (await _stats(runtime, session_ids[3]))["chunk_calls"] == 0
    finally:
        gate.set()
        policy.release()
        with contextlib.suppress(BaseException):
            await asyncio.wait_for(asyncio.gather(*inflight), timeout=5.0)
        await runtime.gateway.stop()
        await runtime.aclose()


# --------------------------------------------------------------------- Assertion 8


async def test_assertion_8_env_failure_is_isolated(
    local_runtime: LocalRuntime, fake_env_spec: Any
) -> None:
    """Assertion 8: after one session's env raises, other sessions remain usable and the process doesn't exit.

    Args:
        local_runtime: In-process runtime.
        fake_env_spec: env spec factory.
    """
    gateway = local_runtime.gateway
    # Different env_config → different digest → an independent env pool, so the failure
    # naturally only affects this one session.
    broken_spec = fake_env_spec(episode_length=64, fail_on_step=1)
    healthy_spec = fake_env_spec(episode_length=64)
    (broken,) = await open_sessions(local_runtime, broken_spec, key_prefix="broken")
    (healthy,) = await open_sessions(local_runtime, healthy_spec, key_prefix="healthy")
    await gateway.reset([broken, healthy], ResetSpec(seed=1))

    results = await gateway.policy_step([broken, healthy], POLICY)
    assert isinstance(results[0], Err)
    assert results[0].error.code is ErrorCode.ENV_FAILURE
    assert "injected failure" in results[0].error.message
    assert isinstance(results[1], Ok)
    assert unwrap(results[1]).executed_horizon == 4

    # The process is alive, the Gateway is alive, and the session hasn't been declared dead.
    assert (await gateway.get_session(broken)).state is SessionState.READY
    assert (await gateway.get_session(healthy)).state is SessionState.READY

    # The broken session can also continue on its own (fake only injects the failure on chunk_step #1).
    recovered = unwrap((await gateway.policy_step([broken], POLICY))[0])
    assert recovered.executed_horizon == 4
    for _ in range(3):
        assert isinstance((await gateway.policy_step([healthy], POLICY))[0], Ok)
    assert (await _stats(local_runtime, healthy))["chunk_calls"] == 4
    assert isinstance((await gateway.close_sessions([broken, healthy]))[0], Ok)
