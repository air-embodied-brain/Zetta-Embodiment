"""Real heartbeat and ``LOST`` semantics.

Historical context: ``register_env_worker`` was previously called only once at
launch time, and ``EnvWorkerRegistry.heartbeat_at`` was never updated afterward
because ``refresh_worker_registry`` had no call site. As a result, ``stale_ranks()``
would declare **all** ranks unreachable after ``heartbeat_timeout_seconds`` elapsed,
and every ``READY`` session on them would be flipped to ``LOST`` — i.e. ``LOST`` was
triggered purely by **wall-clock time**, independent of whether the worker was
actually alive. This went unnoticed in earlier runs because presets set the timeout
far longer than a single validation pass (60 s for the a100 preset, 3600 s for the
zetta seam preset).

So the heartbeat mechanism was made real: the Gateway periodically sends
``EnvOperation.HEARTBEAT`` over the **control channel**, and a success refreshes the
registry. This file focuses on three things:

1. A live worker's sessions stay ``READY`` well past ``heartbeat_timeout_seconds``
   (regression: removing the heartbeat loop would fail this);
2. When one rank stops responding to control messages, **only its own** sessions
   flip to ``LOST``, while other ranks keep operating normally;
3. ``LOST`` is not auto-replayed and is not recoverable — it can only be closed and
   recreated.

The heartbeat **deliberately avoids** ``WorkerGroup`` methods: ``WorkerGroupFuncResult``
sends ``os.kill(pid, SIGUSR1)`` to kill the whole job on a remote exception, which
would mean "one dead rank takes down the driver too" if used for liveness probing —
exactly what the GPU-side validation needs to avoid. One test case pins this behavior
down explicitly.
"""

from __future__ import annotations

import asyncio
import contextlib
from typing import Any

import pytest

from rollout_runtime.api.enums import EnvOperation, ErrorCode, SessionState
from rollout_runtime.api.errors import InvalidTransition
from rollout_runtime.api.internal import ControlEnvelope
from rollout_runtime.api.messages import EnvWorkerInfo, PolicyRequest, ResetSpec
from rollout_runtime.api.result import Err
from rollout_runtime.launch.local import build_local_components

from .conftest import local_runtime_config, open_sessions


async def build_two_rank_runtime(
    transport_kind: str,
    *,
    heartbeat_timeout: float = 0.6,
    heartbeat_interval: float = 0.05,
) -> Any:
    """Start an in-process runtime with 2 env ranks.

    Args:
        transport_kind: ``"inproc"`` or ``"ray_channel"``.
        heartbeat_timeout: heartbeat timeout in seconds.
        heartbeat_interval: liveness-probe interval in seconds.

    Returns:
        A started ``LocalRuntime``.
    """
    config = local_runtime_config(
        transport_kind,
        env_worker={"num_ranks": 2, "max_sessions_per_rank": 1},
        gateway={
            "heartbeat_timeout_seconds": heartbeat_timeout,
            "heartbeat_interval_seconds": heartbeat_interval,
            "maintenance_interval_seconds": 0.05,
            "default_lease_seconds": 600.0,
        },
    )
    runtime = build_local_components(config)
    await runtime.start()
    return runtime


async def wait_until(predicate: Any, timeout: float = 5.0) -> bool:
    """Poll until a condition holds.

    Args:
        predicate: zero-arg predicate.
        timeout: upper bound in seconds.

    Returns:
        True if the condition held, False on timeout.
    """
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        if predicate():
            return True
        await asyncio.sleep(0.02)
    return predicate()


async def test_a_live_worker_keeps_its_sessions_ready_past_the_timeout(
    transport_kind,
) -> None:
    """A live worker: sessions remain ``READY`` even after 3x the heartbeat timeout has elapsed.

    This is a regression test for a real defect: ``heartbeat_at`` was never updated,
    so once the timeout passed, every session would be declared ``LOST`` by wall
    clock alone (the ``local_fake`` timeout is 5 s, while the a100 preset is 60 s, so
    a single validation pass happened to dodge this).

    Args:
        transport_kind: transport under test.
    """
    runtime = await build_two_rank_runtime(
        transport_kind, heartbeat_timeout=0.6, heartbeat_interval=0.05
    )
    gateway = runtime.gateway
    try:
        session_ids = await open_sessions(
            runtime, _spec(pool_size=1, episode_length=64), key_prefix="live"
        )
        before = gateway.heartbeat_ok_count

        await asyncio.sleep(2.0)  # > 3× timeout

        status = await gateway.get_session(session_ids[0])
        assert status.state is SessionState.READY, (
            "a live worker must not be reaped by the wall clock; "
            f"heartbeat_ok went {before} -> {gateway.heartbeat_ok_count}"
        )
        assert gateway.heartbeat_ok_count > before, "the heartbeat loop never ran"
        assert gateway.heartbeat_failure_count == 0
        # Still operable after this (not just "the state field is unchanged").
        results = await gateway.reset(session_ids, ResetSpec(seed=1))
        assert not [item for item in results if isinstance(item, Err)]
    finally:
        with contextlib.suppress(BaseException):
            await gateway.stop()
        with contextlib.suppress(BaseException):
            await runtime.aclose()


async def test_probe_env_workers_reports_per_rank_health(
    local_runtime, fake_env_spec
) -> None:
    """``probe_env_workers`` reports per-rank health and does not raise for an unresponsive rank.

    Args:
        local_runtime: in-process runtime fixture.
        fake_env_spec: env spec factory.
    """
    gateway = local_runtime.gateway
    outcomes = await gateway.probe_env_workers()
    assert outcomes == {0: True}

    async def never_answers(_envelope: ControlEnvelope) -> Any:
        await asyncio.sleep(3600)

    local_runtime.env_workers[0].handle_control = never_answers  # type: ignore[assignment]
    gateway._heartbeat_probe_timeout = 0.2  # no need to wait out the full probe window in this test
    outcomes = await gateway.probe_env_workers()
    assert outcomes == {0: False}
    assert gateway.heartbeat_failure_count >= 1


async def test_the_heartbeat_never_uses_worker_group_methods(local_runtime) -> None:
    """The heartbeat only uses the control channel (``WorkerGroup`` remote exceptions send ``SIGUSR1`` and kill the job).

    Args:
        local_runtime: in-process runtime fixture.
    """
    gateway = local_runtime.gateway
    seen: list[tuple[str, EnvOperation]] = []
    transport = gateway.transport
    original_control = transport.send_control
    original_command = transport.send_command

    async def record_control(rank: int, envelope: ControlEnvelope) -> Any:
        seen.append(("control", envelope.operation))
        return await original_control(rank, envelope)

    async def record_command(rank: int, envelope: Any) -> Any:
        seen.append(("command", envelope.operation))
        return await original_command(rank, envelope)

    transport.send_control = record_control  # type: ignore[assignment]
    transport.send_command = record_command  # type: ignore[assignment]
    try:
        await gateway.probe_env_workers()
    finally:
        transport.send_control = original_control  # type: ignore[assignment]
        transport.send_command = original_command  # type: ignore[assignment]

    # The background heartbeat loop may also have sent a round in the meantime (more likely
    # to interleave under the slower ray transport), so the assertion checks the **shape**:
    # everything went through the control channel, everything was HEARTBEAT, and no command
    # was sent at all.
    assert seen, "probe_env_workers sent nothing"
    assert set(seen) == {("control", EnvOperation.HEARTBEAT)}, seen


async def test_the_worker_answers_heartbeats_with_its_worker_info(
    local_runtime,
) -> None:
    """The worker's ``HEARTBEAT`` handler replies with a fresh ``EnvWorkerInfo``.

    Args:
        local_runtime: in-process runtime fixture.
    """
    worker = local_runtime.env_workers[0]
    reply = await worker.handle_control(
        ControlEnvelope(
            request_id="req-heartbeat-probe",
            operation=EnvOperation.HEARTBEAT,
            session_id=None,
        )
    )
    assert reply.ok
    assert isinstance(reply.value, EnvWorkerInfo)
    assert reply.value.worker_rank == worker.worker_rank
    assert reply.value.heartbeat_at > 0.0


async def test_a_silent_rank_does_not_drag_healthy_ranks_down(transport_kind) -> None:
    """A **hung** rank must not cause healthy ranks to also be declared ``LOST`` (a defect found by independent audit).

    The original implementation combined two issues that caused false positives:
    (1) ``_heartbeat_probe_timeout`` was computed as ``max(1.0, min(timeout, 5.0),
    interval)``, with no upper bound relative to ``timeout`` (the ``local_fake`` 5 s
    timeout produced a 5 s probe window); (2) ``_heartbeat_loop`` serially awaited
    ``probe_env_workers()``, which itself ``gather``s all ranks — so the effective
    heartbeat cadence became ``interval + probe_timeout``, and once that exceeded
    ``heartbeat_timeout``, **a healthy rank's ``heartbeat_at`` could expire between
    two refreshes**. The consequence wasn't just "slow" but "wrong": an innocent
    session got declared ``LOST``, and ``LOST`` is not recoverable.

    Args:
        transport_kind: transport under test.
    """
    runtime = await build_two_rank_runtime(
        transport_kind, heartbeat_timeout=0.6, heartbeat_interval=0.05
    )
    gateway = runtime.gateway
    try:
        victim_spec = _spec(pool_size=1, episode_length=32)
        healthy_spec = _spec(pool_size=1, episode_length=33)
        first = await open_sessions(runtime, victim_spec, key_prefix="v")
        second = await open_sessions(runtime, healthy_spec, key_prefix="h")
        await gateway.reset(first + second, ResetSpec(seed=1))
        victim_rank = gateway.sessions.get(first[0]).worker_rank
        healthy_rank = gateway.sessions.get(second[0]).worker_rank
        assert victim_rank != healthy_rank

        async def never_answers(_envelope: ControlEnvelope) -> Any:
            await asyncio.sleep(3600)

        runtime.env_workers[victim_rank].handle_control = never_answers  # type: ignore[assignment]

        # Wait well past 3x the timeout: the unreachable one must flip to LOST, and the
        # healthy one must **always** stay READY.
        assert await wait_until(
            lambda: gateway.sessions.get(first[0]).state is SessionState.LOST,
            timeout=8.0,
        )
        for _ in range(20):  # keep observing for another ~2 s (~3x timeout)
            await asyncio.sleep(0.1)
            state = gateway.sessions.get(second[0]).state
            assert state is SessionState.READY, (
                "a hung rank must not slow the heartbeat cadence enough to reap a "
                f"healthy rank; healthy session became {state.name} "
                f"(probe_timeout={gateway._heartbeat_probe_timeout}, "
                f"interval={gateway._heartbeat_interval}, "
                f"timeout={gateway.workers._timeout})"
            )
        assert gateway.workers.get(healthy_rank).healthy
        # Actually still able to step (not just "the state field checks out").
        results = await gateway.policy_step(second, PolicyRequest(policy_id="fake"))
        assert not [item for item in results if isinstance(item, Err)], results
    finally:
        with contextlib.suppress(BaseException):
            await gateway.stop()
        with contextlib.suppress(BaseException):
            await runtime.aclose()


async def test_the_probe_window_is_bounded_by_the_heartbeat_timeout() -> None:
    """The probe window must be ≤ one third of the timeout, and **not** bounded upward by ``heartbeat_interval``."""
    from rollout_runtime.gateway.gateway import RuntimeGateway

    gateway = RuntimeGateway(
        heartbeat_timeout_seconds=5.0, heartbeat_interval_seconds=1.0
    )
    assert gateway._heartbeat_probe_timeout <= 5.0 / 3.0 + 1e-9
    # The interval-upper-bounded formula (pre-audit behavior) would compute 4.0; it no longer applies.
    greedy = RuntimeGateway(
        heartbeat_timeout_seconds=5.0, heartbeat_interval_seconds=4.0
    )
    assert greedy._heartbeat_probe_timeout <= 5.0 / 3.0 + 1e-9
    # Still capped at 5 s even when the timeout is very large.
    generous = RuntimeGateway(
        heartbeat_timeout_seconds=3600.0, heartbeat_interval_seconds=10.0
    )
    assert generous._heartbeat_probe_timeout == 5.0


def test_load_config_rejects_an_interval_that_is_too_close_to_the_timeout() -> None:
    """Reject configs where the interval is too close to (or ≥) the timeout at load time, since a single probe jitter would otherwise reap a healthy rank."""
    from rollout_runtime.config.schema import load_config

    with pytest.raises(ValueError, match="heartbeat_interval_seconds"):
        load_config(
            {
                "gateway": {
                    "heartbeat_timeout_seconds": 5.0,
                    "heartbeat_interval_seconds": 4.0,
                }
            }
        )
    # A third of the timeout is an allowed boundary.
    ok = load_config(
        {
            "gateway": {
                "heartbeat_timeout_seconds": 30.0,
                "heartbeat_interval_seconds": 10.0,
            }
        }
    )
    assert ok.gateway.heartbeat_interval_seconds == 10.0


async def test_only_the_dead_rank_loses_its_sessions(transport_kind) -> None:
    """When one rank stops answering control messages, **only its own** sessions flip to ``LOST``.

    Also asserts: ``LOST`` is not auto-replayed (no new env step occurs), is not
    recoverable, and other ranks continue serving normally.

    Args:
        transport_kind: transport under test.
    """
    runtime = await build_two_rank_runtime(transport_kind)
    gateway = runtime.gateway
    try:
        # One session per rank: ``max_sessions_per_rank=1`` guarantees the two sessions land on two ranks.
        spec_a = _spec(pool_size=1, episode_length=32)
        spec_b = _spec(pool_size=1, episode_length=33)
        first = await open_sessions(runtime, spec_a, key_prefix="a")
        second = await open_sessions(runtime, spec_b, key_prefix="b")
        await gateway.reset(first + second, ResetSpec(seed=1))
        rank_of_first = gateway.sessions.get(first[0]).worker_rank
        rank_of_second = gateway.sessions.get(second[0]).worker_rank
        assert rank_of_first != rank_of_second, "sessions did not spread across ranks"

        victim = runtime.env_workers[rank_of_first]

        async def never_answers(_envelope: ControlEnvelope) -> Any:
            await asyncio.sleep(3600)

        # Cut off only this one rank's **control**-channel replies: equivalent to its
        # actor dying (heartbeat timeout), without affecting the other rank.
        victim.handle_control = never_answers  # type: ignore[assignment]

        assert await wait_until(
            lambda: gateway.sessions.get(first[0]).state is SessionState.LOST,
            timeout=8.0,
        ), (
            "the session on the silent rank never became LOST "
            f"(state={gateway.sessions.get(first[0]).state.name})"
        )

        # (a) Only its own session flips to LOST.
        assert gateway.sessions.get(second[0]).state is SessionState.READY
        # (b) The other rank keeps serving normally (not just "the state
        # field checks out" but genuinely still able to step).
        healthy_results = await gateway.policy_step(
            second, PolicyRequest(policy_id="fake")
        )
        assert not [item for item in healthy_results if isinstance(item, Err)], (
            healthy_results
        )
        # (c) Not auto-replayed: operations on a LOST session are rejected
        # outright, with no new env step.
        steps_before = victim.pools.pools
        rejected = await gateway.policy_step(first, PolicyRequest(policy_id="fake"))
        assert isinstance(rejected[0], Err)
        assert rejected[0].error.code is ErrorCode.SESSION_NOT_READY
        assert rejected[0].error.detail.get("state") == "LOST"
        assert victim.pools.pools is steps_before
        # (d) Not recoverable: only close then recreate is allowed.
        with pytest.raises(InvalidTransition):
            gateway.sessions.transition(first[0], SessionState.READY)
        closed = await gateway.close_sessions(first)
        assert not [item for item in closed if isinstance(item, Err)]
        assert gateway.sessions.get(first[0]).state is SessionState.CLOSED
    finally:
        with contextlib.suppress(BaseException):
            await gateway.stop()
        with contextlib.suppress(BaseException):
            await runtime.aclose()


async def test_a_recovered_rank_can_serve_new_sessions(transport_kind) -> None:
    """A rank becomes healthy again once it starts answering, and can serve
    a **new** session (the old one remains ``LOST``).

    Args:
        transport_kind: The transport under test.
    """
    runtime = await build_two_rank_runtime(transport_kind)
    gateway = runtime.gateway
    try:
        spec = _spec(pool_size=1, episode_length=32)
        session_ids = await open_sessions(runtime, spec, key_prefix="rec")
        rank = gateway.sessions.get(session_ids[0]).worker_rank
        worker = runtime.env_workers[rank]
        original = worker.handle_control

        async def never_answers(_envelope: ControlEnvelope) -> Any:
            await asyncio.sleep(3600)

        worker.handle_control = never_answers  # type: ignore[assignment]
        assert await wait_until(
            lambda: gateway.sessions.get(session_ids[0]).state is SessionState.LOST,
            timeout=8.0,
        )
        assert not gateway.workers.get(rank).healthy

        worker.handle_control = original  # type: ignore[assignment]
        assert await wait_until(lambda: gateway.workers.get(rank).healthy, timeout=8.0)
        # The old session is still LOST (not recovered), but the rank can
        # already accept new work -- provided close also released the
        # worker-side binding (``_release_lost_binding``), otherwise the
        # ``max_sessions_per_rank=1`` slot leaks permanently.
        assert gateway.sessions.get(session_ids[0]).state is SessionState.LOST
        await gateway.close_sessions(session_ids)
        assert session_ids[0] not in worker.sessions, (
            "closing a LOST session on a recovered rank must release the binding, "
            "otherwise the pre-allocated slot leaks forever (plan D6)"
        )
        fresh = await open_sessions(runtime, spec, key_prefix="fresh")
        assert gateway.sessions.get(fresh[0]).state is SessionState.READY
    finally:
        with contextlib.suppress(BaseException):
            await gateway.stop()
        with contextlib.suppress(BaseException):
            await runtime.aclose()


def _spec(*, pool_size: int, episode_length: int) -> Any:
    """Build a fake env spec (same defaults as conftest's factory).

    Args:
        pool_size: Number of pool slots.
        episode_length: Episode length; changing it switches to a separate
            pool (it enters the digest).

    Returns:
        ``EnvSpecMsg``.
    """
    from rollout_runtime.api.messages import EnvSpecMsg

    return EnvSpecMsg(
        env_family="fake",
        env_config={
            "action_dim": 7,
            "chunk_size": 4,
            "episode_length": episode_length,
            "image_height": 16,
            "image_width": 16,
            "state_dim": 8,
        },
        pool_size=pool_size,
    )
