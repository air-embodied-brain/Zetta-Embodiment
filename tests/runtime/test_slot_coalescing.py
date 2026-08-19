"""Real coalescing by ``SlotGroupCoalescer`` and the masking semantics of
``lockstep_vector``.

These cases run the **real** ``run_lockstep_chunk`` (the algorithm shared by
all three real families) and the real ``RuntimeEnvWorker`` /
``RuntimeGateway``; only the environment itself is the fake backend. Each of
the four masking semantics has assertions:

1. Every lane in the pool must be reset first, otherwise ``chunk_step``
   returns ``SESSION_NOT_READY``;
2. An absent lane is carried forward by the hold action and counted as
   ``masked_steps``; what ``observe`` reads is the **refreshed** frame;
3. Group-wide early stop: the whole group stops as soon as any lane newly terminates.
4. An absent lane's state is **read back** into ``SessionSlot``:
   ``policy_step`` does not run inference on a stale frame, a masked
   termination is seen by its own session, and a ``per_slot`` pool never
   pays the read-back cost.

Four points settled during a later review of point 4 also have
assertions: ``run_episode`` folds a termination that occurred while absent
into a **normal termination** (otherwise ``eval_adapter`` would record it as
``Err`` -> ``valid=False``, dropping a genuine success from the success
rate's denominator); the core side backstops accounting for a ``frozen``
participant, so read-back is no longer the sole line of defense; one sync
only calls ``observe`` once; frames are no longer refreshed against the core
once the episode has ended.

Also included are assertions about coalescing itself: N sessions'
same-tick ``policy_step`` genuinely produces only **one** ``chunk_step``;
a ``per_slot`` pool never coalesces; different chunk lengths never enter
the same group.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import numpy as np
import pytest

from rollout_runtime.api.enums import EnvOperation, ErrorCode, Priority
from rollout_runtime.api.errors import RuntimeApiError
from rollout_runtime.api.ids import RequestId, SessionId
from rollout_runtime.api.internal import CommandEnvelope
from rollout_runtime.api.messages import (
    EnvSpecMsg,
    EpisodeRequest,
    PolicyRequest,
    ResetSpec,
)
from rollout_runtime.api.result import Err, unwrap
from rollout_runtime.core.env_execution import (
    LOCKSTEP_VECTOR_FORM,
    PER_SLOT_FORM,
)
from rollout_runtime.launch.local import build_local_components  # noqa: I001
from rollout_runtime.workers.env_worker import SlotGroupCoalescer
from tests.runtime.conftest import local_runtime_config, open_sessions

POOL_SIZE = 4
CHUNK = 4


def lockstep_env_spec(pool_size: int = POOL_SIZE, **overrides: Any) -> EnvSpecMsg:
    """Build a vector-form fake env spec.

    ``core_form`` is a key of ``env_config``, so it **enters the digest**:
    a vector pool and a per_slot pool are therefore always two distinct
    pools.

    Args:
        pool_size: Pool capacity (= number of lanes).
        **overrides: Overrides for ``env_config``.

    Returns:
        The env spec.
    """
    config: dict[str, Any] = {
        "action_dim": 7,
        "chunk_size": CHUNK,
        "episode_length": 64,
        "image_height": 8,
        "image_width": 8,
        "state_dim": 8,
        "core_form": LOCKSTEP_VECTOR_FORM,
    }
    config.update(overrides)
    return EnvSpecMsg(env_family="fake", env_config=config, pool_size=pool_size)


# ------------------------------------------------------------------ Pure-function groups


def _envelope(session: str, pool_key: str, chunk_len: int) -> CommandEnvelope:
    """Build a command envelope carrying ``pool_key`` / ``chunk_len``.

    Args:
        session: Session id.
        pool_key: Pool key.
        chunk_len: Chunk length.

    Returns:
        The command envelope.
    """
    return CommandEnvelope(
        request_id=RequestId(f"req-{session}"),
        session_id=SessionId(session),
        binding_token=None,
        episode_id=None,
        operation_seq=None,
        operation=EnvOperation.ACTION_STEP,
        deadline=None,
        priority=Priority.BATCH,
        payload={"pool_key": pool_key, "chunk_len": chunk_len},
        trace_context={},
    )


def test_coalesce_groups_by_pool_and_chunk_length() -> None:
    """The grouping key is ``(pool_key, chunk_len)``; different chunk
    lengths never end up in the same group."""
    coalescer = SlotGroupCoalescer(enabled=True)
    commands = [
        _envelope("a", "pool-1", 4),
        _envelope("b", "pool-1", 4),
        _envelope("c", "pool-1", 8),
        _envelope("d", "pool-2", 4),
    ]
    groups = coalescer.coalesce(commands)
    shapes = sorted(
        sorted(str(command.session_id) for command in group) for group in groups
    )
    assert shapes == [["a", "b"], ["c"], ["d"]]


def test_coalesce_disabled_keeps_every_command_alone() -> None:
    """With ``enabled=False``, it falls back to v1's stub semantics: every
    command occupies its own group."""
    coalescer = SlotGroupCoalescer(enabled=False)
    commands = [_envelope("a", "pool-1", 4), _envelope("b", "pool-1", 4)]
    assert [len(group) for group in coalescer.coalesce(commands)] == [1, 1]


async def test_submit_fires_one_group_when_all_lanes_join() -> None:
    """Fires as soon as everyone is present: 3 slots trigger only **one**
    execute, and each gets back its own share of the result."""
    coalescer = SlotGroupCoalescer(enabled=True, window_seconds=5.0)
    calls: list[list[int]] = []

    async def execute(slots: list[int], blocks: list[Any]) -> list[Any]:
        calls.append(list(slots))
        return [f"outcome-{slot}" for slot in slots]

    results = await asyncio.gather(
        *[
            coalescer.submit(
                pool_key="pool",
                slot_index=slot,
                block=np.zeros((CHUNK, 7), dtype=np.float32),
                chunk_len=CHUNK,
                expected=3,
                execute=execute,
            )
            for slot in (0, 1, 2)
        ]
    )
    assert len(calls) == 1, f"expected one coalesced chunk_step, got {calls}"
    assert sorted(calls[0]) == [0, 1, 2]
    assert sorted(results) == ["outcome-0", "outcome-1", "outcome-2"]
    stats = coalescer.stats()
    assert stats["groups_executed"] == 1
    assert stats["coalesced_commands"] == 3
    assert stats["max_group_size"] == 3
    assert stats["window_timeouts"] == 0


async def test_submit_falls_back_to_the_window_when_a_lane_is_absent() -> None:
    """When a lane is absent, once the window elapses it must still be
    sent, recording one ``window_timeouts``."""
    coalescer = SlotGroupCoalescer(enabled=True, window_seconds=0.05)
    calls: list[list[int]] = []

    async def execute(slots: list[int], blocks: list[Any]) -> list[Any]:
        calls.append(list(slots))
        return list(slots)

    result = await coalescer.submit(
        pool_key="pool",
        slot_index=1,
        block=np.zeros((CHUNK, 7), dtype=np.float32),
        chunk_len=CHUNK,
        expected=4,
        execute=execute,
    )
    assert result == 1
    assert calls == [[1]]
    assert coalescer.stats()["window_timeouts"] == 1


async def test_group_failure_reaches_every_waiter() -> None:
    """When the whole group fails, every waiter in the group must get the
    same exception (D5: never silently swallowed)."""
    coalescer = SlotGroupCoalescer(enabled=True, window_seconds=5.0)

    async def execute(slots: list[int], blocks: list[Any]) -> list[Any]:
        raise RuntimeError("group blew up")

    async def submit(slot: int) -> BaseException | None:
        try:
            await coalescer.submit(
                pool_key="pool",
                slot_index=slot,
                block=np.zeros((CHUNK, 7), dtype=np.float32),
                chunk_len=CHUNK,
                expected=2,
                execute=execute,
            )
        except BaseException as exc:  # noqa: BLE001 - this case needs the exception itself
            return exc
        return None

    errors = await asyncio.gather(submit(0), submit(1))
    assert all(isinstance(error, RuntimeError) for error in errors), errors
    assert {str(error) for error in errors} == {"group blew up"}


# -------------------------------------------------- Real coalescing and masking through the worker


async def test_lockstep_pool_really_coalesces_one_chunk_step(
    transport_kind: str,
) -> None:
    """4 sessions' same-tick ``policy_step`` -> the execution core is
    called only once."""
    config = local_runtime_config(
        transport_kind, env_worker={"max_sessions_per_rank": POOL_SIZE}
    )
    runtime = build_local_components(config)
    await runtime.start()
    try:
        spec = lockstep_env_spec()
        sessions = await open_sessions(runtime, spec, POOL_SIZE, key_prefix="lock")
        assert not [
            item.error
            for item in await runtime.gateway.reset(sessions, ResetSpec(seed=1))
            if isinstance(item, Err)
        ]
        worker = runtime.env_workers[0]
        pool = next(iter(worker.pools.pools.values()))
        assert pool.lockstep is True
        assert pool.core.core_form == LOCKSTEP_VECTOR_FORM
        before = pool.core.coalesced_group_count
        results = await runtime.gateway.policy_step(sessions, PolicyRequest())
        assert not [item.error for item in results if isinstance(item, Err)]
        # One coalesced batch = the execution core only gained one more chunk_step call.
        assert pool.core.coalesced_group_count == before + 1
        stats = worker.coalescer.stats()
        assert stats["max_group_size"] == POOL_SIZE
        horizons = {unwrap(item).executed_horizon for item in results}
        assert horizons == {CHUNK}, horizons
        for item in results:
            info = unwrap(item).info
            assert sorted(info["coalesced_slots"]) == list(range(POOL_SIZE))
            assert info["masked_slots"] == []
            assert info["core_form"] == LOCKSTEP_VECTOR_FORM
    finally:
        await runtime.gateway.stop()
        await runtime.aclose()


async def test_per_slot_pool_never_coalesces(transport_kind: str) -> None:
    """A ``per_slot`` pool always takes its own ``chunk_step``, even when
    4 sessions share the same tick."""
    config = local_runtime_config(
        transport_kind, env_worker={"max_sessions_per_rank": POOL_SIZE}
    )
    runtime = build_local_components(config)
    await runtime.start()
    try:
        spec = lockstep_env_spec(core_form=PER_SLOT_FORM)
        sessions = await open_sessions(runtime, spec, POOL_SIZE, key_prefix="perslot")
        await runtime.gateway.reset(sessions, ResetSpec(seed=1))
        worker = runtime.env_workers[0]
        pool = next(iter(worker.pools.pools.values()))
        assert pool.lockstep is False
        results = await runtime.gateway.policy_step(sessions, PolicyRequest())
        assert not [item.error for item in results if isinstance(item, Err)]
        # The coalescer was never used even once (a per_slot pool takes the
        # single-command path directly).
        assert worker.coalescer.stats()["groups_executed"] == 0
        for item in results:
            assert "coalesced_slots" not in unwrap(item).info
    finally:
        await runtime.gateway.stop()
        await runtime.aclose()


async def test_lockstep_requires_every_lane_to_be_reset(transport_kind: str) -> None:
    """Masking semantic 1: refuse to advance the whole vector if any lane
    has never been reset."""
    config = local_runtime_config(
        transport_kind, env_worker={"max_sessions_per_rank": POOL_SIZE}
    )
    runtime = build_local_components(config)
    await runtime.start()
    try:
        spec = lockstep_env_spec()
        sessions = await open_sessions(runtime, spec, POOL_SIZE, key_prefix="partial")
        worker = runtime.env_workers[0]
        # Only reset the first two lanes; the other two were never reset.
        # Slot allocation order is a result of concurrent admission, so the
        # expected value must be looked up from the worker's session
        # table, not hardcoded.
        await runtime.gateway.reset(sessions[:2], ResetSpec(seed=1))
        expected_unstarted = sorted(
            worker.sessions[session].slot_index for session in sessions[2:]
        )
        results = await runtime.gateway.policy_step(sessions[:2], PolicyRequest())
        errors = [item.error for item in results if isinstance(item, Err)]
        assert errors, "stepping a half-reset lockstep pool must be refused"
        assert {error.code for error in errors} == {ErrorCode.SESSION_NOT_READY}
        assert "must be reset before the first chunk_step" in errors[0].message
        assert errors[0].detail["unstarted_slots"] == expected_unstarted
    finally:
        await runtime.gateway.stop()
        await runtime.aclose()


async def test_absent_lane_is_masked_and_observe_tells_the_truth(
    transport_kind: str,
) -> None:
    """Masking semantic 2: an absent lane really is advanced, and
    ``observe`` must report the new frame."""
    config = local_runtime_config(
        transport_kind,
        env_worker={"max_sessions_per_rank": POOL_SIZE, "coalesce_window_ms": 30.0},
    )
    runtime = build_local_components(config)
    await runtime.start()
    try:
        spec = lockstep_env_spec(pool_size=2)
        sessions = await open_sessions(runtime, spec, 2, key_prefix="mask")
        await runtime.gateway.reset(sessions, ResetSpec(seed=1))
        worker = runtime.env_workers[0]
        pool = next(iter(worker.pools.pools.values()))
        slots = {worker.sessions[s].slot_index: s for s in sessions}
        absent_slot = max(slots)
        stale = unwrap((await runtime.gateway.observe([slots[absent_slot]]))[0])
        # Only submit one lane; the other is absent and gets carried
        # forward by the hold action once the window elapses.
        driver = slots[min(slots)]
        result = unwrap(
            (await runtime.gateway.policy_step([driver], PolicyRequest()))[0]
        )
        assert result.info["masked_slots"] == [absent_slot]
        masked_lane = pool.core._slots[absent_slot]
        assert masked_lane.masked_steps == result.executed_horizon > 0
        assert pool.core.total_masked_steps == masked_lane.masked_steps
        # Key: the observation must honestly reflect "it moved," and must
        # not palm off a stale cache on the caller.
        fresh = unwrap((await runtime.gateway.observe([slots[absent_slot]]))[0])
        assert fresh.step_index > stale.step_index
        assert worker.coalescer.stats()["window_timeouts"] >= 1
    finally:
        await runtime.gateway.stop()
        await runtime.aclose()


async def test_group_early_stops_on_the_first_new_termination(
    transport_kind: str,
) -> None:
    """Masking semantic 3: the whole group stops as soon as any lane newly
    terminates, with each lane's ``executed_horizon`` reported honestly."""
    config = local_runtime_config(
        transport_kind, env_worker={"max_sessions_per_rank": 2}
    )
    runtime = build_local_components(config)
    await runtime.start()
    try:
        # episode_length=2 and chunk=4: terminates at step 2, so the whole
        # group can only advance 2 steps.
        spec = lockstep_env_spec(pool_size=2, episode_length=2)
        sessions = await open_sessions(runtime, spec, 2, key_prefix="early")
        await runtime.gateway.reset(sessions, ResetSpec(seed=1))
        results = await runtime.gateway.policy_step(sessions, PolicyRequest())
        assert not [item.error for item in results if isinstance(item, Err)]
        for item in results:
            step = unwrap(item)
            # The chunk requested 4 steps, but both lanes terminated at
            # step 2 -> the whole group stops at step 2.
            assert step.executed_horizon == 2, step.executed_horizon
            assert step.terminated is True
            assert step.info["group_early_stopped"] is True
            assert step.info["requested_horizon"] == CHUNK
            assert len(step.per_step or []) == 2
            assert [record.step_index for record in step.per_step or []] == [1, 2]
    finally:
        await runtime.gateway.stop()
        await runtime.aclose()


async def test_mixed_chunk_lengths_are_refused_inside_one_group() -> None:
    """The same group must share one chunk length (otherwise it would need
    a second round of masking along the time dimension)."""
    from rollout_runtime.backends.fake.env import FakeEnvCore, register_fake_env_family

    register_fake_env_family()
    core = FakeEnvCore()
    core.build(lockstep_env_spec(pool_size=2), num_envs=2)
    core.reset([0, 1], ResetSpec(seed=1))
    with pytest.raises(Exception) as excinfo:
        core.chunk_step(
            [0, 1],
            [
                np.zeros((4, 7), dtype=np.float32),
                np.zeros((2, 7), dtype=np.float32),
            ],
        )
    assert "single chunk length" in str(excinfo.value)
    core.close()


async def test_a_pool_is_built_exactly_once_under_concurrent_bindings(
    transport_kind: str,
) -> None:
    """A second real defect exposed on GPU: concurrent ``create_sessions``
    could build the same pool N times.

    The original implementation was
    ``await asyncio.to_thread(pools.ensure_pool, spec)``, while bindings
    are admitted concurrently, so N tasks would all see "the pool doesn't
    exist yet" and each build one, with the later build overwriting the
    earlier -- the earlier session ends up holding a slot in the orphaned
    pool, so two sessions land on **the same lane**. The symptom is highly
    misleading: ``reset`` all succeed, but ``chunk_step`` reports "a lane
    was never reset."

    Earlier validation missed this because the ``max_sessions_per_rank``
    of two a100 presets was 1 (an EGL limitation).
    """
    config = local_runtime_config(
        transport_kind, env_worker={"max_sessions_per_rank": POOL_SIZE}
    )
    runtime = build_local_components(config)
    await runtime.start()
    try:
        worker = runtime.env_workers[0]
        original = worker.pools.ensure_pool
        builds: list[str] = []

        def counting_ensure_pool(env_spec: EnvSpecMsg) -> Any:
            """Record one real pool build, and stretch out the build
            window to amplify the race."""
            builds.append(env_spec.digest())
            time.sleep(0.05)
            return original(env_spec)

        worker.pools.ensure_pool = counting_ensure_pool  # type: ignore[method-assign]
        spec = lockstep_env_spec()
        sessions = await open_sessions(runtime, spec, POOL_SIZE, key_prefix="race")
        # The same digest is only allowed to be genuinely built once.
        assert builds == [spec.digest()], builds
        # Each session gets a **mutually distinct** lane, filling the whole pool.
        slots = sorted(worker.sessions[session].slot_index for session in sessions)
        assert slots == list(range(POOL_SIZE)), slots
        assert len(worker.pools.pools) == 1
        # Once every reset has succeeded, the vector pool's precondition
        # (every lane reset) should be satisfied.
        assert not [
            item.error
            for item in await runtime.gateway.reset(sessions, ResetSpec(seed=1))
            if isinstance(item, Err)
        ]
        results = await runtime.gateway.policy_step(sessions, PolicyRequest())
        assert not [item.error for item in results if isinstance(item, Err)]
    finally:
        await runtime.gateway.stop()
        await runtime.aclose()


async def test_a_cancelled_leader_still_wakes_its_followers() -> None:
    """A capacity leak found during an independent audit: a follower would
    hang forever if the leader gets cancelled.

    A hung follower keeps occupying ``slot.active_op``, and
    ``recover_expired_sessions`` explicitly skips a session with an
    ``active_op``, so that slot would never return to the pool.
    """
    coalescer = SlotGroupCoalescer(enabled=True, window_seconds=5.0)

    async def execute(slots: list[Any], blocks: list[Any]) -> list[Any]:
        return list(slots)

    async def submit(slot: int) -> Any:
        return await coalescer.submit(
            pool_key="pool",
            slot_index=slot,
            block=np.zeros((CHUNK, 7), dtype=np.float32),
            chunk_len=CHUNK,
            expected=3,  # Never fills up; the leader will sit at the window
            execute=execute,
        )

    leader = asyncio.create_task(submit(0))
    follower = asyncio.create_task(submit(1))
    # Let both tasks enter the waiting state.
    await asyncio.sleep(0.05)
    leader.cancel()
    with pytest.raises(asyncio.CancelledError):
        await leader
    # The follower must be woken up (not hang forever).
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(follower, timeout=2.0)


async def test_a_short_result_list_does_not_strand_waiters() -> None:
    """When the executor returns the wrong number of results, every waiter
    in the group must get an explicit error."""
    coalescer = SlotGroupCoalescer(enabled=True, window_seconds=5.0)

    async def execute(slots: list[Any], blocks: list[Any]) -> list[Any]:
        return ["only-one"]  # 2 slots but only 1 result returned

    async def submit(slot: int) -> BaseException | None:
        try:
            await coalescer.submit(
                pool_key="pool",
                slot_index=slot,
                block=np.zeros((CHUNK, 7), dtype=np.float32),
                chunk_len=CHUNK,
                expected=2,
                execute=execute,
            )
        except BaseException as exc:  # noqa: BLE001 - this case needs the exception itself
            return exc
        return None

    errors = await asyncio.wait_for(asyncio.gather(submit(0), submit(1)), timeout=3.0)
    assert all(error is not None for error in errors), errors
    assert all(
        isinstance(error, RuntimeApiError) and error.info.code is ErrorCode.INTERNAL
        for error in errors
    ), errors


async def test_a_masked_lane_that_finishes_reports_its_termination(
    transport_kind: str,
) -> None:
    """A real defect found during an independent audit: a lane carried
    forward by the hold action terminated, but nobody recorded it.

    The consequences are two-fold: (1) its session would never see that
    termination (while "success is judged only by the environment's
    termination signal" is a hard rule, and maniskill's success predicate
    isn't even monotonic, so missing one step means losing one success);
    (2) since it was never frozen, it would be treated as a "new
    termination" in the **next** group, early-stopping the whole group.
    """
    config = local_runtime_config(
        transport_kind,
        env_worker={"max_sessions_per_rank": 2, "coalesce_window_ms": 30.0},
    )
    runtime = build_local_components(config)
    await runtime.start()
    try:
        # episode_length=4, chunk=4: one group can push even the absent lane to termination.
        spec = lockstep_env_spec(pool_size=2, episode_length=4)
        sessions = await open_sessions(runtime, spec, 2, key_prefix="maskfin")
        await runtime.gateway.reset(sessions, ResetSpec(seed=1))
        worker = runtime.env_workers[0]
        pool = next(iter(worker.pools.pools.values()))
        slots = {worker.sessions[s].slot_index: s for s in sessions}
        driver, absent = min(slots), max(slots)
        result = unwrap(
            (await runtime.gateway.policy_step([slots[driver]], PolicyRequest()))[0]
        )
        assert result.info["masked_slots"] == [absent]
        masked = pool.core._slots[absent]
        assert masked.masked_steps == 4
        # The two key points: the termination was recorded, and it was frozen.
        assert masked.terminated is True
        assert masked.frozen is True
        # So the next group is not early-stopped by this stale lane: the
        # driving side has itself terminated too, and stepping again
        # should get EPISODE_TERMINATED rather than a fake "took 1 step" result.
        again = (await runtime.gateway.policy_step([slots[driver]], PolicyRequest()))[0]
        assert isinstance(again, Err)
        assert again.error.code is ErrorCode.EPISODE_TERMINATED
    finally:
        await runtime.gateway.stop()
        await runtime.aclose()


async def test_a_masked_lane_never_infers_on_a_stale_frame(
    transport_kind: str,
) -> None:
    """The next ``policy_step`` for an absent lane must run inference on
    the **new** frame.

    The original implementation only patched "read the core rather than
    the worker cache" inside ``observe``; ``policy_step`` still used
    ``slot.last_observation`` -- and ``_step_slot`` only updates the slot
    of the submitter, so a lane carried forward by the hold action would
    run inference on a stale frame. This uses one property of the fake
    policy to make it assertable: its action chunk is a deterministic
    function of ``observation.step_index``, so recording the step_index
    carried by every request reveals whether the model was fed a new or old frame.
    """
    config = local_runtime_config(
        transport_kind,
        env_worker={"max_sessions_per_rank": 2, "coalesce_window_ms": 30.0},
    )
    runtime = build_local_components(config)
    await runtime.start()
    try:
        spec = lockstep_env_spec(pool_size=2)
        sessions = await open_sessions(runtime, spec, 2, key_prefix="stalefr")
        await runtime.gateway.reset(sessions, ResetSpec(seed=1))
        worker = runtime.env_workers[0]
        pool = next(iter(worker.pools.pools.values()))
        slots = {worker.sessions[s].slot_index: s for s in sessions}
        driver_slot, absent_slot = min(slots), max(slots)

        # Record the observation.step_index fed to the model for each session.
        core = runtime.rollout_workers[0].policy
        seen: list[tuple[str, int]] = []
        original = core.ainfer_batch

        async def spy(requests: list[Any]) -> Any:
            for request in requests:
                seen.append(
                    (str(request.session_id), int(request.observation.step_index))
                )
            return await original(requests)

        core.ainfer_batch = spy  # type: ignore[method-assign]

        # First group: only the driver submits; absent is absent and gets
        # carried forward for CHUNK steps.
        first = unwrap(
            (await runtime.gateway.policy_step([slots[driver_slot]], PolicyRequest()))[
                0
            ]
        )
        assert first.info["masked_slots"] == [absent_slot]
        masked_lane = pool.core._slots[absent_slot]
        # Snapshot it: the second group will make this lane advance on its
        # own too, so the assertion must be against the value from before
        # entering inference.
        pushed_to = int(masked_lane.step_index)
        assert pushed_to == first.executed_horizon > 0

        # Second group: switch to absent submitting. What it feeds the
        # model must be the core side's new step_index.
        seen.clear()
        second = unwrap(
            (await runtime.gateway.policy_step([slots[absent_slot]], PolicyRequest()))[
                0
            ]
        )
        assert second.executed_horizon > 0
        absent_seen = [
            step for session, step in seen if session == str(slots[absent_slot])
        ]
        assert absent_seen, "the absent lane must have reached the policy exactly once"
        assert absent_seen[0] == pushed_to, (
            "policy_step fed the model a stale frame: it sent step_index "
            f"{absent_seen[0]} while hold actions had already pushed the lane to "
            f"{pushed_to}"
        )
        assert absent_seen[0] > 0
    finally:
        await runtime.gateway.stop()
        await runtime.aclose()


async def test_masked_termination_reaches_its_own_session(
    transport_kind: str,
) -> None:
    """A lane carried forward to termination -- its own session must be
    able to see that termination.

    The core side already recorded it honestly (``lane.terminated`` /
    ``lane.frozen``), but the worker's ``SessionSlot`` could not see it,
    so ``_require_running_episode`` would let it through and
    ``chunk_step`` would take the ``if lane.frozen: continue`` branch for
    that lane, returning a fake result with ``executed_horizon=0`` and
    ``terminated`` false -- equivalent to discarding a success that
    already happened, while "success is judged only by the environment's
    termination signal" is a hard rule.
    """
    config = local_runtime_config(
        transport_kind,
        env_worker={"max_sessions_per_rank": 2, "coalesce_window_ms": 30.0},
    )
    runtime = build_local_components(config)
    await runtime.start()
    try:
        # episode_length=4, chunk=4: one group can push even the absent lane to termination.
        spec = lockstep_env_spec(pool_size=2, episode_length=4)
        sessions = await open_sessions(runtime, spec, 2, key_prefix="maskterm")
        await runtime.gateway.reset(sessions, ResetSpec(seed=1))
        worker = runtime.env_workers[0]
        pool = next(iter(worker.pools.pools.values()))
        slots = {worker.sessions[s].slot_index: s for s in sessions}
        driver_slot, absent_slot = min(slots), max(slots)

        unwrap(
            (await runtime.gateway.policy_step([slots[driver_slot]], PolicyRequest()))[
                0
            ]
        )
        assert pool.core._slots[absent_slot].terminated is True

        # Key: the absent party stepping again must get EPISODE_TERMINATED,
        # not a fake 0-step success.
        again = (
            await runtime.gateway.policy_step([slots[absent_slot]], PolicyRequest())
        )[0]
        assert isinstance(again, Err)
        assert again.error.code is ErrorCode.EPISODE_TERMINATED

        # The termination flag was genuinely read back into SessionSlot
        # (``_lockstep_lane_count`` relies on it to exclude lanes that have
        # finished, otherwise every subsequent group would wait in vain
        # until the window times out).
        absent_session = slots[absent_slot]
        assert worker.sessions[absent_session].terminated is True
        assert all(slot.terminated for slot in worker.sessions.values())
        assert worker._lockstep_lane_count(pool.pool_key) == 1
    finally:
        await runtime.gateway.stop()
        await runtime.aclose()


async def test_per_slot_pools_never_pay_for_the_lockstep_read_back(
    transport_kind: str,
) -> None:
    """A ``per_slot`` pool must never be touched by the read-back logic:
    slots are independent envs and are never advanced by anyone else."""
    config = local_runtime_config(
        transport_kind, env_worker={"max_sessions_per_rank": 2}
    )
    runtime = build_local_components(config)
    await runtime.start()
    try:
        spec = lockstep_env_spec(pool_size=2, core_form=PER_SLOT_FORM)
        sessions = await open_sessions(runtime, spec, 2, key_prefix="perslotrb")
        await runtime.gateway.reset(sessions, ResetSpec(seed=1))
        worker = runtime.env_workers[0]
        pool = next(iter(worker.pools.pools.values()))
        assert pool.lockstep is False
        calls = 0
        original = pool.core.lane_status

        def counting(slots: Any) -> Any:
            nonlocal calls
            calls += 1
            return original(slots)

        pool.core.lane_status = counting  # type: ignore[method-assign]
        results = await runtime.gateway.policy_step(sessions, PolicyRequest())
        assert not [item for item in results if isinstance(item, Err)]
        assert calls == 0, "per_slot pools must not read lane status at all"
    finally:
        await runtime.gateway.stop()
        await runtime.aclose()


async def test_a_masked_termination_ends_run_episode_as_a_terminated_episode(
    transport_kind: str,
) -> None:
    """A termination pushed through while absent, ``run_episode`` must
    fold it as a **normal termination** rather than an ``Err``.

    This termination is discovered by read-back **inside** the loop
    (synced right after inference, before ``chunk_step``), so
    ``_policy_step_inner`` raises ``EPISODE_TERMINATED``. If it leaked out
    unchanged, ``eval_adapter`` would record it as ``Err`` ->
    ``valid=False``, wiping a genuine environment success out of the
    success rate's denominator along with it -- while "success is judged
    only by the environment's termination signal" is a hard rule.

    The timing is aligned like this: hook ``_step_slot``, and before the
    absent party actually submits, first let the other lane run one group
    on its own to completion. This is exactly what happens on the real
    path when "inference latency crosses a coalescing window," just
    without relying on a timing race.
    """
    config = local_runtime_config(
        transport_kind,
        env_worker={"max_sessions_per_rank": 2, "coalesce_window_ms": 30.0},
    )
    runtime = build_local_components(config)
    await runtime.start()
    try:
        # episode_length=CHUNK: one group can push even the absent lane to termination.
        spec = lockstep_env_spec(pool_size=2, episode_length=CHUNK)
        sessions = await open_sessions(runtime, spec, 2, key_prefix="runepterm")
        await runtime.gateway.reset(sessions, ResetSpec(seed=1))
        worker = runtime.env_workers[0]
        pool = next(iter(worker.pools.pools.values()))
        slots = {worker.sessions[s].slot_index: s for s in sessions}
        absent_slot = max(slots)
        driver, absent = slots[min(slots)], slots[absent_slot]

        original_step = worker._step_slot
        driven = False

        async def step_after_the_other_lane_ran(
            slot: Any, block: Any, **kwargs: Any
        ) -> Any:
            nonlocal driven
            if not driven and slot.session_id == absent:
                driven = True
                unwrap(
                    (await runtime.gateway.policy_step([driver], PolicyRequest()))[0]
                )
            return await original_step(slot, block, **kwargs)

        worker._step_slot = step_after_the_other_lane_ran  # type: ignore[method-assign]

        result = unwrap(
            (
                await runtime.gateway.run_episode(
                    [absent], EpisodeRequest(max_steps=4, policy=PolicyRequest())
                )
            )[0]
        )
        assert driven, "the hook never fired; the timing this test needs did not happen"
        assert pool.core._slots[absent_slot].terminated is True
        assert result.terminated is True, (
            "a masked termination was reported as 'still running'; "
            f"stop_reason={result.stop_reason}"
        )
        assert result.stop_reason == "terminated"
        # No action was actually executed in that step, so it doesn't count.
        assert result.num_policy_steps == 0
        assert result.executed_horizon == 0
    finally:
        await runtime.gateway.stop()
        await runtime.aclose()


async def test_a_frozen_participant_still_gets_an_honest_outcome(
    transport_kind: str,
) -> None:
    """Core-side backstop: even when a ``frozen`` lane genuinely comes in
    as a participant, it must not return a fake "still running" result.

    The worker's read-back (masking semantic 4) should already keep it
    out, but that is best-effort -- ``_sync_lockstep_pool`` swallows sync
    failures, with the benefit that a single sync failure doesn't turn the
    operation into an error. The precondition is that the core itself
    also honestly accounts for it: ``executed_horizon=0`` (it executed
    zero steps) plus a genuine termination flag.
    """
    config = local_runtime_config(
        transport_kind, env_worker={"max_sessions_per_rank": 2}
    )
    runtime = build_local_components(config)
    await runtime.start()
    try:
        spec = lockstep_env_spec(pool_size=2, episode_length=CHUNK)
        sessions = await open_sessions(runtime, spec, 2, key_prefix="frozenp")
        await runtime.gateway.reset(sessions, ResetSpec(seed=1))
        worker = runtime.env_workers[0]
        pool = next(iter(worker.pools.pools.values()))
        slots = {worker.sessions[s].slot_index: s for s in sessions}
        driver_slot, absent_slot = min(slots), max(slots)
        unwrap(
            (await runtime.gateway.policy_step([slots[driver_slot]], PolicyRequest()))[
                0
            ]
        )
        lane = pool.core._slots[absent_slot]
        assert lane.frozen is True

        # Call the core directly, bypassing the worker's
        # `_require_running_episode` -- simulating "the sync never happened."
        block = np.zeros((1, 7), dtype=np.float32)
        outcome = pool.core.chunk_step([absent_slot], [block])[0]
        assert outcome.executed_horizon == 0
        assert outcome.terminated is True, (
            "a frozen participant reported terminated=False: that drops a real "
            "environment success"
        )
        assert outcome.info["frozen_participant"] is True
    finally:
        await runtime.gateway.stop()
        await runtime.aclose()


async def test_one_sync_refreshes_every_stale_lane_in_a_single_observe(
    transport_kind: str,
) -> None:
    """One sync merges the whole pool's stale lanes into **a single**
    ``observe`` call, rather than calling once per lane.

    ``EnvExecutionCore.observe`` has had a batch signature since it was
    introduced; calling it lane by lane would mean paying the pool lock
    and thread hop once per lane, and the pool lock on a vector pool is
    shared by all lanes.
    """
    config = local_runtime_config(
        transport_kind,
        env_worker={"max_sessions_per_rank": 3, "coalesce_window_ms": 30.0},
    )
    runtime = build_local_components(config)
    await runtime.start()
    try:
        spec = lockstep_env_spec(pool_size=3)
        sessions = await open_sessions(runtime, spec, 3, key_prefix="batchobs")
        await runtime.gateway.reset(sessions, ResetSpec(seed=1))
        worker = runtime.env_workers[0]
        pool = next(iter(worker.pools.pools.values()))
        slots = {worker.sessions[s].slot_index: s for s in sessions}
        driver_slot = min(slots)
        absent_slots = sorted(index for index in slots if index != driver_slot)

        # Only let the driver submit: the other two lanes both get carried
        # forward by the hold action, so both go stale.
        result = unwrap(
            (await runtime.gateway.policy_step([slots[driver_slot]], PolicyRequest()))[
                0
            ]
        )
        assert result.info["masked_slots"] == absent_slots

        calls: list[list[int]] = []
        original = pool.core.observe

        def counting(slot_indices: Any) -> Any:
            calls.append([int(index) for index in slot_indices])
            return original(slot_indices)

        pool.core.observe = counting  # type: ignore[method-assign]
        fresh = unwrap((await runtime.gateway.observe([slots[absent_slots[0]]]))[0])
        assert calls == [absent_slots], (
            f"one sync must read every stale lane in a single observe call, got {calls}"
        )
        assert fresh.step_index == result.executed_horizon > 0
    finally:
        await runtime.gateway.stop()
        await runtime.aclose()


async def test_a_finished_lane_stops_paying_for_frame_refreshes(
    transport_kind: str,
) -> None:
    """No more frame refreshes against the core once the episode has ended.

    The core side counts ``masked_steps`` **unconditionally** for absent
    lanes (regardless of ``frozen``), so a finished lane's watermark keeps
    rising with every subsequent group. If frames were refreshed based on
    the watermark, every finished lane would pay for one wasted
    ``observe`` in every subsequent group; and those movements belong to
    **someone else's** episode, so overlaying them onto this session's
    final frame would be dishonest.
    """
    config = local_runtime_config(
        transport_kind, env_worker={"max_sessions_per_rank": 2}
    )
    runtime = build_local_components(config)
    await runtime.start()
    try:
        spec = lockstep_env_spec(pool_size=2, episode_length=CHUNK)
        sessions = await open_sessions(runtime, spec, 2, key_prefix="finlane")
        await runtime.gateway.reset(sessions, ResetSpec(seed=1))
        worker = runtime.env_workers[0]
        pool = next(iter(worker.pools.pools.values()))
        slots = {worker.sessions[s].slot_index: s for s in sessions}
        driver_slot, absent_slot = min(slots), max(slots)
        unwrap(
            (await runtime.gateway.policy_step([slots[driver_slot]], PolicyRequest()))[
                0
            ]
        )
        absent_session = slots[absent_slot]
        # First let it sync its termination flag. **The sync that
        # discovers the termination** must still refresh the frame,
        # refreshing precisely the frame at the moment of termination --
        # otherwise masking semantic 2's "never pretend it didn't move"
        # would break on the very last step.
        at_termination = unwrap((await runtime.gateway.observe([absent_session]))[0])
        assert worker.sessions[absent_session].terminated is True
        assert at_termination.step_index == CHUNK, (
            "the frame at the terminating step was never read back: "
            f"step_index={at_termination.step_index}"
        )

        # Afterward, the core-side lane keeps getting advanced by other
        # groups, but this session's frame should no longer move along with it.
        calls = 0
        original = pool.core.observe

        def counting(slot_indices: Any) -> Any:
            nonlocal calls
            calls += 1
            return original(slot_indices)

        pool.core.observe = counting  # type: ignore[method-assign]
        block = np.zeros((CHUNK, 7), dtype=np.float32)
        pool.core.chunk_step([driver_slot], [block])
        assert pool.core._slots[absent_slot].masked_steps > CHUNK
        again = unwrap((await runtime.gateway.observe([absent_session]))[0])
        assert calls == 0, "a finished lane must not pay for a frame refresh"
        assert again.step_index == at_termination.step_index
    finally:
        await runtime.gateway.stop()
        await runtime.aclose()
