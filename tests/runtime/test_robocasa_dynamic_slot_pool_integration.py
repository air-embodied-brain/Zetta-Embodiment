# Copyright (c) 2026 Zetta Contributors
"""``EnvPool.dynamic``/``acquire_or_grow``/``shrink_idle`` against the **real**
``RobocasaCurrentCore`` (not the ``fake`` family's ``DynamicSlotPool`` stand-in that
``tests/runtime/test_dynamic_slot_pool.py`` uses).

Why a separate file instead of extending ``test_dynamic_slot_pool.py``: that file's
fixtures build a ``local_fake``-preset ``LocalRuntime`` (``env_family="fake"``).
Proving robocasa's ``add_slot``/``remove_slot``/``slot_count`` actually make
``EnvPool.dynamic`` true and get called by ``EnvPool.acquire_or_grow``/
``shrink_idle`` requires standing up a ``LocalRuntime`` with ``env_family="robocasa"``
instead -- the whole point of this task is confirming the real backend integrates,
not re-confirming the protocol-level scheduling logic (already covered against the
fake backend).

The real RoboCasa/robosuite simulator is not installed in this environment, so
``RoboCasaSession._ensure_environment`` is monkeypatched with the same minimal fake
gym env ``tests/runtime/test_robocasa_current_backend.py`` uses. ``policy_backend``
is ``fake`` (not ``groot``): ``FakePolicyCore.infer_batch`` only depends on
``(session_id, episode_id, step_index)``, never observation content
(``rollout_runtime/backends/fake/policy.py`` module docstring), so it accepts real
robocasa ``Observation`` payloads without needing a real GR00T checkpoint.
"""

from __future__ import annotations

import contextlib
from collections.abc import AsyncIterator
from typing import Any

import numpy as np
import pytest

from robots.robocasa import session_core
from rollout_runtime.api.messages import CreateSessionRequest, EnvSpecMsg
from rollout_runtime.api.result import Err, Ok
from rollout_runtime.config.schema import RuntimeConfig
from rollout_runtime.launch.local import LocalRuntime, build_local_components
from tests.runtime.conftest import open_sessions


class _FakeEnv:
    """Same minimal fake gym env as ``test_robocasa_current_backend.py``."""

    def __init__(self) -> None:
        self.actions: list[Any] = []
        self._step_index = 0

    def reset(self, seed: int) -> tuple[dict[str, Any], dict]:
        del seed
        self._step_index = 0
        return self._observation(), {"success": False}

    def step(self, action: dict[str, np.ndarray]) -> tuple[Any, ...]:
        self.actions.append(action)
        self._step_index += 1
        terminated = self._step_index >= 3
        return (
            self._observation(),
            1.0 if terminated else 0.0,
            terminated,
            False,
            {"success": terminated},
        )

    def close(self) -> None:
        pass

    def _observation(self) -> dict[str, Any]:
        return {
            "video.robot0_agentview_left": np.zeros((4, 4, 3), dtype=np.uint8),
            "video.robot0_agentview_right": np.full((4, 4, 3), 9, dtype=np.uint8),
            "video.robot0_eye_in_hand": np.full((4, 4, 3), 7, dtype=np.uint8),
            "state.end_effector_position_relative": np.array(
                [0.1, 0.2, 0.3], dtype=np.float32
            ),
            "state.gripper_qpos": np.array([0.0, 1.0], dtype=np.float32),
            "task_descriptions": ["move the pan"],
        }


@pytest.fixture
def fake_ensure_environment(monkeypatch: pytest.MonkeyPatch):
    """Stub ``RoboCasaSession._ensure_environment`` (see the unit test file's twin
    fixture for the full rationale).
    """
    envs: dict[int, _FakeEnv] = {}

    def _fake_ensure_environment(self, task: str, split: str) -> None:
        env = _FakeEnv()
        envs[id(self)] = env
        self.env = env
        self.identity = (task, split)

    monkeypatch.setattr(
        session_core.RoboCasaSession,
        "_ensure_environment",
        _fake_ensure_environment,
    )
    return envs


def _robocasa_runtime_config(*, max_dynamic_pool_size: int = 4) -> RuntimeConfig:
    """A minimal in-process ``robocasa`` config: ``fake`` policy backend, no GPU."""
    config = RuntimeConfig()
    config.env_family = "robocasa"
    config.env_config = {
        "task": "SlideDishwasherRack",
        "require_isolated_renderer": False,
    }
    config.transport.kind = "inproc"
    config.env_worker.placement_strategy = "node"
    # The robocasa family declares needs_accelerator_override=True (RoboCasaSession
    # rendering forces MUJOCO_GL=egl); EnvWorkerRegistry.select_rank only schedules
    # onto a rank that declares an accelerator -- the stubbed fake environment used
    # in this test doesn't actually need a GPU, but has_accelerator=True must still
    # be declared explicitly, otherwise scheduling is blocked at the rank-selection
    # step (not the failure mode this test is meant to cover; see the field
    # description on EnvWorkerConfig.accelerator_present()).
    config.env_worker.has_accelerator = True
    config.env_worker.max_sessions_per_rank = max_dynamic_pool_size
    config.env_worker.default_pool_size = 1
    config.rollout_worker.policy_id = "fake"
    config.rollout_worker.policy_family = "fake"
    config.rollout_worker.policy_backend = "fake"
    config.rollout_worker.device = "cpu"
    config.rollout_worker.dtype = "float32"
    config.rollout_worker.policy_config = {"action_dim": 12}
    return config


@pytest.fixture
async def robocasa_runtime(
    fake_ensure_environment,
) -> AsyncIterator[LocalRuntime]:
    """A ``LocalRuntime`` wired to the real ``robocasa`` family (stubbed simulator)."""
    runtime = build_local_components(_robocasa_runtime_config())
    await runtime.start()
    try:
        yield runtime
    finally:
        with contextlib.suppress(BaseException):
            await runtime.gateway.stop()
        await runtime.aclose()


def _robocasa_spec(*, max_dynamic_pool_size: int | None) -> EnvSpecMsg:
    return EnvSpecMsg(
        env_family="robocasa",
        env_config={
            "task": "SlideDishwasherRack",
            "require_isolated_renderer": False,
        },
        pool_size=1,
        max_dynamic_pool_size=max_dynamic_pool_size,
    )


# ------------------------------------------------------- EnvPool integration


async def test_env_pool_dynamic_is_true_once_the_real_robocasa_core_is_built(
    robocasa_runtime: LocalRuntime,
) -> None:
    """``EnvPool.dynamic`` probes for the three methods ``add_slot``/``remove_slot``/
    ``slot_count`` -- before these methods existed, ``dynamic`` was always false;
    now that the real ``RobocasaCurrentCore`` is wired in, it must be true.
    """
    spec = _robocasa_spec(max_dynamic_pool_size=4)
    (session,) = await open_sessions(robocasa_runtime, spec, key_prefix="d0")

    pool = robocasa_runtime.env_workers[0].pools.find(spec.digest())
    assert pool is not None
    assert pool.lockstep is False, "robocasa only declares per_slot"
    assert pool.dynamic is True, (
        "RobocasaCurrentCore now implements DynamicSlotPool; EnvPool.dynamic must "
        "detect it instead of falling back to the fixed-pool D6 semantics"
    )

    assert isinstance((await robocasa_runtime.gateway.close_sessions([session]))[0], Ok)


async def test_acquire_or_grow_calls_add_slot_instead_of_quota_exceeded_when_full(
    robocasa_runtime: LocalRuntime,
) -> None:
    """When the pool is full but ``max_dynamic_pool_size`` has not been reached, the
    second session must trigger ``RobocasaCurrentCore.add_slot`` to actually append
    an independent ``RoboCasaSession``, instead of being outright rejected with
    ``QUOTA_EXCEEDED`` by ``_reserve_cold_create_locked`` (the pre-fix behavior when
    ``RobocasaCurrentCore`` lacked the ``DynamicSlotPool`` protocol implementation).
    """
    spec = _robocasa_spec(max_dynamic_pool_size=3)
    (first,) = await open_sessions(robocasa_runtime, spec, key_prefix="d1")

    pool = robocasa_runtime.env_workers[0].pools.find(spec.digest())
    assert pool is not None
    assert pool.pool_size == 1

    created = await robocasa_runtime.gateway.create_sessions(
        [
            CreateSessionRequest(
                application_id="test",
                client_session_key="d1-grow",
                env_spec=spec,
                default_policy_id="fake",
            )
        ]
    )
    assert isinstance(created[0], Ok), created[0]
    second = created[0].value.session_id

    assert pool.pool_size == 2, "pool must have grown from 1 to 2 slots via add_slot"
    # The new slot is a genuinely independent RoboCasaSession (not the same session serving two requests).
    sessions = [slot.session for slot in pool.core._slots]
    assert len({id(session) for session in sessions}) == 2

    for session in (first, second):
        assert isinstance(
            (await robocasa_runtime.gateway.close_sessions([session]))[0], Ok
        )


async def test_acquire_or_grow_still_rejects_past_max_dynamic_pool_size(
    robocasa_runtime: LocalRuntime,
) -> None:
    """After growing to ``max_dynamic_pool_size``, further growth is still explicitly rejected instead of growing unbounded."""
    spec = _robocasa_spec(max_dynamic_pool_size=2)
    sessions = await open_sessions(robocasa_runtime, spec, count=2, key_prefix="d2")

    pool = robocasa_runtime.env_workers[0].pools.find(spec.digest())
    assert pool is not None
    assert pool.pool_size == 2

    from rollout_runtime.api.enums import ErrorCode

    denied = (
        await robocasa_runtime.gateway.create_sessions(
            [
                CreateSessionRequest(
                    application_id="test",
                    client_session_key="d2-overflow",
                    env_spec=spec,
                    default_policy_id="fake",
                )
            ]
        )
    )[0]
    assert isinstance(denied, Err)
    assert denied.error.code is ErrorCode.QUOTA_EXCEEDED
    assert pool.pool_size == 2

    for session in sessions:
        assert isinstance(
            (await robocasa_runtime.gateway.close_sessions([session]))[0], Ok
        )


async def test_shrink_idle_calls_remove_slot_and_closes_the_underlying_session(
    robocasa_runtime: LocalRuntime,
) -> None:
    """When ``EnvPool.shrink_idle`` shrinks a trailing idle slot, it must actually call
    ``RobocasaCurrentCore.remove_slot`` (closing the underlying ``RoboCasaSession``),
    not merely drop the index from ``warm_free_slots``.
    """
    spec = _robocasa_spec(max_dynamic_pool_size=4)
    sessions = await open_sessions(robocasa_runtime, spec, count=3, key_prefix="d3")

    pool = robocasa_runtime.env_workers[0].pools.find(spec.digest())
    assert pool is not None
    assert pool.pool_size == 3
    trailing_session_obj = pool.core._slots[-1].session

    for session in sessions:
        assert isinstance(
            (await robocasa_runtime.gateway.close_sessions([session]))[0], Ok
        )
    assert pool.in_use == 0

    removed = await pool.shrink_idle()
    assert removed == 2, "must shrink back down to the initial pool_size=1 floor"
    assert pool.pool_size == 1
    # remove_slot() really did call close_environment() (session.env becoming None is the probe).
    assert trailing_session_obj.env is None


# ------------------------------------------------------- Boundary tests


async def test_grow_to_max_shrink_and_grow_again_via_the_real_env_pool(
    robocasa_runtime: LocalRuntime,
) -> None:
    """``EnvPool``/``LocalRuntime``-level version of the lesson learned from LIBERO:
    grow to the ceiling, shrink all the way down, then grow again -- all through the
    real create_sessions/close_sessions/shrink_idle call chain rather than poking
    ``RobocasaCurrentCore``'s internal methods directly -- confirming that the full
    scheduling chain (``acquire_or_grow`` -> ``core.add_slot``, ``shrink_idle`` ->
    ``core.remove_slot``) has no index out-of-bounds/misalignment issues on the real
    robocasa core.
    """
    spec = _robocasa_spec(max_dynamic_pool_size=4)

    # Grow to the ceiling: 1 initial + 3 dynamically added = 4.
    sessions = await open_sessions(robocasa_runtime, spec, count=4, key_prefix="d4")
    pool = robocasa_runtime.env_workers[0].pools.find(spec.digest())
    assert pool is not None
    assert pool.pool_size == 4
    assert pool.in_use == 4

    # Once at the ceiling, another request must be explicitly rejected (no silent reuse/overflow).
    from rollout_runtime.api.enums import ErrorCode

    denied = (
        await robocasa_runtime.gateway.create_sessions(
            [
                CreateSessionRequest(
                    application_id="test",
                    client_session_key="d4-overflow",
                    env_spec=spec,
                    default_policy_id="fake",
                )
            ]
        )
    )[0]
    assert isinstance(denied, Err)
    assert denied.error.code is ErrorCode.QUOTA_EXCEEDED

    # Release all and shrink back to the initial pool_size=1.
    for session in sessions:
        assert isinstance(
            (await robocasa_runtime.gateway.close_sessions([session]))[0], Ok
        )
    removed = await pool.shrink_idle()
    assert removed == 3
    assert pool.pool_size == 1

    # Grow again: the new session reuses the warm slot left behind by the shrink
    # (index 0, pool_size still 1 -- EnvPool.acquire_or_grow prefers reusing
    # warm_free_slots and does not skip the reuse path just because there is an
    # add_slot history), proving that the slot left after shrinking is still a
    # fully normal, reusable slot rather than some "half-shrunk" bad state. Drive
    # it with an actual reset + close to prove it truly works, not merely that it
    # "looks present but raises IndexError on access."
    (fresh,) = await open_sessions(robocasa_runtime, spec, key_prefix="d4-again")
    assert pool.pool_size == 1, "reusing the surviving warm slot must not grow the pool"
    from rollout_runtime.api.messages import ResetSpec

    reset_result = (
        await robocasa_runtime.gateway.reset([fresh], ResetSpec(seed=1))
    )[0]
    assert isinstance(reset_result, Ok), reset_result
    assert isinstance(
        (await robocasa_runtime.gateway.close_sessions([fresh]))[0], Ok
    )
    assert pool.pool_size == 1

    # Now actually occupy the single pool_size=1 slot, then request one more to
    # genuinely trigger add_slot, confirming that the growth path still correctly
    # appends at index 1 (not 2 or some other stale index) even after a round of
    # growth followed by shrinkage -- the exact lesson from LIBERO was that the
    # "index returned by reuse" and "real capacity" bookkeeping fell out of sync.
    (occupant,) = await open_sessions(robocasa_runtime, spec, key_prefix="d4-occupy")
    assert pool.pool_size == 1
    (grown,) = await open_sessions(robocasa_runtime, spec, key_prefix="d4-grown")
    assert pool.pool_size == 2
    grown_index = robocasa_runtime.env_workers[0].sessions[grown].slot_index
    assert grown_index == 1, "the freshly grown slot must land on index 1, not a stale one"

    for session in (occupant, grown):
        assert isinstance(
            (await robocasa_runtime.gateway.close_sessions([session]))[0], Ok
        )
