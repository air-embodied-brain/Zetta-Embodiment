"""Semantics and "no session leakage" for the Gym Adapter (``test_gym_adapter.py``).

The acceptance criterion: ``RuntimeGymEnv`` running 100 steps must not leak
sessions. This also verifies that the Adapter only speaks through
``RuntimeClient`` (it cannot get at transport / worker), and that session
and episode are two distinct things (multiple ``reset`` calls reuse the same
session).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from rollout_runtime.adapters.gym_adapter import RuntimeGymEnv
from rollout_runtime.api.enums import ErrorCode, SessionState
from rollout_runtime.api.errors import RuntimeApiError
from rollout_runtime.launch.local import LocalRuntime


async def test_gym_env_runs_100_steps_without_leaking_sessions(
    local_runtime: LocalRuntime, fake_env_spec: Any
) -> None:
    """After running 100 full steps (spanning multiple episodes), there is
    still only one session.

    Args:
        local_runtime: Same-process runtime.
        fake_env_spec: Env spec factory.
    """
    gateway = local_runtime.gateway
    worker = local_runtime.env_workers[0]
    env = RuntimeGymEnv(
        gateway,
        fake_env_spec(episode_length=12),
        application_id="gym-test",
        policy_id="fake",
    )

    observation, info = await env.reset(seed=1)
    assert observation.step_index == 0
    assert info["episode_id"] == 1
    session_id = env.session_id
    assert session_id is not None
    assert len(worker.sessions) == 1

    total_steps = 0
    episodes = 1
    while total_steps < 100:
        observation, reward, terminated, truncated, step_info = await env.step()
        total_steps += step_info["executed_horizon"]
        assert observation.session_id == session_id
        assert isinstance(reward, float)
        if terminated or truncated:
            observation, info = await env.reset(seed=1 + episodes)
            episodes += 1
            assert observation.step_index == 0
            assert info["episode_id"] == episodes

    assert total_steps >= 100
    assert episodes > 1, "100 steps must span more than one episode"
    # Only one session, one env slot throughout: session and episode are two
    # distinct things.
    assert len(worker.sessions) == 1
    assert env.session_id == session_id
    assert len(gateway.sessions) == 1
    status = await gateway.get_session(session_id)
    assert status.state is SessionState.READY
    assert status.episode_id == episodes

    await env.close()
    assert worker.sessions == {}
    assert (await gateway.get_session(session_id)).state is SessionState.CLOSED
    assert env.session_id is None


async def test_gym_env_accepts_explicit_actions(
    local_runtime: LocalRuntime, fake_env_spec: Any
) -> None:
    """``step(action)`` goes through ``action_step``, ``step(None)`` goes
    through ``policy_step``.

    Args:
        local_runtime: Same-process runtime.
        fake_env_spec: Env spec factory.
    """
    env = RuntimeGymEnv(
        local_runtime.gateway, fake_env_spec(episode_length=64), policy_id="fake"
    )
    async with env:
        await env.reset(seed=3)

        single = np.full(7, 0.5, dtype=np.float32)
        observation, _, _, _, info = await env.step(single)
        assert info["executed_horizon"] == 1
        assert observation.step_index == 1
        assert observation.extras["last_action_checksum"] == pytest.approx(0.5 * 7)
        assert "model_version" not in info

        chunk = np.zeros((3, 7), dtype=np.float32)
        observation, _, _, _, info = await env.step(chunk)
        assert info["executed_horizon"] == 3
        assert observation.step_index == 4

        observation, _, _, _, info = await env.step()
        assert info["executed_horizon"] == 4
        assert info["model_version"] == "fake-v1"
        assert observation.step_index == 8
        assert env.step_count == 3
        assert env.episode_count == 1


async def test_gym_env_reset_options_and_close_are_idempotent(
    local_runtime: LocalRuntime, fake_env_spec: Any
) -> None:
    """``reset``'s options are promoted to ``ResetSpec`` fields, and
    ``close`` can be called repeatedly.

    Args:
        local_runtime: Same-process runtime.
        fake_env_spec: Env spec factory.
    """
    env = RuntimeGymEnv(
        local_runtime.gateway, fake_env_spec(episode_length=16), policy_id="fake"
    )
    observation, _ = await env.reset(
        seed=5, options={"task_id": 3, "instruction": "put the cube down"}
    )
    assert observation.instruction == "put the cube down"
    assert observation.extras["task_id"] == 3

    await env.close()
    await env.close()
    assert local_runtime.env_workers[0].sessions == {}

    with pytest.raises(RuntimeApiError) as excinfo:
        await env.reset()
    assert excinfo.value.info.code is ErrorCode.SESSION_NOT_READY


async def test_gym_env_never_touches_transport_or_workers(
    local_runtime: LocalRuntime, fake_env_spec: Any
) -> None:
    """The Adapter only holds a ``RuntimeClient``: it cannot get at
    transport, nor at workers.

    Args:
        local_runtime: Same-process runtime.
        fake_env_spec: Env spec factory.
    """
    env = RuntimeGymEnv(local_runtime.gateway, fake_env_spec(), policy_id="fake")
    attributes = {
        name: value
        for name, value in vars(env).items()
        if not callable(value) and name != "_client"
    }
    assert not any(
        "transport" in name or "worker" in name or "channel" in name
        for name in attributes
    )
    # The only external dependency satisfies the RuntimeClient protocol.
    from rollout_runtime.api.client import RuntimeClient

    assert isinstance(local_runtime.gateway, RuntimeClient)
