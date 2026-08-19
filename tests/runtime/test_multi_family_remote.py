"""Remote acceptance for the maniskill family: real maniskill + Gym Adapter +
batch eval.

Three acceptance criteria are landed here as re-runnable test cases:

1. **Family smoke + capability matches actual behavior**: the real
   ``ManiskillEnv``'s 5-key schema, ``seed_options`` reset signature,
   ``gpu_batched`` device form, and rejection of undeclared extensions;
2. **The Gym Adapter runs successfully on a non-libero family** (proving
   the Runtime is decoupled from Agent / LIBERO semantics);
3. **Batch eval**: ``EvaluationAdapter`` genuinely coalesces batches on a
   vector pool, and reports episodes/hour.

How to run (mirroring image ``rlinf/rlinf:agentic-rlinf0.3-maniskill_libero``):

```bash
docker exec -w /workspace/Zetta-Embodiment \
  -e PYTHONPATH=/workspace/Zetta-Embodiment \
  -e ZETTA_RLINF_ROOT=/workspace/Zetta-Embodiment/third_party/rlinf \
  -e CUDA_VISIBLE_DEVICES=0 runtime-container \
  /opt/venv/openpi/bin/python -m pytest tests/runtime/test_multi_family_remote.py -m remote -q -s
```

``RR_MANISKILL_ENV_ID`` can switch tasks (default ``PickCube-v1``).
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from typing import Any

import numpy as np
import pytest
import pytest_asyncio

from rollout_runtime.adapters.eval_adapter import EvaluationAdapter, EvaluationTask
from rollout_runtime.adapters.gym_adapter import RuntimeGymEnv
from rollout_runtime.api.enums import ErrorCode
from rollout_runtime.api.messages import EnvSpecMsg, PolicyRequest
from rollout_runtime.api.result import Err
from rollout_runtime.core.env_execution import (
    LOCKSTEP_VECTOR_FORM,
    PER_SLOT_FORM,
)
from rollout_runtime.core.env_registry import behavior_for

pytestmark = pytest.mark.remote

MANISKILL_FAMILY = "maniskill"
ACTION_DIM = 7
CHUNK = 4
LANES = 4


def _env_config(**overrides: Any) -> dict[str, Any]:
    """maniskill's ``env_config`` (real hardware).

    Args:
        **overrides: Overrides.

    Returns:
        The config dict.
    """
    config: dict[str, Any] = {
        "env_id": os.environ.get("RR_MANISKILL_ENV_ID", "PickCube-v1"),
        "obs_mode": "rgb",
        "control_mode": "pd_ee_delta_pose",
        "sim_backend": "gpu",
        "camera_height": 128,
        "camera_width": 128,
        "wrap_obs_mode": "simple",
        "reward_mode": "raw",
        "max_episode_steps": 60,
        "action_dim": ACTION_DIM,
        "chunk_size": CHUNK,
        "action_model_type": "openpi",
        "action_policy": "panda_wristcam",
    }
    config.update(overrides)
    return config


def _runtime_config(*, lanes: int, core_form: str) -> Any:
    """Build a 1x1 inproc + fake policy real-hardware config.

    The fake policy is deliberate: what needs verifying here is "the family
    plugs in correctly, the declaration matches actual behavior, and
    coalescing genuinely happens," while this image has no maniskill VLA
    weights.

    Args:
        lanes: Pool capacity.
        core_form: Execution core form.

    Returns:
        ``RuntimeConfig``.
    """
    from rollout_runtime.config.schema import load_config

    config = load_config("local_fake")
    config.env_family = MANISKILL_FAMILY
    config.env_config = _env_config(core_form=core_form)
    config.transport.kind = "inproc"
    config.transport.command_timeout_seconds = 600.0
    config.env_worker.max_sessions_per_rank = lanes
    config.env_worker.default_pool_size = lanes
    # maniskill is gpu_batched: the worker must honestly report that it
    # "occupies an accelerator," otherwise the Gateway's serves() would
    # judge it unable to serve this family.
    config.env_worker.has_accelerator = True
    config.env_worker.coalesce_slot_groups = True
    config.env_worker.coalesce_window_ms = 500.0
    config.rollout_worker.policy_backend = "fake"
    config.rollout_worker.policy_id = "fake"
    config.rollout_worker.policy_family = "fake"
    config.rollout_worker.scheduler.max_wait_ms = 0.0
    config.rollout_worker.scheduler.max_batch_size = 8
    config.admission.max_sessions_per_application = 64
    config.admission.max_total_inflight_operations = 512
    config.admission.max_inflight_operations_per_application = 256
    return config


@pytest_asyncio.fixture(loop_scope="module")
async def maniskill_vector_runtime() -> AsyncIterator[Any]:
    """A vector-form real maniskill runtime (module scope, builds one
    sapien scene).

    Yields:
        A ``start()``-ed ``LocalRuntime``.
    """
    from rollout_runtime.launch.local import build_local_components

    runtime = build_local_components(
        _runtime_config(lanes=LANES, core_form=LOCKSTEP_VECTOR_FORM)
    )
    await runtime.start()
    try:
        yield runtime
    finally:
        await runtime.gateway.stop()
        await runtime.aclose()


@pytest_asyncio.fixture(loop_scope="module")
async def maniskill_single_runtime() -> AsyncIterator[Any]:
    """A single-slot ``per_slot`` real maniskill runtime (for the Gym Adapter).

    Yields:
        A ``start()``-ed ``LocalRuntime``.
    """
    from rollout_runtime.launch.local import build_local_components

    runtime = build_local_components(_runtime_config(lanes=1, core_form=PER_SLOT_FORM))
    await runtime.start()
    try:
        yield runtime
    finally:
        await runtime.gateway.stop()
        await runtime.aclose()


def _spec(*, lanes: int, core_form: str) -> EnvSpecMsg:
    """Build an env spec.

    Args:
        lanes: Pool capacity.
        core_form: Execution core form.

    Returns:
        The env spec.
    """
    return EnvSpecMsg(
        env_family=MANISKILL_FAMILY,
        env_config=_env_config(core_form=core_form),
        pool_size=lanes,
    )


@pytest.mark.asyncio(loop_scope="module")
async def test_assertion_1_maniskill_capability_matches_the_real_behaviour(
    maniskill_vector_runtime: Any,
) -> None:
    """Acceptance 1: every item in the declaration table matches the real
    maniskill."""
    from rollout_runtime.api.messages import CreateSessionRequest, ResetSpec
    from rollout_runtime.api.result import unwrap

    runtime = maniskill_vector_runtime
    behavior = behavior_for(MANISKILL_FAMILY)
    worker = runtime.env_workers[0]
    capability = worker.capabilities()[MANISKILL_FAMILY]
    assert capability.needs_accelerator is behavior.needs_accelerator is True
    assert worker.worker_info().has_accelerator is True
    assert capability.core_forms == behavior.core_forms
    assert capability.supports_coalescing is True
    assert capability.extensions == frozenset()

    spec = _spec(lanes=LANES, core_form=LOCKSTEP_VECTOR_FORM)
    created = await runtime.gateway.create_sessions(
        [
            CreateSessionRequest(
                application_id="m6",
                client_session_key=f"cap-{index}",
                env_spec=spec,
                default_policy_id="fake",
                lease_seconds=900.0,
            )
            for index in range(LANES)
        ]
    )
    assert not [item.error for item in created if isinstance(item, Err)]
    sessions = [unwrap(item).session_id for item in created]
    pool = next(iter(worker.pools.pools.values()))
    # The declared form == the form actually built; the vector form = one
    # env with pool_size lanes.
    assert pool.core.core_form == LOCKSTEP_VECTOR_FORM
    assert pool.lockstep is True
    assert len(pool.core._envs) == 1
    assert len(pool.core._lanes) == LANES

    resets = await runtime.gateway.reset(sessions, ResetSpec(seed=1))
    assert not [item.error for item in resets if isinstance(item, Err)]
    first = unwrap(resets[0]).observation
    assert first is not None
    assert first.main_image is not None
    assert first.main_image.dtype == "uint8"
    assert tuple(first.main_image.shape) == (128, 128, 3)
    assert len(first.state) > 0
    assert first.extras["core_form"] == LOCKSTEP_VECTOR_FORM
    print(
        f"[M6-assert1] family={MANISKILL_FAMILY} env_id={_env_config()['env_id']} "
        f"image={tuple(first.main_image.shape)} state_dim={len(first.state)} "
        f"lanes={len(pool.core._lanes)} accel={capability.needs_accelerator}"
    )

    # An undeclared extension is rejected as declared (not crashed).
    extension = (
        await runtime.gateway.extension_call(
            [sessions[0]], "libero", "render_camera", {}
        )
    )[0]
    assert isinstance(extension, Err)
    assert extension.error.code is ErrorCode.UNSUPPORTED_EXTENSION
    print(f"[M6-assert1] undeclared extension -> {extension.error.code.name}")

    # The same pool, same tick genuinely coalesces: one policy_step only
    # advances the execution core by one group.
    before = pool.core.coalesced_group_count
    results = await runtime.gateway.policy_step(sessions, PolicyRequest())
    assert not [item.error for item in results if isinstance(item, Err)]
    assert pool.core.coalesced_group_count == before + 1
    info = unwrap(results[0]).info
    assert sorted(info["coalesced_slots"]) == list(range(LANES))
    assert info["masked_slots"] == []
    horizons = [unwrap(item).executed_horizon for item in results]
    print(
        f"[M6-assert1] coalesced group: slots={sorted(info['coalesced_slots'])} "
        f"horizons={horizons} groups={pool.core.coalesced_group_count} "
        f"stats={worker.coalescer.stats()}"
    )
    await runtime.gateway.close_sessions(sessions)


@pytest.mark.asyncio(loop_scope="module")
async def test_assertion_3_gym_adapter_runs_on_a_non_libero_family(
    maniskill_single_runtime: Any,
) -> None:
    """Acceptance 3: the Gym Adapter runs successfully on maniskill -- the
    Runtime does not depend on LIBERO / Agent semantics."""
    runtime = maniskill_single_runtime
    env = RuntimeGymEnv(
        runtime.gateway,
        _spec(lanes=1, core_form=PER_SLOT_FORM),
        application_id="gym-m6",
        policy_id="fake",
    )
    observation, info = await env.reset(seed=3)
    assert observation.main_image is not None
    assert tuple(observation.main_image.shape) == (128, 128, 3)
    assert info["episode_id"] == 1
    rewards = 0.0
    for _ in range(3):
        # Explicit action (gymnasium semantics), bypassing the policy.
        observation, reward, terminated, truncated, step_info = await env.step(
            np.zeros((CHUNK, ACTION_DIM), dtype=np.float32)
        )
        rewards += reward
        assert step_info["executed_horizon"] == CHUNK
        if terminated or truncated:
            break
    # A policy-driven step (action=None -> policy_step).
    observation, reward, terminated, truncated, step_info = await env.step(None)
    assert step_info["executed_horizon"] >= 1
    print(
        f"[M6-assert3] gym on {MANISKILL_FAMILY}: steps={env.step_count} "
        f"episodes={env.episode_count} reward_sum={rewards:.4f} "
        f"last_horizon={step_info['executed_horizon']}"
    )
    await env.close()
    assert env.session_id is None


@pytest.mark.asyncio(loop_scope="module")
async def test_assertion_2_eval_adapter_reports_episodes_per_hour(
    maniskill_vector_runtime: Any,
) -> None:
    """Half of acceptance 2: batch eval runs a task list to completion on
    the vector pool and reports episodes/hour.

    See the runtime validation notes for the comparison basis against
    rlinf's native eval: the success rate is judged only by the
    environment's termination signal, with valid / invalid episodes
    tallied separately.
    """
    runtime = maniskill_vector_runtime
    adapter = EvaluationAdapter(
        runtime.gateway,
        _spec(lanes=LANES, core_form=LOCKSTEP_VECTOR_FORM),
        application_id="eval-m6",
        concurrency=LANES,
        max_steps=4,
        sink_id="mem:m6-eval",
    )
    report = await adapter.run_episodes(
        [EvaluationTask(task_id=None, seed=seed) for seed in range(LANES)],
        PolicyRequest(policy_id="fake"),
    )
    assert report.attempted == LANES
    assert report.invalid == 0, report.error_counts
    assert report.valid == LANES
    worker = runtime.env_workers[0]
    pool = next(iter(worker.pools.pools.values()))
    stats = worker.coalescer.stats()
    assert stats["max_group_size"] == LANES
    records = worker.sinks.memory("mem:m6-eval")
    assert len(records) == sum(outcome.num_policy_steps for outcome in report.outcomes)
    print(
        f"[M6-assert2] eval on {MANISKILL_FAMILY}: {report.summary()} "
        f"coalescer={stats} masked_steps={pool.core.total_masked_steps}"
    )
