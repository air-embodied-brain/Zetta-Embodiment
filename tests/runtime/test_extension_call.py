"""Real-hardware acceptance test: real libero + pi0.5 (``test_extension_call.py``).

All test cases are marked ``@pytest.mark.remote``: the local `.venv-runtime` does not
have mujoco / robosuite / rlinf, so it is removed by default via ``runtime_ci.sh``'s
``-m "not remote"``; run it on a configured GPU host with
``pytest tests/runtime/test_extension_call.py -q -m remote``.

Covers five acceptance assertions plus four LIBERO privileged methods:

1. The observation's 5-key schema and dtype/layout after ``reset`` (uint8 HWC / float32
   state / str instruction);
2. ``policy_step`` returns finite values of shape ``[chunk, 7]``, and
   ``executed_horizon`` is consistent with ``per_step``;
3. Stepping after termination is rejected (``EPISODE_TERMINATED``), and after ``reset``
   the ``episode_id`` increments and the episode can continue;
4. Multiple consecutive episodes in the same session behave correctly (verifies that
   ``create_session`` and ``reset`` are separate);
5. The four LIBERO methods in ``extension_call`` return structures matching the legacy
   ``LiberoEnvClient``.

Environment variables:
- ``RR_PI05_MODEL_PATH`` overrides the weights path in the preset (do not commit
  machine-specific paths to the repository);
- ``RR_LIBERO_SUITE`` / ``RR_LIBERO_TASK_ID`` override the task;
- ``RR_RUNTIME_CONFIG`` overrides the preset name (default ``a100_libero_pi05``).
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from typing import Any

import numpy as np
import pytest
import pytest_asyncio

from rollout_runtime.api.enums import ErrorCode
from rollout_runtime.api.messages import (
    CreateSessionRequest,
    EnvSpecMsg,
    PolicyRequest,
    ResetSpec,
)
from rollout_runtime.api.result import Err
from rollout_runtime.config.schema import load_config
from rollout_runtime.core import payload as payload_module

pytestmark = [
    pytest.mark.remote,
    # Module-scoped fixtures need a module-scoped event loop: pytest-asyncio 1.x builds
    # a loop per test case by default, and a module-scoped async fixture would fail
    # that assertion outright.
    pytest.mark.asyncio(loop_scope="module"),
]

ACTION_DIM = 7


def _config() -> Any:
    """Load and narrow the preset according to environment variables (single rank, single session, enough for acceptance testing).

    Returns:
        ``RuntimeConfig``.
    """
    config = load_config(os.environ.get("RR_RUNTIME_CONFIG", "a100_libero_pi05"))
    config.env_worker.num_ranks = 1
    config.rollout_worker.num_ranks = 1
    config.env_worker.max_sessions_per_rank = 1
    config.cluster.component_placement = {}
    config.env_worker.placement_strategy = "node"
    config.rollout_worker.placement_strategy = "node"
    config.transport.kind = "inproc"
    config.rollout_worker.scheduler.max_wait_ms = 0.0
    model_path = os.environ.get("RR_PI05_MODEL_PATH")
    if model_path:
        config.rollout_worker.policy_config = {
            **dict(config.rollout_worker.policy_config),
            "model_path": model_path,
        }
    env_config = dict(config.env_config)
    if os.environ.get("RR_LIBERO_SUITE"):
        env_config["task_suite_name"] = os.environ["RR_LIBERO_SUITE"]
    if os.environ.get("RR_LIBERO_TASK_ID"):
        env_config["task_id"] = int(os.environ["RR_LIBERO_TASK_ID"])
    # Keep the episode shorter so acceptance testing runs faster; the terminated
    # semantics remain unchanged.
    env_config["max_episode_steps"] = int(
        os.environ.get("RR_LIBERO_MAX_STEPS", env_config.get("max_episode_steps", 512))
    )
    config.env_config = env_config
    return config


@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def libero_runtime() -> AsyncIterator[Any]:
    """Start a single-rank real runtime (libero env + pi0.5 policy), **shared across the entire module**.

    Deliberately module-scoped: pi0.5 weights are about 5.5 GB, and rebuilding per test
    case would exhaust GPU memory on a shared GPU (this is exactly how the first round
    of testing hit a ``torch.OutOfMemoryError``), and each ``load`` takes over 70s.
    Sessions are created and closed per test case by the ``libero_session`` fixture, so
    the pool capacity (``pool_size=1``) is not held across test cases.

    Yields:
        A ``LocalRuntime`` that has already been ``start()``-ed.
    """
    from rollout_runtime.launch.local import build_local_components

    runtime = build_local_components(_config())
    async with runtime:
        yield runtime


@pytest_asyncio.fixture(loop_scope="module")
async def libero_session(libero_runtime: Any, request: Any) -> AsyncIterator[Any]:
    """Create a session for the current test case and close it immediately when done.

    Args:
        libero_runtime: Module-scoped runtime.
        request: pytest request object (its name is used as the ``client_session_key``).

    Yields:
        ``SessionId``.
    """
    session_id = await _create_session(libero_runtime, request.node.name)
    try:
        yield session_id
    finally:
        await libero_runtime.gateway.close_sessions([session_id])


async def _create_session(runtime: Any, key: str) -> Any:
    """Create a session and return the ``session_id``.

    Args:
        runtime: ``LocalRuntime``.
        key: ``client_session_key``.

    Returns:
        ``SessionId``.
    """
    config = runtime.config
    results = await runtime.gateway.create_sessions(
        [
            CreateSessionRequest(
                application_id="m4",
                client_session_key=key,
                env_spec=EnvSpecMsg(
                    env_family=config.env_family,
                    env_config=dict(config.env_config),
                    pool_size=1,
                ),
                default_policy_id=config.rollout_worker.policy_id,
                lease_seconds=config.gateway.default_lease_seconds,
            )
        ]
    )
    assert not isinstance(results[0], Err), results[0]
    return results[0].value.session_id


async def test_assertion_1_reset_observation_schema(
    libero_runtime: Any, libero_session: Any
) -> None:
    """Assertion 1: the observation's 5-key schema and dtype/layout are correct after reset.

    Args:
        libero_runtime: Real runtime fixture.
        libero_session: This test case's session.
    """
    session_id = libero_session
    results = await libero_runtime.gateway.reset(
        [session_id], ResetSpec(task_id=None, seed=1)
    )
    assert not isinstance(results[0], Err), results[0]
    observation = results[0].value.observation
    assert observation is not None

    main = payload_module.decode_payload(observation.main_image)
    wrist = payload_module.decode_payload(observation.wrist_image)
    height = int(libero_runtime.config.env_config["camera_height"])
    width = int(libero_runtime.config.env_config["camera_width"])
    for image in (main, wrist):
        assert image.dtype == np.uint8, "images must be uint8 HWC"
        assert image.shape == (height, width, 3)
    assert observation.state and all(
        isinstance(value, float) for value in observation.state
    )
    assert np.isfinite(np.asarray(observation.state, dtype=np.float32)).all()
    assert isinstance(observation.instruction, str) and observation.instruction.strip()
    assert observation.extras["env_family"] == "libero"
    assert observation.step_index == 0
    print(
        f"[M4-assert1] instruction={observation.instruction!r} "
        f"state_dim={len(observation.state)} image={main.shape} "
        f"task_id={observation.extras['task_id']} "
        f"reset_state_id={observation.extras['reset_state_id']}"
    )


async def test_assertion_2_policy_step_shape_and_horizon(
    libero_runtime: Any, libero_session: Any
) -> None:
    """Assertion 2: ``policy_step`` returns finite values, and ``executed_horizon`` is consistent with ``per_step``.

    Args:
        libero_runtime: Real runtime fixture.
        libero_session: This test case's session.
    """
    session_id = libero_session
    await libero_runtime.gateway.reset([session_id], ResetSpec(seed=1))
    policy = PolicyRequest(policy_id=libero_runtime.config.rollout_worker.policy_id)
    previous_step = 0
    for _ in range(3):
        results = await libero_runtime.gateway.policy_step([session_id], policy)
        assert not isinstance(results[0], Err), results[0]
        step = results[0].value
        assert step.side_effect_applied is True
        assert step.executed_horizon >= 1
        assert step.per_step is not None, "libero declares per_step_obs_available"
        assert len(step.per_step) == step.executed_horizon
        assert step.info["per_step_obs_available"] is True
        assert step.info["chunk_obs_layout"] == "per_step"
        assert step.observation is not None
        assert step.observation.step_index == previous_step + step.executed_horizon
        previous_step = step.observation.step_index
        assert np.isfinite(np.asarray(step.observation.state, dtype=np.float32)).all()
        print(
            f"[M4-assert2] horizon={step.executed_horizon} "
            f"reward={step.reward} step_index={step.observation.step_index} "
            f"model_version={step.info.get('model_version')}"
        )
        if step.terminated or step.truncated:
            break


async def test_assertion_2b_action_chunk_is_finite_seven_dim(
    libero_runtime: Any,
    libero_session: Any,
) -> None:
    """Action side of assertion 2: the action chunk produced by the policy is a finite ``[chunk, 7]`` array.

    Calling ``extension_call`` directly does not give access to actions, so this test
    goes through the RolloutWorker core to verify the shape and values across the
    "model output -> payload -> decode" pipeline.

    Args:
        libero_runtime: Real runtime fixture.
        libero_session: This test case's session.
    """
    from rollout_runtime.api.ids import EpisodeId, OperationSeq, RequestId
    from rollout_runtime.api.internal import InferenceRequest

    session_id = libero_session
    reset = await libero_runtime.gateway.reset([session_id], ResetSpec(seed=1))
    observation = reset[0].value.observation
    core = libero_runtime.policies[0]
    request = InferenceRequest(
        request_id=RequestId("probe-1"),
        session_id=session_id,
        episode_id=EpisodeId(1),
        operation_seq=OperationSeq(1),
        policy_id=libero_runtime.config.rollout_worker.policy_id,
        observation=observation,
        inference_parameters={"mode": "eval"},
        routing_token="env:0",
        compat_key="probe",
    )
    responses = core.infer_batch([request])
    assert responses[0].error is None, responses[0].error
    block = payload_module.decode_payload(responses[0].actions)
    assert block.ndim == 2 and block.shape[1] == ACTION_DIM
    assert block.dtype == np.float32
    assert np.isfinite(block).all()
    print(
        f"[M4-assert2b] actions={block.shape} "
        f"min={float(block.min()):.4f} max={float(block.max()):.4f} "
        f"model_version={responses[0].model_version}"
    )


async def test_assertion_3_and_4_multiple_episodes_in_one_session(
    libero_runtime: Any,
    libero_session: Any,
) -> None:
    """Assertions 3 + 4: stepping after termination is rejected, ``episode_id`` increments after reset, and multiple episodes run correctly in the same session.

    Args:
        libero_runtime: Real runtime fixture.
        libero_session: This test case's session.
    """
    session_id = libero_session
    policy = PolicyRequest(policy_id=libero_runtime.config.rollout_worker.policy_id)
    episode_ids: list[int] = []
    for episode in range(2):
        results = await libero_runtime.gateway.reset(
            [session_id], ResetSpec(seed=episode + 1)
        )
        assert not isinstance(results[0], Err), results[0]
        episode_ids.append(int(results[0].value.episode_id))
        for _ in range(2):
            step = await libero_runtime.gateway.policy_step([session_id], policy)
            assert not isinstance(step[0], Err), step[0]
    assert episode_ids == sorted(set(episode_ids)), "episode_id must increase"
    assert episode_ids[1] > episode_ids[0]
    print(f"[M4-assert34] episode_ids={episode_ids}")

    # Assertion 3's "rejected after termination": run the episode to truncation, then step again.
    max_steps = int(libero_runtime.config.env_config["max_episode_steps"])
    chunk = int(libero_runtime.config.env_config.get("chunk_size", 10))
    await libero_runtime.gateway.reset([session_id], ResetSpec(seed=3))
    finished = False
    for _ in range(max_steps // max(1, chunk) + 2):
        results = await libero_runtime.gateway.policy_step([session_id], policy)
        if isinstance(results[0], Err):
            assert results[0].error.code is ErrorCode.EPISODE_TERMINATED
            finished = True
            break
        if results[0].value.terminated or results[0].value.truncated:
            after = await libero_runtime.gateway.policy_step([session_id], policy)
            assert isinstance(after[0], Err)
            assert after[0].error.code is ErrorCode.EPISODE_TERMINATED
            finished = True
            break
    assert finished, "episode never reached a terminal state"
    revived = await libero_runtime.gateway.reset([session_id], ResetSpec(seed=4))
    assert not isinstance(revived[0], Err), revived[0]
    assert int(revived[0].value.episode_id) > episode_ids[-1]


async def test_assertion_5_extension_call_matches_legacy(
    libero_runtime: Any, libero_session: Any
) -> None:
    """Assertion 5: the four LIBERO privileged methods' structure matches the legacy ``LiberoEnvClient``.

    Args:
        libero_runtime: Real runtime fixture.
        libero_session: This test case's session.
    """
    gateway = libero_runtime.gateway
    session_id = libero_session

    # By design, EXTENSION_CALL does not require a prior reset; calling
    # get_camera_meta / cached_image before reset is expected to be valid.
    meta_before = await gateway.extension_call(
        [session_id], "libero", "get_camera_meta", {"height": 256, "width": 256}
    )
    assert not isinstance(meta_before[0], Err), meta_before[0]

    await gateway.reset([session_id], ResetSpec(seed=1))

    meta = (
        await gateway.extension_call(
            [session_id], "libero", "get_camera_meta", {"height": 256, "width": 256}
        )
    )[0]
    assert not isinstance(meta, Err), meta
    assert set(meta.value) == {
        "camera_name",
        "height",
        "width",
        "intrinsic_K",
        "extrinsic_cam2world",
        "depth_near",
        "depth_far",
    }
    intrinsic = np.asarray(meta.value["intrinsic_K"], dtype=np.float64)
    extrinsic = np.asarray(meta.value["extrinsic_cam2world"], dtype=np.float64)
    assert intrinsic.shape == (3, 3)
    assert extrinsic.shape == (4, 4)
    assert meta.value["depth_far"] > meta.value["depth_near"] > 0.0
    print(
        f"[M4-assert5] camera_meta near/far={meta.value['depth_near']:.4f}/"
        f"{meta.value['depth_far']:.4f}"
    )

    rendered = (
        await gateway.extension_call(
            [session_id],
            "libero",
            "render_camera",
            {"camera_name": "agentview", "height": 128, "width": 128, "depth": True},
        )
    )[0]
    assert not isinstance(rendered, Err), rendered
    image = payload_module.decode_payload(rendered.value["image"])
    assert image.shape == (128, 128, 3) and image.dtype == np.uint8
    depth = payload_module.decode_payload(rendered.value["depth"])
    assert depth.shape == (128, 128)
    print(f"[M4-assert5] render_camera rgb={image.shape} depth={depth.shape}")

    cached = (await gateway.extension_call([session_id], "libero", "cached_image", {}))[
        0
    ]
    assert not isinstance(cached, Err), cached
    assert cached.value["available"] is True
    cached_image = payload_module.decode_payload(cached.value["image"])
    assert cached_image.dtype == np.uint8
    assert cached_image.shape == tuple(cached.value["shape"])

    contacts = (
        await gateway.extension_call(
            [session_id],
            "libero",
            "privileged_contacts",
            {"include_all_contacts": False, "max_contacts": 64},
        )
    )[0]
    assert not isinstance(contacts, Err), contacts
    report = contacts.value
    assert report["available"] is True, report.get("reason")
    # Field set from legacy robots/libero/privileged_sensors.py::collect_privileged_contacts.
    assert {
        "available",
        "status",
        "source",
        "real_world_analogue",
        "current_state_only",
        "trajectory_collision_certificate",
        "total_contact_count",
        "robot_contact_count",
        "returned_contact_count",
        "truncated",
        "robot_geom_count",
        "force_available",
        "contacts",
    } <= set(report)
    assert report["current_state_only"] is True
    assert report["trajectory_collision_certificate"] is False
    assert report["robot_geom_count"] > 0
    for entry in report["contacts"]:
        assert {
            "contact_index",
            "geom1",
            "geom2",
            "involves_robot",
            "distance_m",
            "position_world",
            "normal_world",
        } <= set(entry)
        assert entry["involves_robot"] is True
    print(
        f"[M4-assert5] contacts total={report['total_contact_count']} "
        f"robot={report['robot_contact_count']} "
        f"geoms={report['robot_geom_count']} force={report['force_available']}"
    )

    unsupported = (
        await gateway.extension_call([session_id], "libero", "teleport", {})
    )[0]
    assert isinstance(unsupported, Err)
    assert unsupported.error.code is ErrorCode.UNSUPPORTED_EXTENSION


async def test_payload_oversize_counters_are_visible(
    libero_runtime: Any, libero_session: Any
) -> None:
    """Record the payload size and oversize count for a real 256x256x3 dual-camera PNG.

    This is not a gate, it's a **data collection point**: check the numbers here before
    tuning ``inline_threshold_bytes``; don't adjust the threshold by feel.

    Args:
        libero_runtime: Real runtime fixture.
        libero_session: This test case's session.
    """
    session_id = libero_session
    results = await libero_runtime.gateway.reset([session_id], ResetSpec(seed=1))
    observation = results[0].value.observation
    sizes = {
        "main_image": len(observation.main_image.data),
        "wrist_image": len(observation.wrist_image.data),
    }
    stats = payload_module.stats()
    print(
        f"[M4-payload] main={sizes['main_image']}B wrist={sizes['wrist_image']}B "
        f"threshold={payload_module.INLINE_THRESHOLD_BYTES}B "
        f"encoded={stats.encoded_count}/{stats.encoded_bytes}B "
        f"oversize={stats.oversize_count}/{stats.oversize_bytes}B"
    )
    assert sizes["main_image"] > 0 and sizes["wrist_image"] > 0
    assert (
        sizes["main_image"] + sizes["wrist_image"]
        < payload_module.REQUEST_PAYLOAD_LIMIT_BYTES
    )
