"""Normalization and libero adapter shape (``test_env_family_normalization.py``).

This file does **not** run the simulator: `.venv-runtime` has no mujoco /
robosuite / rlinf, so a minimal ``rlinf`` stub is used to pin down the three
kinds of family divergence in ``LiberoEnvCore``. Real-hardware behavior is
cross-checked by ``test_extension_call.py`` (``@pytest.mark.remote``) on a
configured GPU host.

Covers three of the family-divergence axes:

1. ``chunk_step`` return length: both ``per_step`` (libero) and ``final_only``
   (robotwin-style) forms come out of the single ``normalize_chunk_outcome``
   exit point;
2. ``reset`` signature: libero uses ``reset(env_idx, reset_state_ids)``, and
   the reset state id formula must match the legacy
   ``first_id + seed % trials`` exactly (a prerequisite for parity);
3. action preprocessing: must go through rlinf's ``prepare_actions`` family
   branch, with correct ``env_type`` / ``model_type`` / shapes.

Also covers the four privileged extensions (structure and rejection of
undeclared methods) and the capability table.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any

import numpy as np
import pytest

from rollout_runtime.api.enums import ErrorCode
from rollout_runtime.api.errors import RuntimeApiError
from rollout_runtime.api.ids import EpisodeId, SessionId
from rollout_runtime.api.messages import EnvSpecMsg, Observation, ResetSpec
from rollout_runtime.core import payload as payload_module
from rollout_runtime.core.env_execution import (
    LOCKSTEP_VECTOR_FORM,
    PER_SLOT_FORM,
    EnvFamilyBehavior,
    normalize_chunk_outcome,
)
from rollout_runtime.core.env_registry import (
    ENV_FAMILY_BEHAVIORS,
    LIBERO_ENV_FAMILY,
    LIBERO_EXTENSIONS,
    behavior_for,
    capability_from_behavior,
    get_env_family,
)
from tests.runtime.libero_stub import (
    ACTION_DIM,
    IMAGE_SIZE,
    STATE_DIM,
    TRIALS_PER_TASK,
    stub_rlinf,
)
from tests.runtime.libero_stub import (
    PREPARE_ACTION_CALLS as _PREPARE_ACTION_CALLS,
)
from tests.runtime.libero_stub import (
    StubInnerEnv as _StubInnerEnv,
)
from tests.runtime.libero_stub import (
    StubLiberoEnv as _StubLiberoEnv,
)

__all__ = ["stub_rlinf"]


def _build_core(env_config: dict[str, Any] | None = None, *, pool_size: int = 1) -> Any:
    """Build a ``LiberoEnvCore`` that has already been ``build``-ed.

    Args:
        env_config: family-private config overrides.
        pool_size: number of slots in the pool.

    Returns:
        ``LiberoEnvCore``.
    """
    from rollout_runtime.backends.rlinf_env import LiberoEnvCore

    config = {
        "task_suite_name": "libero_10",
        "task_id": 0,
        "camera_height": IMAGE_SIZE,
        "camera_width": IMAGE_SIZE,
        "action_dim": ACTION_DIM,
        "chunk_size": 4,
        # Unit tests use a simulator stub and must not depend on live host
        # NVML state. GPU-guard tests opt back in explicitly below.
        "gpu_memory_reserve_mib": 0,
        **(env_config or {}),
    }
    spec = EnvSpecMsg(
        env_family=LIBERO_ENV_FAMILY, env_config=config, pool_size=pool_size
    )
    core = LiberoEnvCore()
    core.build(spec, num_envs=pool_size, seed_offset=0, total_num_processes=1)
    return core


def _observation(step_index: int) -> Observation:
    """Build a minimal ``Observation``.

    Args:
        step_index: step number.

    Returns:
        ``Observation``.
    """
    return Observation(
        session_id=SessionId("sess-x"),
        episode_id=EpisodeId(1),
        step_index=step_index,
        main_image=payload_module.encode_image(np.zeros((2, 2, 3), dtype=np.uint8)),
        state=[0.0],
        instruction="stub",
    )


# ---------------------------------------------------- Divergence 2: obs_list length


def test_per_step_family_keeps_one_record_per_executed_step() -> None:
    """``per_step`` family: ``per_step`` has the same length as the actual number of executed steps, and does not include per-frame images by default."""
    behavior = behavior_for(LIBERO_ENV_FAMILY)
    frames = [_observation(index) for index in range(1, 4)]
    outcome = normalize_chunk_outcome(
        behavior=behavior,
        final_observation=frames[-1],
        step_observations=frames,
        rewards=[0.0, 0.0, 1.0],
        terminations=[False, False, True],
        truncations=[False, False, False],
        requested_horizon=8,
    )
    assert outcome.per_step_obs_available is True
    assert outcome.per_step is not None
    assert [record.step_index for record in outcome.per_step] == [1, 2, 3]
    # Payload budget: per-frame observations are not included in per_step by default (a size threshold applies).
    assert all(record.observation is None for record in outcome.per_step)
    assert outcome.executed_horizon == 3
    assert outcome.info["requested_horizon"] == 8
    assert outcome.reward == pytest.approx(1.0)
    assert outcome.terminated is True
    assert outcome.info["step_observations_included"] is False


def test_per_step_family_can_opt_into_frames() -> None:
    """When ``include_step_observations`` is enabled, per-frame images are included (equivalent to legacy's return_all_frames)."""
    frames = [_observation(1), _observation(2)]
    outcome = normalize_chunk_outcome(
        behavior=behavior_for(LIBERO_ENV_FAMILY),
        final_observation=frames[-1],
        step_observations=frames,
        rewards=[0.0, 0.0],
        terminations=[False, False],
        truncations=[False, False],
        requested_horizon=2,
        include_step_observations=True,
    )
    assert outcome.per_step is not None
    assert all(record.observation is not None for record in outcome.per_step)
    assert outcome.info["step_observations_included"] is True


def test_final_only_family_reports_no_per_step_frames() -> None:
    """``final_only`` family (robotwin-style): only the final frame, ``per_step`` is ``None``.

    This is the reason this normalization exists: a robotwin submission
    returns only 1 obs for the entire chunk, so upstream code cannot treat
    libero's chunk length as a universal contract.
    """
    behavior = behavior_for("robotwin")
    assert behavior.chunk_obs_layout == "final_only"
    outcome = normalize_chunk_outcome(
        behavior=behavior,
        final_observation=_observation(4),
        step_observations=None,
        rewards=[0.0, 0.0, 0.0, 1.0],
        terminations=[False, False, False, True],
        truncations=[False, False, False, False],
        requested_horizon=4,
    )
    assert outcome.per_step_obs_available is False
    assert outcome.per_step is None
    assert outcome.executed_horizon == 4
    assert outcome.observation is not None
    assert outcome.observation.step_index == 4


@pytest.mark.parametrize(
    ("family", "step_observations", "message"),
    [
        (LIBERO_ENV_FAMILY, None, "passed no step_observations"),
        ("robotwin", [_observation(1)], "final_only"),
    ],
)
def test_layout_mismatch_is_rejected(
    family: str, step_observations: list[Observation] | None, message: str
) -> None:
    """When the declared and actual ``obs_list`` shape contradict each other, an error must be raised instead of silently producing a wrong contract.

    Args:
        family: family name.
        step_observations: the per-step observations passed in.
        message: expected fragment in the exception message.
    """
    with pytest.raises(ValueError, match=message):
        normalize_chunk_outcome(
            behavior=behavior_for(family),
            final_observation=_observation(1),
            step_observations=step_observations,
            rewards=[0.0],
            terminations=[False],
            truncations=[False],
            requested_horizon=1,
        )


def test_per_step_lengths_must_agree() -> None:
    """Immediate error when the three per-step sequence lengths disagree."""
    with pytest.raises(ValueError, match="per-step lengths disagree"):
        normalize_chunk_outcome(
            behavior=behavior_for(LIBERO_ENV_FAMILY),
            final_observation=_observation(1),
            step_observations=[_observation(1)],
            rewards=[0.0],
            terminations=[False, False],
            truncations=[False],
            requested_horizon=1,
        )


def test_behavior_table_rejects_unknown_axis_values() -> None:
    """Enum fields of the divergence axes only accept confirmed values."""
    with pytest.raises(ValueError, match="chunk_obs_layout"):
        EnvFamilyBehavior(
            env_family="x",
            env_type="x",
            reset_signature="seed_options",
            chunk_obs_layout="sometimes",
            action_layout="numpy_env_chunk_dim",
            device_kind="cpu_subproc",
        )


def test_declared_families_cover_the_five_reset_signatures() -> None:
    """The declared table covers the four live reset signatures."""
    signatures = {
        behavior.reset_signature for behavior in ENV_FAMILY_BEHAVIORS.values()
    }
    assert signatures == {
        "env_idx_reset_state_ids",  # libero
        "env_idx_env_seeds",  # robotwin
        "task_seed_split_payload",  # RoboCasaSession (runtime v3 design §4)
        "seed_options",  # maniskill
    }
    # maniskill is GPU-batched (a device-kind divergence), libero is a CPU
    # subprocess: **do not** assign it an accelerator.
    # RoboCasaSession is single-process numpy (not a batched GPU tensor), but
    # rendering forces MUJOCO_GL=egl, so it genuinely needs a GPU
    # (needs_accelerator_override, see env_registry.py).
    assert behavior_for("maniskill").needs_accelerator is True
    assert behavior_for(LIBERO_ENV_FAMILY).needs_accelerator is False
    assert behavior_for("robocasa").needs_accelerator is True
    # robotwin is a CPU subprocess pool too, but SAPIEN will not build a
    # renderer without a GPU, so it carries the same explicit override.
    assert behavior_for("robotwin").needs_accelerator is True


def test_robotwin_is_the_only_final_only_family() -> None:
    """``final_only`` exists for robotwin; that fact must stay in the table.

    ``normalize_chunk_outcome``'s whole reason to exist is that one family
    submits a chunk and sees only its last frame. If this list ever comes back
    empty, the normalization has lost its only real caller and the guard in
    ``normalize_chunk_outcome`` is no longer exercised by anything but the
    ``fake`` backend's impersonation.
    """
    assert behavior_for("robotwin").chunk_obs_layout == "final_only"
    assert [
        name
        for name, behavior in ENV_FAMILY_BEHAVIORS.items()
        if behavior.chunk_obs_layout == "final_only"
    ] == ["robotwin"]


def test_every_declared_family_now_ships_an_adapter() -> None:
    """No family is declaration-only any more, robotwin included.

    This is the inverse of the assertion this file used to carry: robotwin was
    declared without an adapter, and ``get_env_family`` was expected to raise
    ``UNSUPPORTED_ENV_SPEC`` with ``declared=True``. ``rlinf_robotwin.py``
    closes that gap, so the expectation flips -- every declared family must now
    resolve to a registered adapter.
    """
    from rollout_runtime.backends import ENV_BACKENDS, register_env_family_for

    for name in ENV_FAMILY_BEHAVIORS:
        assert name in ENV_BACKENDS, f"{name} is declared but has no backend entry"
        adapter = register_env_family_for(name)
        assert adapter.env_family == name
        assert get_env_family(name) is adapter


def test_unknown_family_still_reports_unsupported_env_spec() -> None:
    """The declaration-only path stays live for a future family."""
    with pytest.raises(RuntimeApiError) as excinfo:
        get_env_family("not_a_family")
    assert excinfo.value.info.code is ErrorCode.UNSUPPORTED_ENV_SPEC
    assert excinfo.value.info.detail["declared"] is False


def test_capability_projection_matches_the_behavior() -> None:
    """capability is a projection of behavior (the Gateway uses it to decide ``UNSUPPORTED_ENV_SPEC``)."""
    capability = capability_from_behavior(behavior_for(LIBERO_ENV_FAMILY))
    assert capability.env_family == LIBERO_ENV_FAMILY
    assert capability.per_step_obs_available is True
    assert capability.supports_reset_state_id is True
    assert capability.needs_accelerator is False
    assert capability.extensions == LIBERO_EXTENSIONS


# ------------------------------------------------- Divergence 1: libero's reset signature


def test_reset_uses_the_libero_signature_and_the_legacy_reset_state_id(
    stub_rlinf: type[_StubLiberoEnv],
) -> None:
    """``reset(env_idx, reset_state_ids)``, with the id formula matching legacy exactly.

    legacy ``robots/libero/env_server.py::make_env`` uses
    ``first_id = sum(trials[:task_id])``, ``rid = first_id + seed % trials[task_id]``.
    Parity depends on this alignment, so it is checked here per task per seed.

    Args:
        stub_rlinf: rlinf stub fixture.
    """
    core = _build_core()
    env = core._slots[0].env  # noqa: SLF001 - asserting family call shape requires inspecting the stub
    assert env.is_start is False, (
        "is_start must be cleared at build time, otherwise LiberoEnv.reset overwrites "
        "the first episode's reset_state_ids"
    )

    first_ids = [0, TRIALS_PER_TASK[0], TRIALS_PER_TASK[0] + TRIALS_PER_TASK[1]]
    for task_id, first_id in enumerate(first_ids):
        for seed in (0, 1, 9):
            core.reset([0], ResetSpec(task_id=task_id, seed=seed))
            call = env.reset_calls[-1]
            assert call["env_idx"] == [0]
            expected = first_id + (seed % TRIALS_PER_TASK[task_id])
            assert call["reset_state_ids"] == [expected]

    # An explicit reset_state_id takes priority over (task_id, seed).
    core.reset([0], ResetSpec(task_id=0, seed=3, reset_state_id=11))
    assert env.reset_calls[-1]["reset_state_ids"] == [11]
    core.close()


def test_reset_rejects_out_of_range_task_id(
    stub_rlinf: type[_StubLiberoEnv],
) -> None:
    """An out-of-range ``task_id`` returns ``INVALID_ARGUMENT`` instead of blowing up later in the simulator.

    Args:
        stub_rlinf: rlinf stub fixture.
    """
    core = _build_core()
    with pytest.raises(RuntimeApiError) as excinfo:
        core.reset([0], ResetSpec(task_id=99, seed=0))
    assert excinfo.value.info.code is ErrorCode.INVALID_ARGUMENT
    core.close()


def test_reset_produces_the_five_key_schema(
    stub_rlinf: type[_StubLiberoEnv],
) -> None:
    """After reset, the observation's dtype / layout are correct (local version of an acceptance assertion).

    Args:
        stub_rlinf: rlinf stub fixture.
    """
    core = _build_core()
    observation = core.reset([0], ResetSpec(task_id=1, seed=2))[0]
    assert observation.main_image is not None
    assert observation.wrist_image is not None
    main = payload_module.decode_payload(observation.main_image)
    assert main.dtype == np.uint8
    assert main.shape == (IMAGE_SIZE, IMAGE_SIZE, 3)
    assert len(observation.state) == STATE_DIM
    assert all(isinstance(value, float) for value in observation.state)
    assert isinstance(observation.instruction, str) and observation.instruction
    assert observation.extras["env_family"] == LIBERO_ENV_FAMILY
    assert observation.extras["task_id"] == 1
    assert observation.step_index == 0
    # observe only reads the cache, it does not touch the environment.
    before = len(core._slots[0].env.step_actions)  # noqa: SLF001
    assert core.observe([0])[0] is observation
    assert len(core._slots[0].env.step_actions) == before  # noqa: SLF001
    core.close()


def test_per_slot_reset_reserves_gpu_memory_only_until_scene_is_ready(
    stub_rlinf: type[_StubLiberoEnv], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Cold resets reserve once per slot while warm resets skip the ledger."""
    from rollout_runtime.backends import rlinf_env

    core = _build_core({"gpu_memory_reserve_mib": 800}, pool_size=2)
    reservations: list[tuple[int, int | None]] = []

    @contextmanager
    def record_reservation(requested_mib: int, *, device: int | None = None):
        reservations.append((requested_mib, device))
        yield

    monkeypatch.setattr(rlinf_env, "reserve_gpu_memory", record_reservation)

    core.reset([0], ResetSpec(task_id=0, seed=0))
    core.reset([0], ResetSpec(task_id=0, seed=1))
    core.reset([1], ResetSpec(task_id=0, seed=2))
    core.reset([1], ResetSpec(task_id=0, seed=3))

    assert [requested for requested, _device in reservations] == [800, 800]
    assert [slot.gpu_scene_ready for slot in core._slots] == [True, True]  # noqa: SLF001
    core.close()


def test_failed_per_slot_cold_reset_reserves_again_on_retry(
    stub_rlinf: type[_StubLiberoEnv], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed first reset must not make the slot look warm."""
    from rollout_runtime.backends import rlinf_env

    core = _build_core({"gpu_memory_reserve_mib": 800})
    reservations = 0

    @contextmanager
    def record_reservation(requested_mib: int, *, device: int | None = None):
        del requested_mib, device
        nonlocal reservations
        reservations += 1
        yield

    monkeypatch.setattr(rlinf_env, "reserve_gpu_memory", record_reservation)
    env = core._slots[0].env  # noqa: SLF001
    original_reset = env.reset
    attempts = 0

    def fail_once(*, env_idx: np.ndarray, reset_state_ids: np.ndarray):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("cold reset failed")
        return original_reset(env_idx=env_idx, reset_state_ids=reset_state_ids)

    monkeypatch.setattr(env, "reset", fail_once)
    with pytest.raises(RuntimeError, match="cold reset failed"):
        core.reset([0], ResetSpec(task_id=0, seed=0))
    assert core._slots[0].gpu_scene_ready is False  # noqa: SLF001

    core.reset([0], ResetSpec(task_id=0, seed=0))
    assert reservations == 2
    assert core._slots[0].gpu_scene_ready is True  # noqa: SLF001
    core.close()


def test_gpu_guard_rejection_is_resource_exhausted_and_slot_stays_cold(
    stub_rlinf: type[_StubLiberoEnv], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Admission rejection is retryable and cannot consume the cold marker."""
    from rollout_runtime.backends import rlinf_env
    from zetta.utils.gpu_memory_guard import GpuMemoryExhausted

    core = _build_core({"gpu_memory_reserve_mib": 800})
    attempts = 0

    @contextmanager
    def reject_once(requested_mib: int, *, device: int | None = None):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise GpuMemoryExhausted(
                device=0,
                requested_mib=requested_mib,
                available_mib=128,
            )
        del device
        yield

    monkeypatch.setattr(rlinf_env, "reserve_gpu_memory", reject_once)
    with pytest.raises(RuntimeApiError) as excinfo:
        core.reset([0], ResetSpec(task_id=0, seed=0))
    assert excinfo.value.info.code is ErrorCode.RESOURCE_EXHAUSTED
    assert excinfo.value.info.detail["resource"] == "gpu_memory"
    assert excinfo.value.info.detail["slot_index"] == 0
    assert core._slots[0].gpu_scene_ready is False  # noqa: SLF001

    core.reset([0], ResetSpec(task_id=0, seed=0))
    assert attempts == 2
    assert core._slots[0].gpu_scene_ready is True  # noqa: SLF001
    core.close()


def test_observe_before_reset_is_rejected(
    stub_rlinf: type[_StubLiberoEnv],
) -> None:
    """``observe`` before reset returns ``SESSION_NOT_READY``.

    Args:
        stub_rlinf: rlinf stub fixture.
    """
    core = _build_core()
    with pytest.raises(RuntimeApiError) as excinfo:
        core.observe([0])
    assert excinfo.value.info.code is ErrorCode.SESSION_NOT_READY
    core.close()


# ------------------------------------------------- Divergence 4: action preprocessing


def test_chunk_step_goes_through_prepare_actions(
    stub_rlinf: type[_StubLiberoEnv],
) -> None:
    """Actions must go through rlinf's ``prepare_actions`` family branch, with correct parameters and shapes.

    Args:
        stub_rlinf: rlinf stub fixture.
    """
    core = _build_core({"action_model_type": "openvla_oft"})
    core.reset([0], ResetSpec(task_id=0, seed=0))
    block = np.tile(
        np.arange(ACTION_DIM, dtype=np.float32) + 1.0, (3, 1)
    )  # [3, action_dim]
    outcome = core.chunk_step([0], [block])[0]

    assert _PREPARE_ACTION_CALLS, "prepare_actions was never called"
    call = _PREPARE_ACTION_CALLS[-1]
    assert call["env_type"] == LIBERO_ENV_FAMILY
    assert call["model_type"] == "openvla_oft"
    assert call["num_action_chunks"] == 3
    assert call["action_dim"] == ACTION_DIM
    # The family branch requires [num_envs, chunk, action_dim].
    assert call["shape"] == (1, 3, ACTION_DIM)

    env = core._slots[0].env  # noqa: SLF001
    assert len(env.step_actions) == 3
    for index, sent in enumerate(env.step_actions):
        assert sent.shape == (1, ACTION_DIM)
        # The stub's preprocessing negates the last dimension -> proof that what's sent to the environment is the **preprocessed** action.
        assert sent[0, -1] == pytest.approx(-block[index, -1])
        assert sent[0, 0] == pytest.approx(block[index, 0])
    assert outcome.executed_horizon == 3
    assert outcome.per_step is not None
    assert [record.step_index for record in outcome.per_step] == [1, 2, 3]
    core.close()


def test_chunk_step_stops_at_the_first_termination(
    stub_rlinf: type[_StubLiberoEnv],
) -> None:
    """Stopping mid-chunk on termination: LIBERO-PRO raises if stepped after termination (same semantics as legacy).

    Args:
        stub_rlinf: rlinf stub fixture.
    """
    stub_rlinf.terminate_at = 2
    core = _build_core()
    core.reset([0], ResetSpec(task_id=0, seed=0))
    block = np.zeros((6, ACTION_DIM), dtype=np.float32)
    outcome = core.chunk_step([0], [block])[0]
    assert outcome.executed_horizon == 2, "must not keep stepping a terminated episode"
    assert outcome.terminated is True
    assert outcome.info["requested_horizon"] == 6
    assert len(core._slots[0].env.step_actions) == 2  # noqa: SLF001
    assert outcome.reward == pytest.approx(1.0)
    core.close()


def test_chunk_step_rejects_wrong_action_shape(
    stub_rlinf: type[_StubLiberoEnv],
) -> None:
    """A wrong action shape returns ``INVALID_ARGUMENT``.

    Args:
        stub_rlinf: rlinf stub fixture.
    """
    core = _build_core()
    core.reset([0], ResetSpec(task_id=0, seed=0))
    with pytest.raises(RuntimeApiError) as excinfo:
        core.chunk_step([0], [np.zeros((3, ACTION_DIM + 1), dtype=np.float32)])
    assert excinfo.value.info.code is ErrorCode.INVALID_ARGUMENT
    core.close()


def test_chunk_step_before_reset_is_rejected(
    stub_rlinf: type[_StubLiberoEnv],
) -> None:
    """Stepping before reset returns ``SESSION_NOT_READY``.

    Args:
        stub_rlinf: rlinf stub fixture.
    """
    core = _build_core()
    with pytest.raises(RuntimeApiError) as excinfo:
        core.chunk_step([0], [np.zeros((1, ACTION_DIM), dtype=np.float32)])
    assert excinfo.value.info.code is ErrorCode.SESSION_NOT_READY
    core.close()


# --------------------------------------------------------- Divergence 5/6: pool and extensions


def test_each_slot_gets_an_independent_libero_env(
    stub_rlinf: type[_StubLiberoEnv],
) -> None:
    """``pool_size=N`` builds N independent ``LiberoEnv`` instances; stepping one slot does not touch others.

    ``LiberoEnv.step`` calls ``self.env.step(actions)`` **without an id**, i.e.
    a single step advances every env in the pool. Runtime slots are mutually
    independent sessions, so it must be one slot per env; vectorized
    coalescing is handled separately by ``SlotGroupCoalescer``.

    Args:
        stub_rlinf: rlinf stub fixture.
    """
    core = _build_core(pool_size=3)
    envs = [slot.env for slot in core._slots]  # noqa: SLF001
    assert len({id(env) for env in envs}) == 3
    assert [env.num_envs for env in envs] == [1, 1, 1]
    assert [env.seed_offset for env in envs] == [0, 1, 2]

    core.reset([0, 1, 2], ResetSpec(task_id=0, seed=0))
    core.chunk_step([1], [np.zeros((2, ACTION_DIM), dtype=np.float32)])
    assert [len(env.step_actions) for env in envs] == [0, 2, 0]
    core.close()
    assert all(env.env.closed for env in envs)


def test_slot_index_outside_the_pool_is_rejected(
    stub_rlinf: type[_StubLiberoEnv],
) -> None:
    """An out-of-range slot returns ``INVALID_ARGUMENT`` (the pool does not grow).

    Args:
        stub_rlinf: rlinf stub fixture.
    """
    core = _build_core()
    with pytest.raises(RuntimeApiError) as excinfo:
        core.reset([1], ResetSpec(seed=0))
    assert excinfo.value.info.code is ErrorCode.INVALID_ARGUMENT
    core.close()


def test_the_four_privileged_extensions_return_legacy_shaped_results(
    stub_rlinf: type[_StubLiberoEnv],
) -> None:
    """All four methods are callable, with a structure aligned to the legacy ``LiberoEnvClient``.

    Args:
        stub_rlinf: rlinf stub fixture.
    """
    core = _build_core()

    # It is valid for get_camera_meta / cached_image to be called before reset.
    meta = core.extension(0, "libero", "get_camera_meta", {"height": 128, "width": 128})
    assert set(meta) == {
        "camera_name",
        "height",
        "width",
        "intrinsic_K",
        "extrinsic_cam2world",
        "depth_near",
        "depth_far",
    }
    assert meta["height"] == 128
    assert core.extension(0, "libero", "cached_image", {}) == {
        "available": False,
        "image": None,
        "shape": [],
    }

    core.reset([0], ResetSpec(task_id=0, seed=0))
    cached = core.extension(0, "libero", "cached_image", {})
    assert cached["available"] is True
    assert cached["shape"] == [IMAGE_SIZE, IMAGE_SIZE, 3]
    assert payload_module.decode_payload(cached["image"]).dtype == np.uint8

    rendered = core.extension(
        0, "libero", "render_camera", {"height": 16, "width": 16, "depth": True}
    )
    assert rendered["available"] is True
    assert (rendered["height"], rendered["width"]) == (16, 16)
    image = payload_module.decode_payload(rendered["image"])
    assert image.shape == (16, 16, 3) and image.dtype == np.uint8
    depth = payload_module.decode_payload(rendered["depth"])
    assert depth.shape == (16, 16) and depth.dtype == np.float32

    contacts = core.extension(
        0, "libero", "privileged_contacts", {"include_all_contacts": True}
    )
    assert contacts["available"] is True
    assert contacts["status"] == "ok"
    assert contacts["source"] == "privileged_mujoco_contact_proxy"
    assert contacts["current_state_only"] is True
    assert contacts["trajectory_collision_certificate"] is False
    assert contacts["total_contact_count"] == 2
    assert contacts["robot_contact_count"] == 1
    assert contacts["returned_contact_count"] == 2
    entry = contacts["contacts"][0]
    assert entry["geom1"] == "robot0_link0"
    assert entry["involves_robot"] is True
    assert entry["normal_world"] == [0.0, 1.0, 2.0]
    assert entry["position_world"] == [0.1, 0.2, 0.3]

    # By default, only returns contacts involving the robot.
    filtered = core.extension(0, "libero", "privileged_contacts", {})
    assert filtered["returned_contact_count"] == 1
    core.close()


def test_render_extension_seam_does_not_shadow_the_real_render(
    stub_rlinf: type[_StubLiberoEnv],
) -> None:
    """When ``rr_extension`` is not present, ``render`` falls through
    unchanged to the env's own implementation.

    This guarantees the extension pass-through has zero effect on
    simulation semantics.

    Args:
        stub_rlinf: rlinf stub fixture.
    """
    core = _build_core()
    worker = core._slots[0].env.env.workers[0]  # noqa: SLF001
    assert worker.render(mode="rgb_array") == "original-render"
    assert worker.env.render_calls == [{"mode": "rgb_array"}]
    core.close()


def test_render_extension_seam_survives_a_reconfigure(
    stub_rlinf: type[_StubLiberoEnv],
) -> None:
    """The extension is installed on the **class**, so it stays valid even
    after ``reconfigure`` replaces the env instance inside the subprocess.

    This is a real defect hit on a multi-GPU host: rlinf's libero worker
    loop, upon receiving ``"reconfigure"``, calls ``env.close()`` and then
    ``env = OffScreenRenderEnv(**data)``
    (``rlinf/envs/libero/venv.py:156-160``), and ``LiberoEnv._reconfigure``
    always takes this path on a task change. An instance-level forwarding
    layer would silently disappear, with the symptom being
    ``privileged_contacts`` returning ``None``.

    Args:
        stub_rlinf: rlinf stub fixture.
    """
    core = _build_core()
    worker = core._slots[0].env.env.workers[0]  # noqa: SLF001
    # Simulate the worker loop's reconfigure: same class, brand-new
    # instance, **without** calling install_runtime_extensions again.
    worker.env = _StubInnerEnv()
    report = core.extension(0, "libero", "privileged_contacts", {})
    assert isinstance(report, dict)
    assert report["available"] is True, report.get("reason")
    assert report["status"] == "ok"
    core.close()


def test_unknown_extension_is_unsupported_not_a_crash(
    stub_rlinf: type[_StubLiberoEnv],
) -> None:
    """An undeclared method / namespace returns ``UNSUPPORTED_EXTENSION``.

    Args:
        stub_rlinf: rlinf stub fixture.
    """
    core = _build_core()
    for namespace, method in (("libero", "teleport"), ("fake", "ping")):
        with pytest.raises(RuntimeApiError) as excinfo:
            core.extension(0, namespace, method, {})
        assert excinfo.value.info.code is ErrorCode.UNSUPPORTED_EXTENSION
        assert excinfo.value.info.detail["supported"] == sorted(LIBERO_EXTENSIONS)
    core.close()


# ----------------------------------------------------------------- Family configuration


def test_env_config_aliases_and_unknown_keys() -> None:
    """Legacy key names from earlier presets remain usable; a misspelled
    key must raise an error rather than silently creating a new pool."""
    from rollout_runtime.backends.rlinf_env import LiberoEnvConfig

    config = LiberoEnvConfig.from_mapping(
        {
            "suite": "libero_goal",
            "episode_length": 300,
            "image_height": 128,
            "image_width": 64,
        }
    )
    assert config.task_suite_name == "libero_goal"
    assert config.max_episode_steps == 300
    assert (config.camera_height, config.camera_width) == (128, 64)
    assert LiberoEnvConfig.from_mapping({}).gpu_memory_reserve_mib == 800

    with pytest.raises(RuntimeApiError) as excinfo:
        LiberoEnvConfig.from_mapping({"image_hieght": 256})
    assert excinfo.value.info.code is ErrorCode.INVALID_ARGUMENT
    assert excinfo.value.info.detail["unknown_keys"] == ["image_hieght"]


def test_num_steps_wait_must_match_the_hardcoded_warmup() -> None:
    """``num_steps_wait`` can only be declared as 15: the warm-up loop is
    hardcoded in rlinf."""
    from rollout_runtime.backends.rlinf_env import (
        LIBERO_WARMUP_STEPS,
        LiberoEnvConfig,
    )

    assert LIBERO_WARMUP_STEPS == 15
    assert LiberoEnvConfig.from_mapping({"num_steps_wait": 15}).num_steps_wait == 15
    with pytest.raises(RuntimeApiError) as excinfo:
        LiberoEnvConfig.from_mapping({"num_steps_wait": 10})
    assert excinfo.value.info.code is ErrorCode.INVALID_ARGUMENT
    assert "hardcoded" in excinfo.value.info.message


def test_rlinf_cfg_projection_matches_the_legacy_env_server() -> None:
    """The projected rlinf cfg matches the key fields of the legacy
    ``build_env_cfg`` (a prerequisite for parity)."""
    from rollout_runtime.backends.rlinf_env import LiberoEnvConfig

    cfg = LiberoEnvConfig.from_mapping(
        {"task_suite_name": "libero_10", "max_episode_steps": 512}
    ).to_rlinf_cfg()
    assert cfg["env_type"] == "libero"
    assert cfg["is_eval"] is True
    assert cfg["auto_reset"] is False
    assert cfg["ignore_terminations"] is False
    assert cfg["use_fixed_reset_state_ids"] is True
    assert cfg["use_ordered_reset_state_ids"] is True
    assert cfg["specific_reset_id"] is None
    assert cfg["group_size"] == 1
    assert cfg["reset_gripper_open"] is True
    assert cfg["init_params"]["camera_depths"] is True
    assert cfg["init_params"]["ignore_done"] is True
    assert cfg["init_params"]["horizon"] == 512 + 1000


def test_pool_build_rejects_zero_slots(stub_rlinf: type[_StubLiberoEnv]) -> None:
    """``num_envs < 1`` returns ``INVALID_ARGUMENT``.

    Args:
        stub_rlinf: rlinf stub fixture.
    """
    from rollout_runtime.backends.rlinf_env import LiberoEnvCore

    spec = EnvSpecMsg(env_family=LIBERO_ENV_FAMILY, env_config={}, pool_size=1)
    with pytest.raises(RuntimeApiError) as excinfo:
        LiberoEnvCore().build(spec, num_envs=0, seed_offset=0, total_num_processes=1)
    assert excinfo.value.info.code is ErrorCode.INVALID_ARGUMENT


# ------------------------------------------------- The ``lockstep_vector`` form for libero


def test_libero_accepts_the_core_form_key_and_builds_one_vector_env(
    stub_rlinf: type,
) -> None:
    """``core_form`` must be a field of ``LiberoEnvConfig``.

    A defect exposed on GPU: unknown keys in ``env_config`` are always
    rejected (deliberately -- a misspelled key would silently produce a
    new pool), and libero's config class at the time **did not** have a
    ``core_form`` field, so the vector-form preset got
    ``INVALID_ARGUMENT: unknown libero env config keys: ['core_form']``
    directly.
    """
    from rollout_runtime.backends.rlinf_env import LiberoEnvConfig

    assert LiberoEnvConfig.from_mapping({}).core_form == PER_SLOT_FORM
    assert (
        LiberoEnvConfig.from_mapping({"core_form": LOCKSTEP_VECTOR_FORM}).core_form
        == LOCKSTEP_VECTOR_FORM
    )
    core = _build_core({"core_form": LOCKSTEP_VECTOR_FORM}, pool_size=3)
    assert core.core_form == LOCKSTEP_VECTOR_FORM
    # One vector env per pool (vs. per_slot's "one env per slot").
    assert len(core._envs) == 1
    assert core._envs[0].num_envs == 3
    assert [slot.lane_index for slot in core._slots] == [0, 1, 2]
    assert core._envs[0].is_start is False
    core.close()


def test_vector_reset_reserves_once_and_failed_cold_reset_retries(
    stub_rlinf: type[_StubLiberoEnv], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The shared vector scene becomes warm only after reset succeeds."""
    from rollout_runtime.backends import rlinf_env

    core = _build_core(
        {"core_form": LOCKSTEP_VECTOR_FORM, "gpu_memory_reserve_mib": 800},
        pool_size=2,
    )
    reservations = 0

    @contextmanager
    def record_reservation(requested_mib: int, *, device: int | None = None):
        del requested_mib, device
        nonlocal reservations
        reservations += 1
        yield

    monkeypatch.setattr(rlinf_env, "reserve_gpu_memory", record_reservation)
    env = core._envs[0]  # noqa: SLF001
    original_reset = env.reset
    attempts = 0

    def fail_once(*, env_idx: np.ndarray, reset_state_ids: np.ndarray):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("vector cold reset failed")
        return original_reset(env_idx=env_idx, reset_state_ids=reset_state_ids)

    monkeypatch.setattr(env, "reset", fail_once)
    with pytest.raises(RuntimeError, match="vector cold reset failed"):
        core.reset([0], ResetSpec(task_id=0, seed=0))
    assert core._vector_gpu_scene_ready is False  # noqa: SLF001

    core.reset([0], ResetSpec(task_id=0, seed=0))
    core.reset([1], ResetSpec(task_id=0, seed=1))
    assert reservations == 2
    assert core._vector_gpu_scene_ready is True  # noqa: SLF001
    core.close()


def test_libero_vector_form_coalesces_and_masks_absent_lanes(stub_rlinf: type) -> None:
    """Vector form: one advance covers every lane, and absent lanes are
    carried through by the hold action and counted."""
    core = _build_core({"core_form": LOCKSTEP_VECTOR_FORM}, pool_size=3)
    core.reset([0, 1, 2], ResetSpec(task_id=0, seed=1))
    # reset is **one** call covering all three lanes (libero's reset
    # already supports an env_idx subset).
    assert core._envs[0].reset_calls[-1]["env_idx"] == [0, 1, 2]
    outcomes = core.chunk_step(
        [0, 2], [np.zeros((4, ACTION_DIM), dtype=np.float32)] * 2
    )
    assert [outcome.executed_horizon for outcome in outcomes] == [4, 4]
    # Within one chunk_step, the env was stepped 4 times, each time as a
    # full batch of [3, action_dim].
    assert len(core._envs[0].step_actions) == 4
    assert core._envs[0].step_actions[0].shape == (3, ACTION_DIM)
    assert outcomes[0].info["coalesced_slots"] == [0, 2]
    assert outcomes[0].info["masked_slots"] == [1]
    assert core._slots[1].masked_steps == 4
    assert core.total_masked_steps == 4
    assert core.coalesced_group_count == 1
    # Hold action: all-zero displacement, gripper -1 (reset_gripper_open
    # defaults to true). The stub's prepare_actions negates the last
    # dimension, so the dimension fed to the env is +1.
    masked_row = core._envs[0].step_actions[0][1]
    assert list(masked_row[:6]) == [0.0] * 6
    assert masked_row[-1] == pytest.approx(1.0)
    core.close()


def test_libero_vector_form_refuses_non_standard_variants(stub_rlinf: type) -> None:
    """The vector form cannot stop only one lane, while LIBERO-PRO/PLUS
    raise if stepped after termination -> rejected at build time."""
    with pytest.raises(RuntimeApiError) as excinfo:
        _build_core(
            {"core_form": LOCKSTEP_VECTOR_FORM, "libero_variant": "pro"}, pool_size=2
        )
    assert excinfo.value.info.code is ErrorCode.INVALID_ARGUMENT
    assert "libero_variant='standard'" in excinfo.value.info.message


def test_libero_vector_form_needs_every_lane_reset(stub_rlinf: type) -> None:
    """Masking semantic 1: a lane that has never been reset cannot be
    touched, not even for a single step."""
    core = _build_core({"core_form": LOCKSTEP_VECTOR_FORM}, pool_size=2)
    core.reset([0], ResetSpec(task_id=0, seed=1))
    with pytest.raises(RuntimeApiError) as excinfo:
        core.chunk_step([0], [np.zeros((2, ACTION_DIM), dtype=np.float32)])
    assert excinfo.value.info.code is ErrorCode.SESSION_NOT_READY
    assert excinfo.value.info.detail["unstarted_slots"] == [1]
    core.close()
