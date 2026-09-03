# Copyright (c) 2026 Zetta Contributors
"""The robotwin family: ``final_only`` normalization and the D1/D3 mappings.

``.venv-runtime`` has no ``robotwin`` package or SAPIEN, so a stub env drives
the **real** ``RobotwinEnvCore``. What that pins down locally is exactly the
part that is ours rather than the simulator's: that a submitted chunk produces
no per-step records, that the left/right wrist frames land in the fields the
policy expects, that the open-loop execute horizon truncates instead of
silently over-running, and that the capability table matches the behaviour.

What the stub cannot cover is whether these parameters actually boot RoboTwin;
that is the GPU smoke test's job (``plan/robotwin_s0_findings.md``).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from rollout_runtime.api.enums import ErrorCode
from rollout_runtime.api.errors import RuntimeApiError
from rollout_runtime.api.messages import EnvSpecMsg, ResetSpec
from rollout_runtime.core import payload as payload_module
from rollout_runtime.core.env_execution import LOCKSTEP_VECTOR_FORM, PER_SLOT_FORM
from rollout_runtime.core.env_registry import behavior_for

ACTION_DIM = 14
IMAGE_H = 6
IMAGE_W = 8
MAIN_FILL = 33
LEFT_FILL = 11
RIGHT_FILL = 22


def _frame(fill: int) -> np.ndarray:
    """Build a uniformly filled uint8 RGB frame.

    Args:
        fill: The fill value, used to tell the three views apart.

    Returns:
        An ``IMAGE_H x IMAGE_W x 3`` uint8 array.
    """
    return np.full((IMAGE_H, IMAGE_W, 3), fill, dtype=np.uint8)


class _StubRoboTwinEnv:
    """Stand-in for the vendored ``RoboTwinEnv``.

    Reproduces the parts of the contract the adapter depends on: the
    ``reset(env_idx, env_seeds)`` signature, and a ``chunk_step`` that consumes
    the whole chunk and returns a **length-1** ``obs_list`` alongside
    chunk-shaped reward/flag matrices whose only populated column is the last.
    """

    instances: list[_StubRoboTwinEnv] = []

    def __init__(
        self,
        cfg: Any,
        num_envs: int,
        seed_offset: int,
        total_num_processes: int,
        worker_info: Any,
        record_metrics: bool = True,
    ) -> None:
        """Record construction arguments and register the instance."""
        self.cfg = cfg
        self.num_envs = num_envs
        self.seed_offset = seed_offset
        self.total_num_processes = total_num_processes
        self.is_start = True
        self.reset_calls: list[tuple[Any, Any]] = []
        self.chunk_calls: list[np.ndarray] = []
        self.closed = False
        self.wrist_names = ["left_wrist_image", "right_wrist_image"]
        self.terminate = False
        type(self).instances.append(self)

    def _payload(self) -> dict[str, Any]:
        """Build one observation payload in the family's own layout.

        Returns:
            The 4-key payload plus the additive ``wrist_image_names``.
        """
        fills = {"left_wrist_image": LEFT_FILL, "right_wrist_image": RIGHT_FILL}
        wrist_stack = (
            np.stack([_frame(fills[name]) for name in self.wrist_names])[None, ...]
            if self.wrist_names
            else None
        )
        return {
            "main_images": _frame(MAIN_FILL)[None, ...],
            "wrist_images": wrist_stack,
            # float64 on purpose: this is what RoboTwin really returns, and the
            # adapter must land it as float32 in the Observation.
            "states": np.arange(ACTION_DIM, dtype=np.float64)[None, ...],
            "task_descriptions": ["pick the bottle with the correct arm"],
            "wrist_image_names": list(self.wrist_names),
        }

    def reset(self, env_idx: Any = None, env_seeds: Any = None):
        """Record the reset and return the initial payload."""
        self.reset_calls.append(
            (env_idx, None if env_seeds is None else list(env_seeds))
        )
        return self._payload(), {}

    def chunk_step(self, chunk_actions: Any):
        """Consume the whole chunk and return a single final frame."""
        array = np.asarray(chunk_actions)
        self.chunk_calls.append(array)
        chunk = int(array.shape[1])
        rewards = np.zeros((1, chunk), dtype=np.float32)
        rewards[0, -1] = 1.5
        terminations = np.zeros((1, chunk), dtype=bool)
        truncations = np.zeros((1, chunk), dtype=bool)
        if self.terminate:
            terminations[0, -1] = True
        infos = [{"success": np.array([self.terminate])}]
        return [self._payload()], rewards, terminations, truncations, infos

    def offload(self, clear_cache: bool = True) -> None:
        """Mark the env released."""
        self.closed = True


@pytest.fixture
def robotwin_backend(monkeypatch: pytest.MonkeyPatch):
    """Point the adapter's lazy env lookup at the stub.

    Args:
        monkeypatch: pytest's patcher.

    Returns:
        The backend module.
    """
    import rollout_runtime.backends.rlinf_robotwin as backend_module

    _StubRoboTwinEnv.instances = []
    monkeypatch.setattr(backend_module, "_robotwin_env_class", lambda: _StubRoboTwinEnv)
    return backend_module


def robotwin_spec(pool_size: int = 1, **overrides: Any) -> EnvSpecMsg:
    """Build a robotwin env spec.

    Args:
        pool_size: Pool capacity (unused by the spec itself, kept for parity).
        **overrides: ``env_config`` overrides.

    Returns:
        The env spec.
    """
    config: dict[str, Any] = {
        "task_name": "adjust_bottle",
        "assets_path": "/workspace/RoboTwin",
        "action_dim": ACTION_DIM,
        "chunk_size": 4,
    }
    config.update(overrides)
    return EnvSpecMsg(env_family="robotwin", env_config=config)


def _build(backend_module, *, num_envs: int = 1, **overrides: Any):
    """Build a core against the stub.

    Args:
        backend_module: The adapter module.
        num_envs: Pool size.
        **overrides: ``env_config`` overrides.

    Returns:
        The built core.
    """
    core = backend_module.RobotwinEnvCore()
    core.build(robotwin_spec(**overrides), num_envs=num_envs)
    return core


# ------------------------------------------------------------------ behaviour


def test_declaration_matches_the_measured_family() -> None:
    """The six axes are what was verified against RLinf 9ad44393 on hardware."""
    behavior = behavior_for("robotwin")
    assert behavior.chunk_obs_layout == "final_only"
    assert behavior.per_step_obs_available is False
    assert behavior.reset_signature == "env_idx_env_seeds"
    assert behavior.action_layout == "numpy_env_chunk_dim"
    assert behavior.needs_accelerator is True, "SAPIEN cannot render without a GPU"
    assert behavior.core_forms == frozenset({PER_SLOT_FORM})
    assert behavior.supports_coalescing is False
    assert behavior.max_pool_size == 16


def test_capability_declares_reset_state_ids(robotwin_backend) -> None:
    """RoboTwin's ``env_seeds`` *is* a reset-state selector, so it is declared.

    This is the opposite call from maniskill, where ``reset_state_id`` is
    accepted by the signature but ignored by most tasks. Here the seed fully
    determines the scene, and the campaign protocol's paired same-seed gate
    depends on being able to pin it.
    """
    capability = robotwin_backend.robotwin_env_capability()
    assert capability.env_family == "robotwin"
    assert capability.supports_reset_state_id is True
    assert capability.per_step_obs_available is False
    assert capability.needs_accelerator is True
    assert capability.supports_auto_reset is False
    assert capability.extensions == frozenset()


def test_lockstep_form_is_rejected(robotwin_backend) -> None:
    """Only ``per_slot`` is declared, so asking for a vector pool is an error."""
    core = robotwin_backend.RobotwinEnvCore()
    with pytest.raises(RuntimeApiError) as excinfo:
        core.build(robotwin_spec(core_form=LOCKSTEP_VECTOR_FORM), num_envs=1)
    assert excinfo.value.info.code is ErrorCode.INVALID_ARGUMENT


def test_pool_size_is_capped_at_the_measured_ceiling(robotwin_backend) -> None:
    """The SAPIEN buffer ceiling is a declared limit, not a runtime surprise."""
    core = robotwin_backend.RobotwinEnvCore()
    with pytest.raises(RuntimeApiError) as excinfo:
        core.build(robotwin_spec(), num_envs=17)
    assert excinfo.value.info.code is ErrorCode.INVALID_ARGUMENT
    assert excinfo.value.info.detail["max_pool_size"] == 16


def test_chunk_step_produces_no_per_step_records(robotwin_backend) -> None:
    """``final_only``: one submitted chunk, one frame, no per-step records.

    This is the whole reason ``normalize_chunk_outcome`` exists. ``per_step``
    must be ``None`` -- not an empty list, which downstream code would read as
    "records exist and there were none".
    """
    core = _build(robotwin_backend)
    core.reset([0], ResetSpec(seed=123))
    outcome = core.chunk_step([0], [np.zeros((4, ACTION_DIM), dtype=np.float32)])[0]

    assert outcome.per_step is None
    assert outcome.per_step_obs_available is False
    assert outcome.executed_horizon == 4
    assert outcome.reward == pytest.approx(1.5)
    assert outcome.info["chunk_obs_layout"] == "final_only"
    # The family got the whole chunk in a single call.
    env = _StubRoboTwinEnv.instances[0]
    assert len(env.chunk_calls) == 1
    assert env.chunk_calls[0].shape == (1, 4, ACTION_DIM)


def test_wrist_views_are_split_by_name_not_position(robotwin_backend) -> None:
    """D1: left wrist -> ``wrist_image``, right wrist -> ``extra_view_images[0]``.

    The mapping reads the family's ``wrist_image_names``. Getting it backwards
    would not raise anywhere -- it would just hand the policy a mirrored view --
    so it is pinned by comparing against the exact encoded frames.
    """
    core = _build(robotwin_backend)
    observation = core.reset([0], ResetSpec(seed=1))[0]

    assert observation.main_image == payload_module.encode_image(_frame(MAIN_FILL))
    assert observation.wrist_image == payload_module.encode_image(_frame(LEFT_FILL))
    assert len(observation.extra_view_images) == 1
    assert observation.extra_view_images[0] == payload_module.encode_image(
        _frame(RIGHT_FILL)
    )


def test_wrist_split_follows_a_right_only_configuration(robotwin_backend) -> None:
    """A right-wrist-only stack must not be mistaken for a left wrist.

    Upstream stacks whichever wrists exist, so slice 0 is the right wrist here.
    Position-based mapping would silently put it in ``wrist_image``.
    """
    core = _build(robotwin_backend)
    env = _StubRoboTwinEnv.instances[0]
    env.wrist_names = ["right_wrist_image"]

    observation = core.reset([0], ResetSpec(seed=1))[0]
    assert observation.wrist_image is None
    assert observation.extra_view_images[0] == payload_module.encode_image(
        _frame(RIGHT_FILL)
    )


def test_no_wrist_cameras_yields_no_wrist_views(robotwin_backend) -> None:
    """``collect_wrist_camera: false`` must degrade cleanly, not crash."""
    core = _build(robotwin_backend)
    env = _StubRoboTwinEnv.instances[0]
    env.wrist_names = []

    observation = core.reset([0], ResetSpec(seed=1))[0]
    assert observation.wrist_image is None
    assert observation.extra_view_images == []


def test_state_is_float32_even_though_robotwin_returns_float64(
    robotwin_backend,
) -> None:
    """RoboTwin's ``state`` is float64; the schema digest compares dtypes."""
    core = _build(robotwin_backend)
    observation = core.reset([0], ResetSpec(seed=1))[0]
    assert len(observation.state) == ACTION_DIM
    assert all(isinstance(value, float) for value in observation.state)
    assert observation.state[:3] == [0.0, 1.0, 2.0]


def test_execute_horizon_truncates_the_submitted_chunk(robotwin_backend) -> None:
    """D3: the model may emit 50 actions while only the first N are executed.

    The discard is the configured open-loop replan, so it must be visible in
    the outcome rather than inferred from a length mismatch.
    """
    core = _build(robotwin_backend, execute_horizon=3)
    core.reset([0], ResetSpec(seed=1))
    outcome = core.chunk_step([0], [np.zeros((10, ACTION_DIM), dtype=np.float32)])[0]

    env = _StubRoboTwinEnv.instances[0]
    assert env.chunk_calls[0].shape == (1, 3, ACTION_DIM)
    assert outcome.executed_horizon == 3
    assert outcome.info["requested_horizon"] == 10
    assert outcome.info["discarded_actions"] == 7
    assert core.total_discarded_actions == 7


def test_execute_horizon_leaves_a_short_chunk_alone(robotwin_backend) -> None:
    """A chunk shorter than the horizon is submitted whole, with no discard."""
    core = _build(robotwin_backend, execute_horizon=8)
    core.reset([0], ResetSpec(seed=1))
    outcome = core.chunk_step([0], [np.zeros((2, ACTION_DIM), dtype=np.float32)])[0]
    assert outcome.executed_horizon == 2
    assert "discarded_actions" not in outcome.info


def test_reset_state_id_wins_over_seed(robotwin_backend) -> None:
    """Seed precedence: ``reset_state_id`` > ``seed`` > the family's schedule."""
    core = _build(robotwin_backend)
    env = _StubRoboTwinEnv.instances[0]

    core.reset([0], ResetSpec(seed=7, reset_state_id=4242))
    assert env.reset_calls[-1] == (None, [4242])

    core.reset([0], ResetSpec(seed=7))
    assert env.reset_calls[-1] == (None, [7])

    core.reset([0], ResetSpec())
    assert env.reset_calls[-1] == (None, None)


def test_termination_freezes_the_lane(robotwin_backend) -> None:
    """A terminating chunk sets the flag and stops the lane."""
    core = _build(robotwin_backend)
    core.reset([0], ResetSpec(seed=1))
    _StubRoboTwinEnv.instances[0].terminate = True

    outcome = core.chunk_step([0], [np.zeros((4, ACTION_DIM), dtype=np.float32)])[0]
    assert outcome.terminated is True
    assert outcome.info["success"] is True
    assert core.lane_status([0])[0].terminated is True


def test_wrong_action_width_is_rejected(robotwin_backend) -> None:
    """A 7-dim single-arm chunk reaching a bimanual env must fail loudly."""
    core = _build(robotwin_backend)
    core.reset([0], ResetSpec(seed=1))
    with pytest.raises(RuntimeApiError) as excinfo:
        core.chunk_step([0], [np.zeros((4, 7), dtype=np.float32)])
    assert excinfo.value.info.code is ErrorCode.INVALID_ARGUMENT


def test_chunk_step_before_reset_is_rejected(robotwin_backend) -> None:
    """Stepping an un-reset slot is ``SESSION_NOT_READY``, not a crash."""
    core = _build(robotwin_backend)
    with pytest.raises(RuntimeApiError) as excinfo:
        core.chunk_step([0], [np.zeros((4, ACTION_DIM), dtype=np.float32)])
    assert excinfo.value.info.code is ErrorCode.SESSION_NOT_READY


def test_extensions_are_declaratively_rejected(robotwin_backend) -> None:
    """The family declares no extensions, so every call is refused cleanly."""
    core = _build(robotwin_backend)
    with pytest.raises(RuntimeApiError) as excinfo:
        core.extension(0, "robotwin", "snapshot", {})
    assert excinfo.value.info.code is ErrorCode.UNSUPPORTED_EXTENSION


def test_unknown_config_keys_are_rejected(robotwin_backend) -> None:
    """A typo would otherwise allocate a second SAPIEN pool via the digest."""
    with pytest.raises(RuntimeApiError) as excinfo:
        robotwin_backend.RobotwinEnvConfig.from_mapping(
            {"assets_path": "/x", "taskname": "adjust_bottle"}
        )
    assert excinfo.value.info.code is ErrorCode.INVALID_ARGUMENT
    assert excinfo.value.info.detail["unknown_keys"] == ["taskname"]


def test_missing_assets_path_is_rejected(robotwin_backend) -> None:
    """``assets_path`` has no sensible default and its absence fails late."""
    with pytest.raises(RuntimeApiError):
        robotwin_backend.RobotwinEnvConfig.from_mapping({"task_name": "adjust_bottle"})


def test_task_config_defaults_to_mplib_and_disables_data_collection(
    robotwin_backend,
) -> None:
    """The planner default is mplib, and the family must not write its own data."""
    config = robotwin_backend.RobotwinEnvConfig.from_mapping(
        {"assets_path": "/workspace/RoboTwin"}
    )
    task_config = config.task_config()
    assert task_config["planner_backend"] == "mplib"
    assert task_config["collect_data"] is False
    assert task_config["camera"]["collect_wrist_camera"] is True


def test_pool_builds_one_env_per_slot(robotwin_backend) -> None:
    """``per_slot``: N slots means N single-lane envs, each seeded distinctly."""
    core = _build(robotwin_backend, num_envs=3)
    assert len(_StubRoboTwinEnv.instances) == 3
    assert [env.num_envs for env in _StubRoboTwinEnv.instances] == [1, 1, 1]
    assert [env.seed_offset for env in _StubRoboTwinEnv.instances] == [0, 1, 2]
    # The family's "first reset decides its own state" branch must be off.
    assert all(env.is_start is False for env in _StubRoboTwinEnv.instances)

    core.close()
    # SAPIEN teardown segfaults, so `close` drops references without calling it.
    assert all(env.closed is False for env in _StubRoboTwinEnv.instances)


def test_close_does_not_tear_down_sapien_by_default(robotwin_backend) -> None:
    """A segfaulting teardown would turn `close_sessions` into a worker restart.

    ``VectorEnv.close()`` crashes the process during SAPIEN teardown, and a
    segfault cannot be caught, so the pool releases its references instead and
    lets the subprocesses go when the worker exits.
    """
    core = _build(robotwin_backend)
    core.close()
    assert core.closed is True
    assert _StubRoboTwinEnv.instances[0].closed is False


def test_teardown_on_close_is_available_as_an_escape_hatch(robotwin_backend) -> None:
    """The real teardown stays reachable for a RoboTwin build that fixed it."""
    core = _build(robotwin_backend, teardown_on_close=True)
    core.close()
    assert _StubRoboTwinEnv.instances[0].closed is True
