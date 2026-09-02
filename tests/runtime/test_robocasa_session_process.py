# Copyright (c) 2026 Zetta Contributors
"""``robots/robocasa/session_process.py`` -- process-level isolation (Path B).

Covers runtime comparison test design §0.2's fix: when a single Ray env-worker
rank holds more than one ``RoboCasaSession`` (``pool_size > 1``), each slot's
session must run in its own OS subprocess instead of sharing this process's
thread pool with the others, because real py-spy evidence (2026-08-17,
``PickPlaceCounterToStove``) showed two threads genuinely concurrently inside
native robosuite/MuJoCo/EGL calls during ``reset()``, producing either
``EGL_BAD_ACCESS`` crashes (4/10 observed) or full deadlocks (prior ``OpenDrawer``
investigation) depending on scheduling luck.

The real RoboCasa/robosuite simulator is not installed in this environment, so
these tests use ``_FakeRoboCasaSession`` (a module-level ``RoboCasaSession``
subclass) instead of the real one. It must be importable by module path rather
than a local closure or a ``pytest.MonkeyPatch``-applied stub, because
``multiprocessing``'s ``spawn`` start method pickles the child process's target
and keyword arguments by reference and re-imports everything fresh in a brand
new interpreter -- a monkeypatch applied in the parent test process (the pattern
``tests/runtime/test_robocasa_current_backend.py`` uses for the non-isolated
path) never reaches the child.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from robots.robocasa import session_core
from robots.robocasa.session_process import (
    RemoteRoboCasaSession,
    RemoteSessionCrashed,
    spawn_robocasa_subprocess,
)
from rollout_runtime.api.enums import ErrorCode
from rollout_runtime.api.errors import RuntimeApiError
from rollout_runtime.api.messages import EnvSpecMsg, ResetSpec
from rollout_runtime.backends.robocasa_current import (
    RobocasaCurrentConfig,
    RobocasaCurrentCore,
)


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


class _FakeRoboCasaSession(session_core.RoboCasaSession):
    """``RoboCasaSession`` with ``_ensure_environment`` stubbed to a fake env.

    Must stay at module level (see module docstring): the child process spawned
    by ``spawn_robocasa_subprocess(session_factory=_FakeRoboCasaSession)``
    re-imports this class fresh by its ``module:qualname`` pickle reference, so
    the override lives on the class itself rather than being monkeypatched onto
    the real ``RoboCasaSession`` from the parent test process.
    """

    def _ensure_environment(self, task: str, split: str) -> None:  # noqa: D102
        self.env = _FakeEnv()
        self.identity = (task, split)


class _CrashingRoboCasaSession(session_core.RoboCasaSession):
    """Raises inside ``reset`` to exercise the child-crash/error-reporting path."""

    def reset(self, payload: dict[str, Any]) -> dict[str, Any]:  # noqa: D102
        raise RuntimeError("simulated native crash during reset")


class _KillsProcessRoboCasaSession(session_core.RoboCasaSession):
    """Hard-exits the child process to simulate a segfault-style crash."""

    def reset(self, payload: dict[str, Any]) -> dict[str, Any]:  # noqa: D102
        import os

        os._exit(1)


@pytest.fixture
def remote_session():
    """Spawn a ``RemoteRoboCasaSession`` against the fake env; close it after."""
    remote = spawn_robocasa_subprocess(
        camera_size=4,
        max_steps=1000,
        cold_reset_lock=None,
        require_isolated_renderer=False,
        session_factory=_FakeRoboCasaSession,
    )
    try:
        yield remote
    finally:
        remote.close_environment()


def _spec(**overrides: Any) -> EnvSpecMsg:
    config: dict[str, Any] = {
        "task": "SlideDishwasherRack",
        "require_isolated_renderer": False,
        "process_isolation": True,
    }
    config.update(overrides)
    return EnvSpecMsg(env_family="robocasa", env_config=config, pool_size=1)


def test_config_accepts_process_isolation_flag_and_defaults_to_false() -> None:
    default = RobocasaCurrentConfig.from_mapping({"task": "X"})
    assert default.process_isolation is False

    enabled = RobocasaCurrentConfig.from_mapping(
        {"task": "X", "process_isolation": True}
    )
    assert enabled.process_isolation is True


def test_spawn_and_reset_round_trip_through_the_subprocess(remote_session) -> None:
    result = remote_session.reset(
        {"task": "SlideDishwasherRack", "split": "target", "seed": 5}
    )
    assert result["observation"]["state"]["state.end_effector_position_relative"] == [
        pytest.approx(0.1),
        pytest.approx(0.2),
        pytest.approx(0.3),
    ]
    assert remote_session.is_alive is True


def test_execute_chunk_round_trip_through_the_subprocess(remote_session) -> None:
    remote_session.reset({"task": "SlideDishwasherRack", "split": "target", "seed": 1})
    result = remote_session.execute_chunk(
        {
            "actions": [[0.0] * 12, [0.0] * 12, [0.0] * 12],
            "critic_rules": [],
            "interrupt_on_proposal": True,
            "capture_event_images": False,
            "enable_task_program": False,
        }
    )
    assert result["executed_horizon"] == 3
    assert result["terminated"] is True


def test_observe_encoded_returns_jpeg_quantized_arrays_not_full_resolution(
    remote_session,
) -> None:
    """The child must JPEG-quantize (D9 forbids it from also PNG-encoding, see
    ``session_process.py``'s module docstring: PNG transport encoding needs
    ``rollout_runtime.core.payload``, which ``robots/robocasa/**`` may not
    import except through ``run_rollout.py``). ``RobocasaCurrentCore.
    _encode_camera`` applies the PNG step back in the parent process.
    """
    remote_session.reset({"task": "SlideDishwasherRack", "split": "target", "seed": 1})
    encoded = remote_session.observe_encoded(
        camera_keys=[
            "video.robot0_agentview_left",
            "video.robot0_agentview_right",
            "video.robot0_eye_in_hand",
        ]
    )
    main = encoded["images"]["video.robot0_agentview_left"]
    # Still a raw uint8 HxWx3 array (not an InlineBytes/PNG payload), but
    # already JPEG-quantized -- identical pixels to what the non-isolated path
    # produces before its own PNG transport-encoding step.
    assert isinstance(main, np.ndarray)
    assert main.dtype == np.uint8
    expected = session_core.jpeg_lossy_rgb_frame(np.zeros((4, 4, 3), dtype=np.uint8))
    np.testing.assert_array_equal(main, expected)
    assert encoded["step_index"] == 0
    assert encoded["attestation"]["official_success"] is False
    assert encoded["task_descriptions"] == ["move the pan"]


def test_snapshot_and_finalize_round_trip_through_the_subprocess(
    remote_session,
) -> None:
    remote_session.reset({"task": "SlideDishwasherRack", "split": "target", "seed": 1})
    snapshot = remote_session.snapshot(include_images=False)
    assert snapshot["task"] == "SlideDishwasherRack"
    finalized = remote_session.finalize_episode_artifacts()
    assert finalized["finalized"] is True


def test_close_environment_terminates_the_child_process(remote_session) -> None:
    remote_session.reset({"task": "SlideDishwasherRack", "split": "target", "seed": 1})
    pid = remote_session._process.pid
    remote_session.close_environment()
    assert remote_session.is_alive is False
    # Calling it again must be a no-op, not raise.
    remote_session.close_environment()
    import os

    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)


def test_calling_a_closed_remote_session_raises_crashed() -> None:
    remote = spawn_robocasa_subprocess(
        camera_size=4,
        max_steps=1000,
        cold_reset_lock=None,
        require_isolated_renderer=False,
        session_factory=_FakeRoboCasaSession,
    )
    remote.close_environment()
    with pytest.raises(RemoteSessionCrashed):
        remote.reset({"task": "SlideDishwasherRack", "split": "target", "seed": 1})


def test_an_exception_inside_the_child_surfaces_as_remote_session_crashed() -> None:
    remote = spawn_robocasa_subprocess(
        camera_size=4,
        max_steps=1000,
        cold_reset_lock=None,
        require_isolated_renderer=False,
        session_factory=_CrashingRoboCasaSession,
    )
    try:
        with pytest.raises(RemoteSessionCrashed) as excinfo:
            remote.reset({"task": "X", "split": "target", "seed": 1})
        assert "simulated native crash during reset" in str(excinfo.value)
        # The child process itself must still be alive: a normal Python
        # exception inside one RPC must not kill the RPC loop (a failed
        # execute_chunk on episode N must not prevent the next reset from
        # reaching the same warm session).
        assert remote.is_alive is True
    finally:
        remote.close_environment()


def test_a_dead_child_process_surfaces_as_remote_session_crashed_not_a_hang() -> None:
    """This is the actual failure mode Path B must turn into a clean error.

    A native crash (segfault, unhandled EGL abort, etc.) kills the OS process
    outright rather than raising a catchable Python exception; the parent must
    detect that and raise promptly instead of hanging the caller's
    ``asyncio.to_thread`` slot (or, worse, the whole rank) the way the
    historical Python-thread-based deadlock did.
    """
    remote = spawn_robocasa_subprocess(
        camera_size=4,
        max_steps=1000,
        cold_reset_lock=None,
        require_isolated_renderer=False,
        session_factory=_KillsProcessRoboCasaSession,
    )
    try:
        with pytest.raises(RemoteSessionCrashed):
            remote.reset({"task": "X", "split": "target", "seed": 1})
        assert remote.is_alive is False
    finally:
        remote.close_environment()


def test_robocasa_current_core_builds_remote_sessions_when_process_isolation_is_on() -> (
    None
):
    core = RobocasaCurrentCore()
    core.build(_spec(), num_envs=1, seed_offset=0)
    try:
        assert isinstance(core._slots[0].session, RemoteRoboCasaSession)
    finally:
        core.close()


def test_robocasa_current_core_default_still_builds_in_process_sessions() -> None:
    """Regression guard: the default config path must never change.

    ``process_isolation`` defaults to ``False``; omitting it from ``env_config``
    must keep building a same-process ``RoboCasaSession``, exactly like before
    this feature existed (``test_robocasa_current_backend.py`` covers this in
    depth; this is a narrow cross-check that the two backends stay switchable
    behind one flag rather than accidentally becoming two divergent code paths).
    """
    core = RobocasaCurrentCore()
    core.build(_spec(process_isolation=False), num_envs=1, seed_offset=0)
    try:
        assert isinstance(core._slots[0].session, session_core.RoboCasaSession)
        assert not isinstance(core._slots[0].session, RemoteRoboCasaSession)
    finally:
        core.close()


def test_robocasa_current_core_end_to_end_reset_and_chunk_step_with_isolation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The full ``RobocasaCurrentCore`` surface must behave the same whether or
    not a slot's session lives in a subprocess -- this is the point of the
    ``observe_encoded``/``getattr`` branch in ``_observation``.
    """

    def _factory_build(
        self, env_spec, *, num_envs, seed_offset=0, total_num_processes=1
    ):  # type: ignore[no-untyped-def]
        from rollout_runtime.backends.robocasa_current import (
            RobocasaCurrentConfig as _Config,
        )
        from rollout_runtime.backends.robocasa_current import (
            _RobocasaSlot,
        )

        self.config = _Config.from_mapping(env_spec.env_config)
        self.env_spec = env_spec
        self.seed_offset = seed_offset
        self._slots = [
            _RobocasaSlot(
                session=spawn_robocasa_subprocess(
                    camera_size=self.config.camera_size,
                    max_steps=self.config.max_steps,
                    cold_reset_lock=self.config.cold_reset_lock,
                    require_isolated_renderer=self.config.require_isolated_renderer,
                    session_factory=_FakeRoboCasaSession,
                )
            )
            for _ in range(num_envs)
        ]
        del total_num_processes

    monkeypatch.setattr(RobocasaCurrentCore, "build", _factory_build)
    core = RobocasaCurrentCore()
    core.build(_spec(), num_envs=1, seed_offset=0)
    try:
        observations = core.reset([0], ResetSpec(seed=5))
        obs = observations[0]
        assert obs.main_image is not None
        assert obs.wrist_image is not None
        assert len(obs.extra_view_images) == 1
        assert obs.instruction == "move the pan"
        assert obs.extras["raw_state"][
            "state.end_effector_position_relative"
        ] == pytest.approx([0.1, 0.2, 0.3])

        outcome = core.chunk_step([0], [np.zeros((3, 12), dtype=np.float32)])[0]
        assert outcome.executed_horizon == 3
        assert outcome.terminated is True
        assert outcome.observation is not None
        assert outcome.observation.main_image is not None
    finally:
        core.close()


def test_slot_out_of_range_is_invalid_argument_when_isolated() -> None:
    core = RobocasaCurrentCore()
    core.build(_spec(), num_envs=1, seed_offset=0)
    try:
        with pytest.raises(RuntimeApiError) as excinfo:
            core.observe([5])
        assert excinfo.value.info.code is ErrorCode.INVALID_ARGUMENT
    finally:
        core.close()
