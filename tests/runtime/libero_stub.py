"""``rlinf`` / LIBERO stub (shared by ``tests/runtime``).

``.venv-runtime`` does not have mujoco / robosuite / rlinf, so a minimal
``rlinf`` stub is used locally to drive the **real** ``LiberoEnvCore``: family
divergences (reset signature, early chunk termination, action preprocessing),
the structure of the five privileged extensions, and the seam shape can all be
pinned down locally, while real-hardware behavior is re-verified by
``test_extension_call.py`` / ``test_legacy_parity.py`` (``@pytest.mark.remote``)
on a configured GPU host.

These stubs previously lived in ``test_env_family_normalization.py``;
``test_zetta_seam.py`` needs the same stub set (the seam also goes through
``LiberoEnvCore``), so it was extracted here to be shared, **with the content
left unchanged verbatim**.
"""

from __future__ import annotations

import importlib.machinery
import sys
import types
from typing import Any

import numpy as np
import pytest

__all__ = [
    "ACTION_DIM",
    "IMAGE_SIZE",
    "PREPARE_ACTION_CALLS",
    "STATE_DIM",
    "TRIALS_PER_TASK",
    "StubInnerEnv",
    "StubLiberoEnv",
    "install_rlinf_stub",
    "stub_module",
]


def _stub_module(name: str) -> types.ModuleType:
    """Create a stub module with a ``__spec__``.

    ``rlinf_bootstrap.ensure_rlinf_importable`` uses
    ``importlib.util.find_spec("rlinf")`` to determine "whether it is already
    importable", and ``find_spec`` raises ``ValueError`` for modules where
    ``__spec__ is None``. So the stub must carry a spec.

    Args:
        name: Module name.

    Returns:
        The stub module.
    """
    module = types.ModuleType(name)
    module.__spec__ = importlib.machinery.ModuleSpec(name, None)
    return module


IMAGE_SIZE = 8
STATE_DIM = 8
ACTION_DIM = 7
TRIALS_PER_TASK = (5, 7, 3)
"""Trials per task for the stub benchmark, used to verify the
``first_id + seed % trials`` segmentation."""


# --------------------------------------------------------------------- rlinf stub


class _StubTaskSuite:
    """Minimal LIBERO benchmark stub."""

    def get_num_tasks(self) -> int:
        """Number of tasks.

        Returns:
            Number of tasks.
        """
        return len(TRIALS_PER_TASK)

    def get_task_init_states(self, task_id: int) -> list[int]:
        """List of initial states for a given task.

        Args:
            task_id: Task index.

        Returns:
            A placeholder list whose length equals that task's trial count.
        """
        return list(range(TRIALS_PER_TASK[task_id]))


class _StubSimModel:
    """MuJoCo model stub (only provides what's needed for contact naming)."""

    ngeom = 3

    def geom_id2name(self, index: int) -> str:
        """geom id → name.

        Args:
            index: geom id.

        Returns:
            The name; id 0 is the robot geom.
        """
        return {0: "robot0_link0", 1: "table_top", 2: "gripper0_finger"}[int(index)]


class _StubContact:
    """A single contact stub.

    Attributes:
        geom1: First geom id.
        geom2: Second geom id.
        dist: Penetration distance.
        pos: World coordinates.
        frame: Contact frame (first 3 components are the normal).
    """

    def __init__(self, geom1: int, geom2: int) -> None:
        """Initialize.

        Args:
            geom1: First geom id.
            geom2: Second geom id.
        """
        self.geom1 = geom1
        self.geom2 = geom2
        self.dist = -0.001
        self.pos = np.array([0.1, 0.2, 0.3])
        self.frame = np.arange(9, dtype=np.float64)


class _StubSimData:
    """MuJoCo data stub."""

    def __init__(self) -> None:
        """Initialize two contacts: one involving the robot, one not."""
        self.contact = [_StubContact(0, 1), _StubContact(1, 1)]
        self.ncon = len(self.contact)


class _StubSim:
    """robosuite ``sim`` stub."""

    def __init__(self) -> None:
        """Initialize model / data."""
        self.model = _StubSimModel()
        self.data = _StubSimData()


class _StubInnerEnv:
    """LIBERO env stub running in the subprocess (``install_runtime_extensions``
    is attached to it)."""

    def __init__(self) -> None:
        """Initialize sim and render counters."""
        self.sim = _StubSim()
        self.robots: list[Any] = []
        self.render_calls: list[dict[str, Any]] = []

    def render(self, **kwargs: Any) -> str:
        """Original ``render`` (wrapped by the extension forwarding layer).

        Args:
            **kwargs: Ignored.

        Returns:
            A fixed string used to prove "falls back to the original render
            when rr_extension is absent".
        """
        self.render_calls.append(dict(kwargs))
        return "original-render"


class _StubWorker:
    """venv worker stub: forwards ``render`` / camera commands to the
    subprocess-side env."""

    def __init__(self, env: Any) -> None:
        """Initialize.

        Args:
            env: The subprocess-side env.
        """
        self.env = env

    def render(self, **kwargs: Any) -> Any:
        """Corresponds to rlinf ``_worker``'s ``render`` command.

        Args:
            **kwargs: Passed through to the env's ``render``.

        Returns:
            The env's ``render`` return value.
        """
        return self.env.render(**kwargs)

    def close(self) -> None:
        """Cleanup (the stub holds no resources)."""


class _StubVenv:
    """``ReconfigureSubprocEnv`` stub.

    Attributes:
        workers: One worker per slot (fixed at 1 for this core).
        closed: Whether it has been closed.
    """

    def __init__(self, env_fns: list[Any]) -> None:
        """Construct workers from the factories (the factories have already
        been wrapped by ``wrap_env_factories``).

        Args:
            env_fns: List of env factories.
        """
        self.workers = [_StubWorker(factory()) for factory in env_fns]
        self.closed = False

    def close(self) -> None:
        """Mark as closed."""
        self.closed = True


class _StubLiberoEnv:
    """Stub for ``rlinf.envs.libero.libero_env.LiberoEnv``.

    Only implements the interface that ``LiberoEnvCore`` actually uses:
    the constructor signature, ``is_start``, ``task_suite``,
    ``reset(env_idx, reset_state_ids)``, ``step(actions, auto_reset=)``,
    ``elapsed_steps``, ``get_camera_meta`` / ``render_camera``, ``get_env_fns``.

    Attributes:
        reset_calls: Records the keyword arguments of every ``reset`` call,
            used to assert the family signature.
        step_actions: Records the actions received by every ``step`` call,
            used to assert the action preprocessing result.
        terminate_at: The step (1-indexed) at which ``terminated`` is reported;
            ``None`` means never terminate.
    """

    terminate_at: int | None = None

    def __init__(
        self,
        cfg: Any,
        num_envs: int,
        seed_offset: int,
        total_num_processes: int,
        worker_info: Any,
    ) -> None:
        """Initialize.

        Args:
            cfg: omegaconf configuration.
            num_envs: Number of envs (fixed at 1 for this core).
            seed_offset: Seed offset.
            total_num_processes: Total number of processes.
            worker_info: Ignored.
        """
        # The ``per_slot`` form must be "one LiberoEnv(num_envs=1) per slot";
        # the ``lockstep_vector`` form deliberately uses one vector env with
        # num_envs=pool_size.
        assert num_envs >= 1
        self.cfg = cfg
        self.num_envs = num_envs
        self.seed_offset = seed_offset
        self.total_num_processes = total_num_processes
        self.worker_info = worker_info
        self.is_start = True
        self.task_suite = _StubTaskSuite()
        self.task_ids = np.zeros(num_envs, dtype=np.int64)
        self.trial_ids = np.zeros(num_envs, dtype=np.int64)
        self.task_descriptions = ["stub: pick the cube"] * num_envs
        self._elapsed = np.zeros(num_envs, dtype=np.int32)
        self.reset_calls: list[dict[str, Any]] = []
        self.step_actions: list[np.ndarray] = []
        # LiberoEnv stores the raw obs coming back from the subprocess here
        # during reset / step; the ``libero.raw_obs`` extension reads it
        # (legacy env_server.py:294 reads the same thing).
        self.current_raw_obs: list[dict[str, Any]] | None = None
        self.env = _StubVenv(self.get_env_fns())

    # -- rlinf interface

    def get_env_fns(self) -> list[Any]:
        """Return the list of env factories.

        Returns:
            A single-element list (one slot, one env, for this core).
        """
        return [_StubInnerEnv for _ in range(self.num_envs)]

    @property
    def elapsed_steps(self) -> np.ndarray:
        """Number of steps executed so far.

        Returns:
            An array of shape ``[1]``.
        """
        return self._elapsed

    def reset(
        self, env_idx: Any = None, reset_state_ids: Any = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """The libero family's reset signature.

        Args:
            env_idx: Env index.
            reset_state_ids: Reset state id.

        Returns:
            ``(obs, infos)``.
        """
        lanes = np.asarray(env_idx).reshape(-1).tolist()
        ids = np.asarray(reset_state_ids).reshape(-1).tolist()
        self.reset_calls.append({"env_idx": lanes, "reset_state_ids": ids})
        for lane, reset_state_id in zip(lanes, ids, strict=True):
            reset_state_id = int(reset_state_id)
            task_id = 0
            pivot = 0
            for index, trials in enumerate(TRIALS_PER_TASK):
                if pivot <= reset_state_id < pivot + trials:
                    task_id = index
                    break
                pivot += trials
            self.task_ids[int(lane)] = task_id
            self.trial_ids[int(lane)] = reset_state_id - pivot
            self._elapsed[int(lane)] = 0
        return self._obs(), {}

    def step(
        self, actions: Any, auto_reset: bool = True
    ) -> tuple[dict[str, Any], Any, Any, Any, dict[str, Any]]:
        """A single step.

        Args:
            actions: Action of shape ``[1, action_dim]``.
            auto_reset: Must be ``False`` (the Runtime drives resets via the
                session).

        Returns:
            ``(obs, reward, terminated, truncated, infos)``.
        """
        assert auto_reset is False, "runtime must drive resets itself"
        block = np.asarray(actions)
        self.step_actions.append(block.copy())
        self._elapsed = self._elapsed + 1
        step_index = int(self._elapsed[0])
        terminated = self.terminate_at is not None and step_index >= self.terminate_at
        lanes = self.num_envs
        return (
            self._obs(),
            np.full(lanes, 1.0 if terminated else 0.0, dtype=np.float32),
            np.full(lanes, terminated, dtype=bool),
            np.zeros(lanes, dtype=bool),
            {},
        )

    def get_camera_meta(
        self, camera_name: str = "agentview", height: int = 256, width: int = 256
    ) -> dict[str, Any]:
        """Camera intrinsics/extrinsics stub.

        Args:
            camera_name: Camera name.
            height: Height.
            width: Width.

        Returns:
            Same structure as rlinf venv's ``get_camera_meta``.
        """
        return {
            "camera_name": camera_name,
            "height": height,
            "width": width,
            "intrinsic_K": [[1.0, 0.0, 0.5], [0.0, 1.0, 0.5], [0.0, 0.0, 1.0]],
            "extrinsic_cam2world": [[1.0, 0.0, 0.0, 0.0]] * 4,
            "depth_near": 0.01,
            "depth_far": 5.0,
        }

    def render_camera(
        self,
        camera_name: str = "agentview",
        height: int = 1024,
        width: int = 1024,
        depth: bool = False,
    ) -> Any:
        """Arbitrary camera rendering stub.

        Args:
            camera_name: Camera name.
            height: Height.
            width: Width.
            depth: Whether to also return depth.

        Returns:
            An RGB array, or ``(rgb, depth)``.
        """
        rgb = np.full((height, width, 3), 7, dtype=np.uint8)
        if depth:
            return rgb, np.full((height, width), 0.5, dtype=np.float32)
        return rgb

    # -- internal

    def _obs(self) -> dict[str, Any]:
        """Construct an obs following the 5-key schema.

        Returns:
            ``main_images`` / ``wrist_images`` / ``states`` / ``task_descriptions``.
        """
        step = int(self._elapsed[0])
        lanes = self.num_envs
        main = np.full((lanes, IMAGE_SIZE, IMAGE_SIZE, 3), step % 256, dtype=np.uint8)
        wrist = np.full(
            (lanes, IMAGE_SIZE, IMAGE_SIZE, 3), (step * 3) % 256, dtype=np.uint8
        )
        # The shape mirrors LIBERO's raw obs dict (including 2D depth, 1D
        # proprioception, and object *_pos keys), so the seam's raw_obs
        # decoding can be asserted key-by-key locally.
        entry = {
            "agentview_image": main[0],
            "agentview_depth": np.full(
                (IMAGE_SIZE, IMAGE_SIZE, 1), 0.25, dtype=np.float32
            ),
            "robot0_eye_in_hand_image": wrist[0],
            "robot0_eye_in_hand_depth": np.full(
                (IMAGE_SIZE, IMAGE_SIZE, 1), 0.5, dtype=np.float32
            ),
            "robot0_eef_pos": np.array([0.1, 0.2, 0.3 + step * 0.01]),
            # xyzw (scipy convention), deliberately with one negative
            # component: the 8-dim state's axisangle cannot recover this
            # sign, which is exactly why the raw_obs extension is needed.
            "robot0_eef_quat": np.array([0.0, 0.0, -0.7071, 0.7071]),
            "robot0_gripper_qpos": np.array([0.04, -0.04]),
            "akita_black_bowl_1_pos": np.array([0.5, 0.1, 0.9]),
            "plate_1_pos": np.array([0.6, 0.2, 0.9]),
            "robot0_joint_pos": np.zeros(7),
        }
        self.current_raw_obs = [dict(entry) for _ in range(lanes)]
        return {
            "main_images": main,
            "wrist_images": wrist,
            "states": np.full((lanes, STATE_DIM), float(step), dtype=np.float32),
            "task_descriptions": list(self.task_descriptions),
        }


_PREPARE_ACTION_CALLS: list[dict[str, Any]] = []
"""Call log for the ``prepare_actions`` stub."""


def _stub_prepare_actions(
    raw_chunk_actions: Any,
    env_type: str,
    model_type: str,
    num_action_chunks: int,
    action_dim: int,
    **kwargs: Any,
) -> np.ndarray:
    """Stub for ``rlinf.envs.action_utils.prepare_actions``.

    Deliberately behaves the same as the openvla branch (negating the last
    dimension), so tests can assert "the action actually went through family
    preprocessing" instead of being passed through unchanged.

    Args:
        raw_chunk_actions: Action of shape ``[1, chunk, dim]``.
        env_type: Family name.
        model_type: Model type.
        num_action_chunks: Chunk length.
        action_dim: Action dimension.
        **kwargs: Remaining parameters.

    Returns:
        The preprocessed action.
    """
    _PREPARE_ACTION_CALLS.append(
        {
            "env_type": env_type,
            "model_type": model_type,
            "num_action_chunks": num_action_chunks,
            "action_dim": action_dim,
            "shape": tuple(int(dim) for dim in np.asarray(raw_chunk_actions).shape),
        }
    )
    out = np.array(raw_chunk_actions, dtype=np.float32, copy=True)
    out[..., -1] = -out[..., -1]
    return out


def install_rlinf_stub(monkeypatch: pytest.MonkeyPatch) -> type[_StubLiberoEnv]:
    """Replace the two lazy imports of ``rlinf`` with stubs.

    Args:
        monkeypatch: pytest fixture.

    Returns:
        The stub env class (tests can modify its ``terminate_at``).
    """
    _PREPARE_ACTION_CALLS.clear()

    class _StubEnv(_StubLiberoEnv):
        """One copy per test case, to avoid cross-contamination of
        ``terminate_at``."""

        def get_env_fns(self) -> list[Any]:
            from rollout_runtime.backends import libero_privileged

            return libero_privileged.wrap_env_factories(super().get_env_fns())

    import rollout_runtime.backends.rlinf_env as backend_module
    import zetta.compat.actions as action_module

    monkeypatch.setattr(backend_module, "_libero_env_class", lambda: _StubEnv)
    monkeypatch.setattr(action_module, "prepare_actions", _stub_prepare_actions)

    modules = {
        "rlinf": _stub_module("rlinf"),
        "rlinf.envs": _stub_module("rlinf.envs"),
        "rlinf.envs.libero": _stub_module("rlinf.envs.libero"),
        "rlinf.envs.libero.libero_env": _stub_module("rlinf.envs.libero.libero_env"),
        "rlinf.envs.action_utils": _stub_module("rlinf.envs.action_utils"),
    }
    modules["rlinf.envs.libero.libero_env"].LiberoEnv = _StubEnv  # type: ignore[attr-defined]
    modules["rlinf.envs.action_utils"].prepare_actions = _stub_prepare_actions  # type: ignore[attr-defined]
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    return _StubEnv


@pytest.fixture
def stub_rlinf(monkeypatch: pytest.MonkeyPatch) -> type[_StubLiberoEnv]:
    """Fixture form of ``install_rlinf_stub`` (test cases fetch the stub
    under this name).

    Args:
        monkeypatch: pytest fixture.

    Returns:
        The stub env class (tests can modify its ``terminate_at``).
    """
    return install_rlinf_stub(monkeypatch)


# Public aliases (the module internals keep the original underscore naming,
# left unchanged verbatim).
stub_module = _stub_module
StubLiberoEnv = _StubLiberoEnv
StubInnerEnv = _StubInnerEnv
PREPARE_ACTION_CALLS = _PREPARE_ACTION_CALLS
