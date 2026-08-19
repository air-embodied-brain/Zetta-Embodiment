"""Divergences and the two execution forms of the new maniskill family.

``.venv-runtime`` does not have mani_skill / robosuite, so this uses a minimal
``rlinf`` stub to drive the **real** ``ManiskillEnvCore``: the family
signature (one of the four ``reset`` forms), the family branch of action
preprocessing, GPU tensor device differences, the 5-key observation schema,
and both the ``per_slot`` / ``lockstep_vector`` execution forms can all be
pinned down locally. Real-hardware behavior is re-verified by
``test_multi_family_remote.py`` (``@pytest.mark.remote``) on a configured GPU
host.

What the stub does not cover is "whether these parameters can actually boot a
real simulator" — that is the job of the GPU smoke test.

The rlinf-based backend for the robocasa family (``rlinf_robocasa.py``) has
been dropped as part of the Rollout Runtime v3 migration in favor of the
current branch's ``RoboCasaSession``/GR00T business logic; the corresponding
stubs and test cases were removed together with it. Tests for the new
backend, ``robocasa_current.py``, live elsewhere.
"""

from __future__ import annotations

import sys
from typing import Any

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from rollout_runtime.api.enums import ErrorCode
from rollout_runtime.api.errors import RuntimeApiError
from rollout_runtime.api.messages import EnvSpecMsg, ResetSpec
from rollout_runtime.core.env_execution import (
    LOCKSTEP_VECTOR_FORM,
    PER_SLOT_FORM,
)
from rollout_runtime.core.env_registry import behavior_for
from tests.runtime.libero_stub import stub_module

ACTION_DIM = 7
IMAGE = 8
PREPARE_CALLS: list[dict[str, Any]] = []


# ------------------------------------------------------------------- shared stub


def _prepare_actions_stub(
    raw_chunk_actions: Any,
    env_type: str,
    model_type: str,
    num_action_chunks: int,
    action_dim: int,
    **kwargs: Any,
) -> Any:
    """Stub for ``prepare_actions``: performs an **observable** transformation per
    family.

    - maniskill: identity (the real ``panda*`` branch is also identity), but
      records one call.

    Args:
        raw_chunk_actions: ``[num_envs, chunk, dim]`` actions.
        env_type: Family name.
        model_type: Model type.
        num_action_chunks: Chunk length.
        action_dim: Action dimension.
        **kwargs: ``policy`` / ``action_scale`` etc.

    Returns:
        The preprocessed actions.
    """
    PREPARE_CALLS.append(
        {
            "env_type": env_type,
            "model_type": model_type,
            "num_action_chunks": num_action_chunks,
            "action_dim": action_dim,
            "shape": tuple(int(dim) for dim in np.asarray(raw_chunk_actions).shape),
            **{key: kwargs[key] for key in ("policy", "action_scale") if key in kwargs},
        }
    )
    return np.asarray(raw_chunk_actions, dtype=np.float32)


class _StubManiskillEnv:
    """Stub for ``ManiskillEnv`` (the shape of the GPU-batched family).

    Attributes:
        reset_calls: Keyword args for each ``reset``, used to assert the
            ``seed_options`` signature.
        step_actions: Actions received by each ``step`` call (torch tensor).
        terminate_at: The step (1-indexed) at which lane 0 terminates;
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
        record_metrics: bool = True,
    ) -> None:
        """Initialize.

        Args:
            cfg: omegaconf config.
            num_envs: Number of lanes.
            seed_offset: Seed offset.
            total_num_processes: Total number of processes.
            worker_info: Ignored.
            record_metrics: Ignored.
        """
        self.cfg = cfg
        self.num_envs = num_envs
        self.seed_offset = seed_offset
        self.total_num_processes = total_num_processes
        self.is_start = True
        self.device = torch.device("cpu")
        self.reset_calls: list[dict[str, Any]] = []
        self.step_actions: list[Any] = []
        self._elapsed = torch.zeros(num_envs, dtype=torch.long)

    @property
    def elapsed_steps(self) -> Any:
        """Number of steps executed so far (per lane).

        Returns:
            A tensor of shape ``[num_envs]``.
        """
        return self._elapsed

    @property
    def instruction(self) -> list[str]:
        """Language instruction.

        Returns:
            One per lane.
        """
        return [f"maniskill: lane {index}" for index in range(self.num_envs)]

    def reset(self, *, seed: Any = None, options: Any = None) -> tuple[Any, dict]:
        """The maniskill family's reset signature (keyword ``seed`` / ``options``).

        Args:
            seed: A seed or list of seeds.
            options: A dict containing ``env_idx`` (for subset resets).

        Returns:
            ``(obs, infos)``.
        """
        self.reset_calls.append({"seed": seed, "options": options})
        lanes = (options or {}).get("env_idx")
        if lanes is not None:
            for lane in np.asarray(lanes.cpu() if hasattr(lanes, "cpu") else lanes):
                self._elapsed[int(lane)] = 0
        else:
            self._elapsed[:] = 0
        return self._obs(), {}

    def step(self, actions: Any, auto_reset: bool = True) -> tuple[Any, ...]:
        """A single step (advances the whole batch, matching the real family).

        Args:
            actions: ``[num_envs, action_dim]`` actions.
            auto_reset: Must be ``False``.

        Returns:
            ``(obs, reward, terminated, truncated, infos)``, all as torch
            tensors.
        """
        assert auto_reset is False, "runtime must drive resets itself"
        self.step_actions.append(actions)
        self._elapsed = self._elapsed + 1
        step_index = int(self._elapsed[0])
        terminated = torch.zeros(self.num_envs, dtype=torch.bool)
        if self.terminate_at is not None and step_index >= self.terminate_at:
            terminated[0] = True
        return (
            self._obs(),
            torch.full((self.num_envs,), 0.5, dtype=torch.float32),
            terminated,
            torch.zeros(self.num_envs, dtype=torch.bool),
            {},
        )

    def close(self) -> None:
        """Cleanup (the stub holds no resources)."""

    def _obs(self) -> dict[str, Any]:
        """Build an obs with the 5-key schema (maniskill's ``simple`` branch has no
        wrist camera).

        Returns:
            The obs dict.
        """
        step = int(self._elapsed[0])
        return {
            "main_images": torch.full(
                (self.num_envs, IMAGE, IMAGE, 3), step % 256, dtype=torch.uint8
            ),
            "extra_view_images": torch.full(
                (self.num_envs, 1, IMAGE, IMAGE, 3), 3, dtype=torch.uint8
            ),
            "states": torch.full((self.num_envs, 9), float(step), dtype=torch.float32),
        }


@pytest.fixture
def stub_families(monkeypatch: pytest.MonkeyPatch) -> dict[str, type]:
    """Replace maniskill's lazy import with the stub.

    Args:
        monkeypatch: pytest fixture.

    Returns:
        ``{"maniskill": stub class}``.
    """
    PREPARE_CALLS.clear()

    class _Maniskill(_StubManiskillEnv):
        """One instance per test case, to avoid ``terminate_at`` cross-contamination."""

    import rollout_runtime.backends.rlinf_maniskill as backend_module
    import zetta.compat.actions as action_module

    monkeypatch.setattr(backend_module, "_maniskill_env_class", lambda: _Maniskill)
    monkeypatch.setattr(action_module, "prepare_actions", _prepare_actions_stub)

    modules = {
        "rlinf": stub_module("rlinf"),
        "rlinf.envs": stub_module("rlinf.envs"),
        "rlinf.envs.maniskill": stub_module("rlinf.envs.maniskill"),
        "rlinf.envs.maniskill.maniskill_env": stub_module(
            "rlinf.envs.maniskill.maniskill_env"
        ),
        "rlinf.envs.action_utils": stub_module("rlinf.envs.action_utils"),
    }
    modules["rlinf.envs.maniskill.maniskill_env"].ManiskillEnv = _Maniskill  # type: ignore[attr-defined]
    modules["rlinf.envs.action_utils"].prepare_actions = _prepare_actions_stub  # type: ignore[attr-defined]
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    return {"maniskill": _Maniskill}


def maniskill_spec(pool_size: int = 1, **overrides: Any) -> EnvSpecMsg:
    """A maniskill env spec.

    Args:
        pool_size: Pool capacity.
        **overrides: Overrides for ``env_config``.

    Returns:
        The env spec.
    """
    config: dict[str, Any] = {
        "env_id": "PickCube-v1",
        "action_dim": ACTION_DIM,
        "chunk_size": 3,
        "camera_height": IMAGE,
        "camera_width": IMAGE,
    }
    config.update(overrides)
    return EnvSpecMsg(env_family="maniskill", env_config=config, pool_size=pool_size)


# ------------------------------------------------------------------- declaration consistency


def test_new_families_declare_their_divergences() -> None:
    """The declarations for new families match the actual facts."""
    maniskill = behavior_for("maniskill")
    assert maniskill.reset_signature == "seed_options"
    assert maniskill.device_kind == "gpu_batched"
    assert maniskill.action_layout == "torch_env_chunk_dim"
    assert maniskill.needs_accelerator is True
    assert maniskill.per_step_obs_available is True
    assert maniskill.extensions == frozenset()

    assert maniskill.supports_coalescing is True
    assert maniskill.core_forms == frozenset({PER_SLOT_FORM, LOCKSTEP_VECTOR_FORM})


# ----------------------------------------------------------------- maniskill


def test_maniskill_reset_uses_the_seed_options_signature(
    stub_families: dict[str, type],
) -> None:
    """maniskill's ``reset`` uses keyword ``seed`` / ``options``; subsets rely on
    ``options["env_idx"]``."""
    from rollout_runtime.backends.rlinf_maniskill import ManiskillEnvCore

    core = ManiskillEnvCore()
    core.build(maniskill_spec(pool_size=2, core_form=LOCKSTEP_VECTOR_FORM), num_envs=2)
    assert core.core_form == LOCKSTEP_VECTOR_FORM
    observations = core.reset([0, 1], ResetSpec(seed=7))
    env = core._envs[0]
    assert env.is_start is False, "is_start must be cleared at build"
    call = env.reset_calls[-1]
    assert call["seed"] == [7, 8]
    assert [int(item) for item in call["options"]["env_idx"]] == [0, 1]
    # 5-key schema: uint8 HWC main view + float32 state + non-empty instruction.
    assert observations[0].main_image is not None
    assert observations[0].main_image.shape == (IMAGE, IMAGE, 3)
    assert observations[0].main_image.dtype == "uint8"
    assert observations[0].wrist_image is None  # the simple branch has no wrist camera
    assert len(observations[0].extra_view_images) == 1
    assert len(observations[0].state) == 9
    assert observations[0].instruction == "maniskill: lane 0"
    assert observations[1].instruction == "maniskill: lane 1"
    assert observations[0].extras["core_form"] == LOCKSTEP_VECTOR_FORM
    assert observations[1].extras["lane_index"] == 1
    core.close()


def test_maniskill_actions_reach_the_env_as_device_tensors(
    stub_families: dict[str, type],
) -> None:
    """Divergences 4 + 5: actions pass through the family's ``prepare_actions`` and
    are fed to the env as device tensors."""
    from rollout_runtime.backends.rlinf_maniskill import ManiskillEnvCore

    core = ManiskillEnvCore()
    core.build(maniskill_spec(), num_envs=1)
    core.reset([0], ResetSpec(seed=1))
    outcome = core.chunk_step([0], [np.ones((3, ACTION_DIM), dtype=np.float32)])[0]
    assert outcome.executed_horizon == 3
    assert outcome.per_step_obs_available is True
    assert len(outcome.per_step or []) == 3
    assert [record.step_index for record in outcome.per_step or []] == [1, 2, 3]
    call = PREPARE_CALLS[-1]
    assert call["env_type"] == "maniskill"
    assert call["policy"] == "panda_wristcam"
    assert call["shape"] == (1, 3, ACTION_DIM)
    env = core._envs[0]
    assert isinstance(env.step_actions[0], torch.Tensor)
    assert env.step_actions[0].dtype == torch.float32
    core.close()


def test_maniskill_chunk_stops_at_the_first_termination(
    stub_families: dict[str, type],
) -> None:
    """``executed_horizon`` is the **actual** number of steps taken, not the
    requested chunk length."""
    from rollout_runtime.backends.rlinf_maniskill import ManiskillEnvCore

    stub_families["maniskill"].terminate_at = 2
    core = ManiskillEnvCore()
    core.build(maniskill_spec(), num_envs=1)
    core.reset([0], ResetSpec(seed=1))
    outcome = core.chunk_step([0], [np.zeros((3, ACTION_DIM), dtype=np.float32)])[0]
    assert outcome.executed_horizon == 2
    assert outcome.terminated is True
    assert outcome.info["requested_horizon"] == 3
    core.close()


def test_maniskill_rejects_undeclared_extensions(
    stub_families: dict[str, type],
) -> None:
    """maniskill's ``extensions`` is an empty set, so any extension call returns
    ``UNSUPPORTED_EXTENSION``."""
    from rollout_runtime.backends.rlinf_maniskill import ManiskillEnvCore

    core = ManiskillEnvCore()
    core.build(maniskill_spec(), num_envs=1)
    with pytest.raises(RuntimeApiError) as excinfo:
        core.extension(0, "libero", "render_camera", {})
    assert excinfo.value.info.code is ErrorCode.UNSUPPORTED_EXTENSION
    core.close()


# ------------------------------------------------------- form selection and digest semantics


def test_core_form_enters_the_digest_so_the_two_forms_never_share_a_pool() -> None:
    """The two forms are two physically distinct pools, so ``core_form`` must be
    part of the digest."""
    per_slot = maniskill_spec(core_form=PER_SLOT_FORM)
    vector = maniskill_spec(core_form=LOCKSTEP_VECTOR_FORM)
    assert per_slot.digest() != vector.digest()
    # Omitting core_form is behaviorally equivalent to per_slot, but the
    # digest still differs (the config literal is different) — this does not
    # affect correctness: the default pool and an explicitly declared
    # per_slot pool have the same semantics.
    assert maniskill_spec().digest() != per_slot.digest()


def test_unsupported_core_form_is_refused_at_build(
    stub_families: dict[str, type],
) -> None:
    """The form must be declared by the family; a typo fails immediately with
    ``INVALID_ARGUMENT`` rather than blowing up later at runtime."""
    from rollout_runtime.backends.rlinf_maniskill import ManiskillEnvCore

    core = ManiskillEnvCore()
    with pytest.raises(RuntimeApiError) as excinfo:
        core.build(maniskill_spec(core_form="magic"), num_envs=1)
    assert excinfo.value.info.code is ErrorCode.INVALID_ARGUMENT
    assert excinfo.value.info.detail["requested_core_form"] == "magic"


def test_per_slot_form_builds_one_env_per_slot(
    stub_families: dict[str, type],
) -> None:
    """``per_slot`` form: 3 slots -> 3 independent envs, each with a
    different seed offset."""
    from rollout_runtime.backends.rlinf_maniskill import ManiskillEnvCore

    core = ManiskillEnvCore()
    core.build(
        maniskill_spec(pool_size=3, core_form=PER_SLOT_FORM), num_envs=3, seed_offset=1
    )
    assert len(core._envs) == 3
    assert [env.seed_offset for env in core._envs] == [3, 4, 5]
    assert [lane.lane_index for lane in core._lanes] == [0, 0, 0]
    core.reset([1], ResetSpec(seed=1))
    core.chunk_step([1], [np.zeros((3, ACTION_DIM), dtype=np.float32)])
    # Only slot 1's env was stepped; the other two never took a step.
    assert [len(env.step_actions) for env in core._envs] == [0, 3, 0]
    core.close()


def test_lockstep_form_builds_one_shared_env(
    stub_families: dict[str, type],
) -> None:
    """``lockstep_vector`` form: 3 slots share **one** env, each occupying one lane."""
    from rollout_runtime.backends.rlinf_maniskill import ManiskillEnvCore

    core = ManiskillEnvCore()
    core.build(
        maniskill_spec(pool_size=3, core_form=LOCKSTEP_VECTOR_FORM),
        num_envs=3,
        seed_offset=1,
    )
    assert len(core._envs) == 1
    assert core._envs[0].num_envs == 3
    assert [lane.lane_index for lane in core._lanes] == [0, 1, 2]
    core.reset([0, 1, 2], ResetSpec(seed=1))
    outcomes = core.chunk_step(
        [0, 2], [np.zeros((3, ACTION_DIM), dtype=np.float32)] * 2
    )
    assert len(outcomes) == 2
    # One vector advance covers all 3 lanes; the non-participating slot 1
    # is carried through 3 steps by the hold action.
    assert tuple(core._envs[0].step_actions[0].shape) == (3, ACTION_DIM)
    assert core._lanes[1].masked_steps == 3
    assert core.total_masked_steps == 3
    assert outcomes[0].info["masked_slots"] == [1]
    assert outcomes[0].info["coalesced_slots"] == [0, 2]
    core.close()


def test_gpu_batched_family_needs_a_rank_that_declares_an_accelerator() -> None:
    """A real defect exposed on GPU: ``has_accelerator`` used to be hardcoded
    to ``False``, so maniskill could never get a rank.

    ``EnvWorkerRegistry.serves()``'s rule is "a family requiring
    ``needs_accelerator`` can only land on a rank with
    ``has_accelerator``," so the worker must report it **honestly**. The
    default inference is ``placement_strategy != "node"`` (``node`` means N
    processes on CPU).
    """
    from rollout_runtime.backends import register_env_family_for
    from rollout_runtime.config.schema import EnvWorkerConfig
    from rollout_runtime.gateway.worker_registry import EnvWorkerRegistry, WorkerEntry
    from rollout_runtime.workers.env_worker import RuntimeEnvWorker

    register_env_family_for("maniskill")
    register_env_family_for("libero")

    # Default inference: node -> no accelerator; packed -> has accelerator;
    # explicit declaration takes priority.
    assert EnvWorkerConfig(placement_strategy="node").accelerator_present() is False
    assert EnvWorkerConfig(placement_strategy="packed").accelerator_present() is True
    assert (
        EnvWorkerConfig(
            placement_strategy="node", has_accelerator=True
        ).accelerator_present()
        is True
    )

    registry = EnvWorkerRegistry()
    spec = maniskill_spec()
    cpu_only = RuntimeEnvWorker(
        supported_families=("maniskill",), has_accelerator=False
    )
    with_gpu = RuntimeEnvWorker(supported_families=("maniskill",), has_accelerator=True)
    assert cpu_only.worker_info().has_accelerator is False
    assert with_gpu.worker_info().has_accelerator is True
    assert registry.serves(WorkerEntry(info=cpu_only.worker_info()), spec) is False
    assert registry.serves(WorkerEntry(info=with_gpu.worker_info()), spec) is True
    # libero does not need an accelerator, so either kind of rank can serve
    # it (do **not** assign it a GPU).
    libero_worker = RuntimeEnvWorker(
        supported_families=("libero",), has_accelerator=False
    )
    libero_spec = EnvSpecMsg(env_family="libero", env_config={}, pool_size=1)
    assert (
        registry.serves(WorkerEntry(info=libero_worker.worker_info()), libero_spec)
        is True
    )
