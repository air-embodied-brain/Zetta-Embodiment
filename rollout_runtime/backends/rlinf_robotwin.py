# Copyright (c) 2026 Zetta Contributors
"""RoboTwin 2.0 family adapter.

**Does not reimplement the env**: ``zetta.envs.robotwin.environment.RoboTwinEnv``
(vendored from RLinf ``9ad44393``) is driven as-is. This module translates the
Runtime's session/slot semantics into family calls and normalizes the family's
output through ``core.env_execution.normalize_chunk_outcome``, the single exit
point.

Its points of divergence from the other rlinf families -- all of them purely
**declarative**, so nothing upstream needs an ``if env_family == ...``:

1. ``chunk_obs_layout="final_only"``. RoboTwin is the **only** such family.
   ``RoboTwinEnv.chunk_step`` hands the whole chunk to ``venv.step`` in one call
   and returns an ``obs_list`` of length 1 no matter how long the chunk was, so
   this adapter calls the family's own ``chunk_step`` rather than stepping lane
   by lane the way libero/maniskill do. The consequence is real and deliberate
   (see D7 in ``plan/robotwin_support_plan.md``): there are **no intermediate
   frames**, so ``ChunkOutcome.per_step`` is ``None`` and Cluster/Diagnose
   evidence is chunk-granular rather than step-granular.
2. ``reset_signature="env_idx_env_seeds"``: ``reset(env_idx, env_seeds)``, with
   the seed list carrying the *scene* identity. RoboTwin's seed **is** its
   reset-state id, so unlike maniskill this family declares
   ``supports_reset_state_id=True`` -- and that declaration is load-bearing,
   because the campaign protocol's held-out seed discipline depends on being
   able to pin an exact scene.
3. ``action_layout="numpy_env_chunk_dim"`` at 14 dims (ALOHA bimanual:
   6 joints + 1 gripper per arm), not 7.
4. ``device_kind="cpu_subproc"`` **with** ``needs_accelerator_override=True``:
   ``VectorEnv`` is a plain multiprocessing pool, not a batched-GPU-tensor
   family, but SAPIEN rendering will not initialise without a GPU (a container
   lacking the ``graphics`` driver capability fails with
   ``failed to find a rendering device``). Same shape as RoboCasa's override.
5. ``per_slot`` only: ``VectorEnv`` already owns its own subprocess fan-out, so
   there is no second "several lanes in one process" form to expose.
6. **Wrist unstacking (D1).** The family stacks left/right wrist frames into one
   ``[num_envs, n, H, W, C]`` ``wrist_images`` tensor, while the Runtime's
   ``Observation`` has a single ``wrist_image`` plus ``extra_view_images``. The
   split happens here: left wrist -> ``wrist_image``, right wrist ->
   ``extra_view_images[0]``. The mapping is driven by the family's
   ``wrist_image_names``, never by position, and it enters ``obs_schema_digest``
   -- so getting it wrong does not raise, it silently feeds the policy a
   mirrored view. ``tests/runtime/test_robotwin_family.py`` pins it.
7. **Open-loop execute horizon (D3).** ``num_action_chunks`` for the published
   RoboTwin Pi0.5 checkpoint is 50, and with ``final_only`` that would mean one
   observation per 50 simulator steps -- too coarse for the runtime Critic to
   attribute anything. ``execute_horizon`` submits only the first N actions of
   each chunk and discards the rest, which is ordinary open-loop replanning: the
   model is untouched, and more frequent replanning generally helps rather than
   hurts.

Dependency surface: the vendored env + ``robotwin`` + SAPIEN + torch + numpy,
all **lazily imported inside functions**.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from rollout_runtime.api.enums import ErrorCode
from rollout_runtime.api.errors import RuntimeApiError, make_error
from rollout_runtime.api.messages import (
    EnvFamilyCapability,
    EnvSpecMsg,
    Observation,
    ResetSpec,
)
from rollout_runtime.backends.rlinf_family import (
    LaneState,
    LaneStatus,
    lane_statuses,
    observation_from_payload,
    to_numpy,
)
from rollout_runtime.core.env_execution import (
    PER_SLOT_FORM,
    ChunkOutcome,
    EnvFamilyBehavior,
    normalize_chunk_outcome,
)
from rollout_runtime.core.env_registry import (
    ROBOTWIN_ENV_FAMILY,
    behavior_for,
    capability_from_behavior,
    register_env_family,
    requested_core_form,
)

__all__ = [
    "ROBOTWIN_ENV_FAMILY",
    "RobotwinEnvConfig",
    "RobotwinEnvCore",
    "RobotwinEnvFamily",
    "register_robotwin_env_family",
    "robotwin_env_capability",
]

ROBOTWIN_ACTION_DIM = 14
"""ALOHA bimanual action width: 6 joints + 1 gripper, per arm."""

LEFT_WRIST_KEY = "left_wrist_image"
"""The family's name for the frame that becomes ``Observation.wrist_image``."""

RIGHT_WRIST_KEY = "right_wrist_image"
"""The family's name for the frame that becomes ``extra_view_images[0]``."""


@dataclasses.dataclass
class RobotwinEnvConfig:
    """RoboTwin family private config (corresponds to ``EnvSpecMsg.env_config``).

    Field names follow RLinf's ``env/robotwin_*.yaml`` so a published RLinf
    experiment can be transcribed without renaming anything.

    Attributes:
        task_name: RoboTwin task id, e.g. ``adjust_bottle``.
        embodiment: RoboTwin embodiment spec. A one-element list picks a single
            dual-arm robot (``["aloha-agilex"]``); a three-element list picks
            two arms plus their separation (``["piper", "piper", 0.6]``).
        assets_path: The RoboTwin **repository root**. Not its ``assets/``
            subdirectory -- RoboTwin joins ``assets/...`` onto this internally,
            and pointing at the subdirectory yields a confusing
            ``.../assets/assets/...`` ``FileNotFoundError``.
        seeds_path: Optional curated success-seed JSON. When present the family
            draws reset seeds from it instead of sampling; explicit seeds in a
            ``ResetSpec`` still win over both.
        planner_backend: ``mplib`` or ``curobo``. ``mplib`` is the default here
            rather than RoboTwin's own ``curobo`` default: curobo needs a CUDA
            toolchain to build, RLinf's own RoboTwin configs all select mplib,
            and the validated image ships no curobo at all.
        step_lim: RoboTwin's internal per-task planning step limit.
        max_episode_steps: Outer truncation horizon, checked by the family
            against its own elapsed-step counter.
        center_crop: Whether to centre-crop and resize frames to 224x224.
            RoboTwin's cameras deliver 240x320, so this is also the resolution
            switch.
        collect_head_camera: Whether the head camera is rendered.
        collect_wrist_camera: Whether both wrist cameras are rendered. Off means
            ``wrist_image``/``extra_view_images`` come back empty.
        head_camera_type: Camera model for the head view.
        wrist_camera_type: Camera model for the wrist views.
        domain_randomization: RoboTwin's randomization block, passed through
            verbatim.
        reward_coef: Scale applied by the family's custom reward.
        use_custom_reward: Whether to derive reward from the termination flag
            rather than the environment's own.
        use_rel_reward: Whether the custom reward is a per-step difference.
        ignore_terminations: Whether to force termination flags to false and run
            a fixed horizon. RLinf's published eval sets this, which is why its
            reported ``episode_len`` is always the full horizon.
        auto_reset: Whether the family may reset itself. The Runtime drives
            resets through sessions, so this is fixed to false.
        use_fixed_reset_state_ids: Whether the family pins its own reset-state
            pool.
        is_eval: The family's eval flag; only consulted on its auto-reset path.
        group_size: rlinf's group replication; the Runtime binds one session per
            lane, so this is fixed to 1.
        seed: Family base seed; the effective seed still adds ``seed_offset``.
        action_dim: Action width; must be 14.
        chunk_size: Declared chunk length (actual execution follows the supplied
            actions and ``execute_horizon``).
        execute_horizon: How many actions of each chunk to actually submit;
            ``None`` submits all of them. See divergence 7.
        action_model_type: Model type recorded for ``prepare_actions``.
        save_video: Whether the family writes video.
        video_base_dir: Where the family writes video when enabled.
        teardown_on_close: Whether ``close`` really tears the SAPIEN pool down.
            Off by default because that teardown **segfaults**; see
            :meth:`RobotwinEnvCore.close`. Turn it on only against a RoboTwin
            build where the crash is fixed.
        core_form: Execution core form; only ``per_slot`` is declared.
        extra_task_config: Extra keys merged into RoboTwin's ``task_config``.
    """

    task_name: str = "adjust_bottle"
    embodiment: list[Any] = dataclasses.field(default_factory=lambda: ["aloha-agilex"])
    assets_path: str = ""
    seeds_path: str | None = None
    planner_backend: str = "mplib"
    step_lim: int = 200
    max_episode_steps: int = 200
    center_crop: bool = False
    collect_head_camera: bool = True
    collect_wrist_camera: bool = True
    head_camera_type: str = "D435"
    wrist_camera_type: str = "D435"
    domain_randomization: dict[str, Any] = dataclasses.field(default_factory=dict)
    reward_coef: float = 1.0
    use_custom_reward: bool = True
    use_rel_reward: bool = False
    ignore_terminations: bool = False
    auto_reset: bool = False
    use_fixed_reset_state_ids: bool = False
    is_eval: bool = True
    group_size: int = 1
    seed: int = 0
    action_dim: int = ROBOTWIN_ACTION_DIM
    chunk_size: int = 16
    execute_horizon: int | None = None
    action_model_type: str = "openpi"
    save_video: bool = False
    video_base_dir: str = "/tmp/rr_robotwin_video"
    teardown_on_close: bool = False
    core_form: str = PER_SLOT_FORM
    extra_task_config: dict[str, Any] = dataclasses.field(default_factory=dict)

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any] | None) -> RobotwinEnvConfig:
        """Construct a config from an ``env_config`` dict.

        Unknown keys are rejected rather than ignored: ``env_config`` feeds
        ``EnvSpecMsg.digest()``, so a typo would silently allocate a second
        pool -- and for RoboTwin a pool is a set of SAPIEN subprocesses plus
        their share of a hard GPU memory budget, which is far worse to diagnose
        than an error.

        Args:
            config: The family-private config; ``None`` means all defaults.

        Returns:
            The structured config.

        Raises:
            RuntimeApiError: An unknown key is present, or a value violates a
                family invariant (``INVALID_ARGUMENT``).
        """
        if not config:
            return cls()
        known = {field.name for field in dataclasses.fields(cls)}
        unknown = sorted(key for key in config if key not in known)
        if unknown:
            raise RuntimeApiError(
                make_error(
                    ErrorCode.INVALID_ARGUMENT,
                    f"unknown robotwin env config keys: {unknown}",
                    unknown_keys=unknown,
                    known_keys=sorted(known),
                )
            )
        instance = cls(**dict(config))
        instance._validate()
        return instance

    def _validate(self) -> None:
        """Check the family invariants.

        Raises:
            RuntimeApiError: ``group_size`` is not 1, ``action_dim`` is not 14,
                ``execute_horizon`` is not positive, or ``assets_path`` is
                missing (``INVALID_ARGUMENT``).
        """
        if self.group_size != 1:
            raise RuntimeApiError(
                make_error(
                    ErrorCode.INVALID_ARGUMENT,
                    "robotwin group_size must be 1: the Runtime binds one session "
                    "per lane, so rlinf's group replication has no meaning here",
                    group_size=self.group_size,
                )
            )
        if self.action_dim != ROBOTWIN_ACTION_DIM:
            raise RuntimeApiError(
                make_error(
                    ErrorCode.INVALID_ARGUMENT,
                    f"robotwin is bimanual and requires action_dim="
                    f"{ROBOTWIN_ACTION_DIM}",
                    action_dim=self.action_dim,
                )
            )
        if self.execute_horizon is not None and self.execute_horizon < 1:
            raise RuntimeApiError(
                make_error(
                    ErrorCode.INVALID_ARGUMENT,
                    "robotwin execute_horizon must be >= 1 when set",
                    execute_horizon=self.execute_horizon,
                )
            )
        if not self.assets_path:
            raise RuntimeApiError(
                make_error(
                    ErrorCode.INVALID_ARGUMENT,
                    "robotwin assets_path must point at the RoboTwin repository "
                    "root (not its assets/ subdirectory)",
                )
            )

    def task_config(self) -> dict[str, Any]:
        """Build RoboTwin's ``task_config`` block.

        Returns:
            A plain dict suitable for ``VectorEnv(task_config=...)``.
        """
        return {
            "task_name": self.task_name,
            "step_lim": int(self.step_lim),
            "planner_backend": self.planner_backend,
            "render_freq": 0,
            "episode_num": 100,
            "use_seed": False,
            "save_freq": 15,
            "embodiment": list(self.embodiment),
            "language_num": 100,
            "domain_randomization": dict(self.domain_randomization),
            "camera": {
                "head_camera_type": self.head_camera_type,
                "wrist_camera_type": self.wrist_camera_type,
                "collect_head_camera": bool(self.collect_head_camera),
                "collect_wrist_camera": bool(self.collect_wrist_camera),
            },
            "data_type": {
                "rgb": True,
                "third_view": False,
                "depth": False,
                "pointcloud": False,
                "observer": False,
                "endpose": False,
                "qpos": True,
                "mesh_segmentation": False,
                "actor_segmentation": False,
            },
            "pcd_down_sample_num": 1024,
            "pcd_crop": True,
            "save_path": "./data",
            "clear_cache_freq": 1,
            # The Runtime owns episode artifacts; the family must not also write
            # its own dataset next to them.
            "collect_data": False,
            "eval_video_log": bool(self.save_video),
            **dict(self.extra_task_config),
        }

    def to_rlinf_cfg(self) -> Any:
        """Project into the config object ``RoboTwinEnv(cfg=...)`` expects.

        Returns:
            A ``DictConfig`` when omegaconf is importable, otherwise a plain
            ``SimpleNamespace``-backed equivalent. The vendored env reads the
            config through ``getattr``/``.get``, so both work; falling back
            keeps the core constructible in a test environment without
            omegaconf.
        """
        payload = {
            "env_type": ROBOTWIN_ENV_FAMILY,
            "seed": int(self.seed),
            "auto_reset": bool(self.auto_reset),
            "ignore_terminations": bool(self.ignore_terminations),
            "use_rel_reward": bool(self.use_rel_reward),
            "use_custom_reward": bool(self.use_custom_reward),
            "use_fixed_reset_state_ids": bool(self.use_fixed_reset_state_ids),
            "is_eval": bool(self.is_eval),
            "group_size": int(self.group_size),
            "reward_coef": float(self.reward_coef),
            "max_episode_steps": int(self.max_episode_steps),
            "center_crop": bool(self.center_crop),
            "assets_path": self.assets_path,
            "seeds_path": self.seeds_path,
            "video_cfg": {
                "save_video": bool(self.save_video),
                "info_on_video": bool(self.save_video),
                "video_base_dir": self.video_base_dir,
            },
            "task_config": self.task_config(),
        }
        try:
            from omegaconf import OmegaConf
        except ImportError:
            return _PlainConfig(payload)
        return OmegaConf.create(payload)


class _PlainConfig:
    """Minimal attribute/``get`` view over a nested dict.

    Stands in for an OmegaConf ``DictConfig`` when omegaconf is absent, which is
    the case in the minimal test environment. Only the access patterns the
    vendored env actually uses are supported: attribute reads, ``.get(key,
    default)``, and nested dicts behaving the same way.
    """

    def __init__(self, payload: Mapping[str, Any]) -> None:
        """Wrap a mapping.

        Args:
            payload: The configuration mapping.
        """
        self._payload = dict(payload)

    def __getattr__(self, name: str) -> Any:
        """Read a key as an attribute.

        Args:
            name: The key.

        Returns:
            The value, wrapped when it is itself a mapping.

        Raises:
            AttributeError: The key is absent.
        """
        payload = object.__getattribute__(self, "_payload")
        if name not in payload:
            raise AttributeError(name)
        value = payload[name]
        return _PlainConfig(value) if isinstance(value, Mapping) else value

    def get(self, name: str, default: Any = None) -> Any:
        """Read a key with a default.

        Args:
            name: The key.
            default: Returned when the key is absent.

        Returns:
            The value, wrapped when it is itself a mapping.
        """
        if name not in self._payload:
            return default
        value = self._payload[name]
        return _PlainConfig(value) if isinstance(value, Mapping) else value

    def to_container(self) -> dict[str, Any]:
        """Return the underlying plain dict.

        Returns:
            The wrapped mapping.
        """
        return dict(self._payload)


def _robotwin_env_class() -> type:
    """Lazily fetch ``RoboTwinEnv``.

    Not imported at module top level: the module needs ``gymnasium`` and, on
    construction, the ``robotwin`` package, neither of which exists in the
    minimal test environment. A top-level import would break collection of the
    whole ``tests/runtime`` suite.

    Returns:
        The ``RoboTwinEnv`` class.
    """
    from zetta.envs.robotwin.environment import RoboTwinEnv

    return RoboTwinEnv


def _close_env(env: Any, *, teardown: bool) -> None:
    """Release one family env.

    Args:
        env: The family env, or ``None``.
        teardown: Whether to actually call the family's teardown. When false
            this is a no-op and the SAPIEN subprocesses are left to be reaped
            when the worker process exits -- see :meth:`RobotwinEnvCore.close`
            for why that is the default.
    """
    if env is None or not teardown:
        return
    for method in ("offload", "close"):
        closer = getattr(env, method, None)
        if callable(closer):
            try:
                closer()
                return
            except BaseException:  # noqa: BLE001 - teardown must not mask the real error
                continue


class RobotwinEnvCore:
    """The RoboTwin family's ``EnvExecutionCore`` (blocking/synchronous)."""

    def __init__(self) -> None:
        """Initialize a not-yet-``build``-ed execution core."""
        self.config = RobotwinEnvConfig()
        self.env_spec: EnvSpecMsg | None = None
        self.seed_offset = 0
        self.closed = False
        self.total_chunk_calls = 0
        self.total_env_steps = 0
        self.total_discarded_actions = 0
        self._core_form = PER_SLOT_FORM
        self._lanes: list[LaneState] = []
        self._envs: list[Any] = []

    @property
    def behavior(self) -> EnvFamilyBehavior:
        """The robotwin family's declaration across the six divergence axes.

        Returns:
            The family declaration.
        """
        return behavior_for(ROBOTWIN_ENV_FAMILY)

    @property
    def core_form(self) -> str:
        """This core instance's form.

        Returns:
            Always ``per_slot`` for this family.
        """
        return self._core_form

    # ------------------------------------------------- Construction and release

    def build(
        self,
        env_spec: EnvSpecMsg,
        *,
        num_envs: int,
        seed_offset: int = 0,
        total_num_processes: int = 1,
    ) -> None:
        """Construct one ``RoboTwinEnv`` per slot.

        Args:
            env_spec: The environment spec.
            num_envs: Number of slots in the pool.
            seed_offset: Seed offset for this rank.
            total_num_processes: Total processes sharing the seed pool.

        Raises:
            RuntimeApiError: ``num_envs`` is invalid or exceeds the family's
                declared ceiling (``INVALID_ARGUMENT``), or the family failed
                to construct (``ENV_FAILURE``).
        """
        if num_envs < 1:
            raise RuntimeApiError(
                make_error(
                    ErrorCode.INVALID_ARGUMENT, f"num_envs must be >= 1, got {num_envs}"
                )
            )
        behavior = self.behavior
        if behavior.max_pool_size is not None and num_envs > behavior.max_pool_size:
            raise RuntimeApiError(
                make_error(
                    ErrorCode.INVALID_ARGUMENT,
                    f"robotwin pools are capped at {behavior.max_pool_size} slots: "
                    "SAPIEN raises 'cannot create buffer' once the total number of "
                    "rendered cameras exhausts device memory, and the ceiling is a "
                    "whole-device budget rather than a per-rank one",
                    num_envs=num_envs,
                    max_pool_size=behavior.max_pool_size,
                )
            )
        config = RobotwinEnvConfig.from_mapping(env_spec.env_config)
        core_form = requested_core_form(env_spec, behavior)
        env_class = _robotwin_env_class()
        lanes: list[LaneState] = []
        envs: list[Any] = []
        try:
            for slot_index in range(num_envs):
                envs.append(
                    self._make_env(
                        env_class,
                        config,
                        seed_offset=seed_offset * num_envs + slot_index,
                        total_num_processes=max(1, total_num_processes) * num_envs,
                    )
                )
                lanes.append(LaneState(env_index=slot_index, lane_index=0))
        except RuntimeApiError:
            for env in envs:
                _close_env(env, teardown=config.teardown_on_close)
            raise
        except BaseException as exc:
            for env in envs:
                _close_env(env, teardown=config.teardown_on_close)
            raise RuntimeApiError(
                make_error(
                    ErrorCode.ENV_FAILURE,
                    "failed to build the robotwin env pool: "
                    f"{type(exc).__name__}: {exc}",
                    task_name=config.task_name,
                    num_envs=num_envs,
                    core_form=core_form,
                )
            ) from exc
        self.config = config
        self.env_spec = env_spec
        self.seed_offset = seed_offset
        self._core_form = core_form
        self._lanes = lanes
        self._envs = envs
        self.closed = False

    def _make_env(
        self,
        env_class: type,
        config: RobotwinEnvConfig,
        *,
        seed_offset: int,
        total_num_processes: int,
    ) -> Any:
        """Construct one single-lane ``RoboTwinEnv`` and clear ``is_start``.

        ``is_start`` follows the same reasoning as libero and maniskill: the
        family takes a "decide the reset state myself" branch on the very first
        reset, while the Runtime always supplies explicit parameters, so the
        first episode must follow the same path as every later one.

        Args:
            env_class: ``RoboTwinEnv``.
            config: The family config.
            seed_offset: Seed offset for this slot.
            total_num_processes: Total processes for the seed partition.

        Returns:
            The constructed env.
        """
        env = env_class(
            cfg=config.to_rlinf_cfg(),
            num_envs=1,
            seed_offset=seed_offset,
            total_num_processes=total_num_processes,
            worker_info=None,
        )
        env.is_start = False
        return env

    def close(self) -> None:
        """Drop the pool's references without tearing SAPIEN down.

        ``RoboTwinEnv.offload()`` -> ``VectorEnv.close()`` **segfaults** during
        SAPIEN/subprocess teardown (reproduced on every S3 run: the crash lands
        after the last episode's result and before the summary line, see
        ``plan/robotwin_s0_findings.md``). A segfault cannot be caught in
        process, so calling it would take the whole EnvWorker down every time a
        session pool is released -- turning an ordinary ``close_sessions`` into
        a worker restart.

        So the default is to release the references and let the SAPIEN
        subprocesses be reaped when the worker process exits. That matches how
        the Runtime actually uses this family: pools are pre-allocated per
        ``EnvSpecMsg`` digest and do not grow, so a worker builds a small fixed
        number of them over its lifetime rather than churning through pools.

        The real teardown stays reachable through
        ``env_config.teardown_on_close`` for a RoboTwin build where the crash is
        fixed.
        """
        for env in self._envs:
            _close_env(env, teardown=self.config.teardown_on_close)
        self._lanes = []
        self._envs = []
        self.closed = True

    # ------------------------------------------------------------- Operations

    def reset(self, slots: Sequence[int], reset_spec: ResetSpec) -> list[Observation]:
        """Reset the given slots.

        The family signature is ``reset(env_idx, env_seeds)``. Each slot owns a
        single-lane env, so ``env_idx`` stays ``None`` (meaning "all lanes of
        this env") and the scene is selected purely by ``env_seeds``.

        Seed precedence is ``reset_state_id`` > ``seed`` > the family's own
        schedule. RoboTwin's seed *is* its scene identity, so an explicit
        ``reset_state_id`` must win over the curated success-seed rotation --
        that is what makes a paired same-seed gate reproducible.

        Args:
            slots: The slot indices.
            reset_spec: Episode initialization parameters.

        Returns:
            The initial observations, in the same order as ``slots``.
        """
        observations: list[Observation] = []
        for slot_index in slots:
            lane = self._require_lane(slot_index)
            env = self._envs[lane.env_index]
            env_seeds = self._resolve_seeds(reset_spec)
            payload, _info = env.reset(env_idx=None, env_seeds=env_seeds)
            lane.begin_episode()
            lane.instruction = self._instruction(payload, lane, reset_spec)
            lane.extras = {
                "task_name": self.config.task_name,
                "reset_state_id": (int(env_seeds[0]) if env_seeds is not None else -1),
            }
            observations.append(self._observation(slot_index, payload))
        return observations

    def _resolve_seeds(self, reset_spec: ResetSpec) -> list[int] | None:
        """Pick the seed list for one single-lane reset.

        Args:
            reset_spec: Episode initialization parameters.

        Returns:
            A one-element seed list, or ``None`` to let the family use its own
            schedule.
        """
        if reset_spec.reset_state_id is not None:
            return [int(reset_spec.reset_state_id)]
        if reset_spec.seed is not None:
            return [int(reset_spec.seed)]
        return None

    def observe(self, slots: Sequence[int]) -> list[Observation]:
        """Read the cached observation without changing environment state.

        Args:
            slots: The slot indices.

        Returns:
            Observations in the same order as ``slots``.

        Raises:
            RuntimeApiError: The slot has not been reset yet
                (``SESSION_NOT_READY``).
        """
        observations: list[Observation] = []
        for slot_index in slots:
            lane = self._require_lane(slot_index)
            if lane.last_observation is None:
                raise RuntimeApiError(
                    make_error(
                        ErrorCode.SESSION_NOT_READY,
                        f"robotwin slot {slot_index} has not been reset yet",
                        slot_index=slot_index,
                    )
                )
            observations.append(lane.last_observation)
        return observations

    def lane_status(self, slots: Sequence[int]) -> list[LaneStatus]:
        """Read lane lifecycle snapshots.

        Args:
            slots: The slot indices.

        Returns:
            Snapshots in the same order as ``slots``.
        """
        return lane_statuses(self._lanes, slots)

    def chunk_step(
        self, slots: Sequence[int], chunk_actions: Sequence[np.ndarray]
    ) -> list[ChunkOutcome]:
        """Execute an action chunk on the given slots.

        Args:
            slots: The slot indices.
            chunk_actions: Each slot's ``[chunk, 14]`` actions.

        Returns:
            Normalized results in the same order as ``slots``.

        Raises:
            RuntimeApiError: The number of slots does not match the number of
                action blocks (``INVALID_ARGUMENT``).
        """
        if len(slots) != len(chunk_actions):
            raise RuntimeApiError(
                make_error(
                    ErrorCode.INVALID_ARGUMENT,
                    f"chunk_step got {len(slots)} slots but "
                    f"{len(chunk_actions)} action blocks",
                )
            )
        return [
            self._chunk_step_one(slot_index, actions)
            for slot_index, actions in zip(slots, chunk_actions, strict=True)
        ]

    def extension(
        self, slot: int, namespace: str, method: str, args: dict[str, Any]
    ) -> dict[str, Any]:
        """RoboTwin declares no privileged extensions.

        Args:
            slot: The slot index.
            namespace: Extension namespace.
            method: Extension method name.
            args: Method arguments.

        Returns:
            Never returns.

        Raises:
            RuntimeApiError: Always ``UNSUPPORTED_EXTENSION`` -- the family's
                ``extensions`` set is empty, so this is declaratively rejected
                rather than crashed.
        """
        raise RuntimeApiError(
            make_error(
                ErrorCode.UNSUPPORTED_EXTENSION,
                f"robotwin declares no extensions; got {namespace}.{method}",
                namespace=namespace,
                method=method,
                slot=slot,
            )
        )

    # ---------------------------------------------------------------- Helpers

    def _require_lane(self, slot_index: int) -> LaneState:
        """Look up a lane, rejecting out-of-range slots.

        Args:
            slot_index: The slot index.

        Returns:
            The lane state.

        Raises:
            RuntimeApiError: Index out of range (``INVALID_ARGUMENT``). Pools
                are pre-allocated and do not grow.
        """
        if not 0 <= slot_index < len(self._lanes):
            raise RuntimeApiError(
                make_error(
                    ErrorCode.INVALID_ARGUMENT,
                    f"slot {slot_index} is outside the pool "
                    f"(size {len(self._lanes)}); pools do not grow",
                    slot_index=slot_index,
                    pool_size=len(self._lanes),
                )
            )
        return self._lanes[slot_index]

    def _instruction(
        self, payload: dict[str, Any], lane: LaneState, reset_spec: ResetSpec
    ) -> str:
        """Resolve the task instruction for a lane.

        Args:
            payload: The family's observation dict.
            lane: Lane state.
            reset_spec: Episode initialization parameters.

        Returns:
            The instruction string; falls back to the task name.
        """
        if reset_spec.instruction:
            return str(reset_spec.instruction)
        descriptions = payload.get("task_descriptions") or []
        if lane.lane_index < len(descriptions):
            return str(descriptions[lane.lane_index])
        return self.config.task_name

    def _split_wrist_views(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Unstack the family's wrist tensor onto the Runtime's obs fields (D1).

        The family returns ``wrist_images`` shaped ``[num_envs, n, H, W, C]``
        with ``n`` in ``{1, 2}``, plus ``wrist_image_names`` saying which wrist
        each slice is. The Runtime instead has one ``wrist_image`` and a list of
        ``extra_view_images``, so the left wrist becomes ``wrist_image`` and the
        right wrist becomes ``extra_view_images[0]``.

        The mapping is driven by the **names**, never by position: with a
        right-wrist-only configuration slice 0 is the right wrist, and treating
        it as the left one would hand the policy a mirrored view without any
        error surfacing.

        Args:
            payload: The family's observation dict.

        Returns:
            A shallow copy with ``wrist_images`` / ``extra_view_images``
            rewritten for ``observation_from_payload``.
        """
        rewritten = dict(payload)
        wrist_raw = payload.get("wrist_images")
        if wrist_raw is None:
            rewritten["wrist_images"] = None
            rewritten["extra_view_images"] = None
            return rewritten

        stack = to_numpy(wrist_raw)
        names = list(payload.get("wrist_image_names") or [])
        if not names:
            # No self-description: fall back to the family's documented order
            # (left first, then right) rather than guessing per-slice.
            names = [LEFT_WRIST_KEY, RIGHT_WRIST_KEY][: stack.shape[1]]
        index_by_name = {name: position for position, name in enumerate(names)}

        left_position = index_by_name.get(LEFT_WRIST_KEY)
        right_position = index_by_name.get(RIGHT_WRIST_KEY)
        rewritten["wrist_images"] = (
            stack[:, left_position] if left_position is not None else None
        )
        rewritten["extra_view_images"] = (
            stack[:, right_position] if right_position is not None else None
        )
        return rewritten

    def _observation(self, slot_index: int, payload: dict[str, Any]) -> Observation:
        """Convert a family obs dict into an ``Observation`` and cache it.

        ``observation_from_payload`` casts the state to float32, which RoboTwin
        needs: its ``state`` arrives as float64 and ``obs_schema_digest``
        compares dtypes across families.

        Args:
            slot_index: The slot index.
            payload: The family's observation dict.

        Returns:
            An ``Observation``.
        """
        return observation_from_payload(
            payload=self._split_wrist_views(payload),
            lane=self._lanes[slot_index],
            slot_index=slot_index,
            env_family=ROBOTWIN_ENV_FAMILY,
            core_form=self._core_form,
        )

    def _validate_block(self, slot_index: int, actions: np.ndarray) -> np.ndarray:
        """Validate one slot's action shape and apply the execute horizon.

        Args:
            slot_index: The slot index.
            actions: The actions.

        Returns:
            A ``[executed_chunk, 14]`` float32 array, truncated to
            ``execute_horizon`` when one is configured.

        Raises:
            RuntimeApiError: The shape is wrong (``INVALID_ARGUMENT``).
        """
        block = np.asarray(actions, dtype=np.float32)
        if block.ndim != 2 or block.shape[1] != self.config.action_dim:
            raise RuntimeApiError(
                make_error(
                    ErrorCode.INVALID_ARGUMENT,
                    f"robotwin expects [chunk, {self.config.action_dim}] actions, got "
                    f"shape {tuple(int(dim) for dim in block.shape)}",
                    action_dim=self.config.action_dim,
                    slot_index=slot_index,
                )
            )
        if block.shape[0] < 1:
            raise RuntimeApiError(
                make_error(
                    ErrorCode.INVALID_ARGUMENT,
                    "robotwin chunk must contain at least one action",
                    slot_index=slot_index,
                )
            )
        horizon = self.config.execute_horizon
        if horizon is not None and block.shape[0] > horizon:
            self.total_discarded_actions += int(block.shape[0]) - horizon
            block = block[:horizon]
        return block

    def _prepared_actions(self, batch: np.ndarray) -> np.ndarray:
        """Run the batch through the family's ``prepare_actions``.

        Args:
            batch: Actions shaped ``[1, chunk, 14]``.

        Returns:
            A float32 numpy array in the family's expected layout.
        """
        from zetta.compat.actions import prepare_actions

        prepared = prepare_actions(
            batch,
            env_type=ROBOTWIN_ENV_FAMILY,
            model_type=self.config.action_model_type,
            num_action_chunks=int(batch.shape[1]),
            action_dim=int(batch.shape[2]),
        )
        return np.asarray(prepared, dtype=np.float32)

    def _chunk_step_one(self, slot_index: int, actions: np.ndarray) -> ChunkOutcome:
        """Submit one chunk to one lane and normalize the ``final_only`` result.

        Unlike libero/maniskill this **does** call the family's own
        ``chunk_step``: RoboTwin executes a submitted chunk inside a single
        ``venv.step`` and cannot be driven step by step without changing what
        the simulator does, so there is no per-step frame to collect and no
        early stop to detect mid-chunk. ``executed_horizon`` is therefore the
        submitted chunk length, and ``step_observations`` is ``None``.

        Args:
            slot_index: The slot index.
            actions: Actions shaped ``[chunk, 14]``.

        Returns:
            The normalized result.

        Raises:
            RuntimeApiError: The slot has not been reset
                (``SESSION_NOT_READY``) or the action shape is wrong
                (``INVALID_ARGUMENT``).
        """
        lane = self._require_lane(slot_index)
        lane.chunk_calls += 1
        self.total_chunk_calls += 1
        if not lane.started:
            raise RuntimeApiError(
                make_error(
                    ErrorCode.SESSION_NOT_READY,
                    f"robotwin slot {slot_index} has not been reset yet",
                    slot_index=slot_index,
                )
            )
        requested_horizon = int(np.asarray(actions).shape[0])
        block = self._validate_block(slot_index, actions)
        executed = int(block.shape[0])
        env = self._envs[lane.env_index]

        prepared = self._prepared_actions(block[None, ...])
        obs_list, chunk_rewards, chunk_terms, chunk_truncs, infos_list = env.chunk_step(
            prepared
        )

        rewards = self._lane_row(chunk_rewards, executed, float)
        terminations = self._lane_row(chunk_terms, executed, bool)
        truncations = self._lane_row(chunk_truncs, executed, bool)

        lane.step_index += executed
        lane.env_steps += executed
        lane.terminated = lane.terminated or any(terminations)
        lane.truncated = lane.truncated or any(truncations)
        if lane.terminated or lane.truncated:
            lane.frozen = True
        self.total_env_steps += executed

        payload = obs_list[-1] if obs_list else {}
        observation = self._observation(slot_index, payload)

        info = self._chunk_info(infos_list)
        info["executed_horizon"] = executed
        if executed != requested_horizon:
            # The discard is the configured open-loop replan (D3), not a
            # silent truncation: record it so a rollout record can show it.
            info["discarded_actions"] = requested_horizon - executed

        return normalize_chunk_outcome(
            behavior=self.behavior,
            final_observation=observation,
            # final_only: RoboTwin never produces intermediate frames.
            step_observations=None,
            rewards=rewards,
            terminations=terminations,
            truncations=truncations,
            requested_horizon=requested_horizon,
            info=info,
        )

    @staticmethod
    def _lane_row(matrix: Any, executed: int, caster: Any) -> list[Any]:
        """Read lane 0's row out of a ``[1, chunk]`` family tensor.

        The family returns chunk-shaped reward/flag matrices whose only
        populated column is the last one, so the row maps directly onto the
        per-step sequences ``normalize_chunk_outcome`` expects.

        Args:
            matrix: A ``[1, chunk]`` tensor or array.
            executed: Expected row length.
            caster: ``float`` or ``bool``.

        Returns:
            A list of length ``executed``.
        """
        array = to_numpy(matrix).reshape(1, -1)[0]
        values = [caster(array[index]) for index in range(min(executed, array.size))]
        while len(values) < executed:
            values.append(caster(0))
        return values

    @staticmethod
    def _chunk_info(infos_list: Sequence[Any]) -> dict[str, Any]:
        """Extract a small, JSON-friendly info dict from the family's infos.

        Only the success flag is carried across: the family's info also holds
        whole observation tensors on the auto-reset path, which must never end
        up inside a ``StepResult``.

        Args:
            infos_list: The family's per-chunk info list.

        Returns:
            A plain dict.
        """
        if not infos_list:
            return {}
        raw = infos_list[-1]
        if not isinstance(raw, Mapping):
            return {}
        info: dict[str, Any] = {}
        success = raw.get("success")
        if success is not None:
            array = to_numpy(success).reshape(-1)
            if array.size:
                info["success"] = bool(array[0])
        return info


def robotwin_env_capability() -> EnvFamilyCapability:
    """Return the robotwin family's capability declaration.

    Returns:
        An ``EnvFamilyCapability``: **no** per-step observations
        (``final_only``), needs an accelerator, no privileged extensions,
        ``per_slot`` only, and ``reset_state_id`` **supported** -- RoboTwin's
        ``env_seeds`` is exactly a reset-state selector, which the campaign
        protocol's seed discipline relies on.
    """
    return capability_from_behavior(
        behavior_for(ROBOTWIN_ENV_FAMILY),
        supports_auto_reset=False,
        supports_reset_state_id=True,
    )


class RobotwinEnvFamily:
    """The RoboTwin family's ``EnvFamilyAdapter``."""

    @property
    def env_family(self) -> str:
        """Family name.

        Returns:
            ``"robotwin"``.
        """
        return ROBOTWIN_ENV_FAMILY

    @property
    def capability(self) -> EnvFamilyCapability:
        """The family's capability declaration.

        Returns:
            The capability table entry.
        """
        return robotwin_env_capability()

    def create_core(self) -> RobotwinEnvCore:
        """Create a not-yet-``build``-ed execution core.

        Returns:
            A ``RobotwinEnvCore`` instance.
        """
        return RobotwinEnvCore()


def register_robotwin_env_family(*, replace: bool = True) -> RobotwinEnvFamily:
    """Register the robotwin family into ``ENV_FAMILY_REGISTRY``.

    Args:
        replace: Whether to allow overwriting an existing registration under
            the same name.

    Returns:
        The registered family adapter.
    """
    adapter = RobotwinEnvFamily()
    register_env_family(adapter, replace=replace)
    return adapter
