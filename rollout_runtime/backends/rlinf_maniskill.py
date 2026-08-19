"""ManiSkill family adapter.

**Does not reimplement rlinf's env**: ``rlinf.envs.maniskill.maniskill_env.ManiskillEnv``
is reused as-is, along with the maniskill branch of
``envs.action_utils.prepare_actions`` and the 5-key schema of ``_wrap_obs``. This
module does exactly three things: translate Runtime's session/slot semantics into
family calls, normalize family output through
``core.env_execution.normalize_chunk_outcome`` (the single output point), and build
pools according to the two ``core_form`` options.

Its four points of divergence from libero (all purely **declarative** — the layer
above never needs an if):

1. The ``reset`` signature is ``reset(seed=..., options=...)`` (``seed_options``),
   with subset reset relying on ``options["env_idx"]``;
2. ``device_kind="gpu_batched"``: obs / reward / termination flags are all torch
   tensors on the GPU, so ``needs_accelerator`` is true and the EnvWorker must
   allocate a GPU;
3. ``action_layout="torch_env_chunk_dim"``: actions are converted to a device tensor
   before being fed to the family;
4. No privileged extensions (the five under D8 are LIBERO-exclusive); any
   undeclared method uniformly gets ``UNSUPPORTED_EXTENSION``.

``wrap_obs_mode`` defaults to ``"simple"``: rlinf's default branch reads
``sensor_data["3rd_view_camera"]`` (only present for SimplerEnv / bridge tasks),
while generic ManiSkill tasks (e.g. ``PickCube-v1``) name their camera
``base_camera``, which is exactly what the ``"simple"`` branch reads.

Dependency surface: rlinf + mani_skill + torch + numpy, all **lazily imported
inside functions**.
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
    run_lockstep_chunk,
    to_numpy,
    to_scalar,
)
from rollout_runtime.core.env_execution import (
    LOCKSTEP_VECTOR_FORM,
    PER_SLOT_FORM,
    ChunkOutcome,
    EnvFamilyBehavior,
    normalize_chunk_outcome,
)
from rollout_runtime.core.env_registry import (
    MANISKILL_ENV_FAMILY,
    behavior_for,
    capability_from_behavior,
    register_env_family,
    requested_core_form,
)

__all__ = [
    "MANISKILL_ENV_FAMILY",
    "ManiskillEnvConfig",
    "ManiskillEnvCore",
    "ManiskillEnvFamily",
    "maniskill_env_capability",
    "register_maniskill_env_family",
]


@dataclasses.dataclass(kw_only=True)
class ManiskillEnvConfig:
    """ManiSkill family private config (corresponds to ``EnvSpecMsg.env_config``).

    Attributes:
        env_id: The ManiSkill task id (the ``id`` for ``gym.make``).
        obs_mode: ManiSkill observation mode; ``wrap_obs_mode="simple"`` only
            recognizes ``state`` / ``rgb``.
        control_mode: Control mode; ``pd_ee_delta_pose`` is 7-dimensional
            (3 translation + 3 rotation + 1 gripper).
        sim_backend: ``"gpu"`` / ``"physx_cuda"`` etc., left to ManiSkill to
            interpret.
        robot_uids: Robot model; ``None`` uses the task's default.
        camera_height: Sensor camera height.
        camera_width: Sensor camera width.
        wrap_obs_mode: rlinf ``_wrap_obs``'s branch (``"simple"`` reads
            ``base_camera``).
        reward_mode: rlinf ``_calc_step_reward``'s branch; ``"raw"`` uses the
            environment reward directly (the default branch needs SimplerEnv-only
            info keys like ``is_src_obj_grasped``).
        use_full_state: Whether the ``"simple"`` branch's state takes the full
            state.
        max_episode_steps: The outer truncation step count (fed into
            ``gym.make``, enforced by gymnasium's TimeLimit).
        auto_reset: Whether to let the family auto-reset itself; Runtime drives
            resets via sessions, so this is fixed to false.
        ignore_terminations: Whether to ignore the environment's termination
            signal.
        use_rel_reward: Whether to use relative reward.
        use_fixed_reset_state_ids: Whether to use the family's fixed reset-state
            pool.
        group_size: rlinf's grouping size; Runtime uses one session per lane, so
            this is fixed to 1.
        seed: The family's base seed; the actual seed still adds
            ``seed_offset``.
        action_dim: Action dimension.
        chunk_size: The declared action-chunk length (actual execution follows
            the number of rows in the supplied actions).
        action_model_type: The model type that produced the actions, fed into
            ``prepare_actions``.
        action_policy: ``prepare_actions``'s ``policy`` parameter. Values
            containing ``panda`` take the identity branch
            (``action_utils.py``'s ``if "panda" in policy: return``), which is
            correct for the case where "the model already outputs actions in the
            environment's action space".
        action_scale: ``prepare_actions``'s scaling (only takes effect on the
            non-identity branch).
        return_all_frames: Whether ``per_step`` includes per-step observations
            (the payload is large, so this is off by default).
        save_video: Whether to have the family write video.
        core_form: The execution core form (``core.env_execution.CORE_FORMS``).
        extra_init_params: Extra key-value pairs passed through to
            ``gym.make``.
    """

    env_id: str = "PickCube-v1"
    obs_mode: str = "rgb"
    control_mode: str = "pd_ee_delta_pose"
    sim_backend: str = "gpu"
    robot_uids: str | None = None
    camera_height: int = 128
    camera_width: int = 128
    wrap_obs_mode: str = "simple"
    reward_mode: str = "raw"
    use_full_state: bool = False
    max_episode_steps: int = 100
    auto_reset: bool = False
    ignore_terminations: bool = False
    use_rel_reward: bool = False
    use_fixed_reset_state_ids: bool = False
    group_size: int = 1
    seed: int = 0
    action_dim: int = 7
    chunk_size: int = 4
    action_model_type: str = "openpi"
    action_policy: str = "panda_wristcam"
    action_scale: float = 1.0
    return_all_frames: bool = False
    save_video: bool = False
    core_form: str = PER_SLOT_FORM
    extra_init_params: dict[str, Any] = dataclasses.field(default_factory=dict)

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any] | None) -> ManiskillEnvConfig:
        """Construct a config from an ``env_config`` dict.

        Unknown keys are always rejected: ``env_config`` feeds into
        ``EnvSpecMsg.digest()``, and a typo'd key would silently produce a new
        pool (for a GPU family, that means an extra sapien scene), which is far
        harder to diagnose than an error.

        Args:
            config: The family-private config; ``None`` means all defaults.

        Returns:
            The structured config.

        Raises:
            RuntimeApiError: An unknown key is present, or ``group_size`` is not
                1 (``INVALID_ARGUMENT``).
        """
        if not config:
            return cls()
        known = {field.name for field in dataclasses.fields(cls)}
        unknown = sorted(key for key in config if key not in known)
        if unknown:
            raise RuntimeApiError(
                make_error(
                    ErrorCode.INVALID_ARGUMENT,
                    f"unknown maniskill env config keys: {unknown}",
                    unknown_keys=unknown,
                    known_keys=sorted(known),
                )
            )
        instance = cls(**dict(config))
        if instance.group_size != 1:
            raise RuntimeApiError(
                make_error(
                    ErrorCode.INVALID_ARGUMENT,
                    "maniskill group_size must be 1: Runtime binds one session per lane, "
                    "so rlinf's group replication has no meaning here",
                    group_size=instance.group_size,
                )
            )
        return instance

    def to_rlinf_cfg(self, *, num_envs: int) -> Any:
        """Project into the omegaconf config needed by ``ManiskillEnv(cfg=...)``.

        Args:
            num_envs: The number of lanes for this env (``ManiskillEnv`` writes
                it into ``init_params``).

        Returns:
            A ``DictConfig``.

        Raises:
            RuntimeApiError: ``omegaconf`` is unavailable (``ENV_FAILURE``).
        """
        try:
            from omegaconf import OmegaConf
        except ImportError as exc:  # pragma: no cover - always installed in production
            raise RuntimeApiError(
                make_error(
                    ErrorCode.ENV_FAILURE,
                    "omegaconf is required to build the maniskill env cfg",
                )
            ) from exc
        init_params: dict[str, Any] = {
            "id": self.env_id,
            "obs_mode": self.obs_mode,
            "control_mode": self.control_mode,
            "sim_backend": self.sim_backend,
            "num_envs": int(num_envs),
            "sensor_configs": {
                "height": int(self.camera_height),
                "width": int(self.camera_width),
            },
            # Must go into init_params: ``ManiskillEnv`` never reads
            # ``cfg.max_episode_steps``; truncation is enforced by gymnasium's
            # TimeLimit wrapper via ``gym.make(max_episode_steps=...)``. Setting
            # it only in cfg would be a config value with no effect (found by an
            # independent audit).
            "max_episode_steps": int(self.max_episode_steps),
            **dict(self.extra_init_params),
        }
        if self.robot_uids:
            init_params["robot_uids"] = self.robot_uids
        return OmegaConf.create(
            {
                "env_type": MANISKILL_ENV_FAMILY,
                "seed": int(self.seed),
                "auto_reset": bool(self.auto_reset),
                "ignore_terminations": bool(self.ignore_terminations),
                "use_rel_reward": bool(self.use_rel_reward),
                "use_full_state": bool(self.use_full_state),
                "use_fixed_reset_state_ids": bool(self.use_fixed_reset_state_ids),
                "group_size": int(self.group_size),
                "wrap_obs_mode": self.wrap_obs_mode,
                "reward_mode": self.reward_mode,
                "max_episode_steps": int(self.max_episode_steps),
                "video_cfg": {
                    "save_video": bool(self.save_video),
                    "info_on_video": bool(self.save_video),
                    "video_base_dir": "/tmp/rr_maniskill_video",
                },
                "init_params": init_params,
            }
        )


def _maniskill_env_class() -> type:
    """Lazily fetch ``ManiskillEnv``.

    The class is not imported at module top level: the local
    ``.venv-runtime`` has no rlinf / mani_skill, and a top-level import would
    make the whole ``tests/runtime`` suite fail even at collection time.

    Returns:
        The ``ManiskillEnv`` class.
    """
    from zetta.envs.maniskill.environment import ManiskillEnv

    return ManiskillEnv


class ManiskillEnvCore:
    """The ManiSkill family's ``EnvExecutionCore`` (blocking/synchronous, driven
    by ``asyncio.to_thread``)."""

    def __init__(self) -> None:
        """Initialize a not-yet-``build``-ed execution core."""
        self.config = ManiskillEnvConfig()
        self.env_spec: EnvSpecMsg | None = None
        self.seed_offset = 0
        self.closed = False
        self.total_chunk_calls = 0
        self.total_env_steps = 0
        self.total_masked_steps = 0
        self.coalesced_group_count = 0
        self._core_form = PER_SLOT_FORM
        self._lanes: list[LaneState] = []
        self._envs: list[Any] = []

    @property
    def behavior(self) -> EnvFamilyBehavior:
        """The maniskill family's declaration on the six divergence points.

        Returns:
            The family declaration.
        """
        return behavior_for(MANISKILL_ENV_FAMILY)

    @property
    def core_form(self) -> str:
        """This core instance's form.

        Returns:
            ``per_slot`` or ``lockstep_vector``.
        """
        return self._core_form

    # -------------------------------------------------------------- Construction and release

    def build(
        self,
        env_spec: EnvSpecMsg,
        *,
        num_envs: int,
        seed_offset: int = 0,
        total_num_processes: int = 1,
    ) -> None:
        """Construct an env pool according to the spec.

        Args:
            env_spec: The environment spec (``env_config.core_form`` selects the
                form).
            num_envs: The number of slots in the pool (baked in at construction
                time).
            seed_offset: The seed offset for this rank.
            total_num_processes: Total number of processes participating in the
                split.

        Raises:
            RuntimeApiError: ``num_envs`` is invalid or the form is unsupported
                (``INVALID_ARGUMENT``), or the family failed to construct
                (``ENV_FAILURE``).
        """
        if num_envs < 1:
            raise RuntimeApiError(
                make_error(
                    ErrorCode.INVALID_ARGUMENT, f"num_envs must be >= 1, got {num_envs}"
                )
            )
        config = ManiskillEnvConfig.from_mapping(env_spec.env_config)
        core_form = requested_core_form(env_spec, self.behavior)
        env_class = _maniskill_env_class()
        lanes: list[LaneState] = []
        envs: list[Any] = []
        try:
            if core_form == LOCKSTEP_VECTOR_FORM:
                envs.append(
                    self._make_env(
                        env_class,
                        config,
                        num_envs=num_envs,
                        seed_offset=seed_offset,
                        total_num_processes=max(1, total_num_processes),
                    )
                )
                lanes = [
                    LaneState(env_index=0, lane_index=lane) for lane in range(num_envs)
                ]
            else:
                for slot_index in range(num_envs):
                    envs.append(
                        self._make_env(
                            env_class,
                            config,
                            num_envs=1,
                            seed_offset=seed_offset * num_envs + slot_index,
                            total_num_processes=max(1, total_num_processes) * num_envs,
                        )
                    )
                    lanes.append(LaneState(env_index=slot_index, lane_index=0))
        except RuntimeApiError:
            for env in envs:
                _close_env(env)
            raise
        except BaseException as exc:
            for env in envs:
                _close_env(env)
            raise RuntimeApiError(
                make_error(
                    ErrorCode.ENV_FAILURE,
                    "failed to build the maniskill env pool: "
                    f"{type(exc).__name__}: {exc}",
                    env_id=config.env_id,
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
        config: ManiskillEnvConfig,
        *,
        num_envs: int,
        seed_offset: int,
        total_num_processes: int,
    ) -> Any:
        """Construct one ``ManiskillEnv`` and clear ``is_start``.

        ``is_start`` follows the same reasoning as in libero: the family takes
        a "decide the reset state itself" branch on the first reset, while
        Runtime always supplies explicit parameters on every reset, so the
        first episode must follow the same path as subsequent ones.

        Args:
            env_class: ``ManiskillEnv``.
            config: The family config.
            num_envs: The number of lanes for this env.
            seed_offset: The seed offset.
            total_num_processes: Total number of processes.

        Returns:
            The constructed env.
        """
        env = env_class(
            cfg=config.to_rlinf_cfg(num_envs=num_envs),
            num_envs=num_envs,
            seed_offset=seed_offset,
            total_num_processes=total_num_processes,
            worker_info=None,
        )
        env.is_start = False
        return env

    def close(self) -> None:
        """Release every sapien scene."""
        for env in self._envs:
            _close_env(env)
        self._lanes = []
        self._envs = []
        self.closed = True

    # ------------------------------------------------------------------ Operations

    def reset(self, slots: Sequence[int], reset_spec: ResetSpec) -> list[Observation]:
        """Reset the given slots (the family signature is
        ``reset(seed=, options=)``).

        Subset reset goes through ``options["env_idx"]`` — ManiSkill's partial
        reset entry point, which is **a different signature** from libero's
        positional ``env_idx`` argument, hence the two separate entries in the
        declaration table.

        Args:
            slots: The slot indices.
            reset_spec: Episode initialization parameters (``seed`` /
                ``reset_state_id``).

        Returns:
            The initial observations, in the same order as ``slots``.
        """
        import torch

        by_slot: dict[int, Observation] = {}
        groups: dict[int, list[int]] = {}
        for slot_index in slots:
            lane = self._require_lane(slot_index)
            groups.setdefault(lane.env_index, []).append(slot_index)
        for env_index, slot_list in groups.items():
            env = self._envs[env_index]
            lane_indices = [self._lanes[index].lane_index for index in slot_list]
            seed = int(reset_spec.seed) if reset_spec.seed is not None else None
            options: dict[str, Any] = {
                "env_idx": torch.tensor(
                    lane_indices, dtype=torch.long, device=env.device
                )
            }
            if reset_spec.reset_state_id is not None:
                options["episode_id"] = torch.tensor(
                    [int(reset_spec.reset_state_id)] * len(lane_indices),
                    dtype=torch.long,
                    device=env.device,
                )
            payload, _info = env.reset(
                # The seed is computed per **lane**, not per "position within
                # this request": the Gateway resets sessions one at a time, so
                # the same ResetSpec must give the same lane the same seed
                # regardless of how it is batched (found by an independent
                # audit: previously reset([0,1,2]) gave [7,8,9] while
                # reset([1]) gave [7]).
                seed=(
                    [seed + int(lane) for lane in lane_indices]
                    if seed is not None
                    else None
                ),
                options=options,
            )
            for slot_index in slot_list:
                lane = self._lanes[slot_index]
                lane.begin_episode()
                lane.instruction = self._instruction(env, lane, reset_spec)
                lane.extras = {
                    "env_id": self.config.env_id,
                    "reset_state_id": (
                        int(reset_spec.reset_state_id)
                        if reset_spec.reset_state_id is not None
                        else -1
                    ),
                }
                by_slot[slot_index] = self._observation(slot_index, payload)
        return [by_slot[slot_index] for slot_index in slots]

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
                        f"maniskill slot {slot_index} has not been reset yet",
                        slot_index=slot_index,
                    )
                )
            observations.append(lane.last_observation)
        return observations

    def lane_status(self, slots: Sequence[int]) -> list[LaneStatus]:
        """Read lane lifecycle snapshots (``LaneStatusReader``, the readback
        channel for masking semantics 2).

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
            chunk_actions: Each slot's ``[chunk, action_dim]`` actions.

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
        if self._core_form == LOCKSTEP_VECTOR_FORM:
            return self._chunk_step_lockstep(slots, chunk_actions)
        return [
            self._chunk_step_one(slot_index, actions)
            for slot_index, actions in zip(slots, chunk_actions, strict=True)
        ]

    def extension(
        self, slot: int, namespace: str, method: str, args: dict[str, Any]
    ) -> dict[str, Any]:
        """Maniskill declares no privileged extensions.

        Args:
            slot: The slot index.
            namespace: Extension namespace.
            method: Extension method name.
            args: Method arguments.

        Returns:
            Never returns.

        Raises:
            RuntimeApiError: Always ``UNSUPPORTED_EXTENSION`` — maniskill's
                ``extensions`` in the declaration table is an empty set, so this
                is "declaratively rejected" rather than "crashed".
        """
        raise RuntimeApiError(
            make_error(
                ErrorCode.UNSUPPORTED_EXTENSION,
                f"maniskill declares no extensions; got {namespace}.{method}",
                namespace=namespace,
                method=method,
                supported=[],
            )
        )

    # ------------------------------------------------------------------ Internal

    def _require_lane(self, slot_index: int) -> LaneState:
        """Get lane state and validate the index.

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
                    f"(size {len(self._lanes)}); pools do not grow (plan D6)",
                    slot_index=slot_index,
                    pool_size=len(self._lanes),
                )
            )
        return self._lanes[slot_index]

    def _instruction(self, env: Any, lane: LaneState, reset_spec: ResetSpec) -> str:
        """Get the task instruction.

        Args:
            env: The family env.
            lane: Lane state.
            reset_spec: Episode initialization parameters.

        Returns:
            The instruction string; falls back to the task id when the family
            has no language instruction.
        """
        if reset_spec.instruction:
            return str(reset_spec.instruction)
        try:
            instruction = env.instruction
        except BaseException:  # noqa: BLE001 - most ManiSkill tasks have no language instruction
            return self.config.env_id
        if isinstance(instruction, str):
            return instruction
        if isinstance(instruction, Sequence) and instruction:
            index = min(lane.lane_index, len(instruction) - 1)
            return str(instruction[index])
        return self.config.env_id

    def _observation(self, slot_index: int, payload: dict[str, Any]) -> Observation:
        """Convert a family obs dict into an ``Observation`` and update the
        cache.

        Args:
            slot_index: The slot index.
            payload: The output of ``ManiskillEnv._wrap_obs``.

        Returns:
            An ``Observation`` (``session_id`` / ``episode_id`` are stamped by
            the worker).
        """
        return observation_from_payload(
            payload=payload,
            lane=self._lanes[slot_index],
            slot_index=slot_index,
            env_family=MANISKILL_ENV_FAMILY,
            core_form=self._core_form,
        )

    def _prepared_actions(self, batch: np.ndarray, chunk_len: int) -> Any:
        """Run through the family's ``prepare_actions`` and convert to a device
        tensor.

        Args:
            batch: The actions, shape ``[num_envs, chunk, action_dim]``.
            chunk_len: The chunk length.

        Returns:
            A float32 tensor on the device.
        """
        import torch
        from zetta.compat.actions import prepare_actions

        prepared = prepare_actions(
            batch,
            env_type=MANISKILL_ENV_FAMILY,
            model_type=self.config.action_model_type,
            num_action_chunks=chunk_len,
            action_dim=int(batch.shape[2]),
            action_scale=float(self.config.action_scale),
            policy=self.config.action_policy,
        )
        device = self._envs[0].device
        if isinstance(prepared, torch.Tensor):
            return prepared.to(device=device, dtype=torch.float32)
        return torch.as_tensor(
            np.asarray(prepared, dtype=np.float32), device=device, dtype=torch.float32
        )

    def _validate_block(self, slot_index: int, actions: np.ndarray) -> np.ndarray:
        """Validate a single slot's action shape.

        Args:
            slot_index: The slot index.
            actions: The actions.

        Returns:
            A ``[chunk, action_dim]`` float32 array.

        Raises:
            RuntimeApiError: The shape is wrong (``INVALID_ARGUMENT``).
        """
        block = np.asarray(actions, dtype=np.float32)
        if block.ndim != 2 or block.shape[1] != self.config.action_dim:
            raise RuntimeApiError(
                make_error(
                    ErrorCode.INVALID_ARGUMENT,
                    f"maniskill expects [chunk, {self.config.action_dim}] actions, got "
                    f"shape {tuple(int(dim) for dim in block.shape)}",
                    action_dim=self.config.action_dim,
                    slot_index=slot_index,
                )
            )
        return block

    def _chunk_step_one(self, slot_index: int, actions: np.ndarray) -> ChunkOutcome:
        """``per_slot`` form: drive one lane step by step, stopping at the first
        termination signal.

        Same reasoning as libero: does not call the family's own
        ``chunk_step``, because it has no early stop, and
        ``executed_horizon`` must be the **actual** number of steps.

        Args:
            slot_index: The slot index.
            actions: Actions, shape ``[chunk, action_dim]``.

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
                    f"maniskill slot {slot_index} has not been reset yet",
                    slot_index=slot_index,
                )
            )
        block = self._validate_block(slot_index, actions)
        env = self._envs[lane.env_index]
        prepared = self._prepared_actions(block[None, ...], int(block.shape[0]))
        rewards: list[float] = []
        terminations: list[bool] = []
        truncations: list[bool] = []
        per_step_info: list[dict[str, Any]] = []
        frames: list[Observation] = []
        payload: dict[str, Any] | None = None
        for index in range(int(prepared.shape[1])):
            if lane.terminated or lane.truncated:
                break
            payload, reward, terminated, truncated, _info = env.step(
                prepared[:, index], auto_reset=False
            )
            lane.step_index = int(to_numpy(env.elapsed_steps).reshape(-1)[0])
            lane.env_steps += 1
            self.total_env_steps += 1
            lane.terminated = bool(to_scalar(terminated))
            lane.truncated = bool(to_scalar(truncated))
            rewards.append(to_scalar(reward))
            terminations.append(lane.terminated)
            truncations.append(lane.truncated)
            per_step_info.append({"step_index": lane.step_index})
            frames.append(self._observation(slot_index, payload))
        if lane.terminated or lane.truncated:
            lane.frozen = True
        final = (
            frames[-1]
            if frames
            else (lane.last_observation or self._observation(slot_index, payload or {}))
        )
        return normalize_chunk_outcome(
            behavior=self.behavior,
            final_observation=final,
            step_observations=frames,
            rewards=rewards,
            terminations=terminations,
            truncations=truncations,
            requested_horizon=int(block.shape[0]),
            per_step_info=per_step_info,
            include_step_observations=self.config.return_all_frames,
            info={
                "chunk_calls": lane.chunk_calls,
                "env_id": self.config.env_id,
                "core_form": self._core_form,
            },
        )

    def _chunk_step_lockstep(
        self, slots: Sequence[int], chunk_actions: Sequence[np.ndarray]
    ) -> list[ChunkOutcome]:
        """``lockstep_vector`` form: merge same-pool same-tick calls into one
        GPU advance step.

        Args:
            slots: The slots participating in this group.
            chunk_actions: Action blocks, in the same order as ``slots``.

        Returns:
            Normalized results in the same order as ``slots``.
        """
        for slot_index in slots:
            self._require_lane(slot_index)
        self.total_chunk_calls += 1
        self.coalesced_group_count += 1
        env = self._envs[0]

        def _step(actions: Any) -> tuple[Any, Any, Any, Any, Any]:
            self.total_env_steps += 1
            return env.step(actions, auto_reset=False)

        def _elapsed(lane_index: int) -> int:
            return int(to_numpy(env.elapsed_steps).reshape(-1)[lane_index])

        outcomes, _stats = run_lockstep_chunk(
            behavior=self.behavior,
            core_form=self._core_form,
            lanes=self._lanes,
            slots=list(slots),
            blocks=list(chunk_actions),
            action_dim=int(self.config.action_dim),
            hold_action=np.zeros(int(self.config.action_dim), dtype=np.float32),
            prepare=self._prepared_actions,
            step=_step,
            elapsed_steps=_elapsed,
            observe=self._observation,
            include_step_observations=self.config.return_all_frames,
            chunk_info=lambda slot_index: {
                "chunk_calls": self._lanes[slot_index].chunk_calls,
                "env_id": self.config.env_id,
                "masked_steps": self._lanes[slot_index].masked_steps,
            },
        )
        self.total_masked_steps = sum(lane.masked_steps for lane in self._lanes)
        return outcomes


def _close_env(env: Any) -> None:
    """Best-effort close of one ``ManiskillEnv``; never raises.

    Args:
        env: The ``ManiskillEnv`` instance.
    """
    inner = getattr(env, "env", None)
    for candidate in (inner, env):
        close = getattr(candidate, "close", None)
        if callable(close):
            try:
                close()
            except BaseException:  # noqa: BLE001 - cleanup must not raise
                pass
            return


def maniskill_env_capability() -> EnvFamilyCapability:
    """Return the maniskill family's capability declaration.

    Returns:
        An ``EnvFamilyCapability``: per-step obs available, **needs an
        accelerator** (GPU-batched), no privileged extensions, both
        ``core_form`` options supported, and ``reset_state_id`` **not**
        supported.
    """
    return capability_from_behavior(
        behavior_for(MANISKILL_ENV_FAMILY),
        supports_auto_reset=False,
        # **Not declared**: ``reset_state_id``. The adapter can only pass it
        # through as ``options["episode_id"]``, and only SimplerEnv-style
        # tasks (with `total_num_trials` / `xyz_configs`) actually read it;
        # generic ManiSkill tasks (e.g. the preset's ``PickCube-v1``) silently
        # ignore it. Declaring a capability that "doesn't actually take hold"
        # is far worse than not declaring it (found by an independent audit).
        supports_reset_state_id=False,
    )


class ManiskillEnvFamily:
    """The ManiSkill family's ``EnvFamilyAdapter``."""

    @property
    def env_family(self) -> str:
        """Family name.

        Returns:
            ``"maniskill"``.
        """
        return MANISKILL_ENV_FAMILY

    @property
    def capability(self) -> EnvFamilyCapability:
        """The family's capability declaration.

        Returns:
            The capability table entry.
        """
        return maniskill_env_capability()

    def create_core(self) -> ManiskillEnvCore:
        """Create a not-yet-``build``-ed execution core.

        Returns:
            A ``ManiskillEnvCore`` instance.
        """
        return ManiskillEnvCore()


def register_maniskill_env_family(*, replace: bool = True) -> ManiskillEnvFamily:
    """Register the maniskill family into ``ENV_FAMILY_REGISTRY``.

    Args:
        replace: Whether to allow overwriting an existing registration under
            the same name.

    Returns:
        The registered family adapter.
    """
    adapter = ManiskillEnvFamily()
    register_env_family(adapter, replace=replace)
    return adapter
