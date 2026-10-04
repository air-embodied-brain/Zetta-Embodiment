# Copyright (c) 2026 Zetta Contributors
"""BEHAVIOR / OmniGibson Runtime family adapter.

BEHAVIOR is a single-R1Pro, GPU-rendered environment.  It therefore uses the
same ``per_slot`` execution form as RoboCasa and RoboTwin, while returning a
normal per-step observation after every action.  OmniGibson is imported only
when a pool is built; importing Runtime and running the contract suite remains
possible on CPU-only machines.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from rollout_runtime.api.enums import ErrorCode
from rollout_runtime.api.errors import RuntimeApiError, make_error
from rollout_runtime.api.messages import EnvFamilyCapability, EnvSpecMsg, Observation, ResetSpec
from rollout_runtime.backends.rlinf_family import LaneState, LaneStatus, lane_statuses, observation_from_payload
from rollout_runtime.core.env_execution import PER_SLOT_FORM, ChunkOutcome, EnvFamilyBehavior, normalize_chunk_outcome
from rollout_runtime.core.env_registry import (
    BEHAVIOR_ENV_FAMILY,
    behavior_for,
    capability_from_behavior,
    register_env_family,
    requested_core_form,
)
from zetta.envs.behavior.contracts import (
    ACTION_CONTRACT,
    BEHAVIOR_ACTION_DIM,
    DEFAULT_IMAGE_SIZE,
    contract_digest,
    extract_behavior_payload,
    observation_contract,
    validate_behavior_actions,
)

__all__ = [
    "BEHAVIOR_ENV_FAMILY",
    "BehaviorEnvConfig",
    "BehaviorEnvCore",
    "BehaviorEnvFamily",
    "behavior_env_capability",
    "register_behavior_env_family",
]


@dataclasses.dataclass(kw_only=True)
class BehaviorEnvConfig:
    """Family-private configuration for one BEHAVIOR activity."""

    activity_name: str = "put_bowl_on_table"
    scene_model: str = "Rs_int"
    robot_type: str = "R1Pro"
    obs_modalities: tuple[str, ...] = ("rgb", "proprio")
    image_size: int = DEFAULT_IMAGE_SIZE
    main_camera: str = "eyes"
    wrist_cameras: tuple[str, str] = ("left_wrist", "right_wrist")
    state_source: str = "proprio"
    # ``None`` exposes the full R1Pro vector for the Pi0.5 transform. Pi0
    # checkpoints trained on the compact 8-value state can set this to 8.
    state_dim: int | None = None
    action_normalize: bool = True
    max_episode_steps: int = 500
    auto_reset: bool = False
    seed: int = 0
    action_dim: int = BEHAVIOR_ACTION_DIM
    chunk_size: int = 16
    action_model_type: str = "behavior"
    return_all_frames: bool = False
    save_video: bool = False
    env_factory: str = "omnigibson:Environment"
    omnigibson_config: dict[str, Any] = dataclasses.field(default_factory=dict)
    core_form: str = PER_SLOT_FORM

    def __post_init__(self) -> None:
        if self.auto_reset:
            raise ValueError("Runtime owns resets; behavior auto_reset must be false")
        if self.action_dim != BEHAVIOR_ACTION_DIM:
            raise ValueError(f"BEHAVIOR R1Pro requires action_dim={BEHAVIOR_ACTION_DIM}")
        if not isinstance(self.activity_name, str) or not self.activity_name:
            raise ValueError("activity_name must be a non-empty string")
        if not isinstance(self.scene_model, str) or not self.scene_model:
            raise ValueError("scene_model must be a non-empty string")
        if not isinstance(self.robot_type, str) or not self.robot_type:
            raise ValueError("robot_type must be a non-empty string")
        if isinstance(self.obs_modalities, str) or not self.obs_modalities:
            raise ValueError("obs_modalities must be a non-empty sequence")
        if "rgb" not in self.obs_modalities or "proprio" not in self.obs_modalities:
            raise ValueError("BEHAVIOR requires rgb and proprio observations")
        for name in ("image_size", "max_episode_steps", "chunk_size"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if len(self.wrist_cameras) != 2 or any(not name for name in self.wrist_cameras):
            raise ValueError("wrist_cameras must contain two camera names")
        observation_contract(
            image_size=self.image_size,
            main_camera=self.main_camera,
            wrist_cameras=tuple(self.wrist_cameras),
            state_source=self.state_source,
            state_dim=self.state_dim,
        )

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any] | None) -> "BehaviorEnvConfig":
        if not config:
            return cls()
        known = {field.name for field in dataclasses.fields(cls)}
        unknown = sorted(key for key in config if key not in known)
        if unknown:
            raise RuntimeApiError(
                make_error(
                    ErrorCode.INVALID_ARGUMENT,
                    f"unknown behavior env config keys: {unknown}",
                    unknown_keys=unknown,
                    known_keys=sorted(known),
                )
            )
        values = dict(config)
        if "obs_modalities" in values:
            values["obs_modalities"] = tuple(values["obs_modalities"])
        if "wrist_cameras" in values:
            values["wrist_cameras"] = tuple(values["wrist_cameras"])
        try:
            return cls(**values)
        except (TypeError, ValueError) as exc:
            raise RuntimeApiError(make_error(ErrorCode.INVALID_ARGUMENT, str(exc))) from exc

    def observation_digest(self) -> str:
        return contract_digest(
            observation_contract(
                image_size=self.image_size,
                main_camera=self.main_camera,
                wrist_cameras=tuple(self.wrist_cameras),
                state_source=self.state_source,
                state_dim=self.state_dim,
            )
        )

    def environment_config(self) -> dict[str, Any]:
        """Return the lazy wrapper config without mutating the Runtime config."""
        return {
            "activity_name": self.activity_name,
            "scene_model": self.scene_model,
            "robot_type": self.robot_type,
            "obs_modalities": tuple(self.obs_modalities),
            "image_size": self.image_size,
            "main_camera": self.main_camera,
            "wrist_cameras": tuple(self.wrist_cameras),
            "state_source": self.state_source,
            "state_dim": self.state_dim,
            "action_normalize": self.action_normalize,
            "max_episode_steps": self.max_episode_steps,
            "env_factory": self.env_factory,
            "omnigibson_config": dict(self.omnigibson_config),
        }


def _behavior_env_class() -> type:
    """Lazily import the optional environment wrapper."""
    from zetta.envs.behavior.environment import BehaviorEnv

    return BehaviorEnv


class BehaviorEnvCore:
    """Blocking, per-slot execution core for one BEHAVIOR activity."""

    def __init__(self) -> None:
        self.config = BehaviorEnvConfig()
        self.env_spec: EnvSpecMsg | None = None
        self.seed_offset = 0
        self.closed = False
        self.total_chunk_calls = 0
        self.total_env_steps = 0
        self._core_form = PER_SLOT_FORM
        self._lanes: list[LaneState] = []
        self._envs: list[Any] = []

    @property
    def behavior(self) -> EnvFamilyBehavior:
        return behavior_for(BEHAVIOR_ENV_FAMILY)

    @property
    def core_form(self) -> str:
        return self._core_form

    def build(
        self,
        env_spec: EnvSpecMsg,
        *,
        num_envs: int,
        seed_offset: int = 0,
        total_num_processes: int = 1,
    ) -> None:
        del total_num_processes
        if num_envs < 1:
            raise RuntimeApiError(make_error(ErrorCode.INVALID_ARGUMENT, "num_envs must be >= 1"))
        config = BehaviorEnvConfig.from_mapping(env_spec.env_config)
        core_form = requested_core_form(env_spec, self.behavior)
        if core_form != PER_SLOT_FORM:
            raise RuntimeApiError(
                make_error(
                    ErrorCode.INVALID_ARGUMENT,
                    "BEHAVIOR supports only the per_slot execution form",
                    requested_core_form=core_form,
                )
            )
        env_class = _behavior_env_class()
        envs: list[Any] = []
        try:
            for _ in range(num_envs):
                envs.append(env_class(config.environment_config()))
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
                    f"failed to build the BEHAVIOR env pool: {type(exc).__name__}: {exc}",
                    activity_name=config.activity_name,
                    num_envs=num_envs,
                )
            ) from exc
        self.config = config
        self.env_spec = env_spec
        self.seed_offset = seed_offset
        self._core_form = core_form
        self._envs = envs
        self._lanes = [LaneState(env_index=index, lane_index=0) for index in range(num_envs)]
        self.closed = False

    def close(self) -> None:
        for env in self._envs:
            _close_env(env)
        self._envs.clear()
        self._lanes.clear()
        self.closed = True

    def reset(self, slots: Sequence[int], reset_spec: ResetSpec) -> list[Observation]:
        by_slot: dict[int, Observation] = {}
        for slot_index in slots:
            lane = self._require_lane(slot_index)
            env = self._envs[lane.env_index]
            seed = reset_spec.seed
            if seed is None:
                seed = self.config.seed + self.seed_offset + slot_index
            try:
                raw_obs, _info = env.reset(seed=int(seed), options=reset_spec.options)
                payload = extract_behavior_payload(
                    raw_obs,
                    main_camera=self.config.main_camera,
                    wrist_cameras=tuple(self.config.wrist_cameras),
                    image_size=self.config.image_size,
                    state_dim=self.config.state_dim,
                )
            except RuntimeApiError:
                raise
            except BaseException as exc:
                raise RuntimeApiError(
                    make_error(
                        ErrorCode.ENV_FAILURE,
                        f"BEHAVIOR reset failed: {type(exc).__name__}: {exc}",
                        slot_index=slot_index,
                        activity_name=self.config.activity_name,
                    )
                ) from exc
            lane.begin_episode()
            lane.instruction = reset_spec.instruction or str(getattr(env, "instruction", self.config.activity_name))
            lane.extras = {
                "activity_name": self.config.activity_name,
                "scene_model": self.config.scene_model,
                "robot_type": self.config.robot_type,
                "observation_contract": self.config.observation_digest(),
                "action_contract": ACTION_CONTRACT,
                "seed": int(seed),
            }
            lane.step_index = int(getattr(env, "elapsed_steps", 0))
            by_slot[slot_index] = self._observation(slot_index, payload)
        return [by_slot[slot_index] for slot_index in slots]

    def observe(self, slots: Sequence[int]) -> list[Observation]:
        result: list[Observation] = []
        for slot_index in slots:
            lane = self._require_lane(slot_index)
            if lane.last_observation is None:
                raise RuntimeApiError(
                    make_error(
                        ErrorCode.SESSION_NOT_READY,
                        f"behavior slot {slot_index} has not been reset yet",
                        slot_index=slot_index,
                    )
                )
            result.append(lane.last_observation)
        return result

    def lane_status(self, slots: Sequence[int]) -> list[LaneStatus]:
        return lane_statuses(self._lanes, slots)

    def chunk_step(
        self, slots: Sequence[int], chunk_actions: Sequence[np.ndarray]
    ) -> list[ChunkOutcome]:
        if len(slots) != len(chunk_actions):
            raise RuntimeApiError(
                make_error(
                    ErrorCode.INVALID_ARGUMENT,
                    f"chunk_step got {len(slots)} slots but {len(chunk_actions)} action blocks",
                )
            )
        return [
            self._chunk_step_one(slot_index, block)
            for slot_index, block in zip(slots, chunk_actions, strict=True)
        ]

    def extension(self, slot: int, namespace: str, method: str, args: dict[str, Any]) -> dict[str, Any]:
        del slot, args
        raise RuntimeApiError(
            make_error(
                ErrorCode.UNSUPPORTED_EXTENSION,
                f"BEHAVIOR declares no extensions; got {namespace}.{method}",
                namespace=namespace,
                method=method,
                supported=[],
            )
        )

    def _require_lane(self, slot_index: int) -> LaneState:
        if not 0 <= slot_index < len(self._lanes):
            raise RuntimeApiError(
                make_error(
                    ErrorCode.INVALID_ARGUMENT,
                    f"slot {slot_index} is outside the pool (size {len(self._lanes)})",
                    slot_index=slot_index,
                    pool_size=len(self._lanes),
                )
            )
        return self._lanes[slot_index]

    def _observation(self, slot_index: int, payload: dict[str, Any]) -> Observation:
        return observation_from_payload(
            payload=payload,
            lane=self._lanes[slot_index],
            slot_index=slot_index,
            env_family=BEHAVIOR_ENV_FAMILY,
            core_form=self._core_form,
        )

    def _chunk_step_one(self, slot_index: int, actions: np.ndarray) -> ChunkOutcome:
        lane = self._require_lane(slot_index)
        if not lane.started:
            raise RuntimeApiError(
                make_error(
                    ErrorCode.SESSION_NOT_READY,
                    f"behavior slot {slot_index} has not been reset yet",
                    slot_index=slot_index,
                )
            )
        try:
            block = validate_behavior_actions(actions)
        except (TypeError, ValueError) as exc:
            raise RuntimeApiError(
                make_error(ErrorCode.INVALID_ARGUMENT, str(exc), slot_index=slot_index)
            ) from exc
        lane.chunk_calls += 1
        self.total_chunk_calls += 1
        env = self._envs[lane.env_index]
        rewards: list[float] = []
        terminations: list[bool] = []
        truncations: list[bool] = []
        frames: list[Observation] = []
        infos: list[dict[str, Any]] = []
        for action in block:
            if lane.terminated or lane.truncated:
                break
            try:
                raw_obs, reward, terminated, truncated, info = env.step(action)
                payload = extract_behavior_payload(
                    raw_obs,
                    main_camera=self.config.main_camera,
                    wrist_cameras=tuple(self.config.wrist_cameras),
                    image_size=self.config.image_size,
                    state_dim=self.config.state_dim,
                )
            except BaseException as exc:
                raise RuntimeApiError(
                    make_error(
                        ErrorCode.ENV_FAILURE,
                        f"BEHAVIOR step failed: {type(exc).__name__}: {exc}",
                        slot_index=slot_index,
                    )
                ) from exc
            lane.step_index = int(getattr(env, "elapsed_steps", lane.step_index + 1))
            lane.env_steps += 1
            self.total_env_steps += 1
            lane.terminated = bool(terminated)
            lane.truncated = bool(truncated)
            rewards.append(float(reward))
            terminations.append(lane.terminated)
            truncations.append(lane.truncated)
            step_info = dict(info) if isinstance(info, Mapping) else {}
            infos.append(step_info)
            frames.append(self._observation(slot_index, payload))
        if lane.terminated or lane.truncated:
            lane.frozen = True
        final = frames[-1] if frames else lane.last_observation
        if final is None:
            raise RuntimeApiError(make_error(ErrorCode.ENV_FAILURE, "BEHAVIOR produced no observation"))
        return normalize_chunk_outcome(
            behavior=self.behavior,
            final_observation=final,
            step_observations=frames,
            rewards=rewards,
            terminations=terminations,
            truncations=truncations,
            requested_horizon=int(block.shape[0]),
            per_step_info=infos,
            include_step_observations=self.config.return_all_frames,
            info={
                "chunk_calls": lane.chunk_calls,
                "activity_name": self.config.activity_name,
                "core_form": self._core_form,
                "last_info": infos[-1] if infos else {},
            },
        )


def _close_env(env: Any) -> None:
    close = getattr(env, "close", None)
    if callable(close):
        try:
            close()
        except BaseException:  # noqa: BLE001 - cleanup must not mask the original error
            pass


def behavior_env_capability() -> EnvFamilyCapability:
    return capability_from_behavior(
        behavior_for(BEHAVIOR_ENV_FAMILY),
        supports_auto_reset=False,
        supports_reset_state_id=False,
    )


class BehaviorEnvFamily:
    @property
    def env_family(self) -> str:
        return BEHAVIOR_ENV_FAMILY

    @property
    def capability(self) -> EnvFamilyCapability:
        return behavior_env_capability()

    def create_core(self) -> BehaviorEnvCore:
        return BehaviorEnvCore()


def register_behavior_env_family(*, replace: bool = True) -> BehaviorEnvFamily:
    adapter = BehaviorEnvFamily()
    register_env_family(adapter, replace=replace)
    return adapter
