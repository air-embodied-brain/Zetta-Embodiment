# Copyright (c) 2026 Zetta Contributors
"""Runtime adapter for the pinned Genie Sim G2 pick-block task.

Each core owns one supervised Isaac process. Per-step records contain the
observed joint state; only the final observation carries all three images.
"""

from __future__ import annotations

import threading
from collections.abc import Sequence
from typing import Any

import numpy as np

from rollout_runtime.api.enums import ErrorCode
from rollout_runtime.api.errors import RuntimeApiError, make_error
from rollout_runtime.api.ids import EpisodeId, SessionId
from rollout_runtime.api.messages import EnvSpecMsg, Observation, ResetSpec
from rollout_runtime.core.env_execution import PER_SLOT_FORM, normalize_chunk_outcome
from rollout_runtime.core.env_registry import (
    GENIESIM_ENV_FAMILY,
    behavior_for,
    capability_from_behavior,
    register_env_family,
)
from rollout_runtime.core.payload import encode_image
from zetta.envs.geniesim.config import (
    CAMERA_NAMES,
    STATE_GROUPS,
    GenieSimConfig,
    flatten_state,
)
from zetta.envs.geniesim.process import GenieSimProcess, GenieSimProcessError
from zetta.envs.geniesim.session import validate_actions


def _invalid(message: str) -> RuntimeApiError:
    return RuntimeApiError(make_error(ErrorCode.INVALID_ARGUMENT, message))


class GenieSimEnvCore:
    def __init__(self) -> None:
        self.config = None
        self.process = None
        self._observation = None
        self._limits = None
        self._result = None
        self._seed = None
        self._finished = False
        self._lock = threading.RLock()

    @property
    def core_form(self):
        return PER_SLOT_FORM

    @property
    def behavior(self):
        return behavior_for(GENIESIM_ENV_FAMILY)

    def build(
        self,
        env_spec: EnvSpecMsg,
        *,
        num_envs: int,
        seed_offset: int = 0,
        total_num_processes: int = 1,
    ) -> None:
        del seed_offset, total_num_processes
        if num_envs != 1:
            raise _invalid("geniesim supports exactly one slot per pool")
        if self.process is not None:
            raise _invalid("geniesim core is already built")
        try:
            self.config = GenieSimConfig.from_mapping(env_spec.env_config)
        except ValueError as error:
            raise _invalid(str(error)) from error
        self._start()

    def _start(self) -> None:
        try:
            self.process = GenieSimProcess(self.config)
        except GenieSimProcessError as error:
            raise RuntimeApiError(
                make_error(ErrorCode.ENV_FAILURE, str(error))
            ) from error

    def _slots(self, slots: Sequence[int], *, started: bool = True) -> None:
        if list(slots) != [0]:
            raise _invalid("geniesim operations require the single slot [0]")
        if self.process is None or (started and self._observation is None):
            raise RuntimeApiError(
                make_error(ErrorCode.SESSION_NOT_READY, "geniesim requires reset")
            )

    def _request(
        self, method: str, payload: dict[str, Any], **kwargs
    ) -> dict[str, Any]:
        try:
            result = self.process.request(method, payload, **kwargs)
        except GenieSimProcessError as error:
            self._observation = None
            raise RuntimeApiError(
                make_error(ErrorCode.ENV_FAILURE, str(error))
            ) from error
        try:
            result["observation"] = self._convert(result["observation"])
            for name in ("terminated", "truncated", "success"):
                if type(result["result"][name]) is not bool:
                    raise ValueError(f"invalid native result flag {name}")
            if result["result"]["control_steps"] != result["observation"].step_index:
                raise ValueError("native result and observation step counters disagree")
            if method == "reset":
                limits = np.asarray(result["action_limits"], dtype=np.float64)
                validate_actions(limits.mean(axis=1)[None, :], limits)
                if result["observation"].step_index != 0:
                    raise ValueError("native reset returned a nonzero step counter")
            else:
                steps = result["steps"]
                if not 1 <= len(steps) <= len(payload["actions"]):
                    raise ValueError("native result has an invalid executed horizon")
                for offset, step in enumerate(steps, 1):
                    if step["step_index"] != self._observation.step_index + offset:
                        raise ValueError("native per-step counter is not contiguous")
                if steps[-1]["step_index"] != result["observation"].step_index:
                    raise ValueError(
                        "native final observation has the wrong step counter"
                    )
                result["step_observations"] = [
                    self._convert(step, include_images=False) for step in steps
                ]
            return result
        except (KeyError, TypeError, ValueError, IndexError) as error:
            self.process.abort(f"invalid {method} response: {error}")
            self._observation = None
            raise RuntimeApiError(
                make_error(
                    ErrorCode.ENV_FAILURE, f"invalid native {method} response: {error}"
                )
            ) from error

    def reset(self, slots: Sequence[int], reset_spec: ResetSpec) -> list[Observation]:
        with self._lock:
            self._slots(slots, started=False)
            if (
                reset_spec.options
                or reset_spec.reset_state_id is not None
                or reset_spec.task_id not in (None, 0)
                or reset_spec.instruction is not None
            ):
                raise _invalid(
                    "geniesim supports seed-only reset for its fixed task profile"
                )
            seed = 0 if reset_spec.seed is None else reset_spec.seed
            if type(seed) is not int or not 0 <= seed < 2**32:
                raise _invalid("seed must be an integer in [0, 2**32)")
            self._observation = None
            if self._seed is not None and self._seed != seed:
                # Generalization mutates the USD scene. Rebuild on a changed
                # seed so perturbations cannot accumulate across episodes.
                self.close()
                self._start()
            result = self._request(
                "reset", {"seed": seed}, timeout=self.config.reset_timeout_seconds
            )
            self._limits = result["action_limits"]
            self._result = result["result"]
            self._observation = result["observation"]
            self._observation.extras["action_limits"] = self._limits
            self._seed = seed
            self._finished = False
            return [self._observation]

    def _convert(
        self, raw: dict[str, Any], *, include_images: bool = True
    ) -> Observation:
        images = []
        if include_images:
            for name in CAMERA_NAMES:
                entry = raw["images"][name]
                shape = entry["shape"]
                if (
                    not isinstance(shape, list)
                    or len(shape) != 3
                    or shape[2] != 3
                    or any(type(n) is not int or n <= 0 or n > 4096 for n in shape)
                ):
                    raise ValueError(f"invalid {name} image shape")
                array = np.frombuffer(entry["data"], dtype=np.uint8).reshape(shape)
                images.append(encode_image(array))
        return Observation(
            session_id=SessionId(""),
            episode_id=EpisodeId(0),
            step_index=raw["step_index"],
            main_image=images[0] if images else None,
            wrist_image=images[1] if images else None,
            extra_view_images=images[2:] if images else [],
            state=flatten_state(raw["states"]),
            instruction=raw.get("instruction", ""),
            extras={
                "raw_state": raw["states"],
                "state_groups": [name for name, _ in STATE_GROUPS],
                "camera_slots": {
                    "main_image": "head",
                    "wrist_image": "left_hand",
                    "extra_view_images": ["right_hand"],
                },
                "action_units": "absolute_joint_radians",
                "physics_between_requests": "free_running",
                "seed": raw.get("seed", self._seed),
                "official_success": raw.get("official_success", False),
                "evaluation_step": raw.get("evaluation_step", 0),
                "official_result": raw.get("official_result", {}),
            },
        )

    def observe(self, slots: Sequence[int]) -> list[Observation]:
        with self._lock:
            self._slots(slots)
            return [self._observation]

    def chunk_step(self, slots: Sequence[int], chunk_actions: Sequence[np.ndarray]):
        with self._lock:
            self._slots(slots)
            if len(chunk_actions) != 1:
                raise _invalid("geniesim expects exactly one action block")
            if self._finished:
                raise RuntimeApiError(
                    make_error(
                        ErrorCode.SESSION_NOT_READY, "episode ended; reset is required"
                    )
                )
            try:
                actions = validate_actions(chunk_actions[0], self._limits)
            except (ValueError, TypeError) as error:
                raise _invalid(str(error)) from error
            result = self._request("step", {"actions": actions.tolist()})
            steps = result["steps"]
            self._result = result["result"]
            self._observation = result["observation"]
            self._finished = bool(
                self._result["terminated"] or self._result["truncated"]
            )
            return [
                normalize_chunk_outcome(
                    behavior=self.behavior,
                    final_observation=self._observation,
                    step_observations=result["step_observations"],
                    rewards=[step["reward"] for step in steps],
                    terminations=[step["terminated"] for step in steps],
                    truncations=[step["truncated"] for step in steps],
                    per_step_info=[
                        {
                            key: value
                            for key, value in step.items()
                            if key not in {"reward", "terminated", "truncated"}
                        }
                        for step in steps
                    ],
                    requested_horizon=len(actions),
                    info=dict(self._result),
                )
            ]

    def extension(
        self, slot: int, namespace: str, method: str, args: dict[str, Any]
    ) -> dict[str, Any]:
        with self._lock:
            self._slots([slot])
            if f"{namespace}.{method}" != "geniesim.result":
                raise RuntimeApiError(
                    make_error(
                        ErrorCode.UNSUPPORTED_EXTENSION,
                        f"unsupported extension {namespace}.{method}",
                    )
                )
            if args:
                raise _invalid("geniesim.result accepts no arguments")
            return {
                **self._result,
                "process_id": self.process.pid,
                "artifact_directory": str(self.process.directory),
                "runtime": dict(self.process.metadata),
                "action_limits": self._limits,
            }

    def close(self) -> None:
        with self._lock:
            process, self.process = self.process, None
            self._observation = None
            if process is not None:
                try:
                    process.close()
                except GenieSimProcessError as error:
                    raise RuntimeApiError(
                        make_error(ErrorCode.ENV_FAILURE, str(error))
                    ) from error


class GenieSimFamily:
    env_family = GENIESIM_ENV_FAMILY

    @property
    def capability(self):
        return capability_from_behavior(
            behavior_for(self.env_family), supports_reset_state_id=False
        )

    def create_core(self) -> GenieSimEnvCore:
        return GenieSimEnvCore()


def register_geniesim_env_family(*, replace: bool = True) -> GenieSimFamily:
    family = GenieSimFamily()
    register_env_family(family, replace=replace)
    return family
