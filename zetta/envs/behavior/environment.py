# Copyright (c) 2026 Zetta Contributors
"""Thin, lazy OmniGibson wrapper for BEHAVIOR tasks.

The wrapper intentionally owns only the Gym-like lifecycle.  Observation
normalization and the Runtime slot semantics live in the backend so the external
simulator remains optional for imports and contract tests.
"""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from typing import Any

import numpy as np


class BehaviorEnv:
    """Single-lane BEHAVIOR environment backed by OmniGibson."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        self.config = dict(config)
        self._env = self._create_env()
        self._elapsed_steps = 0
        self._last_info: dict[str, Any] = {}

    def _create_env(self) -> Any:
        factory_path = str(self.config.get("env_factory", "omnigibson:Environment"))
        module_name, separator, attr_name = factory_path.partition(":")
        if not separator:
            module_name, attr_name = factory_path.rsplit(".", 1)
        try:
            module = importlib.import_module(module_name)
            factory = getattr(module, attr_name)
        except (ImportError, AttributeError) as exc:
            raise RuntimeError(
                "BEHAVIOR requires OmniGibson. Install the optional behavior "
                "dependencies or set env_config.env_factory to a compatible "
                "Gym-like factory."
            ) from exc
        return factory(configs=self._omnigibson_config())

    def _omnigibson_config(self) -> dict[str, Any]:
        config = {
            "scene": {
                "type": "InteractiveTraversableScene",
                "scene_model": self.config["scene_model"],
            },
            "robots": [
                {
                    "type": self.config["robot_type"],
                    "obs_modalities": list(self.config["obs_modalities"]),
                    "action_normalize": bool(self.config["action_normalize"]),
                }
            ],
            "task": {
                "type": "BehaviorTask",
                "activity_name": self.config["activity_name"],
            },
            "render": {
                "viewer_width": int(self.config["image_size"]),
                "viewer_height": int(self.config["image_size"]),
            },
        }
        extra = self.config.get("omnigibson_config")
        if extra:
            if not isinstance(extra, Mapping):
                raise ValueError("omnigibson_config must be a mapping")
            config.update(dict(extra))
        return config

    @property
    def elapsed_steps(self) -> int:
        return self._elapsed_steps

    @property
    def instruction(self) -> str:
        task = getattr(self._env, "task", None)
        for candidate in (
            getattr(task, "language_instruction", None),
            getattr(task, "instruction", None),
            getattr(task, "activity_name", None),
            self.config.get("activity_name"),
        ):
            if isinstance(candidate, str) and candidate:
                return candidate
        return str(self.config.get("activity_name", "BEHAVIOR task"))

    def reset(
        self,
        *,
        seed: int | None = None,
        options: Mapping[str, Any] | None = None,
    ) -> tuple[Any, dict[str, Any]]:
        del options  # OmniGibson's task state is selected by activity + seed.
        reset = getattr(self._env, "reset")
        try:
            result = reset(seed=seed) if seed is not None else reset()
        except TypeError:
            # Older OmniGibson versions expose a no-argument reset.
            result = reset()
        self._elapsed_steps = 0
        return self._normalize_reset(result)

    def step(self, action: np.ndarray) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        result = self._env.step(action)
        if not isinstance(result, tuple) or len(result) not in {4, 5}:
            raise RuntimeError(
                "BEHAVIOR environment must return a Gym 4- or 5-tuple from step"
            )
        if len(result) == 5:
            obs, reward, terminated, truncated, info = result
        else:
            obs, reward, done, info = result
            terminated, truncated = bool(done), False
        self._elapsed_steps += 1
        if self._elapsed_steps >= int(self.config["max_episode_steps"]) and not terminated:
            truncated = True
        info = dict(info) if isinstance(info, Mapping) else {}
        self._last_info = info
        return obs, float(np.asarray(reward).reshape(-1)[0]), bool(terminated), bool(truncated), info

    @staticmethod
    def _normalize_reset(result: Any) -> tuple[Any, dict[str, Any]]:
        if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], Mapping):
            return result[0], dict(result[1])
        return result, {}

    def close(self) -> None:
        close = getattr(self._env, "close", None)
        if callable(close):
            close()


__all__ = ["BehaviorEnv"]
