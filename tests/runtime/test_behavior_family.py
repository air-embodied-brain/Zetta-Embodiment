"""Simulator-free contract coverage for the BEHAVIOR family."""

from __future__ import annotations

import numpy as np
import pytest

from rollout_runtime.api.errors import RuntimeApiError
from rollout_runtime.api.messages import EnvSpecMsg, ResetSpec
from rollout_runtime.backends import register_env_family_for
from rollout_runtime.core.env_registry import behavior_for, get_env_family


def _raw_observation() -> dict:
    image = np.zeros((32, 40, 3), dtype=np.uint8)
    return {
        "robot": {
            "rgb": {
                "eyes": image,
                "left_wrist": image + 1,
                "right_wrist": image + 2,
            },
            "proprio": np.arange(256, dtype=np.float32),
        }
    }


class _FakeBehaviorEnv:
    def __init__(self, config):
        self.config = config
        self.elapsed_steps = 0
        self.instruction = "put the bowl on the table"
        self.closed = False

    def reset(self, *, seed=None, options=None):
        assert seed is not None
        assert options == {}
        self.elapsed_steps = 0
        return _raw_observation(), {"seed": seed}

    def step(self, action):
        assert action.shape == (23,)
        self.elapsed_steps += 1
        return _raw_observation(), 1.0, self.elapsed_steps >= 2, False, {"success": self.elapsed_steps >= 2}

    def close(self):
        self.closed = True


def test_behavior_declaration_and_registration_are_lazy():
    declaration = behavior_for("behavior")
    assert declaration.env_family == "behavior"
    assert declaration.reset_signature == "seed_options"
    assert declaration.chunk_obs_layout == "per_step"
    assert declaration.action_layout == "numpy_env_chunk_dim"
    assert declaration.needs_accelerator is True
    adapter = register_env_family_for("behavior")
    assert adapter.env_family == "behavior"
    assert get_env_family("behavior") is adapter


def test_behavior_config_rejects_unknown_and_ambiguous_values():
    from rollout_runtime.backends.behavior import BehaviorEnvConfig

    with pytest.raises(RuntimeApiError, match="unknown behavior env config keys"):
        BehaviorEnvConfig.from_mapping({"typo": 1})
    with pytest.raises(ValueError, match="action_dim"):
        BehaviorEnvConfig(action_dim=7)
    with pytest.raises(ValueError, match="auto_reset"):
        BehaviorEnvConfig(auto_reset=True)


def test_behavior_core_reset_step_and_close_without_simulator(monkeypatch):
    import rollout_runtime.backends.behavior as backend

    monkeypatch.setattr(backend, "_behavior_env_class", lambda: _FakeBehaviorEnv)
    from rollout_runtime.backends.behavior import BehaviorEnvConfig, BehaviorEnvCore

    config = BehaviorEnvConfig(chunk_size=3)
    spec = EnvSpecMsg(
        env_family="behavior",
        env_config={field.name: getattr(config, field.name) for field in __import__("dataclasses").fields(config)},
    )
    core = BehaviorEnvCore()
    core.build(spec, num_envs=1)
    reset = core.reset([0], ResetSpec(seed=13))
    assert reset[0].instruction == "put the bowl on the table"
    assert len(reset[0].state) == 256
    outcome = core.chunk_step([0], [np.zeros((3, 23), dtype=np.float32)])[0]
    assert outcome.executed_horizon == 2
    assert outcome.terminated is True
    assert outcome.reward == pytest.approx(2.0)
    core.close()
    assert core.closed is True
