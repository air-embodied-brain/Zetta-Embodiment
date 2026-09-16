# Copyright (c) 2026 Zetta Contributors
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch

from zetta.envs.maniskill.environment import ManiskillEnv, extract_termination_from_info


def test_extract_termination_uses_success_and_fail_flags() -> None:
    info = {
        "success": torch.tensor([True, False]),
        "fail": torch.tensor([False, True]),
    }
    actual = extract_termination_from_info(info, num_envs=2, device="cpu")
    torch.testing.assert_close(actual, torch.tensor([True, True]))


def test_reset_preserves_explicit_seed_without_options():
    seen = []
    env = ManiskillEnv.__new__(ManiskillEnv)
    env.seed = 999
    env.cfg = SimpleNamespace(goal_visibility="task_default")
    env.use_fixed_reset_state_ids = False
    env.env = SimpleNamespace(reset=lambda **kwargs: (seen.append(kwargs) or {}, {}))
    env._show_goal_site_visual = lambda: None
    env._wrap_obs = lambda obs, infos: obs
    env._reset_metrics = lambda *args: None
    env.reset(seed=31)
    assert seen == [{"seed": 31, "options": {}}]


def test_reset_first_rgb_contains_visible_goal_after_scene_reconfiguration():
    env = ManiskillEnv.__new__(ManiskillEnv)
    env.cfg = SimpleNamespace(goal_visibility="visible")
    env.seed = 0
    env.use_fixed_reset_state_ids = False
    visible = False

    def show_goal():
        nonlocal visible
        visible = True

    env.env = SimpleNamespace(
        reset=lambda **kwargs: ({"goal_visible": False}, {}),
        unwrapped=SimpleNamespace(
            obs_mode="rgb", get_obs=lambda: {"goal_visible": visible}
        ),
    )
    env._show_goal_site_visual = show_goal
    env._wrap_obs = lambda obs, infos: obs
    env._reset_metrics = lambda *args: None
    obs, _ = env.reset(seed=31)
    assert obs["goal_visible"]


def test_proprio_profile_excludes_task_truth_and_preserves_raw_observation():
    env = ManiskillEnv.__new__(ManiskillEnv)
    env.cfg = SimpleNamespace(
        wrap_obs_mode="simple", observation_profile="panda_proprio_v1"
    )
    env.use_full_state = False
    robot = SimpleNamespace(
        get_qpos=lambda: torch.ones(1, 9), get_qvel=lambda: torch.zeros(1, 9)
    )
    env.env = SimpleNamespace(
        unwrapped=SimpleNamespace(obs_mode="rgb", agent=SimpleNamespace(robot=robot))
    )
    raw = {
        "sensor_data": {
            "base_camera": {"rgb": torch.zeros(1, 8, 8, 3, dtype=torch.uint8)}
        },
        "sensor_param": {},
        "extra": {"goal_pos": torch.ones(1, 3) * 100},
    }
    result = env._wrap_obs(raw)
    assert result["states"].shape == (1, 18)
    assert "sensor_data" in raw and "sensor_param" in raw
    assert result["states"].max().item() == 1


def test_native_action_validation_does_not_modify_input():
    from zetta.envs.maniskill.contracts import validate_native_actions

    actions = np.zeros((2, 7), dtype=np.float32)
    result = validate_native_actions(actions)
    result[0, 0] = 1
    assert actions[0, 0] == 0


def test_state_mode_uses_declared_proprio_profile():
    env = ManiskillEnv.__new__(ManiskillEnv)
    env.cfg = SimpleNamespace(
        wrap_obs_mode="simple", observation_profile="panda_proprio_v1"
    )
    robot = SimpleNamespace(
        get_qpos=lambda: torch.ones(1, 9), get_qvel=lambda: torch.zeros(1, 9)
    )
    env.env = SimpleNamespace(
        unwrapped=SimpleNamespace(obs_mode="state", agent=SimpleNamespace(robot=robot))
    )
    result = env._wrap_obs(torch.full((1, 42), 100.0))
    assert result["states"].shape == (1, 18)
    assert result["states"].max().item() == 1


def test_prepared_physx_avoids_implicit_home_download(tmp_path, monkeypatch):
    import ctypes

    from zetta.envs.maniskill.utils import prepare_physx_gpu

    loaded = []
    physx = ModuleType("sapien.physx")
    physx.is_gpu_enabled = lambda: False
    physx._enable_gpu = lambda: loaded.append("enabled")
    package = ModuleType("sapien")
    package.physx = physx
    monkeypatch.setitem(sys.modules, "sapien", package)
    monkeypatch.setitem(sys.modules, "sapien.physx", physx)
    library = tmp_path / "libPhysXGpu_64.so"
    monkeypatch.setenv("SAPIEN_PHYSX_GPU_LIBRARY", str(library))
    monkeypatch.setattr(ctypes, "CDLL", lambda name, mode: loaded.append(name))
    with pytest.raises(FileNotFoundError, match="prepared PhysX"):
        prepare_physx_gpu()
    assert loaded == []
    library.write_bytes(b"test library")
    prepare_physx_gpu()
    assert loaded == ["libcuda.so", str(library), "enabled"]
