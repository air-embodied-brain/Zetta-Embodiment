from __future__ import annotations

import threading
from types import SimpleNamespace

import numpy as np

from robots.robocasa.stable_renderer import (
    close_persistent_rgb_renderer,
    install_persistent_rgb_renderer,
)


class _FakeRenderer:
    instances: list["_FakeRenderer"] = []

    def __init__(self, model, *, height: int, width: int, max_geom: int) -> None:
        self.height = height
        self.width = width
        self.closed = False
        self.camera = None
        self.instances.append(self)

    def update_scene(self, data, *, camera, scene_option) -> None:
        self.camera = camera

    def render(self) -> np.ndarray:
        value = 10 if self.camera == "left" else 20
        return np.full((self.height, self.width, 3), value, dtype=np.uint8)

    def close(self) -> None:
        self.closed = True


class _FakeMjSim:
    def __init__(self) -> None:
        self.model = SimpleNamespace(
            _model=object(),
            vis=SimpleNamespace(global_=SimpleNamespace(offwidth=1, offheight=1)),
        )
        self.data = SimpleNamespace(_data=object())
        self._render_context_offscreen = SimpleNamespace(vopt=object())

    def render(self, *args, **kwargs):
        return "fallback"


def test_persistent_renderer_reuses_context_and_returns_owned_frames() -> None:
    _FakeRenderer.instances.clear()
    binding = SimpleNamespace(MjSim=_FakeMjSim, _MjSim_render_lock=threading.RLock())
    mujoco = SimpleNamespace(Renderer=_FakeRenderer)
    result = install_persistent_rgb_renderer(
        binding_utils_module=binding, mujoco_module=mujoco
    )
    simulator = _FakeMjSim()

    left = simulator.render(32, 32, camera_name="left")
    right = simulator.render(32, 32, camera_name="right")

    assert result["mode"] == "persistent_mujoco_renderer_per_sim"
    assert len(_FakeRenderer.instances) == 1
    assert left.flags.owndata and left.flags.c_contiguous
    assert right.flags.owndata and right.flags.c_contiguous
    assert left.mean() == 10
    assert right.mean() == 20
    assert simulator.render(32, 32, camera_name="left", depth=True) == "fallback"
    assert close_persistent_rgb_renderer(simulator) is True
    assert _FakeRenderer.instances[0].closed is True


def test_persistent_renderer_recreates_only_when_dimensions_change() -> None:
    _FakeRenderer.instances.clear()

    class AnotherFakeMjSim(_FakeMjSim):
        pass

    binding = SimpleNamespace(
        MjSim=AnotherFakeMjSim, _MjSim_render_lock=threading.RLock()
    )
    mujoco = SimpleNamespace(Renderer=_FakeRenderer)
    install_persistent_rgb_renderer(
        binding_utils_module=binding, mujoco_module=mujoco
    )
    simulator = AnotherFakeMjSim()

    simulator.render(32, 32, camera_name="left")
    simulator.render(64, 64, camera_name="left")

    assert len(_FakeRenderer.instances) == 2
    assert _FakeRenderer.instances[0].closed is True
    assert _FakeRenderer.instances[1].closed is False
