from __future__ import annotations

import threading
from types import SimpleNamespace

import numpy as np
import pytest

from robots.robocasa.stable_renderer import (
    close_persistent_rgb_renderer,
    install_persistent_rgb_renderer,
    install_egl_context_cleanup,
)


class _FakeOffscreenContext:
    def __init__(self) -> None:
        self.camera_id = None
        self.calls: list[tuple[str, int, int, int | None]] = []

    def render(
        self,
        *,
        width: int,
        height: int,
        camera_id: int | None,
        segmentation: bool,
    ) -> None:
        assert segmentation is False
        self.camera_id = camera_id
        self.calls.append(("render", width, height, camera_id))

    def read_pixels(
        self, width: int, height: int, *, depth: bool, segmentation: bool
    ) -> np.ndarray:
        assert depth is False and segmentation is False
        self.calls.append(("read", width, height, self.camera_id))
        value = 10 if self.camera_id == 1 else 20
        return np.full((height, width, 3), value, dtype=np.uint8)


class _FakeMjSim:
    def __init__(self) -> None:
        self.model = SimpleNamespace(
            _model=object(),
            vis=SimpleNamespace(global_=SimpleNamespace(offwidth=1, offheight=1)),
            camera_name2id=lambda name: {"left": 1, "right": 2}[name],
        )
        self.data = SimpleNamespace(_data=object())
        self._render_context_offscreen = _FakeOffscreenContext()

    def render(self, *args, **kwargs):
        return "fallback"


def test_native_renderer_serializes_render_readback_and_returns_owned_frames() -> None:
    binding = SimpleNamespace(MjSim=_FakeMjSim, _MjSim_render_lock=threading.RLock())
    result = install_persistent_rgb_renderer(binding_utils_module=binding)
    simulator = _FakeMjSim()

    left = simulator.render(32, 32, camera_name="left")
    right = simulator.render(32, 32, camera_name="right")

    assert result["mode"] == "serialized_robosuite_offscreen_context"
    assert left.flags.owndata and left.flags.c_contiguous
    assert right.flags.owndata and right.flags.c_contiguous
    assert left.mean() == 10
    assert right.mean() == 20
    assert simulator._render_context_offscreen.calls == [
        ("render", 32, 32, 1),
        ("read", 32, 32, 1),
        ("render", 32, 32, 2),
        ("read", 32, 32, 2),
    ]
    assert simulator.render(32, 32, camera_name="left", depth=True) == "fallback"
    assert close_persistent_rgb_renderer(simulator) is False


def test_native_renderer_installer_is_idempotent() -> None:
    class AnotherFakeMjSim:
        def render(self, *args, **kwargs):
            return "fallback"

    binding = SimpleNamespace(MjSim=AnotherFakeMjSim, _MjSim_render_lock=threading.RLock())
    first = install_persistent_rgb_renderer(binding_utils_module=binding)
    second = install_persistent_rgb_renderer(binding_utils_module=binding)

    assert first["already_installed"] is False
    assert second["already_installed"] is True


@pytest.mark.parametrize("previous_id", [None, 1, 2])
def test_deferred_cleanup_uses_owner_and_restores_other_context(previous_id) -> None:
    # Contexts 1 and 2 can contain identical numeric GL resource names.
    # Freeing context 1's resources while 2 is current would corrupt 2.
    events = []
    previous = SimpleNamespace(address=previous_id) if previous_id else None
    current = [previous]
    owner = SimpleNamespace(address=1)

    class FakeEGL:
        EGL_DRAW = "draw"
        EGL_READ = "read"
        eglGetCurrentContext = staticmethod(lambda: current[0])
        eglGetCurrentDisplay = staticmethod(lambda: "display")
        eglGetCurrentSurface = staticmethod(lambda kind: kind)

        @staticmethod
        def eglMakeCurrent(display, draw, read, context):
            assert (display, draw, read) == ("display", "draw", "read")
            current[0] = context
            events.append("restore")
            return True

    class GLContext:
        _context = owner

        def make_current(self):
            current[0] = owner
            events.append("bind_owner")

        def free(self):
            assert current[0] is owner
            current[0] = None
            self._context = None
            events.append("free_gl")

    class Context:
        pass

    def free_resources():
        assert current[0] is owner
        events.append("free_resources")

    binding = SimpleNamespace(_MUJOCO_GL="egl", MjRenderContext=Context)
    assert install_egl_context_cleanup(binding, egl_module=FakeEGL)["installed"]
    assert install_egl_context_cleanup(binding, egl_module=FakeEGL)["already_installed"]
    context = Context()
    context.gl_ctx = GLContext()
    context.con = SimpleNamespace(free=free_resources)
    context.__del__()
    context.__del__()  # No double free.
    assert events[:3] == ["bind_owner", "free_resources", "free_gl"]
    if previous_id == 2:
        assert events == ["bind_owner", "free_resources", "free_gl", "restore"]
        assert current[0] is previous
    else:
        assert len(events) == 3
        assert current[0] is None


def test_context_cleanup_tolerates_failed_initialization() -> None:
    class Context:
        pass

    binding = SimpleNamespace(_MUJOCO_GL="egl", MjRenderContext=Context)
    install_egl_context_cleanup(binding, egl_module=SimpleNamespace())
    context = Context()
    context.__del__()  # No gl_ctx / con allocated yet.
