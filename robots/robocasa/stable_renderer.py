# Copyright (c) 2026 Zetta Contributors
"""Owned-context EGL cleanup and serialized native RGB readback.

Deferred destruction of an old RoboSuite context must not free its GL resource
names in the current (new episode's) context. Those names are context-local.
Preserve the current binding across cleanup, including GC during rendering.
"""

from __future__ import annotations

from typing import Any

import numpy as np

# Keep the public installer name for compatibility with existing runtime code.
PATCH_MARKER = "zetta_serialized_native_rgb_renderer_v2"
_RENDERER_ATTR = "_zetta_persistent_rgb_renderer"
_RENDERER_SIZE_ATTR = "_zetta_persistent_rgb_renderer_size"


def install_egl_context_cleanup(
    binding_utils_module: Any, *, egl_module: Any | None = None
) -> dict[str, Any]:
    """Fix RoboSuite's destructor without modifying the shared installation.

Do not take the render lock here: cyclic GC can invoke this destructor while
the same thread holds that lock. EGL current-context state is thread-local.
"""
    if getattr(binding_utils_module, "_MUJOCO_GL", None) != "egl":
        return {"installed": False, "reason": "not_egl"}
    context_class = binding_utils_module.MjRenderContext
    if getattr(context_class, "_zetta_owned_egl_cleanup", False):
        return {"installed": True, "already_installed": True}
    if egl_module is None:
        from OpenGL import EGL as egl_module
    egl = egl_module

    def owned_context_destroy(self: Any) -> None:
        gl_context = getattr(self, "gl_ctx", None)
        owner = getattr(gl_context, "_context", None)
        if not owner:
            return  # Also safe for repeated / partially initialized cleanup.
        previous = egl.eglGetCurrentContext()
        display = egl.eglGetCurrentDisplay()
        draw = egl.eglGetCurrentSurface(egl.EGL_DRAW)
        read = egl.eglGetCurrentSurface(egl.EGL_READ)
        restore = bool(previous) and previous.address != owner.address
        try:
            gl_context.make_current()
            render_context = getattr(self, "con", None)
            if render_context is not None:
                render_context.free()
                del self.con
            gl_context.free()
        finally:
            if restore and not egl.eglMakeCurrent(display, draw, read, previous):
                raise RuntimeError("failed to restore EGL context after cleanup")

    context_class.__del__ = owned_context_destroy
    context_class._zetta_owned_egl_cleanup = True
    return {"installed": True, "already_installed": False}


def install_persistent_rgb_renderer(
    *,
    binding_utils_module: Any | None = None,
    mujoco_module: Any | None = None,
) -> dict[str, Any]:
    """Install an idempotent process-local MjSim RGB render patch."""

    if binding_utils_module is None:
        from robosuite.utils import binding_utils as binding_utils_module
    # Kept as an injectable compatibility argument for callers and older
    # tests. The v2 backend intentionally creates no additional Renderer.
    del mujoco_module

    cleanup = install_egl_context_cleanup(binding_utils_module)

    sim_class = binding_utils_module.MjSim
    existing = getattr(sim_class, "_zetta_rgb_renderer_patch", None)
    if existing == PATCH_MARKER:
        return {
            "installed": True,
            "already_installed": True,
            "mode": "serialized_robosuite_offscreen_context",
            "marker": PATCH_MARKER,
            "context_cleanup": cleanup,
        }

    original_render = sim_class.render
    render_lock = binding_utils_module._MjSim_render_lock

    def persistent_render(
        self: Any,
        width: int | None = None,
        height: int | None = None,
        *,
        camera_name: str | None = None,
        depth: bool = False,
        mode: str = "offscreen",
        device_id: int = -1,
        segmentation: bool = False,
    ) -> Any:
        if depth or segmentation or mode != "offscreen":
            return original_render(
                self,
                width,
                height,
                camera_name=camera_name,
                depth=depth,
                mode=mode,
                device_id=device_id,
                segmentation=segmentation,
            )
        if width is None or height is None:
            raise ValueError("persistent RGB rendering requires width and height")
        context = self._render_context_offscreen
        if context is None:
            raise RuntimeError("offscreen render context is not initialized")

        with render_lock:
            camera_id = (
                None
                if camera_name is None
                else self.model.camera_name2id(camera_name)
            )
            context.render(
                width=int(width),
                height=int(height),
                camera_id=camera_id,
                segmentation=False,
            )
            frame = context.read_pixels(
                int(width), int(height), depth=False, segmentation=False
            )
            return np.array(
                frame, dtype=np.uint8, order="C", copy=True
            )

    persistent_render.__name__ = original_render.__name__
    persistent_render.__doc__ = original_render.__doc__
    setattr(sim_class, "_zetta_original_render", original_render)
    setattr(sim_class, "_zetta_rgb_renderer_patch", PATCH_MARKER)
    sim_class.render = persistent_render
    return {
        "installed": True,
        "already_installed": False,
        "mode": "serialized_robosuite_offscreen_context",
        "marker": PATCH_MARKER,
        "context_cleanup": cleanup,
    }


def close_persistent_rgb_renderer(simulator: Any) -> bool:
    """Clean up a legacy v1 renderer if one exists on the simulator.

    The v2 backend borrows RoboSuite's context, whose lifetime is owned by the
    simulator, so normally there is nothing to close here.
    """

    renderer = getattr(simulator, _RENDERER_ATTR, None)
    if renderer is None:
        return False
    renderer.close()
    delattr(simulator, _RENDERER_ATTR)
    if hasattr(simulator, _RENDERER_SIZE_ATTR):
        delattr(simulator, _RENDERER_SIZE_ATTR)
    return True
