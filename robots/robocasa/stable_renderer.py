# Copyright (c) 2026 Zetta Contributors
"""Process-local persistent RGB renderer for long RoboSuite episodes.

The installed RoboSuite patch isolates RGB from the shared depth framebuffer,
but creates and destroys one ``mujoco.Renderer`` for every camera observation.
On EGL this eventually produced camera swaps, coherent-but-wrong viewpoints,
and, through the old live encoder path, pixel noise in both RoboCasa and
LIBERO. Reuse one renderer for the lifetime of each MjSim and copy the rendered
pixels before returning them.
"""

from __future__ import annotations

from typing import Any

import numpy as np

PATCH_MARKER = "zetta_persistent_rgb_renderer_v1"
_RENDERER_ATTR = "_zetta_persistent_rgb_renderer"
_RENDERER_SIZE_ATTR = "_zetta_persistent_rgb_renderer_size"


def install_persistent_rgb_renderer(
    *,
    binding_utils_module: Any | None = None,
    mujoco_module: Any | None = None,
) -> dict[str, Any]:
    """Install an idempotent process-local MjSim RGB render patch."""

    if binding_utils_module is None:
        from robosuite.utils import binding_utils as binding_utils_module
    if mujoco_module is None:
        import mujoco as mujoco_module

    sim_class = binding_utils_module.MjSim
    existing = getattr(sim_class, "_zetta_rgb_renderer_patch", None)
    if existing == PATCH_MARKER:
        return {
            "installed": True,
            "already_installed": True,
            "mode": "persistent_mujoco_renderer_per_sim",
            "marker": PATCH_MARKER,
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
        if self._render_context_offscreen is None:
            raise RuntimeError("offscreen render context is not initialized")

        with render_lock:
            self.model.vis.global_.offwidth = max(
                width, self.model.vis.global_.offwidth
            )
            self.model.vis.global_.offheight = max(
                height, self.model.vis.global_.offheight
            )
            requested_size = (int(height), int(width))
            renderer = getattr(self, _RENDERER_ATTR, None)
            renderer_size = getattr(self, _RENDERER_SIZE_ATTR, None)
            if renderer is None or renderer_size != requested_size:
                if renderer is not None:
                    renderer.close()
                renderer = mujoco_module.Renderer(
                    self.model._model,
                    height=height,
                    width=width,
                    max_geom=10000,
                )
                setattr(self, _RENDERER_ATTR, renderer)
                setattr(self, _RENDERER_SIZE_ATTR, requested_size)
            renderer.update_scene(
                self.data._data,
                camera=camera_name if camera_name is not None else -1,
                scene_option=self._render_context_offscreen.vopt,
            )
            return np.array(
                renderer.render()[::-1], dtype=np.uint8, order="C", copy=True
            )

    persistent_render.__name__ = original_render.__name__
    persistent_render.__doc__ = original_render.__doc__
    setattr(sim_class, "_zetta_original_render", original_render)
    setattr(sim_class, "_zetta_rgb_renderer_patch", PATCH_MARKER)
    sim_class.render = persistent_render
    return {
        "installed": True,
        "already_installed": False,
        "mode": "persistent_mujoco_renderer_per_sim",
        "marker": PATCH_MARKER,
    }


def close_persistent_rgb_renderer(simulator: Any) -> bool:
    """Close and detach a renderer installed on one MjSim instance."""

    renderer = getattr(simulator, _RENDERER_ATTR, None)
    if renderer is None:
        return False
    renderer.close()
    delattr(simulator, _RENDERER_ATTR)
    if hasattr(simulator, _RENDERER_SIZE_ATTR):
        delattr(simulator, _RENDERER_SIZE_ATTR)
    return True
