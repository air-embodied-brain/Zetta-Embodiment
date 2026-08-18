"""Resolve ``MUJOCO_EGL_DEVICE_ID`` from the requested CUDA device.

EGL enumerates devices independently of ``CUDA_VISIBLE_DEVICES``, and the EGL
device order is not guaranteed to match the CUDA ordinal order. Pinning a
process to CUDA device N (via ``--cuda-device``) without also pointing MuJoCo
at the *matching* EGL device makes MuJoCo render on the wrong device — or on a
software/DRM-only EGL entry — and env_server crashes. This module bridges the
two enumerations.
"""
from __future__ import annotations

import ctypes
import os

from rpent.utils.logging import get_logger

logger = get_logger("egl")

_EGL_EXTENSIONS = 0x3055
_EGL_CUDA_DEVICE_NV = 0x323A


def cuda_to_egl_map() -> dict[int, int]:
    """Return ``{cuda_ordinal: egl_device_index}`` for every real NVIDIA EGL
    device. Software/DRM-only EGL entries are skipped, so values always point
    at GPU devices.

    ``cuda_ordinal`` is the *global* CUDA device index (``EGL_CUDA_DEVICE_NV``),
    which the driver reports regardless of ``CUDA_VISIBLE_DEVICES``.
    """
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    from mujoco.egl import egl_ext as EGL
    from OpenGL import EGL as GLEGL
    from OpenGL.EGL.EXT.device_base import eglQueryDeviceStringEXT

    # eglQueryDeviceAttribEXT isn't exposed by mujoco.egl; grab the raw pointer.
    proc = GLEGL.eglGetProcAddress("eglQueryDeviceAttribEXT")
    if not proc:
        raise RuntimeError("eglQueryDeviceAttribEXT unavailable (need EGL_EXT_device_query)")
    _query = ctypes.cast(int(proc), ctypes.CFUNCTYPE(
        ctypes.c_uint, ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_ssize_t)))

    mapping: dict[int, int] = {}
    for egl_index, dev in enumerate(EGL.eglQueryDevicesEXT()):
        ext = (eglQueryDeviceStringEXT(dev, _EGL_EXTENSIONS) or b"").decode(errors="replace")
        if "EGL_NV_device_cuda" not in ext:
            continue                                      # software / DRM-only — skip
        addr = ctypes.cast(dev, ctypes.c_void_p).value
        val = ctypes.c_ssize_t(-1)
        if _query(ctypes.c_void_p(addr), _EGL_CUDA_DEVICE_NV, ctypes.byref(val)) == 1:
            if val.value >= 0:
                mapping[int(val.value)] = egl_index
    return mapping


def configure_egl_device(cuda_device: int) -> None:
    """Set ``MUJOCO_EGL_DEVICE_ID`` to the EGL device matching *cuda_device*.

    *cuda_device* is a single integer CUDA device ordinal (as passed to
    ``--cuda-device``).

    No-op when ``MUJOCO_EGL_DEVICE_ID`` is already set.  If the EGL/CUDA
    mapping cannot be built, a warning is logged and MuJoCo falls back to its
    default EGL device selection.
    """
    existing = os.environ.get("MUJOCO_EGL_DEVICE_ID")
    if existing is not None:
        logger.warning(
            "MUJOCO_EGL_DEVICE_ID=%s already set; keeping it "
            "(ignoring --cuda-device=%s)", existing, cuda_device,
        )
        return
    cuda_ordinal = cuda_device

    try:
        mapping = cuda_to_egl_map()
    except Exception as e:
        logger.warning("EGL device probe failed (%s); leaving MUJOCO_EGL_DEVICE_ID unset", e)
        return

    egl_index = mapping.get(cuda_ordinal)
    if egl_index is None:
        logger.warning(
            "CUDA device %d has no matching EGL device (map=%s); "
            "leaving MUJOCO_EGL_DEVICE_ID unset",
            cuda_ordinal, mapping,
        )
        return

    os.environ["MUJOCO_EGL_DEVICE_ID"] = str(egl_index)
    if egl_index != cuda_ordinal:
        logger.info(
            "EGL device order != CUDA order: CUDA %d -> EGL %d "
            "(set MUJOCO_EGL_DEVICE_ID=%d)",
            cuda_ordinal, egl_index, egl_index,
        )
    else:
        logger.debug("MUJOCO_EGL_DEVICE_ID=%d (CUDA %d)", egl_index, cuda_ordinal)
