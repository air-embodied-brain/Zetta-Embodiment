# Copyright (c) 2026 Zetta Contributors
import torch


def prepare_physx_gpu() -> None:
    """Load the explicitly prepared GPU library without SAPIEN's home-directory download."""
    import ctypes
    import os
    from pathlib import Path

    library = os.environ.get("SAPIEN_PHYSX_GPU_LIBRARY")
    if not library:
        return
    import sapien.physx

    if sapien.physx.is_gpu_enabled():
        return
    path = Path(library)
    if not path.is_file():
        raise FileNotFoundError(f"prepared PhysX GPU library is missing: {path}")
    ctypes.CDLL("libcuda.so", ctypes.RTLD_GLOBAL)
    ctypes.CDLL(str(path), ctypes.RTLD_LOCAL)
    # Same final activation as the pinned SAPIEN 3.0.2 enable_gpu implementation.
    sapien.physx._enable_gpu()


def allow_pci_render_backend() -> None:
    """Let ManiSkill accept a full ``pci:<domain>:<bus>:<slot>.<func>`` render backend.

    ManiSkill documents ``pci:...`` as the way to pick a renderer on machines
    without CUDA, but ``parse_backend_device_id`` splits the string on every
    colon and unpacks exactly two values, so a real PCI address raises
    ``ValueError`` before SAPIEN sees it. Keep such a backend intact and let
    SAPIEN resolve it; everything else keeps ManiSkill's own parsing.

    Idempotent, and a no-op for a ManiSkill that is absent or parses backend
    strings some other way.
    """
    try:
        from mani_skill.envs.utils.system import backend
    except ImportError:
        return

    original_parse = getattr(backend, "parse_backend_device_id", None)
    if original_parse is None or getattr(original_parse, "_rlinf_pci_patched", False):
        return

    def parse_backend_device_id(device_backend):
        if isinstance(device_backend, str) and device_backend.startswith("pci:"):
            return device_backend, None
        return original_parse(device_backend)

    parse_backend_device_id._rlinf_pci_patched = True
    backend.parse_backend_device_id = parse_backend_device_id


def recursive_to_own(obj):
    if isinstance(obj, torch.Tensor):
        return obj.clone() if obj.is_shared() else obj
    elif isinstance(obj, list):
        return [recursive_to_own(elem) for elem in obj]
    elif isinstance(obj, tuple):
        return tuple(recursive_to_own(elem) for elem in obj)
    elif isinstance(obj, dict):
        return {k: recursive_to_own(v) for k, v in obj.items()}
    else:
        return obj


def get_batch_rng_state(batched_rng):
    state = {
        "rngs": batched_rng.rngs,
    }
    return state


def set_batch_rng_state(state: dict):
    from mani_skill.envs.utils.randomization.batched_rng import BatchedRNG

    return BatchedRNG.from_rngs(state["rngs"])
