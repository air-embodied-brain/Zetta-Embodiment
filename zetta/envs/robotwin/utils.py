"""Helpers for the RoboTwin environment, derived from RLinf.

Two pieces are adapted rather than copied verbatim:

- :func:`center_crop_image` replaces RLinf's TensorFlow implementation
  (``rlinf/envs/utils.py``) with a NumPy/Pillow one. The upstream helper pulls
  in ``tensorflow`` purely to call ``tf.image.crop_and_resize`` on a single
  frame; TensorFlow is not a Zetta dependency and adding it for one bilinear
  resample is not worth a multi-hundred-megabyte wheel.
- :func:`ensure_spawn_start_method` wraps the bare
  ``mp.set_start_method("spawn", force=True)`` that upstream executes as an
  import-time-ish side effect inside ``RoboTwinEnv._init_env``.
"""

from __future__ import annotations

import math
import warnings
from typing import Any

import numpy as np

__all__ = [
    "center_crop_image",
    "ensure_spawn_start_method",
    "partition_success_seeds",
]

CROP_SCALE = 0.9
"""Fraction of the **original image area** kept by the centre crop.

Matches RLinf's ``crop_and_resize`` default. The linear crop fraction is
``sqrt(CROP_SCALE)``, applied to both axes.
"""

CROP_OUTPUT_SIZE = (224, 224)
"""Size the crop is resampled to, matching RLinf's hard-coded ``(224, 224)``.

RoboTwin's cameras deliver 240x320 frames, so this step is also what brings a
raw observation to the resolution the Pi0/Pi0.5 image transforms expect.
"""


def center_crop_image(
    image: Any,
    *,
    crop_scale: float = CROP_SCALE,
    output_size: tuple[int, int] = CROP_OUTPUT_SIZE,
) -> np.ndarray:
    """Centre-crop to ``crop_scale`` of the original area, then resize.

    Mirrors ``rlinf.envs.utils.center_crop_image`` -> ``crop_and_resize``: crop
    a centred box covering ``crop_scale`` of the area (so ``sqrt(crop_scale)``
    of each side) and bilinearly resample it to ``output_size``. Upstream
    round-trips through float32 inside TensorFlow and saturates back to uint8;
    Pillow's bilinear resize on uint8 is numerically very close and avoids the
    dependency. The contract that matters downstream -- output shape, dtype and
    channel count -- is identical.

    Args:
        image: An ``HxWx3`` uint8 array, or anything ``np.asarray`` accepts.
        crop_scale: Fraction of the original area to keep, clamped to
            ``[0, 1]``.
        output_size: ``(width, height)`` of the resampled result.

    Returns:
        An ``output_size`` uint8 RGB array.

    Raises:
        ValueError: The input does not have a height and a width.
    """
    from PIL import Image

    array = np.asarray(image)
    if array.ndim < 2:
        raise ValueError(
            f"center_crop_image expects an image with H and W, got shape {array.shape}"
        )

    height, width = array.shape[:2]
    fraction = math.sqrt(min(max(float(crop_scale), 0.0), 1.0))
    crop_h = max(1, int(round(height * fraction)))
    crop_w = max(1, int(round(width * fraction)))
    top = (height - crop_h) // 2
    left = (width - crop_w) // 2
    cropped = array[top : top + crop_h, left : left + crop_w]

    picture = Image.fromarray(np.ascontiguousarray(cropped)).convert("RGB")
    return np.asarray(picture.resize(output_size, Image.BILINEAR))


def ensure_spawn_start_method() -> str:
    """Make the multiprocessing start method ``spawn``, idempotently.

    RoboTwin's ``VectorEnv`` runs each lane in its own process and requires the
    ``spawn`` start method; upstream therefore calls
    ``mp.set_start_method("spawn", force=True)`` unconditionally inside
    ``_init_env``. That is a **process-global** side effect: under the Rollout
    Runtime the env core is built inside a Ray worker that has already made its
    own multiprocessing choices, and silently forcing a different method there
    is exactly the kind of action-at-a-distance that is painful to debug.

    So the force is kept -- RoboTwin genuinely does not work without it -- but
    it becomes explicit, idempotent, and loud when it actually changes
    something.

    Returns:
        The start method in effect after the call (always ``"spawn"``).

    Warns:
        UserWarning: A different start method was already active and has been
            overridden.
    """
    import torch.multiprocessing as mp

    current = mp.get_start_method(allow_none=True)
    if current == "spawn":
        return "spawn"
    if current is not None:
        warnings.warn(
            f"RoboTwin requires the 'spawn' multiprocessing start method; "
            f"overriding the process-global start method {current!r}. "
            "Anything else in this process that depends on "
            f"{current!r} will be affected.",
            UserWarning,
            stacklevel=2,
        )
    mp.set_start_method("spawn", force=True)
    return "spawn"


def partition_success_seeds(
    success_seeds,
    *,
    base_seed: int,
    seed_offset: int,
    total_num_processes: int,
    num_group: int,
):
    """Shuffle the curated success seeds and return this worker's slice.

    Copied from ``rlinf/envs/robotwin/seed_utils.py``. The shuffle is driven by
    ``base_seed`` alone, so every worker derives the *same* global permutation
    and then takes a disjoint contiguous slice of it -- that is what keeps the
    per-worker seed sets non-overlapping without any coordination.

    Args:
        success_seeds: 1-D tensor of seeds known to be solvable for the task.
        base_seed: Seed of the global permutation; must be identical across
            workers.
        seed_offset: This worker's index.
        total_num_processes: Number of workers sharing the seed pool.
        num_group: Group size the slice length is truncated to a multiple of.

    Returns:
        This worker's seed slice, truncated to a whole number of groups.
    """
    import torch

    global_generator = torch.Generator()
    global_generator.manual_seed(base_seed)
    shuffled_indices = torch.randperm(success_seeds.numel(), generator=global_generator)
    shuffled_seeds = success_seeds[shuffled_indices]

    seeds_per_worker = shuffled_seeds.numel() // total_num_processes
    start = seed_offset * seeds_per_worker
    end = start + seeds_per_worker
    worker_seeds = shuffled_seeds[start:end]

    keep_count = (worker_seeds.numel() // num_group) * num_group
    return worker_seeds[:keep_count]
