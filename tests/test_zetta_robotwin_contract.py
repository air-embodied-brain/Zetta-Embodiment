# Copyright (c) 2026 Zetta Contributors
"""Contract tests for the vendored RoboTwin env helpers.

These cover the three pieces of ``zetta/envs/robotwin/`` that are reachable
without a simulator: the TensorFlow-free centre crop, the success-seed
partition, and the multiprocessing start-method guard. The env class itself
needs ``robotwin`` + SAPIEN and is exercised on a GPU host instead.
"""

from __future__ import annotations

import numpy as np
import pytest

from zetta.envs.robotwin.utils import (
    CROP_OUTPUT_SIZE,
    CROP_SCALE,
    center_crop_image,
)


def _frame(height: int = 240, width: int = 320) -> np.ndarray:
    """Build a deterministic uint8 RGB frame.

    Args:
        height: Frame height.
        width: Frame width.

    Returns:
        An ``HxWx3`` uint8 array with a distinguishable gradient.
    """
    ys = np.linspace(0, 255, height, dtype=np.float64)[:, None]
    xs = np.linspace(0, 255, width, dtype=np.float64)[None, :]
    red = np.broadcast_to(ys, (height, width))
    green = np.broadcast_to(xs, (height, width))
    blue = (ys + xs) / 2.0
    return np.stack([red, green, blue], axis=-1).astype(np.uint8)


def test_center_crop_maps_robotwin_frames_to_the_model_resolution() -> None:
    """RoboTwin's native 240x320 frame becomes the 224x224 the Pi0 stack wants."""
    cropped = center_crop_image(_frame())
    assert cropped.shape == (CROP_OUTPUT_SIZE[1], CROP_OUTPUT_SIZE[0], 3)
    assert cropped.dtype == np.uint8


def test_center_crop_keeps_the_centre_of_the_frame() -> None:
    """The crop is centred, so a bright centre patch survives and a corner does not."""
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    frame[110:130, 150:170] = 255  # centre patch
    cropped = center_crop_image(frame)
    assert cropped.max() == 255

    corner_only = np.zeros((240, 320, 3), dtype=np.uint8)
    corner_only[0:4, 0:4] = 255  # outside sqrt(0.9) of each side
    assert center_crop_image(corner_only).max() == 0


def test_center_crop_scale_is_an_area_fraction() -> None:
    """``CROP_SCALE`` is an area fraction, so each side keeps ``sqrt(CROP_SCALE)``."""
    assert 0.0 < CROP_SCALE <= 1.0
    # A square input makes the linear fraction directly observable: with
    # crop_scale=0.25 exactly half of each side is kept.
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    frame[25:75, 25:75] = 255
    kept = center_crop_image(frame, crop_scale=0.25, output_size=(50, 50))
    assert kept.min() == 255


def test_center_crop_accepts_grayscale_and_returns_rgb() -> None:
    """A single-channel frame is widened to RGB rather than rejected."""
    gray = np.full((240, 320), 128, dtype=np.uint8)
    out = center_crop_image(gray)
    assert out.shape == (224, 224, 3)


def test_center_crop_rejects_a_non_image() -> None:
    """A 1-D array has no height/width and must fail loudly."""
    with pytest.raises(ValueError, match="H and W"):
        center_crop_image(np.zeros(8, dtype=np.uint8))


def test_success_seed_partition_is_disjoint_and_deterministic() -> None:
    """Every worker derives the same permutation and takes a disjoint slice."""
    torch = pytest.importorskip("torch")
    from zetta.envs.robotwin.utils import partition_success_seeds

    seeds = torch.arange(100, 140, dtype=torch.long)
    slices = [
        partition_success_seeds(
            seeds,
            base_seed=7,
            seed_offset=offset,
            total_num_processes=4,
            num_group=2,
        )
        for offset in range(4)
    ]

    assert all(s.numel() == 10 for s in slices)
    flat = torch.cat(slices)
    assert flat.numel() == len(set(flat.tolist())), "worker slices overlap"

    again = partition_success_seeds(
        seeds, base_seed=7, seed_offset=1, total_num_processes=4, num_group=2
    )
    assert torch.equal(again, slices[1])

    shifted = partition_success_seeds(
        seeds, base_seed=8, seed_offset=1, total_num_processes=4, num_group=2
    )
    assert not torch.equal(shifted, slices[1]), "base_seed must drive the shuffle"


def test_success_seed_partition_truncates_to_whole_groups() -> None:
    """A slice that does not divide by ``num_group`` is trimmed, not padded."""
    torch = pytest.importorskip("torch")
    from zetta.envs.robotwin.utils import partition_success_seeds

    seeds = torch.arange(0, 21, dtype=torch.long)  # 21 seeds, 2 workers -> 10 each
    got = partition_success_seeds(
        seeds, base_seed=1, seed_offset=0, total_num_processes=2, num_group=4
    )
    assert got.numel() == 8  # 10 trimmed down to a multiple of 4


def test_spawn_guard_is_idempotent() -> None:
    """Calling the guard twice leaves ``spawn`` in place and warns only on a change."""
    pytest.importorskip("torch")
    import warnings

    import torch.multiprocessing as mp

    from zetta.envs.robotwin.utils import ensure_spawn_start_method

    original = mp.get_start_method(allow_none=True)
    try:
        assert ensure_spawn_start_method() == "spawn"
        assert mp.get_start_method(allow_none=True) == "spawn"

        # Second call must be a silent no-op: the override warning is only for
        # the case where a different method was actually displaced.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            assert ensure_spawn_start_method() == "spawn"
        assert not caught, f"unexpected warning(s): {[str(w.message) for w in caught]}"
    finally:
        if original is not None:
            mp.set_start_method(original, force=True)


def test_spawn_guard_warns_when_it_displaces_another_method() -> None:
    """Overriding a live start method is a process-global act and must be loud."""
    pytest.importorskip("torch")
    import multiprocessing
    import warnings

    import torch.multiprocessing as mp

    from zetta.envs.robotwin.utils import ensure_spawn_start_method

    if "fork" not in multiprocessing.get_all_start_methods():
        pytest.skip("platform has no 'fork' start method to displace")

    original = mp.get_start_method(allow_none=True)
    try:
        mp.set_start_method("fork", force=True)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            assert ensure_spawn_start_method() == "spawn"
        assert any("spawn" in str(w.message) for w in caught)
    finally:
        if original is not None:
            mp.set_start_method(original, force=True)


def test_prepare_actions_passes_robotwin_chunks_through_unchanged() -> None:
    """RoboTwin consumes ALOHA joint targets directly, so the transform is identity.

    The delta-vs-absolute split (deltas on the 6 joints, absolute on the
    gripper, per arm) already happened inside the data config's
    ``DeltaActions``/``AbsoluteActions``; re-applying anything here would
    double-count it.
    """
    pytest.importorskip("torch")
    from zetta.compat.actions import prepare_actions

    chunk = np.arange(2 * 3 * 14, dtype=np.float32).reshape(2, 3, 14)
    out = prepare_actions(
        chunk,
        env_type="robotwin",
        model_type="openpi",
        num_action_chunks=3,
        action_dim=14,
    )
    assert np.array_equal(np.asarray(out), chunk)


def test_prepare_actions_rejects_single_arm_width_for_robotwin() -> None:
    """A 7-dim chunk reaching a bimanual env is the mistake worth failing on.

    Without an explicit branch this would ride the function's fall-through
    identity return and silently feed half-width actions to a 14-DoF robot.
    """
    pytest.importorskip("torch")
    from zetta.compat.actions import prepare_actions

    with pytest.raises(ValueError, match="bimanual"):
        prepare_actions(
            np.zeros((1, 4, 7), dtype=np.float32),
            env_type="robotwin",
            model_type="openpi",
            num_action_chunks=4,
            action_dim=7,
        )


def test_prepare_actions_rejects_a_mismatched_trailing_axis() -> None:
    """A declared 14 with a 7-wide array must not pass either."""
    pytest.importorskip("torch")
    from zetta.compat.actions import prepare_actions

    with pytest.raises(ValueError, match="trailing dimension"):
        prepare_actions(
            np.zeros((1, 4, 7), dtype=np.float32),
            env_type="robotwin",
            model_type="openpi",
            num_action_chunks=4,
            action_dim=14,
        )
