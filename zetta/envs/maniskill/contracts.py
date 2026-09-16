# Copyright (c) 2026 Zetta Contributors
"""Versioned observation and action contracts for native ManiSkill Panda tasks."""

from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np

ACTION_CONTRACT = "maniskill_panda_pd_ee_delta_pose_normalized_v1"
PROPRIO_PROFILE = "panda_proprio_v1"
LEGACY_PROFILE = "legacy_flatten_v1"
STATE_FIELDS = tuple(
    f"{field}.{index}" for field in ("qpos", "qvel") for index in range(9)
)


def observation_contract(
    *, profile: str, main_camera: str, wrist_camera: str | None, goal_visibility: str
) -> dict[str, Any]:
    if profile not in {PROPRIO_PROFILE, LEGACY_PROFILE}:
        raise ValueError(f"unsupported ManiSkill observation profile: {profile}")
    if goal_visibility not in {"task_default", "visible"}:
        raise ValueError("goal_visibility must be task_default or visible")
    if not main_camera or main_camera == wrist_camera:
        raise ValueError("main and wrist cameras must be named and distinct")
    return {
        "schema_version": 1,
        "profile": profile,
        "state_fields": list(STATE_FIELDS)
        if profile == PROPRIO_PROFILE
        else "task_defined",
        "state_units": "joint position radians/meters; joint velocity radians/meters per second",
        "main_camera": main_camera,
        "wrist_camera": wrist_camera,
        "extra_cameras": "lexicographic_camera_id",
        "image_layout": "uint8_HWC_RGB",
        "goal_visibility": goal_visibility,
    }


def contract_digest(value: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def validate_native_actions(actions: Any) -> np.ndarray:
    """Validate normalized Panda EE commands without silently clipping them."""
    block = np.asarray(actions, dtype=np.float32)
    if block.ndim != 2 or block.shape[0] < 1 or block.shape[1] != 7:
        raise ValueError(f"expected nonempty [chunk, 7] actions, got {block.shape}")
    if not np.isfinite(block).all():
        raise ValueError("ManiSkill actions must be finite")
    if (np.abs(block) > 1.0).any():
        raise ValueError("normalized ManiSkill actions must be in [-1, 1]")
    return block.copy()
