# Copyright (c) 2026 Zetta Contributors
"""Small, simulator-independent BEHAVIOR rollout contract."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from zetta.envs.behavior.contracts import ACTION_CONTRACT, BEHAVIOR_ACTION_DIM

ACTION_DIM = BEHAVIOR_ACTION_DIM
TASK_CONTRACT = {
    "schema_version": 1,
    "environment": "behavior",
    "robot": "R1Pro",
    "action_contract": ACTION_CONTRACT,
    "observation_profile": "r1pro_rgb_proprio_v1",
    "core_form": "per_slot",
}


def validate_observation(observation: Any) -> None:
    """Validate the fields a BEHAVIOR policy rollout must receive."""
    if observation.main_image is None or observation.wrist_image is None:
        raise ValueError("BEHAVIOR requires main and wrist images")
    if len(observation.extra_view_images) < 1:
        raise ValueError("BEHAVIOR requires the second wrist view in extra_view_images")
    if len(observation.state) < ACTION_DIM:
        raise ValueError(f"BEHAVIOR requires at least {ACTION_DIM} state values")
    if any(not math.isfinite(float(value)) for value in observation.state):
        raise ValueError("BEHAVIOR state must be finite")


def validate_actions(actions: Any) -> np.ndarray:
    block = np.asarray(actions, dtype=np.float32)
    if block.ndim != 2 or block.shape[0] < 1 or block.shape[1] != ACTION_DIM:
        raise ValueError(f"expected [chunk, {ACTION_DIM}] BEHAVIOR actions")
    if not np.isfinite(block).all():
        raise ValueError("BEHAVIOR actions must be finite")
    return block
