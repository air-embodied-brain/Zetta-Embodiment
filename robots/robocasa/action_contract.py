# Copyright (c) 2026 RPent Contributors
"""Canonical RoboCasa action contract used by proposal tools.

The public RoboCasa Gym wrapper consumes a mapping of named action arrays.  A
flat vector is accepted only as a transport convenience; it is converted to
that public mapping before the environment is mutated.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

COMPONENT_SIZES = {
    "end_effector_position": 3,
    "end_effector_rotation": 3,
    "gripper_close": 1,
    "base_motion": 4,
    "control_mode": 1,
}
FLAT_ACTION_SIZE = sum(COMPONENT_SIZES.values())


@dataclass(frozen=True)
class ActionScale:
    """Optional normalized-command scales; unit scale preserves VLA output."""

    end_effector_position: float = 1.0
    end_effector_rotation: float = 1.0
    base_xy: float = 1.0
    base_yaw: float = 1.0
    torso: float = 1.0

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any] | None) -> "ActionScale":
        values = dict(payload or {})
        result = cls(
            end_effector_position=float(values.get("end_effector_position", 1.0)),
            end_effector_rotation=float(values.get("end_effector_rotation", 1.0)),
            base_xy=float(values.get("base_xy", 1.0)),
            base_yaw=float(values.get("base_yaw", 1.0)),
            torso=float(values.get("torso", 1.0)),
        )
        if any(value <= 0 or not np.isfinite(value) for value in result.as_tuple()):
            raise ValueError("all RoboCasa action scales must be finite and positive")
        return result

    def as_tuple(self) -> tuple[float, ...]:
        return (
            self.end_effector_position,
            self.end_effector_rotation,
            self.base_xy,
            self.base_yaw,
            self.torso,
        )


def _component(
    value: Any,
    *,
    name: str,
    size: int,
    lower: float,
    upper: float,
) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape == (1, size):
        array = array[0]
    if array.shape == () and size == 1:
        array = array.reshape(1)
    if array.shape != (size,) or not np.isfinite(array).all():
        raise ValueError(f"{name} must contain exactly {size} finite values")
    if np.any(array < lower) or np.any(array > upper):
        raise ValueError(f"{name} values must be within [{lower}, {upper}]")
    return array


def _mapping_from_flat(value: Sequence[Any] | np.ndarray) -> dict[str, Any]:
    flat = np.asarray(value, dtype=np.float64)
    if flat.shape != (FLAT_ACTION_SIZE,):
        raise ValueError(
            f"flat RoboCasa action must have {FLAT_ACTION_SIZE} values, got {flat.shape}"
        )
    cursor = 0
    result: dict[str, Any] = {}
    for name, size in COMPONENT_SIZES.items():
        result[name] = flat[cursor : cursor + size]
        cursor += size
    return result


def canonical_action(
    value: Mapping[str, Any] | Sequence[Any] | np.ndarray,
    *,
    scale: ActionScale | None = None,
) -> dict[str, np.ndarray]:
    """Validate and convert one command to RoboCasa's named Gym mapping."""

    source: Mapping[str, Any]
    if isinstance(value, Mapping):
        nested = value.get("action")
        source = nested if isinstance(nested, Mapping) else value
    else:
        source = _mapping_from_flat(value)
    scales = scale or ActionScale()
    position = _component(
        source.get(
            "end_effector_position",
            source.get("action.end_effector_position", [0.0, 0.0, 0.0]),
        ),
        name="end_effector_position",
        size=3,
        lower=-1.0,
        upper=1.0,
    )
    rotation = _component(
        source.get(
            "end_effector_rotation",
            source.get("action.end_effector_rotation", [0.0, 0.0, 0.0]),
        ),
        name="end_effector_rotation",
        size=3,
        lower=-1.0,
        upper=1.0,
    )
    gripper = _component(
        source.get("gripper_close", source.get("action.gripper_close", [0.0])),
        name="gripper_close",
        size=1,
        lower=0.0,
        upper=1.0,
    )
    base = _component(
        source.get("base_motion", source.get("action.base_motion", [0.0] * 4)),
        name="base_motion",
        size=4,
        lower=-1.0,
        upper=1.0,
    )
    control = _component(
        source.get("control_mode", source.get("action.control_mode", [0.0])),
        name="control_mode",
        size=1,
        lower=0.0,
        upper=1.0,
    )
    position = np.clip(position * scales.end_effector_position, -1.0, 1.0)
    rotation = np.clip(rotation * scales.end_effector_rotation, -1.0, 1.0)
    base = np.clip(
        base
        * np.asarray(
            [scales.base_xy, scales.base_xy, scales.base_yaw, scales.torso],
            dtype=np.float64,
        ),
        -1.0,
        1.0,
    )
    return {
        "action.end_effector_position": position,
        "action.end_effector_rotation": rotation,
        "action.gripper_close": gripper,
        "action.base_motion": base,
        "action.control_mode": control,
    }


def serializable_action(value: Mapping[str, Any]) -> dict[str, list[float]]:
    """Return a stable JSON form for hashes and trajectory artifacts."""

    canonical = canonical_action(value)
    return {
        name: np.asarray(component, dtype=np.float64).astype(float).tolist()
        for name, component in canonical.items()
    }


def zero_action(*, gripper_close: float = 0.0) -> dict[str, np.ndarray]:
    return canonical_action({"gripper_close": [gripper_close]})
