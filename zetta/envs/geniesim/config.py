# Copyright (c) 2026 Zetta Contributors
"""The supported single-task Genie Sim deployment contract."""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

GENIESIM_REVISION = "6ca11c7593ecf6b7dae58c28fd19f4a789a0dd46"
ASSET_REVISION = "a0813aad7c16165daffbe6c2737754e0344809ee"
ISAACSIM_VERSION = "5.1.0.0"
TASK_NAME = "g2op_if_pick_block_color"
STATE_GROUPS = (
    ("left_arm", 7),
    ("right_arm", 7),
    ("left_gripper", 1),
    ("right_gripper", 1),
    ("waist", 5),
    ("head", 0),
)
CAMERA_NAMES = ("head", "left_hand", "right_hand")
ACTION_DIM = 16
MAX_CHUNK = 64


@dataclasses.dataclass(frozen=True, kw_only=True)
class GenieSimConfig:
    python_executable: str
    geniesim_root: str
    assets_root: str
    output_root: str
    task: str = TASK_NAME
    gpu_id: int = 0
    max_steps: int = 300
    startup_timeout_seconds: float = 600.0
    reset_timeout_seconds: float = 900.0
    request_timeout_seconds: float = 120.0
    close_timeout_seconds: float = 60.0
    library_paths: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in (
            "python_executable",
            "geniesim_root",
            "assets_root",
            "output_root",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"geniesim {name} must be a nonempty path")
            path = Path(value).expanduser()
            # Resolving the interpreter symlink would bypass its virtualenv.
            path = path.absolute() if name == "python_executable" else path.resolve()
            object.__setattr__(self, name, str(path))
        if self.task != TASK_NAME:
            raise ValueError(f"geniesim currently supports only {TASK_NAME!r}")
        if type(self.gpu_id) is not int or self.gpu_id < 0:
            raise ValueError("gpu_id must be a nonnegative physical GPU index")
        if type(self.max_steps) is not int or self.max_steps < 1:
            raise ValueError("max_steps must be a positive integer")
        for name in (
            "startup_timeout_seconds",
            "reset_timeout_seconds",
            "request_timeout_seconds",
            "close_timeout_seconds",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"{name} must be finite and positive")
        if not isinstance(self.library_paths, (list, tuple)) or any(
            not isinstance(path, str) or not path for path in self.library_paths
        ):
            raise ValueError("library_paths must be an array of directory paths")
        object.__setattr__(
            self,
            "library_paths",
            tuple(
                str(Path(path).expanduser().resolve()) for path in self.library_paths
            ),
        )

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> GenieSimConfig:
        payload = dict(value)
        if payload.pop("core_form", "per_slot") != "per_slot":
            raise ValueError("geniesim supports only per_slot")
        unknown = set(payload) - {field.name for field in dataclasses.fields(cls)}
        if unknown:
            raise ValueError(f"unknown geniesim configuration keys: {sorted(unknown)}")
        try:
            return cls(**payload)
        except TypeError as error:
            raise ValueError(f"invalid geniesim configuration: {error}") from error

    def to_dict(self) -> dict[str, Any]:
        result = dataclasses.asdict(self)
        result["library_paths"] = list(self.library_paths)
        return result


def flatten_state(states: Mapping[str, Any]) -> list[float]:
    values = []
    for name, width in STATE_GROUPS:
        group = states.get(name)
        if not isinstance(group, (list, tuple)) or len(group) != width:
            raise ValueError(f"expected {width} {name} joint values")
        for value in group:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"nonnumeric {name} joint value")
            if not math.isfinite(value):
                raise ValueError(f"nonfinite {name} joint value")
            values.append(float(value))
    return values
