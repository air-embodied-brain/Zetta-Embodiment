# Copyright (c) 2026 Zetta Contributors
"""Task program matching the fixed dishwasher proposal contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from robots.robocasa.action_contract import canonical_action, serializable_action

CONTACT_PUSH_TOOL = "robocasa.slide_dishwasher.vla.contact_push"
GUARDED_SUFFIX_TOOL = "robocasa.slide_dishwasher.guarded_vla.terminal_suffix"
BASE_ASSIST_TOOL = "robocasa.slide_dishwasher.base_assisted.terminal_push"
TASK_PROGRESS_CRITIC = "robocasa.slide_dishwasher.critic.task_progress"
INTEGRITY_GATE = "robocasa.slide_dishwasher.integrity"
ROLE1_AGENT = "robocasa.slide_dishwasher.role1_agent"

PROGRAM_COMPONENTS = (
    CONTACT_PUSH_TOOL,
    GUARDED_SUFFIX_TOOL,
    BASE_ASSIST_TOOL,
    TASK_PROGRESS_CRITIC,
    INTEGRITY_GATE,
    ROLE1_AGENT,
)


def _state_number(state: Mapping[str, Any], key: str) -> float | None:
    value = state.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if np.isfinite(number) else None


def _axis(state: Mapping[str, Any]) -> np.ndarray:
    value = np.asarray(
        state.get("privileged.dishwasher.rack.success_direction_base"),
        dtype=np.float64,
    )
    if value.shape != (3,) or not np.isfinite(value).all():
        raise ValueError("authoritative dishwasher success axis is unavailable")
    norm = float(np.linalg.norm(value))
    if norm < 1e-9:
        raise ValueError("authoritative dishwasher success axis is degenerate")
    return value / norm


def _action_translation(action: Mapping[str, Any] | Sequence[Any]) -> np.ndarray:
    canonical = canonical_action(action)
    return np.asarray(canonical["action.end_effector_position"], dtype=np.float64)


def guard_terminal_suffix(
    actions: Sequence[Mapping[str, Any] | Sequence[Any]],
    *,
    state: Mapping[str, Any],
    minimum_projection: float = 0.05,
) -> list[dict[str, list[float]]]:
    """Clamp only reverse success-axis translation in an approved VLA suffix."""

    if not actions:
        raise ValueError("guarded suffix requires at least one pending VLA action")
    axis = _axis(state)
    transformed: list[dict[str, list[float]]] = []
    for raw in actions:
        canonical = canonical_action(raw)
        position = np.asarray(
            canonical["action.end_effector_position"], dtype=np.float64
        )
        projection = float(np.dot(position, axis))
        if projection < minimum_projection:
            position = position + (minimum_projection - projection) * axis
            canonical["action.end_effector_position"] = np.clip(position, -1.0, 1.0)
        transformed.append(serializable_action(canonical))
    return transformed


def base_assisted_terminal_action(
    *,
    state: Mapping[str, Any],
    base_command: float = 1.0,
    arm_retract_command: float = 0.05,
) -> dict[str, list[float]]:
    """Propose positive world advance while unloading arm configuration."""

    axis = _axis(state)
    base = np.zeros(4, dtype=np.float64)
    base[:2] = np.clip(axis[:2] * base_command, -1.0, 1.0)
    arm = np.clip(-axis * arm_retract_command, -1.0, 1.0)
    return serializable_action(
        canonical_action(
            {
                "end_effector_position": arm,
                "base_motion": base,
                "gripper_close": [1.0],
            }
        )
    )


@dataclass(slots=True)
class SlideDishwasherProgramState:
    """Episode-local proposal-only critic with tool-epoch coupling memory."""

    last_residual: float | None = None
    cumulative_progress: float = 0.0
    recent_progress: list[float] = field(default_factory=list)
    coupled: bool = False
    epoch: int = 0

    def reset(self, state: Mapping[str, Any]) -> None:
        self.last_residual = _state_number(
            state, "privileged.dishwasher.rack.residual_to_success"
        )
        self.cumulative_progress = 0.0
        self.recent_progress.clear()
        self.coupled = False
        self.epoch += 1

    def before_action(
        self,
        action: Mapping[str, Any] | Sequence[Any],
        *,
        state: Mapping[str, Any],
        step_index: int,
    ) -> list[dict[str, Any]]:
        proposals: list[dict[str, Any]] = []
        # Operators report that the available collision detector has a
        # high false-positive rate.  Preserve its privileged telemetry for
        # offline diagnosis, but never let it independently reject an action
        # or wake Role1 in the default online program.
        projection = float(np.dot(_action_translation(action), _axis(state)))
        collapsed = bool(self.recent_progress) and self.recent_progress[-1] <= 0.002
        if self.coupled and collapsed and projection <= -0.05:
            proposals.append(
                {
                    "rule_id": "slide_dishwasher.premature_disengagement",
                    "step_index": step_index,
                    "feature": "pending_action.success_axis_projection",
                    "observed_value": projection,
                    "proposal": "pause the pending suffix for multimodal Role1 review",
                    "safety_only": False,
                    "environment_write": False,
                    "program_component": TASK_PROGRESS_CRITIC,
                    "tool_epoch": self.epoch,
                }
            )
        return proposals

    def after_action(
        self,
        *,
        state: Mapping[str, Any],
        step_index: int,
        at_chunk_boundary: bool,
    ) -> list[dict[str, Any]]:
        residual = _state_number(
            state, "privileged.dishwasher.rack.residual_to_success"
        )
        if residual is None:
            return []
        progress = (
            max(0.0, self.last_residual - residual)
            if self.last_residual is not None
            else 0.0
        )
        self.last_residual = residual
        self.cumulative_progress += progress
        self.recent_progress.append(progress)
        self.recent_progress = self.recent_progress[-8:]
        if (
            self.cumulative_progress >= 0.025
            and bool(state.get("privileged.dishwasher.rack.target_contact", False))
        ):
            self.coupled = True
        proposals: list[dict[str, Any]] = []
        if not at_chunk_boundary:
            return proposals
        remaining = _state_number(
            state, "privileged.dishwasher.rack.remaining_to_success_m"
        )
        if self.coupled and remaining is not None and remaining <= 0.04:
            proposals.append(
                {
                    "rule_id": "slide_dishwasher.terminal_handoff",
                    "step_index": step_index,
                    "feature": "privileged.dishwasher.rack.remaining_to_success_m",
                    "observed_value": remaining,
                    "proposal": "review guarded terminal suffix eligibility at boundary",
                    "safety_only": False,
                    "environment_write": False,
                    "program_component": TASK_PROGRESS_CRITIC,
                    "tool_epoch": self.epoch,
                }
            )
        elif (
            self.coupled
            and len(self.recent_progress) >= 6
            and sum(self.recent_progress[-6:]) < 0.008
        ):
            proposals.append(
                {
                    "rule_id": "slide_dishwasher.progress_stagnation",
                    "step_index": step_index,
                    "feature": "privileged.dishwasher.rack.residual_to_success",
                    "observed_value": residual,
                    "proposal": "request Role1 review at the current chunk boundary",
                    "safety_only": False,
                    "environment_write": False,
                    "program_component": TASK_PROGRESS_CRITIC,
                    "tool_epoch": self.epoch,
                }
            )
        return proposals


__all__ = [
    "BASE_ASSIST_TOOL",
    "CONTACT_PUSH_TOOL",
    "GUARDED_SUFFIX_TOOL",
    "INTEGRITY_GATE",
    "PROGRAM_COMPONENTS",
    "ROLE1_AGENT",
    "SlideDishwasherProgramState",
    "TASK_PROGRESS_CRITIC",
    "base_assisted_terminal_action",
    "guard_terminal_suffix",
]
