# Copyright (c) 2026 RPent Contributors
"""Proposal-only dishwasher critic for observable task behavior."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _residuals(history: Any) -> list[float]:
    if not isinstance(history, Sequence) or isinstance(history, (str, bytes)):
        raise ValueError("history must be an array")
    result = []
    for index, item in enumerate(history):
        if not isinstance(item, Mapping):
            raise ValueError(f"history[{index}] must be an object")
        raw = item.get("rack_residual", item.get("residual_to_success"))
        result.append(_finite(raw, f"history[{index}].rack_residual"))
    return result


def _proposal(
    *, classification: str, kind: str, reason: str, evidence: dict[str, Any]
) -> dict[str, Any]:
    return {
        "status": "proposal",
        "triggered": True,
        "classification": classification,
        "proposal": {
            "kind": kind,
            "reason": reason,
            "evidence": evidence,
            "authority": "proposal_only",
            "environment_write": False,
        },
        "authority": "proposal_only",
        "environment_write": False,
    }


def evaluate_slide_dishwasher_rack(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Classify reverse motion, post-progress stagnation, or live handoff.

    This function owns no simulator and cannot execute, replace, or terminate an
    action. All thresholds are supplied in the frozen candidate payload.
    """

    residuals = _residuals(payload.get("history"))
    if len(residuals) < 2:
        return {
            "status": "insufficient_evidence",
            "triggered": False,
            "authority": "proposal_only",
            "environment_write": False,
        }
    minimum_progress = _finite(
        payload.get("minimum_progress", 0.03), "minimum_progress"
    )
    reverse_threshold = _finite(
        payload.get("reverse_projection_threshold", 0.05),
        "reverse_projection_threshold",
    )
    stagnation_threshold = _finite(
        payload.get("stagnation_threshold", 1e-3), "stagnation_threshold"
    )
    stagnation_window = int(payload.get("stagnation_window", 4))
    if stagnation_window < 2:
        raise ValueError("stagnation_window must be at least two")
    progress = residuals[0] - residuals[-1]
    productive_transitions = sum(
        first - second >= minimum_progress
        for first, second in zip(residuals, residuals[1:], strict=False)
    )
    coupling_established = bool(payload.get("coupling_established")) or (
        productive_transitions >= 2
    )

    pending = payload.get("pending_action_position")
    direction = payload.get("success_direction")
    if pending is not None or direction is not None:
        action = np.asarray(pending, dtype=np.float64)
        axis = np.asarray(direction, dtype=np.float64)
        if action.shape != (3,) or axis.shape != (3,):
            raise ValueError("pending action and success direction must be 3-vectors")
        axis_norm = float(np.linalg.norm(axis))
        if (
            not np.isfinite(action).all()
            or not np.isfinite(axis).all()
            or axis_norm <= 0
        ):
            raise ValueError("pending action projection inputs are invalid")
        projection = float(np.dot(action, axis / axis_norm))
        if coupling_established and projection < -reverse_threshold:
            return _proposal(
                classification="premature_disengagement",
                kind="premature_disengagement",
                reason="Pending motion reverses verified task progress.",
                evidence={
                    "progress_guard_armed": True,
                    "guard_source": "live_progress_and_coupling_history",
                    "task_direction_projection": projection,
                    "reverse_projection_threshold": reverse_threshold,
                },
            )

    if len(residuals) >= stagnation_window and progress >= minimum_progress:
        recent = residuals[-stagnation_window:]
        if max(recent) - min(recent) <= stagnation_threshold:
            return _proposal(
                classification="vla_stagnation_after_episode_progress",
                kind="regenerate_plan",
                reason="Progress was established but the recent VLA window stalled.",
                evidence={
                    "coupling_established": coupling_established,
                    "global_signed_progress": progress,
                    "recent_residual_span": max(recent) - min(recent),
                    "window": stagnation_window,
                },
            )

    if coupling_established and not bool(
        payload.get("within_supported_terminal_horizon", True)
    ):
        return _proposal(
            classification="productive_coupling_handoff_ready",
            kind="switch_tool",
            reason="Live productive coupling is confirmed before the terminal horizon.",
            evidence={
                "coupling_established": True,
                "handoff_timing": "while_live_coupling_is_confirmed",
                "global_signed_progress": progress,
                "within_supported_terminal_horizon": False,
            },
        )

    return {
        "status": "clear",
        "triggered": False,
        "classification": None,
        "authority": "proposal_only",
        "environment_write": False,
        "evidence": {
            "coupling_established": coupling_established,
            "global_signed_progress": progress,
        },
    }


__all__ = ["evaluate_slide_dishwasher_rack"]
