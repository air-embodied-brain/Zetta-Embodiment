# Copyright (c) 2026 Zetta Contributors
from __future__ import annotations

from robots.robocasa.dishwasher_critic import evaluate_slide_dishwasher_rack


def _history(*residuals: float):
    return [{"rack_residual": value, "contact": False} for value in residuals]


def test_fixed_reference_fixture_detects_post_progress_vla_stagnation() -> None:
    result = evaluate_slide_dishwasher_rack(
        {
            "history": _history(1.0, 0.8, 0.6, 0.6, 0.6, 0.6),
            "stagnation_window": 4,
            "stagnation_threshold": 1e-6,
            "minimum_progress": 0.05,
        }
    )
    assert result["classification"] == "vla_stagnation_after_episode_progress"
    assert result["proposal"]["kind"] == "regenerate_plan"
    assert result["proposal"]["evidence"]["global_signed_progress"] == 0.4
    assert result["environment_write"] is False


def test_fixed_reference_fixture_detects_live_terminal_handoff() -> None:
    result = evaluate_slide_dishwasher_rack(
        {
            "history": _history(1.0, 0.9, 0.8),
            "minimum_progress": 0.05,
            "within_supported_terminal_horizon": False,
        }
    )
    assert result["classification"] == "productive_coupling_handoff_ready"
    assert result["proposal"]["kind"] == "switch_tool"
    assert (
        result["proposal"]["evidence"]["handoff_timing"]
        == "while_live_coupling_is_confirmed"
    )
    assert result["proposal"]["authority"] == "proposal_only"


def test_fixed_reference_fixture_rejects_wrong_direction_without_execution() -> None:
    result = evaluate_slide_dishwasher_rack(
        {
            "history": _history(1.0, 0.8, 0.6),
            "minimum_progress": 0.05,
            "pending_action_position": [-0.5, 0.1, 0.0],
            "success_direction": [1.0, 0.0, 0.0],
            "reverse_projection_threshold": 0.05,
        }
    )
    assert result["classification"] == "premature_disengagement"
    assert result["proposal"]["kind"] == "premature_disengagement"
    assert result["proposal"]["evidence"]["task_direction_projection"] == -0.5
    assert result["proposal"]["environment_write"] is False


def test_single_frame_cannot_trigger_temporal_rule() -> None:
    result = evaluate_slide_dishwasher_rack({"history": _history(0.7)})
    assert result["status"] == "insufficient_evidence"
    assert result["triggered"] is False
