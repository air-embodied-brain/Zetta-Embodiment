# Copyright (c) 2026 Zetta Contributors
"""The RoboTwin bimanual action contract.

Two invariants carry most of the weight here, and both are things a single-arm
robot never has to think about:

1. **Every command names an arm.** There is no default hand.
2. **"Hold" is not zero.** RoboTwin consumes absolute joint targets, so the arm
   you are not commanding must be filled from its measured state; a zero vector
   commands every joint to angle 0 and throws the arm across the table.
"""

from __future__ import annotations

import numpy as np
import pytest

from robots.robotwin.action_contract import (
    ACTION_DIM,
    ARM_SLICES,
    ARMS,
    GRIPPER_OFFSET,
    JOINTS_PER_ARM,
    ArmSelectionError,
    RoboTwinAction,
    action_from_flat,
    compose_action,
    hold_action,
    normalize_arm,
)


def _state() -> np.ndarray:
    """Build a distinguishable 14-dim joint state.

    Returns:
        Left half filled with 0.1, right half with -0.2, grippers at 0.5.
    """
    state = np.zeros(ACTION_DIM, dtype=np.float64)
    state[ARM_SLICES["left"]] = 0.1
    state[ARM_SLICES["right"]] = -0.2
    state[ARM_SLICES["left"]][GRIPPER_OFFSET] = 0.5
    state[GRIPPER_OFFSET] = 0.5
    state[7 + GRIPPER_OFFSET] = 0.5
    return state


def _half(joint: float, gripper: float = 0.5) -> list[float]:
    """Build one arm's 7-slot command.

    Args:
        joint: Value for every joint.
        gripper: Gripper opening.

    Returns:
        A 7-element command.
    """
    return [joint] * JOINTS_PER_ARM + [gripper]


def test_layout_is_declared_once() -> None:
    """The slices are the single source of truth for the bimanual layout."""
    assert ACTION_DIM == 14
    assert ARMS == ("left", "right")
    assert ARM_SLICES["left"] == slice(0, 7)
    assert ARM_SLICES["right"] == slice(7, 14)
    covered = list(range(ACTION_DIM))
    assert covered[ARM_SLICES["left"]] + covered[ARM_SLICES["right"]] == covered


def test_an_unnamed_arm_is_an_error_not_a_default() -> None:
    """On a two-armed robot, an unnamed arm is a bug."""
    for bad in ("", None):
        with pytest.raises(ArmSelectionError):
            normalize_arm(bad)  # type: ignore[arg-type]
    with pytest.raises(ArmSelectionError, match="unknown arm"):
        normalize_arm("port")
    assert normalize_arm("LEFT") == "left"
    assert normalize_arm(" Both ") == "both"


def test_hold_action_repeats_the_measured_state_rather_than_zeroing() -> None:
    """The identity command for an absolute-target robot is the current state."""
    state = _state()
    action = hold_action(state)
    assert np.allclose(action.as_array(), state.astype(np.float32))
    assert action.commanded == ARMS
    assert not np.allclose(action.as_array(), np.zeros(ACTION_DIM))


def test_single_arm_command_holds_the_other_arm_at_its_measured_pose() -> None:
    """This is the trap the contract exists to close.

    Commanding only the left arm must leave the right arm exactly where it was,
    not at joint angle zero.
    """
    state = _state()
    action = compose_action(state, left=_half(0.9, gripper=0.0))

    assert action.commanded == ("left",)
    assert action.arm_half("left") == tuple(_half(0.9, gripper=0.0))
    # The uncommanded arm is a copy of the measurement, not zeros.
    assert action.arm_half("right") == tuple(state[ARM_SLICES["right"]])
    assert action.arm_half("right") != tuple([0.0] * (JOINTS_PER_ARM + 1))


def test_both_arms_can_be_commanded_together() -> None:
    """A genuinely bimanual command records both arms."""
    action = compose_action(_state(), left=_half(0.3), right=_half(-0.4, gripper=1.0))
    assert action.commanded == ("left", "right")
    assert action.arm_half("right")[GRIPPER_OFFSET] == pytest.approx(1.0)


def test_compose_requires_at_least_one_arm() -> None:
    """A no-op composition is a mistake; holding both must be explicit."""
    with pytest.raises(ArmSelectionError, match="at least one arm"):
        compose_action(_state())


def test_compose_requires_the_measured_state() -> None:
    """The state is mandatory precisely so the idle arm cannot be guessed."""
    with pytest.raises(ValueError, match="14 values"):
        compose_action(np.zeros(7), left=_half(0.1))


def test_malformed_arm_halves_are_rejected() -> None:
    """Width, finiteness, joint range and gripper range are all checked."""
    state = _state()
    with pytest.raises(ValueError, match="7 values"):
        compose_action(state, left=[0.1] * 6)
    with pytest.raises(ValueError, match="non-finite"):
        compose_action(state, left=_half(float("nan")))
    with pytest.raises(ValueError, match="rad"):
        compose_action(state, left=_half(100.0))
    with pytest.raises(ValueError, match="gripper"):
        compose_action(state, left=_half(0.1, gripper=3.0))


def test_flat_actions_record_which_arms_they_claim() -> None:
    """The VLA path emits the whole vector; intent is still recorded."""
    flat = np.linspace(-0.5, 0.5, ACTION_DIM)
    assert action_from_flat(flat).commanded == ARMS
    assert action_from_flat(flat, arm="left").commanded == ("left",)
    with pytest.raises(ValueError, match="14 values"):
        action_from_flat(np.zeros(7))


def test_action_serializes_with_its_arm_provenance() -> None:
    """An audit record must show which arms a command owned."""
    payload = compose_action(_state(), right=_half(0.2)).public_dict()
    assert payload["contract"] == "robotwin_bimanual_joint_v1"
    assert payload["commanded_arms"] == ["right"]
    assert len(payload["values"]) == ACTION_DIM


def test_arm_half_rejects_both() -> None:
    """``both`` is a command selector, not an addressable half."""
    with pytest.raises(ArmSelectionError, match="single arm"):
        hold_action(_state()).arm_half("both")


def test_action_must_command_something() -> None:
    """A constructed action with no commanded arm is malformed."""
    with pytest.raises(ArmSelectionError):
        RoboTwinAction(values=tuple([0.0] * ACTION_DIM), commanded=())
