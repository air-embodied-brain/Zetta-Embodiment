# Copyright (c) 2026 Zetta Contributors
"""Canonical RoboTwin action contract used by proposal tools.

RoboTwin 2.0 drives an ALOHA-style **bimanual** robot: one action is 14 numbers,
``[left 6 joints, left gripper, right 6 joints, right gripper]``, and they are
**absolute joint targets**, not deltas.

Two consequences shape everything in this module.

**Every command must name an arm.** A single-arm primitive copied from a
single-arm robot has nowhere to say which of the two hands it means, and the
tool catalog it lands in is content-hashed into a campaign manifest and cannot
be edited afterwards. So :data:`ARM_SLICES` is the one place the layout is
written down, and every constructor here takes an explicit ``arm``.

**"Do nothing" is not zero.** Because the targets are absolute, sending zeros to
the arm you are not commanding does not hold it still -- it commands every joint
to angle 0 and flings the arm across the table. The idle arm must be filled from
the *current measured state*, which is why :func:`compose_action` requires the
observed state and refuses to guess it.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

ACTION_DIM = 14
"""Full bimanual action width."""

JOINTS_PER_ARM = 6
"""Revolute joints per arm; the 7th slot of each half is the gripper."""

ARM_SLICES: Mapping[str, slice] = {
    "left": slice(0, 7),
    "right": slice(7, 14),
}
"""The **only** place the bimanual layout is written down.

``[left 6 joints, left gripper, right 6 joints, right gripper]``, matching
``LeRobotAlohaDataConfig``'s delta mask
(``[True]*6 + [False] + [True]*6 + [False]``) and RoboTwin's own state vector.
"""

GRIPPER_OFFSET = JOINTS_PER_ARM
"""Index of the gripper within one arm's 7-slot half."""

ARMS: tuple[str, ...] = ("left", "right")
"""The individually addressable arms, in canonical order."""

Arm = Literal["left", "right", "both"]

JOINT_LIMIT = float(np.pi)
"""Symmetric bound accepted for a joint target, in radians.

Deliberately generous: RoboTwin's per-embodiment URDF limits are narrower and
are enforced by the simulator's own controller. This bound exists to catch a
command that is *structurally* wrong -- unnormalised, in the wrong unit, or
carrying a stray sentinel -- before it reaches the environment actor.
"""

GRIPPER_RANGE = (0.0, 1.0)
"""Normalised gripper command range; 0 is fully closed, 1 fully open."""


class ArmSelectionError(ValueError):
    """An action names no arm, or an arm the robot does not have."""


def normalize_arm(arm: str) -> str:
    """Validate and canonicalise an arm selector.

    Args:
        arm: ``"left"``, ``"right"`` or ``"both"``, in any case.

    Returns:
        The lower-cased selector.

    Raises:
        ArmSelectionError: The selector is missing or unknown. There is no
            default: on a two-armed robot an unnamed arm is a bug, not a
            preference.
    """
    if not arm:
        raise ArmSelectionError(
            "RoboTwin is bimanual: every action must name an arm "
            f"({', '.join(ARMS)}, or 'both')"
        )
    selector = str(arm).strip().lower()
    if selector not in {*ARMS, "both"}:
        raise ArmSelectionError(
            f"unknown arm {arm!r}; expected one of {[*ARMS, 'both']}"
        )
    return selector


def _half(value: Any, *, arm: str) -> np.ndarray:
    """Validate one arm's 7-slot command half.

    Args:
        value: A 7-element sequence, ``[6 joint targets, gripper]``.
        arm: The arm being validated, for the error message.

    Returns:
        A float64 array of length 7.

    Raises:
        ValueError: The half is the wrong width, non-finite, or out of range.
    """
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if array.shape != (JOINTS_PER_ARM + 1,):
        raise ValueError(
            f"{arm} arm command must have {JOINTS_PER_ARM + 1} values "
            f"({JOINTS_PER_ARM} joints + gripper), got shape {array.shape}"
        )
    if not np.isfinite(array).all():
        raise ValueError(f"{arm} arm command contains non-finite values")
    joints = array[:JOINTS_PER_ARM]
    if np.any(np.abs(joints) > JOINT_LIMIT):
        raise ValueError(
            f"{arm} arm joint targets must lie within +-{JOINT_LIMIT:.4f} rad; "
            "values outside it usually mean the command is unnormalised or in "
            "the wrong unit"
        )
    gripper = float(array[GRIPPER_OFFSET])
    low, high = GRIPPER_RANGE
    if not low <= gripper <= high:
        raise ValueError(f"{arm} gripper command {gripper} is outside [{low}, {high}]")
    return array


@dataclass(frozen=True, slots=True)
class RoboTwinAction:
    """One validated bimanual action.

    Attributes:
        values: The 14 absolute joint targets.
        commanded: The arms this action actually commands; the others are
            holding their measured position. Recorded so an audit can tell a
            deliberate hold from an accidental one.
    """

    values: tuple[float, ...]
    commanded: tuple[str, ...]

    def __post_init__(self) -> None:
        """Validate the packed action.

        Raises:
            ValueError: Wrong width, non-finite values, or an unknown arm in
                ``commanded``.
        """
        if len(self.values) != ACTION_DIM:
            raise ValueError(
                f"RoboTwin action must have {ACTION_DIM} values, got {len(self.values)}"
            )
        if not all(np.isfinite(self.values)):
            raise ValueError("RoboTwin action contains non-finite values")
        unknown = sorted(set(self.commanded) - set(ARMS))
        if unknown:
            raise ValueError(f"unknown arm(s) in commanded: {unknown}")
        if not self.commanded:
            raise ArmSelectionError("an action must command at least one arm")

    def arm_half(self, arm: str) -> tuple[float, ...]:
        """Return one arm's 7-slot half.

        Args:
            arm: ``"left"`` or ``"right"``.

        Returns:
            The 7 values for that arm.

        Raises:
            ArmSelectionError: ``arm`` is not an individual arm.
        """
        selector = normalize_arm(arm)
        if selector == "both":
            raise ArmSelectionError("arm_half needs a single arm, not 'both'")
        return tuple(self.values[ARM_SLICES[selector]])

    def as_array(self) -> np.ndarray:
        """Return the action as a float32 array.

        Returns:
            A ``[14]`` float32 array, the layout the env adapter expects.
        """
        return np.asarray(self.values, dtype=np.float32)

    def public_dict(self) -> dict[str, Any]:
        """Return the stable representation used in audit records.

        Returns:
            A JSON-friendly dict.
        """
        return {
            "schema_version": 1,
            "contract": "robotwin_bimanual_joint_v1",
            "commanded_arms": list(self.commanded),
            "values": [float(value) for value in self.values],
        }


def hold_action(current_state: Sequence[float]) -> RoboTwinAction:
    """Build the action that keeps both arms exactly where they are.

    The identity command for an absolute-target robot is "repeat the measured
    state", never a zero vector.

    Args:
        current_state: The observed 14-dim joint state.

    Returns:
        An action holding both arms.
        ``_state_vector`` rejects a state that is not a finite 14-vector.
    """
    state = _state_vector(current_state)
    return RoboTwinAction(values=tuple(state), commanded=ARMS)


def compose_action(
    current_state: Sequence[float],
    *,
    left: Sequence[float] | None = None,
    right: Sequence[float] | None = None,
) -> RoboTwinAction:
    """Compose a bimanual action from one or both arm commands.

    Any arm left as ``None`` is filled from ``current_state`` -- it holds its
    measured position. ``current_state`` is required precisely so that a
    single-arm command cannot silently zero the other arm.

    Args:
        current_state: The observed 14-dim joint state.
        left: The left arm's 7-slot command, or ``None`` to hold.
        right: The right arm's 7-slot command, or ``None`` to hold.

    Returns:
        The composed action. ``_half`` and ``_state_vector`` reject a malformed
        command half or state.

    Raises:
        ArmSelectionError: Neither arm was commanded.
    """
    if left is None and right is None:
        raise ArmSelectionError(
            "compose_action must command at least one arm; use hold_action() "
            "to deliberately hold both"
        )
    values = _state_vector(current_state).copy()
    commanded: list[str] = []
    for arm, command in (("left", left), ("right", right)):
        if command is None:
            continue
        values[ARM_SLICES[arm]] = _half(command, arm=arm)
        commanded.append(arm)
    return RoboTwinAction(values=tuple(values), commanded=tuple(commanded))


def action_from_flat(
    value: Sequence[float] | np.ndarray, *, arm: Arm = "both"
) -> RoboTwinAction:
    """Wrap an already-complete 14-vector, recording which arms it owns.

    This is the transport convenience for a policy that emits the full
    bimanual vector directly (the VLA path); ``arm`` records intent so an audit
    can still tell whether a single-arm tool produced it.

    Args:
        value: A 14-element action.
        arm: The arm(s) this command is claimed to control.

    Returns:
        The validated action. ``normalize_arm`` rejects an unknown ``arm``.

    Raises:
        ValueError: The vector is the wrong width or non-finite.
    """
    selector = normalize_arm(arm)
    flat = np.asarray(value, dtype=np.float64).reshape(-1)
    if flat.shape != (ACTION_DIM,):
        raise ValueError(
            f"flat RoboTwin action must have {ACTION_DIM} values, got {flat.shape}"
        )
    if not np.isfinite(flat).all():
        raise ValueError("flat RoboTwin action contains non-finite values")
    commanded = ARMS if selector == "both" else (selector,)
    return RoboTwinAction(values=tuple(flat), commanded=commanded)


def _state_vector(current_state: Sequence[float]) -> np.ndarray:
    """Validate an observed joint-state vector.

    Args:
        current_state: The observed state.

    Returns:
        A float64 array of length 14.

    Raises:
        ValueError: The state is the wrong width or non-finite.
    """
    state = np.asarray(current_state, dtype=np.float64).reshape(-1)
    if state.shape != (ACTION_DIM,):
        raise ValueError(
            f"RoboTwin state must have {ACTION_DIM} values, got shape {state.shape}"
        )
    if not np.isfinite(state).all():
        raise ValueError("RoboTwin state contains non-finite values")
    return state
