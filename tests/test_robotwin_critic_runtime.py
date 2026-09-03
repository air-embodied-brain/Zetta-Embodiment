# Copyright (c) 2026 Zetta Contributors
"""The RoboTwin Critic feature plane.

Two properties matter more than the individual numbers.

**Every motion feature is per-arm.** A single scalar cannot say which hand
stalled, and a recovery that does not name an arm cannot be executed, so the
plane must let a rule address one hand.

**The dwell unit is a chunk, not a simulator step.** RoboTwin is ``final_only``,
so a rule's ``dwell_steps`` counts chunk boundaries here while it counts
simulator steps on LIBERO. That factor-of-horizon difference is invisible in the
rule itself, which is why the plane publishes the conversion.
"""

from __future__ import annotations

import numpy as np
import pytest

from robots.robotwin.action_contract import ACTION_DIM, ARM_SLICES, GRIPPER_OFFSET
from robots.robotwin.critic_runtime import (
    ACTIVE_ARM_FEATURE,
    GRANULARITY_FEATURE,
    MOTION_EPSILON,
    SIM_STEPS_PER_CHUNK_FEATURE,
    arm_from_proposal,
    critic_rules_from_payload,
    describe_dwell_semantics,
    extract_robotwin_critic_features,
    next_stall_counts,
)
from zetta.evolution.critic import TemporalCritic


def _state(left: float = 0.0, right: float = 0.0, grip: float = 0.5) -> list[float]:
    """Build a 14-dim joint state with per-arm joint values.

    Args:
        left: Value for every left joint.
        right: Value for every right joint.
        grip: Gripper opening for both arms.

    Returns:
        The state vector.
    """
    state = np.zeros(ACTION_DIM, dtype=np.float64)
    state[ARM_SLICES["left"]] = left
    state[ARM_SLICES["right"]] = right
    state[GRIPPER_OFFSET] = grip
    state[7 + GRIPPER_OFFSET] = grip
    return list(state)


def _features(current, previous=None, *, stalls=None, chunk_index=1, horizon=25):
    """Extract features with the common arguments filled in.

    Args:
        current: The chunk-final state.
        previous: The previous chunk-final state.
        stalls: Running stall counters.
        chunk_index: Chunk index.
        horizon: Executed horizon.

    Returns:
        The feature dict.
    """
    return extract_robotwin_critic_features(
        {"state": current},
        chunk_index=chunk_index,
        executed_horizon=horizon,
        reward=0.0,
        terminated=False,
        truncated=False,
        previous_state=previous,
        stall_counts=stalls,
    )


def test_evidence_granularity_is_declared_in_the_plane() -> None:
    """A rule can refuse to fire on the wrong evidence unit."""
    features = _features(_state())
    assert features[GRANULARITY_FEATURE] == "chunk"
    assert features[SIM_STEPS_PER_CHUNK_FEATURE] == 25


def test_dwell_semantics_note_states_the_conversion() -> None:
    """The note belongs in the manifest next to the frozen rules."""
    note = describe_dwell_semantics(execute_horizon=25)
    assert note["dwell_unit"] == "chunk"
    assert note["sim_steps_per_chunk"] == 25
    assert "25x" in note["note"]


def test_motion_is_reported_per_arm() -> None:
    """Only the arm that moved reports motion."""
    features = _features(_state(left=0.3), previous=_state())
    assert features["robotwin.arm.left.joint_motion"] > MOTION_EPSILON
    assert features["robotwin.arm.right.joint_motion"] == pytest.approx(0.0)
    assert features[ACTIVE_ARM_FEATURE] == "left"


def test_active_arm_is_none_when_neither_moved() -> None:
    """A held robot must not be reported as driving an arm."""
    features = _features(_state(), previous=_state())
    assert features[ACTIVE_ARM_FEATURE] == "none"
    assert features["robotwin.arm.motion_ratio"] == pytest.approx(0.0)


def test_motion_ratio_flags_one_sided_effort() -> None:
    """A two-handed task attempted with one hand is visible as a ratio near 1."""
    one_sided = _features(_state(left=0.4), previous=_state())
    balanced = _features(_state(left=0.4, right=0.4), previous=_state())
    assert one_sided["robotwin.arm.motion_ratio"] == pytest.approx(1.0)
    assert balanced["robotwin.arm.motion_ratio"] == pytest.approx(0.0)


def test_stall_counters_advance_per_arm_and_reset_on_motion() -> None:
    """A stall is evidence about one hand, so the counter is per hand."""
    stalls = {"left": 0, "right": 0}
    previous = _state()
    for _ in range(3):
        features = _features(_state(right=0.0), previous=previous, stalls=stalls)
        stalls = next_stall_counts(features)
    assert stalls == {"left": 3, "right": 3}

    moved = _features(_state(left=0.5), previous=previous, stalls=stalls)
    assert next_stall_counts(moved)["left"] == 0
    assert next_stall_counts(moved)["right"] == 4


def test_first_chunk_cannot_evidence_a_stall() -> None:
    """With no predecessor there was no opportunity to move.

    Counting it would let ``dwell_steps: 1`` fire before any motion was possible.
    """
    features = _features(_state(), previous=None)
    assert next_stall_counts(features) == {"left": 0, "right": 0}


def test_a_missing_state_degrades_instead_of_raising() -> None:
    """The Critic is proposal-only; it must never be able to abort an episode."""
    features = extract_robotwin_critic_features(
        {},
        chunk_index=0,
        executed_horizon=25,
        reward=0.0,
        terminated=False,
        truncated=False,
    )
    assert features["robotwin.state_available"] is False
    assert features[ACTIVE_ARM_FEATURE] == "none"

    malformed = _features([0.0] * 7, previous=_state())
    assert malformed["robotwin.state_available"] is False


def test_features_drive_the_shared_temporal_evaluator() -> None:
    """The plane is only useful if a frozen rule can actually consume it."""
    rules = critic_rules_from_payload(
        [
            {
                "rule_id": "left-arm-stalled",
                "title": "Left arm has not moved for two chunks",
                "feature": "robotwin.arm.left.stalled_chunks",
                "operator": "ge",
                "threshold": 2,
                "dwell_steps": 1,
                "cooldown_steps": 0,
                "proposal": "recover the left arm",
                "evidence_ids": ["robotwin.arm.left.joint_motion"],
                "activation_conditions": [
                    {
                        "feature": GRANULARITY_FEATURE,
                        "operator": "eq",
                        "threshold": "chunk",
                    }
                ],
            }
        ]
    )
    critic = TemporalCritic(rules)

    stalls = {"left": 0, "right": 0}
    previous = _state()
    proposals: list[dict] = []
    for index in range(3):
        features = _features(
            _state(), previous=previous, stalls=stalls, chunk_index=index
        )
        stalls = next_stall_counts(features)
        proposals = critic.evaluate(features, step_index=index)
    assert proposals, "a stalled left arm must eventually raise a proposal"
    assert proposals[0]["rule_id"] == "left-arm-stalled"


def test_proposal_arm_is_read_from_either_shape() -> None:
    """Proposals carry the arm at the top level or inside ``details``."""
    assert arm_from_proposal({"arm": "LEFT"}) == "left"
    assert arm_from_proposal({"details": {"arm": "right"}}) == "right"
    assert arm_from_proposal({"rule_id": "x"}) is None
    # An unusable arm reads as absent, so the Actor rejects rather than guesses.
    assert arm_from_proposal({"arm": "port"}) is None
