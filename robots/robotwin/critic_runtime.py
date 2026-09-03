# Copyright (c) 2026 Zetta Contributors
"""Bimanual, chunk-granular Critic features for audited RoboTwin rollouts.

The evaluator itself is shared (:class:`zetta.evolution.critic.TemporalCritic`);
what is environment-specific is the feature plane, and RoboTwin's differs from
every other family here in two ways.

**Evidence is chunk-granular.** RoboTwin is the only ``final_only`` family: one
``chunk_step`` submits the whole chunk and returns exactly one observation, so
there are no intermediate frames and a feature can only ever be computed between
*chunk-final* frames. Every counter below therefore advances once per chunk.

That has a consequence sharp enough to deserve its own field. ``CriticRule``'s
``dwell_steps`` and ``cooldown_steps`` are unit-less integers, and on LIBERO they
count **simulator steps**. Here the same number counts **chunks**, so a rule
copied across families silently changes meaning by a factor of the execute
horizon -- ``dwell_steps: 10`` is 10 simulator steps on LIBERO and 250 on
RoboTwin at a horizon of 25. :data:`SIM_STEPS_PER_CHUNK_FEATURE` is published so
a rule can state the conversion instead of assuming it, and
:func:`describe_dwell_semantics` produces the note that belongs in a campaign
manifest.

**Every feature that concerns motion is per-arm.** A single ``eef_motion``
number cannot say *which hand* stopped moving, and a recovery that does not name
an arm cannot be executed (see
:mod:`robots.robotwin.action_contract`). So the plane is mirrored across
``left`` and ``right``, and :data:`ACTIVE_ARM_FEATURE` reports which arm the
chunk actually moved.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from robots.robotwin.action_contract import ACTION_DIM, ARM_SLICES, ARMS, GRIPPER_OFFSET
from zetta.evolution.models import CriticPredicate, CriticRule

FEATURE_PREFIX = "robotwin"
"""Every feature this module publishes is namespaced under it."""

SIM_STEPS_PER_CHUNK_FEATURE = f"{FEATURE_PREFIX}.chunk.sim_steps_per_chunk"
"""Feature naming how many simulator steps one chunk advanced.

Published so a frozen rule can convert a dwell expressed in simulator steps into
this family's chunk unit, rather than inheriting a factor-of-horizon error from
a rule written for a per-step family.
"""

ACTIVE_ARM_FEATURE = f"{FEATURE_PREFIX}.active_arm"
"""Feature naming the arm that moved more during the chunk (``left``/``right``/``none``)."""

GRANULARITY_FEATURE = f"{FEATURE_PREFIX}.evidence.granularity"
"""Constant ``"chunk"``; lets a rule refuse to fire on the wrong evidence unit."""

MOTION_EPSILON = 1e-4
"""Below this joint-space L2 delta an arm counts as not having moved.

Chosen against RoboTwin's joint targets in radians: an arm genuinely holding
position reports deltas at the 1e-6 level, while the smallest deliberate motion
in a 25-step chunk is orders of magnitude above 1e-4.
"""


def critic_rules_from_payload(
    payload: Sequence[Mapping[str, Any]],
) -> tuple[CriticRule, ...]:
    """Rehydrate frozen critic rules from their JSON form.

    Args:
        payload: The manifest's serialized rules.

    Returns:
        The rules, with predicates reconstructed.
    """
    rules: list[CriticRule] = []
    for item in payload:
        value = dict(item)
        value["evidence_ids"] = tuple(value.get("evidence_ids", ()))
        value["activation_conditions"] = tuple(
            CriticPredicate(**dict(condition))
            for condition in value.get("activation_conditions", ())
        )
        rules.append(CriticRule(**value))
    return tuple(rules)


def describe_dwell_semantics(*, execute_horizon: int) -> dict[str, Any]:
    """Describe what a rule's ``dwell_steps`` means on this family.

    Belongs in the campaign manifest next to the frozen rules: without it a
    reader cannot tell whether a dwell was written in simulator steps or chunks,
    and the two differ by ``execute_horizon``.

    Args:
        execute_horizon: Simulator steps executed per chunk.

    Returns:
        A JSON-friendly note.
    """
    return {
        "evidence_granularity": "chunk",
        "dwell_unit": "chunk",
        "sim_steps_per_chunk": int(execute_horizon),
        "note": (
            "RoboTwin is final_only: features exist only at chunk-final frames, "
            "so dwell_steps and cooldown_steps count chunks. A rule copied from "
            "a per-step family means "
            f"{int(execute_horizon)}x more simulator time here."
        ),
    }


def _arm_state(state: np.ndarray, arm: str) -> np.ndarray:
    """Slice one arm's 7-slot half out of a joint state.

    Args:
        state: The 14-dim joint state.
        arm: ``"left"`` or ``"right"``.

    Returns:
        The arm's 7 values.
    """
    return state[ARM_SLICES[arm]]


def _as_state(value: Sequence[float] | None) -> np.ndarray | None:
    """Coerce an observed state into a validated array.

    Args:
        value: A 14-dim state, or ``None``.

    Returns:
        The array, or ``None`` when the input is absent or malformed. A
        malformed state is treated as missing rather than raising: the Critic is
        proposal-only and must never be able to abort an episode.
    """
    if value is None:
        return None
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if array.shape != (ACTION_DIM,) or not np.isfinite(array).all():
        return None
    return array


def robotwin_state_features(state: Sequence[float] | None) -> dict[str, Any]:
    """The part of the Critic feature plane one state determines by itself.

    ``states.jsonl`` publishes these, and ``lifecycle._observed_critic_features``
    turns whatever it finds there into the vocabulary Stage2 may bind a critic
    rule to.  Sharing this function with
    :func:`extract_robotwin_critic_features` is what keeps the two from drifting:
    a name Stage2 is offered is by construction a name the runtime Critic can
    resolve.

    Motion and stall features are deliberately **not** here.  They are chunk
    deltas, and a campaign records ``states.jsonl`` per simulator step; emitting
    them from a per-step timeline under the Critic's names would advertise
    values the Critic never evaluates.

    Args:
        state: A 14-dim joint state, or ``None``.

    Returns:
        The state-determined features, under their runtime names.
    """

    parsed = _as_state(state)
    features: dict[str, Any] = {
        f"{FEATURE_PREFIX}.state_available": parsed is not None,
    }
    for arm in ARMS:
        prefix = f"{FEATURE_PREFIX}.arm.{arm}"
        features[f"{prefix}.gripper"] = (
            0.0 if parsed is None else float(_arm_state(parsed, arm)[GRIPPER_OFFSET])
        )
    return features


def extract_robotwin_critic_features(
    observation: Mapping[str, Any],
    *,
    chunk_index: int,
    executed_horizon: int,
    reward: float,
    terminated: bool,
    truncated: bool,
    previous_state: Sequence[float] | None = None,
    stall_counts: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    """Expose the audited, chunk-granular Critic feature plane.

    Args:
        observation: The chunk-final observation; ``state`` must carry the
            14-dim joint vector.
        chunk_index: Index of the chunk just executed.
        executed_horizon: Simulator steps this chunk advanced.
        reward: The chunk's reward.
        terminated: Termination flag.
        truncated: Truncation flag.
        previous_state: The previous chunk-final joint state, if any.
        stall_counts: Running per-arm stall counters from the caller, so the
            feature plane stays a pure function of its inputs.

    Returns:
        A flat, dotted-name feature dict for
        :class:`zetta.evolution.critic.TemporalCritic`.
    """
    state = _as_state(observation.get("state"))
    previous = _as_state(previous_state)
    counters = dict(stall_counts or {})

    features: dict[str, Any] = {
        GRANULARITY_FEATURE: "chunk",
        SIM_STEPS_PER_CHUNK_FEATURE: int(executed_horizon),
        f"{FEATURE_PREFIX}.chunk.index": int(chunk_index),
        f"{FEATURE_PREFIX}.chunk.reward": float(reward),
        f"{FEATURE_PREFIX}.chunk.terminated": bool(terminated),
        f"{FEATURE_PREFIX}.chunk.truncated": bool(truncated),
        # Shared with states.jsonl so the published vocabulary and the evaluated
        # plane cannot use two names for one quantity.
        **robotwin_state_features(state),
    }

    motions: dict[str, float] = {}
    for arm in ARMS:
        prefix = f"{FEATURE_PREFIX}.arm.{arm}"
        if state is None:
            features[f"{prefix}.joint_motion"] = 0.0
            features[f"{prefix}.gripper_motion"] = 0.0
            features[f"{prefix}.stalled_chunks"] = int(counters.get(arm, 0))
            motions[arm] = 0.0
            continue
        half = _arm_state(state, arm)
        if previous is None:
            joint_motion = 0.0
            gripper_motion = 0.0
        else:
            previous_half = _arm_state(previous, arm)
            joint_motion = float(
                np.linalg.norm(half[:GRIPPER_OFFSET] - previous_half[:GRIPPER_OFFSET])
            )
            gripper_motion = float(
                abs(half[GRIPPER_OFFSET] - previous_half[GRIPPER_OFFSET])
            )
        features[f"{prefix}.joint_motion"] = joint_motion
        features[f"{prefix}.gripper_motion"] = gripper_motion
        motions[arm] = joint_motion
        stalled = int(counters.get(arm, 0))
        # The very first chunk has no predecessor, so it cannot evidence a
        # stall; counting it would let a dwell of 1 fire before any motion was
        # ever possible.
        if previous is not None and joint_motion < MOTION_EPSILON:
            stalled += 1
        elif previous is not None:
            stalled = 0
        features[f"{prefix}.stalled_chunks"] = stalled

    if not motions or max(motions.values()) < MOTION_EPSILON:
        features[ACTIVE_ARM_FEATURE] = "none"
    else:
        features[ACTIVE_ARM_FEATURE] = max(motions, key=lambda arm: motions[arm])
    features[f"{FEATURE_PREFIX}.arm.motion_ratio"] = _motion_ratio(motions)
    return features


def _motion_ratio(motions: Mapping[str, float]) -> float:
    """Report how one-sided the chunk's motion was.

    Args:
        motions: Per-arm joint motion magnitudes.

    Returns:
        ``0.0`` when both arms moved equally (or neither did) and ``1.0`` when
        only one did. A rule can use this to notice that a task needing both
        hands is being attempted with one.
    """
    values = [float(value) for value in motions.values()]
    total = sum(values)
    if total < MOTION_EPSILON:
        return 0.0
    return float(abs(values[0] - values[1]) / total) if len(values) == 2 else 0.0


def next_stall_counts(features: Mapping[str, Any]) -> dict[str, int]:
    """Read the per-arm stall counters back out of a feature plane.

    The extractor is a pure function, so the caller carries the counters between
    chunks; this is the accessor that keeps the key names in one place.

    Args:
        features: A feature dict from :func:`extract_robotwin_critic_features`.

    Returns:
        Per-arm stall counts.
    """
    return {
        arm: int(features.get(f"{FEATURE_PREFIX}.arm.{arm}.stalled_chunks", 0))
        for arm in ARMS
    }


def arm_from_proposal(proposal: Mapping[str, Any]) -> str | None:
    """Read the arm a critic proposal is about, if it names one.

    A RoboTwin recovery cannot execute without an arm, so a proposal that omits
    it is not actionable; the caller decides whether that is a rejection or a
    request for clarification.

    Args:
        proposal: A critic proposal row.

    Returns:
        The canonical arm name, or ``None`` when the proposal names none.
    """
    from robots.robotwin.action_contract import ArmSelectionError, normalize_arm

    candidate = proposal.get("arm")
    if candidate is None:
        details = proposal.get("details")
        if isinstance(details, Mapping):
            candidate = details.get("arm")
    if candidate is None:
        return None
    try:
        return normalize_arm(str(candidate))
    except ArmSelectionError:
        return None
