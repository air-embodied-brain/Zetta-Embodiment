# Copyright (c) 2026 Zetta Contributors
"""Environment-neutral Critic features for audited LIBERO-Pro rollouts."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from zetta.evolution.models import CriticPredicate, CriticRule


def critic_rules_from_payload(
    payload: Sequence[Mapping[str, Any]],
) -> tuple[CriticRule, ...]:
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


def extract_libero_critic_features(
    observation: Mapping[str, Any],
    *,
    step_index: int,
    reward: float,
    terminated: bool,
    truncated: bool,
    privileged_state: Mapping[str, Any] | None = None,
    action: Sequence[float] | None = None,
    previous_eef: Sequence[float] | None = None,
) -> dict[str, Any]:
    """Expose the audited Critic-only feature plane to learned rules.

    The first eight RLinF ``states`` values are the same robot quantities used
    by the existing LIBERO primitives: EEF position followed by robot state and
    the two finger positions.  Simulator truth is accepted only from the
    dedicated privileged sensor and is never inserted into the Actor/VLA
    observation.
    """

    states = np.asarray(observation.get("states", ()), dtype=np.float64).reshape(-1)
    if states.size < 8 or not np.isfinite(states[:8]).all():
        raise ValueError("LIBERO critic requires eight finite robot state values")
    features: dict[str, Any] = {
        "episode.step_index": int(step_index),
        "episode.reward": float(reward),
        "episode.terminated": bool(terminated),
        "episode.truncated": bool(truncated),
        "robot.eef.x": float(states[0]),
        "robot.eef.y": float(states[1]),
        "robot.eef.z": float(states[2]),
        "robot.gripper.opening": float(abs(states[6]) + abs(states[7])),
        "robot.eef.delta_available": previous_eef is not None,
        "command.available": action is not None,
    }
    for index, value in enumerate(states):
        if np.isfinite(value):
            features[f"robot.state.{index}"] = float(value)

    if previous_eef is not None:
        previous = np.asarray(previous_eef, dtype=np.float64).reshape(-1)
        if previous.size != 3 or not np.isfinite(previous).all():
            raise ValueError("previous LIBERO EEF position must be three finite values")
        realized = states[:3] - previous
        for axis, value in zip(("x", "y", "z"), realized, strict=True):
            features[f"robot.eef.delta.{axis}"] = float(value)
        features["robot.eef.motion_m"] = float(np.linalg.norm(realized))

    if action is not None:
        command = np.asarray(action, dtype=np.float64).reshape(-1)
        if command.size != 7 or not np.isfinite(command).all():
            raise ValueError("LIBERO Critic action must contain seven finite values")
        for axis, value in zip(("x", "y", "z"), command[:3], strict=True):
            features[f"command.translation.{axis}"] = float(value)
        for axis, value in zip(("x", "y", "z"), command[3:6], strict=True):
            features[f"command.rotation.{axis}"] = float(value)
        command_norm = float(np.linalg.norm(command[:3]))
        features["command.translation.norm"] = command_norm
        features["command.rotation.norm"] = float(np.linalg.norm(command[3:6]))
        features["command.gripper"] = float(command[6])
        if previous_eef is not None:
            realized = states[:3] - np.asarray(previous_eef, dtype=np.float64)
            realized_norm = float(np.linalg.norm(realized))
            directional = command_norm > 1e-9 and realized_norm > 1e-9
            features["command.realization.direction_available"] = directional
            features["command.realization.eef_motion_m"] = realized_norm
            features["command.realization.stalled"] = (
                command_norm >= 0.05 and realized_norm <= 0.0005
            )
            if directional:
                features["command.realization.direction_cosine"] = float(
                    np.dot(command[:3], realized) / (command_norm * realized_norm)
                )

    for key, value in (privileged_state or {}).items():
        name = str(key)
        if not name.startswith("privileged."):
            raise ValueError(f"invalid LIBERO privileged feature name: {name}")
        if isinstance(value, (bool, str)):
            features[name] = value
        elif isinstance(value, (int, float, np.integer, np.floating)):
            number = float(value)
            if np.isfinite(number):
                features[name] = int(value) if isinstance(value, (int, np.integer)) else number
    return features


LIBERO_CRITIC_FEATURES = tuple(
    [
        "episode.step_index",
        "episode.reward",
        "episode.terminated",
        "episode.truncated",
        "robot.eef.x",
        "robot.eef.y",
        "robot.eef.z",
        "robot.gripper.opening",
        "robot.eef.delta_available",
        "robot.eef.delta.x",
        "robot.eef.delta.y",
        "robot.eef.delta.z",
        "robot.eef.motion_m",
        "command.available",
        "command.translation.x",
        "command.translation.y",
        "command.translation.z",
        "command.translation.norm",
        "command.rotation.x",
        "command.rotation.y",
        "command.rotation.z",
        "command.rotation.norm",
        "command.gripper",
        "command.realization.direction_available",
        "command.realization.direction_cosine",
        "command.realization.eef_motion_m",
        "command.realization.stalled",
        "privileged.available",
        "privileged.task.semantic_available",
        "privileged.task.success",
        "privileged.task.goal.predicate_count",
        "privileged.task.goal.evaluable_count",
        "privileged.task.goal.satisfied_count",
        "privileged.task.goal.progress_available",
        "privileged.task.goal.progress",
        "privileged.task.primary_relation",
        "privileged.task.primary_relation_satisfied",
        "privileged.task.target_identity_confidence",
        "privileged.task.manipulated_object.name",
        "privileged.task.manipulated_object.position.x",
        "privileged.task.manipulated_object.position.y",
        "privileged.task.manipulated_object.position.z",
        "privileged.task.manipulated_object.distance_to_eef_m",
        "privileged.task.manipulated_object.distance_to_target_m",
        "privileged.task.manipulated_object.gripper_contact",
        "privileged.task.manipulated_object.robot_contact",
        "privileged.task.manipulated_object.grasped",
        "privileged.task.manipulated_object.mechanical_engagement",
        "privileged.task.manipulated_object.coupled",
        "privileged.task.manipulated_object.ever_grasped",
        "privileged.task.manipulated_object.retained",
        "privileged.task.manipulated_object.released_now",
        "privileged.task.manipulated_object.ever_released",
        "privileged.task.manipulated_object.in_target",
        "privileged.task.manipulated_object.target_progress_available",
        "privileged.task.manipulated_object.target_distance_delta_m",
        "privileged.task.manipulated_object.target_progress_m",
        "privileged.task.stage.index",
        "privileged.task.stage.name",
        "privileged.task.target.name",
        "privileged.task.target.position.x",
        "privileged.task.target.position.y",
        "privileged.task.target.position.z",
        "privileged.task.target.distance_to_eef_m",
        "privileged.task.target.gripper_contact",
        "privileged.task.target.robot_contact",
        "privileged.task.target.is_nearest_entity",
        "privileged.task.target.distance_rank",
        "privileged.task.nearest_entity.name",
        "privileged.task.nearest_entity.distance_m",
        "privileged.contact.robot.count",
        "privileged.contact.gripper.count",
        "privileged.contact.force_available",
        "privileged.contact.max_normal_force_n",
        "privileged.joint.count",
    ]
    + [f"robot.state.{index}" for index in range(64)]
)


__all__ = [
    "LIBERO_CRITIC_FEATURES",
    "critic_rules_from_payload",
    "extract_libero_critic_features",
]
