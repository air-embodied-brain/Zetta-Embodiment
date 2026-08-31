# Copyright (c) 2026 Zetta Contributors
"""Standalone Critic rule evaluator (a ``rollout_runtime``-exclusive reimplementation).

This is an equivalent reimplementation of ``zetta.evolution.critic.TemporalCritic``
and ``zetta.evolution.models.{CriticRule,CriticPredicate}``. The layering
guard (``tests/runtime/test_layering.py`` rule 1) forbids ``rollout_runtime/**``
from importing ``zetta`` / ``robots``, so this cannot directly import and
reuse that code; instead it maintains an independent copy kept in exact
semantic alignment.

Alignment baselines (**must remain in exact semantic alignment, must not
evolve independently**):

- ``zetta/evolution/critic.py``: evaluation algorithm of
  ``TemporalCritic``/``resolve_feature``;
- ``zetta/evolution/models.py``: field shape of ``CriticRule``/``CriticPredicate``;
- ``robots/libero/critic_runtime.py``: evaluation semantics of
  ``critic_rules_from_payload``/``extract_libero_critic_features``.

Any change to the evaluation logic on either side must be mirrored on the
other, otherwise the two implementations would compute different proposal
timings given the same ``critic_rules`` payload and the same trajectory --
this is precisely why the three-way comparison requires field-by-field
verification.

Dependency surface: stdlib + numpy (layering guard rule 1).
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "CriticPredicate",
    "CriticRule",
    "TemporalCritic",
    "canonical_rules_fingerprint",
    "critic_rules_from_payload",
    "extract_libero_critic_features",
    "resolve_feature",
]


# --------------------------------------------------------------------- Rule shapes


@dataclass(frozen=True)
class CriticPredicate:
    """One activation condition (fields aligned one-to-one with
    ``zetta.evolution.models.CriticPredicate``).

    Attributes:
        feature: Dotted feature name (e.g. ``"privileged.task.success"``).
        operator: One of ``eq`` / ``ne`` / ``lt`` / ``le`` / ``gt`` / ``ge``.
        threshold: Comparison threshold.
    """

    feature: str
    operator: str
    threshold: Any

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dict representation.

        Returns:
            ``{"feature": ..., "operator": ..., "threshold": ...}``.
        """
        return {
            "feature": self.feature,
            "operator": self.operator,
            "threshold": self.threshold,
        }


@dataclass(frozen=True)
class CriticRule:
    """A frozen Critic rule (fields aligned one-to-one with
    ``zetta.evolution.models.CriticRule``).

    The field set deliberately retains only what ``TemporalCritic.evaluate``
    and the ``critic-recovery.json`` deployment configuration actually use;
    the upstream ``zetta`` side has additional audit/evidence-related fields
    (``evidence_ids`` etc.) that are not needed for evaluation here and are
    therefore not copied.

    Attributes:
        rule_id: Rule identifier, returned as-is in the proposal.
        feature: The dotted feature name being evaluated.
        operator: ``stagnant`` / ``eq`` / ``ne`` / ``lt`` / ``le`` / ``gt`` / ``ge``.
        threshold: Comparison threshold (for ``stagnant``, the maximum
            allowed historical fluctuation).
        dwell_steps: Dwell window length (used as the historical window for
            ``stagnant``; used as the qualifying-count target for other operators).
        cooldown_steps: Number of cooldown steps after a single hit.
        activation_conditions: Preconditions that must all hold before evaluation.
        proposal: Human-readable description attached on a hit, passed
            through to the proposal.
        safety_only: Whether this is a safety-constraint rule (only affects
            proposal metadata, not the evaluation algorithm).
        title: Optional title, only affects proposal metadata.
    """

    rule_id: str
    feature: str
    operator: str
    threshold: Any
    dwell_steps: int = 1
    cooldown_steps: int = 0
    activation_conditions: tuple[CriticPredicate, ...] = ()
    proposal: str = ""
    safety_only: bool = False
    title: str = ""


def critic_rules_from_payload(payload: Any) -> tuple[CriticRule, ...]:
    """Parse a JSON-compatible payload (such as the ``critic_rules`` field of
    ``critic-recovery.json``) into a tuple of ``CriticRule``.

    The parsing semantics are aligned exactly with
    ``robots.libero.critic_runtime.critic_rules_from_payload``: extra keys
    (e.g. ``evidence_ids``) are ignored rather than raising an error, so that
    both implementations can share the same deployment configuration JSON.

    Args:
        payload: Array of ``critic_rules``, each item a mapping.

    Returns:
        A frozen tuple of rules.

    Raises:
        ValueError: ``payload`` is not an array, or an item is missing a
            required field.
    """
    if not isinstance(payload, (list, tuple)):
        raise ValueError("critic_rules must be an array")
    rules: list[CriticRule] = []
    for item in payload:
        if not isinstance(item, dict):
            raise ValueError("each critic rule must be a mapping")
        value = dict(item)
        conditions = tuple(
            CriticPredicate(
                feature=str(condition["feature"]),
                operator=str(condition["operator"]),
                threshold=condition["threshold"],
            )
            for condition in value.get("activation_conditions", ())
        )
        rules.append(
            CriticRule(
                rule_id=str(value["rule_id"]),
                feature=str(value["feature"]),
                operator=str(value["operator"]),
                threshold=value["threshold"],
                dwell_steps=int(value.get("dwell_steps", 1)),
                cooldown_steps=int(value.get("cooldown_steps", 0)),
                activation_conditions=conditions,
                proposal=str(value.get("proposal", "")),
                safety_only=bool(value.get("safety_only", False)),
                title=str(value.get("title", "")),
            )
        )
    return tuple(rules)


def canonical_rules_fingerprint(payload: Any) -> str:
    """Compute the sha256 fingerprint of a canonicalized JSON representation
    of the ``critic_rules`` payload.

    Aligned exactly with ``zetta.evolution.jsonio.canonical_sha256``
    (``ensure_ascii=False``, ``sort_keys=True``, ``separators=(",", ":")``,
    ``default=str``, with a trailing ``"\\n"`` appended), used to detect
    "``critic_rules`` changed within the same episode" (aligned with the
    rejection semantics of the legacy ``LiberoEnvClient._configure_critic``).
    The layering guard forbids importing ``zetta`` directly, hence this
    independent implementation.

    Args:
        payload: Array of ``critic_rules`` (JSON-compatible structure).

    Returns:
        64-character hex sha256 digest.
    """
    encoded = (
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        + "\n"
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


# --------------------------------------------------------------------- Evaluation


def resolve_feature(observation: dict[str, Any], dotted_path: str) -> Any:
    """Look up a value from the feature dict by dotted path (aligned exactly
    with ``zetta.evolution.critic.resolve_feature``).

    Tries the whole path as a flat key lookup first (this is exactly what
    ``extract_libero_critic_features`` produces, e.g. a flat key
    ``"privileged.task.success"``); only if that misses does it split on
    ``.`` and descend level by level, to remain compatible with nested-dict
    diagnostic fixtures.

    Args:
        observation: Feature dict.
        dotted_path: Dotted feature name.

    Returns:
        The corresponding value.

    Raises:
        KeyError: Neither lookup path found a value.
    """
    if dotted_path in observation:
        return observation[dotted_path]
    value: Any = observation
    for part in dotted_path.split("."):
        if isinstance(value, dict) and part in value:
            value = value[part]
        else:
            raise KeyError(f"critic feature is unavailable: {dotted_path}")
    return value


def _numeric(value: Any) -> float:
    """Convert a value to a finite float, rejecting bool and non-numeric types.

    Args:
        value: The value to convert.

    Returns:
        A finite float.

    Raises:
        TypeError: ``value`` is a bool or a non-numeric type.
        ValueError: The converted result is not finite.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"critic numeric operator received {type(value).__name__}")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("critic feature must be finite")
    return number


@dataclass
class _RuleState:
    """Cross-chunk persistent state for a single rule (aligned with
    ``zetta.evolution.critic._RuleState``).

    Attributes:
        consecutive: Count of consecutive condition satisfactions.
        cooldown_remaining: Remaining cooldown steps.
        history: The sliding history window used by the ``stagnant`` operator.
    """

    consecutive: int = 0
    cooldown_remaining: int = 0
    history: list[Any] = field(default_factory=list)


class TemporalCritic:
    """Evaluates frozen rules; only produces proposals, never mutates
    environment state.

    Aligned exactly with the evaluation algorithm of
    ``zetta.evolution.critic.TemporalCritic``: the state dict is indexed by
    ``rule_id`` and persists across multiple ``evaluate`` calls (should be
    constructed only once per episode and bound to the corresponding
    slot/session; see the integration point in ``rlinf_env.py``).
    """

    def __init__(self, rules: tuple[CriticRule, ...]) -> None:
        """Construct the evaluator.

        Args:
            rules: The frozen set of rules.
        """
        self.rules = rules
        self._state = {rule.rule_id: _RuleState() for rule in rules}

    def reset(self) -> None:
        """Clear cross-chunk state for all rules (called at the start of a new episode)."""
        self._state = {rule.rule_id: _RuleState() for rule in self.rules}

    def evaluate(
        self, observation: dict[str, Any], *, step_index: int
    ) -> list[dict[str, Any]]:
        """Evaluate all rules against one frame of features, returning the
        list of proposals that fired.

        Args:
            observation: Flat feature dict produced by
                ``extract_libero_critic_features``.
            step_index: The current physical action's step number, passed
                through into the proposal.

        Returns:
            List of proposals that fired; an empty list means no rule fired
            at this step.
        """
        proposals: list[dict[str, Any]] = []
        for rule in self.rules:
            state = self._state[rule.rule_id]
            if not all(
                self._predicate(condition, observation)
                for condition in rule.activation_conditions
            ):
                # Clear history when activation conditions are not met:
                # avoids pre-contact history spuriously triggering rules
                # that are only meaningful post-contact.
                state.consecutive = 0
                state.cooldown_remaining = 0
                state.history.clear()
                continue
            value = resolve_feature(observation, rule.feature)
            state.history.append(value)
            if len(state.history) > rule.dwell_steps:
                state.history.pop(0)
            if state.cooldown_remaining > 0:
                state.cooldown_remaining -= 1
                state.consecutive = 0
                continue
            condition = self._condition(rule, value, state.history)
            state.consecutive = state.consecutive + 1 if condition else 0
            required_consecutive = 1 if rule.operator == "stagnant" else rule.dwell_steps
            if state.consecutive >= required_consecutive:
                proposals.append(
                    {
                        "rule_id": rule.rule_id,
                        "step_index": step_index,
                        "feature": rule.feature,
                        "observed_value": value,
                        "activation_conditions": [
                            {
                                **condition.as_dict(),
                                "observed_value": resolve_feature(
                                    observation, condition.feature
                                ),
                            }
                            for condition in rule.activation_conditions
                        ],
                        "proposal": rule.proposal,
                        "safety_only": rule.safety_only,
                        "environment_write": False,
                    }
                )
                state.consecutive = 0
                state.cooldown_remaining = rule.cooldown_steps
        return proposals

    @classmethod
    def _predicate(cls, predicate: CriticPredicate, observation: dict[str, Any]) -> bool:
        """Evaluate a single activation condition.

        Args:
            predicate: The activation condition.
            observation: Feature dict.

        Returns:
            Whether it is satisfied.
        """
        value = resolve_feature(observation, predicate.feature)
        if predicate.operator == "eq":
            return value == predicate.threshold
        if predicate.operator == "ne":
            return value != predicate.threshold
        observed = _numeric(value)
        threshold = _numeric(predicate.threshold)
        return {
            "lt": observed < threshold,
            "le": observed <= threshold,
            "gt": observed > threshold,
            "ge": observed >= threshold,
        }[predicate.operator]

    @staticmethod
    def _condition(rule: CriticRule, value: Any, history: list[Any]) -> bool:
        """Evaluate the rule's primary operator (operators other than
        ``stagnant`` share the same comparison logic as activation conditions).

        Args:
            rule: The rule.
            value: The observed value at this step.
            history: The sliding history window.

        Returns:
            Whether the primary condition is satisfied.
        """
        if rule.operator == "stagnant":
            if len(history) < rule.dwell_steps:
                return False
            numbers = [_numeric(item) for item in history]
            return max(numbers) - min(numbers) <= _numeric(rule.threshold)
        if rule.operator == "eq":
            return value == rule.threshold
        if rule.operator == "ne":
            return value != rule.threshold
        observed = _numeric(value)
        threshold = _numeric(rule.threshold)
        return {
            "lt": observed < threshold,
            "le": observed <= threshold,
            "gt": observed > threshold,
            "ge": observed >= threshold,
        }[rule.operator]


# ------------------------------------------------------------- LIBERO feature extraction


def extract_libero_critic_features(
    observation: dict[str, Any],
    *,
    step_index: int,
    reward: float,
    terminated: bool,
    truncated: bool,
    privileged_state: dict[str, Any] | None = None,
    action: Any = None,
    previous_eef: Any = None,
) -> dict[str, Any]:
    """Compress one frame of obs + privileged state into a Critic feature
    plane (aligned exactly with
    ``robots.libero.critic_runtime.extract_libero_critic_features``).

    Args:
        observation: The family's 5-key obs (must at least contain ``"states"``).
        step_index: The current physical action step number.
        reward: The reward at this step.
        terminated: Whether this step is terminal.
        truncated: Whether this step is truncated.
        privileged_state: The ``privileged.*`` dict returned by the
            ``libero.critic_state`` extension.
        action: The raw action executed this step (7-dim).
        previous_eef: The previous step's EEF position (3-dim), used to
            compute the actually realized displacement.

    Returns:
        Flat feature dict, keys are all dotted paths.

    Raises:
        ValueError: ``states`` has fewer than eight elements, or
            ``action``/``previous_eef`` has the wrong dimensionality.
    """
    import numpy as np

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
