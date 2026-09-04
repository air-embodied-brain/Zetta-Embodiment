# Copyright (c) 2026 Zetta Contributors
"""Proposal-only temporal critic evaluator used inside chunk execution."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from zetta.evolution.models import CriticPredicate, CriticRule


def resolve_feature(observation: dict[str, Any], dotted_path: str) -> Any:
    # RoboCasa / GR00T observations intentionally use flattened canonical
    # names such as ``privileged.dishwasher.rack.residual_to_success``.  A
    # nested diagnostic fixture may use the equivalent dictionary hierarchy,
    # so support both without rewriting the frozen rule.
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
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"critic numeric operator received {type(value).__name__}")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("critic feature must be finite")
    return number


@dataclass
class _RuleState:
    consecutive: int = 0
    cooldown_remaining: int = 0
    history: list[Any] = field(default_factory=list)


class TemporalCritic:
    """Evaluate frozen rules; emits proposals but never mutates an environment."""

    def __init__(self, rules: tuple[CriticRule, ...]) -> None:
        self.rules = rules
        self._state = {rule.rule_id: _RuleState() for rule in rules}

    def reset(self) -> None:
        self._state = {rule.rule_id: _RuleState() for rule in self.rules}

    def evaluate(
        self, observation: dict[str, Any], *, step_index: int
    ) -> list[dict[str, Any]]:
        proposals: list[dict[str, Any]] = []
        for rule in self.rules:
            state = self._state[rule.rule_id]
            if not all(
                self._predicate(condition, observation)
                for condition in rule.activation_conditions
            ):
                # A false activation guard starts a fresh temporal epoch. This
                # prevents pre-contact history satisfying a post-contact rule.
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
            required_consecutive = (
                1 if rule.operator == "stagnant" else rule.dwell_steps
            )
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
    def _predicate(
        cls, predicate: CriticPredicate, observation: dict[str, Any]
    ) -> bool:
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
