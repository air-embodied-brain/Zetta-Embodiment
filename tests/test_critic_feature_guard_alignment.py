# Copyright (c) 2026 Zetta Contributors
from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

from rollout_runtime.backends.libero_critic import (
    CriticPredicate as RuntimeCriticPredicate,
)
from rollout_runtime.backends.libero_critic import CriticRule as RuntimeCriticRule
from rollout_runtime.backends.libero_critic import (
    TemporalCritic as RuntimeTemporalCritic,
)
from zetta.evolution.critic import TemporalCritic as EvolutionTemporalCritic
from zetta.evolution.models import CriticPredicate, CriticRule


def _evolution_critic() -> EvolutionTemporalCritic:
    return EvolutionTemporalCritic(
        (
            CriticRule(
                rule_id="eef-motion",
                title="EEF motion is low",
                feature="robot.eef.motion_m",
                operator="le",
                threshold=0.0,
                dwell_steps=1,
                cooldown_steps=0,
                proposal="recover EEF motion",
                evidence_ids=("segment-eef-motion",),
                activation_conditions=(
                    CriticPredicate(
                        feature="robot.eef.delta_available",
                        operator="eq",
                        threshold=True,
                    ),
                ),
            ),
        )
    )


def _runtime_critic() -> RuntimeTemporalCritic:
    return RuntimeTemporalCritic(
        (
            RuntimeCriticRule(
                rule_id="eef-motion",
                title="EEF motion is low",
                feature="robot.eef.motion_m",
                operator="le",
                threshold=0.0,
                dwell_steps=1,
                cooldown_steps=0,
                proposal="recover EEF motion",
                activation_conditions=(
                    RuntimeCriticPredicate(
                        feature="robot.eef.delta_available",
                        operator="eq",
                        threshold=True,
                    ),
                ),
            ),
        )
    )


@pytest.mark.parametrize(
    "critic_factory",
    [_evolution_critic, _runtime_critic],
    ids=["evolution", "libero-runtime"],
)
def test_activation_guard_controls_primary_feature_availability(
    critic_factory: Callable[[], Any],
) -> None:
    critic = critic_factory()

    assert (
        critic.evaluate(
            {"robot.eef.delta_available": False},
            step_index=1,
        )
        == []
    )

    with pytest.raises(
        KeyError,
        match="critic feature is unavailable: robot.eef.motion_m",
    ):
        critic.evaluate(
            {"robot.eef.delta_available": True},
            step_index=2,
        )

    proposals = critic.evaluate(
        {
            "robot.eef.delta_available": True,
            "robot.eef.motion_m": 0.0,
        },
        step_index=3,
    )
    assert [proposal["rule_id"] for proposal in proposals] == ["eef-motion"]
    assert proposals[0]["activation_conditions"][0]["observed_value"] is True
