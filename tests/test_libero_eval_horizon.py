# Copyright (c) 2026 RPent Contributors
from __future__ import annotations

import pytest

from robots.libero.env_server import build_env_cfg
from robots.libero.evolution_defaults import libero_horizon_contract


@pytest.mark.parametrize(
    ("suite", "policy_actions", "episode_cap"),
    [
        ("libero_10_task", 520, 530),
        ("libero_10_swap", 520, 530),
        ("libero_goal_task", 300, 310),
        ("libero_goal_swap", 300, 310),
    ],
)
def test_libero_pro_horizon_follows_base_suite(
    suite: str, policy_actions: int, episode_cap: int
) -> None:
    contract = libero_horizon_contract(suite)

    assert contract["policy_action_horizon"] == policy_actions
    assert contract["wait_steps"] == 10
    assert contract["max_episode_steps"] == episode_cap
    assert contract["is_standard"] is True


def test_explicit_nonstandard_horizon_is_preserved_and_flagged() -> None:
    contract = libero_horizon_contract(
        "libero_10_task", max_actions=300, wait_steps=10
    )

    assert contract["standard_policy_action_horizon"] == 520
    assert contract["policy_action_horizon"] == 300
    assert contract["max_episode_steps"] == 310
    assert contract["is_standard"] is False


def test_unknown_libero_suite_has_no_silent_horizon_fallback() -> None:
    with pytest.raises(ValueError, match="no official OpenPI action horizon"):
        libero_horizon_contract("libero_unknown")


def test_rlinf_truncates_before_robosuite_internal_horizon():
    cfg = build_env_cfg(max_episode_steps=530)

    assert cfg.max_episode_steps == 530
    assert cfg.max_steps_per_rollout_epoch == 530
    assert cfg.init_params.horizon > cfg.max_episode_steps
    assert cfg.init_params.horizon == 1530
