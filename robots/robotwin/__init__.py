# Copyright (c) 2026 Zetta Contributors
"""RoboTwin 2.0 bimanual runtime used by rollout-evolution campaigns.

Only ``run_rollout`` may import ``rollout_runtime`` (enforced by
``tests/runtime/test_layering.py``); everything else here is contract-level and
simulator-free, so it imports cleanly in the minimal test environment.
"""

from robots.robotwin.action_contract import ARM_SLICES, ARMS, RoboTwinAction
from robots.robotwin.critic_runtime import extract_robotwin_critic_features
from robots.robotwin.recovery_controller import RecoveryController
from robots.robotwin.role1_actor import (
    ArmAwareRole1,
    Role1Decision,
    Role1EpisodeActor,
)
from robots.robotwin.role1_agent import ModelBackedRole1, Role1DecisionStore
from robots.robotwin.tool_catalog import DEFAULT_ROBOTWIN_TOOL_CATALOG

__all__ = [
    "ARMS",
    "ARM_SLICES",
    "DEFAULT_ROBOTWIN_TOOL_CATALOG",
    "ArmAwareRole1",
    "ModelBackedRole1",
    "RecoveryController",
    "RoboTwinAction",
    "Role1Decision",
    "Role1DecisionStore",
    "Role1EpisodeActor",
    "extract_robotwin_critic_features",
]
