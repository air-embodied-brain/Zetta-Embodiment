# Copyright (c) 2026 Zetta Contributors
"""Frozen native Panda task and recovery contracts, independent of Runtime."""

from __future__ import annotations

import dataclasses
import math
from typing import Any

from zetta.envs.maniskill.contracts import (
    ACTION_CONTRACT,
    PROPRIO_PROFILE,
    STATE_FIELDS,
)
from zetta.evolution.models import CandidateBundle, RecoveryStep, SafetyLayerConfig

TASK_ID = "PickCube-v1"
TASK_LANGUAGE = "Move the red cube to the green target position and hold it still."
SAFETY_LAYER = SafetyLayerConfig(
    action_contract=ACTION_CONTRACT,
    control_limits="normalized_finite_7d_reject_v1",
    joint_limit_shield="native_panda_pd_controller_v1",
)
EXECUTION_CONTRACT = {
    "critic": "proposal_only_per_control_step",
    "role1": "bounded_contract_review_v1",
    "actor": "deterministic_frozen_recovery_program",
    "reentry": "critic evaluation is suspended during active recovery",
    "step_stop_when": "budget_exhausted_or_success",
    "recovery_stop_condition": "official_success_or_budget",
    "recovery_fallback": "resume_vla",
    "success": "official info.success; termination alone is not success",
    "physics": "advances_only_on_action_step",
    "core_form": "per_slot",
    "reset_comparison": "maniskill_initial_state_sha256_v1",
    "reset_scene": "reconfigure_each_episode",
}


@dataclasses.dataclass(frozen=True)
class TaskContract:
    """The first verified task profile; additional tasks need their own contracts."""

    schema_version: int = 1
    env_id: str = TASK_ID
    instruction: str = TASK_LANGUAGE
    robot_uids: str = "panda"
    max_episode_steps: int = 50
    mani_skill_version: str = "3.0.1"
    sapien_version: str = "3.0.2"
    control_frequency: int = 20
    simulation_frequency: int = 100
    observation_profile: str = PROPRIO_PROFILE
    goal_visibility: str = "visible"
    action_contract: str = ACTION_CONTRACT
    policy_model_version: str = ""
    checkpoint_sha256: str = ""
    norm_stats_sha256: str = ""

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> TaskContract:
        contract = cls(**value)
        reference = cls()
        for field in (
            "schema_version",
            "env_id",
            "robot_uids",
            "max_episode_steps",
            "mani_skill_version",
            "sapien_version",
            "control_frequency",
            "simulation_frequency",
            "observation_profile",
            "goal_visibility",
            "action_contract",
        ):
            if getattr(contract, field) != getattr(reference, field):
                raise ValueError(
                    f"unsupported native ManiSkill task contract field: {field}"
                )
        if (
            not contract.instruction.strip()
            or not contract.policy_model_version.strip()
        ):
            raise ValueError(
                "task instruction and frozen policy model version are required"
            )
        for name in ("checkpoint_sha256", "norm_stats_sha256"):
            digest = getattr(contract, name)
            if len(digest) != 64 or any(v not in "0123456789abcdef" for v in digest):
                raise ValueError(f"{name} must be a lowercase SHA256 digest")
        return contract

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def diagnosis_contract(self) -> dict[str, Any]:
        return {
            "suite": "maniskill",
            "task": self.env_id,
            "language": self.instruction,
            "native_profile": self.as_dict(),
        }

    def validate_env(self, config: dict[str, Any]) -> None:
        required = {
            "env_id": self.env_id,
            "robot_uids": self.robot_uids,
            "max_episode_steps": self.max_episode_steps,
            "observation_profile": self.observation_profile,
            "goal_visibility": self.goal_visibility,
            "control_mode": "pd_ee_delta_pose",
            "action_dim": 7,
            "core_form": "per_slot",
            "obs_mode": "rgb",
            "auto_reset": False,
            "ignore_terminations": False,
            "use_full_state": False,
            "return_all_frames": True,
            "camera_height": 128,
            "camera_width": 128,
            "main_camera": "base_camera",
            "wrist_camera": None,
            "use_rel_reward": False,
        }
        for name, expected in required.items():
            if config.get(name) != expected:
                raise ValueError(f"campaign requires env_config.{name}={expected!r}")
        if config.get("extra_init_params", {}) != {
            "enhanced_determinism": True,
            "reconfiguration_freq": 1,
        }:
            raise ValueError(
                "formal task requires enhanced_determinism, reconfiguration_freq=1 and no task overrides"
            )


def observation_features(state: list[float], *, step_index: int) -> dict[str, Any]:
    if len(state) != len(STATE_FIELDS) or any(not math.isfinite(v) for v in state):
        raise ValueError("panda_proprio_v1 requires 18 finite joint states")
    return {
        "episode.step": step_index,
        **{
            f"maniskill.{name}": float(value)
            for name, value in zip(STATE_FIELDS, state, strict=True)
        },
        "maniskill.joint_speed": math.sqrt(sum(v * v for v in state[9:16])),
        "maniskill.gripper_width": float(state[7] + state[8]),
    }


FEATURE_NAMES = frozenset(observation_features([0.0] * 18, step_index=0))


def _schema(properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": properties,
        "required": required,
    }


_BUDGET = {"type": "integer", "minimum": 1, "maximum": 50}
_ACTION = {
    "type": "array",
    "minItems": 7,
    "maxItems": 7,
    "items": {"type": "number", "minimum": -1, "maximum": 1},
}
TOOL_CATALOG = {
    "schema_version": 1,
    "environment": "maniskill",
    "task": TASK_ID,
    "execution_contract": EXECUTION_CONTRACT,
    "features": sorted(FEATURE_NAMES),
    "tools": [
        {
            "name": "maniskill.vla",
            "description": "Replan from current RGB and proprioception for a bounded number of control steps.",
            "input_schema": _schema(
                {
                    "max_steps": _BUDGET,
                    "instruction": {"type": "string", "minLength": 1},
                    "actions_per_chunk": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 50,
                    },
                },
                ["max_steps"],
            ),
        },
        {
            "name": "maniskill.ee_delta",
            "description": "Repeat an explicit normalized Panda EE controller command. Translation is in the robot root frame; rotation follows the pinned native controller, and gripper -1 closes / +1 opens. This is not a meter/radian API.",
            "input_schema": _schema(
                {"max_steps": _BUDGET, "action": _ACTION}, ["max_steps", "action"]
            ),
        },
        {
            "name": "maniskill.hold",
            "description": "Zero EE delta while preserving the last commanded gripper target for a bounded number of steps.",
            "input_schema": _schema({"max_steps": _BUDGET}, ["max_steps"]),
        },
    ],
}


def validate_step(step: RecoveryStep) -> None:
    schemas = {v["name"]: v["input_schema"] for v in TOOL_CATALOG["tools"]}
    if step.tool not in schemas:
        raise ValueError(f"unsupported ManiSkill tool: {step.tool}")
    schema = schemas[step.tool]
    if set(step.parameters) - set(schema["properties"]) or set(
        schema["required"]
    ) - set(step.parameters):
        raise ValueError("missing or unsupported recovery parameters")
    budget = step.parameters["max_steps"]
    if type(budget) is not int or not 1 <= budget <= 50:
        raise ValueError("recovery budget must be an integer in [1, 50]")
    if step.tool == "maniskill.ee_delta":
        from zetta.envs.maniskill.contracts import validate_native_actions

        validate_native_actions([step.parameters["action"]])
    if step.tool == "maniskill.vla":
        horizon = step.parameters.get("actions_per_chunk", 4)
        if type(horizon) is not int or not 1 <= horizon <= 50:
            raise ValueError("actions_per_chunk must be in [1, 50]")
        prompt = step.parameters.get("instruction")
        if prompt is not None and (not isinstance(prompt, str) or not prompt.strip()):
            raise ValueError("instruction must be nonempty")
    if step.stop_when != "budget_exhausted_or_success":
        raise ValueError("unsupported step stop_when")


def validate_bundle(bundle: CandidateBundle) -> None:
    if bundle.tool_plugin is not None:
        raise ValueError("native ManiSkill does not execute tool plugins")
    for rule in bundle.critic_rules:
        if any(
            condition.operator == "stagnant" for condition in rule.activation_conditions
        ):
            raise ValueError("activation conditions require instantaneous operators")
        for predicate in (rule, *rule.activation_conditions):
            if predicate.feature not in FEATURE_NAMES:
                raise ValueError(f"unavailable critic feature: {predicate.feature}")
            if predicate.operator not in {
                "lt",
                "le",
                "gt",
                "ge",
                "eq",
                "ne",
                "stagnant",
            }:
                raise ValueError("unsupported critic operator")
            if not isinstance(predicate.threshold, (int, float)) or not math.isfinite(
                predicate.threshold
            ):
                raise ValueError("critic threshold must be finite")
    for recovery in bundle.recovery_rules:
        if (
            recovery.stop_condition != "official_success_or_budget"
            or recovery.fallback != "resume_vla"
        ):
            raise ValueError("unsupported recovery stop condition or fallback")
        for step in recovery.steps:
            validate_step(step)
