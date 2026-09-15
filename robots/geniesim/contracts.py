# Copyright (c) 2026 Zetta Contributors
"""Frozen recovery vocabulary for the G2 joint-control profile."""

from __future__ import annotations

import math
from typing import Any

from zetta.envs.geniesim.config import TASK_NAME
from zetta.evolution.models import CandidateBundle, RecoveryStep, SafetyLayerConfig

SAFETY_LAYER = SafetyLayerConfig(
    action_contract="geniesim_g2_absolute_joint_radians_16_v1",
    control_limits="native_articulation_limits_reject_v1",
    joint_limit_shield="native_articulation_limits_reject_v1",
)

EXECUTION_CONTRACT = {
    "actor": "deterministic_frozen_recovery_program",
    "critic": "proposal_only_per_control_step",
    "activation": "critic activation_conditions are the executable prerequisites",
    "reentry": "critic evaluation is suspended during active recovery",
    "step_stop_when": "budget_exhausted_or_success",
    "recovery_stop_condition": "official_success_or_budget",
    "recovery_fallback": "resume_vla",
    "prose": "precondition and safety_constraints describe the frozen program",
    "success": "official scores.E2E == 1 only",
    "physics": "free_running_between_requests",
    "reset_comparison": "geniesim_g2_joint_reset_v1: scenario identity plus 0.02 rad per joint; cameras audited separately",
}

TOOL_CATALOG = {
    "schema_version": 1,
    "environment": "geniesim",
    "task": TASK_NAME,
    "execution_contract": EXECUTION_CONTRACT,
    "tools": [
        {
            "name": "geniesim.vla",
            "description": "Replan a bounded joint-action program from current RGB and proprioception; both arms remain policy-controlled.",
            "parameters": {
                "instruction": "optional nonempty prompt; omitted uses official task instruction",
                "actions_per_chunk": "integer 1..64, default 5",
                "max_steps": "required integer 1..300, total control-step budget",
            },
            "input_schema": {
                "type": "object",
                "additionalProperties": False,
                "required": ["max_steps"],
                "properties": {
                    "instruction": {"type": "string", "minLength": 1},
                    "actions_per_chunk": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 64,
                    },
                    "max_steps": {"type": "integer", "minimum": 1, "maximum": 300},
                },
            },
        },
        {
            "name": "geniesim.hold",
            "description": "Hold all 16 observed native joints for a bounded control-step budget.",
            "parameters": {"max_steps": "required integer 1..64"},
            "input_schema": {
                "type": "object",
                "additionalProperties": False,
                "required": ["max_steps"],
                "properties": {
                    "max_steps": {"type": "integer", "minimum": 1, "maximum": 64}
                },
            },
        },
    ],
}


def observation_features(
    state: list[float], *, step_index: int, extras: dict[str, Any]
) -> dict[str, Any]:
    if len(state) != 21 or any(
        isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
        for v in state
    ):
        raise ValueError("expected 21 finite Genie Sim joint states")
    features = {
        f"geniesim.joint.{index}": float(value) for index, value in enumerate(state)
    }
    features.update(
        {
            "episode.step": step_index,
            "geniesim.official_success": bool(extras.get("official_success", False)),
            "geniesim.evaluation_step": int(extras.get("evaluation_step", 0)),
        }
    )
    return features


FEATURE_NAMES = frozenset(observation_features([0.0] * 21, step_index=0, extras={}))


def validate_step(step: RecoveryStep) -> None:
    if step.tool not in {"geniesim.vla", "geniesim.hold"}:
        raise ValueError(f"unsupported Genie Sim recovery tool: {step.tool}")
    parameters = step.parameters
    allowed = (
        {"max_steps", "instruction", "actions_per_chunk"}
        if step.tool == "geniesim.vla"
        else {"max_steps"}
    )
    if set(parameters) - allowed:
        raise ValueError("unsupported Genie Sim recovery parameters")
    maximum = 300 if step.tool == "geniesim.vla" else 64
    budget = parameters.get("max_steps")
    if type(budget) is not int or not 1 <= budget <= maximum:
        raise ValueError(f"recovery max_steps must be an integer in [1, {maximum}]")
    if step.tool == "geniesim.vla":
        horizon = parameters.get("actions_per_chunk", 5)
        if type(horizon) is not int or not 1 <= horizon <= 64:
            raise ValueError("recovery actions_per_chunk must be in [1, 64]")
        instruction = parameters.get("instruction")
        if instruction is not None and (
            not isinstance(instruction, str) or not instruction.strip()
        ):
            raise ValueError("recovery instruction must be nonempty")
    if step.stop_when != "budget_exhausted_or_success":
        raise ValueError("unsupported Genie Sim recovery step stop_when")


def validate_bundle(bundle: CandidateBundle) -> None:
    if bundle.tool_plugin is not None:
        raise ValueError("Genie Sim does not execute tool-plugin bundles")
    for rule in bundle.critic_rules:
        for predicate in (rule, *rule.activation_conditions):
            if predicate.feature not in FEATURE_NAMES:
                raise ValueError(
                    f"unsupported Genie Sim critic feature: {predicate.feature}"
                )
            if predicate.operator not in {
                "lt",
                "le",
                "gt",
                "ge",
                "eq",
                "ne",
                "stagnant",
            }:
                raise ValueError("unsupported Genie Sim critic operator")
            if predicate.operator not in {"eq", "ne"} and (
                isinstance(predicate.threshold, bool)
                or not isinstance(predicate.threshold, (int, float))
                or not math.isfinite(predicate.threshold)
            ):
                raise ValueError("critic numeric threshold must be finite")
    for recovery in bundle.recovery_rules:
        if (
            recovery.stop_condition != "official_success_or_budget"
            or recovery.fallback != "resume_vla"
        ):
            raise ValueError(
                "unsupported Genie Sim recovery stop condition or fallback"
            )
        for step in recovery.steps:
            validate_step(step)
