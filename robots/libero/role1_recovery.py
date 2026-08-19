# Copyright (c) 2026 Zetta Contributors
"""Episode-level Role1 approval for frozen LIBERO recovery steps."""

from __future__ import annotations

import base64
import hashlib
import io
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import imageio.v3 as iio
import numpy as np

from robots.libero.evolution_defaults import LIBERO_PRIVILEGED_EVIDENCE_DEFAULT
from robots.libero.tool_catalog import role1_tool_name, short_tool_name
from robots.robocasa.role1_agent import (
    CriticProposal,
    Role1Event,
    Role1ModelAdapter,
    ToolProposal,
)
from zetta.evolution.jsonio import atomic_write_json, canonical_sha256


class LiberoRecoveryActorError(RuntimeError):
    """Role1 did not authorize an executable frozen LIBERO recovery step."""


@dataclass(frozen=True, slots=True)
class LiberoRecoveryResult:
    decision_id: str
    selected_tool: str
    executed_horizon: int
    result: dict[str, Any]


def _is_private_geometry_field(key: object) -> bool:
    """Return true for fields that can reveal an absolute simulator pose."""

    normalized = str(key).strip().lower().replace("-", "_")
    segments = normalized.replace(".", "_").split("_")
    return bool(
        {"position", "orientation", "pose", "xyz", "quat", "quaternion", "pos"}
        & set(segments)
    ) or "target_offset" in normalized


def _sanitize_actor_visible(value: Any) -> Any:
    """Recursively remove private geometry from Role1-visible payloads."""

    if isinstance(value, Mapping):
        return {
            str(key): _sanitize_actor_visible(item)
            for key, item in value.items()
            if not _is_private_geometry_field(key)
        }
    if isinstance(value, (list, tuple)):
        return [_sanitize_actor_visible(item) for item in value]
    return value


def _image_payloads(observation: Mapping[str, Any]) -> dict[str, str]:
    result: dict[str, str] = {}
    for source, label in (("main_images", "agentview"), ("wrist_images", "wrist")):
        value = observation.get(source)
        if value is None:
            continue
        array = np.asarray(value, dtype=np.uint8)
        if array.ndim != 3 or array.shape[-1] != 3:
            continue
        buffer = io.BytesIO()
        iio.imwrite(buffer, array, extension=".png")
        result[label] = "data:image/png;base64," + base64.b64encode(
            buffer.getvalue()
        ).decode("ascii")
    return result


def _sanitized_critic_observations(
    *, primitives: Any, step_index: int
) -> dict[str, Any] | None:
    """Expose bounded Critic scalars while withholding coordinate motion oracles."""

    chunk_info = getattr(primitives, "_last_chunk_info", {})
    if not isinstance(chunk_info, Mapping):
        return None
    records = chunk_info.get("step_records", ())
    if not isinstance(records, (list, tuple)) or not records:
        return None
    latest = records[-1]
    if not isinstance(latest, Mapping):
        return None
    state = latest.get("state")
    if not isinstance(state, Mapping):
        return None
    fields: dict[str, Any] = {}
    for raw_key, value in sorted(state.items(), key=lambda item: str(item[0])):
        key = str(raw_key)
        if key.startswith("privileged."):
            # Distances, predicates, contact, force and joint progress are
            # allowed evidence. Absolute object/EEF coordinates are withheld.
            if _is_private_geometry_field(key):
                continue
        elif key.startswith("command.realization.") or key in {
            "command.translation.norm",
            "command.rotation.norm",
            "robot.gripper.opening",
            "robot.eef.motion_m",
            "episode.terminated",
            "episode.truncated",
        }:
            pass
        else:
            continue
        if isinstance(value, float) and not math.isfinite(value):
            continue
        if isinstance(value, (bool, int, float, str)):
            fields[key] = value
        if len(fields) >= 96:
            break
    if not fields:
        return None
    return {
        "source": "libero.critic",
        "authorization": "audited_privileged_scalar_evidence",
        "step_index": int(step_index),
        "fields": fields,
    }


class LiberoRole1RecoveryActor:
    """Make Role1 the sole caller of mutating LIBERO recovery primitives."""

    def __init__(
        self,
        *,
        adapter: Role1ModelAdapter,
        audit_root: str | Path,
        allowed_tools: Sequence[str],
        allow_privileged_evidence: bool = LIBERO_PRIVILEGED_EVIDENCE_DEFAULT,
    ) -> None:
        self.adapter = adapter
        self.audit_root = Path(audit_root)
        self.audit_root.mkdir(parents=True, exist_ok=True)
        self.allowed_tools = frozenset(str(value) for value in allowed_tools)
        self.allow_privileged_evidence = bool(allow_privileged_evidence)
        if not self.allowed_tools:
            raise ValueError("LIBERO recovery Actor requires a frozen tool allowlist")

    def decide_and_execute(
        self,
        *,
        task: str,
        step_index: int,
        observation: Mapping[str, Any],
        critic_values: Sequence[Mapping[str, Any]],
        recovery_context: Mapping[str, Any],
        primitives: Any,
    ) -> LiberoRecoveryResult:
        current_step = recovery_context.get("current_step")
        if not isinstance(current_step, Mapping):
            raise LiberoRecoveryActorError("active recovery has no current step")
        required_tool = str(current_step.get("tool", "")).strip()
        if required_tool not in self.allowed_tools:
            raise LiberoRecoveryActorError("recovery step is outside the frozen allowlist")
        parameters = current_step.get("parameters", {})
        if not isinstance(parameters, Mapping):
            raise LiberoRecoveryActorError("recovery parameters must be an object")

        images = _image_payloads(observation)
        image_references = {
            name: "sha256:" + hashlib.sha256(payload.encode()).hexdigest()
            for name, payload in images.items()
        }
        for row in critic_values:
            if _is_private_geometry_field(row.get("feature", "")) or any(
                _is_private_geometry_field(condition.get("feature", ""))
                for condition in row.get("activation_conditions", ())
                if isinstance(condition, Mapping)
            ):
                raise LiberoRecoveryActorError(
                    "Critic attempted to expose private geometry to Role1"
                )
        critic_proposals = tuple(
            CriticProposal(
                proposal_id=f"critic-{row.get('rule_id', index)}-{index}",
                reject_current_action=True,
                reason=str(row.get("proposal", "execute frozen recovery")),
                evidence=(f"critic:{row.get('rule_id', index)}",),
                details={
                    "rule_id": row.get("rule_id"),
                    "feature": row.get("feature"),
                    "observed_value": row.get("observed_value"),
                    "activation_conditions": [
                        {
                            "feature": condition.get("feature"),
                            "operator": condition.get("operator"),
                            "threshold": condition.get("threshold"),
                            "observed_value": condition.get("observed_value"),
                        }
                        for condition in row.get("activation_conditions", ())
                        if isinstance(condition, Mapping)
                    ],
                },
            )
            for index, row in enumerate(critic_values)
        )
        required_role1_tool = role1_tool_name(required_tool)
        vla_role1_tool = role1_tool_name("vla_execute")
        interrupted_proposal = ToolProposal(
            proposal_id=f"libero-vla-interrupted-{step_index:06d}",
            tool=vla_role1_tool,
            proposal={
                "status": "rejected_by_critic",
                "proposal_role": "interrupted_current_tool",
                "required_replacement_tool": required_role1_tool,
                "recovery_id": recovery_context.get("recovery_id"),
                "current_step": _sanitize_actor_visible(current_step),
            },
            evidence=("live-critic-proposal",),
        )
        recovery_proposal = ToolProposal(
            proposal_id=f"libero-frozen-recovery-{step_index:06d}",
            tool=required_role1_tool,
            proposal={
                "status": "ready_for_role1_approval",
                "proposal_role": "frozen_recovery_alternative",
                "recovery_id": recovery_context.get("recovery_id"),
                "current_step": _sanitize_actor_visible(current_step),
            },
            evidence=("active-frozen-recovery", "live-critic-proposal"),
        )
        task_state = {
            "libero_terminated": bool(primitives.env.episode_terminated),
            "episode_truncated": bool(primitives.env.episode_truncated),
            "active_recovery": _sanitize_actor_visible(recovery_context),
        }
        if self.allow_privileged_evidence:
            critic_observations = _sanitized_critic_observations(
                primitives=primitives, step_index=step_index
            )
            if critic_observations is not None:
                task_state["critic_observations"] = critic_observations
            task_state["privileged_recovery_state"] = _recovery_privileged_state(
                primitives
            )
        event = Role1Event(
            event_id=f"libero-recovery-{step_index:06d}",
            task=task,
            step_index=step_index,
            current_stage="recover",
            # The active frozen step is the executable recovery alternative;
            # the interrupted VLA is retained only as an explicitly rejected
            # proposal for auditability.
            current_tool=required_role1_tool,
            allowed_stages=("recover", "verify", "closed_loop_execution"),
            allowed_tools=tuple(
                dict.fromkeys((required_role1_tool, vla_role1_tool))
            ),
            image_references=image_references,
            task_state=task_state,
            tool_proposals=(recovery_proposal, interrupted_proposal),
            critic_proposals=critic_proposals,
        )
        effect = self.adapter.decide(event, image_payloads=images)
        if effect.termination.approved or effect.direct_action is not None:
            raise LiberoRecoveryActorError(
                "Role1 cannot terminate or bypass a frozen recovery step"
            )
        selected = effect.selected_tool or event.current_tool
        if selected != required_role1_tool:
            raise LiberoRecoveryActorError(
                "Role1 did not select the current frozen recovery tool"
            )
        selected_short = short_tool_name(selected)
        handler = getattr(primitives, selected_short, None)
        if handler is None or not callable(handler):
            raise LiberoRecoveryActorError("frozen recovery tool is not executable")
        effective_parameters = dict(parameters)
        if effect.proposal_disposition == "modify":
            modifications = dict(effect.modifications)
            replacement = modifications.get("parameters")
            if not isinstance(replacement, Mapping):
                raise LiberoRecoveryActorError(
                    "modified recovery requires modifications.parameters"
                )
            effective_parameters.update(dict(replacement))
        begin_recovery_step = getattr(primitives, "begin_recovery_step", None)
        if not callable(begin_recovery_step):
            raise LiberoRecoveryActorError(
                "LIBERO primitives cannot acknowledge the active Critic proposal"
            )
        begin_recovery_step()
        before = int(primitives.env.episode_steps)
        result = handler(**effective_parameters)
        after = int(primitives.env.episode_steps)
        if not isinstance(result, dict):
            result = {"value": result}
        public_result = _sanitize_actor_visible(result)
        executed = after - before
        no_op_verified = bool(result.get("no_op_verified"))
        environment_advanced = bool(result.get("environment_advanced", executed > 0))
        if executed < 1 and not primitives.env.episode_terminated and not (
            no_op_verified and not environment_advanced
        ):
            raise LiberoRecoveryActorError("recovery tool produced no environment action")
        audit = {
            "event_id": event.event_id,
            "decision_id": effect.decision_id,
            "selected_tool": selected_short,
            "base_parameters_sha256": canonical_sha256(dict(parameters)),
            "parameters_sha256": canonical_sha256(effective_parameters),
            "role1_modified_parameters": effect.proposal_disposition == "modify",
            "executed_horizon": executed,
            "no_op_verified": no_op_verified,
            "environment_advanced": environment_advanced,
            "result": public_result,
            "environment_write_owner": "libero_role1_recovery_actor",
        }
        atomic_write_json(
            self.audit_root / f"step-{step_index:06d}-{effect.decision_id}.json",
            audit,
            overwrite=False,
        )
        return LiberoRecoveryResult(
            decision_id=effect.decision_id,
            selected_tool=selected_short,
            executed_horizon=executed,
            result=public_result,
        )


def _recovery_privileged_state(primitives: Any) -> dict[str, Any]:
    """Expose only bounded, non-geometric semantic state to Role1."""
    reader = getattr(getattr(primitives, "env", None), "privileged_critic_state", None)
    if not callable(reader):
        return {"available": False}
    try:
        state = reader()
    except Exception as exc:  # pragma: no cover - transport-specific failure
        return {"available": False, "error": type(exc).__name__}
    if not isinstance(state, Mapping):
        return {"available": False}
    result: dict[str, Any] = {}
    for raw_key, value in state.items():
        key = str(raw_key)
        if not key.startswith(("privileged.task.", "privileged.available")):
            continue
        if _is_private_geometry_field(key):
            continue
        if isinstance(value, float) and not math.isfinite(value):
            continue
        if isinstance(value, (bool, int, float, str)):
            result[key] = value
    return result or {"available": False}


__all__ = [
    "LiberoRecoveryActorError",
    "LiberoRecoveryResult",
    "LiberoRole1RecoveryActor",
]
