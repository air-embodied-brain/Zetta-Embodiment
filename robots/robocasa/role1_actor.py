# Copyright (c) 2026 RPent Contributors
"""Single-writer Actor that applies only persisted Role1 decisions.

The GR00T client and every proposal tool remain proposal producers.
This module is the only bridge from a persisted Role1 effect to an action chunk
returned to the environment owner.  It intentionally has no environment
client, so a failed decision cannot partially mutate the simulator.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from robots.robocasa.role1_agent import (
    CriticProposal,
    Role1DecisionStore,
    Role1Event,
    Role1ModelAdapter,
    ToolProposal,
)
from robots.robocasa.tool_bindings import TaskToolBinding
from robots.robocasa.tool_runtime import InvocationPolicy
from rpent.evolution.jsonio import atomic_write_json, canonical_sha256

VLA_TOOL = "robocasa.vla.groot"


class Role1ActorError(RuntimeError):
    """No audited action can be produced without violating the contract."""


class ProposalToolRuntime(Protocol):
    """Minimal proposal-only runtime interface consumed by the Actor."""

    def invoke(
        self,
        name: str,
        payload: Mapping[str, Any],
        *,
        policy: InvocationPolicy | None = None,
    ) -> dict[str, Any]: ...


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    if isinstance(value, list):
        return [_thaw(item) for item in value]
    return value


def _blind(value: Any) -> Any:
    """Remove scheduler identity before constructing a fail-closed event."""

    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            tokens = {
                token
                for token in str(key).lower().replace("-", "_").split("_")
                if token
            }
            if tokens.intersection({"seed", "rng", "schedule"}):
                continue
            result[str(key)] = _blind(item)
        return result
    if isinstance(value, (list, tuple)):
        return [_blind(item) for item in value]
    return value


def _image_references(observation: Mapping[str, Any]) -> dict[str, str]:
    images = observation.get("images", {})
    if not isinstance(images, Mapping):
        return {}
    return {
        str(name): "sha256:" + hashlib.sha256(str(value).encode()).hexdigest()
        for name, value in images.items()
    }


def _proposal_summary(result: Mapping[str, Any]) -> dict[str, Any]:
    """Bound model context while the full result remains in the audit file."""

    omitted: dict[str, Any] = {}
    summary: dict[str, Any] = {}
    for key, value in result.items():
        if key in {"points", "mask", "masks", "depth", "images", "observation"}:
            length = len(value) if isinstance(value, Sequence) else None
            omitted[str(key)] = {
                "length": length,
                "sha256": canonical_sha256(value),
            }
        else:
            summary[str(key)] = value
    if omitted:
        summary["omitted_artifacts"] = omitted
    return summary


def _critic_proposals(
    values: Sequence[Mapping[str, Any]],
) -> tuple[CriticProposal, ...]:
    proposals: list[CriticProposal] = []
    for index, value in enumerate(values):
        rule_id = str(value.get("rule_id", f"critic-{index}"))
        reason = str(
            value.get("proposal", value.get("reason", "reject current action"))
        )
        evidence = value.get("evidence", ())
        if isinstance(evidence, Mapping):
            evidence = [f"critic:{rule_id}:{canonical_sha256(evidence)[:16]}"]
        if not isinstance(evidence, Sequence) or isinstance(evidence, (str, bytes)):
            evidence = [f"critic:{rule_id}"]
        evidence_values = tuple(str(item) for item in evidence if str(item).strip())
        proposals.append(
            CriticProposal(
                proposal_id=f"critic-{rule_id}-{index}",
                reject_current_action=True,
                reason=reason,
                evidence=evidence_values or (f"critic:{rule_id}",),
                details={
                    "rule_id": rule_id,
                    "feature": value.get("feature"),
                    "observed": value.get("observed"),
                },
            )
        )
    return tuple(proposals)


def _authorized_recovery_tools(
    recovery_suggestions: Sequence[Mapping[str, Any]],
) -> frozenset[str]:
    """Extract only tools explicitly frozen into triggered recovery rules."""

    tools: set[str] = set()
    for recovery in recovery_suggestions:
        steps = recovery.get("steps", ())
        if not isinstance(steps, Sequence) or isinstance(steps, (str, bytes)):
            continue
        for step in steps:
            if isinstance(step, Mapping) and str(step.get("tool", "")).strip():
                tools.add(str(step["tool"]))
    return frozenset(tools)


@dataclass(frozen=True, slots=True)
class Role1ActorResult:
    actions: tuple[Any, ...]
    terminate: bool
    termination_reason: str
    selected_tool: str | None
    decision_ids: tuple[str, ...]


class Role1EpisodeActor:
    """Resolve one action boundary through proposal-only tools and Role1."""

    def __init__(
        self,
        *,
        adapter: Role1ModelAdapter,
        decision_store: Role1DecisionStore,
        tool_runtime: ProposalToolRuntime,
        binding: TaskToolBinding,
        audit_root: str | Path,
        allow_privileged: bool = True,
        maximum_decisions_without_action: int = 4,
    ) -> None:
        if maximum_decisions_without_action < 1:
            raise ValueError("maximum_decisions_without_action must be positive")
        if adapter.store is not decision_store:
            raise ValueError("adapter and Actor must share one decision store")
        self.adapter = adapter
        self.decision_store = decision_store
        self.tool_runtime = tool_runtime
        self.binding = binding
        self.audit_root = Path(audit_root)
        self.allow_privileged = allow_privileged
        self.maximum_decisions_without_action = maximum_decisions_without_action

    def _persist_tool_result(
        self,
        *,
        step_index: int,
        event_index: int,
        tool: str,
        result: Mapping[str, Any],
    ) -> Path:
        path = (
            self.audit_root
            / "tool_proposals"
            / f"step-{step_index:06d}-event-{event_index:02d}.json"
        )
        atomic_write_json(
            path,
            {
                "tool": tool,
                "proposal_only": True,
                "environment_write": False,
                "result": dict(result),
            },
            overwrite=False,
        )
        return path

    def decide_action(
        self,
        *,
        task: str,
        step_index: int,
        observation_response: Mapping[str, Any],
        vla_actions: Sequence[Any],
        vla_metadata: Mapping[str, Any],
        critic_values: Sequence[Mapping[str, Any]] = (),
        recovery_suggestions: Sequence[Mapping[str, Any]] = (),
        active_recovery: Mapping[str, Any] | None = None,
        current_stage: str = "closed_loop_execution",
    ) -> Role1ActorResult:
        if task != self.binding.task:
            raise Role1ActorError("Actor task does not match its frozen tool binding")
        container = observation_response.get("observation", observation_response)
        if not isinstance(container, Mapping):
            raise Role1ActorError("observation must be an object")
        actions = tuple(_thaw(item) for item in vla_actions)
        if not actions:
            raise Role1ActorError("VLA proposal contains no actions")
        safe_meta = {
            key: _blind(value)
            for key, value in vla_metadata.items()
            if "seed" not in str(key).lower() and "rng" not in str(key).lower()
        }
        current_tool = self.binding.vla_tool_name
        proposal_result: dict[str, Any] = {
            "actions": list(actions),
            "horizon": len(actions),
            **safe_meta,
        }
        proposed_actions: tuple[Any, ...] | None = actions
        critics = _critic_proposals(critic_values)
        recovery_tools = _authorized_recovery_tools(recovery_suggestions)
        required_recovery_tool: str | None = None
        frozen_recovery_parameters: dict[str, Any] = {}
        if active_recovery is not None:
            current_step = active_recovery.get("current_step")
            if not isinstance(current_step, Mapping):
                raise Role1ActorError("active recovery has no current step")
            required_recovery_tool = str(current_step.get("tool", "")).strip()
            if not required_recovery_tool or required_recovery_tool not in recovery_tools:
                raise Role1ActorError("active recovery step is not frozen and authorized")
            parameters = current_step.get("parameters", {})
            if not isinstance(parameters, Mapping):
                raise Role1ActorError("active recovery parameters must be an object")
            frozen_recovery_parameters = dict(_thaw(parameters))
        candidate_only = frozenset(self.binding.candidate_tool_names)
        allowed_tools = tuple(
            tool
            for tool in self.binding.tool_names
            if tool not in candidate_only or tool in recovery_tools
        )
        if required_recovery_tool is not None:
            # Role1Event requires the currently proposed VLA tool to remain in
            # its allowlist.  Keep it visible as rejected context, then enforce
            # below that the only executable choice is the frozen recovery
            # step.  This preserves an honest proposal record without letting
            # VLA bypass recovery.
            allowed_tools = tuple(
                dict.fromkeys((current_tool, required_recovery_tool))
            )
        history: list[Mapping[str, Any]] = []
        decision_ids: list[str] = []

        for event_index in range(self.maximum_decisions_without_action):
            proposal_id = f"tool-{step_index:06d}-{event_index:02d}"
            artifact = self._persist_tool_result(
                step_index=step_index,
                event_index=event_index,
                tool=current_tool,
                result=proposal_result,
            )
            proposal = ToolProposal(
                proposal_id=proposal_id,
                tool=current_tool,
                proposal=_proposal_summary(proposal_result),
                evidence=(f"artifact:{artifact.name}",),
            )
            task_state = _blind(container.get("state", {}))
            if recovery_suggestions:
                task_state = {
                    **task_state,
                    "candidate_recovery_rules": _blind(recovery_suggestions),
                    "active_recovery": _blind(active_recovery),
                }
            event = Role1Event(
                event_id=f"step-{step_index:06d}-event-{event_index:02d}",
                task=task,
                step_index=step_index,
                current_stage=current_stage,
                current_tool=current_tool,
                allowed_stages=(
                    "observe",
                    "approach",
                    "engage",
                    "recover",
                    "verify",
                    "closed_loop_execution",
                ),
                allowed_tools=allowed_tools,
                image_references=_image_references(container),
                task_state=task_state,
                tool_proposals=(proposal,),
                critic_proposals=critics,
                history=tuple(history),
            )
            image_payloads = container.get("images", {})
            if not isinstance(image_payloads, Mapping):
                raise Role1ActorError("observation images must be an object")
            effect = self.adapter.decide(event, image_payloads=image_payloads)
            persisted = self.decision_store.load(effect.decision_id)
            decision = persisted.decision
            decision_ids.append(effect.decision_id)
            history.append(
                {
                    "decision_id": effect.decision_id,
                    "action_kind": effect.action_kind,
                    "selected_tool": effect.selected_tool,
                    "persisted_digest": effect.persisted_digest,
                }
            )
            if effect.termination.approved:
                if required_recovery_tool is not None:
                    raise Role1ActorError(
                        "Role1 cannot terminate before executing the current "
                        "frozen recovery step"
                    )
                return Role1ActorResult(
                    actions=(),
                    terminate=True,
                    termination_reason=effect.termination.reason,
                    selected_tool=None,
                    decision_ids=tuple(decision_ids),
                )
            if effect.direct_action is not None:
                if required_recovery_tool is not None:
                    raise Role1ActorError(
                        "Role1 direct_action cannot bypass the current frozen "
                        "recovery step"
                    )
                return Role1ActorResult(
                    actions=(_thaw(effect.direct_action),),
                    terminate=False,
                    termination_reason="",
                    selected_tool=None,
                    decision_ids=tuple(decision_ids),
                )

            selected_tool = effect.selected_tool or current_tool
            if selected_tool not in allowed_tools:
                raise Role1ActorError(
                    "Role1 selected a candidate-only tool without a triggered, "
                    "frozen recovery-rule authorization"
                )
            if (
                required_recovery_tool is not None
                and selected_tool != required_recovery_tool
            ):
                raise Role1ActorError(
                    "Role1 must execute the current frozen recovery step before "
                    "returning control to VLA"
                )
            if (
                selected_tool == current_tool
                and decision.proposal_disposition == "accept"
                and proposed_actions
            ):
                return Role1ActorResult(
                    actions=proposed_actions,
                    terminate=False,
                    termination_reason="",
                    selected_tool=selected_tool,
                    decision_ids=tuple(decision_ids),
                )
            if selected_tool in {VLA_TOOL, self.binding.vla_tool_name}:
                if decision.proposal_disposition != "accept":
                    raise Role1ActorError(
                        "Role1 rejected or modified the VLA proposal without "
                        "materializing a replacement action or tool proposal"
                    )
                return Role1ActorResult(
                    actions=actions,
                    terminate=False,
                    termination_reason="",
                    selected_tool=selected_tool,
                    decision_ids=tuple(decision_ids),
                )

            modifications = _thaw(decision.modifications)
            parameters = modifications.get("parameters", {})
            if not isinstance(parameters, Mapping):
                raise Role1ActorError("modifications.parameters must be an object")
            payload = {**dict(parameters), **frozen_recovery_parameters}
            payload.setdefault("observation", _thaw(container))
            payload.setdefault("pending_vla_actions", list(actions))
            payload.setdefault("task", task)
            payload.setdefault("step_index", step_index)
            result = self.tool_runtime.invoke(
                selected_tool,
                payload,
                policy=InvocationPolicy(
                    allow=frozenset(allowed_tools),
                    allow_privileged=self.allow_privileged,
                ),
            )
            current_tool = selected_tool
            proposal_result = dict(result)
            proposed = result.get("action")
            proposed_many = result.get("actions")
            proposed_actions = (
                tuple(proposed_many)
                if isinstance(proposed_many, list) and proposed_many
                else (proposed,)
                if isinstance(proposed, Mapping)
                else None
            )
            critics = ()

        raise Role1ActorError(
            "Role1 exceeded the bounded proposal loop without an audited action"
        )


__all__ = [
    "Role1ActorError",
    "Role1ActorResult",
    "Role1EpisodeActor",
    "VLA_TOOL",
]
