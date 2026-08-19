# Copyright (c) 2026 Zetta Contributors
"""Event-driven Role1 high-level authority for RoboCasa.

This is a clean-room behavioral contract.  Tools may propose actions and a
critic may reject the current action, but neither may select the next tool,
recovery, stage, or termination.  A validated Role1 decision remains inert
until its immutable audit artifact has been published successfully.

The module intentionally has no simulator client and exposes no environment
mutation method.  The episode Actor consumes an activated :class:`Role1Effect`
and remains responsible for applying the selected command.
"""

from __future__ import annotations

import base64
import copy
import hashlib
import io
import json
import math
import re
import time
import traceback
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal

from robots.robocasa.action_contract import serializable_action
from robots.robocasa.tool_catalog import (
    DEFAULT_ROBOCASA_TOOL_CATALOG,
    ToolCatalog,
)
from zetta.evolution.jsonio import atomic_write_json, canonical_json_bytes, read_json
from zetta.planner.base import build_planner
from zetta.tools.toolkit import Toolkit

ProposalDisposition = Literal["accept", "reject", "modify"]
ActionKind = Literal[
    "continue",
    "switch",
    "recover",
    "restage",
    "regenerate",
    "replace",
    "terminate",
]

PROPOSAL_DISPOSITIONS = frozenset({"accept", "reject", "modify"})
ACTION_KINDS = frozenset(
    {"continue", "switch", "recover", "restage", "regenerate", "replace", "terminate"}
)
_ACTION_ALIASES = {
    "switch_tool": "switch",
    "regenerate_grasp": "regenerate",
    "execute_action": "replace",
    "terminate_episode": "terminate",
}
_DIRECT_ACTION_KEYS = frozenset(
    {
        "end_effector_position",
        "end_effector_rotation",
        "gripper_close",
        "base_motion",
        "control_mode",
    }
)
_PREFIXED_DIRECT_ACTION_KEYS = frozenset(f"action.{key}" for key in _DIRECT_ACTION_KEYS)
_FORBIDDEN_KEY_TOKENS = frozenset(
    {
        "seed",
        "seeds",
        "rng",
        "schedule",
        "future_schedule",
        "policy_rng",
        "environment_seed",
    }
)
_FORBIDDEN_TEXT = re.compile(
    r"(?:\b(?:environment[_ -]?seed|policy[_ -]?rng|future[_ -]?schedule)\b|"
    r"\bseed\s*[:=#_-]?\s*\d+\b)",
    re.IGNORECASE,
)
_TOOL_FORBIDDEN_CONTROL_KEYS = frozenset(
    {
        "execute",
        "executed",
        "apply",
        "applied",
        "environment_write",
        "termination",
        "termination_required",
        "selected_stage",
        "selected_tool",
    }
)
_CRITIC_FORBIDDEN_KEYS = _TOOL_FORBIDDEN_CONTROL_KEYS | frozenset(
    {"action", "direct_action", "replacement_action", "recovery"}
)
_READ_ONLY_CAPABILITY_PREFIXES = (
    "perception.",
    "geometry.",
    "verification.",
)


class Role1ContractError(ValueError):
    """A Role1 input or decision violates the authority contract."""


class Role1PersistenceError(RuntimeError):
    """An immutable decision artifact is absent, changed, or conflicting."""


class DecisionNotPersistedError(Role1PersistenceError):
    """A caller attempted to activate an inert, not-yet-persisted decision."""


class Role1ModelError(RuntimeError):
    """The Role1 model invocation failed before an effect could be activated."""


def _plain_json(value: Any) -> Any:
    """Deep-copy JSON data while rejecting lossy or non-finite values."""

    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise Role1ContractError("all object keys must be strings")
            result[key] = _plain_json(item)
        return result
    if isinstance(value, (list, tuple)):
        return [_plain_json(item) for item in value]
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise Role1ContractError("numbers must be finite")
        return value
    raise Role1ContractError(f"value is not JSON serializable: {type(value).__name__}")


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _freeze(item) for key, item in value.items()}
        )
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return copy.deepcopy(value)


def _tokens(key: str) -> set[str]:
    lowered = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", key).lower()
    return {part for part in re.split(r"[^a-z0-9]+", lowered) if part}


def _assert_seed_blind(value: Any, *, path: str = "input") -> None:
    """Reject explicit experiment identity and future-schedule leakage."""

    if isinstance(value, Mapping):
        for key, item in value.items():
            lowered = str(key).lower()
            tokens = _tokens(str(key))
            if lowered in _FORBIDDEN_KEY_TOKENS or tokens.intersection({"seed", "rng"}):
                raise Role1ContractError(
                    f"seed or RNG metadata is forbidden at {path}.{key}"
                )
            if "future" in tokens and "schedule" in tokens:
                raise Role1ContractError(
                    f"future schedule is forbidden at {path}.{key}"
                )
            _assert_seed_blind(item, path=f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _assert_seed_blind(item, path=f"{path}[{index}]")
    elif isinstance(value, str) and _FORBIDDEN_TEXT.search(value):
        raise Role1ContractError(
            f"seed, RNG, or future schedule text is forbidden at {path}"
        )


def _nonempty(value: Any, name: str) -> str:
    result = str(value).strip()
    if not result:
        raise Role1ContractError(f"{name} must not be empty")
    return result


def _evidence(value: Any, name: str = "evidence") -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise Role1ContractError(f"{name} must be an array of strings")
    result = tuple(_nonempty(item, name) for item in value)
    if not result or len(result) > 32:
        raise Role1ContractError(f"{name} must contain between 1 and 32 entries")
    return result


def _reject_control_claims(value: Any, *, critic: bool, path: str) -> None:
    forbidden = _CRITIC_FORBIDDEN_KEYS if critic else _TOOL_FORBIDDEN_CONTROL_KEYS
    if isinstance(value, Mapping):
        for key, item in value.items():
            lowered = str(key).lower()
            if lowered in forbidden:
                if lowered == "environment_write" and item is False:
                    continue
                raise Role1ContractError(f"proposal cannot claim {lowered!r} at {path}")
            if lowered in {"authority", "action_authority"}:
                required = "reject_current_action_only" if critic else "proposal_only"
                if item != required:
                    raise Role1ContractError(f"proposal authority must be {required!r}")
            _reject_control_claims(item, critic=critic, path=f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _reject_control_claims(item, critic=critic, path=f"{path}[{index}]")


@dataclass(frozen=True, slots=True)
class ToolProposal:
    """One inert tool result for Role1 review."""

    proposal_id: str
    tool: str
    proposal: Mapping[str, Any]
    evidence: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "proposal_id", _nonempty(self.proposal_id, "proposal_id")
        )
        object.__setattr__(self, "tool", _nonempty(self.tool, "tool"))
        plain = _plain_json(self.proposal)
        _assert_seed_blind(plain, path="tool_proposal")
        _reject_control_claims(plain, critic=False, path="tool_proposal")
        object.__setattr__(self, "proposal", _freeze(plain))
        object.__setattr__(self, "evidence", _evidence(self.evidence))

    @classmethod
    def from_payload(cls, value: Mapping[str, Any]) -> "ToolProposal":
        payload = _plain_json(value)
        proposal_id = payload.pop("proposal_id", payload.pop("id", ""))
        source = payload.pop("source", "tool")
        if source != "tool":
            raise Role1ContractError("tool proposal source must be 'tool'")
        tool = payload.pop("tool", "")
        evidence = payload.pop("evidence", [f"tool:{tool}"])
        if payload.pop("proposal_only", True) is not True:
            raise Role1ContractError("tool proposal must be proposal_only")
        if payload.pop("environment_write", False) is not False:
            raise Role1ContractError("tool proposal cannot write the environment")
        proposal = payload.pop("proposal", None)
        if proposal is not None and payload:
            raise Role1ContractError(
                "tool proposal cannot mix nested and top-level payloads"
            )
        return cls(
            proposal_id=str(proposal_id),
            tool=str(tool),
            proposal=payload if proposal is None else proposal,
            evidence=tuple(evidence),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "proposal_id": self.proposal_id,
            "source": "tool",
            "tool": self.tool,
            "proposal": _thaw(self.proposal),
            "evidence": list(self.evidence),
            "proposal_only": True,
            "environment_write": False,
        }


@dataclass(frozen=True, slots=True)
class CriticProposal:
    """A critic may only advise whether the current action should be rejected."""

    proposal_id: str
    reject_current_action: bool
    reason: str
    evidence: tuple[str, ...]
    details: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "proposal_id", _nonempty(self.proposal_id, "proposal_id")
        )
        if not isinstance(self.reject_current_action, bool):
            raise Role1ContractError("reject_current_action must be boolean")
        object.__setattr__(self, "reason", _nonempty(self.reason, "critic reason"))
        object.__setattr__(self, "evidence", _evidence(self.evidence))
        plain = _plain_json(self.details)
        _assert_seed_blind(plain, path="critic_proposal")
        _reject_control_claims(plain, critic=True, path="critic_proposal")
        object.__setattr__(self, "details", _freeze(plain))

    @classmethod
    def from_payload(cls, value: Mapping[str, Any]) -> "CriticProposal":
        payload = _plain_json(value)
        proposal_id = payload.pop("proposal_id", payload.pop("id", ""))
        source = payload.pop("source", "critic")
        if source != "critic":
            raise Role1ContractError("critic proposal source must be 'critic'")
        reject = payload.pop("reject_current_action", None)
        reason = payload.pop("reason", "")
        evidence = payload.pop("evidence", [str(reason)])
        if payload.pop("proposal_only", True) is not True:
            raise Role1ContractError("critic output must be proposal_only")
        if payload.pop("environment_write", False) is not False:
            raise Role1ContractError("critic output cannot write the environment")
        authority = payload.pop("action_authority", "reject_current_action_only")
        if authority != "reject_current_action_only":
            raise Role1ContractError(
                "critic authority must be reject_current_action_only"
            )
        return cls(
            proposal_id=str(proposal_id),
            reject_current_action=reject,
            reason=str(reason),
            evidence=tuple(evidence),
            details=payload.pop("details", payload),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "proposal_id": self.proposal_id,
            "source": "critic",
            "reject_current_action": self.reject_current_action,
            "reason": self.reason,
            "evidence": list(self.evidence),
            "details": _thaw(self.details),
            "proposal_only": True,
            "action_authority": "reject_current_action_only",
            "environment_write": False,
        }


@dataclass(frozen=True, slots=True)
class Role1Event:
    """Seed-blind multimodal context at one high-level decision boundary."""

    event_id: str
    task: str
    step_index: int
    current_stage: str | None
    current_tool: str | None
    allowed_stages: tuple[str, ...]
    allowed_tools: tuple[str, ...]
    image_references: Mapping[str, str] = field(default_factory=dict)
    task_state: Mapping[str, Any] = field(default_factory=dict)
    tool_proposals: tuple[ToolProposal, ...] = ()
    critic_proposals: tuple[CriticProposal, ...] = ()
    history: tuple[Mapping[str, Any], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "event_id", _nonempty(self.event_id, "event_id"))
        object.__setattr__(self, "task", _nonempty(self.task, "task"))
        if (
            not isinstance(self.step_index, int)
            or isinstance(self.step_index, bool)
            or self.step_index < 0
        ):
            raise Role1ContractError("step_index must be a non-negative integer")
        stages = tuple(_nonempty(item, "allowed_stage") for item in self.allowed_stages)
        tools = tuple(_nonempty(item, "allowed_tool") for item in self.allowed_tools)
        if not stages or len(stages) != len(set(stages)):
            raise Role1ContractError("allowed_stages must be non-empty and unique")
        if len(tools) != len(set(tools)):
            raise Role1ContractError("allowed_tools must be unique")
        if self.current_stage is not None and self.current_stage not in stages:
            raise Role1ContractError("current_stage is not allowed")
        if self.current_tool is not None and self.current_tool not in tools:
            raise Role1ContractError("current_tool is not allowed")
        images = _plain_json(self.image_references)
        if any(
            not str(key).strip() or not isinstance(value, str) or not value.strip()
            for key, value in images.items()
        ):
            raise Role1ContractError(
                "image_references must map names to non-empty references"
            )
        state = _plain_json(self.task_state)
        history = _plain_json(list(self.history))
        _assert_seed_blind(
            {
                "event_id": self.event_id,
                "task": self.task,
                "images": images,
                "task_state": state,
                "history": history,
            }
        )
        proposal_ids = [
            item.proposal_id for item in (*self.tool_proposals, *self.critic_proposals)
        ]
        if len(proposal_ids) != len(set(proposal_ids)):
            raise Role1ContractError(
                "proposal_id values must be unique within an event"
            )
        if any(item.tool not in tools for item in self.tool_proposals):
            raise Role1ContractError(
                "tool proposal uses a tool outside the event allowlist"
            )
        object.__setattr__(self, "allowed_stages", stages)
        object.__setattr__(self, "allowed_tools", tools)
        object.__setattr__(self, "image_references", _freeze(images))
        object.__setattr__(self, "task_state", _freeze(state))
        object.__setattr__(self, "history", tuple(_freeze(item) for item in history))

    @property
    def proposal_ids(self) -> tuple[str, ...]:
        return tuple(
            item.proposal_id for item in (*self.tool_proposals, *self.critic_proposals)
        )

    @classmethod
    def from_payload(cls, value: Mapping[str, Any]) -> "Role1Event":
        payload = _plain_json(value)
        allowed = {
            "event_id",
            "task",
            "step_index",
            "current_stage",
            "current_tool",
            "allowed_stages",
            "allowed_tools",
            "image_references",
            "task_state",
            "tool_proposals",
            "critic_proposals",
            "history",
        }
        unknown = set(payload) - allowed
        if unknown:
            raise Role1ContractError(f"unknown Role1 event fields: {sorted(unknown)}")
        return cls(
            event_id=payload.get("event_id", ""),
            task=payload.get("task", ""),
            step_index=payload.get("step_index", -1),
            current_stage=payload.get("current_stage"),
            current_tool=payload.get("current_tool"),
            allowed_stages=tuple(payload.get("allowed_stages", ())),
            allowed_tools=tuple(payload.get("allowed_tools", ())),
            image_references=payload.get("image_references", {}),
            task_state=payload.get("task_state", {}),
            tool_proposals=tuple(
                ToolProposal.from_payload(item)
                for item in payload.get("tool_proposals", ())
            ),
            critic_proposals=tuple(
                CriticProposal.from_payload(item)
                for item in payload.get("critic_proposals", ())
            ),
            history=tuple(payload.get("history", ())),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "task": self.task,
            "step_index": self.step_index,
            "current_stage": self.current_stage,
            "current_tool": self.current_tool,
            "allowed_stages": list(self.allowed_stages),
            "allowed_tools": list(self.allowed_tools),
            "image_references": _thaw(self.image_references),
            "task_state": _thaw(self.task_state),
            "tool_proposals": [item.to_dict() for item in self.tool_proposals],
            "critic_proposals": [item.to_dict() for item in self.critic_proposals],
            "history": _thaw(self.history),
        }


@dataclass(frozen=True, slots=True)
class TerminationDecision:
    approved: bool = False
    reason: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.approved, bool):
            raise Role1ContractError("termination.approved must be boolean")
        if self.approved and not self.reason.strip():
            raise Role1ContractError("approved termination requires a reason")
        if not self.approved and self.reason:
            raise Role1ContractError("unapproved termination must have an empty reason")

    def to_dict(self) -> dict[str, Any]:
        return {"approved": self.approved, "reason": self.reason}


@dataclass(frozen=True, slots=True)
class Role1Decision:
    """A validated but not necessarily persisted high-level decision."""

    decision_id: str
    event_id: str
    proposal_disposition: ProposalDisposition
    action_kind: ActionKind
    selected_stage: str | None
    selected_tool: str | None
    direct_action: Mapping[str, Any] | None
    termination: TerminationDecision
    evidence: tuple[str, ...]
    confidence: float
    rationale: str
    proposal_ids: tuple[str, ...]
    modifications: Mapping[str, Any] = field(default_factory=dict)

    @property
    def intent(self) -> str:
        return self.action_kind

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision_id": self.decision_id,
            "event_id": self.event_id,
            "authority": "role1_high_level_agent",
            "proposal_disposition": self.proposal_disposition,
            "action_kind": self.action_kind,
            "selected_stage": self.selected_stage,
            "selected_tool": self.selected_tool,
            "direct_action": None
            if self.direct_action is None
            else _thaw(self.direct_action),
            "termination": self.termination.to_dict(),
            "evidence": list(self.evidence),
            "confidence": self.confidence,
            "rationale": self.rationale,
            "proposal_ids": list(self.proposal_ids),
            "modifications": _thaw(self.modifications),
            "tool_authority": "proposal_only",
            "critic_action_authority": "reject_current_action_only",
            "environment_write": False,
        }


def _canonical_direct_action(value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise Role1ContractError("direct_action must be an object")
    keys = frozenset(str(key) for key in value)
    if keys not in {_DIRECT_ACTION_KEYS, _PREFIXED_DIRECT_ACTION_KEYS}:
        raise Role1ContractError(
            "direct_action must contain exactly the five canonical components"
        )
    try:
        return _freeze(serializable_action(value))
    except (TypeError, ValueError) as exc:
        raise Role1ContractError(f"invalid direct_action: {exc}") from exc


def _decision_identity_payload(value: Mapping[str, Any]) -> dict[str, Any]:
    return {key: item for key, item in value.items() if key != "decision_id"}


def validate_role1_decision(
    raw: Mapping[str, Any],
    *,
    event: Role1Event,
    catalog: ToolCatalog = DEFAULT_ROBOCASA_TOOL_CATALOG,
) -> Role1Decision:
    """Validate a model decision without selecting or coercing a policy."""

    payload = _plain_json(raw)
    _assert_seed_blind(payload, path="decision")
    allowed = {
        "decision_id",
        "event_id",
        "proposal_disposition",
        "action_kind",
        "intent",
        "selected_stage",
        "selected_tool",
        "direct_action",
        "termination",
        "evidence",
        "evidence_used",
        "confidence",
        "rationale",
        "proposal_ids",
        "modifications",
    }
    unknown = set(payload) - allowed
    if unknown:
        raise Role1ContractError(f"unknown Role1 decision fields: {sorted(unknown)}")
    event_id = str(payload.get("event_id", event.event_id))
    if event_id != event.event_id:
        raise Role1ContractError("decision event_id does not match its event")
    disposition = str(payload.get("proposal_disposition", ""))
    if disposition not in PROPOSAL_DISPOSITIONS:
        raise Role1ContractError(
            "proposal_disposition must be accept, reject, or modify"
        )
    raw_kind = payload.get("action_kind", payload.get("intent", ""))
    if (
        "action_kind" in payload
        and "intent" in payload
        and payload["action_kind"] != payload["intent"]
    ):
        raise Role1ContractError("action_kind and intent disagree")
    action_kind = _ACTION_ALIASES.get(str(raw_kind), str(raw_kind))
    if action_kind not in ACTION_KINDS:
        raise Role1ContractError(f"unknown Role1 action_kind: {action_kind!r}")
    selected_stage = payload.get("selected_stage")
    selected_stage = None if selected_stage in {None, ""} else str(selected_stage)
    selected_tool = payload.get("selected_tool")
    selected_tool = None if selected_tool in {None, ""} else str(selected_tool)
    if selected_stage is not None and selected_stage not in event.allowed_stages:
        raise Role1ContractError("selected_stage is outside the event allowlist")
    if selected_tool is not None:
        if selected_tool not in event.allowed_tools:
            raise Role1ContractError("selected_tool is outside the event allowlist")
        try:
            spec = catalog.get(selected_tool)
        except KeyError as exc:
            raise Role1ContractError(
                "selected_tool is absent from the frozen tool catalog"
            ) from exc
        read_only = all(
            capability.startswith(_READ_ONLY_CAPABILITY_PREFIXES)
            for capability in spec.capabilities
        )
        if not spec.proposal_only and not read_only:
            raise Role1ContractError(
                "selected control tool must have a proposal-only contract"
            )
    proposal_ids_raw = payload.get("proposal_ids", event.proposal_ids)
    if not isinstance(proposal_ids_raw, Sequence) or isinstance(
        proposal_ids_raw, (str, bytes)
    ):
        raise Role1ContractError("proposal_ids must be an array")
    proposal_ids = tuple(_nonempty(item, "proposal_id") for item in proposal_ids_raw)
    if len(proposal_ids) != len(set(proposal_ids)):
        raise Role1ContractError("proposal_ids must be unique")
    unknown_proposals = set(proposal_ids) - set(event.proposal_ids)
    if unknown_proposals:
        raise Role1ContractError(
            f"decision cites unknown proposals: {sorted(unknown_proposals)}"
        )
    if event.proposal_ids and not proposal_ids:
        raise Role1ContractError("a proposal event requires explicit proposal review")
    direct_action = payload.get("direct_action")
    if direct_action is not None:
        direct_action = _canonical_direct_action(direct_action)
    termination_raw = payload.get("termination", {"approved": False, "reason": ""})
    if not isinstance(termination_raw, Mapping) or set(termination_raw) != {
        "approved",
        "reason",
    }:
        raise Role1ContractError("termination must contain exactly approved and reason")
    termination = TerminationDecision(
        approved=termination_raw["approved"],
        reason=str(termination_raw["reason"]),
    )
    modifications = _plain_json(payload.get("modifications", {}))
    _reject_control_claims(modifications, critic=False, path="modifications")
    if disposition == "accept" and modifications:
        raise Role1ContractError("accept cannot include modifications")
    if disposition == "modify" and not modifications and action_kind == "continue":
        raise Role1ContractError("modify must describe or select a concrete change")
    if disposition == "reject" and action_kind == "continue":
        raise Role1ContractError(
            "reject cannot silently continue the rejected proposal"
        )

    if action_kind == "continue":
        if selected_stage not in {None, event.current_stage}:
            raise Role1ContractError("continue cannot change stage")
        if selected_tool not in {None, event.current_tool}:
            raise Role1ContractError("continue cannot change tool")
        if direct_action is not None:
            raise Role1ContractError("continue cannot contain a direct action")
    elif action_kind == "switch":
        if selected_tool is None or selected_tool == event.current_tool:
            raise Role1ContractError("switch requires a different selected_tool")
        if selected_stage not in {None, event.current_stage}:
            raise Role1ContractError("switch cannot change stage")
        if direct_action is not None:
            raise Role1ContractError("switch cannot contain a direct action")
    elif action_kind == "restage":
        if selected_stage is None or selected_stage == event.current_stage:
            raise Role1ContractError("restage requires a different selected_stage")
        if selected_tool is None or selected_tool == event.current_tool:
            raise Role1ContractError(
                "restage requires a different executable selected_tool"
            )
        if direct_action is not None:
            raise Role1ContractError("restage cannot contain a direct action")
    elif action_kind == "regenerate":
        if selected_tool is None:
            raise Role1ContractError("regenerate requires a selected proposal tool")
        if selected_stage not in {None, event.current_stage}:
            raise Role1ContractError("regenerate cannot implicitly change stage")
        if direct_action is not None:
            raise Role1ContractError("regenerate cannot contain a direct action")
    elif action_kind in {"recover", "replace"}:
        if (selected_tool is None) == (direct_action is None):
            raise Role1ContractError(
                f"{action_kind} requires exactly one tool or direct_action"
            )
    elif action_kind == "terminate":
        if not termination.approved:
            raise Role1ContractError("terminate requires explicit approval")
        if selected_tool is not None or direct_action is not None:
            raise Role1ContractError("terminate cannot select a tool or action")

    if action_kind != "terminate" and termination.approved:
        raise Role1ContractError("only terminate may approve termination")
    cited_critic_rejections = {
        item.proposal_id
        for item in event.critic_proposals
        if item.reject_current_action and item.proposal_id in proposal_ids
    }
    cited_frozen_recovery_alternatives = {
        item.proposal_id
        for item in event.tool_proposals
        if item.proposal_id in proposal_ids
        and item.proposal.get("proposal_role") == "frozen_recovery_alternative"
    }
    if (
        cited_critic_rejections
        and disposition == "accept"
        and action_kind == "continue"
        and not cited_frozen_recovery_alternatives
    ):
        raise Role1ContractError(
            "accepting a critic rejection requires an explicit alternative"
        )
    evidence = _evidence(payload.get("evidence", payload.get("evidence_used", ())))
    confidence = payload.get("confidence")
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        raise Role1ContractError("confidence must be numeric")
    confidence = float(confidence)
    if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
        raise Role1ContractError("confidence must be within [0, 1]")
    rationale = _nonempty(payload.get("rationale", ""), "rationale")

    normalized = {
        "event_id": event_id,
        "proposal_disposition": disposition,
        "action_kind": action_kind,
        "selected_stage": selected_stage,
        "selected_tool": selected_tool,
        "direct_action": None if direct_action is None else _thaw(direct_action),
        "termination": termination.to_dict(),
        "evidence": list(evidence),
        "confidence": confidence,
        "rationale": rationale,
        "proposal_ids": list(proposal_ids),
        "modifications": modifications,
    }
    decision_id = str(payload.get("decision_id", "")).strip()
    if not decision_id:
        digest = hashlib.sha256(canonical_json_bytes(normalized)).hexdigest()[:24]
        decision_id = f"role1-{digest}"
    elif re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", decision_id) is None:
        raise Role1ContractError("decision_id must be a safe immutable artifact name")
    return Role1Decision(
        decision_id=decision_id,
        event_id=event_id,
        proposal_disposition=disposition,  # type: ignore[arg-type]
        action_kind=action_kind,  # type: ignore[arg-type]
        selected_stage=selected_stage,
        selected_tool=selected_tool,
        direct_action=None if direct_action is None else _freeze(_thaw(direct_action)),
        termination=termination,
        evidence=evidence,
        confidence=confidence,
        rationale=rationale,
        proposal_ids=proposal_ids,
        modifications=_freeze(modifications),
    )


@dataclass(frozen=True, slots=True)
class PendingRole1Decision:
    event: Role1Event
    decision: Role1Decision
    envelope: Mapping[str, Any]
    digest: str

    def activate(self) -> "Role1Effect":
        raise DecisionNotPersistedError(
            "Role1 decision must be persisted before activation"
        )


@dataclass(frozen=True, slots=True)
class PersistedRole1Decision:
    path: Path
    digest: str
    event: Role1Event
    decision: Role1Decision


@dataclass(frozen=True, slots=True)
class Role1Effect:
    """Audited high-level instruction for the separate episode Actor."""

    decision_id: str
    event_id: str
    proposal_disposition: ProposalDisposition
    action_kind: ActionKind
    selected_stage: str | None
    selected_tool: str | None
    direct_action: Mapping[str, Any] | None
    termination: TerminationDecision
    modifications: Mapping[str, Any]
    persisted_digest: str


class Role1DecisionStore:
    """Immutable, idempotent persistence gate for Role1 decisions."""

    def __init__(
        self,
        root: str | Path,
        *,
        catalog: ToolCatalog = DEFAULT_ROBOCASA_TOOL_CATALOG,
    ) -> None:
        self.root = Path(root).resolve()
        self.catalog = catalog

    def prepare(
        self, event: Role1Event, raw: Mapping[str, Any]
    ) -> PendingRole1Decision:
        unknown_tools = set(event.allowed_tools) - set(self.catalog.names())
        if unknown_tools:
            raise Role1ContractError(
                f"event allowlist contains unknown tools: {sorted(unknown_tools)}"
            )
        decision = validate_role1_decision(raw, event=event, catalog=self.catalog)
        envelope = {
            "schema_version": 1,
            "authority": "role1_high_level_agent",
            "event": event.to_dict(),
            "decision": decision.to_dict(),
        }
        digest = hashlib.sha256(canonical_json_bytes(envelope)).hexdigest()
        return PendingRole1Decision(event, decision, _freeze(envelope), digest)

    def persist(self, pending: PendingRole1Decision) -> PersistedRole1Decision:
        if not isinstance(pending, PendingRole1Decision):
            raise Role1PersistenceError("persist requires a prepared Role1 decision")
        target = self.root / f"{pending.decision.decision_id}.json"
        try:
            written = atomic_write_json(
                target, _thaw(pending.envelope), overwrite=False
            )
        except FileExistsError as exc:
            raise Role1PersistenceError(
                "decision_id already exists with different content"
            ) from exc
        if written != pending.digest:
            raise Role1PersistenceError("persisted decision digest mismatch")
        return PersistedRole1Decision(target, written, pending.event, pending.decision)

    def load(self, decision_id: str) -> PersistedRole1Decision:
        identity = _nonempty(decision_id, "decision_id")
        path = (self.root / f"{identity}.json").resolve()
        if path.parent != self.root:
            raise Role1PersistenceError("decision path escapes its store")
        if not path.is_file():
            raise DecisionNotPersistedError(f"decision is not persisted: {identity}")
        envelope = read_json(path)
        digest = hashlib.sha256(canonical_json_bytes(envelope)).hexdigest()
        if not isinstance(envelope, Mapping) or envelope.get("schema_version") != 1:
            raise Role1PersistenceError("invalid persisted Role1 envelope")
        if envelope.get("authority") != "role1_high_level_agent":
            raise Role1PersistenceError("persisted artifact has invalid authority")
        try:
            event = Role1Event.from_payload(envelope["event"])
            decision_payload = dict(envelope["decision"])
            for fixed in (
                "authority",
                "tool_authority",
                "critic_action_authority",
                "environment_write",
            ):
                decision_payload.pop(fixed, None)
            decision = validate_role1_decision(
                decision_payload, event=event, catalog=self.catalog
            )
        except (KeyError, TypeError, Role1ContractError) as exc:
            raise Role1PersistenceError(
                "persisted Role1 decision failed validation"
            ) from exc
        if decision.decision_id != identity:
            raise Role1PersistenceError(
                "persisted decision_id does not match its filename"
            )
        return PersistedRole1Decision(path, digest, event, decision)

    def activate(self, persisted: PersistedRole1Decision | str) -> Role1Effect:
        identity = (
            persisted if isinstance(persisted, str) else persisted.decision.decision_id
        )
        verified = self.load(identity)
        if (
            isinstance(persisted, PersistedRole1Decision)
            and verified.digest != persisted.digest
        ):
            raise Role1PersistenceError("persisted decision changed after publication")
        decision = verified.decision
        return Role1Effect(
            decision_id=decision.decision_id,
            event_id=decision.event_id,
            proposal_disposition=decision.proposal_disposition,
            action_kind=decision.action_kind,
            selected_stage=decision.selected_stage,
            selected_tool=decision.selected_tool,
            direct_action=decision.direct_action,
            termination=decision.termination,
            modifications=decision.modifications,
            persisted_digest=verified.digest,
        )


ROLE1_SYSTEM_CONTRACT = """You are the sole high-level Role1 decision authority
for one event in a robot episode. Tool outputs are proposals only. A critic may
only recommend rejecting the current action; it cannot execute, replace,
recover, switch, or terminate. Select no default tool and make no implicit
fallback. Use only the event's allowed stages and tools. Do not request, infer,
or mention experiment identity, random generators, or future rollout order.
Return exactly one JSON object and no Markdown or surrounding text. The object
must explicitly contain event_id, proposal_disposition, action_kind,
selected_stage, selected_tool, direct_action, termination, evidence,
confidence, rationale, proposal_ids, and modifications. Valid dispositions are
accept, reject, and modify. Valid action kinds are continue, switch, recover,
restage, regenerate, replace, and terminate. A direct action, when present,
must contain exactly the five canonical bounded RoboCasa action components.
Recover and replace require exactly one selected tool or direct action. Restage
requires both a different stage and a different executable selected tool; it
is not a passive request to observe again. Never reject a proposal without
materializing an allowed replacement tool or direct action. If no executable
alternative exists, do not use recover, replace, restage, or regenerate.
Only terminate may approve termination. When selecting a tool that needs an
invocation payload, place exactly that payload under modifications.parameters;
the tool remains proposal-only and a later Role1 event must approve any action
it proposes. If termination.approved is false, termination.reason must be the
empty string; put any explanation in rationale instead. When accepting an
existing proposal, modifications must be an empty object: cite its proposal_id
but never repeat its action arrays, hashes, or invocation payload. Only a
modify decision may provide replacement parameters under modifications. A tool
proposal explicitly marked proposal_role=frozen_recovery_alternative is an
executable alternative to the interrupted Critic-rejected chunk even when it
uses the same VLA tool; accept+continue must cite both proposal IDs. If that
frozen alternative names a different executable tool than current_tool, use
switch (or recover when selecting the recovery tool) instead of continue;
continue may only retain current_tool. When the read_role1_image tool is
available, you must call it on at least one current
image reference before returning the decision; image-reference text alone is
not visual evidence."""

_MODEL_REQUIRED_FIELDS = frozenset(
    {
        "event_id",
        "proposal_disposition",
        "selected_stage",
        "selected_tool",
        "direct_action",
        "termination",
        "evidence",
        "confidence",
        "rationale",
        "proposal_ids",
        "modifications",
    }
)
_MODEL_ROLES = frozenset({"assistant", "model", "codex_sdk", "claude_agent_sdk"})


def _strict_model_json(text: str) -> dict[str, Any]:
    """Parse one bare JSON object, rejecting duplicate keys and extra text."""

    source = text.strip()
    if not source:
        raise Role1ContractError("Role1 model returned no textual decision")

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise Role1ContractError(f"Role1 model repeated JSON key: {key}")
            result[key] = value
        return result

    try:
        value = json.loads(source, object_pairs_hook=unique_object)
    except Role1ContractError:
        raise
    except json.JSONDecodeError as exc:
        raise Role1ContractError(
            "Role1 model must return exactly one bare JSON object"
        ) from exc
    if not isinstance(value, dict):
        raise Role1ContractError("Role1 model output must be a JSON object")
    return value


def _model_text(messages: Any) -> str:
    if not isinstance(messages, list):
        raise Role1ContractError("planner messages must be an array")
    parts: list[str] = []
    seen_parts: set[str] = set()

    def append_text(text: str) -> None:
        normalized = text.strip()
        if normalized and normalized not in seen_parts:
            parts.append(normalized)
            seen_parts.add(normalized)

    for message in messages:
        if not isinstance(message, Mapping):
            raise Role1ContractError("planner message must be an object")
        if str(message.get("role", "")).lower() not in _MODEL_ROLES:
            continue
        content = message.get("content")
        if isinstance(content, str):
            append_text(content)
            continue
        if isinstance(content, list):
            for block in content:
                if (
                    isinstance(block, Mapping)
                    and block.get("type") == "text"
                    and isinstance(block.get("text"), str)
                    and str(block["text"]).strip()
                ):
                    append_text(str(block["text"]))
            continue
        raise Role1ContractError("model message content has an unsupported shape")
    return "\n".join(parts)


def _model_payload(
    event: Role1Event,
    *,
    catalog: ToolCatalog = DEFAULT_ROBOCASA_TOOL_CATALOG,
) -> dict[str, Any]:
    """Render only current, seed-blind evidence and an output shape reminder."""

    return {
        "event": event.to_dict(),
        "tool_contracts": [
            catalog.get(name).public_dict() for name in event.allowed_tools
        ],
        "required_output": {
            "event_id": event.event_id,
            "proposal_disposition": "accept|reject|modify",
            "action_kind": (
                "continue|switch|recover|restage|regenerate|replace|terminate"
            ),
            "selected_stage": "allowed stage or null",
            "selected_tool": "allowed proposal-only tool or null",
            "direct_action": "five canonical action components or null",
            "termination": {
                "approved": "boolean",
                "reason": "non-empty only when approved=true; otherwise empty",
            },
            "evidence": ["current-event evidence references"],
            "confidence": "number in [0,1]",
            "rationale": "non-empty string",
            "proposal_ids": ["reviewed current-event proposal ids"],
            "modifications": (
                "{} for accept/reject; for modify only, replacement parameters "
                "without copying an existing proposal payload"
            ),
        },
    }


class Role1ModelAdapter:
    """Obtain one audited raw decision and cross the persistence gate.

    A planner instance can be injected for tests. Otherwise the existing
    :func:`build_planner` factory constructs either the API or Codex backend.
    The adapter never repairs invalid model output and never chooses a tool.
    """

    def __init__(
        self,
        *,
        store: Role1DecisionStore,
        output_root: str | Path,
        planner_type: Literal["api", "codex"] = "codex",
        model: str | None = None,
        reasoning_effort: str | None = None,
        base_url: str | None = None,
        max_tokens: int = 4096,
        timeout_s: int = 600,
        max_turns: int = 2,
        toolkit_mode: Literal["empty", "read_only"] = "empty",
        require_visual_review: bool = False,
        planner: Any | None = None,
        planner_factory: Any = build_planner,
    ) -> None:
        if planner_type not in {"api", "codex"}:
            raise ValueError("Role1ModelAdapter supports only api or codex planners")
        if toolkit_mode not in {"empty", "read_only"}:
            raise ValueError("toolkit_mode must be empty or read_only")
        if max_tokens <= 0 or timeout_s <= 0 or max_turns <= 0:
            raise ValueError("model limits must be positive")
        self.store = store
        self.output_root = Path(output_root).resolve()
        self.planner_type = planner_type
        self.model = model
        if reasoning_effort not in {None, "low", "medium", "high", "xhigh"}:
            raise ValueError("unsupported Role1 reasoning_effort")
        self.reasoning_effort = reasoning_effort
        self.base_url = base_url
        self.max_tokens = max_tokens
        self.timeout_s = timeout_s
        self.max_turns = max_turns
        self.toolkit_mode = toolkit_mode
        self.require_visual_review = bool(require_visual_review)
        self._planner = planner
        self._planner_factory = planner_factory

    def _new_invocation_dir(self) -> Path:
        self.output_root.mkdir(parents=True, exist_ok=True)
        for _ in range(8):
            path = self.output_root / f"invocation-{uuid.uuid4().hex}"
            try:
                path.mkdir()
            except FileExistsError:
                continue
            return path
        raise Role1PersistenceError("could not allocate an invocation directory")

    def _toolkit(
        self,
        *,
        event: Role1Event,
        image_payloads: Mapping[str, str] | None,
        viewed_references: set[str] | None = None,
    ) -> Toolkit:
        toolkit = Toolkit()
        names = {"describe_tools"} if self.toolkit_mode == "read_only" else set()
        if image_payloads is not None:
            payloads = {str(name): str(value) for name, value in image_payloads.items()}
            references = dict(event.image_references)
            if set(payloads) != set(references):
                raise Role1ModelError(
                    "private Role1 images do not match event image references"
                )
            by_reference: dict[str, str] = {}
            for name, payload in payloads.items():
                expected = "sha256:" + hashlib.sha256(payload.encode()).hexdigest()
                if references[name] != expected:
                    raise Role1ModelError(
                        "private Role1 image failed reference integrity validation"
                    )
                by_reference[expected] = payload

            def read_role1_image(reference: str) -> dict[str, Any]:
                value = by_reference.get(str(reference))
                if value is None:
                    raise KeyError("unknown Role1 image reference")
                if not value.startswith("data:") or "," not in value:
                    raise ValueError("Role1 image must use a data URL")
                encoded = value.split(",", 1)[1]
                raw = base64.b64decode(encoded, validate=True)
                import imageio.v3 as iio

                image = iio.imread(io.BytesIO(raw))
                png = iio.imwrite("<bytes>", image, extension=".png")
                if viewed_references is not None:
                    viewed_references.add(str(reference))
                return {
                    "reference": reference,
                    "kind": "current_observation_image",
                    "_image_bytes": png,
                }

            toolkit.add_tool(
                "read_role1_image",
                {
                    "name": "read_role1_image",
                    "description": (
                        "Read one current-observation image using an opaque "
                        "sha256 reference from image_references."
                    ),
                    "input_schema": {
                        "type": "object",
                        "properties": {"reference": {"type": "string"}},
                        "required": ["reference"],
                    },
                },
                read_role1_image,
            )
            names.add("read_role1_image")
        toolkit.retain_tools(names)
        return toolkit

    def _planner_for(self, invocation: Path) -> Any:
        if self._planner is not None:
            return self._planner
        return self._planner_factory(
            self.planner_type,
            output_dir=invocation,
            recipe_tag="role1-decision",
            env_name="robocasa",
            base_url=self.base_url,
            model=self.model,
            reasoning_effort=self.reasoning_effort,
            max_tokens=self.max_tokens,
            planner_timeout_s=self.timeout_s,
            no_images=False,
        )

    @staticmethod
    def _write_result_artifacts(invocation: Path, result: Any) -> None:
        messages = getattr(result, "messages", [])
        stats = getattr(result, "stats", {})
        error = getattr(result, "error", None)
        atomic_write_json(invocation / "planner_messages.json", messages)
        atomic_write_json(invocation / "planner_stats.json", stats)
        atomic_write_json(invocation / "planner_error.json", {"error": error})
        atomic_write_json(
            invocation / "planner_result.json",
            {"messages": messages, "stats": stats, "error": error},
        )

    @staticmethod
    def _write_failure(invocation: Path, phase: str, exc: BaseException) -> None:
        target = invocation / "failure.json"
        if target.exists():
            return
        atomic_write_json(
            target,
            {
                "phase": phase,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": "".join(
                    traceback.format_exception(type(exc), exc, exc.__traceback__)
                ),
            },
        )

    def decide(
        self,
        event: Role1Event,
        *,
        image_payloads: Mapping[str, str] | None = None,
    ) -> Role1Effect:
        """Call the model and return only an already-persisted Role1 effect."""

        if not isinstance(event, Role1Event):
            raise Role1ModelError("Role1ModelAdapter requires a validated Role1Event")
        invocation = self._new_invocation_dir()
        payload = _model_payload(event, catalog=self.store.catalog)
        _assert_seed_blind(payload, path="model_payload")
        user_message = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        atomic_write_json(
            invocation / "input.json",
            {
                "system_contract": ROLE1_SYSTEM_CONTRACT,
                "user_payload": payload,
                "planner_type": self.planner_type,
                "model": self.model,
                "reasoning_effort": self.reasoning_effort,
                "toolkit_mode": self.toolkit_mode,
                "require_visual_review": self.require_visual_review,
            },
        )
        viewed_references: set[str] = set()
        toolkit = self._toolkit(
            event=event,
            image_payloads=image_payloads,
            viewed_references=viewed_references,
        )
        result: Any | None = None
        phase = "planner"
        planner_started_at: float | None = None
        planner_started_monotonic: float | None = None
        planner_timing_written = False
        try:
            planner = self._planner_for(invocation)
            planner_started_at = time.time()
            planner_started_monotonic = time.monotonic()
            result = planner.solve(
                system_prompt=ROLE1_SYSTEM_CONTRACT,
                user_message=user_message,
                toolkit=toolkit,
                max_turns=self.max_turns,
            )
            visual_required = bool(self.require_visual_review and image_payloads)
            visual_review = {
                "required": visual_required,
                "available_cameras": sorted(event.image_references),
                "viewed_cameras": sorted(
                    name
                    for name, reference in event.image_references.items()
                    if reference in viewed_references
                ),
                "viewed_references": sorted(viewed_references),
                "completed": bool(viewed_references) or not visual_required,
            }
            atomic_write_json(
                invocation / "visual_review.json",
                visual_review,
                overwrite=False,
            )
            if visual_required and not viewed_references:
                raise Role1ContractError(
                    "Role1 must read at least one current image before deciding"
                )
            atomic_write_json(
                invocation / "planner_timing.json",
                {
                    "phase": "model_inference",
                    "status": "completed",
                    "model": self.model,
                    "reasoning_effort": self.reasoning_effort,
                    "started_at_unix_s": planner_started_at,
                    "finished_at_unix_s": time.time(),
                    "elapsed_s": max(
                        0.0, time.monotonic() - planner_started_monotonic
                    ),
                },
                overwrite=False,
            )
            planner_timing_written = True
            self._write_result_artifacts(invocation, result)
            if (
                self.reasoning_effort is not None
                and self.model is not None
                and result.stats.get("model") != self.model
            ):
                raise Role1ModelError("planner did not attest the frozen model")
            if (
                self.reasoning_effort is not None
                and result.stats.get("reasoning_effort") != self.reasoning_effort
            ):
                raise Role1ModelError(
                    "planner did not attest the frozen reasoning_effort"
                )
            if getattr(result, "error", None):
                raise Role1ModelError(f"Role1 planner failed: {result.error}")
            phase = "model_output"
            raw = _strict_model_json(_model_text(getattr(result, "messages", None)))
            required = set(_MODEL_REQUIRED_FIELDS)
            if "action_kind" not in raw and "intent" not in raw:
                required.add("action_kind")
            missing = sorted(required - set(raw))
            if missing:
                raise Role1ContractError(
                    f"Role1 model omitted required fields: {missing}"
                )
            atomic_write_json(invocation / "model_output.json", raw)
            phase = "prepare"
            pending = self.store.prepare(event, raw)
            phase = "persist"
            persisted = self.store.persist(pending)
            phase = "activate"
            effect = self.store.activate(persisted)
            atomic_write_json(
                invocation / "completion.json",
                {
                    "decision_id": effect.decision_id,
                    "decision_digest": persisted.digest,
                    "decision_path": str(persisted.path),
                    "activated_after_persistence": True,
                },
            )
            return effect
        except Exception as exc:
            if (
                planner_started_at is not None
                and planner_started_monotonic is not None
                and not planner_timing_written
            ):
                try:
                    atomic_write_json(
                        invocation / "planner_timing.json",
                        {
                            "phase": "model_inference",
                            "status": "failed",
                            "model": self.model,
                            "reasoning_effort": self.reasoning_effort,
                            "started_at_unix_s": planner_started_at,
                            "finished_at_unix_s": time.time(),
                            "elapsed_s": max(
                                0.0,
                                time.monotonic() - planner_started_monotonic,
                            ),
                            "error_type": type(exc).__name__,
                        },
                        overwrite=False,
                    )
                except Exception:
                    pass
            if result is None:
                try:
                    atomic_write_json(invocation / "planner_messages.json", [])
                    atomic_write_json(invocation / "planner_stats.json", {})
                    atomic_write_json(
                        invocation / "planner_error.json",
                        {"error": f"{type(exc).__name__}: {exc}"},
                    )
                except Exception:
                    pass
            try:
                self._write_failure(invocation, phase, exc)
            except Exception:
                pass
            if isinstance(exc, Role1ModelError):
                raise
            raise Role1ModelError(
                f"Role1 invocation failed closed during {phase}: {type(exc).__name__}"
            ) from exc
        finally:
            toolkit.close()

    __call__ = decide


__all__ = [
    "ACTION_KINDS",
    "PROPOSAL_DISPOSITIONS",
    "CriticProposal",
    "DecisionNotPersistedError",
    "PendingRole1Decision",
    "PersistedRole1Decision",
    "Role1ContractError",
    "Role1Decision",
    "Role1DecisionStore",
    "Role1Effect",
    "Role1Event",
    "Role1ModelAdapter",
    "Role1ModelError",
    "Role1PersistenceError",
    "ROLE1_SYSTEM_CONTRACT",
    "TerminationDecision",
    "ToolProposal",
    "validate_role1_decision",
]
