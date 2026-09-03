# Copyright (c) 2026 Zetta Contributors
"""Model-backed Role1 for RoboTwin: one audited decision per chunk boundary.

The protocol's role split is what this file has to protect. The Critic proposes,
Role1 rules, and only the Actor writes actions -- so this module obtains a
verdict and persists it, and it never chooses a tool, never repairs invalid
model output, and never touches the simulator.

Three checks are specific to a bimanual robot and are the reason this is not a
generic adapter:

1. **The model may not invent a hand.** A verdict must echo the arm it is ruling
   on, and that arm must be the one the Critic's evidence concerned. Letting the
   model redirect a left-arm observation into a right-arm recovery would produce
   an audit trail that reads as compliant while authorising something the
   evidence never supported.
2. **A proposal that names no arm cannot be accepted.** A RoboTwin recovery is
   not executable without one, so accepting it would promise an action nobody
   can take.
3. **The payload is chunk-granular and says so.** This family sees one frame per
   chunk, so a verdict must not be asked to reason about per-step motion it was
   never shown.

Seed blindness is enforced on the way out: the payload must carry no seed, RNG
or future-schedule metadata, because a Role1 that can see the schedule is no
longer making an online decision.
"""

from __future__ import annotations

import json
import re
import time
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from robots.robotwin.action_contract import ARMS, ArmSelectionError, normalize_arm
from robots.robotwin.critic_runtime import GRANULARITY_FEATURE, arm_from_proposal
from robots.robotwin.role1_actor import ROLE1_SYSTEM_CONTRACT, Role1Decision
from zetta.evolution.jsonio import atomic_write_json, canonical_sha256

_FORBIDDEN_KEY_TOKENS = frozenset({"seed", "rng", "master_seed", "policy_rng"})
"""Key names that would leak experiment identity into an online decision."""

_TOKEN_SPLIT = re.compile(r"[^a-z0-9]+")
"""Split an identifier into lower-case tokens."""

_FORBIDDEN_TEXT = re.compile(r"\b(?:seed|rng)\s*[:=]\s*\d+", re.IGNORECASE)
"""Free text that embeds a concrete seed value."""

VALID_DISPOSITIONS = ("accept", "reject")
"""The verdicts Role1 may return for a Critic proposal."""


class Role1ContractError(ValueError):
    """The model's output violates the frozen Role1 contract."""


class Role1ModelError(RuntimeError):
    """The model could not be reached, or returned nothing usable."""


def _tokens(value: str) -> set[str]:
    """Split an identifier into lower-case tokens.

    Args:
        value: The identifier.

    Returns:
        Its tokens.
    """
    return {token for token in _TOKEN_SPLIT.split(value.lower()) if token}


def assert_seed_blind(value: Any, *, path: str = "input") -> None:
    """Reject explicit experiment identity and future-schedule leakage.

    Args:
        value: The payload to inspect, recursively.
        path: Location prefix used in the error message.

    Raises:
        Role1ContractError: The payload carries seed/RNG metadata, a future
            schedule, or free text embedding a seed value.
    """
    if isinstance(value, Mapping):
        for key, item in value.items():
            lowered = str(key).lower()
            tokens = _tokens(str(key))
            if lowered in _FORBIDDEN_KEY_TOKENS or tokens & {"seed", "rng"}:
                raise Role1ContractError(
                    f"seed or RNG metadata is forbidden at {path}.{key}"
                )
            if "future" in tokens and "schedule" in tokens:
                raise Role1ContractError(
                    f"future schedule is forbidden at {path}.{key}"
                )
            assert_seed_blind(item, path=f"{path}.{key}")
    elif isinstance(value, list | tuple):
        for index, item in enumerate(value):
            assert_seed_blind(item, path=f"{path}[{index}]")
    elif isinstance(value, str) and _FORBIDDEN_TEXT.search(value):
        raise Role1ContractError(f"seed value is forbidden in text at {path}")


@dataclass(frozen=True, slots=True)
class Role1Event:
    """One validated decision boundary, as the model will see it.

    Attributes:
        event_id: Stable identifier for this boundary.
        task: The RoboTwin task.
        chunk_index: Index of the chunk boundary.
        proposals: The Critic's proposals, most significant first.
        features: The Critic feature plane for this boundary.
        arm_scoped_tools: Tools the task binding declares as arm-scoped.
    """

    event_id: str
    task: str
    chunk_index: int
    proposals: tuple[Mapping[str, Any], ...]
    features: Mapping[str, Any] = field(default_factory=dict)
    arm_scoped_tools: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Validate the event.

        Raises:
            Role1ContractError: The event has no proposal to rule on.
        """
        if not self.proposals:
            raise Role1ContractError("a Role1 event must carry a Critic proposal")

    @property
    def leading_proposal(self) -> Mapping[str, Any]:
        """The proposal Role1 is being asked to rule on.

        Returns:
            The first proposal.
        """
        return self.proposals[0]

    @property
    def evidence_arm(self) -> str | None:
        """The arm the leading proposal's evidence concerns.

        Returns:
            The canonical arm name, or ``None`` when the proposal names none.
        """
        return arm_from_proposal(self.leading_proposal)

    def model_payload(self) -> dict[str, Any]:
        """Build the seed-blind payload handed to the model.

        Returns:
            A JSON-friendly payload.
        """
        payload = {
            "event_id": self.event_id,
            "task": self.task,
            "chunk_index": int(self.chunk_index),
            "robot": {
                "kind": "bimanual",
                "arms": list(ARMS),
                "action_dim": 14,
                "action_space": "absolute joint targets",
            },
            "evidence": {
                "granularity": str(self.features.get(GRANULARITY_FEATURE, "chunk")),
                "note": (
                    "One observation per chunk. There are no intermediate "
                    "frames; do not reason about per-step motion."
                ),
                "features": {
                    key: value
                    for key, value in sorted(self.features.items())
                    if isinstance(value, int | float | str | bool)
                },
            },
            "critic_proposals": [
                {
                    "rule_id": str(row.get("rule_id", "")),
                    "proposal": str(row.get("proposal", row.get("reason", ""))),
                    "arm": arm_from_proposal(row),
                    "feature": row.get("feature"),
                    "observed": row.get("observed"),
                }
                for row in self.proposals
            ],
            "arm_scoped_tools": list(self.arm_scoped_tools),
            "response_contract": {
                "dispositions": list(VALID_DISPOSITIONS),
                "required_keys": ["event_id", "proposal_disposition", "arm", "reason"],
                "arm_rule": (
                    "`arm` must equal the arm named by the proposal you are "
                    "ruling on. If that proposal names no arm, the only valid "
                    "disposition is reject."
                ),
            },
        }
        assert_seed_blind(payload, path="model_payload")
        return payload


class Role1DecisionStore:
    """Append-only persistence for Role1 verdicts.

    A verdict that is not durably recorded before it is acted on cannot be
    audited afterwards, so the Actor is only ever handed decisions that this
    store has already written.
    """

    def __init__(self, root: str | Path) -> None:
        """Initialize the store.

        Args:
            root: Directory decisions are written into.
        """
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._records: list[dict[str, Any]] = []

    def persist(
        self, *, event: Role1Event, decision: Role1Decision, raw: Mapping[str, Any]
    ) -> dict[str, Any]:
        """Write one verdict and return its record.

        Args:
            event: The boundary that was decided.
            decision: The parsed verdict.
            raw: The model's raw JSON object.

        Returns:
            The persisted record, carrying its own content hash.
        """
        record = {
            "schema_version": 1,
            "event_id": event.event_id,
            "task": event.task,
            "chunk_index": int(event.chunk_index),
            "evidence_arm": event.evidence_arm,
            "decision": decision.public_dict(),
            "raw_model_object": dict(raw),
        }
        record["record_sha256"] = canonical_sha256(record)
        atomic_write_json(self.root / f"{event.event_id}.json", record, overwrite=False)
        self._records.append(record)
        return record

    @property
    def records(self) -> tuple[dict[str, Any], ...]:
        """Every verdict persisted by this store.

        Returns:
            The records in write order.
        """
        return tuple(self._records)


def strict_model_json(text: str) -> dict[str, Any]:
    """Parse exactly one bare JSON object, rejecting repairs and extra text.

    Args:
        text: The model's textual response.

    Returns:
        The parsed object.

    Raises:
        Role1ContractError: The response is empty, is not a single bare object,
            or repeats a key.
    """
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
        raise Role1ContractError("Role1 model must return a JSON object")
    return value


def model_text(messages: Any) -> str:
    """Extract the assistant text from a planner result.

    Args:
        messages: The planner's message list.

    Returns:
        The concatenated distinct text parts.

    Raises:
        Role1ContractError: The messages are not a list of objects.
    """
    if not isinstance(messages, list):
        raise Role1ContractError("planner messages must be an array")
    parts: list[str] = []
    seen: set[str] = set()
    for message in messages:
        if not isinstance(message, Mapping):
            raise Role1ContractError("planner message must be an object")
        content = message.get("content")
        chunks = content if isinstance(content, list) else [content]
        for chunk in chunks:
            text = (
                chunk.get("text")
                if isinstance(chunk, Mapping)
                else chunk
                if isinstance(chunk, str)
                else None
            )
            if isinstance(text, str) and text.strip() and text.strip() not in seen:
                parts.append(text.strip())
                seen.add(text.strip())
    return "\n".join(parts)


def parse_decision(raw: Mapping[str, Any], *, event: Role1Event) -> Role1Decision:
    """Validate one raw model object against the RoboTwin Role1 contract.

    Args:
        raw: The model's JSON object.
        event: The boundary being decided.

    Returns:
        The validated verdict.

    Raises:
        Role1ContractError: A required key is missing, the disposition is
            unknown, the event id does not match, the arm is absent or does not
            match the evidence, or an arm-less proposal was accepted.
    """
    missing = [
        key
        for key in ("event_id", "proposal_disposition", "reason")
        if not str(raw.get(key, "")).strip()
    ]
    if missing:
        raise Role1ContractError(f"Role1 decision is missing keys: {missing}")
    if str(raw["event_id"]) != event.event_id:
        raise Role1ContractError("Role1 decision answers a different event")
    disposition = str(raw["proposal_disposition"]).strip().lower()
    if disposition not in VALID_DISPOSITIONS:
        raise Role1ContractError(
            f"unknown proposal_disposition {disposition!r}; "
            f"expected one of {list(VALID_DISPOSITIONS)}"
        )
    accepted = disposition == "accept"
    evidence_arm = event.evidence_arm
    supplied = raw.get("arm")
    arm: str | None = None
    if supplied is not None and str(supplied).strip():
        try:
            arm = normalize_arm(str(supplied))
        except ArmSelectionError as exc:
            raise Role1ContractError(f"Role1 decision names an unusable arm: {exc}")

    if accepted:
        if evidence_arm is None:
            raise Role1ContractError(
                "Role1 accepted a proposal that names no arm; a RoboTwin "
                "recovery cannot be executed without one"
            )
        if arm is None:
            raise Role1ContractError(
                "Role1 accepted an arm-scoped proposal without naming the arm"
            )
        if arm != evidence_arm:
            # The model may not redirect the evidence to the other hand: the
            # audit would read as compliant while authorising something the
            # Critic never observed.
            raise Role1ContractError(
                f"Role1 ruled on the {arm} arm but the evidence concerns the "
                f"{evidence_arm} arm"
            )
    return Role1Decision(
        accepted=accepted,
        reason=str(raw["reason"]).strip(),
        proposal_id=str(event.leading_proposal.get("rule_id") or "") or None,
        arm=arm if accepted else (arm or evidence_arm),
    )


class ModelBackedRole1:
    """A ``Role1Decider`` backed by the shared planner stack.

    The adapter never repairs invalid model output: a verdict that does not meet
    the contract is a contract failure, and silently fixing it would hide the
    fact that the model was not following the rules the campaign froze.
    """

    def __init__(
        self,
        *,
        store: Role1DecisionStore,
        binding: Any,
        output_root: str | Path,
        planner_type: Literal["api", "codex"] = "codex",
        model: str | None = None,
        reasoning_effort: str | None = None,
        base_url: str | None = None,
        max_tokens: int = 4096,
        timeout_s: int = 600,
        max_turns: int = 2,
        planner: Any | None = None,
        planner_factory: Any = None,
        fail_closed: bool = True,
    ) -> None:
        """Initialize the adapter.

        Args:
            store: Where verdicts are persisted.
            binding: The task's frozen tool binding.
            output_root: Directory for per-invocation artifacts.
            planner_type: ``"api"`` or ``"codex"``.
            model: Model identifier.
            reasoning_effort: Reasoning effort, when the backend supports it.
            base_url: Override base URL.
            max_tokens: Response budget.
            timeout_s: Planner timeout.
            max_turns: Maximum planner turns.
            planner: Injected planner, for tests.
            planner_factory: Factory used when no planner is injected.
            fail_closed: Whether a model or contract failure becomes a rejection
                rather than an exception. Rejecting continues the episode on the
                policy's own chunk, which is the safe direction: a Role1 that
                cannot be reached must not be able to authorise a recovery.

        Raises:
            ValueError: A limit is not positive, or the planner type is unknown.
        """
        if planner_type not in {"api", "codex"}:
            raise ValueError("ModelBackedRole1 supports only api or codex planners")
        if max_tokens <= 0 or timeout_s <= 0 or max_turns <= 0:
            raise ValueError("model limits must be positive")
        self.store = store
        self.binding = binding
        self.output_root = Path(output_root)
        self.planner_type = planner_type
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.base_url = base_url
        self.max_tokens = max_tokens
        self.timeout_s = timeout_s
        self.max_turns = max_turns
        self._planner = planner
        self._planner_factory = planner_factory
        self.fail_closed = fail_closed

    def _build_planner(self, invocation: Path) -> Any:
        """Return the planner for one invocation.

        Args:
            invocation: Directory for this invocation's artifacts.

        Returns:
            The planner instance.
        """
        if self._planner is not None:
            return self._planner
        factory = self._planner_factory
        if factory is None:
            from zetta.planner.base import build_planner

            factory = build_planner
        return factory(
            self.planner_type,
            output_dir=invocation,
            recipe_tag="role1-decision",
            env_name="robotwin",
            base_url=self.base_url,
            model=self.model,
            reasoning_effort=self.reasoning_effort,
            max_tokens=self.max_tokens,
            planner_timeout_s=self.timeout_s,
            no_images=True,
        )

    def decide(
        self,
        *,
        task: str,
        chunk_index: int,
        features: Mapping[str, Any],
        proposals: Sequence[Mapping[str, Any]],
    ) -> Role1Decision:
        """Obtain, validate and persist one verdict.

        Args:
            task: The task under evaluation.
            chunk_index: The chunk boundary index.
            features: The Critic feature plane.
            proposals: The Critic's proposals.

        Returns:
            The verdict. On a model or contract failure with ``fail_closed``,
            a rejection carrying the failure reason; with ``fail_closed`` off,
            :meth:`_failed` re-raises the original ``Role1ModelError`` or
            ``Role1ContractError`` instead.
        """
        if not proposals:
            return Role1Decision(accepted=False, reason="no critic proposal")
        event = Role1Event(
            event_id=f"role1-{chunk_index:06d}-{uuid.uuid4().hex[:8]}",
            task=task,
            chunk_index=int(chunk_index),
            proposals=tuple(dict(row) for row in proposals),
            features=dict(features),
            arm_scoped_tools=tuple(
                getattr(self.binding, "arm_scoped_tool_names", ()) or ()
            ),
        )
        invocation = self.output_root / event.event_id
        invocation.mkdir(parents=True, exist_ok=True)
        payload = event.model_payload()
        atomic_write_json(
            invocation / "input.json",
            {
                "system_contract": ROLE1_SYSTEM_CONTRACT,
                "user_payload": payload,
                "planner_type": self.planner_type,
                "model": self.model,
            },
            overwrite=False,
        )
        started = time.monotonic()
        try:
            planner = self._build_planner(invocation)
            result = planner.solve(
                system_prompt=ROLE1_SYSTEM_CONTRACT,
                user_message=json.dumps(payload, ensure_ascii=False, sort_keys=True),
                toolkit=None,
                max_turns=self.max_turns,
            )
            raw = strict_model_json(model_text(getattr(result, "messages", None)))
            decision = parse_decision(raw, event=event)
        except (Role1ContractError, Role1ModelError) as exc:
            return self._failed(event, invocation, exc, started)
        except Exception as exc:  # noqa: BLE001 - any planner failure is a Role1 failure
            return self._failed(event, invocation, Role1ModelError(str(exc)), started)
        atomic_write_json(
            invocation / "timing.json",
            {"elapsed_s": round(time.monotonic() - started, 3)},
            overwrite=False,
        )
        self.store.persist(event=event, decision=decision, raw=raw)
        return decision

    def _failed(
        self,
        event: Role1Event,
        invocation: Path,
        exc: BaseException,
        started: float,
    ) -> Role1Decision:
        """Record a failure and either reject or re-raise.

        Args:
            event: The boundary being decided.
            invocation: The invocation directory.
            exc: The failure.
            started: Monotonic start time.

        Returns:
            A rejection, when failing closed. With ``fail_closed`` off the
            original failure is re-raised unchanged, so a campaign that wants
            Role1 outages to be loud gets them.
        """
        atomic_write_json(
            invocation / "failure.json",
            {
                "error_type": type(exc).__name__,
                "error": str(exc),
                "elapsed_s": round(time.monotonic() - started, 3),
            },
            overwrite=False,
        )
        if not self.fail_closed:
            raise exc
        # Rejecting continues on the policy's own chunk. The opposite default --
        # accepting when Role1 is unreachable -- would let an outage authorise
        # recoveries nobody ruled on.
        decision = Role1Decision(
            accepted=False,
            reason=f"role1 unavailable: {type(exc).__name__}: {exc}",
            proposal_id=str(event.leading_proposal.get("rule_id") or "") or None,
            arm=event.evidence_arm,
        )
        self.store.persist(event=event, decision=decision, raw={"error": str(exc)})
        return decision


__all__ = [
    "ModelBackedRole1",
    "Role1ContractError",
    "Role1DecisionStore",
    "Role1Event",
    "Role1ModelError",
    "assert_seed_blind",
    "model_text",
    "parse_decision",
    "strict_model_json",
]
