# Copyright (c) 2026 Zetta Contributors
"""The model-backed RoboTwin Role1.

The checks worth pinning are the ones a generic adapter would not have. Role1 is
the decision authority, so the failure that matters is not "the model said
something malformed" -- that is caught by strict parsing -- but "the model said
something well-formed that quietly exceeds its authority": redirecting a
left-arm observation into a right-arm recovery, or authorising a recovery for a
proposal that named no hand at all.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from robots.robotwin.critic_runtime import GRANULARITY_FEATURE
from robots.robotwin.role1_agent import (
    ModelBackedRole1,
    Role1ContractError,
    Role1DecisionStore,
    Role1Event,
    assert_seed_blind,
    model_text,
    parse_decision,
    strict_model_json,
)
from robots.robotwin.tool_bindings import binding_for_task


def _event(arm: str | None = "left", **overrides) -> Role1Event:
    """Build a decision event.

    Args:
        arm: The arm the proposal's evidence concerns, or ``None``.
        **overrides: Field overrides.

    Returns:
        The event.
    """
    proposal = {"rule_id": "left-arm-stalled", "proposal": "recover the arm"}
    if arm is not None:
        proposal["arm"] = arm
    payload = {
        "event_id": "role1-000001-abcdef12",
        "task": "adjust_bottle",
        "chunk_index": 1,
        "proposals": (proposal,),
        "features": {GRANULARITY_FEATURE: "chunk", "robotwin.chunk.index": 1},
    }
    payload.update(overrides)
    return Role1Event(**payload)


class _FakePlanner:
    """A planner returning a canned assistant message."""

    def __init__(self, text: str | None = None, raises: Exception | None = None):
        """Configure the fake.

        Args:
            text: The assistant text to return.
            raises: An exception to raise instead.
        """
        self.text = text
        self.raises = raises
        self.calls: list[dict] = []

    def solve(self, *, system_prompt, user_message, toolkit, max_turns):
        """Return the canned response."""
        self.calls.append(
            {"system_prompt": system_prompt, "user_message": user_message}
        )
        if self.raises is not None:
            raise self.raises
        return SimpleNamespace(
            messages=[{"role": "assistant", "content": [{"text": self.text}]}],
            stats={},
            error=None,
        )


def _adapter(tmp_path: Path, planner: _FakePlanner, **overrides) -> ModelBackedRole1:
    """Build an adapter over a fake planner.

    Args:
        tmp_path: Test directory.
        planner: The fake planner.
        **overrides: Constructor overrides.

    Returns:
        The adapter.
    """
    kwargs = {
        "store": Role1DecisionStore(tmp_path / "decisions"),
        "binding": binding_for_task("adjust_bottle"),
        "output_root": tmp_path / "invocations",
        "planner": planner,
    }
    kwargs.update(overrides)
    return ModelBackedRole1(**kwargs)


def _reply(event: Role1Event, **fields) -> str:
    """Build a well-formed model reply.

    Args:
        event: The event being answered.
        **fields: Field overrides.

    Returns:
        The JSON text.
    """
    payload = {
        "event_id": event.event_id,
        "proposal_disposition": "accept",
        "arm": "left",
        "reason": "the left arm has not moved for two chunks",
    }
    payload.update(fields)
    return json.dumps(payload)


# ------------------------------------------------------------ seed blindness


def test_seed_and_rng_metadata_are_refused() -> None:
    """A Role1 that can see the schedule is no longer deciding online."""
    with pytest.raises(Role1ContractError, match="seed or RNG"):
        assert_seed_blind({"seed": 7})
    with pytest.raises(Role1ContractError, match="seed or RNG"):
        assert_seed_blind({"outer": [{"policy_rng": 3}]})
    with pytest.raises(Role1ContractError, match="future schedule"):
        assert_seed_blind({"future_schedule": ["a"]})
    with pytest.raises(Role1ContractError, match="seed value"):
        assert_seed_blind({"note": "use seed=1234 next"})
    assert_seed_blind({"chunk_index": 3, "arm": "left"})


def test_model_payload_is_seed_blind_and_states_the_granularity() -> None:
    """The model must not be asked to reason about frames it was never shown."""
    payload = _event().model_payload()
    assert payload["evidence"]["granularity"] == "chunk"
    assert "no intermediate frames" in payload["evidence"]["note"]
    assert payload["robot"]["action_space"] == "absolute joint targets"
    assert_seed_blind(payload)


# ----------------------------------------------------------- strict parsing


def test_only_one_bare_json_object_is_accepted() -> None:
    """No repair: a response that is not the contract is a contract failure."""
    assert strict_model_json('{"a": 1}') == {"a": 1}
    for bad in ("", "not json", "[1,2]", '{"a":1} trailing'):
        with pytest.raises(Role1ContractError):
            strict_model_json(bad)


def test_repeated_keys_are_refused() -> None:
    """A duplicate key makes the verdict ambiguous, so it is not a verdict."""
    with pytest.raises(Role1ContractError, match="repeated JSON key"):
        strict_model_json('{"arm": "left", "arm": "right"}')


def test_model_text_joins_distinct_assistant_parts() -> None:
    """Planner backends differ in how they chunk a reply."""
    messages = [
        {"role": "assistant", "content": [{"text": "one"}, {"text": "one"}]},
        {"role": "assistant", "content": "two"},
    ]
    assert model_text(messages) == "one\ntwo"
    with pytest.raises(Role1ContractError):
        model_text("not a list")


# ------------------------------------------------------- decision contract


def test_a_well_formed_acceptance_is_parsed() -> None:
    """The happy path still has to carry the arm it ruled on."""
    event = _event(arm="left")
    decision = parse_decision(json.loads(_reply(event)), event=event)
    assert decision.accepted is True
    assert decision.arm == "left"
    assert decision.proposal_id == "left-arm-stalled"


def test_the_model_may_not_redirect_the_evidence_to_the_other_hand() -> None:
    """This is the check a single-arm adapter has no reason to have.

    Accepting a right-arm recovery on left-arm evidence would leave an audit
    trail that reads as compliant while authorising something the Critic never
    observed.
    """
    event = _event(arm="left")
    with pytest.raises(Role1ContractError, match="evidence concerns the left arm"):
        parse_decision(json.loads(_reply(event, arm="right")), event=event)


def test_accepting_an_arm_less_proposal_is_a_contract_failure() -> None:
    """A RoboTwin recovery without an arm is not executable."""
    event = _event(arm=None)
    with pytest.raises(Role1ContractError, match="names no arm"):
        parse_decision(json.loads(_reply(event, arm=None)), event=event)


def test_accepting_without_naming_the_arm_is_refused() -> None:
    """The verdict must state which hand it authorised."""
    event = _event(arm="left")
    payload = json.loads(_reply(event))
    payload.pop("arm")
    with pytest.raises(Role1ContractError, match="without naming the arm"):
        parse_decision(payload, event=event)


def test_a_rejection_needs_no_arm() -> None:
    """Rejecting is always available, including when the evidence named no hand."""
    event = _event(arm=None)
    payload = json.loads(_reply(event, proposal_disposition="reject", arm=None))
    decision = parse_decision(payload, event=event)
    assert decision.accepted is False


def test_unknown_disposition_and_wrong_event_are_refused() -> None:
    """A verdict must answer this boundary, with one of the frozen verdicts."""
    event = _event()
    with pytest.raises(Role1ContractError, match="unknown proposal_disposition"):
        parse_decision(
            json.loads(_reply(event, proposal_disposition="modify")), event=event
        )
    with pytest.raises(Role1ContractError, match="different event"):
        parse_decision(json.loads(_reply(event, event_id="other")), event=event)


def test_missing_required_keys_are_named(tmp_path: Path) -> None:
    """The error says which keys were missing, not merely that it failed."""
    event = _event()
    with pytest.raises(Role1ContractError, match="missing keys"):
        parse_decision({"event_id": event.event_id}, event=event)


# -------------------------------------------------------------- the adapter


def test_adapter_persists_before_returning(tmp_path: Path) -> None:
    """A verdict that was acted on but never recorded cannot be audited."""
    planner = _FakePlanner()
    adapter = _adapter(tmp_path, planner)
    event_ids = []

    def _capture(**kwargs):
        # The event id is generated inside decide(), so answer whatever it asks.
        payload = json.loads(kwargs["user_message"])
        event_ids.append(payload["event_id"])
        planner.text = json.dumps(
            {
                "event_id": payload["event_id"],
                "proposal_disposition": "accept",
                "arm": "left",
                "reason": "stalled",
            }
        )
        return _FakePlanner.solve(planner, **kwargs)

    planner.solve = _capture  # type: ignore[method-assign]
    decision = adapter.decide(
        task="adjust_bottle",
        chunk_index=1,
        features={GRANULARITY_FEATURE: "chunk"},
        proposals=[{"rule_id": "left-arm-stalled", "arm": "left"}],
    )
    assert decision.accepted is True
    records = adapter.store.records
    assert len(records) == 1
    assert records[0]["evidence_arm"] == "left"
    assert len(records[0]["record_sha256"]) == 64
    assert (tmp_path / "decisions" / f"{event_ids[0]}.json").is_file()


def test_a_planner_failure_rejects_rather_than_authorising(tmp_path: Path) -> None:
    """Failing closed is the safe direction.

    The opposite default -- accepting when Role1 cannot be reached -- would let
    an outage authorise recoveries nobody ruled on.
    """
    adapter = _adapter(tmp_path, _FakePlanner(raises=RuntimeError("boom")))
    decision = adapter.decide(
        task="adjust_bottle",
        chunk_index=0,
        features={},
        proposals=[{"rule_id": "left-arm-stalled", "arm": "left"}],
    )
    assert decision.accepted is False
    assert "role1 unavailable" in decision.reason
    assert adapter.store.records, "a failed verdict is still an audited verdict"


def test_a_contract_violation_also_rejects(tmp_path: Path) -> None:
    """Malformed output is a rejection, never a silently repaired acceptance."""
    adapter = _adapter(tmp_path, _FakePlanner(text="I think you should recover."))
    decision = adapter.decide(
        task="adjust_bottle",
        chunk_index=0,
        features={},
        proposals=[{"rule_id": "left-arm-stalled", "arm": "left"}],
    )
    assert decision.accepted is False
    assert "Role1ContractError" in decision.reason


def test_fail_closed_off_re_raises(tmp_path: Path) -> None:
    """A campaign that wants failures loud can have them."""
    adapter = _adapter(
        tmp_path, _FakePlanner(raises=RuntimeError("boom")), fail_closed=False
    )
    with pytest.raises(Exception, match="boom"):
        adapter.decide(
            task="adjust_bottle",
            chunk_index=0,
            features={},
            proposals=[{"rule_id": "x", "arm": "left"}],
        )


def test_no_proposal_means_no_model_call(tmp_path: Path) -> None:
    """There is nothing to rule on, so nothing should be spent ruling."""
    planner = _FakePlanner()
    adapter = _adapter(tmp_path, planner)
    decision = adapter.decide(
        task="adjust_bottle", chunk_index=0, features={}, proposals=[]
    )
    assert decision.accepted is False
    assert planner.calls == []


def test_adapter_satisfies_the_actor_protocol(tmp_path: Path) -> None:
    """The Actor must accept it wherever the deterministic reference goes."""
    from robots.robotwin.role1_actor import Role1EpisodeActor

    adapter = _adapter(tmp_path, _FakePlanner(raises=RuntimeError("offline")))
    actor = Role1EpisodeActor(
        decider=adapter, binding=binding_for_task("adjust_bottle")
    )
    result = actor.decide_action(
        task="adjust_bottle",
        chunk_index=0,
        observation={"state": [0.0] * 14},
        vla_actions=[[0.0] * 14],
        critic_proposals=[{"rule_id": "left-arm-stalled", "arm": "left"}],
    )
    assert result.source == "vla"
    assert result.decision is not None and result.decision.accepted is False
