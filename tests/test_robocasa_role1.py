# Copyright (c) 2026 RPent Contributors
from __future__ import annotations

import base64
import hashlib
import io
import json
from dataclasses import FrozenInstanceError, replace
from types import SimpleNamespace
from typing import Any

import pytest
from PIL import Image

from robots.robocasa.role1_agent import (
    ROLE1_SYSTEM_CONTRACT,
    CriticProposal,
    DecisionNotPersistedError,
    Role1ContractError,
    Role1DecisionStore,
    Role1Event,
    Role1ModelAdapter,
    Role1ModelError,
    Role1PersistenceError,
    ToolProposal,
    validate_role1_decision,
)

MOVE = "robocasa.control.move_to"
GRIP = "robocasa.gripper.set"
VLA = "robocasa.vla.groot"


def test_role1_contract_keeps_accept_output_small_and_unambiguous() -> None:
    assert "termination.reason must be the\nempty string" in ROLE1_SYSTEM_CONTRACT
    assert "modifications must be an empty object" in ROLE1_SYSTEM_CONTRACT
    assert "never repeat its action arrays" in ROLE1_SYSTEM_CONTRACT
    assert "must call it on at least one current" in ROLE1_SYSTEM_CONTRACT


class FakePlanner:
    def __init__(
        self,
        output: dict[str, object] | str,
        *,
        error: str | None = None,
        raises: Exception | None = None,
    ) -> None:
        self.output = output
        self.error = error
        self.raises = raises
        self.calls: list[dict[str, Any]] = []
        self.messages: list[dict[str, Any]] = []

    def solve(self, **kwargs: Any) -> SimpleNamespace:
        self.calls.append(kwargs)
        if self.raises is not None:
            raise self.raises
        text = self.output if isinstance(self.output, str) else json.dumps(self.output)
        self.messages = [
            {"role": "user", "content": "unaltered planner-side history"},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "thinking",
                        "thinking": "private model reasoning artifact",
                    },
                    {"type": "text", "text": text},
                ],
            },
        ]
        return SimpleNamespace(
            messages=self.messages,
            stats={"total_input_tokens": 123, "total_output_tokens": 45},
            error=self.error,
        )


def _action() -> dict[str, list[float]]:
    return {
        "end_effector_position": [0.1, 0.0, -0.1],
        "end_effector_rotation": [0.0, 0.0, 0.0],
        "gripper_close": [1.0],
        "base_motion": [0.0, 0.0, 0.0, 0.0],
        "control_mode": [0.0],
    }


def _event(*, critic: bool = False) -> Role1Event:
    tool = ToolProposal(
        proposal_id="tool-p1",
        tool=MOVE,
        proposal={"action": _action(), "status": "candidate"},
        evidence=("camera:left@step-4",),
    )
    critics = (
        (
            CriticProposal(
                proposal_id="critic-p1",
                reject_current_action=True,
                reason="contact progress stopped",
                evidence=("state-window:3-4",),
            ),
        )
        if critic
        else ()
    )
    return Role1Event(
        event_id="event-4",
        task="SlideDishwasherRack",
        step_index=4,
        current_stage="engage",
        current_tool=MOVE,
        allowed_stages=("approach", "engage", "pull", "verify"),
        allowed_tools=(MOVE, GRIP, VLA),
        image_references={"left": "artifact://episode/camera-left/0004.jpg"},
        task_state={"phase": "contact", "progress": 0.31},
        tool_proposals=(tool,),
        critic_proposals=critics,
        history=({"decision_id": "role1-prior", "outcome": "observed"},),
    )


def _decision(**updates: object) -> dict[str, object]:
    value: dict[str, object] = {
        "proposal_disposition": "accept",
        "action_kind": "continue",
        "selected_stage": "engage",
        "selected_tool": MOVE,
        "direct_action": None,
        "termination": {"approved": False, "reason": ""},
        "evidence": ["tool-p1", "camera:left@step-4"],
        "confidence": 0.75,
        "rationale": "The proposed bounded motion remains consistent with current evidence.",
        "proposal_ids": ["tool-p1"],
        "modifications": {},
    }
    value.update(updates)
    return value


def _model_decision(**updates: object) -> dict[str, object]:
    return _decision(event_id="event-4", **updates)


def test_role1_collapses_exact_duplicate_final_text_from_planner_transport(
    tmp_path,
) -> None:
    decision = _model_decision()
    text = json.dumps(decision)

    class DuplicateFinalPlanner(FakePlanner):
        def solve(self, **kwargs: Any) -> SimpleNamespace:
            self.calls.append(kwargs)
            self.messages = [
                {"role": "user", "content": "planner input"},
                {"role": "assistant", "content": [{"type": "text", "text": text}]},
                {"role": "assistant", "content": text},
            ]
            return SimpleNamespace(messages=self.messages, stats={}, error=None)

    result = Role1ModelAdapter(
        store=Role1DecisionStore(tmp_path / "decisions"),
        output_root=tmp_path / "invocations",
        planner=DuplicateFinalPlanner(decision),
    ).decide(_event())
    assert result.event_id == "event-4"


def test_role1_still_rejects_distinct_multiple_model_outputs(tmp_path) -> None:
    decision = _model_decision()

    class MultipleOutputPlanner(FakePlanner):
        def solve(self, **kwargs: Any) -> SimpleNamespace:
            self.calls.append(kwargs)
            self.messages = [
                {"role": "assistant", "content": json.dumps(decision)},
                {"role": "assistant", "content": json.dumps({"other": True})},
            ]
            return SimpleNamespace(messages=self.messages, stats={}, error=None)

    adapter = Role1ModelAdapter(
        store=Role1DecisionStore(tmp_path / "decisions"),
        output_root=tmp_path / "invocations",
        planner=MultipleOutputPlanner(decision),
    )
    with pytest.raises(Role1ModelError) as error:
        adapter.decide(_event())
    assert isinstance(error.value.__cause__, Role1ContractError)


def test_event_supports_multimodal_state_proposals_and_history() -> None:
    event = _event(critic=True)
    payload = event.to_dict()
    assert payload["image_references"]["left"].startswith("artifact://")
    assert payload["task_state"]["progress"] == 0.31
    assert payload["tool_proposals"][0]["proposal_only"] is True
    assert payload["tool_proposals"][0]["environment_write"] is False
    assert (
        payload["critic_proposals"][0]["action_authority"]
        == "reject_current_action_only"
    )
    assert payload["history"][0]["decision_id"] == "role1-prior"
    with pytest.raises(FrozenInstanceError):
        event.step_index = 5  # type: ignore[misc]


def test_role1_is_the_only_high_level_authority() -> None:
    event = _event(critic=True)
    decision = validate_role1_decision(
        _decision(
            proposal_disposition="modify",
            action_kind="recover",
            selected_stage="engage",
            selected_tool=GRIP,
            proposal_ids=["tool-p1", "critic-p1"],
            modifications={"reason": "replace the rejected current motion"},
        ),
        event=event,
    )
    payload = decision.to_dict()
    assert payload["authority"] == "role1_high_level_agent"
    assert payload["tool_authority"] == "proposal_only"
    assert payload["critic_action_authority"] == "reject_current_action_only"
    assert payload["environment_write"] is False
    assert decision.action_kind == "recover"


def test_critic_cannot_execute_replace_or_terminate() -> None:
    with pytest.raises(Role1ContractError, match="cannot claim"):
        CriticProposal.from_payload(
            {
                "proposal_id": "critic-bad",
                "reject_current_action": True,
                "reason": "unsafe",
                "evidence": ["collision"],
                "replacement_action": _action(),
            }
        )
    with pytest.raises(Role1ContractError, match="authority"):
        CriticProposal.from_payload(
            {
                "proposal_id": "critic-bad",
                "reject_current_action": True,
                "reason": "unsafe",
                "evidence": ["collision"],
                "action_authority": "execute",
            }
        )


def test_tool_cannot_claim_environment_write_or_termination() -> None:
    with pytest.raises(Role1ContractError, match="cannot write"):
        ToolProposal.from_payload(
            {
                "proposal_id": "tool-bad",
                "tool": MOVE,
                "proposal_only": True,
                "environment_write": True,
                "proposal": {"action": _action()},
            }
        )
    with pytest.raises(Role1ContractError, match="cannot claim"):
        ToolProposal(
            proposal_id="tool-bad",
            tool=MOVE,
            proposal={"termination": {"approved": True}},
            evidence=("bad",),
        )


def test_critic_rejection_requires_explicit_role1_disposition() -> None:
    event = _event(critic=True)
    with pytest.raises(Role1ContractError, match="critic rejection"):
        validate_role1_decision(
            _decision(proposal_ids=["tool-p1", "critic-p1"]),
            event=event,
        )
    with pytest.raises(Role1ContractError, match="executable selected_tool"):
        validate_role1_decision(
            _decision(
                proposal_disposition="accept",
                action_kind="restage",
                selected_stage="approach",
                selected_tool=None,
                proposal_ids=["critic-p1"],
            ),
            event=event,
        )
    accepted_rejection = validate_role1_decision(
        _decision(
            proposal_disposition="accept",
            action_kind="restage",
            selected_stage="approach",
            selected_tool=GRIP,
            proposal_ids=["critic-p1"],
        ),
        event=event,
    )
    assert accepted_rejection.action_kind == "restage"


def test_decision_schema_fails_closed_for_unknown_fields_and_tools() -> None:
    event = _event()
    with pytest.raises(Role1ContractError, match="unknown Role1 decision fields"):
        validate_role1_decision(_decision(run_now=True), event=event)
    with pytest.raises(Role1ContractError, match="allowlist"):
        validate_role1_decision(
            _decision(action_kind="switch", selected_tool="robocasa.missing"),
            event=event,
        )
    unknown_allowlist = Role1Event(
        event_id="event-unknown-tool",
        task="SlideDishwasherRack",
        step_index=0,
        current_stage="approach",
        current_tool=None,
        allowed_stages=("approach",),
        allowed_tools=("robocasa.missing",),
    )
    with pytest.raises(Role1ContractError, match="unknown tools"):
        Role1DecisionStore("unused").prepare(
            unknown_allowlist,
            _decision(
                proposal_disposition="modify",
                action_kind="regenerate",
                selected_stage="approach",
                selected_tool="robocasa.missing",
                proposal_ids=[],
                modifications={"request": "new proposal"},
            ),
        )


@pytest.mark.parametrize(
    "leak",
    [
        {"environment_seed": 12},
        {"policy_rng": 99},
        {"nested": {"future_schedule": [1, 2]}},
        {"note": "retry seed=17 next"},
    ],
)
def test_event_is_seed_and_future_schedule_blind(leak: dict[str, object]) -> None:
    with pytest.raises(Role1ContractError, match="seed|RNG|schedule"):
        Role1Event(
            event_id="event-blind",
            task="SlideDishwasherRack",
            step_index=0,
            current_stage="approach",
            current_tool=None,
            allowed_stages=("approach",),
            allowed_tools=(MOVE,),
            task_state=leak,
        )


def test_decision_is_seed_blind_too() -> None:
    with pytest.raises(Role1ContractError, match="seed"):
        validate_role1_decision(
            _decision(rationale="Use the successful policy seed-3 behavior."),
            event=_event(),
        )


def test_direct_action_is_exact_canonical_and_bounded() -> None:
    event = _event()
    decision = validate_role1_decision(
        _decision(
            proposal_disposition="modify",
            action_kind="replace",
            selected_tool=None,
            direct_action=_action(),
            modifications={"action": "bounded replacement selected by Role1"},
        ),
        event=event,
    )
    assert set(decision.direct_action or {}) == {
        "action.end_effector_position",
        "action.end_effector_rotation",
        "action.gripper_close",
        "action.base_motion",
        "action.control_mode",
    }
    invalid = _action()
    invalid["end_effector_position"] = [1.01, 0.0, 0.0]
    with pytest.raises(Role1ContractError, match="invalid direct_action"):
        validate_role1_decision(
            _decision(
                proposal_disposition="modify",
                action_kind="replace",
                selected_tool=None,
                direct_action=invalid,
                modifications={"action": "bad"},
            ),
            event=event,
        )


def test_termination_requires_explicit_role1_approval_and_reason() -> None:
    event = _event()
    with pytest.raises(Role1ContractError, match="terminate requires explicit"):
        validate_role1_decision(
            _decision(
                proposal_disposition="reject",
                action_kind="terminate",
                selected_tool=None,
            ),
            event=event,
        )
    decision = validate_role1_decision(
        _decision(
            proposal_disposition="reject",
            action_kind="terminate",
            selected_stage=None,
            selected_tool=None,
            termination={
                "approved": True,
                "reason": "authoritative terminal state observed",
            },
        ),
        event=event,
    )
    assert decision.termination.approved is True


def test_prepared_decision_is_inert_until_immutable_persistence(tmp_path) -> None:
    store = Role1DecisionStore(tmp_path / "decisions")
    pending = store.prepare(_event(), _decision())
    with pytest.raises(DecisionNotPersistedError):
        pending.activate()
    assert not hasattr(store, "step")
    assert not hasattr(pending, "execute")
    persisted = store.persist(pending)
    effect = store.activate(persisted)
    assert effect.decision_id == pending.decision.decision_id
    assert effect.selected_tool == MOVE
    assert effect.persisted_digest == pending.digest
    artifact = json.loads(persisted.path.read_text(encoding="utf-8"))
    assert artifact["authority"] == "role1_high_level_agent"
    assert artifact["decision"]["environment_write"] is False


def test_decision_id_is_deterministic_and_persistence_is_idempotent(tmp_path) -> None:
    store = Role1DecisionStore(tmp_path)
    first = store.prepare(_event(), _decision())
    second = store.prepare(_event(), _decision())
    assert first.decision.decision_id == second.decision.decision_id
    assert first.digest == second.digest
    assert store.persist(first).digest == store.persist(second).digest
    changed = store.prepare(
        _event(),
        _decision(
            decision_id=first.decision.decision_id,
            rationale="A different decision must not reuse an immutable identity.",
        ),
    )
    with pytest.raises(Role1PersistenceError, match="different content"):
        store.persist(changed)


def test_model_supplied_decision_id_cannot_escape_the_store(tmp_path) -> None:
    adapter = Role1ModelAdapter(
        store=Role1DecisionStore(tmp_path / "decisions"),
        output_root=tmp_path / "invocations",
        planner=FakePlanner(_model_decision(decision_id="../../outside")),
    )
    with pytest.raises(Role1ModelError, match="prepare"):
        adapter.decide(_event())
    assert not (tmp_path / "outside.json").exists()


def test_activation_revalidates_disk_and_fails_closed_on_tampering(tmp_path) -> None:
    store = Role1DecisionStore(tmp_path)
    persisted = store.persist(store.prepare(_event(), _decision()))
    artifact = json.loads(persisted.path.read_text(encoding="utf-8"))
    artifact["decision"]["selected_tool"] = "robocasa.missing"
    persisted.path.write_text(json.dumps(artifact), encoding="utf-8")
    with pytest.raises(Role1PersistenceError):
        store.activate(persisted)


def test_persisted_event_and_decision_round_trip(tmp_path) -> None:
    store = Role1DecisionStore(tmp_path)
    pending = store.prepare(
        _event(critic=True),
        _decision(
            proposal_disposition="reject",
            action_kind="restage",
            selected_stage="approach",
            selected_tool=GRIP,
            proposal_ids=["tool-p1", "critic-p1"],
        ),
    )
    persisted = store.persist(pending)
    loaded = store.load(persisted.decision.decision_id)
    assert loaded.event.to_dict() == pending.event.to_dict()
    assert loaded.decision.to_dict() == pending.decision.to_dict()


def test_model_adapter_preserves_full_messages_and_returns_persisted_effect(
    tmp_path,
) -> None:
    planner = FakePlanner(_model_decision())
    store = Role1DecisionStore(tmp_path / "decisions")
    adapter = Role1ModelAdapter(
        store=store,
        output_root=tmp_path / "invocations",
        planner=planner,
        toolkit_mode="empty",
    )
    effect = adapter.decide(_event())
    assert effect.selected_tool == MOVE
    assert list((tmp_path / "decisions").glob("*.json"))
    invocations = list((tmp_path / "invocations").glob("invocation-*"))
    assert len(invocations) == 1
    invocation = invocations[0]
    assert (
        json.loads((invocation / "planner_messages.json").read_text())
        == planner.messages
    )
    assert json.loads((invocation / "planner_stats.json").read_text()) == {
        "total_input_tokens": 123,
        "total_output_tokens": 45,
    }
    assert json.loads((invocation / "planner_error.json").read_text()) == {
        "error": None
    }
    timing = json.loads((invocation / "planner_timing.json").read_text())
    assert timing["phase"] == "model_inference"
    assert timing["status"] == "completed"
    assert timing["elapsed_s"] >= 0
    assert timing["finished_at_unix_s"] >= timing["started_at_unix_s"]
    assert (
        json.loads((invocation / "completion.json").read_text())[
            "activated_after_persistence"
        ]
        is True
    )
    call = planner.calls[0]
    assert call["toolkit"].get_tools_spec() == []
    assert "future_schedule" not in call["user_message"]
    assert "environment_seed" not in call["user_message"]


def test_role1_formal_model_contract_is_attested(tmp_path) -> None:
    planner = FakePlanner(_model_decision())
    original_solve = planner.solve

    def solve(**kwargs: Any) -> SimpleNamespace:
        result = original_solve(**kwargs)
        result.stats.update({"model": "openai:gpt-5.6-sol", "reasoning_effort": "high"})
        return result

    planner.solve = solve  # type: ignore[method-assign]
    root = tmp_path / "formal-role1"
    Role1ModelAdapter(
        store=Role1DecisionStore(root / "decisions"),
        output_root=root / "invocations",
        model="openai:gpt-5.6-sol",
        reasoning_effort="high",
        planner=planner,
    ).decide(_event())
    invocation = next((root / "invocations").glob("invocation-*"))
    frozen = json.loads((invocation / "input.json").read_text())
    assert frozen["model"] == "openai:gpt-5.6-sol"
    assert frozen["reasoning_effort"] == "high"


def test_role1_adapter_exposes_verified_current_image_by_opaque_reference(
    tmp_path,
) -> None:
    buffer = io.BytesIO()
    Image.new("RGB", (1, 1), (255, 0, 0)).save(buffer, format="PNG")
    image = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()
    reference = "sha256:" + hashlib.sha256(image.encode()).hexdigest()
    payload = _event().to_dict()
    payload["image_references"] = {"left": reference}
    event = Role1Event.from_payload(payload)
    adapter = Role1ModelAdapter(
        store=Role1DecisionStore(tmp_path / "decisions"),
        output_root=tmp_path / "invocations",
        planner=FakePlanner(_model_decision()),
    )
    toolkit = adapter._toolkit(event=event, image_payloads={"left": image})
    result = toolkit.execute_tool("read_role1_image", {"reference": reference})
    assert result.result["kind"] == "current_observation_image"
    assert any(block["type"] == "image" for block in result.content_blocks)
    assert image not in json.dumps(result.result, default=str)
    with pytest.raises(Role1ModelError, match="integrity"):
        adapter._toolkit(
            event=event,
            image_payloads={"left": "data:image/png;base64,AAAA"},
        )


def test_formal_role1_fails_closed_without_actual_visual_review(tmp_path) -> None:
    buffer = io.BytesIO()
    Image.new("RGB", (1, 1), (255, 0, 0)).save(buffer, format="PNG")
    image = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()
    reference = "sha256:" + hashlib.sha256(image.encode()).hexdigest()
    payload = _event().to_dict()
    payload["image_references"] = {"left": reference}
    event = Role1Event.from_payload(payload)
    adapter = Role1ModelAdapter(
        store=Role1DecisionStore(tmp_path / "decisions"),
        output_root=tmp_path / "invocations",
        planner=FakePlanner(_model_decision()),
        require_visual_review=True,
    )
    with pytest.raises(Role1ModelError) as error:
        adapter.decide(event, image_payloads={"left": image})
    assert isinstance(error.value.__cause__, Role1ContractError)
    invocation = next((tmp_path / "invocations").glob("invocation-*"))
    review = json.loads((invocation / "visual_review.json").read_text())
    assert review == {
        "available_cameras": ["left"],
        "completed": False,
        "required": True,
        "viewed_cameras": [],
        "viewed_references": [],
    }


def test_formal_role1_records_camera_read_before_decision(tmp_path) -> None:
    buffer = io.BytesIO()
    Image.new("RGB", (1, 1), (0, 255, 0)).save(buffer, format="PNG")
    image = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()
    reference = "sha256:" + hashlib.sha256(image.encode()).hexdigest()
    payload = _event().to_dict()
    payload["image_references"] = {"left": reference}
    event = Role1Event.from_payload(payload)

    class VisualPlanner(FakePlanner):
        def solve(self, **kwargs: Any) -> SimpleNamespace:
            result = kwargs["toolkit"].execute_tool(
                "read_role1_image", {"reference": reference}
            )
            assert any(block["type"] == "image" for block in result.content_blocks)
            return super().solve(**kwargs)

    adapter = Role1ModelAdapter(
        store=Role1DecisionStore(tmp_path / "decisions"),
        output_root=tmp_path / "invocations",
        planner=VisualPlanner(_model_decision()),
        require_visual_review=True,
    )
    adapter.decide(event, image_payloads={"left": image})
    invocation = next((tmp_path / "invocations").glob("invocation-*"))
    review = json.loads((invocation / "visual_review.json").read_text())
    assert review["required"] is True
    assert review["completed"] is True
    assert review["viewed_cameras"] == ["left"]
    assert review["viewed_references"] == [reference]


def test_read_only_model_toolkit_exposes_no_mutating_tool(tmp_path) -> None:
    planner = FakePlanner(_model_decision())
    adapter = Role1ModelAdapter(
        store=Role1DecisionStore(tmp_path / "decisions"),
        output_root=tmp_path / "invocations",
        planner=planner,
        toolkit_mode="read_only",
    )
    adapter(_event())
    names = {item["name"] for item in planner.calls[0]["toolkit"].get_tools_spec()}
    assert names == {"describe_tools"}
    assert "write_text_file" not in names
    assert "finish" not in names


def test_model_error_is_audited_and_does_not_persist_a_decision(tmp_path) -> None:
    planner = FakePlanner(_model_decision(), error="CapacityUnavailable")
    decision_root = tmp_path / "decisions"
    invocation_root = tmp_path / "invocations"
    adapter = Role1ModelAdapter(
        store=Role1DecisionStore(decision_root),
        output_root=invocation_root,
        planner=planner,
    )
    with pytest.raises(Role1ModelError, match="planner failed"):
        adapter.decide(_event())
    assert not decision_root.exists() or not list(decision_root.glob("*.json"))
    invocation = next(invocation_root.glob("invocation-*"))
    assert (
        json.loads((invocation / "planner_messages.json").read_text())
        == planner.messages
    )
    assert json.loads((invocation / "planner_error.json").read_text()) == {
        "error": "CapacityUnavailable"
    }
    assert json.loads((invocation / "failure.json").read_text())["phase"] == "planner"
    assert not (invocation / "model_output.json").exists()
    assert not (invocation / "completion.json").exists()


def test_planner_exception_is_audited_and_fails_closed(tmp_path) -> None:
    planner = FakePlanner(_model_decision(), raises=TimeoutError("API timed out"))
    adapter = Role1ModelAdapter(
        store=Role1DecisionStore(tmp_path / "decisions"),
        output_root=tmp_path / "invocations",
        planner=planner,
    )
    with pytest.raises(Role1ModelError, match="planner"):
        adapter.decide(_event())
    invocation = next((tmp_path / "invocations").glob("invocation-*"))
    error = json.loads((invocation / "planner_error.json").read_text())["error"]
    assert "TimeoutError" in error
    assert "API timed out" in error
    assert not (tmp_path / "decisions").exists()


@pytest.mark.parametrize(
    "raw_text",
    [
        "```json\n{}\n```",
        '{"first": 1}\n{"second": 2}',
        '{"event_id": "event-4", "event_id": "event-4"}',
    ],
)
def test_model_output_must_be_exactly_one_strict_json_object(
    tmp_path, raw_text: str
) -> None:
    adapter = Role1ModelAdapter(
        store=Role1DecisionStore(tmp_path / "decisions"),
        output_root=tmp_path / "invocations",
        planner=FakePlanner(raw_text),
    )
    with pytest.raises(Role1ModelError, match="model_output"):
        adapter.decide(_event())
    assert not (tmp_path / "decisions").exists()


def test_model_adapter_never_fills_an_omitted_tool(tmp_path) -> None:
    raw = _model_decision(
        action_kind="switch",
        selected_tool=None,
        proposal_disposition="modify",
        modifications={"intent": "switch without selecting a target"},
    )
    adapter = Role1ModelAdapter(
        store=Role1DecisionStore(tmp_path / "decisions"),
        output_root=tmp_path / "invocations",
        planner=FakePlanner(raw),
    )
    with pytest.raises(Role1ModelError, match="prepare"):
        adapter.decide(_event())
    assert not (tmp_path / "decisions").exists()


def test_model_adapter_persists_before_activation(tmp_path) -> None:
    order: list[str] = []

    class OrderedStore(Role1DecisionStore):
        def prepare(self, event, raw):  # type: ignore[no-untyped-def]
            order.append("prepare")
            return super().prepare(event, raw)

        def persist(self, pending):  # type: ignore[no-untyped-def]
            order.append("persist")
            persisted = super().persist(pending)
            assert persisted.path.is_file()
            return persisted

        def activate(self, persisted):  # type: ignore[no-untyped-def]
            order.append("activate")
            assert persisted.path.is_file()
            return super().activate(persisted)

    adapter = Role1ModelAdapter(
        store=OrderedStore(tmp_path / "decisions"),
        output_root=tmp_path / "invocations",
        planner=FakePlanner(_model_decision()),
    )
    adapter.decide(_event())
    assert order == ["prepare", "persist", "activate"]


def test_injected_planner_factory_receives_api_configuration(tmp_path) -> None:
    planner = FakePlanner(_model_decision())
    calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def factory(*args: Any, **kwargs: Any) -> FakePlanner:
        calls.append((args, kwargs))
        return planner

    adapter = Role1ModelAdapter(
        store=Role1DecisionStore(tmp_path / "decisions"),
        output_root=tmp_path / "invocations",
        planner_type="api",
        model="openai:gpt-test",
        base_url="http://provider.invalid/v1",
        planner_factory=factory,
    )
    adapter.decide(_event())
    args, kwargs = calls[0]
    assert args == ("api",)
    assert kwargs["model"] == "openai:gpt-test"
    assert kwargs["base_url"] == "http://provider.invalid/v1"


def test_role1_may_select_read_only_observation_tool() -> None:
    observation_tool = "robocasa.observation.view_driver_state"
    event = replace(
        _event(),
        allowed_tools=(MOVE, observation_tool),
    )
    decision = validate_role1_decision(
        _decision(
            action_kind="switch",
            selected_tool=observation_tool,
        ),
        event=event,
    )
    assert decision.selected_tool == observation_tool
