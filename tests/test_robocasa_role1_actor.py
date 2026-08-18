# Copyright (c) 2026 RPent Contributors
from __future__ import annotations

import json
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from robots.robocasa import run_rollout
from robots.robocasa.role1_actor import Role1ActorError, Role1EpisodeActor
from robots.robocasa.role1_agent import (
    Role1ContractError,
    Role1DecisionStore,
    Role1ModelError,
)
from robots.robocasa.run_rollout import (
    _advance_recovery_after_chunk,
    _is_role1_method_failure,
    _role1_artifact_index,
    _role1_inference_heartbeat,
)
from robots.robocasa.slide_dishwasher_program import (
    CONTACT_PUSH_TOOL,
    GUARDED_SUFFIX_TOOL,
)
from robots.robocasa.tool_bindings import binding_for_task


def _decision(event: Any, **updates: Any) -> dict[str, Any]:
    value = {
        "event_id": event.event_id,
        "proposal_disposition": "accept",
        "action_kind": "continue",
        "selected_stage": event.current_stage,
        "selected_tool": event.current_tool,
        "direct_action": None,
        "termination": {"approved": False, "reason": ""},
        "evidence": [event.tool_proposals[0].proposal_id],
        "confidence": 0.8,
        "rationale": "Current proposal is supported by the audited observation.",
        "proposal_ids": list(event.proposal_ids),
        "modifications": {},
    }
    value.update(updates)
    return value


def test_role1_inference_heartbeat_reports_live_reasoning(tmp_path: Path) -> None:
    heartbeat = tmp_path / "heartbeat.jsonl"
    with _role1_inference_heartbeat(
        heartbeat,
        interval_s=0.01,
        step_index=17,
    ):
        time.sleep(0.035)
    rows = [
        json.loads(line)
        for line in heartbeat.read_text(encoding="utf-8").splitlines()
    ]
    assert len(rows) >= 2
    assert {row["phase"] for row in rows} == {"role1_inference"}
    assert {row["step_index"] for row in rows} == {17}

    actor_heartbeat = tmp_path / "actor-heartbeat.jsonl"
    with _role1_inference_heartbeat(
        actor_heartbeat,
        interval_s=0.01,
        step_index=18,
        phase="role1_actor",
    ):
        time.sleep(0.025)
    actor_rows = [
        json.loads(line)
        for line in actor_heartbeat.read_text(encoding="utf-8").splitlines()
    ]
    assert {row["phase"] for row in actor_rows} == {"role1_actor"}
    assert {row["step_index"] for row in actor_rows} == {18}


def test_interrupted_recovery_chunk_does_not_advance_frozen_step(
    tmp_path: Path,
) -> None:
    class Controller:
        calls: list[dict[str, Any]] = []

        def complete_current_step(self, **values: Any) -> None:
            self.calls.append(values)

    controller = Controller()
    tools = tmp_path / "tools.jsonl"
    advanced = _advance_recovery_after_chunk(
        recovery_controller=controller,  # type: ignore[arg-type]
        selected_tool=CONTACT_PUSH_TOOL,
        environment_step=11,
        result={
            "executed_horizon": 1,
            "critic_proposals": [{"rule_id": "critic-repeat"}],
        },
        tools_path=tools,
    )
    assert advanced is False
    assert controller.calls == []
    audit = json.loads(tools.read_text(encoding="utf-8"))
    assert audit["recovery_advanced"] is False

    advanced = _advance_recovery_after_chunk(
        recovery_controller=controller,  # type: ignore[arg-type]
        selected_tool=CONTACT_PUSH_TOOL,
        environment_step=14,
        result={"executed_horizon": 3, "critic_proposals": []},
        tools_path=tools,
    )
    assert advanced is True
    assert controller.calls == [
        {
            "selected_tool": CONTACT_PUSH_TOOL,
            "environment_step": 14,
            "executed_horizon": 3,
        }
    ]


class FakeAdapter:
    def __init__(
        self, store: Role1DecisionStore, decisions: list[dict[str, Any]]
    ) -> None:
        self.store = store
        self.decisions = decisions
        self.events: list[Any] = []
        self.image_payloads: list[dict[str, str]] = []

    def decide(
        self, event: Any, *, image_payloads: dict[str, str] | None = None
    ) -> Any:
        self.events.append(event)
        self.image_payloads.append(dict(image_payloads or {}))
        updates = self.decisions.pop(0)
        pending = self.store.prepare(event, _decision(event, **updates))
        return self.store.activate(self.store.persist(pending))


class FakeToolRuntime:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def invoke(
        self, name: str, payload: dict[str, Any], *, policy: Any
    ) -> dict[str, Any]:
        self.calls.append((name, payload))
        return {
            "action": {
                "end_effector_position": [0.2, 0.0, 0.0],
                "end_effector_rotation": [0.0, 0.0, 0.0],
                "gripper_close": [1.0],
                "base_motion": [0.0, 0.0, 0.0, 0.0],
                "control_mode": [0.0],
            },
            "proposal_only": True,
            "environment_write": False,
        }


def _observation() -> dict[str, Any]:
    return {
        "observation": {
            "state": {"progress": 0.2, "policy_rng": 999},
            "images": {"left": "data:image/png;base64,AAAA"},
        }
    }


def test_actor_accepts_vla_only_after_persisted_role1_decision(tmp_path: Path) -> None:
    store = Role1DecisionStore(tmp_path / "decisions")
    adapter = FakeAdapter(store, [{}])
    runtime = FakeToolRuntime()
    actor = Role1EpisodeActor(
        adapter=adapter,  # type: ignore[arg-type]
        decision_store=store,
        tool_runtime=runtime,  # type: ignore[arg-type]
        binding=binding_for_task("SlideDishwasherRack"),
        audit_root=tmp_path / "audit",
    )
    actions = [[0.0] * 12, [0.1] * 12]
    result = actor.decide_action(
        task="SlideDishwasherRack",
        step_index=0,
        observation_response=_observation(),
        vla_actions=actions,
        vla_metadata={"inference_seed": 123, "action_chunk_sha256": "a" * 64},
    )
    assert result.actions == tuple(actions)
    assert len(result.decision_ids) == 1
    assert adapter.image_payloads == [{"left": "data:image/png;base64,AAAA"}]
    assert runtime.calls == []
    event = adapter.events[0].to_dict()
    assert "policy_rng" not in event["task_state"]
    assert "inference_seed" not in event["tool_proposals"][0]["proposal"]
    assert list((tmp_path / "decisions").glob("*.json"))


def test_actor_uses_global_step_boundary_for_tool_proposal_artifacts(
    tmp_path: Path,
) -> None:
    store = Role1DecisionStore(tmp_path / "decisions")
    adapter = FakeAdapter(store, [{}, {}])
    actor = Role1EpisodeActor(
        adapter=adapter,  # type: ignore[arg-type]
        decision_store=store,
        tool_runtime=FakeToolRuntime(),  # type: ignore[arg-type]
        binding=binding_for_task("SlideDishwasherRack"),
        audit_root=tmp_path / "audit",
    )
    for step_index in (0, 8):
        actor.decide_action(
            task="SlideDishwasherRack",
            step_index=step_index,
            observation_response=_observation(),
            vla_actions=[[0.0] * 12],
            vla_metadata={"action_chunk_sha256": "d" * 64},
        )
    paths = sorted((tmp_path / "audit" / "tool_proposals").glob("*.json"))
    assert [path.name for path in paths] == [
        "step-000000-event-00.json",
        "step-000008-event-00.json",
    ]


def test_role1_artifact_index_is_flat_complete_and_stable(tmp_path: Path) -> None:
    role1 = tmp_path / "role1"
    first = role1 / "actor" / "tool_proposals" / "step-000000-event-00.json"
    second = role1 / "invocations" / "invocation-0000" / "planner_messages.json"
    first.parent.mkdir(parents=True)
    second.parent.mkdir(parents=True)
    first.write_text("{}", encoding="utf-8")
    second.write_text("[]", encoding="utf-8")

    assert _role1_artifact_index(role1) == {
        "role1:actor/tool_proposals/step-000000-event-00.json": str(first),
        "role1:invocations/invocation-0000/planner_messages.json": str(second),
    }


def test_tool_action_requires_a_second_persisted_role1_approval(tmp_path: Path) -> None:
    store = Role1DecisionStore(tmp_path / "decisions")
    adapter = FakeAdapter(
        store,
        [
            {
                "proposal_disposition": "modify",
                "action_kind": "switch",
                "selected_tool": GUARDED_SUFFIX_TOOL,
                "modifications": {
                    "parameters": {
                        "current_position": [0.0, 0.0, 0.0],
                        "target_position": [0.2, 0.0, 0.0],
                    }
                },
            },
            {},
        ],
    )
    runtime = FakeToolRuntime()
    actor = Role1EpisodeActor(
        adapter=adapter,  # type: ignore[arg-type]
        decision_store=store,
        tool_runtime=runtime,  # type: ignore[arg-type]
        binding=binding_for_task("SlideDishwasherRack"),
        audit_root=tmp_path / "audit",
    )
    result = actor.decide_action(
        task="SlideDishwasherRack",
        step_index=3,
        observation_response=_observation(),
        vla_actions=[[0.0] * 12],
        vla_metadata={"action_chunk_sha256": "b" * 64},
        recovery_suggestions=(
            {
                "recovery_id": "guarded-terminal",
                "steps": [{"tool": GUARDED_SUFFIX_TOOL, "parameters": {}}],
            },
        ),
    )
    assert len(result.decision_ids) == 2
    assert result.selected_tool == GUARDED_SUFFIX_TOOL
    assert len(result.actions) == 1
    assert len(runtime.calls) == 1
    assert len(list((tmp_path / "decisions").glob("*.json"))) == 2


def test_candidate_only_tool_is_hidden_without_triggered_recovery(
    tmp_path: Path,
) -> None:
    store = Role1DecisionStore(tmp_path / "decisions")
    adapter = FakeAdapter(
        store,
        [
            {
                "proposal_disposition": "modify",
                "action_kind": "switch",
                "selected_tool": GUARDED_SUFFIX_TOOL,
            }
        ],
    )
    actor = Role1EpisodeActor(
        adapter=adapter,  # type: ignore[arg-type]
        decision_store=store,
        tool_runtime=FakeToolRuntime(),  # type: ignore[arg-type]
        binding=binding_for_task("SlideDishwasherRack"),
        audit_root=tmp_path / "audit",
    )
    with pytest.raises(Role1ContractError, match="outside the event allowlist"):
        actor.decide_action(
            task="SlideDishwasherRack",
            step_index=3,
            observation_response=_observation(),
            vla_actions=[[0.0] * 12],
            vla_metadata={"action_chunk_sha256": "e" * 64},
        )
    assert adapter.events[0].allowed_tools == (CONTACT_PUSH_TOOL,)


@pytest.mark.parametrize(
    "decision,error",
    [
        (
            {
                "action_kind": "terminate",
                "selected_tool": None,
                "termination": {"approved": True, "reason": "stop now"},
            },
            "cannot terminate",
        ),
        (
            {
                "proposal_disposition": "modify",
                "action_kind": "recover",
                "selected_tool": None,
                "direct_action": {
                    "end_effector_position": [0.0, 0.0, 0.0],
                    "end_effector_rotation": [0.0, 0.0, 0.0],
                    "gripper_close": [1.0],
                    "base_motion": [0.0, 0.0, 0.0, 0.0],
                    "control_mode": [0.0],
                },
            },
            "direct_action cannot bypass",
        ),
    ],
)
def test_active_recovery_cannot_be_bypassed(
    tmp_path: Path, decision: dict[str, Any], error: str
) -> None:
    store = Role1DecisionStore(tmp_path / "decisions")
    adapter = FakeAdapter(store, [decision])
    actor = Role1EpisodeActor(
        adapter=adapter,  # type: ignore[arg-type]
        decision_store=store,
        tool_runtime=FakeToolRuntime(),  # type: ignore[arg-type]
        binding=binding_for_task("SlideDishwasherRack"),
        audit_root=tmp_path / "audit",
    )
    recovery = {
        "recovery_id": "guarded-terminal",
        "steps": [{"tool": GUARDED_SUFFIX_TOOL, "parameters": {}}],
    }
    with pytest.raises(Role1ActorError, match=error):
        actor.decide_action(
            task="SlideDishwasherRack",
            step_index=3,
            observation_response=_observation(),
            vla_actions=[[0.0] * 12],
            vla_metadata={"action_chunk_sha256": "f" * 64},
            recovery_suggestions=(recovery,),
            active_recovery={
                "recovery_id": "guarded-terminal",
                "current_step": {"tool": GUARDED_SUFFIX_TOOL, "parameters": {}},
            },
        )


def test_actor_never_executes_a_rejected_vla_chunk(tmp_path: Path) -> None:
    store = Role1DecisionStore(tmp_path / "decisions")
    adapter = FakeAdapter(
        store,
        [
            {
                "proposal_disposition": "reject",
                "action_kind": "regenerate",
                "selected_tool": CONTACT_PUSH_TOOL,
                "modifications": {"instruction": "try a different contact strategy"},
            }
        ],
    )
    actor = Role1EpisodeActor(
        adapter=adapter,  # type: ignore[arg-type]
        decision_store=store,
        tool_runtime=FakeToolRuntime(),  # type: ignore[arg-type]
        binding=binding_for_task("SlideDishwasherRack"),
        audit_root=tmp_path / "audit",
    )
    with pytest.raises(Role1ActorError, match="rejected or modified"):
        actor.decide_action(
            task="SlideDishwasherRack",
            step_index=0,
            observation_response=_observation(),
            vla_actions=[[0.0] * 12],
            vla_metadata={"action_chunk_sha256": "c" * 64},
        )


def test_strict_gen0_skips_role1_and_records_normal_termination_as_valid_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class FakeVla:
        def __init__(self, *_: Any, **__: Any) -> None:
            pass

        def act(self, *_: Any, **__: Any) -> tuple[list[list[float]], dict[str, Any]]:
            return [[0.0] * 12], {"action_chunk_sha256": "a" * 64}

    class FakeAdapterForRollout:
        def __init__(self, *_: Any, **__: Any) -> None:
            pass

    class FailingActor:
        def __init__(self, *_: Any, **__: Any) -> None:
            raise AssertionError("strict Gen0 must not initialize Role1")

        def decide_action(self, **_: Any) -> Any:
            raise AssertionError("Role1 must not inspect Gen0 merely because chunks == 0")

    class FakeEnvironment:
        def __init__(self) -> None:
            self.reset_payload: dict[str, Any] | None = None
            self.execute_payload: dict[str, Any] | None = None

        def reset(self, **payload: Any) -> dict[str, Any]:
            self.reset_payload = payload
            return {
                "observation": {"state": {}, "images": {}},
                "authoritative_success": False,
                "terminated": False,
                "truncated": False,
                "task_program_enabled": False,
                "critic_rule_count": 0,
            }

        def observation(self, **_: Any) -> dict[str, Any]:
            return {"observation": {"state": {}, "images": {}}}

        def execute_chunk(self, actions: Any, **payload: Any) -> dict[str, Any]:
            self.execute_payload = {"actions": actions, **payload}
            return {
                "executed_horizon": 1,
                "steps": [
                    {
                        "step_index": 1,
                        "applied_action": {},
                        "action_sha256": "a" * 64,
                        "state": {},
                        "reward": 0.0,
                        "official_success": False,
                        "terminated": True,
                        "truncated": False,
                        "proposal_rule_ids": [],
                    }
                ],
                "critic_proposals": [],
                "authoritative_success": False,
                "terminated": True,
                "truncated": False,
                "task_program_enabled": False,
                "critic_rule_count": 0,
            }

        def finalize_episode(self) -> dict[str, Any]:
            return {"video_paths": {}}

        def release(self) -> dict[str, Any]:
            return {"binding_released": True, "released_generation": 0}

    monkeypatch.setattr(run_rollout, "Gr00tClient", FakeVla)
    monkeypatch.setattr(run_rollout, "Role1ModelAdapter", FakeAdapterForRollout)
    monkeypatch.setattr(run_rollout, "Role1EpisodeActor", FailingActor)
    monkeypatch.setattr(
        run_rollout,
        "build_episode_visual_artifacts",
        lambda **_: {"artifacts": {}, "artifact_sha256": {}},
    )
    args = SimpleNamespace(
        output_dir=str(tmp_path / "attempt"),
        result_file=str(tmp_path / "attempt" / "episode_record.json"),
        bundle="none",
        bundle_sha256="none",
        baseline_mode="strict_pure_vla",
        safety_layer="interface_contract_v1",
        role1_planner="api",
        role1_model="openai:gpt-5.6-sol",
        reasoning_effort="high",
        role1_max_tokens=128,
        role1_timeout_s=10.0,
        role1_heartbeat_s=0.01,
        role1_max_turns=1,
        role1_max_decisions_per_action=2,
        allow_privileged_tools=True,
        vla_endpoint="http://127.0.0.1:1",
        vla_timeout_s=10.0,
        task="SlideDishwasherRack",
        split="target",
        seed=1,
        policy_rng=2,
        instruction=None,
        max_actions=8,
        actions_per_chunk=8,
        logical_id="method-failure-valid-zero",
        generation=0,
        attempt_index=0,
    )
    environment = FakeEnvironment()
    record = run_rollout._run_with_environment(args, environment)  # type: ignore[arg-type]
    assert record.status == "valid"
    assert record.success is False
    assert record.failure_segment is not None
    assert record.artifact_index["role1_decisions"] == 0
    assert record.artifact_index["terminated"] is True
    assert record.artifact_index["task_program_enabled"] is False
    assert environment.reset_payload is not None
    assert environment.reset_payload["enable_task_program"] is False
    assert environment.execute_payload is not None
    assert environment.execute_payload["critic_rules"] == []
    assert environment.execute_payload["interrupt_on_proposal"] is False


def test_pure_vla_rollout_does_not_resolve_task_binding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class FakeVla:
        def __init__(self, *_: Any, **__: Any) -> None:
            pass

    class FakeEnvironment:
        def reset(self, **_: Any) -> dict[str, Any]:
            return {
                "observation": {"state": {}, "images": {}},
                "authoritative_success": True,
                "task_program_enabled": False,
                "critic_rule_count": 0,
                "terminated": False,
                "truncated": False,
            }

        def finalize_episode(self) -> dict[str, Any]:
            return {"video_paths": {}}

        def release(self) -> dict[str, Any]:
            return {"binding_released": True, "released_generation": 0}

    def fail_if_binding_is_resolved(_: str) -> Any:
        raise AssertionError("pure VLA must not resolve an RPent task binding")

    monkeypatch.setattr(run_rollout, "Gr00tClient", FakeVla)
    monkeypatch.setattr(run_rollout, "binding_for_task", fail_if_binding_is_resolved)
    monkeypatch.setattr(
        run_rollout,
        "build_episode_visual_artifacts",
        lambda **_: {"artifacts": {}, "artifact_sha256": {}},
    )
    args = SimpleNamespace(
        output_dir=str(tmp_path / "attempt"),
        result_file=str(tmp_path / "attempt" / "episode_record.json"),
        bundle="none",
        bundle_sha256="none",
        role1_planner="none",
        role1_model=None,
        reasoning_effort=None,
        role1_max_tokens=128,
        role1_timeout_s=10.0,
        role1_max_turns=1,
        role1_max_decisions_per_action=2,
        allow_privileged_tools=False,
        tool_runtime="harness",
        harness_root="/must/not/be/read",
        vla_endpoint="http://127.0.0.1:1",
        vla_timeout_s=10.0,
        task="PickPlaceDrawerToCounter",
        split="target",
        seed=100,
        policy_rng=100,
        instruction=None,
        max_actions=8,
        actions_per_chunk=8,
        logical_id="pure-vla-unbound-task",
        generation=0,
        attempt_index=0,
    )
    record = run_rollout._run_with_environment(args, FakeEnvironment())  # type: ignore[arg-type]
    assert record.status == "valid"
    assert record.success is True
    assert record.artifact_index["tool_runtime"] == {
        "backend": "none_pure_vla",
        "tool_count": 0,
        "tool_names": [],
        "manifest_sha256": None,
    }
    assert (tmp_path / "attempt" / "tool_events.jsonl").read_text() == ""
    assert not (tmp_path / "attempt" / "role1").exists()


def test_rollout_counts_invalid_model_contract_as_valid_zero() -> None:
    try:
        raise Role1ContractError("model omitted an executable alternative")
    except Role1ContractError as cause:
        error = Role1ModelError(
            "Role1 invocation failed closed during prepare: Role1ContractError"
        )
        error.__cause__ = cause
    assert _is_role1_method_failure(error)
    assert not _is_role1_method_failure(Role1ModelError("provider unavailable"))
