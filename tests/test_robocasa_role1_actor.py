# Copyright (c) 2026 Zetta Contributors
from __future__ import annotations

import json
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
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
from rollout_runtime.api.ids import EpisodeId, RequestId, SessionId
from rollout_runtime.api.messages import (
    Observation,
    PerStepRecord,
    PolicyInferResult,
    SessionHandle,
    SessionStatus,
    StepResult,
)
from rollout_runtime.api.result import ok
from rollout_runtime.core.payload import decode_array, encode_array


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

ACTION_DIM = 12


def _snapshot_payload(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "observation": {"state": {}, "images": {}, "image_sha256": {}},
        "task": "SlideDishwasherRack",
        "split": "target",
        "step_index": 0,
        "terminated": False,
        "truncated": False,
        "reward": 0.0,
        "official_success": False,
        "success_latched": False,
        "success_first_step": None,
        "authoritative_success": False,
        "bundle_sha256": None,
        "task_program_enabled": False,
        "critic_rule_count": 0,
        "video_paths": {},
        "renderer": {},
    }
    payload.update(overrides)
    return payload


class _FakeRuntimeClient:
    """``RemoteRuntimeClient``-shaped stand-in: one session, no simulator.

    Attributes:
        calls: Ordered operation names, so a test can assert *which* call path
            the loop took (``policy_step`` vs ``policy_infer`` + ``action_step``).
        reset_options: The ``ResetSpec.options`` the rollout sent.
        policy_requests: Every ``PolicyRequest`` the rollout sent.
        executed_chunks: Action chunks that reached ``action_step``.
        closed: Whether ``close_sessions`` succeeded.
    """

    def __init__(
        self,
        *,
        chunks_before_termination: int = 1,
        critic_proposals_by_chunk: dict[int, list[dict[str, Any]]] | None = None,
        snapshot: dict[str, Any] | None = None,
    ) -> None:
        self.chunks_before_termination = int(chunks_before_termination)
        self.critic_proposals_by_chunk = critic_proposals_by_chunk or {}
        self.snapshot_payload = snapshot or _snapshot_payload()
        self.calls: list[str] = []
        self.reset_options: dict[str, Any] | None = None
        self.policy_requests: list[Any] = []
        self.executed_chunks: list[list[list[float]]] = []
        self.closed = False
        self.session_id = SessionId("session-fake")
        self._chunk_index = 0
        self._step_index = 0

    # -------------------------------------------------------------- Control plane

    async def create_sessions(self, requests: Any) -> list[Any]:
        self.calls.append("create_sessions")
        self.create_request = list(requests)[0]
        return [
            ok(
                SessionHandle(
                    session_id=self.session_id,
                    application_id="zetta-robocasa",
                    env_spec_digest="d" * 64,
                    default_policy_id=self.create_request.default_policy_id,
                    lease_expiration=time.time() + 3600.0,
                    gateway_epoch=1,
                )
            )
        ]

    async def renew_sessions(self, session_ids: Any, lease_seconds: float) -> list[Any]:
        self.calls.append("renew_sessions")
        return [
            ok(
                SessionStatus(
                    session_id=self.session_id,
                    state="READY",  # type: ignore[arg-type]
                    lease_expiration=time.time() + lease_seconds,
                )
            )
        ]

    async def close_sessions(self, session_ids: Any) -> list[Any]:
        self.calls.append("close_sessions")
        self.closed = True
        return [ok(None)]

    async def aclose(self) -> None:
        self.calls.append("aclose")

    # -------------------------------------------------------------- Action plane

    async def reset(self, session_ids: Any, reset_spec: Any) -> list[Any]:
        self.calls.append("reset")
        self.reset_options = dict(reset_spec.options)
        self.reset_seed = reset_spec.seed
        return [
            ok(
                StepResult(
                    request_id=RequestId("request-reset"),
                    session_id=self.session_id,
                    episode_id=EpisodeId(1),
                    observation=Observation(
                        session_id=self.session_id,
                        episode_id=EpisodeId(1),
                        step_index=0,
                    ),
                    info={"reset": True, "episode_id": 1},
                    side_effect_applied=True,
                )
            )
        ]

    async def extension_call(
        self, session_ids: Any, namespace: str, method: str, args: dict[str, Any]
    ) -> list[Any]:
        self.calls.append(f"{namespace}.{method}")
        if method == "snapshot":
            return [ok(dict(self.snapshot_payload))]
        if method == "finalize_episode":
            return [ok({"finalized": True, "video_paths": {}, "video_manifest": None})]
        raise AssertionError(f"unexpected extension {namespace}.{method}")

    async def policy_step(self, session_ids: Any, policy_request: Any) -> list[Any]:
        self.calls.append("policy_step")
        self.policy_requests.append(policy_request)
        return [ok(self._step_result())]

    async def policy_infer(self, session_ids: Any, policy_request: Any) -> list[Any]:
        self.calls.append("policy_infer")
        self.policy_requests.append(policy_request)
        block = np.zeros((1, ACTION_DIM), dtype=np.float32)
        return [
            ok(
                PolicyInferResult(
                    request_id=RequestId("request-infer"),
                    session_id=self.session_id,
                    episode_id=EpisodeId(1),
                    actions=encode_array(block),
                    model_version="fake-v1",
                    auxiliary_outputs={"chunk": 1},
                    observation_step_index=self._step_index,
                    info={"policy_id": "groot"},
                )
            )
        ]

    async def action_step(self, session_ids: Any, actions: Any) -> list[Any]:
        self.calls.append("action_step")
        block = decode_array(list(actions)[0])
        self.executed_chunks.append([[float(v) for v in row] for row in block])
        return [ok(self._step_result())]

    # -------------------------------------------------------------- Internal

    def _step_result(self) -> StepResult:
        self._chunk_index += 1
        self._step_index += 1
        terminated = self._chunk_index >= self.chunks_before_termination
        proposals = self.critic_proposals_by_chunk.get(self._chunk_index, [])
        return StepResult(
            request_id=RequestId(f"request-chunk-{self._chunk_index}"),
            session_id=self.session_id,
            episode_id=EpisodeId(1),
            observation=Observation(
                session_id=self.session_id,
                episode_id=EpisodeId(1),
                step_index=self._step_index,
            ),
            reward=0.0,
            terminated=terminated,
            truncated=False,
            executed_horizon=1,
            per_step=[
                PerStepRecord(
                    step_index=self._step_index,
                    reward=0.0,
                    terminated=terminated,
                    truncated=False,
                    info={
                        "applied_action": {"action.gripper_close": [0.0]},
                        "action_sha256": "a" * 64,
                        "observation_sha256": "b" * 64,
                        "raw_state": {},
                        "official_success": False,
                        "success_latched": False,
                        "proposal_rule_ids": [
                            str(item["rule_id"]) for item in proposals
                        ],
                    },
                )
            ],
            info={
                "critic_proposals": proposals,
                "critic_rule_count": len(self.critic_proposals_by_chunk and proposals),
                "task_program_enabled": False,
                "authoritative_success": False,
                "official_success": False,
                "success_latched": False,
                "success_first_step": None,
                "video_paths": {},
                "environment_write_owner": "robocasa_session",
                "model_version": "fake-v1",
                "policy_id": "groot",
            },
            side_effect_applied=True,
        )


def _rollout_args(tmp_path: Path, **overrides: Any) -> SimpleNamespace:
    attempt = tmp_path / "attempt"
    args = SimpleNamespace(
        runtime_url="http://runtime.test",
        runtime_token=None,
        policy_id="groot",
        operation_timeout_s=30.0,
        session_timeout_s=30.0,
        session_lease_s=3600.0,
        output_dir=str(attempt),
        result_file=str(attempt / "episode_record.json"),
        bundle="none",
        bundle_sha256="none",
        baseline_mode="strict_pure_vla",
        safety_layer="interface_contract_v1",
        role1_planner="none",
        role1_model=None,
        reasoning_effort=None,
        role1_max_tokens=128,
        role1_timeout_s=10.0,
        role1_heartbeat_s=0.01,
        role1_max_turns=1,
        role1_max_decisions_per_action=2,
        allow_privileged_tools=True,
        tool_runtime="builtin",
        harness_root=None,
        task="SlideDishwasherRack",
        split="target",
        seed=1,
        policy_rng=2,
        instruction=None,
        max_actions=8,
        actions_per_chunk=8,
        camera_size=256,
        env_max_steps=1000,
        require_isolated_renderer=True,
        process_isolation=False,
        env_pool_size=1,
        env_max_pool_size=None,
        logical_id="stage7-loop",
        generation=0,
        attempt_index=0,
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


async def test_strict_gen0_uses_policy_step_and_scores_termination_as_valid_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Gen0 goes through the atomic ``policy_step``; Role1 is never initialized, and a normal termination is recorded as a valid zero score."""

    class FailingAdapter:
        def __init__(self, *_: Any, **__: Any) -> None:
            raise AssertionError("strict Gen0 must not initialize Role1")

    class FailingActor:
        def __init__(self, *_: Any, **__: Any) -> None:
            raise AssertionError("strict Gen0 must not initialize Role1")

    monkeypatch.setattr(run_rollout, "Role1ModelAdapter", FailingAdapter)
    monkeypatch.setattr(run_rollout, "Role1EpisodeActor", FailingActor)
    monkeypatch.setattr(
        run_rollout,
        "build_episode_visual_artifacts",
        lambda **_: {"artifacts": {}, "artifact_sha256": {}},
    )
    client = _FakeRuntimeClient(chunks_before_termination=1)
    args = _rollout_args(tmp_path, role1_planner="api", logical_id="gen0-valid-zero")

    record = await run_rollout.run(args, client=client)

    assert record.status == "valid"
    assert record.success is False
    assert record.failure_segment is not None
    assert record.artifact_index["role1_decisions"] == 0
    assert record.artifact_index["terminated"] is True
    assert record.artifact_index["task_program_enabled"] is False
    # The only call chain: create -> reset -> snapshot -> policy_step -> finalize -> close
    assert client.calls == [
        "create_sessions",
        "reset",
        "robocasa.snapshot",
        "policy_step",
        "robocasa.finalize_episode",
        "close_sessions",
    ]
    # Gen0 has no bundle, so there is no Critic rule to dispatch
    assert client.reset_options is not None
    assert client.reset_options["critic_rules"] == []
    assert client.reset_options["interrupt_on_proposal"] is False
    assert client.reset_options["enable_task_program"] is False
    assert client.reset_seed == 1
    assert client.policy_requests[0].actions_per_chunk == 8
    assert client.policy_requests[0].inference_parameters == {
        "seed": run_rollout._chunk_seed(2, 0)
    }
    assert record.artifact_index["environment_release"]["session_closed"] is True


async def test_pure_vla_rollout_does_not_resolve_task_binding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail_if_binding_is_resolved(_: str) -> Any:
        raise AssertionError("pure VLA must not resolve an Zetta task binding")

    monkeypatch.setattr(run_rollout, "binding_for_task", fail_if_binding_is_resolved)
    monkeypatch.setattr(
        run_rollout,
        "build_episode_visual_artifacts",
        lambda **_: {"artifacts": {}, "artifact_sha256": {}},
    )
    client = _FakeRuntimeClient(
        snapshot=_snapshot_payload(authoritative_success=True, success_latched=True)
    )
    args = _rollout_args(
        tmp_path,
        role1_planner="none",
        tool_runtime="harness",
        harness_root="/must/not/be/read",
        allow_privileged_tools=False,
        task="PickPlaceDrawerToCounter",
        seed=100,
        policy_rng=100,
        logical_id="pure-vla-unbound-task",
    )

    record = await run_rollout.run(args, client=client)

    assert record.status == "valid"
    # After reset, authoritative_success is already true, so not a single chunk should run
    assert record.success is True
    assert "policy_step" not in client.calls
    assert record.artifact_index["tool_runtime"] == {
        "backend": "none_pure_vla",
        "tool_count": 0,
        "tool_names": [],
        "manifest_sha256": None,
    }
    assert (tmp_path / "attempt" / "tool_events.jsonl").read_text() == ""
    assert not (tmp_path / "attempt" / "role1").exists()


def _critic_rule() -> Any:
    from zetta.evolution.models import CriticRule

    return CriticRule(
        rule_id="critic-1",
        title="rack residual stagnant",
        feature="privileged.dishwasher.rack.residual_to_success",
        operator="gt",
        threshold=0.05,
        dwell_steps=1,
        cooldown_steps=0,
        proposal="stop and re-approach the rack",
        evidence_ids=("evidence-1",),
    )


def _bundle_file(tmp_path: Path) -> tuple[Path, Any]:
    from zetta.evolution.jsonio import atomic_write_json
    from zetta.evolution.models import CandidateBundle, RecoveryRule, RecoveryStep

    recovery = RecoveryRule(
        recovery_id="recovery-1",
        title="re-approach with a guarded push",
        trigger_rule_ids=("critic-1",),
        precondition="rack residual above the frozen threshold",
        steps=(
            RecoveryStep(
                tool=CONTACT_PUSH_TOOL,
                parameters={"distance_m": 0.02},
                stop_when="rack residual below threshold",
            ),
        ),
        safety_constraints=("no base motion",),
        stop_condition="rack fully seated",
        fallback="return control to the VLA",
        evidence_ids=("evidence-1",),
    )
    bundle = CandidateBundle(
        candidate_id="candidate-stage7",
        generation=1,
        parent_sha256=None,
        diagnosis_sha256="3" * 64,
        causal_hypothesis="one auditable failure mechanism",
        mechanism_change="one frozen critic rule with a bounded recovery",
        validation_plan="paired same-seed measurement",
        critic_rules=(_critic_rule(),),
        recovery_rules=(recovery,),
    )
    path = tmp_path / "bundle.json"
    atomic_write_json(path, bundle.as_dict(), overwrite=False)
    return path, bundle


async def test_active_bundle_sends_critic_rules_and_switches_to_action_step_for_role1(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The complete active_bundle loop: rule dispatch -> Critic proposal -> Role1 review -> action_step.

    This validates two behavioral changes in one pass:

    1. ``critic_rules`` actually reach the environment via ``ResetSpec.options``
       (previously hardcoded as an empty list, leaving the Critic permanently
       silent and giving ``active_bundle`` no semantics on the runtime path);
    2. The chunk where Role1 is allowed to intervene goes through
       ``policy_infer`` + ``action_step``, executing Role1's rewritten action
       instead of the policy's original action (``policy_step`` is atomic and
       its action block never passes through the client, so Role1 cannot
       review it).
    """
    bundle_path, bundle = _bundle_file(tmp_path)
    reviewed_actions = [[0.25] * ACTION_DIM]

    class FakeAdapterForRollout:
        def __init__(self, *_: Any, **__: Any) -> None:
            pass

    class ReviewingActor:
        def __init__(self, *_: Any, **__: Any) -> None:
            self.seen: list[dict[str, Any]] = []

        def decide_action(self, **kwargs: Any) -> Any:
            self.seen.append(kwargs)
            return SimpleNamespace(
                decision_ids=("decision-1",),
                selected_tool=CONTACT_PUSH_TOOL,
                terminate=False,
                termination_reason=None,
                actions=tuple(reviewed_actions),
            )

    actors: list[ReviewingActor] = []

    def _make_actor(*args: Any, **kwargs: Any) -> ReviewingActor:
        actor = ReviewingActor()
        actors.append(actor)
        return actor

    monkeypatch.setattr(run_rollout, "Role1ModelAdapter", FakeAdapterForRollout)
    monkeypatch.setattr(run_rollout, "Role1EpisodeActor", _make_actor)
    monkeypatch.setattr(
        run_rollout,
        "build_episode_visual_artifacts",
        lambda **_: {"artifacts": {}, "artifact_sha256": {}},
    )
    proposal = {
        "rule_id": "critic-1",
        "step_index": 1,
        "proposal": "stop and re-approach the rack",
    }
    client = _FakeRuntimeClient(
        chunks_before_termination=2,
        critic_proposals_by_chunk={1: [proposal]},
    )
    args = _rollout_args(
        tmp_path,
        role1_planner="api",
        baseline_mode="active_bundle",
        bundle=str(bundle_path),
        bundle_sha256=bundle.sha256,
        generation=1,
        logical_id="active-bundle-role1",
    )

    record = await run_rollout.run(args, client=client)

    assert record.status == "valid"
    # Frozen rules are dispatched once via reset (constant within the episode, so not resent per chunk)
    assert client.reset_options is not None
    assert client.reset_options["critic_rules"] == [
        rule.as_dict() for rule in bundle.critic_rules
    ]
    assert client.reset_options["interrupt_on_proposal"] is True
    assert client.reset_options["bundle_sha256"] == bundle.sha256
    # First chunk has no proposal -> atomic policy_step; second chunk has a proposal -> infer + action_step
    assert client.calls == [
        "create_sessions",
        "reset",
        "robocasa.snapshot",
        "policy_step",
        "policy_infer",
        "robocasa.snapshot",
        "action_step",
        "robocasa.finalize_episode",
        "close_sessions",
    ]
    assert client.executed_chunks == [reviewed_actions]
    assert record.artifact_index["role1_decisions"] == 1
    assert record.artifact_index["active_bundle_sha256"] == bundle.sha256
    # What Role1 sees is the previous round's Critic proposal along with the actual VLA action block
    assert len(actors) == 1
    seen = actors[0].seen[0]
    assert [item["rule_id"] for item in seen["critic_values"]] == ["critic-1"]
    assert seen["vla_metadata"]["source"] == "policy_infer"
    assert len(seen["vla_metadata"]["action_chunk_sha256"]) == 64
    events = [
        json.loads(line)
        for line in (tmp_path / "attempt" / "tool_events.jsonl")
        .read_text()
        .splitlines()
        if line.strip()
    ]
    assert [event["type"] for event in events] == ["role1_action_boundary"]
    assert events[0]["environment_write"] is False
