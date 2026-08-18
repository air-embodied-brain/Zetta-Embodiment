# Copyright (c) 2026 RPent Contributors
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from robots.libero.env_client import _TIMEOUT_S
from robots.libero.evolution_defaults import (
    add_privileged_evidence_argument,
    privileged_evidence_enabled,
)
from robots.libero.role1_recovery import (
    LiberoRecoveryActorError,
    LiberoRole1RecoveryActor,
)
from robots.libero.run_evolution_rollout import (
    REPOSITORY_ROOT,
    _exception_traceback,
    _frozen_subprocess_environment,
    _require_expected_task_language,
    _role1_inference_heartbeat,
)
from robots.libero.tool_catalog import DEFAULT_LIBERO_ROLE1_TOOL_CATALOG
from robots.libero.tools import (
    LiberoPrimitives,
    _owned_rgb_frame,
    _validate_owned_video_frames,
)
from robots.robocasa.role1_agent import (
    CriticProposal,
    Role1DecisionStore,
    Role1Event,
    ToolProposal,
)
from rpent.evolution.jsonio import canonical_sha256, read_json
from rpent.utils.rpc import RpcError
from scripts.evolution.prepare_libero_campaign import (
    _load_task_contract,
    prepare,
)
from scripts.evolution.prepare_libero_role1_smoke import prepare as prepare_role1_smoke


def test_recorded_rgb_frame_owns_renderer_memory() -> None:
    renderer_buffer = np.zeros((8, 8, 3), dtype=np.uint8)
    frame = _owned_rgb_frame(renderer_buffer, name="agentview")
    renderer_buffer[:] = 255
    assert frame.flags.owndata
    assert frame.flags.c_contiguous
    assert int(frame.max()) == 0


def test_rollout_requires_live_language_to_match_frozen_contract() -> None:
    expected = "Pick up the cup and place it in the caddy"

    assert (
        _require_expected_task_language(
            "  PICK up the cup  and place it in the caddy  ", expected
        )
        == "PICK up the cup  and place it in the caddy"
    )
    with pytest.raises(RuntimeError, match="changed after reset"):
        _require_expected_task_language("pick up the book", expected)
    with pytest.raises(RuntimeError, match="is empty"):
        _require_expected_task_language(None, expected)


def test_campaign_probes_authoritative_task_language(monkeypatch) -> None:
    args = SimpleNamespace(
        suite="libero_10_task",
        task_id=5,
        runtime_python=Path(sys.executable),
        task_language=None,
        task_contract=None,
    )

    monkeypatch.setattr(
        "scripts.evolution.prepare_libero_campaign._probe_task_language",
        lambda _args: "pick up the cup and place it in the back compartment of the caddy",
    )

    contract = _load_task_contract(args, "libero_10_task/task5")

    assert contract["suite"] == "libero_10_task"
    assert contract["task_id"] == 5
    assert contract["language"] == (
        "pick up the cup and place it in the back compartment of the caddy"
    )
    assert contract["normalized_language"] == contract["language"]
    assert contract["language_sha256"] == canonical_sha256(
        {"language": contract["normalized_language"]}
    )


def test_libero_video_export_persists_aligned_source_frames(tmp_path: Path) -> None:
    primitives = object.__new__(LiberoPrimitives)
    primitives._recording = True
    primitives._frames = []
    primitives._wrist_frames = []
    for step in range(3):
        primitives.record_frame(
            {
                "main_images": np.full((16, 16, 3), step * 20, dtype=np.uint8),
                "wrist_images": np.full(
                    (16, 16, 3), 100 + step * 20, dtype=np.uint8
                ),
            }
        )

    report = primitives.stop_recording_and_save(
        str(tmp_path / "videos" / "episode.mp4"), fps=10
    )

    manifest = read_json(Path(report["source_manifest"]))
    assert manifest["status"] == "complete"
    assert manifest["frame_count"] == 3
    raw = Path(manifest["raw_frame_directory"])
    assert len(list(raw.glob("*_agentview.jpg"))) == 3
    assert len(list(raw.glob("*_wrist.jpg"))) == 3


def test_libero_video_validation_rejects_synchronized_corruption() -> None:
    black = np.zeros((8, 8, 3), dtype=np.uint8)
    white = np.full((8, 8, 3), 255, dtype=np.uint8)
    report = _validate_owned_video_frames([black, white], [black, white])
    assert report["status"] == "invalid"
    assert report["errors"][-1]["kind"] == "source_frame_corruption"


class _Role1Adapter:
    def __init__(
        self,
        *,
        selected_tool: str,
        proposal_disposition: str = "accept",
        modifications: dict[str, Any] | None = None,
    ) -> None:
        self.selected_tool = selected_tool
        self.proposal_disposition = proposal_disposition
        self.modifications = dict(modifications or {})
        self.events: list[Any] = []
        self.images: list[dict[str, str]] = []

    def decide(self, event: Any, *, image_payloads: dict[str, str] | None = None) -> Any:
        self.events.append(event)
        self.images.append(dict(image_payloads or {}))
        return SimpleNamespace(
            decision_id="decision-1",
            selected_tool=self.selected_tool,
            proposal_disposition=self.proposal_disposition,
            modifications=self.modifications,
            termination=SimpleNamespace(approved=False),
            direct_action=None,
        )


class _RecoveryPrimitives:
    def __init__(self) -> None:
        self.env = SimpleNamespace(
            episode_steps=4,
            episode_terminated=False,
            episode_truncated=False,
            privileged_critic_state=lambda: {
                "privileged.available": True,
                "privileged.task.semantic_available": True,
                "privileged.task.manipulated_object.name": "cream_cheese_1",
                "privileged.task.manipulated_object.distance_to_target_m": 0.12,
                "privileged.task.manipulated_object.position.x": 1.0,
                "privileged.task.manipulated_object.orientation.w": 1.0,
                "privileged.task.manipulated_object.target_offset.y": 0.4,
                "privileged.task.target.position.z": 0.8,
            },
        )
        self.calls: list[dict[str, Any]] = []
        self._last_chunk_info = {
            "step_records": [
                {
                    "state": {
                        "episode.terminated": False,
                        "privileged.task.goal.progress": 0.4,
                        "privileged.task.manipulated_object.distance_to_target_m": 0.12,
                        "privileged.task.manipulated_object.position.x": 1.0,
                        "privileged.task.manipulated_object.retained": True,
                        "command.realization.stalled": False,
                    }
                }
            ]
        }
        self._last_critic_proposals = [{"rule_id": "critic-1"}]

    def begin_recovery_step(self) -> None:
        self._last_critic_proposals = []

    def move_to(self, **parameters: Any) -> dict[str, Any]:
        self.calls.append(dict(parameters))
        self.env.episode_steps += 3
        return {"success": True}


def _observation() -> dict[str, np.ndarray]:
    image = np.zeros((12, 16, 3), dtype=np.uint8)
    return {"main_images": image, "wrist_images": image.copy()}


def _recovery_context() -> dict[str, Any]:
    return {
        "recovery_id": "recovery-1",
        "current_step": {
            "tool": "move_to",
            "parameters": {"xyz": [0.1, 0.2, 0.3]},
            "stop_when": "done",
        },
    }


def test_libero_role1_is_only_recovery_environment_writer(tmp_path: Path) -> None:
    adapter = _Role1Adapter(selected_tool="libero.move_to")
    primitives = _RecoveryPrimitives()
    actor = LiberoRole1RecoveryActor(
        adapter=adapter,  # type: ignore[arg-type]
        audit_root=tmp_path / "audit",
        allowed_tools=("move_to",),
    )

    result = actor.decide_and_execute(
        task="libero_goal_task/task6",
        step_index=4,
        observation=_observation(),
        critic_values=(
            {
                "rule_id": "critic-1",
                "proposal": "recover",
                "feature": "robot.eef.z",
                "observed_value": 0.1,
                "activation_conditions": [
                    {
                        "feature": "privileged.task.manipulated_object.retained",
                        "operator": "eq",
                        "threshold": True,
                        "observed_value": True,
                    }
                ],
            },
        ),
        recovery_context=_recovery_context(),
        primitives=primitives,
    )

    assert result.selected_tool == "move_to"
    assert result.executed_horizon == 3
    assert primitives.calls == [{"xyz": [0.1, 0.2, 0.3]}]
    assert adapter.events[0].tool_proposals[0].tool == "libero.move_to"
    assert set(adapter.images[0]) == {"agentview", "wrist"}
    activation = adapter.events[0].critic_proposals[0].details[
        "activation_conditions"
    ]
    assert [dict(row) for row in activation] == [
        {
            "feature": "privileged.task.manipulated_object.retained",
            "operator": "eq",
            "threshold": True,
            "observed_value": True,
        }
    ]
    state = adapter.events[0].task_state
    assert state["critic_observations"]["fields"][
        "privileged.task.goal.progress"
    ] == 0.4
    assert "privileged.task.manipulated_object.position.x" not in state[
        "critic_observations"
    ]["fields"]
    assert [row.tool for row in adapter.events[0].tool_proposals] == [
        "libero.move_to",
        "libero.vla_execute",
    ]
    assert adapter.events[0].tool_proposals[0].proposal["proposal_role"] == (
        "frozen_recovery_alternative"
    )
    assert adapter.events[0].tool_proposals[1].proposal["status"] == (
        "rejected_by_critic"
    )
    assert state["privileged_recovery_state"] == {
        "privileged.available": True,
        "privileged.task.semantic_available": True,
        "privileged.task.manipulated_object.name": "cream_cheese_1",
        "privileged.task.manipulated_object.distance_to_target_m": 0.12,
    }
    assert list((tmp_path / "audit").glob("*.json"))


def test_recovery_rule_suppression_filters_only_active_trigger() -> None:
    primitives = LiberoPrimitives.__new__(LiberoPrimitives)
    primitives._suppressed_recovery_rule_ids = set()
    proposals = [
        {"rule_id": "active-trigger", "safety_only": False},
        {"rule_id": "safety-stop", "safety_only": True},
    ]

    with primitives.suppress_recovery_rules({"active-trigger"}):
        assert primitives._filter_suppressed_proposals(proposals) == [
            {"rule_id": "safety-stop", "safety_only": True}
        ]

    assert primitives._filter_suppressed_proposals(proposals) == proposals


def test_libero_role1_recursively_withholds_geometry_from_payload_and_result(
    tmp_path: Path,
) -> None:
    adapter = _Role1Adapter(selected_tool="libero.privileged_pick_place")
    primitives = _RecoveryPrimitives()

    def privileged_pick_place(**_parameters: Any) -> dict[str, Any]:
        primitives.env.episode_steps += 4
        return {
            "name": "privileged_pick_place",
            "status": "placed",
            "primary_relation_satisfied": False,
            "approach": {
                "target_xyz": [0.1, 0.2, 0.3],
                "final_eef_pos": [0.11, 0.21, 0.31],
                "final_dist_m": 0.02,
            },
            "nested": [
                {
                    "object_pose": {"x": 0.1, "y": 0.2, "z": 0.3},
                    "grasp_verified": True,
                }
            ],
        }

    primitives.privileged_pick_place = privileged_pick_place  # type: ignore[attr-defined]
    actor = LiberoRole1RecoveryActor(
        adapter=adapter,  # type: ignore[arg-type]
        audit_root=tmp_path / "audit",
        allowed_tools=("privileged_pick_place",),
    )
    recovery_context = {
        "recovery_id": "recovery-privileged",
        "current_step": {
            "tool": "privileged_pick_place",
            "parameters": {
                "grasp_offset_xyz": [0.0, -0.036, 0.038],
                "max_steps_per_move": 48,
            },
        },
        "private_pose": {"x": 1.0, "y": 2.0, "z": 3.0},
    }

    result = actor.decide_and_execute(
        task="libero_goal_task/task3",
        step_index=4,
        observation=_observation(),
        critic_values=(),
        recovery_context=recovery_context,
        primitives=primitives,
    )

    event = adapter.events[0]
    assert event.task_state["active_recovery"] == {
        "recovery_id": "recovery-privileged",
        "current_step": {
            "tool": "privileged_pick_place",
            "parameters": {"max_steps_per_move": 48},
        },
    }
    assert event.tool_proposals[0].proposal["current_step"] == {
        "tool": "privileged_pick_place",
        "parameters": {"max_steps_per_move": 48},
    }
    assert result.result == {
        "name": "privileged_pick_place",
        "status": "placed",
        "primary_relation_satisfied": False,
        "approach": {"final_dist_m": 0.02},
        "nested": [{"grasp_verified": True}],
    }
    audit = read_json(next((tmp_path / "audit").glob("*.json")))
    assert audit["result"] == result.result


def test_libero_role1_rejects_private_geometry_critic_evidence(
    tmp_path: Path,
) -> None:
    adapter = _Role1Adapter(selected_tool="libero.move_to")
    actor = LiberoRole1RecoveryActor(
        adapter=adapter,  # type: ignore[arg-type]
        audit_root=tmp_path / "audit",
        allowed_tools=("move_to",),
    )

    with pytest.raises(LiberoRecoveryActorError, match="private geometry"):
        actor.decide_and_execute(
            task="libero_goal_task/task3",
            step_index=4,
            observation=_observation(),
            critic_values=(
                {
                    "rule_id": "private-position-rule",
                    "feature": "privileged.task.manipulated_object.position.x",
                    "observed_value": 0.1,
                    "activation_conditions": [],
                },
            ),
            recovery_context=_recovery_context(),
            primitives=_RecoveryPrimitives(),
        )

    assert adapter.events == []


def test_libero_role1_cannot_bypass_frozen_recovery(tmp_path: Path) -> None:
    actor = LiberoRole1RecoveryActor(
        adapter=_Role1Adapter(selected_tool="libero.vla_execute"),  # type: ignore[arg-type]
        audit_root=tmp_path / "audit",
        allowed_tools=("move_to", "vla_execute"),
    )
    with pytest.raises(LiberoRecoveryActorError, match="did not select"):
        actor.decide_and_execute(
            task="libero_goal_task/task6",
            step_index=4,
            observation=_observation(),
            critic_values=(),
            recovery_context=_recovery_context(),
            primitives=_RecoveryPrimitives(),
        )


def test_libero_role1_modified_recovery_parameters_reach_actor(tmp_path: Path) -> None:
    adapter = _Role1Adapter(
        selected_tool="libero.move_to",
        proposal_disposition="modify",
        modifications={"parameters": {"xyz": [0.4, 0.5, 0.6]}},
    )
    primitives = _RecoveryPrimitives()
    actor = LiberoRole1RecoveryActor(
        adapter=adapter,  # type: ignore[arg-type]
        audit_root=tmp_path / "audit",
        allowed_tools=("move_to",),
    )
    actor.decide_and_execute(
        task="libero_goal_task/task6",
        step_index=4,
        observation=_observation(),
        critic_values=(),
        recovery_context=_recovery_context(),
        primitives=primitives,
    )
    assert primitives.calls == [{"xyz": [0.4, 0.5, 0.6]}]
    audit = read_json(next((tmp_path / "audit").glob("*.json")))
    assert audit["role1_modified_parameters"] is True
    assert audit["base_parameters_sha256"] != audit["parameters_sha256"]


def test_libero_role1_store_validates_namespaced_recovery_tools(
    tmp_path: Path,
) -> None:
    event = Role1Event(
        event_id="libero-recovery-000004",
        task="libero_goal_task/task6",
        step_index=4,
        current_stage="recover",
        current_tool="libero.vla_execute",
        allowed_stages=("recover",),
        allowed_tools=("libero.vla_execute", "libero.move_to"),
        image_references={},
        task_state={"libero_terminated": False},
        tool_proposals=(
            ToolProposal(
                proposal_id="tool-proposal",
                tool="libero.vla_execute",
                proposal={
                    "status": "interrupted_by_critic",
                    "proposal_role": "frozen_recovery_alternative",
                },
                evidence=("live-critic-proposal",),
            ),
        ),
        critic_proposals=(
            CriticProposal(
                proposal_id="critic-proposal",
                reject_current_action=True,
                reason="execute the frozen recovery",
                evidence=("critic:test",),
                details={},
            ),
        ),
    )
    store = Role1DecisionStore(
        tmp_path / "decisions",
        catalog=DEFAULT_LIBERO_ROLE1_TOOL_CATALOG,
    )
    accepted = store.prepare(
        event,
        {
            "event_id": event.event_id,
            "proposal_disposition": "accept",
            "action_kind": "continue",
            "selected_stage": "recover",
            "selected_tool": "libero.vla_execute",
            "direct_action": None,
            "termination": {"approved": False, "reason": ""},
            "evidence": ["critic:test"],
            "confidence": 0.9,
            "rationale": "Accept the explicit frozen recovery alternative.",
            "proposal_ids": ["tool-proposal", "critic-proposal"],
            "modifications": {},
        },
    )
    accepted_effect = store.activate(store.persist(accepted))
    assert accepted_effect.proposal_disposition == "accept"
    assert accepted_effect.selected_tool == "libero.vla_execute"
    pending = store.prepare(
        event,
        {
            "event_id": event.event_id,
            "proposal_disposition": "reject",
            "action_kind": "switch",
            "selected_stage": None,
            "selected_tool": "libero.move_to",
            "direct_action": None,
            "termination": {"approved": False, "reason": ""},
            "evidence": ["critic:test"],
            "confidence": 0.9,
            "rationale": "The frozen recovery requires the bounded move tool.",
            "proposal_ids": ["tool-proposal", "critic-proposal"],
            "modifications": {},
        },
    )
    persisted = store.persist(pending)
    effect = store.activate(persisted)
    assert effect.selected_tool == "libero.move_to"


def test_libero_role1_recovery_alternative_switch_is_contract_valid() -> None:
    from robots.robocasa.role1_agent import validate_role1_decision

    event = Role1Event(
        event_id="libero-recovery-switch",
        task="libero_goal_task/task7",
        step_index=11,
        current_stage="recover",
        current_tool="libero.vla_execute",
        allowed_stages=("recover",),
        allowed_tools=("libero.vla_execute", "libero.move_to"),
        image_references={},
        task_state={},
        tool_proposals=(
            ToolProposal(
                proposal_id="frozen-move",
                tool="libero.move_to",
                proposal={"proposal_role": "frozen_recovery_alternative"},
                evidence=("live-critic-proposal",),
            ),
        ),
        critic_proposals=(),
    )
    decision = validate_role1_decision(
        {
            "event_id": event.event_id,
            "proposal_disposition": "accept",
            "action_kind": "switch",
            "selected_stage": "recover",
            "selected_tool": "libero.move_to",
            "direct_action": None,
            "termination": {"approved": False, "reason": ""},
            "evidence": ["live-critic-proposal"],
            "confidence": 0.9,
            "rationale": "Use the frozen recovery tool.",
            "proposal_ids": ["frozen-move"],
            "modifications": {},
        },
        event=event,
        catalog=DEFAULT_LIBERO_ROLE1_TOOL_CATALOG,
    )
    assert decision.action_kind == "switch"
    assert decision.selected_tool == "libero.move_to"


def test_libero_role1_heartbeat_is_live_during_long_reasoning(tmp_path: Path) -> None:
    import json
    import time

    path = tmp_path / "heartbeat.jsonl"
    with _role1_inference_heartbeat(path, interval_s=0.01, step_index=9):
        time.sleep(0.035)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) >= 2
    assert {row["phase"] for row in rows} == {"role1_inference"}
    assert {row["step_index"] for row in rows} == {9}

    actor_path = tmp_path / "actor-heartbeat.jsonl"
    with _role1_inference_heartbeat(
        actor_path,
        interval_s=0.01,
        step_index=10,
        phase="role1_actor",
    ):
        time.sleep(0.025)
    actor_rows = [
        json.loads(line)
        for line in actor_path.read_text(encoding="utf-8").splitlines()
    ]
    assert {row["phase"] for row in actor_rows} == {"role1_actor"}
    assert {row["step_index"] for row in actor_rows} == {10}


def test_libero_infra_result_preserves_elapsed_and_partial_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import time

    import robots.libero.run_evolution_rollout as module

    output = tmp_path / "attempt-00"
    result = output / "result.json"

    def fail(args: Any) -> Any:
        (output / "trajectory").mkdir(parents=True)
        (output / "role1" / "invocations" / "invocation-test").mkdir(
            parents=True
        )
        (output / "heartbeat.jsonl").write_text("{}\n", encoding="utf-8")
        (output / "trajectory" / "chunks.jsonl").write_text(
            "{}\n", encoding="utf-8"
        )
        (output / "role1" / "invocations" / "invocation-test" / "input.json").write_text(
            "{}", encoding="utf-8"
        )
        time.sleep(0.01)
        raise RuntimeError("synthetic transport failure")

    monkeypatch.setattr(module, "_run", fail)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_evolution_rollout.py",
            "--suite",
            "libero_goal_task",
            "--task-id",
            "6",
            "--task",
            "libero_goal_task/task6",
            "--seed",
            "1",
            "--policy-rng",
            "2",
            "--logical-id",
            "infra-test",
            "--attempt-index",
            "0",
            "--generation",
            "0",
            "--baseline-mode",
            "strict_pure_vla",
            "--output-dir",
            str(output),
            "--result-file",
            str(result),
            "--vla-endpoint",
            "http://127.0.0.1:1",
            "--vla-gpu",
            "7",
            "--allowed-environment-gpus",
            "6",
        ],
    )
    assert module.main() == 2
    payload = read_json(result)
    assert payload["status"] == "infra_invalid"
    assert payload["elapsed_s"] >= 0.01
    assert payload["artifact_index"]["heartbeat"].endswith("heartbeat.jsonl")
    assert payload["artifact_index"]["chunks"].endswith("chunks.jsonl")
    assert any(
        key.startswith("partial:role1/")
        for key in payload["artifact_index"]
    )


def test_role1_smoke_bundle_is_explicitly_non_formal(tmp_path: Path) -> None:
    output = tmp_path / "bundle.json"
    report = prepare_role1_smoke(output, trigger_step=7)
    payload = read_json(output)
    attestation = read_json(output.with_suffix(".attestation.json"))
    assert report == attestation
    assert report["development_only"] is True
    assert report["bundle_sha256"] == canonical_sha256(payload)
    assert payload["critic_rules"][0]["feature"] == "episode.step_index"
    assert payload["critic_rules"][0]["threshold"] == 7
    assert payload["recovery_rules"][0]["steps"][0]["tool"] == "set_gripper"


def test_libero_child_server_is_bound_to_frozen_worktree(tmp_path: Path) -> None:
    old = tmp_path / "old-overlay"
    result = _frozen_subprocess_environment(
        {"PYTHONPATH": f"{old}{os.pathsep}{REPOSITORY_ROOT}"}
    )
    entries = result["PYTHONPATH"].split(os.pathsep)
    assert Path(entries[0]).resolve() == REPOSITORY_ROOT
    assert entries.count(str(REPOSITORY_ROOT)) == 1
    assert str(old) in entries


class _PrimitiveEnv:
    return_all_frames = True
    episode_terminated = False
    episode_truncated = False

    def __init__(self) -> None:
        self.episode_steps = 0

    def critic_chunk_step(self, actions: Any, **_kwargs: Any):
        self.episode_steps += 1
        obs = {
            "states": np.zeros(8, dtype=np.float32),
            "main_images": np.zeros((2, 2, 3), dtype=np.uint8),
            "wrist_images": np.zeros((2, 2, 3), dtype=np.uint8),
        }
        return [obs], [0.0], [False], [False], {
            "executed_horizon": 1,
            "critic_rule_count": 1,
            "critic_proposals": [{"rule_id": "critic-1"}],
        }


class _PrimitiveModel:
    def predict_action_batch(self, _obs: Any, **_kwargs: Any):
        return np.zeros((5, 7), dtype=np.float32), {}


class _PureVlaPrimitiveEnv:
    return_all_frames = True
    episode_terminated = False
    episode_truncated = False

    def __init__(self) -> None:
        self.episode_steps = 0
        self.plain_chunk_calls = 0

    def chunk_step(self, actions: Any, **_kwargs: Any):
        horizon = len(actions)
        self.episode_steps += horizon
        self.plain_chunk_calls += 1
        obs = {
            "states": np.zeros(8, dtype=np.float32),
            "main_images": np.zeros((2, 2, 3), dtype=np.uint8),
            "wrist_images": np.zeros((2, 2, 3), dtype=np.uint8),
        }
        return [obs] * horizon, [0.0] * horizon, [False] * horizon, [False] * horizon, {
            "executed_horizon": horizon,
        }

    def critic_chunk_step(self, _actions: Any, **_kwargs: Any):
        raise AssertionError("empty Gen0 critic must not use privileged path")


class _NonFinitePrimitiveModel:
    def predict_action_batch(self, _obs: Any, **_kwargs: Any):
        actions = np.zeros((5, 7), dtype=np.float32)
        actions[0, 0] = np.nan
        return actions, {}


class _SemanticJointProbeEnv:
    """Tiny physical proxy for the bounded semantic-joint controller."""

    episode_terminated = False
    episode_truncated = False

    def __init__(self) -> None:
        self.episode_steps = 0
        self.qpos = 0.03
        self.qvel = 0.03
        self.eef = np.asarray([0.0, 0.0, 0.95], dtype=np.float32)

    def _obs(self) -> dict[str, Any]:
        return {
            "states": np.r_[self.eef, np.zeros(5, dtype=np.float32)],
            "main_images": np.zeros((2, 2, 3), dtype=np.uint8),
            "wrist_images": np.zeros((2, 2, 3), dtype=np.uint8),
        }

    def privileged_semantic_joint_plan(self, **_kwargs: Any) -> dict[str, Any]:
        return {
            "joint": "flat_stove_1_button",
            "qpos": self.qpos,
            "qvel": self.qvel,
            "range_lower": -0.005,
            "range_upper": 2.1,
            "goal_satisfied": self.qpos <= -0.003,
            "approach_position_world": [0.0, 0.0, 1.02],
            "press_position_world": [0.0, 0.0, 0.95],
            # Already signed for the requested lower endpoint.
            "tangent_direction_world": [-1.0, 0.0, 0.0],
        }

    def step(self, action: Any):
        action = np.asarray(action, dtype=np.float32)
        self.eef = self.eef + action[:3] * 0.05
        # A lower tangent brakes positive angular velocity and then drives the
        # joint through the strict lower endpoint.
        self.qvel += float(action[0]) * 0.03 - self.qvel * 0.02
        self.qpos += self.qvel * 0.2
        self.episode_steps += 1
        return self._obs(), 0.0, False, False, {}


class _SemanticJointRotatingTangentEnv(_SemanticJointProbeEnv):
    def privileged_semantic_joint_plan(self, **kwargs: Any) -> dict[str, Any]:
        result = super().privileged_semantic_joint_plan(**kwargs)
        # Model the radial-frame ambiguity seen when the EEF crosses the
        # fixture center during a physical sweep.
        result["tangent_direction_world"] = [
            -1.0 if self.episode_steps % 2 == 0 else 1.0,
            0.0,
            0.0,
        ]
        return result


class _SemanticJointActionTraceEnv(_SemanticJointRotatingTangentEnv):
    def __init__(self) -> None:
        super().__init__()
        self.actions: list[np.ndarray] = []

    def step(self, action: Any):
        self.actions.append(np.asarray(action, dtype=np.float32).copy())
        return super().step(action)


class _SemanticSlideActionTraceEnv(_SemanticJointActionTraceEnv):
    def privileged_semantic_joint_plan(self, **kwargs: Any) -> dict[str, Any]:
        result = super().privileged_semantic_joint_plan(**kwargs)
        result["joint_type"] = "slide"
        return result


def test_semantic_joint_interact_probes_existing_contact_before_reposition() -> None:
    env = _SemanticJointProbeEnv()
    primitives = LiberoPrimitives(
        env, _PrimitiveModel(), object(), allow_privileged_actions=True
    )  # type: ignore[arg-type]
    primitives.set_obs(env._obs())

    result = primitives.semantic_joint_interact(
        "flat_stove_1", "button", direction="lower", max_sweep_steps=36
    )

    assert result["joint_goal_satisfied"] is True
    assert result["direct_contact_steps"] > 0
    assert result["recontacted"] is False
    assert result["direction_reversals"] == 0
    assert result["steps_used"] == result["sweep_steps"]


def test_semantic_joint_interact_keeps_one_tangent_frame_per_sweep() -> None:
    env = _SemanticJointRotatingTangentEnv()
    primitives = LiberoPrimitives(
        env, _PrimitiveModel(), object(), allow_privileged_actions=True
    )  # type: ignore[arg-type]
    primitives.set_obs(env._obs())

    result = primitives.semantic_joint_interact(
        "flat_stove_1", "button", direction="lower", max_sweep_steps=36
    )

    assert result["joint_goal_satisfied"] is True
    assert result["recontacted"] is False
    assert result["direction_reversals"] == 0


def test_semantic_joint_interact_keeps_short_range_mode_across_continuation() -> None:
    env = _SemanticJointActionTraceEnv()
    primitives = LiberoPrimitives(
        env, _PrimitiveModel(), object(), allow_privileged_actions=True
    )  # type: ignore[arg-type]
    primitives.set_obs(env._obs())

    result = primitives.semantic_joint_interact(
        "flat_stove_1", "button", direction="lower", max_sweep_steps=36
    )

    assert result["joint_goal_satisfied"] is True
    assert len(env.actions) == result["sweep_steps"]
    # The rotating probe flips its tangent every step.  A short-range
    # primitive must nevertheless keep the initial tangent through the second
    # continuation sweep after the 12-step direct-contact budget.
    assert all(float(action[0]) < 0.0 for action in env.actions)


def test_semantic_joint_interact_keeps_gripper_open_for_slide_push() -> None:
    env = _SemanticSlideActionTraceEnv()
    primitives = LiberoPrimitives(
        env, _PrimitiveModel(), object(), allow_privileged_actions=True
    )  # type: ignore[arg-type]
    primitives.set_obs(env._obs())

    result = primitives.semantic_joint_interact(
        "wooden_cabinet_1", "top_level", direction="lower", max_sweep_steps=36
    )

    assert result["joint_goal_satisfied"] is True
    assert env.actions
    assert all(float(action[2]) < 0.0 for action in env.actions)
    assert all(float(action[6]) == pytest.approx(-1.0) for action in env.actions)


def test_vla_recovery_tool_yields_after_first_critic_interruption() -> None:
    env = _PrimitiveEnv()
    primitives = LiberoPrimitives(env, _PrimitiveModel(), object())  # type: ignore[arg-type]
    obs = {
        "states": np.zeros(8, dtype=np.float32),
        "main_images": np.zeros((2, 2, 3), dtype=np.uint8),
        "wrist_images": np.zeros((2, 2, 3), dtype=np.uint8),
    }
    primitives.set_obs(obs)
    primitives.configure_critic(
        [
            {
                "rule_id": "critic-1",
                "feature": "robot.eef.z",
                "operator": "ge",
                "threshold": 0,
            }
        ]
    )

    result = primitives.vla_execute(prompt="recover", max_chunks=4)

    assert result["chunks_used"] == 1
    assert result["actions_executed"] == 1
    assert primitives.critic_interrupted() is True


def test_libero_action_contract_rejects_nonfinite_vla_action() -> None:
    env = _PrimitiveEnv()
    primitives = LiberoPrimitives(
        env, _NonFinitePrimitiveModel(), object()  # type: ignore[arg-type]
    )
    primitives.set_obs(
        {
            "states": np.zeros(8, dtype=np.float32),
            "main_images": np.zeros((2, 2, 3), dtype=np.uint8),
            "wrist_images": np.zeros((2, 2, 3), dtype=np.uint8),
        }
    )
    primitives.configure_critic([])

    with pytest.raises(ValueError, match="finite"):
        primitives.vla_execute(prompt="act", max_chunks=1)

    assert env.episode_steps == 0


def test_empty_gen0_critic_uses_plain_vla_path_and_remains_frozen() -> None:
    env = _PureVlaPrimitiveEnv()
    primitives = LiberoPrimitives(
        env, _PrimitiveModel(), object()  # type: ignore[arg-type]
    )
    primitives.set_obs(
        {
            "states": np.zeros(8, dtype=np.float32),
            "main_images": np.zeros((2, 2, 3), dtype=np.uint8),
            "wrist_images": np.zeros((2, 2, 3), dtype=np.uint8),
        }
    )
    primitives.configure_critic([])

    primitives._vlm_chunk("act", actions_per_chunk=2)

    assert primitives._critic_frozen is True
    assert primitives._critic_configured is False
    assert primitives._last_chunk_info["executed_horizon"] == 2
    assert "critic_rule_count" not in primitives._last_chunk_info
    assert env.plain_chunk_calls == 1
    with pytest.raises(ValueError, match="cannot change within an episode"):
        primitives.configure_critic(
            [{"rule_id": "late-rule", "feature": "episode.step_index"}]
        )


@pytest.mark.parametrize("task_id", range(10))
def test_all_goal_tasks_preregister_branch_wide_evolution_defaults(
    tmp_path: Path,
    task_id: int,
) -> None:
    root = tmp_path / "campaign"
    repo = Path(__file__).resolve().parents[1]
    args = SimpleNamespace(
        output_root=root,
        campaign_id="libero-test",
        repository_root=repo,
        runtime_python=Path(sys.executable),
        code_commit="0" * 40,
        suite="libero_goal_task",
        task_id=task_id,
        task=None,
        task_language=f"authoritative language for task {task_id}",
        task_contract=None,
        generation=0,
        parent_bundle=None,
        master_seed=7,
        schedule_from_manifest=None,
        rollout_count=2,
        heldout_count=2,
        population_size=100,
        initial_logical_slots=2,
        maximum_logical_slots=4,
        continuous_logical_slots=1,
        maximum_api_concurrency=4,
        episode_timeout_s=2700,
        no_progress_timeout_s=180,
        target_valid_episodes_per_hour=25.0,
        max_infrastructure_attempts=3,
        vla_endpoint="http://127.0.0.1:18811",
        environment_gpus=(5, 6),
        vla_gpu=7,
        max_actions=300,
        wait_steps=10,
        actions_per_chunk=5,
        role1_planner="api",
        agent_model="gpt-5.6-sol",
        role1_model="gpt-5.6-sol",
        reasoning_effort="high",
        role1_max_tokens=4096,
        role1_timeout_s=1200,
        role1_heartbeat_s=15.0,
        role1_max_turns=2,
        max_recovery_actor_calls=16,
    )

    prepare(args)  # type: ignore[arg-type]
    manifest = read_json(root / "manifest.json")
    catalog = read_json(root / "tool-catalog.json")

    assert manifest["environment"] == "libero_pro"
    assert manifest["task"] == f"libero_goal_task/task{task_id}"
    assert manifest["baseline_mode"] == "strict_pure_vla"
    assert manifest["active_bundle_sha256"] is None
    assert manifest["model"] == "gpt-5.6-sol"
    assert manifest["reasoning_effort"] == "high"
    assert manifest["runtime"]["rollout_requires_api"] is False
    assert manifest["runtime"]["candidate_rollout_requires_api"] is True
    assert manifest["runtime"]["rollout_requires_environment_slot"] is False
    assert manifest["runtime"]["subsequent_rollout_count"] == 10
    assert manifest["runtime"]["evolution_policy"]["skip_regression_gate"] is True
    assert manifest["runtime"]["libero_privileged_evidence"] is True
    assert manifest["runtime"]["evaluation_horizon"] == {
        "schema_version": 1,
        "protocol": "openpi_libero_suite_horizon_v1",
        "source": "https://github.com/Physical-Intelligence/openpi/blob/main/examples/libero/main.py",
        "suite": "libero_goal_task",
        "base_suite": "libero_goal",
        "standard_policy_action_horizon": 300,
        "policy_action_horizon": 300,
        "wait_steps": 10,
        "max_episode_steps": 310,
        "standard_wait_steps": 10,
        "is_standard": True,
    }
    assert manifest["runtime"]["runtime_device_contract"] == {
        "schema_version": 1,
        "contract": "libero_env_vla_physical_gpu_isolation_v1",
        "environment_gpus": [5, 6],
        "vla_gpu": 7,
        "same_gpu_forbidden": True,
    }
    contract = manifest["runtime"]["task_contract"]
    assert contract == read_json(root / "task-contract.json")
    assert contract == catalog["task_binding"]["authoritative_task_contract"]
    assert contract["language"] == f"authoritative language for task {task_id}"
    assert manifest["runtime"]["rollout_command"][-2:] == [
        "--expected-task-language",
        contract["language"],
    ]
    preregistration = read_json(root / "preregistration.json")
    assert preregistration["task_contract"] == contract
    assert preregistration["task_contract_sha256"] == canonical_sha256(contract)
    assert "--allow-privileged-evidence" in manifest["runtime"]["rollout_command"]
    assert "--allowed-environment-gpus" in manifest["runtime"]["rollout_command"]
    assert "--vla-gpu" in manifest["runtime"]["rollout_command"]
    assert (
        manifest["runtime"]["evolution_policy"]["maximum_total_candidate_rounds"]
        == 15
    )
    assert (
        manifest["runtime"]["evolution_policy"]["diagnosis_max_artifact_reads"]
        == 24
    )
    assert (
        manifest["runtime"]["evolution_policy"]["cluster_max_artifact_reads"]
        == 24
    )
    assert "--visual-overview-frames" in manifest["runtime"]["rollout_command"]
    assert "17" in manifest["runtime"]["rollout_command"]
    assert (
        manifest["runtime"]["evolution_policy"][
            "provisional_min_diagnosis_confidence"
        ]
        == 0.0
    )
    assert (
        manifest["runtime"]["evolution_policy"][
            "defer_inconclusive_for_provisional"
        ]
        is True
    )
    assert manifest["runtime"]["rollout_command"][0] == str(
        Path(os.path.abspath(sys.executable))
    )
    assert manifest["safety_layer"]["action_contract"] == "libero_finite_7d_action_v1"
    assert catalog["notes"]["collision_heuristic_default"] is False
    assert catalog["notes"]["critic_privileged_plane"]["role1_visibility"] is True
    assert catalog["suite"] == "libero_goal_task"
    assert catalog["task_id"] == task_id
    assert catalog["task_binding"]["success_criterion"] == "libero_terminated"
    assert "semantic_joint_interact" in catalog["task_binding"]["tool_names"]
    assert any(
        row["name"] == "semantic_joint_interact" for row in catalog["tools"]
    )


def test_libero_privileged_evidence_default_has_explicit_opt_out() -> None:
    parser = argparse.ArgumentParser()
    add_privileged_evidence_argument(parser, help_text="test")

    assert parser.parse_args([]).allow_privileged_evidence is True
    assert (
        parser.parse_args(["--no-allow-privileged-evidence"])
        .allow_privileged_evidence
        is False
    )
    assert privileged_evidence_enabled(SimpleNamespace()) is True
    assert "libero.semantic_joint_interact" in {
        item.name for item in DEFAULT_LIBERO_ROLE1_TOOL_CATALOG.select()
    }


def test_libero_campaign_reuses_frozen_manifest_schedule(tmp_path: Path) -> None:
    source = tmp_path / "source-manifest.json"
    source_payload = {
        "campaign_id": "source-campaign",
        "code_commit": "1" * 40,
        "task": "libero_goal_task/task6",
        "rollout_seeds": [9, 3],
        "heldout_seeds": [8, 4],
        "policy_rng_by_seed": {"9": 90, "3": 30, "8": 80, "4": 40},
    }
    source.write_text(__import__("json").dumps(source_payload), encoding="utf-8")
    root = tmp_path / "campaign"
    repo = Path(__file__).resolve().parents[1]
    args = SimpleNamespace(
        output_root=root,
        campaign_id="libero-reused-schedule",
        repository_root=repo,
        runtime_python=Path(sys.executable),
        code_commit="2" * 40,
        suite="libero_goal_task",
        task_id=6,
        task=None,
        task_language="authoritative language for task 6",
        task_contract=None,
        generation=0,
        parent_bundle=None,
        master_seed=None,
        schedule_from_manifest=source,
        rollout_count=2,
        heldout_count=2,
        population_size=100,
        initial_logical_slots=2,
        maximum_logical_slots=4,
        continuous_logical_slots=1,
        maximum_api_concurrency=4,
        episode_timeout_s=2700,
        no_progress_timeout_s=180,
        target_valid_episodes_per_hour=25.0,
        max_infrastructure_attempts=3,
        vla_endpoint="http://127.0.0.1:18811",
        environment_gpus=(6,),
        vla_gpu=7,
        max_actions=300,
        wait_steps=10,
        actions_per_chunk=5,
        role1_planner="api",
        agent_model="gpt-5.6-sol",
        role1_model="gpt-5.6-sol",
        reasoning_effort="high",
        role1_max_tokens=4096,
        role1_timeout_s=1200,
        role1_heartbeat_s=15.0,
        role1_max_turns=2,
        max_recovery_actor_calls=16,
    )

    report = prepare(args)  # type: ignore[arg-type]
    manifest = read_json(root / "manifest.json")

    assert manifest["rollout_seeds"] == [9, 3]
    assert manifest["heldout_seeds"] == [8, 4]
    assert manifest["policy_rng_by_seed"] == source_payload["policy_rng_by_seed"]
    assert report["schedule_provenance"]["source_campaign_id"] == "source-campaign"
    assert report["schedule_provenance"]["source_manifest_file_sha256"]


def test_formal_long_campaign_rejects_nonstandard_horizon(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[1]
    args = SimpleNamespace(
        output_root=tmp_path / "campaign",
        campaign_id="nonstandard-long",
        repository_root=repo,
        runtime_python=Path(sys.executable),
        code_commit="0" * 40,
        suite="libero_10_task",
        task_id=0,
        max_actions=300,
        wait_steps=10,
    )

    with pytest.raises(ValueError, match="official suite horizon"):
        prepare(args)  # type: ignore[arg-type]


def test_libero_campaign_preregisters_fixed_heldout_seed_block(tmp_path: Path) -> None:
    args = SimpleNamespace(
        schedule_from_manifest=None,
        master_seed=17,
        fixed_heldout_seeds="1-20",
        task=None,
        suite="libero_goal_swap",
        task_id=0,
        rollout_count=50,
        heldout_count=20,
        population_size=1000,
    )
    from scripts.evolution.prepare_libero_campaign import _load_frozen_schedule

    rollout, heldout, policy_rng, provenance = _load_frozen_schedule(args)

    assert heldout == tuple(range(1, 21))
    assert set(rollout).isdisjoint(heldout)
    assert len(rollout) == 50
    assert len(policy_rng) == 70
    assert provenance is None


@pytest.mark.parametrize(
    ("rollout", "heldout", "policy", "match"),
    [
        ([1, 1], [2, 3], {"1": 1, "2": 2, "3": 3}, "duplicate"),
        ([1, 2], [2, 3], {"1": 1, "2": 2, "3": 3}, "overlap"),
        ([1, 2], [3, 4], {"1": 1, "2": 2, "3": 3}, "exactly cover"),
    ],
)
def test_libero_campaign_rejects_invalid_reused_schedule(
    tmp_path: Path,
    rollout: list[int],
    heldout: list[int],
    policy: dict[str, int],
    match: str,
) -> None:
    import json

    source = tmp_path / "source.json"
    source.write_text(
        json.dumps(
            {
                "task": "libero_goal_task/task6",
                "rollout_seeds": rollout,
                "heldout_seeds": heldout,
                "policy_rng_by_seed": policy,
            }
        ),
        encoding="utf-8",
    )
    args = SimpleNamespace(
        schedule_from_manifest=source,
        master_seed=None,
        task=None,
        suite="libero_goal_task",
        task_id=6,
        rollout_count=2,
        heldout_count=2,
    )
    from scripts.evolution.prepare_libero_campaign import _load_frozen_schedule

    with pytest.raises(ValueError, match=match):
        _load_frozen_schedule(args)


def test_rollout_exception_traceback_preserves_remote_rpc_traceback() -> None:
    try:
        raise RpcError(
            "env.critic_chunk_step",
            "",
            traceback="Traceback (most recent call last):\\n  remote failure\\n",
        )
    except RpcError as exc:
        rendered = _exception_traceback(exc)

    assert "--- remote RPC traceback ---" in rendered
    assert "remote failure" in rendered


def test_libero_reset_timeout_allows_cold_shared_storage_initialization() -> None:
    assert _TIMEOUT_S["env.reset"] == 900.0
