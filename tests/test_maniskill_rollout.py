# Copyright (c) 2026 Zetta Contributors
"""Control-step rollout, recovery authority and infrastructure classification."""

from __future__ import annotations

import dataclasses
import json
from types import SimpleNamespace as NS

import numpy as np
import pytest

from robots.maniskill.contracts import TaskContract
from robots.maniskill.role1 import BoundedRole1
from robots.maniskill.run_rollout import build_parser, run
from rollout_runtime.api.result import Ok
from rollout_runtime.backends.rlinf_maniskill import ManiskillEnvConfig
from rollout_runtime.core.payload import decode_array, encode_array
from zetta.envs.maniskill.contracts import ACTION_CONTRACT
from zetta.evolution.jsonio import atomic_write_json
from zetta.evolution.models import (
    CandidateBundle,
    CriticRule,
    RecoveryRule,
    RecoveryStep,
)


def bundle(budget=2):
    return CandidateBundle(
        candidate_id="wiring-only",
        generation=0,
        parent_sha256=None,
        diagnosis_sha256="a" * 64,
        causal_hypothesis="test wiring",
        mechanism_change="hold at step 1",
        validation_plan="unit test only",
        critic_rules=(
            CriticRule(
                rule_id="step",
                title="test",
                feature="episode.step",
                operator="ge",
                threshold=1,
                dwell_steps=1,
                cooldown_steps=100,
                proposal="hold",
                evidence_ids=("test",),
            ),
        ),
        recovery_rules=(
            RecoveryRule(
                recovery_id="hold",
                title="test",
                trigger_rule_ids=("step",),
                precondition="test",
                steps=(
                    RecoveryStep(
                        tool="maniskill.hold",
                        parameters={"max_steps": budget},
                        stop_when="budget_exhausted_or_success",
                    ),
                ),
                safety_constraints=("bounded",),
                stop_condition="official_success_or_budget",
                fallback="resume_vla",
                evidence_ids=("test",),
            ),
        ),
    )


class Client:
    def __init__(self, task, *, bad_ack=None, success=False):
        self.task, self.bad_ack, self.success = task, bad_ack, success
        self.steps = 0
        self.actions = []
        self.inferences = []
        self.closed = False

    def observation(self):
        return NS(
            step_index=self.steps,
            state=[0.0] * 18,
            instruction=self.task.instruction,
            extras={
                "mani_skill_version": "3.0.1",
                "sapien_version": "3.0.2",
                "control_frequency": 20,
                "simulation_frequency": 100,
                "initial_state_sha256": "f" * 64,
                "action_contract": ACTION_CONTRACT,
                "observation_contract": "test-observation",
            },
        )

    async def create_sessions(self, requests):
        return [Ok(NS(session_id="one"))]

    async def reset(self, ids, spec):
        return [
            Ok(NS(observation=self.observation(), terminated=False, truncated=False))
        ]

    async def policy_infer(self, ids, request):
        self.inferences.append(self.steps)
        ack = {
            "policy_rng_acknowledged": True,
            "checkpoint_sha256": self.task.checkpoint_sha256,
            "norm_stats_sha256": self.task.norm_stats_sha256,
            "action_contract": ACTION_CONTRACT,
            "observation_contract": "test-observation",
        }
        if self.bad_ack:
            ack[self.bad_ack] = None
        actions = np.zeros((request.actions_per_chunk, 7), dtype=np.float32)
        actions[:, 0] = 0.25
        actions[:, -1] = -1
        return [
            Ok(
                NS(
                    model_version=self.task.policy_model_version,
                    auxiliary_outputs=ack,
                    observation_step_index=self.steps,
                    actions=encode_array(actions),
                )
            )
        ]

    async def action_step(self, ids, actions):
        self.actions.append(decode_array(actions[0])[0])
        self.steps += 1
        ended = self.steps == 4
        return [
            Ok(
                NS(
                    observation=self.observation(),
                    executed_horizon=1,
                    reward=0.0,
                    terminated=ended and self.success,
                    truncated=ended and not self.success,
                    info={"success": ended and self.success, "fail": False},
                )
            )
        ]

    async def close_sessions(self, ids):
        self.closed = True
        return [Ok(None)]


def args_for(tmp_path, candidate=None):
    config = ManiskillEnvConfig(
        robot_uids="panda",
        goal_visibility="visible",
        return_all_frames=True,
        extra_init_params={"enhanced_determinism": True, "reconfiguration_freq": 1},
    )
    task = TaskContract(
        policy_model_version="fixture",
        checkpoint_sha256="a" * 64,
        norm_stats_sha256="b" * 64,
    )
    atomic_write_json(tmp_path / "env.json", dataclasses.asdict(config))
    atomic_write_json(tmp_path / "task.json", task.as_dict())
    arguments = [
        "--runtime-url",
        "http://localhost:1",
        "--env-config",
        str(tmp_path / "env.json"),
        "--task-contract",
        str(tmp_path / "task.json"),
        "--seed",
        "31",
        "--policy-rng",
        "42",
        "--output-dir",
        str(tmp_path / "episode"),
        "--no-capture-frames",
    ]
    if candidate:
        atomic_write_json(tmp_path / "bundle.json", candidate.as_dict())
        arguments += [
            "--bundle",
            str(tmp_path / "bundle.json"),
            "--bundle-sha256",
            candidate.sha256,
        ]
    return build_parser().parse_args(arguments), task


@pytest.mark.parametrize("success", [False, True])
async def test_baseline_uses_official_result_and_closes(tmp_path, success):
    args, task = args_for(tmp_path)
    client = Client(task, success=success)
    result = await run(args, client=client)
    assert result.status == "valid", result.invalid_reason
    assert result.success is success
    assert client.closed and len(client.actions) == 4
    assert result.artifact_index["candidate_intervention"] is False


async def test_recovery_clears_pending_policy_actions_and_preserves_gripper(tmp_path):
    args, task = args_for(tmp_path, bundle())
    client = Client(task)
    result = await run(args, client=client)
    assert result.status == "valid", result.invalid_reason
    assert result.artifact_index["recovery_steps_executed"] == 2
    assert client.inferences == [0, 3]
    np.testing.assert_array_equal(client.actions[1], [0, 0, 0, 0, 0, 0, -1])
    events = [
        json.loads(line)
        for line in (tmp_path / "episode/tools.jsonl").read_text().splitlines()
    ]
    assert events[0]["accepted"] and events[0]["environment_write"] is False


async def test_role1_rejects_over_budget_without_recovery(tmp_path):
    args, task = args_for(tmp_path, bundle(50))
    client = Client(task)
    result = await run(args, client=client)
    assert result.status == "valid", result.invalid_reason
    assert result.artifact_index["candidate_intervention"] is False
    assert all(action[0] == 0.25 for action in client.actions)
    assert (
        BoundedRole1().review(
            [{"rule_id": "step"}], bundle(2).recovery_rules, remaining_steps=1
        )["accepted"]
        is False
    )


@pytest.mark.parametrize(
    "bad_ack",
    [
        "policy_rng_acknowledged",
        "action_contract",
        "observation_contract",
        "checkpoint_sha256",
        "norm_stats_sha256",
    ],
)
async def test_policy_mismatch_is_infrastructure_invalid(tmp_path, bad_ack):
    args, task = args_for(tmp_path)
    client = Client(task, bad_ack=bad_ack)
    result = await run(args, client=client)
    assert result.status == "infra_invalid" and result.success is None
    assert client.closed and not client.actions


async def test_tampered_task_does_not_open_environment(tmp_path):
    args, task = args_for(tmp_path)
    args.task_contract_sha256 = "0" * 64
    client = Client(task)
    result = await run(args, client=client)
    assert result.status == "infra_invalid"
    assert "digest mismatch" in result.invalid_reason
    assert not client.closed and not client.actions
