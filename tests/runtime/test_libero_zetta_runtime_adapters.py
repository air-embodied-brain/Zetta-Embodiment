# Copyright (c) 2026 Zetta Contributors
"""Unit tests for the Zetta LIBERO runtime facade, against a fake
``RuntimeClient``. No Ray/GPU/simulator is required.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from rollout_runtime.adapters.zetta.runtime_env_client import (
    LiberoRuntimeEnvClient,
    RuntimeOperationError,
    SyncRuntimeLoop,
)
from rollout_runtime.adapters.zetta.runtime_policy_client import LiberoRuntimeVLAClient
from rollout_runtime.api.ids import SessionId
from rollout_runtime.api.messages import Observation, PerStepRecord, StepResult
from rollout_runtime.api.result import Err, Ok, err, ok
from rollout_runtime.api.errors import ErrorCode, make_error
from rollout_runtime.core.payload import InlineBytes, PayloadCodec, encode_array, encode_image


@pytest.fixture
def loop() -> Any:
    instance = SyncRuntimeLoop()
    yield instance
    instance.close()


def _observation(*, step_index: int, image: np.ndarray, instruction: str) -> Observation:
    return Observation(
        session_id=SessionId("session-a"),
        episode_id=1,
        step_index=step_index,
        main_image=encode_image(image),
        wrist_image=encode_image(image),
        state=[1.0, 2.0, 3.0],
        instruction=instruction,
    )


class _FakeRuntimeClient:
    """Records the last call and returns a scripted result."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, Any]] = []
        self.next_step_result: StepResult | None = None
        self.next_extension_result: Any = None

    async def reset(self, ids, spec):
        self.calls.append(("reset", spec))
        return [ok(self.next_step_result)]

    async def action_step(self, ids, actions):
        self.calls.append(("action_step", actions))
        return [ok(self.next_step_result)]

    async def extension_call(self, ids, namespace, method, args):
        self.calls.append(("extension_call", (namespace, method, args)))
        return [ok(self.next_extension_result)]

    async def policy_infer(self, ids, policy_request):
        self.calls.append(("policy_infer", policy_request))
        return [ok(self.next_extension_result)]


def test_reset_freezes_critic_rules_into_reset_spec_options(loop: Any) -> None:
    client = _FakeRuntimeClient()
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    client.next_step_result = StepResult(
        request_id="req-1",
        session_id=SessionId("session-a"),
        observation=_observation(step_index=0, image=image, instruction="pick up the cup"),
    )
    env = LiberoRuntimeEnvClient(client, SessionId("session-a"), loop=loop)

    obs, info = env.reset(
        task_id=3,
        seed=100,
        critic_rules=[{"rule_id": "r1", "feature": "episode.step_index"}],
    )

    assert client.calls[0][0] == "reset"
    spec = client.calls[0][1]
    assert spec.task_id == 3
    assert spec.seed == 100
    assert spec.options["critic_rules"] == [
        {"rule_id": "r1", "feature": "episode.step_index"}
    ]
    assert obs["task_descriptions"] == "pick up the cup"
    assert obs["states"].tolist() == [1.0, 2.0, 3.0]
    assert obs["main_images"].shape == (4, 4, 3)
    assert env.episode_terminated is False
    assert env.episode_steps == 0


def test_chunk_step_updates_episode_bookkeeping_and_decodes_observation(loop: Any) -> None:
    client = _FakeRuntimeClient()
    image = np.full((4, 4, 3), 7, dtype=np.uint8)
    client.next_step_result = StepResult(
        request_id="req-2",
        session_id=SessionId("session-a"),
        observation=_observation(step_index=3, image=image, instruction="pick up the cup"),
        reward=1.0,
        terminated=True,
        truncated=False,
        executed_horizon=3,
        info={"critic_proposals": [{"rule_id": "r1"}], "critic_rule_count": 1},
    )
    env = LiberoRuntimeEnvClient(client, SessionId("session-a"), loop=loop)
    env.reset(task_id=0, seed=0)

    obs, reward, terminated, truncated, info = env.chunk_step(
        np.zeros((5, 7), dtype=np.float32)
    )

    assert client.calls[-1][0] == "action_step"
    assert env.episode_steps == 3
    assert env.episode_terminated is True
    assert bool(np.asarray(terminated).any())
    assert info["critic_rule_count"] == 1
    assert obs["main_images"].mean() == pytest.approx(7.0)

    # Regression guard: actions must be wire-encoded PayloadRefs, not raw
    # numpy arrays (the runtime's msgpack codec rejects ndarray objects
    # outright — this exact bug shipped once and was only caught by a real
    # A100 GPU run through rollout_runtime.cli serve --launch ray, not any
    # unit test, because a hand-written fake client happily accepts anything).
    sent_actions = client.calls[-1][1]
    assert isinstance(sent_actions[0], InlineBytes)
    assert sent_actions[0].codec is PayloadCodec.RAW

    with pytest.raises(AssertionError, match="after the episode signaled"):
        env.chunk_step(np.zeros((1, 7), dtype=np.float32))


def test_critic_chunk_step_is_an_alias_for_chunk_step(loop: Any) -> None:
    client = _FakeRuntimeClient()
    image = np.zeros((2, 2, 3), dtype=np.uint8)
    client.next_step_result = StepResult(
        request_id="req-3",
        session_id=SessionId("session-a"),
        observation=_observation(step_index=1, image=image, instruction="x"),
        executed_horizon=1,
    )
    env = LiberoRuntimeEnvClient(client, SessionId("session-a"), loop=loop)
    env.reset(task_id=0, seed=0, critic_rules=[{"rule_id": "r1"}])

    obs, _reward, _term, _trunc, _info = env.critic_chunk_step(
        np.zeros((1, 7), dtype=np.float32), critic_rules=[{"rule_id": "r1"}]
    )

    assert client.calls[-1][0] == "action_step"
    assert obs["task_descriptions"] == "x"


def test_privileged_extension_calls_use_the_libero_namespace(loop: Any) -> None:
    client = _FakeRuntimeClient()
    client.next_extension_result = {"available": True, "goal_progress": 0.4}
    env = LiberoRuntimeEnvClient(client, SessionId("session-a"), loop=loop)

    result = env.privileged_critic_state(reset_tracker=True)

    assert result == {"available": True, "goal_progress": 0.4}
    _, (namespace, method, args) = client.calls[-1]
    assert namespace == "libero"
    assert method == "critic_state"
    assert args == {"reset_tracker": True}


def test_raw_obs_decodes_array_payloads_and_keeps_scalars(loop: Any) -> None:
    client = _FakeRuntimeClient()
    client.next_extension_result = {
        "available": True,
        "scalars": {"task_language": "pick up the cup"},
        "arrays": {"robot0_eef_quat": encode_array(np.array([0.0, 0.0, 0.0, 1.0]))},
    }
    env = LiberoRuntimeEnvClient(client, SessionId("session-a"), loop=loop)

    result = env.raw_obs()

    assert result["task_language"] == "pick up the cup"
    assert np.asarray(result["robot0_eef_quat"]).tolist() == [0.0, 0.0, 0.0, 1.0]


def test_raw_obs_raises_when_unavailable(loop: Any) -> None:
    client = _FakeRuntimeClient()
    client.next_extension_result = {"available": False, "reason": "not reset yet"}
    env = LiberoRuntimeEnvClient(client, SessionId("session-a"), loop=loop)

    with pytest.raises(RuntimeOperationError, match="not reset yet"):
        env.raw_obs()


def test_runtime_operation_error_on_err_result(loop: Any) -> None:
    client = _FakeRuntimeClient()

    async def reset(ids, spec):
        return [err(make_error(ErrorCode.ENV_FAILURE, "simulator crashed"))]

    client.reset = reset  # type: ignore[method-assign]
    env = LiberoRuntimeEnvClient(client, SessionId("session-a"), loop=loop)

    with pytest.raises(RuntimeOperationError, match="ENV_FAILURE"):
        env.reset(task_id=0, seed=0)


def test_predict_action_batch_returns_chunk_and_metadata(loop: Any) -> None:
    client = _FakeRuntimeClient()

    class _FakePolicyResult:
        actions = encode_array(np.zeros((5, 7), dtype=np.float32))
        model_version = "pi05-libero-sft"
        observation_step_index = 4
        auxiliary_outputs: dict[str, Any] = {}
        info = {"policy_id": "pi05"}

    client.next_extension_result = _FakePolicyResult()
    model = LiberoRuntimeVLAClient(client, SessionId("session-a"), loop=loop, policy_id="pi05")

    actions, metadata = model.predict_action_batch({"states": np.zeros(8)}, mode="eval")

    assert actions.shape == (5, 7)
    assert metadata["horizon"] == 5
    assert metadata["policy_id"] == "pi05"
    assert client.calls[-1][0] == "policy_infer"
    request = client.calls[-1][1]
    assert request.inference_parameters["mode"] == "eval"
