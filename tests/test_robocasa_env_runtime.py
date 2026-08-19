# Copyright (c) 2026 Zetta Contributors
from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from robots.robocasa.action_contract import (
    FLAT_ACTION_SIZE,
    ActionScale,
    canonical_action,
    serializable_action,
)
from robots.robocasa.env_client import RoboCasaEnvClient
from robots.robocasa.env_server import BoundedThreadingHTTPServer, RoboCasaSession
from robots.robocasa.operation_protocol import IdempotentWriteRegistry, payload_sha256


def test_flat_vla_action_becomes_named_robocasa_mapping() -> None:
    flat = [0.1, -0.2, 0.3, 0.4, 0.0, -0.4, 1.0, 0.2, 0.3, 0.4, 0.5, 1.0]
    assert len(flat) == FLAT_ACTION_SIZE == 12
    action = canonical_action(flat)
    assert set(action) == {
        "action.end_effector_position",
        "action.end_effector_rotation",
        "action.gripper_close",
        "action.base_motion",
        "action.control_mode",
    }
    assert action["action.end_effector_position"].tolist() == pytest.approx(
        [0.1, -0.2, 0.3]
    )
    assert action["action.gripper_close"].tolist() == [1.0]


def test_action_contract_rejects_wrong_shape_and_scales_safely() -> None:
    with pytest.raises(ValueError, match="must have 12"):
        canonical_action([0.0] * 11)
    with pytest.raises(ValueError, match="within"):
        canonical_action({"gripper_close": [-1.0]})
    scaled = canonical_action(
        {"end_effector_position": [0.75, 0.0, 0.0]},
        scale=ActionScale(end_effector_position=2.0),
    )
    assert scaled["action.end_effector_position"].tolist() == [1.0, 0.0, 0.0]


class _FakeEnvironment:
    def __init__(self) -> None:
        self.actions: list[dict[str, np.ndarray]] = []

    def step(self, action: dict[str, np.ndarray]) -> tuple[Any, ...]:
        self.actions.append(action)
        observation = {
            "state.end_effector_position_relative": np.zeros(3),
            "frame": np.zeros((4, 4, 3), dtype=np.uint8),
        }
        return observation, 1.0, False, False, {"success": True}

    def close(self) -> None:
        pass


class _OfficialSignalEnvironment(_FakeEnvironment):
    def __init__(
        self,
        *,
        official_sequence: list[bool],
        reward: float,
        terminated: bool,
    ) -> None:
        super().__init__()
        self.official_sequence = list(official_sequence)
        self.reward_value = reward
        self.terminated_value = terminated
        self.current_official = False

    def _check_success(self) -> bool:
        return self.current_official

    def step(self, action: dict[str, np.ndarray]) -> tuple[Any, ...]:
        self.actions.append(action)
        if self.official_sequence:
            self.current_official = self.official_sequence.pop(0)
        observation = {
            "state.end_effector_position_relative": np.zeros(3),
            "frame": np.zeros((4, 4, 3), dtype=np.uint8),
        }
        return (
            observation,
            self.reward_value,
            self.terminated_value,
            False,
            {"success": not self.current_official},
        )


class _PreActionInterruptProgram:
    def before_action(self, *args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        return [{"rule_id": "task.pre_action", "proposal": "pause"}]

    def after_action(self, *args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        raise AssertionError("an interrupted action must not reach post-action review")


def test_server_side_chunk_is_actor_owned_and_uses_named_mapping() -> None:
    session = RoboCasaSession(
        camera_size=4,
        max_steps=10,
        require_isolated_renderer=False,
    )
    fake = _FakeEnvironment()
    session.env = fake
    session.identity = ("SlideDishwasherRack", "target")
    session.observation = {
        "state.end_effector_position_relative": np.zeros(3),
        "frame": np.zeros((4, 4, 3), dtype=np.uint8),
    }
    result = session.execute_chunk(
        {
            "actions": [[0.0] * 12],
            "critic_rules": [],
            "interrupt_on_proposal": False,
            "capture_event_images": False,
        }
    )
    assert result["executed_horizon"] == 1
    assert result["authoritative_success"] is True
    assert result["steps"][0]["terminated"] is False
    assert result["environment_write_owner"] == "robocasa_session"
    assert len(fake.actions) == 1
    assert serializable_action(fake.actions[0]) == result["steps"][0]["applied_action"]
    assert "state" in result["steps"][0]

    finalized = session.finalize_episode_artifacts()
    assert finalized["finalized"] is True
    assert session.env is fake


def test_server_task_program_interrupts_before_environment_write() -> None:
    session = RoboCasaSession(
        camera_size=4,
        max_steps=10,
        require_isolated_renderer=False,
    )
    fake = _FakeEnvironment()
    session.env = fake
    session.identity = ("SlideDishwasherRack", "target")
    session.observation = {
        "state.end_effector_position_relative": np.zeros(3),
        "frame": np.zeros((4, 4, 3), dtype=np.uint8),
    }
    session.task_program = _PreActionInterruptProgram()  # type: ignore[assignment]

    result = session.execute_chunk(
        {
            "actions": [[0.0] * 12],
            "critic_rules": [],
            "interrupt_on_proposal": True,
            "capture_event_images": False,
            "enable_task_program": True,
        }
    )

    assert result["executed_horizon"] == 0
    assert result["critic_proposals"] == [
        {"rule_id": "task.pre_action", "proposal": "pause"}
    ]
    assert fake.actions == []
    assert session.step_index == 0


def test_normal_termination_without_terminal_reward_is_not_success() -> None:
    session = RoboCasaSession(
        camera_size=4,
        max_steps=10,
        require_isolated_renderer=False,
    )
    session.terminated = True
    session.reward = 0.0
    assert session.authoritative_success is False


@pytest.mark.parametrize(
    ("official", "reward", "terminated", "expected"),
    (
        (True, 1.0, False, True),
        (False, 0.0, True, False),
        (True, 0.0, False, True),
        (False, 1.0, True, False),
    ),
)
def test_authoritative_success_uses_official_task_api_not_reward_or_termination(
    official: bool,
    reward: float,
    terminated: bool,
    expected: bool,
) -> None:
    session = RoboCasaSession(
        camera_size=4,
        max_steps=10,
        require_isolated_renderer=False,
    )
    fake = _OfficialSignalEnvironment(
        official_sequence=[official], reward=reward, terminated=terminated
    )
    session.env = fake
    session.identity = ("SlideDishwasherRack", "target")
    session.observation = {
        "state.end_effector_position_relative": np.zeros(3),
        "frame": np.zeros((4, 4, 3), dtype=np.uint8),
    }
    result = session.execute_chunk(
        {
            "actions": [[0.0] * 12],
            "critic_rules": [],
            "interrupt_on_proposal": False,
            "capture_event_images": False,
        }
    )
    assert result["official_success"] is official
    assert result["authoritative_success"] is expected


def test_official_success_is_sticky_and_stops_at_first_success() -> None:
    session = RoboCasaSession(
        camera_size=4,
        max_steps=10,
        require_isolated_renderer=False,
    )
    fake = _OfficialSignalEnvironment(
        official_sequence=[True, False], reward=0.0, terminated=False
    )
    session.env = fake
    session.identity = ("SlideDishwasherRack", "target")
    session.observation = {
        "state.end_effector_position_relative": np.zeros(3),
        "frame": np.zeros((4, 4, 3), dtype=np.uint8),
    }
    result = session.execute_chunk(
        {
            "actions": [[0.0] * 12, [0.0] * 12],
            "critic_rules": [],
            "interrupt_on_proposal": False,
            "capture_event_images": False,
        }
    )
    assert result["executed_horizon"] == 1
    assert result["success_latched"] is True
    assert result["success_first_step"] == 1
    fake.current_official = False
    assert session.authoritative_success is True


def test_nonfinite_simulator_observation_fails_hard_safety() -> None:
    session = RoboCasaSession(
        camera_size=4,
        max_steps=10,
        require_isolated_renderer=False,
    )
    fake = _OfficialSignalEnvironment(
        official_sequence=[False], reward=0.0, terminated=False
    )
    session.env = fake
    session.identity = ("SlideDishwasherRack", "target")
    session.observation = {
        "state.end_effector_position_relative": np.zeros(3),
        "frame": np.zeros((4, 4, 3), dtype=np.uint8),
    }

    def bad_step(action: dict[str, np.ndarray]) -> tuple[Any, ...]:
        return (
            {"state.bad": np.asarray([float("nan")])},
            0.0,
            False,
            False,
            {"success": False},
        )

    fake.step = bad_step  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="non-finite observation"):
        session.execute_chunk(
            {
                "actions": [[0.0] * 12],
                "critic_rules": [],
                "interrupt_on_proposal": False,
                "capture_event_images": False,
            }
        )


def test_positive_infinity_depth_background_is_not_simulation_failure() -> None:
    session = RoboCasaSession(
        camera_size=4,
        max_steps=10,
        require_isolated_renderer=False,
    )
    fake = _OfficialSignalEnvironment(
        official_sequence=[False], reward=0.0, terminated=False
    )
    session.env = fake
    session.identity = ("SlideDishwasherRack", "target")
    session.observation = {
        "state.end_effector_position_relative": np.zeros(3),
        "frame": np.zeros((4, 4, 3), dtype=np.uint8),
    }

    def background_depth_step(action: dict[str, np.ndarray]) -> tuple[Any, ...]:
        return (
            {
                "state.end_effector_position_relative": np.zeros(3),
                "video.robot0_agentview_left_depth": np.asarray(
                    [[1.0, float("inf")]], dtype=np.float32
                ),
            },
            0.0,
            False,
            False,
            {"success": False},
        )

    fake.step = background_depth_step  # type: ignore[method-assign]

    result = session.execute_chunk(
        {
            "actions": [[0.0] * 12],
            "critic_rules": [],
            "interrupt_on_proposal": False,
            "capture_event_images": False,
        }
    )
    assert result["executed_horizon"] == 1


@pytest.mark.parametrize("bad_value", [float("nan"), float("-inf")])
def test_invalid_depth_values_fail_hard_safety(bad_value: float) -> None:
    session = RoboCasaSession(
        camera_size=4,
        max_steps=10,
        require_isolated_renderer=False,
    )
    fake = _OfficialSignalEnvironment(
        official_sequence=[False], reward=0.0, terminated=False
    )
    session.env = fake
    session.identity = ("SlideDishwasherRack", "target")
    session.observation = {
        "state.end_effector_position_relative": np.zeros(3),
        "frame": np.zeros((4, 4, 3), dtype=np.uint8),
    }

    def bad_depth_step(action: dict[str, np.ndarray]) -> tuple[Any, ...]:
        return (
            {
                "state.end_effector_position_relative": np.zeros(3),
                "video.robot0_agentview_left_depth": np.asarray(
                    [[1.0, bad_value]], dtype=np.float32
                ),
            },
            0.0,
            False,
            False,
            {"success": False},
        )

    fake.step = bad_depth_step  # type: ignore[method-assign]

    with pytest.raises(RuntimeError, match="invalid depth observation"):
        session.execute_chunk(
            {
                "actions": [[0.0] * 12],
                "critic_rules": [],
                "interrupt_on_proposal": False,
                "capture_event_images": False,
            }
        )


class _RecordingEnvClient(RoboCasaEnvClient):
    def __init__(self) -> None:
        super().__init__("http://127.0.0.1:1")
        self.binding_token = "binding-" + "b" * 16
        self.writes: list[tuple[str, dict[str, Any], bool]] = []

    def _request(
        self,
        path: str,
        *,
        payload: dict[str, Any] | None = None,
        retry_transport: bool = False,
    ) -> dict[str, Any]:
        assert payload is not None
        self.writes.append((path, payload, retry_transport))
        return {
            "binding_released": path == "/release",
            "released_generation": 0 if path == "/release" else None,
            "_operation": {
                **payload["_operation"],
                "outcome": "COMMITTED",
                "side_effect_applied": True,
            },
        }


def test_client_binds_digest_and_monotonic_sequence_to_every_write() -> None:
    client = _RecordingEnvClient()
    client.reset(task="SlideDishwasherRack", seed=7)
    client.execute_chunk([[0.0] * 12])
    client.finalize_episode()
    client.release()

    reset_path, reset_payload, reset_retry = client.writes[0]
    assert reset_path == "/reset"
    assert reset_retry is True
    reset_envelope = reset_payload["_operation"]
    assert reset_envelope["operation_seq"] == 0
    assert reset_envelope["payload_sha256"] == payload_sha256(reset_payload)

    episode_id = reset_envelope["episode_id"]
    assert [item[1]["_operation"]["operation_seq"] for item in client.writes] == [
        0,
        1,
        2,
        3,
    ]
    assert {item[1]["_operation"]["episode_id"] for item in client.writes} == {
        episode_id
    }
    assert all(item[2] is True for item in client.writes)
    assert client.episode_id is None
    assert client.binding_token is None


class _RegistryEnvClient(RoboCasaEnvClient):
    def __init__(self, registry: IdempotentWriteRegistry) -> None:
        super().__init__("http://127.0.0.1:1")
        self.registry = registry

    def _request(
        self,
        path: str,
        *,
        payload: dict[str, Any] | None = None,
        retry_transport: bool = False,
    ) -> dict[str, Any]:
        del retry_transport
        if path == "/health":
            return {"write_protocol": self.registry.state}
        assert payload is not None
        terminal = self.registry.execute(
            path,
            payload,
            lambda: {"finalized": path in {"/finalize_episode", "/release"}},
            side_effect_marker=lambda: 0,
            release_binding=path == "/release",
        )
        return terminal.payload


def test_two_independent_rollout_clients_reuse_one_persistent_slot() -> None:
    registry = IdempotentWriteRegistry()
    first = _RegistryEnvClient(registry)
    first.reset(task="SlideDishwasherRack", seed=1)
    first.finalize_episode()
    first.release()
    assert registry.state["phase"] == "FREE"
    assert registry.state["generation"] == 1

    second = _RegistryEnvClient(registry)
    second.reset(task="SlideDishwasherRack", seed=2)
    assert registry.state["phase"] == "EPISODE_ACTIVE"
    assert second.session_id != first.session_id


def test_http_admission_rejects_before_spawning_excess_handler() -> None:
    class _UnusedHandler:
        pass

    class _Request:
        def __init__(self) -> None:
            self.response = b""

        def sendall(self, value: bytes) -> None:
            self.response += value

    server = BoundedThreadingHTTPServer(("127.0.0.1", 0), _UnusedHandler, limit=1)
    request = _Request()
    server.shutdown_request = lambda _request: None  # type: ignore[method-assign]
    try:
        assert server._admission.acquire(blocking=False) is True  # noqa: SLF001
        server.process_request(request, ("127.0.0.1", 1))
        assert request.response.startswith(b"HTTP/1.1 503")
        assert b"queue_full" in request.response
    finally:
        server._admission.release()  # noqa: SLF001
        server.server_close()
