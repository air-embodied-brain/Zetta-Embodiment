# Copyright (c) 2026 Zetta Contributors
"""Contract tests for the Cosmos-Lite remote policy backend."""

from __future__ import annotations

import dataclasses
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from rollout_runtime.api.enums import ErrorCode
from rollout_runtime.api.ids import EpisodeId, OperationSeq, RequestId, SessionId
from rollout_runtime.api.internal import InferenceRequest
from rollout_runtime.api.messages import Observation
from rollout_runtime.backends import build_policy_core, policy_compat_constraints
from rollout_runtime.backends.cosmos_lite import (
    COSMOS_LITE_V030_REVISION,
    CosmosLitePolicyConfig,
    CosmosLitePolicyCore,
    _pack_message,
    _unpack_message,
)
from rollout_runtime.config.schema import load_config
from rollout_runtime.core import payload as payload_module

FIXTURE = (
    Path(__file__).parent
    / "fixtures"
    / "cosmos_lite"
    / "resolved_deployment_config.json"
)
MANIFEST_SHA256 = "a" * 64


class _FakeTransport:
    """In-memory stand-in for one OpenPI WebSocket connection."""

    def __init__(self) -> None:
        self.connected = False
        self.closed = False
        self.connect_calls = 0
        self.infer_calls: list[tuple[dict[str, Any], float]] = []
        self.metadata: dict[str, Any] = {}
        self.response: Any = {
            "action": np.zeros((32, 8), dtype=np.float32),
            "server_timing": {"infer_ms": 12.5},
        }
        self.failure: BaseException | None = None

    def connect(self) -> dict[str, Any]:
        self.connect_calls += 1
        self.connected = True
        return dict(self.metadata)

    def infer(self, observation: dict[str, Any], *, timeout_s: float) -> dict[str, Any]:
        self.infer_calls.append((observation, timeout_s))
        if self.failure is not None:
            raise self.failure
        return self.response

    def close(self) -> None:
        self.closed = True
        self.connected = False


def _config(**overrides: Any) -> CosmosLitePolicyConfig:
    values: dict[str, Any] = {
        "resolved_config_path": str(FIXTURE),
        "expected_manifest_sha256": MANIFEST_SHA256,
        "image_layout": "single",
    }
    values.update(overrides)
    return CosmosLitePolicyConfig(**values)


def _request(index: int = 0, **overrides: Any) -> InferenceRequest:
    image = payload_module.encode_image(np.full((4, 6, 3), index, dtype=np.uint8))
    observation = Observation(
        session_id=SessionId(f"sess-{index}"),
        episode_id=EpisodeId(1),
        step_index=index,
        main_image=image,
        wrist_image=payload_module.encode_image(
            np.full((4, 6, 3), index + 1, dtype=np.uint8)
        ),
        extra_view_images=[
            payload_module.encode_image(np.full((4, 6, 3), index + 2, dtype=np.uint8))
        ],
        state=[float(value) for value in range(8)],
        instruction="put the banana in the bowl",
    )
    values: dict[str, Any] = {
        "request_id": RequestId(f"req-{index}"),
        "session_id": SessionId(f"sess-{index}"),
        "episode_id": EpisodeId(1),
        "operation_seq": OperationSeq(index + 1),
        "policy_id": "cosmos_lite",
        "observation": observation,
        "routing_token": "env:0",
        "compat_key": "compat",
    }
    values.update(overrides)
    return InferenceRequest(**values)


def _loaded_core(transport: _FakeTransport, **overrides: Any) -> CosmosLitePolicyCore:
    core = CosmosLitePolicyCore(
        _config(**overrides), transport_factory=lambda _config: transport
    )
    core.load()
    return core


def test_openpi_msgpack_numpy_contract_round_trips_arrays() -> None:
    request = {
        "prompt": "move",
        "observation/image": np.arange(18, dtype=np.uint8).reshape(2, 3, 3),
        "observation/joint_position": np.arange(7, dtype=np.float32),
    }
    decoded = _unpack_message(_pack_message(request))
    assert decoded["prompt"] == "move"
    np.testing.assert_array_equal(
        decoded["observation/image"], request["observation/image"]
    )
    np.testing.assert_array_equal(
        decoded["observation/joint_position"],
        request["observation/joint_position"],
    )


def test_load_verifies_identity_and_connects() -> None:
    transport = _FakeTransport()
    core = _loaded_core(transport)
    assert transport.connect_calls == 1
    assert core.loaded is True
    assert core.model_version.startswith("cosmos-lite:cosmos3_policy:")
    core.load()
    assert transport.connect_calls == 1
    core.close()
    assert transport.closed is True


def test_interface_contract_changes_the_derived_model_version() -> None:
    single = _loaded_core(_FakeTransport(), image_layout="single")
    three_view = _loaded_core(_FakeTransport(), image_layout="robolab_three_view")
    try:
        assert single.model_version != three_view.model_version
    finally:
        single.close()
        three_view.close()


def test_model_version_cannot_override_verified_identity() -> None:
    with pytest.raises(ValueError, match="derived from its verified deployment"):
        build_policy_core(
            backend="cosmos_lite",
            policy_config=dataclasses.asdict(_config()),
            model_version="manual-label",
        )


def test_identity_mismatch_fails_before_connecting(tmp_path: Path) -> None:
    record = json.loads(FIXTURE.read_text(encoding="utf-8"))
    record["model"]["manifest_sha256"] = "b" * 64
    resolved = tmp_path / "resolved.json"
    resolved.write_text(json.dumps(record), encoding="utf-8")
    transport = _FakeTransport()
    core = CosmosLitePolicyCore(
        _config(resolved_config_path=str(resolved)),
        transport_factory=lambda _config: transport,
    )
    with pytest.raises(ValueError, match="manifest mismatch"):
        core.load()
    assert transport.connect_calls == 0


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda record: record["repository"].update({"dirty": True}),
            "repository must be clean",
        ),
        (
            lambda record: record.update({"fallback_decisions": ["use_sdpa"]}),
            "contains runtime fallbacks",
        ),
        (
            lambda record: record["effective"]["sampling"].update(
                {"deterministic_seed": False}
            ),
            "deterministic_seed=true",
        ),
        (
            lambda record: record["effective"]["sampling"].update(
                {"deterministic_seed": "false"}
            ),
            "deterministic_seed=true",
        ),
        (
            lambda record: record["bundle"].update({"manifest_sha256": "b" * 64}),
            "bundle identity does not match",
        ),
        (
            lambda record: record["runtime_probe"].update({"cuda_available": False}),
            "cuda_available=true",
        ),
    ],
)
def test_unverified_deployment_is_rejected_before_connecting(
    tmp_path: Path, mutate: Any, message: str
) -> None:
    record = json.loads(FIXTURE.read_text(encoding="utf-8"))
    mutate(record)
    resolved = tmp_path / "resolved.json"
    resolved.write_text(json.dumps(record), encoding="utf-8")
    transport = _FakeTransport()
    core = CosmosLitePolicyCore(
        _config(resolved_config_path=str(resolved)),
        transport_factory=lambda _config: transport,
    )
    with pytest.raises(ValueError, match=message):
        core.load()
    assert transport.connect_calls == 0


def test_single_view_request_maps_to_upstream_and_returns_actions() -> None:
    transport = _FakeTransport()
    core = _loaded_core(transport)
    request = _request(
        instruction_override="override prompt",
        inference_parameters={"mode": "eval", "seed": 0, "num_steps": 2},
    )
    response = core.infer_batch([request])[0]
    assert response.error is None
    assert response.model_version == core.model_version
    actions = payload_module.decode_payload(response.actions)
    assert actions.shape == (32, 8)
    assert actions.dtype == np.float32
    upstream, timeout_s = transport.infer_calls[0]
    assert upstream["prompt"] == "override prompt"
    assert upstream["observation/image"].shape == (4, 6, 3)
    np.testing.assert_array_equal(
        upstream["observation/joint_position"], np.arange(7, dtype=np.float32)
    )
    np.testing.assert_array_equal(
        upstream["observation/gripper_position"], np.array([7], dtype=np.float32)
    )
    assert timeout_s == pytest.approx(120.0)
    assert response.auxiliary_outputs["server_timing"] == {"infer_ms": 12.5}
    assert (
        response.auxiliary_outputs["cosmos_lite_identity"]["manifest_sha256"]
        == MANIFEST_SHA256
    )


def test_three_view_layout_uses_upstream_robolab_keys() -> None:
    transport = _FakeTransport()
    core = _loaded_core(transport, image_layout="robolab_three_view")
    response = core.infer_batch([_request()])[0]
    assert response.error is None
    upstream = transport.infer_calls[0][0]
    assert "observation/image" not in upstream
    assert upstream["observation/wrist_image_left"][0, 0, 0] == 1
    assert upstream["observation/exterior_image_1_left"][0, 0, 0] == 0
    assert upstream["observation/exterior_image_2_left"][0, 0, 0] == 2


@pytest.mark.parametrize(
    ("response", "message"),
    [
        ({}, "no 'action'"),
        ({"action": np.zeros((31, 8), dtype=np.float32)}, "shape mismatch"),
        ({"action": np.full((32, 8), np.nan, dtype=np.float32)}, "NaN or Inf"),
    ],
)
def test_invalid_service_response_is_policy_failure(
    response: dict[str, Any], message: str
) -> None:
    transport = _FakeTransport()
    transport.response = response
    core = _loaded_core(transport)
    result = core.infer_batch([_request()])[0]
    assert result.error is not None
    assert result.error.code is ErrorCode.POLICY_FAILURE
    assert message in result.error.message


def test_invalid_request_isolated_without_calling_service() -> None:
    transport = _FakeTransport()
    core = _loaded_core(transport)
    original = _request().observation
    invalid = dataclasses.replace(original, state=[0.0] * 7)
    result = core.infer_batch([_request(observation=invalid)])[0]
    assert result.error is not None
    assert result.error.code is ErrorCode.INVALID_ARGUMENT
    assert transport.infer_calls == []


def test_non_numeric_state_is_invalid_argument() -> None:
    transport = _FakeTransport()
    core = _loaded_core(transport)
    original = _request().observation
    invalid = dataclasses.replace(original, state=["bad"] * 8)
    result = core.infer_batch([_request(observation=invalid)])[0]
    assert result.error is not None
    assert result.error.code is ErrorCode.INVALID_ARGUMENT
    assert "state is not numeric" in result.error.message
    assert transport.infer_calls == []


def test_timeout_is_deadline_exceeded_and_is_not_retried() -> None:
    transport = _FakeTransport()
    transport.failure = TimeoutError("server timed out")
    core = _loaded_core(transport)
    result = core.infer_batch([_request(deadline=time.time() + 1.0)])[0]
    assert result.error is not None
    assert result.error.code is ErrorCode.DEADLINE_EXCEEDED
    assert len(transport.infer_calls) == 1


def test_server_fixed_sampling_parameter_mismatch_is_rejected() -> None:
    transport = _FakeTransport()
    core = _loaded_core(transport)
    result = core.infer_batch([_request(inference_parameters={"guidance": 4.0})])[0]
    assert result.error is not None
    assert result.error.code is ErrorCode.INVALID_ARGUMENT
    assert "fixed by the Cosmos-Lite service" in result.error.message


def test_insecure_non_loopback_endpoint_is_rejected() -> None:
    with pytest.raises(ValueError, match="restricted to loopback"):
        _config(endpoint="ws://10.0.0.8:8000")


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("device", "cuda:0", "requires device='cpu'"),
        ("dtype", "float16", "requires dtype='float32'"),
    ],
)
def test_remote_client_execution_contract_is_fixed(
    key: str, value: str, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _config(**{key: value})


def test_backend_registration_and_constraints() -> None:
    config = dataclasses.asdict(_config())
    core = build_policy_core(
        backend="cosmos_lite",
        policy_config=config,
        device="cpu",
        dtype="float32",
        policy_family="cosmos_lite",
        action_dim=8,
        actions_per_chunk=32,
    )
    assert isinstance(core, CosmosLitePolicyCore)
    constraints = policy_compat_constraints(backend="cosmos_lite", policy_config=config)
    assert constraints["action_dim"] == 8
    assert constraints["actions_per_chunk"] == 32
    assert constraints["expected_manifest_sha256"] == MANIFEST_SHA256


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("policy_family", "fake", "policy_family='cosmos_lite'"),
        ("num_ranks", 2, "requires num_ranks=1"),
        ("max_concurrent_inferences", 2, "requires max_concurrent_inferences=1"),
        ("scheduler.max_batch_size", 2, "scheduler.max_batch_size=1"),
        ("scheduler.max_wait_ms", 1.0, "scheduler.max_wait_ms=0"),
    ],
)
def test_runtime_config_requires_single_request_scheduling(
    field: str, value: Any, message: str
) -> None:
    rollout = {
        "policy_id": "cosmos_lite",
        "policy_family": "cosmos_lite",
        "policy_backend": "cosmos_lite",
        "num_ranks": 1,
        "max_concurrent_inferences": 1,
        "policy_config": {
            "resolved_config_path": str(FIXTURE),
            "expected_manifest_sha256": MANIFEST_SHA256,
        },
        "scheduler": {"max_batch_size": 1, "max_wait_ms": 0.0},
    }
    if field.startswith("scheduler."):
        rollout["scheduler"][field.removeprefix("scheduler.")] = value
    else:
        rollout[field] = value
    with pytest.raises(ValueError, match=message):
        load_config({"rollout_worker": rollout})


def test_hot_weight_update_is_rejected() -> None:
    transport = _FakeTransport()
    core = _loaded_core(transport)
    with pytest.raises(RuntimeError, match="does not support runtime weight updates"):
        core.update_weights("new-version")


def test_config_is_pinned_to_cosmos_lite_v030() -> None:
    config = _config()
    assert config.expected_repository_revision == COSMOS_LITE_V030_REVISION


def test_packaged_preset_selects_cosmos_lite_backend() -> None:
    config = load_config("cosmos_lite_remote")
    assert config.rollout_worker.policy_backend == "cosmos_lite"
    assert config.rollout_worker.policy_family == "cosmos_lite"
    assert config.rollout_worker.device == "cpu"
    assert config.rollout_worker.dtype == "float32"
    assert config.rollout_worker.scheduler.max_batch_size == 1
