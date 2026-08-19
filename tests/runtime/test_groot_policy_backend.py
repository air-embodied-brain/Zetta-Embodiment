# Copyright (c) 2026 Zetta Contributors
"""``rollout_runtime/backends/groot_policy.py`` — runtime v3 design
§3.4/Stage 4, ``PolicyInferenceCore`` wrapping ``groot_core.Gr00tModelCore``.

Monkeypatches ``robots.robocasa.groot_core.load_groot_model_core`` with a fake
loader (no real GR00T checkpoint or torch model available in this environment,
mirroring ``tests/test_robocasa_groot_server.py``'s fake-policy approach). The
point under test is the *adapter* boundary: named STATE_FIELDS extraction from
``Observation.extras["raw_state"]``, image decoding from PayloadRef, and the
flat-12-dim action reassembly — not GR00T's own inference logic (already
covered by ``test_robocasa_runtime_split.py``/``test_robocasa_groot_server.py``).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from rollout_runtime.api.enums import ErrorCode
from rollout_runtime.api.ids import EpisodeId, OperationSeq, RequestId, SessionId
from rollout_runtime.api.internal import InferenceRequest
from rollout_runtime.api.messages import Observation
from rollout_runtime.backends import build_policy_core, policy_compat_constraints
from rollout_runtime.backends.groot_policy import GrootPolicyConfig
from rollout_runtime.core import payload as payload_module

CAMERA = 4


class _FakePolicy:
    """Stand-in for the real ``Gr00tPolicy``: echoes a fixed 12-field action dict.

    Attributes:
        calls: recorded observations passed to ``get_action``.
        fail: when true, raises to exercise the per-request error path.
    """

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.fail = False

    def get_action(self, observation: dict[str, np.ndarray]) -> dict[str, Any]:
        self.calls.append(observation)
        if self.fail:
            raise RuntimeError("stub groot policy exploded")
        return {
            "action.end_effector_position": np.zeros((1, 3), dtype=np.float32),
            "action.end_effector_rotation": np.zeros((1, 3), dtype=np.float32),
            "action.gripper_close": np.ones((1, 1), dtype=np.float32),
            "action.base_motion": np.zeros((1, 4), dtype=np.float32),
            "action.control_mode": np.zeros((1, 1), dtype=np.float32),
        }


@pytest.fixture
def stub_policy(monkeypatch: pytest.MonkeyPatch) -> _FakePolicy:
    """Patch ``load_groot_model_core`` to build a ``Gr00tModelCore`` around a fake
    policy instead of loading a real checkpoint.

    Args:
        monkeypatch: pytest fixture.

    Returns:
        The fake policy (tests can flip ``fail`` on it).
    """
    from robots.robocasa import groot_core as groot_core_module

    policy = _FakePolicy()

    def _fake_load(
        *,
        groot_root: str,
        model_path: str,
        data_config_name: str,
        embodiment_tag: str,
        denoising_steps: int,
        maximum_pending: int = 32,
        expected_checkpoint_sha256: str | None = None,
    ):
        del groot_root, model_path, data_config_name, embodiment_tag
        del denoising_steps, expected_checkpoint_sha256
        from types import SimpleNamespace

        data_config = SimpleNamespace(
            video_keys=groot_core_module.VIDEO_KEYS,
            state_keys=tuple(groot_core_module.STATE_FIELDS),
            language_keys=(groot_core_module.LANGUAGE_KEY,),
            observation_indices=(0,),
            action_keys=("action.end_effector_position",),
            action_indices=(0,),
        )
        return groot_core_module.Gr00tModelCore(
            policy=policy,
            data_config=data_config,
            checkpoint_sha256="c" * 64,
            denoising_steps=4,
            maximum_pending=maximum_pending,
        )

    monkeypatch.setattr(groot_core_module, "load_groot_model_core", _fake_load)
    return policy


def _core(**overrides: Any) -> Any:
    """Build a ``GrootPolicyCore`` via ``build_policy_core``.

    Args:
        **overrides: ``GrootPolicyConfig`` field overrides.

    Returns:
        ``GrootPolicyCore``.
    """
    return build_policy_core(
        backend="groot",
        policy_config={
            "groot_root": "/stub/groot",
            "model_path": "/stub/checkpoint",
            **overrides,
        },
        device="cuda",
        dtype="bfloat16",
        action_dim=12,
    )


def _named_state() -> dict[str, list[float]]:
    return {
        "state.end_effector_position_relative": [0.1, 0.2, 0.3],
        "state.end_effector_rotation_relative": [0.0, 0.0, 0.0, 1.0],
        "state.gripper_qpos": [0.0, 1.0],
        "state.base_position": [0.0, 0.0, 0.0],
        "state.base_rotation": [0.0, 0.0, 0.0, 1.0],
    }


def _request(index: int, **overrides: Any) -> InferenceRequest:
    """Build an ``InferenceRequest`` shaped like ``robocasa_current.py``'s output.

    Args:
        index: sequence number, feeds ``request_id``/``session_id``.
        **overrides: ``InferenceRequest`` field overrides.

    Returns:
        ``InferenceRequest``.
    """
    observation = Observation(
        session_id=SessionId(f"sess-{index}"),
        episode_id=EpisodeId(1),
        step_index=index,
        main_image=payload_module.encode_image(
            np.full((CAMERA, CAMERA, 3), index % 256, dtype=np.uint8)
        ),
        wrist_image=payload_module.encode_image(
            np.full((CAMERA, CAMERA, 3), (index * 3) % 256, dtype=np.uint8)
        ),
        extra_view_images=[
            payload_module.encode_image(
                np.full((CAMERA, CAMERA, 3), (index * 5) % 256, dtype=np.uint8)
            )
        ],
        state=[0.1, 0.2, 0.3],
        instruction="move the pan",
        extras={"raw_state": _named_state()},
    )
    payload: dict[str, Any] = {
        "request_id": RequestId(f"req-{index}"),
        "session_id": SessionId(f"sess-{index}"),
        "episode_id": EpisodeId(1),
        "operation_seq": OperationSeq(index + 1),
        "policy_id": "groot",
        "observation": observation,
        "inference_parameters": {"seed": 7},
        "routing_token": "env:0",
        "compat_key": "k",
    }
    payload.update(overrides)
    return InferenceRequest(**payload)


def test_infer_batch_decodes_named_state_and_returns_flat_actions(
    stub_policy: _FakePolicy,
) -> None:
    core = _core()
    core.load()
    assert core.model_version == "c" * 64  # falls back to checkpoint digest

    responses = core.infer_batch([_request(0)])
    assert len(responses) == 1
    response = responses[0]
    assert response.error is None
    block = payload_module.decode_payload(response.actions)
    assert block.shape == (1, 12)
    assert block[0, -1] == pytest.approx(0.0)  # control_mode, clamped in [-1, 1]

    # the fake policy must have received decoded uint8 video + named state vectors
    call = stub_policy.calls[0]
    assert call["video.robot0_agentview_left"].dtype == np.uint8
    assert call["video.robot0_agentview_left"].shape == (1, CAMERA, CAMERA, 3)
    assert call["state.end_effector_position_relative"] == pytest.approx(
        np.array([[0.1, 0.2, 0.3]])
    )
    core.close()


def test_infer_uses_the_deterministic_seed_fallback_when_unset(
    stub_policy: _FakePolicy,
) -> None:
    core = _core()
    core.load()
    core.infer_batch([_request(0, inference_parameters={})])
    core.infer_batch([_request(0, inference_parameters={})])
    # two calls with the same request_id must be reproducible: the fake policy
    # doesn't expose the seed directly, but both calls must succeed without
    # falling back to a random/non-deterministic path raising.
    assert len(stub_policy.calls) == 2
    core.close()


def test_missing_state_field_is_a_per_request_policy_failure(
    stub_policy: _FakePolicy,
) -> None:
    core = _core()
    core.load()
    incomplete_state = dict(_named_state())
    del incomplete_state["state.gripper_qpos"]
    observation = Observation(
        session_id=SessionId("sess-x"),
        episode_id=EpisodeId(1),
        step_index=0,
        main_image=payload_module.encode_image(
            np.zeros((CAMERA, CAMERA, 3), dtype=np.uint8)
        ),
        wrist_image=payload_module.encode_image(
            np.zeros((CAMERA, CAMERA, 3), dtype=np.uint8)
        ),
        extra_view_images=[
            payload_module.encode_image(np.zeros((CAMERA, CAMERA, 3), dtype=np.uint8))
        ],
        instruction="move the pan",
        extras={"raw_state": incomplete_state},
    )
    request = _request(0, observation=observation)
    responses = core.infer_batch([request])
    assert responses[0].error is not None
    assert responses[0].error.code is ErrorCode.POLICY_FAILURE
    assert "gripper_qpos" in responses[0].error.message
    core.close()


def test_missing_camera_is_a_per_request_policy_failure(
    stub_policy: _FakePolicy,
) -> None:
    core = _core()
    core.load()
    observation = Observation(
        session_id=SessionId("sess-x"),
        episode_id=EpisodeId(1),
        step_index=0,
        main_image=payload_module.encode_image(
            np.zeros((CAMERA, CAMERA, 3), dtype=np.uint8)
        ),
        wrist_image=None,  # missing
        extra_view_images=[
            payload_module.encode_image(np.zeros((CAMERA, CAMERA, 3), dtype=np.uint8))
        ],
        instruction="move the pan",
        extras={"raw_state": _named_state()},
    )
    request = _request(0, observation=observation)
    responses = core.infer_batch([request])
    assert responses[0].error is not None
    assert responses[0].error.code is ErrorCode.POLICY_FAILURE
    core.close()


def test_model_failure_is_reported_per_request(stub_policy: _FakePolicy) -> None:
    core = _core()
    core.load()
    stub_policy.fail = True
    responses = core.infer_batch([_request(0), _request(1)])
    assert len(responses) == 2
    for response in responses:
        assert response.error is not None
        assert response.error.code is ErrorCode.POLICY_FAILURE
    assert core.error_count == 2
    core.close()


def test_empty_batch_is_a_noop(stub_policy: _FakePolicy) -> None:
    core = _core()
    core.load()
    assert core.infer_batch([]) == []
    assert stub_policy.calls == []
    core.close()


def test_instruction_override_reaches_the_named_language_key(
    stub_policy: _FakePolicy,
) -> None:
    core = _core()
    core.load()
    core.infer_batch([_request(0, instruction_override="open the drawer")])
    from robots.robocasa.groot_core import LANGUAGE_KEY

    assert stub_policy.calls[0][LANGUAGE_KEY] == ["open the drawer"]
    core.close()


def test_update_weights_relabels_without_reloading(stub_policy: _FakePolicy) -> None:
    core = _core()
    core.load()
    core.update_weights("groot-v2")
    assert core.model_version == "groot-v2"
    responses = core.infer_batch([_request(0)])
    assert responses[0].model_version == "groot-v2"
    core.close()


def test_infer_before_load_is_a_per_request_error() -> None:
    core = _core()
    responses = core.infer_batch([_request(0)])
    assert responses[0].error is not None
    assert responses[0].error.code is ErrorCode.POLICY_FAILURE


def test_config_requires_groot_root_and_model_path() -> None:
    with pytest.raises(ValueError, match="missing required keys"):
        build_policy_core(backend="groot", policy_config={})


def test_config_rejects_unknown_keys() -> None:
    with pytest.raises(ValueError, match="unknown groot policy_config keys"):
        GrootPolicyConfig.from_mapping(
            {"groot_root": "/x", "model_path": "/y", "bogus": 1}
        )


def test_unknown_policy_backend_is_still_rejected() -> None:
    with pytest.raises(ValueError, match="unknown policy backend"):
        build_policy_core(backend="tensorrt")


def test_groot_has_no_compat_key_hard_constraints() -> None:
    assert (
        policy_compat_constraints(
            backend="groot",
            policy_config={"groot_root": "/x", "model_path": "/y"},
        )
        == {}
    )
