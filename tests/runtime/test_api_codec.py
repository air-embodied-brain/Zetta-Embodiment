"""Protocol codec tests.

Assertion focus: msgpack round-trip fidelity, ``EnvSpecMsg.digest()`` stability,
and error normalization.
"""

from __future__ import annotations

import dataclasses

import pytest

from rollout_runtime.api import codec, wire
from rollout_runtime.api.enums import (
    EnvOperation,
    ErrorCode,
    OperationState,
    Priority,
    SessionState,
)
from rollout_runtime.api.errors import (
    RETRYABLE_ERROR_CODES,
    InvalidTransition,
    RuntimeApiError,
    make_error,
    normalize_exception,
)
from rollout_runtime.api.ids import (
    BindingToken,
    EpisodeId,
    OperationSeq,
    RequestId,
    SessionId,
    new_request_id,
    new_session_id,
)
from rollout_runtime.api.internal import (
    ActionResponse,
    CommandEnvelope,
    ControlEnvelope,
    InferenceRequest,
    ResultEnvelope,
    make_routing_token,
    parse_routing_token,
)
from rollout_runtime.api.messages import (
    CreateSessionRequest,
    EnvFamilyCapability,
    EnvSpecMsg,
    EnvWorkerInfo,
    EpisodeRequest,
    EpisodeResult,
    Observation,
    PerStepRecord,
    PolicyRequest,
    ResetSpec,
    SessionStatus,
    StepResult,
    WorkerSummary,
)
from rollout_runtime.api.payload_ref import InlineBytes, ObjectRefId, PayloadCodec
from rollout_runtime.api.result import Err, Ok, err, ok, unwrap

SESSION = SessionId("sess-abc")
REQUEST = RequestId("req-abc")


def _observation() -> Observation:
    return Observation(
        session_id=SESSION,
        episode_id=EpisodeId(3),
        step_index=7,
        main_image=InlineBytes(
            codec=PayloadCodec.PNG, shape=(2, 2, 3), dtype="uint8", data=b"\x89PNGfake"
        ),
        wrist_image=None,
        extra_view_images=[
            InlineBytes(
                codec=PayloadCodec.RAW,
                shape=(4,),
                dtype="uint8",
                data=b"\x01\x02\x03\x04",
            )
        ],
        state=[0.0, 1.5, -2.25],
        instruction="pick up the black bowl",
        extras={"nested": {"a": 1, "b": [1, 2, 3]}, "flag": True},
    )


def _step_result() -> StepResult:
    return StepResult(
        request_id=REQUEST,
        session_id=SESSION,
        binding_token=BindingToken("bind-1"),
        episode_id=EpisodeId(3),
        operation_seq=OperationSeq(11),
        observation=_observation(),
        reward=2.5,
        terminated=True,
        truncated=False,
        info={"libero_terminated": True},
        side_effect_applied=True,
        executed_horizon=4,
        per_step=[
            PerStepRecord(step_index=index, reward=float(index), terminated=index == 3)
            for index in range(4)
        ],
        error=None,
    )


ROUND_TRIP_CASES = [
    EnvSpecMsg(env_family="fake", env_config={"a": 1, "b": [1, 2]}, pool_size=2),
    EnvFamilyCapability(
        env_family="libero",
        per_step_obs_available=True,
        supports_reset_state_id=True,
        extensions=frozenset({"libero.render_camera", "libero.get_camera_meta"}),
    ),
    CreateSessionRequest(
        application_id="zetta",
        client_session_key="task3-seed1",
        env_spec=EnvSpecMsg(env_family="libero", env_config={"suite": "libero_10"}),
        default_policy_id="pi05",
        lease_seconds=120.0,
        metadata={"trace_id": "t-1"},
    ),
    SessionStatus(
        session_id=SESSION,
        state=SessionState.READY,
        lease_expiration=1234.5,
        episode_id=EpisodeId(2),
        active_operation=REQUEST,
        next_operation_seq=OperationSeq(9),
        worker_summary=WorkerSummary(worker_rank=3, group_name="env", step_index=5),
        error=None,
    ),
    ResetSpec(task_id=3, seed=1, instruction="go", reset_state_id=4, options={"x": 1}),
    PolicyRequest(
        policy_id="pi05",
        inference_parameters={"num_steps": 4, "mode": "eval", "temperature": 0.7},
        actions_per_chunk=8,
        action_postprocess={"translation_scale": 0.05},
    ),
    EpisodeRequest(max_steps=32, policy=PolicyRequest(policy_id="pi05"), deadline=99.0),
    _observation(),
    _step_result(),
    EpisodeResult(
        request_id=REQUEST,
        session_id=SESSION,
        episode_id=EpisodeId(1),
        num_policy_steps=5,
        executed_horizon=20,
        total_reward=3.0,
        terminated=True,
        stop_reason="terminated",
        last_observation=_observation(),
    ),
    EnvWorkerInfo(
        worker_rank=2,
        group_name="env",
        node_id="node-a",
        capabilities={"fake": EnvFamilyCapability(env_family="fake")},
        served_env_digests=["d1", "d2"],
        max_sessions=8,
        has_accelerator=False,
    ),
    CommandEnvelope(
        request_id=REQUEST,
        session_id=SESSION,
        binding_token=BindingToken("bind-1"),
        episode_id=EpisodeId(3),
        operation_seq=OperationSeq(4),
        operation=EnvOperation.POLICY_STEP,
        deadline=12.5,
        priority=Priority.BATCH,
        payload={"policy_request": PolicyRequest(policy_id="pi05")},
        trace_context={"traceparent": "00-abc"},
    ),
    ControlEnvelope(
        request_id=REQUEST,
        operation=EnvOperation.CREATE_BINDING,
        session_id=SESSION,
        payload={"env_spec": EnvSpecMsg(env_family="fake"), "lease_expiration": 5.0},
    ),
    ResultEnvelope(
        request_id=REQUEST,
        session_id=SESSION,
        operation=EnvOperation.RESET,
        state=OperationState.SUCCEEDED,
        value=_step_result(),
        side_effect_applied=True,
        worker_summary=WorkerSummary(worker_rank=1),
    ),
    InferenceRequest(
        request_id=REQUEST,
        session_id=SESSION,
        binding_token=BindingToken("bind-1"),
        episode_id=EpisodeId(3),
        operation_seq=OperationSeq(4),
        policy_id="pi05",
        observation=_observation(),
        inference_parameters={"num_steps": 4},
        routing_token=make_routing_token("env", 2),
        compat_key="c" * 64,
        deadline=None,
        priority=Priority.INTERACTIVE,
        application_id="zetta",
    ),
    ActionResponse(
        request_id=REQUEST,
        session_id=SESSION,
        actions=InlineBytes(
            codec=PayloadCodec.RAW, shape=(2, 7), dtype="float32", data=b"\x00" * 56
        ),
        model_version="v3",
        auxiliary_outputs={"value": 0.5},
    ),
    ActionResponse(
        request_id=REQUEST,
        session_id=SESSION,
        actions=ObjectRefId(id="obj-1", shape=(2, 7), dtype="float32", meta={"k": "v"}),
        model_version="v3",
    ),
]


@pytest.mark.parametrize(
    "message", ROUND_TRIP_CASES, ids=[type(case).__name__ for case in ROUND_TRIP_CASES]
)
def test_msgpack_round_trip_is_lossless(message: object) -> None:
    """msgpack round-trip must be field-for-field equal, including nested
    dataclasses, enums, and bytes.

    Args:
        message: The protocol message under test.
    """
    payload = wire.encode_bytes(message)
    assert isinstance(payload, bytes)
    restored = wire.decode_bytes(payload, type(message))
    assert restored == message
    assert type(restored) is type(message)


def test_round_trip_without_hint_uses_type_tag() -> None:
    """Without a type hint, the ``"@"`` tag alone can still restore the concrete type."""
    message = _step_result()
    restored = wire.decode_bytes(wire.encode_bytes(message))
    assert restored == message


def test_payload_ref_union_dispatches_on_tag() -> None:
    """``PayloadRef`` is a union of two dataclasses; decoding must dispatch by tag."""
    inline = ActionResponse(
        request_id=REQUEST,
        session_id=SESSION,
        actions=InlineBytes(
            codec=PayloadCodec.PNG, shape=(1,), dtype="uint8", data=b"\x00"
        ),
    )
    ref = dataclasses.replace(
        inline, actions=ObjectRefId(id="o", shape=(1,), dtype="uint8")
    )
    assert isinstance(
        wire.decode_bytes(wire.encode_bytes(inline), ActionResponse).actions,
        InlineBytes,
    )
    assert isinstance(
        wire.decode_bytes(wire.encode_bytes(ref), ActionResponse).actions, ObjectRefId
    )


def test_result_wrappers_round_trip() -> None:
    """The generic payloads of ``Ok`` / ``Err`` must also be restorable."""
    good = ok(_step_result())
    bad = err(make_error(ErrorCode.QUEUE_FULL, "full"))
    assert wire.decode_bytes(wire.encode_bytes(good)) == good
    assert wire.decode_bytes(wire.encode_bytes(bad)) == bad
    assert unwrap(good) == _step_result()
    with pytest.raises(RuntimeApiError):
        unwrap(bad)
    assert Ok(1).is_ok is True
    assert Err(make_error(ErrorCode.INTERNAL)).is_ok is False


def test_enums_travel_as_names() -> None:
    """Enums travel on the wire as member names; renaming a member breaks compatibility."""
    encoded = codec.encode(Priority.BACKGROUND)
    assert encoded == "BACKGROUND"
    assert codec.decode(encoded, Priority) is Priority.BACKGROUND


def test_tuple_shape_stays_a_tuple() -> None:
    """``shape`` is declared as ``tuple`` and must not degrade to a list after round-trip."""
    inline = InlineBytes(
        codec=PayloadCodec.RAW, shape=(2, 3, 4), dtype="uint8", data=b"\x00" * 24
    )
    restored = wire.decode_bytes(wire.encode_bytes(inline), InlineBytes)
    assert restored.shape == (2, 3, 4)
    assert isinstance(restored.shape, tuple)


def test_unregistered_type_tag_is_rejected() -> None:
    """A forged type tag must raise an error rather than silently becoming a dict."""
    with pytest.raises(KeyError):
        codec.decode({"@": "NotAMessage", "x": 1})


def test_encode_rejects_unsupported_types() -> None:
    """Unsupported types must raise an explicit error."""

    class Opaque:
        pass

    with pytest.raises(TypeError):
        codec.encode(Opaque())


def test_env_spec_digest_is_stable_and_order_insensitive() -> None:
    """The digest must be stable across instances and independent of dict key order."""
    first = EnvSpecMsg(
        env_family="libero",
        env_config={"suite": "libero_10", "task": 3},
        pool_size=1,
    )
    reordered = EnvSpecMsg(
        env_family="libero",
        env_config={"task": 3, "suite": "libero_10"},
        pool_size=1,
    )
    assert first.digest() == reordered.digest()
    assert len(first.digest()) == 64
    assert first.digest() == EnvSpecMsg(**dataclasses.asdict(first)).digest()


def test_env_spec_digest_covers_pool_semantics_only() -> None:
    """The digest covers family / config / pool size; ``resource_hints`` only affects
    placement and is excluded from the digest."""
    base = EnvSpecMsg(env_family="libero", env_config={"suite": "libero_10"})
    assert base.digest() != dataclasses.replace(base, env_family="maniskill").digest()
    assert base.digest() != dataclasses.replace(base, pool_size=2).digest()
    assert (
        base.digest()
        != dataclasses.replace(base, env_config={"suite": "libero_90"}).digest()
    )
    assert (
        base.digest()
        == dataclasses.replace(base, resource_hints={"node_group": "n1"}).digest()
    )


def test_canonical_json_is_deterministic_for_bytes() -> None:
    """Structures containing bytes must also produce a deterministic canonical representation."""
    inline = InlineBytes(
        codec=PayloadCodec.RAW, shape=(1,), dtype="uint8", data=b"\xff"
    )
    assert codec.canonical_json(inline) == codec.canonical_json(inline)
    assert codec.digest(inline) == codec.digest(inline)
    assert codec.digest(inline, prefix="a") != codec.digest(inline, prefix="b")


def test_routing_token_round_trip() -> None:
    """``routing_token`` must be reversible; an invalid format must raise an error."""
    token = make_routing_token("env-group", 7)
    assert token == "env-group:7"
    assert parse_routing_token(token) == ("env-group", 7)
    with pytest.raises(ValueError, match="malformed routing token"):
        parse_routing_token("nope")


def test_id_generators_have_expected_prefixes() -> None:
    """The id prefix is a protocol convention and must not be changed casually."""
    assert new_session_id().startswith("sess-")
    assert new_request_id().startswith("req-")
    assert new_session_id() != new_session_id()


def test_normalize_exception_maps_known_families() -> None:
    """Exception normalization must map common exceptions to deterministic error codes."""
    assert normalize_exception(TimeoutError("late")).code is ErrorCode.DEADLINE_EXCEEDED
    assert normalize_exception(ValueError("bad")).code is ErrorCode.INVALID_ARGUMENT
    assert (
        normalize_exception(NotImplementedError("m2")).code
        is ErrorCode.UNSUPPORTED_EXTENSION
    )
    assert normalize_exception(RuntimeError("boom")).code is ErrorCode.INTERNAL


def test_normalize_exception_preserves_runtime_api_error() -> None:
    """``RuntimeApiError`` already carries normalized info and should not be reclassified."""
    original = make_error(ErrorCode.STALE_BINDING, "old token")
    info = normalize_exception(RuntimeApiError(original))
    assert info == original
    escalated = normalize_exception(RuntimeApiError(original), side_effect_applied=True)
    assert escalated.side_effect_applied is True
    assert escalated.code is ErrorCode.STALE_BINDING


def test_normalize_exception_records_traceback() -> None:
    """The traceback summary must be recorded in detail and have a bounded length."""
    try:
        raise RuntimeError("deep failure")
    except RuntimeError as exc:
        info = normalize_exception(exc)
    assert info.detail["exception"] == "RuntimeError"
    assert "deep failure" in info.detail["traceback"]
    assert len(info.detail["traceback"]) <= 2000


def test_retryable_flag_follows_error_code() -> None:
    """``retryable`` defaults to a value derived from the error code; system-level errors
    are never retryable."""
    assert make_error(ErrorCode.QUEUE_FULL).retryable is True
    assert make_error(ErrorCode.WORKER_LOST).retryable is False
    assert make_error(ErrorCode.ENV_FAILURE).retryable is False
    assert ErrorCode.WORKER_LOST not in RETRYABLE_ERROR_CODES


def test_invalid_transition_carries_states() -> None:
    """``InvalidTransition`` must be able to identify the entity and both states."""
    exc = InvalidTransition("sess-1", SessionState.CLOSED, SessionState.READY)
    assert exc.info.code is ErrorCode.SESSION_NOT_READY
    assert exc.info.detail["from_state"] == "CLOSED"
    assert exc.info.detail["to_state"] == "READY"
    assert "sess-1" in str(exc)


def test_error_info_round_trip() -> None:
    """The error payload itself must also be able to travel over the wire."""
    info = make_error(
        ErrorCode.STALE_BINDING, "token mismatch", session_id="sess-1", attempts=2
    )
    assert wire.decode_bytes(wire.encode_bytes(info)) == info
