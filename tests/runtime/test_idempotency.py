"""Idempotency tests.

Assertion focus: same ``request_id`` + same request returns the cached
result; different request -> ``IDEMPOTENCY_CONFLICT``; ``client_session_key``
idempotency.
"""

from __future__ import annotations

import pytest

from rollout_runtime.api.enums import (
    TERMINAL_OPERATION_STATES,
    EnvOperation,
    ErrorCode,
    OperationState,
    SessionState,
)
from rollout_runtime.api.errors import InvalidTransition, RuntimeApiError, make_error
from rollout_runtime.api.ids import BindingToken, RequestId, SessionId
from rollout_runtime.api.messages import (
    CreateSessionRequest,
    EnvSpecMsg,
    PolicyRequest,
    ResetSpec,
    StepResult,
)
from rollout_runtime.gateway.operation_registry import (
    ALLOWED_OPERATION_TRANSITIONS,
    OperationRegistry,
    request_digest,
)
from rollout_runtime.gateway.session_manager import SessionManager
from tests.runtime.conftest import FakeClock

SESSION = SessionId("sess-1")
OTHER_SESSION = SessionId("sess-2")
REQ = RequestId("req-1")


def _registry(clock: FakeClock, ttl: float = 300.0) -> OperationRegistry:
    return OperationRegistry(time_source=clock, result_ttl_seconds=ttl)


def _step(reward: float = 1.0) -> StepResult:
    return StepResult(request_id=REQ, session_id=SESSION, reward=reward)


# ---------------------------------------------------------- request_id idempotency


def test_same_request_id_same_payload_returns_existing_record(
    clock: FakeClock,
) -> None:
    """Same id, same request: the second ``begin`` does not create a new
    record, and returns the same one.

    Args:
        clock: Controllable time source.
    """
    registry = _registry(clock)
    spec = ResetSpec(task_id=3, seed=1)
    first, created_first = registry.begin(
        REQ, session_id=SESSION, operation=EnvOperation.RESET, payload={"spec": spec}
    )
    second, created_second = registry.begin(
        REQ, session_id=SESSION, operation=EnvOperation.RESET, payload={"spec": spec}
    )
    assert created_first is True
    assert created_second is False
    assert first is second
    assert len(registry) == 1


def test_same_request_id_returns_cached_result(clock: FakeClock) -> None:
    """The result of a terminal record must be retrievable directly on the
    second call (effectively-once).

    Args:
        clock: Controllable time source.
    """
    registry = _registry(clock)
    payload = {"spec": ResetSpec(task_id=3)}
    registry.begin(
        REQ, session_id=SESSION, operation=EnvOperation.RESET, payload=payload
    )
    registry.mark_running(REQ, worker_rank=2)
    registry.succeed(REQ, _step(reward=5.0))

    record, created = registry.begin(
        REQ, session_id=SESSION, operation=EnvOperation.RESET, payload=payload
    )
    assert created is False
    assert record.is_terminal
    assert record.state is OperationState.SUCCEEDED
    assert record.value == _step(reward=5.0)
    assert record.worker_rank == 2


@pytest.mark.parametrize(
    "mutation",
    [
        pytest.param({"payload": {"spec": ResetSpec(task_id=4)}}, id="payload"),
        pytest.param({"operation": EnvOperation.OBSERVE}, id="operation"),
        pytest.param({"session_id": OTHER_SESSION}, id="session"),
    ],
)
def test_same_request_id_different_request_conflicts(
    clock: FakeClock, mutation: dict[str, object]
) -> None:
    """Same id, different request must raise ``IDEMPOTENCY_CONFLICT``,
    rather than silently executing a second semantics.

    Args:
        clock: Controllable time source.
        mutation: The field used to create the difference.
    """
    registry = _registry(clock)
    base = {
        "session_id": SESSION,
        "operation": EnvOperation.RESET,
        "payload": {"spec": ResetSpec(task_id=3)},
    }
    registry.begin(REQ, **base)  # type: ignore[arg-type]
    with pytest.raises(RuntimeApiError) as excinfo:
        registry.begin(REQ, **{**base, **mutation})  # type: ignore[arg-type]
    assert excinfo.value.info.code is ErrorCode.IDEMPOTENCY_CONFLICT
    assert excinfo.value.info.retryable is False
    assert excinfo.value.info.detail["request_id"] == REQ


def test_request_digest_is_stable_and_discriminating() -> None:
    """The request fingerprint must be stable across instances, and must
    distinguish different parameters."""
    spec = ResetSpec(task_id=3, seed=1, options={"a": 1, "b": 2})
    reordered = ResetSpec(task_id=3, seed=1, options={"b": 2, "a": 1})
    assert request_digest(EnvOperation.RESET, SESSION, spec) == request_digest(
        EnvOperation.RESET, SESSION, reordered
    )
    assert request_digest(EnvOperation.RESET, SESSION, spec) != request_digest(
        EnvOperation.RESET, SESSION, ResetSpec(task_id=4)
    )
    assert request_digest(
        EnvOperation.POLICY_STEP, SESSION, PolicyRequest(policy_id="a")
    ) != request_digest(EnvOperation.POLICY_STEP, SESSION, PolicyRequest(policy_id="b"))


def test_different_request_ids_are_independent(clock: FakeClock) -> None:
    """Different ids execute independently even with the same parameters.

    Args:
        clock: Controllable time source.
    """
    registry = _registry(clock)
    payload = {"spec": ResetSpec(task_id=3)}
    _, first_created = registry.begin(
        RequestId("a"),
        session_id=SESSION,
        operation=EnvOperation.RESET,
        payload=payload,
    )
    _, second_created = registry.begin(
        RequestId("b"),
        session_id=SESSION,
        operation=EnvOperation.RESET,
        payload=payload,
    )
    assert first_created and second_created
    assert len(registry) == 2


# ---------------------------------------------------------- Operation state machine


def test_operation_transition_table_terminals_are_closed() -> None:
    """Terminal states cannot be transitioned again."""
    for state in TERMINAL_OPERATION_STATES:
        assert ALLOWED_OPERATION_TRANSITIONS[state] == frozenset()


def test_terminal_operation_cannot_transition_again(clock: FakeClock) -> None:
    """Changing the state again on an already-succeeded operation must
    raise ``InvalidTransition``.

    Args:
        clock: Controllable time source.
    """
    registry = _registry(clock)
    registry.begin(REQ, session_id=SESSION, operation=EnvOperation.RESET)
    registry.succeed(REQ, _step())
    with pytest.raises(InvalidTransition):
        registry.mark_running(REQ)
    with pytest.raises(InvalidTransition):
        registry.fail(REQ, make_error(ErrorCode.INTERNAL))


def test_status_reports_lifecycle(clock: FakeClock) -> None:
    """Status queries cover ACCEPTED -> QUEUED -> RUNNING -> SUCCEEDED.

    Args:
        clock: Controllable time source.
    """
    registry = _registry(clock)
    registry.begin(REQ, session_id=SESSION, operation=EnvOperation.POLICY_STEP)
    assert registry.status(REQ).state is OperationState.ACCEPTED
    registry.mark_queued(REQ)
    assert registry.status(REQ).state is OperationState.QUEUED
    clock.advance(1.0)
    registry.mark_running(REQ)
    status = registry.status(REQ)
    assert status.state is OperationState.RUNNING
    assert status.updated_at > status.created_at
    registry.succeed(REQ, _step())
    assert registry.status(REQ).state is OperationState.SUCCEEDED


def test_unknown_request_id_is_invalid_argument(clock: FakeClock) -> None:
    """Unknown request_id -> ``INVALID_ARGUMENT``.

    Args:
        clock: Controllable time source.
    """
    registry = _registry(clock)
    with pytest.raises(RuntimeApiError) as excinfo:
        registry.status(RequestId("nope"))
    assert excinfo.value.info.code is ErrorCode.INVALID_ARGUMENT
    assert registry.find(RequestId("nope")) is None


def test_failure_propagates_side_effect_flag(clock: FakeClock) -> None:
    """A failure must also honestly flag whether the side effect has
    occurred.

    Args:
        clock: Controllable time source.
    """
    registry = _registry(clock)
    registry.begin(REQ, session_id=SESSION, operation=EnvOperation.ACTION_STEP)
    registry.mark_running(REQ)
    info = make_error(ErrorCode.ENV_FAILURE, "mujoco blew up", side_effect_applied=True)
    record = registry.fail(REQ, info)
    assert record.side_effect_applied is True
    assert registry.status(REQ).side_effect_applied is True


# ---------------------------------------------------------- The four cancellation states


def test_cancel_before_dispatch_reports_no_side_effect(clock: FakeClock) -> None:
    """Not yet dispatched: cancel directly, ``side_effect_applied=false``.

    Args:
        clock: Controllable time source.
    """
    registry = _registry(clock)
    registry.begin(REQ, session_id=SESSION, operation=EnvOperation.POLICY_STEP)
    outcome = registry.request_cancel(REQ)
    assert outcome.state is OperationState.CANCELLED
    assert outcome.side_effect_applied is False
    assert registry.status(REQ).state is OperationState.CANCELLED


def test_cancel_while_running_is_best_effort(clock: FakeClock) -> None:
    """Already dispatched: only registers intent, state remains
    ``RUNNING``, and the EnvWorker decides whether it can stop.

    Args:
        clock: Controllable time source.
    """
    registry = _registry(clock)
    registry.begin(REQ, session_id=SESSION, operation=EnvOperation.POLICY_STEP)
    registry.mark_running(REQ)
    outcome = registry.request_cancel(REQ)
    assert outcome.state is OperationState.RUNNING
    assert "best effort" in outcome.message
    assert registry.get(REQ).cancel_requested is True

    # The env step already started and finished: not rolled back, honestly
    # report side_effect_applied=true.
    registry.succeed(REQ, _step(), side_effect_applied=True)
    assert registry.status(REQ).side_effect_applied is True


def test_cancel_after_terminal_returns_existing_outcome(clock: FakeClock) -> None:
    """Already terminal: cancel is an idempotent no-op.

    Args:
        clock: Controllable time source.
    """
    registry = _registry(clock)
    registry.begin(REQ, session_id=SESSION, operation=EnvOperation.RESET)
    registry.succeed(REQ, _step(), side_effect_applied=True)
    outcome = registry.request_cancel(REQ)
    assert outcome.state is OperationState.SUCCEEDED
    assert outcome.side_effect_applied is True
    assert "already finished" in outcome.message


def test_worker_lost_yields_outcome_unknown(clock: FakeClock) -> None:
    """Worker lost: ``OUTCOME_UNKNOWN``, must not be replayed automatically.

    Args:
        clock: Controllable time source.
    """
    registry = _registry(clock)
    registry.begin(REQ, session_id=SESSION, operation=EnvOperation.POLICY_STEP)
    registry.mark_running(REQ)
    record = registry.mark_outcome_unknown(REQ, "rank 3 gone")
    assert record.state is OperationState.OUTCOME_UNKNOWN
    assert record.error is not None
    assert record.error.code is ErrorCode.WORKER_LOST
    assert record.error.retryable is False


# ---------------------------------------------------------- TTL cache


def test_terminal_results_expire_after_ttl(clock: FakeClock) -> None:
    """Terminal results are cleaned up according to TTL; non-terminal
    records are never cleaned up.

    Args:
        clock: Controllable time source.
    """
    registry = _registry(clock, ttl=60.0)
    registry.begin(RequestId("done"), session_id=SESSION, operation=EnvOperation.RESET)
    registry.succeed(RequestId("done"), _step())
    registry.begin(
        RequestId("running"), session_id=SESSION, operation=EnvOperation.POLICY_STEP
    )
    registry.mark_running(RequestId("running"))

    clock.advance(59.0)
    assert registry.purge() == []
    clock.advance(2.0)
    assert registry.purge() == [RequestId("done")]
    assert registry.find(RequestId("running")) is not None


# ---------------------------------------------------- client_session_key idempotency


def _create_request(key: str, *, family: str = "fake") -> CreateSessionRequest:
    return CreateSessionRequest(
        application_id="zetta",
        client_session_key=key,
        env_spec=EnvSpecMsg(env_family=family),
        lease_seconds=60.0,
    )


def test_client_session_key_is_idempotent(clock: FakeClock) -> None:
    """A second creation with the same application and key reuses the
    existing session.

    Args:
        clock: Controllable time source.
    """
    manager = SessionManager(time_source=clock)
    first, created_first = manager.create(_create_request("task3-seed1"))
    manager.commit_binding(
        first.session_id, worker_rank=0, binding_token=BindingToken("b1")
    )
    second, created_second = manager.create(_create_request("task3-seed1"))
    assert created_first is True
    assert created_second is False
    assert second.session_id == first.session_id
    assert len(manager) == 1


def test_client_session_key_is_scoped_per_application(clock: FakeClock) -> None:
    """The idempotency key is isolated per application; the same key from a
    different tenant does not interfere.

    Args:
        clock: Controllable time source.
    """
    manager = SessionManager(time_source=clock)
    first, _ = manager.create(_create_request("shared"))
    other = CreateSessionRequest(
        application_id="other-app",
        client_session_key="shared",
        env_spec=EnvSpecMsg(env_family="fake"),
        lease_seconds=60.0,
    )
    second, created = manager.create(other)
    assert created is True
    assert second.session_id != first.session_id


def test_client_session_key_is_reusable_after_close(clock: FakeClock) -> None:
    """After a session closes, the same key can be reused to create a new
    session, getting a new session_id.

    Args:
        clock: Controllable time source.
    """
    manager = SessionManager(time_source=clock)
    first, _ = manager.create(_create_request("task3-seed1"))
    manager.commit_binding(
        first.session_id, worker_rank=0, binding_token=BindingToken("b1")
    )
    manager.begin_close(first.session_id)
    manager.finish_close(first.session_id)

    second, created = manager.create(_create_request("task3-seed1"))
    assert created is True
    assert second.session_id != first.session_id
    assert second.state is SessionState.CREATING


def test_client_session_key_idempotency_ignores_payload_drift(
    clock: FakeClock,
) -> None:
    """When the idempotency key hits, the existing session takes precedence
    and is never rebuilt with a new env_spec.

    This differs from ``request_id`` conflict detection: the session
    idempotency key means "I want that session," not "replay this request."
    """
    manager = SessionManager(time_source=clock)
    first, _ = manager.create(_create_request("k", family="fake"))
    manager.commit_binding(
        first.session_id, worker_rank=0, binding_token=BindingToken("b1")
    )
    second, created = manager.create(_create_request("k", family="libero"))
    assert created is False
    assert second.env_spec.env_family == "fake"
