"""Session lifecycle state machine tests.

Focus of assertions: illegal state transitions raise ``InvalidTransition``, lease
expiry, and ``LOST`` is not recoverable.
"""

from __future__ import annotations

import asyncio

import pytest

from rollout_runtime.api.enums import ErrorCode, SessionState
from rollout_runtime.api.errors import InvalidTransition, RuntimeApiError, make_error
from rollout_runtime.api.ids import BindingToken, EpisodeId, RequestId, SessionId
from rollout_runtime.api.messages import CreateSessionRequest, EnvSpecMsg, WorkerSummary
from rollout_runtime.gateway.session_manager import (
    ALLOWED_SESSION_TRANSITIONS,
    SessionManager,
)
from tests.runtime.conftest import FakeClock

ALL_STATES = tuple(SessionState)


def _request(key: str = "k1", *, application_id: str = "zetta") -> CreateSessionRequest:
    return CreateSessionRequest(
        application_id=application_id,
        client_session_key=key,
        env_spec=EnvSpecMsg(env_family="fake", env_config={"n": 1}),
        default_policy_id="fake",
        lease_seconds=60.0,
    )


def _manager(clock: FakeClock) -> SessionManager:
    return SessionManager(
        gateway_epoch=42, time_source=clock, error_retention_seconds=300.0
    )


def _ready(manager: SessionManager, key: str = "k1") -> SessionId:
    record, created = manager.create(_request(key))
    assert created
    manager.commit_binding(
        record.session_id, worker_rank=0, binding_token=BindingToken("bind-1")
    )
    return record.session_id


# ------------------------------------------------------------------ The transition table itself


def test_transition_table_matches_plan() -> None:
    """The transition table is exactly the five edges plus the terminal-state cleanup edges, no more, no less."""
    assert ALLOWED_SESSION_TRANSITIONS == {
        SessionState.CREATING: frozenset({SessionState.READY, SessionState.FAILED}),
        SessionState.READY: frozenset({SessionState.CLOSING, SessionState.LOST}),
        SessionState.CLOSING: frozenset({SessionState.CLOSED}),
        SessionState.FAILED: frozenset({SessionState.CLOSED}),
        SessionState.LOST: frozenset({SessionState.CLOSED}),
        SessionState.CLOSED: frozenset(),
    }
    assert set(ALLOWED_SESSION_TRANSITIONS) == set(ALL_STATES)


def test_happy_path_creating_ready_closing_closed(clock: FakeClock) -> None:
    """Normal lifecycle: ``CREATING -> READY -> CLOSING -> CLOSED``.

    Args:
        clock: Controllable time source.
    """
    manager = _manager(clock)
    record, created = manager.create(_request())
    assert created
    assert record.state is SessionState.CREATING
    assert record.episode_id is None

    handle = manager.commit_binding(
        record.session_id, worker_rank=3, binding_token=BindingToken("bind-9")
    )
    assert handle.state is SessionState.READY
    assert handle.worker_rank == 3
    assert handle.binding_token == BindingToken("bind-9")

    assert manager.begin_close(record.session_id).state is SessionState.CLOSING
    assert manager.finish_close(record.session_id).state is SessionState.CLOSED


def test_handle_hides_internal_routing(clock: FakeClock) -> None:
    """``SessionHandle`` does not expose rank / slot / binding_token.

    Args:
        clock: Controllable time source.
    """
    manager = _manager(clock)
    session_id = _ready(manager)
    handle = manager.get(session_id).handle()
    fields = set(vars(handle))
    assert "worker_rank" not in fields
    assert "binding_token" not in fields
    assert "slot_index" not in fields
    assert handle.gateway_epoch == 42
    assert (
        handle.env_spec_digest
        == EnvSpecMsg(env_family="fake", env_config={"n": 1}).digest()
    )


@pytest.mark.parametrize(
    ("from_state", "to_state"),
    [
        (from_state, to_state)
        for from_state in ALL_STATES
        for to_state in ALL_STATES
        if to_state not in ALLOWED_SESSION_TRANSITIONS[from_state]
    ],
)
def test_illegal_transitions_raise(
    clock: FakeClock, from_state: SessionState, to_state: SessionState
) -> None:
    """Every edge outside the transition table must raise ``InvalidTransition``.

    Args:
        clock: Controllable time source.
        from_state: Starting state.
        to_state: Target state.
    """
    manager = _manager(clock)
    record, _ = manager.create(_request())
    record.state = from_state
    with pytest.raises(InvalidTransition) as excinfo:
        manager.transition(record.session_id, to_state)
    assert excinfo.value.from_state is from_state
    assert excinfo.value.to_state is to_state


def test_lost_session_is_not_recoverable(clock: FakeClock) -> None:
    """v1 does not recover ``LOST``: a newly created environment cannot
    resume the original physical state.

    Args:
        clock: Controllable time source.
    """
    manager = _manager(clock)
    session_id = _ready(manager)
    record = manager.mark_lost(session_id)
    assert record.state is SessionState.LOST
    assert record.error is not None
    assert record.error.code is ErrorCode.WORKER_LOST

    with pytest.raises(InvalidTransition):
        manager.transition(session_id, SessionState.READY)
    with pytest.raises(InvalidTransition):
        manager.begin_close(session_id)
    # Only the caller can close it and create a new one.
    assert manager.finish_close(session_id).state is SessionState.CLOSED


def test_failed_session_keeps_error_for_queries(clock: FakeClock) -> None:
    """A ``CREATING`` allocation failure -> ``FAILED``, with the error kept
    briefly for queries.

    Args:
        clock: Controllable time source.
    """
    manager = _manager(clock)
    record, _ = manager.create(_request())
    info = make_error(ErrorCode.UNSUPPORTED_ENV_SPEC, "no such family")
    manager.fail(record.session_id, info)
    status = manager.status(record.session_id)
    assert status.state is SessionState.FAILED
    assert status.error == info


# ------------------------------------------------------------------ Readiness validation


def test_require_ready_rejects_non_ready_states(clock: FakeClock) -> None:
    """Any operation request on a non-``READY`` session -> ``SESSION_NOT_READY``.

    Args:
        clock: Controllable time source.
    """
    manager = _manager(clock)
    record, _ = manager.create(_request())
    with pytest.raises(RuntimeApiError) as excinfo:
        manager.require_ready(record.session_id)
    assert excinfo.value.info.code is ErrorCode.SESSION_NOT_READY
    assert excinfo.value.info.detail["state"] == "CREATING"


def test_ready_does_not_imply_reset(clock: FakeClock) -> None:
    """``READY`` only means resources are bound; observe/step must be
    blocked before reset.

    Args:
        clock: Controllable time source.
    """
    manager = _manager(clock)
    session_id = _ready(manager)
    with pytest.raises(RuntimeApiError, match="no episode"):
        manager.require_episode(session_id)

    manager.set_episode(session_id, EpisodeId(1))
    assert manager.require_episode(session_id).episode_id == 1


def test_unknown_session_raises_unknown_session(clock: FakeClock) -> None:
    """An unknown session -> ``UNKNOWN_SESSION``.

    Args:
        clock: Controllable time source.
    """
    manager = _manager(clock)
    with pytest.raises(RuntimeApiError) as excinfo:
        manager.get(SessionId("sess-nope"))
    assert excinfo.value.info.code is ErrorCode.UNKNOWN_SESSION
    assert manager.find(SessionId("sess-nope")) is None


# ------------------------------------------------------------------ Leases


def test_lease_expiry_uses_injected_clock(clock: FakeClock) -> None:
    """Once a lease elapses, it appears in the expired list.

    Args:
        clock: Controllable time source.
    """
    manager = _manager(clock)
    session_id = _ready(manager)
    assert manager.expired_sessions() == []
    clock.advance(59.0)
    assert manager.expired_sessions() == []
    clock.advance(2.0)
    assert [record.session_id for record in manager.expired_sessions()] == [session_id]


def test_renew_extends_lease_and_checks_ownership(clock: FakeClock) -> None:
    """Renewal must extend the expiration time, and validate caller ownership.

    Args:
        clock: Controllable time source.
    """
    manager = _manager(clock)
    session_id = _ready(manager)
    clock.advance(120.0)
    assert manager.expired_sessions()

    record = manager.renew(session_id, 60.0)
    assert record.lease_expiration == clock.now + 60.0
    assert manager.expired_sessions() == []

    with pytest.raises(RuntimeApiError, match="does not belong"):
        manager.renew(session_id, 60.0, application_id="other-app")
    with pytest.raises(RuntimeApiError) as excinfo:
        manager.renew(session_id, 0.0)
    assert excinfo.value.info.code is ErrorCode.INVALID_ARGUMENT


def test_create_rejects_non_positive_lease(clock: FakeClock) -> None:
    """An invalid lease duration must be rejected at creation time.

    Args:
        clock: Controllable time source.
    """
    manager = _manager(clock)
    bad = CreateSessionRequest(
        application_id="zetta",
        client_session_key="k",
        env_spec=EnvSpecMsg(env_family="fake"),
        lease_seconds=0.0,
    )
    with pytest.raises(RuntimeApiError) as excinfo:
        manager.create(bad)
    assert excinfo.value.info.code is ErrorCode.INVALID_ARGUMENT


# ------------------------------------------------------------ Operation sequence and lock


def test_operation_seq_is_monotonic(clock: FakeClock) -> None:
    """``operation_seq`` increments independently per session, starting at 1.

    Args:
        clock: Controllable time source.
    """
    manager = _manager(clock)
    first = manager.get(_ready(manager, "a"))
    second = manager.get(_ready(manager, "b"))
    assert [manager.allocate_operation_seq(first) for _ in range(3)] == [1, 2, 3]
    assert manager.allocate_operation_seq(second) == 1
    assert manager.status(first.session_id).next_operation_seq == 4


def test_active_operation_blocks_concurrent_mutation(clock: FakeClock) -> None:
    """The same session can have only one mutating operation at a time
    (the second line of defense).

    Args:
        clock: Controllable time source.
    """
    manager = _manager(clock)
    record = manager.get(_ready(manager))
    manager.begin_operation(record, RequestId("req-1"))
    with pytest.raises(RuntimeApiError) as excinfo:
        manager.begin_operation(record, RequestId("req-2"))
    assert excinfo.value.info.code is ErrorCode.SESSION_NOT_READY

    manager.end_operation(record, RequestId("req-1"))
    assert record.active_operation is None
    manager.begin_operation(record, RequestId("req-2"))


def test_per_session_lock_serializes_mutations(clock: FakeClock) -> None:
    """The per-session lock genuinely serializes: two interleaved
    coroutines never overlap the critical section.

    Args:
        clock: Controllable time source.
    """
    manager = _manager(clock)
    record = manager.get(_ready(manager))
    order: list[str] = []

    async def mutate(name: str) -> None:
        async with record.lock:
            order.append(f"{name}-enter")
            await asyncio.sleep(0)
            order.append(f"{name}-exit")

    async def main() -> None:
        await asyncio.gather(mutate("a"), mutate("b"))

    asyncio.run(main())
    assert order in (
        ["a-enter", "a-exit", "b-enter", "b-exit"],
        ["b-enter", "b-exit", "a-enter", "a-exit"],
    )


# ------------------------------------------------------------------ Cleanup and observation


def test_purge_removes_closed_immediately_and_errors_after_retention(
    clock: FakeClock,
) -> None:
    """``CLOSED`` is purged immediately; ``FAILED`` / ``LOST`` are retained
    for 300s for querying.

    Args:
        clock: Controllable time source.
    """
    manager = _manager(clock)
    closed = _ready(manager, "closed")
    manager.begin_close(closed)
    manager.finish_close(closed)

    failed_record, _ = manager.create(_request("failed"))
    manager.fail(failed_record.session_id, make_error(ErrorCode.ENV_FAILURE, "boom"))

    assert manager.purge_terminal() == [closed]
    assert manager.find(failed_record.session_id) is not None

    clock.advance(299.0)
    assert manager.purge_terminal() == []
    clock.advance(2.0)
    assert manager.purge_terminal() == [failed_record.session_id]
    assert len(manager) == 0


def test_sessions_on_rank_and_active_counts(clock: FakeClock) -> None:
    """Both per-rank and per-application statistics count only active states.

    Args:
        clock: Controllable time source.
    """
    manager = _manager(clock)
    first = _ready(manager, "a")
    second = _ready(manager, "b")
    assert {record.session_id for record in manager.sessions_on_rank(0)} == {
        first,
        second,
    }
    assert manager.active_session_count("zetta") == 2

    manager.begin_close(first)
    manager.finish_close(first)
    assert [record.session_id for record in manager.sessions_on_rank(0)] == [second]
    assert manager.active_session_count("zetta") == 1


def test_worker_summary_cache_is_optional(clock: FakeClock) -> None:
    """The worker summary is a non-authoritative cache; ``None`` does not
    overwrite the existing value.

    Args:
        clock: Controllable time source.
    """
    manager = _manager(clock)
    session_id = _ready(manager)
    manager.set_worker_summary(session_id, WorkerSummary(worker_rank=0, step_index=5))
    manager.set_worker_summary(session_id, None)
    summary = manager.status(session_id).worker_summary
    assert summary is not None
    assert summary.step_index == 5
