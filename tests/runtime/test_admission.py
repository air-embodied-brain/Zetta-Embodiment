"""Admission tests.

Assertion focus: authentication, quota, deadline, backpressure rejection.
"""

from __future__ import annotations

import pytest

from rollout_runtime.api.enums import ErrorCode
from rollout_runtime.api.errors import RuntimeApiError
from rollout_runtime.api.ids import SessionId
from rollout_runtime.api.messages import EnvFamilyCapability, EnvSpecMsg, EnvWorkerInfo
from rollout_runtime.gateway.admission import AdmissionConfig, AdmissionController
from rollout_runtime.gateway.worker_registry import EnvWorkerRegistry
from tests.runtime.conftest import FakeClock

APP = "zetta"
SESSION = SessionId("sess-1")


def _controller(clock: FakeClock, **overrides: object) -> AdmissionController:
    config = AdmissionConfig(**overrides)  # type: ignore[arg-type]
    return AdmissionController(config, time_source=clock)


def _expect(
    error_code: ErrorCode, call: object, *args: object, **kwargs: object
) -> None:
    with pytest.raises(RuntimeApiError) as excinfo:
        call(*args, **kwargs)  # type: ignore[operator]
    assert excinfo.value.info.code is error_code


# ------------------------------------------------------------------ Authentication


def test_auth_disabled_by_default(clock: FakeClock) -> None:
    """By default, the token is not verified (embedded form).

    Args:
        clock: Controllable time source.
    """
    controller = _controller(clock)
    controller.authenticate(APP, None)
    controller.admit_session(APP)


def test_empty_application_id_is_rejected(clock: FakeClock) -> None:
    """``application_id`` is the ownership key for quota and auditing, and
    cannot be empty.

    Args:
        clock: Controllable time source.
    """
    controller = _controller(clock)
    _expect(ErrorCode.INVALID_ARGUMENT, controller.authenticate, "", None)


def test_auth_required_checks_token(clock: FakeClock) -> None:
    """Once authentication is enabled, the token must match; failure falls
    under ``INVALID_ARGUMENT`` with a reason tag.

    There is no dedicated ``UNAUTHENTICATED`` error code.

    Args:
        clock: Controllable time source.
    """
    controller = _controller(clock, require_auth=True, tokens={APP: "secret"})
    controller.authenticate(APP, "secret")

    with pytest.raises(RuntimeApiError) as excinfo:
        controller.authenticate(APP, "wrong")
    assert excinfo.value.info.code is ErrorCode.INVALID_ARGUMENT
    assert excinfo.value.info.detail["reason"] == "authentication"

    _expect(ErrorCode.INVALID_ARGUMENT, controller.authenticate, "unknown-app", "x")
    _expect(ErrorCode.INVALID_ARGUMENT, controller.authenticate, APP, None)


# ------------------------------------------------------------------ Session quota


def test_session_quota_is_enforced_per_application(clock: FakeClock) -> None:
    """Per-application concurrent session limit.

    Args:
        clock: Controllable time source.
    """
    controller = _controller(clock, max_sessions_per_application=2)
    controller.admit_session(APP)
    controller.admit_session(APP)
    with pytest.raises(RuntimeApiError) as excinfo:
        controller.admit_session(APP)
    info = excinfo.value.info
    assert info.code is ErrorCode.QUOTA_EXCEEDED
    assert info.retryable is True
    assert info.detail["limit"] == 2
    assert info.detail["current"] == 2

    # Another tenant is unaffected.
    controller.admit_session("other-app")


def test_releasing_a_session_frees_quota(clock: FakeClock) -> None:
    """Quota becomes reusable after release, and the count never goes negative.

    Args:
        clock: Controllable time source.
    """
    controller = _controller(clock, max_sessions_per_application=1)
    controller.admit_session(APP)
    controller.release_session(APP)
    controller.admit_session(APP)
    controller.release_session(APP)
    controller.release_session(APP)
    assert controller.snapshot().sessions_per_application == {}


# ------------------------------------------------------------------ Deadline


def test_elapsed_deadline_is_rejected(clock: FakeClock) -> None:
    """An already-elapsed deadline is rejected directly with
    ``DEADLINE_EXCEEDED``, without dispatch.

    Args:
        clock: Controllable time source.
    """
    controller = _controller(clock)
    controller.admit_operation(APP, SESSION, deadline=clock.now + 10.0)
    controller.complete_operation(APP, SESSION)

    with pytest.raises(RuntimeApiError) as excinfo:
        controller.admit_operation(APP, SESSION, deadline=clock.now)
    assert excinfo.value.info.code is ErrorCode.DEADLINE_EXCEEDED
    assert excinfo.value.info.retryable is True

    clock.advance(20.0)
    _expect(
        ErrorCode.DEADLINE_EXCEEDED,
        controller.admit_operation,
        APP,
        SESSION,
        deadline=clock.now - 1.0,
    )


def test_no_deadline_means_no_check(clock: FakeClock) -> None:
    """``deadline=None`` performs no timing check.

    Args:
        clock: Controllable time source.
    """
    controller = _controller(clock)
    controller.admit_operation(APP, SESSION, deadline=None)


# ------------------------------------------------------------------ Backpressure


def test_full_downstream_queue_is_rejected_not_queued(clock: FakeClock) -> None:
    """A full queue is explicitly rejected with ``QUEUE_FULL``, **never
    queued** (backpressure propagates to the entry point).

    Args:
        clock: Controllable time source.
    """
    controller = _controller(clock)
    with pytest.raises(RuntimeApiError) as excinfo:
        controller.admit_operation(APP, SESSION, queue_depth=2, queue_capacity=2)
    info = excinfo.value.info
    assert info.code is ErrorCode.QUEUE_FULL
    assert info.retryable is True
    assert info.detail["queue_depth"] == 2
    assert info.detail["queue_capacity"] == 2

    controller.admit_operation(APP, SESSION, queue_depth=1, queue_capacity=2)


def test_global_inflight_watermark_is_queue_full(clock: FakeClock) -> None:
    """Hitting the global in-flight watermark also counts as ``QUEUE_FULL``.

    Args:
        clock: Controllable time source.
    """
    controller = _controller(
        clock,
        max_total_inflight_operations=2,
        max_inflight_operations_per_session=10,
        max_inflight_operations_per_application=10,
    )
    controller.admit_operation(APP, SESSION)
    controller.admit_operation(APP, SESSION)
    _expect(ErrorCode.QUEUE_FULL, controller.admit_operation, APP, SESSION)

    controller.complete_operation(APP, SESSION)
    controller.admit_operation(APP, SESSION)


# ------------------------------------------------------------------ In-flight quota


def test_per_application_inflight_quota(clock: FakeClock) -> None:
    """Per-tenant in-flight limit, to prevent one application from filling
    up the Gateway.

    Args:
        clock: Controllable time source.
    """
    controller = _controller(
        clock,
        max_inflight_operations_per_application=2,
        max_inflight_operations_per_session=10,
    )
    controller.admit_operation(APP, SESSION)
    controller.admit_operation(APP, SessionId("sess-2"))
    _expect(
        ErrorCode.QUOTA_EXCEEDED,
        controller.admit_operation,
        APP,
        SessionId("sess-3"),
    )
    controller.admit_operation("other-app", SessionId("sess-4"))


def test_per_session_inflight_quota(clock: FakeClock) -> None:
    """Per-session in-flight limit (serialization of mutating operations is
    handled separately by the per-session lock).

    Args:
        clock: Controllable time source.
    """
    controller = _controller(clock, max_inflight_operations_per_session=1)
    controller.admit_operation(APP, SESSION)
    with pytest.raises(RuntimeApiError) as excinfo:
        controller.admit_operation(APP, SESSION)
    assert excinfo.value.info.code is ErrorCode.QUOTA_EXCEEDED
    assert excinfo.value.info.detail["session_id"] == SESSION

    controller.admit_operation(APP, SessionId("sess-other"))


def test_completion_releases_inflight_counters(clock: FakeClock) -> None:
    """The count returns to zero on completion, with no empty entries left
    in the snapshot.

    Args:
        clock: Controllable time source.
    """
    controller = _controller(clock)
    controller.admit_operation(APP, SESSION)
    snapshot = controller.snapshot()
    assert snapshot.total_inflight == 1
    assert snapshot.inflight_per_session[str(SESSION)] == 1

    controller.complete_operation(APP, SESSION)
    snapshot = controller.snapshot()
    assert snapshot.total_inflight == 0
    assert snapshot.inflight_per_application == {}
    assert snapshot.inflight_per_session == {}


def test_forget_session_drops_leaked_inflight(clock: FakeClock) -> None:
    """Closing a session clears any residual in-flight count, avoiding a
    quota leak.

    Args:
        clock: Controllable time source.
    """
    controller = _controller(clock, max_inflight_operations_per_session=4)
    controller.admit_operation(APP, SESSION)
    controller.admit_operation(APP, SESSION)
    controller.forget_session(SESSION)
    snapshot = controller.snapshot()
    assert snapshot.total_inflight == 0
    assert str(SESSION) not in snapshot.inflight_per_session


def test_rejections_are_counted_by_code(clock: FakeClock) -> None:
    """Rejections must be counted by error code, for metrics export.

    Args:
        clock: Controllable time source.
    """
    controller = _controller(clock, max_sessions_per_application=0)
    for _ in range(3):
        with pytest.raises(RuntimeApiError):
            controller.admit_session(APP)
    with pytest.raises(RuntimeApiError):
        controller.admit_operation(APP, SESSION, deadline=clock.now - 1)
    counts = controller.rejected_counts()
    assert counts["QUOTA_EXCEEDED"] == 3
    assert counts["DEADLINE_EXCEEDED"] == 1


# -------------------------------------------------- Worker-side capacity and capability admission


def _worker(
    rank: int,
    *,
    families: tuple[str, ...] = ("fake",),
    max_sessions: int = 2,
    node_id: str = "node-a",
    accelerator: bool = False,
    needs_accelerator: bool = False,
    served: tuple[str, ...] = (),
) -> EnvWorkerInfo:
    return EnvWorkerInfo(
        worker_rank=rank,
        group_name="env",
        node_id=node_id,
        capabilities={
            family: EnvFamilyCapability(
                env_family=family, needs_accelerator=needs_accelerator
            )
            for family in families
        },
        served_env_digests=list(served),
        max_sessions=max_sessions,
        has_accelerator=accelerator,
    )


def test_unknown_env_family_is_unsupported_env_spec(clock: FakeClock) -> None:
    """No one can serve the family -> ``UNSUPPORTED_ENV_SPEC``, rather than
    exploding at runtime.

    Args:
        clock: Controllable time source.
    """
    registry = EnvWorkerRegistry(time_source=clock)
    registry.register(_worker(0))
    _expect(
        ErrorCode.UNSUPPORTED_ENV_SPEC,
        registry.select_rank,
        EnvSpecMsg(env_family="maniskill"),
    )


def test_no_worker_at_all_is_worker_lost(clock: FakeClock) -> None:
    """No healthy rank -> ``WORKER_LOST``.

    Args:
        clock: Controllable time source.
    """
    registry = EnvWorkerRegistry(time_source=clock)
    _expect(ErrorCode.WORKER_LOST, registry.select_rank, EnvSpecMsg(env_family="fake"))


def test_registry_ignores_static_max_sessions_and_chooses_least_loaded_worker(
    clock: FakeClock,
) -> None:
    """The Gateway schedules only by worker-level load, and no longer uses
    ``max_sessions`` as a capacity gate.

    Args:
        clock: Controllable time source.
    """
    registry = EnvWorkerRegistry(time_source=clock)
    registry.register(_worker(0, max_sessions=1))
    spec = EnvSpecMsg(env_family="fake")
    assert registry.select_rank(spec) == 0
    registry.acquire(0, spec.digest())
    assert registry.select_rank(spec) == 0

    registry.release(0)
    assert registry.select_rank(spec) == 0


def test_selection_uses_worker_load_before_served_digest(clock: FakeClock) -> None:
    """Same-digest pool information no longer drives Gateway placement; the
    less-loaded rank is preferred.

    Args:
        clock: Controllable time source.
    """
    registry = EnvWorkerRegistry(time_source=clock)
    spec = EnvSpecMsg(env_family="fake")
    registry.register(_worker(0, max_sessions=4))
    registry.register(_worker(1, max_sessions=4, served=(spec.digest(),)))
    assert registry.select_rank(spec) == 0

    registry.acquire(0, spec.digest())
    assert registry.select_rank(spec) == 1


def test_pre_reserve_prevents_concurrent_stampede(clock: FakeClock) -> None:
    """Consecutive select/acquire calls spread concurrent creation requests
    across the currently emptiest rank."""
    registry = EnvWorkerRegistry(time_source=clock)
    spec = EnvSpecMsg(env_family="fake")
    registry.register(_worker(0, max_sessions=1))
    registry.register(_worker(1, max_sessions=1))

    first = registry.select_rank(spec)
    registry.acquire(first, spec.digest())
    second = registry.select_rank(spec)
    registry.acquire(second, spec.digest())

    assert [first, second] == [0, 1]
    assert registry.snapshot()[0].active_sessions == 1
    assert registry.snapshot()[1].active_sessions == 1


def test_can_create_slot_backoff_filters_and_release_restores(
    clock: FakeClock,
) -> None:
    """The backoff after a structured OOM skips that rank, and it is
    restored once the release succeeds."""
    registry = EnvWorkerRegistry(time_source=clock)
    spec = EnvSpecMsg(env_family="fake")
    registry.register(_worker(0))
    registry.register(_worker(1))

    registry.mark_cannot_create_slot(0)
    assert registry.select_rank(spec) == 1

    registry.release(0)
    assert registry.select_rank(spec) == 0


def test_accelerator_requirement_filters_ranks(clock: FakeClock) -> None:
    """A family requiring an accelerator cannot land on a rank without one.

    Args:
        clock: Controllable time source.
    """
    registry = EnvWorkerRegistry(time_source=clock)
    registry.register(
        _worker(0, families=("maniskill",), needs_accelerator=True, accelerator=False)
    )
    _expect(
        ErrorCode.UNSUPPORTED_ENV_SPEC,
        registry.select_rank,
        EnvSpecMsg(env_family="maniskill"),
    )
    registry.register(
        _worker(1, families=("maniskill",), needs_accelerator=True, accelerator=True)
    )
    assert registry.select_rank(EnvSpecMsg(env_family="maniskill")) == 1


def test_node_group_hint_filters_ranks(clock: FakeClock) -> None:
    """``resource_hints["node_group"]`` is a hard filter condition.

    Args:
        clock: Controllable time source.
    """
    registry = EnvWorkerRegistry(time_source=clock)
    registry.register(_worker(0, node_id="node-a"))
    registry.register(_worker(1, node_id="node-b"))
    spec = EnvSpecMsg(env_family="fake", resource_hints={"node_group": "node-b"})
    assert registry.select_rank(spec) == 1


def test_stale_heartbeat_marks_rank_unhealthy(clock: FakeClock) -> None:
    """A rank whose heartbeat timed out must be removed, and any session on
    it transitioned by the Gateway to ``LOST``.

    Args:
        clock: Controllable time source.
    """
    registry = EnvWorkerRegistry(time_source=clock, heartbeat_timeout_seconds=10.0)
    registry.register(_worker(0))
    assert registry.stale_ranks() == []

    clock.advance(11.0)
    assert registry.stale_ranks() == [0]
    registry.mark_unhealthy(0)
    _expect(ErrorCode.WORKER_LOST, registry.select_rank, EnvSpecMsg(env_family="fake"))

    registry.heartbeat(0)
    assert registry.stale_ranks() == []
    assert registry.select_rank(EnvSpecMsg(env_family="fake")) == 0
