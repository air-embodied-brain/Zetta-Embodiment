"""Dynamic slot scaling for the ``per_slot`` form (``core.env_execution.DynamicSlotPool``).

Coverage:

- When the pool is full and ``max_dynamic_pool_size`` allows growth,
  ``create_binding`` automatically appends a slot instead of explicitly
  rejecting (contrast with
  ``test_fault_isolation.py::test_pool_exhaustion_is_rejected_explicitly``'s
  "reject if growth is unsupported" branch);
- Once ``max_dynamic_pool_size`` is reached, requests are still explicitly
  rejected and the pool does not grow unbounded;
- Idle trailing slots are shrunk by ``shrink_idle``, but never below the
  initial ``pool_size``;
- Slots blocked by an in-use middle slot are not shrunk (to avoid index
  misalignment);
- The ``lockstep_vector`` form still enforces the fixed-pool D6 semantics
  even if ``max_dynamic_pool_size`` is declared (``add_slot`` / ``remove_slot``
  are rejected outright; the pool never grows);
- Under the served form, ``ServeLimits.max_pool_size`` clamps both
  ``pool_size`` and ``max_dynamic_pool_size``, so a client cannot bypass the
  server-side cap by declaring a larger dynamic growth target.
"""

from __future__ import annotations

from typing import Any

from rollout_runtime.api.enums import ErrorCode
from rollout_runtime.api.ids import SessionId
from rollout_runtime.api.messages import CreateSessionRequest, EnvSpecMsg
from rollout_runtime.api.result import Err, Ok
from rollout_runtime.launch.local import LocalRuntime
from rollout_runtime.serve.app import ServeLimits, _with_identity
from tests.runtime.conftest import open_sessions


async def test_pool_grows_on_demand_up_to_max_dynamic_pool_size(
    local_runtime: LocalRuntime, fake_env_spec: Any
) -> None:
    """When the pool is full and ``max_dynamic_pool_size`` has not been reached, it automatically grows by one slot."""
    spec = fake_env_spec(pool_size=1, max_dynamic_pool_size=3)
    (first,) = await open_sessions(local_runtime, spec, key_prefix="p0")

    created = await local_runtime.gateway.create_sessions(
        [
            CreateSessionRequest(
                application_id="test",
                client_session_key="p1-0",
                env_spec=spec,
                default_policy_id="fake",
            )
        ]
    )
    assert isinstance(created[0], Ok), created[0]
    second = created[0].value.session_id

    pool = local_runtime.env_workers[0].pools.find(spec.digest())
    assert pool is not None
    assert pool.pool_size == 2, "pool must have grown from 1 to 2 slots"
    assert pool.in_use == 2

    assert isinstance((await local_runtime.gateway.close_sessions([first]))[0], Ok)
    assert isinstance((await local_runtime.gateway.close_sessions([second]))[0], Ok)


async def test_pool_still_rejects_once_max_dynamic_pool_size_is_reached(
    local_runtime: LocalRuntime, fake_env_spec: Any
) -> None:
    """Once grown to ``max_dynamic_pool_size``, requests are still explicitly rejected and the pool does not grow unbounded."""
    spec = fake_env_spec(pool_size=1, max_dynamic_pool_size=2)
    sessions = await open_sessions(local_runtime, spec, count=2, key_prefix="q")

    pool = local_runtime.env_workers[0].pools.find(spec.digest())
    assert pool is not None
    assert pool.pool_size == 2

    denied = (
        await local_runtime.gateway.create_sessions(
            [
                CreateSessionRequest(
                    application_id="test",
                    client_session_key="q-overflow",
                    env_spec=spec,
                    default_policy_id="fake",
                )
            ]
        )
    )[0]
    assert isinstance(denied, Err)
    assert denied.error.code is ErrorCode.QUOTA_EXCEEDED
    assert "max_dynamic_pool_size" in denied.error.message
    assert pool.pool_size == 2, "pool must not grow past the declared max"

    for session in sessions:
        assert isinstance((await local_runtime.gateway.close_sessions([session]))[0], Ok)


async def test_shrink_idle_removes_trailing_free_slots_but_not_below_initial_size(
    local_runtime: LocalRuntime, fake_env_spec: Any
) -> None:
    """Idle trailing slots are shrunk, with the floor set to the initial ``pool_size`` at construction time."""
    spec = fake_env_spec(pool_size=1, max_dynamic_pool_size=4)
    sessions = await open_sessions(local_runtime, spec, count=3, key_prefix="r")

    pool = local_runtime.env_workers[0].pools.find(spec.digest())
    assert pool is not None
    assert pool.pool_size == 3

    for session in sessions:
        assert isinstance((await local_runtime.gateway.close_sessions([session]))[0], Ok)
    assert pool.in_use == 0

    removed = await pool.shrink_idle()
    assert removed == 2, "must shrink back down to the initial pool_size=1 floor"
    assert pool.pool_size == 1

    # The floor is the initial pool_size; it never shrinks to 0. Shrinking again is idempotent.
    removed_again = await pool.shrink_idle()
    assert removed_again == 0
    assert pool.pool_size == 1


async def test_shrink_idle_does_not_remove_slots_blocked_by_an_in_use_middle_slot(
    local_runtime: LocalRuntime, fake_env_spec: Any
) -> None:
    """When a trailing slot is still in use, the other idle slots are not shrunk (to avoid index misalignment)."""
    spec = fake_env_spec(pool_size=1, max_dynamic_pool_size=4)
    sessions = await open_sessions(local_runtime, spec, count=3, key_prefix="s")

    worker = local_runtime.env_workers[0]
    pool = worker.pools.find(spec.digest())
    assert pool is not None
    assert pool.pool_size == 3

    # Do not assume creation order equals slot allocation order (scheduling may
    # differ across transports): query the worker-side state directly to find
    # the session that actually holds the trailing index (``pool_size - 1``),
    # and only release the other two so the trailing slot stays occupied.
    trailing_index = pool.pool_size - 1
    trailing_session: SessionId | None = None
    for session_id in sessions:
        slot = worker.sessions.get(SessionId(session_id))
        if slot is not None and slot.slot_index == trailing_index:
            trailing_session = SessionId(session_id)
            break
    assert trailing_session is not None, "trailing slot must be held by some session"

    for session_id in sessions:
        if SessionId(session_id) != trailing_session:
            assert isinstance(
                (await local_runtime.gateway.close_sessions([session_id]))[0], Ok
            )
    assert pool.in_use == 1

    removed = await pool.shrink_idle()
    # The trailing slot is still in use, blocking every idle slot ahead of it: nothing is shrunk.
    assert removed == 0
    assert pool.pool_size == 3

    assert isinstance(
        (await local_runtime.gateway.close_sessions([trailing_session]))[0], Ok
    )


async def test_lockstep_vector_pool_stays_fixed_even_with_max_dynamic_pool_size(
    local_runtime: LocalRuntime, fake_env_spec: Any
) -> None:
    """The ``lockstep_vector`` form maintains fixed D6 pool semantics even when a growth cap is declared."""
    spec = fake_env_spec(
        pool_size=2,
        max_dynamic_pool_size=8,
        core_form="lockstep_vector",
    )
    sessions = await open_sessions(local_runtime, spec, count=2, key_prefix="t")

    pool = local_runtime.env_workers[0].pools.find(spec.digest())
    assert pool is not None
    assert pool.lockstep is True
    assert pool.dynamic is False, "lockstep_vector core must not expose DynamicSlotPool"
    assert pool.pool_size == 2

    denied = (
        await local_runtime.gateway.create_sessions(
            [
                CreateSessionRequest(
                    application_id="test",
                    client_session_key="t-overflow",
                    env_spec=spec,
                    default_policy_id="fake",
                )
            ]
        )
    )[0]
    assert isinstance(denied, Err)
    assert denied.error.code is ErrorCode.QUOTA_EXCEEDED
    assert "dynamic growth is not supported" in denied.error.message
    assert pool.pool_size == 2, "lockstep_vector pool must never grow past pool_size"

    for session in sessions:
        assert isinstance((await local_runtime.gateway.close_sessions([session]))[0], Ok)


def test_served_identity_clamps_max_dynamic_pool_size_to_server_limit() -> None:
    """Under the served form, ``max_dynamic_pool_size`` is subject to the same server-side hard cap as ``pool_size``.

    Regression guard: ``EnvSpecMsg.max_dynamic_pool_size`` (the client's
    declared growth intent) and ``ServeLimits.max_pool_size`` (the server's
    hard clamp on ``pool_size``) are two distinct concepts. Because their
    names are similar they can easily be mistaken for the same field and left
    unclamped -- which would let a client bypass server-side rate limiting and
    silently grow the pool to an unbounded size at runtime.
    """
    limits = ServeLimits(max_pool_size=4)
    request = CreateSessionRequest(
        application_id="ignored",
        client_session_key="k",
        env_spec=EnvSpecMsg(
            env_family="fake", pool_size=10, max_dynamic_pool_size=100
        ),
    )
    adjusted = _with_identity(request, "resolved-app", limits)
    assert adjusted.env_spec.pool_size == 4
    assert adjusted.env_spec.max_dynamic_pool_size == 4


def test_served_identity_leaves_pool_size_under_limit_untouched() -> None:
    """Declarations already within the server-side cap must be left untouched (to avoid affecting normal requests)."""
    limits = ServeLimits(max_pool_size=8)
    request = CreateSessionRequest(
        application_id="ignored",
        client_session_key="k",
        env_spec=EnvSpecMsg(
            env_family="fake", pool_size=2, max_dynamic_pool_size=5
        ),
    )
    adjusted = _with_identity(request, "resolved-app", limits)
    assert adjusted.env_spec.pool_size == 2
    assert adjusted.env_spec.max_dynamic_pool_size == 5


def test_effective_max_pool_size_defaults_to_pool_size() -> None:
    """When ``max_dynamic_pool_size`` is not set, dynamic growth is disabled by default (backward compatible with D6)."""
    spec = EnvSpecMsg(env_family="fake", pool_size=3)
    assert spec.effective_max_pool_size() == 3

    spec_with_smaller_max = EnvSpecMsg(
        env_family="fake", pool_size=3, max_dynamic_pool_size=1
    )
    assert spec_with_smaller_max.effective_max_pool_size() == 3, (
        "max_dynamic_pool_size below pool_size must not shrink the effective max"
    )


def test_pool_size_change_alone_does_not_affect_digest_semantics() -> None:
    """``max_dynamic_pool_size`` does not affect the digest: it only affects growth capacity, not pool reuse semantics."""
    base = EnvSpecMsg(env_family="fake", env_config={"a": 1}, pool_size=2)
    with_growth = EnvSpecMsg(
        env_family="fake", env_config={"a": 1}, pool_size=2, max_dynamic_pool_size=8
    )
    assert base.digest() == with_growth.digest()
