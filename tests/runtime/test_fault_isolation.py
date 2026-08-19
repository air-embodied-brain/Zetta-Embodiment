"""Dedicated verification: errors must never be raised across a WorkerGroup boundary.

Why this is a hard constraint: on a remote exception, ``WorkerGroupFuncResult._wait_for_results``
sets ``Cluster._run_failed = True`` and calls ``os.kill(pid, SIGUSR1)`` — **a single failed
request kills the entire job**. Therefore every public method of ``RuntimeEnvWorker`` /
``RuntimeRolloutWorker`` must be a total function: it must internally use
``try/except BaseException`` and normalize any exception into a returned
``ResultEnvelope(error=...)`` or ``ActionResponse(error=...)``.

Coverage: env raises / env hangs / policy fails per-request / policy raises as a whole /
invalid commands and control messages / invalid payload. Each case additionally asserts
that "other sessions remain usable" and "both workers are still serving".
"""

from __future__ import annotations

import asyncio
from typing import Any

import numpy as np

from rollout_runtime.api.enums import (
    EnvOperation,
    ErrorCode,
    OperationState,
    SessionState,
)
from rollout_runtime.api.ids import new_request_id
from rollout_runtime.api.internal import (
    CommandEnvelope,
    ControlEnvelope,
    InferenceRequest,
    make_routing_token,
)
from rollout_runtime.api.messages import (
    CreateSessionRequest,
    Observation,
    PolicyRequest,
    ResetSpec,
)
from rollout_runtime.api.result import Err, Ok, unwrap
from rollout_runtime.backends.fake.policy import FakePolicyConfig, FakePolicyCore
from rollout_runtime.core import payload as payload_module
from rollout_runtime.launch.local import LocalRuntime, build_local_components
from rollout_runtime.workers.rollout_worker import RuntimeRolloutWorker
from tests.runtime.conftest import local_runtime_config, open_sessions

POLICY = PolicyRequest(policy_id="fake")
"""Shared inference parameters."""


async def _healthy_pair(
    runtime: LocalRuntime, fake_env_spec: Any, **broken_config: Any
) -> tuple[Any, Any]:
    """Create one "broken env" session and one "healthy env" session, then reset each.

    Args:
        runtime: The in-process runtime.
        fake_env_spec: The env spec factory.
        **broken_config: The ``env_config`` injected into the broken session.

    Returns:
        ``(broken session, healthy session)``.
    """
    broken_spec = fake_env_spec(episode_length=64, **broken_config)
    healthy_spec = fake_env_spec(episode_length=64)
    (broken,) = await open_sessions(runtime, broken_spec, key_prefix="broken")
    (healthy,) = await open_sessions(runtime, healthy_spec, key_prefix="healthy")
    await runtime.gateway.reset([broken, healthy], ResetSpec(seed=1))
    return broken, healthy


async def test_env_exception_never_crosses_the_worker_boundary(
    local_runtime: LocalRuntime, fake_env_spec: Any
) -> None:
    """Env raises an exception -> ``ENV_FAILURE``, while the Gateway and other sessions
    remain intact.

    Args:
        local_runtime: The in-process runtime.
        fake_env_spec: The env spec factory.
    """
    gateway = local_runtime.gateway
    broken, healthy = await _healthy_pair(local_runtime, fake_env_spec, fail_on_step=1)

    failed = (await gateway.policy_step([broken], POLICY))[0]
    assert isinstance(failed, Err)
    assert failed.error.code is ErrorCode.ENV_FAILURE
    assert failed.error.retryable is False
    # Normalization preserves the exception type and traceback summary for debugging,
    # but the exception itself is never leaked.
    assert failed.error.detail["exception"] == "RuntimeError"
    assert "traceback" in failed.error.detail

    assert isinstance((await gateway.policy_step([healthy], POLICY))[0], Ok)
    assert (await gateway.get_session(broken)).state is SessionState.READY
    assert local_runtime.env_workers[0].serving is True
    assert local_runtime.rollout_workers[0].serving is True


async def test_env_reset_failure_is_isolated(
    local_runtime: LocalRuntime, fake_env_spec: Any
) -> None:
    """A ``reset`` failure also affects only that session, and never leaves a half-open
    episode.

    Args:
        local_runtime: The in-process runtime.
        fake_env_spec: The env spec factory.
    """
    gateway = local_runtime.gateway
    broken_spec = fake_env_spec(fail_on_reset=True)
    healthy_spec = fake_env_spec()
    (broken,) = await open_sessions(local_runtime, broken_spec, key_prefix="rb")
    (healthy,) = await open_sessions(local_runtime, healthy_spec, key_prefix="rh")

    results = await gateway.reset([broken, healthy], ResetSpec(seed=1))
    assert isinstance(results[0], Err)
    assert results[0].error.code is ErrorCode.ENV_FAILURE
    assert isinstance(results[1], Ok)

    status = await gateway.get_session(broken)
    assert status.state is SessionState.READY
    assert status.episode_id is None
    follow_up = (await gateway.policy_step([broken], POLICY))[0]
    assert isinstance(follow_up, Err)
    assert follow_up.error.code is ErrorCode.SESSION_NOT_READY


async def test_hanging_env_step_times_out_without_blocking_others(
    local_runtime: LocalRuntime, fake_env_spec: Any
) -> None:
    """A hanging env step times out as ``DEADLINE_EXCEEDED`` on schedule, without
    affecting other sessions.

    ``hang_timeout_seconds`` is deliberately bounded: an indefinitely blocking thread
    would make the ``ThreadPoolExecutor`` join hang when the interpreter exits.

    Args:
        local_runtime: The in-process runtime.
        fake_env_spec: The env spec factory.
    """
    gateway = local_runtime.gateway
    broken, healthy = await _healthy_pair(
        local_runtime,
        fake_env_spec,
        hang_on_step=1,
        hang_timeout_seconds=0.2,
    )
    hung = asyncio.create_task(gateway.policy_step([broken], POLICY))

    # Other sessions continue to be served normally while one is hanging (the env call
    # runs on a thread, so the event loop is not blocked).
    for _ in range(2):
        assert isinstance((await gateway.policy_step([healthy], POLICY))[0], Ok)

    result = (await hung)[0]
    assert isinstance(result, Err)
    assert result.error.code is ErrorCode.DEADLINE_EXCEEDED
    assert isinstance((await gateway.policy_step([healthy], POLICY))[0], Ok)


async def test_policy_failure_is_reported_per_request(
    fake_env_spec: Any, transport_kind: str
) -> None:
    """Policy fails per-request -> ``POLICY_FAILURE``, without affecting other requests
    on the same rank.

    Args:
        fake_env_spec: The env spec factory.
        transport_kind: The transport under test.
    """
    runtime = build_local_components(local_runtime_config(transport_kind))
    await runtime.start()
    try:
        gateway = runtime.gateway
        spec = fake_env_spec(pool_size=2, episode_length=64)
        session_ids = await open_sessions(runtime, spec, count=2)
        await gateway.reset(session_ids, ResetSpec(seed=1))
        runtime.policies[0].config.fail_sessions = frozenset({str(session_ids[0])})

        results = await gateway.policy_step(session_ids, POLICY)
        assert isinstance(results[0], Err)
        assert results[0].error.code is ErrorCode.POLICY_FAILURE
        assert isinstance(results[1], Ok)
        # The policy failure occurs before the env step, so there is no side effect.
        assert results[0].error.side_effect_applied is False
        stats = unwrap(
            (await gateway.extension_call([session_ids[0]], "fake", "stats", {}))[0]
        )
        assert stats["chunk_calls"] == 0

        runtime.policies[0].config.fail_sessions = frozenset()
        assert isinstance((await gateway.policy_step([session_ids[0]], POLICY))[0], Ok)
    finally:
        await runtime.gateway.stop()
        await runtime.aclose()


async def test_rollout_worker_survives_a_raising_policy_core() -> None:
    """When the policy core itself raises, ``infer_batch`` still normalizes per-request,
    and the worker does not die."""

    class ExplodingPolicy(FakePolicyCore):
        """A policy core that raises on every inference call."""

        async def ainfer_batch(self, requests: list[InferenceRequest]) -> Any:
            """Always raises.

            Args:
                requests: The inference requests.

            Raises:
                RuntimeError: Injected unconditionally.
            """
            del requests
            raise RuntimeError("policy exploded")

    worker = RuntimeRolloutWorker(policy=ExplodingPolicy(FakePolicyConfig()))
    worker.init_worker({})
    request = InferenceRequest(
        request_id=new_request_id(),
        session_id="sess-x",  # type: ignore[arg-type]
        policy_id="fake",
        observation=Observation(
            session_id="sess-x",  # type: ignore[arg-type]
            episode_id=1,  # type: ignore[arg-type]
            step_index=0,
        ),
        routing_token=make_routing_token("env", 0),
        compat_key="k",
    )
    responses = await worker.infer_batch([request])
    assert len(responses) == 1
    assert responses[0].error is not None
    assert responses[0].error.code is ErrorCode.INTERNAL
    assert responses[0].request_id == request.request_id

    # Same behavior with no policy attached: an error response is returned rather than raised.
    naked = RuntimeRolloutWorker()
    empty = await naked.infer_batch([request])
    assert empty[0].error is not None
    assert await naked.infer_batch([]) == []


async def test_worker_handlers_are_total_functions(
    local_runtime: LocalRuntime, fake_env_spec: Any
) -> None:
    """Invalid commands / control messages / payloads always return an envelope with an
    error, never raise.

    Args:
        local_runtime: The in-process runtime.
        fake_env_spec: The env spec factory.
    """
    worker = local_runtime.env_workers[0]
    (session_id,) = await open_sessions(local_runtime, fake_env_spec())
    await local_runtime.gateway.reset([session_id], ResetSpec(seed=1))

    unknown_session = await worker.handle_command(
        CommandEnvelope(
            request_id=new_request_id(),
            session_id="sess-does-not-exist",  # type: ignore[arg-type]
            operation=EnvOperation.OBSERVE,
        )
    )
    assert unknown_session.error is not None
    assert unknown_session.error.code is ErrorCode.UNKNOWN_SESSION
    assert unknown_session.state is OperationState.FAILED

    # A control-plane operation received on the command channel: error, not a crash.
    wrong_channel = await worker.handle_command(
        CommandEnvelope(
            request_id=new_request_id(),
            session_id=session_id,
            operation=EnvOperation.CREATE_BINDING,
        )
    )
    assert wrong_channel.error is not None
    assert wrong_channel.error.code is ErrorCode.INVALID_ARGUMENT

    missing_payload = await worker.handle_command(
        CommandEnvelope(
            request_id=new_request_id(),
            session_id=session_id,
            operation=EnvOperation.ACTION_STEP,
            payload={},
        )
    )
    assert missing_payload.error is not None
    assert missing_payload.error.code is ErrorCode.INVALID_ARGUMENT

    non_finite = await worker.handle_command(
        CommandEnvelope(
            request_id=new_request_id(),
            session_id=session_id,
            operation=EnvOperation.ACTION_STEP,
            payload={
                "actions": payload_module.encode_array(
                    np.full((1, 7), np.nan, dtype=np.float32)
                )
            },
        )
    )
    assert non_finite.error is not None
    assert non_finite.error.code is ErrorCode.INVALID_ARGUMENT

    wrong_dim = await worker.handle_command(
        CommandEnvelope(
            request_id=new_request_id(),
            session_id=session_id,
            operation=EnvOperation.ACTION_STEP,
            payload={
                "actions": payload_module.encode_array(
                    np.zeros((1, 3), dtype=np.float32)
                )
            },
        )
    )
    assert wrong_dim.error is not None
    assert wrong_dim.error.code is ErrorCode.INVALID_ARGUMENT

    unsupported_control = await worker.handle_control(
        ControlEnvelope(request_id=new_request_id(), operation=EnvOperation.OBSERVE)
    )
    assert unsupported_control.error is not None
    assert unsupported_control.error.code is ErrorCode.INVALID_ARGUMENT

    # The environment is still usable.
    assert isinstance(
        (await local_runtime.gateway.policy_step([session_id], POLICY))[0], Ok
    )


async def test_unsupported_extension_does_not_crash(
    local_runtime: LocalRuntime, fake_env_spec: Any
) -> None:
    """An undeclared extension method returns ``UNSUPPORTED_EXTENSION`` without crashing.

    Args:
        local_runtime: The in-process runtime.
        fake_env_spec: The env spec factory.
    """
    gateway = local_runtime.gateway
    (session_id,) = await open_sessions(local_runtime, fake_env_spec())
    await gateway.reset([session_id], ResetSpec(seed=1))

    missing = (await gateway.extension_call([session_id], "fake", "nope", {}))[0]
    assert isinstance(missing, Err)
    assert missing.error.code is ErrorCode.UNSUPPORTED_EXTENSION
    wrong_namespace = (
        await gateway.extension_call([session_id], "libero", "ping", {})
    )[0]
    assert isinstance(wrong_namespace, Err)
    assert wrong_namespace.error.code is ErrorCode.UNSUPPORTED_EXTENSION

    pong = unwrap(
        (await gateway.extension_call([session_id], "fake", "ping", {"n": 1}))[0]
    )
    assert pong["pong"] is True
    assert pong["echo"] == {"n": 1}


async def test_pool_exhaustion_is_rejected_explicitly(
    local_runtime: LocalRuntime, fake_env_spec: Any
) -> None:
    """When pool capacity is exhausted, explicitly reject with ``QUOTA_EXCEEDED``
    rather than silently expanding the pool.

    Args:
        local_runtime: The in-process runtime.
        fake_env_spec: The env spec factory.
    """
    spec = fake_env_spec(pool_size=1)
    (first,) = await open_sessions(local_runtime, spec, key_prefix="p0")

    denied = (
        await local_runtime.gateway.create_sessions(
            [
                CreateSessionRequest(
                    application_id="test",
                    client_session_key="p1-0",
                    env_spec=spec,
                    default_policy_id="fake",
                )
            ]
        )
    )[0]
    assert isinstance(denied, Err)
    assert denied.error.code is ErrorCode.QUOTA_EXCEEDED
    # When max_dynamic_pool_size is undeclared, it defaults to pool_size (dynamic pool
    # growth is off by default), so once the pool is full the request **must** be
    # explicitly rejected rather than silently expanded — the specific reason depends
    # on whether this core implements DynamicSlotPool (the fake core does, so this is
    # "limit reached"; families that do not support expansion produce a different
    # message), but either reason must appear in the error info to aid debugging.
    assert denied.error.detail.get("pool_size") == 1
    assert denied.error.detail.get("max_dynamic_pool_size") == 1
    # The first session remains intact; once released, its slot can be reused.
    assert isinstance((await local_runtime.gateway.close_sessions([first]))[0], Ok)
    (reused,) = await open_sessions(local_runtime, spec, key_prefix="p2")
    assert reused != first
    pool = local_runtime.env_workers[0].pools.find(spec.digest())
    assert pool is not None
    assert pool.in_use == 1
