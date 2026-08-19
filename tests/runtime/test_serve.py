"""Authentication, ownership validation, and untrusted-input containment for
``rollout-runtime serve``.

Covers three hard requirements plus the untrusted-input checklist:

- bind to a **non-loopback** address without ``RR_AUTH_TOKEN`` set → **refuse to
  start** (and this must happen before the runtime is built);
- wrong token / missing ``Authorization`` header / non-Bearer scheme →
  ``INVALID_ARGUMENT`` + ``detail.reason="authentication"`` (**no new
  ErrorCode**), HTTP 401;
- ``application_id`` is determined by the token; whatever is in the request body
  has **zero effect**;
- ownership validation for session / request: another caller's id and a
  nonexistent id must be **indistinguishable** to the caller;
- clamping of ``lease_seconds`` / ``pool_size`` / ``max_steps`` / body size;
- outbound responses strip ``detail.traceback`` by default.

The body is always msgpack via ``api.wire`` (the finalized serialization form),
so assertions here operate directly on protocol objects with no second schema.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest
import pytest_asyncio

from rollout_runtime.api import wire
from rollout_runtime.api.enums import ErrorCode, SessionState
from rollout_runtime.api.errors import RuntimeApiError
from rollout_runtime.api.messages import (
    CreateSessionRequest,
    EnvSpecMsg,
    EpisodeRequest,
    PolicyRequest,
    ResetSpec,
)
from rollout_runtime.api.result import Err, Ok
from rollout_runtime.serve.app import ServeLimits, build_app
from rollout_runtime.serve.auth import (
    AUTH_TOKEN_ENV,
    ServeSecurityError,
    TokenAuthority,
    is_loopback_host,
)
from rollout_runtime.serve.server import ServeOptions, build_served_runtime

TOKEN_A = "token-for-team-a"
TOKEN_B = "token-for-team-b"
TOKENS = {AUTH_TOKEN_ENV: f"teamA:{TOKEN_A},teamB:{TOKEN_B}"}


def env_spec(pool_size: int = 2) -> EnvSpecMsg:
    """Build an env spec for the fake family.

    Args:
        pool_size: Number of pool slots.

    Returns:
        ``EnvSpecMsg``.
    """
    return EnvSpecMsg(
        env_family="fake",
        env_config={"chunk_size": 2, "episode_length": 64},
        pool_size=pool_size,
    )


def create_body(key: str, **overrides: Any) -> bytes:
    """Build the msgpack body for one ``POST /v1/sessions`` request.

    Args:
        key: ``client_session_key``.
        **overrides: Fields overriding ``CreateSessionRequest``.

    Returns:
        msgpack bytes.
    """
    fields: dict[str, Any] = {
        "application_id": "",
        "client_session_key": key,
        "env_spec": env_spec(),
        "lease_seconds": 60.0,
    }
    fields.update(overrides)
    return wire.encode_bytes({"requests": [CreateSessionRequest(**fields)]})


@pytest_asyncio.fixture(loop_scope="function")
async def served() -> AsyncIterator[Any]:
    """Start an authenticated served runtime (fake backend, inproc).

    Yields:
        ``(ServedRuntime, httpx.AsyncClient)``.
    """
    runtime = await build_served_runtime(
        ServeOptions(
            config="local_fake",
            host="127.0.0.1",
            gateway_epoch=7,
            limits=ServeLimits(
                max_lease_seconds=120.0,
                max_pool_size=2,
                max_episode_steps=3,
                max_body_bytes=4096,
            ),
        ),
        environ=dict(TOKENS),
    )
    transport = httpx.ASGITransport(app=runtime.app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://serve.test"
    ) as client:
        yield runtime, client
    await runtime.aclose()


def bearer(token: str) -> dict[str, str]:
    """Build the ``Authorization`` header.

    Args:
        token: Bearer token.

    Returns:
        Header dict.
    """
    return {"Authorization": f"Bearer {token}"}


def decode(response: httpx.Response) -> Any:
    """Decode an msgpack response body.

    Args:
        response: HTTP response.

    Returns:
        Protocol object.
    """
    return wire.decode_bytes(response.content)


# ------------------------------------------------------------ startup-time security


def test_loopback_detection_covers_names_and_ipv6() -> None:
    """Loopback detection: ``127.0.0.1`` / ``::1`` / ``localhost`` are loopback,
    everything else is not."""
    assert is_loopback_host("127.0.0.1")
    assert is_loopback_host("127.5.6.7")
    assert is_loopback_host("::1")
    assert is_loopback_host("[::1]")
    assert is_loopback_host("localhost")
    assert not is_loopback_host("0.0.0.0")
    assert not is_loopback_host("192.168.1.10")
    assert not is_loopback_host("")
    assert not is_loopback_host("runtime.internal")


def test_non_loopback_bind_without_a_token_refuses_to_start() -> None:
    """**Negative case**: binding ``0.0.0.0`` without ``RR_AUTH_TOKEN`` must refuse
    to start."""
    with pytest.raises(ServeSecurityError) as excinfo:
        TokenAuthority.from_environment(host="0.0.0.0", environ={})
    assert AUTH_TOKEN_ENV in str(excinfo.value)


def test_non_loopback_bind_with_a_token_is_allowed() -> None:
    """Non-loopback bind is only allowed once a token is provided, and the token
    table is parsed into multiple tenants."""
    authority = TokenAuthority.from_environment(host="0.0.0.0", environ=dict(TOKENS))
    assert authority.enabled
    assert authority.application_ids == ["teamA", "teamB"]


def test_loopback_bind_without_a_token_starts_without_auth() -> None:
    """Loopback + no token allows startup, but identity is still assigned by the
    server (not self-declared by the client)."""
    authority = TokenAuthority.from_environment(host="127.0.0.1", environ={})
    assert not authority.enabled
    assert authority.resolve(None) == "local"


def test_malformed_token_table_is_rejected() -> None:
    """A blank token / duplicate application_id must always refuse to start,
    rather than silently dropping one entry."""
    with pytest.raises(ServeSecurityError):
        TokenAuthority.from_environment(
            host="0.0.0.0", environ={AUTH_TOKEN_ENV: "teamA:"}
        )
    with pytest.raises(ServeSecurityError):
        TokenAuthority.from_environment(
            host="0.0.0.0", environ={AUTH_TOKEN_ENV: "teamA:x,teamA:y"}
        )


async def test_refusal_happens_before_anything_is_built(monkeypatch) -> None:
    """Refusing to start must happen **before** the runtime is built (otherwise
    minutes of weight-loading would be burned first).

    Args:
        monkeypatch: pytest fixture used to swap the launcher with an exploding
            stub.
    """
    called: list[str] = []

    def explode(*_args: Any, **_kwargs: Any) -> Any:
        called.append("build")
        raise AssertionError("the runtime must not be built when auth is missing")

    monkeypatch.setattr("rollout_runtime.launch.local.build_local_components", explode)
    with pytest.raises(ServeSecurityError):
        await build_served_runtime(
            ServeOptions(config="local_fake", host="0.0.0.0"), environ={}
        )
    assert called == []


# ------------------------------------------------------------ request-time authentication


async def test_requests_without_a_bearer_header_are_rejected(served) -> None:
    """**Negative case**: a missing ``Authorization`` header → 401 +
    ``reason="authentication"``.

    Args:
        served: fixture.
    """
    _runtime, client = served
    response = await client.post("/v1/sessions", content=create_body("k"))
    assert response.status_code == 401
    assert response.headers["WWW-Authenticate"] == "Bearer"
    info = decode(response)
    assert info.code is ErrorCode.INVALID_ARGUMENT
    assert info.detail["reason"] == "authentication"


async def test_a_wrong_token_is_rejected(served) -> None:
    """**Negative case**: a wrong token must be rejected, and the error message
    must not echo the token back.

    Args:
        served: fixture.
    """
    _runtime, client = served
    response = await client.post(
        "/v1/sessions", content=create_body("k"), headers=bearer("not-a-real-token")
    )
    assert response.status_code == 401
    info = decode(response)
    assert info.code is ErrorCode.INVALID_ARGUMENT
    assert info.detail["reason"] == "authentication"
    assert "not-a-real-token" not in info.message


async def test_a_non_bearer_scheme_is_rejected(served) -> None:
    """Schemes such as ``Basic`` are not accepted (the token must go through
    Bearer).

    Args:
        served: fixture.
    """
    _runtime, client = served
    response = await client.post(
        "/v1/sessions",
        content=create_body("k"),
        headers={"Authorization": f"Basic {TOKEN_A}"},
    )
    assert response.status_code == 401


async def test_livez_is_the_only_unauthenticated_endpoint(served) -> None:
    """``/livez`` requires no authentication and carries no topology
    information; ``/healthz`` and ``/metrics`` both require authentication.

    Args:
        served: fixture.
    """
    _runtime, client = served
    alive = await client.get("/livez")
    assert alive.status_code == 200
    assert alive.json() == {"status": "ok"}
    assert (await client.get("/healthz")).status_code == 401
    assert (await client.get("/metrics")).status_code == 401


# ------------------------------------------------- application_id is not client-supplied


async def test_application_id_comes_from_the_token_not_the_request_body(
    served,
) -> None:
    """``application_id`` in the request body has zero effect: quota and
    ownership are both recorded against the token's tenant.

    ``CreateSessionRequest.application_id`` is client-supplied in the embedded
    form, but in the served form it must be overridden by the token, otherwise
    one tenant could impersonate another tenant to consume its quota.

    Args:
        served: fixture.
    """
    runtime, client = served
    response = await client.post(
        "/v1/sessions",
        content=create_body("k1", application_id="teamB", auth_token=TOKEN_B),
        headers=bearer(TOKEN_A),
    )
    assert response.status_code == 200
    results = decode(response)
    assert isinstance(results[0], Ok), results[0]
    handle = results[0].value
    # Identity is teamA (from the token), not teamB as written in the request body.
    assert handle.application_id == "teamA"
    record = runtime.gateway.sessions.get(handle.session_id)
    assert record.application_id == "teamA"
    snapshot = runtime.gateway.admission.snapshot()
    assert snapshot.sessions_per_application == {"teamA": 1}
    assert handle.gateway_epoch == 7


async def test_sessions_of_another_tenant_are_reported_as_unknown(served) -> None:
    """Operating on another tenant's session across tenants returns
    ``UNKNOWN_SESSION`` (does not leak existence).

    Args:
        served: fixture.
    """
    _runtime, client = served
    created = decode(
        await client.post(
            "/v1/sessions", content=create_body("owned"), headers=bearer(TOKEN_A)
        )
    )
    session_id = created[0].value.session_id

    stolen = await client.post(
        "/v1/reset",
        content=wire.encode_bytes(
            {"session_ids": [session_id], "reset_spec": ResetSpec(seed=1)}
        ),
        headers=bearer(TOKEN_B),
    )
    assert stolen.status_code == 200
    results = decode(stolen)
    assert isinstance(results[0], Err)
    assert results[0].error.code is ErrorCode.UNKNOWN_SESSION

    # The exact same error code and message shape as "a session that doesn't exist".
    missing = decode(
        await client.post(
            "/v1/reset",
            content=wire.encode_bytes(
                {"session_ids": ["sess-doesnotexist"], "reset_spec": ResetSpec()}
            ),
            headers=bearer(TOKEN_B),
        )
    )
    assert missing[0].error.code is ErrorCode.UNKNOWN_SESSION
    assert results[0].error.code.name == missing[0].error.code.name, (
        "foreign and missing sessions must be indistinguishable"
    )

    # The real owner can still use it (ownership validation doesn't also block
    # the owner's own session).
    own = decode(
        await client.post(
            "/v1/reset",
            content=wire.encode_bytes(
                {"session_ids": [session_id], "reset_spec": ResetSpec(seed=1)}
            ),
            headers=bearer(TOKEN_A),
        )
    )
    assert isinstance(own[0], Ok), own[0]


async def test_get_session_of_another_tenant_is_404(served) -> None:
    """Ownership validation on the single-object endpoint is projected as 404.

    Args:
        served: fixture.
    """
    _runtime, client = served
    created = decode(
        await client.post(
            "/v1/sessions", content=create_body("owned"), headers=bearer(TOKEN_A)
        )
    )
    session_id = created[0].value.session_id
    assert (
        await client.get(f"/v1/sessions/{session_id}", headers=bearer(TOKEN_A))
    ).status_code == 200
    foreign = await client.get(f"/v1/sessions/{session_id}", headers=bearer(TOKEN_B))
    assert foreign.status_code == 404
    assert decode(foreign).code is ErrorCode.UNKNOWN_SESSION


async def test_request_status_and_cancel_are_scoped_to_the_owner(served) -> None:
    """``request_id`` must also be ownership-validated: guessing another
    caller's idempotency key must not allow reading the result or cancelling.

    Args:
        served: fixture.
    """
    _runtime, client = served
    created = decode(
        await client.post(
            "/v1/sessions", content=create_body("owned"), headers=bearer(TOKEN_A)
        )
    )
    session_id = created[0].value.session_id
    request_id = "req-owned-by-team-a"
    await client.post(
        "/v1/reset",
        content=wire.encode_bytes(
            {
                "session_ids": [session_id],
                "reset_spec": ResetSpec(seed=1),
                "request_ids": [request_id],
            }
        ),
        headers=bearer(TOKEN_A),
    )
    mine = await client.get(f"/v1/requests/{request_id}", headers=bearer(TOKEN_A))
    assert mine.status_code == 200
    assert decode(mine).request_id == request_id

    for method, path in (
        ("get", f"/v1/requests/{request_id}"),
        ("post", f"/v1/requests/{request_id}/cancel"),
    ):
        response = await getattr(client, method)(path, headers=bearer(TOKEN_B))
        assert response.status_code == 404
        assert decode(response).code is ErrorCode.UNKNOWN_SESSION


# ------------------------------------------------------------ input containment


async def test_lease_and_pool_size_are_clamped(served) -> None:
    """``lease_seconds`` and ``pool_size`` are clamped by server-side limits.

    Args:
        served: fixture.
    """
    runtime, client = served
    created = decode(
        await client.post(
            "/v1/sessions",
            content=wire.encode_bytes(
                {
                    "requests": [
                        CreateSessionRequest(
                            application_id="",
                            client_session_key="greedy",
                            env_spec=EnvSpecMsg(
                                env_family="fake",
                                env_config={"chunk_size": 2},
                                pool_size=999,
                            ),
                            lease_seconds=10**9,
                        )
                    ]
                }
            ),
            headers=bearer(TOKEN_A),
        )
    )
    assert isinstance(created[0], Ok), created[0]
    handle = created[0].value
    record = runtime.gateway.sessions.get(handle.session_id)
    assert record.env_spec.pool_size == 2, "pool_size must be clamped to max_pool_size"
    lease_left = handle.lease_expiration - record.created_at
    assert lease_left <= 121.0, f"lease was not clamped: {lease_left}"


async def test_a_foreign_env_family_is_rejected(served) -> None:
    """Only the family declared in the effective config is allowed.

    Args:
        served: fixture.
    """
    _runtime, client = served
    created = decode(
        await client.post(
            "/v1/sessions",
            content=wire.encode_bytes(
                {
                    "requests": [
                        CreateSessionRequest(
                            application_id="",
                            client_session_key="alien",
                            env_spec=EnvSpecMsg(env_family="libero", env_config={}),
                        )
                    ]
                }
            ),
            headers=bearer(TOKEN_A),
        )
    )
    assert isinstance(created[0], Err)
    assert created[0].error.code is ErrorCode.UNSUPPORTED_ENV_SPEC


async def test_episode_max_steps_is_clamped(served) -> None:
    """``EpisodeRequest.max_steps`` is clamped to prevent a long in-worker loop
    from occupying a slot.

    Args:
        served: fixture.
    """
    _runtime, client = served
    created = decode(
        await client.post(
            "/v1/sessions", content=create_body("ep"), headers=bearer(TOKEN_A)
        )
    )
    session_id = created[0].value.session_id
    await client.post(
        "/v1/reset",
        content=wire.encode_bytes(
            {"session_ids": [session_id], "reset_spec": ResetSpec(seed=1)}
        ),
        headers=bearer(TOKEN_A),
    )
    results = decode(
        await client.post(
            "/v1/run_episode",
            content=wire.encode_bytes(
                {
                    "session_ids": [session_id],
                    "episode_request": EpisodeRequest(
                        max_steps=10_000, policy=PolicyRequest(policy_id="fake")
                    ),
                }
            ),
            headers=bearer(TOKEN_A),
        )
    )
    assert isinstance(results[0], Ok), results[0]
    assert results[0].value.num_policy_steps <= 3


async def test_an_oversized_body_is_rejected_before_decoding(served) -> None:
    """A body exceeding ``--max-body-bytes`` is rejected outright, without
    entering msgpack decoding.

    Args:
        served: fixture.
    """
    _runtime, client = served
    response = await client.post(
        "/v1/sessions",
        content=b"\x00" * 8192,
        headers=bearer(TOKEN_A),
    )
    assert response.status_code == 400
    info = decode(response)
    assert info.detail["reason"] == "body_too_large"


async def test_a_malformed_body_becomes_invalid_argument(served) -> None:
    """Malformed msgpack is normalized to ``INVALID_ARGUMENT``, not a 500 HTML
    page.

    Args:
        served: fixture.
    """
    _runtime, client = served
    response = await client.post(
        "/v1/sessions", content=b"\xc1\xc1\xc1", headers=bearer(TOKEN_A)
    )
    assert response.status_code == 400
    assert decode(response).code is ErrorCode.INVALID_ARGUMENT


async def test_tracebacks_are_stripped_from_responses_by_default(served) -> None:
    """Outbound redaction: ``detail.traceback`` is not returned to the caller by
    default.

    Args:
        served: fixture.
    """
    _runtime, client = served
    created = decode(
        await client.post(
            "/v1/sessions", content=create_body("tb"), headers=bearer(TOKEN_A)
        )
    )
    session_id = created[0].value.session_id
    # policy_step without a prior reset → SESSION_NOT_READY, going through the
    # normalize_exception path.
    results = decode(
        await client.post(
            "/v1/policy_step",
            content=wire.encode_bytes({"session_ids": [session_id]}),
            headers=bearer(TOKEN_A),
        )
    )
    assert isinstance(results[0], Err)
    assert results[0].error.code is ErrorCode.SESSION_NOT_READY
    assert "traceback" not in results[0].error.detail


# ------------------------------------------------------------ normal path


async def test_auth_is_checked_before_any_body_is_read(served) -> None:
    """**Authenticate first, read the body second**: an unauthenticated caller
    must never reach the msgpack decoder.

    An ordering bug found by an independent audit: ``identify()`` used to live
    inside ``batch_call``, i.e. after ``read_body()``, so a ``POST /v1/reset``
    without ``Authorization`` and with a garbage body returned 400 "not valid
    msgpack" (an oracle for "is the body well-formed") instead of 401.

    Args:
        served: fixture.
    """
    _runtime, client = served
    for path in (
        "/v1/reset",
        "/v1/observe",
        "/v1/policy_step",
        "/v1/policy_infer",
        "/v1/action_step",
        "/v1/run_episode",
        "/v1/extension_call",
        "/v1/sessions/renew",
        "/v1/sessions/close",
        "/v1/sessions",
    ):
        response = await client.post(path, content=b"\xc1\xc1\xc1")
        assert response.status_code == 401, f"{path} leaked past auth: {response.text}"
        info = decode(response)
        assert info.detail["reason"] == "authentication", path


async def test_a_chunked_body_is_capped_without_buffering_everything(served) -> None:
    """A chunked body (no ``Content-Length``) must also be capped.

    ``await request.body()`` buffers the whole stream into memory before
    measuring its length: an independent audit measured driver RSS growing by
    868 MB for a 419 MB chunked body before returning 400, even though the
    declared limit is 16 MiB. The gateway is the data plane's single asyncio
    writer, so OOM-killing it would take down every tenant.

    Args:
        served: fixture.
    """
    _runtime, client = served
    sent = 0

    async def stream() -> Any:
        nonlocal sent
        for _ in range(64):  # 64 KiB, far beyond the fixture's 4096 limit
            sent += 1024
            yield b"\x00" * 1024

    response = await client.post(
        "/v1/sessions", content=stream(), headers=bearer(TOKEN_A)
    )
    assert response.status_code == 400
    assert decode(response).detail["reason"] == "body_too_large"
    # Reading stops right as the limit is exceeded: it must not absorb the
    # entire 64 KiB.
    assert sent <= 4096 + 2048, f"the whole stream was buffered: {sent} bytes"


def test_a_non_ascii_token_is_an_authentication_failure() -> None:
    """A non-ASCII token must be normalized into an authentication failure,
    not an internal ``TypeError``.

    ``hmac.compare_digest`` requires ASCII-only str; previously passing a str
    directly would raise TypeError, which got normalized into a 400 with
    `detail` carrying `exception=TypeError` and CPython's error string --
    leaking internal implementation details to an unauthenticated caller, and
    also dropping `reason="authentication"` (found by an independent audit).

    This test calls ``TokenAuthority.resolve`` directly: httpx itself refuses
    to send non-ASCII headers, but a real HTTP client can (Starlette decodes
    them as latin-1), so the server must handle this input on its own.
    """
    authority = TokenAuthority.from_environment(host="0.0.0.0", environ=dict(TOKENS))
    for header in ("Bearer tökA", "Bearer 令牌", f"Bearer {TOKEN_A}ä"):
        with pytest.raises(RuntimeApiError) as excinfo:
            authority.resolve(header)
        info = excinfo.value.info
        assert info.code is ErrorCode.INVALID_ARGUMENT
        assert info.detail["reason"] == "authentication"
        assert "TypeError" not in info.message
    # The correct token still passes (the fix didn't break the comparison).
    assert authority.resolve(f"Bearer {TOKEN_A}") == "teamA"


def test_token_table_uses_the_injected_environment_only() -> None:
    """The token table only reads the injected environ; the process
    environment must not override it.

    Previously `_parse_tokens` read `os.environ` directly for
    `RR_AUTH_APPLICATION_ID`, so under single-tenant configuration, tenant
    ownership could become inconsistent with `fallback_application_id`, and
    the process environment could silently change quota and audit ownership
    (found by an independent audit).
    """
    import os

    os.environ["RR_AUTH_APPLICATION_ID"] = "fromProcessEnv"
    try:
        authority = TokenAuthority.from_environment(
            host="0.0.0.0",
            environ={"RR_AUTH_TOKEN": "s3cr3t", "RR_AUTH_APPLICATION_ID": "teamA"},
        )
        assert authority.tokens == {"teamA": "s3cr3t"}
        assert authority.fallback_application_id == "teamA"
        assert authority.resolve("Bearer s3cr3t") == "teamA"
    finally:
        os.environ.pop("RR_AUTH_APPLICATION_ID", None)


def test_a_blank_token_table_refuses_to_start() -> None:
    """Providing ``RR_AUTH_TOKEN`` but with no parseable entries → refuse to
    start (do not silently disable authentication)."""
    for blank in (",", "   ", ",,"):
        with pytest.raises(ServeSecurityError):
            TokenAuthority.from_environment(
                host="127.0.0.1", environ={AUTH_TOKEN_ENV: blank}
            )
    # Not setting it at all is still allowed (loopback dev mode).
    assert not TokenAuthority.from_environment(host="127.0.0.1", environ={}).enabled


def test_the_unauthenticated_path_table_is_explicit() -> None:
    """"Who can skip authentication" is declared in exactly one place, so
    adding an endpoint cannot accidentally miss authentication."""
    from rollout_runtime.serve.app import _UNAUTHENTICATED_PATHS

    assert _UNAUTHENTICATED_PATHS == frozenset({"/livez"})


async def test_the_full_lifecycle_works_over_http(served) -> None:
    """create → reset → observe → policy_step → close, entirely over HTTP.

    Args:
        served: fixture.
    """
    runtime, client = served
    created = decode(
        await client.post(
            "/v1/sessions", content=create_body("life"), headers=bearer(TOKEN_A)
        )
    )
    session_id = created[0].value.session_id
    body = wire.encode_bytes({"session_ids": [session_id]})

    reset = decode(
        await client.post(
            "/v1/reset",
            content=wire.encode_bytes(
                {"session_ids": [session_id], "reset_spec": ResetSpec(seed=3)}
            ),
            headers=bearer(TOKEN_A),
        )
    )
    assert isinstance(reset[0], Ok), reset[0]
    assert reset[0].value.episode_id == 1

    observed = decode(
        await client.post("/v1/observe", content=body, headers=bearer(TOKEN_A))
    )
    assert isinstance(observed[0], Ok)
    assert observed[0].value.main_image is not None

    stepped = decode(
        await client.post(
            "/v1/policy_step",
            content=wire.encode_bytes(
                {
                    "session_ids": [session_id],
                    "policy_request": PolicyRequest(policy_id="fake"),
                }
            ),
            headers=bearer(TOKEN_A),
        )
    )
    assert isinstance(stepped[0], Ok), stepped[0]
    step = stepped[0].value
    # chunk length is given by the **policy backend** (the rollout worker's
    # config), not by the session's env_spec, so this asserts self-consistency:
    # actual step count == number of per-step records.
    assert step.executed_horizon > 0
    assert step.per_step is not None
    assert len(step.per_step) == step.executed_horizon

    status = decode(
        await client.get(f"/v1/sessions/{session_id}", headers=bearer(TOKEN_A))
    )
    assert status.state is SessionState.READY

    closed = decode(
        await client.post("/v1/sessions/close", content=body, headers=bearer(TOKEN_A))
    )
    assert isinstance(closed[0], Ok)
    assert runtime.gateway.sessions.get(session_id).state is SessionState.CLOSED


async def test_batch_order_survives_the_ownership_filter(served) -> None:
    """When ownership validation blocks the middle session, the output order
    still matches the input order.

    Args:
        served: fixture.
    """
    _runtime, client = served
    mine: list[str] = []
    for index in range(2):
        created = decode(
            await client.post(
                "/v1/sessions",
                content=create_body(f"order-{index}"),
                headers=bearer(TOKEN_A),
            )
        )
        mine.append(created[0].value.session_id)
    theirs = decode(
        await client.post(
            "/v1/sessions",
            # Use a different env_config: the pool is reused based on
            # ``EnvSpecMsg.digest()`` and ``pool_size`` is part of that digest,
            # and teamA's two sessions have already filled the 2-slot pool.
            content=create_body(
                "theirs",
                env_spec=EnvSpecMsg(
                    env_family="fake",
                    env_config={"chunk_size": 2, "episode_length": 65},
                    pool_size=2,
                ),
            ),
            headers=bearer(TOKEN_B),
        )
    )[0].value.session_id

    ordered = [mine[0], theirs, mine[1]]
    results = decode(
        await client.post(
            "/v1/reset",
            content=wire.encode_bytes(
                {"session_ids": ordered, "reset_spec": ResetSpec(seed=1)}
            ),
            headers=bearer(TOKEN_A),
        )
    )
    assert len(results) == 3
    assert isinstance(results[0], Ok), results[0]
    assert isinstance(results[1], Err)
    assert results[1].error.code is ErrorCode.UNKNOWN_SESSION
    assert isinstance(results[2], Ok), results[2]
    assert results[0].value.session_id == mine[0]
    assert results[2].value.session_id == mine[1]


def test_build_app_needs_no_ray_or_rlinf() -> None:
    """The serve layer only touches the Runtime through ``RuntimeClient``
    (a runtime cross-check of the layering guarantee)."""
    import rollout_runtime.serve.app as module

    source = module.__file__
    assert source is not None
    text = open(source, encoding="utf-8").read()
    for forbidden in ("import ray", "import rlinf", "import torch"):
        assert forbidden not in text
    assert callable(build_app)
