"""Contract tests for ``RemoteRuntimeClient``: run against a **real ASGI app**.

Deliberately does not mock HTTP: the entire value of this client is "being
consistent with ``serve/app.py``'s wire contract", and mocking the transport
would defeat the very thing being verified. So ``httpx.ASGITransport`` is
used to wire in the real FastAPI app, going through real msgpack encoding/
decoding and real authentication and ownership checks.

The backend is ``local_fake`` (no GPU, no LIBERO), so these cases can run
locally.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest
import pytest_asyncio

from rollout_runtime.api.client import RuntimeClient
from rollout_runtime.api.enums import ErrorCode
from rollout_runtime.api.errors import RuntimeApiError
from rollout_runtime.api.ids import RequestId, SessionId
from rollout_runtime.api.messages import (
    CreateSessionRequest,
    EnvSpecMsg,
    PolicyRequest,
    ResetSpec,
)
from rollout_runtime.api.result import Err
from rollout_runtime.serve.app import ServeLimits
from rollout_runtime.serve.client import RemoteRuntime, RemoteRuntimeClient
from rollout_runtime.serve.server import ServeOptions, build_served_runtime

TOKEN = "tok-a"
OTHER_TOKEN = "tok-b"
TOKENS = {"RR_AUTH_TOKEN": f"teamA:{TOKEN},teamB:{OTHER_TOKEN}"}


def env_spec(pool_size: int = 1) -> EnvSpecMsg:
    """Construct an env spec for the fake family."""
    return EnvSpecMsg(
        env_family="fake",
        env_config={"chunk_size": 2, "episode_length": 64},
        pool_size=pool_size,
    )


@pytest_asyncio.fixture(loop_scope="function")
async def remote() -> AsyncIterator[Any]:
    """Start a served runtime, and yield its ``RemoteRuntimeClient``.

    Yields:
        ``(runtime, client, other_client)``.
    """
    runtime = await build_served_runtime(
        ServeOptions(
            config="local_fake",
            host="127.0.0.1",
            gateway_epoch=3,
            limits=ServeLimits(
                max_lease_seconds=120.0,
                max_pool_size=2,
                max_episode_steps=3,
                max_body_bytes=1 << 20,
            ),
        ),
        environ=dict(TOKENS),
    )
    transport = httpx.ASGITransport(app=runtime.app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://serve.test"
    ) as raw, httpx.AsyncClient(
        transport=httpx.ASGITransport(app=runtime.app), base_url="http://serve.test"
    ) as raw_other:
        client = RemoteRuntimeClient("http://serve.test", token=TOKEN, client=raw)
        other = RemoteRuntimeClient(
            "http://serve.test", token=OTHER_TOKEN, client=raw_other
        )
        yield runtime, client, other
    await runtime.aclose()


class TestProtocolConformance:
    def test_satisfies_the_runtime_client_protocol(self) -> None:
        """Structural matching: missing even one method should fail."""
        client = RemoteRuntimeClient("http://x", token="t")
        assert isinstance(client, RuntimeClient)

    def test_base_url_is_normalised(self) -> None:
        assert RemoteRuntimeClient("http://x:1/").base_url == "http://x:1"

    def test_environment_proxies_are_not_trusted(self) -> None:
        """Explicit runtime endpoints are isolated from ambient transport settings."""
        client = RemoteRuntimeClient("http://x:1")
        assert client._client.trust_env is False


@pytest.mark.asyncio(loop_scope="function")
class TestLifecycle:
    async def test_livez_and_start(self, remote: Any) -> None:
        _runtime, client, _other = remote
        assert (await client.livez())["status"] == "ok"
        await RemoteRuntime(client).start()

    async def test_create_reset_policy_step_close(self, remote: Any) -> None:
        _runtime, client, _other = remote
        created = await client.create_sessions(
            [
                CreateSessionRequest(
                    application_id="ignored-by-server",
                    client_session_key="remote-1",
                    env_spec=env_spec(),
                    lease_seconds=60.0,
                )
            ]
        )
        assert not isinstance(created[0], Err), created[0]
        handle = created[0].value
        # Identity comes from the token, not the request body.
        assert handle.application_id == "teamA"
        session_id = handle.session_id

        reset = await client.reset([session_id], ResetSpec(seed=1))
        assert not isinstance(reset[0], Err), reset[0]

        stepped = await client.policy_step([session_id], PolicyRequest())
        assert not isinstance(stepped[0], Err), stepped[0]

        status = await client.get_session(session_id)
        assert status.session_id == session_id

        closed = await client.close_sessions([session_id])
        assert not isinstance(closed[0], Err), closed[0]

    async def test_observe_and_renew(self, remote: Any) -> None:
        _runtime, client, _other = remote
        created = await client.create_sessions(
            [
                CreateSessionRequest(
                    application_id="",
                    client_session_key="remote-observe",
                    env_spec=env_spec(),
                    lease_seconds=60.0,
                )
            ]
        )
        session_id = created[0].value.session_id
        await client.reset([session_id], ResetSpec(seed=1))

        observed = await client.observe([session_id])
        assert not isinstance(observed[0], Err), observed[0]

        renewed = await client.renew_sessions([session_id], 90.0)
        assert not isinstance(renewed[0], Err), renewed[0]

    async def test_aclose_does_not_touch_the_shared_server(self, remote: Any) -> None:
        """The server must still be alive after the client closes — the
        shared runtime doesn't belong to any single agent."""
        runtime, client, other = remote
        await RemoteRuntime(client).aclose()
        assert (await other.livez())["status"] == "ok"
        assert runtime.gateway is not None

    async def test_remote_runtime_has_no_transport_attribute(self, remote: Any) -> None:
        """The server-side transport count is not accessible to the client;
        it should not pretend to have one."""
        _runtime, client, _other = remote
        assert not hasattr(RemoteRuntime(client), "transport")

    async def test_gateway_has_no_stop_so_teardown_skips_it(self, remote: Any) -> None:
        """``RuntimeSeam._teardown`` distinguishes the two forms via
        ``hasattr(gateway, "stop")``."""
        _runtime, client, _other = remote
        assert not hasattr(client, "stop")


@pytest.mark.asyncio(loop_scope="function")
class TestErrorSemantics:
    async def test_per_item_failure_is_an_err_not_an_exception(
        self, remote: Any
    ) -> None:
        """Per-item failures in a batch endpoint go through 200 + body
        ``Err``, and must not be collapsed into an exception."""
        _runtime, client, _other = remote
        results = await client.reset([SessionId("does-not-exist")], ResetSpec(seed=1))
        assert isinstance(results[0], Err)
        assert results[0].error.code is ErrorCode.UNKNOWN_SESSION

    async def test_another_tenant_cannot_touch_the_session(self, remote: Any) -> None:
        _runtime, client, other = remote
        created = await client.create_sessions(
            [
                CreateSessionRequest(
                    application_id="",
                    client_session_key="remote-owned",
                    env_spec=env_spec(),
                    lease_seconds=60.0,
                )
            ]
        )
        session_id = created[0].value.session_id
        stolen = await other.reset([session_id], ResetSpec(seed=1))
        assert isinstance(stolen[0], Err)
        # Does not leak existence.
        assert stolen[0].error.code is ErrorCode.UNKNOWN_SESSION

    async def test_whole_request_failure_raises(self, remote: Any) -> None:
        """Only whole-request failures (here, details for a nonexistent
        session) raise ``RuntimeApiError``."""
        _runtime, client, _other = remote
        with pytest.raises(RuntimeApiError) as excinfo:
            await client.get_session(SessionId("nope"))
        assert excinfo.value.info.code is ErrorCode.UNKNOWN_SESSION

    async def test_missing_token_is_rejected(self, remote: Any) -> None:
        runtime, _client, _other = remote
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=runtime.app),
            base_url="http://serve.test",
        ) as raw:
            anonymous = RemoteRuntimeClient("http://serve.test", client=raw)
            with pytest.raises(RuntimeApiError):
                await anonymous.get_session(SessionId("whatever"))

    async def test_unknown_request_id_raises(self, remote: Any) -> None:
        _runtime, client, _other = remote
        with pytest.raises(RuntimeApiError):
            await client.get_request_status(RequestId("no-such-request"))

    async def test_transport_failure_is_reported_as_internal(self) -> None:
        """When unable to connect, give a recognizable reason instead of a
        raw httpx exception."""
        client = RemoteRuntimeClient(
            "http://127.0.0.1:1", token="t", connect_timeout_s=0.2
        )
        try:
            with pytest.raises(RuntimeApiError) as excinfo:
                await client.livez()
            assert excinfo.value.info.detail.get("reason") == "client_transport"
        finally:
            await client.aclose()
