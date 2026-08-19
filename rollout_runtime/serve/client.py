"""HTTP implementation of ``RuntimeClient``: connects to an **already-running**
``rollout-runtime serve``.

This is the missing client-side half of the served mode. ``serve/app.py``
wraps the same ``RuntimeGateway`` with HTTP (D2), but before this module
existed there was no client-side implementation in the repository —
``RuntimeClient`` in ``api/client.py`` was only a ``Protocol``, and the only
implementation was the in-process ``RuntimeGateway``. As a result, zetta's
``--runtime rollout`` could only call ``build_local_components`` to build its
own private runtime, so N concurrent agents meant N runtimes and N sets of
weights — **cross-session batching and weight sharing were completely
unavailable**. This module fills in that missing half: N agents connect to
the **same** runtime, with 4 sets of weights serving N sessions.

Why this lives in ``serve/`` rather than ``api/``: only ``wire.py`` under
``api/**`` is allowed to be imported by third parties
(``test_layering.py::API_THIRD_PARTY_ALLOWLIST``), but the HTTP client needs
``httpx``. ``serve/`` follows the same rules as ``gateway/`` (rule 3b: no
touching ray / rlinf / torch), and httpx does not violate that.

## Contract with the server (mirrors ``serve/app.py`` point by point)

- The body is always ``api.wire`` msgpack, ``Content-Type: application/msgpack``;
- Identity relies on ``Authorization: Bearer <token>``; any ``application_id``
  in the request body has **zero effect**;
- **Per-item failures in batch endpoints go through 200 + an ``Err`` in the
  body**, not an HTTP status code; only "the whole request failed" is
  non-200, in which case ``RuntimeApiError`` is raised. This is protocol
  semantics (partial success within a batch, §2.3); the client must not
  flatten it into an exception.

## Three deliberate design choices

1. **Never shut down the server.** The remote runtime is shared; the client
   only has ``aclose()`` (closing its own HTTP connections) and
   ``close_sessions()`` (closing its own sessions). No method calls into the
   server's gateway to shut it down — one agent exiting must not take down
   other tenants.
2. **Isolate environment transport settings** (``trust_env=False``). The
   runtime address is provided explicitly by the caller; the client should
   not be implicitly rewritten by HTTP transport configuration in the
   process environment.
3. **Tiered timeouts.** ``create_sessions`` has to build a LIBERO subprocess
   on the server side (tens of seconds to minutes measured in real
   environments), ``policy_step`` is a second-scale GPU forward pass, and
   ``get_session`` is millisecond-scale. A single uniform short timeout
   would break session creation, while a single uniform long timeout would
   make failures hang for a long time.
"""

from __future__ import annotations

import contextlib
from collections.abc import Sequence
from typing import Any

import httpx

from rollout_runtime.api import wire
from rollout_runtime.api.enums import ErrorCode
from rollout_runtime.api.errors import RuntimeApiError, make_error
from rollout_runtime.api.ids import RequestId, SessionId
from rollout_runtime.api.messages import (
    CancelOutcome,
    CreateSessionRequest,
    EpisodeRequest,
    EpisodeResult,
    Observation,
    OperationStatus,
    PolicyInferResult,
    PolicyRequest,
    ResetSpec,
    SessionHandle,
    SessionStatus,
    StepResult,
)
from rollout_runtime.api.payload_ref import PayloadRef
from rollout_runtime.api.result import Result
from rollout_runtime.serve.app import MSGPACK_CONTENT_TYPE

__all__ = [
    "DEFAULT_CONNECT_TIMEOUT_S",
    "DEFAULT_OPERATION_TIMEOUT_S",
    "DEFAULT_SESSION_TIMEOUT_S",
    "RemoteRuntime",
    "RemoteRuntimeClient",
]

DEFAULT_CONNECT_TIMEOUT_S = 10.0
"""Upper bound for establishing a TCP connection."""

DEFAULT_OPERATION_TIMEOUT_S = 900.0
"""Read timeout for the operation plane (``reset`` / ``policy_step`` / ``action_step`` ...).

Same order of magnitude as ``transport.command_timeout_seconds=600`` in
``zetta_libero_pi05.yaml``, with margin: a server-side timeout only returns
``DEADLINE_EXCEEDED`` (the operation itself is not cancelled per §3.1.2);
the client disconnecting earlier than that would just lose that error code.
"""

DEFAULT_SESSION_TIMEOUT_S = 1800.0
"""Read timeout for ``create_sessions``: the server must build the env pool (real LIBERO's reconfigure is slow)."""


class RemoteRuntimeClient:
    """HTTP implementation of ``RuntimeClient`` (structurally matched via ``@runtime_checkable``).

    Attributes:
        base_url: The service address, with any trailing slash removed.
    """

    def __init__(
        self,
        base_url: str,
        *,
        token: str | None = None,
        connect_timeout_s: float = DEFAULT_CONNECT_TIMEOUT_S,
        operation_timeout_s: float = DEFAULT_OPERATION_TIMEOUT_S,
        session_timeout_s: float = DEFAULT_SESSION_TIMEOUT_S,
        max_connections: int = 128,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        """Initialize the client (does not send any request).

        Args:
            base_url: ``http://host:port``.
            token: The bearer token; ``None`` means no ``Authorization``
                header is sent (only viable for loopback connections where
                the server has not set ``RR_AUTH_TOKEN``, see
                ``serve/auth.py``).
            connect_timeout_s: The connect timeout.
            operation_timeout_s: The operation-plane read timeout.
            session_timeout_s: The ``create_sessions`` read timeout.
            max_connections: The connection pool limit; N concurrent sessions
                will issue concurrent requests, and too small a limit will
                make them queue up against each other.
            client: Inject a custom ``AsyncClient`` (used in tests with
                ``ASGITransport``). If supplied, the caller is responsible
                for closing it; ``aclose()`` will not touch it.
        """
        self.base_url = base_url.rstrip("/")
        self._token = token
        self._operation_timeout = float(operation_timeout_s)
        self._session_timeout = float(session_timeout_s)
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(
            base_url=self.base_url,
            # See module design point 2: use only the explicitly supplied endpoint.
            trust_env=False,
            timeout=httpx.Timeout(
                connect=float(connect_timeout_s),
                read=float(operation_timeout_s),
                write=float(operation_timeout_s),
                pool=float(operation_timeout_s),
            ),
            limits=httpx.Limits(
                max_connections=int(max_connections),
                max_keepalive_connections=int(max_connections),
            ),
        )

    # ------------------------------------------------------------------ transport

    def _headers(self, *, with_body: bool) -> dict[str, str]:
        """Construct request headers.

        Args:
            with_body: Whether an msgpack body is attached.

        Returns:
            The header dictionary.
        """
        headers: dict[str, str] = {"Accept": MSGPACK_CONTENT_TYPE}
        if with_body:
            headers["Content-Type"] = MSGPACK_CONTENT_TYPE
        if self._token is not None:
            headers["Authorization"] = f"Bearer {self._token}"
        return headers

    async def _request(
        self,
        method: str,
        path: str,
        *,
        body: Any = None,
        timeout: float | None = None,
        hint: Any = None,
    ) -> Any:
        """Send one request and decode the body.

        Args:
            method: The HTTP method.
            path: The path (relative to ``base_url``).
            body: The request body to msgpack-encode; ``None`` means no body.
            timeout: Override the read timeout.
            hint: The decoding type annotation; ``None`` uses dynamic
                decoding driven by the ``"@"`` tag.

        Returns:
            The decoded object.

        Raises:
            RuntimeApiError: Non-200 (the whole request failed), or a
                transport-layer exception. Per-item failures are **not**
                raised here — they are ``Err`` values inside a 200 body.
        """
        data = wire.encode_bytes(body) if body is not None else None
        try:
            response = await self._client.request(
                method,
                path,
                content=data,
                headers=self._headers(with_body=data is not None),
                timeout=(
                    httpx.Timeout(
                        connect=DEFAULT_CONNECT_TIMEOUT_S,
                        read=float(timeout),
                        write=float(timeout),
                        pool=float(timeout),
                    )
                    if timeout is not None
                    else None
                ),
            )
        except httpx.TimeoutException as exc:
            raise RuntimeApiError(
                make_error(
                    ErrorCode.DEADLINE_EXCEEDED,
                    f"{method} {path} timed out: {exc}",
                    reason="client_timeout",
                )
            ) from exc
        except httpx.HTTPError as exc:
            # ErrorCode has no UNAVAILABLE; "cannot reach the shared service"
            # is closest to INTERNAL in the protocol, so `reason` is used to
            # distinguish the transport layer, letting callers decide
            # whether to retry based on `reason` rather than `code`.
            raise RuntimeApiError(
                make_error(
                    ErrorCode.INTERNAL,
                    f"{method} {path} failed: {type(exc).__name__}: {exc}",
                    reason="client_transport",
                )
            ) from exc
        if response.status_code != 200:
            raise self._error_from(response)
        return wire.decode_bytes(response.content, hint)

    def _error_from(self, response: httpx.Response) -> RuntimeApiError:
        """Reconstruct a ``RuntimeApiError`` from a non-200 response.

        On a total request failure, the server returns msgpack-encoded
        ``RuntimeErrorInfo`` (via the ``normalize_errors`` middleware in
        ``serve/app.py``), but it may also be FastAPI's own JSON (e.g. a
        nonexistent route), so a decoding failure degrades to a plain
        ``INTERNAL`` error carrying the status code, instead of raising
        another decoding exception.

        Args:
            response: The non-200 response.

        Returns:
            The corresponding ``RuntimeApiError``.
        """
        with contextlib.suppress(Exception):
            info = wire.decode_bytes(response.content)
            if hasattr(info, "code") and hasattr(info, "message"):
                return RuntimeApiError(info)
        detail = response.text[:400] if response.content else ""
        return RuntimeApiError(
            make_error(
                ErrorCode.INTERNAL,
                f"HTTP {response.status_code} from {response.request.url.path}: {detail}",
                reason="client_http_status",
            )
        )

    async def aclose(self) -> None:
        """Close this client's HTTP connection pool.

        This **does not** affect the remote runtime — it is shared. Sessions
        must be closed individually via ``close_sessions``. If an external
        ``AsyncClient`` was injected, it is not closed (ownership stays with
        the caller).
        """
        if self._owns_client:
            await self._client.aclose()

    async def livez(self) -> dict[str, str]:
        """Liveness probe (the only endpoint that requires no authentication).

        Returns:
            ``{"status": "ok"}``.

        Raises:
            RuntimeApiError: The service is unreachable.
        """
        try:
            response = await self._client.get(
                "/livez", timeout=httpx.Timeout(DEFAULT_CONNECT_TIMEOUT_S)
            )
        except httpx.HTTPError as exc:
            raise RuntimeApiError(
                make_error(
                    ErrorCode.INTERNAL,
                    f"/livez unreachable at {self.base_url}: {type(exc).__name__}: {exc}",
                    reason="client_transport",
                )
            ) from exc
        if response.status_code != 200:
            raise self._error_from(response)
        return dict(response.json())

    # ------------------------------------------------------------------ control plane

    async def create_sessions(
        self, requests: Sequence[CreateSessionRequest]
    ) -> list[Result[SessionHandle]]:
        """Batch-create sessions.

        Args:
            requests: The create requests; ``application_id`` is determined
                by the token, and any value in the body has zero effect.

        Returns:
            Per-item results in the same order as the input.
        """
        return list(
            await self._request(
                "POST",
                "/v1/sessions",
                body={"requests": list(requests)},
                timeout=self._session_timeout,
            )
        )

    async def get_session(self, session_id: SessionId) -> SessionStatus:
        """Query a session's status.

        Args:
            session_id: The target session.

        Returns:
            The ``SessionStatus``.
        """
        return await self._request("GET", f"/v1/sessions/{session_id}")

    async def renew_sessions(
        self, session_ids: Sequence[SessionId], lease_seconds: float
    ) -> list[Result[SessionStatus]]:
        """Batch-renew leases.

        Args:
            session_ids: The target sessions.
            lease_seconds: The new lease duration (clamped by the server).

        Returns:
            Per-item results in the same order as the input.
        """
        return list(
            await self._request(
                "POST",
                "/v1/sessions/renew",
                body={
                    "session_ids": list(session_ids),
                    "lease_seconds": float(lease_seconds),
                },
            )
        )

    async def close_sessions(
        self, session_ids: Sequence[SessionId]
    ) -> list[Result[None]]:
        """Batch-close **your own** sessions.

        Args:
            session_ids: The target sessions.

        Returns:
            Per-item results in the same order as the input.
        """
        return list(
            await self._request(
                "POST",
                "/v1/sessions/close",
                body={"session_ids": list(session_ids)},
            )
        )

    # ------------------------------------------------------------------ operation plane

    async def reset(
        self,
        session_ids: Sequence[SessionId],
        reset_spec: ResetSpec,
        *,
        request_ids: Sequence[RequestId] | None = None,
    ) -> list[Result[StepResult]]:
        """Batch-reset episodes.

        Args:
            session_ids: The target sessions.
            reset_spec: Episode initialization parameters.
            request_ids: Idempotency identifiers.

        Returns:
            Per-item results in the same order as the input.
        """
        body: dict[str, Any] = {
            "session_ids": list(session_ids),
            "reset_spec": reset_spec,
        }
        if request_ids is not None:
            body["request_ids"] = list(request_ids)
        return list(
            await self._request(
                "POST", "/v1/reset", body=body, timeout=self._session_timeout
            )
        )

    async def observe(
        self,
        session_ids: Sequence[SessionId],
        *,
        consistency: str = "linearizable",
    ) -> list[Result[Observation]]:
        """Batch-read the current observation, without changing environment state.

        Args:
            session_ids: The target sessions.
            consistency: ``"linearizable"`` or ``"eventual"``.

        Returns:
            Per-item results in the same order as the input.
        """
        return list(
            await self._request(
                "POST",
                "/v1/observe",
                body={
                    "session_ids": list(session_ids),
                    "consistency": str(consistency),
                },
            )
        )

    async def action_step(
        self,
        session_ids: Sequence[SessionId],
        actions: Sequence[PayloadRef],
        *,
        request_ids: Sequence[RequestId] | None = None,
    ) -> list[Result[StepResult]]:
        """Batch-execute externally supplied action chunks.

        Args:
            session_ids: The target sessions.
            actions: A ``[chunk, action_dim] float32`` payload per session.
            request_ids: Idempotency identifiers.

        Returns:
            Per-item results in the same order as the input.
        """
        body: dict[str, Any] = {
            "session_ids": list(session_ids),
            "actions": list(actions),
        }
        if request_ids is not None:
            body["request_ids"] = list(request_ids)
        return list(await self._request("POST", "/v1/action_step", body=body))

    async def policy_step(
        self,
        session_ids: Sequence[SessionId],
        policy_request: PolicyRequest,
        *,
        request_ids: Sequence[RequestId] | None = None,
    ) -> list[Result[StepResult]]:
        """Batch-execute the atomic "observe -> infer -> chunk_step" sequence.

        Args:
            session_ids: The target sessions.
            policy_request: Inference parameters.
            request_ids: Idempotency identifiers.

        Returns:
            Per-item results in the same order as the input.
        """
        body: dict[str, Any] = {
            "session_ids": list(session_ids),
            "policy_request": policy_request,
        }
        if request_ids is not None:
            body["request_ids"] = list(request_ids)
        return list(await self._request("POST", "/v1/policy_step", body=body))

    async def policy_infer(
        self,
        session_ids: Sequence[SessionId],
        policy_request: PolicyRequest,
        *,
        request_ids: Sequence[RequestId] | None = None,
    ) -> list[Result[PolicyInferResult]]:
        """Batch-execute "observe -> infer" without executing ``chunk_step``.

        Args:
            session_ids: The target sessions.
            policy_request: Inference parameters.
            request_ids: Idempotency identifiers (not consumed by this
                endpoint on the server side; kept to satisfy the protocol
                signature).

        Returns:
            Per-item results in the same order as the input.
        """
        body: dict[str, Any] = {
            "session_ids": list(session_ids),
            "policy_request": policy_request,
        }
        if request_ids is not None:
            body["request_ids"] = list(request_ids)
        return list(await self._request("POST", "/v1/policy_infer", body=body))

    async def run_episode(
        self,
        session_ids: Sequence[SessionId],
        episode_request: EpisodeRequest,
        *,
        request_ids: Sequence[RequestId] | None = None,
    ) -> list[Result[EpisodeResult]]:
        """Batch-execute a full episode (the server clamps ``max_steps``).

        Args:
            session_ids: The target sessions.
            episode_request: Episode parameters.
            request_ids: Idempotency identifiers.

        Returns:
            Per-item results in the same order as the input.
        """
        body: dict[str, Any] = {
            "session_ids": list(session_ids),
            "episode_request": episode_request,
        }
        if request_ids is not None:
            body["request_ids"] = list(request_ids)
        return list(await self._request("POST", "/v1/run_episode", body=body))

    async def extension_call(
        self,
        session_ids: Sequence[SessionId],
        namespace: str,
        method: str,
        args: dict[str, Any],
        *,
        request_ids: Sequence[RequestId] | None = None,
    ) -> list[Result[dict[str, Any]]]:
        """Invoke a family-specific extension.

        Args:
            session_ids: The target sessions.
            namespace: The extension namespace (e.g. ``"libero"``).
            method: The extension method name.
            args: The method arguments.
            request_ids: Idempotency identifiers (not consumed by this
                endpoint on the server side; kept to satisfy the protocol
                signature).

        Returns:
            Per-item results in the same order as the input.
        """
        body: dict[str, Any] = {
            "session_ids": list(session_ids),
            "namespace": str(namespace),
            "method": str(method),
            "args": dict(args),
        }
        if request_ids is not None:
            body["request_ids"] = list(request_ids)
        return list(await self._request("POST", "/v1/extension_call", body=body))

    # ------------------------------------------------------------------ status and cancellation

    async def get_request_status(self, request_id: RequestId) -> OperationStatus:
        """Query an operation's status.

        Args:
            request_id: The request identifier.

        Returns:
            The ``OperationStatus``.
        """
        return await self._request("GET", f"/v1/requests/{request_id}")

    async def cancel_request(self, request_id: RequestId) -> CancelOutcome:
        """Best-effort cancellation of an operation.

        Args:
            request_id: The request identifier.

        Returns:
            The ``CancelOutcome``.
        """
        return await self._request("POST", f"/v1/requests/{request_id}/cancel")


class RemoteRuntime:
    """Wraps a ``RemoteRuntimeClient`` into the runtime shape expected by ``RuntimeSeam``.

    ``RuntimeSeam._serve()`` uses the trio ``start()`` / ``gateway`` /
    ``aclose()`` for local runtimes (and also calls ``gateway.stop()``). In
    the remote form:

    - ``start()`` only performs a single ``/livez`` probe, and builds
      **no** worker;
    - ``gateway`` is the HTTP client itself;
    - ``aclose()`` only closes the connection pool;
    - there is **no** ``transport`` attribute, so the ray_channel byte
      counters in ``snapshot_extra_metrics`` are naturally absent (those
      counters live in the server process; the client cannot obtain them,
      and should not pretend to).

    Attributes:
        gateway: The ``RemoteRuntimeClient`` instance.
    """

    def __init__(self, client: RemoteRuntimeClient):
        """Initialize.

        Args:
            client: An already-constructed HTTP client.
        """
        self.gateway = client

    async def start(self) -> None:
        """Probe the remote service for liveness.

        Raises:
            RuntimeApiError: The service is unreachable.
        """
        await self.gateway.livez()

    async def aclose(self) -> None:
        """Close the client's connection pool (does not affect the remote runtime)."""
        await self.gateway.aclose()
