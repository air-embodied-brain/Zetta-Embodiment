"""HTTP endpoints for the served mode.

Design decision D2: embedded and served **share the same ``RuntimeGateway``
class**. This layer does exactly four things and never steps into the data
plane:

1. **Authentication and identity**: ``Authorization: Bearer`` -> ``application_id``;
   any ``application_id`` in the request body has **zero effect**;
2. **Ownership checks**: reply ``UNKNOWN_SESSION`` when a session / request
   does not belong to the caller (never leaking existence);
3. **Convergence of untrusted input**: clamping ``lease_seconds`` /
   ``pool_size`` / ``max_steps`` / body size;
4. **Encoding/decoding**: request and response bodies are always ``api.wire``
   msgpack (the finalized serialization format), so there is no second
   schema here, and no "HTTP layer drifting from the protocol."

``/metrics`` exposes ``gateway.metrics.registry``; ``/livez`` is the only
endpoint that requires no authentication (it only returns
``{"status": "ok"}``, with no topology information).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from typing import Any

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse

from rollout_runtime.api import codec, wire
from rollout_runtime.api.enums import ErrorCode
from rollout_runtime.api.errors import (
    RuntimeApiError,
    RuntimeErrorInfo,
    make_error,
    normalize_exception,
)
from rollout_runtime.api.ids import RequestId, SessionId
from rollout_runtime.api.messages import (
    CreateSessionRequest,
    EpisodeRequest,
    PolicyRequest,
    ResetSpec,
)
from rollout_runtime.api.payload_ref import PayloadRef
from rollout_runtime.api.result import Err, err
from rollout_runtime.gateway.gateway import RuntimeGateway
from rollout_runtime.serve.auth import TokenAuthority

__all__ = ["MSGPACK_CONTENT_TYPE", "ServeLimits", "build_app", "http_status_for"]

MSGPACK_CONTENT_TYPE = "application/msgpack"
"""Body content type for the Runtime API over HTTP (finalized: v1 serialization is msgpack)."""

_STATUS_BY_CODE = {
    ErrorCode.UNKNOWN_SESSION: 404,
    ErrorCode.INVALID_ARGUMENT: 400,
    ErrorCode.UNSUPPORTED_EXTENSION: 400,
    ErrorCode.UNSUPPORTED_ENV_SPEC: 400,
    ErrorCode.SESSION_NOT_READY: 409,
    ErrorCode.EPISODE_TERMINATED: 409,
    ErrorCode.STALE_BINDING: 409,
    ErrorCode.IDEMPOTENCY_CONFLICT: 409,
    ErrorCode.CANCELLED: 409,
    ErrorCode.QUEUE_FULL: 429,
    ErrorCode.QUOTA_EXCEEDED: 429,
    ErrorCode.DEADLINE_EXCEEDED: 504,
    ErrorCode.WORKER_LOST: 503,
}
"""**Projection** from ``ErrorCode`` to HTTP status code (not a second error surface).

Only used for "the entire request failed" cases. Per-item failures in batch
endpoints always go through 200 + an ``Err`` in the body, because "partial
success within a batch, no cross-session transactions" is the protocol
semantics and must not be flattened by an HTTP status code.
"""


_UNAUTHENTICATED_PATHS = frozenset({"/livez"})
"""The only path that does not require ``Authorization`` (it only returns
``{"status": "ok"}``, with no topology information).

Authentication happens in middleware (see ``normalize_errors`` in
``build_app``), so this set is the **single** declaration of "who may skip
authentication" — adding an endpoint can never accidentally skip it.
"""


def _too_large(limit: int) -> RuntimeApiError:
    """Construct a "body too large" error.

    Args:
        limit: The byte limit.

    Returns:
        A ``RuntimeApiError`` with ``reason="body_too_large"``.
    """
    return RuntimeApiError(
        make_error(
            ErrorCode.INVALID_ARGUMENT,
            f"request body exceeds {limit} bytes",
            reason="body_too_large",
        )
    )


def http_status_for(info: RuntimeErrorInfo) -> int:
    """Project a normalized error onto an HTTP status code.

    Args:
        info: The normalized error.

    Returns:
        The HTTP status code; authentication failures
        (``detail.reason == "authentication"``) are 401, others follow
        ``_STATUS_BY_CODE``, and anything not listed is 500.
    """
    if info.detail.get("reason") == "authentication":
        return 401
    return _STATUS_BY_CODE.get(info.code, 500)


@dataclasses.dataclass(frozen=True, kw_only=True)
class ServeLimits:
    """Clamps applied to untrusted input in the served mode.

    Attributes:
        max_lease_seconds: Upper bound on ``lease_seconds``, preventing a
            long lease from pinning down an env slot.
        max_pool_size: Upper bound on ``env_spec.pool_size``, preventing a
            single create call from spinning up hundreds of simulator
            processes.
        max_episode_steps: Upper bound on ``EpisodeRequest.max_steps``,
            preventing a long loop inside a worker from holding a slot.
        max_body_bytes: Upper bound on the HTTP body. The 8 MiB in §2.5 is
            the **per-request payload** budget, not a body size limit, and
            uvicorn does not limit the body by default, so this is a
            separate guard.
        max_batch_sessions: Upper bound on the number of sessions allowed in
            a single batch call.
        allowed_env_families: Allowed ``env_family`` values; empty means no
            restriction.
        include_traceback: Whether to also return ``detail.traceback`` to the
            caller (the default is **not** to, since ``normalize_exception``
            would write internal paths into it).
    """

    max_lease_seconds: float = 3600.0
    max_pool_size: int = 8
    max_episode_steps: int = 1000
    max_body_bytes: int = 16 * 1024 * 1024
    max_batch_sessions: int = 256
    allowed_env_families: frozenset[str] = frozenset()
    include_traceback: bool = False


def _sanitize(info: RuntimeErrorInfo, *, include_traceback: bool) -> RuntimeErrorInfo:
    """Outbound scrubbing: strips ``detail.traceback`` by default.

    Args:
        info: The normalized error.
        include_traceback: If true, return unchanged.

    Returns:
        The scrubbed error.
    """
    if include_traceback or "traceback" not in info.detail:
        return info
    detail = {key: value for key, value in info.detail.items() if key != "traceback"}
    return dataclasses.replace(info, detail=detail)


def _sanitize_results(results: Sequence[Any], *, include_traceback: bool) -> list[Any]:
    """Scrub each item of a batch result.

    Args:
        results: ``list[Result[T]]``.
        include_traceback: If true, return unchanged.

    Returns:
        The scrubbed list.
    """
    return [
        err(_sanitize(item.error, include_traceback=include_traceback))
        if isinstance(item, Err)
        else item
        for item in results
    ]


class _Body:
    """A decoded msgpack request body, with fields retrieved via type annotations."""

    def __init__(self, data: Any) -> None:
        """Initialize.

        Args:
            data: The raw structure decoded from msgpack.

        Raises:
            RuntimeApiError: If the top level is not a map.
        """
        if not isinstance(data, dict):
            raise RuntimeApiError(
                make_error(
                    ErrorCode.INVALID_ARGUMENT,
                    "request body must be a msgpack map",
                )
            )
        self._data = data

    def get(self, key: str, hint: Any = None, default: Any = None) -> Any:
        """Decode a field according to its annotation.

        Args:
            key: The field name.
            hint: The target type annotation.
            default: The default value.

        Returns:
            The decoded value.

        Raises:
            RuntimeApiError: The field structure does not match the annotation.
        """
        if key not in self._data or self._data[key] is None:
            return default
        try:
            return codec.decode(self._data[key], hint)
        except RuntimeApiError:
            raise
        except BaseException as exc:  # noqa: BLE001 - network input, normalize into an argument error
            raise RuntimeApiError(
                make_error(
                    ErrorCode.INVALID_ARGUMENT,
                    f"cannot decode field {key!r}: {exc}",
                    field=key,
                )
            ) from exc

    def require(self, key: str, hint: Any = None) -> Any:
        """Retrieve a required field.

        Args:
            key: The field name.
            hint: The target type annotation.

        Returns:
            The decoded value.

        Raises:
            RuntimeApiError: The field is missing.
        """
        value = self.get(key, hint)
        if value is None:
            raise RuntimeApiError(
                make_error(
                    ErrorCode.INVALID_ARGUMENT,
                    f"missing required field: {key!r}",
                    field=key,
                )
            )
        return value


def build_app(
    gateway: RuntimeGateway,
    *,
    authority: TokenAuthority,
    limits: ServeLimits | None = None,
) -> Any:
    """Construct the FastAPI application.

    Args:
        gateway: An already-started Gateway (embedded and served share the
            same class, D2).
        authority: The authenticator.
        limits: Clamps on untrusted input; ``None`` uses the defaults.

    Returns:
        The FastAPI instance.
    """
    bounds = limits or ServeLimits()
    app = FastAPI(
        title="Rollout Runtime",
        version="v1",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    app.state.gateway = gateway
    app.state.authority = authority
    app.state.limits = bounds

    # ---------------------------------------------------------------- infrastructure

    def pack(value: Any, status_code: int = 200) -> Response:
        return Response(
            content=wire.encode_bytes(value),
            media_type=MSGPACK_CONTENT_TYPE,
            status_code=status_code,
        )

    def fail(info: RuntimeErrorInfo) -> Response:
        clean = _sanitize(info, include_traceback=bounds.include_traceback)
        status = http_status_for(clean)
        headers = (
            {"WWW-Authenticate": "Bearer"}
            if clean.detail.get("reason") == "authentication"
            else None
        )
        return Response(
            content=wire.encode_bytes(clean),
            media_type=MSGPACK_CONTENT_TYPE,
            status_code=status,
            headers=headers,
        )

    async def read_body(request: Request) -> _Body:
        declared = request.headers.get("content-length")
        if declared is not None and declared.isdigit():
            if int(declared) > bounds.max_body_bytes:
                raise _too_large(bounds.max_body_bytes)
        # Read the body **streamed**, stopping immediately once the limit is
        # exceeded. ``await request.body()`` would buffer the entire stream
        # into memory before measuring its length, so a chunked request (no
        # Content-Length) could let the driver consume unbounded memory —
        # an independent audit measured RSS growing by 868 MB for a 419 MB
        # chunked body before it was rejected with 400, even though the
        # declared limit was 16 MiB. The Gateway is the asyncio single writer
        # for the data plane (D2), so OOM-killing it would take down every
        # tenant.
        chunks: list[bytes] = []
        total = 0
        async for chunk in request.stream():
            if not chunk:
                continue
            total += len(chunk)
            if total > bounds.max_body_bytes:
                raise _too_large(bounds.max_body_bytes)
            chunks.append(chunk)
        raw = b"".join(chunks)
        if not raw:
            return _Body({})
        try:
            # ``wire.decode_bytes`` performs **dynamic** decoding driven by
            # the ``"@"`` tag; as long as the client packs protocol objects,
            # what we get back here is already a dataclass instance. Running
            # it through ``codec.encode`` once flattens it back into a raw
            # structure, and everything downstream decodes uniformly against
            # the **target type annotation**: this way both "the client sent
            # a tagged object" and "the client hand-rolled an untagged map"
            # go through the same code path without two separate branches.
            return _Body(codec.encode(wire.decode_bytes(raw)))
        except RuntimeApiError:
            raise
        except BaseException as exc:  # noqa: BLE001 - network input
            raise RuntimeApiError(
                make_error(
                    ErrorCode.INVALID_ARGUMENT,
                    f"request body is not valid msgpack: {exc}",
                )
            ) from exc

    def identify(request: Request) -> str:
        """Retrieve the caller's identity (the **sole** source of ``application_id``).

        The identity is already resolved and stored in ``request.state`` by
        the middleware (**authenticate first, read the body second**); this
        just retrieves it. When called directly, it degrades to resolving it
        on the spot.

        Args:
            request: The HTTP request.

        Returns:
            The ``application_id``.
        """
        cached = getattr(request.state, "application_id", None)
        if isinstance(cached, str):
            return cached
        return authority.resolve(request.headers.get("authorization"))

    def owned_sessions(
        application_id: str, session_ids: Sequence[SessionId]
    ) -> dict[int, RuntimeErrorInfo]:
        """Pick out the indices that do not belong to the caller.

        Args:
            application_id: The caller's identity.
            session_ids: The sessions in the request.

        Returns:
            A mapping from index to ``UNKNOWN_SESSION`` error; empty means
            every session's ownership checks out.
        """
        rejected: dict[int, RuntimeErrorInfo] = {}
        for index, session_id in enumerate(session_ids):
            record = gateway.sessions.find(session_id)
            if record is None or record.application_id != application_id:
                # Deliberately using UNKNOWN_SESSION: someone else's session
                # and a nonexistent session must be indistinguishable to the
                # caller, otherwise this endpoint becomes a session id
                # existence oracle.
                rejected[index] = make_error(
                    ErrorCode.UNKNOWN_SESSION,
                    f"unknown session: {session_id}",
                    session_id=str(session_id),
                    reason="ownership",
                )
        return rejected

    def check_batch(session_ids: Sequence[SessionId]) -> None:
        if len(session_ids) > bounds.max_batch_sessions:
            raise RuntimeApiError(
                make_error(
                    ErrorCode.INVALID_ARGUMENT,
                    f"batch of {len(session_ids)} sessions exceeds the server limit "
                    f"({bounds.max_batch_sessions})",
                )
            )

    async def batch_call(
        request: Request,
        session_ids: Sequence[SessionId],
        call: Any,
    ) -> Response:
        """Ownership check + call + per-item scrubbing (preserving input order).

        Args:
            request: The HTTP request.
            session_ids: The target sessions.
            call: ``async (owned_ids, owned_indices) -> list[Result]``;
                ``owned_indices`` lets the endpoint slice its own per-session
                arguments (``actions`` / ``request_ids``) by index, so it
                does **not depend on session_id uniqueness**.

        Returns:
            The msgpack response.
        """
        application_id = identify(request)
        check_batch(session_ids)
        rejected = owned_sessions(application_id, session_ids)
        allowed_indices = [
            index for index in range(len(session_ids)) if index not in rejected
        ]
        allowed = [session_ids[index] for index in allowed_indices]
        produced = list(await call(allowed, allowed_indices)) if allowed else []
        if len(produced) != len(allowed):
            raise RuntimeApiError(
                make_error(
                    ErrorCode.INTERNAL,
                    f"gateway returned {len(produced)} results for {len(allowed)} "
                    "sessions; batch order cannot be reconstructed",
                )
            )
        merged: list[Any] = []
        cursor = 0
        for index in range(len(session_ids)):
            if index in rejected:
                merged.append(err(rejected[index]))
            else:
                merged.append(produced[cursor])
                cursor += 1
        return pack(
            _sanitize_results(merged, include_traceback=bounds.include_traceback)
        )

    @app.middleware("http")
    async def normalize_errors(request: Request, call_next: Any) -> Response:
        """**Authenticate first**, then normalize any leaked exception into a protocol error.

        Placing authentication in middleware is deliberate: it guarantees
        "resolve identity" happens before **any** body read or msgpack
        decoding, and it automatically applies to any endpoint added in the
        future. An independent audit confirmed the original implementation's
        fail-open ordering — `identify()` used to live inside `batch_call`,
        i.e. after `read_body()`, so an **unauthenticated** caller could
        still reach the msgpack decoder and obtain a "is the body valid"
        oracle (`POST /v1/reset` with no `Authorization` and a garbage body
        returned 400 instead of 401).

        Args:
            request: The HTTP request.
            call_next: The downstream handler.

        Returns:
            The HTTP response.
        """
        try:
            if request.url.path not in _UNAUTHENTICATED_PATHS:
                request.state.application_id = authority.resolve(
                    request.headers.get("authorization")
                )
            return await call_next(request)
        except RuntimeApiError as exc:
            return fail(exc.info)
        except BaseException as exc:  # noqa: BLE001 - don't let fastapi return a 500 HTML page
            return fail(normalize_exception(exc))

    # ---------------------------------------------------------------- observability endpoints

    @app.get("/livez")
    async def livez() -> dict[str, str]:
        """Unauthenticated liveness probe (returns **only** ok, no topology information).

        Returns:
            ``{"status": "ok"}``.
        """
        return {"status": "ok"}

    @app.get("/healthz")
    async def healthz(request: Request) -> Response:
        """Authenticated health information (epoch and rank health distribution).

        Args:
            request: The HTTP request.

        Returns:
            A JSON response.
        """
        identify(request)
        gateway.refresh_metrics()
        healthy = sum(1 for entry in gateway.workers if entry.healthy)
        return JSONResponse(
            {
                "status": "ok",
                "gateway_epoch": gateway.gateway_epoch,
                "auth": "bearer" if authority.enabled else "disabled",
                "env_ranks": len(gateway.workers),
                "env_ranks_healthy": healthy,
                "heartbeat_ok": gateway.heartbeat_ok_count,
                "heartbeat_failed": gateway.heartbeat_failure_count,
            }
        )

    @app.get("/metrics")
    async def metrics(request: Request) -> Response:
        """Prometheus scrape endpoint (visible after authentication: metrics include per-tenant and per-session counts).

        Args:
            request: The HTTP request.

        Returns:
            Prometheus text; 503 if ``prometheus_client`` is missing.
        """
        identify(request)
        gateway.refresh_metrics()
        rendered = gateway.metrics.render()
        if rendered is None:
            return Response(
                content=b"prometheus_client is not installed; install zetta[runtime]\n",
                media_type="text/plain; charset=utf-8",
                status_code=503,
            )
        body, content_type = rendered
        return Response(content=body, media_type=content_type)

    # ---------------------------------------------------------------- session plane

    @app.post("/v1/sessions")
    async def create_sessions(request: Request) -> Response:
        """Batch-create sessions; ``application_id`` is determined by the token.

        Args:
            request: The HTTP request.

        Returns:
            msgpack-encoded ``list[Result[SessionHandle]]``.
        """
        application_id = identify(request)
        body = await read_body(request)
        requests = body.require("requests", list[CreateSessionRequest])
        if len(requests) > bounds.max_batch_sessions:
            raise RuntimeApiError(
                make_error(
                    ErrorCode.INVALID_ARGUMENT,
                    f"batch of {len(requests)} requests exceeds the server limit "
                    f"({bounds.max_batch_sessions})",
                )
            )
        prepared: list[Any] = []
        failures: dict[int, RuntimeErrorInfo] = {}
        for index, item in enumerate(requests):
            try:
                prepared.append(_with_identity(item, application_id, bounds))
            except RuntimeApiError as exc:
                # Per-item failure: one non-compliant spec should not take
                # down the whole batch (batch semantics per §2.3).
                failures[index] = exc.info
        produced = list(await gateway.create_sessions(prepared)) if prepared else []
        merged: list[Any] = []
        cursor = 0
        for index in range(len(requests)):
            if index in failures:
                merged.append(err(failures[index]))
            else:
                merged.append(produced[cursor])
                cursor += 1
        return pack(
            _sanitize_results(merged, include_traceback=bounds.include_traceback)
        )

    @app.get("/v1/sessions/{session_id}")
    async def get_session(session_id: str, request: Request) -> Response:
        """Query a single session's status.

        Args:
            session_id: The session identifier.
            request: The HTTP request.

        Returns:
            msgpack-encoded ``SessionStatus``.

        Raises:
            RuntimeApiError: The session does not exist or does not belong to
                the caller (``UNKNOWN_SESSION``).
        """
        application_id = identify(request)
        target = SessionId(session_id)
        rejected = owned_sessions(application_id, [target])
        if rejected:
            raise RuntimeApiError(rejected[0])
        return pack(await gateway.get_session(target))

    @app.post("/v1/sessions/renew")
    async def renew_sessions(request: Request) -> Response:
        """Batch-renew leases (``lease_seconds`` is clamped).

        Args:
            request: The HTTP request.

        Returns:
            msgpack-encoded ``list[Result[SessionStatus]]``.
        """
        body = await read_body(request)
        session_ids = body.require("session_ids", list[SessionId])
        lease = min(
            float(body.get("lease_seconds", float, bounds.max_lease_seconds)),
            bounds.max_lease_seconds,
        )
        return await batch_call(
            request,
            session_ids,
            lambda ids, _indices: gateway.renew_sessions(ids, lease),
        )

    @app.post("/v1/sessions/close")
    async def close_sessions(request: Request) -> Response:
        """Batch-close sessions.

        Args:
            request: The HTTP request.

        Returns:
            msgpack-encoded ``list[Result[None]]``.
        """
        body = await read_body(request)
        session_ids = body.require("session_ids", list[SessionId])
        return await batch_call(
            request, session_ids, lambda ids, _indices: gateway.close_sessions(ids)
        )

    # ---------------------------------------------------------------- operation plane

    @app.post("/v1/reset")
    async def reset(request: Request) -> Response:
        """Batch reset.

        Args:
            request: The HTTP request.

        Returns:
            msgpack-encoded ``list[Result[StepResult]]``.
        """
        body = await read_body(request)
        session_ids = body.require("session_ids", list[SessionId])
        spec = body.get("reset_spec", ResetSpec, ResetSpec())
        request_ids = _checked_request_ids(body, len(session_ids))
        return await batch_call(
            request,
            session_ids,
            lambda ids, indices: gateway.reset(
                ids, spec, request_ids=_slice(request_ids, indices)
            ),
        )

    @app.post("/v1/observe")
    async def observe(request: Request) -> Response:
        """Batch-read observations.

        Args:
            request: The HTTP request.

        Returns:
            msgpack-encoded ``list[Result[Observation]]``.
        """
        body = await read_body(request)
        session_ids = body.require("session_ids", list[SessionId])
        consistency = str(body.get("consistency", str, "linearizable"))
        if consistency not in ("linearizable", "eventual"):
            raise RuntimeApiError(
                make_error(
                    ErrorCode.INVALID_ARGUMENT,
                    f"unknown consistency mode: {consistency!r}",
                )
            )
        return await batch_call(
            request,
            session_ids,
            lambda ids, _indices: gateway.observe(ids, consistency=consistency),  # type: ignore[arg-type]
        )

    @app.post("/v1/action_step")
    async def action_step(request: Request) -> Response:
        """Batch-execute externally supplied action chunks.

        Args:
            request: The HTTP request.

        Returns:
            msgpack-encoded ``list[Result[StepResult]]``.
        """
        body = await read_body(request)
        session_ids = body.require("session_ids", list[SessionId])
        actions = body.require("actions", list[PayloadRef])
        if len(actions) != len(session_ids):
            raise RuntimeApiError(
                make_error(
                    ErrorCode.INVALID_ARGUMENT,
                    f"actions length {len(actions)} != session_ids length "
                    f"{len(session_ids)}",
                )
            )
        request_ids = _checked_request_ids(body, len(session_ids))
        return await batch_call(
            request,
            session_ids,
            lambda ids, indices: gateway.action_step(
                ids,
                [actions[index] for index in indices],
                request_ids=_slice(request_ids, indices),
            ),
        )

    @app.post("/v1/policy_step")
    async def policy_step(request: Request) -> Response:
        """Batch-execute the atomic ``policy_step`` operation.

        Args:
            request: The HTTP request.

        Returns:
            msgpack-encoded ``list[Result[StepResult]]``.
        """
        body = await read_body(request)
        session_ids = body.require("session_ids", list[SessionId])
        policy = body.get("policy_request", PolicyRequest, PolicyRequest())
        request_ids = _checked_request_ids(body, len(session_ids))
        return await batch_call(
            request,
            session_ids,
            lambda ids, indices: gateway.policy_step(
                ids, policy, request_ids=_slice(request_ids, indices)
            ),
        )

    @app.post("/v1/policy_infer")
    async def policy_infer(request: Request) -> Response:
        """Batch-execute "observe -> infer" without touching the environment.

        Args:
            request: The HTTP request.

        Returns:
            msgpack-encoded ``list[Result[PolicyInferResult]]``.
        """
        body = await read_body(request)
        session_ids = body.require("session_ids", list[SessionId])
        policy = body.get("policy_request", PolicyRequest, PolicyRequest())
        return await batch_call(
            request,
            session_ids,
            lambda ids, _indices: gateway.policy_infer(ids, policy),
        )

    @app.post("/v1/run_episode")
    async def run_episode(request: Request) -> Response:
        """Batch-execute a full episode (``max_steps`` is clamped).

        Args:
            request: The HTTP request.

        Returns:
            msgpack-encoded ``list[Result[EpisodeResult]]``.
        """
        body = await read_body(request)
        session_ids = body.require("session_ids", list[SessionId])
        episode = body.require("episode_request", EpisodeRequest)
        if episode.max_steps > bounds.max_episode_steps:
            episode = dataclasses.replace(episode, max_steps=bounds.max_episode_steps)
        return await batch_call(
            request,
            session_ids,
            lambda ids, _indices: gateway.run_episode(ids, episode),
        )

    @app.post("/v1/extension_call")
    async def extension_call(request: Request) -> Response:
        """Invoke a family-specific extension.

        Args:
            request: The HTTP request.

        Returns:
            msgpack-encoded ``list[Result[dict]]``.
        """
        body = await read_body(request)
        session_ids = body.require("session_ids", list[SessionId])
        namespace = str(body.require("namespace", str))
        method = str(body.require("method", str))
        args = body.get("args", dict[str, Any], {})
        return await batch_call(
            request,
            session_ids,
            lambda ids, _indices: gateway.extension_call(
                ids, namespace, method, dict(args)
            ),
        )

    # ---------------------------------------------------------------- status and cancellation

    @app.get("/v1/requests/{request_id}")
    async def get_request_status(request_id: str, request: Request) -> Response:
        """Query operation status (ownership is validated via the operation's owning session).

        Args:
            request_id: The request identifier.
            request: The HTTP request.

        Returns:
            msgpack-encoded ``OperationStatus``.

        Raises:
            RuntimeApiError: The operation does not exist or does not belong
                to the caller.
        """
        application_id = identify(request)
        target = RequestId(request_id)
        _require_owned_request(gateway, application_id, target)
        return pack(await gateway.get_request_status(target))

    @app.post("/v1/requests/{request_id}/cancel")
    async def cancel_request(request_id: str, request: Request) -> Response:
        """Best-effort cancellation of an operation.

        Args:
            request_id: The request identifier.
            request: The HTTP request.

        Returns:
            msgpack-encoded ``CancelOutcome``.

        Raises:
            RuntimeApiError: The operation does not exist or does not belong
                to the caller.
        """
        application_id = identify(request)
        target = RequestId(request_id)
        _require_owned_request(gateway, application_id, target)
        return pack(await gateway.cancel_request(target))

    return app


def _checked_request_ids(body: _Body, expected: int) -> list[RequestId] | None:
    """Retrieve the optional idempotency identifiers and validate the length.

    Args:
        body: The decoded request body.
        expected: The length of ``session_ids``.

    Returns:
        The list of idempotency identifiers; ``None`` if not provided.

    Raises:
        RuntimeApiError: The length does not match ``session_ids``.
    """
    request_ids = body.get("request_ids", list[RequestId])
    if request_ids is None:
        return None
    if len(request_ids) != expected:
        raise RuntimeApiError(
            make_error(
                ErrorCode.INVALID_ARGUMENT,
                f"request_ids length {len(request_ids)} != session_ids length "
                f"{expected}",
            )
        )
    return list(request_ids)


def _slice(
    request_ids: list[RequestId] | None, indices: Sequence[int]
) -> list[RequestId] | None:
    """Slice out the idempotency identifiers that are still in-batch after the ownership check, by index.

    Args:
        request_ids: The original list, or ``None``.
        indices: The indices that passed the ownership check.

    Returns:
        A sublist in the same order as ``indices``; still ``None`` if the
        input was ``None``.
    """
    if request_ids is None:
        return None
    return [request_ids[index] for index in indices]


def _require_owned_request(
    gateway: RuntimeGateway, application_id: str, request_id: RequestId
) -> None:
    """Validate that a ``request_id`` belongs to the caller.

    ``OperationRegistry`` is a global table, so guessing someone else's
    ``request_id`` could read their result or cancel their in-flight
    operation; ownership is therefore determined via the operation's owning
    session.

    Args:
        gateway: The Gateway.
        application_id: The caller's identity.
        request_id: The request identifier.

    Raises:
        RuntimeApiError: The operation does not exist, has no session, or the
            session does not belong to the caller (always
            ``UNKNOWN_SESSION``, without distinction, to avoid becoming an
            existence oracle).
    """
    record = gateway.operations.find(request_id)
    session_id = record.session_id if record is not None else None
    session = gateway.sessions.find(session_id) if session_id else None
    if session is None or session.application_id != application_id:
        raise RuntimeApiError(
            make_error(
                ErrorCode.UNKNOWN_SESSION,
                f"unknown request: {request_id}",
                request_id=str(request_id),
                reason="ownership",
            )
        )


def _with_identity(
    request: CreateSessionRequest, application_id: str, limits: ServeLimits
) -> CreateSessionRequest:
    """Write the server-resolved identity and clamps into the create request.

    ``application_id`` / ``auth_token`` in the request body have **zero
    effect**: the former is overridden and the latter is cleared (the served
    mode only trusts HTTP headers). ``lease_seconds`` / ``pool_size`` /
    ``max_dynamic_pool_size`` are all clamped to within
    ``limits.max_pool_size`` — the latter declares ``EnvSpecMsg``'s dynamic
    scale-up ceiling; not clamping it would let a client bypass the hard
    ``pool_size`` cap and quietly grow the pool back to an unbounded size at
    runtime.

    Args:
        request: The create request submitted by the client.
        application_id: The identity resolved from the token.
        limits: The server-side limits.

    Returns:
        A create request that can be handed to the Gateway.

    Raises:
        RuntimeApiError: ``env_family`` is not in the allowed list.
    """
    spec = request.env_spec
    if limits.allowed_env_families and spec.env_family not in (
        limits.allowed_env_families
    ):
        raise RuntimeApiError(
            make_error(
                ErrorCode.UNSUPPORTED_ENV_SPEC,
                f"env_family {spec.env_family!r} is not served here",
                env_family=spec.env_family,
                allowed=sorted(limits.allowed_env_families),
            )
        )
    if spec.pool_size > limits.max_pool_size:
        spec = dataclasses.replace(spec, pool_size=limits.max_pool_size)
    if (
        spec.max_dynamic_pool_size is not None
        and spec.max_dynamic_pool_size > limits.max_pool_size
    ):
        spec = dataclasses.replace(
            spec, max_dynamic_pool_size=limits.max_pool_size
        )
    return dataclasses.replace(
        request,
        application_id=application_id,
        auth_token=None,
        lease_seconds=min(request.lease_seconds, limits.max_lease_seconds),
        env_spec=spec,
    )
