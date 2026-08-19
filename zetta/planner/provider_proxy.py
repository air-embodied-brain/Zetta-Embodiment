# Copyright (c) 2026 Zetta Contributors
"""Loopback Responses API proxy backed by the ordered provider pool.

The Codex SDK accepts one base URL and one API key.  This proxy keeps that
single local interface while selecting a configured upstream URL/key for every
Responses request.  It binds only to loopback and never writes request bodies,
upstream URLs, or credentials to logs.
"""

from __future__ import annotations

import json
import os
import queue
import secrets
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlencode, urlsplit

import httpx

from zetta.planner.provider_pool import (
    ProviderPoolConfig,
    ProviderRouteSpec,
    _SharedCooldowns,
    is_retryable_provider_error,
)
from zetta.utils.logging import get_logger

logger = get_logger("provider_proxy")

BROKER_URL_ENV = "ZETTA_API_PROVIDER_BROKER_URL"
BROKER_API_KEY_ENV = "ZETTA_API_PROVIDER_BROKER_API_KEY"
BROKER_MAX_CONCURRENCY_ENV = "ZETTA_API_PROVIDER_BROKER_MAX_CONCURRENCY"
BROKER_QUEUE_CAPACITY_ENV = "ZETTA_API_PROVIDER_BROKER_QUEUE_CAPACITY"
DEFAULT_BROKER_CONCURRENCY = 8
MAXIMUM_BROKER_CONCURRENCY = 20
DEFAULT_BROKER_QUEUE_CAPACITY = 256

_RETRYABLE_STATUSES = {401, 403, 404, 408, 409, 425, 429, 500, 502, 503, 504}
_FORWARDED_RESPONSE_HEADERS = {
    "content-type",
    "openai-processing-ms",
    "retry-after",
    "x-request-id",
}


@dataclass
class _QueuedRequest:
    """One authenticated request waiting for central dispatch."""

    handler: BaseHTTPRequestHandler
    body: bytes
    done: threading.Event = field(default_factory=threading.Event)


def load_provider_broker_connection(
    env: Mapping[str, str] | None = None,
) -> tuple[str, str] | None:
    """Load the external central-broker URL and local credential as one unit."""
    values = env if env is not None else os.environ
    url = values.get(BROKER_URL_ENV, "").strip()
    api_key = values.get(BROKER_API_KEY_ENV, "").strip()
    if not url and not api_key:
        return None
    if not url or not api_key:
        raise ValueError(
            f"{BROKER_URL_ENV} and {BROKER_API_KEY_ENV} must be configured together"
        )
    parsed = urlsplit(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError(f"{BROKER_URL_ENV} must be an absolute HTTP(S) URL")
    return url.rstrip("/"), api_key


class ProviderPoolProxy:
    """A bounded FIFO broker whose worker threads alone contact providers.

    The HTTP handler never selects a route or sends an upstream request.  It
    validates and enqueues work, then waits for one of the fixed dispatcher
    workers.  Running one standalone instance therefore gives every Role1 and
    Codex process a single queue, a single health view, and a hard global
    concurrency ceiling.
    """

    def __init__(
        self,
        config: ProviderPoolConfig,
        *,
        timeout_seconds: float = 900.0,
        host: str = "127.0.0.1",
        port: int = 0,
        api_key: str | None = None,
        max_concurrency: int = DEFAULT_BROKER_CONCURRENCY,
        queue_capacity: int = DEFAULT_BROKER_QUEUE_CAPACITY,
    ) -> None:
        if not 1 <= max_concurrency <= MAXIMUM_BROKER_CONCURRENCY:
            raise ValueError(
                "max_concurrency must be between 1 and "
                f"{MAXIMUM_BROKER_CONCURRENCY}"
            )
        if queue_capacity < 1:
            raise ValueError("queue_capacity must be positive")
        self._config = config
        self._timeout_seconds = timeout_seconds
        self._host = host
        self._port = port
        self._max_concurrency = max_concurrency
        self._queue_capacity = queue_capacity
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self._workers: list[threading.Thread] = []
        self._queue: queue.Queue[_QueuedRequest | None] = queue.Queue(
            maxsize=queue_capacity
        )
        self._stats_lock = threading.Lock()
        self._accepted = 0
        self._completed = 0
        self._rejected = 0
        self._inflight = 0
        self._max_inflight_observed = 0
        self._cooldowns = _SharedCooldowns(config.state_file, clock=time.time)
        self.api_key = api_key or secrets.token_urlsafe(32)
        if not self.api_key.strip():
            raise ValueError("api_key must be non-empty")

    def start(self) -> str:
        """Start the loopback server and return its base URL."""
        if self._server is not None:
            raise RuntimeError("provider-pool proxy is already running")
        proxy = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def do_GET(self) -> None:  # noqa: N802 - stdlib handler API
                if self.path.rstrip("/") == "/health":
                    self._send_json(
                        HTTPStatus.OK,
                        {"ok": True, "broker": proxy.stats()},
                    )
                elif self.path.rstrip("/") == "/stats":
                    self._send_json(HTTPStatus.OK, proxy.stats())
                else:
                    self._send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})

            def do_POST(self) -> None:  # noqa: N802 - stdlib handler API
                if not self.path.split("?", 1)[0].rstrip("/").endswith("/responses"):
                    self._send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})
                    return
                try:
                    content_length = int(self.headers.get("content-length", "0"))
                except ValueError:
                    self._send_json(
                        HTTPStatus.BAD_REQUEST, {"error": "invalid content-length"}
                    )
                    return
                if content_length <= 0 or content_length > 64 * 1024 * 1024:
                    self._send_json(
                        HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                        {"error": "invalid request size"},
                    )
                    return
                body = self.rfile.read(content_length)
                if not secrets.compare_digest(
                    self.headers.get("authorization", ""),
                    f"Bearer {proxy.api_key}",
                ):
                    self._send_json(HTTPStatus.UNAUTHORIZED, {"error": "unauthorized"})
                    return
                request = _QueuedRequest(handler=self, body=body)
                try:
                    proxy._queue.put_nowait(request)
                except queue.Full:
                    with proxy._stats_lock:
                        proxy._rejected += 1
                    self._send_json(
                        HTTPStatus.SERVICE_UNAVAILABLE,
                        {
                            "error": {
                                "type": "provider_broker_queue_full",
                                "message": "provider request queue is full",
                            }
                        },
                    )
                    return
                with proxy._stats_lock:
                    proxy._accepted += 1
                request.done.wait()

            def _send_json(self, status: int, value: dict[str, Any]) -> None:
                payload = json.dumps(value, separators=(",", ":")).encode()
                self.send_response(int(status))
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(payload)))
                self.send_header("connection", "close")
                self.end_headers()
                self.wfile.write(payload)
                self.close_connection = True

            def log_message(self, _format: str, *args: Any) -> None:
                # BaseHTTPRequestHandler would otherwise log the local request
                # path and client details to stderr.
                return

        self._queue = queue.Queue(maxsize=self._queue_capacity)
        with self._stats_lock:
            self._accepted = 0
            self._completed = 0
            self._rejected = 0
            self._inflight = 0
            self._max_inflight_observed = 0
        self._server = ThreadingHTTPServer((self._host, self._port), Handler)
        self._server.daemon_threads = True
        self._workers = [
            threading.Thread(
                target=self._dispatch_worker,
                name=f"zetta-provider-dispatch-{index}",
                daemon=True,
            )
            for index in range(self._max_concurrency)
        ]
        for worker in self._workers:
            worker.start()
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name="zetta-provider-pool-proxy",
            daemon=True,
        )
        self._thread.start()
        host, port = self._server.server_address[:2]
        return f"http://{host}:{port}"

    def stop(self) -> None:
        """Stop the proxy and wait briefly for its server thread."""
        if self._server is None:
            return
        self._server.shutdown()
        self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=5)
        self._queue.join()
        for _worker in self._workers:
            self._queue.put(None)
        for worker in self._workers:
            worker.join(timeout=5)
        self._server = None
        self._thread = None
        self._workers = []

    def stats(self) -> dict[str, int]:
        """Return a credential-free snapshot of queue and dispatch state."""
        with self._stats_lock:
            return {
                "route_count": len(self._config.routes),
                "max_concurrency": self._max_concurrency,
                "queue_capacity": self._queue_capacity,
                "queue_depth": self._queue.qsize(),
                "inflight": self._inflight,
                "max_inflight_observed": self._max_inflight_observed,
                "accepted": self._accepted,
                "completed": self._completed,
                "rejected": self._rejected,
            }

    def _dispatch_worker(self) -> None:
        """Drain the central FIFO; only this method sends provider requests."""
        while True:
            request = self._queue.get()
            if request is None:
                self._queue.task_done()
                return
            with self._stats_lock:
                self._inflight += 1
                self._max_inflight_observed = max(
                    self._max_inflight_observed, self._inflight
                )
            try:
                self._forward(request.handler, request.body)
            except Exception as exc:
                logger.warning(
                    "provider broker request failed locally (%s)", type(exc).__name__
                )
                try:
                    request.handler._send_json(  # type: ignore[attr-defined]
                        HTTPStatus.BAD_GATEWAY,
                        {"error": "provider broker request failed"},
                    )
                except (BrokenPipeError, ConnectionResetError, OSError):
                    pass
            finally:
                with self._stats_lock:
                    self._inflight -= 1
                    self._completed += 1
                request.done.set()
                self._queue.task_done()

    def _forward(self, handler: BaseHTTPRequestHandler, original_body: bytes) -> None:
        attempted: set[str] = set()
        while len(attempted) < len(self._config.routes):
            route = self._next_route(attempted)
            attempted.add(route.route_id)
            body = _body_for_route(original_body, route)
            headers = _headers_for_route(route)
            try:
                with httpx.Client(
                    timeout=httpx.Timeout(self._timeout_seconds),
                    follow_redirects=True,
                ) as client:
                    request = client.build_request(
                        "POST", _responses_url(route), headers=headers, content=body
                    )
                    response = client.send(request, stream=True)
                    if response.status_code in _RETRYABLE_STATUSES:
                        status_code = response.status_code
                        cooldown = max(
                            route.cooldown_seconds,
                            _retry_after(response.headers) or 0.0,
                        )
                        response.close()
                        self._cooldowns.mark_failure(route.route_id, cooldown)
                        logger.warning(
                            "provider route %s unavailable (HTTP %d); cooling for %.1fs and falling back",
                            route.name,
                            status_code,
                            cooldown,
                        )
                        continue
                    self._cooldowns.mark_success(route.route_id)
                    _relay_response(handler, response)
                    return
            except Exception as exc:
                if not is_retryable_provider_error(exc):
                    raise
                self._cooldowns.mark_failure(route.route_id, route.cooldown_seconds)
                logger.warning(
                    "provider route %s unavailable (%s); cooling for %.1fs and falling back",
                    route.name,
                    type(exc).__name__,
                    route.cooldown_seconds,
                )
        _send_exhausted(handler)

    def _next_route(self, attempted: set[str]) -> ProviderRouteSpec:
        remaining = [
            route for route in self._config.routes if route.route_id not in attempted
        ]
        while True:
            cooldowns = self._cooldowns.cooldown_until(
                [route.route_id for route in remaining]
            )
            now = time.time()
            for route in remaining:
                if cooldowns.get(route.route_id, 0.0) <= now:
                    return route
            delay = max(0.0, min(cooldowns.values()) - now)
            if delay:
                logger.info(
                    "all remaining provider routes are cooling; waiting %.1fs", delay
                )
                time.sleep(delay)


def _responses_url(route: ProviderRouteSpec) -> str:
    if route.provider.startswith("azure"):
        query = urlencode({"api-version": route.api_version or "2025-04-01-preview"})
        return f"{route.base_url}/openai/responses?{query}"
    return f"{route.base_url}/responses"


def _headers_for_route(route: ProviderRouteSpec) -> dict[str, str]:
    headers = {"content-type": "application/json", "accept": "text/event-stream"}
    if route.provider.startswith("azure"):
        headers["api-key"] = route.api_key
    else:
        headers["authorization"] = f"Bearer {route.api_key}"
    return headers


def _body_for_route(body: bytes, route: ProviderRouteSpec) -> bytes:
    try:
        value = json.loads(body)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return body
    if not isinstance(value, dict):
        return body
    model_name = route.model.partition(":")[2] or route.model
    value["model"] = model_name
    return json.dumps(value, separators=(",", ":")).encode()


def _relay_response(handler: BaseHTTPRequestHandler, response: httpx.Response) -> None:
    try:
        handler.send_response(response.status_code)
        for key, value in response.headers.items():
            if key.lower() in _FORWARDED_RESPONSE_HEADERS:
                handler.send_header(key, value)
        handler.send_header("connection", "close")
        handler.end_headers()
        # Upstreams may gzip even streaming Responses payloads.  The broker
        # deliberately does not forward Content-Encoding, so it must relay
        # decoded bytes rather than a raw compressed body.
        for chunk in response.iter_bytes():
            if chunk:
                handler.wfile.write(chunk)
                handler.wfile.flush()
    except (BrokenPipeError, ConnectionResetError):
        pass
    finally:
        response.close()
        handler.close_connection = True


def _send_exhausted(handler: BaseHTTPRequestHandler) -> None:
    payload = json.dumps(
        {
            "error": {
                "type": "provider_pool_exhausted",
                "message": "all configured model provider routes are unavailable",
            }
        },
        separators=(",", ":"),
    ).encode()
    handler.send_response(HTTPStatus.SERVICE_UNAVAILABLE)
    handler.send_header("content-type", "application/json")
    handler.send_header("content-length", str(len(payload)))
    handler.send_header("connection", "close")
    handler.end_headers()
    handler.wfile.write(payload)
    handler.close_connection = True


def _retry_after(headers: httpx.Headers) -> float | None:
    try:
        value = float(headers.get("retry-after", ""))
    except ValueError:
        return None
    return value if value >= 0 else None
