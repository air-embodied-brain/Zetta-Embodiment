# Copyright (c) 2026 Zetta Contributors
"""Persistent RoboCasa environment with server-side chunk execution.

The implementation is clean-room and uses RoboCasa's public Gym action
mapping.  One process owns one persistent environment and isolated renderer;
many such processes may share a GPU.
"""

from __future__ import annotations

import argparse
import json
import os
import threading
import traceback
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from robots.robocasa.operation_protocol import IdempotentWriteRegistry
from robots.robocasa.session_core import (
    CAMERA_KEYS,
    RoboCasaSession,
    SimulationHealthError,
    isolated_renderer_status,
)

__all__ = [
    "BoundedThreadingHTTPServer",
    "RoboCasaSession",
    "SimulationHealthError",
    "isolated_renderer_status",
    "serve",
    "main",
]

MAXIMUM_REQUEST_BYTES = 8 * 1024 * 1024


class BoundedThreadingHTTPServer(ThreadingHTTPServer):
    """Reject excess requests before creating another handler thread."""

    daemon_threads = True

    def __init__(self, server_address: tuple[str, int], handler: Any, *, limit: int):
        super().__init__(server_address, handler)
        self._admission = threading.BoundedSemaphore(limit)

    def process_request(self, request: Any, client_address: Any) -> None:
        if not self._admission.acquire(blocking=False):
            body = b'{"error":"queue_full","retry_after_s":0.25}'
            response = (
                b"HTTP/1.1 503 Service Unavailable\r\n"
                b"Content-Type: application/json\r\n"
                + f"Content-Length: {len(body)}\r\n".encode()
                + b"Retry-After: 1\r\nConnection: close\r\n\r\n"
                + body
            )
            try:
                request.sendall(response)
            finally:
                self.shutdown_request(request)
            return
        try:
            super().process_request(request, client_address)
        except Exception:
            self._admission.release()
            raise

    def process_request_thread(self, request: Any, client_address: Any) -> None:
        try:
            super().process_request_thread(request, client_address)
        finally:
            self._admission.release()


def serve(
    *,
    host: str,
    port: int,
    camera_size: int,
    max_steps: int,
    cold_reset_lock: str | None,
    require_isolated_renderer: bool,
    maximum_inflight_requests: int = 2,
) -> None:
    if maximum_inflight_requests < 1:
        raise ValueError("maximum_inflight_requests must be positive")
    session = RoboCasaSession(
        camera_size=camera_size,
        max_steps=max_steps,
        cold_reset_lock=cold_reset_lock,
        require_isolated_renderer=require_isolated_renderer,
    )
    write_registry = IdempotentWriteRegistry()

    class Handler(BaseHTTPRequestHandler):
        def _send(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
            body = json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")
            self.send_response(int(status))
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _payload(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length", "0"))
            if length < 0 or length > MAXIMUM_REQUEST_BYTES:
                raise ValueError("request body exceeds the configured limit")
            value = (
                json.loads(self.rfile.read(length).decode("utf-8")) if length else {}
            )
            if not isinstance(value, dict):
                raise ValueError("request body must be an object")
            return value

        def do_GET(self) -> None:
            if self.path == "/health":
                self._send(
                    HTTPStatus.OK,
                    {
                        "status": "healthy",
                        "persistent": True,
                        "renderer": session.renderer_status,
                        "gpu_visible": os.environ.get("CUDA_VISIBLE_DEVICES"),
                        "egl_device": os.environ.get("MUJOCO_EGL_DEVICE_ID"),
                        "write_protocol": write_registry.state,
                    },
                )
            elif self.path == "/schema":
                self._send(
                    HTTPStatus.OK,
                    {
                        "reset": "POST /reset",
                        "observation": "POST /observation",
                        "execute_chunk": "POST /execute_chunk",
                        "finalize_episode": "POST /finalize_episode",
                        "release": "POST /release",
                        "critic_semantics": "proposal_only",
                        "environment_writer": "session_only",
                        "action_contract": "named_robocasa_gym_mapping",
                        "flat_action_size": 12,
                        "camera_keys": list(CAMERA_KEYS),
                        "write_protocol": "binding+episode+sequence+request_digest",
                        "maximum_inflight_requests": maximum_inflight_requests,
                    },
                )
            else:
                self._send(HTTPStatus.NOT_FOUND, {"error": "not_found"})

        def do_POST(self) -> None:
            try:
                payload = self._payload()

                def marker() -> tuple[Any, ...]:
                    return (
                        id(session.env),
                        session.identity,
                        session.step_index,
                        session.observation is not None,
                    )

                if self.path == "/reset":
                    terminal = write_registry.execute(
                        self.path,
                        payload,
                        lambda: session.reset(payload),
                        side_effect_marker=marker,
                    )
                    self._send(terminal.status, terminal.payload)
                    return
                elif self.path == "/observation":
                    result = session.snapshot(
                        include_images=bool(payload.get("include_images", True))
                    )
                elif self.path == "/execute_chunk":
                    terminal = write_registry.execute(
                        self.path,
                        payload,
                        lambda: session.execute_chunk(payload),
                        side_effect_marker=marker,
                    )
                    self._send(terminal.status, terminal.payload)
                    return
                elif self.path == "/finalize_episode":
                    terminal = write_registry.execute(
                        self.path,
                        payload,
                        session.finalize_episode_artifacts,
                        side_effect_marker=marker,
                    )
                    self._send(terminal.status, terminal.payload)
                    return
                elif self.path == "/release":
                    terminal = write_registry.execute(
                        self.path,
                        payload,
                        session.finalize_episode_artifacts,
                        side_effect_marker=marker,
                        release_binding=True,
                    )
                    self._send(terminal.status, terminal.payload)
                    return
                elif self.path == "/close":
                    terminal = write_registry.execute(
                        self.path,
                        payload,
                        lambda: session.close_environment() or {"closed": True},
                        side_effect_marker=marker,
                    )
                    self._send(terminal.status, terminal.payload)
                    return
                else:
                    self._send(HTTPStatus.NOT_FOUND, {"error": "not_found"})
                    return
                self._send(HTTPStatus.OK, result)
            except ValueError as exc:
                self._send(
                    HTTPStatus.CONFLICT,
                    {"error": type(exc).__name__, "detail": str(exc)},
                )
            except Exception as exc:
                traceback.print_exc()
                self._send(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    {"error": type(exc).__name__, "detail": str(exc)},
                )

        def log_message(self, format: str, *args: Any) -> None:
            print(f"{self.client_address[0]} - {format % args}", flush=True)

    server = BoundedThreadingHTTPServer(
        (host, port), Handler, limit=maximum_inflight_requests
    )
    try:
        server.serve_forever()
    finally:
        session.close_environment()


def main() -> None:
    parser = argparse.ArgumentParser(description="Persistent RoboCasa chunk server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18800)
    parser.add_argument("--camera-size", type=int, default=256)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--cold-reset-lock")
    parser.add_argument("--maximum-inflight-requests", type=int, default=2)
    parser.add_argument(
        "--allow-shared-renderer",
        action="store_true",
        help="diagnostic only; production requires an isolated MuJoCo renderer",
    )
    args = parser.parse_args()
    serve(
        host=args.host,
        port=args.port,
        camera_size=args.camera_size,
        max_steps=args.max_steps,
        cold_reset_lock=args.cold_reset_lock,
        require_isolated_renderer=not args.allow_shared_renderer,
        maximum_inflight_requests=args.maximum_inflight_requests,
    )


if __name__ == "__main__":
    main()
