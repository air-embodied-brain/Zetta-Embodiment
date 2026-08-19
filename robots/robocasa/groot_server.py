# Copyright (c) 2026 Zetta Contributors
"""Loopback-only persistent GR00T N1.5 service for RoboCasa.

This adapter implements the RoboCasa GR00T wire contract while adding bounded
admission, immutable checkpoint identity and secret-free health metadata.
Inference remains serialized because the upstream policy mutates process-global
RNG state for request-level replay.
"""

from __future__ import annotations

import argparse
import json
import traceback
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from robots.robocasa.groot_core import (
    Gr00tModelCore,
    Gr00tRuntime,
    checkpoint_digest,
    load_groot_model_core,
    parse_inference_seed,
)

__all__ = [
    "Gr00tModelCore",
    "Gr00tRuntime",
    "checkpoint_digest",
    "parse_inference_seed",
    "serve",
    "main",
]

MAX_REQUEST_BYTES = 128 * 1024 * 1024


def _load_runtime(args: argparse.Namespace) -> Gr00tModelCore:
    return load_groot_model_core(
        groot_root=args.groot_root,
        model_path=args.model_path,
        data_config_name=args.data_config,
        embodiment_tag=args.embodiment_tag,
        denoising_steps=args.denoising_steps,
        maximum_pending=args.maximum_pending,
        expected_checkpoint_sha256=args.expected_checkpoint_sha256,
    )


def serve(*, host: str, port: int, runtime: Gr00tModelCore) -> None:
    if host not in {"127.0.0.1", "localhost"}:
        raise ValueError("GR00T service must bind loopback")

    class Handler(BaseHTTPRequestHandler):
        def _send(self, status: HTTPStatus, value: dict[str, Any]) -> None:
            body = json.dumps(value, separators=(",", ":"), allow_nan=False).encode()
            self.send_response(int(status))
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            if self.path == "/health":
                self._send(HTTPStatus.OK, runtime.health)
            elif self.path == "/schema":
                self._send(HTTPStatus.OK, runtime.schema)
            else:
                self._send(HTTPStatus.NOT_FOUND, {"error": "not_found"})

        def do_POST(self) -> None:
            if self.path != "/act":
                self._send(HTTPStatus.NOT_FOUND, {"error": "not_found"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0 or length > MAX_REQUEST_BYTES:
                    raise ValueError("request body length is invalid")
                payload = json.loads(self.rfile.read(length).decode())
                if not isinstance(payload, dict):
                    raise ValueError("request body must be an object")
                actions, metadata = runtime.act(payload)
                self._send(HTTPStatus.OK, {"actions": actions, "metadata": metadata})
            except RuntimeError as exc:
                status = (
                    HTTPStatus.TOO_MANY_REQUESTS
                    if "admission" in str(exc)
                    else HTTPStatus.INTERNAL_SERVER_ERROR
                )
                self._send(status, {"error": type(exc).__name__})
            except Exception as exc:
                traceback.print_exc()
                self._send(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    {"error": type(exc).__name__},
                )

        def log_message(self, format: str, *args: Any) -> None:
            del format, args

    server = ThreadingHTTPServer((host, port), Handler)
    try:
        server.serve_forever()
    finally:
        server.server_close()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--groot-root", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18811)
    parser.add_argument("--data-config", default="panda_omron")
    parser.add_argument("--embodiment-tag", default="new_embodiment")
    parser.add_argument("--denoising-steps", type=int, default=4)
    parser.add_argument("--maximum-pending", type=int, default=32)
    parser.add_argument("--expected-checkpoint-sha256")
    return parser


def main() -> None:
    args = _parser().parse_args()
    runtime = _load_runtime(args)
    serve(host=args.host, port=args.port, runtime=runtime)


if __name__ == "__main__":
    main()
