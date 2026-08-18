# Copyright (c) 2026 RPent Contributors
"""Loopback-only persistent GR00T N1.5 service for RoboCasa.

This adapter implements the RoboCasa GR00T wire contract while adding bounded
admission, immutable checkpoint identity and secret-free health metadata.
Inference remains serialized because the upstream policy mutates process-global
RNG state for request-level replay.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import threading
import time
import traceback
import uuid
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Mapping

import numpy as np

MAX_REQUEST_BYTES = 128 * 1024 * 1024


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()


def parse_inference_seed(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("request seed must be an integer")
    if value < 0 or value > 2**31 - 1:
        raise ValueError("request seed must be in [0, 2**31 - 1]")
    return value


def checkpoint_digest(root: str | Path) -> str:
    """Hash model config, index and weight shards without trusting filenames."""

    root = Path(root).resolve()
    required = (root / "config.json", root / "model.safetensors.index.json")
    if any(not path.is_file() for path in required):
        raise ValueError("checkpoint is missing config or safetensors index")
    files = [*required, *sorted(root.glob("model-*.safetensors"))]
    if len(files) <= len(required):
        raise ValueError("checkpoint has no model weight shards")
    rows = []
    for path in files:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            while chunk := stream.read(8 * 1024 * 1024):
                digest.update(chunk)
        rows.append(
            {
                "name": path.name,
                "size": path.stat().st_size,
                "sha256": digest.hexdigest(),
            }
        )
    return hashlib.sha256(_canonical_json(rows)).hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        if not np.isfinite(value).all():
            raise ValueError("policy output contains non-finite values")
        return value.tolist()
    if isinstance(value, np.generic):
        scalar = value.item()
        if isinstance(scalar, float) and not np.isfinite(scalar):
            raise ValueError("policy output contains non-finite values")
        return scalar
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _observation_arrays(value: Any) -> dict[str, np.ndarray]:
    if not isinstance(value, Mapping):
        raise ValueError("request requires an observation object")
    result: dict[str, np.ndarray] = {}
    for raw_key, raw_value in value.items():
        key = str(raw_key)
        if key.startswith("video."):
            array = np.asarray(raw_value, dtype=np.uint8)
            if array.ndim != 4 or array.shape[0] != 1 or array.shape[-1] != 3:
                raise ValueError(f"{key} must have shape [1,H,W,3]")
        elif key.startswith("state."):
            array = np.asarray(raw_value, dtype=np.float32)
            if not np.isfinite(array).all():
                raise ValueError(f"{key} contains non-finite values")
        else:
            array = np.asarray(raw_value)
        result[key] = array
    return result


def _seed_process(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class Gr00tRuntime:
    """One loaded policy with serialized, bounded inference admission."""

    def __init__(
        self,
        *,
        policy: Any,
        data_config: Any,
        checkpoint_sha256: str,
        denoising_steps: int,
        maximum_pending: int = 32,
    ) -> None:
        if maximum_pending < 1:
            raise ValueError("maximum_pending must be positive")
        self.policy = policy
        self.data_config = data_config
        self.checkpoint_sha256 = checkpoint_sha256
        self.denoising_steps = denoising_steps
        self._admission = threading.BoundedSemaphore(maximum_pending)
        self._inference_lock = threading.Lock()

    @property
    def schema(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "observation": {
                "video": list(self.data_config.video_keys),
                "state": list(self.data_config.state_keys),
                "language": list(self.data_config.language_keys),
                "observation_indices": list(self.data_config.observation_indices),
            },
            "action": {
                "keys": list(self.data_config.action_keys),
                "action_indices": list(self.data_config.action_indices),
            },
            "checkpoint_sha256": self.checkpoint_sha256,
        }

    @property
    def health(self) -> dict[str, Any]:
        return {
            "status": "healthy",
            "model": "GR00T_N1_5",
            "checkpoint_sha256": self.checkpoint_sha256,
            "denoising_steps": self.denoising_steps,
            "serialized_inference": True,
            "request_seed_supported": True,
        }

    def act(self, payload: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        seed = parse_inference_seed(payload.get("seed"))
        observation = _observation_arrays(payload.get("observation"))
        admitted = time.monotonic()
        if not self._admission.acquire(blocking=False):
            raise RuntimeError("inference admission is full")
        try:
            with self._inference_lock:
                started = time.monotonic()
                _seed_process(seed)
                action = self.policy.get_action(observation)
                finished = time.monotonic()
        finally:
            self._admission.release()
        result = _jsonable(action)
        if not isinstance(result, dict):
            raise ValueError("GR00T policy must return an action object")
        return result, {
            "request_id": uuid.uuid4().hex,
            "checkpoint_sha256": self.checkpoint_sha256,
            "queue_latency_s": started - admitted,
            "inference_latency_s": finished - started,
        }


def _load_runtime(args: argparse.Namespace) -> Gr00tRuntime:
    root = args.groot_root.expanduser().resolve()
    model = args.model_path.expanduser().resolve()
    if not root.is_dir() or not model.is_dir():
        raise RuntimeError("GR00T source or checkpoint directory is missing")
    identity = checkpoint_digest(model)
    if args.expected_checkpoint_sha256 and identity != args.expected_checkpoint_sha256:
        raise RuntimeError("checkpoint digest differs from the frozen manifest")
    sys.path.insert(0, str(root))
    from gr00t.experiment.data_config import DATA_CONFIG_MAP
    from gr00t.model.policy import Gr00tPolicy

    data_config = DATA_CONFIG_MAP[args.data_config]
    policy = Gr00tPolicy(
        model_path=str(model),
        modality_config=data_config.modality_config(),
        modality_transform=data_config.transform(),
        embodiment_tag=args.embodiment_tag,
        denoising_steps=args.denoising_steps,
    )
    return Gr00tRuntime(
        policy=policy,
        data_config=data_config,
        checkpoint_sha256=identity,
        denoising_steps=args.denoising_steps,
        maximum_pending=args.maximum_pending,
    )


def serve(*, host: str, port: int, runtime: Gr00tRuntime) -> None:
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
