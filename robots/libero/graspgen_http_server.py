"""Loopback HTTP bridge from RPent's proposal schema to GraspGen ZMQ."""

from __future__ import annotations

import argparse
import json
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import numpy as np


class GraspGenBridge:
    def __init__(self, host: str, port: int, timeout_s: float) -> None:
        from grasp_gen.serving.zmq_client import GraspGenClient

        self._client = GraspGenClient(
            host=host,
            port=port,
            timeout_ms=int(timeout_s * 1000),
            wait_for_server=True,
        )
        self._lock = threading.Lock()
        self._metadata = self._client.get_metadata()

    def health(self) -> dict[str, Any]:
        with self._lock:
            ready = self._client.health_check()
        return {
            "ok": ready,
            "ready": ready,
            "status": "ready" if ready else "backend_unavailable",
            "backend": "nvlabs-graspgen-zmq",
            "metadata": self._metadata,
            "proposal_only": True,
        }

    def propose(self, payload: dict[str, Any]) -> dict[str, Any]:
        points = np.asarray(payload.get("point_cloud"), dtype=np.float32)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("point_cloud must have shape Nx3")
        if not 32 <= len(points) <= 65_536 or not np.isfinite(points).all():
            raise ValueError("point_cloud must contain 32..65536 finite points")
        max_candidates = int(
            payload.get("topk_num_grasps", payload.get("max_candidates", 16))
        )
        if not 1 <= max_candidates <= 64:
            raise ValueError("max_candidates must be in [1,64]")

        started = time.perf_counter()
        with self._lock:
            grasps, scores = self._client.infer(
                points,
                num_grasps=max(200, max_candidates * 4),
                topk_num_grasps=max_candidates,
                min_grasps=min(40, max_candidates),
                max_tries=3,
                remove_outliers=bool(payload.get("remove_outliers", False)),
            )
        candidates = [
            {
                "transform_model": pose.astype(float).tolist(),
                "score": float(score),
            }
            for pose, score in zip(grasps[:max_candidates], scores[:max_candidates])
        ]
        return {
            "ok": True,
            "ready": True,
            "status": "ok",
            "grasps": candidates,
            "proposal_only": True,
            "environment_advanced": False,
            "latency_s": round(time.perf_counter() - started, 3),
            "evidence": {
                "backend": "nvlabs-graspgen-zmq",
                "input_point_count": len(points),
                "collision_filter_requested": bool(
                    payload.get("filter_collisions", True)
                ),
                "collision_filter_applied": False,
                "outlier_filter_applied": bool(
                    payload.get("remove_outliers", False)
                ),
                "collision_filter_note": (
                    "RPent separately validates current simulator contacts; this "
                    "sensor-only bridge does not receive a privileged scene mesh."
                ),
            },
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18093)
    parser.add_argument("--zmq-host", default="127.0.0.1")
    parser.add_argument("--zmq-port", type=int, default=5558)
    parser.add_argument("--timeout-s", type=float, default=420.0)
    args = parser.parse_args()
    bridge = GraspGenBridge(args.zmq_host, args.zmq_port, args.timeout_s)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args: Any) -> None:
            return

        def _write(self, status: HTTPStatus, value: dict[str, Any]) -> None:
            body = json.dumps(value).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            if self.path != "/health":
                self._write(HTTPStatus.NOT_FOUND, {"error": "not found"})
                return
            self._write(HTTPStatus.OK, bridge.health())

        def do_POST(self) -> None:
            if self.path not in {"/generate", "/propose"}:
                self._write(HTTPStatus.NOT_FOUND, {"error": "not found"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length))
                self._write(HTTPStatus.OK, bridge.propose(payload))
            except Exception as exc:
                self._write(
                    HTTPStatus.BAD_REQUEST,
                    {"ok": False, "error": f"{type(exc).__name__}: {exc}"},
                )

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.serve_forever()


if __name__ == "__main__":
    main()
