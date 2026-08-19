# Copyright (c) 2026 Zetta Contributors
from __future__ import annotations

import base64
import io
import json
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Iterator

import numpy as np
import pytest
from PIL import Image

from robots.robocasa.groot_client import Gr00tClient


def _image() -> str:
    buffer = io.BytesIO()
    Image.new("RGB", (4, 5), (12, 34, 56)).save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def _observation() -> dict[str, Any]:
    image = _image()
    return {
        "observation": {
            "state": {
                "state.end_effector_position_relative": [0.1, 0.2, 0.3],
                "state.end_effector_rotation_relative": [0.0, 0.0, 0.0, 1.0],
                "state.gripper_qpos": [0.04, -0.04],
                "state.base_position": [1.0, 2.0, 0.7],
                "state.base_rotation": [0.0, 0.0, 0.0, 1.0],
            },
            "images": {
                "video.robot0_agentview_left": image,
                "video.robot0_agentview_right": image,
                "video.robot0_eye_in_hand": image,
            },
        }
    }


def _schema() -> dict[str, Any]:
    return {
        "observation": {
            "video": [
                "video.robot0_agentview_left",
                "video.robot0_agentview_right",
                "video.robot0_eye_in_hand",
            ],
            "state": [
                "state.end_effector_position_relative",
                "state.end_effector_rotation_relative",
                "state.gripper_qpos",
                "state.base_position",
                "state.base_rotation",
            ],
            "language": ["annotation.human.task_description"],
        },
        "action": {
            "keys": [
                "action.end_effector_position",
                "action.end_effector_rotation",
                "action.gripper_close",
                "action.base_motion",
                "action.control_mode",
            ]
        },
    }


@contextmanager
def _server(
    *, schema: dict[str, Any] | None = None, actions: dict[str, Any] | None = None
) -> Iterator[tuple[str, type[BaseHTTPRequestHandler]]]:
    class Handler(BaseHTTPRequestHandler):
        last_payload: dict[str, Any] | None = None
        schema_value = _schema() if schema is None else schema
        actions_value = actions or {
            "action.end_effector_position": [[0.1, 0.0, 0.0], [2.0, 0.0, 0.0]],
            "action.end_effector_rotation": [[0.0, 0.0, 0.0], [0.0, 0.1, 0.0]],
            "action.gripper_close": [[-1.0], [1.0]],
            "action.base_motion": [[0.0] * 4, [0.0] * 4],
            "action.control_mode": [[-1.0], [1.0]],
        }

        def _send(self, value: dict[str, Any]) -> None:
            body = json.dumps(value).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            assert self.path == "/schema"
            self._send(type(self).schema_value)

        def do_POST(self) -> None:
            assert self.path == "/act"
            size = int(self.headers["Content-Length"])
            type(self).last_payload = json.loads(self.rfile.read(size))
            self._send(type(self).actions_value)

        def log_message(self, format: str, *args: Any) -> None:
            del format, args

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{httpd.server_port}", Handler
    finally:
        httpd.shutdown()
        thread.join(timeout=2)
        httpd.server_close()


def test_groot_client_validates_schema_and_preserves_full_chunk() -> None:
    with _server() as (url, handler):
        actions, metadata = Gr00tClient(url).act(
            _observation(), instruction="Slide the rack.", inference_seed=123
        )
    assert len(actions) == 2
    assert actions[1][0] == 1.0
    assert actions[0][6] == 0.0
    assert actions[1][6] == 1.0
    assert metadata["horizon"] == 2
    assert metadata["clamped_values"] == 1
    assert metadata["schema_sha256"]
    assert "annotation.human.task_description" in metadata["observation_field_sha256"]
    assert handler.last_payload is not None
    assert handler.last_payload["seed"] == 123
    assert np.asarray(
        handler.last_payload["observation"]["video.robot0_agentview_left"]
    ).shape == (1, 5, 4, 3)


@pytest.mark.parametrize("seed", [True, 1.5, -1, 2**31])
def test_groot_client_rejects_invalid_policy_rng(seed: Any) -> None:
    with _server() as (url, _):
        with pytest.raises(ValueError, match="inference_seed"):
            Gr00tClient(url).act(
                _observation(), instruction="Slide the rack.", inference_seed=seed
            )


def test_groot_client_fails_closed_on_schema_or_nonfinite_action() -> None:
    broken = _schema()
    broken["action"]["keys"].remove("action.control_mode")
    with _server(schema=broken) as (url, _):
        with pytest.raises(ValueError, match="schema omitted action"):
            Gr00tClient(url).act(
                _observation(), instruction="Slide the rack.", inference_seed=1
            )
    actions = {
        "action.end_effector_position": [[float("nan"), 0.0, 0.0]],
        "action.end_effector_rotation": [[0.0, 0.0, 0.0]],
        "action.gripper_close": [[0.0]],
        "action.base_motion": [[0.0] * 4],
        "action.control_mode": [[0.0]],
    }
    with _server(actions=actions) as (url, _):
        with pytest.raises(ValueError, match="non-finite"):
            Gr00tClient(url).act(
                _observation(), instruction="Slide the rack.", inference_seed=1
            )
