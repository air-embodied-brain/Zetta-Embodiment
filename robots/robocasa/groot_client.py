# Copyright (c) 2026 Zetta Contributors
"""Deterministic GR00T HTTP client for RoboCasa observations."""

from __future__ import annotations

import base64
import hashlib
import io
import json
import time
import urllib.error
import urllib.request
from typing import Any

import numpy as np

LANGUAGE_KEY = "annotation.human.task_description"
VIDEO_KEYS = (
    "video.robot0_agentview_left",
    "video.robot0_agentview_right",
    "video.robot0_eye_in_hand",
)
STATE_FIELDS = {
    "state.end_effector_position_relative": 3,
    "state.end_effector_rotation_relative": 4,
    "state.gripper_qpos": 2,
    "state.base_position": 3,
    "state.base_rotation": 4,
}
ACTION_FIELDS = {
    "end_effector_position": 3,
    "end_effector_rotation": 3,
    "gripper_close": 1,
    "base_motion": 4,
    "control_mode": 1,
}


def _inference_seed(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("GR00T inference_seed must be an integer")
    if value < 0 or value > 2**31 - 1:
        raise ValueError("GR00T inference_seed must be in [0, 2**31 - 1]")
    return value


def _decode_data_image(value: str) -> np.ndarray:
    import imageio.v3 as iio

    if not value.startswith("data:") or "," not in value:
        raise ValueError("GR00T camera observation must be a data URL")
    payload = base64.b64decode(value.split(",", 1)[1], validate=True)
    image = np.asarray(iio.imread(io.BytesIO(payload)))
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError("GR00T camera observation must be HxWx3 RGB")
    return image.astype(np.uint8, copy=False)


def _vector(value: Any, size: int, key: str) -> list[float]:
    array = np.asarray(value, dtype=np.float32)
    if array.shape == (1, size):
        array = array[0]
    if array.shape != (size,):
        raise ValueError(f"{key} must have shape ({size},), got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{key} contains non-finite values")
    return array.astype(float).tolist()


class Gr00tClient:
    def __init__(
        self,
        base_url: str,
        *,
        timeout_s: float = 60.0,
        verify_schema: bool = True,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self.verify_schema = verify_schema
        self._schema: dict[str, Any] | None = None

    def _request_json(
        self, path: str, *, payload: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        body = (
            None
            if payload is None
            else json.dumps(payload, separators=(",", ":")).encode("utf-8")
        )
        request = urllib.request.Request(
            self.base_url + path,
            data=body,
            headers={"Accept": "application/json", "Content-Type": "application/json"},
            method="POST" if body is not None else "GET",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                value = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"GR00T HTTP {exc.code}: {detail[:500]}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"GR00T request failed: {exc}") from exc
        if not isinstance(value, dict):
            raise ValueError("GR00T response must be an object")
        return value

    def schema(self) -> dict[str, Any]:
        value = self._request_json("/schema")
        observation = value.get("observation")
        action = value.get("action")
        if not isinstance(observation, dict) or not isinstance(action, dict):
            raise ValueError("GR00T schema omitted observation or action contract")
        observed = {
            str(item)
            for key in ("video", "state", "language")
            for item in observation.get(key, ())
        }
        required_observation = set(VIDEO_KEYS) | set(STATE_FIELDS) | {LANGUAGE_KEY}
        if not required_observation.issubset(observed):
            missing = sorted(required_observation - observed)
            raise ValueError(f"GR00T schema omitted observation keys: {missing}")
        action_keys = {str(item) for item in action.get("keys", ())}
        required_actions = {f"action.{key}" for key in ACTION_FIELDS}
        if not required_actions.issubset(action_keys):
            missing = sorted(required_actions - action_keys)
            raise ValueError(f"GR00T schema omitted action keys: {missing}")
        self._schema = value
        return value

    def act(
        self,
        observation_response: dict[str, Any],
        *,
        instruction: str,
        inference_seed: int,
    ) -> tuple[list[list[float]], dict[str, Any]]:
        validated_seed = _inference_seed(inference_seed)
        if self.verify_schema and self._schema is None:
            self.schema()
        container = observation_response.get("observation", observation_response)
        state = container.get("state") or {}
        images = container.get("images") or {}
        packed: dict[str, Any] = {}
        manifests: dict[str, str] = {}
        for key, size in STATE_FIELDS.items():
            vector = _vector(state.get(key), size, key)
            packed[key] = [vector]
            manifests[key] = hashlib.sha256(
                np.asarray(vector, dtype=np.float32).tobytes()
            ).hexdigest()
        packed[LANGUAGE_KEY] = [instruction]
        for key in VIDEO_KEYS:
            image = _decode_data_image(str(images.get(key, "")))
            packed[key] = [image.tolist()]
            manifests[key] = hashlib.sha256(image.tobytes()).hexdigest()
        manifests[LANGUAGE_KEY] = hashlib.sha256(instruction.encode()).hexdigest()
        started = time.monotonic()
        result = self._request_json(
            "/act", payload={"observation": packed, "seed": validated_seed}
        )
        latency_s = time.monotonic() - started
        action_object = result.get("actions", result)
        arrays: dict[str, np.ndarray] = {}
        horizon: int | None = None
        for key, size in ACTION_FIELDS.items():
            value = action_object.get(f"action.{key}", action_object.get(key))
            array = np.asarray(value, dtype=np.float32)
            if array.ndim == 3 and array.shape[0] == 1:
                array = array[0]
            if array.shape == (size,):
                array = array[None, :]
            if array.ndim != 2 or array.shape[1] != size:
                raise ValueError(f"GR00T action {key} has invalid shape {array.shape}")
            if not np.isfinite(array).all():
                raise ValueError(f"GR00T action {key} contains non-finite values")
            if horizon is None:
                horizon = len(array)
            elif horizon != len(array):
                raise ValueError("GR00T action fields disagree on horizon")
            arrays[key] = array
        if not horizon:
            raise ValueError("GR00T returned an empty action chunk")
        actions = []
        clamped_values = 0
        for index in range(horizon):
            gripper = 1.0 if arrays["gripper_close"][index, 0] > 0 else 0.0
            control = 1.0 if arrays["control_mode"][index, 0] > 0 else 0.0
            action = np.concatenate(
                [
                    arrays["end_effector_position"][index],
                    arrays["end_effector_rotation"][index],
                    np.asarray([gripper], dtype=np.float32),
                    arrays["base_motion"][index],
                    np.asarray([control], dtype=np.float32),
                ]
            )
            clipped = np.clip(action, -1.0, 1.0)
            clamped_values += int(np.count_nonzero(clipped != action))
            actions.append(clipped.astype(float).tolist())
        return actions, {
            "inference_seed": validated_seed,
            "observation_field_sha256": manifests,
            "action_chunk_sha256": hashlib.sha256(
                np.asarray(actions, dtype=np.float32).tobytes()
            ).hexdigest(),
            "horizon": horizon,
            "latency_s": latency_s,
            "clamped_values": clamped_values,
            "schema_sha256": (
                hashlib.sha256(
                    json.dumps(
                        self._schema, sort_keys=True, separators=(",", ":")
                    ).encode()
                ).hexdigest()
                if self._schema is not None
                else None
            ),
        }
