# Copyright (c) 2026 Zetta Contributors
"""RGB/joint VLA adapter for Genie Sim's pinned CoRobot WebSocket protocol.

The G2 omnipicker uses JOINT_ABS arms and sign-inverted grippers on the
wire. Isaac remains the only authority for native joint limits. EEF, depth,
history, head and waist control require a different environment profile.
"""

from __future__ import annotations

import dataclasses
import io
import math
import threading
import time
from collections.abc import Mapping
from typing import Any
from urllib.parse import urlsplit

import msgpack
import numpy as np

from rollout_runtime.api.enums import ErrorCode
from rollout_runtime.api.errors import make_error
from rollout_runtime.api.internal import ActionResponse, InferenceRequest
from rollout_runtime.core.payload import decode_payload, encode_array


# G2_omnipicker's two native gripper joints are bounded to [0, pi/4]. The
# upstream model occasionally overshoots this saturation point by a few
# milliradians; clamp only these physical end stops and leave arm joints
# untouched so Isaac remains the authority for arm safety.
GENIESIM_GRIPPER_LIMIT = math.pi / 4


@dataclasses.dataclass(frozen=True, kw_only=True)
class GenieSimPolicyConfig:
    endpoint: str
    model_version: str
    action_dim: int = 16
    actions_per_chunk: int = 5
    connect_timeout_s: float = 15.0
    request_timeout_s: float = 120.0
    maximum_payload_bytes: int = 16 * 1024 * 1024
    require_policy_rng_ack: bool = True
    device: str = "cpu"
    dtype: str = "float32"

    def __post_init__(self) -> None:
        url = urlsplit(self.endpoint)
        if url.scheme not in {"ws", "wss"} or not url.hostname:
            raise ValueError("geniesim policy endpoint must be a ws:// or wss:// URL")
        if url.username or url.password or url.query or url.fragment:
            raise ValueError("policy endpoint must not contain credentials or query")
        if not isinstance(self.model_version, str) or not self.model_version.strip():
            raise ValueError("geniesim policy requires a frozen model_version")
        if self.action_dim != 16 or type(self.action_dim) is not int:
            raise ValueError("geniesim policy action_dim must be 16")
        if (
            type(self.actions_per_chunk) is not int
            or not 1 <= self.actions_per_chunk <= 64
        ):
            raise ValueError("actions_per_chunk must be in [1, 64]")
        for name in ("connect_timeout_s", "request_timeout_s"):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if (
            type(self.maximum_payload_bytes) is not int
            or self.maximum_payload_bytes < 1
        ):
            raise ValueError("maximum_payload_bytes must be a positive integer")
        if type(self.require_policy_rng_ack) is not bool:
            raise ValueError("require_policy_rng_ack must be boolean")
        if self.device != "cpu" or self.dtype != "float32":
            raise ValueError("remote geniesim policy requires cpu / float32 transport")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> GenieSimPolicyConfig:
        unknown = set(value) - {field.name for field in dataclasses.fields(cls)}
        if unknown:
            raise ValueError(
                f"unknown geniesim policy configuration: {sorted(unknown)}"
            )
        return cls(**value)

    def compat_key_constraints(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def _image(ref: Any) -> dict[str, Any]:
    from PIL import Image

    if ref is None:
        raise ValueError("geniesim VLA requires all three RGB cameras")
    pixels = np.asarray(decode_payload(ref))
    if pixels.dtype != np.uint8 or pixels.ndim != 3 or pixels.shape[2] != 3:
        raise ValueError("geniesim VLA camera must be HWC uint8 RGB")
    stream = io.BytesIO()
    Image.fromarray(pixels).save(stream, format="JPEG", quality=95)
    return {
        "encoding": "JPEG",
        "image_data": stream.getvalue(),
        "height": pixels.shape[0],
        "width": pixels.shape[1],
    }


def build_payload(request: InferenceRequest, *, episode_index: int) -> dict[str, Any]:
    obs = request.observation
    state = np.asarray(obs.state)
    if (
        state.shape != (21,)
        or state.dtype.kind not in "fiu"
        or not np.isfinite(state).all()
    ):
        raise ValueError("geniesim VLA requires 21 finite native joint states")
    if obs.extras.get("action_units") != "absolute_joint_radians":
        raise ValueError("geniesim VLA requires absolute_joint_radians observations")
    if len(obs.extra_view_images) != 1:
        raise ValueError("geniesim VLA requires exactly one right-hand image")
    prompt = (
        request.instruction_override
        if request.instruction_override is not None
        else obs.instruction
    )
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("geniesim VLA requires a task instruction")
    parameters = request.inference_parameters
    unknown = set(parameters) - {"mode", "seed", "actions_per_chunk"}
    if unknown or parameters.get("mode", "eval") != "eval":
        raise ValueError(f"unsupported geniesim inference parameters: {parameters}")
    timestamp = time.time_ns()
    params = {
        "timestamps": {"head": timestamp, "states": timestamp},
        "images": {
            "head": _image(obs.main_image),
            "hand_left": _image(obs.wrist_image),
            "hand_right": _image(obs.extra_view_images[0]),
        },
        "states": {
            "arm_joint_states": state[:14].tolist(),
            "gripper_states": (-state[14:16]).tolist(),
            "waist_joint_states": state[16:21].tolist(),
            "head_joint_states": [],
        },
        "prompt": prompt,
        "robot_type": "G2_omnipicker",
        "task_name": "pick_block_color",
        "episode_idx": episode_index,
        "episode_done": False,
        "task_progress": [],
    }
    if "seed" in parameters:
        seed = parameters["seed"]
        if type(seed) is not int or not 0 <= seed < 2**32:
            raise ValueError("policy seed must be an integer in [0, 2**32)")
        # Optional server extension for paired Campaign sampling. A server
        # must acknowledge it; stock CoRobot serving makes no RNG guarantee.
        params["policy_rng"] = seed
    return {"method": "infer", "params": params}


def parse_actions(response: Any) -> tuple[np.ndarray, Mapping[str, Any]]:
    if not isinstance(response, Mapping) or response.get("error"):
        raise ValueError("geniesim policy returned an error or non-object response")
    result = response.get("result")
    if not isinstance(result, Mapping):
        raise ValueError("geniesim policy response has no result object")
    history = result.get("history") or {}
    if not isinstance(history, Mapping):
        raise ValueError("invalid geniesim policy history request")
    if any(
        result.get(key)
        for key in ("head", "waist", "need_depth", "hist_frame_interval")
    ) or history.get("interval", 0):
        raise ValueError(
            "this geniesim profile cannot provide depth/history or head/waist control"
        )
    parts = []
    for name, width in (
        ("left_arm", 7),
        ("right_arm", 7),
        ("left_effector", 1),
        ("right_effector", 1),
    ):
        value = result.get(name)
        if name.endswith("arm"):
            if not isinstance(value, Mapping) or value.get("kind") != "JOINT_ABS":
                raise ValueError(f"{name} must explicitly use JOINT_ABS")
            value = value.get("values")
        array = np.asarray(value)
        if (
            array.ndim != 2
            or array.shape[1] != width
            or array.dtype.kind not in "fiu"
            or not np.isfinite(array).all()
        ):
            raise ValueError(f"invalid {name} action array")
        parts.append(array)
    horizon = len(parts[0])
    if not 1 <= horizon <= 64 or any(len(part) != horizon for part in parts):
        raise ValueError("geniesim policy action horizons must match and be in [1, 64]")
    block = np.concatenate(parts, axis=1).astype(np.float32)
    block[:, 14:16] *= -1
    block[:, 14:16] = np.clip(block[:, 14:16], 0.0, GENIESIM_GRIPPER_LIMIT)
    if not np.isfinite(block).all():
        raise ValueError("geniesim policy action overflows float32")
    return block, result


class GenieSimPolicyCore:
    policy_family = "geniesim_vla"

    def __init__(self, config: GenieSimPolicyConfig, *, connect: Any = None) -> None:
        self.config = config
        self._connect = connect
        self._socket = None
        self._episode = None
        self._episode_index = -1
        self._lock = threading.RLock()
        self._loaded = False

    @property
    def model_version(self) -> str:
        return self.config.model_version

    @property
    def device(self) -> str:
        return self.config.device

    @property
    def dtype(self) -> str:
        return self.config.dtype

    def load(self) -> None:
        if self._connect is None:
            from websockets.sync.client import connect

            self._connect = connect
        self._loaded = True

    def update_weights(self, model_version: str) -> None:
        if model_version != self.model_version:
            raise ValueError(
                "restart geniesim policy with a newly frozen model_version"
            )

    def _disconnect(self) -> None:
        socket, self._socket = self._socket, None
        if socket is not None:
            socket.close()

    def close(self) -> None:
        with self._lock:
            self._loaded = False
            self._disconnect()

    def infer_batch(self, requests: list[InferenceRequest]) -> list[ActionResponse]:
        with self._lock:
            return [self._infer(request) for request in requests]

    def _infer(self, request: InferenceRequest) -> ActionResponse:
        response = {
            "request_id": request.request_id,
            "session_id": request.session_id,
            "binding_token": request.binding_token,
            "episode_id": request.episode_id,
            "operation_seq": request.operation_seq,
            "model_version": self.model_version,
        }
        code = ErrorCode.INVALID_ARGUMENT
        try:
            if not self._loaded:
                raise RuntimeError("geniesim policy core is not loaded")
            if request.model_version_hint not in (None, self.model_version):
                raise ValueError("requested model_version does not match frozen policy")
            episode = (request.session_id, request.episode_id)
            if episode != self._episode:
                self._disconnect()
                self._episode = episode
                self._episode_index += 1
            payload = build_payload(request, episode_index=self._episode_index)
            horizon = request.inference_parameters.get(
                "actions_per_chunk", self.config.actions_per_chunk
            )
            if type(horizon) is not int or not 1 <= horizon <= 64:
                raise ValueError("actions_per_chunk must be in [1, 64]")
            data = msgpack.packb(payload, use_bin_type=True)
            if len(data) > self.config.maximum_payload_bytes:
                raise ValueError(
                    "geniesim policy request exceeds maximum_payload_bytes"
                )
            code = ErrorCode.POLICY_FAILURE
            remaining = self.config.request_timeout_s
            if request.deadline is not None:
                remaining = min(remaining, request.deadline - time.time())
            if remaining <= 0:
                raise TimeoutError("geniesim policy request deadline expired")
            started = time.monotonic()
            if self._socket is None:
                self._socket = self._connect(
                    self.config.endpoint,
                    compression=None,
                    max_size=self.config.maximum_payload_bytes,
                    open_timeout=min(remaining, self.config.connect_timeout_s),
                    close_timeout=1.0,
                )
            remaining -= time.monotonic() - started
            if remaining <= 0:
                raise TimeoutError("geniesim policy deadline expired while connecting")
            self._socket.send(data)
            raw = self._socket.recv(timeout=remaining)
            if not isinstance(raw, bytes):
                raise ValueError("geniesim policy response must be binary msgpack")
            block, result = parse_actions(msgpack.unpackb(raw, raw=False))
            version = result.get("model_version")
            if version is not None and version != self.model_version:
                raise ValueError("server model_version differs from frozen policy")
            seed = request.inference_parameters.get("seed")
            if seed is not None and self.config.require_policy_rng_ack:
                if (
                    type(result.get("policy_rng")) is not int
                    or result["policy_rng"] != seed
                    or version != self.model_version
                ):
                    raise ValueError(
                        "Campaign requires server acknowledgement of policy_rng and model_version"
                    )
            return ActionResponse(
                **response,
                actions=encode_array(block[:horizon]),
                auxiliary_outputs={
                    "protocol": "geniesim_corobot_joint_abs_v1",
                    "policy_rng_acknowledged": seed is not None
                    and type(result.get("policy_rng")) is int
                    and result["policy_rng"] == seed
                    and version == self.model_version,
                },
            )
        except Exception as error:
            if code == ErrorCode.POLICY_FAILURE:
                self._disconnect()
            return ActionResponse(
                **response, error=make_error(code, str(error), retryable=False)
            )
