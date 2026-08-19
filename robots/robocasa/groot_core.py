# Copyright (c) 2026 Zetta Contributors
"""GR00T N1.5 model loading, inference, and data contract: pure logic.

Extracted from ``groot_server.py`` (model loading + checkpoint-hash-verified
inference) and ``groot_client.py`` (the wire data contract: ``STATE_FIELDS`` /
``ACTION_FIELDS`` and their shape conversions), per
``runtime v3 design`` §4, Stage 3.

``Gr00tModelCore`` has no dependency on ``http.server``: ``load()`` loads the
checkpoint once (with the same digest verification ``groot_server.py`` always
did) and ``act()`` takes the same JSON-shaped observation dict the HTTP wire
contract used, so ``groot_server.py`` continues to expose it as a standalone
debugging service and ``rollout_runtime/backends/groot_policy.py`` (Stage 4)
can load one instance per rank and call it in-process, with no HTTP hop and no
re-serialization through ``groot_client.py``'s ``STATE_FIELDS`` /
``ACTION_FIELDS`` contract (that contract's shape-conversion rules move here
verbatim so both call sites keep reading the same source of truth).
"""

from __future__ import annotations

import hashlib
import json
import random
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Mapping

import numpy as np

MAX_REQUEST_BYTES = 128 * 1024 * 1024

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


def vector_from_state(value: Any, size: int, key: str) -> list[float]:
    """Validate and flatten one ``STATE_FIELDS`` entry to a ``[size]`` vector.

    This is ``groot_client.py``'s ``_vector`` helper, moved verbatim: it
    accepts either a bare ``(size,)`` array or the batched ``(1, size)`` shape
    the HTTP wire contract used, so callers migrating off JSON payloads keep
    the same tolerance.
    """

    array = np.asarray(value, dtype=np.float32)
    if array.shape == (1, size):
        array = array[0]
    if array.shape != (size,):
        raise ValueError(f"{key} must have shape ({size},), got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{key} contains non-finite values")
    return array.astype(float).tolist()


def decode_data_url_image(value: str) -> np.ndarray:
    """Decode one ``data:image/...`` URL to an ``HxWx3`` uint8 array.

    Moved verbatim from ``groot_client.py``'s ``_decode_data_image``.
    """

    import base64
    import io

    import imageio.v3 as iio

    if not value.startswith("data:") or "," not in value:
        raise ValueError("GR00T camera observation must be a data URL")
    payload = base64.b64decode(value.split(",", 1)[1], validate=True)
    image = np.asarray(iio.imread(io.BytesIO(payload)))
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError("GR00T camera observation must be HxWx3 RGB")
    return image.astype(np.uint8, copy=False)


def action_dict_to_flat_chunks(
    action_object: Mapping[str, Any],
) -> tuple[list[list[float]], int]:
    """Convert the GR00T ``ACTION_FIELDS`` response into flat 12-dim chunks.

    Moved verbatim from ``groot_client.py``'s ``Gr00tClient.act`` tail: reads
    ``ACTION_FIELDS``, validates per-field shape/finiteness, and concatenates
    them into the same clamp-to-``[-1, 1]`` 12-dim action layout
    ``robots/robocasa/action_contract.py`` expects.

    Returns:
        ``(actions, clamped_values)`` where ``actions`` is
        ``[horizon][12]`` and ``clamped_values`` counts values that hit the
        clamp.
    """

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
    return actions, clamped_values


class Gr00tModelCore:
    """One loaded GR00T policy with serialized, bounded inference admission.

    This is ``groot_server.py``'s ``Gr00tRuntime``, renamed to match
    ``runtime v3 design`` §3.4's naming for the pure-logic core
    (``groot_server.py`` keeps re-exporting the old name for its HTTP shell).
    """

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


# Backward-compatible alias: ``groot_server.py`` and existing tests import
# ``Gr00tRuntime``. Keep this until Stage 4 finishes retiring the old name.
Gr00tRuntime = Gr00tModelCore


def load_groot_model_core(
    *,
    groot_root: str | Path,
    model_path: str | Path,
    data_config_name: str,
    embodiment_tag: str,
    denoising_steps: int,
    maximum_pending: int = 32,
    expected_checkpoint_sha256: str | None = None,
) -> Gr00tModelCore:
    """Load one GR00T checkpoint into a ``Gr00tModelCore``.

    Moved from ``groot_server.py``'s ``_load_runtime``: verifies the
    checkpoint digest against ``expected_checkpoint_sha256`` when supplied,
    then constructs the upstream ``Gr00tPolicy`` and wraps it.
    """

    root = Path(groot_root).expanduser().resolve()
    model = Path(model_path).expanduser().resolve()
    if not root.is_dir() or not model.is_dir():
        raise RuntimeError("GR00T source or checkpoint directory is missing")
    identity = checkpoint_digest(model)
    if expected_checkpoint_sha256 and identity != expected_checkpoint_sha256:
        raise RuntimeError("checkpoint digest differs from the frozen manifest")
    sys.path.insert(0, str(root))
    from gr00t.experiment.data_config import DATA_CONFIG_MAP
    from gr00t.model.policy import Gr00tPolicy

    data_config = DATA_CONFIG_MAP[data_config_name]
    policy = Gr00tPolicy(
        model_path=str(model),
        modality_config=data_config.modality_config(),
        modality_transform=data_config.transform(),
        embodiment_tag=embodiment_tag,
        denoising_steps=denoising_steps,
    )
    return Gr00tModelCore(
        policy=policy,
        data_config=data_config,
        checkpoint_sha256=identity,
        denoising_steps=denoising_steps,
        maximum_pending=maximum_pending,
    )
