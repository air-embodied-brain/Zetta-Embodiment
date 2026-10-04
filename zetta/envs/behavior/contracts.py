# Copyright (c) 2026 Zetta Contributors
"""Pure observation and action contracts for the BEHAVIOR family.

The simulator is deliberately absent from this module.  Keeping conversion and
validation here makes the default test suite useful on machines that do not have
OmniGibson, Isaac, or the BEHAVIOR asset bundle installed.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

import numpy as np

ACTION_CONTRACT = "behavior_r1pro_joint_23_v1"
OBSERVATION_PROFILE = "r1pro_rgb_proprio_v1"
BEHAVIOR_ACTION_DIM = 23
DEFAULT_IMAGE_SIZE = 224


def observation_contract(
    *,
    image_size: int,
    main_camera: str,
    wrist_cameras: tuple[str, str],
    state_source: str,
    state_dim: int | None = None,
) -> dict[str, Any]:
    """Return the versioned observation schema used by the policy bridge."""
    if type(image_size) is not int or image_size < 1:
        raise ValueError("image_size must be a positive integer")
    if not main_camera:
        raise ValueError("main_camera must be named")
    if len(wrist_cameras) != 2 or any(not name for name in wrist_cameras):
        raise ValueError("wrist_cameras must contain two named cameras")
    if not state_source:
        raise ValueError("state_source must be named")
    if state_dim is not None and (type(state_dim) is not int or state_dim < 1):
        raise ValueError("state_dim must be a positive integer when set")
    return {
        "schema_version": 1,
        "profile": OBSERVATION_PROFILE,
        "image_size": image_size,
        "main_camera": main_camera,
        "wrist_cameras": list(wrist_cameras),
        "state_source": state_source,
        "state_dim": state_dim,
        "image_layout": "uint8_HWC_RGB",
        "state_layout": "r1pro_full_proprio_then_policy_slice",
        "action_dim": BEHAVIOR_ACTION_DIM,
    }


def contract_digest(value: Mapping[str, Any]) -> str:
    """Hash a JSON-compatible contract with stable ordering."""
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def validate_behavior_actions(actions: Any) -> np.ndarray:
    """Validate a nonempty chunk of finite 23-D R1Pro commands."""
    block = np.asarray(actions, dtype=np.float32)
    if block.ndim != 2 or block.shape[0] < 1 or block.shape[1] != BEHAVIOR_ACTION_DIM:
        raise ValueError(
            f"expected nonempty [chunk, {BEHAVIOR_ACTION_DIM}] actions, got {block.shape}"
        )
    if not np.isfinite(block).all():
        raise ValueError("BEHAVIOR actions must be finite")
    return block.copy()


def _to_numpy(value: Any) -> Any:
    detach = getattr(value, "detach", None)
    if callable(detach):
        value = detach()
    cpu = getattr(value, "cpu", None)
    if callable(cpu):
        value = cpu()
    numpy = getattr(value, "numpy", None)
    return numpy() if callable(numpy) else value


def _as_image(value: Any) -> np.ndarray | None:
    """Convert a candidate image to contiguous uint8 HWC RGB."""
    try:
        image = np.asarray(_to_numpy(value))
    except Exception:  # noqa: BLE001 - malformed simulator payload is skipped
        return None
    if image.ndim == 4 and image.shape[0] == 1:
        image = image[0]
    if image.ndim != 3:
        return None
    if image.shape[0] in {1, 3, 4} and image.shape[-1] not in {1, 3, 4}:
        image = np.moveaxis(image, 0, -1)
    if image.shape[-1] == 1:
        image = np.repeat(image, 3, axis=-1)
    elif image.shape[-1] > 3:
        image = image[..., :3]
    if image.shape[-1] != 3:
        return None
    if np.issubdtype(image.dtype, np.floating):
        scale = 255.0 if np.nanmax(image, initial=0.0) <= 1.0 else 1.0
        image = image * scale
    return np.ascontiguousarray(np.clip(image, 0, 255).astype(np.uint8))


def _walk(value: Any, path: str = ""):
    if isinstance(value, Mapping):
        for key, child in value.items():
            key_path = f"{path}.{key}" if path else str(key)
            yield key_path, child
            yield from _walk(child, key_path)
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            key_path = f"{path}[{index}]"
            yield key_path, child
            yield from _walk(child, key_path)


def extract_behavior_payload(
    raw_obs: Any,
    *,
    main_camera: str,
    wrist_cameras: tuple[str, str],
    image_size: int,
    state_dim: int | None = None,
) -> dict[str, Any]:
    """Extract the five-key Runtime payload from common OmniGibson shapes.

    OmniGibson releases have used both named camera mappings and nested robot
    observation dictionaries.  We retain camera names in the contract and use a
    deterministic name/path search, so a release-specific nesting change does not
    silently swap the two wrist views.
    """
    candidates: list[tuple[str, np.ndarray]] = []
    for path, value in _walk(raw_obs):
        image = _as_image(value)
        if image is not None:
            candidates.append((path, image))
    if not candidates:
        raise ValueError("BEHAVIOR observation does not contain an RGB image")

    def pick(name: str, *, fallback_index: int | None = None) -> np.ndarray:
        lowered = name.lower()
        for path, image in candidates:
            if path.lower().split(".")[-1].replace("/", "") == lowered:
                return image
            if lowered in path.lower():
                return image
        if fallback_index is not None and fallback_index < len(candidates):
            return candidates[fallback_index][1]
        return candidates[0][1]

    main = pick(main_camera, fallback_index=0)
    left = pick(wrist_cameras[0], fallback_index=1)
    right = pick(wrist_cameras[1], fallback_index=2)

    state: np.ndarray | None = None
    for path, value in _walk(raw_obs):
        if any(token in path.lower() for token in ("proprio", "proprioception", "joint_position")):
            try:
                candidate = np.asarray(_to_numpy(value), dtype=np.float32).reshape(-1)
            except Exception:  # noqa: BLE001 - try the next candidate
                continue
            if candidate.size >= BEHAVIOR_ACTION_DIM:
                state = candidate
                break
    if state is None:
        for path, value in _walk(raw_obs):
            if "state" in path.lower():
                try:
                    candidate = np.asarray(_to_numpy(value), dtype=np.float32).reshape(-1)
                except Exception:  # noqa: BLE001 - try the next candidate
                    continue
                if candidate.size >= BEHAVIOR_ACTION_DIM:
                    state = candidate
                    break
    if state is None:
        raise ValueError(
            "BEHAVIOR observation does not contain a full R1Pro proprioception vector"
        )
    if state_dim is not None:
        if state_dim > state.size:
            raise ValueError(
                f"requested BEHAVIOR state_dim={state_dim} exceeds proprioception size {state.size}"
            )
        state = state[:state_dim]

    def resize(image: np.ndarray) -> np.ndarray:
        if image.shape[:2] == (image_size, image_size):
            return image
        try:
            from PIL import Image

            return np.asarray(Image.fromarray(image).resize((image_size, image_size)))
        except ImportError:
            # Keep the contract deterministic even in the simulator-free test env.
            y = np.linspace(0, image.shape[0] - 1, image_size).round().astype(int)
            x = np.linspace(0, image.shape[1] - 1, image_size).round().astype(int)
            return np.ascontiguousarray(image[y][:, x])

    return {
        "main_images": resize(main)[None, ...],
        # Runtime's common schema has one dedicated wrist slot plus a list of
        # extra views. Keep the right wrist in that list so policy adapters can
        # reconstruct the ordered two-view tensor without ambiguity.
        "wrist_images": resize(left)[None, ...],
        "extra_view_images": resize(right)[None, None, ...],
        "states": state[None, ...],
        "task_descriptions": [],
    }


__all__ = [
    "ACTION_CONTRACT",
    "BEHAVIOR_ACTION_DIM",
    "DEFAULT_IMAGE_SIZE",
    "OBSERVATION_PROFILE",
    "contract_digest",
    "extract_behavior_payload",
    "observation_contract",
    "validate_behavior_actions",
]
