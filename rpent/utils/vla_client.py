"""Thin client wrapping the Pi0.5 VLA RPC server.

The server lifecycle is the caller's responsibility: bring up
``robots/libero/vla_server.py`` (or any compatible ``predict`` /
``healthz`` implementation) before constructing this client.

Wire schema (see also ``vla_server.py``):

    call("predict", kwargs={
        "instruction": "<task_descriptions>",
        "images": {
            "main":  {"format": "png", "data": "<base64>"},
            "wrist": {"format": "png", "data": "<base64>"},  # optional
            "extra": {"format": "png", "data": "<base64>"},  # optional
        },
        "state": [[s0..sN]],           # shape [B, state_dim]
        "mode":  "eval",
    })
    -> {"actions": [[[a0..a6], ...]], "shape": [B, chunk, action_dim], "dtype": "float32"}
"""

from __future__ import annotations

import base64
import io
from typing import Any

import numpy as np

from rpent.utils.rpc import RpcClient


# Pi0.5 inference can legitimately approach the generic HTTP RPC timeout.
PREDICT_TIMEOUT_S = 120.0


def _prepare_policy_image(img: np.ndarray, *, name: str) -> np.ndarray:
    """Validate and normalize one policy camera image for PNG transport.

    A constant image is never a useful LIBERO policy observation. Treat it as
    an infrastructure failure instead of silently asking the VLA to act on an
    all-black EGL frame.
    """

    arr = np.asarray(img)
    if arr.ndim != 3 or arr.shape[-1] != 3 or arr.size == 0:
        raise ValueError(f"{name} expected non-empty shape [H,W,3]; got {arr.shape}")
    if not np.issubdtype(arr.dtype, np.number):
        raise ValueError(f"{name} must be numeric; got dtype {arr.dtype}")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} contains non-finite pixels")

    source_min = float(arr.min())
    source_max = float(arr.max())
    if np.issubdtype(arr.dtype, np.floating):
        if 0.0 <= source_min and source_max <= 1.0:
            arr = np.rint(arr * 255.0)
        elif not (0.0 <= source_min and source_max <= 255.0):
            raise ValueError(
                f"{name} floating pixels must be in [0,1] or [0,255]; "
                f"got min={source_min:g}, max={source_max:g}"
            )
        else:
            arr = np.rint(arr)
    elif source_min < 0.0 or source_max > 255.0:
        raise ValueError(
            f"{name} integer pixels must be in [0,255]; "
            f"got min={source_min:g}, max={source_max:g}"
        )

    normalized = np.ascontiguousarray(arr, dtype=np.uint8)
    normalized_min = int(normalized.min())
    normalized_max = int(normalized.max())
    if normalized_min == normalized_max:
        raise ValueError(
            f"{name} is constant after uint8 normalization "
            f"(shape={normalized.shape}, value={normalized_min}); "
            "check the camera and EGL device mapping"
        )
    return normalized


def _png_b64(img: np.ndarray, *, name: str = "policy_image") -> str:
    import imageio.v2 as imageio

    arr = _prepare_policy_image(img, name=name)
    buf = io.BytesIO()
    imageio.imwrite(buf, arr, format="png")
    return base64.b64encode(buf.getvalue()).decode("ascii")


class VLAClient:
    """Client wrapping a remote Pi0.5 VLA over any :class:`RpcClient` transport.

    Only call site is ``LiberoPrimitives``, which uses one method:
    ``predict_action_batch(env_obs, mode="eval")``.
    """

    def __init__(self, client: RpcClient):
        self._client = client

    def healthz(self, *, timeout_s: float | None = None) -> dict[str, Any]:
        return self._client.call("healthz", timeout_s=timeout_s)

    def predict_action_batch(
        self,
        env_obs: dict[str, Any],
        mode: str = "eval",
        **kwargs,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        main_images = np.asarray(env_obs["main_images"])
        images: dict[str, Any] = {
            "main": {
                "format": "png",
                "data": _png_b64(main_images, name="main_images"),
            },
        }
        for src_key, wire_key in (
            ("wrist_images", "wrist"),
            ("extra_view_images", "extra"),
        ):
            view = env_obs.get(src_key)
            if view is None:
                continue
            arr = np.asarray(view)
            if arr.size == 0:
                continue
            images[wire_key] = {
                "format": "png",
                "data": _png_b64(arr, name=src_key),
            }

        states = np.asarray(env_obs["states"]).astype(np.float32)
        if states.ndim != 1:
            raise ValueError(
                f"states must be single-env shape [state_dim]; got {states.shape}"
            )

        inference_parameters = kwargs.get("inference_parameters")
        wire_kwargs: dict[str, Any] = {
            "instruction": env_obs.get("task_descriptions") or "",
            "images": images,
            # vla_server's wire still expects [B, state_dim]
            "state": [states.tolist()],
            "mode": mode,
        }
        if inference_parameters:
            if not isinstance(inference_parameters, dict):
                raise ValueError("inference_parameters must be an object")
            wire_kwargs["parameters"] = inference_parameters

        payload = self._client.call(
            "predict",
            kwargs=wire_kwargs,
            timeout_s=PREDICT_TIMEOUT_S,
        )
        # Wire returns [B=1, chunk, action_dim]; strip B so callers see
        # [chunk, action_dim] without thinking in num_envs.
        actions = np.asarray(payload["actions"], dtype=np.float32)[0]
        metadata = payload.get("metadata", {})
        return actions, metadata if isinstance(metadata, dict) else {}
