"""Transport-independent client for Zetta's SAM3 segmentation service."""

from __future__ import annotations

import base64
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np

from zetta.utils.rpc import RpcClient


@dataclass(frozen=True)
class Sam3Result:
    """One top-ranked SAM3 segmentation result."""

    found: bool
    score: float | None = None
    box: list[float] | None = None
    mask: np.ndarray | None = None
    mask_shape: tuple[int, int] | None = None
    reason: str | None = None


class Sam3Client:
    """Client wrapping the SAM3 service over any :class:`RpcClient`."""

    def __init__(self, client: RpcClient, *, timeout_s: float = 120.0) -> None:
        self._client = client
        self._timeout_s = timeout_s

    def segment(
        self,
        image_path: str | Path,
        *,
        text_prompt: str | None = None,
        point: list[int] | None = None,
        min_score: float = 0.2,
    ) -> Sam3Result:
        """Segment an image using exactly one text prompt or positive point.

        Point coordinates use Zetta's image convention: ``[row, col]``.
        """
        prompt = text_prompt.strip() if isinstance(text_prompt, str) else None
        has_text = bool(prompt)
        has_point = point is not None
        if has_text == has_point:
            raise ValueError("segment requires exactly one of text_prompt or point")
        if has_point and (not isinstance(point, list) or len(point) != 2):
            raise ValueError("point must be [row, col]")
        if not 0.0 <= float(min_score) <= 1.0:
            raise ValueError("min_score must be between 0 and 1")

        image_bytes = Path(image_path).read_bytes()
        body: dict[str, Any] = {
            "image_base64": base64.b64encode(image_bytes).decode("ascii"),
            "min_score": float(min_score),
        }
        if has_text:
            body["text_prompt"] = prompt
        else:
            body["point"] = [int(point[0]), int(point[1])]

        payload = self._client.call(
            "segment",
            kwargs=body,
            timeout_s=self._timeout_s,
        )
        return self._decode_result(payload)

    @staticmethod
    def _decode_result(payload: Any) -> Sam3Result:
        if not isinstance(payload, dict) or not isinstance(payload.get("found"), bool):
            raise RuntimeError(f"invalid SAM3 segment response: {payload!r}")

        score = payload.get("score")
        if score is not None:
            score = float(score)
        box = payload.get("box")
        if box is not None:
            if not isinstance(box, list) or len(box) != 4:
                raise RuntimeError(f"invalid SAM3 box: {box!r}")
            box = [float(value) for value in box]

        if not payload["found"]:
            return Sam3Result(
                found=False,
                score=score,
                box=box,
                reason=str(payload.get("reason") or "SAM3 found no mask"),
            )

        encoded_mask = payload.get("mask_png_base64")
        shape = payload.get("mask_shape")
        if not isinstance(encoded_mask, str) or not encoded_mask:
            raise RuntimeError("SAM3 response marked found but omitted mask_png_base64")
        if (
            not isinstance(shape, list)
            or len(shape) != 2
            or not all(isinstance(value, int) and value > 0 for value in shape)
        ):
            raise RuntimeError(f"invalid SAM3 mask_shape: {shape!r}")

        try:
            raw = base64.b64decode(encoded_mask, validate=True)
            decoded = np.asarray(imageio.imread(io.BytesIO(raw)))
        except Exception as exc:
            raise RuntimeError(f"could not decode SAM3 PNG mask: {exc}") from exc
        if decoded.ndim == 3:
            decoded = decoded[..., 0]
        expected_shape = (shape[0], shape[1])
        if decoded.shape != expected_shape:
            raise RuntimeError(
                "SAM3 mask shape mismatch: "
                f"response={expected_shape}, decoded={decoded.shape}"
            )

        mask = decoded > 0
        return Sam3Result(
            found=True,
            score=score,
            box=box,
            mask=mask,
            mask_shape=expected_shape,
        )


class UnavailableSam3Client:
    """Deterministic no-motion fallback when gated SAM3 weights are unavailable."""

    def __init__(self, reason: str) -> None:
        self._reason = str(reason)

    def segment(
        self,
        image_path: str | Path,
        *,
        text_prompt: str | None = None,
        point: list[int] | None = None,
        min_score: float = 0.2,
    ) -> Sam3Result:
        # Retain the same argument validation as the real client so disabling
        # the service does not hide malformed planner calls.
        prompt = text_prompt.strip() if isinstance(text_prompt, str) else None
        if bool(prompt) == (point is not None):
            raise ValueError("segment requires exactly one of text_prompt or point")
        if not 0.0 <= float(min_score) <= 1.0:
            raise ValueError("min_score must be between 0 and 1")
        return Sam3Result(found=False, reason=self._reason)
