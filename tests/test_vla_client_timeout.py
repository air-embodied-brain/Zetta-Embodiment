from __future__ import annotations

import numpy as np
import pytest

from rpent.utils.vla_client import PREDICT_TIMEOUT_S, VLAClient, _prepare_policy_image


class _RecordingClient:
    def __init__(self) -> None:
        self.timeout_s = None

    def call(self, method, args=(), kwargs=None, *, timeout_s=None):
        assert method == "predict"
        self.timeout_s = timeout_s
        return {
            "actions": np.zeros((1, 5, 7), dtype=np.float32).tolist(),
            "metadata": {},
        }


def test_predict_uses_inference_specific_timeout():
    transport = _RecordingClient()
    client = VLAClient(transport)
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    image[0, 0] = (1, 2, 3)
    actions, metadata = client.predict_action_batch(
        {
            "main_images": image,
            "states": np.zeros(8, dtype=np.float32),
            "task_descriptions": "test",
        }
    )
    assert transport.timeout_s == PREDICT_TIMEOUT_S
    assert PREDICT_TIMEOUT_S > 30.0
    assert actions.shape == (5, 7)
    assert metadata == {}


def test_policy_image_rejects_constant_egl_frame():
    with pytest.raises(ValueError, match="constant.*EGL device mapping"):
        _prepare_policy_image(
            np.zeros((8, 8, 3), dtype=np.uint8),
            name="main_images",
        )


def test_policy_image_normalizes_unit_float_pixels():
    image = np.zeros((2, 2, 3), dtype=np.float32)
    image[0, 0] = (0.25, 0.5, 1.0)

    normalized = _prepare_policy_image(image, name="main_images")

    assert normalized.dtype == np.uint8
    assert normalized.flags.c_contiguous
    assert normalized[0, 0].tolist() == [64, 128, 255]
