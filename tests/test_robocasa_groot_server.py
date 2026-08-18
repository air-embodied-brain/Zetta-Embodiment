# Copyright (c) 2026 RPent Contributors
from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np
import pytest

from robots.robocasa.groot_server import (
    Gr00tRuntime,
    _observation_arrays,
    checkpoint_digest,
    parse_inference_seed,
)


class _Policy:
    def get_action(self, observation):
        assert observation["video.left"].dtype == np.uint8
        return {"action.x": np.asarray([[0.25]], dtype=np.float32)}


def _config():
    return SimpleNamespace(
        video_keys=("video.left",),
        state_keys=("state.pose",),
        language_keys=("annotation.human.task_description",),
        observation_indices=(0,),
        action_keys=("action.x",),
        action_indices=(0,),
    )


@pytest.mark.parametrize("value", [True, 1.2, -1, 2**31])
def test_server_rejects_invalid_request_seed(value) -> None:
    with pytest.raises(ValueError):
        parse_inference_seed(value)


def test_checkpoint_digest_changes_with_weight_content(tmp_path) -> None:
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    (tmp_path / "model.safetensors.index.json").write_text("{}", encoding="utf-8")
    shard = tmp_path / "model-00001-of-00001.safetensors"
    shard.write_bytes(b"first")
    first = checkpoint_digest(tmp_path)
    shard.write_bytes(b"second")
    assert checkpoint_digest(tmp_path) != first


def test_runtime_returns_public_metrics_and_schema(monkeypatch) -> None:
    torch_stub = SimpleNamespace(
        manual_seed=lambda _seed: None,
        cuda=SimpleNamespace(
            is_available=lambda: False,
            manual_seed_all=lambda _seed: None,
        ),
    )
    monkeypatch.setitem(sys.modules, "torch", torch_stub)
    runtime = Gr00tRuntime(
        policy=_Policy(),
        data_config=_config(),
        checkpoint_sha256="a" * 64,
        denoising_steps=4,
    )
    actions, metadata = runtime.act(
        {
            "seed": 3,
            "observation": {
                "video.left": np.zeros((1, 4, 5, 3), dtype=np.uint8).tolist(),
                "state.pose": [[0.0, 1.0]],
                "annotation.human.task_description": ["Slide the rack."],
            },
        }
    )
    assert actions == {"action.x": [[0.25]]}
    assert metadata["checkpoint_sha256"] == "a" * 64
    assert metadata["queue_latency_s"] >= 0
    assert runtime.schema["action"]["keys"] == ["action.x"]
    assert "path" not in str(runtime.health).lower()


def test_observation_validation_rejects_non_rgb_or_nonfinite_state() -> None:
    with pytest.raises(ValueError, match="shape"):
        _observation_arrays({"video.left": [[[0]]]})
    with pytest.raises(ValueError, match="non-finite"):
        _observation_arrays({"state.pose": [[float("nan")]]})
