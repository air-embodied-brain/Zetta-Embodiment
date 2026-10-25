# Copyright (c) 2026 Zetta Contributors
"""``FlashRtPolicyCore.infer_batch``'s batched-CUDA-graph dispatch path.

``flash_rt`` itself is a GPU-only third-party package (see
``FlashRtPolicyCore.load``'s docstring), so it cannot be exercised in the
simulator-free CI suite. What *can* be pinned without it is the dispatch
logic added for the "FlashRT batched inference" roadmap item: whether
``infer_batch`` drives ``model.pipeline``'s batched primitives
(``select_batch_size`` / ``set_prompt_batch`` / ``calibrate_batch`` /
``infer_batch``) or falls back to the per-request ``model.predict()`` loop,
and that a single request's failure inside a batched call only fails that
request (D5). ``_StubPipeline`` / ``_StubModel`` stand in for
``flash_rt.load_model()``'s return value.
"""

from __future__ import annotations

import numpy as np
import pytest

from rollout_runtime.api.internal import InferenceRequest
from rollout_runtime.api.messages import Observation
from rollout_runtime.backends.flashrt_policy import FlashRtPolicyConfig, FlashRtPolicyCore
from rollout_runtime.core import payload as payload_module

ACTION_DIM = 7
CHUNK = 10


class _StubPipeline:
    """Stands in for ``pi05_rtx``'s batched primitives."""

    def __init__(self) -> None:
        self.calibrated = False
        self.batch_size: int | None = None
        self.select_batch_size_calls: list[int] = []
        self.prompt_batches: list[list[str]] = []
        self.calibrate_calls = 0
        self.last_observations: list[dict] | None = None

    def set_batched_mode(self, *, enable: bool = True, batch_size: int = 1) -> None:
        """Real frontends expose this; ``select_batch_size`` is our only
        caller, but ``_supports_batched_predict`` still probes for it."""

    def select_batch_size(self, batch_size: int) -> None:
        self.select_batch_size_calls.append(batch_size)
        if self.batch_size != batch_size:
            self.calibrated = False
        self.batch_size = batch_size

    def set_prompt_batch(self, prompts: list) -> None:
        self.prompt_batches.append(list(prompts))

    def calibrate_batch(self, sample_observations: list) -> None:
        self.calibrate_calls += 1
        self.calibrated = True

    def infer_batch(self, observations: list) -> list[dict]:
        self.last_observations = observations
        return [
            {"actions": np.ones((CHUNK, ACTION_DIM), dtype=np.float32) * (i + 1)}
            for i in range(len(observations))
        ]


class _StubModel:
    """Stands in for ``flash_rt.load_model()``'s return value."""

    def __init__(self, pipeline: object | None) -> None:
        self.pipeline = pipeline
        self.predict_calls: list[dict] = []

    def predict(self, **kwargs) -> np.ndarray:
        self.predict_calls.append(kwargs)
        return np.zeros((CHUNK, ACTION_DIM), dtype=np.float32)


def _make_core(*, enable_batched_infer: bool, pipeline: object | None) -> FlashRtPolicyCore:
    config = FlashRtPolicyConfig(checkpoint="/fake/ckpt", enable_batched_infer=enable_batched_infer)
    core = FlashRtPolicyCore(config)
    core.model = _StubModel(pipeline)
    core.loaded = True
    return core


def _request(index: int) -> InferenceRequest:
    image = np.zeros((224, 224, 3), dtype=np.uint8)
    observation = Observation(
        session_id=f"sess-{index}",
        episode_id=f"ep-{index}",
        step_index=0,
        main_image=payload_module.encode_payload(image),
        instruction=f"do task {index}",
    )
    return InferenceRequest(
        request_id=f"req-{index}",
        session_id=f"sess-{index}",
        policy_id="flashrt",
        observation=observation,
        routing_token="route",
        compat_key="k",
    )


def test_batched_path_used_when_enabled_and_multiple_requests() -> None:
    """A >1-request batch, with a batching-capable pipeline and the opt-in
    flag set, goes through ``select_batch_size`` / ``set_prompt_batch`` /
    ``calibrate_batch`` / ``infer_batch`` instead of ``model.predict``."""
    pipeline = _StubPipeline()
    core = _make_core(enable_batched_infer=True, pipeline=pipeline)
    requests = [_request(0), _request(1), _request(2)]

    responses = core.infer_batch(requests)

    assert pipeline.select_batch_size_calls == [3]
    assert pipeline.prompt_batches == [["do task 0", "do task 1", "do task 2"]]
    assert pipeline.calibrate_calls == 1  # first time this width is seen
    assert core.model.predict_calls == []  # per-request loop was not used
    assert len(responses) == 3
    for i, response in enumerate(responses):
        assert response.error is None
        actions = payload_module.decode_payload(response.actions)
        assert np.allclose(actions, (i + 1))


def test_batched_path_reuses_calibration_for_a_stable_width() -> None:
    """A second batch at the same width skips ``calibrate_batch`` again."""
    pipeline = _StubPipeline()
    core = _make_core(enable_batched_infer=True, pipeline=pipeline)
    core.infer_batch([_request(0), _request(1)])
    core.infer_batch([_request(2), _request(3)])

    assert pipeline.select_batch_size_calls == [2, 2]
    assert pipeline.calibrate_calls == 1


def test_singleton_batch_never_uses_the_batched_path() -> None:
    """A batch of one request always falls back to ``model.predict``, even
    with the batched pipeline available and the flag enabled (mirrors the
    existing single-request-loop behavior exactly)."""
    pipeline = _StubPipeline()
    core = _make_core(enable_batched_infer=True, pipeline=pipeline)

    responses = core.infer_batch([_request(0)])

    assert pipeline.select_batch_size_calls == []
    assert len(core.model.predict_calls) == 1
    assert responses[0].error is None


def test_batched_path_disabled_by_default() -> None:
    """Without the opt-in flag, ``infer_batch`` still loops over
    ``model.predict`` even for a multi-request batch and a
    batching-capable pipeline (preserves existing default behavior)."""
    pipeline = _StubPipeline()
    core = _make_core(enable_batched_infer=False, pipeline=pipeline)

    responses = core.infer_batch([_request(0), _request(1)])

    assert pipeline.select_batch_size_calls == []
    assert len(core.model.predict_calls) == 2
    assert len(responses) == 2


def test_batched_path_falls_back_when_pipeline_lacks_batched_primitives() -> None:
    """A pipeline without the batched primitives (e.g. a non-pi05 frontend)
    is rejected by ``_supports_batched_predict`` and the per-request loop
    is used instead, even with the flag enabled."""

    class _BareObject:
        pass

    core = _make_core(enable_batched_infer=True, pipeline=_BareObject())

    responses = core.infer_batch([_request(0), _request(1)])

    assert len(core.model.predict_calls) == 2
    assert len(responses) == 2


def test_batch_level_failure_falls_back_to_per_request_loop() -> None:
    """If the batched pipeline raises (e.g. mid-flight CUDA error), the
    whole batch falls back to the per-request loop rather than losing all
    responses (D5: a batch-level failure must not become a full outage)."""

    class _RaisingPipeline(_StubPipeline):
        def infer_batch(self, observations: list) -> list[dict]:
            raise RuntimeError("simulated CUDA graph replay failure")

    core = _make_core(enable_batched_infer=True, pipeline=_RaisingPipeline())

    responses = core.infer_batch([_request(0), _request(1)])

    assert len(core.model.predict_calls) == 2
    assert len(responses) == 2
    assert all(response.error is None for response in responses)


def test_one_non_finite_response_does_not_fail_the_whole_batch() -> None:
    """D5 inside the batched path: one request's non-finite actions become
    an error response for that request only; the rest still succeed."""

    class _PartlyBrokenPipeline(_StubPipeline):
        def infer_batch(self, observations: list) -> list[dict]:
            self.last_observations = observations
            return [
                {"actions": np.zeros((CHUNK, ACTION_DIM), dtype=np.float32)},
                {"actions": np.full((CHUNK, ACTION_DIM), np.nan, dtype=np.float32)},
            ]

    core = _make_core(enable_batched_infer=True, pipeline=_PartlyBrokenPipeline())

    responses = core.infer_batch([_request(0), _request(1)])

    assert responses[0].error is None
    assert responses[1].error is not None
    assert "flashrt inference failed" in responses[1].error.message
