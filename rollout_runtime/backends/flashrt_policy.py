# Copyright (c) 2026 Zetta Contributors
"""``PolicyInferenceCore`` for FlashRT (https://github.com/flashrt-project/FlashRT).

FlashRT is a standalone, pip-installable (from source; see ``load()``)
realtime inference engine for small-batch VLA control (Pi0 / Pi0.5 /
Pi0-FAST / GROOT N1.6 / N1.7). Unlike ``groot_policy.py`` /
``robocasa_current.py``, this backend has **no application-layer coupling**:
it only ever imports the third-party ``flash_rt`` package, never
``robots``/``zetta`` (see ``docs/architecture.md`` § Known tech debt --
this module is deliberately not on that list).

This module does not reimplement any inference logic. It does two things
only: call ``flash_rt.load_model(...)`` once during ``load()`` (weights +
kernel calibration + CUDA Graph capture happen inside FlashRT, ~3s on the
first call, sub-frame graph replay after), and translate the Runtime's
``InferenceRequest``/``Observation`` into the ``images``/``prompt``/``state``
keyword arguments expected by ``FlashRtModel.predict`` and back into an
``ActionResponse``.

Key divergences (same shape as ``groot_policy.py``):

1. **FlashRT's documented ``predict`` call takes one observation at a
   time** (``model.predict(images=[...], prompt=..., state=...)``); there is
   no batched-observation entry point in the *public* ``load_model()``
   API. ``infer_batch`` therefore loops over ``predict`` per request by
   default, mirroring the GR00T backend's divergence 1. When
   ``config.enable_batched_infer`` is set, this backend instead drives the
   frontend's own batched primitives (``set_batched_mode`` /
   ``set_prompt_batch`` / ``calibrate_batch`` / ``infer_batch``, exposed
   today by the Pi0.5 ``pi05_rtx`` torch frontend) through the
   ``FlashRtModel.pipeline`` escape hatch, replaying one CUDA graph per
   scheduler batch instead of one per request -- see
   ``_supports_batched_predict`` / ``_infer_batched``. Any request count
   the batched pipeline does not support (batch of 1, or a frontend
   without these primitives) still falls back to the per-request loop.
2. **State is only sent for configs that consume it**: Pi0 and GROOT read a
   continuous proprioceptive ``state`` vector; Pi0.5 and Pi0-FAST encode
   state (if any) in the prompt prefix and do not take a ``state=`` kwarg.
   ``_STATE_CONSUMING_CONFIGS`` records this so the backend never sends an
   argument FlashRT's frontend does not accept.
3. **Images come from ``PayloadRef``**: ``Observation.main_image`` /
   ``wrist_image`` / ``extra_view_images`` are decoded via
   ``payload_module.decode_payload`` into uint8 ``[H, W, C]`` arrays before
   being passed as the ``images=[...]`` list, main view first.
4. **Images are resized to FlashRT's fixed ``224x224`` frontend buffer
   before ``predict()``**: the env renders LIBERO cameras at
   ``camera_height``/``camera_width`` (256x256 in the LIBERO-PRO presets),
   the same raw resolution ``zetta_openpi`` (``rlinf_policy.py`` ->
   ``build_openpi_model``) receives -- that path resizes internally via
   ``openpi.shared.image_tools.resize_with_pad_torch`` to
   ``IMAGE_RESOLUTION=(224, 224)`` right before the vision tower.
   FlashRT's ``pi05_rtx`` frontend has no such internal step (its
   ``_stack_images``/``_fill_img_buf`` copy straight into a fixed
   ``(num_views, 224, 224, 3)`` CUDA buffer), so skipping the resize here
   either crashes on a shape mismatch or silently feeds a wrong-resolution
   frame into the buffer. ``_resize_with_pad_uint8`` replicates
   ``resize_with_pad_torch``'s exact algorithm (bilinear, ``align_corners=
   False``, round + clamp back to uint8, zero-pad any residual aspect-ratio
   slack) so both backends' models see bit-for-bit the same 224x224 input
   for the square LIBERO camera frames used here.

Dependency surface: depends only on the third-party ``flash_rt`` package
(numpy + torch or jax, per ``framework``), lazily imported inside ``load()``
so importing this module never requires FlashRT to be installed. The image
resize helper also lazily imports ``torch`` (only reached when
``framework="torch"``, which is the only framework this module has been
exercised with) so the module remains importable without either dependency.
"""

from __future__ import annotations

import logging
import dataclasses
from collections.abc import Mapping
from typing import Any

import numpy as np

from rollout_runtime.api.enums import ErrorCode
from rollout_runtime.api.errors import make_error
from rollout_runtime.api.internal import ActionResponse, InferenceRequest
from rollout_runtime.core import payload as payload_module

__all__ = ["FLASHRT_POLICY_FAMILY", "FlashRtPolicyConfig", "FlashRtPolicyCore"]

_IMG_HW = 224
"""FlashRT's fixed frontend image buffer size (``flash_rt.frontends.torch.
pi05_rtx.IMG_HW``); duplicated here (rather than imported) to keep this
module importable without FlashRT installed."""


def _resize_with_pad_uint8(image: np.ndarray, height: int = _IMG_HW, width: int = _IMG_HW) -> np.ndarray:
    """Resize a uint8 ``[H, W, C]`` image to ``height``x``width`` without
    distortion, replicating ``openpi.shared.image_tools.resize_with_pad_torch``
    bit-for-bit (bilinear, ``align_corners=False``, round + clamp, zero-pad
    any leftover aspect-ratio slack) so FlashRT and ``zetta_openpi`` feed the
    model the same pixels for the same input frame.

    Args:
        image: uint8 array, ``[H, W, C]``.
        height: Target height (FlashRT's fixed frontend buffer height).
        width: Target width (FlashRT's fixed frontend buffer width).

    Returns:
        uint8 array, ``[height, width, C]``.
    """
    import torch
    import torch.nn.functional as F  # noqa: N812

    cur_height, cur_width = image.shape[0], image.shape[1]
    if cur_height == height and cur_width == width:
        return image
    tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)  # [1, C, H, W]
    ratio = max(cur_width / width, cur_height / height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)
    resized = F.interpolate(
        tensor.to(torch.float32), size=(resized_height, resized_width), mode="bilinear", align_corners=False
    )
    resized = torch.round(resized).clamp(0, 255).to(torch.uint8)
    pad_h0, remainder_h = divmod(height - resized_height, 2)
    pad_h1 = pad_h0 + remainder_h
    pad_w0, remainder_w = divmod(width - resized_width, 2)
    pad_w1 = pad_w0 + remainder_w
    padded = F.pad(resized, (pad_w0, pad_w1, pad_h0, pad_h1), value=0)
    return padded.squeeze(0).permute(1, 2, 0).contiguous().numpy()


FLASHRT_POLICY_FAMILY = "flashrt"
"""The family name corresponding to ``core.policy_inference.BATCHABLE_PARAM_KEYS``.

Only ``noise_seed`` is mixable (same posture as ``pi0``/``pi05``): FlashRT's
CUDA Graph capture bakes in the model config (``pi0``/``pi05``/``groot``/
``groot_n17``/``pi0fast``) and checkpoint at ``load()`` time, so those never
vary within one core's lifetime and need no ``compat_key`` entry beyond the
default ``policy_id``/``model_version`` fields already covered there.
"""

_BATCHED_PIPELINE_ATTRS = (
    "set_batched_mode",
    "select_batch_size",
    "set_prompt_batch",
    "calibrate_batch",
    "infer_batch",
)
"""Frontend attributes ``_supports_batched_predict`` requires. Present on
FlashRT's Pi0.5 ``pi05_rtx`` torch frontend today; a frontend missing any of
these falls back to the per-request ``predict()`` loop."""

_STATE_CONSUMING_CONFIGS = frozenset({"pi0", "groot", "groot_n17"})
"""``config`` values whose FlashRT frontend accepts a ``state=`` kwarg.

Pi0.5 (``pi05``) and Pi0-FAST (``pi0fast``) encode state in the prompt
prefix (or omit it) and do not take this argument; passing it would raise a
``TypeError`` inside ``flash_rt``.
"""


@dataclasses.dataclass(kw_only=True)
class FlashRtPolicyConfig:
    """Private configuration for the FlashRT policy backend.

    Fields correspond one-to-one with the keyword arguments of
    ``flash_rt.load_model``, plus reporting-only fields for ``compat_key``.

    Attributes:
        checkpoint: Checkpoint directory (safetensors for ``framework="torch"``,
            Orbax for ``framework="jax"``).
        config: Model variant: ``"pi0"``, ``"pi05"``, ``"groot"``,
            ``"groot_n17"``, or ``"pi0fast"``.
        framework: ``"torch"`` or ``"jax"``; both share one kernel binary.
        autotune: Kernel autotune level (``0``=off, ``3``=default,
            ``5``=thorough).
        decode_cuda_graph: Whether to capture the Pi0-FAST decode loop as a
            CUDA Graph (max-performance mode for fixed-prompt deployments).
        decode_graph_steps: Action tokens captured per inference when
            ``decode_cuda_graph`` is set.
        action_dim: Reported action dimensionality, used only for
            diagnostics via ``compat_key`` (FlashRT itself determines the
            real output shape per ``config``: e.g. ``(10, 7)`` for
            Pi0/Pi0.5, ``(50, 128)`` for GROOT).
        model_version: Reported model version; ``None`` falls back to
            ``config``.
        device: Device identifier, goes into ``compat_key``. FlashRT
            auto-dispatches to the best-matching hardware backend by
            inspecting ``torch.cuda.get_device_capability()`` at
            ``load_model`` time, so this field is reporting-only.
        dtype: Compute precision name, goes into ``compat_key``.
        calibration_data_path: Path to an ``.npz`` file with a single real
            observation (keys: ``main_image`` uint8 ``[H, W, C]``,
            optionally ``wrist_image``/``extra_view_images`` in the same
            layout, ``state`` float array, ``instruction`` string) captured
            from a real LIBERO ``reset()`` -- see
            ``scripts/profile/capture_libero_calibration_frame.py``.
            ``load()`` resizes and normalizes it exactly like a real
            inference request and feeds it through ``model.predict()``
            once, so FlashRT's FP8 calibration statistics come from a real
            camera frame instead of ``flash_rt``'s implicit fallback (the
            first ``predict()`` call auto-calibrates on whatever it is
            given, which would otherwise be the first live request).
            ``None`` skips this warmup: FlashRT still auto-calibrates on
            the first live ``infer_batch`` call, silently absorbing the
            calibration overhead into that request's latency instead of
            ``load()``'s.
        enable_batched_infer: Opt-in (default ``False``, preserving the
            existing per-request-loop behavior). When set and the loaded
            frontend exposes the batched primitives listed in
            ``_BATCHED_PIPELINE_ATTRS``, ``infer_batch`` replays one CUDA
            graph for the whole scheduler batch instead of one per
            request, at the cost of a one-time (~3s) calibration the first
            time a given batch width is seen (``scheduler.max_batch_size``
            should be paired with this so the width stabilizes quickly).
            Batches whose size the pipeline cannot serve this way (e.g. a
            singleton batch, or a frontend missing these primitives) still
            go through the per-request loop.
    """

    checkpoint: str
    config: str = "pi05"
    framework: str = "torch"
    autotune: int = 3
    decode_cuda_graph: bool = False
    decode_graph_steps: int = 46
    action_dim: int = 7
    model_version: str | None = None
    device: str = "cuda"
    dtype: str = "bfloat16"
    calibration_data_path: str | None = None
    enable_batched_infer: bool = False

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any] | None) -> FlashRtPolicyConfig:
        """Construct configuration from a ``policy_config`` dict.

        Args:
            config: Family-private configuration.

        Returns:
            Structured configuration.

        Raises:
            ValueError: Missing required ``checkpoint``, or an unknown key
                is present.
        """
        payload = dict(config or {})
        if "checkpoint" not in payload:
            raise ValueError("flashrt policy_config missing required key: 'checkpoint'")
        known = {field.name for field in dataclasses.fields(cls)}
        unknown = sorted(set(payload) - known)
        if unknown:
            raise ValueError(f"unknown flashrt policy_config keys: {unknown}")
        return cls(**payload)


class FlashRtPolicyCore:
    """Inference core for FlashRT (synchronous, blocking; same posture as
    ``GrootPolicyCore`` / ``RlinfPolicyCore``).

    FlashRT's ``predict`` is a blocking call (CUDA Graph replay), so this
    core provides no ``ainfer_batch`` -- ``RuntimeRolloutWorker`` calls
    ``infer_batch`` via ``asyncio.to_thread``.
    """

    def __init__(self, config: FlashRtPolicyConfig) -> None:
        """Initialize a not-yet-``load``ed inference core.

        Args:
            config: FlashRT configuration.
        """
        self.config = config
        self.model: Any = None
        self.loaded = False
        self.closed = False
        self.batch_calls = 0
        self.request_count = 0
        self.error_count = 0
        self._model_version = config.model_version or config.config

    # ------------------------------------------------------------ Protocol attributes

    @property
    def model_version(self) -> str:
        """The current model version.

        Returns:
            The version identifier: the explicitly configured value,
            otherwise ``config.config`` (e.g. ``"pi05"``).
        """
        return self._model_version

    @property
    def device(self) -> str:
        """The device the model resides on.

        Returns:
            Device identifier.
        """
        return self.config.device

    @property
    def dtype(self) -> str:
        """The model's compute precision.

        Returns:
            Dtype name.
        """
        return self.config.dtype

    @property
    def policy_family(self) -> str:
        """The policy family name.

        Returns:
            ``"flashrt"``, determines the ``compat_key`` batchable whitelist.
        """
        return FLASHRT_POLICY_FAMILY

    # ---------------------------------------------------------------- Lifecycle

    def load(self) -> None:
        """Load the FlashRT model once (kernel calibration + CUDA Graph
        capture happen inside ``flash_rt.load_model``; ~3s on the first
        call, sub-frame graph replay after).

        When ``config.calibration_data_path`` is set, immediately follows
        with one real ``model.predict()`` call built from that real
        observation (same resize/normalize path as ``infer_batch``) so
        FlashRT's FP8 calibration is driven by a real camera frame rather
        than the first live request.

        Raises:
            ImportError: The ``flash_rt`` package is not installed. FlashRT
                ships no PyPI wheel: build it from source (``cmake ..
                && make -j``) per
                https://github.com/flashrt-project/FlashRT#build--install,
                then ``pip install -e .`` its checkout.
        """
        import flash_rt

        self.model = flash_rt.load_model(
            checkpoint=self.config.checkpoint,
            config=self.config.config,
            framework=self.config.framework,
            autotune=self.config.autotune,
            **(
                {
                    "decode_cuda_graph": self.config.decode_cuda_graph,
                    "decode_graph_steps": self.config.decode_graph_steps,
                }
                if self.config.config == "pi0fast"
                else {}
            ),
        )
        if self.config.calibration_data_path:
            self.model.predict(**self._load_calibration_kwargs(self.config.calibration_data_path))
        self.loaded = True

    def _load_calibration_kwargs(self, path: str) -> dict[str, Any]:
        """Build ``model.predict`` kwargs from a real-observation ``.npz``
        (see ``FlashRtPolicyConfig.calibration_data_path``).

        Args:
            path: Path to the ``.npz`` calibration sample.

        Returns:
            The same ``{"images": [...], "prompt": str, ...}`` shape
            ``_build_kwargs`` produces for a live request, so calibration
            exercises exactly the code path ``infer_batch`` will use.
        """
        with np.load(path, allow_pickle=True) as data:
            images = [np.asarray(data["main_image"], dtype=np.uint8)]
            if "wrist_image" in data:
                images.append(np.asarray(data["wrist_image"], dtype=np.uint8))
            if "extra_view_images" in data:
                extra = data["extra_view_images"]
                images.extend(np.asarray(view, dtype=np.uint8) for view in extra)
            instruction = str(data["instruction"])
            state = data["state"] if "state" in data else None
        kwargs: dict[str, Any] = {
            "images": [_resize_with_pad_uint8(image) for image in images],
            "prompt": instruction,
        }
        if state is not None and self.config.config in _STATE_CONSUMING_CONFIGS:
            kwargs["state"] = list(np.asarray(state))
        return kwargs

    def update_weights(self, model_version: str) -> None:
        """Record the expected model version label (FlashRT's CUDA Graph
        capture is tied to the checkpoint loaded at ``load()`` time; hot
        weight-swapping is not supported).

        Args:
            model_version: Target version label, used only for reporting;
                it does not trigger a checkpoint reload.
        """
        self._model_version = model_version

    def close(self) -> None:
        """Release resources (drops the reference; ``flash_rt`` has no
        explicit close)."""
        self.model = None
        self.closed = True

    # ------------------------------------------------------------------ Inference

    def infer_batch(self, requests: list[InferenceRequest]) -> list[ActionResponse]:
        """Dispatch a scheduler batch to FlashRT.

        When ``config.enable_batched_infer`` is set and the batch has more
        than one request and the loaded frontend supports the batched
        primitives (see ``_supports_batched_predict``), this replays one
        CUDA graph for the whole batch via ``_infer_batched``. Otherwise
        (the common case today) it calls ``model.predict`` once per
        request (divergence 1: FlashRT's public ``load_model()`` API has
        no batched-observation entry point).

        Args:
            requests: List of requests sharing a ``compat_key``.

        Returns:
            Per-request responses in the same order as the input; a single
            request's failure only affects that request (D5), and this
            method never raises.
        """
        if not requests:
            return []
        self.batch_calls += 1
        self.request_count += len(requests)
        if (
            self.config.enable_batched_infer
            and len(requests) > 1
            and self._supports_batched_predict()
        ):
            try:
                return self._infer_batched(requests)
            except BaseException:  # noqa: BLE001 - batch-level failure: fall back to per-request loop
                logging.getLogger(__name__).warning(
                    "batched FlashRT inference failed; falling back to per-request loop",
                    exc_info=True,
                )
        return [self._infer_one(request) for request in requests]

    def _supports_batched_predict(self) -> bool:
        """Whether the loaded frontend exposes FlashRT's batched-inference
        primitives (``_BATCHED_PIPELINE_ATTRS``) via ``model.pipeline``.

        Returns:
            ``True`` for frontends with real B=N CUDA-graph batching (the
            Pi0.5 ``pi05_rtx`` torch frontend today); ``False`` otherwise,
            in which case ``infer_batch`` uses the per-request loop.
        """
        pipeline = getattr(self.model, "pipeline", None)
        if pipeline is None:
            return False
        return all(hasattr(pipeline, attr) for attr in _BATCHED_PIPELINE_ATTRS)

    def _infer_batched(self, requests: list[InferenceRequest]) -> list[ActionResponse]:
        """Run one batched ``predict`` for the whole scheduler batch.

        Drives the frontend's batched primitives directly through
        ``model.pipeline`` (documented as an escape hatch for exactly this
        kind of harness in ``VLAModel.pipeline``'s docstring):
        ``select_batch_size`` picks (or, first time for this width,
        builds) a batched pipeline of width ``len(requests)``,
        ``set_prompt_batch`` uploads per-slot prompts, ``calibrate_batch``
        runs once per width (first time it is built -- subsequent calls at
        an already-``calibrated`` width are near-free), and ``infer_batch``
        replays the batch's CUDA graph once.

        Args:
            requests: List of requests sharing a ``compat_key``; becomes
                the batched pipeline's width for this call.

        Returns:
            Per-request responses in the same order as ``requests``; a
            single request's non-finite output only fails that request
            (D5) even though the underlying compute already ran for the
            whole batch.

        Raises:
            BaseException: Any failure in building kwargs or driving the
                batched pipeline (width selection, prompt upload,
                calibration, or the graph replay itself) -- the caller
                (``infer_batch``) catches this and falls back to the
                per-request loop for this call.
        """
        pipeline = self.model.pipeline
        kwargs_list = [self._build_kwargs(request) for request in requests]
        pipeline.select_batch_size(len(requests))
        pipeline.set_prompt_batch([kwargs["prompt"] for kwargs in kwargs_list])
        observations = [self._kwargs_to_observation(kwargs) for kwargs in kwargs_list]
        if not getattr(pipeline, "calibrated", False):
            pipeline.calibrate_batch([observations[0]])
        results = pipeline.infer_batch(observations)
        return [
            self._response_from_actions(request, entry["actions"])
            for request, entry in zip(requests, results)
        ]

    @staticmethod
    def _kwargs_to_observation(kwargs: dict[str, Any]) -> dict[str, Any]:
        """Translate ``_build_kwargs``' output into the observation dict
        shape ``pipeline.infer_batch`` expects (``images`` and, for
        state-consuming configs, ``state``).

        Args:
            kwargs: Output of ``_build_kwargs``.

        Returns:
            ``{"images": [...], "state": [...]}`` (``state`` omitted when
            absent from ``kwargs``).
        """
        observation: dict[str, Any] = {"images": kwargs["images"]}
        if "state" in kwargs:
            observation["state"] = kwargs["state"]
        return observation

    def _infer_one(self, request: InferenceRequest) -> ActionResponse:
        try:
            kwargs = self._build_kwargs(request)
            actions = self.model.predict(**kwargs)
        except BaseException as exc:  # noqa: BLE001 - D5: per-request normalization, never leak
            return self._error_response(request, exc)
        return self._response_from_actions(request, actions)

    def _response_from_actions(self, request: InferenceRequest, actions: Any) -> ActionResponse:
        """Build a success or per-request error ``ActionResponse`` from a
        raw actions array (D5: never raises, one bad array only fails its
        own request).

        Args:
            request: The request the actions correspond to.
            actions: Raw actions returned by FlashRT for this request.

        Returns:
            A success ``ActionResponse``, or an error one if ``actions``
            is non-finite.
        """
        try:
            block = np.asarray(actions, dtype=np.float32)
            if not np.isfinite(block).all():
                raise ValueError("flashrt policy produced non-finite actions")
        except BaseException as exc:  # noqa: BLE001 - D5: per-request normalization, never leak
            return self._error_response(request, exc)
        return ActionResponse(
            request_id=request.request_id,
            session_id=request.session_id,
            binding_token=request.binding_token,
            episode_id=request.episode_id,
            operation_seq=request.operation_seq,
            actions=payload_module.encode_array(block),
            model_version=self._model_version,
            auxiliary_outputs={
                "chunk": int(block.shape[0]),
                "compat_key": request.compat_key,
            },
        )

    def _error_response(self, request: InferenceRequest, exc: BaseException) -> ActionResponse:
        """Build the per-request error ``ActionResponse`` for a failed
        inference (D5: never raises out of ``infer_batch``).

        Args:
            request: The request that failed.
            exc: The exception raised while producing its actions.

        Returns:
            An ``ActionResponse`` carrying a ``POLICY_FAILURE`` error.
        """
        self.error_count += 1
        return ActionResponse(
            request_id=request.request_id,
            session_id=request.session_id,
            binding_token=request.binding_token,
            episode_id=request.episode_id,
            operation_seq=request.operation_seq,
            model_version=self._model_version,
            error=make_error(
                ErrorCode.POLICY_FAILURE,
                f"flashrt inference failed: {exc}",
                policy_id=request.policy_id,
                session_id=request.session_id,
            ),
        )

    def _build_kwargs(self, request: InferenceRequest) -> dict[str, Any]:
        """Translate an ``InferenceRequest`` into ``model.predict`` kwargs.

        Args:
            request: Inference request.

        Returns:
            ``{"images": [...], "prompt": str, "state": [...]}`` (``state``
            omitted for configs that do not consume it -- see
            ``_STATE_CONSUMING_CONFIGS``); every image is resized to
            FlashRT's fixed 224x224 buffer via ``_resize_with_pad_uint8``
            (divergence 4 in the module docstring).

        Raises:
            ValueError: The observation is missing the main image.
        """
        observation = request.observation
        if observation.main_image is None:
            raise ValueError("observation is missing the required main_image")
        images = [payload_module.decode_payload(observation.main_image)]
        if observation.wrist_image is not None:
            images.append(payload_module.decode_payload(observation.wrist_image))
        images.extend(
            payload_module.decode_payload(ref) for ref in observation.extra_view_images
        )
        images = [_resize_with_pad_uint8(image) for image in images]
        instruction = (
            request.instruction_override
            if request.instruction_override is not None
            else observation.instruction
        )
        kwargs: dict[str, Any] = {"images": images, "prompt": instruction}
        if self.config.config in _STATE_CONSUMING_CONFIGS:
            kwargs["state"] = list(observation.state)
        return kwargs
