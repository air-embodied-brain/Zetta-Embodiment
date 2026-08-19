"""``PolicyInferenceCore`` for the GR00T family.

**Does not reimplement GR00T inference logic**: ``robots.robocasa.groot_core.Gr00tModelCore``
is reused as-is -- it already implements "load checkpoint in-process (with
hash verification) + inference" (``groot_server.py`` just deploys it as a
standalone process). This module does two things only: call
``groot_core.load_groot_model_core`` during ``load()`` to load the weights
once (one resident copy per RolloutWorker rank, no longer needing a
separately operated ``groot_server.py`` process), and translate the
Runtime's ``InferenceRequest``/``Observation`` into the named dict payload
expected by ``Gr00tModelCore.act`` and back.

Key divergences:

1. **``Gr00tModelCore.act`` handles only one request at a time** (internally
   serialized with a ``threading.BoundedSemaphore`` plus a single lock; see
   the ``groot_core.py`` docstring: "upstream policy mutates process-global
   RNG state for request-level replay"). Therefore ``infer_batch`` loops
   over ``act`` for each request in the batch rather than concatenating them
   -- GR00T itself does not support batched inference, and faking batch
   semantics would only obscure this real limitation.
2. **Named observation fields come from ``Observation.extras["raw_state"]``**,
   not the flat ``Observation.state`` vector: ``robocasa_current.py``
   preserves the original named state dict in ``extras`` (e.g.
   ``state.end_effector_position_relative``); this module reads each key by
   ``groot_core.STATE_FIELDS`` and validates shape with
   ``groot_core.vector_from_state`` -- the same contract as ``groot_client.py``
   (the HTTP client) reading the same named fields, just without the
   JSON/HTTP round trip.
3. **Images are decoded from ``PayloadRef``**: ``Observation.main_image`` /
   ``wrist_image`` / ``extra_view_images`` are encoded payloads that need
   ``payload_module.decode_image`` to restore them into uint8 arrays before
   feeding them to GR00T's ``video.*`` keys.
4. **Action chunks are converted back to the 12-dim flat action** with
   ``groot_core.action_dict_to_flat_chunks`` -- the same conversion logic
   found at the end of ``groot_client.py``'s ``Gr00tClient.act``, same
   source, same result.

Dependency surface: depends only on ``robots.robocasa.groot_core`` (numpy +
torch, both lazily imported by that module itself), no rlinf. ``robots`` is
not imported before ``load()`` (lazy import, same posture as the other
backends).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import Any

import numpy as np

from rollout_runtime.api.enums import ErrorCode
from rollout_runtime.api.errors import make_error
from rollout_runtime.api.internal import ActionResponse, InferenceRequest
from rollout_runtime.core import payload as payload_module

__all__ = ["GROOT_POLICY_FAMILY", "GrootPolicyConfig", "GrootPolicyCore"]

GROOT_POLICY_FAMILY = "groot"
"""The family name corresponding to ``core.policy_inference.BATCHABLE_PARAM_KEYS``.

Not explicitly declared in ``BATCHABLE_PARAM_KEYS``, so it only inherits the
default whitelist (``do_sample``/``repetition_penalty``/``seed``/``temperature``/
``top_k``/``top_p``) -- GR00T's ``seed`` is a per-request hard constraint (see
``groot_core.parse_inference_seed``) and should not be treated as a
"batchable" parameter, but ``infer_batch`` in this module itself calls
``act`` serially per request (divergence 1) and never actually batches, so
the compat_key bucketing outcome does not affect correctness, only which
requests get counted as "the same batch" for reporting purposes.
"""


@dataclasses.dataclass(kw_only=True)
class GrootPolicyConfig:
    """Private configuration for the GR00T policy (corresponds to
    ``rollout_worker.policy_config``).

    Fields correspond one-to-one with the keyword arguments of
    ``groot_core.load_groot_model_core``.

    Attributes:
        groot_root: GR00T source root directory (fed to ``sys.path.insert``).
        model_path: Checkpoint directory.
        data_config_name: Key into ``DATA_CONFIG_MAP`` (e.g. ``"panda_omron"``).
        embodiment_tag: Embodiment tag.
        denoising_steps: Number of flow-sampling steps.
        maximum_pending: Upper bound on concurrent inference permits.
        expected_checkpoint_sha256: Expected checkpoint digest; ``None`` skips verification.
        action_dim: Reported action dimensionality (``groot_core.action_dict_to_flat_chunks``
            always outputs 12 dims; this field is only for diagnostics via
            ``compat_key``).
        model_version: Reported model version; ``None`` uses the checkpoint digest.
        device: Device identifier, goes into ``compat_key``.
        dtype: Compute precision name, goes into ``compat_key``.
    """

    groot_root: str
    model_path: str
    data_config_name: str = "panda_omron"
    embodiment_tag: str = "new_embodiment"
    denoising_steps: int = 4
    maximum_pending: int = 32
    expected_checkpoint_sha256: str | None = None
    action_dim: int = 12
    model_version: str | None = None
    device: str = "cuda"
    dtype: str = "bfloat16"

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any] | None) -> GrootPolicyConfig:
        """Construct configuration from a ``policy_config`` dict.

        Args:
            config: Family-private configuration.

        Returns:
            Structured configuration.

        Raises:
            ValueError: Missing required ``groot_root``/``model_path``, or an
                unknown key is present.
        """
        payload = dict(config or {})
        missing = [key for key in ("groot_root", "model_path") if key not in payload]
        if missing:
            raise ValueError(f"groot policy_config missing required keys: {missing}")
        known = {field.name for field in dataclasses.fields(cls)}
        unknown = sorted(set(payload) - known)
        if unknown:
            raise ValueError(f"unknown groot policy_config keys: {unknown}")
        return cls(**payload)


class GrootPolicyCore:
    """Inference core for the GR00T family (synchronous, blocking; same
    posture as ``RlinfPolicyCore``).

    ``Gr00tModelCore`` itself is a blocking call (synchronous
    ``policy.get_action``), so this core provides no ``ainfer_batch`` --
    ``RuntimeRolloutWorker`` calls ``infer_batch`` via ``asyncio.to_thread``,
    consistent with other synchronous cores (see the "env uses blocking,
    policy prefers async" split described in the
    ``core.policy_inference.PolicyInferenceCore`` docstring: GR00T, due to
    single-process serialized inference, naturally leans toward the
    env-like blocking category).
    """

    def __init__(self, config: GrootPolicyConfig) -> None:
        """Initialize a not-yet-``load``ed inference core.

        Args:
            config: GR00T configuration.
        """
        self.config = config
        self.core: Any = None
        self.loaded = False
        self.closed = False
        self.batch_calls = 0
        self.request_count = 0
        self.error_count = 0
        self._model_version = config.model_version or ""

    # ------------------------------------------------------------ Protocol attributes

    @property
    def model_version(self) -> str:
        """The current model version.

        Returns:
            The version identifier: the explicitly configured value,
            otherwise the checkpoint digest (only available after ``load()``).
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
            ``"groot"``, determines the ``compat_key`` batchable whitelist.
        """
        return GROOT_POLICY_FAMILY

    # ---------------------------------------------------------------- Lifecycle

    def load(self) -> None:
        """Load the GR00T checkpoint once (one resident copy per rank, no
        longer needing a separate process)."""
        from robots.robocasa.groot_core import load_groot_model_core

        self.core = load_groot_model_core(
            groot_root=self.config.groot_root,
            model_path=self.config.model_path,
            data_config_name=self.config.data_config_name,
            embodiment_tag=self.config.embodiment_tag,
            denoising_steps=self.config.denoising_steps,
            maximum_pending=self.config.maximum_pending,
            expected_checkpoint_sha256=self.config.expected_checkpoint_sha256,
        )
        if not self._model_version:
            self._model_version = self.core.checkpoint_sha256
        self.loaded = True

    def update_weights(self, model_version: str) -> None:
        """Record the expected model version label (the GR00T core does not
        support hot-swapping weights).

        Args:
            model_version: Target version label, used only for reporting;
                it does not trigger a checkpoint reload.
        """
        self._model_version = model_version

    def close(self) -> None:
        """Release resources (``Gr00tModelCore`` has no explicit close; only sets a flag)."""
        self.core = None
        self.closed = True

    # ------------------------------------------------------------------ Inference

    def infer_batch(self, requests: list[InferenceRequest]) -> list[ActionResponse]:
        """Call ``Gr00tModelCore.act`` once per request (divergence 1: GR00T
        does not support true batching).

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
        return [self._infer_one(request) for request in requests]

    def _infer_one(self, request: InferenceRequest) -> ActionResponse:
        try:
            payload = self._build_payload(request)
            actions, metadata = self.core.act(payload)
            flat_chunks, clamped_values = _action_dict_to_flat_chunks(actions)
            block = np.asarray(flat_chunks, dtype=np.float32)
            if not np.isfinite(block).all():
                raise ValueError("groot policy produced non-finite actions")
        except BaseException as exc:  # noqa: BLE001 - D5: per-request normalization, never leak
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
                    f"groot inference failed: {exc}",
                    policy_id=request.policy_id,
                    session_id=request.session_id,
                ),
            )
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
                "clamped_values": int(clamped_values),
                **{
                    key: value
                    for key, value in metadata.items()
                    if key in ("queue_latency_s", "inference_latency_s", "request_id")
                },
            },
        )

    def _build_payload(self, request: InferenceRequest) -> dict[str, Any]:
        """Translate an ``InferenceRequest`` into the named payload expected by ``Gr00tModelCore.act``.

        Args:
            request: Inference request.

        Returns:
            ``{"seed": int, "observation": {"video.*": ..., "state.*": ..., <instruction key>: ...}}``.

        Raises:
            ValueError: The observation is missing a required named state
                field, or an image slot is empty.
        """
        from robots.robocasa.groot_core import LANGUAGE_KEY, STATE_FIELDS, VIDEO_KEYS

        observation = request.observation
        raw_state = observation.extras.get("raw_state") or {}
        state_payload: dict[str, Any] = {}
        for key, size in STATE_FIELDS.items():
            if key not in raw_state:
                raise ValueError(
                    f"observation is missing required GR00T state field {key!r}"
                )
            state_payload[key] = [_vector_from_state(raw_state[key], size, key)]

        image_slots = {
            VIDEO_KEYS[0]: observation.main_image,
            VIDEO_KEYS[1]: (
                observation.extra_view_images[0]
                if observation.extra_view_images
                else None
            ),
            VIDEO_KEYS[2]: observation.wrist_image,
        }
        video_payload: dict[str, Any] = {}
        for key, ref in image_slots.items():
            if ref is None:
                raise ValueError(f"observation is missing required GR00T video {key!r}")
            image = payload_module.decode_image(ref)
            video_payload[key] = [image.tolist()]

        instruction = (
            request.instruction_override
            if request.instruction_override is not None
            else observation.instruction
        )
        seed = request.inference_parameters.get("seed")
        if not isinstance(seed, int) or isinstance(seed, bool):
            seed = _deterministic_seed(request.request_id)
        return {
            "seed": int(seed),
            "observation": {
                **video_payload,
                **state_payload,
                LANGUAGE_KEY: [instruction],
            },
        }


def _deterministic_seed(request_id: Any) -> int:
    """Derive a deterministic seed in ``[0, 2**31 - 1]`` from ``request_id``.

    Fallback for when ``request.inference_parameters`` does not explicitly
    give a ``seed``: the same ``request_id`` always yields the same seed
    (reproducible replay), rather than a different one on every call (the
    same "derive a deterministic value from request_id" approach used by
    ``fake/policy.py::_delay_for``).

    Args:
        request_id: Request identifier.

    Returns:
        An integer seed in ``[0, 2**31 - 1]``.
    """
    import hashlib

    digest = hashlib.sha256(str(request_id).encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % (2**31 - 1)


def _vector_from_state(value: Any, size: int, key: str) -> list[float]:
    from robots.robocasa.groot_core import vector_from_state

    return vector_from_state(value, size, key)


def _action_dict_to_flat_chunks(
    action_object: Mapping[str, Any],
) -> tuple[list[list[float]], int]:
    from robots.robocasa.groot_core import action_dict_to_flat_chunks

    return action_dict_to_flat_chunks(action_object)
