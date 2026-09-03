"""rlinf implementation of ``PolicyInferenceCore`` (supports the openpi / pi0.5 pipeline first).

Extraction source and scope follow two lists exactly:

| Reused (extracted from ``workers/rollout/hf/huggingface_worker.py``) | Landing point |
|---|---|
| Model loading / precision / ``torch_compile`` / ``cuda_graph`` (:141-193) | ``RlinfPolicyCore.load`` |
| Sampling parameter assembly (:195-232) -> per-request ``inference_parameters`` | ``_predict_kwargs`` |
| ``predict`` body + SupportedModel branch (:471-554) | ``_predict_batch`` |
| Result construction (:579-610) -> generic ``ActionResponse`` | ``infer_batch`` |
| Weight sync (:630-676) | ``update_weights`` |
| offload / reload (:859-880) | ``offload`` / ``reload`` |
| ``get_model`` / ``BasePolicy`` (models/__init__.py:273-345, base_policy.py:33-107) | ``_build_model`` |

Explicitly **excluded**: ``get_bootstrap_values``, advantage / return / loss,
the trainer's trajectory sharding, DAgger / expert / RLT / DSRL -- these are
RL-only semantics. This core only does "a batch of observations -> a batch
of ``[chunk, action_dim]`` actions."

Three hard constraints (the "known batching constraints") are implemented here:

1. ``cuda_graph`` bakes in the batch size (``mlp_policy.py:392-437``), so
   enabling it requires also supplying ``SchedulerConfig.fixed_batch_size``,
   or ``load`` refuses outright; this core pads to a fixed batch and records
   the real request count in ``auxiliary_outputs``.
2. ``openvla_oft``'s ``padding="max_length"`` is fixed -> goes into
   ``compat_key`` (``constraints``, see ``compat_key_constraints``).
3. ``_merge_obs_batches`` (:910-960) requires the obs dict structure to be
   completely identical -> guaranteed by ``obs_schema_digest``; this core
   double-checks with ``observations_to_env_output`` and reports errors
   per-request on mismatch.

D5 applies **per-request** here: ``infer_batch`` never raises; any failure
becomes an ``ActionResponse(error=...)``. A remote exception would trigger
``os.kill(pid, SIGUSR1)``, killing the whole job.

This core is **synchronous and blocking** (a real GPU forward pass is
inherently so), called via ``asyncio.to_thread`` by
``RuntimeRolloutWorker`` -- "only synchronous cores fall back to ``to_thread``."
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from rollout_runtime.api.enums import ErrorCode
from rollout_runtime.api.errors import make_error
from rollout_runtime.api.internal import ActionResponse, InferenceRequest
from rollout_runtime.core import payload as payload_module
from rollout_runtime.core.obs_schema import (
    obs_schema_digest,
    observations_to_env_output,
)

__all__ = [
    "OPENPI_MODEL_TYPES",
    "RlinfPolicyConfig",
    "RlinfPolicyCore",
]

OPENPI_MODEL_TYPES = frozenset({"openpi", "openpi_pytorch"})
"""``model_type`` values currently supported: the openpi (pi0 / pi0.5) pipeline."""

_PRECISION_TO_DTYPE = {
    None: "float32",
    "null": "float32",
    "fp32": "float32",
    "float32": "float32",
    "fp16": "float16",
    "float16": "float16",
    "bf16": "bfloat16",
    "bfloat16": "bfloat16",
}
"""Mapping from ``precision`` string to dtype name, used only for reporting and ``compat_key``."""

# Column indices for ``gripper_close`` and ``control_mode`` in RoboCasa's
# 12-dim flat action contract (``robots.robocasa.action_contract``),
# corresponding to the ``COMPONENT_SIZES`` order: end_effector_position(3) +
# end_effector_rotation(3) + gripper_close(1) + base_motion(4) + control_mode(1).
_ROBOCASA_GRIPPER_CLOSE_INDEX = 6
_ROBOCASA_CONTROL_MODE_INDEX = 11


def _remap_robocasa_gripper_and_control_mode(block: np.ndarray) -> None:
    """Remap openpi/pi0's gripper/control_mode outputs into RoboCasa's value domain.

    openpi/pi0-family models output ``gripper_close`` and ``control_mode`` in
    the same ``[-1, 1]`` convention as position/rotation/base_motion, but
    ``robots.robocasa.action_contract.canonical_action`` requires ``[0, 1]``
    for these two channels (unlike other channels, it does not clip before
    validating -- out of range raises ``ValueError`` directly). The GR00T
    path (``robots.robocasa.groot_core``) already has a dedicated action
    conversion; this adds the equivalent linear mapping for the generic
    openpi/pi0 backend, active only when ``action_dim == 12`` (RoboCasa's
    flat action contract), leaving other 7-dim families like LIBERO
    unaffected. Modifies in place; the caller must ensure ``block`` is
    writable.

    Args:
        block: A ``[chunk, 12]`` action block, modified in place.
    """
    for column in (_ROBOCASA_GRIPPER_CLOSE_INDEX, _ROBOCASA_CONTROL_MODE_INDEX):
        block[:, column] = np.clip((block[:, column] + 1.0) / 2.0, 0.0, 1.0)


@dataclasses.dataclass(kw_only=True)
class RlinfPolicyConfig:
    """Policy backend configuration (corresponds to
    ``RolloutWorkerConfig.policy_config``).

    Defaults are taken from ``examples/embodiment/config/model/pi0_5.yaml``,
    so a minimal configuration only needs to supply ``model_path``.

    Attributes:
        model_type: Value from rlinf's ``SupportedModel``; only the openpi
            family is currently supported.
        model_path: Weights directory.
        precision: ``null`` / ``bf16`` / ``fp16`` / ``fp32``; must be
            ``null`` for openpi.
        num_action_chunks: Length of the action chunk output per model call.
        action_dim: Environment-side action dimensionality.
        num_steps: Number of diffusion/flow sampling steps.
        use_proprio: Whether to use proprioceptive state.
        add_value_head: Whether to include a value head (the Runtime only
            does inference, off by default).
        is_lora: Whether LoRA is used.
        lora_rank: LoRA rank.
        lora_path: Trained LoRA directory.
        load_to_device: Whether to move to the accelerator inside ``get_model``.
        ckpt_path: Optional ``.pt`` weights, loaded after the model is built.
        family_params: Family-private parameter block (openpi's
            ``config_name`` / ``num_images_in_input`` etc.), embedded into
            ``get_model``'s cfg under the ``model_type`` key.
        enable_torch_compile: Whether to ``torch.compile``.
        torch_compile_mode: Compile mode.
        enable_cuda_graph: Whether to capture a CUDA graph; requires
            ``fixed_batch_size``.
        cuda_graph_batch_size: Fixed CUDA graph batch size, must equal
            ``SchedulerConfig.fixed_batch_size``.
        default_mode: Default ``mode`` for ``predict_action_batch``.
        model_version: Reported model version, stamped into ``ActionResponse``
            per request.
        policy_family: Family name from ``BATCHABLE_PARAM_KEYS``.
        device: Device identifier (goes into ``compat_key``).
        dtype: Compute precision name; ``None`` means infer from ``precision``.
        unnorm_key: Action denormalization key (needed for openvla family;
            ignored by openpi).
        stack_wrist_views: Whether to re-stack ``wrist_images`` and
            ``extra_view_images`` into one ``[B, N_IMG, H, W, C]`` tensor before
            the forward pass. Required by the ALOHA pipeline, which reads the
            left and right wrist as ``wrist_images[0]`` / ``wrist_images[1]``
            and ignores ``extra_view_image`` entirely; see
            :func:`_stack_wrist_views`.
    """

    model_type: str = "openpi"
    model_path: str = ""
    precision: str | None = None
    num_action_chunks: int = 10
    action_dim: int = 7
    num_steps: int = 5
    use_proprio: bool = True
    add_value_head: bool = False
    is_lora: bool = False
    lora_rank: int = 32
    lora_path: str | None = None
    load_to_device: bool = True
    ckpt_path: str | None = None
    family_params: dict[str, Any] = dataclasses.field(
        default_factory=lambda: {
            "config_name": "pi05_libero",
            "num_images_in_input": 2,
            "noise_level": 0.5,
            "train_expert_only": True,
            "noise_method": "flow_sde",
            "value_after_vlm": False,
            "value_vlm_mode": "mean_token",
            "detach_critic_input": None,
            "use_dsrl": False,
        }
    )
    enable_torch_compile: bool = False
    torch_compile_mode: str = "max-autotune-no-cudagraphs"
    enable_cuda_graph: bool = False
    cuda_graph_batch_size: int | None = None
    default_mode: str = "eval"
    model_version: str = "pi05"
    policy_family: str = "openpi"
    device: str = "cuda"
    dtype: str | None = None
    unnorm_key: str = ""
    stack_wrist_views: bool = False

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any] | None) -> RlinfPolicyConfig:
        """Construct configuration from a ``policy_config`` dict.

        Args:
            config: Policy-private configuration; ``None`` uses all defaults.

        Returns:
            Structured configuration.

        Raises:
            ValueError: An unknown key is present, or ``model_type`` is
                outside the currently supported set.
        """
        instance = cls(**dict(config or {}))
        if instance.model_type not in OPENPI_MODEL_TYPES:
            raise ValueError(
                f"model_type {instance.model_type!r} is out of M4 scope; "
                f"supported: {sorted(OPENPI_MODEL_TYPES)} "
                "(other VLA families land in M6)"
            )
        return instance

    def resolved_dtype(self) -> str:
        """Return the dtype name used for reporting.

        Returns:
            Dtype name (explicit ``dtype`` takes priority, otherwise
            inferred from ``precision``).
        """
        if self.dtype:
            return self.dtype
        key = self.precision.lower() if isinstance(self.precision, str) else None
        return _PRECISION_TO_DTYPE.get(key, "float32")

    def to_model_cfg(self) -> Any:
        """Project into the omegaconf configuration expected by ``rlinf.models.get_model``.

        Returns:
            ``DictConfig``.
        """
        from omegaconf import OmegaConf

        payload: dict[str, Any] = {
            "model_type": self.model_type,
            "model_path": self.model_path,
            "precision": self.precision,
            "num_action_chunks": int(self.num_action_chunks),
            "action_dim": int(self.action_dim),
            "num_steps": int(self.num_steps),
            "use_proprio": bool(self.use_proprio),
            "add_value_head": bool(self.add_value_head),
            "is_lora": bool(self.is_lora),
            "lora_rank": int(self.lora_rank),
            "load_to_device": bool(self.load_to_device),
            "device": self.device,
        }
        if self.lora_path:
            payload["lora_path"] = self.lora_path
        if self.unnorm_key:
            payload["unnorm_key"] = self.unnorm_key
        family = dict(self.family_params)
        family.setdefault("action_chunk", int(self.num_action_chunks))
        family.setdefault("num_steps", int(self.num_steps))
        family.setdefault("action_env_dim", int(self.action_dim))
        family.setdefault("add_value_head", bool(self.add_value_head))
        # openpi's family parameters live at cfg.openpi.* in rlinf;
        # openpi_pytorch is isomorphic.
        payload[self.model_type] = family
        return OmegaConf.create(payload)

    def compat_key_constraints(self) -> dict[str, Any]:
        """Return the hard constraints to put into ``compat_key``.

        Returns:
            Constraint dict: the fixed ``cuda_graph`` batch,
            ``openvla_oft``'s ``padding="max_length"``, and the action chunk length.
        """
        constraints: dict[str, Any] = {
            "actions_per_chunk": int(self.num_action_chunks),
            "action_dim": int(self.action_dim),
        }
        if self.enable_cuda_graph:
            constraints["fixed_batch_size"] = int(self.cuda_graph_batch_size or 0)
        if self.model_type == "openvla_oft":
            constraints["padding"] = "max_length"
        return constraints


class RlinfPolicyCore:
    """rlinf HF policy inference core (synchronous, blocking)."""

    def __init__(self, config: RlinfPolicyConfig | None = None) -> None:
        """Initialize a not-yet-``load``ed inference core.

        Args:
            config: Policy configuration; ``None`` uses defaults (openpi / pi0.5).
        """
        self.config = config or RlinfPolicyConfig()
        self.model: Any = None
        self.loaded = False
        self.closed = False
        self.offloaded = False
        self.batch_calls = 0
        self.request_count = 0
        self.error_count = 0
        self.padded_request_count = 0
        self._model_version = self.config.model_version

    # ------------------------------------------------------------ Protocol attributes

    @property
    def model_version(self) -> str:
        """The current model version.

        Returns:
            The version identifier; changes after ``update_weights``.
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
        return self.config.resolved_dtype()

    @property
    def policy_family(self) -> str:
        """The policy family name.

        Returns:
            Family identifier, determines the ``compat_key`` batchable whitelist.
        """
        return self.config.policy_family

    # ---------------------------------------------------------------- Lifecycle

    def load(self) -> None:
        """Load the model on the accelerator assigned to the current rank.

        In order: ``get_model`` -> optional ``ckpt_path`` -> ``eval()`` ->
        optional ``torch_compile`` -> optional ``capture_cuda_graph``.

        Raises:
            ValueError: ``cuda_graph`` is enabled without a fixed batch size
                (which would otherwise bake the batch size in).
            RuntimeError: ``get_model`` did not recognize ``model_type``.
        """
        if self.loaded:
            return
        config = self.config
        if config.enable_cuda_graph and not config.cuda_graph_batch_size:
            raise ValueError(
                "enable_cuda_graph requires cuda_graph_batch_size (and the matching "
                "SchedulerConfig.fixed_batch_size): CUDA graph capture bakes the batch "
                "size in (mlp_policy.py:392-437)"
            )
        import torch

        from zetta.policies.openpi.factory import build_openpi_model

        model = build_openpi_model(config.to_model_cfg())
        if model is None:
            raise RuntimeError(
                f"OpenPI factory returned None for model_type={config.model_type!r}"
            )
        if config.ckpt_path:
            model.load_state_dict(torch.load(config.ckpt_path))
        model.eval()
        if config.enable_torch_compile:
            model.enable_torch_compile(mode=config.torch_compile_mode)
        if config.enable_cuda_graph:
            batch = int(config.cuda_graph_batch_size or 0)
            model.capture_cuda_graph(train_batch_size=batch, eval_batch_size=batch)
        self.model = model
        self.loaded = True

    def update_weights(self, model_version: str) -> None:
        """Switch weight versions at a batch boundary.

        The Runtime only serves inference; there is no actor group to pull
        weights from, so the v1 semantics are: if ``model_version`` is of
        the form ``"file:<path>"``, reload the ``state_dict`` from that
        ``.pt`` file; otherwise only change the version label (for A/B
        bookkeeping and ``compat_key`` bucketing). A real ``WeightSyncer``
        (huggingface_worker.py:630-676) awaits the Runtime being wired into
        the training loop, which is out of scope here.

        Args:
            model_version: Target version; ``"file:<path>"`` triggers an
                actual reload.
        """
        if model_version.startswith("file:") and self.model is not None:
            import torch

            self.model.load_state_dict(torch.load(model_version[len("file:") :]))
        self._model_version = model_version

    def offload(self) -> None:
        """Move the model to CPU and clear the GPU memory cache (huggingface_worker.py:859-880)."""
        if self.model is None or self.offloaded:
            return
        import torch

        self.model.to("cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.offloaded = True

    def reload(self) -> None:
        """Move the model back to the accelerator."""
        if self.model is None or not self.offloaded:
            return
        self.model.to(self.config.device)
        self.offloaded = False

    def close(self) -> None:
        """Release the model and GPU memory."""
        model = self.model
        self.model = None
        self.loaded = False
        self.closed = True
        if model is None:
            return
        release = getattr(model, "release_cuda_graph", None)
        if callable(release):
            try:
                release()
            except BaseException:  # noqa: BLE001 - cleanup must never raise
                pass
        del model
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except BaseException:  # noqa: BLE001 - cleanup must never raise
            pass

    # ------------------------------------------------------------------ Inference

    def infer_batch(self, requests: list[InferenceRequest]) -> list[ActionResponse]:
        """Execute inference on a pre-bucketed batch (a total function, D5).

        Args:
            requests: List of requests sharing a ``compat_key``.

        Returns:
            Per-request responses in the same order as the input; any
            failure is reported as ``error`` and this method never raises.
        """
        if not requests:
            return []
        self.batch_calls += 1
        self.request_count += len(requests)
        try:
            actions = self._predict_batch(requests)
        except BaseException as exc:  # noqa: BLE001 - D5: per-request normalization, never leak
            self.error_count += len(requests)
            return [self._error_response(request, exc) for request in requests]
        responses: list[ActionResponse] = []
        for index, request in enumerate(requests):
            block = np.ascontiguousarray(
                np.asarray(actions[index], dtype=np.float32).reshape(
                    -1, int(self.config.action_dim)
                )
            )
            if int(self.config.action_dim) == 12:
                _remap_robocasa_gripper_and_control_mode(block)
            if not np.isfinite(block).all():
                self.error_count += 1
                responses.append(
                    ActionResponse(
                        request_id=request.request_id,
                        session_id=request.session_id,
                        binding_token=request.binding_token,
                        episode_id=request.episode_id,
                        operation_seq=request.operation_seq,
                        model_version=self._model_version,
                        error=make_error(
                            ErrorCode.POLICY_FAILURE,
                            "policy produced non-finite actions",
                            policy_id=request.policy_id,
                            session_id=request.session_id,
                        ),
                    )
                )
                continue
            responses.append(
                ActionResponse(
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
                        "batch_size": len(requests),
                        "model_type": self.config.model_type,
                    },
                )
            )
        return responses

    # ------------------------------------------------------------------ Internal

    def _predict_kwargs(self, requests: Sequence[InferenceRequest]) -> dict[str, Any]:
        """Assemble keyword arguments for ``predict_action_batch`` (sampling
        parameter assembly).

        Requests sharing a ``compat_key`` are guaranteed identical on these
        keys (``canonicalize_inference_parameters`` already put them into
        the key), so taking the first request's values suffices.

        Args:
            requests: The batch of requests.

        Returns:
            Keyword arguments.
        """
        parameters = dict(requests[0].inference_parameters)
        mode = str(parameters.get("mode", self.config.default_mode))
        return {"mode": mode, "compute_values": False}

    def _predict_batch(self, requests: list[InferenceRequest]) -> np.ndarray:
        """Build env_obs, run one ``predict_action_batch`` call, and return ``[B, chunk, dim]``.

        Args:
            requests: List of requests sharing a ``compat_key``.

        Returns:
            numpy actions, dim 0 in the same order and length as ``requests``
            (padding already trimmed off).

        Raises:
            RuntimeError: The model has not been ``load``ed yet.
        """
        if self.model is None:
            raise RuntimeError("policy core has not been loaded")
        import torch

        from zetta.compat.tensors import to_tensor

        observations = [request.observation for request in requests]
        reference = obs_schema_digest(observations[0])
        for index, observation in enumerate(observations[1:], start=1):
            if obs_schema_digest(observation) != reference:
                raise ValueError(
                    "batch contains mixed observation schemas at index "
                    f"{index}; compat_key should have separated them"
                )
        env_obs = observations_to_env_output(observations)
        if self.config.stack_wrist_views:
            env_obs = _stack_wrist_views(env_obs)
        instructions = list(env_obs["task_descriptions"])
        for index, request in enumerate(requests):
            if request.instruction_override:
                instructions[index] = request.instruction_override

        real = len(requests)
        pad_to = real
        if self.config.enable_cuda_graph:
            pad_to = int(self.config.cuda_graph_batch_size or real)
            if real > pad_to:
                raise ValueError(
                    f"batch of {real} exceeds the captured cuda_graph batch {pad_to}; "
                    "SchedulerConfig.fixed_batch_size must match"
                )
        payload: dict[str, Any] = {}
        for key in ("main_images", "wrist_images", "extra_view_images", "states"):
            array = env_obs.get(key)
            if array is None:
                payload[key] = None
                continue
            padded = _pad_batch(np.asarray(array), pad_to)
            payload[key] = to_tensor(padded)
        payload["task_descriptions"] = instructions + [instructions[-1]] * (
            pad_to - real
        )
        if pad_to > real:
            self.padded_request_count += pad_to - real

        with torch.no_grad():
            actions, _result = self.model.predict_action_batch(
                env_obs=payload, **self._predict_kwargs(requests)
            )
        if hasattr(actions, "detach"):
            actions = actions.detach().to(torch.float32).cpu().numpy()
        array = np.asarray(actions, dtype=np.float32)
        return array[:real]

    def _error_response(
        self, request: InferenceRequest, exc: BaseException
    ) -> ActionResponse:
        """Normalize an exception into a single failed response (D5).

        Args:
            request: The request that triggered the failure.
            exc: The caught exception.

        Returns:
            A response carrying ``POLICY_FAILURE``.
        """
        return ActionResponse(
            request_id=request.request_id,
            session_id=request.session_id,
            binding_token=request.binding_token,
            episode_id=request.episode_id,
            operation_seq=request.operation_seq,
            model_version=self._model_version,
            error=make_error(
                ErrorCode.POLICY_FAILURE,
                f"{type(exc).__name__}: {exc}",
                policy_id=request.policy_id,
                session_id=request.session_id,
                model_type=self.config.model_type,
            ),
        )


def _stack_wrist_views(env_obs: dict[str, Any]) -> dict[str, Any]:
    """Re-stack the wrist views into the single ``[B, N_IMG, H, W, C]`` tensor.

    The rlinf 5-key schema allows ``wrist_images`` to be either
    ``[N_ENV, H, W, C]`` or ``[N_ENV, N_IMG, H, W, C]``
    (``rlinf/data/schema/embodied_types.py::prepare_observations``), and the
    ALOHA pipeline **requires** the second form: ``aloha_policy._decode_aloha``
    indexes ``wrist_images[0]`` as the left wrist and ``wrist_images[1]`` as the
    right one, and never reads ``observation/extra_view_image`` at all.

    The Runtime's ``Observation`` cannot carry that form -- ``wrist_image`` is a
    single ``PayloadRef`` -- so a bimanual family splits its two wrists across
    ``wrist_image`` (left) and ``extra_view_images[0]`` (right), and this
    function is the exact inverse, applied just before the model sees the batch.
    The split and this re-stack are a matched pair; changing one without the
    other silently feeds the model a mangled wrist tensor rather than raising
    (RoboTwin's first hardware run produced a
    ``expected input[2, 224, 224, 224] to have 3 channels`` deep inside the
    vision tower).

    A no-op when there is nothing to stack, so it is safe to leave enabled for a
    single-wrist configuration.

    Args:
        env_obs: The five-key batch dict from ``observations_to_env_output``.

    Returns:
        A shallow copy with ``wrist_images`` stacked and ``extra_view_images``
        cleared; the input unchanged when either view is absent.
    """
    wrist = env_obs.get("wrist_images")
    extra = env_obs.get("extra_view_images")
    if wrist is None or extra is None:
        return env_obs
    wrist_array = np.asarray(wrist)
    extra_array = np.asarray(extra)
    if wrist_array.ndim == 4:
        # [B, H, W, C] -> [B, 1, H, W, C]
        wrist_array = wrist_array[:, None, ...]
    if extra_array.ndim == 4:
        extra_array = extra_array[:, None, ...]
    stacked = dict(env_obs)
    stacked["wrist_images"] = np.concatenate([wrist_array, extra_array], axis=1)
    stacked["extra_view_images"] = None
    return stacked


def _pad_batch(array: np.ndarray, batch_size: int) -> np.ndarray:
    """Pad the batch dimension to ``batch_size`` (repeating the last entry),
    used for CUDA graph's fixed batch size.

    Args:
        array: Array whose dim 0 is the batch dimension.
        batch_size: Target batch size.

    Returns:
        The padded array; returned as-is if no padding is needed.
    """
    real = int(array.shape[0])
    if batch_size <= real:
        return array
    pad = np.repeat(array[-1:], batch_size - real, axis=0)
    return np.concatenate([array, pad], axis=0)
