"""Fake policy backend.

``FakePolicyCore`` implements ``core.policy_inference.PolicyInferenceCore``,
returning predictable ``[chunk, action_dim] float32`` output: the action is
determined solely by ``(session_id, episode_id, step_index)``, so the same
observation always yields the same action chunk -- letting idempotency and
out-of-order assertions do exact comparisons.

Three kinds of injection:

| Config / method | Behavior | Used for |
|---|---|---|
| ``delay_seconds`` + ``jitter_seconds`` | Deterministic delay derived from ``request_id`` | Out-of-order responses |
| ``fail_sessions`` / ``fail_policy_ids`` | Per-request ``POLICY_FAILURE``, never raises | Error isolation (D5) |
| ``hold()`` / ``release()`` | Inference hangs until released by the test | Backpressure ``QUEUE_FULL``, the "waiting on inference" cancellation state |

Unlike the env backend, waiting here uses **asyncio** (``ainfer_batch``):
inference is semantically "waiting on a remote service" and must be
cancellable -- this is what makes it possible for the EnvWorker to stop
waiting when an operation has already been dispatched but is still waiting
on inference. ``infer_batch`` (synchronous, pure computation) is kept as
well, satisfying the Protocol and letting real backends reuse the same
"synchronous core + ``to_thread``" shape.
"""

from __future__ import annotations

import asyncio
import dataclasses
import hashlib
from collections.abc import Callable

import numpy as np

from rollout_runtime.api.enums import ErrorCode
from rollout_runtime.api.errors import make_error
from rollout_runtime.api.internal import ActionResponse, InferenceRequest
from rollout_runtime.core import payload as payload_module

__all__ = ["FAKE_POLICY_FAMILY", "FakePolicyConfig", "FakePolicyCore"]

FAKE_POLICY_FAMILY = "fake"
"""The family name corresponding to ``core.policy_inference.BATCHABLE_PARAM_KEYS``."""


@dataclasses.dataclass(kw_only=True)
class FakePolicyConfig:
    """Fake policy configuration.

    Attributes:
        action_dim: Action dimensionality (v1's VLA is 7-dim).
        actions_per_chunk: Default action chunk length.
        model_version: Reported model version, stamped into ``ActionResponse``
            per request.
        device: Device identifier, goes into ``compat_key``.
        dtype: Compute precision, goes into ``compat_key``.
        policy_family: Policy family name, determines the batchable parameter whitelist.
        delay_seconds: Base delay per batch.
        jitter_seconds: Upper bound on the additional delay derived from
            ``request_id`` (deterministic, no random numbers used).
        fail_sessions: Requests for these sessions return ``POLICY_FAILURE``.
        fail_policy_ids: Requests for these policy_ids return ``POLICY_FAILURE``.
    """

    action_dim: int = 7
    actions_per_chunk: int = 4
    model_version: str = "fake-v1"
    device: str = "cpu"
    dtype: str = "float32"
    policy_family: str = FAKE_POLICY_FAMILY
    delay_seconds: float = 0.0
    jitter_seconds: float = 0.0
    fail_sessions: frozenset[str] = frozenset()
    fail_policy_ids: frozenset[str] = frozenset()


class FakePolicyCore:
    """A deterministic inference core with injectable delay, failure, and hangs."""

    def __init__(self, config: FakePolicyConfig | None = None) -> None:
        """Initialize.

        Args:
            config: Fake policy configuration; ``None`` uses defaults.
        """
        self.config = config or FakePolicyConfig()
        self.loaded = False
        self.closed = False
        self.batch_calls = 0
        self.request_count = 0
        self.inflight = 0
        self.entered_inference = asyncio.Event()
        self.delay_hook: Callable[[InferenceRequest], float] | None = None
        self._model_version = self.config.model_version
        self._gate: asyncio.Event | None = None

    # ------------------------------------------------------------ Protocol attributes

    @property
    def model_version(self) -> str:
        """The current model version.

        Returns:
            The version identifier.
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
            Family identifier, determines the ``compat_key`` batchable whitelist.
        """
        return self.config.policy_family

    # ---------------------------------------------------------------- Lifecycle

    def load(self) -> None:
        """Load the model (the fake backend only sets a flag)."""
        self.loaded = True

    def update_weights(self, model_version: str) -> None:
        """Switch weight versions at a batch boundary.

        Args:
            model_version: Target version.
        """
        self._model_version = model_version

    def close(self) -> None:
        """Release resources and unblock any hung inference."""
        self.release()
        self.closed = True

    # ------------------------------------------------------------------ Hang gate

    def hold(self) -> asyncio.Event:
        """Hang subsequent inference calls until ``release()``.

        Returns:
            The gate event; the caller can ``set()`` it to let calls through.
        """
        gate = asyncio.Event()
        self._gate = gate
        return gate

    def release(self) -> None:
        """Release inference calls hung by ``hold()``."""
        gate = self._gate
        self._gate = None
        if gate is not None:
            gate.set()

    @property
    def holding(self) -> bool:
        """Whether currently in a hung state.

        Returns:
            True if ``hold()`` was called and ``release()`` has not been called yet.
        """
        return self._gate is not None

    # ------------------------------------------------------------------ Inference

    async def ainfer_batch(
        self, requests: list[InferenceRequest]
    ) -> list[ActionResponse]:
        """Asynchronous inference: pass through the gate and delay first,
        then run the synchronous computation.

        Args:
            requests: List of requests sharing a ``compat_key``.

        Returns:
            Per-request responses in the same order as the input.
        """
        self.inflight += len(requests)
        self.entered_inference.set()
        try:
            gate = self._gate
            if gate is not None:
                await gate.wait()
            delay = max((self._delay_for(request) for request in requests), default=0.0)
            if delay > 0:
                await asyncio.sleep(delay)
            return self.infer_batch(requests)
        finally:
            self.inflight -= len(requests)

    def infer_batch(self, requests: list[InferenceRequest]) -> list[ActionResponse]:
        """Execute inference on a pre-bucketed batch (pure computation, no waiting).

        A single request's failure only affects that request: the error goes
        into ``ActionResponse.error``, and this method never raises (D5).

        Args:
            requests: List of requests sharing a ``compat_key``.

        Returns:
            Per-request responses in the same order as the input.
        """
        self.batch_calls += 1
        responses: list[ActionResponse] = []
        for request in requests:
            self.request_count += 1
            if (
                str(request.session_id) in self.config.fail_sessions
                or request.policy_id in self.config.fail_policy_ids
            ):
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
                            "fake policy injected failure for "
                            f"session {request.session_id}",
                            policy_id=request.policy_id,
                            session_id=request.session_id,
                        ),
                    )
                )
                continue
            actions = self._actions_for(request)
            responses.append(
                ActionResponse(
                    request_id=request.request_id,
                    session_id=request.session_id,
                    binding_token=request.binding_token,
                    episode_id=request.episode_id,
                    operation_seq=request.operation_seq,
                    actions=payload_module.encode_array(actions),
                    model_version=self._model_version,
                    auxiliary_outputs={
                        "chunk": int(actions.shape[0]),
                        "compat_key": request.compat_key,
                    },
                )
            )
        return responses

    # ------------------------------------------------------------------ Internal

    def _chunk_for(self, request: InferenceRequest) -> int:
        requested = request.inference_parameters.get("actions_per_chunk")
        if isinstance(requested, int) and requested > 0:
            return requested
        return self.config.actions_per_chunk

    def _actions_for(self, request: InferenceRequest) -> np.ndarray:
        """Generate a deterministic action chunk.

        Args:
            request: The inference request.

        Returns:
            ``[chunk, action_dim] float32``, values in ``[-1, 1)``.
        """
        key = (
            f"{request.session_id}|{request.episode_id}|"
            f"{request.observation.step_index}|{request.policy_id}"
        )
        seed = int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:16], 16)
        generator = np.random.default_rng(seed)
        chunk = self._chunk_for(request)
        return generator.uniform(
            -1.0, 1.0, size=(chunk, self.config.action_dim)
        ).astype(np.float32)

    def _delay_for(self, request: InferenceRequest) -> float:
        if self.delay_hook is not None:
            return float(self.delay_hook(request))
        base = self.config.delay_seconds
        if self.config.jitter_seconds <= 0:
            return base
        digest = hashlib.sha256(str(request.request_id).encode("utf-8")).hexdigest()
        fraction = int(digest[:4], 16) / 0xFFFF
        return base + self.config.jitter_seconds * fraction
