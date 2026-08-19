"""The ``RuntimeClient`` protocol.

Every entry point is ``async`` and batch-shaped; partial success is allowed
within a batch, and there are no cross-session transactions. Both the
Gateway (``gateway.gateway.RuntimeGateway``) and every Adapter treat this as
the contract; Adapters must never bypass the Gateway to touch the transport
or workers directly.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal, Protocol, runtime_checkable

from rollout_runtime.api.ids import RequestId, SessionId
from rollout_runtime.api.messages import (
    CancelOutcome,
    CreateSessionRequest,
    EpisodeRequest,
    EpisodeResult,
    Observation,
    OperationStatus,
    PolicyInferResult,
    PolicyRequest,
    ResetSpec,
    SessionHandle,
    SessionStatus,
    StepResult,
)
from rollout_runtime.api.payload_ref import PayloadRef
from rollout_runtime.api.result import Result

__all__ = ["Consistency", "RuntimeClient"]

Consistency = Literal["linearizable", "eventual"]
"""The consistency mode for ``observe``.

``linearizable`` must be ordered after already-accepted mutating commands
(takes a per-session lock); ``eventual`` allows skipping the lock and reading
a cached summary.
"""


@runtime_checkable
class RuntimeClient(Protocol):
    """The sole external interface of the Runtime."""

    async def create_sessions(
        self, requests: Sequence[CreateSessionRequest]
    ) -> list[Result[SessionHandle]]:
        """Create sessions in batch.

        Each session's handle is published only after all creation steps
        (including the binding returned by the EnvWorker) are complete.

        Args:
            requests: The creation requests, in the order corresponding to
                the return value.

        Returns:
            Per-item results in the same order as the input.
        """
        ...

    async def get_session(self, session_id: SessionId) -> SessionStatus:
        """Query the status of a session.

        Args:
            session_id: The target session.

        Returns:
            The logical lifecycle, lease, and a non-authoritative worker
            summary.
        """
        ...

    async def renew_sessions(
        self, session_ids: Sequence[SessionId], lease_seconds: float
    ) -> list[Result[SessionStatus]]:
        """Renew leases in batch.

        Args:
            session_ids: The target sessions.
            lease_seconds: The new lease length.

        Returns:
            Per-item results in the same order as the input.
        """
        ...

    async def reset(
        self,
        session_ids: Sequence[SessionId],
        reset_spec: ResetSpec,
        *,
        request_ids: Sequence[RequestId] | None = None,
    ) -> list[Result[StepResult]]:
        """Reset episodes in batch.

        Args:
            session_ids: The target sessions.
            reset_spec: The episode initialization parameters (the same for
                every session in this batch).
            request_ids: Idempotency keys; ``None`` means the implementation
                generates them.

        Returns:
            Per-item results in the same order as the input; successful
            items carry the new ``episode_id`` and the initial observation.
        """
        ...

    async def observe(
        self,
        session_ids: Sequence[SessionId],
        *,
        consistency: Consistency = "linearizable",
    ) -> list[Result[Observation]]:
        """Read the current observation in batch, without changing
        environment state.

        Args:
            session_ids: The target sessions.
            consistency: The consistency mode.

        Returns:
            Per-item results in the same order as the input.
        """
        ...

    async def action_step(
        self,
        session_ids: Sequence[SessionId],
        actions: Sequence[PayloadRef],
        *,
        request_ids: Sequence[RequestId] | None = None,
    ) -> list[Result[StepResult]]:
        """Execute externally supplied action chunks in batch.

        Args:
            session_ids: The target sessions.
            actions: Each session's ``[chunk, action_dim] float32`` payload.
            request_ids: Idempotency keys; ``None`` means the implementation
                generates them.

        Returns:
            Per-item results in the same order as the input.
        """
        ...

    async def policy_step(
        self,
        session_ids: Sequence[SessionId],
        policy_request: PolicyRequest,
        *,
        request_ids: Sequence[RequestId] | None = None,
    ) -> list[Result[StepResult]]:
        """Execute the atomic "observe -> infer -> chunk_step" operation in
        batch.

        This is a single atomic environment command to the Gateway: the
        Gateway never touches the RolloutWorker.

        Args:
            session_ids: The target sessions.
            policy_request: The inference parameters.
            request_ids: Idempotency keys; ``None`` means the implementation
                generates them.

        Returns:
            Per-item results in the same order as the input.
        """
        ...

    async def policy_infer(
        self,
        session_ids: Sequence[SessionId],
        policy_request: PolicyRequest,
        *,
        request_ids: Sequence[RequestId] | None = None,
    ) -> list[Result[PolicyInferResult]]:
        """Execute "observe -> infer" in batch, **without** executing
        ``chunk_step``.

        This is the first half of ``policy_step``, semantically equivalent
        to ``observe`` plus one model forward pass: it only reads the cached
        observation and does not change environment state, so it does not
        allocate an ``operation_seq`` and does not take the per-session
        mutation lock. After obtaining the action chunk, the caller can
        post-process it before handing it to ``action_step`` for execution
        -- this mirrors the shape of the legacy
        ``LiberoPrimitives._vlm_chunk`` (with the planner controlling
        ``translation_scale`` / ``action_clip`` / ``actions_per_chunk``),
        and is how "the primitive's loop and stop condition do not belong
        in the Runtime Core" is realized.

        Args:
            session_ids: The target sessions.
            policy_request: The inference parameters.
            request_ids: Idempotency keys; ``None`` means the implementation
                generates them.

        Returns:
            Per-item results in the same order as the input.
        """
        ...

    async def run_episode(
        self,
        session_ids: Sequence[SessionId],
        episode_request: EpisodeRequest,
        *,
        request_ids: Sequence[RequestId] | None = None,
    ) -> list[Result[EpisodeResult]]:
        """Execute a full episode in batch.

        The loop runs entirely inside the EnvWorker; transitions go through
        the sink and are not returned to the Gateway.

        Args:
            session_ids: The target sessions.
            episode_request: The episode parameters.
            request_ids: Idempotency keys; ``None`` means the implementation
                generates them.

        Returns:
            Per-item results in the same order as the input.
        """
        ...

    async def extension_call(
        self,
        session_ids: Sequence[SessionId],
        namespace: str,
        method: str,
        args: dict[str, Any],
        *,
        request_ids: Sequence[RequestId] | None = None,
    ) -> list[Result[dict[str, Any]]]:
        """Call a family-specific namespaced extension.

        An undeclared method returns ``UNSUPPORTED_EXTENSION`` rather than
        crashing.

        Args:
            session_ids: The target sessions.
            namespace: The extension namespace (e.g. ``"libero"``).
            method: The extension method name.
            args: The method arguments.
            request_ids: Idempotency keys; ``None`` means the implementation
                generates them.

        Returns:
            Per-item results in the same order as the input.
        """
        ...

    async def close_sessions(
        self, session_ids: Sequence[SessionId]
    ) -> list[Result[None]]:
        """Close sessions and release env slots in batch.

        Args:
            session_ids: The target sessions.

        Returns:
            Per-item results in the same order as the input.
        """
        ...

    async def get_request_status(self, request_id: RequestId) -> OperationStatus:
        """Query the status of an operation.

        Args:
            request_id: The request identifier.

        Returns:
            The operation status; the error behavior for an unknown id is
            up to the implementation.
        """
        ...

    async def cancel_request(self, request_id: RequestId) -> CancelOutcome:
        """Best-effort attempt to cancel an operation.

        An RPC timeout does not mean cancellation; an env step that has
        already started is not rolled back.

        Args:
            request_id: The request identifier.

        Returns:
            The cancellation result, including ``side_effect_applied``.
        """
        ...
