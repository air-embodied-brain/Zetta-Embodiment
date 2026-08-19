"""Runtime-internal messages.

These messages are not exposed to upstream applications: ``CommandEnvelope``
/ ``ControlEnvelope`` / ``ResultEnvelope`` travel over the Gateway <->
EnvWorker transport, while ``InferenceRequest`` / ``ActionResponse`` travel
over the EnvWorker <-> RolloutWorker request plane.
"""

from __future__ import annotations

import dataclasses
from typing import Any

from rollout_runtime.api import codec
from rollout_runtime.api.enums import EnvOperation, OperationState, Priority
from rollout_runtime.api.errors import RuntimeErrorInfo
from rollout_runtime.api.ids import (
    BindingToken,
    EpisodeId,
    OperationSeq,
    RequestId,
    SessionId,
)
from rollout_runtime.api.messages import Observation, WorkerSummary
from rollout_runtime.api.payload_ref import PayloadRef

__all__ = [
    "ActionResponse",
    "CommandEnvelope",
    "ControlEnvelope",
    "InferenceRequest",
    "ResultEnvelope",
    "make_routing_token",
    "parse_routing_token",
]


def make_routing_token(group_name: str, worker_rank: int) -> str:
    """Construct the reply-routing identifier for an EnvWorker.

    Args:
        group_name: The EnvWorker group name.
        worker_rank: The EnvWorker rank.

    Returns:
        A string of the form ``"{group_name}:{rank}"``.
    """
    return f"{group_name}:{worker_rank}"


def parse_routing_token(token: str) -> tuple[str, int]:
    """Parse a reply-routing identifier.

    Args:
        token: The output of ``make_routing_token``.

    Returns:
        A ``(group_name, worker_rank)`` tuple.

    Raises:
        ValueError: The format is malformed.
    """
    group_name, _, rank_text = token.rpartition(":")
    if not group_name or not rank_text.isdigit():
        raise ValueError(f"malformed routing token: {token!r}")
    return group_name, int(rank_text)


@dataclasses.dataclass(frozen=True, kw_only=True)
class CommandEnvelope:
    """A single-session command sent from the Gateway to an EnvWorker.

    ``payload`` carries operation-specific parameters (``ResetSpec`` /
    ``PolicyRequest`` / actions, etc.); these values must be either
    registered protocol dataclasses or msgpack-native types.

    Attributes:
        request_id: The idempotency key.
        session_id: The target session.
        binding_token: The binding identifier recorded by the Gateway; the
            EnvWorker side re-validates it (mismatch -> ``STALE_BINDING``).
        episode_id: The expected episode; a response with an episode less
            than the worker's current value is dropped and counted.
        operation_seq: The mutating-operation sequence number; ``None`` for
            read-only operations.
        operation: The operation type.
        deadline: An absolute unix timestamp.
        priority: The scheduling priority.
        payload: The operation parameters.
        trace_context: The forwarded tracing context.
    """

    request_id: RequestId
    session_id: SessionId
    binding_token: BindingToken | None = None
    episode_id: EpisodeId | None = None
    operation_seq: OperationSeq | None = None
    operation: EnvOperation
    deadline: float | None = None
    priority: Priority = Priority.INTERACTIVE
    payload: dict[str, Any] = dataclasses.field(default_factory=dict)
    trace_context: dict[str, Any] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass(frozen=True, kw_only=True)
class ControlEnvelope:
    """A message that travels over the independent high-priority control
    channel.

    Cancel / heartbeat / shutdown / binding management must not be blocked
    behind ordinary environment commands.

    Attributes:
        request_id: The idempotency key.
        operation: The control operation type.
        session_id: The target session (``None`` for heartbeat-like
            operations).
        payload: The control parameters.
        deadline: An absolute unix timestamp.
        trace_context: The forwarded tracing context.
    """

    request_id: RequestId
    operation: EnvOperation
    session_id: SessionId | None = None
    payload: dict[str, Any] = dataclasses.field(default_factory=dict)
    deadline: float | None = None
    trace_context: dict[str, Any] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass(frozen=True, kw_only=True)
class ResultEnvelope:
    """A worker's response to a command / control message.

    Workers never raise; failures always go through the ``error`` field.

    Attributes:
        request_id: The corresponding request.
        session_id: The owning session.
        operation: The operation type.
        state: The terminal state of the operation (or ``RUNNING`` meaning
            accepted but not yet complete).
        value: The operation payload (``StepResult`` / ``Observation`` /
            ``EpisodeResult`` / dict, etc.).
        error: The normalized error.
        side_effect_applied: Whether the environment side effect has already
            occurred.
        worker_summary: A non-authoritative worker summary.
    """

    request_id: RequestId
    session_id: SessionId | None = None
    operation: EnvOperation
    state: OperationState = OperationState.SUCCEEDED
    value: Any = None
    error: RuntimeErrorInfo | None = None
    side_effect_applied: bool = False
    worker_summary: WorkerSummary | None = None

    @property
    def ok(self) -> bool:
        """Whether this is a success.

        Returns:
            ``True`` when ``error`` is empty and the state is ``SUCCEEDED``.
        """
        return self.error is None and self.state is OperationState.SUCCEEDED


@dataclasses.dataclass(frozen=True, kw_only=True)
class InferenceRequest:
    """An inference request sent from an EnvWorker to a RolloutWorker.

    Attributes:
        request_id: The idempotency key (same id as the env command that
            triggered it).
        session_id: The owning session.
        binding_token: The binding identifier.
        episode_id: The owning episode.
        operation_seq: The operation sequence number that triggered it.
        policy_id: The target policy.
        model_version_hint: The expected model version; ``None`` means any
            current version is acceptable.
        observation: The inference input.
        instruction_override: An override instruction.
        inference_parameters: The inference parameters (normalized before
            entering ``compat_key``).
        routing_token: The reply route (``make_routing_token``).
        compat_key: The batching compatibility key (see
            ``core.policy_inference``).
        deadline: An absolute unix timestamp.
        priority: The scheduling priority.
        application_id: The tenant identifier, used for scheduler fairness
            limits.
    """

    request_id: RequestId
    session_id: SessionId
    binding_token: BindingToken | None = None
    episode_id: EpisodeId | None = None
    operation_seq: OperationSeq | None = None
    policy_id: str
    model_version_hint: str | None = None
    observation: Observation
    instruction_override: str | None = None
    inference_parameters: dict[str, Any] = dataclasses.field(default_factory=dict)
    routing_token: str
    compat_key: str
    deadline: float | None = None
    priority: Priority = Priority.INTERACTIVE
    application_id: str = ""


@dataclasses.dataclass(frozen=True, kw_only=True)
class ActionResponse:
    """The action chunk returned from a RolloutWorker to an EnvWorker.

    Attributes:
        request_id: The corresponding request.
        session_id: The owning session.
        binding_token: The binding identifier, used for late-response
            detection.
        episode_id: The owning episode, used for late-response detection.
        operation_seq: The corresponding operation sequence number.
        actions: The ``[chunk, action_dim] float32`` payload.
        model_version: The model version actually used, tagged per request
            (the version fence).
        auxiliary_outputs: Additional outputs such as value / logprob.
        error: The normalized error.
    """

    request_id: RequestId
    session_id: SessionId
    binding_token: BindingToken | None = None
    episode_id: EpisodeId | None = None
    operation_seq: OperationSeq | None = None
    actions: PayloadRef | None = None
    model_version: str = ""
    auxiliary_outputs: dict[str, Any] = dataclasses.field(default_factory=dict)
    error: RuntimeErrorInfo | None = None


codec.register_messages(globals())
