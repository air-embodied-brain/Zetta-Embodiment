"""External-facing protocol messages of the Runtime API.

All are ``@dataclass(frozen=True, kw_only=True)`` and depend only on stdlib.
Byte-level serialization is in ``api.wire``; structural encoding/decoding is
in ``api.codec``.
"""

from __future__ import annotations

import dataclasses
from typing import Any

from rollout_runtime.api import codec
from rollout_runtime.api.enums import EnvOperation, OperationState, SessionState
from rollout_runtime.api.errors import RuntimeErrorInfo
from rollout_runtime.api.ids import (
    BindingToken,
    EpisodeId,
    OperationSeq,
    RequestId,
    SessionId,
)
from rollout_runtime.api.payload_ref import PayloadRef

__all__ = [
    "CancelOutcome",
    "CreateSessionRequest",
    "EnvFamilyCapability",
    "EnvSpecMsg",
    "EnvWorkerInfo",
    "EpisodeRequest",
    "EpisodeResult",
    "Observation",
    "OperationStatus",
    "PerStepRecord",
    "PolicyInferResult",
    "PolicyRequest",
    "ResetSpec",
    "SessionHandle",
    "SessionStatus",
    "StepResult",
    "WorkerSummary",
]

ENV_SPEC_DIGEST_DOMAIN = "rollout_runtime/env_spec/v1"
"""The domain-separation prefix for ``EnvSpecMsg.digest()``, to avoid
collisions with other digest spaces."""


@dataclasses.dataclass(frozen=True, kw_only=True)
class EnvSpecMsg:
    """The environment specification.

    ``env_config`` is interpreted by the ``EnvFamilyAdapter``; the Gateway
    **does not interpret its contents**, only using the digest for pool
    reuse and compatibility grouping.

    Attributes:
        env_family: The family name (``"libero"`` / ``"maniskill"`` /
            ``"robotwin"`` / ``"fake"`` etc.).
        env_config: Family-private configuration passed straight through to
            the adapter.
        pool_size: The initial slot count of the env pool for this spec
            (fixed at construction time for the ``lockstep_vector`` form and
            never grows dynamically; for the ``per_slot`` form this is the
            initial value, which can be dynamically scaled within
            ``[pool_size, max_dynamic_pool_size]`` as needed, see
            ``core.env_execution.DynamicSlotPool``).
        max_dynamic_pool_size: The upper bound on the slot count the
            ``per_slot`` form is allowed to scale up to. Defaults to
            ``pool_size``, i.e. dynamic scaling is disabled by default,
            preserving the fixed-pool behavior; only setting it explicitly
            above ``pool_size`` enables on-demand scaling. Has no effect for
            an execution core that does not provide the ``DynamicSlotPool``
            capability (including every ``lockstep_vector`` core); the pool
            still stays fixed at ``pool_size``.
            **Do not confuse this with** ``serve.app.ServeLimits.max_pool_size``:
            the latter is a hard clamp the server applies to the
            client-declared ``pool_size`` (to prevent a single create from
            spinning up hundreds of simulator processes); in the served
            form, ``_with_identity`` clamps this field with the same bound.
        resource_hints: Placement hints such as
            ``{"node_group": ..., "accelerator": bool}``.
    """

    env_family: str
    env_config: dict[str, Any] = dataclasses.field(default_factory=dict)
    pool_size: int = 1
    max_dynamic_pool_size: int | None = None
    resource_hints: dict[str, Any] = dataclasses.field(default_factory=dict)

    def effective_max_pool_size(self) -> int:
        """Resolve the dynamic scaling upper bound, falling back to
        ``pool_size`` when unset (no scaling).

        Returns:
            The effective slot-count upper bound, always ``>= pool_size``.
        """
        if self.max_dynamic_pool_size is None:
            return self.pool_size
        return max(self.pool_size, self.max_dynamic_pool_size)

    def digest(self) -> str:
        """Return a stable env spec digest.

        Only covers the fields that determine whether the "same env pool
        can be shared": the family, the family config, and the initial pool
        size. ``max_dynamic_pool_size`` and ``resource_hints`` are not part
        of the digest: the former only affects whether the pool can scale at
        runtime, not what the current pool looks like (a scaled-up pool
        still serves the same batch of sessions); the latter only affects
        placement.

        Returns:
            A 64-character hex sha256 digest.
        """
        return codec.digest(
            {
                "env_family": self.env_family,
                "env_config": self.env_config,
                "pool_size": self.pool_size,
            },
            prefix=ENV_SPEC_DIGEST_DOMAIN,
        )


@dataclasses.dataclass(frozen=True, kw_only=True)
class EnvFamilyCapability:
    """An env family's declared capabilities.

    The Gateway validates against this at ``create_session`` time, returning
    ``UNSUPPORTED_ENV_SPEC`` for unsupported families instead of failing at
    runtime.

    Attributes:
        env_family: The family name.
        per_step_obs_available: Whether ``chunk_step`` returns a per-step
            observation (robotwin is the "no" case).
        supports_auto_reset: Whether the environment auto-resets after
            termination.
        supports_reset_state_id: Whether ``ResetSpec.reset_state_id`` is
            supported.
        extensions: The set of supported ``extension_call`` method full
            names (``"ns.method"``).
        needs_accelerator: Whether this must be placed on a rank with an
            accelerator.
        core_forms: The execution-core forms this family can build (a
            subset of ``core.env_execution.CORE_FORMS``).
        supports_coalescing: Whether it can build a vectorized-form pool
            (i.e. whether ``SlotGroupCoalescer`` could ever truly batch it).
    """

    env_family: str
    per_step_obs_available: bool = True
    supports_auto_reset: bool = False
    supports_reset_state_id: bool = False
    extensions: frozenset[str] = frozenset()
    needs_accelerator: bool = False
    core_forms: frozenset[str] = frozenset({"per_slot"})
    supports_coalescing: bool = False


@dataclasses.dataclass(frozen=True, kw_only=True)
class CreateSessionRequest:
    """A session creation request.

    Attributes:
        application_id: The owner for auth, quota, metrics, and audit.
        client_session_key: The caller-supplied idempotency key, unique
            within the application.
        env_spec: The environment specification.
        default_policy_id: Used when ``policy_step`` does not specify a
            policy.
        lease_seconds: The lease length.
        metadata: Pass-through fields such as ``tool_call_id`` /
            ``trace_id``.
        auth_token: The auth token; parsed from the ``Authorization`` header
            by the served form.
    """

    application_id: str
    client_session_key: str
    env_spec: EnvSpecMsg
    default_policy_id: str | None = None
    lease_seconds: float = 300.0
    metadata: dict[str, Any] = dataclasses.field(default_factory=dict)
    auth_token: str | None = None


@dataclasses.dataclass(frozen=True, kw_only=True)
class SessionHandle:
    """The externally published session handle.

    Explicitly **does not include** the EnvWorker rank / env slot /
    ``binding_token``: those are internal routing details, to avoid a
    worker restart or placement change breaking the upstream protocol.

    Attributes:
        session_id: The session identifier.
        application_id: The owning application.
        env_spec_digest: The environment spec digest.
        default_policy_id: The default policy.
        lease_expiration: The lease expiration unix timestamp.
        gateway_epoch: The Gateway instance epoch, used to identify a
            Gateway restart.
    """

    session_id: SessionId
    application_id: str
    env_spec_digest: str
    default_policy_id: str | None
    lease_expiration: float
    gateway_epoch: int


@dataclasses.dataclass(frozen=True, kw_only=True)
class WorkerSummary:
    """A status summary from the EnvWorker side.

    Non-authoritative: the Gateway can only cache the worker's most recent
    summary; the sole source of truth for environment execution state is
    always the EnvWorker.

    Attributes:
        worker_rank: The EnvWorker rank.
        group_name: The worker group name.
        pool_key: The env pool key (usually the env spec digest).
        slot_index: The env slot index.
        step_index: The number of steps executed in the current episode.
        terminated: The termination flag from the most recent step.
        truncated: The truncation flag from the most recent step.
        updated_at: When the summary was generated.
    """

    worker_rank: int
    group_name: str = ""
    pool_key: str = ""
    slot_index: int = -1
    step_index: int = 0
    terminated: bool = False
    truncated: bool = False
    updated_at: float = 0.0


@dataclasses.dataclass(frozen=True, kw_only=True)
class SessionStatus:
    """The result of a session query.

    Attributes:
        session_id: The session identifier.
        state: The Gateway-side logical lifecycle.
        lease_expiration: The lease expiration time.
        episode_id: The current episode (``None`` if not yet reset).
        active_operation: The operation currently executing.
        next_operation_seq: The next ``operation_seq`` to be assigned.
        worker_summary: A non-authoritative worker summary, for observation
            only.
        error: The retained error for ``FAILED`` / ``LOST``.
    """

    session_id: SessionId
    state: SessionState
    lease_expiration: float
    episode_id: EpisodeId | None = None
    active_operation: RequestId | None = None
    next_operation_seq: OperationSeq = OperationSeq(1)
    worker_summary: WorkerSummary | None = None
    error: RuntimeErrorInfo | None = None


@dataclasses.dataclass(frozen=True, kw_only=True)
class ResetSpec:
    """Episode initialization parameters.

    Attributes:
        task_id: The task index.
        seed: The random seed.
        instruction: An override task instruction.
        reset_state_id: The initial state to specify, when the family
            supports it (LIBERO's ``specific_reset_id``).
        options: The remaining family-private reset parameters.
    """

    task_id: int | None = None
    seed: int | None = None
    instruction: str | None = None
    reset_state_id: int | None = None
    options: dict[str, Any] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass(frozen=True, kw_only=True)
class PolicyRequest:
    """Parameters for one policy inference call.

    Attributes:
        policy_id: The target policy; ``None`` uses the session's
            ``default_policy_id``.
        instruction_override: Overrides the instruction in the observation.
        inference_parameters: ``{"num_steps": ..., "mode": "eval", ...}``;
            after normalization, participates in ``compat_key`` (see
            ``core.policy_inference``).
        actions_per_chunk: The requested action-chunk length.
        action_postprocess: ``{"translation_scale": .., "action_clip": ..}``.
    """

    policy_id: str | None = None
    instruction_override: str | None = None
    inference_parameters: dict[str, Any] = dataclasses.field(default_factory=dict)
    actions_per_chunk: int | None = None
    action_postprocess: dict[str, Any] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass(frozen=True, kw_only=True)
class EpisodeRequest:
    """A request to execute a full episode.

    Attributes:
        max_steps: The maximum number of policy_step calls.
        policy: The inference parameters.
        deadline: An absolute unix timestamp deadline.
        sink_id: The transition sink identifier; transitions are not
            returned to the Gateway.
        stop_on_truncation: Whether truncation also ends the episode.
    """

    max_steps: int
    policy: PolicyRequest = dataclasses.field(default_factory=PolicyRequest)
    deadline: float | None = None
    sink_id: str | None = None
    stop_on_truncation: bool = True


@dataclasses.dataclass(frozen=True, kw_only=True)
class Observation:
    """The unified observation schema (aligned with
    ``EnvOutput.prepare_observations``).

    Attributes:
        session_id: The owning session.
        episode_id: The owning episode.
        step_index: The step index this observation corresponds to.
        main_image: The main-view image (uint8 HWC).
        wrist_image: The wrist-camera image.
        extra_view_images: Images from the remaining views.
        state: The robot state vector.
        instruction: The task instruction.
        extras: Family-private supplementary fields.
    """

    session_id: SessionId
    episode_id: EpisodeId
    step_index: int
    main_image: PayloadRef | None = None
    wrist_image: PayloadRef | None = None
    extra_view_images: list[PayloadRef] = dataclasses.field(default_factory=list)
    state: list[float] = dataclasses.field(default_factory=list)
    instruction: str = ""
    extras: dict[str, Any] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass(frozen=True, kw_only=True)
class PerStepRecord:
    """A single-step record within a chunk.

    Only meaningful when the family's ``per_step_obs_available`` is true.

    Attributes:
        step_index: The step index.
        reward: The reward for this step.
        terminated: The termination flag for this step.
        truncated: The truncation flag for this step.
        observation: The observation for this step (may be omitted to save
            bandwidth).
        info: Info for this step.
    """

    step_index: int
    reward: float = 0.0
    terminated: bool = False
    truncated: bool = False
    observation: Observation | None = None
    info: dict[str, Any] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass(frozen=True, kw_only=True)
class StepResult:
    """The result of one environment-changing operation.

    Attributes:
        request_id: The corresponding request.
        session_id: The owning session.
        binding_token: The binding identifier at execution time.
        episode_id: The episode at execution time.
        operation_seq: The operation sequence number at execution time.
        observation: The observation after execution.
        reward: The cumulative reward for the chunk.
        terminated: The termination flag.
        truncated: The truncation flag.
        info: Family-private information.
        side_effect_applied: Whether the environment side effect has
            already occurred.
        executed_horizon: The number of environment steps actually
            executed.
        per_step: Per-step records; ``None`` for the robotwin family.
        error: The normalized error on failure.
    """

    request_id: RequestId
    session_id: SessionId
    binding_token: BindingToken | None = None
    episode_id: EpisodeId | None = None
    operation_seq: OperationSeq | None = None
    observation: Observation | None = None
    reward: float = 0.0
    terminated: bool = False
    truncated: bool = False
    info: dict[str, Any] = dataclasses.field(default_factory=dict)
    side_effect_applied: bool = False
    executed_horizon: int = 0
    per_step: list[PerStepRecord] | None = None
    error: RuntimeErrorInfo | None = None


@dataclasses.dataclass(frozen=True, kw_only=True)
class PolicyInferResult:
    """The result of ``policy_infer``: runs inference only, without touching
    the environment.

    This is the **first half** of ``policy_step``. It exists because the
    legacy Agent-side seam must be able to proceed in two steps: first get
    the action chunk so ``LiberoPrimitives`` can apply planner-controlled
    post-processing (``translation_scale`` / ``action_clip`` /
    ``actions_per_chunk``), then hand the **post-processed** actions to
    ``action_step`` for execution. Combining this into a single
    ``policy_step`` would bypass this post-processing (pi0.5's output has
    indeed been observed to exceed ``[-1, 1]``, with a minimum around
    ``-1.0072``), which is why this read-only operation was added instead of
    pushing the primitive's loop down into the Runtime (explicitly excluded
    from v1).

    Attributes:
        request_id: The corresponding request.
        session_id: The owning session.
        binding_token: The binding identifier at execution time.
        episode_id: The episode at execution time.
        actions: The ``[chunk, action_dim] float32`` action-chunk payload.
        model_version: The model version that produced this action chunk.
        auxiliary_outputs: Additional model outputs (value / logprob etc.,
            family-private).
        observation_step_index: The ``step_index`` of the observation used
            for inference.
        info: Additional information (``policy_id`` etc.).
        error: The normalized error on failure.
    """

    request_id: RequestId
    session_id: SessionId
    binding_token: BindingToken | None = None
    episode_id: EpisodeId | None = None
    actions: PayloadRef | None = None
    model_version: str = ""
    auxiliary_outputs: dict[str, Any] = dataclasses.field(default_factory=dict)
    observation_step_index: int = 0
    info: dict[str, Any] = dataclasses.field(default_factory=dict)
    error: RuntimeErrorInfo | None = None


@dataclasses.dataclass(frozen=True, kw_only=True)
class EpisodeResult:
    """The result of ``run_episode``.

    Transitions go through the sink and are not returned with the result,
    so the Gateway does not need to store the trajectory.

    Attributes:
        request_id: The corresponding request.
        session_id: The owning session.
        episode_id: The owning episode.
        num_policy_steps: The number of policy_step calls executed.
        executed_horizon: The cumulative number of environment steps.
        total_reward: The cumulative reward.
        terminated: The termination flag.
        truncated: The truncation flag.
        stop_reason: ``"terminated"`` / ``"truncated"`` / ``"max_steps"`` /
            ``"deadline"`` / ``"cancelled"`` / ``"error"``.
        sink_id: The transition sink identifier.
        last_observation: The final observation.
        info: Aggregated information.
        error: The normalized error on failure.
    """

    request_id: RequestId
    session_id: SessionId
    episode_id: EpisodeId | None = None
    num_policy_steps: int = 0
    executed_horizon: int = 0
    total_reward: float = 0.0
    terminated: bool = False
    truncated: bool = False
    stop_reason: str = ""
    sink_id: str | None = None
    last_observation: Observation | None = None
    info: dict[str, Any] = dataclasses.field(default_factory=dict)
    error: RuntimeErrorInfo | None = None


@dataclasses.dataclass(frozen=True, kw_only=True)
class OperationStatus:
    """The return value of ``get_request_status``.

    Attributes:
        request_id: The request identifier.
        session_id: The owning session.
        operation: The operation type.
        state: The operation state.
        created_at: When it was registered.
        updated_at: When it was last updated.
        side_effect_applied: Whether the environment side effect has
            already occurred.
        error: The normalized error on failure.
    """

    request_id: RequestId
    session_id: SessionId | None
    operation: EnvOperation
    state: OperationState
    created_at: float = 0.0
    updated_at: float = 0.0
    side_effect_applied: bool = False
    error: RuntimeErrorInfo | None = None


@dataclasses.dataclass(frozen=True, kw_only=True)
class CancelOutcome:
    """The return value of ``cancel_request`` (the four cancellation states).

    Attributes:
        request_id: The request identifier.
        state: The operation state after cancellation.
        side_effect_applied: Whether the environment side effect has
            already occurred.
        message: An explanatory message.
    """

    request_id: RequestId
    state: OperationState
    side_effect_applied: bool = False
    message: str = ""


@dataclasses.dataclass(frozen=True, kw_only=True)
class EnvWorkerInfo:
    """Capabilities and capacity an EnvWorker registers with the Gateway
    (``register_env_worker``).

    Attributes:
        worker_rank: The rank.
        group_name: The worker group name.
        node_id: The hosting node identifier, used for locality selection.
        capabilities: A mapping from family name to capability declaration.
        served_env_digests: The env spec digests for which a pool has
            already been built (basis for stickiness and reuse).
        max_sessions: The maximum number of sessions this rank can host.
        active_sessions: The current number of sessions.
        has_accelerator: Whether an accelerator is visible.
        heartbeat_at: The time of the most recent heartbeat.
    """

    worker_rank: int
    group_name: str = ""
    node_id: str = ""
    capabilities: dict[str, EnvFamilyCapability] = dataclasses.field(
        default_factory=dict
    )
    served_env_digests: list[str] = dataclasses.field(default_factory=list)
    max_sessions: int = 1
    active_sessions: int = 0
    has_accelerator: bool = False
    heartbeat_at: float = 0.0


codec.register_messages(globals())
