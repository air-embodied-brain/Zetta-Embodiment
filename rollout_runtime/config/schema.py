# Copyright (c) 2026 Zetta Contributors
"""Runtime configuration schema.

Dataclass schema + omegaconf loading. ``omegaconf`` is imported lazily, only
inside ``load_config``, so pure dataclass usage (tests, embedded mode) does
not depend on it.

Placement-related field values are constrained by the environment:
``PackedPlacementStrategy`` is unavailable with 0 local accelerators, and
``NodePlacementStrategy`` must be used instead.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

from rollout_runtime.gateway.admission import AdmissionConfig
from rollout_runtime.workers.batch_scheduler import SchedulerConfig

__all__ = [
    "ClusterConfig",
    "EnvWorkerConfig",
    "GatewayConfig",
    "PayloadConfig",
    "RolloutWorkerConfig",
    "RuntimeConfig",
    "TransportConfig",
    "load_config",
    "preset_path",
]


@dataclasses.dataclass(kw_only=True)
class ClusterConfig:
    """Ray cluster and placement configuration (used by ``launch/ray_launch.py``).

    The shape of ``component_placement`` is **identical** to rlinf's
    ``cfg.cluster.component_placement``, so it can be fed directly into
    ``HybridComponentPlacement`` (``rlinf/utils/placement.py:86``) without
    translating between two configuration formats:

    ```yaml
    cluster:
      num_nodes: 1
      component_placement:
        env:     {node_group: node, placement: "0-1:0-7"}
        rollout: {placement: "0-7"}
    ```

    Attributes:
        num_nodes: The number of cluster nodes; ``0`` means use every
            connected node.
        component_placement: The per-component placement declaration; when
            empty, each group follows its own ``placement_strategy`` via
            ``NodePlacementStrategy`` (the only strategy available with 0
            local accelerators).
    """

    num_nodes: int = 1
    component_placement: dict[str, Any] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass(kw_only=True)
class GatewayConfig:
    """Gateway configuration.

    Attributes:
        gateway_epoch: The Gateway epoch.
        maintenance_interval_seconds: The interval for lease reclamation and
            heartbeat inspection.
        default_lease_seconds: The default lease duration.
        error_retention_seconds: How long ``FAILED`` / ``LOST`` records are
            retained.
        result_ttl_seconds: How long operation results are cached.
        heartbeat_timeout_seconds: The worker heartbeat timeout.
        heartbeat_interval_seconds: The active probing interval. The Gateway
            sends a ``HEARTBEAT`` control message to each rank at this
            interval; a rank is only declared unreachable after consecutive
            failures reach ``heartbeat_timeout_seconds``. Must be
            significantly smaller than the timeout; ``0`` disables
            heartbeats (in which case ``heartbeat_at`` is only updated once
            at registration, and every rank ends up declared unreachable by
            the wall clock).
        max_concurrency_per_rank: The per-rank command concurrency limit.
        metrics_namespace: The prometheus metric prefix.
    """

    gateway_epoch: int = 1
    maintenance_interval_seconds: float = 1.0
    default_lease_seconds: float = 300.0
    error_retention_seconds: float = 300.0
    result_ttl_seconds: float = 300.0
    heartbeat_timeout_seconds: float = 30.0
    heartbeat_interval_seconds: float = 10.0
    max_concurrency_per_rank: int = 8
    metrics_namespace: str = "rr"


@dataclasses.dataclass(kw_only=True)
class TransportConfig:
    """Transport configuration.

    Attributes:
        kind: ``"inproc"`` or ``"ray_channel"`` (the default).
        command_queue_size: The command channel maxsize (a bounded queue is
            the source of backpressure).
        control_queue_size: The control channel maxsize.
        result_queue_size: The result flow-back channel maxsize.
        infer_request_queue_size: The ``rr_infer_req`` maxsize.
        infer_response_queue_size: The ``rr_infer_resp`` maxsize.
        command_timeout_seconds: The wait limit for a single command; a
            timeout only returns ``DEADLINE_EXCEEDED`` and does not cancel
            the environment operation.
    """

    kind: str = "inproc"
    command_queue_size: int = 64
    control_queue_size: int = 16
    result_queue_size: int = 64
    infer_request_queue_size: int = 64
    infer_response_queue_size: int = 64
    command_timeout_seconds: float = 120.0


@dataclasses.dataclass(kw_only=True)
class EnvWorkerConfig:
    """EnvWorker group configuration.

    Attributes:
        group_name: The worker group name (channel naming depends on it).
        num_ranks: The number of ranks.
        placement_strategy: ``"node"`` (local/CPU) or ``"packed"`` (multi-GPU).
        max_sessions_per_rank: The number of sessions a single rank can host.
        default_pool_size: The default for ``EnvSpecMsg.pool_size`` (design
            decision D6: v1 defaults to 1).
        seed_offset: The seed offset for this group.
        coalesce_slot_groups: Whether to enable real batching via
            ``SlotGroupCoalescer``. Only effective for pools declaring
            ``core_form="lockstep_vector"``; ``per_slot`` pools always take
            the single-item path.
        coalesce_window_ms: The upper bound (in milliseconds) of the
            batching window; since a pool fires as soon as all in-flight
            lanes are ready, this is only a timeout safeguard.
        has_accelerator: Whether ranks in this group occupy an accelerator.
            ``None`` means infer it from
            ``placement_strategy != "node"`` (``node`` is N processes on
            CPU). The Gateway uses it to block "a GPU-batched family landing
            on a rank without a card" — when ``EnvFamilyCapability
            .needs_accelerator`` is true but this is false, ``select_rank``
            simply will not pick that rank. This field must not be
            hardcoded to ``False``, otherwise environments requiring an
            accelerator would never get a rank.
    """

    group_name: str = "env"
    num_ranks: int = 1
    placement_strategy: str = "node"
    max_sessions_per_rank: int = 8
    default_pool_size: int = 1
    seed_offset: int = 0
    coalesce_slot_groups: bool = True
    coalesce_window_ms: float = 200.0
    has_accelerator: bool | None = None

    def accelerator_present(self) -> bool:
        """Whether ranks in this group occupy an accelerator.

        Returns:
            The explicit declaration takes priority; otherwise it is
            inferred from ``placement_strategy != "node"``.
        """
        if self.has_accelerator is not None:
            return bool(self.has_accelerator)
        return self.placement_strategy != "node"


@dataclasses.dataclass(kw_only=True)
class RolloutWorkerConfig:
    """RolloutWorker group configuration.

    Attributes:
        group_name: The worker group name.
        num_ranks: The number of ranks (one GPU per rank, one full VLA copy).
        placement_strategy: ``"node"`` or ``"packed"``.
        tensor_parallel_size: The tensor parallel degree; fixed at 1 for v1.
        pipeline_parallel_size: The pipeline parallel degree; fixed at 1 for v1.
        policy_id: The default policy identifier.
        policy_family: The policy family, which decides the allowed
            mixed-batching parameter set.
        policy_backend: ``"fake"``, ``"zetta_openpi"``, ``"groot"``, or
            ``"cosmos_lite"``.
            Both launchers read this via ``backends.build_policy_core``.
        policy_config: Backend-private policy configuration (for the
            ``rlinf`` backend, these are ``RlinfPolicyConfig``'s fields:
            ``model_path`` / ``num_action_chunks`` / ``family_params`` /
            ``enable_cuda_graph`` ...).
        device: The device identifier, feeding into ``compat_key``.
        dtype: The compute precision, feeding into ``compat_key``.
        max_concurrent_inferences: The upper bound on simultaneously
            in-flight inference requests for a single rank. This is the
            in-flight gate, also the upstream source of backpressure
            (requests within the limit are picked up immediately; excess
            ones stay in the bounded ``rr_infer_req``); the batching
            semantics are taken over by ``InferenceBatchScheduler``'s
            ``max_batch_size``.
        scheduler: The batch scheduler configuration.
    """

    group_name: str = "rollout"
    num_ranks: int = 1
    placement_strategy: str = "node"
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    policy_id: str = "fake"
    policy_family: str = "fake"
    policy_backend: str = "fake"
    policy_config: dict[str, Any] = dataclasses.field(default_factory=dict)
    device: str = "cpu"
    dtype: str = "float32"
    max_concurrent_inferences: int = 8
    scheduler: SchedulerConfig = dataclasses.field(default_factory=SchedulerConfig)


@dataclasses.dataclass(kw_only=True)
class PayloadConfig:
    """Payload budget configuration.

    Attributes:
        inline_threshold_bytes: Exceeding this counts as an oversize
            warning, but v1 still inlines it.
        request_limit_bytes: The hard per-request limit; exceeding it raises
            ``INVALID_ARGUMENT``.
        image_codec: ``"png"`` or ``"raw"``.
    """

    inline_threshold_bytes: int = 256 * 1024
    request_limit_bytes: int = 8 * 1024 * 1024
    image_codec: str = "png"


@dataclasses.dataclass(kw_only=True)
class RuntimeConfig:
    """Top-level runtime configuration.

    Attributes:
        env_family: The default env family.
        env_config: The default family configuration.
        cluster: The Ray cluster and placement configuration.
        gateway: The Gateway configuration.
        transport: The transport configuration.
        env_worker: The EnvWorker group configuration.
        rollout_worker: The RolloutWorker group configuration.
        admission: The admission configuration.
        payload: The payload budget configuration.
    """

    env_family: str = "fake"
    env_config: dict[str, Any] = dataclasses.field(default_factory=dict)
    cluster: ClusterConfig = dataclasses.field(default_factory=ClusterConfig)
    gateway: GatewayConfig = dataclasses.field(default_factory=GatewayConfig)
    transport: TransportConfig = dataclasses.field(default_factory=TransportConfig)
    env_worker: EnvWorkerConfig = dataclasses.field(default_factory=EnvWorkerConfig)
    rollout_worker: RolloutWorkerConfig = dataclasses.field(
        default_factory=RolloutWorkerConfig
    )
    admission: AdmissionConfig = dataclasses.field(default_factory=AdmissionConfig)
    payload: PayloadConfig = dataclasses.field(default_factory=PayloadConfig)


def preset_path(name: str) -> Path:
    """Return the path to a built-in preset.

    Args:
        name: The preset file name (the ``.yaml`` extension may be omitted).

    Returns:
        The preset file path.

    Raises:
        FileNotFoundError: The preset does not exist.
    """
    filename = name if name.endswith((".yaml", ".yml")) else f"{name}.yaml"
    path = Path(__file__).parent / "presets" / filename
    if not path.is_file():
        available = sorted(
            item.name for item in (Path(__file__).parent / "presets").glob("*.yaml")
        )
        raise FileNotFoundError(
            f"unknown runtime preset {name!r}; available: {available}"
        )
    return path


def load_config(source: str | Path | dict[str, Any] | None = None) -> RuntimeConfig:
    """Load configuration from a preset name, a YAML path, or a dict.

    Args:
        source: A preset name / file path / override dict; ``None`` returns
            the default configuration.

    Returns:
        The structured configuration.
    """
    if source is None:
        return RuntimeConfig()

    from omegaconf import OmegaConf

    schema = OmegaConf.structured(RuntimeConfig)
    # The config dataclass is deliberately **not** frozen: omegaconf 2.3.1
    # marks a frozen structured config as read-only, and merge would
    # immediately raise ReadonlyConfigError, plus it cannot clear the
    # read-only marker on leaf nodes. Read-only is still explicitly cleared
    # here once, for compatibility with any third-party frozen config block
    # that might get merged in later.
    OmegaConf.set_readonly(schema, False)
    if isinstance(source, dict):
        overrides = OmegaConf.create(source)
    else:
        path = Path(source)
        if not path.exists():
            path = preset_path(str(source))
        overrides = OmegaConf.load(path)
    merged = OmegaConf.merge(schema, overrides)
    config: RuntimeConfig = OmegaConf.to_object(merged)  # type: ignore[assignment]
    _validate(config)
    return config


def _validate(config: RuntimeConfig) -> None:
    """Load-time validation for combinations that "would silently become a correctness issue if misconfigured".

    Currently there is only one rule: the heartbeat interval must be
    significantly smaller than the heartbeat timeout. A real defect measured
    on a multi-GPU host: ``local_fake`` (5 s timeout / 1 s interval) paired
    with a **too-long liveness probing window** meant a single hung rank
    could stretch the heartbeat cadence past the timeout, causing **a
    healthy rank to also be declared unreachable**, with its sessions
    transitioning to ``LOST`` (which is unrecoverable). The probing window
    itself has already been self-clamped to ``timeout/3`` (see
    ``RuntimeGateway.__init__``); this rule additionally blocks the
    obviously-wrong "interval >= timeout" configuration: matching the same
    posture as `serve`'s "refuse to start when the token is missing" —
    better to fail loudly at load time than to silently declare things dead
    at runtime.

    Args:
        config: The merged configuration.

    Raises:
        ValueError: ``heartbeat_interval_seconds`` is not significantly
            smaller than ``heartbeat_timeout_seconds`` (must be at most a
            third of the timeout, leaving room for three probes).
    """
    gateway = config.gateway
    interval = gateway.heartbeat_interval_seconds
    timeout = gateway.heartbeat_timeout_seconds
    if interval > 0 and interval * 3 > timeout:
        raise ValueError(
            "gateway.heartbeat_interval_seconds must be at most a third of "
            f"gateway.heartbeat_timeout_seconds (got interval={interval}, "
            f"timeout={timeout}): a single missed probe would otherwise be enough to "
            "declare a live rank lost, and LOST sessions are not recoverable"
        )
    rollout = config.rollout_worker
    if rollout.policy_backend == "cosmos_lite":
        if rollout.policy_family != "cosmos_lite":
            raise ValueError(
                "cosmos_lite backend requires rollout_worker.policy_family="
                "'cosmos_lite'"
            )
        if rollout.num_ranks != 1:
            raise ValueError("initial cosmos_lite backend requires num_ranks=1")
        if rollout.max_concurrent_inferences != 1:
            raise ValueError(
                "initial cosmos_lite backend requires max_concurrent_inferences=1"
            )
        if rollout.scheduler.max_batch_size != 1:
            raise ValueError(
                "initial cosmos_lite backend requires scheduler.max_batch_size=1"
            )
        if rollout.scheduler.max_wait_ms != 0:
            raise ValueError(
                "initial cosmos_lite backend requires scheduler.max_wait_ms=0"
            )
