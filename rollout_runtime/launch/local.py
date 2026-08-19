"""Local launcher, supporting two data planes.

Packs the Gateway, the EnvWorker group, and the RolloutWorker group into
**a single process, a single event loop**. ``transport.kind`` decides the
data plane:

- ``inproc``: ``InProcTransport`` + ``InProcInferenceChannel``, pure asyncio;
- ``ray_channel``: a real rlinf ``Channel`` actor (command / control /
  result / inference request / inference response, five in total) —
  **worker objects still live in this process** — so the e2e assertions can
  be re-run unchanged under real Channel semantics, while tests can still
  reach the fake backend directly to inject latency and faults.

Topology (``num_ranks`` comes from config, defaulting to 1x1 locally):

```text
RuntimeGateway ──transport──> RuntimeEnvWorker[rank]
                                     │  put rr_infer_req["pending"]
                                     v
                            InferenceChannel (inproc or Channel)
                                     │  get (multiple rollout ranks compete = work stealing)
                                     v
                            RuntimeRolloutWorker[rank] ──put rr_infer_resp[routing_token]──┐
                                     ^                                                     │
                                     └──────────── EnvWorker's drain wakes the Future ◀────┘
```

See ``launch/ray_launch.py`` for the form where workers actually run in
separate Ray processes.
"""

from __future__ import annotations

import contextlib
import dataclasses
from typing import Any

from rollout_runtime.backends import (
    build_policy_core,
    policy_compat_constraints,
    register_env_family_for,
)
from rollout_runtime.config.schema import RuntimeConfig, load_config
from rollout_runtime.gateway.gateway import RuntimeGateway
from rollout_runtime.transport.inproc import InProcInferenceChannel, InProcTransport
from rollout_runtime.workers.env_worker import RuntimeEnvWorker
from rollout_runtime.workers.rollout_worker import RuntimeRolloutWorker

__all__ = ["LocalRuntime", "build_local_runtime", "build_local_components"]


def _payload_stats_snapshot() -> dict[str, int]:
    """Flatten ``core.payload.stats()`` into a field dictionary (for the Gateway's metric sampling).

    ``core.payload`` is imported here rather than in ``gateway/``: that would
    drag numpy into the Gateway's dependency surface (the layering of
    ``PayloadRef`` has already been finalized to avoid this).

    Returns:
        ``encoded_count`` / ``encoded_bytes`` / ``decoded_count`` /
        ``decoded_bytes`` / ``oversize_count`` / ``oversize_bytes``.
    """
    from rollout_runtime.core.payload import stats

    return dataclasses.asdict(stats())


@dataclasses.dataclass
class LocalRuntime:
    """All components of an in-process runtime, exposed so tests can reach the fake backend directly.

    Attributes:
        config: The effective configuration.
        gateway: The control entry point (a ``RuntimeClient`` implementation).
        env_workers: The EnvWorker group.
        rollout_workers: The RolloutWorker group.
        channel: The inference request-plane channel (an ``InferenceChannel``
            implementation).
        transport: The Gateway <-> EnvWorker transport.
        policies: The inference core for each rollout rank (used by tests to
            inject latency / hangs).
        endpoints: The command endpoint for each env rank in the
            ray_channel form; empty for inproc.
    """

    config: RuntimeConfig
    gateway: RuntimeGateway
    env_workers: list[RuntimeEnvWorker]
    rollout_workers: list[RuntimeRolloutWorker]
    channel: Any
    transport: Any
    policies: list[Any]
    endpoints: list[Any] = dataclasses.field(default_factory=list)
    _serve_tasks: list[Any] = dataclasses.field(default_factory=list)
    _closed: bool = False

    @property
    def transport_kind(self) -> str:
        """The current transport kind.

        Returns:
            ``"inproc"`` or ``"ray_channel"``.
        """
        return self.config.transport.kind

    async def start(self) -> None:
        """Start the resident tasks of both worker groups, the Gateway's
        housekeeping, and register the workers.

        The only difference between the two transports is how the EnvWorker
        is driven: for inproc, ``InProcTransport`` directly ``await``s the
        handler; for ray_channel, the command / control / reap loops in
        ``run()`` pull work from the Channel.
        """
        import asyncio

        loop = asyncio.get_running_loop()
        ray_mode = self.transport_kind == "ray_channel"
        for index, worker in enumerate(self.env_workers):
            channels: dict[str, Any] = {"inference": self.channel}
            if ray_mode:
                channels["commands"] = self.endpoints[index]
            worker.init_worker(channels)
            if ray_mode:
                self._serve_tasks.append(loop.create_task(worker.run()))
            else:
                worker.start_server()
        for worker in self.rollout_workers:
            worker.init_worker({"inference": self.channel})
            self._serve_tasks.append(
                loop.create_task(worker.serve(self.channel, self.channel))
            )
        if ray_mode:
            await self.transport.start()
        for worker in self.env_workers:
            await self.gateway.register_env_worker(worker.worker_info())
        # Observation injection. Both sources live in **this process**, so
        # the numbers are complete: worker objects stay in the driver
        # process even in the ``ray_channel`` form. ``launch/ray_launch.py``,
        # which really puts workers into separate Ray actors, cannot reach
        # them there, so it **does not inject** rather than guessing a number.
        self.gateway.set_payload_stats_source(_payload_stats_snapshot)
        self.gateway.set_scheduler_stats_source(self._scheduler_stats_snapshot)
        await self.gateway.start()

    def _scheduler_stats_snapshot(self) -> dict[str, float]:
        """Sum up the scheduler counters across every rollout rank in this process (batch utilization).

        Returns:
            The combined ``batch_count`` / ``batched_count`` / ``queue_depth``.
        """
        total = {"batch_count": 0.0, "batched_count": 0.0, "queue_depth": 0.0}
        for worker in self.rollout_workers:
            scheduler = getattr(worker, "scheduler", None)
            if scheduler is None:
                continue
            total["batch_count"] += float(scheduler.batch_count)
            total["batched_count"] += float(scheduler.batched_count)
            total["queue_depth"] += float(scheduler.queue_depth)
        return total

    async def aclose(self) -> None:
        """Shut down in order: rollout -> env -> transport -> channels (idempotent)."""
        if self._closed:
            return
        self._closed = True
        for worker in self.rollout_workers:
            with contextlib.suppress(BaseException):
                await worker.stop()
        for worker in self.env_workers:
            with contextlib.suppress(BaseException):
                await worker.aclose()
        for task in self._serve_tasks:
            task.cancel()
        for task in self._serve_tasks:
            with contextlib.suppress(BaseException):
                await task
        self._serve_tasks.clear()
        if self.transport_kind == "ray_channel":
            for endpoint in self.endpoints:
                with contextlib.suppress(BaseException):
                    endpoint.close()
            with contextlib.suppress(BaseException):
                await self.transport.close()
            with contextlib.suppress(BaseException):
                await self.channel.aclose()
            with contextlib.suppress(BaseException):
                self.transport.shutdown()
            with contextlib.suppress(BaseException):
                self.channel.shutdown()
        else:
            self.channel.close()

    async def __aenter__(self) -> LocalRuntime:
        """Start and return self.

        Returns:
            The started ``LocalRuntime``.
        """
        await self.start()
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        """Stop the Gateway and all workers.

        Args:
            *exc_info: Ignored.
        """
        await self.gateway.stop()
        await self.aclose()

    def refresh_worker_registry(self) -> None:
        """Re-report worker capability and capacity (a heartbeat equivalent)."""
        for worker in self.env_workers:
            self.gateway.workers.register(worker.worker_info())


def _build_ray_plane(runtime_config: RuntimeConfig, env_ranks: list[int]) -> Any:
    """Build the Ray Channel data plane (a local Cluster + five channels).

    Args:
        runtime_config: The effective configuration.
        env_ranks: The list of EnvWorker ranks.

    Returns:
        ``(transport, channel, endpoints)``.
    """
    from rollout_runtime.transport.ray_channel import (
        ChannelNames,
        RayChannelTransport,
        RayChannelWorkerEndpoint,
        RayInferenceChannel,
    )
    from zetta.runtime.ray.bootstrap import ensure_ray_initialized

    ensure_ray_initialized()

    transport_conf = runtime_config.transport
    names = ChannelNames.build(
        env_group_name=runtime_config.env_worker.group_name,
        rollout_group_name=runtime_config.rollout_worker.group_name,
        gateway_epoch=runtime_config.gateway.gateway_epoch,
    )
    transport = RayChannelTransport.create(
        names=names,
        worker_ranks=env_ranks,
        command_queue_size=transport_conf.command_queue_size,
        control_queue_size=transport_conf.control_queue_size,
        result_queue_size=transport_conf.result_queue_size,
        command_timeout_seconds=transport_conf.command_timeout_seconds,
    )
    channel = RayInferenceChannel.create(
        names=names,
        request_queue_size=transport_conf.infer_request_queue_size,
        response_queue_size=transport_conf.infer_response_queue_size,
    )
    # In the single-process topology, workers and the Gateway share the same
    # batch of Channel handles: this saves three Channel.connect calls per
    # rank (each of which is a round of WorkerGroup.from_group_name plus a
    # remote maxsize query). See launch/ray_launch.py for the real
    # multi-process connection method.
    endpoints = [
        RayChannelWorkerEndpoint(
            worker_rank=rank,
            command_queue=transport._commands,  # noqa: SLF001 - same-process shared view
            control_queue=transport._controls,  # noqa: SLF001
            result_queue=transport._results,  # noqa: SLF001
        )
        for rank in env_ranks
    ]
    return transport, channel, endpoints


def build_local_components(
    config: str | RuntimeConfig | dict[str, Any] | None = None,
) -> LocalRuntime:
    """Build (but do not start) all components of an in-process runtime, according to config.

    ``config.transport.kind`` decides the data plane:

    - ``"inproc"``: ``InProcTransport`` + ``InProcInferenceChannel``, pure asyncio;
    - ``"ray_channel"``: a real rlinf ``Channel`` actor (command / control /
      result / inference request / inference response, five in total) —
      **worker objects still live in this process**. This lets the 8 e2e
      assertions be re-run unchanged under real Channel semantics
      (cross-process serialization, bounded queues, ``QueueFull`` backpressure,
      result flow-back), while tests can still reach the fake backend
      directly to inject latency and faults. See ``launch/ray_launch.py``
      for the form where workers actually run in separate Ray processes.

    Args:
        config: A preset name / yaml path / override dict / already
            constructed ``RuntimeConfig``; ``None`` uses the defaults.

    Returns:
        A ``LocalRuntime`` that has not been ``start()``-ed yet.

    Raises:
        ValueError: ``transport.kind`` is not one of the two implemented kinds.
    """
    runtime_config = (
        config if isinstance(config, RuntimeConfig) else load_config(config)
    )
    register_env_family_for(runtime_config.env_family)

    env_conf = runtime_config.env_worker
    rollout_conf = runtime_config.rollout_worker
    kind = runtime_config.transport.kind
    if kind not in ("inproc", "ray_channel"):
        raise ValueError(
            f"unknown transport kind {kind!r}; expected 'inproc' or 'ray_channel'"
        )

    policy_constraints = policy_compat_constraints(
        backend=rollout_conf.policy_backend,
        policy_config=dict(rollout_conf.policy_config),
    )
    env_workers = [
        RuntimeEnvWorker(
            worker_rank=rank,
            group_name=env_conf.group_name,
            node_id="local",
            max_sessions=env_conf.max_sessions_per_rank,
            seed_offset=env_conf.seed_offset + rank,
            total_num_processes=env_conf.num_ranks,
            default_policy_id=rollout_conf.policy_id,
            policy_family=rollout_conf.policy_family,
            policy_device=rollout_conf.device,
            policy_dtype=rollout_conf.dtype,
            policy_constraints=policy_constraints,
            supported_families=(runtime_config.env_family,),
            coalesce_slot_groups=env_conf.coalesce_slot_groups,
            coalesce_window_ms=env_conf.coalesce_window_ms,
            has_accelerator=env_conf.accelerator_present(),
            reap_interval_seconds=runtime_config.gateway.maintenance_interval_seconds,
        )
        for rank in range(env_conf.num_ranks)
    ]

    policies = [
        build_policy_core(
            backend=rollout_conf.policy_backend,
            policy_config=dict(rollout_conf.policy_config),
            device=rollout_conf.device,
            dtype=rollout_conf.dtype,
            policy_family=rollout_conf.policy_family,
            action_dim=int(runtime_config.env_config.get("action_dim", 7)),
            actions_per_chunk=int(runtime_config.env_config.get("chunk_size", 4)),
        )
        for _ in range(rollout_conf.num_ranks)
    ]
    rollout_workers = [
        RuntimeRolloutWorker(
            worker_rank=rank,
            group_name=rollout_conf.group_name,
            scheduler_config=rollout_conf.scheduler,
            policy=policies[rank],
            max_concurrent_inferences=rollout_conf.max_concurrent_inferences,
        )
        for rank in range(rollout_conf.num_ranks)
    ]

    endpoints: list[Any] = []
    if kind == "ray_channel":
        transport, channel, endpoints = _build_ray_plane(
            runtime_config, [worker.worker_rank for worker in env_workers]
        )
    else:
        channel = InProcInferenceChannel(
            request_queue_size=runtime_config.transport.infer_request_queue_size,
            response_queue_size=runtime_config.transport.infer_response_queue_size,
        )
        transport = InProcTransport(
            {worker.worker_rank: worker for worker in env_workers}
        )

    gateway_conf = runtime_config.gateway
    gateway = RuntimeGateway(
        transport=transport,
        admission=runtime_config.admission,
        gateway_epoch=gateway_conf.gateway_epoch,
        maintenance_interval_seconds=gateway_conf.maintenance_interval_seconds,
        error_retention_seconds=gateway_conf.error_retention_seconds,
        result_ttl_seconds=gateway_conf.result_ttl_seconds,
        heartbeat_timeout_seconds=gateway_conf.heartbeat_timeout_seconds,
        heartbeat_interval_seconds=gateway_conf.heartbeat_interval_seconds,
        max_concurrency_per_rank=gateway_conf.max_concurrency_per_rank,
        default_lease_seconds=gateway_conf.default_lease_seconds,
        metrics_namespace=gateway_conf.metrics_namespace,
    )
    return LocalRuntime(
        config=runtime_config,
        gateway=gateway,
        env_workers=env_workers,
        rollout_workers=rollout_workers,
        channel=channel,
        transport=transport,
        policies=policies,
        endpoints=endpoints,
    )


async def build_local_runtime(
    config: str | RuntimeConfig | dict[str, Any] | None = None,
) -> RuntimeGateway:
    """Build and start an in-process runtime, returning the Gateway.

    Releasing workers and channels is hooked into the Gateway's shutdown
    hook, so the caller only needs to call ``await gateway.stop()`` for a
    complete shutdown. Use ``build_local_components`` when direct access to
    the fake backend is needed.

    Args:
        config: A preset name / yaml path / override dict / already
            constructed ``RuntimeConfig``.

    Returns:
        The started ``RuntimeGateway``.
    """
    runtime = build_local_components(config)
    await runtime.start()
    runtime.gateway.add_shutdown_hook(runtime.aclose)
    return runtime.gateway
