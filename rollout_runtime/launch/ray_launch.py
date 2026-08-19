"""Ray launcher: starts both worker groups via ``Cluster`` + placement.

The pattern follows ``third_party/rlinf/examples/embodiment/train_async.py:80-95``:

```python
env_group = RayEnvWorker.create_group(config, names).launch(
    cluster, name=config.env_worker.group_name, placement_strategy=env_placement
)
```

**This is the only place in the whole runtime that subclasses rlinf's
``Worker``.** Reason: ``.venv-runtime`` does not have rlinf installed, and if
``workers/env_worker.py`` declared ``class RuntimeEnvWorker(Worker)`` at the
top level, the local inproc tests in M1/M2 would not even be able to import
it. That's why ``RuntimeEnvWorker`` / ``RuntimeRolloutWorker`` are plain
classes, wrapped here in a Ray shell: the shell handles placement, channel
connections, and lifecycle, without duplicating a single line of business
logic.

Division of labor with ``launch/local.py``:

| | ``local.py`` | this module |
|---|---|---|
| worker process | same process as the Gateway (tests can reach the fake backend directly) | separate Ray actor process |
| transport | ``inproc`` or ``ray_channel`` | ``ray_channel`` only |
| use case | e2e assertions, smoke tests | benchmarks, multi-rank, GPU deployments |

This layer matters a great deal for design decision D5:
``WorkerGroupFuncResult._wait_for_results`` sends ``SIGUSR1`` to the driver
process on a remote exception — a single method raising can kill the entire
job. Every remote method on the shell is therefore a total function; a
failure only ever returns ``{"ok": False, "error": ...}``.
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
from typing import Any

from rollout_runtime.api.messages import EnvWorkerInfo
from rollout_runtime.api.wire import decode_bytes, encode_bytes
from rollout_runtime.config.schema import RuntimeConfig, load_config
from rollout_runtime.gateway.gateway import RuntimeGateway
from rollout_runtime.transport.ray_channel import (
    ChannelNames,
    RayChannelTransport,
    RayChannelWorkerEndpoint,
    RayInferenceChannel,
)
from zetta.runtime.ray.bootstrap import ensure_ray_initialized
from zetta.runtime.ray.group import ZettaWorkerGroup
from zetta.runtime.ray.placement import parse_component_placement

__all__ = [
    "RayRuntime",
    "build_ray_components",
    "build_ray_runtime",
    "launch_runtime_workers",
    "worker_base_class",
]


def worker_base_class() -> type:
    """Return the small rank-aware base used by Zetta's actor shells."""

    class WorkerBase:
        def __init__(self, rank: int) -> None:
            self._rank = rank
            self._node_id = 0

    return WorkerBase


def _placement_for(
    component: str, *, config: RuntimeConfig, cluster: Any, num_ranks: int
) -> Any:
    """Compute the placement strategy for a component.

    Two branches:

    - The component is declared in ``cluster.component_placement`` -> use
      rlinf's ``HybridComponentPlacement`` (multi-GPU, cross-node, driven by
      ``node_group`` / ``placement``);
    - Not declared -> ``NodePlacementStrategy([0] * num_ranks)``, the only
      strategy available with 0 local accelerators.

    ``{env,rollout}_worker.placement_strategy`` is read here: declaring
    ``"packed"`` (multi-GPU) without providing a corresponding entry in
    ``cluster.component_placement`` is a **configuration error** and must
    fail explicitly rather than silently falling back to single-node —
    otherwise, on a GPU machine, all ranks would quietly be squeezed into
    the default placement and the throughput numbers would be worthless.

    Args:
        component: ``"env"`` or ``"rollout"``.
        config: The runtime configuration.
        cluster: The rlinf ``Cluster``.
        num_ranks: The rank count for this group.

    Returns:
        The ``PlacementStrategy``.

    Raises:
        ValueError: The group declares a placement strategy other than
            ``"node"``, but ``cluster.component_placement`` has no
            corresponding entry.
    """
    declared = config.cluster.component_placement or {}
    if component in declared:
        return parse_component_placement(
            dict(declared[component]), num_ranks=num_ranks
        )
    group_conf = getattr(config, f"{component}_worker", None)
    strategy_name = getattr(group_conf, "placement_strategy", "node")
    if strategy_name != "node":
        raise ValueError(
            f"{component}_worker.placement_strategy={strategy_name!r} needs an entry in "
            f"cluster.component_placement[{component!r}] "
            "(see config/presets/a100_libero.yaml); without a placement declaration only "
            "'node' works"
        )
    return parse_component_placement(
        None, num_ranks=max(1, num_ranks), strategy="node"
    )


def _build_env_worker_shell() -> type:
    """Construct the ``RayEnvWorker`` class (subclassing rlinf ``Worker``).

    The class is deliberately defined inside a function: the module top
    level must not contain ``class X(Worker)``, otherwise importing this
    module would require rlinf. Ray transfers class definitions via
    ``cloudpickle``, so a class defined inside a function can still be used
    to launch an actor.

    Returns:
        The ``RayEnvWorker`` class.
    """
    base = worker_base_class()

    class RayEnvWorker(base):  # type: ignore[misc, valid-type]
        """Ray shell for ``RuntimeEnvWorker``."""

        def __init__(
            self, rank: int, config: RuntimeConfig, names: dict[str, str]
        ) -> None:
            """Initialize (runs inside the actor process).

            Args:
                config: The runtime configuration.
                names: The five channel names.
            """
            super().__init__(rank)
            self._config = config
            self._names = ChannelNames(**names)
            self._runtime: Any = None
            self._endpoint: Any = None
            self._channel: Any = None
            self._task: asyncio.Task[None] | None = None

        async def init_worker(self) -> dict[str, Any]:
            """Build ``RuntimeEnvWorker`` and connect the command / control / result / inference channels.

            Returns:
                ``{"ok": bool, "error": str}``; never raises (D5).
            """
            try:
                from rollout_runtime.backends import (
                    policy_compat_constraints,
                    register_env_family_for,
                )
                from rollout_runtime.workers.env_worker import RuntimeEnvWorker

                register_env_family_for(self._config.env_family)
                env_conf = self._config.env_worker
                rollout_conf = self._config.rollout_worker
                transport_conf = self._config.transport
                self._runtime = RuntimeEnvWorker(
                    worker_rank=self._rank,
                    group_name=env_conf.group_name,
                    node_id=f"node-{self._node_id}"
                    if hasattr(self, "_node_id")
                    else "ray",
                    max_sessions=env_conf.max_sessions_per_rank,
                    seed_offset=env_conf.seed_offset + self._rank,
                    total_num_processes=env_conf.num_ranks,
                    default_policy_id=rollout_conf.policy_id,
                    policy_family=rollout_conf.policy_family,
                    policy_device=rollout_conf.device,
                    policy_dtype=rollout_conf.dtype,
                    policy_constraints=policy_compat_constraints(
                        backend=rollout_conf.policy_backend,
                        policy_config=dict(rollout_conf.policy_config),
                    ),
                    supported_families=(self._config.env_family,),
                    coalesce_slot_groups=env_conf.coalesce_slot_groups,
                    coalesce_window_ms=env_conf.coalesce_window_ms,
                    has_accelerator=env_conf.accelerator_present(),
                    reap_interval_seconds=(
                        self._config.gateway.maintenance_interval_seconds
                    ),
                )
                self._endpoint = RayChannelWorkerEndpoint.connect(
                    worker_rank=self._rank,
                    names=self._names,
                    command_queue_size=transport_conf.command_queue_size,
                    control_queue_size=transport_conf.control_queue_size,
                    result_queue_size=transport_conf.result_queue_size,
                )
                self._channel = RayInferenceChannel.connect(
                    names=self._names,
                    request_queue_size=transport_conf.infer_request_queue_size,
                    response_queue_size=transport_conf.infer_response_queue_size,
                )
                self._runtime.init_worker(
                    {"inference": self._channel, "commands": self._endpoint}
                )
                return {"ok": True, "error": ""}
            except BaseException as exc:  # noqa: BLE001 - D5: an exception here would kill the whole job
                return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

        async def start_server(self) -> dict[str, Any]:
            """Start the three loops of ``RuntimeEnvWorker.run()``.

            Returns:
                ``{"ok": bool, "error": str}``; never raises (D5).
            """
            try:
                loop = asyncio.get_running_loop()
                self._task = loop.create_task(self._runtime.run())
                return {"ok": True, "error": ""}
            except BaseException as exc:  # noqa: BLE001 - D5
                return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

        async def worker_info_payload(self) -> bytes:
            """Encode ``EnvWorkerInfo`` as msgpack for Gateway registration.

            Returns:
                ``api.wire`` bytes; empty on failure.
            """
            try:
                return encode_bytes(self._runtime.worker_info())
            except BaseException:  # noqa: BLE001 - D5
                return b""

        async def stop_server(self) -> dict[str, Any]:
            """Stop the loops and release sessions and pools.

            Returns:
                ``{"ok": bool, "error": str}``; never raises (D5).
            """
            try:
                if self._runtime is not None:
                    self._runtime.stop_server()
                if self._task is not None:
                    self._task.cancel()
                    self._task = None
                return {"ok": True, "error": ""}
            except BaseException as exc:  # noqa: BLE001 - D5
                return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    return RayEnvWorker


def _build_rollout_worker_shell() -> type:
    """Construct the ``RayRolloutWorker`` class (subclassing rlinf ``Worker``).

    Returns:
        The ``RayRolloutWorker`` class.
    """
    base = worker_base_class()

    class RayRolloutWorker(base):  # type: ignore[misc, valid-type]
        """Ray shell for ``RuntimeRolloutWorker``."""

        def __init__(
            self, rank: int, config: RuntimeConfig, names: dict[str, str]
        ) -> None:
            """Initialize (runs inside the actor process).

            Args:
                config: The runtime configuration.
                names: The five channel names.
            """
            super().__init__(rank)
            self._config = config
            self._names = ChannelNames(**names)
            self._runtime: Any = None
            self._channel: Any = None
            self._task: asyncio.Task[None] | None = None

        async def init_worker(self) -> dict[str, Any]:
            """Load this rank's policy and connect to the request plane.

            The backend is decided by ``rollout_worker.policy_backend``
            (``fake`` or the rlinf backend); ``backends.build_policy_core``
            is the sole resolution point.

            Returns:
                ``{"ok": bool, "error": str}``; never raises (D5).
            """
            try:
                from rollout_runtime.backends import build_policy_core
                from rollout_runtime.workers.rollout_worker import RuntimeRolloutWorker

                rollout_conf = self._config.rollout_worker
                transport_conf = self._config.transport
                policy = build_policy_core(
                    backend=rollout_conf.policy_backend,
                    policy_config=dict(rollout_conf.policy_config),
                    device=rollout_conf.device,
                    dtype=rollout_conf.dtype,
                    policy_family=rollout_conf.policy_family,
                    action_dim=int(self._config.env_config.get("action_dim", 7)),
                    actions_per_chunk=int(self._config.env_config.get("chunk_size", 4)),
                )
                self._runtime = RuntimeRolloutWorker(
                    worker_rank=self._rank,
                    group_name=rollout_conf.group_name,
                    scheduler_config=rollout_conf.scheduler,
                    policy=policy,
                    max_concurrent_inferences=rollout_conf.max_concurrent_inferences,
                )
                self._channel = RayInferenceChannel.connect(
                    names=self._names,
                    request_queue_size=transport_conf.infer_request_queue_size,
                    response_queue_size=transport_conf.infer_response_queue_size,
                )
                self._runtime.init_worker({"inference": self._channel})
                return {"ok": True, "error": ""}
            except BaseException as exc:  # noqa: BLE001 - D5
                return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

        async def start_server(self) -> dict[str, Any]:
            """Start ``serve()`` (receive loop ‖ batch execution loop).

            Returns:
                ``{"ok": bool, "error": str}``; never raises (D5).
            """
            try:
                loop = asyncio.get_running_loop()
                self._task = loop.create_task(
                    self._runtime.serve(self._channel, self._channel)
                )
                return {"ok": True, "error": ""}
            except BaseException as exc:  # noqa: BLE001 - D5
                return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

        async def counters(self) -> dict[str, Any]:
            """Report this rank's serving counters (used by bench).

            Returns:
                A dictionary of counters; empty on failure.
            """
            try:
                worker = self._runtime
                return {
                    "rank": self._rank,
                    "received": worker.received_count,
                    "responded": worker.responded_count,
                    "batches": worker.scheduler.batch_count,
                    "batched": worker.scheduler.batched_count,
                    "dropped_routes": worker.dropped_route_count,
                }
            except BaseException:  # noqa: BLE001 - D5
                return {}

        async def stop_server(self) -> dict[str, Any]:
            """Stop receiving, drain, and release the model.

            Returns:
                ``{"ok": bool, "error": str}``; never raises (D5).
            """
            try:
                if self._task is not None:
                    self._task.cancel()
                    self._task = None
                return {"ok": True, "error": ""}
            except BaseException as exc:  # noqa: BLE001 - D5
                return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    return RayRolloutWorker


def launch_runtime_workers(
    config: RuntimeConfig, cluster: Any, names: ChannelNames
) -> tuple[Any, Any]:
    """Launch both the env and rollout worker groups according to their placement.

    Args:
        config: The runtime configuration.
        cluster: The rlinf ``Cluster``.
        names: The five channel names.

    Returns:
        ``(env_group, rollout_group)``.
    """
    env_cls = _build_env_worker_shell()
    rollout_cls = _build_rollout_worker_shell()
    payload = names.as_dict()
    env_group = ZettaWorkerGroup.launch(
        env_cls,
        config,
        payload,
        num_ranks=config.env_worker.num_ranks,
        name=config.env_worker.group_name,
        placements=_placement_for(
            "env", config=config, cluster=cluster, num_ranks=config.env_worker.num_ranks
        ),
    )
    try:
        rollout_group = ZettaWorkerGroup.launch(
            rollout_cls,
            config,
            payload,
            num_ranks=config.rollout_worker.num_ranks,
            name=config.rollout_worker.group_name,
            placements=_placement_for(
                "rollout",
                config=config,
                cluster=cluster,
                num_ranks=config.rollout_worker.num_ranks,
            ),
        )
    except BaseException:
        with contextlib.suppress(BaseException):
            env_group.shutdown()
        raise
    return env_group, rollout_group


@dataclasses.dataclass
class RayRuntime:
    """A runtime running on real Ray worker processes.

    Attributes:
        config: The effective configuration.
        gateway: The control entry point.
        cluster: The rlinf ``Cluster``.
        env_group: The EnvWorker group.
        rollout_group: The RolloutWorker group.
        transport: The ``RayChannelTransport``.
        channel: The Gateway-side view of the request plane (only used to
            read watermarks and counters).
        names: The five channel names.
    """

    config: RuntimeConfig
    gateway: RuntimeGateway
    cluster: Any
    env_group: Any
    rollout_group: Any
    transport: RayChannelTransport
    channel: RayInferenceChannel
    names: ChannelNames
    #: The number of ranks actually observed per group at ``start()`` time
    #: (e.g. ``{"env": 8, "rollout": 4}``). This exists because the
    #: placement declaration decides the process count and ``num_ranks`` can
    #: be silently ignored — any claim in a performance report about "how
    #: many copies of the weights" must be backed by a measurement, not by
    #: copying the configuration.
    observed_ranks: dict[str, int] = dataclasses.field(default_factory=dict)
    _closed: bool = False

    async def start(self) -> None:
        """init -> start_server -> register workers -> start transport and Gateway.

        Raises:
            RuntimeError: Any rank's ``init_worker`` / ``start_server`` failed
                (the shell turns exceptions into return values, so the real
                reason is available here instead of the process being killed
                by SIGUSR1), or a group's actual rank count does not match
                ``num_ranks``.
        """
        for group, label in ((self.env_group, "env"), (self.rollout_group, "rollout")):
            for stage in ("init_worker", "start_server"):
                outcomes = await asyncio.to_thread(
                    lambda group=group, stage=stage: getattr(group, stage)().wait()
                )
                if stage == "init_worker":
                    # One return value per rank, so this is the measured process count.
                    self.observed_ranks[label] = len(outcomes)
                failures = [
                    item.get("error")
                    for item in outcomes
                    if not (isinstance(item, dict) and item.get("ok"))
                ]
                if failures:
                    raise RuntimeError(f"{label}.{stage} failed: {failures}")
        # Guard for the rollout side, matching the env-side guard below. The
        # env side is covered by the registration count from
        # worker_info_payload; rollout is not registered with the Gateway
        # (the Gateway never touches RolloutWorker), so it must be checked
        # separately here: the placement string in
        # ``cluster.component_placement['rollout']`` enumerates **hardware
        # ranks**, which decides the process count, and
        # ``rollout_worker.num_ranks`` is silently ignored. The consequence
        # is not a crash but **loading extra copies of the weights** (each
        # pi0.5 copy is roughly 7.5 GiB) and invalidating the "N weight
        # copies serve M sessions" conclusion — e.g. when the a100 preset
        # declares placement "0-7", setting num_ranks to 4 will not limit
        # the launch to only 4 ranks.
        observed_rollout = self.observed_ranks.get("rollout", 0)
        expected_rollout = self.config.rollout_worker.num_ranks
        if observed_rollout != expected_rollout:
            raise RuntimeError(
                f"rollout group started {observed_rollout} rank(s) but "
                f"rollout_worker.num_ranks={expected_rollout}; the placement "
                "declaration decides the process count. To run fewer rollout ranks than "
                "cluster.component_placement['rollout'] enumerates, narrow that "
                "placement string instead of only lowering num_ranks (each extra rank "
                "loads its own policy weights)"
            )
        await self.transport.start()
        payloads = await asyncio.to_thread(
            lambda: self.env_group.worker_info_payload().wait()
        )
        registered = 0
        for payload in payloads:
            if not payload:
                continue
            info = decode_bytes(payload, EnvWorkerInfo)
            await self.gateway.register_env_worker(info)
            registered += 1
        expected = self.config.env_worker.num_ranks
        if registered != expected:
            # A real defect found on a multi-GPU host: the placement string
            # in ``cluster.component_placement`` enumerates **hardware
            # ranks**, which decides the process count, and ``num_ranks`` is
            # silently ignored (``node_group: node`` + ``placement: "0"``
            # only launches 1 env process). The symptom is the 2nd
            # ``create_session`` getting ``QUOTA_EXCEEDED``, which is hard to
            # trace back to placement, hence the explicit failure here.
            raise RuntimeError(
                f"env group registered {registered} rank(s) but env_worker.num_ranks="
                f"{expected}; the placement declaration decides the process count. For "
                "CPU env workers drop cluster.component_placement['env'] and keep "
                "env_worker.placement_strategy='node' (NodePlacementStrategy gives one "
                "process per rank on the node)"
            )
        # Observation injection is **deliberately left empty** here: payload
        # encoding/decoding and the inference scheduler both live in
        # separate Ray actor processes, so ``core.payload.stats()`` read
        # from the driver process would only cover a small slice of the
        # driver side, and reporting it would be misread as a global value.
        # Missing is better than misleading.
        await self.gateway.start()

    async def refresh_worker_registry(self) -> None:
        """Re-poll each rank's capability and capacity (a heartbeat equivalent)."""
        payloads = await asyncio.to_thread(
            lambda: self.env_group.worker_info_payload().wait()
        )
        for payload in payloads:
            if payload:
                self.gateway.workers.register(decode_bytes(payload, EnvWorkerInfo))

    async def rollout_counters(self) -> list[dict[str, Any]]:
        """Collect serving counters from each rollout rank.

        Returns:
            A per-rank dictionary of counters.
        """
        return await asyncio.to_thread(lambda: self.rollout_group.counters().wait())

    async def aclose(self) -> None:
        """Stop workers, close channels, and reclaim actors (idempotent)."""
        if self._closed:
            return
        self._closed = True
        for group in (self.rollout_group, self.env_group):
            with contextlib.suppress(BaseException):
                await asyncio.to_thread(self._stop_live_ranks, group)
        with contextlib.suppress(BaseException):
            await self.transport.close()
        with contextlib.suppress(BaseException):
            await self.channel.aclose()
        self._kill_groups()
        with contextlib.suppress(BaseException):
            self.transport.shutdown()
        with contextlib.suppress(BaseException):
            self.channel.shutdown()

    @staticmethod
    def _stop_live_ranks(group: Any) -> None:
        """Call ``stop_server()`` only on ranks that are **still alive** (the shutdown side of D5).

        A real defect measured on a multi-GPU host: the original
        implementation simply called ``group.stop_server().wait()``, but if
        even one actor in the group had already died (deliberately via
        ``ray.kill`` on an env rank in one of the failure scenarios),
        ``WorkerGroupFuncResult._wait_for_results`` would receive an
        ``ActorDiedError`` -> set ``Cluster._run_failed`` ->
        ``os.kill(pid, SIGUSR1)``, **killing the driver during shutdown**.
        The outer ``contextlib.suppress`` cannot help here: the signal is
        sent by an rlinf daemon thread, not raised as an exception. Hence
        each actor is probed for liveness individually first, and
        ``stop_server()`` is only called rank by rank on the ones that are
        still alive.

        Args:
            group: The rlinf ``WorkerGroup``.
        """
        live = group.live_ranks(timeout=10)
        # **Deliberately no "call the whole group if everything looks alive"
        # fast path**: probing and stopping are not atomic, and an actor can
        # easily die in the gap between them during shutdown, which would
        # put us right back on the whole-group `stop_server()` ->
        # `ActorDiedError` -> `SIGUSR1` driver-kill path (a TOCTOU flagged by
        # an independent audit). Calling rank by rank costs a few extra RPCs,
        # which is fine during shutdown.
        for rank in live:
            with contextlib.suppress(BaseException):
                group.invoke_on(rank, "stop_server").wait()

    def _kill_groups(self) -> None:
        """Reclaim the actors of both groups.

        rlinf's ``WorkerGroup`` has no destroy API, and actor names
        (``{group_name}:{rank}``) are globally unique: without reclaiming
        them, a second ``launch`` of a group with the same name in the same
        process fails immediately with ``ActorAlreadyExistsError``, and the
        actors would keep hanging around (confirmed by testing).
        """
        for group in (self.rollout_group, self.env_group):
            with contextlib.suppress(BaseException):
                group.shutdown()

    async def __aenter__(self) -> RayRuntime:
        """Start and return self.

        Returns:
            The started ``RayRuntime``.
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


def build_ray_components(
    config: str | RuntimeConfig | dict[str, Any] | None = None,
) -> RayRuntime:
    """Build (but do not start) a runtime running on Ray.

    Args:
        config: A preset name / yaml path / override dict / already
            constructed ``RuntimeConfig``.

    Returns:
        A ``RayRuntime`` that has not been ``start()``-ed yet.

    Raises:
        ValueError: ``transport.kind`` is not ``ray_channel``.
    """
    runtime_config = (
        config if isinstance(config, RuntimeConfig) else load_config(config)
    )
    if runtime_config.transport.kind != "ray_channel":
        raise ValueError(
            "ray_launch requires transport.kind='ray_channel', got "
            f"{runtime_config.transport.kind!r}; use launch/local.py for inproc"
        )
    cluster = ensure_ray_initialized()
    names = ChannelNames.build(
        env_group_name=runtime_config.env_worker.group_name,
        rollout_group_name=runtime_config.rollout_worker.group_name,
        gateway_epoch=runtime_config.gateway.gateway_epoch,
    )
    transport_conf = runtime_config.transport
    transport = RayChannelTransport.create(
        names=names,
        worker_ranks=list(range(runtime_config.env_worker.num_ranks)),
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
    try:
        env_group, rollout_group = launch_runtime_workers(runtime_config, cluster, names)
    except BaseException:
        # Actor launch is transactional: no partially-created groups or channels
        # should survive a placement/name/resource failure.
        with contextlib.suppress(BaseException):
            transport.shutdown()
        with contextlib.suppress(BaseException):
            channel.shutdown()
        raise
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
    return RayRuntime(
        config=runtime_config,
        gateway=gateway,
        cluster=cluster,
        env_group=env_group,
        rollout_group=rollout_group,
        transport=transport,
        channel=channel,
        names=names,
    )


async def build_ray_runtime(
    config: str | RuntimeConfig | dict[str, Any] | None = None,
) -> RuntimeGateway:
    """Build and start a Ray runtime, returning the Gateway (same shape as ``build_local_runtime``).

    Releasing workers and channels is hooked into the Gateway's shutdown
    hook, so the caller only needs to call ``await gateway.stop()``. Use
    ``build_ray_components`` when direct access to the groups is needed.

    Args:
        config: A preset name / yaml path / override dict / already
            constructed ``RuntimeConfig``.

    Returns:
        The started ``RuntimeGateway``.
    """
    runtime = build_ray_components(config)
    try:
        await runtime.start()
    except BaseException:
        await runtime.aclose()
        raise
    runtime.gateway.add_shutdown_hook(runtime.aclose)
    return runtime.gateway
