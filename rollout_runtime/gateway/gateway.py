"""``RuntimeGateway``: the sole implementation of the Runtime API.

D2: the first version is a **single-writer asyncio object living inside the
driving process**, not a Ray Worker. A single writer directly satisfies the
requirement that "the Gateway and EnvWorker never co-mutate the same session
state," and the Gateway needs no GPU. Both deployment shapes share the same
class: ``embedded`` (held directly inside the application process) and
``served`` (wrapped with uvicorn behind HTTP).

The Gateway does not execute env steps, does not run model forward passes,
and does not persist trajectories: to it, ``policy_step`` is a single atomic
environment command; only the EnvWorker expands it into
observation -> inference -> ``chunk_step``.

Current status: session / operation / admission / routing / order-preserving
aggregation are all implemented over the transport Protocol; end-to-end runs
have been validated using ``InProcTransport`` plus a fake backend.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Any

from rollout_runtime.api.client import Consistency
from rollout_runtime.api.enums import (
    CONTROL_PLANE_OPERATIONS,
    MUTATING_OPERATIONS,
    EnvOperation,
    ErrorCode,
    OperationState,
    Priority,
    SessionState,
)
from rollout_runtime.api.errors import (
    RuntimeApiError,
    RuntimeErrorInfo,
    is_resource_exhausted,
    make_error,
    normalize_exception,
)
from rollout_runtime.api.ids import (
    BindingToken,
    EpisodeId,
    RequestId,
    SessionId,
    new_request_id,
)
from rollout_runtime.api.internal import (
    CommandEnvelope,
    ControlEnvelope,
    ResultEnvelope,
)
from rollout_runtime.api.messages import (
    CancelOutcome,
    CreateSessionRequest,
    EnvWorkerInfo,
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
from rollout_runtime.api.result import Ok, Result, err, ok
from rollout_runtime.gateway.admission import AdmissionConfig, AdmissionController
from rollout_runtime.gateway.dispatcher import (
    DEFAULT_MAX_CONCURRENCY_PER_RANK,
    CommandDispatcher,
)
from rollout_runtime.gateway.metrics import GatewayMetrics
from rollout_runtime.gateway.operation_registry import (
    DEFAULT_RESULT_TTL_SECONDS,
    OperationRegistry,
)
from rollout_runtime.gateway.plugin import PluginExecutor
from rollout_runtime.gateway.session_manager import (
    DEFAULT_ERROR_RETENTION_SECONDS,
    SessionManager,
    SessionRecord,
)
from rollout_runtime.gateway.worker_registry import (
    DEFAULT_HEARTBEAT_TIMEOUT_SECONDS,
    EnvWorkerRegistry,
)
from rollout_runtime.transport.base import CommandTransport

__all__ = ["RuntimeGateway"]

DEFAULT_MAINTENANCE_INTERVAL_SECONDS = 1.0
"""Interval between lease reclamation and registry inspection passes."""

DEFAULT_HEARTBEAT_INTERVAL_SECONDS = 10.0
"""Active probing interval.

Must be **noticeably smaller** than ``heartbeat_timeout_seconds``, otherwise
a single probing jitter would misjudge a healthy rank as lost. The default
of 10 s leaves three chances within the default 30 s timeout.
"""


class RuntimeGateway:
    """Protocol-agnostic control entry point oriented around sessions."""

    def __init__(
        self,
        *,
        transport: CommandTransport | None = None,
        admission: AdmissionConfig | None = None,
        gateway_epoch: int = 1,
        time_source: Callable[[], float] = time.time,
        maintenance_interval_seconds: float = DEFAULT_MAINTENANCE_INTERVAL_SECONDS,
        metrics: GatewayMetrics | None = None,
        default_priority: Priority = Priority.INTERACTIVE,
        error_retention_seconds: float = DEFAULT_ERROR_RETENTION_SECONDS,
        result_ttl_seconds: float = DEFAULT_RESULT_TTL_SECONDS,
        heartbeat_timeout_seconds: float = DEFAULT_HEARTBEAT_TIMEOUT_SECONDS,
        heartbeat_interval_seconds: float = DEFAULT_HEARTBEAT_INTERVAL_SECONDS,
        max_concurrency_per_rank: int = DEFAULT_MAX_CONCURRENCY_PER_RANK,
        default_lease_seconds: float = 300.0,
        metrics_namespace: str = "rr",
    ) -> None:
        """Initialize the Gateway.

        Args:
            transport: Transport to the EnvWorker; ``None`` means it is not
                yet attached (any operation needing a worker returns
                ``WORKER_LOST``).
            admission: Admission configuration.
            gateway_epoch: Gateway epoch, written into ``SessionHandle``.
            time_source: Time source, injectable for testing.
            maintenance_interval_seconds: Background maintenance interval.
            metrics: Metrics collection; ``None`` creates one (private registry).
            default_priority: Scheduling priority used when not explicitly specified.
            error_retention_seconds: ``FAILED`` / ``LOST`` record retention duration.
            result_ttl_seconds: Operation result cache duration.
            heartbeat_timeout_seconds: Worker heartbeat timeout.
            heartbeat_interval_seconds: Active probing interval; ``0``
                disables the heartbeat loop (in that case ``heartbeat_at`` is
                only updated when registered at the launch layer, and all
                ranks will be judged lost by the wall clock after
                ``heartbeat_timeout_seconds``).
            max_concurrency_per_rank: Per-rank concurrent command limit.
            default_lease_seconds: Default used when a request does not give
                ``lease_seconds``.
            metrics_namespace: Prometheus metric prefix.
        """
        self._now = time_source
        self._gateway_epoch = gateway_epoch
        self._maintenance_interval = maintenance_interval_seconds
        self._heartbeat_interval = heartbeat_interval_seconds
        self._heartbeat_timeout = heartbeat_timeout_seconds
        # Upper bound on the wait for a single probe. **Must be noticeably
        # smaller than `heartbeat_timeout`**, and must not be pushed up by
        # `heartbeat_interval` -- a real defect measured independently: the
        # original formula was `max(1.0, min(timeout, 5.0), interval)`,
        # which had no upper bound on `timeout`, so `local_fake` (timeout 5 s
        # / interval 1 s) computed 5 s, and a single hung rank would turn the
        # heartbeat cadence into interval + probe = 6 s > timeout,
        # **misjudging a healthy rank as lost too**. It is now taken as
        # timeout/3 (clamped to [0.5 s, 5 s]): the 0.5 s floor leaves margin
        # for the first Channel round trip to establish a connection under
        # the ray shape (measured steady-state 1.5 ms, cold start much
        # slower). Also, `probe_env_workers` is no longer chained onto the
        # heartbeat cadence (see `_heartbeat_loop`), so a single hung rank no
        # longer slows down the refresh of other ranks.
        self._heartbeat_probe_timeout = max(
            0.5, min(heartbeat_timeout_seconds / 3.0, 5.0)
        )
        self._default_priority = default_priority
        self._max_concurrency_per_rank = max_concurrency_per_rank
        self.sessions = SessionManager(
            gateway_epoch=gateway_epoch,
            time_source=time_source,
            error_retention_seconds=error_retention_seconds,
            default_lease_seconds=default_lease_seconds,
        )
        self.operations = OperationRegistry(
            time_source=time_source, result_ttl_seconds=result_ttl_seconds
        )
        self.admission = AdmissionController(admission, time_source=time_source)
        self.workers = EnvWorkerRegistry(
            time_source=time_source,
            heartbeat_timeout_seconds=heartbeat_timeout_seconds,
        )
        self.plugins = PluginExecutor()
        self.metrics = metrics or GatewayMetrics(namespace=metrics_namespace)
        self._transport = transport
        self._dispatcher = (
            CommandDispatcher(
                transport, max_concurrency_per_rank=max_concurrency_per_rank
            )
            if transport
            else None
        )
        self.late_result_count = 0
        self.orphan_result_count = 0
        self.heartbeat_ok_count = 0
        self.heartbeat_failure_count = 0
        self._heartbeat_inflight: set[int] = set()
        self._payload_stats_source: Callable[[], Mapping[str, int]] | None = None
        self._scheduler_stats_source: Callable[[], Mapping[str, float]] | None = None
        self._tasks: list[asyncio.Task[None]] = []
        self._shutdown_hooks: list[Callable[[], Awaitable[None]]] = []
        self._running = False
        if transport is not None:
            self._bind_late_results(transport)

    # ---------------------------------------------------------------- Lifecycle

    @property
    def gateway_epoch(self) -> int:
        """Return the Gateway epoch.

        Returns:
            The epoch value.
        """
        return self._gateway_epoch

    @property
    def transport(self) -> CommandTransport | None:
        """Return the current transport.

        Returns:
            The transport, or ``None``.
        """
        return self._transport

    def attach_transport(self, transport: CommandTransport) -> None:
        """Attach a transport (called by the ``launch`` layer once the worker is up).

        Args:
            transport: Transport to the EnvWorker.
        """
        self._transport = transport
        self._dispatcher = CommandDispatcher(
            transport, max_concurrency_per_rank=self._max_concurrency_per_rank
        )
        self._bind_late_results(transport)

    def _bind_late_results(self, transport: CommandTransport) -> None:
        """Register a late-result callback (``LateResultSink``) for transports
        with a result flow-back channel.

        Detected via ``getattr`` rather than by extending the
        ``CommandTransport`` Protocol: ``InProcTransport`` is
        request-response style and has no notion of "the result arriving
        after the caller has already returned," and should not be forced to
        implement an empty method.

        Args:
            transport: Transport to the EnvWorker.
        """
        register = getattr(transport, "set_late_result_handler", None)
        if register is not None:
            register(self._absorb_late_result)

    async def _absorb_late_result(self, result: ResultEnvelope) -> None:
        """Finalize a late result (one that came back only after an RPC timeout).

        When ``transport.command_timeout_seconds`` elapses, the caller only
        receives ``DEADLINE_EXCEEDED``, and **the operation does not enter a
        terminal state**; only when the real result arrives via the result
        channel does this method push it into a terminal state. So a
        "timeout" never misjudges an env step that is still running as
        failed, nor does it replay it.

        Args:
            result: The late result envelope.
        """
        if result.operation in CONTROL_PLANE_OPERATIONS:
            # Control-plane operations (heartbeat / binding / cancel) never
            # enter the ``OperationRegistry``; their late responses are not
            # what "RPC timeout != cancellation" is meant to measure. Letting
            # them through would let ``late_results_total{kind="orphaned"}``
            # be completely dominated by heartbeats during rank jitter.
            return
        self.late_result_count += 1
        record = self.operations.find(result.request_id)
        if record is None or record.is_terminal:
            self.orphan_result_count += 1
            return
        session_record = (
            self.sessions.find(record.session_id) if record.session_id else None
        )
        try:
            if session_record is not None:
                self._finish(
                    session_record,
                    result.request_id,
                    record.operation,
                    result,
                    self._now(),
                )
                return
            if result.ok:
                self.operations.succeed(
                    result.request_id,
                    result.value,
                    side_effect_applied=result.side_effect_applied,
                )
            else:
                self.operations.fail(
                    result.request_id,
                    result.error or self._synthesize_error(result),
                    side_effect_applied=result.side_effect_applied,
                )
        except BaseException:  # noqa: BLE001 - a finalization failure must not sink the drain loop
            self.orphan_result_count += 1

    def add_shutdown_hook(self, hook: Callable[[], Awaitable[None]]) -> None:
        """Register a finalization hook to run inside ``stop()``.

        ``launch/local.py`` uses this to attach worker and channel teardown
        to the Gateway's lifecycle, so that the object returned by
        ``build_local_runtime(config) -> RuntimeGateway`` can fully clean up
        after itself.

        Args:
            hook: A no-argument async callback.
        """
        self._shutdown_hooks.append(hook)

    async def start(self) -> None:
        """Start the background maintenance and heartbeat loops.

        The two loops are deliberately separate: maintenance (lease
        reclamation, lost-worker detection, result TTL cleanup, metrics
        sampling) runs on ``maintenance_interval_seconds``; heartbeat runs on
        ``heartbeat_interval_seconds``. The result flow-back loop belongs to
        the transport (``InProcTransport`` is request-response style and
        does not need one).
        """
        if self._running:
            return
        self._running = True
        self._tasks.append(asyncio.create_task(self._maintenance_loop()))
        if self._heartbeat_interval > 0:
            self._tasks.append(asyncio.create_task(self._heartbeat_loop()))

    async def stop(self) -> None:
        """Stop the background loops and run the finalization hooks."""
        self._running = False
        for task in self._tasks:
            task.cancel()
        for task in self._tasks:
            with contextlib.suppress(asyncio.CancelledError):
                await task
        self._tasks.clear()
        hooks = list(reversed(self._shutdown_hooks))
        self._shutdown_hooks.clear()
        for hook in hooks:
            with contextlib.suppress(BaseException):
                await hook()

    async def __aenter__(self) -> RuntimeGateway:
        """Enter the context and start the background loops.

        Returns:
            Itself.
        """
        await self.start()
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        """Exit the context and stop the background loops.

        Args:
            *exc_info: Ignored.
        """
        await self.stop()

    async def _maintenance_loop(self) -> None:
        while self._running:
            try:
                await asyncio.sleep(self._maintenance_interval)
                await self.recover_expired_sessions()
                self._reap_stale_workers()
                self.operations.purge()
                self.sessions.purge_terminal()
                self.refresh_metrics()
            except asyncio.CancelledError:
                raise
            except BaseException:  # noqa: BLE001 - the maintenance loop must never die
                continue

    async def _heartbeat_loop(self) -> None:
        # Cadence and probing are **decoupled**: each round only
        # `create_task`s and never `await`s the probe result. Otherwise a
        # single hung rank would drag the entire `gather` out to
        # `probe_timeout`, turning the heartbeat cadence into
        # `interval + probe_timeout`; a healthy rank's `heartbeat_at` would
        # then go stale between two refreshes and get misjudged as lost by
        # `_reap_stale_workers` too (a defect measured on `local_fake`
        # independently).
        pending: set[asyncio.Task[dict[int, bool]]] = set()
        while self._running:
            try:
                await asyncio.sleep(self._heartbeat_interval)
                task = asyncio.create_task(self.probe_env_workers())
                pending.add(task)
                task.add_done_callback(pending.discard)
            except asyncio.CancelledError:
                for task in list(pending):
                    task.cancel()
                raise
            except BaseException:  # noqa: BLE001 - the heartbeat loop must never die
                continue

    # ---------------------------------------------------------------- Observation injection

    def set_payload_stats_source(
        self, source: Callable[[], Mapping[str, int]] | None
    ) -> None:
        """Inject a payload-count getter function (the field dict from ``core.payload.stats()``).

        The Gateway itself **does not** import ``core.payload``: that would
        drag numpy into the gateway's dependency surface (this is precisely
        why the structure and encode/decode logic of ``PayloadRef`` are kept
        separate). So it is injected by the launch layer. When the worker is
        in **a different process** (``launch/ray_launch.py``), the driving
        process's count only covers the driver-side encode/decode portion,
        so the metric will read low -- this is documented honestly rather
        than presented as a global value.

        Args:
            source: No-argument function returning ``encoded_count`` /
                ``encoded_bytes`` / ``decoded_count`` / ``decoded_bytes`` /
                ``oversize_count`` / ``oversize_bytes``; ``None`` means no
                collection.
        """
        self._payload_stats_source = source

    def set_scheduler_stats_source(
        self, source: Callable[[], Mapping[str, float]] | None
    ) -> None:
        """Inject an inference scheduler getter function (batch utilization).

        Args:
            source: No-argument function returning ``batch_count`` /
                ``batched_count`` / ``queue_depth`` (pre-summed when there
                are multiple rollout ranks); ``None`` means no collection.
                When the RolloutWorker lives in a separate Ray actor, the
                driving process cannot access it, in which case ``None``
                should be passed rather than guessing a number.
        """
        self._scheduler_stats_source = source

    def refresh_metrics(self) -> None:
        """Sample state-based metrics once (session distribution / in-flight /
        queue watermark / bytes / batch).

        Event-based metrics are already recorded at the point they occur;
        state-based ones have no "point of occurrence," so they are sampled
        once per maintenance-loop iteration and once before each
        ``/metrics`` scrape. **Each sampler gets its own try/except**:
        previously there was a single
        ``except BaseException: return`` wrapping the whole block, so the
        first broken sampler would silently wipe out all subsequent metrics
        (measured independently: making ``observe_sessions`` raise left the
        last-written ``gateway_epoch`` stuck at 0 forever, with no count or
        log at all). Failures are now recorded individually into
        ``rr_metrics_sampling_errors_total{sampler}``; metrics collection
        still never affects the control plane.
        """
        for name, sampler in (
            ("sessions", self._sample_sessions),
            ("admission", self._sample_admission),
            ("workers", self._sample_workers),
            ("late_results", self._sample_late_results),
            ("epoch", self._sample_epoch),
            ("transport", self._sample_transport),
            ("payload", self._sample_payload),
            ("scheduler", self._sample_scheduler),
        ):
            try:
                sampler()
            except BaseException:  # noqa: BLE001 - metrics collection must never affect the control plane
                self.metrics.record_sampling_error(name)

    def _sample_sessions(self) -> None:
        counts: dict[str, int] = {}
        for record in self.sessions:
            key = record.state.name.lower()
            counts[key] = counts.get(key, 0) + 1
        self.metrics.observe_sessions(counts)

    def _sample_admission(self) -> None:
        snapshot = self.admission.snapshot()
        self.metrics.observe_inflight(snapshot.total_inflight)
        self.metrics.observe_rejections(snapshot.rejected)

    def _sample_workers(self) -> None:
        healthy = sum(1 for entry in self.workers if entry.healthy)
        self.metrics.observe_env_workers(
            healthy=healthy, unhealthy=len(self.workers) - healthy
        )

    def _sample_late_results(self) -> None:
        self.metrics.observe_late_results(
            absorbed=self.late_result_count - self.orphan_result_count,
            orphaned=self.orphan_result_count,
        )

    def _sample_epoch(self) -> None:
        self.metrics.observe_gateway_epoch(self._gateway_epoch)

    def _sample_transport(self) -> None:
        transport = self._transport
        depth_of = getattr(transport, "command_depth", None)
        if depth_of is not None:
            depths: dict[int, int] = {}
            for entry in self.workers:
                with contextlib.suppress(BaseException):
                    depths[entry.worker_rank] = int(depth_of(entry.worker_rank))
            self.metrics.observe_queue_depth(depths)
        sent = getattr(transport, "bytes_sent", None)
        received = getattr(transport, "bytes_received", None)
        if sent is not None and received is not None:
            self.metrics.observe_transport_bytes(sent=int(sent), received=int(received))

    def _sample_payload(self) -> None:
        if self._payload_stats_source is not None:
            self.metrics.observe_payload_stats(self._payload_stats_source())

    def _sample_scheduler(self) -> None:
        if self._scheduler_stats_source is not None:
            self.metrics.observe_scheduler(self._scheduler_stats_source())

    # ---------------------------------------------------------------- Worker plane

    async def register_env_worker(self, worker_info: EnvWorkerInfo) -> None:
        """Update the EnvWorker's capability, capacity, heartbeat, and servable EnvSpecs.

        Args:
            worker_info: Worker-reported information.
        """
        self.workers.register(worker_info)

    def _reap_stale_workers(self) -> None:
        for worker_rank in self.workers.stale_ranks():
            self.workers.mark_unhealthy(worker_rank)
            for record in self.sessions.sessions_on_rank(worker_rank):
                if record.state is SessionState.READY:
                    self.sessions.mark_lost(record.session_id)

    async def probe_env_workers(self) -> dict[int, bool]:
        """Send a ``HEARTBEAT`` control message to every registered rank and refresh the registry.

        Why go through the **control channel** rather than a
        ``WorkerGroup`` method: ``WorkerGroupFuncResult._wait_for_results``
        sets ``Cluster._run_failed`` and calls ``os.kill(pid, SIGUSR1)`` on a
        remote exception (including a dead actor) -- using it to probe would
        mean "one dead rank kills the driver too," which defeats exactly the
        thing ``LOST`` semantics are supposed to validate. The control
        channel is a bounded queue plus an independent result flow-back;
        when a rank dies it shows up only as **that one path timing out**,
        with other ranks unaffected.

        A successful probe simply calls ``register()`` (refreshing
        ``heartbeat_at`` and the capacity declaration); a failed probe
        **does nothing at all**, letting ``heartbeat_at`` age naturally so
        that ``_reap_stale_workers`` judges it lost after
        ``heartbeat_timeout_seconds`` and transitions any ``READY`` session
        on it to ``LOST``. In other words, a timeout is the cumulative
        result of "multiple failed probes," not the wall clock itself.

        Returns:
            Mapping of rank to whether this round's probe succeeded.
        """
        dispatcher = self._dispatcher
        if dispatcher is None:
            return {}
        ranks = [
            entry.worker_rank
            for entry in self.workers
            if entry.worker_rank not in self._heartbeat_inflight
        ]
        if not ranks:
            return {}
        # For the wait upper bound see ``_heartbeat_probe_timeout``: never
        # use the transport's 120 s command timeout, or a single dead rank
        # would drag the whole heartbeat loop out for two minutes.
        timeout = self._heartbeat_probe_timeout
        outcomes = await asyncio.gather(
            *(self._probe_rank_once(dispatcher, rank, timeout) for rank in ranks)
        )
        return dict(zip(ranks, outcomes, strict=True))

    async def _probe_rank_once(
        self, dispatcher: CommandDispatcher, worker_rank: int, timeout: float
    ) -> bool:
        """Probe one rank, releasing its in-flight marker **only once it completes itself**.

        Releasing per-rank is intentional: if release only happened after
        the whole ``gather`` finished, a single hung rank would also push
        healthy ranks' next refresh out past ``probe_timeout`` (measured: a
        healthy rank's ``heartbeat_age`` rose from the 3 s interval to the
        5 s probe window). While still far below the timeout, that coupling
        is unnecessary -- a healthy rank's refresh cadence should be
        determined solely by ``heartbeat_interval``.

        Args:
            dispatcher: Command dispatcher.
            worker_rank: Target rank.
            timeout: Upper bound on the wait for this probe (seconds).

        Returns:
            True if the registry was successfully refreshed.
        """
        self._heartbeat_inflight.add(worker_rank)
        try:
            return await self._probe_rank(dispatcher, worker_rank, timeout)
        finally:
            self._heartbeat_inflight.discard(worker_rank)

    async def _probe_rank(
        self, dispatcher: CommandDispatcher, worker_rank: int, timeout: float
    ) -> bool:
        """Probe one rank.

        Args:
            dispatcher: Command dispatcher (control messages do not consume
                the command semaphore).
            worker_rank: Target rank.
            timeout: Upper bound on the wait for this probe (seconds).

        Returns:
            True if the registry was successfully refreshed.
        """
        envelope = ControlEnvelope(
            request_id=new_request_id(),
            operation=EnvOperation.HEARTBEAT,
            session_id=None,
        )
        try:
            result = await asyncio.wait_for(
                dispatcher.send_control(worker_rank, envelope), timeout
            )
        except (asyncio.CancelledError, GeneratorExit):
            raise
        except BaseException:  # noqa: BLE001 - a probe failure is just "no heartbeat," not an error surface
            self.heartbeat_failure_count += 1
            return False
        info = result.value
        if not result.ok or not isinstance(info, EnvWorkerInfo):
            self.heartbeat_failure_count += 1
            return False
        self.workers.register(info)
        self.heartbeat_ok_count += 1
        return True

    async def recover_expired_sessions(self) -> list[SessionId]:
        """Reclaim sessions whose lease has already expired.

        Sessions with an operation in progress are first marked
        ``reap_pending`` and closed only once the operation finishes.

        Returns:
            IDs of sessions actually closed in this round.
        """
        closed: list[SessionId] = []
        for record in self.sessions.expired_sessions():
            if record.active_operation is not None:
                record.reap_pending = True
                continue
            outcome = await self._close_session(record.session_id)
            if isinstance(outcome, Ok):
                closed.append(record.session_id)
        return closed

    # ---------------------------------------------------------------- Session plane

    async def create_sessions(
        self, requests: Sequence[CreateSessionRequest]
    ) -> list[Result[SessionHandle]]:
        """Create sessions in batch.

        Args:
            requests: Creation requests.

        Returns:
            Per-item results in the same order as the input.
        """
        tasks = [
            asyncio.create_task(self._create_session(request)) for request in requests
        ]
        return list(await asyncio.gather(*tasks))

    async def _create_session(
        self, request: CreateSessionRequest
    ) -> Result[SessionHandle]:
        try:
            self.admission.admit_session(request.application_id, request.auth_token)
        except RuntimeApiError as exc:
            self.metrics.record_error(exc.info.code.name)
            return err(exc.info)

        record: SessionRecord | None = None
        last_resource_error: RuntimeErrorInfo | None = None
        try:
            record, created = self.sessions.create(request)
            if not created:
                # client_session_key idempotency: reuse the existing
                # session without occupying additional quota.
                self.admission.release_session(request.application_id)
                return ok(record.handle())
            attempted_workers: set[int] = set()
            while True:
                worker_rank = self.workers.select_rank(
                    record.env_spec,
                    prefer_node=request.env_spec.resource_hints.get("node_group"),
                    attempted_workers=attempted_workers,
                )
                # Pre-reserve capacity before creating the binding: there
                # must be no ``await`` between ``select_rank`` and
                # ``acquire``, otherwise concurrent creates would all see
                # the same "free" rank and pile onto it together.
                self.workers.acquire(worker_rank, record.env_spec_digest)
                try:
                    binding_token = await self._create_binding(record, worker_rank)
                except BaseException as exc:
                    self.workers.release(worker_rank, restore_can_create_slot=False)
                    info = normalize_exception(exc)
                    if is_resource_exhausted(info):
                        self.workers.mark_cannot_create_slot(worker_rank)
                        attempted_workers.add(worker_rank)
                        last_resource_error = info
                        continue
                    raise
                self.sessions.commit_binding(
                    record.session_id,
                    worker_rank=worker_rank,
                    binding_token=binding_token,
                )
                self.metrics.record_operation("create_session", "succeeded")
                return ok(record.handle())
        except BaseException as exc:  # noqa: BLE001 - the batch entry point must not leak exceptions
            info = normalize_exception(exc)
            if last_resource_error is not None and is_resource_exhausted(info):
                info = last_resource_error
            self.admission.release_session(request.application_id)
            if record is not None and record.state is SessionState.CREATING:
                self.sessions.fail(record.session_id, info)
            self.metrics.record_error(info.code.name)
            self.metrics.record_operation("create_session", "failed")
            return err(info)

    async def _release_lost_binding(self, record: SessionRecord) -> None:
        """When closing a ``LOST`` / ``FAILED`` session, make a best effort to
        also release the worker-side binding.

        Why this step is needed: ``LOST`` is judged unilaterally by the
        Gateway (heartbeat timeout), and the worker-side slot is still
        attached to that binding. If the rank has **truly died**, of course
        it cannot be recovered; but if the rank only had "a heartbeat gap
        and then came back," failing to release it would be a **permanent
        capacity leak** (the pool is pre-allocated and does not grow).

        Why it is only sent when the rank is **currently healthy**: sending
        a control message to a dead rank would wait all the way to
        ``transport.command_timeout_seconds`` (120 s by default), stalling
        ``close_sessions``; the slot on a truly dead rank is instead
        reclaimed by the worker's own lease reclamation
        (``recover_expired_sessions``) once the lease expires -- that is the
        correct exit for this path.

        Args:
            record: A session record in ``LOST`` / ``FAILED`` state.
        """
        if (
            self._dispatcher is None
            or record.worker_rank is None
            or record.binding_token is None
        ):
            return
        entry = self.workers.snapshot().get(record.worker_rank)
        if entry is None or not entry.healthy:
            return
        envelope = ControlEnvelope(
            request_id=new_request_id(),
            operation=EnvOperation.RELEASE_BINDING,
            session_id=record.session_id,
            payload={"binding_token": record.binding_token},
        )
        try:
            await asyncio.wait_for(
                self._dispatcher.send_control(record.worker_rank, envelope),
                # Releasing the binding is "best effort," with the window
                # capped by the probe window but never exceeding 2 s: close
                # should not be blocked by a rank that is currently hanging
                # (still considered healthy in the registry) (e.g. 4 LOST
                # sessions serialized x a 10 s default window = 40 s).
                min(2.0, self._heartbeat_probe_timeout),
            )
        except (asyncio.CancelledError, GeneratorExit):
            # Never swallow cancellation: ``gateway.stop()`` or an outer
            # timeout must be able to actually stop.
            raise
        except BaseException:  # noqa: BLE001 - release failure is backstopped by worker-side lease reclamation
            return

    async def _create_binding(
        self, record: SessionRecord, worker_rank: int
    ) -> BindingToken:
        envelope = ControlEnvelope(
            request_id=new_request_id(),
            operation=EnvOperation.CREATE_BINDING,
            session_id=record.session_id,
            payload={
                "env_spec": record.env_spec,
                "lease_expiration": record.lease_expiration,
            },
        )
        result = await self._require_dispatcher().send_control(worker_rank, envelope)
        if not result.ok:
            raise RuntimeApiError(
                result.error
                or make_error(
                    ErrorCode.ENV_FAILURE, "env worker rejected the binding request"
                )
            )
        token = result.value
        if isinstance(token, dict):
            token = token.get("binding_token")
        if not isinstance(token, str) or not token:
            raise RuntimeApiError(
                make_error(
                    ErrorCode.ENV_FAILURE,
                    "env worker returned no binding token",
                    worker_rank=worker_rank,
                )
            )
        return BindingToken(token)

    async def get_session(self, session_id: SessionId) -> SessionStatus:
        """Query session status.

        Args:
            session_id: Target session.

        Returns:
            ``SessionStatus``; the cached worker summary is not treated as
            authoritative environment state.
        """
        return self.sessions.status(session_id)

    async def renew_sessions(
        self, session_ids: Sequence[SessionId], lease_seconds: float
    ) -> list[Result[SessionStatus]]:
        """Renew leases in batch and notify the corresponding EnvWorker watchdog.

        Args:
            session_ids: Target sessions.
            lease_seconds: New lease duration.

        Returns:
            Per-item results in the same order as the input.
        """
        results: list[Result[SessionStatus]] = []
        for session_id in session_ids:
            try:
                record = self.sessions.renew(session_id, lease_seconds)
            except RuntimeApiError as exc:
                self.metrics.record_error(exc.info.code.name)
                results.append(err(exc.info))
                continue
            if self._dispatcher is not None and record.worker_rank is not None:
                await self._dispatcher.send_control(
                    record.worker_rank,
                    ControlEnvelope(
                        request_id=new_request_id(),
                        operation=EnvOperation.RENEW_LEASE,
                        session_id=session_id,
                        payload={"lease_expiration": record.lease_expiration},
                    ),
                )
            results.append(ok(record.status()))
        return results

    async def close_sessions(
        self, session_ids: Sequence[SessionId]
    ) -> list[Result[None]]:
        """Close sessions in batch.

        Args:
            session_ids: Target sessions.

        Returns:
            Per-item results in the same order as the input.
        """
        # Concurrent but **order-preserving** (``gather`` returns in input
        # order): on the ``LOST`` / ``FAILED`` path, ``_close_session`` makes
        # a best-effort attempt to send one ``RELEASE_BINDING``; serializing
        # this would turn N sessions into N x the release window (measured:
        # 4 serialized LOST sessions amplify the wait 4x).
        if not session_ids:
            return []
        return list(
            await asyncio.gather(
                *(self._close_session(session_id) for session_id in session_ids)
            )
        )

    async def _close_session(self, session_id: SessionId) -> Result[None]:
        try:
            record = self.sessions.get(session_id)
        except RuntimeApiError as exc:
            return err(exc.info)
        if record.state is SessionState.CLOSED:
            return ok(None)
        try:
            if record.state in (SessionState.FAILED, SessionState.LOST):
                await self._release_lost_binding(record)
                self.sessions.finish_close(session_id)
            else:
                self.sessions.begin_close(session_id)
                if self._dispatcher is not None and record.worker_rank is not None:
                    await self._dispatcher.send_control(
                        record.worker_rank,
                        ControlEnvelope(
                            request_id=new_request_id(),
                            operation=EnvOperation.RELEASE_BINDING,
                            session_id=session_id,
                            payload={"binding_token": record.binding_token},
                        ),
                    )
                self.sessions.finish_close(session_id)
        except RuntimeApiError as exc:
            self.metrics.record_error(exc.info.code.name)
            return err(exc.info)
        if record.worker_rank is not None:
            self.workers.release(record.worker_rank, restore_can_create_slot=True)
        self.admission.forget_session(session_id)
        self.admission.release_session(record.application_id)
        self.metrics.record_operation("close_session", "succeeded")
        return ok(None)

    # -------------------------------------------------------------- Operation plane

    async def reset(
        self,
        session_ids: Sequence[SessionId],
        reset_spec: ResetSpec,
        *,
        request_ids: Sequence[RequestId] | None = None,
    ) -> list[Result[StepResult]]:
        """Reset episodes in batch.

        Args:
            session_ids: Target sessions.
            reset_spec: Episode initialization parameters.
            request_ids: Idempotency identifiers.

        Returns:
            Per-item results in the same order as the input.
        """
        return await self._dispatch_batch(
            session_ids,
            EnvOperation.RESET,
            {"reset_spec": reset_spec},
            request_ids=request_ids,
        )

    async def observe(
        self,
        session_ids: Sequence[SessionId],
        *,
        consistency: Consistency = "linearizable",
    ) -> list[Result[Observation]]:
        """Read the current observation in batch.

        Args:
            session_ids: Target sessions.
            consistency: ``"linearizable"`` orders after any already-accepted
                mutating command; ``"eventual"`` skips the per-session lock.

        Returns:
            Per-item results in the same order as the input.
        """
        return await self._dispatch_batch(
            session_ids,
            EnvOperation.OBSERVE,
            {"consistency": consistency},
            take_lock=consistency == "linearizable",
        )

    async def action_step(
        self,
        session_ids: Sequence[SessionId],
        actions: Sequence[PayloadRef],
        *,
        request_ids: Sequence[RequestId] | None = None,
    ) -> list[Result[StepResult]]:
        """Execute externally supplied action chunks in batch.

        Args:
            session_ids: Target sessions.
            actions: Action payload for each session.
            request_ids: Idempotency identifiers.

        Returns:
            Per-item results in the same order as the input.
        """
        if len(actions) != len(session_ids):
            info = make_error(
                ErrorCode.INVALID_ARGUMENT,
                f"actions length {len(actions)} != session_ids length {len(session_ids)}",
            )
            return [err(info) for _ in session_ids]
        return await self._dispatch_batch(
            session_ids,
            EnvOperation.ACTION_STEP,
            [{"actions": action} for action in actions],
            request_ids=request_ids,
        )

    async def policy_step(
        self,
        session_ids: Sequence[SessionId],
        policy_request: PolicyRequest,
        *,
        request_ids: Sequence[RequestId] | None = None,
    ) -> list[Result[StepResult]]:
        """Execute the atomic ``policy_step`` in batch.

        The Gateway never touches the RolloutWorker: expanding into
        observation -> inference -> ``chunk_step`` is the EnvWorker's job.

        Args:
            session_ids: Target sessions.
            policy_request: Inference parameters.
            request_ids: Idempotency identifiers.

        Returns:
            Per-item results in the same order as the input.
        """
        return await self._dispatch_batch(
            session_ids,
            EnvOperation.POLICY_STEP,
            {"policy_request": policy_request},
            request_ids=request_ids,
        )

    async def policy_infer(
        self,
        session_ids: Sequence[SessionId],
        policy_request: PolicyRequest,
        *,
        request_ids: Sequence[RequestId] | None = None,
    ) -> list[Result[PolicyInferResult]]:
        """Execute "observe -> inference" in batch, without executing ``chunk_step``.

        A read-only operation: not in ``MUTATING_OPERATIONS``, so it
        **does not allocate** an ``operation_seq``. It still goes through the
        linearizable path (taking the per-session lock, consistent with
        A read-only operation: not in ``MUTATING_OPERATIONS``, so it
        **does not allocate** an ``operation_seq``. It still goes through the
        linearizable path (taking the per-session lock, consistent with
        ``observe(consistency="linearizable")`` and ``EXTENSION_CALL``),
        otherwise it might read a ``last_observation`` that is being
        overwritten by a concurrent ``chunk_step``. The Gateway still never
        touches the RolloutWorker; expanding inference is the EnvWorker's job.

        Args:
            session_ids: Target sessions.
            policy_request: Inference parameters.
            request_ids: Idempotency identifiers.

        Returns:
            Per-item results in the same order as the input.
        """
        return await self._dispatch_batch(
            session_ids,
            EnvOperation.POLICY_INFER,
            {"policy_request": policy_request},
            request_ids=request_ids,
        )

    async def run_episode(
        self,
        session_ids: Sequence[SessionId],
        episode_request: EpisodeRequest,
        *,
        request_ids: Sequence[RequestId] | None = None,
    ) -> list[Result[EpisodeResult]]:
        """Execute a full episode in batch.

        Args:
            session_ids: Target sessions.
            episode_request: Episode parameters.
            request_ids: Idempotency identifiers.

        Returns:
            Per-item results in the same order as the input.
        """
        return await self._dispatch_batch(
            session_ids,
            EnvOperation.RUN_EPISODE,
            {"episode_request": episode_request},
            request_ids=request_ids,
            deadline=episode_request.deadline,
        )

    async def extension_call(
        self,
        session_ids: Sequence[SessionId],
        namespace: str,
        method: str,
        args: dict[str, Any],
        *,
        request_ids: Sequence[RequestId] | None = None,
    ) -> list[Result[dict[str, Any]]]:
        """Invoke a family-specific extension.

        Args:
            session_ids: Target sessions.
            namespace: Extension namespace.
            method: Extension method name.
            args: Method parameters.
            request_ids: Idempotency identifiers.

        Returns:
            Per-item results in the same order as the input.
        """
        return await self._dispatch_batch(
            session_ids,
            EnvOperation.EXTENSION_CALL,
            {"namespace": namespace, "method": method, "args": dict(args)},
            request_ids=request_ids,
        )

    # -------------------------------------------------------------- Status and cancellation

    async def get_request_status(self, request_id: RequestId) -> OperationStatus:
        """Query the operation status.

        Args:
            request_id: Request identifier.

        Returns:
            ``OperationStatus``.
        """
        return self.operations.status(request_id)

    async def cancel_request(self, request_id: RequestId) -> CancelOutcome:
        """Make a best effort to cancel an operation.

        If not yet dispatched, the Gateway cancels it directly; if already
        dispatched, it forwards a best-effort cancel. An RPC timeout is not
        equivalent to a cancellation.

        Args:
            request_id: Request identifier.

        Returns:
            Cancellation outcome, including ``side_effect_applied``.
        """
        try:
            record = self.operations.get(request_id)
        except RuntimeApiError as exc:
            return CancelOutcome(
                request_id=request_id,
                state=OperationState.OUTCOME_UNKNOWN,
                message=exc.info.message,
            )
        outcome = self.operations.request_cancel(request_id)
        if (
            outcome.state is not OperationState.RUNNING
            or self._dispatcher is None
            or record.worker_rank is None
        ):
            return outcome
        reply = await self._dispatcher.send_control(
            record.worker_rank,
            ControlEnvelope(
                request_id=new_request_id(),
                operation=EnvOperation.CANCEL,
                session_id=record.session_id,
                payload={"target_request_id": request_id},
            ),
        )
        detail = reply.value if isinstance(reply.value, dict) else {}
        stage = str(detail.get("stage", "unknown"))
        side_effect_applied = bool(
            detail.get("side_effect_applied", record.side_effect_applied)
        )
        if detail.get("cancelled"):
            message = f"cancel accepted at stage {stage!r}; no env step was started"
        elif side_effect_applied:
            # Third state: the action is not rolled back; wait for it to
            # finish, and honestly report the side effect.
            message = (
                f"env step already started (stage {stage!r}); not rolled back, "
                "the operation will finish"
            )
        else:
            message = f"cancel forwarded to env worker (stage {stage!r}, best effort)"
        current = self.operations.find(request_id)
        return CancelOutcome(
            request_id=request_id,
            state=current.state if current is not None else outcome.state,
            side_effect_applied=side_effect_applied,
            message=message,
        )

    # -------------------------------------------------------------- Internal implementation

    def _require_dispatcher(self) -> CommandDispatcher:
        if self._dispatcher is None:
            raise RuntimeApiError(
                make_error(
                    ErrorCode.WORKER_LOST,
                    "gateway has no transport attached; call attach_transport first",
                )
            )
        return self._dispatcher

    def _cancelled_before_dispatch(self, request_id: RequestId) -> Any:
        """Check whether the operation was already cancelled before dispatch.

        ``OperationRegistry.request_cancel`` immediately sets ``ACCEPTED`` /
        ``QUEUED`` operations to ``CANCELLED`` (terminal). At that point
        ``mark_running`` must not be called again (it would raise
        ``InvalidTransition``), nor should the environment actually be touched.

        Args:
            request_id: Request identifier.

        Returns:
            ``RuntimeErrorInfo`` if it was already cancelled, otherwise ``None``.
        """
        record = self.operations.find(request_id)
        if record is None or not record.is_terminal:
            return None
        return record.error or make_error(
            ErrorCode.CANCELLED,
            f"request {request_id} was cancelled before dispatch",
            side_effect_applied=False,
            request_id=request_id,
        )

    @staticmethod
    def _synthesize_error(result: ResultEnvelope) -> Any:
        """Fill in a normalized error when the worker only reports a
        non-success state without an accompanying error.

        The semantics of ``OUTCOME_UNKNOWN`` are precisely "the worker was
        lost and it cannot be determined whether the step happened," so it
        is filled in as ``WORKER_LOST`` rather than ``INTERNAL``, preserving
        the side-effect flag declared by the worker (must not be replayed
        automatically).

        Args:
            result: Worker response.

        Returns:
            ``RuntimeErrorInfo``.
        """
        if result.state is OperationState.OUTCOME_UNKNOWN:
            return make_error(
                ErrorCode.WORKER_LOST,
                "env worker reported OUTCOME_UNKNOWN; the operation must not be "
                "replayed automatically",
                side_effect_applied=result.side_effect_applied,
                state=result.state.name,
            )
        if result.state is OperationState.CANCELLED:
            return make_error(
                ErrorCode.CANCELLED,
                "env worker reported CANCELLED",
                side_effect_applied=result.side_effect_applied,
            )
        return make_error(
            ErrorCode.INTERNAL,
            f"env worker returned {result.state.name} without an error payload",
            side_effect_applied=result.side_effect_applied,
        )

    def _require_session(
        self, session_id: SessionId, operation: EnvOperation
    ) -> SessionRecord:
        """Select the session precondition based on operation type.

        ``RESET`` only requires ``READY``; ``EXTENSION_CALL`` is a read-only
        family extension (none of the four LIBERO methods need an episode);
        all other operations require that reset has already happened.

        Args:
            session_id: Session identifier.
            operation: Operation type.

        Returns:
            The session record.
        """
        if operation in (EnvOperation.RESET, EnvOperation.EXTENSION_CALL):
            return self.sessions.require_ready(session_id)
        return self.sessions.require_episode(session_id)

    async def _dispatch_batch(
        self,
        session_ids: Sequence[SessionId],
        operation: EnvOperation,
        payload: dict[str, Any] | Sequence[dict[str, Any]],
        *,
        request_ids: Sequence[RequestId] | None = None,
        deadline: float | None = None,
        take_lock: bool = True,
    ) -> list[Result[Any]]:
        if request_ids is not None and len(request_ids) != len(session_ids):
            info = make_error(
                ErrorCode.INVALID_ARGUMENT,
                f"request_ids length {len(request_ids)} != "
                f"session_ids length {len(session_ids)}",
            )
            return [err(info) for _ in session_ids]
        payloads: Sequence[dict[str, Any]] = (
            [dict(payload)] * len(session_ids)
            if isinstance(payload, dict)
            else list(payload)
        )
        tasks = [
            asyncio.create_task(
                self._dispatch_one(
                    session_id,
                    operation,
                    payloads[index],
                    request_id=(
                        request_ids[index]
                        if request_ids is not None
                        else new_request_id()
                    ),
                    deadline=deadline,
                    take_lock=take_lock,
                )
            )
            for index, session_id in enumerate(session_ids)
        ]
        return list(await asyncio.gather(*tasks))

    async def _dispatch_one(
        self,
        session_id: SessionId,
        operation: EnvOperation,
        payload: dict[str, Any],
        *,
        request_id: RequestId,
        deadline: float | None,
        take_lock: bool,
    ) -> Result[Any]:
        started = self._now()
        mutating = operation in MUTATING_OPERATIONS
        application_id = ""
        admitted = False
        try:
            record = self._require_session(session_id, operation)
            application_id = record.application_id
            self.admission.admit_operation(
                application_id, session_id, deadline=deadline
            )
            admitted = True
            op_record, created = self.operations.begin(
                request_id,
                session_id=session_id,
                operation=operation,
                payload=payload,
            )
            if not created:
                # Idempotency hit: return the cached result if terminal,
                # otherwise honestly report the current state.
                if op_record.is_terminal:
                    if op_record.error is not None:
                        return err(op_record.error)
                    return ok(op_record.value)
                return err(
                    make_error(
                        ErrorCode.SESSION_NOT_READY,
                        f"request {request_id} is still {op_record.state.name}",
                        request_id=request_id,
                        state=op_record.state.name,
                    )
                )

            if take_lock:
                async with record.lock:
                    return await self._execute(
                        record,
                        op_record.request_id,
                        operation,
                        payload,
                        deadline,
                        mutating=mutating,
                        started=started,
                    )
            return await self._execute(
                record,
                op_record.request_id,
                operation,
                payload,
                deadline,
                mutating=mutating,
                started=started,
            )
        except BaseException as exc:  # noqa: BLE001 - the batch entry point must not leak exceptions
            info = normalize_exception(exc)
            self.metrics.record_error(info.code.name)
            self.metrics.record_operation(operation.name.lower(), "failed")
            return err(info)
        finally:
            if admitted:
                self.admission.complete_operation(application_id, session_id)

    async def _execute(
        self,
        record: SessionRecord,
        request_id: RequestId,
        operation: EnvOperation,
        payload: dict[str, Any],
        deadline: float | None,
        *,
        mutating: bool,
        started: float,
    ) -> Result[Any]:
        operation_seq = None
        cancelled = self._cancelled_before_dispatch(request_id)
        if cancelled is not None:
            # Cancelled before dispatch: no operation_seq is allocated, and
            # the worker is never touched (the first of the four cancellation states).
            self.metrics.record_operation(operation.name.lower(), "cancelled")
            return err(cancelled)
        if mutating:
            self.sessions.begin_operation(record, request_id)
            operation_seq = self.sessions.allocate_operation_seq(record)
        envelope = CommandEnvelope(
            request_id=request_id,
            session_id=record.session_id,
            binding_token=record.binding_token,
            episode_id=record.episode_id,
            operation_seq=operation_seq,
            operation=operation,
            deadline=deadline,
            priority=self._default_priority,
            payload=payload,
            trace_context={"application_id": record.application_id},
        )
        worker_rank = record.worker_rank
        if worker_rank is None:
            raise RuntimeApiError(
                make_error(
                    ErrorCode.WORKER_LOST,
                    f"session {record.session_id} has no worker binding",
                    session_id=record.session_id,
                )
            )
        dispatcher = self._require_dispatcher()
        self.operations.mark_running(request_id, worker_rank=worker_rank)
        try:
            result = await dispatcher.send(worker_rank, envelope)
        finally:
            if mutating:
                self.sessions.end_operation(record, request_id)
        return self._finish(record, request_id, operation, result, started)

    def _finish(
        self,
        record: SessionRecord,
        request_id: RequestId,
        operation: EnvOperation,
        result: ResultEnvelope,
        started: float,
    ) -> Result[Any]:
        self.sessions.set_worker_summary(record.session_id, result.worker_summary)
        latency = self._now() - started
        if (
            result.state is OperationState.RUNNING
            and result.error is not None
            and result.error.code is ErrorCode.DEADLINE_EXCEEDED
        ):
            # "RPC timeout != cancellation": the caller receives
            # DEADLINE_EXCEEDED, but the operation remains RUNNING in the
            # registry, to be finalized later by the worker's result
            # flow-back (_absorb_late_result).
            self.metrics.record_error(result.error.code.name)
            self.metrics.record_operation(operation.name.lower(), "timeout", latency)
            return err(result.error)
        if result.error is not None or result.state is not OperationState.SUCCEEDED:
            info = result.error or self._synthesize_error(result)
            outcome = "failed"
            if result.state is OperationState.CANCELLED:
                self.operations.cancel(
                    request_id, side_effect_applied=result.side_effect_applied
                )
                outcome = "cancelled"
            elif result.state is OperationState.OUTCOME_UNKNOWN:
                self.operations.mark_outcome_unknown(request_id, info.message)
                outcome = "outcome_unknown"
            else:
                self.operations.fail(
                    request_id, info, side_effect_applied=result.side_effect_applied
                )
            self.metrics.record_error(info.code.name)
            self.metrics.record_operation(operation.name.lower(), outcome, latency)
            return err(info)

        value = result.value
        episode_id = getattr(value, "episode_id", None)
        if isinstance(episode_id, int):
            self.sessions.set_episode(record.session_id, EpisodeId(episode_id))
        self.operations.succeed(
            request_id, value, side_effect_applied=result.side_effect_applied
        )
        self.metrics.record_operation(operation.name.lower(), "succeeded", latency)
        return ok(value)
