"""Gateway metrics.

Falls back to an empty implementation with the same interface when
``prometheus_client`` is unavailable, ensuring the Gateway can still start in
a minimal-dependency environment. Metrics themselves have no side effects:
they only count and never affect scheduling decisions.

Metrics fall into two categories, landed differently:

- **Event-based** (operation / error / payload) are recorded directly with
  ``record_*`` at the point they occur;
- **State-based** (session state distribution, in-flight count, queue
  watermark, batch utilization, transport bytes) have no "point of
  occurrence"; they are periodically **sampled** by
  ``RuntimeGateway.refresh_metrics()`` and written via ``observe_*``. The
  sampling points are once per background maintenance loop iteration and
  once before each ``/metrics`` scrape, so the watermark captured reflects
  that instant.

Cumulative quantities (transport bytes, payload bytes, rejection counts,
batch counts) come from sources that are themselves **monotonic counters**,
but prometheus ``Counter`` can only ``inc``, so ``observe_*`` internally
remembers the previous value and only adds the delta (``_advance``). If the
source is reset (e.g. ``PayloadStats.reset()`` in tests), the delta is
negative and is simply skipped rather than making the counter non-monotonic.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol

__all__ = ["GatewayMetrics", "prometheus_available"]


class _Counter(Protocol):
    def labels(self, *args: str, **kwargs: str) -> Any: ...

    def inc(self, amount: float = 1.0) -> None: ...


class _NoopMetric:
    """Empty implementation used when prometheus is unavailable."""

    def labels(self, *args: str, **kwargs: str) -> _NoopMetric:
        """Return itself, ignoring labels.

        Args:
            *args: Ignored.
            **kwargs: Ignored.

        Returns:
            Itself.
        """
        return self

    def inc(self, amount: float = 1.0) -> None:
        """Ignore the count.

        Args:
            amount: Ignored.
        """

    def observe(self, amount: float) -> None:
        """Ignore the observation.

        Args:
            amount: Ignored.
        """

    def set(self, value: float) -> None:
        """Ignore the assignment.

        Args:
            value: Ignored.
        """


def prometheus_available() -> bool:
    """Whether ``prometheus_client`` is available.

    Returns:
        True if it can be imported.
    """
    try:
        import prometheus_client  # noqa: F401
    except ImportError:
        return False
    return True


class GatewayMetrics:
    """Gateway metrics collection."""

    def __init__(self, *, namespace: str = "rr", registry: Any = None) -> None:
        """Initialize.

        Args:
            namespace: Metric name prefix.
            registry: prometheus registry; ``None`` **creates a private
                ``CollectorRegistry``** rather than using the global default
                registry. A single process might host multiple Gateways
                simultaneously (tests, embedded + served); sharing the
                default registry would raise ``Duplicated timeseries in
                CollectorRegistry`` on the second construction. The
                ``/metrics`` endpoint renders ``self.registry`` directly.
        """
        self._namespace = namespace
        self._last: dict[str, float] = {}
        self._known_session_states: set[str] = set()
        self._known_ranks: set[int] = set()
        try:
            from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram
        except ImportError:
            self.registry = None
            self.operations_total = _NoopMetric()
            self.errors_total = _NoopMetric()
            self.payload_bytes_total = _NoopMetric()
            self.payload_events_total = _NoopMetric()
            self.payload_oversize_bytes_total = _NoopMetric()
            self.operation_latency_seconds = _NoopMetric()
            self.sessions = _NoopMetric()
            self.inflight_operations = _NoopMetric()
            self.admission_rejected_total = _NoopMetric()
            self.command_queue_depth = _NoopMetric()
            self.transport_bytes_total = _NoopMetric()
            self.env_workers = _NoopMetric()
            self.inference_batches_total = _NoopMetric()
            self.inference_batched_requests_total = _NoopMetric()
            self.inference_batch_size = _NoopMetric()
            self.inference_queue_depth = _NoopMetric()
            self.late_results_total = _NoopMetric()
            self.gateway_epoch = _NoopMetric()
            self.sampling_errors_total = _NoopMetric()
            self.enabled = False
            return

        self.registry = registry if registry is not None else CollectorRegistry()
        kwargs = {"registry": self.registry}
        self.operations_total = Counter(
            f"{namespace}_operations_total",
            "Runtime operations by type and outcome",
            ["operation", "outcome"],
            **kwargs,
        )
        self.errors_total = Counter(
            f"{namespace}_errors_total",
            "Runtime errors by normalized error code",
            ["code"],
            **kwargs,
        )
        self.payload_bytes_total = Counter(
            f"{namespace}_payload_bytes_total",
            "Payload bytes moved through the runtime",
            ["direction", "kind"],
            **kwargs,
        )
        self.payload_events_total = Counter(
            f"{namespace}_payload_events_total",
            "Payload encode/decode/oversize events (plan §2.5 decision data)",
            ["kind"],
            **kwargs,
        )
        self.payload_oversize_bytes_total = Counter(
            f"{namespace}_payload_oversize_bytes_total",
            "Payload bytes that crossed the inline threshold (subset of encoded bytes)",
            **kwargs,
        )
        self.operation_latency_seconds = Histogram(
            f"{namespace}_operation_latency_seconds",
            "End-to-end operation latency observed by the gateway",
            ["operation"],
            **kwargs,
        )
        self.sessions = Gauge(
            f"{namespace}_sessions",
            "Sessions by lifecycle state",
            ["state"],
            **kwargs,
        )
        self.inflight_operations = Gauge(
            f"{namespace}_inflight_operations",
            "In-flight operations accepted by the gateway",
            **kwargs,
        )
        self.admission_rejected_total = Counter(
            f"{namespace}_admission_rejected_total",
            "Admission rejections by normalized error code",
            ["code"],
            **kwargs,
        )
        self.command_queue_depth = Gauge(
            f"{namespace}_command_queue_depth",
            "Per-rank command channel depth (bounded queue = backpressure source)",
            ["worker_rank"],
            **kwargs,
        )
        self.transport_bytes_total = Counter(
            f"{namespace}_transport_bytes_total",
            "Wire bytes moved by the command transport",
            ["direction"],
            **kwargs,
        )
        self.env_workers = Gauge(
            f"{namespace}_env_workers",
            "Registered env worker ranks by health",
            ["state"],
            **kwargs,
        )
        self.inference_batches_total = Counter(
            f"{namespace}_inference_batches_total",
            "Inference batches dispatched by the rollout-side scheduler",
            **kwargs,
        )
        self.inference_batched_requests_total = Counter(
            f"{namespace}_inference_batched_requests_total",
            "Inference requests that went into a batch",
            **kwargs,
        )
        self.inference_batch_size = Gauge(
            f"{namespace}_inference_batch_size",
            "Average inference batch size (batched requests / batches)",
            **kwargs,
        )
        self.inference_queue_depth = Gauge(
            f"{namespace}_inference_queue_depth",
            "Requests waiting in the rollout-side scheduler buckets",
            **kwargs,
        )
        self.late_results_total = Counter(
            f"{namespace}_late_results_total",
            "Results that arrived after the caller's RPC deadline",
            ["kind"],
            **kwargs,
        )
        self.gateway_epoch = Gauge(
            f"{namespace}_gateway_epoch",
            "Gateway epoch published in SessionHandle (changes across restarts)",
            **kwargs,
        )
        self.sampling_errors_total = Counter(
            f"{namespace}_metrics_sampling_errors_total",
            "Failures inside a single metrics sampler (never affects the control plane)",
            ["sampler"],
            **kwargs,
        )
        self.enabled = True

    # ------------------------------------------------------------------ Event-based

    def record_operation(
        self, operation: str, outcome: str, latency_seconds: float | None = None
    ) -> None:
        """Record one operation.

        Args:
            operation: Operation name.
            outcome: ``"succeeded"`` / ``"failed"`` / ``"cancelled"`` etc.
            latency_seconds: End-to-end latency.
        """
        self.operations_total.labels(operation=operation, outcome=outcome).inc()
        if latency_seconds is not None:
            self.operation_latency_seconds.labels(operation=operation).observe(
                latency_seconds
            )

    def record_error(self, code: str) -> None:
        """Record one error.

        Args:
            code: Normalized error code name.
        """
        self.errors_total.labels(code=code).inc()

    def record_sampling_error(self, sampler: str) -> None:
        """Record one sampling failure.

        A broken sampler must not fail silently: ``/metrics`` would go stale
        from that point on, and without this count nobody would know.

        Args:
            sampler: Sampler name (``sessions`` / ``payload`` / ``scheduler`` ...).
        """
        self.sampling_errors_total.labels(sampler=sampler).inc()

    def record_payload(self, direction: str, kind: str, nbytes: int) -> None:
        """Record payload byte counts.

        Args:
            direction: ``"in"`` / ``"out"``.
            kind: ``"image"`` / ``"array"`` / ``"action"`` etc.
            nbytes: Byte count.
        """
        if nbytes:
            self.payload_bytes_total.labels(direction=direction, kind=kind).inc(nbytes)

    # ------------------------------------------------------------------ State-based

    def _advance(self, key: str, value: float) -> float:
        """Convert an absolute value from a monotonic source into a delta for this call.

        Args:
            key: Stable identifier for this source.
            value: The source's current cumulative value.

        Returns:
            The delta to ``inc`` this time; returns ``0.0`` if the source was
            reset (the value decreased).
        """
        previous = self._last.get(key, 0.0)
        self._last[key] = value
        delta = value - previous
        return delta if delta > 0 else 0.0

    def observe_sessions(self, counts: Mapping[str, int]) -> None:
        """Write the session state distribution.

        Args:
            counts: State name to count; **states not present are explicitly
                set to 0**, otherwise a state that once appeared would remain
                stuck forever at its last nonzero value.
        """
        self._known_session_states.update(counts)
        for state in self._known_session_states:
            self.sessions.labels(state=state).set(float(counts.get(state, 0)))

    def observe_inflight(self, total: int) -> None:
        """Write the global in-flight operation count.

        Args:
            total: Current in-flight count.
        """
        self.inflight_operations.set(float(total))

    def observe_rejections(self, counts: Mapping[str, int]) -> None:
        """Write admission rejection counts (by error code).

        Args:
            counts: Error code name to cumulative rejection count.
        """
        for code, value in counts.items():
            delta = self._advance(f"reject:{code}", float(value))
            if delta:
                self.admission_rejected_total.labels(code=code).inc(delta)

    def observe_queue_depth(self, depths: Mapping[int, int]) -> None:
        """Write the command queue watermark for each rank.

        Args:
            depths: Rank to queue length.
        """
        self._known_ranks.update(depths)
        for rank in self._known_ranks:
            self.command_queue_depth.labels(worker_rank=str(rank)).set(
                float(depths.get(rank, 0))
            )

    def observe_transport_bytes(self, *, sent: int, received: int) -> None:
        """Write the transport's cumulative wire byte counts.

        Args:
            sent: Bytes sent (commands + control).
            received: Bytes received (result flow-back).
        """
        for direction, value in (("out", sent), ("in", received)):
            delta = self._advance(f"wire:{direction}", float(value))
            if delta:
                self.transport_bytes_total.labels(direction=direction).inc(delta)

    def observe_payload_stats(self, snapshot: Mapping[str, int]) -> None:
        """Write cumulative counts from ``core.payload.stats()``.

        The Gateway itself **does not import** ``core.payload`` (that would
        drag numpy into the gateway's dependency surface), so the snapshot is
        provided by a getter function injected by the launch layer.

        Args:
            snapshot: ``encoded_count`` / ``encoded_bytes`` / ``decoded_count`` /
                ``decoded_bytes`` / ``oversize_count`` / ``oversize_bytes``.
                ``oversize_*`` is a subset of ``encoded_*``, so its byte count
                goes into a separate ``payload_oversize_bytes_total`` rather
                than being added to ``direction="out"``.
        """
        # ``oversize_*`` is a **subset** of ``encoded_*`` (``core/payload.py``
        # accumulates both for the same encode call), so it must never share
        # ``direction="out"`` with encoded -- otherwise ``sum(direction="out")``
        # would double-count the bytes that crossed the threshold. The
        # oversize count gets its own metric, semantically meaning "how many
        # bytes crossed the inline threshold."
        for field, direction, kind in (
            ("encoded_bytes", "out", "encoded"),
            ("decoded_bytes", "in", "decoded"),
        ):
            delta = self._advance(f"payload:{field}", float(snapshot.get(field, 0)))
            if delta:
                self.payload_bytes_total.labels(direction=direction, kind=kind).inc(
                    delta
                )
        oversize_delta = self._advance(
            "payload:oversize_bytes", float(snapshot.get("oversize_bytes", 0))
        )
        if oversize_delta:
            self.payload_oversize_bytes_total.inc(oversize_delta)
        for field in ("encoded_count", "decoded_count", "oversize_count"):
            delta = self._advance(f"payload:{field}", float(snapshot.get(field, 0)))
            if delta:
                self.payload_events_total.labels(kind=field.removesuffix("_count")).inc(
                    delta
                )

    def observe_scheduler(self, snapshot: Mapping[str, float]) -> None:
        """Write the inference scheduler's batch utilization.

        Args:
            snapshot: Aggregated ``batch_count`` / ``batched_count`` /
                ``queue_depth`` values (already summed across rollout ranks
                by the getter function).
        """
        batches = float(snapshot.get("batch_count", 0))
        batched = float(snapshot.get("batched_count", 0))
        batch_delta = self._advance("sched:batch_count", batches)
        if batch_delta:
            self.inference_batches_total.inc(batch_delta)
        batched_delta = self._advance("sched:batched_count", batched)
        if batched_delta:
            self.inference_batched_requests_total.inc(batched_delta)
        # Average batch size **within this sampling window**, not a
        # lifetime mean: the latter would make a change to batching
        # parameters invisible in production (e.g. 1000 size-1 batches
        # followed by 10 size-8 batches only moves the lifetime mean from
        # 1.00 to 1.07), whereas the windowed value is the number that
        # actually matters. Keep the previous value if there are no new
        # batches in this window, so an empty window doesn't zero it out.
        if batch_delta:
            self.inference_batch_size.set(batched_delta / batch_delta)
        self.inference_queue_depth.set(float(snapshot.get("queue_depth", 0)))

    def observe_env_workers(self, *, healthy: int, unhealthy: int) -> None:
        """Write the health distribution of registered ranks.

        Args:
            healthy: Number of healthy ranks.
            unhealthy: Number of ranks marked unhealthy due to heartbeat timeout.
        """
        self.env_workers.labels(state="healthy").set(float(healthy))
        self.env_workers.labels(state="unhealthy").set(float(unhealthy))

    def observe_late_results(self, *, absorbed: int, orphaned: int) -> None:
        """Write cumulative counts of late results (the observable surface for
        "RPC timeout != cancellation").

        Args:
            absorbed: Number of late results that were successfully absorbed.
            orphaned: Number of late results with no corresponding operation left.
        """
        for kind, value in (("absorbed", absorbed), ("orphaned", orphaned)):
            delta = self._advance(f"late:{kind}", float(value))
            if delta:
                self.late_results_total.labels(kind=kind).inc(delta)

    def observe_gateway_epoch(self, epoch: int) -> None:
        """Write the Gateway epoch.

        Args:
            epoch: Current epoch.
        """
        self.gateway_epoch.set(float(epoch))

    # ------------------------------------------------------------------ Export

    def render(self) -> tuple[bytes, str] | None:
        """Render the prometheus text format.

        Returns:
            ``(body, Content-Type)``; returns ``None`` when
            ``prometheus_client`` is unavailable or there is no registry, and
            the caller (``/metrics`` in ``serve``) should respond with 503.
        """
        if not self.enabled or self.registry is None:
            return None
        from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

        return generate_latest(self.registry), CONTENT_TYPE_LATEST
