"""``InferenceBatchScheduler`` (M3 real implementation).

Responsibility boundary: the Scheduler is the serving scheduler **internal** to each
RolloutWorker rank, responsible only for queueing, compatibility bucketing,
micro-batching, fairness, and backpressure; cross-worker transport is handled by the
Ray Channel.

Full M3 requirements:

- Buckets: ``dict[compat_key, deque[PendingRequest]]``;
- Trigger: ``max_batch_size`` or ``max_wait_ms`` (**configurable**, replacing rlinf's
  hardcoded 20ms);
- Ordering: ``Priority`` first, then earliest deadline first within the same priority;
- Fairness: an in-flight cap per ``application_id`` / per session to prevent a single
  tenant from monopolizing capacity;
- Backpressure: once the total queue depth exceeds the threshold, stop getting new
  requests, letting ``rr_infer_req`` naturally back up to maxsize, so the EnvWorker
  side's ``put_nowait`` receives ``QueueFull``;
- Version fence: ``update_weights`` executes between two batches, with
  ``ActionResponse.model_version`` tagged per request.
"""

from __future__ import annotations

import collections
import dataclasses

from rollout_runtime.api.enums import Priority
from rollout_runtime.api.internal import InferenceRequest

__all__ = ["InferenceBatchScheduler", "PendingRequest", "SchedulerConfig"]


@dataclasses.dataclass(kw_only=True)
class SchedulerConfig:
    """Batch scheduler configuration.

    Attributes:
        max_batch_size: Maximum number of requests per batch.
        max_wait_ms: Maximum wait time in milliseconds for batch accumulation
            (replacing rlinf's hardcoded 20ms).
        max_queue_depth: Total queue depth watermark; once exceeded, stop taking new
            requests from the Channel (backpressure).
        max_inflight_per_application: In-flight cap per tenant.
        max_inflight_per_session: In-flight cap per session.
        fixed_batch_size: Policies with cuda_graph enabled must use a fixed batch size
            with padding; ``None`` means a dynamic batch size (``mlp_policy.py:392-437``
            bakes in the batch size).
    """

    max_batch_size: int = 8
    max_wait_ms: float = 20.0
    max_queue_depth: int = 256
    max_inflight_per_application: int = 64
    max_inflight_per_session: int = 4
    fixed_batch_size: int | None = None


@dataclasses.dataclass
class PendingRequest:
    """A queued inference request.

    Attributes:
        request: The inference request.
        enqueued_at: Enqueue time, used for ``max_wait_ms`` evaluation.
        routing_token: Response routing token, recorded when the request is received.
    """

    request: InferenceRequest
    enqueued_at: float
    routing_token: str

    @property
    def priority(self) -> Priority:
        """Return the scheduling priority.

        Returns:
            The priority carried by the request.
        """
        return self.request.priority

    @property
    def sort_key(self) -> tuple[int, float]:
        """Return the sort key: priority first, earliest deadline first within the
        same priority.

        Returns:
            ``(priority_rank, deadline)``; no deadline is treated as positive infinity.
        """
        return (
            self.request.priority.sort_key,
            self.request.deadline
            if self.request.deadline is not None
            else float("inf"),
        )


class InferenceBatchScheduler:
    """Micro-batching scheduler bucketed by ``compat_key``.

    Handles only queueing, bucketing, micro-batching, fairness, and backpressure;
    cross-worker transport is handled by the Ray Channel.

    Fairness is implemented via an **in-flight cap**: when a request enters a batch it
    consumes one quota unit each from its tenant and session, returned on
    ``complete``. Requests whose quota is exhausted stay in the bucket for the next
    round rather than being rejected — rejection is the responsibility of admission,
    not the scheduler.
    """

    def __init__(self, config: SchedulerConfig | None = None) -> None:
        """Initialize.

        Args:
            config: Scheduler configuration; ``None`` uses the default values.
        """
        self.config = config or SchedulerConfig()
        self.buckets: dict[str, collections.deque[PendingRequest]] = {}
        self.inflight_per_application: dict[str, int] = {}
        self.inflight_per_session: dict[str, int] = {}
        self.inflight_total = 0
        self.enqueued_count = 0
        self.batched_count = 0
        self.batch_count = 0
        self.completed_count = 0
        self.fence_version: str | None = None

    @property
    def queue_depth(self) -> int:
        """Return the total queue depth across all buckets.

        Returns:
            Number of queued requests.
        """
        return sum(len(bucket) for bucket in self.buckets.values())

    def accepts_more(self) -> bool:
        """Whether new requests can still be accepted (backpressure watermark check).

        Returns:
            True if below ``max_queue_depth``.
        """
        return self.queue_depth < self.config.max_queue_depth

    # ------------------------------------------------------------------ Version fence

    @property
    def fenced(self) -> bool:
        """Whether currently inside a version switch fence.

        Returns:
            True if a switch has been requested and not yet completed.
        """
        return self.fence_version is not None

    def begin_version_fence(self, model_version: str) -> None:
        """Raise the version fence: ``next_batch`` stops issuing new batches until
        in-flight requests are drained.

        This is the mechanism that ensures ``update_weights`` executes between two
        batches, without mixing versions within the same batch.

        Args:
            model_version: Target version.
        """
        self.fence_version = model_version

    def end_version_fence(self) -> None:
        """Lower the version fence."""
        self.fence_version = None

    # ------------------------------------------------------------------ Queue

    def enqueue(self, pending: PendingRequest) -> None:
        """Place a request into its corresponding ``compat_key`` bucket.

        Args:
            pending: The queued request.
        """
        bucket = self.buckets.get(pending.request.compat_key)
        if bucket is None:
            bucket = collections.deque()
            self.buckets[pending.request.compat_key] = bucket
        bucket.append(pending)
        self.enqueued_count += 1

    def next_batch(self, now: float) -> list[PendingRequest]:
        """Pull out one executable batch.

        Trigger condition: the number of **selectable** requests in a bucket reaches
        ``max_batch_size``, or the earliest selectable request has waited longer than
        ``max_wait_ms``. Batches never span multiple ``compat_key`` buckets; when
        several buckets are ready at once, the one with the highest priority (earliest
        deadline within the same priority) is served first.

        Args:
            now: Current time (a monotonic clock is fine, as long as it shares the
                same source as ``PendingRequest.enqueued_at``).

        Returns:
            A list of requests sharing the same ``compat_key``; empty if no batch is
            executable.
        """
        if self.fenced and self.inflight_total > 0:
            # Do not issue new batches during the fence: wait for in-flight batches
            # to return before switching versions.
            return []
        limit = self.config.fixed_batch_size or self.config.max_batch_size
        limit = max(1, limit)
        wait_seconds = max(0.0, self.config.max_wait_ms / 1000.0)

        best_key: str | None = None
        best_rank: tuple[int, float] | None = None
        best_items: list[PendingRequest] = []
        for compat_key, bucket in self.buckets.items():
            selectable = self._selectable(bucket, limit)
            if not selectable:
                continue
            oldest = min(item.enqueued_at for item in selectable)
            ready = len(selectable) >= limit or (now - oldest) >= wait_seconds
            if not ready:
                continue
            rank = (selectable[0].sort_key[0], selectable[0].sort_key[1])
            if best_rank is None or rank < best_rank:
                best_key, best_rank, best_items = compat_key, rank, selectable
        if best_key is None:
            return []

        batch = best_items[:limit]
        bucket = self.buckets[best_key]
        chosen = {id(item) for item in batch}
        remaining = collections.deque(item for item in bucket if id(item) not in chosen)
        if remaining:
            self.buckets[best_key] = remaining
        else:
            self.buckets.pop(best_key, None)
        for item in batch:
            self._acquire_quota(item)
        self.batch_count += 1
        self.batched_count += len(batch)
        return batch

    def time_until_ready(self, now: float) -> float | None:
        """How long until the next batch can be issued.

        Used by ``execute_batches`` to wait precisely, avoiding polling just to
        respect ``max_wait_ms``.

        Args:
            now: Current time.

        Returns:
            ``0.0`` means a batch can be issued now; a positive number is the seconds
            still to wait; ``None`` means there are no selectable requests (including
            when blocked by fairness quotas or during a version fence).
        """
        if self.fenced and self.inflight_total > 0:
            return None
        limit = max(1, self.config.fixed_batch_size or self.config.max_batch_size)
        wait_seconds = max(0.0, self.config.max_wait_ms / 1000.0)
        best: float | None = None
        for bucket in self.buckets.values():
            selectable = self._selectable(bucket, limit)
            if not selectable:
                continue
            if len(selectable) >= limit:
                return 0.0
            oldest = min(item.enqueued_at for item in selectable)
            remaining = max(0.0, wait_seconds - (now - oldest))
            best = remaining if best is None else min(best, remaining)
        return best

    def complete(self, pending: PendingRequest) -> None:
        """Release fairness counters.

        Args:
            pending: The completed request.
        """
        application_id = pending.request.application_id
        session_id = str(pending.request.session_id)
        if application_id:
            left = self.inflight_per_application.get(application_id, 0) - 1
            if left > 0:
                self.inflight_per_application[application_id] = left
            else:
                self.inflight_per_application.pop(application_id, None)
        left = self.inflight_per_session.get(session_id, 0) - 1
        if left > 0:
            self.inflight_per_session[session_id] = left
        else:
            self.inflight_per_session.pop(session_id, None)
        self.inflight_total = max(0, self.inflight_total - 1)
        self.completed_count += 1

    # ------------------------------------------------------------------ Fairness

    def _selectable(
        self, bucket: collections.deque[PendingRequest], limit: int
    ) -> list[PendingRequest]:
        """Pick up to ``limit`` executable requests from a bucket according to
        fairness quotas.

        Quota is checked **incrementally**: placing 3 requests from the same session
        into one batch must consume 3 session-quota units, otherwise the "no single
        tenant can monopolize capacity" rule would fail to hold within a batch.

        Args:
            bucket: The queue for a given ``compat_key``.
            limit: Per-batch cap.

        Returns:
            Executable requests sorted by ``(priority, deadline, enqueue_time)``.
        """
        candidates = sorted(bucket, key=lambda item: (item.sort_key, item.enqueued_at))
        extra_application: dict[str, int] = {}
        extra_session: dict[str, int] = {}
        picked: list[PendingRequest] = []
        for item in candidates:
            if len(picked) >= limit:
                break
            if not self._has_quota(item, extra_application, extra_session):
                continue
            application_id = item.request.application_id
            if application_id:
                extra_application[application_id] = (
                    extra_application.get(application_id, 0) + 1
                )
            session_id = str(item.request.session_id)
            extra_session[session_id] = extra_session.get(session_id, 0) + 1
            picked.append(item)
        return picked

    def _has_quota(
        self,
        pending: PendingRequest,
        extra_application: dict[str, int] | None = None,
        extra_session: dict[str, int] | None = None,
    ) -> bool:
        application_id = pending.request.application_id
        pending_application = (extra_application or {}).get(application_id, 0)
        if (
            application_id
            and self.inflight_per_application.get(application_id, 0)
            + pending_application
            >= self.config.max_inflight_per_application
        ):
            return False
        session_id = str(pending.request.session_id)
        pending_session = (extra_session or {}).get(session_id, 0)
        return (
            self.inflight_per_session.get(session_id, 0) + pending_session
            < self.config.max_inflight_per_session
        )

    def _acquire_quota(self, pending: PendingRequest) -> None:
        application_id = pending.request.application_id
        if application_id:
            self.inflight_per_application[application_id] = (
                self.inflight_per_application.get(application_id, 0) + 1
            )
        session_id = str(pending.request.session_id)
        self.inflight_per_session[session_id] = (
            self.inflight_per_session.get(session_id, 0) + 1
        )
        self.inflight_total += 1
