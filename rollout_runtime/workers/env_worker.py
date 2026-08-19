"""``RuntimeEnvWorker``.

The M1 deliverable is "complete method signatures + a ``handle_command`` dispatch
skeleton"; the real execution body lands in M2 (fake backend) and M4 (rlinf env).

Three finalized shape constraints:

- **Total functions**: every public method wraps its body in
  ``try/except BaseException``, normalizing exceptions into a
  ``ResultEnvelope(error=...)`` return value, **never raising**. The reason is that
  ``WorkerGroupFuncResult._wait_for_results`` calls ``os.kill(pid, SIGUSR1)`` on a
  remote exception — a single failed request would kill the whole job.
- **Interface isomorphic to rlinf's ``ToolWorker``**
  (``workers/agent/tool_worker.py:33``): ``init_worker(channels)`` /
  ``start_server()`` / ``stop_server()`` + a persistent ``_process_requests()``
  loop.
- **Env slots are pre-allocated pool indices**: ``num_envs`` is baked in at env
  construction time, the pool is built once up front with ``pool_size`` slots, a
  session binds to ``(pool_key, slot_index)``, and ``binding_token`` is an opaque
  uuid.

The late-arrival discard logic mirrors the structure of
``rlinf/workers/env/rtc_env_worker.py:217::RTCEnvWorker.evaluate`` (``chunk_id``
late-rejection + overlapping ``pending_response``).

This module does not import ``ray`` / ``rlinf`` at the top level: subclassing
rlinf's ``Worker`` is wrapped with a lazy import in the M3 launch layer, so that
``.venv-runtime`` (without rlinf) can still run the M2 inproc e2e.
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import json
import time
from collections.abc import Awaitable, Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from rollout_runtime.api.enums import (
    MUTATING_OPERATIONS,
    EnvOperation,
    ErrorCode,
    OperationState,
    Priority,
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
    OperationSeq,
    RequestId,
    SessionId,
    new_binding_token,
)
from rollout_runtime.api.internal import (
    ActionResponse,
    CommandEnvelope,
    ControlEnvelope,
    InferenceRequest,
    ResultEnvelope,
    make_routing_token,
)
from rollout_runtime.api.messages import (
    EnvFamilyCapability,
    EnvSpecMsg,
    EnvWorkerInfo,
    EpisodeRequest,
    EpisodeResult,
    Observation,
    PolicyInferResult,
    PolicyRequest,
    ResetSpec,
    StepResult,
    WorkerSummary,
)
from rollout_runtime.api.payload_ref import PayloadRef
from rollout_runtime.core import payload as payload_module
from rollout_runtime.core.env_execution import LOCKSTEP_VECTOR_FORM, PER_SLOT_FORM
from rollout_runtime.core.env_registry import get_env_family
from rollout_runtime.core.obs_schema import obs_schema_digest
from rollout_runtime.core.policy_inference import compute_compat_key
from rollout_runtime.transport.base import InferenceChannelClosed, TransportClosed

__all__ = [
    "ActiveRequest",
    "EnvPool",
    "EnvPoolManager",
    "InferenceClient",
    "RuntimeEnvWorker",
    "SessionSlot",
    "SlotGroupCoalescer",
    "TransitionSinkHub",
]

STAGE_QUEUED = "queued"
"""The operation has been accepted but has not yet produced any environment side
effect."""

STAGE_INFERENCE = "inference"
"""Waiting for an action from the RolloutWorker: waiting can be stopped and a late
action discarded."""

STAGE_ENV_STEP = "env_step"
"""The env step has started: the action cannot be rolled back and must be waited
out to completion."""

STAGE_DONE = "done"
"""The operation has finished."""


@dataclasses.dataclass
class SessionSlot:
    """Session state on the EnvWorker side (the single source of truth for
    environment execution state).

    Attributes:
        session_id: Session identifier.
        binding_token: Binding identifier; once a slot is released and reused, the
            old token becomes invalid immediately.
        pool_key: Env pool key (env spec digest).
        slot_index: Slot index within the pool.
        episode_id: Current episode; ``None`` means not yet reset.
        step_index: Number of steps already executed in the current episode.
        last_observation: Cached observation; ``observe`` only reads it.
        terminated: Termination flag.
        truncated: Truncation flag.
        lease_expiration: Lease expiration time.
        active_op: The mutating operation currently in progress.
        last_operation_seq: The highest executed operation sequence number, used to
            reject late commands.
        masked_steps_seen: The cumulative ``masked_steps`` for this lane as of the
            last sync (only meaningful on vector pools). Watermark semantics: if the
            core-side value is larger, it means this slot was absent from some
            coalesced group and got carried forward by a hold action, so the
            worker's cached observation is stale and needs a re-read. If equal,
            skip that ``observe`` call, so the common (non-absent) path only ever
            pays for the cheap ``lane_status`` check.
        lock: Per-session lock.
    """

    session_id: SessionId
    binding_token: BindingToken
    pool_key: str
    slot_index: int
    episode_id: EpisodeId | None = None
    step_index: int = 0
    last_observation: Observation | None = None
    terminated: bool = False
    truncated: bool = False
    lease_expiration: float = 0.0
    active_op: RequestId | None = None
    last_operation_seq: int = 0
    masked_steps_seen: int = 0
    lock: asyncio.Lock = dataclasses.field(
        default_factory=asyncio.Lock, repr=False, compare=False
    )

    def summary(self, worker_rank: int, group_name: str = "") -> WorkerSummary:
        """Generate a non-authoritative summary for Gateway observation.

        Args:
            worker_rank: Rank of this worker.
            group_name: Worker group name.

        Returns:
            A ``WorkerSummary``.
        """
        return WorkerSummary(
            worker_rank=worker_rank,
            group_name=group_name,
            pool_key=self.pool_key,
            slot_index=self.slot_index,
            step_index=self.step_index,
            terminated=self.terminated,
            truncated=self.truncated,
        )


@dataclasses.dataclass
class ActiveRequest:
    """Stage tracking for an in-progress operation (the staged cancellation
    semantics).

    Attributes:
        request_id: Request identifier.
        session_id: Owning session.
        operation: Operation type.
        stage: One of ``queued`` / ``inference`` / ``env_step`` / ``done``.
        side_effect_applied: Whether an environment side effect has occurred.
        cancel_requested: Whether a cancellation request has been received.
        inference_future: The inference Future currently being awaited, used to
            stop waiting on cancellation.
    """

    request_id: RequestId
    session_id: SessionId
    operation: EnvOperation
    stage: str = STAGE_QUEUED
    side_effect_applied: bool = False
    cancel_requested: bool = False
    inference_future: asyncio.Future[ActionResponse] | None = None

    @property
    def cancellable(self) -> bool:
        """Whether the current stage can be cancelled without causing a side
        effect.

        Returns:
            True if still in the ``queued`` / ``inference`` stage with no side
            effect yet.
        """
        return not self.side_effect_applied and self.stage in (
            STAGE_QUEUED,
            STAGE_INFERENCE,
        )


class EnvPool:
    """The environment pool corresponding to one env spec (a dynamic warm/cold slot
    model).

    The EnvWorker exclusively owns slot-level facts: idle warm slots, active slots,
    and cold-creation throttling are all maintained here. The Gateway only sees
    worker-level load and a conservative backoff flag.
    """

    def __init__(
        self,
        pool_key: str,
        env_spec: EnvSpecMsg,
        *,
        core: Any = None,
    ) -> None:
        """Initialize.

        Args:
            pool_key: Pool key (env spec digest).
            env_spec: Environment spec.
            core: An already-``build``-ed execution core; ``None`` means the core
                has not been built yet.
        """
        self.pool_key = pool_key
        self.env_spec = env_spec
        self.warm_free_slots: list[int] = list(range(env_spec.pool_size))
        self.active_slots: set[int] = set()
        self.core: Any = core
        # **All** core calls on a vector pool must be serialized: a vector env for
        # a GPU-batched family is a state machine, and concurrent partial resets
        # break it outright (ManiSkill's observed error is "Once _gpu_apply_all is
        # called, you must call _gpu_fetch_all before calling _gpu_apply_all
        # again"). per_slot pools are unaffected (slots are independent envs), so
        # this lock is only used on lockstep pools.
        self.core_lock = asyncio.Lock()
        self.cold_create_semaphore = asyncio.Semaphore(8)
        self._state_lock = asyncio.Lock()
        self._pending_cold_creates = 0
        self._reserved_slot_count = env_spec.pool_size
        self._next_seed_offset_hint = env_spec.pool_size
        self._min_pool_size = env_spec.pool_size

    @property
    def free_slots(self) -> list[int]:
        """Legacy debug name kept for compatibility; returns the warm free slot
        list."""
        return self.warm_free_slots

    @free_slots.setter
    def free_slots(self, value: list[int]) -> None:
        self.warm_free_slots = value

    @property
    def pool_size(self) -> int:
        """Pool capacity (may have been dynamically resized).

        Returns:
            Current total slot count.
        """
        core = self.core
        slot_count = getattr(core, "slot_count", None)
        if callable(slot_count):
            with contextlib.suppress(BaseException):
                return int(slot_count())
        return self.env_spec.pool_size

    @property
    def in_use(self) -> int:
        """Number of occupied slots.

        Returns:
            Occupied slot count.
        """
        return len(self.active_slots)

    @property
    def lockstep(self) -> bool:
        """Whether this pool is in vector form (and therefore can be truly
        coalesced).

        Returns:
            True if ``core_form == "lockstep_vector"``.
        """
        core = self.core
        if core is None:
            return False
        return str(getattr(core, "core_form", PER_SLOT_FORM)) == LOCKSTEP_VECTOR_FORM

    @property
    def dynamic(self) -> bool:
        """Whether this pool's current form supports dynamic resizing (a
        ``DynamicSlotPool``).

        Returns:
            True if in ``per_slot`` form and the core provides ``add_slot`` /
            ``remove_slot`` / ``slot_count`` simultaneously.
        """
        core = self.core
        if core is None or self.lockstep:
            return False
        return (
            callable(getattr(core, "add_slot", None))
            and callable(getattr(core, "remove_slot", None))
            and callable(getattr(core, "slot_count", None))
        )

    def acquire(self) -> int:
        """Take a warm idle slot (synchronous fast path, does not trigger cold
        creation).

        Returns:
            Slot index.

        Raises:
            RuntimeApiError: No free slot available (``QUOTA_EXCEEDED``).
        """
        if not self.warm_free_slots:
            raise RuntimeApiError(
                make_error(
                    ErrorCode.QUOTA_EXCEEDED,
                    f"env pool {self.pool_key[:12]} has no warm free slot",
                    pool_key=self.pool_key,
                    pool_size=self.pool_size,
                )
            )
        slot_index = self.warm_free_slots.pop()
        self.active_slots.add(slot_index)
        return slot_index

    async def acquire_or_grow(self) -> int:
        """Take a slot; prefer reusing a warm slot, otherwise cold-create one on
        demand.

        Returns:
            Slot index.

        Raises:
            RuntimeApiError: ``QUOTA_EXCEEDED`` if the pool is full and cannot grow;
                ``RESOURCE_EXHAUSTED`` if resources are insufficient for cold
                creation.
        """
        async with self._state_lock:
            if self.warm_free_slots:
                slot_index = self.warm_free_slots.pop()
                self.active_slots.add(slot_index)
                return slot_index
            seed_offset = self._reserve_cold_create_locked()
        try:
            async with self.cold_create_semaphore:
                new_index = await self._cold_create_slot(seed_offset)
        except BaseException:
            async with self._state_lock:
                self._pending_cold_creates = max(0, self._pending_cold_creates - 1)
                self._reserved_slot_count = max(
                    self._min_pool_size, self._reserved_slot_count - 1
                )
            raise
        async with self._state_lock:
            self._pending_cold_creates = max(0, self._pending_cold_creates - 1)
            self._reserved_slot_count = max(self._reserved_slot_count, new_index + 1)
            self.active_slots.add(new_index)
            return new_index

    def _reserve_cold_create_locked(self) -> int:
        """Reserve capacity for one cold create while holding ``_state_lock``."""
        if not self.dynamic:
            raise RuntimeApiError(
                make_error(
                    ErrorCode.QUOTA_EXCEEDED,
                    f"env pool {self.pool_key[:12]} is full ({self.pool_size} slots); "
                    "dynamic growth is not supported for this core form/family",
                    pool_key=self.pool_key,
                    pool_size=self.pool_size,
                    max_dynamic_pool_size=self.env_spec.effective_max_pool_size(),
                    dynamic=False,
                )
            )
        reserved_size = self._reserved_slot_count
        max_size = self.env_spec.effective_max_pool_size()
        if reserved_size >= max_size:
            raise RuntimeApiError(
                make_error(
                    ErrorCode.QUOTA_EXCEEDED,
                    f"env pool {self.pool_key[:12]} is full ({self.pool_size} slots); "
                    f"reached max_dynamic_pool_size={max_size}",
                    pool_key=self.pool_key,
                    pool_size=self.pool_size,
                    max_dynamic_pool_size=max_size,
                    dynamic=True,
                )
            )
        seed_offset = self._next_seed_offset_hint
        self._next_seed_offset_hint += 1
        self._reserved_slot_count += 1
        self._pending_cold_creates += 1
        return seed_offset

    async def _cold_create_slot(self, seed_offset: int) -> int:
        """Perform one blocking cold slot creation, normalizing resource-exhaustion
        errors."""
        try:
            return int(await asyncio.to_thread(self.core.add_slot, seed_offset))
        except RuntimeApiError as exc:
            if is_resource_exhausted(exc.info):
                raise
            raise
        except MemoryError as exc:
            raise RuntimeApiError(
                make_error(
                    ErrorCode.RESOURCE_EXHAUSTED,
                    f"failed to create env slot for pool {self.pool_key[:12]}: {exc}",
                    reason="OOM",
                    resource="gpu_memory",
                    pool_key=self.pool_key,
                )
            ) from exc

    def _next_seed_offset(self) -> int:
        """Generate the seed offset for a new slot.

        Returns:
            Seed offset.
        """
        return self._next_seed_offset_hint

    async def grow(self, extra: int = 1) -> int:
        """Explicitly grow the pool by a number of slots (optional; for a
        maintenance loop to pre-warm pool capacity).

        Args:
            extra: Desired number of new slots; the actual number added is capped
                by ``max_dynamic_pool_size``.

        Returns:
            The actual number of slots added.
        """
        if not self.dynamic or extra <= 0:
            return 0
        added = 0
        for _ in range(extra):
            async with self._state_lock:
                if self._reserved_slot_count >= self.env_spec.effective_max_pool_size():
                    break
                seed_offset = self._next_seed_offset_hint
                self._next_seed_offset_hint += 1
                self._reserved_slot_count += 1
                self._pending_cold_creates += 1
            try:
                async with self.cold_create_semaphore:
                    new_index = await self._cold_create_slot(seed_offset)
            except BaseException:
                async with self._state_lock:
                    self._pending_cold_creates = max(0, self._pending_cold_creates - 1)
                    self._reserved_slot_count = max(
                        self._min_pool_size, self._reserved_slot_count - 1
                    )
                raise
            async with self._state_lock:
                self._pending_cold_creates = max(0, self._pending_cold_creates - 1)
                self._reserved_slot_count = max(self._reserved_slot_count, new_index + 1)
                if new_index not in self.active_slots and new_index not in self.warm_free_slots:
                    self.warm_free_slots.append(new_index)
                    self.warm_free_slots.sort()
                    added += 1
        return added

    async def shrink_idle(self, *, keep_at_least: int | None = None) -> int:
        """Shrink a contiguous run of warm-idle slots from the tail (optional; for
        periodic invocation by a maintenance loop).

        Args:
            keep_at_least: The minimum number of slots to keep after shrinking;
                ``None`` means use the initial ``pool_size`` from construction as
                the floor.

        Returns:
            The actual number of slots shrunk.
        """
        if not self.dynamic:
            return 0
        floor = self._min_pool_size if keep_at_least is None else keep_at_least
        removed = 0
        while True:
            async with self._state_lock:
                free_set = set(self.warm_free_slots)
                last_index = self._reserved_slot_count - 1
                if (
                    self._reserved_slot_count <= floor
                    or self._pending_cold_creates
                    or last_index not in free_set
                    or last_index in self.active_slots
                ):
                    return removed
                self.warm_free_slots = [
                    index for index in self.warm_free_slots if index != last_index
                ]
                # Fully symmetric with the cold-create side's "reserve first,
                # then build": the actual `remove_slot` IO (the
                # `asyncio.to_thread` call below) is not lock-protected. If we
                # decremented `_reserved_slot_count` only after it completes,
                # an `acquire_or_grow` arriving almost simultaneously during
                # that window would see that `warm_free_slots` no longer
                # contains this slot (already removed above) and would take
                # the `_reserve_cold_create_locked` path, but the
                # `_reserved_slot_count` it reads would still be the
                # pre-removal value — if it equals `effective_max_pool_size()`
                # it would be wrongly rejected with `QUOTA_EXCEEDED`, even
                # though the quota would free up the moment this shrink
                # finishes. So we decrement the quota inside the lock (in the
                # same critical section as removing from `warm_free_slots`),
                # and roll it back unchanged in the except block below if the
                # actual remove fails (this is also why the `max(...)` for the
                # `_min_pool_size` floor is placed there instead of here:
                # rollback on failure must restore exactly the pre-shrink
                # value, not reapply the "not below the floor" clamp — a
                # rollback is always +1 and can never push the count below the
                # floor).
                self._reserved_slot_count -= 1
            try:
                await asyncio.to_thread(self.core.remove_slot, last_index)
            except BaseException:
                async with self._state_lock:
                    self._reserved_slot_count += 1
                    if last_index not in self.warm_free_slots:
                        self.warm_free_slots.append(last_index)
                        self.warm_free_slots.sort()
                raise
            removed += 1

    def release(self, slot_index: int) -> None:
        """Return a slot to the pool. Repeated release is idempotent."""
        if 0 <= slot_index < self._reserved_slot_count:
            self.active_slots.discard(slot_index)
            if slot_index not in self.warm_free_slots:
                self.warm_free_slots.append(slot_index)
                self.warm_free_slots.sort()

    def close(self) -> None:
        """Close the execution core and clear occupancy."""
        core = self.core
        self.active_slots.clear()
        self.warm_free_slots = list(range(self.pool_size))
        try:
            if core is not None:
                core.close()
        finally:
            self.core = None


class EnvPoolManager:
    """``dict[env_spec_digest, EnvPool]``: reuses environment pools keyed by
    digest."""

    def __init__(self, *, seed_offset: int = 0, total_num_processes: int = 1) -> None:
        """Initialize an empty pool table.

        Args:
            seed_offset: The seed offset for this rank, passed through to
                ``EnvExecutionCore.build``.
            total_num_processes: Total number of processes participating in the
                split.
        """
        self.pools: dict[str, EnvPool] = {}
        self.seed_offset = seed_offset
        self.total_num_processes = total_num_processes

    def find(self, pool_key: str) -> EnvPool | None:
        """Look up a pool by its key.

        Args:
            pool_key: env spec digest.

        Returns:
            The pool, or ``None``.
        """
        return self.pools.get(pool_key)

    def ensure_pool(self, env_spec: EnvSpecMsg) -> EnvPool:
        """Create (on demand) and return an environment pool.

        The pool is built with its initial slots when first needed, sized by
        ``env_spec.pool_size``; specs sharing the same digest hit the same pool.
        When in ``per_slot`` form and the core supports ``DynamicSlotPool``, the
        pool can subsequently be dynamically resized within
        ``[pool_size, effective_max_pool_size()]`` (see ``EnvPool``); the
        ``lockstep_vector`` form keeps a fixed capacity.

        Args:
            env_spec: Environment spec.

        Returns:
            The corresponding pool.
        """
        pool_key = env_spec.digest()
        pool = self.pools.get(pool_key)
        if pool is not None and pool.core is not None:
            return pool
        adapter = get_env_family(env_spec.env_family)
        core = adapter.create_core()
        core.build(
            env_spec,
            num_envs=env_spec.pool_size,
            seed_offset=self.seed_offset,
            total_num_processes=self.total_num_processes,
        )
        pool = EnvPool(pool_key, env_spec, core=core)
        self.pools[pool_key] = pool
        return pool

    def capabilities(self) -> dict[str, EnvFamilyCapability]:
        """Aggregate the family capabilities covered by pools built so far.

        Returns:
            Mapping from family name to capability.
        """
        table: dict[str, EnvFamilyCapability] = {}
        for pool in self.pools.values():
            family = pool.env_spec.env_family
            if family not in table:
                table[family] = get_env_family(family).capability
        return table

    def close(self) -> None:
        """Close every pool; a single pool's close failure does not affect the
        rest (cleanup must not be left half-done).

        An independent audit caught the same class of bug at this seam: a failed
        cleanup path leaked several GB of GPU memory. Here we ``suppress`` per
        pool, ensuring that "one family's close raised" does not leave the
        remaining pools permanently holding onto subprocesses / GPU memory.
        """
        for pool in list(self.pools.values()):
            with contextlib.suppress(BaseException):
                pool.close()
        self.pools.clear()


class InferenceClient:
    """The inference request-side interface on the EnvWorker side.

    ``put`` goes to ``rr_infer_req["pending"]`` (multiple RolloutWorker ranks
    compete on ``get``, giving natural work-stealing), and a background task only
    drains its own ``rr_infer_resp[env_rank]`` key, waking up an ``asyncio.Future``
    by ``request_id``. ``put_nowait`` raising ``QueueFull`` is itself the
    backpressure signal, converted upstream into an admission rejection.
    """

    def __init__(self, *, worker_rank: int, group_name: str = "") -> None:
        """Initialize.

        Args:
            worker_rank: Rank of this worker, used for response routing.
            group_name: Worker group name.
        """
        self.worker_rank = worker_rank
        self.group_name = group_name
        self.routing_token = make_routing_token(group_name, worker_rank)
        self.pending: dict[RequestId, asyncio.Future[ActionResponse]] = {}
        self.late_response_count = 0
        self.queue_full_count = 0
        self.submitted_count = 0
        self.channel: Any = None
        self._expected: dict[
            RequestId, tuple[BindingToken | None, EpisodeId | None]
        ] = {}
        self._drain_task: asyncio.Task[None] | None = None

    def attach(self, channel: Any) -> None:
        """Attach to the inference request channel.

        Args:
            channel: A ``transport.base.InferenceChannel`` implementation.
        """
        self.channel = channel
        register = getattr(channel, "register_route", None)
        if register is not None:
            register(self.routing_token)

    def start(self) -> None:
        """Launch the response-drain background task (only drains its own key)."""
        if self.channel is None or self._drain_task is not None:
            return
        self._drain_task = asyncio.get_running_loop().create_task(self._drain())

    async def stop(self) -> None:
        """Stop the drain task and wake up all waiters."""
        task = self._drain_task
        self._drain_task = None
        if task is not None:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        for future in list(self.pending.values()):
            if not future.done():
                future.cancel()
        self.pending.clear()
        self._expected.clear()

    async def submit(self, request: InferenceRequest) -> ActionResponse:
        """Submit an inference request and wait for the corresponding action.

        ``put_nowait`` raising ``QueueFull`` is itself the backpressure signal,
        translated upstream into a ``QUEUE_FULL`` rejection; requests are never
        allowed to accumulate unbounded in memory.

        Args:
            request: The inference request.

        Returns:
            The action-chunk response.

        Raises:
            RuntimeApiError: No channel attached (``WORKER_LOST``), the request
                side is full (``QUEUE_FULL``), or the channel is closed
                (``WORKER_LOST``).
        """
        if self.channel is None:
            raise RuntimeApiError(
                make_error(
                    ErrorCode.WORKER_LOST,
                    "env worker has no inference channel attached",
                    worker_rank=self.worker_rank,
                )
            )
        loop = asyncio.get_running_loop()
        future: asyncio.Future[ActionResponse] = loop.create_future()
        self.pending[request.request_id] = future
        self._expected[request.request_id] = (request.binding_token, request.episode_id)
        try:
            await self.channel.put_request_nowait(request)
        except asyncio.QueueFull as exc:
            self._forget(request.request_id)
            self.queue_full_count += 1
            depth = getattr(self.channel, "request_depth", -1)
            capacity = getattr(self.channel, "request_capacity", -1)
            raise RuntimeApiError(
                make_error(
                    ErrorCode.QUEUE_FULL,
                    "inference request queue is full "
                    f"({depth}/{capacity}); rejecting instead of queueing",
                    queue_depth=depth,
                    queue_capacity=capacity,
                    session_id=request.session_id,
                )
            ) from exc
        except InferenceChannelClosed as exc:
            self._forget(request.request_id)
            raise RuntimeApiError(
                make_error(
                    ErrorCode.WORKER_LOST,
                    "inference channel is closed",
                    session_id=request.session_id,
                )
            ) from exc
        self.submitted_count += 1
        try:
            return await future
        finally:
            self._forget(request.request_id)

    def deliver(self, response: ActionResponse) -> bool:
        """Deliver a response to its waiting Future.

        Late-arrival detection (mirroring ``rtc_env_worker``'s ``chunk_id``
        rejection structure): if ``request_id`` is no longer in the waiting table
        (cancelled / timed out / duplicate delivery), or ``binding_token`` /
        ``episode_id`` no longer match what was submitted originally, discard it
        and count it, unconditionally.

        Args:
            response: The action-chunk response.

        Returns:
            Whether the Future was successfully woken; late or already-cancelled
            requests return ``False`` and are counted.
        """
        future = self.pending.get(response.request_id)
        expected = self._expected.get(response.request_id)
        if future is None or future.done() or expected is None:
            self.late_response_count += 1
            return False
        if expected != (response.binding_token, response.episode_id):
            self.late_response_count += 1
            return False
        future.set_result(response)
        return True

    def _forget(self, request_id: RequestId) -> None:
        self.pending.pop(request_id, None)
        self._expected.pop(request_id, None)

    async def _drain(self) -> None:
        """Persistently drain this client's own response key, waking Futures by
        ``request_id``."""
        while True:
            try:
                response = await self.channel.get_response(self.routing_token)
            except (asyncio.CancelledError, GeneratorExit):
                raise
            except InferenceChannelClosed:
                return
            except BaseException:  # noqa: BLE001 - the drain loop must never die
                await asyncio.sleep(0)
                continue
            self.deliver(response)


@dataclasses.dataclass
class _CoalesceEntry:
    """One slot's registration entry within the current group.

    Attributes:
        slot_index: Slot index within the pool.
        block: The action, shape ``[chunk, action_dim]``.
        future: The result Future for this slot.
    """

    slot_index: int
    block: Any
    future: asyncio.Future[Any]


@dataclasses.dataclass
class _CoalesceGroup:
    """A pool's coalescing group for the current tick.

    Attributes:
        entries: Slots registered so far.
        ready: The "everyone is here" event, set by the ``expected``-th
            registrant.
        closed: Whether execution has already started; late registrants must open
            a new group.
        expected: The number of lanes expected in this group (the number of lanes
            currently running in the pool).
        chunk_len: This group's chunk length; different lengths go into different
            groups.
    """

    entries: list[_CoalesceEntry] = dataclasses.field(default_factory=list)
    ready: asyncio.Event = dataclasses.field(default_factory=asyncio.Event)
    closed: bool = False
    expected: int = 1
    chunk_len: int = 0


class SlotGroupCoalescer:
    """Coalescing of commands within the same pool and tick (the real
    implementation, M6).

    v1 was a serial stub where "each command has its own group". M6 replaces it
    with **asynchronous rendezvous**: on the worker side each command spawns its
    own task (finalized in M3), so ``chunk_step`` calls for the same pool and tick
    arrive concurrently, and a rendezvous point is required to merge them into a
    single call.

    The shape is "leader + window": the first slot to arrive becomes the leader,
    waits until "every lane currently running in the pool has arrived" or until
    ``window_seconds`` elapses, then issues **one**
    ``core.chunk_step(slots, blocks)`` call (one ``asyncio.to_thread``), and
    distributes the results back per slot. Absent lanes are carried forward by the
    execution core via a hold action and counted (see the masking semantics
    section for details).

    Only enabled for pools declared ``lockstep_vector``: for ``per_slot`` pools
    the execution core loops internally, so coalescing would only tie together
    unrelated sessions to wait on each other, with no benefit.
    """

    def __init__(
        self,
        *,
        enabled: bool = False,
        window_seconds: float = 0.2,
    ) -> None:
        """Initialize.

        Args:
            enabled: Whether coalescing is enabled; when ``False``, ``submit``
                executes each command individually (equivalent to the v1 stub).
            window_seconds: Upper bound on the accumulation window. Once everyone
                is present, the batch fires immediately, so on the normal path
                this is only a timeout safeguard.
        """
        self.enabled = enabled
        self.window_seconds = max(0.0, float(window_seconds))
        self.groups_executed = 0
        self.coalesced_command_count = 0
        self.window_timeout_count = 0
        self.max_group_size = 0
        self._groups: dict[tuple[str, int], _CoalesceGroup] = {}

    def coalesce(self, commands: list[CommandEnvelope]) -> list[list[CommandEnvelope]]:
        """Group commands by ``(pool_key, chunk_len)`` (a pure function, so unit
        tests can call it directly).

        Commands in the same group must share a chunk length: mixing them would
        require masking again along the **time** dimension, which is not worth
        it. ``pool_key`` is read from ``payload["pool_key"]``, falling back to
        ``session_id`` when missing, i.e. "its own group".

        Args:
            commands: Commands to group.

        Returns:
            The grouping result; when ``enabled`` is false, each command occupies
            its own group (v1 stub semantics).
        """
        if not self.enabled:
            return [[command] for command in commands]
        groups: dict[tuple[str, int], list[CommandEnvelope]] = {}
        for command in commands:
            payload = command.payload or {}
            key = (
                str(payload.get("pool_key") or command.session_id),
                int(payload.get("chunk_len") or 0),
            )
            groups.setdefault(key, []).append(command)
        return list(groups.values())

    async def submit(
        self,
        *,
        pool_key: str,
        slot_index: int,
        block: Any,
        chunk_len: int,
        expected: int,
        execute: Callable[[list[int], list[Any]], Awaitable[list[Any]]],
    ) -> Any:
        """Register one slot's action block, wait for the group to finish
        executing, and retrieve this slot's own result.

        Args:
            pool_key: Env pool key (env spec digest).
            slot_index: Slot index within the pool.
            block: The action, shape ``[chunk, action_dim]``.
            chunk_len: This call's chunk length (part of the grouping key).
            expected: The number of lanes currently running in the pool; the batch
                fires immediately once this many have arrived.
            execute: ``(slots, blocks) -> list[ChunkOutcome]``, provided by the
                worker (internally routes through ``_call_core`` to normalize
                family exceptions into ``ENV_FAILURE``).

        Returns:
            This slot's ``ChunkOutcome``.

        Raises:
            BaseException: If the whole group's execution fails (or the leader is
                cancelled), the exception is propagated as-is to **every** waiter
                in the group (already normalized by ``_call_core``); the leader
                itself also receives the same exception from its own future, so
                no future is left dangling.
        """
        if not self.enabled or expected <= 1:
            outcomes = await execute([slot_index], [block])
            self.groups_executed += 1
            self.coalesced_command_count += 1
            self.max_group_size = max(self.max_group_size, 1)
            return outcomes[0]
        key = (pool_key, int(chunk_len))
        group = self._groups.get(key)
        if group is None or group.closed:
            group = _CoalesceGroup(expected=expected, chunk_len=int(chunk_len))
            self._groups[key] = group
        group.expected = min(group.expected, expected) if group.entries else expected
        loop = asyncio.get_running_loop()
        entry = _CoalesceEntry(
            slot_index=slot_index, block=block, future=loop.create_future()
        )
        is_leader = not group.entries
        group.entries.append(entry)
        if len(group.entries) >= group.expected:
            group.ready.set()
        if is_leader:
            # The leader must **unconditionally** wake up every waiter in the
            # group: a dangling follower would keep holding `slot.active_op`
            # forever, and `recover_expired_sessions` explicitly skips sessions
            # with an active_op — that slot would never return to the pool (a
            # capacity leak found by an independent audit).
            try:
                if not group.ready.is_set() and self.window_seconds > 0:
                    try:
                        await asyncio.wait_for(group.ready.wait(), self.window_seconds)
                    except asyncio.TimeoutError:
                        # ``asyncio.TimeoutError``, not the builtin
                        # ``TimeoutError``: on Python 3.10 these are unrelated
                        # classes (unified only from 3.11 onward).
                        self.window_timeout_count += 1
                await self._fire(key, group, execute)
            except BaseException as exc:  # noqa: BLE001 - includes CancelledError
                self._abandon(key, group, exc)
                # No re-raise here: the leader's own future already carries the
                # same exception, and the `await entry.future` below will raise
                # it, which also "retrieves" it, avoiding a
                # "Future exception was never retrieved" warning.
        return await entry.future

    def _abandon(
        self, key: tuple[str, int], group: _CoalesceGroup, exc: BaseException
    ) -> None:
        """Wake every not-yet-settled waiter in the group with the same
        exception, and remove the group.

        Args:
            key: Grouping key.
            group: The coalescing group.
            exc: The exception to deliver to the waiters.
        """
        group.closed = True
        if self._groups.get(key) is group:
            self._groups.pop(key, None)
        for entry in list(group.entries):
            if not entry.future.done():
                entry.future.set_exception(exc)

    async def _fire(
        self,
        key: tuple[str, int],
        group: _CoalesceGroup,
        execute: Callable[[list[int], list[Any]], Awaitable[list[Any]]],
    ) -> None:
        """Execute one group and distribute results to each waiter in it.

        Args:
            key: Grouping key.
            group: The coalescing group.
            execute: The group execution function.
        """
        group.closed = True
        if self._groups.get(key) is group:
            self._groups.pop(key, None)
        entries = list(group.entries)
        slots = [entry.slot_index for entry in entries]
        blocks = [entry.block for entry in entries]
        self.groups_executed += 1
        self.coalesced_command_count += len(entries)
        self.max_group_size = max(self.max_group_size, len(entries))
        outcomes = await execute(slots, blocks)
        # Wake waiters individually even on a length mismatch (``strict=True``
        # would raise **midway** through distribution, leaving the remaining
        # futures permanently pending — an independent audit reproduced this).
        if len(outcomes) != len(entries):
            raise RuntimeApiError(
                make_error(
                    ErrorCode.INTERNAL,
                    f"coalesced chunk_step returned {len(outcomes)} outcome(s) for "
                    f"{len(entries)} slot(s) {slots}",
                    slots=slots,
                )
            )
        for entry, outcome in zip(entries, outcomes, strict=True):
            if not entry.future.done():
                entry.future.set_result(outcome)

    def stats(self) -> dict[str, Any]:
        """Return coalescing statistics (fed into ``worker_info`` and acceptance
        reports).

        Returns:
            Statistics dictionary.
        """
        return {
            "enabled": self.enabled,
            "window_seconds": self.window_seconds,
            "groups_executed": self.groups_executed,
            "coalesced_commands": self.coalesced_command_count,
            "window_timeouts": self.window_timeout_count,
            "max_group_size": self.max_group_size,
            "average_group_size": (
                self.coalesced_command_count / self.groups_executed
                if self.groups_executed
                else 0.0
            ),
        }


class TransitionSinkHub:
    """The transition output of ``run_episode``.

    v1 provides two sink kinds: in-memory (``"mem:<name>"`` or any unregistered
    id) and jsonl (``"jsonl:<path>"``). Transitions go through a sink rather than
    being returned to the Gateway, so the Gateway never has to persist
    trajectories.
    """

    JSONL_PREFIX = "jsonl:"
    """Id prefix for the jsonl sink."""

    def __init__(self) -> None:
        """Initialize an empty sink table."""
        self.sinks: dict[str, Any] = {}
        self.published_count = 0
        self.dropped_count = 0

    def memory(self, sink_id: str) -> list[dict[str, Any]]:
        """Register (or retrieve) an in-memory sink.

        Args:
            sink_id: Sink identifier.

        Returns:
            That sink's record list, which can be asserted on directly in tests.
        """
        records = self.sinks.get(sink_id)
        if not isinstance(records, list):
            records = []
            self.sinks[sink_id] = records
        return records

    def publish(self, sink_id: str | None, record: dict[str, Any]) -> None:
        """Write one transition.

        Args:
            sink_id: Sink identifier; ``None`` means discard and count.
            record: Transition content.
        """
        if sink_id is None:
            self.dropped_count += 1
            return
        if sink_id.startswith(self.JSONL_PREFIX):
            path = Path(sink_id[len(self.JSONL_PREFIX) :])
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
            self.published_count += 1
            return
        self.memory(sink_id).append(record)
        self.published_count += 1


class RuntimeEnvWorker:
    """The data-plane worker holding environment execution state.

    Isomorphic to rlinf's ``ToolWorker``: ``init_worker(channels)`` is called once
    by the launcher, ``start_server()`` launches persistent asyncio tasks (M2 is
    the ``InferenceClient``'s response drain; M3 adds the command channel loop),
    after which the data plane runs entirely through Channels.

    Execution-core calls always go through ``asyncio.to_thread``: a real env is
    blocking (libero CPU subproc), and "once an env step has started it cannot be
    cancelled" is exactly the semantics required.
    """

    def __init__(
        self,
        *,
        worker_rank: int = 0,
        group_name: str = "env",
        node_id: str = "local",
        max_sessions: int = 8,
        seed_offset: int = 0,
        total_num_processes: int = 1,
        default_policy_id: str = "fake",
        policy_family: str = "fake",
        policy_device: str = "cpu",
        policy_dtype: str = "float32",
        policy_constraints: dict[str, Any] | None = None,
        supported_families: Sequence[str] = ("fake",),
        coalesce_slot_groups: bool = True,
        coalesce_window_ms: float = 200.0,
        has_accelerator: bool = False,
        reap_interval_seconds: float = 1.0,
        time_source: Callable[[], float] = time.time,
    ) -> None:
        """Initialize.

        Args:
            worker_rank: Rank of this worker.
            group_name: Worker group name.
            node_id: Identifier of the node this worker runs on, used by the
                Gateway for locality selection.
            max_sessions: Cap on the number of sessions this rank can host.
            seed_offset: The seed offset for this rank, passed through to the
                execution core.
            total_num_processes: Total number of processes participating in the
                split.
            default_policy_id: Default value for ``PolicyRequest.policy_id``.
            policy_family: Policy family name, determining the mixable-batch
                whitelist for ``compat_key``.
            policy_device: Policy device identifier, fed into ``compat_key``.
            policy_dtype: Policy compute precision, fed into ``compat_key``.
            policy_constraints: Hard constraints of the policy (cuda_graph's fixed
                batch, ``openvla_oft``'s ``padding="max_length"``, action-chunk
                length), also fed into ``compat_key``. Declared by the backend
                (``backends.policy_compat_constraints``); the EnvWorker merely
                relays it — since ``compat_key`` is computed on this side, it must
                know it.
            supported_families: The env families this rank is willing to serve.
                **Must be supplied via configuration**: the Gateway uses it to
                decide ``UNSUPPORTED_ENV_SPEC``, and starting from M4, a build
                only registers the adapter for the family in the configuration
                (hardcoding ``("fake",)`` would make the libero form blow up at
                registration time).
            coalesce_slot_groups: Whether to enable ``SlotGroupCoalescer``'s real
                coalescing (M6). Only takes effect for pools declared
                ``lockstep_vector``; ``per_slot`` pools always take the
                single-item path.
            coalesce_window_ms: Upper bound on the accumulation window
                (milliseconds). The batch fires immediately once everyone is
                present, so this is only a timeout safeguard.
            has_accelerator: Whether this rank occupies an accelerator. Used by
                the Gateway's ``serves()`` to prevent "a family needing
                ``needs_accelerator`` landing on a rank without a card"
                (maniskill is the first ``gpu_batched`` family; this field used to
                be hardcoded to ``False``, so it could never get a rank).
            reap_interval_seconds: Interval of the lease-recovery loop in
                ``run()``.
            time_source: Time source; tests can inject one.
        """
        self.worker_rank = worker_rank
        self.group_name = group_name
        self.node_id = node_id
        self.max_sessions = max_sessions
        self.default_policy_id = default_policy_id
        self.policy_family = policy_family
        self.policy_device = policy_device
        self.policy_dtype = policy_dtype
        self.policy_constraints = dict(policy_constraints or {})
        self.supported_families: tuple[str, ...] = tuple(supported_families)
        self.pools = EnvPoolManager(
            seed_offset=seed_offset, total_num_processes=total_num_processes
        )
        self.sessions: dict[SessionId, SessionSlot] = {}
        # One pool-build lock per env spec digest: only one task is ever allowed
        # to actually build a given pool.
        self._pool_locks: dict[str, asyncio.Lock] = {}
        self.inference = InferenceClient(worker_rank=worker_rank, group_name=group_name)
        self.has_accelerator = bool(has_accelerator)
        self.coalescer = SlotGroupCoalescer(
            enabled=bool(coalesce_slot_groups),
            window_seconds=max(0.0, float(coalesce_window_ms)) / 1000.0,
        )
        self.sinks = TransitionSinkHub()
        self.active: dict[RequestId, ActiveRequest] = {}
        self.stale_command_count = 0
        self.rejected_command_count = 0
        self.commands: Any = None
        self.reap_interval_seconds = reap_interval_seconds
        self.reaped_session_count = 0
        self.shrunk_slot_count = 0
        self.command_loop_error_count = 0
        self.control_loop_error_count = 0
        self.run_exit_reason = ""
        self._loops: list[asyncio.Task[None]] = []
        self._inflight: set[asyncio.Task[None]] = set()
        self._now = time_source
        self._serving = False

    # ------------------------------------------------------------ Persistent loops

    def init_worker(self, channels: dict[str, Any] | None = None) -> None:
        """Called once by the launcher to attach channels (interface isomorphic
        to ``ToolWorker``).

        Args:
            channels: Channel table. ``"inference"`` is a
                ``transport.base.InferenceChannel``; ``"commands"`` (M3) is a
                ``transport.base.WorkerCommandEndpoint``, i.e. the worker-side
                view of the three command / control / result-return channels. The
                inproc form does not supply ``"commands"``, because
                ``InProcTransport`` calls the handler directly via ``await``.
        """
        table = channels or {}
        channel = table.get("inference")
        if channel is not None:
            self.inference.attach(channel)
        endpoint = table.get("commands")
        if endpoint is not None:
            self.commands = endpoint

    def start_server(self) -> None:
        """Launch persistent tasks (inference response drain; the command /
        control loops are driven by ``run()``)."""
        self._serving = True
        self.inference.start()

    def stop_server(self) -> None:
        """Set the stop flag; asynchronous cleanup is done by ``aclose``."""
        self._serving = False

    @property
    def serving(self) -> bool:
        """Whether the persistent loop is running.

        Returns:
            True if ``start_server`` has been called and ``stop_server`` has not.
        """
        return self._serving

    async def run(self) -> None:
        """Start the command-receive, control-receive, and resource-reclaim
        loops (M3).

        The three loops are **parallel** tasks — this is the concrete
        implementation of an "independent high-priority control channel": cancel
        / heartbeat / binding go through the control loop and are never queued
        behind a command that is currently running an env step. Each command also
        spawns its own task, so different sessions on the same rank can run
        concurrently (serialization within the same session is doubly guaranteed
        by the Gateway's per-session lock plus the worker-side ``active_op``).

        Never raises: any loop's exception is swallowed and counted, because an
        exception crossing the ``WorkerGroup`` boundary would trigger
        ``os.kill(pid, SIGUSR1)``.
        """
        if self.commands is None:
            self.run_exit_reason = "no-command-endpoint"
            return
        self._serving = True
        self.inference.start()
        loop = asyncio.get_running_loop()
        loops = [
            loop.create_task(self._command_loop()),
            loop.create_task(self._control_loop()),
            loop.create_task(self._reap_loop()),
        ]
        self._loops = loops
        try:
            await asyncio.gather(*loops)
        except asyncio.CancelledError:
            self.run_exit_reason = "cancelled"
            raise
        except BaseException:  # noqa: BLE001 - must never raise across the worker boundary
            self.run_exit_reason = "error"
        finally:
            for task in loops:
                task.cancel()
            self._loops = []

    async def _command_loop(self) -> None:
        """Receive commands → spawn a task to execute → return the result."""
        endpoint = self.commands
        assert endpoint is not None  # noqa: S101 - already checked by run()
        while self._serving:
            try:
                envelope, reply_key = await endpoint.get_command()
            except (asyncio.CancelledError, GeneratorExit):
                raise
            except TransportClosed:
                self.run_exit_reason = "command-channel-closed"
                return
            except BaseException:  # noqa: BLE001 - the receive loop must never die
                self.command_loop_error_count += 1
                await asyncio.sleep(0)
                continue
            self._spawn(self._serve_command(envelope, reply_key))

    async def _control_loop(self) -> None:
        """Receive control messages → spawn a task to execute → return the
        result."""
        endpoint = self.commands
        assert endpoint is not None  # noqa: S101 - already checked by run()
        while self._serving:
            try:
                envelope, reply_key = await endpoint.get_control()
            except (asyncio.CancelledError, GeneratorExit):
                raise
            except TransportClosed:
                self.run_exit_reason = "control-channel-closed"
                return
            except BaseException:  # noqa: BLE001 - the receive loop must never die
                self.control_loop_error_count += 1
                await asyncio.sleep(0)
                continue
            self._spawn(self._serve_control(envelope, reply_key))

    async def _reap_loop(self) -> None:
        """Periodically reclaim sessions whose lease has expired, and shrink idle
        dynamic slots (the third loop).

        Shrinking only takes effect for ``per_slot`` pools that implement
        ``DynamicSlotPool`` (see ``EnvPool.dynamic`` / ``EnvPool.shrink_idle``);
        ``lockstep_vector`` pools and families that do not support dynamic growth
        simply return 0 from ``shrink_idle`` and are unaffected.
        """
        while self._serving:
            await asyncio.sleep(self.reap_interval_seconds)
            try:
                reaped = await self.recover_expired_sessions()
            except (asyncio.CancelledError, GeneratorExit):
                raise
            except BaseException:  # noqa: BLE001 - the reclaim loop must never die
                continue
            self.reaped_session_count += len(reaped)
            try:
                self.shrunk_slot_count += await self._shrink_idle_pools()
            except (asyncio.CancelledError, GeneratorExit):
                raise
            except BaseException:  # noqa: BLE001 - the reclaim loop must never die
                continue

    async def _shrink_idle_pools(self) -> int:
        """Attempt to shrink trailing contiguous idle dynamic slots across all of
        this rank's environment pools.

        Returns:
            Total number of slots actually shrunk this round.
        """
        total = 0
        for pool in list(self.pools.pools.values()):
            total += await pool.shrink_idle()
        return total

    def _spawn(self, coroutine: Any) -> None:
        task = asyncio.get_running_loop().create_task(coroutine)
        self._inflight.add(task)
        task.add_done_callback(self._inflight.discard)

    async def _serve_command(self, envelope: CommandEnvelope, reply_key: str) -> None:
        result = await self.handle_command(envelope)
        endpoint = self.commands
        if endpoint is not None:
            await endpoint.put_result(reply_key, result)

    async def _serve_control(self, envelope: ControlEnvelope, reply_key: str) -> None:
        result = await self.handle_control(envelope)
        endpoint = self.commands
        if endpoint is not None:
            await endpoint.put_result(reply_key, result)

    async def aclose(self) -> None:
        """Release all sessions, pools, and background tasks."""
        self.stop_server()
        for task in list(self._loops):
            task.cancel()
        for task in list(self._loops):
            with contextlib.suppress(BaseException):
                await task
        self._loops = []
        for task in list(self._inflight):
            task.cancel()
        for task in list(self._inflight):
            with contextlib.suppress(BaseException):
                await task
        self._inflight.clear()
        await self.inference.stop()
        endpoint = self.commands
        if endpoint is not None:
            endpoint.close()
        for session_id in list(self.sessions):
            with contextlib.suppress(BaseException):
                await self.release_binding(session_id)
        self.pools.close()

    def worker_info(self) -> EnvWorkerInfo:
        """Generate the capabilities and capacity registered with the Gateway.

        Returns:
            An ``EnvWorkerInfo``; ``capabilities`` covers the families this rank
            can serve.
        """
        return EnvWorkerInfo(
            worker_rank=self.worker_rank,
            group_name=self.group_name,
            node_id=self.node_id,
            capabilities=self.capabilities(),
            served_env_digests=sorted(self.pools.pools),
            max_sessions=self.max_sessions,
            active_sessions=len(self.sessions),
            has_accelerator=self.has_accelerator,
            heartbeat_at=self._now(),
        )

    def capabilities(self) -> dict[str, EnvFamilyCapability]:
        """Return the family capability table declared by this rank.

        Returns:
            Mapping from family name to capability; families outside
            ``supported_families`` do not appear.
        """
        table = dict(self.pools.capabilities())
        for family in self.supported_families:
            if family not in table:
                table[family] = get_env_family(family).capability
        return table

    # -------------------------------------------------------------- Command dispatch

    async def handle_command(self, envelope: CommandEnvelope) -> ResultEnvelope:
        """Handle one environment command (a total function).

        Before dispatch, validate the triple: a ``binding_token`` mismatch →
        ``STALE_BINDING``; an ``episode_id`` or ``operation_seq`` behind the
        worker's current value → discard the late command and count it. Mutating
        operations additionally check ``active_op`` (the Gateway already blocks
        one layer; this is the second line of defense).

        Args:
            envelope: The command.

        Returns:
            A response envelope; never raises.
        """
        slot: SessionSlot | None = None
        active: ActiveRequest | None = None
        mutating = envelope.operation in MUTATING_OPERATIONS
        try:
            slot = self._require_slot(envelope.session_id, envelope.binding_token)
            self._reject_stale(slot, envelope)
            handler = self._COMMAND_HANDLERS.get(envelope.operation)
            if handler is None:
                return self._failure(
                    envelope,
                    make_error(
                        ErrorCode.INVALID_ARGUMENT,
                        f"unsupported command operation: {envelope.operation.name}",
                    ),
                )
            if mutating:
                self._begin_active_slot(slot, envelope)
                active = self._register_active(envelope)
            try:
                if mutating:
                    async with slot.lock:
                        value = await handler(self, slot, envelope, active)
                else:
                    value = await handler(self, slot, envelope, active)
            finally:
                if mutating:
                    self._end_active_slot(slot, envelope)
            return ResultEnvelope(
                request_id=envelope.request_id,
                session_id=envelope.session_id,
                operation=envelope.operation,
                state=OperationState.SUCCEEDED,
                value=value,
                side_effect_applied=self._side_effect_of(value, active, mutating),
                worker_summary=slot.summary(self.worker_rank, self.group_name),
            )
        except BaseException as exc:  # noqa: BLE001 - must never raise across the worker boundary
            info = normalize_exception(
                exc,
                side_effect_applied=active.side_effect_applied
                if active is not None
                else False,
            )
            return self._failure(
                envelope,
                info,
                worker_summary=(
                    slot.summary(self.worker_rank, self.group_name)
                    if slot is not None
                    else None
                ),
            )
        finally:
            if active is not None:
                active.stage = STAGE_DONE
                self.active.pop(envelope.request_id, None)

    async def handle_control(self, envelope: ControlEnvelope) -> ResultEnvelope:
        """Handle one control message (a total function).

        Args:
            envelope: The control message.

        Returns:
            A response envelope; never raises.
        """
        try:
            handler = self._CONTROL_HANDLERS.get(envelope.operation)
            if handler is None:
                return ResultEnvelope(
                    request_id=envelope.request_id,
                    session_id=envelope.session_id,
                    operation=envelope.operation,
                    state=OperationState.FAILED,
                    error=make_error(
                        ErrorCode.INVALID_ARGUMENT,
                        f"unsupported control operation: {envelope.operation.name}",
                    ),
                )
            value = await handler(self, envelope)
            return ResultEnvelope(
                request_id=envelope.request_id,
                session_id=envelope.session_id,
                operation=envelope.operation,
                state=OperationState.SUCCEEDED,
                value=value,
            )
        except BaseException as exc:  # noqa: BLE001 - must never raise across the worker boundary
            return ResultEnvelope(
                request_id=envelope.request_id,
                session_id=envelope.session_id,
                operation=envelope.operation,
                state=OperationState.FAILED,
                error=normalize_exception(exc),
            )

    def _failure(
        self,
        envelope: CommandEnvelope,
        error: RuntimeErrorInfo,
        *,
        worker_summary: WorkerSummary | None = None,
    ) -> ResultEnvelope:
        """Wrap an error into a response envelope, mapping ``CANCELLED`` to the
        corresponding operation state.

        Args:
            envelope: The command that triggered the failure.
            error: The normalized error.
            worker_summary: A non-authoritative worker summary.

        Returns:
            A response envelope with ``state`` of ``CANCELLED`` or ``FAILED``.
        """
        state = (
            OperationState.CANCELLED
            if error.code is ErrorCode.CANCELLED
            else OperationState.FAILED
        )
        return ResultEnvelope(
            request_id=envelope.request_id,
            session_id=envelope.session_id,
            operation=envelope.operation,
            state=state,
            error=error,
            side_effect_applied=error.side_effect_applied,
            worker_summary=worker_summary,
        )

    @staticmethod
    def _side_effect_of(
        value: Any, active: ActiveRequest | None, mutating: bool
    ) -> bool:
        """Determine whether a successful execution produced an environment side
        effect.

        Args:
            value: Operation payload.
            active: Stage tracking record.
            mutating: Whether this is a mutating operation.

        Returns:
            The flag carried by the payload takes precedence; falls back to stage
            tracking when missing, then to "mutating operations are assumed to
            have a side effect".
        """
        declared = getattr(value, "side_effect_applied", None)
        if isinstance(declared, bool):
            return declared
        if active is not None:
            return active.side_effect_applied
        return mutating

    # ------------------------------------------------------------ Slots and stages

    def _require_slot(
        self, session_id: SessionId, binding_token: BindingToken | None
    ) -> SessionSlot:
        """Get a session slot and validate the binding identifier.

        Args:
            session_id: Session identifier.
            binding_token: The binding identifier recorded by the Gateway;
                ``None`` skips validation.

        Returns:
            The session slot.

        Raises:
            RuntimeApiError: The session is unbound (``UNKNOWN_SESSION``), or the
                token has expired (``STALE_BINDING``, since the old token becomes
                invalid immediately once a slot is released and reused).
        """
        slot = self.sessions.get(session_id)
        if slot is None:
            raise RuntimeApiError(
                make_error(
                    ErrorCode.UNKNOWN_SESSION,
                    f"env worker {self.worker_rank} has no binding for {session_id}",
                    session_id=session_id,
                    worker_rank=self.worker_rank,
                )
            )
        if binding_token is not None and binding_token != slot.binding_token:
            self.stale_command_count += 1
            raise RuntimeApiError(
                make_error(
                    ErrorCode.STALE_BINDING,
                    f"binding token mismatch for {session_id}",
                    session_id=session_id,
                )
            )
        return slot

    def _reject_stale(self, slot: SessionSlot, envelope: CommandEnvelope) -> None:
        """Reject a late-arriving command.

        Args:
            slot: The session slot.
            envelope: The command.

        Raises:
            RuntimeApiError: If ``episode_id`` is behind the current episode, or
                ``operation_seq`` is not greater than the highest executed
                sequence number (``STALE_BINDING``); both cases are counted.
        """
        if (
            envelope.episode_id is not None
            and slot.episode_id is not None
            and envelope.episode_id < slot.episode_id
        ):
            self.stale_command_count += 1
            raise RuntimeApiError(
                make_error(
                    ErrorCode.STALE_BINDING,
                    f"late command from episode {envelope.episode_id}; "
                    f"session {slot.session_id} is on episode {slot.episode_id}",
                    session_id=slot.session_id,
                    command_episode_id=envelope.episode_id,
                    current_episode_id=slot.episode_id,
                )
            )
        if (
            envelope.operation_seq is not None
            and envelope.operation_seq <= slot.last_operation_seq
        ):
            self.stale_command_count += 1
            raise RuntimeApiError(
                make_error(
                    ErrorCode.STALE_BINDING,
                    f"late command seq {envelope.operation_seq}; session "
                    f"{slot.session_id} already executed {slot.last_operation_seq}",
                    session_id=slot.session_id,
                    command_operation_seq=envelope.operation_seq,
                    last_operation_seq=slot.last_operation_seq,
                )
            )

    def _begin_active_slot(self, slot: SessionSlot, envelope: CommandEnvelope) -> None:
        if slot.active_op is not None and slot.active_op != envelope.request_id:
            self.rejected_command_count += 1
            raise RuntimeApiError(
                make_error(
                    ErrorCode.SESSION_NOT_READY,
                    f"session {slot.session_id} already runs {slot.active_op}",
                    session_id=slot.session_id,
                    active_operation=slot.active_op,
                )
            )
        slot.active_op = envelope.request_id

    def _end_active_slot(self, slot: SessionSlot, envelope: CommandEnvelope) -> None:
        if slot.active_op == envelope.request_id:
            slot.active_op = None
        if envelope.operation_seq is not None:
            slot.last_operation_seq = max(
                slot.last_operation_seq, int(envelope.operation_seq)
            )

    def _register_active(self, envelope: CommandEnvelope) -> ActiveRequest:
        active = ActiveRequest(
            request_id=envelope.request_id,
            session_id=envelope.session_id,
            operation=envelope.operation,
        )
        self.active[envelope.request_id] = active
        return active

    def _check_cancelled(self, active: ActiveRequest | None) -> None:
        """Check for cancellation intent before producing a side effect.

        Args:
            active: Stage tracking record.

        Raises:
            RuntimeApiError: A cancellation has been registered
                (``CANCELLED``, ``side_effect_applied=False``).
        """
        if active is not None and active.cancel_requested:
            raise RuntimeApiError(
                make_error(
                    ErrorCode.CANCELLED,
                    f"operation {active.request_id} cancelled before the env step",
                    side_effect_applied=False,
                    request_id=active.request_id,
                )
            )

    def _pool_of(self, slot: SessionSlot) -> EnvPool:
        pool = self.pools.find(slot.pool_key)
        if pool is None or pool.core is None:
            raise RuntimeApiError(
                make_error(
                    ErrorCode.ENV_FAILURE,
                    f"env pool {slot.pool_key[:12]} is gone for {slot.session_id}",
                    session_id=slot.session_id,
                )
            )
        return pool

    def _lockstep_lane_count(self, pool_key: str) -> int:
        """Count the number of lanes "currently running" on a vector pool (the
        expected headcount for a coalescing group).

        Running = bound + reset + this episode not yet terminated/truncated.
        Lanes that have already finished do not count: they will never submit
        another command, and waiting for one would stall the whole group until
        the window times out (absent lanes are carried forward by the execution
        core via a hold action).

        This reads ``SessionSlot`` rather than the core, so it **relies on
        ``_sync_lockstep_pool`` having already run before this call**: a lane
        carried to termination by a hold action is only excluded here once it
        has been synced, otherwise every group would have to wait it out to a
        window timeout for nothing. ``_step_slot`` orders things exactly this
        way.

        Args:
            pool_key: The env pool key.

        Returns:
            The number of running lanes, at least 1.
        """
        count = sum(
            1
            for slot in self.sessions.values()
            if slot.pool_key == pool_key
            and slot.episode_id is not None
            and not slot.terminated
            and not slot.truncated
        )
        return max(1, count)

    async def _call_pool_core(
        self,
        pool: EnvPool,
        fn: Callable[..., Any],
        *args: Any,
        side_effect_applied: bool = False,
    ) -> Any:
        """Call a given pool's execution core; serialized on vector pools.

        All slots of a ``lockstep_vector`` pool share one family env, while the
        Gateway dispatches commands concurrently (each command spawns its own
        task, finalized in M3). A GPU-batched family's vector env is a state
        machine: two lanes calling ``reset`` simultaneously collide with
        ManiSkill's ``Once _gpu_apply_all is called, you must call
        _gpu_fetch_all before ...``. So every core call on a vector pool must
        take the pool lock; ``per_slot`` pool slots are independent envs and
        remain fully concurrent as before.

        Args:
            pool: The target pool.
            fn: The execution core method.
            *args: Method arguments.
            side_effect_applied: On failure, honestly annotate whether an
                environment side effect may already have occurred.

        Returns:
            The execution core's return value.
        """
        if not pool.lockstep:
            return await self._call_core(
                fn, *args, side_effect_applied=side_effect_applied
            )
        async with pool.core_lock:
            return await self._call_core(
                fn, *args, side_effect_applied=side_effect_applied
            )

    async def _call_core(
        self, fn: Callable[..., Any], *args: Any, side_effect_applied: bool = False
    ) -> Any:
        """Call the execution core in a thread, normalizing family exceptions
        into ``ENV_FAILURE``.

        The execution core is blocking (a real env is a CPU subproc / GPU
        batched); running it in a thread both avoids blocking the event loop and
        naturally makes "once an env step has started it cannot be cancelled"
        hold true.

        Args:
            fn: The execution core method.
            *args: Method arguments.
            side_effect_applied: On failure, honestly annotate whether an
                environment side effect may already have occurred.

        Returns:
            The execution core's return value.

        Raises:
            RuntimeApiError: Any exception raised by the execution core,
                normalized with ``ENV_FAILURE`` (``TimeoutError`` →
                ``DEADLINE_EXCEEDED``, argument-related exceptions →
                ``INVALID_ARGUMENT``).
        """
        try:
            return await asyncio.to_thread(fn, *args)
        except (asyncio.CancelledError, GeneratorExit):
            raise
        except RuntimeApiError as exc:
            if side_effect_applied and not exc.info.side_effect_applied:
                raise RuntimeApiError(
                    dataclasses.replace(exc.info, side_effect_applied=True)
                ) from exc
            raise
        except BaseException as exc:
            raise RuntimeApiError(
                normalize_exception(
                    exc,
                    default_code=ErrorCode.ENV_FAILURE,
                    side_effect_applied=side_effect_applied,
                )
            ) from exc

    def _stamp(self, observation: Observation, slot: SessionSlot) -> Observation:
        """Stamp an observation produced by the execution core with the session
        identity.

        The execution core only knows slot indices, not sessions / episodes
        (``core`` is session-agnostic); identity fields are filled in by the
        worker.

        Args:
            observation: The observation produced by the execution core.
            slot: The session slot.

        Returns:
            The observation, with ``session_id`` / ``episode_id`` filled in.
        """
        return dataclasses.replace(
            observation,
            session_id=slot.session_id,
            episode_id=slot.episode_id or EpisodeId(0),
        )

    # ------------------------------------------------------------ Environment operations

    async def create_binding(
        self,
        session_id: SessionId,
        env_spec: EnvSpecMsg,
        lease_expiration: float = 0.0,
    ) -> BindingToken:
        """Allocate an env slot for a session already created by the Gateway and
        return an opaque token.

        Args:
            session_id: Session identifier.
            env_spec: Environment spec.
            lease_expiration: Lease expiration time.

        Returns:
            The binding identifier.

        Raises:
            RuntimeApiError: The EnvPool could not allocate a slot.
        """
        existing = self.sessions.get(session_id)
        if existing is not None:
            existing.lease_expiration = lease_expiration or existing.lease_expiration
            return existing.binding_token
        pool = await self._ensure_pool(env_spec)
        slot_index = await pool.acquire_or_grow()
        slot = SessionSlot(
            session_id=session_id,
            binding_token=new_binding_token(),
            pool_key=pool.pool_key,
            slot_index=slot_index,
            lease_expiration=lease_expiration,
        )
        self.sessions[session_id] = slot
        return slot.binding_token

    async def _ensure_pool(self, env_spec: EnvSpecMsg) -> EnvPool:
        """Get or build an environment pool by digest, **building a given digest
        only once** (a race condition found and fixed on GPU in M6).

        Building a pool requires ``await asyncio.to_thread`` (real family
        construction is blocking and can take tens of seconds), while
        ``create_sessions`` accepts requests concurrently (finalized in M3: each
        command spawns its own task). The original implementation called
        ``await asyncio.to_thread(self.pools.ensure_pool, ...)`` directly, so N
        concurrent bindings would **all** see "the pool does not exist yet",
        each build their own pool, and the later build would overwrite the
        earlier one in the table: the earlier session would hold a slot in an
        orphaned pool, while ``_pool_of`` looking up by pool_key would only ever
        find the last pool built — so two sessions would end up on **the same
        lane**.

        Why this was not exposed in M4/M5: both a100 presets had
        ``max_sessions_per_rank`` set to 1 (at the time attributed to an EGL
        limitation, but actually caused by this very race), so a rank only ever
        had one session and could never trigger it. M6's vector pool puts
        multiple lanes on one rank, and the first genuinely concurrent pool build
        surfaced it: the symptom was that all ``reset`` calls succeeded, yet
        ``chunk_step`` reported "lane 1 was never reset", and concurrently
        constructing two sapien envs also corrupted the process's stderr
        (``ValueError: I/O operation on closed file``).

        Args:
            env_spec: Environment spec.

        Returns:
            The pool corresponding to that digest (already ``build``-ed).
        """
        pool_key = env_spec.digest()
        existing = self.pools.find(pool_key)
        if existing is not None and existing.core is not None:
            return existing
        lock = self._pool_locks.get(pool_key)
        if lock is None:
            lock = asyncio.Lock()
            self._pool_locks[pool_key] = lock
        async with lock:
            # Double-check: someone else may have already built it while we
            # waited for the lock.
            existing = self.pools.find(pool_key)
            if existing is not None and existing.core is not None:
                return existing
            return await asyncio.to_thread(self.pools.ensure_pool, env_spec)

    async def release_binding(self, session_id: SessionId) -> None:
        """Release the env slot and delete the session state.

        Args:
            session_id: Session identifier.
        """
        slot = self.sessions.pop(session_id, None)
        if slot is None:
            return
        pool = self.pools.find(slot.pool_key)
        if pool is not None:
            pool.release(slot.slot_index)

    async def renew_lease(self, session_id: SessionId, lease_expiration: float) -> None:
        """Update the session's lease expiration time.

        Args:
            session_id: Session identifier.
            lease_expiration: The new expiration time.
        """
        slot = self.sessions.get(session_id)
        if slot is not None:
            slot.lease_expiration = lease_expiration

    async def recover_expired_sessions(self) -> list[SessionId]:
        """Reclaim sessions whose lease has expired and have no operation in
        progress.

        Returns:
            The reclaimed session ids.
        """
        now = self._now()
        expired = [
            session_id
            for session_id, slot in self.sessions.items()
            if slot.lease_expiration and slot.lease_expiration <= now
            if slot.active_op is None
        ]
        for session_id in expired:
            await self.release_binding(session_id)
        return expired

    async def reset(
        self,
        session_id: SessionId,
        reset_spec: ResetSpec,
        *,
        request_id: RequestId | None = None,
        operation_seq: OperationSeq | None = None,
        active: ActiveRequest | None = None,
    ) -> StepResult:
        """Reset the session and cache the initial observation
        (``episode_id += 1``).

        Args:
            session_id: Session identifier.
            reset_spec: Episode initialization parameters.
            request_id: The request identifier that triggered it.
            operation_seq: The operation sequence number that triggered it.
            active: Stage tracking record.

        Returns:
            A result carrying the new ``episode_id`` and the initial observation.
        """
        slot = self._require_slot(session_id, None)
        self._check_cancelled(active)
        pool = self._pool_of(slot)
        if active is not None:
            active.stage = STAGE_ENV_STEP
        observations = await self._call_pool_core(
            pool, pool.core.reset, [slot.slot_index], reset_spec
        )
        if active is not None:
            active.side_effect_applied = True
        slot.episode_id = EpisodeId((slot.episode_id or 0) + 1)
        slot.step_index = 0
        slot.terminated = False
        slot.truncated = False
        slot.last_observation = self._stamp(observations[0], slot)
        # ``masked_steps_seen`` is deliberately **not** reset to zero: the
        # core-side ``masked_steps`` also accumulates across episodes (see
        # ``LaneState``); zeroing it would only make the watermark lag behind
        # the core. If a difference remains from the previous episode, the
        # first sync after reset will re-read one extra frame (harmless, since
        # both sides already have the same frame at that point) and then
        # self-align.
        return StepResult(
            request_id=request_id or RequestId(""),
            session_id=session_id,
            binding_token=slot.binding_token,
            episode_id=slot.episode_id,
            operation_seq=operation_seq,
            observation=slot.last_observation,
            reward=0.0,
            terminated=False,
            truncated=False,
            info={"reset": True, "episode_id": int(slot.episode_id)},
            side_effect_applied=True,
            executed_horizon=0,
            per_step=None,
        )

    async def _sync_lockstep_slot(self, slot: SessionSlot) -> None:
        """Read back a lane's true state on a vector pool into its
        ``SessionSlot`` (completing the masking semantics).

        A ``lockstep_vector`` pool advances **all** lanes in a single
        ``chunk_step`` call; absent lanes are carried forward by a hold action,
        but ``run_lockstep_chunk`` only returns a ``ChunkOutcome`` to
        participants, and ``_step_slot`` only updates the ``SessionSlot`` of the
        submitter. Skipping this readback has three consequences, none of which
        are "reordering" but rather "stale state":

        1. ``last_observation`` goes stale → ``policy_step`` feeds an old frame
           into inference (the original implementation only patched this in
           ``observe``, not in ``policy_step``);
        2. ``terminated`` / ``truncated`` lag behind → the core-side lane is
           already ``frozen``, but the worker-side ``_require_running_episode``
           still lets it through, so the next ``chunk_step`` call for that lane
           takes the ``if lane.frozen: continue`` branch, returning
           ``executed_horizon=0`` with ``terminated`` still false — **an
           already-successful termination gets silently dropped**, violating the
           hard rule that "success is determined solely by the environment's
           termination signal";
        3. ``_lockstep_lane_count`` keeps counting this already-terminated lane
           as "running", so every subsequent group has to wait it out to the
           ``coalesce_window_ms`` timeout before firing.

        Because point 3 requires **whole-pool** truth, this simply reads back
        all lanes of the pool at once and syncs every ``SessionSlot`` bound to
        that pool: ``lane_status`` does not render or encode images, so reading
        1 lane costs the same as reading 4, and this also saves a second pool
        lock acquisition that a "sync myself, then separately count everyone
        else" approach would need.

        Args:
            slot: The session slot that triggered the sync (its owning pool is
                synced in full).
        """
        pool = self.pools.find(slot.pool_key)
        if pool is None or pool.core is None or not pool.lockstep:
            # A ``per_slot`` pool's slots are independent envs and are never
            # advanced by anyone else.
            return
        await self._sync_lockstep_pool(pool)

    async def _sync_lockstep_pool(self, pool: EnvPool) -> None:
        """Read back the state of every lane in a vector pool at once, and sync
        it to each ``SessionSlot`` bound to that pool.

        Half of being a total function: a readback failure just means "did not
        sync", and must not turn an operation that would otherwise succeed into
        an error, so the exception is swallowed. This is safe precisely because
        **the core side honestly reports frozen participants too**
        (``run_lockstep_chunk`` returns them an outcome with
        ``terminated=True`` and ``executed_horizon=0``), so even if a sync fails
        that termination is not lost — it is just discovered one step later,
        with a slightly less precise error message than a post-sync one would
        have.

        Args:
            pool: The target vector pool.
        """
        bound = [
            slot for slot in self.sessions.values() if slot.pool_key == pool.pool_key
        ]
        if not bound:
            return
        reader = getattr(pool.core, "lane_status", None)
        if reader is None:
            # Declared lockstep but did not implement readback: fall back to
            # "re-read one frame for every running lane", preferring to pay for
            # an extra pool lock over feeding a stale frame to the model.
            # Termination flags cannot be recovered this way (no channel for
            # it), relying entirely on the core's honest reporting for frozen
            # participants as a backstop.
            await self._refresh_lockstep_observations(
                pool, [slot for slot in bound if self._needs_lockstep_frame(slot)]
            )
            return
        try:
            statuses = await self._call_pool_core(
                pool, reader, [slot.slot_index for slot in bound]
            )
        except (asyncio.CancelledError, GeneratorExit):
            raise
        except BaseException:  # noqa: BLE001 - a sync failure must not turn the operation into an error
            return
        stale: list[SessionSlot] = []
        for slot, status in zip(bound, statuses, strict=True):
            if slot.episode_id is None:
                # Not yet reset: no episode state to sync.
                continue
            # First record whether "the worker believed it was still running
            # before this sync": **the very sync that discovers termination**
            # still needs a frame refresh, specifically the frame from the
            # moment of termination (the masking rule of "never pretend nothing
            # moved"); only later syncs skip it.
            was_running = self._needs_lockstep_frame(slot)
            # Termination flags are **monotonic**: only OR them in, never clear
            # a True back to False.
            slot.terminated = slot.terminated or bool(status.terminated)
            slot.truncated = slot.truncated or bool(status.truncated)
            masked = int(status.masked_steps)
            if masked == slot.masked_steps_seen:
                continue
            # The watermark always advances (including for finished lanes),
            # otherwise it would be re-judged "stale" in every subsequent group.
            slot.masked_steps_seen = masked
            if was_running:
                stale.append(slot)
        await self._refresh_lockstep_observations(pool, stale)

    @staticmethod
    def _needs_lockstep_frame(slot: SessionSlot) -> bool:
        """Whether this slot still needs its observation refreshed to follow the
        core side.

        A lane whose episode has **already** finished no longer gets frame
        refreshes: the core-side lane keeps being carried forward by hold
        actions in every subsequent group (``run_lockstep_chunk`` counts absent
        lanes unconditionally, regardless of ``frozen``), but that movement
        belongs to **someone else's** episode, and stamping it onto this
        session's last frame would be dishonest; it would also make every
        finished lane pay for an ``observe`` call in every subsequent group for
        nothing.

        Note that what is being checked is "was it still running **before this
        sync**": the very sync that discovers termination still needs a frame
        refresh, otherwise the frame from the moment of termination would never
        be read back (which is why ``_sync_lockstep_pool`` captures the value
        before the OR).

        Args:
            slot: The session slot.

        Returns:
            Whether a frame refresh is needed.
        """
        return (
            slot.episode_id is not None and not slot.terminated and not slot.truncated
        )

    async def _refresh_lockstep_observations(
        self, pool: EnvPool, slots: Sequence[SessionSlot]
    ) -> None:
        """Re-read frames for several lanes from the execution core **in one
        call** and stamp them (vector-pool specific).

        Issues a single ``observe`` call rather than calling per lane:
        ``EnvExecutionCore.observe`` has had a batch signature since M1, so
        calling it per lane would pay for a pool lock and thread hop per lane.

        Failures are swallowed for the same reason as in
        ``_sync_lockstep_pool``: a failed frame refresh just means "did not get
        refreshed".

        Args:
            pool: The pool this slot belongs to.
            slots: The session slots awaiting a frame refresh.
        """
        if not slots or pool.core is None:
            return
        ordered_slots = sorted(slots, key=lambda slot: slot.slot_index)
        try:
            fresh = await self._call_pool_core(
                pool, pool.core.observe, [slot.slot_index for slot in ordered_slots]
            )
        except (asyncio.CancelledError, GeneratorExit):
            raise
        except BaseException:  # noqa: BLE001 - same reasoning as above
            return
        for slot, observation in zip(ordered_slots, fresh, strict=True):
            slot.last_observation = self._stamp(observation, slot)
            slot.step_index = slot.last_observation.step_index

    async def observe(self, session_id: SessionId) -> Observation:
        """Return the cached observation without touching the environment.

        On a vector pool (``lockstep_vector``), first runs
        ``_sync_lockstep_slot`` to read back state: an absent lane may have been
        carried forward by a hold action, so the worker's cached copy may
        already be stale, and reporting it as "unchanged" would be dishonest.

        Args:
            session_id: Session identifier.

        Returns:
            The current observation.

        Raises:
            RuntimeApiError: Not yet reset (``SESSION_NOT_READY``).
        """
        slot = self._require_slot(session_id, None)
        if slot.last_observation is None:
            raise RuntimeApiError(
                make_error(
                    ErrorCode.SESSION_NOT_READY,
                    f"session {session_id} has no observation; call reset first",
                    session_id=session_id,
                )
            )
        await self._sync_lockstep_slot(slot)
        return slot.last_observation

    async def action_step(
        self,
        session_id: SessionId,
        actions: PayloadRef,
        *,
        request_id: RequestId | None = None,
        operation_seq: OperationSeq | None = None,
        active: ActiveRequest | None = None,
    ) -> StepResult:
        """Convert an external action into an environment action and execute one
        ``chunk_step``.

        Args:
            session_id: Session identifier.
            actions: The action payload, shape ``[chunk, action_dim]``.
            request_id: The request identifier that triggered it.
            operation_seq: The operation sequence number that triggered it.
            active: Stage tracking record.

        Returns:
            The step result.
        """
        payload_module.check_payload_budget(actions, context="action_step")
        block = self._decode_actions(actions)
        return await self.step_env(
            session_id,
            block,
            request_id=request_id,
            operation_seq=operation_seq,
            active=active,
        )

    async def policy_step(
        self,
        session_id: SessionId,
        policy_request: PolicyRequest,
        *,
        request_id: RequestId | None = None,
        operation_seq: OperationSeq | None = None,
        active: ActiveRequest | None = None,
        deadline: float | None = None,
        priority: Priority = Priority.INTERACTIVE,
        application_id: str = "",
    ) -> StepResult:
        """Atomically execute observation → inference → ``chunk_step``.

        Args:
            session_id: Session identifier.
            policy_request: Inference parameters.
            request_id: The request identifier that triggered it.
            operation_seq: The operation sequence number that triggered it.
            active: Stage tracking record.
            deadline: Absolute unix timestamp.
            priority: Scheduling priority.
            application_id: Tenant identifier, used by the scheduler for
                fairness decisions.

        Returns:
            The step result.
        """
        slot = self._require_slot(session_id, None)
        return await self._policy_step_inner(
            slot,
            policy_request,
            request_id=request_id,
            operation_seq=operation_seq,
            active=active,
            deadline=deadline,
            priority=priority,
            application_id=application_id,
        )

    async def _policy_step_inner(
        self,
        slot: SessionSlot,
        policy_request: PolicyRequest,
        *,
        request_id: RequestId | None,
        operation_seq: OperationSeq | None,
        active: ActiveRequest | None,
        deadline: float | None = None,
        priority: Priority = Priority.INTERACTIVE,
        application_id: str = "",
    ) -> StepResult:
        # First read back the vector pool's lane state: while absent, this lane
        # may already have been carried to termination by a hold action, in
        # which case the correct behavior is to report EPISODE_TERMINATED
        # rather than run inference on a stale frame.
        await self._sync_lockstep_slot(slot)
        self._require_running_episode(slot)
        self._check_cancelled(active)
        observation = slot.last_observation
        if observation is None:
            raise RuntimeApiError(
                make_error(
                    ErrorCode.SESSION_NOT_READY,
                    f"session {slot.session_id} has no observation; call reset first",
                    session_id=slot.session_id,
                )
            )
        started_inference = time.perf_counter()
        response = await self.request_inference(
            slot.session_id,
            observation,
            policy_request,
            request_id=request_id,
            operation_seq=operation_seq,
            active=active,
            deadline=deadline,
            priority=priority,
            application_id=application_id,
        )
        inference_latency_s = time.perf_counter() - started_inference
        if response.error is not None:
            raise RuntimeApiError(response.error)
        self._check_cancelled(active)
        block = self._decode_actions(response.actions)
        if policy_request.actions_per_chunk:
            block = block[: policy_request.actions_per_chunk]
        started_env_step = time.perf_counter()
        result = await self._step_slot(
            slot,
            block,
            request_id=request_id,
            operation_seq=operation_seq,
            active=active,
        )
        env_step_latency_s = time.perf_counter() - started_env_step
        return dataclasses.replace(
            result,
            info={
                **result.info,
                "model_version": response.model_version,
                "policy_id": policy_request.policy_id or self.default_policy_id,
                # The client-side policy_step is an atomic call and cannot be
                # split into an inference segment and an env.step segment (a
                # known methodological limit); these two fields relay
                # server-known boundaries without changing the RPC contract
                # itself.
                "inference_latency_s": inference_latency_s,
                "env_step_latency_s": env_step_latency_s,
            },
        )

    async def policy_infer(
        self,
        session_id: SessionId,
        policy_request: PolicyRequest,
        *,
        request_id: RequestId | None = None,
        active: ActiveRequest | None = None,
        deadline: float | None = None,
        priority: Priority = Priority.INTERACTIVE,
        application_id: str = "",
    ) -> PolicyInferResult:
        """Only runs "cached observation → inference", and does **not** execute
        ``chunk_step`` (an M5 addition).

        Similar to ``observe``: read-only, does not allocate an
        ``operation_seq``, and does **not** require the episode to still be in
        progress (the legacy vla_server has no episode concept, and asking for
        one more action after termination should not be an error). The only
        requirement is "already reset, with one observation frame available".

        Args:
            session_id: Session identifier.
            policy_request: Inference parameters.
            request_id: The request identifier that triggered it.
            active: Stage tracking record.
            deadline: Absolute unix timestamp.
            priority: Scheduling priority.
            application_id: Tenant identifier.

        Returns:
            The action-chunk result.

        Raises:
            RuntimeApiError: Not yet reset (``SESSION_NOT_READY``), or inference
                failed (``POLICY_FAILURE``, etc., normalized by the
                RolloutWorker and re-raised as-is).
        """
        slot = self._require_slot(session_id, None)
        self._check_cancelled(active)
        # A read-only operation, but "reading" on a vector pool still requires a
        # readback first: otherwise a stale frame would be sent to inference,
        # and the info would report lagging terminated / truncated flags.
        await self._sync_lockstep_slot(slot)
        observation = slot.last_observation
        if observation is None:
            raise RuntimeApiError(
                make_error(
                    ErrorCode.SESSION_NOT_READY,
                    f"session {slot.session_id} has no observation; call reset first",
                    session_id=slot.session_id,
                )
            )
        response = await self.request_inference(
            slot.session_id,
            observation,
            policy_request,
            request_id=request_id,
            operation_seq=None,
            active=active,
            deadline=deadline,
            priority=priority,
            application_id=application_id,
        )
        if response.error is not None:
            raise RuntimeApiError(response.error)
        return PolicyInferResult(
            request_id=response.request_id,
            session_id=slot.session_id,
            binding_token=slot.binding_token,
            episode_id=slot.episode_id,
            actions=response.actions,
            model_version=response.model_version,
            auxiliary_outputs=dict(response.auxiliary_outputs),
            observation_step_index=int(observation.step_index),
            info={
                "policy_id": policy_request.policy_id or self.default_policy_id,
                "terminated": bool(slot.terminated),
                "truncated": bool(slot.truncated),
            },
        )

    async def request_inference(
        self,
        session_id: SessionId,
        observation: Observation,
        policy_request: PolicyRequest,
        *,
        request_id: RequestId | None = None,
        operation_seq: OperationSeq | None = None,
        active: ActiveRequest | None = None,
        deadline: float | None = None,
        priority: Priority = Priority.INTERACTIVE,
        application_id: str = "",
    ) -> ActionResponse:
        """Submit an inference request to the RolloutWorker and wait for the
        corresponding action.

        The wait can be cancelled: ``cancel_request`` cancels the Future here,
        and a subsequently arriving action is judged late and counted in
        ``InferenceClient.deliver``.

        Args:
            session_id: Session identifier.
            observation: Inference input.
            policy_request: Inference parameters.
            request_id: The request identifier that triggered it.
            operation_seq: The operation sequence number that triggered it.
            active: Stage tracking record.
            deadline: Absolute unix timestamp.
            priority: Scheduling priority.
            application_id: Tenant identifier.

        Returns:
            The action-chunk response.

        Raises:
            RuntimeApiError: Cancelled while waiting (``CANCELLED``, no side
                effect).
        """
        slot = self._require_slot(session_id, None)
        policy_id = policy_request.policy_id or self.default_policy_id
        request = InferenceRequest(
            request_id=request_id or RequestId(""),
            session_id=session_id,
            binding_token=slot.binding_token,
            episode_id=slot.episode_id,
            operation_seq=operation_seq,
            policy_id=policy_id,
            observation=observation,
            instruction_override=policy_request.instruction_override,
            inference_parameters=dict(policy_request.inference_parameters),
            routing_token=self.inference.routing_token,
            compat_key=compute_compat_key(
                policy_id=policy_id,
                model_version="",
                obs_schema_digest=obs_schema_digest(observation),
                inference_parameters=policy_request.inference_parameters,
                device=self.policy_device,
                dtype=self.policy_dtype,
                policy_family=self.policy_family,
                constraints=self.policy_constraints or None,
            ),
            deadline=deadline,
            priority=priority,
            application_id=application_id,
        )
        if active is not None:
            active.stage = STAGE_INFERENCE
        try:
            return await self.inference.submit(request)
        except asyncio.CancelledError as exc:
            if active is not None and active.cancel_requested:
                raise RuntimeApiError(
                    make_error(
                        ErrorCode.CANCELLED,
                        f"operation {request.request_id} cancelled while waiting "
                        "for inference; the late action will be discarded",
                        side_effect_applied=False,
                        request_id=request.request_id,
                    )
                ) from exc
            raise

    async def step_env(
        self,
        session_id: SessionId,
        raw_actions: Any,
        *,
        request_id: RequestId | None = None,
        operation_seq: OperationSeq | None = None,
        active: ActiveRequest | None = None,
    ) -> StepResult:
        """Execute action conversion and ``chunk_step``, updating observation and
        termination state.

        Args:
            session_id: Session identifier.
            raw_actions: The model's raw actions, or an already-decoded
                ``[chunk, action_dim]`` array.
            request_id: The request identifier that triggered it.
            operation_seq: The operation sequence number that triggered it.
            active: Stage tracking record.

        Returns:
            The step result.
        """
        slot = self._require_slot(session_id, None)
        block = (
            raw_actions
            if isinstance(raw_actions, np.ndarray)
            else self._decode_actions(raw_actions)
        )
        return await self._step_slot(
            slot,
            block,
            request_id=request_id,
            operation_seq=operation_seq,
            active=active,
        )

    async def _step_slot(
        self,
        slot: SessionSlot,
        block: np.ndarray,
        *,
        request_id: RequestId | None,
        operation_seq: OperationSeq | None,
        active: ActiveRequest | None,
    ) -> StepResult:
        # Sync once more: after ``_policy_step_inner``'s earlier sync, a full
        # inference round has been awaited (a real VLA can take seconds), during
        # which someone else on the same pool could easily have opened a group
        # that carried this lane to termination. ``action_step`` also enters
        # here, so placing the readback here covers every entry point.
        await self._sync_lockstep_slot(slot)
        self._require_running_episode(slot)
        self._check_cancelled(active)
        pool = self._pool_of(slot)
        if active is not None:
            # After this step, cancellation can only wait for completion: the
            # action cannot be rolled back.
            active.stage = STAGE_ENV_STEP
            active.side_effect_applied = True
        if pool.lockstep and self.coalescer.enabled:
            # Vector pool: merge into one chunk_step call for the same pool and
            # tick (the real coalescing implementation). Absent lanes are
            # carried forward and counted by the execution core via a hold
            # action.
            outcome = await self.coalescer.submit(
                pool_key=slot.pool_key,
                slot_index=slot.slot_index,
                block=block,
                chunk_len=int(np.asarray(block).shape[0]),
                expected=self._lockstep_lane_count(slot.pool_key),
                execute=lambda slots, blocks: self._call_pool_core(
                    pool, pool.core.chunk_step, slots, blocks, side_effect_applied=True
                ),
            )
        else:
            outcomes = await self._call_pool_core(
                pool,
                pool.core.chunk_step,
                [slot.slot_index],
                [block],
                side_effect_applied=True,
            )
            outcome = outcomes[0]
        # The termination flag only ever gets a monotonic OR: this slot's
        # ``terminated`` may have just been set True by a readback (carried to
        # termination while absent), while the core's report for a frozen lane
        # has ``executed_horizon`` of 0; a direct assignment would clear that
        # True back. Zeroing only happens in ``reset``, which does not go
        # through this path.
        slot.terminated = slot.terminated or bool(outcome.terminated)
        slot.truncated = slot.truncated or bool(outcome.truncated)
        if outcome.observation is not None:
            slot.last_observation = self._stamp(outcome.observation, slot)
            slot.step_index = outcome.observation.step_index
        return StepResult(
            request_id=request_id or RequestId(""),
            session_id=slot.session_id,
            binding_token=slot.binding_token,
            episode_id=slot.episode_id,
            operation_seq=operation_seq,
            observation=slot.last_observation,
            reward=float(outcome.reward),
            terminated=slot.terminated,
            truncated=slot.truncated,
            info={
                **outcome.info,
                "per_step_obs_available": outcome.per_step_obs_available,
            },
            side_effect_applied=True,
            executed_horizon=int(outcome.executed_horizon),
            per_step=outcome.per_step,
        )

    async def run_episode(
        self,
        session_id: SessionId,
        episode_request: EpisodeRequest,
        *,
        request_id: RequestId | None = None,
        operation_seq: OperationSeq | None = None,
        active: ActiveRequest | None = None,
        application_id: str = "",
    ) -> EpisodeResult:
        """Loop-execute ``policy_step`` until one of four termination conditions
        is hit.

        Bounded by ``max_steps`` / ``deadline`` / cancel /
        terminated·truncated; transitions go through a sink rather than being
        returned to the Gateway.

        A vector pool adds one extra termination path: this lane may be carried
        to termination by a hold action **while absent**. That termination is
        only discovered by a readback inside the loop, surfacing as
        ``_policy_step_inner`` raising ``EPISODE_TERMINATED``. This is **not**
        an error on the caller's part, nor an invalid episode — "success is
        determined solely by the environment's termination signal" — so it is
        folded here into a normal termination; otherwise ``eval_adapter`` would
        record it as ``Err`` with ``valid=False``, erasing a genuinely
        successful environment outcome from the denominator of the success
        rate. Conversely, an episode that has already ended **before** entering
        this method is still a caller error (consistent with ``policy_step`` /
        ``action_step``), so the entry check syncs once first to make it
        authoritative.

        Args:
            session_id: Session identifier.
            episode_request: Episode parameters.
            request_id: The request identifier that triggered it.
            operation_seq: The operation sequence number that triggered it.
            active: Stage tracking record.
            application_id: Tenant identifier.

        Returns:
            An ``EpisodeResult``; transitions are not returned with the result.

        Raises:
            RuntimeApiError: The episode has already ended on entry
                (``EPISODE_TERMINATED``) or has not been reset
                (``SESSION_NOT_READY``).
        """
        slot = self._require_slot(session_id, None)
        await self._sync_lockstep_slot(slot)
        self._require_running_episode(slot)
        steps = 0
        horizon = 0
        total_reward = 0.0
        stop_reason = "max_steps"
        last: StepResult | None = None
        while True:
            reason = self.should_finish_episode(
                session_id, episode_request, steps_taken=steps, active=active
            )
            if reason is not None:
                stop_reason = reason
                break
            try:
                last = await self._policy_step_inner(
                    slot,
                    episode_request.policy,
                    request_id=request_id,
                    operation_seq=operation_seq,
                    active=active,
                    deadline=episode_request.deadline,
                    application_id=application_id,
                )
            except RuntimeApiError as exc:
                if exc.info.code is not ErrorCode.EPISODE_TERMINATED:
                    raise
                # Carried to termination while absent: the readback has already
                # written the true flag into the slot, and this step executed
                # zero actions (``side_effect_applied=False``), so it does not
                # count as a step and is wrapped up as a normal termination.
                stop_reason = "terminated" if slot.terminated else "truncated"
                break
            steps += 1
            horizon += last.executed_horizon
            total_reward += last.reward
            await self.publish_transition(episode_request, last)
        return EpisodeResult(
            request_id=request_id or RequestId(""),
            session_id=session_id,
            episode_id=slot.episode_id,
            num_policy_steps=steps,
            executed_horizon=horizon,
            total_reward=total_reward,
            terminated=slot.terminated,
            truncated=slot.truncated,
            stop_reason=stop_reason,
            sink_id=episode_request.sink_id,
            last_observation=slot.last_observation,
            info={
                "side_effect_applied": steps > 0,
                "max_steps": episode_request.max_steps,
            },
        )

    async def extension_call(
        self,
        session_id: SessionId,
        namespace: str,
        method: str,
        args: dict[str, Any],
    ) -> dict[str, Any]:
        """Route to the execution core's extension table.

        An undeclared method is reported by the execution core as
        ``UNSUPPORTED_EXTENSION`` rather than crashing; the family's capability
        table already declares its supported surface ahead of this.

        Args:
            session_id: Session identifier.
            namespace: Extension namespace.
            method: Extension method name.
            args: Method arguments.

        Returns:
            A structured result.

        Raises:
            RuntimeApiError: The family does not declare this method
                (``UNSUPPORTED_EXTENSION``).
        """
        slot = self._require_slot(session_id, None)
        pool = self._pool_of(slot)
        capability = get_env_family(pool.env_spec.env_family).capability
        full_name = f"{namespace}.{method}"
        if full_name not in capability.extensions:
            raise RuntimeApiError(
                make_error(
                    ErrorCode.UNSUPPORTED_EXTENSION,
                    f"env family {pool.env_spec.env_family!r} does not declare "
                    f"extension {full_name!r}",
                    namespace=namespace,
                    method=method,
                    supported=sorted(capability.extensions),
                )
            )
        return await self._call_pool_core(
            pool, pool.core.extension, slot.slot_index, namespace, method, dict(args)
        )

    async def publish_transition(
        self, episode_request: EpisodeRequest, step_result: StepResult
    ) -> None:
        """Write one transition to a sink.

        Args:
            episode_request: Episode parameters (containing ``sink_id``).
            step_result: The single-step result.
        """
        self.sinks.publish(
            episode_request.sink_id,
            {
                "session_id": str(step_result.session_id),
                "episode_id": (
                    int(step_result.episode_id)
                    if step_result.episode_id is not None
                    else None
                ),
                "step_index": (
                    step_result.observation.step_index
                    if step_result.observation is not None
                    else None
                ),
                "reward": step_result.reward,
                "terminated": step_result.terminated,
                "truncated": step_result.truncated,
                "executed_horizon": step_result.executed_horizon,
            },
        )

    def should_finish_episode(
        self,
        session_id: SessionId,
        episode_request: EpisodeRequest,
        *,
        steps_taken: int = 0,
        active: ActiveRequest | None = None,
    ) -> str | None:
        """Check terminated / truncated / cancel / deadline / max steps.

        Args:
            session_id: Session identifier.
            episode_request: Episode parameters.
            steps_taken: The number of ``policy_step`` calls already executed.
            active: Stage tracking record, used for cancel evaluation.

        Returns:
            The ``stop_reason`` if it should finish, otherwise ``None``.
        """
        slot = self.sessions.get(session_id)
        if slot is None:
            return "error"
        if slot.terminated:
            return "terminated"
        if slot.truncated and episode_request.stop_on_truncation:
            return "truncated"
        if active is not None and active.cancel_requested:
            return "cancelled"
        if episode_request.deadline is not None and self._now() >= (
            episode_request.deadline
        ):
            return "deadline"
        if steps_taken >= episode_request.max_steps:
            return "max_steps"
        return None

    async def cancel_request(self, request_id: RequestId) -> bool:
        """Cancel an operation that has not yet started its env step; one
        already started can only be waited out.

        Args:
            request_id: The target request.

        Returns:
            Whether it was successfully cancelled before producing a side
            effect.
        """
        active = self.active.get(request_id)
        if active is None:
            return False
        active.cancel_requested = True
        if not active.cancellable:
            return False
        future = self.inference.pending.get(request_id)
        if future is not None and not future.done():
            future.cancel()
        return True

    # ------------------------------------------------------------------ Utilities

    def _require_running_episode(self, slot: SessionSlot) -> None:
        """Require the session to have an unfinished episode.

        Args:
            slot: The session slot.

        Raises:
            RuntimeApiError: Not yet reset (``SESSION_NOT_READY``) or the
                episode has already ended (``EPISODE_TERMINATED``).
        """
        if slot.episode_id is None:
            raise RuntimeApiError(
                make_error(
                    ErrorCode.SESSION_NOT_READY,
                    f"session {slot.session_id} has no episode; call reset first",
                    session_id=slot.session_id,
                )
            )
        if slot.terminated or slot.truncated:
            raise RuntimeApiError(
                make_error(
                    ErrorCode.EPISODE_TERMINATED,
                    f"episode {slot.episode_id} of session {slot.session_id} "
                    f"already finished (terminated={slot.terminated}, "
                    f"truncated={slot.truncated}); call reset to start a new one",
                    session_id=slot.session_id,
                    episode_id=slot.episode_id,
                    terminated=slot.terminated,
                    truncated=slot.truncated,
                )
            )

    @staticmethod
    def _decode_actions(actions: Any) -> np.ndarray:
        """Decode an action payload into ``[chunk, action_dim] float32``.

        Args:
            actions: A ``PayloadRef``, or already an array.

        Returns:
            A contiguous 2D float32 array.

        Raises:
            RuntimeApiError: The payload is missing, has the wrong number of
                dimensions, or contains non-finite values
                (``INVALID_ARGUMENT``).
        """
        if actions is None:
            raise RuntimeApiError(
                make_error(ErrorCode.INVALID_ARGUMENT, "actions payload is missing")
            )
        block = (
            actions
            if isinstance(actions, np.ndarray)
            else payload_module.decode_payload(actions)
        )
        block = np.asarray(block, dtype=np.float32)
        if block.ndim == 1:
            block = block[None, :]
        if block.ndim != 2:
            raise RuntimeApiError(
                make_error(
                    ErrorCode.INVALID_ARGUMENT,
                    "actions must be [chunk, action_dim], got shape "
                    f"{tuple(int(dim) for dim in block.shape)}",
                )
            )
        if not bool(np.all(np.isfinite(block))):
            raise RuntimeApiError(
                make_error(
                    ErrorCode.INVALID_ARGUMENT, "actions contain non-finite values"
                )
            )
        return block

    # ---------------------------------------------------------- Dispatch tables

    async def _cmd_reset(
        self,
        slot: SessionSlot,
        envelope: CommandEnvelope,
        active: ActiveRequest | None,
    ) -> StepResult:
        return await self.reset(
            slot.session_id,
            envelope.payload["reset_spec"],
            request_id=envelope.request_id,
            operation_seq=envelope.operation_seq,
            active=active,
        )

    async def _cmd_observe(
        self,
        slot: SessionSlot,
        envelope: CommandEnvelope,
        active: ActiveRequest | None,
    ) -> Observation:
        del envelope, active
        return await self.observe(slot.session_id)

    async def _cmd_action_step(
        self,
        slot: SessionSlot,
        envelope: CommandEnvelope,
        active: ActiveRequest | None,
    ) -> StepResult:
        return await self.action_step(
            slot.session_id,
            envelope.payload["actions"],
            request_id=envelope.request_id,
            operation_seq=envelope.operation_seq,
            active=active,
        )

    async def _cmd_policy_step(
        self,
        slot: SessionSlot,
        envelope: CommandEnvelope,
        active: ActiveRequest | None,
    ) -> StepResult:
        return await self.policy_step(
            slot.session_id,
            envelope.payload["policy_request"],
            request_id=envelope.request_id,
            operation_seq=envelope.operation_seq,
            active=active,
            deadline=envelope.deadline,
            priority=envelope.priority,
            application_id=str(envelope.trace_context.get("application_id", "")),
        )

    async def _cmd_policy_infer(
        self,
        slot: SessionSlot,
        envelope: CommandEnvelope,
        active: ActiveRequest | None,
    ) -> PolicyInferResult:
        return await self.policy_infer(
            slot.session_id,
            envelope.payload["policy_request"],
            request_id=envelope.request_id,
            active=active,
            deadline=envelope.deadline,
            priority=envelope.priority,
            application_id=str(envelope.trace_context.get("application_id", "")),
        )

    async def _cmd_run_episode(
        self,
        slot: SessionSlot,
        envelope: CommandEnvelope,
        active: ActiveRequest | None,
    ) -> EpisodeResult:
        return await self.run_episode(
            slot.session_id,
            envelope.payload["episode_request"],
            request_id=envelope.request_id,
            operation_seq=envelope.operation_seq,
            active=active,
            application_id=str(envelope.trace_context.get("application_id", "")),
        )

    async def _cmd_extension_call(
        self,
        slot: SessionSlot,
        envelope: CommandEnvelope,
        active: ActiveRequest | None,
    ) -> dict[str, Any]:
        del active
        return await self.extension_call(
            slot.session_id,
            envelope.payload["namespace"],
            envelope.payload["method"],
            envelope.payload.get("args", {}),
        )

    async def _ctl_create_binding(self, envelope: ControlEnvelope) -> dict[str, Any]:
        token = await self.create_binding(
            envelope.session_id,  # type: ignore[arg-type]
            envelope.payload["env_spec"],
            envelope.payload.get("lease_expiration", 0.0),
        )
        return {"binding_token": str(token)}

    async def _ctl_release_binding(self, envelope: ControlEnvelope) -> None:
        await self.release_binding(envelope.session_id)  # type: ignore[arg-type]

    async def _ctl_renew_lease(self, envelope: ControlEnvelope) -> None:
        await self.renew_lease(
            envelope.session_id,  # type: ignore[arg-type]
            envelope.payload.get("lease_expiration", 0.0),
        )

    async def _ctl_cancel(self, envelope: ControlEnvelope) -> dict[str, Any]:
        target = envelope.payload["target_request_id"]
        active = self.active.get(target)
        stage = active.stage if active is not None else "unknown"
        cancelled = await self.cancel_request(target)
        return {
            "cancelled": cancelled,
            "side_effect_applied": (
                active.side_effect_applied if active is not None else False
            ),
            "stage": stage,
        }

    async def _ctl_heartbeat(self, envelope: ControlEnvelope) -> EnvWorkerInfo:
        """Respond with an up-to-date ``EnvWorkerInfo`` (the real heartbeat).

        Goes through the control channel and **not** a ``WorkerGroup`` method:
        ``WorkerGroupFuncResult`` calls ``os.kill(pid, SIGUSR1)`` to kill the
        whole job on a remote exception, so a liveness check must never use it
        — when an actor dies, what we want is "this one path timed out", not
        "the driver dies along with it".

        Args:
            envelope: The control message (for heartbeat-type messages,
                ``session_id`` is ``None``).

        Returns:
            A snapshot of current capabilities and capacity;
            ``heartbeat_at`` is the worker-side time.
        """
        del envelope
        return self.worker_info()

    _COMMAND_HANDLERS = {
        EnvOperation.RESET: _cmd_reset,
        EnvOperation.OBSERVE: _cmd_observe,
        EnvOperation.ACTION_STEP: _cmd_action_step,
        EnvOperation.POLICY_STEP: _cmd_policy_step,
        EnvOperation.POLICY_INFER: _cmd_policy_infer,
        EnvOperation.RUN_EPISODE: _cmd_run_episode,
        EnvOperation.EXTENSION_CALL: _cmd_extension_call,
    }
    """Command dispatch table; keys cover the first seven ``EnvOperation``
    entries."""

    _CONTROL_HANDLERS = {
        EnvOperation.CREATE_BINDING: _ctl_create_binding,
        EnvOperation.RELEASE_BINDING: _ctl_release_binding,
        EnvOperation.RENEW_LEASE: _ctl_renew_lease,
        EnvOperation.CANCEL: _ctl_cancel,
        EnvOperation.HEARTBEAT: _ctl_heartbeat,
    }
    """Control dispatch table; keys cover the last five ``EnvOperation``
    entries."""
