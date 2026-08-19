"""``RuntimeRolloutWorker``.

Runs as a serving system: keeps the policy loaded persistently, consumes
``InferenceRequest`` from multiple EnvWorkers, and routes results back to the
originating EnvWorker by ``routing_token``. **A complete episode never enters the
RolloutWorker** — it only ever sees a continuous stream of ordinary inference
requests.

The M3 shape (M2 only wired things together, without batching):

| Method | Shape |
|---|---|
| ``init_worker`` / ``infer_batch`` / ``send_responses`` | Same as M2 |
| ``serve`` | Two persistent loops: ``receive_requests`` ‖ ``execute_batches`` |
| ``receive_requests`` | Competitively get from ``rr_infer_req["pending"]`` → record ``routing_token`` → **enqueue into the scheduler** |
| ``execute_batches`` | Pull a batch bucketed by ``compat_key`` (``max_batch_size`` / ``max_wait_ms`` / priority / tenant fairness), one task per batch |
| ``update_weights`` | The scheduler's version fence: raise fence → wait for in-flight to drain → swap weights → lower fence |

Fetching is constrained by **two gates**; once either is saturated, no more requests
are pulled from the channel, letting the bounded ``rr_infer_req`` back up to maxsize
and propagating backpressure to the EnvWorker: ``max_concurrent_inferences`` (total
accepted so far, present since M2) and ``SchedulerConfig.max_queue_depth`` (the
scheduler's watermark).

All public methods must be total functions.
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import inspect
from typing import Any

from rollout_runtime.api.errors import normalize_exception
from rollout_runtime.api.ids import RequestId
from rollout_runtime.api.internal import ActionResponse, InferenceRequest
from rollout_runtime.transport.base import InferenceChannelClosed
from rollout_runtime.workers.batch_scheduler import (
    InferenceBatchScheduler,
    PendingRequest,
    SchedulerConfig,
)

__all__ = ["RuntimeRolloutWorker"]

DEFAULT_MAX_CONCURRENT_INFERENCES = 8
"""Cap on the number of inference requests concurrently in flight per rank (the M2
in-flight gate)."""


class RuntimeRolloutWorker:
    """Inference-serving worker for a single GPU rank."""

    def __init__(
        self,
        *,
        worker_rank: int = 0,
        group_name: str = "rollout",
        scheduler_config: SchedulerConfig | None = None,
        policy: Any = None,
        max_concurrent_inferences: int = DEFAULT_MAX_CONCURRENT_INFERENCES,
        weight_update_timeout_seconds: float = 30.0,
    ) -> None:
        """Initialize.

        Args:
            worker_rank: Rank of this worker (corresponds to one GPU,
                ``tensor_parallel_size=1``).
            group_name: Worker group name.
            scheduler_config: Batch scheduler configuration.
            policy: A ``core.policy_inference.PolicyInferenceCore`` implementation.
            max_concurrent_inferences: Cap on requests accepted by this rank (queued +
                in flight), i.e. the admission window for pulling from
                ``rr_infer_req``.
            weight_update_timeout_seconds: Timeout for ``update_weights`` while
                waiting for in-flight requests to drain.
        """
        self.worker_rank = worker_rank
        self.group_name = group_name
        self.scheduler = InferenceBatchScheduler(scheduler_config)
        self.policy: Any = policy
        self.model_version = ""
        self.received_count = 0
        self.responded_count = 0
        self.dropped_route_count = 0
        self.weight_update_count = 0
        self.weight_update_error_count = 0
        self.weight_update_timeout_seconds = weight_update_timeout_seconds
        self._max_concurrent = max(1, max_concurrent_inferences)
        self._slots = asyncio.Semaphore(self._max_concurrent)
        self._wake = asyncio.Event()
        self._idle_sleep = 0.001
        self._routes: dict[RequestId, str] = {}
        self._tasks: set[asyncio.Task[None]] = set()
        self._response_channel: Any = None
        self._serving = False
        self.last_serve_error: Any = None

    @property
    def serving(self) -> bool:
        """Whether the persistent loop is running.

        Returns:
            True if ``serve`` has been called and ``stop`` has not.
        """
        return self._serving

    @property
    def inflight(self) -> int:
        """Number of inference requests currently in flight.

        Returns:
            The in-flight count.
        """
        return len(self._tasks)

    def init_worker(self, channels: dict[str, Any] | None = None) -> None:
        """Load the policy for the current rank and record the response channel.

        Args:
            channels: Channel table; the ``"inference"`` key is a
                ``transport.base.InferenceChannel``.
        """
        channel = (channels or {}).get("inference")
        if channel is not None:
            self._response_channel = channel
        if self.policy is not None:
            self.policy.load()
            self.model_version = self.policy.model_version

    async def serve(self, request_channel: Any, response_channel: Any) -> None:
        """Persistently run the "receive" and "execute batch" loops.

        If either sub-loop exits with an exception, ``serve()`` stops as a whole
        (``asyncio.gather`` semantics); the exception itself is recorded in
        ``last_serve_error`` (a persistent loop must not leak exceptions, but silently
        swallowing diagnostic information is equally harmful — under the
        ``ray_channel`` transport this is the only trace of failure available; the
        caller sees no Python traceback, only in-flight inference requests hanging
        forever).

        Args:
            request_channel: The ``rr_infer_req`` channel.
            response_channel: The ``rr_infer_resp`` channel.
        """
        self._response_channel = response_channel
        self._serving = True
        self.last_serve_error = None
        try:
            await asyncio.gather(
                self.receive_requests(request_channel),
                self.execute_batches(response_channel),
            )
        except asyncio.CancelledError:
            raise
        except BaseException as exc:  # noqa: BLE001 - persistent loop must not leak exceptions
            self.last_serve_error = normalize_exception(exc)
            return
        finally:
            self._serving = False
            self._wake.set()

    async def receive_requests(self, request_channel: Any) -> None:
        """Competitively get from ``rr_infer_req["pending"]``, record the
        ``routing_token``, then enqueue.

        Multiple ranks competing on the same key is natural work-stealing. Fetching is
        constrained by two gates; once either is saturated, stop getting, letting the
        bounded request queue naturally back up to maxsize, which in turn propagates
        backpressure to the EnvWorker:

        - ``max_concurrent_inferences``: total requests accepted by this rank
          (queued + in flight);
        - ``SchedulerConfig.max_queue_depth``: the scheduler's watermark.

        Args:
            request_channel: The ``rr_infer_req`` channel.
        """
        loop = asyncio.get_running_loop()
        while self._serving:
            await self._slots.acquire()
            if not self.scheduler.accepts_more():
                self._slots.release()
                await asyncio.sleep(self._idle_sleep)
                continue
            try:
                request = await request_channel.get_request()
            except (asyncio.CancelledError, GeneratorExit):
                self._slots.release()
                raise
            except InferenceChannelClosed:
                self._slots.release()
                return
            except BaseException:  # noqa: BLE001 - the receive loop must never die
                self._slots.release()
                await asyncio.sleep(0)
                continue
            self.received_count += 1
            self._routes[request.request_id] = request.routing_token
            self.scheduler.enqueue(
                PendingRequest(
                    request=request,
                    enqueued_at=loop.time(),
                    routing_token=request.routing_token,
                )
            )
            self._wake.set()

    def _finish_task(self, task: asyncio.Task[None]) -> None:
        self._tasks.discard(task)

    async def execute_batches(self, response_channel: Any) -> None:
        """Continuously pull and execute batches according to ``max_batch_size`` /
        ``max_wait_ms`` / compatibility / priority.

        No polling: ``scheduler.time_until_ready`` gives the due time for the next
        batch, and the loop waits exactly that long (or until woken by a new
        request), so ``max_wait_ms`` is a genuine batch-accumulation window rather
        than a sampling period.

        Args:
            response_channel: The ``rr_infer_resp`` channel.
        """
        loop = asyncio.get_running_loop()
        while self._serving:
            batch = self.scheduler.next_batch(loop.time())
            if batch:
                self._spawn(self._run_batch(batch, response_channel))
                continue
            self._wake.clear()
            delay = self.scheduler.time_until_ready(loop.time())
            if delay is None:
                await self._wake.wait()
                continue
            if delay <= 0:
                await asyncio.sleep(0)
                continue
            # ``asyncio.TimeoutError``, not the builtin ``TimeoutError``: on
            # Python 3.10 these are unrelated classes (only from 3.11 onward is
            # ``asyncio.TimeoutError`` an alias of the builtin ``TimeoutError``),
            # and ``asyncio.wait_for`` raises the former on timeout. Writing the
            # builtin ``TimeoutError`` here would not be caught by ``suppress`` at
            # all on 3.10; the exception would propagate through
            # ``execute_batches`` → ``asyncio.gather`` → ``serve()``'s
            # ``except BaseException: return``, silently exiting the entire
            # persistent loop, after which all in-flight inference requests hang
            # forever (no timeout protection; the caller sees a deadlock rather
            # than an error).
            with contextlib.suppress(asyncio.TimeoutError, asyncio.CancelledError):
                await asyncio.wait_for(self._wake.wait(), delay)

    def _spawn(self, coroutine: Any) -> None:
        task = asyncio.get_running_loop().create_task(coroutine)
        self._tasks.add(task)
        task.add_done_callback(self._finish_task)

    async def _run_batch(
        self, batch: list[PendingRequest], response_channel: Any
    ) -> None:
        """Execute one batch and send responses (total function).

        Args:
            batch: Queued requests sharing the same ``compat_key``.
            response_channel: The ``rr_infer_resp`` channel.
        """
        requests = [pending.request for pending in batch]
        try:
            responses = await self.infer_batch(requests)
        except asyncio.CancelledError:
            raise
        except BaseException as exc:  # noqa: BLE001 - must never leak
            responses = [self._failed_response(request, exc) for request in requests]
        try:
            await self.send_responses(responses, response_channel)
        except asyncio.CancelledError:
            raise
        except BaseException:  # noqa: BLE001 - response failures must not leak either
            pass
        finally:
            for pending in batch:
                self.scheduler.complete(pending)
                self._slots.release()
            self._wake.set()

    async def infer_batch(
        self, requests: list[InferenceRequest]
    ) -> list[ActionResponse]:
        """Preprocess a batch, run ``predict_action_batch``, and unpack responses per
        request.

        Prefers the execution core's ``ainfer_batch`` (inference waits must be
        cancellable); falls back to ``asyncio.to_thread`` only for synchronous cores,
        so a future HF backend can plug in without changing this shape.

        Args:
            requests: Requests sharing the same ``compat_key``.

        Returns:
            Responses in the same order as the input; a single request's failure only
            affects that request.
        """
        if not requests:
            return []
        core = self.policy
        if core is None:
            error = RuntimeError("rollout worker has no policy loaded")
            return [self._failed_response(request, error) for request in requests]
        try:
            ainfer = getattr(core, "ainfer_batch", None)
            if ainfer is not None:
                responses = await ainfer(requests)
            else:
                responses = await asyncio.to_thread(core.infer_batch, requests)
        except asyncio.CancelledError:
            raise
        except BaseException as exc:  # noqa: BLE001 - normalize per-request even on whole-batch failure
            return [self._failed_response(request, exc) for request in requests]
        version = getattr(core, "model_version", self.model_version)
        self.model_version = version
        return [
            response
            if response.model_version
            else dataclasses.replace(response, model_version=version)
            for response in responses
        ]

    async def send_responses(
        self, responses: list[ActionResponse], response_channel: Any
    ) -> None:
        """Route responses back to the originating EnvWorker's response key by
        ``routing_token``.

        Args:
            responses: Responses to be sent back.
            response_channel: The ``rr_infer_resp`` channel.
        """
        channel = response_channel or self._response_channel
        if channel is None:
            self.dropped_route_count += len(responses)
            return
        for response in responses:
            routing_token = self._routes.pop(response.request_id, None)
            if routing_token is None:
                self.dropped_route_count += 1
                continue
            await channel.put_response_nowait(routing_token, response)
            self.responded_count += 1

    async def update_weights(self, model_version: str) -> bool:
        """Switch weights between two batches; never mix versions within a batch.

        The flow is exactly the scheduler's version fence: raise fence → wait for all
        in-flight batches to respond → swap weights → lower fence. While the fence is
        raised, ``next_batch`` returns empty, so any new batch is guaranteed to use
        the new version, while every response in the old batch still carries the old
        version (``ActionResponse.model_version`` is tagged per request).

        A total function: failures are only counted and return ``False``, never
        raised.

        Args:
            model_version: Target version.

        Returns:
            Whether the switch succeeded.
        """
        loop = asyncio.get_running_loop()
        self.scheduler.begin_version_fence(model_version)
        self._wake.set()
        try:
            deadline = loop.time() + self.weight_update_timeout_seconds
            while self.scheduler.inflight_total > 0 and loop.time() < deadline:
                await asyncio.sleep(0.001)
            if self.scheduler.inflight_total > 0:
                self.weight_update_error_count += 1
                return False
            core = self.policy
            if core is not None:
                update = getattr(core, "update_weights", None)
                if update is not None:
                    outcome = update(model_version)
                    if inspect.isawaitable(outcome):
                        await outcome
                self.model_version = getattr(core, "model_version", model_version)
            else:
                self.model_version = model_version
            self.weight_update_count += 1
            return True
        except asyncio.CancelledError:
            raise
        except BaseException:  # noqa: BLE001 - must never raise across the worker boundary
            self.weight_update_error_count += 1
            return False
        finally:
            self.scheduler.end_version_fence()
            self._wake.set()

    async def stop(self) -> None:
        """Stop accepting new requests, drain in-flight inference, and release model
        resources."""
        self._serving = False
        self._wake.set()
        tasks = list(self._tasks)
        if tasks:
            with contextlib.suppress(BaseException):
                await asyncio.wait(tasks, timeout=5.0)
        for task in list(self._tasks):
            task.cancel()
        for task in list(self._tasks):
            with contextlib.suppress(BaseException):
                await task
        self._tasks.clear()
        self._routes.clear()
        if self.policy is not None:
            with contextlib.suppress(BaseException):
                self.policy.close()

    def _failed_response(
        self, request: InferenceRequest, exc: BaseException
    ) -> ActionResponse:
        """Normalize an exception into a single-request failure response.

        Args:
            request: The request that triggered the failure.
            exc: The caught exception.

        Returns:
            A response carrying ``error``.
        """
        return ActionResponse(
            request_id=request.request_id,
            session_id=request.session_id,
            binding_token=request.binding_token,
            episode_id=request.episode_id,
            operation_seq=request.operation_seq,
            model_version=self.model_version,
            error=normalize_exception(exc),
        )
