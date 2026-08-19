"""Same-process transport and request plane (the e2e foundation).

Two pieces of content:

- ``InProcTransport``: the Gateway <-> EnvWorker command / control path.
  All three parties live in one process and one event loop, so commands
  directly ``await`` the worker's handler.
- ``InProcInferenceChannel``: the EnvWorker <-> RolloutWorker request
  plane, i.e. the asyncio version of ``rr_infer_req`` / ``rr_infer_resp``.
  **Bounded**; when full it raises ``asyncio.QueueFull`` (a backpressure
  signal), and the Ray Channel implementation must preserve the same
  semantics.

Pure stdlib; a full e2e run can complete on a local CPU without ray.

Even though the worker is already a total function, this module adds
another layer of exception normalization: failures within the transport
itself (a missing handler, an event-loop cancellation) must also present
as a ``ResultEnvelope`` carrying an ``error``.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence

from rollout_runtime.api.enums import ErrorCode, OperationState
from rollout_runtime.api.errors import make_error, normalize_exception
from rollout_runtime.api.internal import (
    ActionResponse,
    CommandEnvelope,
    ControlEnvelope,
    InferenceRequest,
    ResultEnvelope,
)
from rollout_runtime.transport.base import CommandHandler, InferenceChannelClosed

__all__ = ["InProcInferenceChannel", "InProcTransport"]


class InProcTransport:
    """A transport that dispatches commands directly to same-process worker
    objects."""

    def __init__(self, handlers: Mapping[int, CommandHandler]) -> None:
        """Initialize.

        Args:
            handlers: A mapping from rank to worker handler.
        """
        self._handlers = dict(handlers)

    def worker_ranks(self) -> Sequence[int]:
        """List the reachable ranks.

        Returns:
            A sorted list of ranks.
        """
        return sorted(self._handlers)

    def register(self, worker_rank: int, handler: CommandHandler) -> None:
        """Register a worker handler.

        Args:
            worker_rank: The rank.
            handler: The worker object.
        """
        self._handlers[worker_rank] = handler

    def _missing(
        self, worker_rank: int, request_id: str, operation: object
    ) -> ResultEnvelope:
        return ResultEnvelope(
            request_id=request_id,  # type: ignore[arg-type]
            operation=operation,  # type: ignore[arg-type]
            state=OperationState.FAILED,
            error=make_error(
                ErrorCode.WORKER_LOST,
                f"no in-process worker registered at rank {worker_rank}",
                worker_rank=worker_rank,
            ),
        )

    async def send_command(
        self, worker_rank: int, envelope: CommandEnvelope
    ) -> ResultEnvelope:
        """Dispatch an environment command.

        Args:
            worker_rank: The target rank.
            envelope: The command.

        Returns:
            The response envelope.
        """
        handler = self._handlers.get(worker_rank)
        if handler is None:
            return self._missing(worker_rank, envelope.request_id, envelope.operation)
        try:
            return await handler.handle_command(envelope)
        except BaseException as exc:  # noqa: BLE001 - must never leak
            return ResultEnvelope(
                request_id=envelope.request_id,
                session_id=envelope.session_id,
                operation=envelope.operation,
                state=OperationState.FAILED,
                error=normalize_exception(exc),
            )

    async def send_control(
        self, worker_rank: int, envelope: ControlEnvelope
    ) -> ResultEnvelope:
        """Dispatch a control message.

        Args:
            worker_rank: The target rank.
            envelope: The control message.

        Returns:
            The response envelope.
        """
        handler = self._handlers.get(worker_rank)
        if handler is None:
            return self._missing(worker_rank, envelope.request_id, envelope.operation)
        try:
            return await handler.handle_control(envelope)
        except BaseException as exc:  # noqa: BLE001 - must never leak
            return ResultEnvelope(
                request_id=envelope.request_id,
                session_id=envelope.session_id,
                operation=envelope.operation,
                state=OperationState.FAILED,
                error=normalize_exception(exc),
            )


class InProcInferenceChannel:
    """The same-process implementation of ``rr_infer_req`` /
    ``rr_infer_resp``.

    The request side has a single logical key (``"pending"``); multiple
    RolloutWorker ranks competing on ``get`` is naturally work-stealing.
    The response side is keyed by ``routing_token``, with each EnvWorker
    only draining its own.

    Both channels are **bounded**: when the request queue is full,
    ``put_request_nowait`` raises ``asyncio.QueueFull``, which the
    EnvWorker turns into a ``QUEUE_FULL`` rejection rather than piling up
    unboundedly.
    """

    def __init__(
        self,
        *,
        request_queue_size: int = 64,
        response_queue_size: int = 64,
    ) -> None:
        """Initialize.

        Args:
            request_queue_size: The maxsize of ``rr_infer_req`` (the
                backpressure watermark).
            response_queue_size: The maxsize of each ``rr_infer_resp`` key.
        """
        self._request_queue_size = max(1, request_queue_size)
        self._response_queue_size = max(1, response_queue_size)
        self._requests: asyncio.Queue[InferenceRequest | None] = asyncio.Queue(
            maxsize=self._request_queue_size
        )
        self._responses: dict[str, asyncio.Queue[ActionResponse | None]] = {}
        self._closed = False
        self.requests_put = 0
        self.requests_rejected = 0
        self.responses_put = 0
        self.responses_dropped = 0

    @property
    def closed(self) -> bool:
        """Whether the channel is closed.

        Returns:
            ``True`` when closed.
        """
        return self._closed

    @property
    def request_depth(self) -> int:
        """The number of currently queued requests.

        Returns:
            The queue length.
        """
        return self._requests.qsize()

    @property
    def request_capacity(self) -> int:
        """The request queue capacity.

        Returns:
            The maxsize.
        """
        return self._request_queue_size

    def register_route(self, routing_token: str) -> None:
        """Pre-create the response key for an EnvWorker.

        Args:
            routing_token: The output of ``make_routing_token``.
        """
        self._response_queue(routing_token)

    def _response_queue(
        self, routing_token: str
    ) -> asyncio.Queue[ActionResponse | None]:
        queue = self._responses.get(routing_token)
        if queue is None:
            queue = asyncio.Queue(maxsize=self._response_queue_size)
            self._responses[routing_token] = queue
        return queue

    async def put_request_nowait(self, request: InferenceRequest) -> None:
        """Non-blockingly submit an inference request.

        This implementation has no ``await`` point at all: the coroutine
        body runs to completion synchronously once awaited, so the
        semantics exactly match the earlier synchronous version
        (``QueueFull`` is raised within the same event-loop tick). See
        ``base.InferenceChannel`` for why the signature is async.

        Args:
            request: The inference request.

        Raises:
            InferenceChannelClosed: The channel is closed.
            asyncio.QueueFull: The request queue is full (a backpressure
                signal).
        """
        if self._closed:
            raise InferenceChannelClosed("inference channel is closed")
        try:
            self._requests.put_nowait(request)
        except asyncio.QueueFull:
            self.requests_rejected += 1
            raise
        self.requests_put += 1

    async def get_request(self) -> InferenceRequest:
        """Get one pending request.

        Returns:
            The inference request.

        Raises:
            InferenceChannelClosed: The channel is closed.
        """
        if self._closed:
            raise InferenceChannelClosed("inference channel is closed")
        item = await self._requests.get()
        if item is None:
            raise InferenceChannelClosed("inference channel is closed")
        return item

    async def put_response_nowait(
        self, routing_token: str, response: ActionResponse
    ) -> None:
        """Non-blockingly submit a response back to the specified
        EnvWorker.

        If the response side is full or the EnvWorker has already
        disappeared, the response is **dropped and counted** rather than
        raised to the RolloutWorker: one disconnected EnvWorker must not
        take down the inference service.

        Args:
            routing_token: The target EnvWorker's routing identifier.
            response: The action-chunk response.
        """
        if self._closed:
            self.responses_dropped += 1
            return
        try:
            self._response_queue(routing_token).put_nowait(response)
        except asyncio.QueueFull:
            self.responses_dropped += 1
            return
        self.responses_put += 1

    async def get_response(self, routing_token: str) -> ActionResponse:
        """Get a response belonging to this endpoint.

        Args:
            routing_token: This endpoint's own routing identifier.

        Returns:
            The action-chunk response.

        Raises:
            InferenceChannelClosed: The channel is closed.
        """
        if self._closed:
            raise InferenceChannelClosed("inference channel is closed")
        item = await self._response_queue(routing_token).get()
        if item is None:
            raise InferenceChannelClosed("inference channel is closed")
        return item

    def close(self) -> None:
        """Close the channel and wake up all waiters with sentinels."""
        if self._closed:
            return
        self._closed = True
        for _ in range(self._request_queue_size):
            try:
                self._requests.put_nowait(None)
            except asyncio.QueueFull:
                break
        for queue in self._responses.values():
            for _ in range(self._response_queue_size):
                try:
                    queue.put_nowait(None)
                except asyncio.QueueFull:
                    break
