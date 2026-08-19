"""The transport abstraction.

The Gateway depends only on the two Protocols here and **knows nothing
about Ray**. The division of labor across the three implementations:

| Implementation | Stage | Description |
|---|---|---|
| ``InProcTransport`` | early | Same-process asyncio, directly awaits the worker's handler |
| ``RayChannelTransport`` | default | A bounded ``Channel`` command queue + an independent high-priority control channel + result flow-back |
| ``RayActorTransport`` | not implemented | Kept only as a spike comparison |

It has been settled that "Channel handles the data plane, WorkerGroup only
handles the control plane": all 12 rlinf runners follow this without
exception, and ``WorkerGroupFuncResult.async_wait()`` polls every 100ms and
spins up a thread per call.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from typing import Protocol, runtime_checkable

from rollout_runtime.api.internal import (
    ActionResponse,
    CommandEnvelope,
    ControlEnvelope,
    InferenceRequest,
    ResultEnvelope,
)

__all__ = [
    "CommandHandler",
    "CommandTransport",
    "InferenceChannel",
    "InferenceChannelClosed",
    "LateResultSink",
    "TransportClosed",
    "WorkerCommandEndpoint",
]


@runtime_checkable
class CommandTransport(Protocol):
    """The Gateway-side sending end."""

    async def send_command(
        self, worker_rank: int, envelope: CommandEnvelope
    ) -> ResultEnvelope:
        """Send an environment command and wait for the response.

        The implementation must be a total function: transport-layer
        failures must also be normalized into a ``ResultEnvelope`` with an
        ``error``, never raised.

        Args:
            worker_rank: The target EnvWorker rank.
            envelope: The command.

        Returns:
            The response envelope.
        """
        ...

    async def send_control(
        self, worker_rank: int, envelope: ControlEnvelope
    ) -> ResultEnvelope:
        """Send a control message over the high-priority control channel.

        Args:
            worker_rank: The target EnvWorker rank.
            envelope: The control message.

        Returns:
            The response envelope.
        """
        ...

    def worker_ranks(self) -> Sequence[int]:
        """List the reachable EnvWorker ranks.

        Returns:
            A sequence of ranks.
        """
        ...


@runtime_checkable
class CommandHandler(Protocol):
    """The worker-side receiving end (implemented by ``RuntimeEnvWorker``)."""

    async def handle_command(self, envelope: CommandEnvelope) -> ResultEnvelope:
        """Handle an environment command.

        Args:
            envelope: The command.

        Returns:
            The response envelope; never raises.
        """
        ...

    async def handle_control(self, envelope: ControlEnvelope) -> ResultEnvelope:
        """Handle a control message.

        Args:
            envelope: The control message.

        Returns:
            The response envelope; never raises.
        """
        ...


class TransportClosed(RuntimeError):
    """The signal received by a blocking reader when the channel has been
    closed.

    Not a protocol error, so it is not part of ``ErrorCode``: a worker's or
    Gateway's long-running loop treats it as a normal shutdown signal.
    """


class InferenceChannelClosed(TransportClosed):
    """The request-plane channel is closed (raised by ``get_request`` /
    ``get_response``)."""


@runtime_checkable
class WorkerCommandEndpoint(Protocol):
    """The worker-side command / control receiving end and result flow-back
    end.

    ``InProcTransport`` awaits the worker's handler directly and doesn't
    need this; the Channel-based implementation has
    ``RuntimeEnvWorker.run()`` receive commands and send back results
    through this Protocol, so ``env_worker.py`` doesn't need to know about
    rlinf.

    Commands and controls travel over **two separate channels**: cancel /
    heartbeat / shutdown must not be queued behind ordinary commands.
    """

    async def get_command(self) -> tuple[CommandEnvelope, str]:
        """Get one environment command.

        Returns:
            A ``(command, reply key)`` tuple.

        Raises:
            TransportClosed: The channel is closed.
        """
        ...

    async def get_control(self) -> tuple[ControlEnvelope, str]:
        """Get one control message.

        Returns:
            A ``(control message, reply key)`` tuple.

        Raises:
            TransportClosed: The channel is closed.
        """
        ...

    async def put_result(self, reply_key: str, envelope: ResultEnvelope) -> None:
        """Write a response back to the result channel.

        Args:
            reply_key: The reply key given by ``get_command`` /
                ``get_control``.
            envelope: The response.
        """
        ...

    def close(self) -> None:
        """Close the endpoint and wake up waiters."""
        ...


@runtime_checkable
class LateResultSink(Protocol):
    """An optional capability only present on transports that have a
    result flow-back channel.

    Once ``transport.command_timeout_seconds`` elapses, the caller only
    gets a ``DEADLINE_EXCEEDED`` and **the operation does not reach a
    terminal state**; the real result arrives later from the result
    channel, at which point the callback registered here finalizes the
    operation (RPC timeout is not the same as cancellation).

    The Gateway probes for this method with ``getattr``, so a transport
    that doesn't implement it (``InProcTransport``) is unaffected.
    """

    def set_late_result_handler(
        self, handler: Callable[[ResultEnvelope], Awaitable[None]]
    ) -> None:
        """Register the finalization callback for late results.

        Args:
            handler: An async callback receiving the late
                ``ResultEnvelope``.
        """
        ...


@runtime_checkable
class InferenceChannel(Protocol):
    """The EnvWorker <-> RolloutWorker request plane.

    Corresponds to two bounded channels:

    ```text
    Channel("rr_infer_req",  maxsize=N)  key="pending"        # multiple rollout ranks compete on get
    Channel("rr_infer_resp", maxsize=M)  key=routing_token    # routed home by key
    ```

    Two hard semantics that both the inproc implementation and the Ray
    Channel implementation must preserve:

    - **Bounded**: when full, ``put_request_nowait`` raises
      ``asyncio.QueueFull``, which the EnvWorker's ``InferenceClient``
      turns into a ``QUEUE_FULL`` rejection upstream, instead of piling up
      unboundedly in memory.
    - **No shared key**: requests share a single ``"pending"`` key
      (work-stealing); responses are keyed by ``routing_token``, and each
      EnvWorker only drains its own.

    The two ``*_nowait`` methods are **async**: rlinf's
    ``Channel.put_nowait`` is ``ray.get`` on the driver path and a
    concurrent ``Future.wait()`` on the worker path, both of which would
    block the caller's event loop; to be non-blocking within the event
    loop while still being able to observe ``QueueFull``, one ``await`` of
    an actor call is required. In the inproc implementation, these two
    coroutines have no ``await`` point at all, so the semantics exactly
    match the earlier synchronous version (the coroutine body runs to
    completion synchronously once awaited).
    """

    async def put_request_nowait(self, request: InferenceRequest) -> None:
        """Non-blockingly submit an inference request.

        Args:
            request: The inference request.
        """
        ...

    async def get_request(self) -> InferenceRequest:
        """Get one pending request (multiple ranks compete for the same
        key).

        Returns:
            The inference request.
        """
        ...

    async def put_response_nowait(
        self, routing_token: str, response: ActionResponse
    ) -> None:
        """Non-blockingly submit a response back to the specified
        EnvWorker.

        Args:
            routing_token: The output of
                ``api.internal.make_routing_token``.
            response: The action-chunk response.
        """
        ...

    async def get_response(self, routing_token: str) -> ActionResponse:
        """Get a response belonging to this endpoint.

        Args:
            routing_token: This endpoint's own routing identifier.

        Returns:
            The action-chunk response.
        """
        ...

    def close(self) -> None:
        """Close the channel and wake up all waiters."""
        ...
