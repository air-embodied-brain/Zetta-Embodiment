"""Ray Channel transport and the request plane.

Four things:

| Class | Role |
|---|---|
| ``RayQueue`` | An asyncio facade over rlinf's ``Channel`` (per-key queues, bounded, ``QueueFull``) |
| ``RayChannelTransport`` | Gateway side: bounded command queue + independent control channel + result flow-back |
| ``RayChannelWorkerEndpoint`` | EnvWorker side: receives commands / receives control / sends back results (used by ``run()``) |
| ``RayInferenceChannel`` | ``rr_infer_req`` / ``rr_infer_resp`` (the request plane) |

Channel naming follows rlinf's convention (``Channel.create(f"Tool-{group}")``
in ``agent_runner.py:104``): ``rr-cmd-{env_group_name}`` /
``rr-ctl-{env_group_name}`` / ``rr-res-{gateway_epoch}``, plus
``rr-infer-req-{rollout_group_name}`` / ``rr-infer-resp-{rollout_group_name}``.
``suffix`` is a per-launch unique run suffix: ``Channel.create`` **connects
to the old channel** when it encounters the same name again (the
``except ValueError`` branch in ``channel.py``), so running multiple rounds
within one process would otherwise reuse a stale queue.

Three rlinf facts drive the code here (all in
``third_party/rlinf/rlinf/scheduler/channel/``):

1. ``Channel.create(maxsize=0)`` is **unbounded** by default. Backpressure
   requires explicitly passing ``maxsize``, otherwise the assertion
   "``QUEUE_FULL`` instead of unbounded accumulation" would silently fail.
   ``ChannelWorker.create_queue`` creates an independent queue per key with
   the same ``maxsize``, which is exactly the "per-key queues, each
   bounded" behavior needed here.
2. In the ``LocalChannel`` branch, ``assert async_op is False``, and
   ``put``/``get`` **busy-wait** (``while ...: continue``) when the queue
   is full/empty. So ``local=True`` is never used here.
3. In the worker context, ``Channel`` goes through ``send/recv`` +
   ``AsyncChannelWork``, whose ``async_wait()`` is
   ``loop.run_in_executor(None, future.wait)`` -- each waiter occupies a
   thread-pool worker, and a few dozen concurrent gets would exhaust the
   default thread pool. When ``_current_worker is None``, it instead goes
   through ``put_via_ray`` / ``get_via_ray``, where
   ``AsyncRayWork.async_wait()`` is just ``await ObjectRef``, i.e. genuine
   native asyncio waiting. **This is why this module always uses
   ``Channel.connect(name, current_worker=None)``**, even when the caller
   itself is an rlinf Worker.
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import uuid
from collections.abc import Awaitable, Callable, Sequence
from typing import Any

from rollout_runtime.api.enums import ErrorCode, OperationState
from rollout_runtime.api.errors import make_error, normalize_exception
from rollout_runtime.api.ids import RequestId
from rollout_runtime.api.internal import (
    ActionResponse,
    CommandEnvelope,
    ControlEnvelope,
    InferenceRequest,
    ResultEnvelope,
)
from rollout_runtime.api.wire import decode_bytes, encode_bytes
from rollout_runtime.transport.base import InferenceChannelClosed, TransportClosed

__all__ = [
    "CMD_CHANNEL_TEMPLATE",
    "CTL_CHANNEL_TEMPLATE",
    "PENDING_KEY",
    "RESP_CHANNEL_TEMPLATE",
    "REQ_CHANNEL_TEMPLATE",
    "RESULT_KEY",
    "RES_CHANNEL_TEMPLATE",
    "ChannelNames",
    "RayChannelTransport",
    "RayChannelWorkerEndpoint",
    "RayInferenceChannel",
    "RayQueue",
]

CMD_CHANNEL_TEMPLATE = "rr-cmd-{env_group_name}"
"""The command-channel naming template."""

CTL_CHANNEL_TEMPLATE = "rr-ctl-{env_group_name}"
"""The control-channel naming template."""

RES_CHANNEL_TEMPLATE = "rr-res-{gateway_epoch}"
"""The result-flow-back channel naming template."""

REQ_CHANNEL_TEMPLATE = "rr-infer-req-{rollout_group_name}"
"""The inference-request channel naming template (``rr_infer_req``)."""

RESP_CHANNEL_TEMPLATE = "rr-infer-resp-{rollout_group_name}"
"""The inference-response channel naming template (``rr_infer_resp``)."""

PENDING_KEY = "pending"
"""The single key of ``rr_infer_req``: multiple rollout ranks competing on
get implements work-stealing."""

RESULT_KEY = "results"
"""The key of the result-flow-back channel (one drain loop per Gateway
epoch)."""

CLOSE_SENTINEL = "__rr_channel_closed__"
"""The ``kind`` of the close sentinel, used to wake up waiters blocked on
``get``."""


def _new_suffix() -> str:
    """Generate a run suffix.

    Returns:
        An 8-character hex string.
    """
    return uuid.uuid4().hex[:8]


@dataclasses.dataclass(frozen=True, kw_only=True)
class ChannelNames:
    """The full set of channel names used by a runtime instance.

    Attributes:
        command: The command channel.
        control: The control channel.
        result: The result-flow-back channel.
        inference_request: The inference-request channel.
        inference_response: The inference-response channel.
    """

    command: str
    control: str
    result: str
    inference_request: str
    inference_response: str

    @classmethod
    def build(
        cls,
        *,
        env_group_name: str = "env",
        rollout_group_name: str = "rollout",
        gateway_epoch: int = 1,
        suffix: str | None = None,
    ) -> ChannelNames:
        """Assemble the five channel names following rlinf convention.

        Args:
            env_group_name: The EnvWorker group name.
            rollout_group_name: The RolloutWorker group name.
            gateway_epoch: The Gateway epoch.
            suffix: The run suffix; ``None`` generates one automatically,
                ``""`` means no suffix.

        Returns:
            The set of channel names.
        """
        tail = _new_suffix() if suffix is None else suffix
        end = f"-{tail}" if tail else ""
        return cls(
            command=CMD_CHANNEL_TEMPLATE.format(env_group_name=env_group_name) + end,
            control=CTL_CHANNEL_TEMPLATE.format(env_group_name=env_group_name) + end,
            result=RES_CHANNEL_TEMPLATE.format(gateway_epoch=gateway_epoch) + end,
            inference_request=REQ_CHANNEL_TEMPLATE.format(
                rollout_group_name=rollout_group_name
            )
            + end,
            inference_response=RESP_CHANNEL_TEMPLATE.format(
                rollout_group_name=rollout_group_name
            )
            + end,
        )

    def as_dict(self) -> dict[str, str]:
        """Export as a plain dict that can be passed through to a worker.

        Returns:
            A mapping from field name to channel name.
        """
        return dataclasses.asdict(self)


def _frame(kind: str, obj: Any, *, reply_key: str = "") -> dict[str, Any]:
    """Pack a protocol object into one channel frame.

    The frame body uses msgpack (``api.wire``) rather than ray's pickle:
    this keeps serialization to a well-defined first version, guarantees
    that only protocol objects ever cross processes, and lets the byte
    count be tallied.

    Args:
        kind: The frame type (``"command"`` / ``"control"`` / ``"result"``
            / ``"request"`` etc.).
        obj: The protocol object.
        reply_key: The result-flow-back key (only present for command /
            control frames).

    Returns:
        The channel frame.
    """
    return {"kind": kind, "reply_key": reply_key, "body": encode_bytes(obj)}


class RayQueue:
    """An asyncio facade over rlinf's ``Channel``: per-key queues, bounded,
    closeable.

    ``put_nowait`` is async: rlinf's method of the same name uses
    ``ray.get`` on the driver path (which blocks the event loop); here it
    is changed to ``await`` the ChannelWorker actor's
    ``put_via_ray(nowait=True)``, and ``asyncio.QueueFull`` comes back
    unchanged as a ``RayTaskError(QueueFull)`` (it is a subclass of
    ``asyncio.QueueFull`` and can be caught directly with ``except``).

    Getting the actor handle uses two private methods of ``Channel``
    (``_get_channel_rank_by_key`` / ``_get_channel_actor``): rlinf has no
    public equivalent, and this subpackage must not modify rlinf itself.
    """

    def __init__(
        self,
        channel: Any,
        *,
        name: str,
        maxsize: int,
        owns_actors: bool = False,
    ) -> None:
        """Initialize.

        Args:
            channel: An already-created or already-connected rlinf
                ``Channel``.
            name: The channel name.
            maxsize: The capacity of each key's queue.
            owns_actors: Whether this object is responsible for
                ``ray.kill``-ing the underlying actors.
        """
        self._channel = channel
        self.name = name
        self.maxsize = maxsize
        self._owns_actors = owns_actors
        self._closed = False
        self._keys: set[str] = set()
        self._waiters: dict[str, int] = {}
        self.bytes_put = 0
        self.bytes_got = 0
        self.items_put = 0
        self.items_got = 0

    # ------------------------------------------------------------------ construction

    @classmethod
    def create(cls, name: str, *, maxsize: int) -> RayQueue:
        """Create a new bounded channel (called by the driver process).

        Args:
            name: The channel name.
            maxsize: The capacity of each key's queue; must be ``>= 1``
                (``maxsize=0`` is **unbounded** in rlinf, which would
                defeat backpressure).

        Returns:
            The channel facade.

        Raises:
            ValueError: ``maxsize < 1``.
        """
        if maxsize < 1:
            raise ValueError(
                f"channel {name!r} needs an explicit maxsize >= 1; "
                "rlinf treats maxsize=0 as unbounded (breaks backpressure)"
            )
        channel_cls = _channel_class()
        channel = channel_cls.create(name=name, maxsize=maxsize)
        return cls(channel, name=name, maxsize=maxsize, owns_actors=True)

    @classmethod
    def connect(
        cls,
        name: str,
        *,
        maxsize: int = 1,
        attempts: int = 40,
        retry_seconds: float = 0.1,
    ) -> RayQueue:
        """Connect to an existing channel.

        ``current_worker=None`` is deliberate: it forces the code path
        through ``put_via_ray`` / ``get_via_ray``, i.e. native asyncio
        ``await ObjectRef`` (see point 3 in the module docstring).

        Retries: ``Channel.connect`` goes through
        ``WorkerGroup.from_group_name``, and a ChannelWorker group's
        registration is not immediately visible to a freshly launched
        worker process -- launching a worker right after
        ``Channel.create`` can occasionally (roughly 1/6 in local testing)
        trigger ``AssertionError: Worker group ... not found`` on the
        worker side. This is a distributed-registration timing issue, not
        a configuration error, hence the backoff retry.

        Args:
            name: The channel name.
            maxsize: The capacity used for local bookkeeping (the real
                capacity is held by the ChannelWorker).
            attempts: The maximum number of attempts.
            retry_seconds: The interval between retries.

        Returns:
            The channel facade.

        Raises:
            TransportClosed: The channel is still unreachable after
                exhausting the retries.
        """
        channel_cls = _channel_class()
        try:
            channel = channel_cls.connect(
                name,
                maxsize=maxsize,
                attempts=attempts,
                retry_seconds=retry_seconds,
            )
        except (RuntimeError, ValueError, KeyError) as exc:
            raise TransportClosed(
                f"channel {name!r} is not reachable after {attempts} attempts: {exc}"
            ) from exc
        return cls(channel, name=name, maxsize=maxsize, owns_actors=False)

    # ------------------------------------------------------------------ send/receive

    def _actor(self, key: str) -> Any:
        del key
        return self._channel.actor

    async def put(self, item: Any, *, key: str, weight: int = 0) -> None:
        """Deliver one item, waiting for space if the queue is full.

        Args:
            item: The item to deliver.
            key: The target key.
            weight: The priority weight (used by rlinf's ``get_batch``).
        """
        self._note(key, item)
        del weight
        await self._channel.put(item, key=key)

    async def put_nowait(self, item: Any, *, key: str, weight: int = 0) -> None:
        """Non-blockingly deliver one item.

        Args:
            item: The item to deliver.
            key: The target key.
            weight: The priority weight.

        Raises:
            asyncio.QueueFull: This key's queue is full (a backpressure
                signal).
        """
        self._note(key, item)
        del weight
        await self._channel.put_nowait(item, key=key)

    async def get(self, *, key: str) -> Any:
        """Get one item, waiting if the queue is empty.

        Args:
            key: The target key.

        Returns:
            The item from the channel.

        Raises:
            TransportClosed: The channel is closed (a sentinel was
                received, or the actor was already reclaimed).
        """
        if self._closed:
            raise TransportClosed(f"channel {self.name!r} is closed")
        self._keys.add(key)
        self._waiters[key] = self._waiters.get(key, 0) + 1
        try:
            item = await self._channel.get(key=key)
        except (asyncio.CancelledError, GeneratorExit):
            raise
        except BaseException as exc:  # noqa: BLE001 - the actor being killed also counts as closed
            if self._closed:
                raise TransportClosed(f"channel {self.name!r} is closed") from exc
            raise
        finally:
            self._waiters[key] = max(0, self._waiters.get(key, 1) - 1)
        if isinstance(item, dict) and item.get("kind") == CLOSE_SENTINEL:
            raise TransportClosed(f"channel {self.name!r} is closed")
        self.items_got += 1
        body = item.get("body") if isinstance(item, dict) else None
        if isinstance(body, bytes):
            self.bytes_got += len(body)
        return item

    def _note(self, key: str, item: Any) -> None:
        self._keys.add(key)
        self.items_put += 1
        body = item.get("body") if isinstance(item, dict) else None
        if isinstance(body, bytes):
            self.bytes_put += len(body)

    def qsize(self, key: str) -> int:
        """Check the watermark of a key.

        Args:
            key: The target key.

        Returns:
            The queue length; returns ``-1`` on query failure (so
            observation never drags down the caller).
        """
        try:
            return int(self._channel.qsize(key))
        except BaseException:  # noqa: BLE001 - observation must not raise
            return -1

    # ------------------------------------------------------------------ finalization

    @property
    def closed(self) -> bool:
        """Whether the channel is closed.

        Returns:
            ``True`` when closed.
        """
        return self._closed

    async def close(self) -> None:
        """Close the channel: send a sentinel for every used key, waking
        up blocked ``get`` calls."""
        if self._closed:
            return
        self._closed = True
        sentinel = {"kind": CLOSE_SENTINEL, "reply_key": "", "body": b""}
        for key in sorted(self._keys):
            for _ in range(max(1, self._waiters.get(key, 0))):
                with contextlib.suppress(BaseException):
                    await self._channel.put_nowait(sentinel, key=key)

    def shutdown(self) -> None:
        """Reclaim the underlying ChannelWorker actors (only the creator
        should call this).

        Running dozens of test cases within one process would otherwise
        accumulate just as many actors, and ray's own reference-counted
        reclamation isn't prompt enough.
        """
        self._closed = True
        if not self._owns_actors:
            return
        with contextlib.suppress(BaseException):
            self._channel.shutdown()


def _channel_class() -> Any:
    """Import Zetta's Ray channel lazily so non-Ray modes stay lightweight."""
    from zetta.runtime.ray.channel import ZettaChannel

    return ZettaChannel


class RayChannelTransport:
    """A ``CommandTransport`` implementation built on rlinf's ``Channel``
    (the default).

    Three channels, each with its own responsibility:

    - ``rr-cmd-*``: a **bounded** command queue keyed by
      ``str(env_rank)``; when full, it returns ``QUEUE_FULL`` directly
      (no queueing);
    - ``rr-ctl-*``: an independent control channel, so cancel /
      heartbeat / binding are never blocked behind environment commands;
    - ``rr-res-*``: the result flow-back, with one drain loop waking up
      waiters by ``request_id``.

    Once ``command_timeout_seconds`` elapses, the caller only gets a
    ``DEADLINE_EXCEEDED``, and **the operation does not reach a terminal
    state** (``state=RUNNING``); when the real result arrives later, the
    callback registered via ``set_late_result_handler`` finalizes it
    ("an RPC timeout is not the same as cancellation").
    """

    def __init__(
        self,
        *,
        command_queue: RayQueue,
        control_queue: RayQueue,
        result_queue: RayQueue,
        worker_ranks: Sequence[int],
        command_timeout_seconds: float = 120.0,
        result_key: str = RESULT_KEY,
    ) -> None:
        """Initialize.

        Args:
            command_queue: The command channel.
            control_queue: The control channel.
            result_queue: The result-flow-back channel.
            worker_ranks: The reachable EnvWorker ranks.
            command_timeout_seconds: The wait cap for a single command;
                ``<= 0`` means no timeout.
            result_key: The result-flow-back key.
        """
        self._commands = command_queue
        self._controls = control_queue
        self._results = result_queue
        self._ranks = list(worker_ranks)
        self._timeout = command_timeout_seconds
        self._result_key = result_key
        self._pending: dict[RequestId, asyncio.Future[ResultEnvelope]] = {}
        self._late_handler: Callable[[ResultEnvelope], Awaitable[None]] | None = None
        self._drain_task: asyncio.Task[None] | None = None
        self._closed = False
        self.command_timeout_count = 0
        self.late_result_count = 0
        self.orphan_result_count = 0
        self.queue_full_count = 0

    # ------------------------------------------------------------------ lifecycle

    @classmethod
    def create(
        cls,
        *,
        names: ChannelNames,
        worker_ranks: Sequence[int],
        command_queue_size: int,
        control_queue_size: int,
        result_queue_size: int,
        command_timeout_seconds: float,
    ) -> RayChannelTransport:
        """Create three new channels and assemble the transport.

        Args:
            names: The set of channel names.
            worker_ranks: The reachable EnvWorker ranks.
            command_queue_size: The command channel's maxsize.
            control_queue_size: The control channel's maxsize.
            result_queue_size: The result channel's maxsize.
            command_timeout_seconds: The wait cap for a single command.

        Returns:
            The transport, not yet started (``await start()`` is still
            needed).
        """
        created: list[RayQueue] = []
        try:
            command_queue = RayQueue.create(names.command, maxsize=command_queue_size)
            created.append(command_queue)
            control_queue = RayQueue.create(names.control, maxsize=control_queue_size)
            created.append(control_queue)
            result_queue = RayQueue.create(names.result, maxsize=result_queue_size)
            created.append(result_queue)
            return cls(
                command_queue=command_queue,
                control_queue=control_queue,
                result_queue=result_queue,
                worker_ranks=worker_ranks,
                command_timeout_seconds=command_timeout_seconds,
            )
        except BaseException:
            for queue in created:
                with contextlib.suppress(BaseException):
                    queue.shutdown()
            raise

    async def start(self) -> None:
        """Start the result-flow-back drain loop."""
        if self._drain_task is None:
            self._drain_task = asyncio.get_running_loop().create_task(self._drain())

    async def close(self) -> None:
        """Close the channels, stop draining, and wake up all waiters."""
        if self._closed:
            return
        self._closed = True
        await self._results.close()
        task = self._drain_task
        self._drain_task = None
        if task is not None:
            task.cancel()
            with contextlib.suppress(BaseException):
                await task
        await self._commands.close()
        await self._controls.close()
        for future in list(self._pending.values()):
            if not future.done():
                future.cancel()
        self._pending.clear()

    def shutdown(self) -> None:
        """Reclaim the actors backing all three channels."""
        for queue in (self._commands, self._controls, self._results):
            queue.shutdown()

    def set_late_result_handler(
        self, handler: Callable[[ResultEnvelope], Awaitable[None]]
    ) -> None:
        """Register the finalization callback for late results (the
        ``LateResultSink``).

        Args:
            handler: An async callback receiving the late
                ``ResultEnvelope``.
        """
        self._late_handler = handler

    # ------------------------------------------------------------------ sending

    def worker_ranks(self) -> Sequence[int]:
        """List the reachable ranks.

        Returns:
            A sequence of ranks.
        """
        return list(self._ranks)

    @property
    def bytes_sent(self) -> int:
        """The total frame-body bytes delivered over the command + control
        channels.

        Returns:
            The byte count.
        """
        return self._commands.bytes_put + self._controls.bytes_put

    @property
    def bytes_received(self) -> int:
        """The total frame-body bytes received over the result channel.

        Returns:
            The byte count.
        """
        return self._results.bytes_got

    def command_depth(self, worker_rank: int) -> int:
        """Check the command-queue watermark for a rank.

        Args:
            worker_rank: The target rank.

        Returns:
            The queue length.
        """
        return self._commands.qsize(str(worker_rank))

    async def send_command(
        self, worker_rank: int, envelope: CommandEnvelope
    ) -> ResultEnvelope:
        """Deliver an environment command and wait for the result to flow
        back.

        Args:
            worker_rank: The target rank.
            envelope: The command.

        Returns:
            The response envelope; transport-layer failures are also
            normalized into an envelope carrying an ``error``.
        """
        return await self._send(
            self._commands,
            key=str(worker_rank),
            kind="command",
            envelope=envelope,
            worker_rank=worker_rank,
        )

    async def send_control(
        self, worker_rank: int, envelope: ControlEnvelope
    ) -> ResultEnvelope:
        """Deliver a control message over the independent control channel
        and wait for the result.

        Args:
            worker_rank: The target rank.
            envelope: The control message.

        Returns:
            The response envelope.
        """
        return await self._send(
            self._controls,
            key=str(worker_rank),
            kind="control",
            envelope=envelope,
            worker_rank=worker_rank,
        )

    async def _send(
        self,
        queue: RayQueue,
        *,
        key: str,
        kind: str,
        envelope: CommandEnvelope | ControlEnvelope,
        worker_rank: int,
    ) -> ResultEnvelope:
        request_id = envelope.request_id
        if self._closed:
            return self._error_envelope(
                envelope,
                make_error(
                    ErrorCode.WORKER_LOST,
                    "ray channel transport is closed",
                    worker_rank=worker_rank,
                ),
            )
        loop = asyncio.get_running_loop()
        future: asyncio.Future[ResultEnvelope] = loop.create_future()
        self._pending[request_id] = future
        try:
            await queue.put_nowait(
                _frame(kind, envelope, reply_key=self._result_key), key=key
            )
        except asyncio.QueueFull:
            self._pending.pop(request_id, None)
            self.queue_full_count += 1
            depth = queue.qsize(key)
            return self._error_envelope(
                envelope,
                make_error(
                    ErrorCode.QUEUE_FULL,
                    f"{kind} queue for rank {worker_rank} is full "
                    f"({depth}/{queue.maxsize}); rejecting instead of queueing",
                    queue_depth=depth,
                    queue_capacity=queue.maxsize,
                    worker_rank=worker_rank,
                ),
            )
        except (asyncio.CancelledError, GeneratorExit):
            self._pending.pop(request_id, None)
            raise
        except BaseException as exc:  # noqa: BLE001 - transport failures must not leak
            self._pending.pop(request_id, None)
            return self._error_envelope(envelope, normalize_exception(exc))
        try:
            if self._timeout > 0:
                return await asyncio.wait_for(future, self._timeout)
            return await future
        except asyncio.TimeoutError:
            # ``asyncio.TimeoutError``, not the builtin ``TimeoutError``: on
            # Python 3.10 the two are unrelated classes (unified only in
            # 3.11), and ``asyncio.wait_for`` raises the former on timeout.
            # An RPC timeout is not the same as cancellation: the operation
            # stays in the registry and is finalized by the result
            # flow-back.
            self._pending.pop(request_id, None)
            self.command_timeout_count += 1
            return ResultEnvelope(
                request_id=request_id,
                session_id=getattr(envelope, "session_id", None),
                operation=envelope.operation,
                state=OperationState.RUNNING,
                error=make_error(
                    ErrorCode.DEADLINE_EXCEEDED,
                    f"waiting for the {kind} result exceeded "
                    f"{self._timeout:.3f}s; the env worker still owns the operation "
                    "(the RPC timeout does not cancel it)",
                    request_id=request_id,
                    worker_rank=worker_rank,
                    timeout_seconds=self._timeout,
                ),
            )
        except (asyncio.CancelledError, GeneratorExit):
            self._pending.pop(request_id, None)
            raise
        except BaseException as exc:  # noqa: BLE001 - must not leak
            self._pending.pop(request_id, None)
            return self._error_envelope(envelope, normalize_exception(exc))

    @staticmethod
    def _error_envelope(
        envelope: CommandEnvelope | ControlEnvelope, error: Any
    ) -> ResultEnvelope:
        return ResultEnvelope(
            request_id=envelope.request_id,
            session_id=getattr(envelope, "session_id", None),
            operation=envelope.operation,
            state=OperationState.FAILED,
            error=error,
        )

    # ------------------------------------------------------------------ flow-back

    async def _drain(self) -> None:
        """The long-running result receiver: wakes up waiters by
        ``request_id``, and hands late results to the callback for
        finalization."""
        while True:
            try:
                frame = await self._results.get(key=self._result_key)
            except (asyncio.CancelledError, GeneratorExit):
                raise
            except TransportClosed:
                return
            except BaseException:  # noqa: BLE001 - the drain loop must never die
                await asyncio.sleep(0)
                continue
            try:
                result = decode_bytes(frame["body"], ResultEnvelope)
            except BaseException:  # noqa: BLE001 - a bad frame is dropped, one frame at a time
                self.orphan_result_count += 1
                continue
            future = self._pending.pop(result.request_id, None)
            if future is not None and not future.done():
                future.set_result(result)
                continue
            self.late_result_count += 1
            handler = self._late_handler
            if handler is None:
                self.orphan_result_count += 1
                continue
            with contextlib.suppress(BaseException):
                await handler(result)


class RayChannelWorkerEndpoint:
    """The EnvWorker side's command / control receiving endpoint (a
    ``WorkerCommandEndpoint`` implementation)."""

    def __init__(
        self,
        *,
        worker_rank: int,
        command_queue: RayQueue,
        control_queue: RayQueue,
        result_queue: RayQueue,
    ) -> None:
        """Initialize.

        Args:
            worker_rank: This worker's rank (= its own command key).
            command_queue: The command channel.
            control_queue: The control channel.
            result_queue: The result-flow-back channel.
        """
        self.worker_rank = worker_rank
        self._commands = command_queue
        self._controls = control_queue
        self._results = result_queue
        self._closed = False
        self.commands_received = 0
        self.controls_received = 0
        self.results_sent = 0
        self.results_dropped = 0

    @classmethod
    def connect(
        cls,
        *,
        worker_rank: int,
        names: ChannelNames,
        command_queue_size: int = 64,
        control_queue_size: int = 16,
        result_queue_size: int = 64,
    ) -> RayChannelWorkerEndpoint:
        """Connect to the three already-existing channels.

        Args:
            worker_rank: This worker's rank.
            names: The set of channel names.
            command_queue_size: The command channel's capacity (local bookkeeping).
            control_queue_size: The control channel's capacity (local bookkeeping).
            result_queue_size: The result channel's capacity (local bookkeeping).

        Returns:
            The worker-side endpoint.
        """
        return cls(
            worker_rank=worker_rank,
            command_queue=RayQueue.connect(names.command, maxsize=command_queue_size),
            control_queue=RayQueue.connect(names.control, maxsize=control_queue_size),
            result_queue=RayQueue.connect(names.result, maxsize=result_queue_size),
        )

    async def get_command(self) -> tuple[CommandEnvelope, str]:
        """Get one environment command belonging to this rank.

        Returns:
            ``(command, reply key)``.

        Raises:
            TransportClosed: The channel is closed.
        """
        frame = await self._commands.get(key=str(self.worker_rank))
        self.commands_received += 1
        return decode_bytes(frame["body"], CommandEnvelope), str(frame["reply_key"])

    async def get_control(self) -> tuple[ControlEnvelope, str]:
        """Get one control message belonging to this rank.

        Returns:
            ``(control message, reply key)``.

        Raises:
            TransportClosed: The channel is closed.
        """
        frame = await self._controls.get(key=str(self.worker_rank))
        self.controls_received += 1
        return decode_bytes(frame["body"], ControlEnvelope), str(frame["reply_key"])

    async def put_result(self, reply_key: str, envelope: ResultEnvelope) -> None:
        """Write the response back to the result channel (blocking delivery:
        the result is never dropped).

        Args:
            reply_key: The flow-back key.
            envelope: The response.
        """
        if self._closed:
            self.results_dropped += 1
            return
        try:
            await self._results.put(_frame("result", envelope), key=reply_key)
        except (asyncio.CancelledError, GeneratorExit):
            raise
        except BaseException:  # noqa: BLE001 - D5: the worker side must never leak
            self.results_dropped += 1
            return
        self.results_sent += 1

    def close(self) -> None:
        """Set the closed flag (the channel itself is closed by its creator)."""
        self._closed = True

    async def aclose(self) -> None:
        """Close the three channel views used by this endpoint."""
        self._closed = True
        for queue in (self._commands, self._controls, self._results):
            await queue.close()


class RayInferenceChannel:
    """The Channel implementation for ``rr_infer_req`` / ``rr_infer_resp``.

    Implements the same ``InferenceChannel`` Protocol as
    ``InProcInferenceChannel``, so no code on either the EnvWorker or
    RolloutWorker side needs to change.
    """

    def __init__(
        self,
        *,
        request_queue: RayQueue,
        response_queue: RayQueue,
        request_queue_size: int = 64,
        response_queue_size: int = 64,
    ) -> None:
        """Initialize.

        Args:
            request_queue: The ``rr_infer_req`` channel.
            response_queue: The ``rr_infer_resp`` channel.
            request_queue_size: The request queue capacity (backpressure watermark).
            response_queue_size: The capacity for each response key.
        """
        self._requests = request_queue
        self._responses = response_queue
        self._request_queue_size = max(1, request_queue_size)
        self._response_queue_size = max(1, response_queue_size)
        self._closed = False
        self.requests_put = 0
        self.requests_rejected = 0
        self.responses_put = 0
        self.responses_dropped = 0

    @classmethod
    def create(
        cls,
        *,
        names: ChannelNames,
        request_queue_size: int,
        response_queue_size: int,
    ) -> RayInferenceChannel:
        """Create the two new request-plane channels.

        Args:
            names: The set of channel names.
            request_queue_size: ``rr_infer_req``'s maxsize.
            response_queue_size: ``rr_infer_resp``'s maxsize.

        Returns:
            The request-plane implementation.
        """
        created: list[RayQueue] = []
        try:
            request_queue = RayQueue.create(
                names.inference_request, maxsize=request_queue_size
            )
            created.append(request_queue)
            response_queue = RayQueue.create(
                names.inference_response, maxsize=response_queue_size
            )
            created.append(response_queue)
            return cls(
                request_queue=request_queue,
                response_queue=response_queue,
                request_queue_size=request_queue_size,
                response_queue_size=response_queue_size,
            )
        except BaseException:
            for queue in created:
                with contextlib.suppress(BaseException):
                    queue.shutdown()
            raise

    @classmethod
    def connect(
        cls,
        *,
        names: ChannelNames,
        request_queue_size: int = 64,
        response_queue_size: int = 64,
    ) -> RayInferenceChannel:
        """Connect to the two already-existing request-plane channels.

        Args:
            names: The set of channel names.
            request_queue_size: The request capacity, for local bookkeeping.
            response_queue_size: The response capacity, for local bookkeeping.

        Returns:
            The request-plane implementation.
        """
        return cls(
            request_queue=RayQueue.connect(
                names.inference_request, maxsize=request_queue_size
            ),
            response_queue=RayQueue.connect(
                names.inference_response, maxsize=response_queue_size
            ),
            request_queue_size=request_queue_size,
            response_queue_size=response_queue_size,
        )

    # ------------------------------------------------------------------ Attributes

    @property
    def closed(self) -> bool:
        """Whether the channel is closed.

        Returns:
            True when closed.
        """
        return self._closed

    @property
    def request_depth(self) -> int:
        """The number of requests currently queued.

        Returns:
            The length of ``rr_infer_req["pending"]``.
        """
        return self._requests.qsize(PENDING_KEY)

    @property
    def request_capacity(self) -> int:
        """The request queue's capacity.

        Returns:
            maxsize.
        """
        return self._request_queue_size

    def register_route(self, routing_token: str) -> None:
        """Pre-register an EnvWorker's response key.

        The ChannelWorker automatically creates the key's queue on the
        first put / get, so this is bookkeeping only.

        Args:
            routing_token: The output of ``make_routing_token``.
        """
        self._responses._keys.add(routing_token)  # noqa: SLF001 - for close to send a sentinel to

    # ------------------------------------------------------------------ Send/receive

    async def put_request_nowait(self, request: InferenceRequest) -> None:
        """Non-blockingly deliver one inference request.

        Args:
            request: The inference request.

        Raises:
            InferenceChannelClosed: The channel is closed.
            asyncio.QueueFull: The request queue is full (the backpressure signal).
        """
        if self._closed:
            raise InferenceChannelClosed("inference channel is closed")
        try:
            await self._requests.put_nowait(_frame("request", request), key=PENDING_KEY)
        except asyncio.QueueFull:
            self.requests_rejected += 1
            raise
        self.requests_put += 1

    async def get_request(self) -> InferenceRequest:
        """Compete for one request from the ``"pending"`` key.

        Returns:
            The inference request.

        Raises:
            InferenceChannelClosed: The channel is closed.
        """
        try:
            frame = await self._requests.get(key=PENDING_KEY)
        except TransportClosed as exc:
            raise InferenceChannelClosed(str(exc)) from exc
        return decode_bytes(frame["body"], InferenceRequest)

    async def put_response_nowait(
        self, routing_token: str, response: ActionResponse
    ) -> None:
        """Non-blockingly deliver the response back to the named EnvWorker;
        drop and count it if full or closed.

        A lost EnvWorker must not sink the inference service (D5), so the
        exception is not raised back to the RolloutWorker here.

        Args:
            routing_token: The routing identifier of the target EnvWorker.
            response: The action-block response.
        """
        if self._closed:
            self.responses_dropped += 1
            return
        try:
            await self._responses.put_nowait(
                _frame("response", response), key=routing_token
            )
        except (asyncio.CancelledError, GeneratorExit):
            raise
        except BaseException:  # noqa: BLE001 - full / lost is always dropped and counted
            self.responses_dropped += 1
            return
        self.responses_put += 1

    async def get_response(self, routing_token: str) -> ActionResponse:
        """Get one response belonging to this caller.

        Args:
            routing_token: This caller's own routing identifier.

        Returns:
            The action-block response.

        Raises:
            InferenceChannelClosed: The channel is closed.
        """
        try:
            frame = await self._responses.get(key=routing_token)
        except TransportClosed as exc:
            raise InferenceChannelClosed(str(exc)) from exc
        return decode_bytes(frame["body"], ActionResponse)

    def close(self) -> None:
        """Set the closed flag (the ``InferenceChannel`` Protocol is synchronous).

        The actual sentinel delivery happens in ``aclose``: the Protocol's
        ``close`` cannot await.
        """
        self._closed = True

    async def aclose(self) -> None:
        """Send sentinels to wake up waiters and close both channels."""
        self._closed = True
        await self._requests.close()
        await self._responses.close()

    def shutdown(self) -> None:
        """Reclaim the actors backing both channels."""
        self._requests.shutdown()
        self._responses.shutdown()
