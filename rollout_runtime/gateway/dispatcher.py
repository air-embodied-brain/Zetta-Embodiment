"""Sticky routing and batch scatter/gather.

Invariant: **output order == input ``session_ids`` order**. Batch APIs are
split internally by the Gateway into per-session commands dispatched
concurrently, then aggregated back in input order; different sessions may
partially succeed independently, with no cross-session transaction.

Per-rank concurrency is throttled with a semaphore to avoid a single rank
being overwhelmed by one batch of requests.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence

from rollout_runtime.api.enums import OperationState
from rollout_runtime.api.errors import normalize_exception
from rollout_runtime.api.internal import (
    CommandEnvelope,
    ControlEnvelope,
    ResultEnvelope,
)
from rollout_runtime.transport.base import CommandTransport

__all__ = ["CommandDispatcher", "DEFAULT_MAX_CONCURRENCY_PER_RANK"]

DEFAULT_MAX_CONCURRENCY_PER_RANK = 8
"""Per-rank concurrent command limit."""


class CommandDispatcher:
    """Sends commands to the sticky-bound EnvWorker and aggregates batch results in order."""

    def __init__(
        self,
        transport: CommandTransport,
        *,
        max_concurrency_per_rank: int = DEFAULT_MAX_CONCURRENCY_PER_RANK,
    ) -> None:
        """Initialize.

        Args:
            transport: Underlying transport.
            max_concurrency_per_rank: Per-rank concurrency limit.
        """
        self._transport = transport
        self._limit = max_concurrency_per_rank
        self._semaphores: dict[int, asyncio.Semaphore] = {}

    @property
    def transport(self) -> CommandTransport:
        """Return the underlying transport.

        Returns:
            The transport instance.
        """
        return self._transport

    def _semaphore(self, worker_rank: int) -> asyncio.Semaphore:
        semaphore = self._semaphores.get(worker_rank)
        if semaphore is None:
            semaphore = asyncio.Semaphore(self._limit)
            self._semaphores[worker_rank] = semaphore
        return semaphore

    async def send(self, worker_rank: int, envelope: CommandEnvelope) -> ResultEnvelope:
        """Dispatch one environment command.

        Args:
            worker_rank: Target rank.
            envelope: The command.

        Returns:
            Result envelope; transport exceptions are also normalized into an
            envelope carrying ``error``.
        """
        try:
            async with self._semaphore(worker_rank):
                return await self._transport.send_command(worker_rank, envelope)
        except BaseException as exc:  # noqa: BLE001 - D5: never leak
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
        """Dispatch one control message (does not consume the command semaphore).

        The control channel must be independent from regular commands, so that
        cancel / heartbeat / shutdown are never blocked by environment commands.

        Args:
            worker_rank: Target rank.
            envelope: Control message.

        Returns:
            Result envelope.
        """
        try:
            return await self._transport.send_control(worker_rank, envelope)
        except BaseException as exc:  # noqa: BLE001 - D5: never leak
            return ResultEnvelope(
                request_id=envelope.request_id,
                session_id=envelope.session_id,
                operation=envelope.operation,
                state=OperationState.FAILED,
                error=normalize_exception(exc),
            )

    async def scatter(
        self, items: Sequence[tuple[int, CommandEnvelope]]
    ) -> list[ResultEnvelope]:
        """Dispatch multiple commands concurrently and return them in input order.

        Args:
            items: Sequence of ``(rank, command)``.

        Returns:
            List of responses in the same order as the input.
        """
        if not items:
            return []
        tasks = [
            asyncio.create_task(self.send(worker_rank, envelope))
            for worker_rank, envelope in items
        ]
        return list(await asyncio.gather(*tasks))
