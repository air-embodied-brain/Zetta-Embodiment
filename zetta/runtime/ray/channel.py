"""Named, bounded, per-key queues backed by a Ray actor."""

from __future__ import annotations

import asyncio
from typing import Any

from zetta.runtime.ray.bootstrap import RAY_NAMESPACE, ensure_ray_initialized

_ACTOR_CLASS = None


def _channel_actor_class():
    """Build the actor class lazily so importing Zetta does not require Ray."""
    global _ACTOR_CLASS
    if _ACTOR_CLASS is not None:
        return _ACTOR_CLASS

    ray = ensure_ray_initialized()

    @ray.remote
    class _ChannelActor:
        def __init__(self, maxsize: int) -> None:
            self.maxsize = maxsize
            self.queues: dict[str, asyncio.Queue[Any]] = {}

        def _queue(self, key: str) -> asyncio.Queue[Any]:
            if key not in self.queues:
                self.queues[key] = asyncio.Queue(maxsize=self.maxsize)
            return self.queues[key]

        async def put(self, item: Any, key: str) -> None:
            await self._queue(key).put(item)

        async def put_nowait(self, item: Any, key: str) -> bool:
            try:
                self._queue(key).put_nowait(item)
            except asyncio.QueueFull:
                return False
            return True

        async def get(self, key: str) -> Any:
            return await self._queue(key).get()

        def qsize(self, key: str) -> int:
            return self._queue(key).qsize()

    _ACTOR_CLASS = _ChannelActor
    return _ACTOR_CLASS


class ZettaChannel:
    """A named Ray actor exposing bounded queues isolated by key."""

    def __init__(self, actor: Any, *, name: str, maxsize: int, owner: bool) -> None:
        self.actor = actor
        self.name = name
        self.maxsize = maxsize
        self._owner = owner
        self._closed = False

    @classmethod
    def create(cls, name: str, *, maxsize: int) -> "ZettaChannel":
        if maxsize < 1:
            raise ValueError("maxsize must be >= 1")
        ensure_ray_initialized()
        actor = _channel_actor_class().options(
            name=name,
            namespace=RAY_NAMESPACE,
        ).remote(maxsize)
        return cls(actor, name=name, maxsize=maxsize, owner=True)

    @classmethod
    def connect(
        cls,
        name: str,
        *,
        maxsize: int = 1,
        attempts: int = 40,
        retry_seconds: float = 0.1,
    ) -> "ZettaChannel":
        ray = ensure_ray_initialized()
        last: BaseException | None = None
        for _ in range(max(1, attempts)):
            try:
                actor = ray.get_actor(name, namespace=RAY_NAMESPACE)
                return cls(actor, name=name, maxsize=maxsize, owner=False)
            except (ValueError, RuntimeError) as exc:
                last = exc
                import time

                time.sleep(retry_seconds)
        raise RuntimeError(f"channel {name!r} is not reachable: {last}")

    async def put(self, item: Any, *, key: str) -> None:
        if self._closed:
            raise RuntimeError(f"channel {self.name!r} is closed")
        await self.actor.put.remote(item, key)

    async def put_nowait(self, item: Any, *, key: str) -> None:
        if self._closed:
            raise RuntimeError(f"channel {self.name!r} is closed")
        accepted = await self.actor.put_nowait.remote(item, key)
        if not accepted:
            raise asyncio.QueueFull

    async def get(self, *, key: str) -> Any:
        if self._closed:
            raise RuntimeError(f"channel {self.name!r} is closed")
        return await self.actor.get.remote(key)

    def qsize(self, key: str) -> int:
        import ray

        return int(ray.get(self.actor.qsize.remote(key)))

    def shutdown(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._owner:
            import ray

            ray.kill(self.actor, no_restart=True)
