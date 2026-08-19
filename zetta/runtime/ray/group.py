"""A compact Ray actor group used by Zetta launchers."""

from __future__ import annotations

import contextlib
import dataclasses
from typing import Any, Sequence

from zetta.runtime.ray.bootstrap import ensure_ray_initialized
from zetta.runtime.ray.placement import PlacementSlot


@dataclasses.dataclass(frozen=True)
class WorkerInfo:
    worker: Any


class GroupCallResult:
    def __init__(self, refs: Sequence[Any]) -> None:
        self._refs = list(refs)

    def wait(self, timeout: float | None = None) -> list[Any]:
        ray = ensure_ray_initialized()
        if timeout is None:
            return list(ray.get(self._refs))
        ready, remaining = ray.wait(
            self._refs, num_returns=len(self._refs), timeout=timeout
        )
        if remaining:
            raise TimeoutError(f"{len(remaining)} worker call(s) did not finish")
        return list(ray.get(ready))


class _RankView:
    def __init__(self, group: "ZettaWorkerGroup", rank: int) -> None:
        self._group = group
        self._rank = rank

    def __getattr__(self, method: str):
        return lambda *args, **kwargs: self._group.invoke_on(
            self._rank, method, *args, **kwargs
        )


class ZettaWorkerGroup:
    def __init__(self, actors: Sequence[Any], *, name: str) -> None:
        self.actors = list(actors)
        self.name = name
        self.worker_info_list = [WorkerInfo(worker=actor) for actor in self.actors]

    @classmethod
    def launch(
        cls,
        worker_cls: type,
        *init_args: Any,
        num_ranks: int,
        name: str,
        placements: Sequence[PlacementSlot],
        **init_kwargs: Any,
    ) -> "ZettaWorkerGroup":
        ray = ensure_ray_initialized()
        if len(placements) != num_ranks:
            raise ValueError(
                f"received {len(placements)} placement slots for {num_ranks} ranks"
            )
        remote_cls = ray.remote(worker_cls)
        actors = []
        nodes = sorted(
            (node for node in ray.nodes() if node.get("Alive")),
            key=lambda node: node.get("NodeID", ""),
        )
        try:
            for rank, slot in enumerate(placements):
                options: dict[str, Any] = {"name": f"{name}:{rank}"}
                if slot.accelerator_rank is not None:
                    options["num_gpus"] = 1
                    # Ray owns CUDA_VISIBLE_DEVICES for GPU actors.  Overriding it
                    # with a physical rank is incorrect when Ray remaps logical GPU
                    # IDs; retain the placement rank as metadata for diagnostics.
                    options["runtime_env"] = {
                        "env_vars": {
                            "ZETTA_ACCELERATOR_RANK": str(slot.accelerator_rank),
                        }
                    }
                if nodes and slot.node_rank < len(nodes):
                    node_resources = nodes[slot.node_rank].get("Resources", {})
                    node_key = next(
                        (key for key in node_resources if key.startswith("node:")), None
                    )
                    if node_key:
                        options["resources"] = {node_key: 0.001}
                actor = remote_cls.options(**options).remote(
                    rank, *init_args, **init_kwargs
                )
                actors.append(actor)
        except BaseException:
            for actor in actors:
                with contextlib.suppress(BaseException):
                    ray.kill(actor, no_restart=True)
            raise
        return cls(actors, name=name)

    def invoke_all(self, method: str, *args: Any, **kwargs: Any) -> GroupCallResult:
        return GroupCallResult(
            [getattr(actor, method).remote(*args, **kwargs) for actor in self.actors]
        )

    def invoke_on(
        self, rank: int, method: str, *args: Any, **kwargs: Any
    ) -> GroupCallResult:
        actor = self.actors[rank]
        return GroupCallResult([getattr(actor, method).remote(*args, **kwargs)])

    def execute_on(self, rank: int) -> _RankView:
        return _RankView(self, rank)

    def __getattr__(self, method: str):
        return lambda *args, **kwargs: self.invoke_all(method, *args, **kwargs)

    def live_ranks(self, timeout: float = 10.0) -> list[int]:
        ray = ensure_ray_initialized()
        live = []
        for rank, actor in enumerate(self.actors):
            try:
                ray.get(actor.__ray_ready__.remote(), timeout=timeout)
            except BaseException:
                continue
            live.append(rank)
        return live

    def shutdown(self) -> None:
        ray = ensure_ray_initialized()
        for actor in self.actors:
            with contextlib.suppress(BaseException):
                ray.kill(actor, no_restart=True)
