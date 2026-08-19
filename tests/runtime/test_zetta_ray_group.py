from __future__ import annotations

import uuid

import pytest

from zetta.runtime.ray.placement import node_slots, parse_component_placement


class CounterWorker:
    def __init__(self, rank: int) -> None:
        self.rank_id = rank

    def rank(self) -> int:
        return self.rank_id


def test_parse_single_node_gpu_range() -> None:
    slots = parse_component_placement({"placement": "0:4-7"}, num_ranks=4)
    assert [(slot.node_rank, slot.accelerator_rank) for slot in slots] == [
        (0, 4),
        (0, 5),
        (0, 6),
        (0, 7),
    ]


def test_node_strategy_repeats_node_without_gpu() -> None:
    slots = parse_component_placement(None, num_ranks=3, strategy="node")
    assert [slot.accelerator_rank for slot in slots] == [None, None, None]


@pytest.mark.ray
def test_group_invokes_every_rank_and_one_rank() -> None:
    ray = pytest.importorskip("ray")
    from zetta.runtime.ray.bootstrap import ensure_ray_initialized
    from zetta.runtime.ray.group import ZettaWorkerGroup

    ensure_ray_initialized()
    group = ZettaWorkerGroup.launch(
        CounterWorker,
        num_ranks=2,
        name=f"counter-group-{uuid.uuid4().hex}",
        placements=node_slots(2),
    )
    try:
        assert sorted(group.invoke_all("rank").wait()) == [0, 1]
        assert group.invoke_on(1, "rank").wait() == [1]
    finally:
        group.shutdown()
        ray.shutdown()
