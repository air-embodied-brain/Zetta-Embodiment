"""Placement parsing for Zetta Ray worker groups."""

from __future__ import annotations

import dataclasses
from typing import Any


@dataclasses.dataclass(frozen=True)
class PlacementSlot:
    node_rank: int
    accelerator_rank: int | None = None


def _expand_range(value: str) -> list[int]:
    result: list[int] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = (int(item) for item in part.split("-", 1))
            result.extend(range(start, end + 1))
        else:
            result.append(int(part))
    return result


def node_slots(num_ranks: int, node_rank: int = 0) -> list[PlacementSlot]:
    return [PlacementSlot(node_rank=node_rank) for _ in range(num_ranks)]


def parse_component_placement(
    declaration: dict[str, Any] | None,
    *,
    num_ranks: int,
    strategy: str = "packed",
) -> list[PlacementSlot]:
    """Translate the placement syntax used by existing presets into slots."""
    if not declaration:
        if strategy != "node":
            raise ValueError(
                f"placement strategy {strategy!r} requires component_placement"
            )
        return node_slots(num_ranks)

    placement = str(declaration.get("placement", "")).strip()
    if not placement:
        raise ValueError("component placement needs a non-empty 'placement'")

    if declaration.get("node_group") == "node" and ":" not in placement:
        slots = [PlacementSlot(node_rank=rank) for rank in _expand_range(placement)]
    elif ":" in placement:
        nodes, accelerators = placement.split(":", 1)
        slots = [
            PlacementSlot(node_rank=node, accelerator_rank=accelerator)
            for node in _expand_range(nodes)
            for accelerator in _expand_range(accelerators)
        ]
    else:
        slots = [
            PlacementSlot(node_rank=0, accelerator_rank=accelerator)
            for accelerator in _expand_range(placement)
        ]

    if len(slots) != num_ranks:
        raise ValueError(
            f"placement describes {len(slots)} slot(s), expected num_ranks={num_ranks}"
        )
    return slots
