# Copyright (c) 2026 RPent Contributors
"""Deterministic, preregistered environment and policy RNG schedules."""

from __future__ import annotations

import hashlib
import random
from typing import Iterable


def policy_rng_seed(*, master_seed: int, task: str, environment_seed: int) -> int:
    payload = f"{master_seed}\0{task}\0{environment_seed}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & 0x7FFFFFFF


def preregister_seed_schedule(
    *,
    master_seed: int,
    task: str,
    rollout_count: int = 50,
    heldout_count: int = 50,
    population: Iterable[int] = range(100_000),
    heldout_seeds: Iterable[int] | None = None,
) -> tuple[tuple[int, ...], tuple[int, ...], dict[str, int]]:
    """Create one frozen, disjoint schedule without exposing ordering to agents."""

    if rollout_count < 1 or heldout_count < 1:
        raise ValueError("rollout_count and heldout_count must be positive")
    candidates = list(dict.fromkeys(int(value) for value in population))
    fixed_heldout = (
        tuple(int(value) for value in heldout_seeds)
        if heldout_seeds is not None
        else None
    )
    if fixed_heldout is not None:
        if len(fixed_heldout) != heldout_count:
            raise ValueError(
                f"fixed heldout schedule has {len(fixed_heldout)} values; "
                f"expected {heldout_count}"
            )
        if len(set(fixed_heldout)) != len(fixed_heldout):
            raise ValueError("fixed heldout schedule contains duplicate seeds")
        missing = set(fixed_heldout) - set(candidates)
        if missing:
            raise ValueError(
                f"fixed heldout seeds are outside the population: {sorted(missing)}"
            )
        protected = set(fixed_heldout)
        candidates = [seed for seed in candidates if seed not in protected]
        if len(candidates) < rollout_count:
            raise ValueError(
                f"seed population has {len(candidates)} rollout values after "
                f"heldout isolation; need {rollout_count}"
            )
    elif len(candidates) < rollout_count + heldout_count:
        needed = rollout_count + heldout_count
        raise ValueError(f"seed population has {len(candidates)} values; need {needed}")
    rng = random.Random(master_seed)
    rng.shuffle(candidates)
    rollout = tuple(candidates[:rollout_count])
    heldout = (
        fixed_heldout
        if fixed_heldout is not None
        else tuple(candidates[rollout_count : rollout_count + heldout_count])
    )
    policy = {
        str(seed): policy_rng_seed(
            master_seed=master_seed,
            task=task,
            environment_seed=seed,
        )
        for seed in (*rollout, *heldout)
    }
    return rollout, heldout, policy
