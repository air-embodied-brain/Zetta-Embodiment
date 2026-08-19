# Copyright (c) 2026 Zetta Contributors
"""Declarative policy for one rollout-evolution campaign.

The lifecycle implementation stays environment agnostic; this module is the
small, serializable contract that controls how many attempts are allowed and
whether the fixed held-out block is a promotion gate or a report-only test.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

HeldoutMode = Literal["test", "validation"]


@dataclass(frozen=True, slots=True)
class EvolutionProtocol:
    """Preregistered, tunable parameters for the required lifecycle.

    ``heldout_seeds`` are always evaluated.  ``heldout_mode=test`` records the
    result without allowing it to reject or promote a candidate; validation
    mode uses the same result as a formal promotion gate.
    """

    rollout_count: int = 50
    heldout_seeds: tuple[int, ...] = field(
        default_factory=lambda: tuple(range(1, 21))
    )
    heldout_mode: HeldoutMode = "test"
    same_seed_pass_rate: float = 0.5
    same_seed_max_rounds: int = 2
    heldout_alpha: float = 0.025
    heldout_min_gain: int = 1
    heldout_min_success_rate: float = 0.0
    heldout_require_significance: bool = True
    heldout_max_rounds: int = 1
    max_infrastructure_attempts: int = 2
    max_candidate_rounds_per_cluster: int = 8
    maximum_target_clusters: int = 2
    maximum_total_candidate_rounds: int = 25
    regression_required: bool = True

    def __post_init__(self) -> None:
        if self.rollout_count < 1:
            raise ValueError("rollout_count must be positive")
        if not self.heldout_seeds:
            raise ValueError("heldout_seeds must not be empty")
        if len(set(self.heldout_seeds)) != len(self.heldout_seeds):
            raise ValueError("heldout_seeds must be unique")
        if self.heldout_mode not in {"test", "validation"}:
            raise ValueError("heldout_mode must be 'test' or 'validation'")
        if not 0 < self.same_seed_pass_rate <= 1:
            raise ValueError("same_seed_pass_rate must be in (0, 1]")
        if not 0 < self.heldout_alpha < 1:
            raise ValueError("heldout_alpha must be in (0, 1)")
        if self.heldout_min_gain < 0:
            raise ValueError("heldout_min_gain must be non-negative")
        if not 0 <= self.heldout_min_success_rate <= 1:
            raise ValueError("heldout_min_success_rate must be in [0, 1]")
        if not isinstance(self.heldout_require_significance, bool):
            raise ValueError("heldout_require_significance must be boolean")
        for name in (
            "same_seed_max_rounds",
            "heldout_max_rounds",
            "max_infrastructure_attempts",
            "max_candidate_rounds_per_cluster",
            "maximum_target_clusters",
            "maximum_total_candidate_rounds",
        ):
            if int(getattr(self, name)) < 1:
                raise ValueError(f"{name} must be positive")

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "EvolutionProtocol":
        value = dict(payload)
        same_seed = value.pop("same_seed", {})
        heldout = value.pop("heldout", {})
        if not isinstance(same_seed, dict) or not isinstance(heldout, dict):
            raise ValueError("same_seed and heldout protocol sections must be objects")
        value.setdefault("same_seed_pass_rate", same_seed.get("pass_rate", 0.5))
        value.setdefault("same_seed_max_rounds", same_seed.get("max_rounds", 2))
        value.setdefault("heldout_mode", heldout.get("mode", "test"))
        value.setdefault("heldout_seeds", tuple(range(1, 21)))
        value.setdefault("heldout_alpha", heldout.get("alpha", 0.025))
        value.setdefault("heldout_min_gain", heldout.get("min_gain", 1))
        value.setdefault(
            "heldout_min_success_rate", heldout.get("min_success_rate", 0.0)
        )
        value.setdefault(
            "heldout_require_significance",
            heldout.get("require_significance", True),
        )
        value.setdefault("heldout_max_rounds", heldout.get("max_rounds", 1))
        if isinstance(value.get("heldout_seeds"), list):
            value["heldout_seeds"] = tuple(int(seed) for seed in value["heldout_seeds"])
        return cls(**value)

    def as_dict(self) -> dict[str, Any]:
        return {
            "rollout_count": self.rollout_count,
            "heldout_seeds": list(self.heldout_seeds),
            "same_seed": {
                "pass_rate": self.same_seed_pass_rate,
                "max_rounds": self.same_seed_max_rounds,
            },
            "heldout": {
                "mode": self.heldout_mode,
                "alpha": self.heldout_alpha,
                "min_gain": self.heldout_min_gain,
                "min_success_rate": self.heldout_min_success_rate,
                "require_significance": self.heldout_require_significance,
                "max_rounds": self.heldout_max_rounds,
            },
            "max_infrastructure_attempts": self.max_infrastructure_attempts,
            "max_candidate_rounds_per_cluster": self.max_candidate_rounds_per_cluster,
            "maximum_target_clusters": self.maximum_target_clusters,
            "maximum_total_candidate_rounds": self.maximum_total_candidate_rounds,
            "regression_required": self.regression_required,
        }

    def runtime_policy(self) -> dict[str, Any]:
        """Return the flat keys consumed by the existing lifecycle code."""

        return {
            "same_seed_pass_rate": self.same_seed_pass_rate,
            "same_seed_max_rounds": self.same_seed_max_rounds,
            "heldout_mode": self.heldout_mode,
            "heldout_alpha": self.heldout_alpha,
            "heldout_min_gain": self.heldout_min_gain,
            "heldout_min_success_rate": self.heldout_min_success_rate,
            "heldout_require_significance": self.heldout_require_significance,
            "heldout_max_rounds": self.heldout_max_rounds,
            "max_candidate_rounds_per_cluster": self.max_candidate_rounds_per_cluster,
            "maximum_target_clusters": self.maximum_target_clusters,
            "maximum_total_candidate_rounds": self.maximum_total_candidate_rounds,
            "skip_regression_gate": not self.regression_required,
        }

    def validate_partition(
        self, rollout_seeds: tuple[int, ...], heldout_seeds: tuple[int, ...]
    ) -> None:
        if len(rollout_seeds) != self.rollout_count:
            raise ValueError(
                f"rollout schedule has {len(rollout_seeds)} seeds; "
                f"expected {self.rollout_count}"
            )
        if tuple(heldout_seeds) != self.heldout_seeds:
            raise ValueError("held-out schedule differs from the protocol")
        if set(rollout_seeds) & set(heldout_seeds):
            raise ValueError("rollout and held-out schedules overlap")


__all__ = ["EvolutionProtocol", "HeldoutMode"]
