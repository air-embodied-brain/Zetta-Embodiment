# Copyright (c) 2026 Zetta Contributors
"""``PolicyInferenceCore`` for GR00T N1.5 on the FlashRT-style CUDA-Graph path.

Identical to ``GrootPolicyCore`` (same observation/action contract, same
``Gr00tModelCore`` pre/post-processing and seed handling) except that, after
the checkpoint is loaded, the model call is routed through
``flashrt_groot_n15.Gr00tN15GraphRunner`` (one captured CUDA Graph per request
shape, flash-attn kept inside the graph). ``groot_core`` is untouched, so the
``groot`` backend remains the reference implementation to compare against.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import Any

from rollout_runtime.backends.groot_policy import GrootPolicyConfig, GrootPolicyCore

__all__ = ["FlashRtGrootPolicyConfig", "FlashRtGrootPolicyCore"]


@dataclasses.dataclass(kw_only=True)
class FlashRtGrootPolicyConfig(GrootPolicyConfig):
    """``GrootPolicyConfig`` plus the graph cache size.

    Attributes:
        max_graphs: Maximum number of shape-specialised CUDA graphs kept
            (one per ``(prompt token count, image tile count)``); at least 1.
    """

    max_graphs: int = 16

    def __post_init__(self) -> None:
        if self.max_graphs < 1:
            raise ValueError(f"flashrt_groot max_graphs must be >= 1, got {self.max_graphs}")

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any] | None) -> FlashRtGrootPolicyConfig:
        payload = dict(config or {})
        missing = [key for key in ("groot_root", "model_path") if key not in payload]
        if missing:
            raise ValueError(f"flashrt_groot policy_config missing required keys: {missing}")
        known = {field.name for field in dataclasses.fields(cls)}
        unknown = sorted(set(payload) - known)
        if unknown:
            raise ValueError(f"unknown flashrt_groot policy_config keys: {unknown}")
        return cls(**payload)


class FlashRtGrootPolicyCore(GrootPolicyCore):
    """``GrootPolicyCore`` whose model call is a replayed CUDA Graph."""

    config: FlashRtGrootPolicyConfig

    def load(self) -> None:
        super().load()
        from rollout_runtime.backends.flashrt_groot_n15 import install_graph_runner

        self.graph_runner = install_graph_runner(
            self.core.policy, max_graphs=self.config.max_graphs
        )
