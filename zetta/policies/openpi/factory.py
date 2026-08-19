"""Narrow OpenPI factory; this is the only policy registry entry Zetta needs."""

from __future__ import annotations

from typing import Any


def build_openpi_model(config: Any, torch_dtype: Any = None):
    model_type = str(getattr(config, "model_type", "openpi")).lower()
    if model_type not in {"openpi", "pi0", "pi05", "pi0.5"}:
        raise ValueError(f"Zetta OpenPI factory only accepts openpi, got {model_type!r}")
    from zetta.policies.openpi import get_model

    return get_model(config, torch_dtype=torch_dtype)
