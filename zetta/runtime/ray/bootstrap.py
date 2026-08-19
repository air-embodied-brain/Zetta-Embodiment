"""Lazy Ray initialization for Zetta runtime components."""

from __future__ import annotations

from typing import Any

RAY_NAMESPACE = "zetta-runtime"


def ensure_ray_initialized(**kwargs: Any):
    """Initialize (or return) the local Ray runtime in Zetta's namespace."""
    import ray

    if not ray.is_initialized():
        options = {
            "namespace": RAY_NAMESPACE,
            "ignore_reinit_error": True,
            "include_dashboard": False,
        }
        options.update(kwargs)
        ray.init(**options)
    return ray
