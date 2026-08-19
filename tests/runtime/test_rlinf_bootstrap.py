"""Regression guard for the former RLinf source bootstrap."""

from pathlib import Path


def test_main_rlinf_source_bootstrap_has_been_removed() -> None:
    root = Path(__file__).resolve().parents[2]
    assert not (root / "rollout_runtime" / "rlinf_bootstrap.py").exists()


def test_ray_bootstrap_uses_zetta_namespace() -> None:
    from zetta.runtime.ray.bootstrap import RAY_NAMESPACE

    assert RAY_NAMESPACE == "zetta-runtime"
