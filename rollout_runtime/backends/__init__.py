"""Env and policy backends, plus the sole resolution point for "select a backend by config."

``fake`` is pure stdlib + numpy; ``rlinf_env`` / ``rlinf_policy`` and
``rlinf_maniskill`` allow rlinf + torch, and must be imported lazily. All
adaptation to rlinf is written here -- **``third_party/rlinf/`` submodules are
never modified**. ``robocasa_current`` / ``groot_policy`` do not touch rlinf
at all; they depend only on the current branch's
``robots.robocasa.session_core`` / ``groot_core``.

Landing order: ``fake`` -> ``libero`` -> ``maniskill``. The originally
planned fourth step was ``robotwin`` (the only all-rlinf family that is
``final_only``, useful for exercising the ``obs_list`` length normalization),
but its package does not exist in any of the verified runtime images, so it
remains "declared but not implemented" (see the runtime validation notes for
details).

The ``robocasa`` family, as part of the Rollout Runtime v3 migration, switches
to the current branch's ``RoboCasaSession``/GR00T business logic; the source
branch's ``rlinf_robocasa.py`` backend built on ``rlinf.envs.robocasa`` has
been dropped. ``robocasa_current.py`` is the new backend.

Both launchers (``launch/local.py`` / ``launch/ray_launch.py``) obtain
backends exclusively through this module's ``register_env_family_for`` and
``build_policy_core``, so "what the config says is what runs" has exactly one
implementation, rather than each launcher writing its own set of `if`s.
"""

from __future__ import annotations

from typing import Any

from rollout_runtime.api.enums import ErrorCode
from rollout_runtime.api.errors import RuntimeApiError, make_error

__all__ = [
    "ENV_BACKENDS",
    "POLICY_BACKENDS",
    "build_policy_core",
    "policy_compat_constraints",
    "register_env_family_for",
]

ENV_BACKENDS = ("fake", "libero", "maniskill", "robocasa")
"""Env families that actually have an adapter in this build.

``robotwin`` **has a declaration in ``ENV_FAMILY_BEHAVIORS`` but no
adapter**: its ``robotwin`` package does not exist in any verified runtime
image, so it remains "declared but not implemented," and
``get_env_family`` returns an explicit ``UNSUPPORTED_ENV_SPEC`` with
``declared=True``.

``robocasa`` is likewise not yet registered here: a new
``robocasa_current.py`` (``EnvExecutionCore``, wrapping the current branch's
``RoboCasaSession``) will be added later.
"""

POLICY_BACKENDS = ("fake", "zetta_openpi", "groot")
"""Optional policy backends: ``fake``, ``zetta_openpi`` (openpi / pi0.5), and
``groot`` (the current branch's GR00T)."""


def register_env_family_for(env_family: str) -> Any:
    """Register the adapter for the named family and return it.

    Args:
        env_family: ``EnvSpecMsg.env_family`` / ``RuntimeConfig.env_family``.

    Returns:
        The registered ``EnvFamilyAdapter``.

    Raises:
        RuntimeApiError: This build has no adapter for the family
            (``UNSUPPORTED_ENV_SPEC``).
    """
    if env_family == "fake":
        from rollout_runtime.backends.fake.env import register_fake_env_family

        return register_fake_env_family(replace=True)
    if env_family == "libero":
        from rollout_runtime.backends.rlinf_env import register_libero_env_family

        return register_libero_env_family(replace=True)
    if env_family == "maniskill":
        from rollout_runtime.backends.rlinf_maniskill import (
            register_maniskill_env_family,
        )

        return register_maniskill_env_family(replace=True)
    if env_family == "robocasa":
        from rollout_runtime.backends.robocasa_current import (
            register_robocasa_current_env_family,
        )

        return register_robocasa_current_env_family(replace=True)
    raise RuntimeApiError(
        make_error(
            ErrorCode.UNSUPPORTED_ENV_SPEC,
            f"no env backend for family {env_family!r} in this build",
            env_family=env_family,
            available=list(ENV_BACKENDS),
        )
    )


def build_policy_core(
    *,
    backend: str,
    policy_config: dict[str, Any] | None = None,
    device: str = "cpu",
    dtype: str = "float32",
    policy_family: str = "fake",
    action_dim: int = 7,
    actions_per_chunk: int = 4,
    model_version: str | None = None,
) -> Any:
    """Build a ``PolicyInferenceCore`` for the named backend.

    Args:
        backend: ``"fake"``, ``"rlinf"``, or ``"groot"``.
        policy_config: Backend-private configuration (``RlinfPolicyConfig``
            fields for the ``rlinf`` backend, ``GrootPolicyConfig`` fields
            for the ``groot`` backend).
        device: Device identifier, goes into ``compat_key``.
        dtype: Compute precision name, goes into ``compat_key``.
        policy_family: Family name, determines the batchable parameter whitelist.
        action_dim: Action dimensionality (used by the fake backend).
        actions_per_chunk: Action chunk length (used by the fake backend).
        model_version: Reported model version; ``None`` uses the backend default.

    Returns:
        A not-yet-``load``ed inference core.

    Raises:
        ValueError: Unknown backend name.
    """
    if backend == "fake":
        from rollout_runtime.backends.fake.policy import (
            FakePolicyConfig,
            FakePolicyCore,
        )

        return FakePolicyCore(
            FakePolicyConfig(
                action_dim=action_dim,
                actions_per_chunk=actions_per_chunk,
                device=device,
                dtype=dtype,
                policy_family=policy_family,
                **({"model_version": model_version} if model_version else {}),
            )
        )
    if backend == "zetta_openpi":
        from rollout_runtime.backends.rlinf_policy import (
            RlinfPolicyConfig,
            RlinfPolicyCore,
        )

        merged = dict(policy_config or {})
        merged.setdefault("device", device)
        merged.setdefault("policy_family", policy_family)
        merged.setdefault("action_dim", action_dim)
        if dtype:
            merged.setdefault("dtype", dtype)
        if model_version:
            merged.setdefault("model_version", model_version)
        return RlinfPolicyCore(RlinfPolicyConfig.from_mapping(merged))
    if backend == "groot":
        from rollout_runtime.backends.groot_policy import (
            GrootPolicyConfig,
            GrootPolicyCore,
        )

        merged = dict(policy_config or {})
        merged.setdefault("device", device)
        merged.setdefault("action_dim", action_dim)
        if dtype:
            merged.setdefault("dtype", dtype)
        if model_version:
            merged.setdefault("model_version", model_version)
        return GrootPolicyCore(GrootPolicyConfig.from_mapping(merged))
    raise ValueError(
        f"unknown policy backend {backend!r}; expected one of {list(POLICY_BACKENDS)}"
    )


def policy_compat_constraints(
    *, backend: str, policy_config: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Return the hard constraints that this policy backend must put into ``compat_key``.

    ``compat_key`` is computed on the **EnvWorker** side (``request_inference``),
    but facts such as "cuda_graph bakes in the batch size" or "``openvla_oft``
    requires ``padding='max_length'``" are only known to the policy backend,
    so this function copies them over to the EnvWorker.

    Args:
        backend: ``"fake"``, ``"rlinf"``, or ``"groot"``.
        policy_config: Backend-private configuration.

    Returns:
        Constraint dictionary; empty for the ``fake``/``groot`` backends
        (GR00T does not support cuda_graph/fixed batch; single-request
        serialized inference needs no additional hard constraints).
    """
    if backend != "zetta_openpi":
        return {}
    from rollout_runtime.backends.rlinf_policy import RlinfPolicyConfig

    return RlinfPolicyConfig.from_mapping(
        dict(policy_config or {})
    ).compat_key_constraints()
