"""The env family registry.

``EnvFamilyAdapter`` absorbs family differences (reset signature,
``chunk_step`` return length, obs extraction, action preprocessing, device,
privileged extensions); each family registers one instance into
``ENV_FAMILY_REGISTRY``. The Gateway validates against the capability table
at ``create_session`` time, returning ``UNSUPPORTED_ENV_SPEC`` for
unsupported cases instead of failing at runtime.

The registry mechanism and Protocol live here; this module also holds the
**declarative branches for real families**
(``ENV_FAMILY_BEHAVIORS``): each family's ``reset`` signature, how many obs
``chunk_step`` returns, whether actions are numpy or torch, whether an
accelerator is needed, and which privileged extensions exist. Separating
declaration from implementation means neither the Gateway nor the
EnvWorker needs an ``if env_family == ...`` branch, and a family that is
"declared but not implemented in this build" (currently only
``robotwin``, whose package is absent from the validated runtime images)
gets a clear ``UNSUPPORTED_ENV_SPEC`` + ``declared=True`` instead of a
``KeyError``.
"""

from __future__ import annotations

import types
from collections.abc import Mapping
from typing import Protocol

from rollout_runtime.api.enums import ErrorCode
from rollout_runtime.api.errors import RuntimeApiError, make_error
from rollout_runtime.api.messages import EnvFamilyCapability, EnvSpecMsg
from rollout_runtime.core.env_execution import (
    LOCKSTEP_VECTOR_FORM,
    PER_SLOT_FORM,
    EnvExecutionCore,
    EnvFamilyBehavior,
)

__all__ = [
    "ENV_FAMILY_BEHAVIORS",
    "ENV_FAMILY_REGISTRY",
    "LIBERO_ENV_FAMILY",
    "MANISKILL_ENV_FAMILY",
    "ROBOCASA_ENV_FAMILY",
    "ROBOCASA_EXTENSIONS",
    "EnvFamilyAdapter",
    "behavior_for",
    "capability_from_behavior",
    "clear_env_families",
    "env_families",
    "get_env_family",
    "register_env_family",
    "requested_core_form",
    "validate_env_spec",
]

LIBERO_ENV_FAMILY = "libero"
"""The libero family name (the only real family shipped initially)."""

MANISKILL_ENV_FAMILY = "maniskill"
"""The maniskill family name (GPU-batched)."""

ROBOCASA_ENV_FAMILY = "robocasa"
"""The robocasa family name (a second family, CPU subprocess)."""

CORE_FORM_CONFIG_KEY = "core_form"
"""The key in ``env_config`` that selects the execution-core form. It
enters ``EnvSpecMsg.digest()``."""

LIBERO_EXTENSIONS = frozenset(
    {
        "libero.privileged_contacts",
        "libero.render_camera",
        "libero.get_camera_meta",
        "libero.cached_image",
        "libero.raw_obs",
        "libero.critic_state",
    }
)
"""The full names of LIBERO's privileged methods.

An undeclared method always returns ``UNSUPPORTED_EXTENSION`` (rather than
crashing). All are read-only, so ``EXTENSION_CALL`` neither occupies
``operation_seq`` nor requires a prior reset.

``libero.critic_state`` mirrors the legacy
``robots/libero/env_client.py::privileged_critic_state``: it reads the
current privileged Critic state independently of ``chunk_step``
(diagnostic scripts and cross-branch parity comparisons need to read it
at any point in time, not only the automatic call inside
``_chunk_step_one`` on every step). The ``reset_tracker`` parameter is
passed through to ``libero_privileged.CRITIC_STATE_METHOD``; the caller
controls whether to clear cross-call history, which is an independent
path from the automatic ``critic_history_pending_reset`` semantics inside
``_chunk_step_one`` -- the two are separate paths that do not interfere
with each other (both are bound to the same ``_rr_critic_history`` state
on the subprocess side, but are cleared at different times by two
separate explicit requests).

``libero.raw_obs`` is a required legacy seam: ``robots/libero/tools.py``
has 10 motion primitives that read ``env.raw_obs()`` directly for
``robot0_eef_quat`` / ``robot0_eef_pos`` / ``robot0_gripper_qpos``, plus
``agentview_image`` / ``agentview_depth`` / ``robot0_eye_in_hand_image`` /
``robot0_eye_in_hand_depth`` and "object keys ending in ``_pos``"
(``view_driver_state`` relies on it to list ``object_names``).
``Observation.state``'s 8 dimensions (eef_pos + quat2axisangle +
gripper_qpos) **carry insufficient information**: axisangle cannot recover
the sign of the quat, and there are no images or depth at all.
"""

ROBOCASA_EXTENSIONS = frozenset(
    {
        "robocasa.snapshot",
        "robocasa.finalize_episode",
    }
)
"""The full names of the robocasa extension methods (runtime v3 design
Stage 7).

Both methods are existing ``RoboCasaSession`` entry points also used by
``env_server.py``; Stage 7 makes them reachable through the sole Runtime
path as well, not only directly over HTTP:

- ``robocasa.snapshot(include_images: bool)`` ->
  ``RoboCasaSession.snapshot``: Role1's visual review
  (``role1_actor.decide_action(observation_response=...)``) and Gen0's
  ``initial_observation_identity`` both need this one payload of
  **raw named state + data-URL images**. ``Observation`` cannot hold it:
  ``state`` is a flat vector (dropping key names), and images are PNG
  ``PayloadRef`` (hashed from a different source than the JPEG data URL in
  the audit record). Going through this extension means the evidence
  bytes Role1 receives are identical before and after migration, without
  having to reimplement the encoding on the application side.
- ``robocasa.finalize_episode()`` ->
  ``RoboCasaSession.finalize_episode_artifacts``: video is only flushed
  to disk here (``_close_video_writers``), and
  ``zetta.evolution.trajectory`` requires the episode video to exist and
  be non-empty. Without this, an episode can never reach finalization.

``release`` is **not** an extension method: in the direct-connection era,
``RoboCasaEnvClient.release()``'s semantics were "return the slot binding
for reuse by the next rollout process", which in the Runtime is
``close_sessions`` (the Gateway's binding + ``EnvPool.release``); adding
another synonymous extension would only create two coexisting release
paths.
"""

ENV_FAMILY_BEHAVIORS: dict[str, EnvFamilyBehavior] = {
    LIBERO_ENV_FAMILY: EnvFamilyBehavior(
        env_family=LIBERO_ENV_FAMILY,
        env_type="libero",
        reset_signature="env_idx_reset_state_ids",
        chunk_obs_layout="per_step",
        action_layout="numpy_env_chunk_dim",
        device_kind="cpu_subproc",
        extensions=LIBERO_EXTENSIONS,
        core_forms=frozenset({PER_SLOT_FORM, LOCKSTEP_VECTOR_FORM}),
        obs_extraction="LiberoEnv._wrap_obs -> EnvOutput.prepare_observations",
    ),
    MANISKILL_ENV_FAMILY: EnvFamilyBehavior(
        env_family=MANISKILL_ENV_FAMILY,
        env_type="maniskill",
        reset_signature="seed_options",
        chunk_obs_layout="per_step",
        action_layout="torch_env_chunk_dim",
        device_kind="gpu_batched",
        # The natural form of a GPU-batched family is a vector: num_envs
        # lanes inside a single sapien scene.
        core_forms=frozenset({PER_SLOT_FORM, LOCKSTEP_VECTOR_FORM}),
        obs_extraction="ManiskillEnv._wrap_obs -> EnvOutput.prepare_observations",
    ),
    ROBOCASA_ENV_FAMILY: EnvFamilyBehavior(
        env_family=ROBOCASA_ENV_FAMILY,
        # Not an rlinf ``SupportedEnvType``: RoboCasaSession is a
        # clean-room implementation that never calls rlinf's
        # ``prepare_actions`` (runtime v3 design).
        env_type="robocasa_session",
        # RoboCasaSession.reset(payload) is a single dict payload
        # (task/seed/split/...), neither env-index style nor
        # gym-keyword style -- it is the native signature of the current
        # branch's robots/robocasa/session_core.py, not the upstream
        # branch's rlinf.envs.robocasa.RobocasaEnv.reset(env_idx, options)
        # (that backend has since been discarded by the migration).
        reset_signature="task_seed_split_payload",
        # RoboCasaSession.execute_chunk loops per step and records one
        # step record per step, so it's per_step rather than final_only;
        # but the per-step record doesn't include images (bandwidth
        # considerations, see the step_record in session_core.py's
        # execute_chunk), so the adapter must read session.observation
        # (the raw numpy dict) separately for images -- they aren't in
        # execute_chunk's return value.
        chunk_obs_layout="per_step",
        # What's received at the adapter boundary is still
        # [chunk, action_dim] numpy (the Runtime's unified shape); the
        # adapter internally converts it to the python list expected by
        # RoboCasaSession.execute_chunk.
        action_layout="numpy_env_chunk_dim",
        # RoboCasaSession is neither pure CPU (rendering forces
        # MUJOCO_GL=egl, requiring a GPU) nor a batched-GPU-tensor family
        # like maniskill (single-process numpy, one gymnasium.make() per
        # session). needs_accelerator_override declares "needs a GPU"
        # independently, instead of borrowing device_kind="gpu_batched"
        # and its batched-tensor semantics that it doesn't actually have.
        device_kind="cpu_subproc",
        needs_accelerator_override=True,
        extensions=ROBOCASA_EXTENSIONS,
        # per_slot only: RoboCasaSession is single-process, single-env,
        # with no "multiple lanes within one process" vectorized form;
        # lockstep_vector (batching within the same pool/tick) has no
        # meaning here.
        core_forms=frozenset({PER_SLOT_FORM}),
        obs_extraction="RoboCasaSession.observation (raw gym dict) -> Observation",
    ),
    # robotwin remains "declared only, not implemented": the `robotwin`
    # package is absent from every validated runtime image, and pulling
    # it in would require an entire asset bundle. Keeping this
    # declaration matters because
    # (a) it is the **only** `final_only` family across all of rlinf
    # (robotwin_env.py:322 submits the whole chunk and only returns 1
    # obs), which is the counter-example fact for normalization; and
    # (b) create_session therefore gets a clear "declared but not
    # implemented in this build" instead of "unknown family".
    "robotwin": EnvFamilyBehavior(
        env_family="robotwin",
        env_type="robotwin",
        reset_signature="env_idx_env_seeds",
        chunk_obs_layout="final_only",
        action_layout="numpy_env_chunk_dim",
        device_kind="cpu_subproc",
        obs_extraction="RoboTwinEnv._wrap_obs -> EnvOutput.prepare_observations",
    ),
}
"""The declaration table of real env families across the six divergence
axes.

``fake`` is not in this table: it carries its own declaration from
``backends/fake/env.py`` and is not an rlinf family.
"""


def behavior_for(env_family: str) -> EnvFamilyBehavior:
    """Look up a declaration by family name.

    Args:
        env_family: The family name.

    Returns:
        The family declaration.

    Raises:
        RuntimeApiError: The family has no declaration
            (``UNSUPPORTED_ENV_SPEC``).
    """
    behavior = ENV_FAMILY_BEHAVIORS.get(env_family)
    if behavior is None:
        raise RuntimeApiError(
            make_error(
                ErrorCode.UNSUPPORTED_ENV_SPEC,
                f"no behavior declared for env family {env_family!r}",
                declared_families=sorted(ENV_FAMILY_BEHAVIORS),
            )
        )
    return behavior


def capability_from_behavior(
    behavior: EnvFamilyBehavior,
    *,
    supports_auto_reset: bool = False,
    supports_reset_state_id: bool | None = None,
) -> EnvFamilyCapability:
    """Project a family declaration into the capability the Gateway uses.

    ``EnvFamilyCapability`` is defined in ``api/messages.py`` (the Gateway
    needs it to decide ``UNSUPPORTED_ENV_SPEC`` without importing numpy),
    so this function performs a one-way core -> api projection.

    Args:
        behavior: The family declaration.
        supports_auto_reset: Whether this family supports auto-reset.
        supports_reset_state_id: Whether ``ResetSpec.reset_state_id`` is
            supported; ``None`` means infer from ``reset_signature``.

    Returns:
        The capability table entry.
    """
    if supports_reset_state_id is None:
        supports_reset_state_id = behavior.reset_signature == "env_idx_reset_state_ids"
    return EnvFamilyCapability(
        env_family=behavior.env_family,
        per_step_obs_available=behavior.per_step_obs_available,
        supports_auto_reset=supports_auto_reset,
        supports_reset_state_id=supports_reset_state_id,
        extensions=behavior.extensions,
        needs_accelerator=behavior.needs_accelerator,
        core_forms=frozenset(behavior.core_forms),
        supports_coalescing=behavior.supports_coalescing,
    )


def requested_core_form(env_spec: EnvSpecMsg, behavior: EnvFamilyBehavior) -> str:
    """Read the requested execution-core form from ``env_config`` and
    validate it against the family declaration.

    The form-selection bit is deliberately placed in ``env_config``
    rather than a new field on ``EnvSpecMsg``: it must enter
    ``EnvSpecMsg.digest()`` (the two forms are two physically distinct
    pools), and ``env_config`` naturally enters the digest. The default is
    ``per_slot``.

    Args:
        env_spec: The environment specification.
        behavior: The family declaration.

    Returns:
        The form name (one of ``CORE_FORMS``).

    Raises:
        RuntimeApiError: The form is unknown, or this family does not
            declare support for it (``INVALID_ARGUMENT``).
    """
    requested = str(
        (env_spec.env_config or {}).get(CORE_FORM_CONFIG_KEY) or PER_SLOT_FORM
    )
    try:
        return behavior.require_core_form(requested)
    except ValueError as exc:
        raise RuntimeApiError(
            make_error(
                ErrorCode.INVALID_ARGUMENT,
                str(exc),
                env_family=behavior.env_family,
                requested_core_form=requested,
                declared_core_forms=sorted(behavior.core_forms),
            )
        ) from exc


class EnvFamilyAdapter(Protocol):
    """The adapter for a single env family."""

    @property
    def env_family(self) -> str:
        """The family name.

        Returns:
            The family identifier, matching ``EnvSpecMsg.env_family``.
        """
        ...

    @property
    def capability(self) -> EnvFamilyCapability:
        """The family capability declaration.

        Returns:
            The capability table entry.
        """
        ...

    def create_core(self) -> EnvExecutionCore:
        """Create an execution-core instance.

        Returns:
            An execution core that has not yet been ``build``.
        """
        ...


_REGISTRY: dict[str, EnvFamilyAdapter] = {}

ENV_FAMILY_REGISTRY: Mapping[str, EnvFamilyAdapter] = types.MappingProxyType(_REGISTRY)
"""A read-only **live** view of the registry.

It changes along with ``register_env_family``, so a reference can be held
at module level.
"""


def register_env_family(adapter: EnvFamilyAdapter, *, replace: bool = False) -> None:
    """Register an env family adapter.

    Args:
        adapter: The family adapter.
        replace: Whether to allow overwriting an existing registration
            under the same name (for tests).

    Raises:
        ValueError: The family name is already registered and
            ``replace`` is false.
    """
    name = adapter.env_family
    if not replace and name in _REGISTRY:
        raise ValueError(f"env family already registered: {name!r}")
    _REGISTRY[name] = adapter


def get_env_family(env_family: str) -> EnvFamilyAdapter:
    """Look up a family adapter by name.

    Args:
        env_family: The family name.

    Returns:
        The family adapter.

    Raises:
        RuntimeApiError: The family is not registered
            (``UNSUPPORTED_ENV_SPEC``).
    """
    adapter = _REGISTRY.get(env_family)
    if adapter is None:
        declared = env_family in ENV_FAMILY_BEHAVIORS
        message = (
            f"env family {env_family!r} is declared in ENV_FAMILY_BEHAVIORS but no "
            "adapter is registered in this build (this build ships libero / "
            "maniskill / robocasa; robotwin stays declaration-only because the "
            "`robotwin` package is absent from the validated runtime images)"
            if declared
            else f"unknown env family: {env_family!r}"
        )
        raise RuntimeApiError(
            make_error(
                ErrorCode.UNSUPPORTED_ENV_SPEC,
                message,
                known_families=sorted(_REGISTRY),
                declared_families=sorted(ENV_FAMILY_BEHAVIORS),
                declared=declared,
            )
        )
    return adapter


def env_families() -> Mapping[str, EnvFamilyAdapter]:
    """Return a read-only view of the registry.

    Returns:
        A mapping from family name to adapter.
    """
    return types.MappingProxyType(_REGISTRY)


def clear_env_families() -> None:
    """Clear the registry (for tests)."""
    _REGISTRY.clear()


def validate_env_spec(
    env_spec: EnvSpecMsg,
    *,
    capabilities: Mapping[str, EnvFamilyCapability] | None = None,
) -> EnvFamilyCapability:
    """Validate that an env spec can be served.

    Args:
        env_spec: The environment specification.
        capabilities: An optional capability table (reported by the
            worker when the Gateway side does not import core);
            ``None`` means query this process's registry.

    Returns:
        The corresponding capability.

    Raises:
        RuntimeApiError: The family is unsupported, or ``pool_size`` is
            invalid.
    """
    if env_spec.pool_size < 1:
        raise RuntimeApiError(
            make_error(
                ErrorCode.INVALID_ARGUMENT,
                f"pool_size must be >= 1, got {env_spec.pool_size}",
            )
        )
    if capabilities is not None:
        capability = capabilities.get(env_spec.env_family)
        if capability is None:
            raise RuntimeApiError(
                make_error(
                    ErrorCode.UNSUPPORTED_ENV_SPEC,
                    f"env family not served here: {env_spec.env_family!r}",
                    known_families=sorted(capabilities),
                )
            )
        return capability
    return get_env_family(env_spec.env_family).capability
