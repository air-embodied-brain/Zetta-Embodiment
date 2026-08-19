"""The environment execution-core interface.

``EnvExecutionCore`` is a thin adapter layer: **it does not rewrite rlinf
env**. 16+ env families have been confirmed to share the same constructor
signature ``(cfg, num_envs, seed_offset, total_num_processes, worker_info)``
plus ``reset/step/chunk_step/update_reset_state_ids/is_start``, so the real
implementation just translates the Runtime's session/slot semantics into
family calls, reusing ``action_utils.prepare_actions`` /
``envs/utils.to_tensor`` / ``venv/venv.py`` / ``EnvOutput`` verbatim.

This module holds three things, all stdlib + numpy:

1. ``ChunkOutcome``: the normalized structure;
2. ``EnvFamilyBehavior``: the **declarative** expression of the "six
   confirmed divergences" table -- family adapters self-describe against
   it, and upstream code no longer has to guess;
3. ``normalize_chunk_outcome``: the sole normalization exit point. libero /
   maniskill / realworld's ``obs_list`` length equals the chunk_size, while
   robotwin submits the whole chunk and only returns 1
   (``robotwin_env.py:322``) -- both come out of here, so upstream code
   never treats one family's chunk length as a universal contract.

The implementation that actually touches rlinf is in
``backends/rlinf_env.py``.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from typing import Any, Protocol

import numpy as np

from rollout_runtime.api.messages import (
    EnvSpecMsg,
    Observation,
    PerStepRecord,
    ResetSpec,
)

__all__ = [
    "ACTION_LAYOUTS",
    "CHUNK_OBS_LAYOUTS",
    "CORE_FORMS",
    "DEVICE_KINDS",
    "LOCKSTEP_VECTOR_FORM",
    "PER_SLOT_FORM",
    "RESET_SIGNATURES",
    "ChunkOutcome",
    "DynamicSlotPool",
    "EnvExecutionCore",
    "EnvFamilyBehavior",
    "LaneStatusReader",
    "normalize_chunk_outcome",
]

RESET_SIGNATURES = (
    "env_idx_reset_state_ids",  # libero: reset(env_idx, reset_state_ids)
    "env_idx_env_seeds",  # robotwin: reset(env_idx, env_seeds)
    "env_idx_options",  # rlinf robocasa (discarded, superseded): reset(env_idx, options)
    "seed_options",  # maniskill / gym: reset(seed=..., options=...)
    "task_seed_split_payload",  # RoboCasaSession: reset({"task", "seed", "split", ...})
)
"""The five confirmed forms of the ``reset`` signature.

The first two take the env index as the first positional argument;
``seed_options`` uses gym-style keywords. ``env_idx_options`` corresponds
to the upstream branch ``rlinf.envs.robocasa.RobocasaEnv.reset(env_idx,
options)`` -- that backend has since been discarded as the runtime v3
design migrated away from it, and this literal is kept only for historical
declaration reference; no adapter in this build uses it.
``task_seed_split_payload`` is the real signature of the current branch's
``RoboCasaSession.reset(payload)``: a single dict payload (``task`` /
``seed`` / ``split`` / ``video_dir`` / ...), which is neither env-index
style nor gym-keyword style -- it is inherently a single-process,
single-session HTTP payload contract, not a native signature for a batched
env pool, and forcing it into any of the first four would obscure that
fact.
"""

PER_SLOT_FORM = "per_slot"
"""One family env with ``num_envs=1`` per slot (the default form)."""

LOCKSTEP_VECTOR_FORM = "lockstep_vector"
"""One family env with ``num_envs=pool_size`` for the whole pool; a single
tick of the same pool composes into one ``chunk_step`` call."""

CORE_FORMS = (PER_SLOT_FORM, LOCKSTEP_VECTOR_FORM)
"""The two forms of an execution core.

The selector is ``EnvSpecMsg.env_config["core_form"]``, and therefore
**enters the digest**: the two forms are two physically distinct pools and
must never collide. Each family declares which forms it supports via
``EnvFamilyBehavior.core_forms``.

- ``per_slot``: slots are fully independent, ``chunk_step`` loops inside
  the core, with no masking concerns;
- ``lockstep_vector``: slots are lanes of the same vectorized env. The
  family's ``step`` advances **all** lanes at once (true for libero /
  maniskill / robocasa), so an absent lane must be padded with a hold
  action and counted honestly; see the three semantic rules.
"""

CHUNK_OBS_LAYOUTS = (
    "per_step",  # libero / maniskill / realworld: len(obs_list) == chunk_size
    "final_only",  # robotwin: submits the whole chunk, len(obs_list) == 1
)
"""The two confirmed forms of the length of ``obs_list`` returned by
``chunk_step``."""

ACTION_LAYOUTS = (
    "numpy_env_chunk_dim",  # [num_envs, chunk, action_dim] numpy (libero)
    "torch_env_chunk_dim",  # same shape but torch (maniskill on GPU)
)
"""The ``chunk_step`` input layout (the shape after the family-specific
branch in ``prepare_actions``)."""

DEVICE_KINDS = (
    "cpu_subproc",  # libero: ReconfigureSubprocEnv, CPU subprocess
    "gpu_batched",  # maniskill: vectorized on GPU
)
"""The device form. The EnvWorker's placement decides whether to assign a
GPU based on this."""


@dataclasses.dataclass(frozen=True, kw_only=True)
class ChunkOutcome:
    """The normalized result of one ``chunk_step`` call on a single slot.

    A confirmed family divergence: libero / maniskill / realworld return an
    ``obs_list`` whose length equals ``chunk_size``, while robotwin submits
    the whole chunk and only returns a list of length 1
    (``robotwin_env.py:322``). ``EnvFamilyAdapter`` is responsible for
    normalizing both into this structure, explicitly declaring which one
    via ``per_step_obs_available`` so upstream code no longer has to guess.

    Attributes:
        observation: The final observation after the chunk executes.
        reward: The cumulative reward within the chunk.
        terminated: The termination flag (true if any step within the
            chunk terminated).
        truncated: The truncation flag.
        executed_horizon: The number of environment steps actually
            executed.
        per_step: Per-step records; ``None`` when
            ``per_step_obs_available`` is false.
        per_step_obs_available: Whether this family returns observations
            per step.
        info: Family-private information.
    """

    observation: Observation | None = None
    reward: float = 0.0
    terminated: bool = False
    truncated: bool = False
    executed_horizon: int = 0
    per_step: list[PerStepRecord] | None = None
    per_step_obs_available: bool = True
    info: dict[str, Any] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass(frozen=True, kw_only=True)
class EnvFamilyBehavior:
    """One env family's declaration across the six divergence axes.

    This table represents the shape of "real family branches": a family
    adapter only declares what it looks like, and the normalization logic
    (``normalize_chunk_outcome``) as well as upstream code (EnvWorker /
    Gateway) only ever read the declaration, never writing an
    ``if env_family == ...`` branch.

    Attributes:
        env_family: The family name (``EnvSpecMsg.env_family``).
        env_type: The value of rlinf's ``SupportedEnvType``, fed to
            ``prepare_actions``.
        reset_signature: One of ``RESET_SIGNATURES``.
        chunk_obs_layout: One of ``CHUNK_OBS_LAYOUTS``.
        action_layout: One of ``ACTION_LAYOUTS``.
        device_kind: One of ``DEVICE_KINDS``.
        extensions: The full ``extension_call`` names this family declares
            support for.
        core_forms: The execution-core forms this family's adapter can
            build (a non-empty subset of ``CORE_FORMS``).
        max_pool_size: The upper bound on slots allowed in a single pool;
            ``None`` means unlimited.
        obs_extraction: A description of the obs-extraction path, used only
            for diagnostics and documentation.
        needs_accelerator_override: Explicitly overrides
            ``needs_accelerator``; ``None`` means infer from
            ``device_kind == "gpu_batched"`` (unchanged default behavior
            for the two real families). ``device_kind`` splits families
            into two kinds based on "whether it does batched GPU tensor
            computation" (cpu_subproc / gpu_batched), but not every family
            that needs an accelerator does batched tensor computation --
            RoboCasa (runtime v3 design) is a single-process numpy env that
            relies on a MuJoCo/EGL renderer to occupy the GPU; it is
            neither CPU-only nor a maniskill-style batched-tensor family,
            so it needs to declare "needs an accelerator" explicitly,
            independent of ``device_kind``.

    Raises:
        ValueError: Any enum field takes a value outside the confirmed set.
    """

    env_family: str
    env_type: str
    reset_signature: str
    chunk_obs_layout: str
    action_layout: str
    device_kind: str
    extensions: frozenset[str] = frozenset()
    core_forms: frozenset[str] = frozenset({PER_SLOT_FORM})
    max_pool_size: int | None = None
    obs_extraction: str = "EnvOutput.prepare_observations"
    needs_accelerator_override: bool | None = None

    def __post_init__(self) -> None:
        """Validate that the four enum fields and ``core_forms`` fall
        within the confirmed value sets.

        Raises:
            ValueError: A value is not in the corresponding constant
                tuple, or ``core_forms`` is empty / contains an unknown
                form.
        """
        for value, allowed, label in (
            (self.reset_signature, RESET_SIGNATURES, "reset_signature"),
            (self.chunk_obs_layout, CHUNK_OBS_LAYOUTS, "chunk_obs_layout"),
            (self.action_layout, ACTION_LAYOUTS, "action_layout"),
            (self.device_kind, DEVICE_KINDS, "device_kind"),
        ):
            if value not in allowed:
                raise ValueError(
                    f"{self.env_family}: unknown {label}={value!r}; "
                    f"expected one of {list(allowed)}"
                )
        if not self.core_forms:
            raise ValueError(
                f"{self.env_family}: core_forms must declare at least one of "
                f"{list(CORE_FORMS)}"
            )
        unknown = sorted(set(self.core_forms) - set(CORE_FORMS))
        if unknown:
            raise ValueError(
                f"{self.env_family}: unknown core_form(s) {unknown}; "
                f"expected a subset of {list(CORE_FORMS)}"
            )

    @property
    def per_step_obs_available(self) -> bool:
        """Whether this family returns observations per step.

        Returns:
            ``True`` when ``chunk_obs_layout == "per_step"``.
        """
        return self.chunk_obs_layout == "per_step"

    @property
    def supports_coalescing(self) -> bool:
        """Whether this family can build a vectorized-form pool (i.e.
        whether it can truly be batched).

        Returns:
            ``True`` when ``core_forms`` contains ``lockstep_vector``.
        """
        return LOCKSTEP_VECTOR_FORM in self.core_forms

    def require_core_form(self, core_form: str) -> str:
        """Validate that a given form is declared as supported by this
        family.

        Args:
            core_form: The requested form.

        Returns:
            The same form name, unchanged.

        Raises:
            ValueError: The form is not in ``CORE_FORMS``, or this family
                does not declare it.
        """
        if core_form not in CORE_FORMS:
            raise ValueError(
                f"{self.env_family}: unknown core_form={core_form!r}; "
                f"expected one of {list(CORE_FORMS)}"
            )
        if core_form not in self.core_forms:
            raise ValueError(
                f"{self.env_family}: core_form={core_form!r} is not declared by this "
                f"family; declared forms are {sorted(self.core_forms)}"
            )
        return core_form

    @property
    def needs_accelerator(self) -> bool:
        """Whether this family's execution core needs an accelerator.

        Returns:
            The value of ``needs_accelerator_override`` when set;
            otherwise inferred from ``device_kind == "gpu_batched"``
            (libero is a CPU subprocess and **must not** be assigned a
            GPU; families like RoboCasa, which need GPU rendering but do
            not perform batched tensor computation, use the override to
            explicitly declare ``True``).
        """
        if self.needs_accelerator_override is not None:
            return self.needs_accelerator_override
        return self.device_kind == "gpu_batched"


def normalize_chunk_outcome(
    *,
    behavior: EnvFamilyBehavior,
    final_observation: Observation,
    step_observations: Sequence[Observation] | None,
    rewards: Sequence[float],
    terminations: Sequence[bool],
    truncations: Sequence[bool],
    requested_horizon: int,
    per_step_info: Sequence[dict[str, Any]] | None = None,
    include_step_observations: bool = False,
    info: dict[str, Any] | None = None,
) -> ChunkOutcome:
    """Normalize a family's raw ``chunk_step`` output into a
    ``ChunkOutcome`` (the sole exit point).

    Both ``obs_list`` forms come out of here:

    - ``per_step`` (libero / maniskill / realworld): ``step_observations``
      has the same length as ``rewards``;
    - ``final_only`` (robotwin): ``step_observations`` must be ``None``,
      ``per_step`` is also ``None`` with
      ``per_step_obs_available=False``, so upstream code never mistakenly
      assumes intermediate frames are available.

    ``executed_horizon`` is always computed from the **actually executed**
    step count (``len(rewards)``), not the requested chunk length: libero
    can terminate partway through a chunk and stop early.

    ``include_step_observations`` defaults to off: a real libero single
    frame is a 256x256x3 dual-camera PNG, and stuffing 8 steps' worth of
    intermediate frames into a ``StepResult`` would consistently exceed the
    256 KiB inline threshold. It corresponds to the legacy
    ``chunk_step(return_all_frames=...)`` semantics, and must be turned on
    explicitly when per-frame data is needed.

    Args:
        behavior: The family declaration.
        final_observation: The final observation after the chunk executes
            (always present).
        step_observations: Per-step observations; must be ``None`` for
            ``final_only`` families.
        rewards: Per-step rewards, with length equal to the number of
            steps actually executed.
        terminations: Per-step termination flags, same length as
            ``rewards``.
        truncations: Per-step truncation flags, same length as ``rewards``.
        requested_horizon: The requested chunk length, only recorded into
            ``info``.
        per_step_info: Per-step family-private information, same length
            as ``rewards`` or ``None``.
        include_step_observations: Whether to place per-step observations
            into ``PerStepRecord``.
        info: Chunk-level family-private information.

    Returns:
        The normalized result.

    Raises:
        ValueError: The per-step sequence lengths disagree, or
            ``step_observations`` contradicts the family's declared
            layout.
    """
    executed = len(rewards)
    if len(terminations) != executed or len(truncations) != executed:
        raise ValueError(
            f"{behavior.env_family}: per-step lengths disagree "
            f"(rewards={executed}, terminations={len(terminations)}, "
            f"truncations={len(truncations)})"
        )
    if per_step_info is not None and len(per_step_info) != executed:
        raise ValueError(
            f"{behavior.env_family}: per_step_info has {len(per_step_info)} entries "
            f"for {executed} executed step(s)"
        )
    per_step_available = behavior.per_step_obs_available
    if per_step_available and step_observations is None:
        raise ValueError(
            f"{behavior.env_family}: declares chunk_obs_layout='per_step' but the "
            "adapter passed no step_observations"
        )
    if not per_step_available and step_observations is not None:
        raise ValueError(
            f"{behavior.env_family}: declares chunk_obs_layout='final_only' but the "
            f"adapter passed {len(step_observations)} step observation(s); "
            "robotwin-style families submit the whole chunk and only see the last frame"
        )
    if per_step_available and len(step_observations or ()) != executed:
        raise ValueError(
            f"{behavior.env_family}: declares chunk_obs_layout='per_step' but returned "
            f"{len(step_observations or ())} observation(s) for {executed} "
            "executed step(s)"
        )

    per_step: list[PerStepRecord] | None = None
    if per_step_available:
        frames = list(step_observations or ())
        per_step = [
            PerStepRecord(
                step_index=frames[index].step_index,
                reward=float(rewards[index]),
                terminated=bool(terminations[index]),
                truncated=bool(truncations[index]),
                observation=frames[index] if include_step_observations else None,
                info=dict(per_step_info[index]) if per_step_info is not None else {},
            )
            for index in range(executed)
        ]
    return ChunkOutcome(
        observation=final_observation,
        reward=float(sum(float(item) for item in rewards)),
        terminated=any(bool(item) for item in terminations),
        truncated=any(bool(item) for item in truncations),
        executed_horizon=executed,
        per_step=per_step,
        per_step_obs_available=per_step_available,
        info={
            **(info or {}),
            "env_family": behavior.env_family,
            "requested_horizon": int(requested_horizon),
            "chunk_obs_layout": behavior.chunk_obs_layout,
            "step_observations_included": bool(
                per_step_available and include_step_observations
            ),
        },
    )


class EnvExecutionCore(Protocol):
    """The environment execution core: completely independent of transport /
    session.

    Implementations are provided per ``env_family`` by
    ``core.env_registry.ENV_FAMILY_REGISTRY``.

    **Must be blocking and synchronous**: ``RuntimeEnvWorker`` always calls
    it via ``asyncio.to_thread``, which is the very foundation of "an env
    step, once started, cannot be cancelled". By contrast,
    ``PolicyInferenceCore`` prefers to offer ``ainfer_batch`` (where waiting
    for inference can be stopped).
    """

    @property
    def behavior(self) -> EnvFamilyBehavior:
        """The declaration of the family this core belongs to, across the
        six divergence axes.

        Returns:
            The family declaration.
        """
        ...

    @property
    def core_form(self) -> str:
        """The form of this core instance (one of ``CORE_FORMS``).

        Before ``build`` this is the value requested by ``env_config`` (or
        the family default); after ``build`` it is the actual form.
        ``per_slot`` slots are independent of one another; ``lockstep_vector``
        slots are lanes of the same vectorized env, so ``chunk_step`` can be
        composed into a single call by ``SlotGroupCoalescer``.

        Returns:
            The form name.
        """
        ...

    def build(
        self,
        env_spec: EnvSpecMsg,
        *,
        num_envs: int,
        seed_offset: int,
        total_num_processes: int,
    ) -> None:
        """Construct the env pool according to the spec.

        ``num_envs`` is the initial slot count at construction time. In
        the ``per_slot`` form, the pool can subsequently grow or shrink via
        ``add_slot`` / ``remove_slot`` (an optional capability, see below);
        in the ``lockstep_vector`` form this number stays fixed, because
        the whole pool shares a single vectorized env instance and
        changing the lane count at runtime is equivalent to rebuilding the
        entire simulated scene.

        Args:
            env_spec: The environment specification.
            num_envs: The initial slot count in the pool.
            seed_offset: The seed offset for this rank.
            total_num_processes: The total number of processes
                participating in the split.
        """
        ...

    def reset(self, slots: Sequence[int], reset_spec: ResetSpec) -> list[Observation]:
        """Reset the specified slots.

        Args:
            slots: The slot indices.
            reset_spec: The episode initialization parameters.

        Returns:
            The initial observations, in the same order as ``slots``.
        """
        ...

    def chunk_step(
        self, slots: Sequence[int], chunk_actions: Sequence[np.ndarray]
    ) -> list[ChunkOutcome]:
        """Execute an action chunk on the specified slots.

        Args:
            slots: The slot indices.
            chunk_actions: Each slot's ``[chunk, action_dim]`` actions.

        Returns:
            The normalized results, in the same order as ``slots``.
        """
        ...

    def observe(self, slots: Sequence[int]) -> list[Observation]:
        """Read the cached observation without changing environment state.

        Args:
            slots: The slot indices.

        Returns:
            The observations, in the same order as ``slots``.
        """
        ...

    def extension(
        self, slot: int, namespace: str, method: str, args: dict[str, Any]
    ) -> dict[str, Any]:
        """Execute a family-specific extension.

        Args:
            slot: The slot index.
            namespace: The extension namespace.
            method: The extension method name.
            args: The method arguments.

        Returns:
            A structured result.
        """
        ...

    def close(self) -> None:
        """Release all environment resources."""
        ...


class DynamicSlotPool(Protocol):
    """An **optional** capability only the ``per_slot`` form can provide:
    appending / closing an independent slot at runtime.

    Analogous in spirit to ``LaneStatusReader``: ``EnvPool`` probes this
    method with ``getattr``, and falls back to the fixed-pool semantics
    (a full pool returns ``QUOTA_EXCEEDED`` directly and never shrinks
    proactively) when the method is not present.

    Why this is only offered for ``per_slot``: in this form, each slot is
    a fully independent family env instance (e.g. libero's
    ``LiberoEnv(num_envs=1)``), sharing no state with each other, so
    appending or closing one slot doesn't affect the rest. In the
    ``lockstep_vector`` form, all slots are lanes of **the same**
    vectorized env; appending/removing a lane is equivalent to
    reconstructing the entire scene with a different ``num_envs`` (GPU
    tensor buffers are allocated by batch size at construction time), so
    this form does not provide this protocol -- when ``EnvPool`` can't
    detect it, the pool stays fixed and scale-up requests still get
    ``QUOTA_EXCEEDED`` as before.
    """

    def add_slot(self, seed_offset: int) -> int:
        """Append an independent slot and return its index.

        The implementation must ensure the new slot is already usable via
        ``reset`` / ``chunk_step`` / ``observe`` before returning; the
        index must equal the total slot count before the append (appended
        at the end, never reusing a gap in the middle).

        Args:
            seed_offset: The seed offset for the new slot; the caller is
                responsible for ensuring no seed collision across slots.

        Returns:
            The index of the new slot.

        Raises:
            RuntimeApiError: The family construction failed
                (``ENV_FAILURE``).
        """
        ...

    def remove_slot(self, slot_index: int) -> None:
        """Close and remove a slot.

        Only the **current last** slot (``slot_index == existing slot
        count - 1``) that is not currently occupied may be removed:
        removing a middle index would misalign subsequent indices and
        break the correspondence between ``EnvPool.free_slots`` and
        ``SessionSlot.slot_index``. The caller (``EnvPool.shrink_idle``)
        guarantees this method is only called when a contiguous run of
        idle slots exists at the end.

        Args:
            slot_index: The index of the slot to remove; must be the
                current last index.

        Raises:
            RuntimeApiError: The index is not the current last one, or the
                slot is still occupied (``INVALID_ARGUMENT``).
        """
        ...

    def slot_count(self) -> int:
        """Return the current total slot count (which may already reflect
        dynamic scaling).

        Returns:
            The total slot count.
        """
        ...


class LaneStatusReader(Protocol):
    """An **optional** capability only cores in the ``lockstep_vector``
    form need to provide (the readback channel for masking semantics 2).

    Analogous in spirit to ``transport.base.LateResultSink``:
    ``RuntimeEnvWorker`` probes this method with ``getattr``, so a
    ``per_slot``-only core (robocasa) is unaffected by not implementing it.

    Why it's needed: a ``lockstep_vector`` pool advances **all** lanes on
    every ``chunk_step`` call, with an absent lane propped up by a hold
    action (semantics 2), but the core only returns a ``ChunkOutcome`` to
    the participants. As a result, the worker-side ``SessionSlot``'s
    ``terminated`` / ``truncated`` / ``step_index`` can fall behind the
    core, leading to inference on a stale frame, or a masked termination's
    success being mistaken for "still running". ``observe`` can recover
    the observation, but not the termination flag -- which is exactly why
    this method exists.

    Implementations should always use
    ``backends.rlinf_family.lane_statuses``; individual families should
    not reimplement it.
    """

    def lane_status(self, slots: Sequence[int]) -> list[Any]:
        """Read a lifecycle snapshot of several lanes without changing
        environment state.

        Must be cheap enough to call before every ``policy_step``: it only
        reads existing lane-status fields, without re-rendering or
        re-encoding images.

        Args:
            slots: The slot indices.

        Returns:
            ``rlinf_family.LaneStatus`` entries, in the same order as
            ``slots``.
        """
        ...
