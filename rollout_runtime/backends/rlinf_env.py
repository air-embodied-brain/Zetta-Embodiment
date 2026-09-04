"""The libero family's ``EnvExecutionCore``.

**Does not reimplement rlinf's env**: ``rlinf.envs.libero.libero_env.LiberoEnv``
is reused as-is, and ``envs.action_utils.prepare_actions`` /
``LiberoEnv._wrap_obs`` (internally routed through ``envs.utils.to_tensor`` and
``EnvOutput.prepare_observations``'s 5-key schema) are also reused as-is. This
module does exactly three things: translate Runtime's session/slot semantics
into family calls, normalize the family's output through
``core.env_execution.normalize_chunk_outcome``, and wire the four LIBERO
privileged methods into ``extension_call``.

Two deliberate departures from rlinf's default usage:

1. **Defaults to one ``LiberoEnv`` per slot (each with ``num_envs=1``)**,
   rather than a single ``LiberoEnv(num_envs=pool_size)``. The reason is that
   ``LiberoEnv.step`` calls ``self.env.step(actions)`` **without an ``id``**,
   i.e. one step advances every env in the pool at once, while Runtime's slots
   are mutually independent sessions. **M6 addition**: a vectorized single
   ``LiberoEnv`` is now also supported, via
   ``env_config.core_form="lockstep_vector"``, which is exactly the form that
   ``SlotGroupCoalescer`` truly coalesces (one ``chunk_step`` for the same pool
   and tick), at the cost of requiring all lanes in the pool to lockstep, with
   absent lanes carried forward by a hold action. The coexistence of the two
   forms and the masking semantics are described where the coalescer is
   implemented.
2. **A chunk is driven step by step via ``LiberoEnv.step``, stopping at the
   first termination signal**, rather than calling ``LiberoEnv.chunk_step``.
   The latter has no early stop, and LIBERO-PRO raises if stepped again after
   task termination (the legacy ``robots/libero/env_server.py:240`` also
   manually stepped for this exact reason). ``step`` itself is an rlinf method,
   so ``_wrap_obs`` / ``_record_metrics`` / ``_calc_step_reward`` all run as
   normal, and ``executed_horizon`` is therefore the **actual** number of
   steps.

Dependency surface: Zetta's built-in LIBERO environment + torch + numpy, with
the simulator dependency kept as a lazy import inside functions.
"""

from __future__ import annotations

import dataclasses
import os
import threading
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from rollout_runtime.api.enums import ErrorCode
from rollout_runtime.api.errors import RuntimeApiError, make_error
from rollout_runtime.api.ids import EpisodeId, SessionId
from rollout_runtime.api.messages import (
    EnvFamilyCapability,
    EnvSpecMsg,
    Observation,
    ResetSpec,
)
from rollout_runtime.backends import libero_privileged
from rollout_runtime.backends.libero_critic import (
    TemporalCritic,
    canonical_rules_fingerprint,
    critic_rules_from_payload,
    extract_libero_critic_features,
)
from rollout_runtime.backends.rlinf_family import (
    LaneStatus,
    lane_statuses,
    run_lockstep_chunk,
)
from rollout_runtime.core import payload as payload_module
from rollout_runtime.core.env_execution import (
    LOCKSTEP_VECTOR_FORM,
    PER_SLOT_FORM,
    ChunkOutcome,
    EnvFamilyBehavior,
    normalize_chunk_outcome,
)
from rollout_runtime.core.env_registry import (
    LIBERO_ENV_FAMILY,
    LIBERO_EXTENSIONS,
    behavior_for,
    capability_from_behavior,
    register_env_family,
    requested_core_form,
)

__all__ = [
    "LIBERO_ENV_FAMILY",
    "LIBERO_EXTENSIONS",
    "LIBERO_NAMESPACE",
    "LIBERO_WARMUP_STEPS",
    "LiberoEnvConfig",
    "LiberoEnvCore",
    "LiberoEnvFamily",
    "libero_env_capability",
    "register_libero_env_family",
]

LIBERO_NAMESPACE = "libero"
"""The namespace for ``extension_call``."""

LIBERO_WARMUP_STEPS = 15
"""The zero-action warm-up step count hardcoded in ``LiberoEnv.reset``
(``libero_env.py``'s ``range(15)``).

It is not configurable. ``env_config.num_steps_wait`` only **declares** this
fact; supplying any other value is rejected, so an experiment manifest never
freezes a parameter that has no effect. Parity investigations must also align
on this.
"""

_ALIASES = {
    "suite": "task_suite_name",
    "episode_length": "max_episode_steps",
    "image_height": "camera_height",
    "image_width": "camera_width",
}
"""Compatibility aliases for ``env_config``: an earlier ``a100_libero.yaml``
used the names on the left."""


@dataclasses.dataclass(kw_only=True)
class LiberoEnvConfig:
    """The libero family's private config (corresponds to
    ``EnvSpecMsg.env_config``).

    Defaults deliberately match the legacy
    ``robots/libero/env_server.py::build_env_cfg`` (``is_eval``,
    ``use_ordered_reset_state_ids``, ``camera_depths``, the ``horizon``
    margin, ``ignore_done``), so parity checks don't need to first explain a
    pile of config differences.

    Attributes:
        task_suite_name: The LIBERO benchmark name (e.g. ``"libero_10"``).
        task_id: The default task index; overridable by ``ResetSpec.task_id``.
        max_episode_steps: The outer truncation step count.
        camera_height: Camera height (fed into
            ``init_params.camera_heights``).
        camera_width: Camera width.
        camera_depths: Whether to also render depth (needed for
            back-projection).
        robots: List of robot models; ``None`` uses the LIBERO default.
        libero_variant: ``"standard"`` / ``"pro"`` / ``"plus"``, mapped onto
            ``LIBERO_TYPE``.
        perturbation_suffix: The LIBERO-PRO / PLUS perturbation subset.
        reset_gripper_open: Whether to open the gripper during warm-up.
        ignore_terminations: Whether to ignore the environment's termination
            signal.
        auto_reset: Whether to let the family auto-reset itself; Runtime
            drives resets via sessions, so this is fixed to false.
        is_eval: rlinf's eval mode (reset does not rebuild the subprocess when
            the task is unchanged, which is much faster).
        use_rel_reward: Whether to use relative reward.
        use_step_penalty: Whether to add a per-step penalty.
        reward_coef: Termination reward coefficient.
        seed: The family's base seed; the actual seed still adds
            ``seed_offset``.
        robosuite_horizon_margin: The extra margin robosuite's internal
            horizon keeps beyond the outer truncation.
        ignore_done: Whether to prevent robosuite from self-latching the
            episode.
        action_dim: Action dimension.
        chunk_size: The declared action-chunk length (actual execution follows
            the number of rows in the supplied actions).
        num_steps_wait: Must equal ``LIBERO_WARMUP_STEPS``.
        action_model_type: The model type that produced the actions, fed into
            ``prepare_actions``.
        return_all_frames: Whether ``per_step`` includes per-step observations
            (the payload is large, so this is off by default).
        save_video: Whether to have the family write video.
        video_base_dir: The family's video directory.
        core_form: The execution core form
            (``core.env_execution.CORE_FORMS``). ``lockstep_vector`` only
            supports ``libero_variant="standard"``, see ``build``'s
            explanation.
        assets_root: Override for the robosuite assets root directory;
            ``None`` means no override.
    """

    task_suite_name: str = "libero_10"
    task_id: int = 0
    max_episode_steps: int = 512
    camera_height: int = 256
    camera_width: int = 256
    camera_depths: bool = True
    robots: list[str] | None = None
    libero_variant: str = "standard"
    perturbation_suffix: str | None = None
    reset_gripper_open: bool = True
    ignore_terminations: bool = False
    auto_reset: bool = False
    is_eval: bool = True
    use_rel_reward: bool = False
    use_step_penalty: bool = False
    reward_coef: float = 1.0
    seed: int = 0
    robosuite_horizon_margin: int = 1000
    ignore_done: bool = True
    action_dim: int = 7
    chunk_size: int = 8
    num_steps_wait: int = LIBERO_WARMUP_STEPS
    action_model_type: str = "openpi"
    return_all_frames: bool = False
    save_video: bool = False
    video_base_dir: str = ""
    core_form: str = PER_SLOT_FORM

    assets_root: str | None = None

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any] | None) -> LiberoEnvConfig:
        """Construct a config from an ``env_config`` dict.

        Unknown keys are always rejected rather than silently ignored:
        ``env_config`` feeds into ``EnvSpecMsg.digest()``, and a typo'd key
        would silently create a new pool (for real libero, that means one more
        subprocess), which is far harder to diagnose than an error.

        Args:
            config: The family-private config; ``None`` means all defaults.

        Returns:
            The structured config.

        Raises:
            RuntimeApiError: An unknown key is present, or
                ``num_steps_wait`` / ``libero_variant`` is set to a value that
                can never take effect (``INVALID_ARGUMENT``).
        """
        if not config:
            return cls()
        known = {field.name for field in dataclasses.fields(cls)}
        normalized: dict[str, Any] = {}
        unknown: list[str] = []
        for key, value in config.items():
            target = _ALIASES.get(key, key)
            if target not in known:
                unknown.append(key)
                continue
            normalized[target] = value
        if unknown:
            raise RuntimeApiError(
                make_error(
                    ErrorCode.INVALID_ARGUMENT,
                    f"unknown libero env config keys: {sorted(unknown)}",
                    unknown_keys=sorted(unknown),
                    known_keys=sorted(known),
                    aliases=dict(_ALIASES),
                )
            )
        if normalized.get("robots") is not None:
            normalized["robots"] = list(normalized["robots"])
        instance = cls(**normalized)
        if instance.num_steps_wait != LIBERO_WARMUP_STEPS:
            raise RuntimeApiError(
                make_error(
                    ErrorCode.INVALID_ARGUMENT,
                    f"num_steps_wait must be {LIBERO_WARMUP_STEPS}: the warm-up loop is "
                    "hardcoded in rlinf LiberoEnv.reset and the submodule is read-only",
                    requested=instance.num_steps_wait,
                    hardcoded=LIBERO_WARMUP_STEPS,
                )
            )
        if instance.libero_variant not in ("standard", "pro", "plus"):
            raise RuntimeApiError(
                make_error(
                    ErrorCode.INVALID_ARGUMENT,
                    f"unknown libero_variant {instance.libero_variant!r}; "
                    "expected 'standard' | 'pro' | 'plus'",
                )
            )
        return instance

    def to_rlinf_cfg(self) -> Any:
        """Project into the omegaconf config needed by ``LiberoEnv(cfg=...)``.

        Returns:
            A ``DictConfig``.

        Raises:
            RuntimeApiError: ``omegaconf`` is unavailable (``ENV_FAILURE``).
        """
        try:
            from omegaconf import OmegaConf
        except ImportError as exc:  # pragma: no cover - always installed in production
            raise RuntimeApiError(
                make_error(
                    ErrorCode.ENV_FAILURE,
                    "omegaconf is required to build the libero env cfg",
                )
            ) from exc
        init_params: dict[str, Any] = {
            "camera_heights": int(self.camera_height),
            "camera_widths": int(self.camera_width),
            "camera_depths": bool(self.camera_depths),
            # rlinf owns the outer truncation; robosuite's own horizon must
            # stay beyond that boundary, otherwise once LIBERO-PRO overwrites
            # horizon-done with task success, the next action would raise
            # immediately.
            "horizon": int(self.max_episode_steps + self.robosuite_horizon_margin),
            "ignore_done": bool(self.ignore_done),
        }
        if self.robots:
            init_params["robots"] = list(self.robots)
        return OmegaConf.create(
            {
                "env_type": LIBERO_ENV_FAMILY,
                "task_suite_name": self.task_suite_name,
                "auto_reset": bool(self.auto_reset),
                "ignore_terminations": bool(self.ignore_terminations),
                "max_steps_per_rollout_epoch": int(self.max_episode_steps),
                "max_episode_steps": int(self.max_episode_steps),
                "use_rel_reward": bool(self.use_rel_reward),
                "use_step_penalty": bool(self.use_step_penalty),
                "reward_coef": float(self.reward_coef),
                "reset_gripper_open": bool(self.reset_gripper_open),
                "is_eval": bool(self.is_eval),
                "seed": int(self.seed),
                "group_size": 1,
                "use_fixed_reset_state_ids": True,
                "use_ordered_reset_state_ids": True,
                # Runtime always supplies reset_state_ids explicitly on every
                # reset, so specific_reset_id is left unset.
                "specific_reset_id": None,
                "libero_variant": self.libero_variant,
                "perturbation_suffix": self.perturbation_suffix,
                "video_cfg": {
                    "save_video": bool(self.save_video),
                    "info_on_video": bool(self.save_video),
                    "video_base_dir": self.video_base_dir or "/tmp/rr_libero_video",
                },
                "init_params": init_params,
            }
        )


def _libero_env_class() -> type:
    """Lazily construct a Runtime subclass of ``LiberoEnv``.

    The only change is ``get_env_fns``: wraps the factory so that the env in
    every subprocess carries an ``rr_extension_call`` and ``render``
    forwarding hook (needed for ``privileged_contacts``). The class definition
    lives inside the function to avoid loading LIBERO/MuJoCo at module import
    time.

    Returns:
        A subclass of ``LiberoEnv``.
    """
    from zetta.envs.libero.environment import LiberoEnv

    class RuntimeLiberoEnv(LiberoEnv):  # type: ignore[misc, valid-type]
        """A ``LiberoEnv`` with the Runtime privileged extensions."""

        def get_env_fns(self) -> list[Any]:
            """Return the list of env factories wrapped with extensions.

            Returns:
                The factory list, in the same order as the parent class.
            """
            return libero_privileged.wrap_env_factories(super().get_env_fns())

    return RuntimeLiberoEnv


def _to_numpy(value: Any) -> np.ndarray:
    """Normalize a torch tensor / numpy array / sequence into a numpy array.

    Args:
        value: The tensor or array returned by the family.

    Returns:
        A numpy array (torch goes through ``detach().cpu()``).
    """
    detach = getattr(value, "detach", None)
    if callable(detach):
        return detach().cpu().numpy()
    return np.asarray(value)


def _scalar(value: Any) -> float:
    """Extract the scalar from a ``[1]``-shaped tensor.

    Args:
        value: A scalar, or a length-1 tensor / array.

    Returns:
        A float value.
    """
    array = _to_numpy(value).reshape(-1)
    return float(array[0]) if array.size else 0.0


@dataclasses.dataclass
class _LiberoSlot:
    """The driver-side state for one slot.

    In ``per_slot`` form, each slot exclusively owns a
    ``LiberoEnv(num_envs=1)`` and has ``lane_index=0``; in
    ``lockstep_vector`` form, all slots share the same
    ``LiberoEnv(num_envs=pool_size)``, and ``lane_index`` is its lane within
    that vector env.

    Attributes:
        env: The ``LiberoEnv`` this slot uses (in vector form, all slots are
            the same object).
        lane_index: This slot's lane index within ``env``.
        process_offset: The absolute ``LiberoEnv(seed_offset=...)`` offset
            exclusively owned by this slot, in the range
            ``[0, total_num_processes)`` (only meaningful in ``per_slot``
            form). Upstream, ``LiberoEnv.__init__`` uses it to index the
            evaluation reset-state pool split by ``total_num_processes``
            (``reset_state_ids_all[seed_offset]``), so it **cannot** be an
            arbitrary non-conflicting number — it must be strictly less than
            the ``total_num_processes`` declared at construction time, and no
            two live slots may occupy the same offset at once. ``add_slot`` /
            ``remove_slot`` use this field to reclaim and reallocate offset
            slots (dynamic slot resizing).
        started: Whether it has already been reset.
        step_index: Number of env steps already executed in the current
            episode.
        terminated: Termination flag.
        truncated: Truncation flag.
        frozen: The episode has ended and has not been reset again (used by
            the vector form's early stop within a group to avoid re-triggering
            on it).
        masked_steps: Number of steps carried forward by a hold action (the
            vector form's masking semantics).
        reset_state_id: The reset state id for the current episode.
        task_id: The current task index.
        trial_id: The current trial index.
        instruction: The current task instruction.
        last_observation: The most recent observation frame (``observe`` only
            reads it).
        last_main_image: The most recent main-view image
            (``libero.cached_image`` reads it).
        chunk_calls: Number of ``chunk_step`` calls.
        env_steps: Cumulative number of env steps (not reset across
            episodes).
        resets: Number of resets.
        critic: The ``TemporalCritic`` instance persisted across chunks;
            ``None`` means this episode did not request Critic evaluation via
            ``ResetSpec.options["critic_rules"]``. Consistent with the legacy
            ``LiberoEnvClient._critic`` lifecycle: an episode only (re)builds
            it once at reset, and the rule set is immutable within the same
            episode (the semantics of ``_configure_critic``, see
            ``_after_reset``).
        critic_rule_fingerprint: A fingerprint of the ``critic_rules`` payload
            in effect for this episode; used to reject misuse where "the rule
            set changed within the same episode" (matching the legacy
            ``_critic_fingerprint``).
        critic_interrupt_on_proposal: Whether to break out of the chunk's
            physical action loop early upon hitting a proposal (matching the
            legacy ``critic_chunk_step``'s ``interrupt_on_proposal``).
        critic_previous_eef: The EEF position (3D) from the previous physical
            action step, used by ``extract_libero_critic_features`` to compute
            genuine displacement. It is seeded from the reset observation so
            realization features are available on the first physical step.
        critic_history_pending_reset: Whether the next read of the
            subprocess's ``critic_state`` extension should request clearing
            cross-call history (``ever_grasped``, etc.). Set true on every
            reset and set false after the first successful read — an explicit
            flag is used instead of checking ``env_steps == 0``, because
            ``env_steps`` accumulates across episodes and is not reset, so it
            cannot identify "this episode's first Critic state collection".
        audit_trace: The per-step audit record accumulated across chunks
            (matching the legacy ``_audit_trace``, read by the
            ``libero.audit_trace`` extension method).
    """

    env: Any
    lane_index: int = 0
    process_offset: int = 0
    started: bool = False
    step_index: int = 0
    terminated: bool = False
    truncated: bool = False
    frozen: bool = False
    masked_steps: int = 0
    reset_state_id: int = -1
    task_id: int = -1
    trial_id: int = -1
    instruction: str = ""
    last_observation: Observation | None = None
    last_main_image: np.ndarray | None = None
    chunk_calls: int = 0
    env_steps: int = 0
    resets: int = 0
    critic: Any = None
    critic_rule_fingerprint: str | None = None
    critic_interrupt_on_proposal: bool = True
    critic_previous_eef: np.ndarray | None = None
    critic_history_pending_reset: bool = True
    audit_trace: list[dict[str, Any]] = dataclasses.field(default_factory=list)


class LiberoEnvCore:
    """The libero family's ``EnvExecutionCore`` (blocking/synchronous, driven
    by ``asyncio.to_thread``).

    Entirely transport/session agnostic: it only knows slot indices, and
    ``Observation``'s ``session_id`` / ``episode_id`` are left blank, stamped
    by the EnvWorker.

    In ``per_slot`` form, additionally implements
    ``core.env_execution.DynamicSlotPool`` (``add_slot`` / ``remove_slot`` /
    ``slot_count``): each slot is a fully independent
    ``LiberoEnv(num_envs=1)``, so a single slot can be appended or closed at
    runtime without affecting the rest. In ``lockstep_vector`` form,
    ``add_slot`` / ``remove_slot`` are simply rejected — in that form all
    slots share the same vectorized env, and there is no such thing as
    "growing or shrinking a single lane".
    """

    def __init__(self) -> None:
        """Initialize a not-yet-``build``-ed execution core."""
        self.config = LiberoEnvConfig()
        self.env_spec: EnvSpecMsg | None = None
        self.seed_offset = 0
        self.total_num_processes = 1
        self.closed = False
        self.total_chunk_calls = 0
        self.total_env_steps = 0
        self.total_masked_steps = 0
        self.coalesced_group_count = 0
        self._core_form = PER_SLOT_FORM
        # Vector form: whether the whole pool has already been fully
        # initialized once (the first reset must cover every lane).
        self._vector_initialized = False
        self._slots: list[_LiberoSlot] = []
        self._slot_mutation_lock = threading.Lock()
        self._envs: list[Any] = []

    @property
    def behavior(self) -> EnvFamilyBehavior:
        """The libero family's declaration on the six divergence points.

        Returns:
            The family declaration.
        """
        return behavior_for(LIBERO_ENV_FAMILY)

    @property
    def core_form(self) -> str:
        """This core instance's form (``per_slot`` or ``lockstep_vector``).

        Returns:
            The form name; defaults to ``per_slot`` before ``build``.
        """
        return self._core_form

    def _hold_action(self) -> np.ndarray:
        """The hold action used for absent lanes in vector form.

        Same shape and semantics as ``LiberoEnv.reset``'s zero-action warm-up:
        all-zero displacement, with the gripper dimension at ``-1`` (open)
        when ``reset_gripper_open`` is set.

        Returns:
            A ``[action_dim]`` float32 action.
        """
        hold = np.zeros(int(self.config.action_dim), dtype=np.float32)
        if self.config.reset_gripper_open:
            hold[-1] = -1.0
        return hold

    # -------------------------------------------------------------- Construction and release

    def build(
        self,
        env_spec: EnvSpecMsg,
        *,
        num_envs: int,
        seed_offset: int = 0,
        total_num_processes: int = 1,
    ) -> None:
        """Construct an env pool according to the spec; the form is selected
        by ``env_config.core_form``.

        - ``per_slot`` (default): ``num_envs`` mutually independent
          ``LiberoEnv(num_envs=1)`` instances, with each slot's seed offset
          being ``seed_offset * num_envs + slot``, guaranteeing no collision
          across ranks or slots;
        - ``lockstep_vector``: **one** ``LiberoEnv(num_envs=num_envs)``, with
          slots as its lanes, and the seed offset passed through to the
          family as-is (it internally uses ``seed_offset`` to pick its own
          reset state pool).

        Args:
            env_spec: The environment spec (``env_config.core_form`` selects
                the form).
            num_envs: The number of slots in the pool (baked in at
                construction time).
            seed_offset: The seed offset for this rank.
            total_num_processes: Total number of processes participating in
                the split.

        Raises:
            RuntimeApiError: ``num_envs`` is invalid, the form is not declared
                by the family, or the vector form is combined with a
                non-``standard`` libero variant (``INVALID_ARGUMENT``); the
                family failed to construct
                （``ENV_FAILURE``）。
        """
        if num_envs < 1:
            raise RuntimeApiError(
                make_error(
                    ErrorCode.INVALID_ARGUMENT, f"num_envs must be >= 1, got {num_envs}"
                )
            )
        config = LiberoEnvConfig.from_mapping(env_spec.env_config)
        core_form = requested_core_form(env_spec, self.behavior)
        if core_form == LOCKSTEP_VECTOR_FORM and config.libero_variant != "standard":
            # A vector form cannot stop only one lane, and LIBERO-PRO / PLUS
            # raise if stepped again after task termination. Better to reject
            # this explicitly at build time than to have a run fail halfway
            # through and turn a success into an ENV_FAILURE.
            raise RuntimeApiError(
                make_error(
                    ErrorCode.INVALID_ARGUMENT,
                    f"core_form={LOCKSTEP_VECTOR_FORM!r} is only supported for "
                    "libero_variant='standard': a vector env cannot stop a single lane "
                    "and LIBERO-PRO/PLUS raise when stepped after termination "
                    "(plan §7); use core_form='per_slot' for this variant",
                    libero_variant=config.libero_variant,
                    core_form=core_form,
                )
            )
        # LIBERO_TYPE is read at the **import time** of rlinf's libero module
        # (a module-level libero_type), so it must land in the process
        # environment before the first import.
        os.environ.setdefault("ROBOT_PLATFORM", "LIBERO")
        os.environ["LIBERO_TYPE"] = config.libero_variant
        if config.perturbation_suffix:
            os.environ.setdefault("LIBERO_PERTURBATION", config.perturbation_suffix)
        if config.assets_root:
            import robosuite.models

            robosuite.models.assets_root = config.assets_root

        env_class = _libero_env_class()
        cfg = config.to_rlinf_cfg()
        slots: list[_LiberoSlot] = []
        envs: list[Any] = []
        # The denominator reused by add_slot in per_slot form; lockstep_vector
        # form does not support dynamic growth (see add_slot's rejection
        # branch), so keeping the default value here just avoids leaving it
        # unassigned.
        total_processes = max(1, total_num_processes)
        try:
            if core_form == LOCKSTEP_VECTOR_FORM:
                # One vector env per pool: slots become its lanes.
                # ``LiberoEnv.step`` advances every lane at once, so
                # SlotGroupCoalescer can merge same-pool same-tick calls into
                # one chunk_step.
                env = env_class(
                    cfg=cfg,
                    num_envs=num_envs,
                    seed_offset=seed_offset,
                    total_num_processes=max(1, total_num_processes),
                    worker_info=None,
                )
                env.is_start = False
                envs.append(env)
                slots = [
                    _LiberoSlot(env=env, lane_index=lane) for lane in range(num_envs)
                ]
            else:
                # The denominator uses effective_max_pool_size rather than
                # num_envs: in per_slot form, LiberoEnv.__init__ uses
                # seed_offset to index an evaluation reset-state pool split by
                # total_num_processes (reset_state_ids_all[seed_offset]), and
                # this denominator determines "how many distinct offsets can
                # exist at most" — if declared only by the initial num_envs, a
                # later add_slot's desired new offset would go straight out of
                # bounds (reproduced on a real multi-GPU host with a real
                # LiberoEnv: IndexError: index 1 is out of bounds for axis 0
                # with size 1). Declaring the denominator by the growth ceiling
                # only causes reset_state_ids_all to be split into a few extra
                # pieces, without constructing any extra subprocess, even if
                # the extra offsets are not used right away.
                max_pool_size = env_spec.effective_max_pool_size()
                total_processes = max(1, total_num_processes) * max_pool_size
                for slot_index in range(num_envs):
                    process_offset = seed_offset * max_pool_size + slot_index
                    env = env_class(
                        cfg=cfg,
                        num_envs=1,
                        seed_offset=process_offset,
                        total_num_processes=total_processes,
                        worker_info=None,
                    )
                    # Runtime always supplies reset_state_ids explicitly on
                    # every reset, while LiberoEnv.reset's ``if self.is_start:``
                    # branch overwrites the ids passed in the first time.
                    # Turning off is_start up front makes the first episode
                    # follow the same path as later episodes (needed for
                    # parity).
                    env.is_start = False
                    envs.append(env)
                    slots.append(
                        _LiberoSlot(
                            env=env, lane_index=0, process_offset=process_offset
                        )
                    )
        except RuntimeApiError:
            for env in envs:
                _close_env(env)
            raise
        except BaseException as exc:
            for env in envs:
                _close_env(env)
            raise RuntimeApiError(
                make_error(
                    ErrorCode.ENV_FAILURE,
                    f"failed to build the libero env pool: {type(exc).__name__}: {exc}",
                    task_suite_name=config.task_suite_name,
                    num_envs=num_envs,
                    core_form=core_form,
                )
            ) from exc
        self.config = config
        self.env_spec = env_spec
        self.seed_offset = seed_offset
        # per_slot form: add_slot uses this as LiberoEnv's
        # total_num_processes, which must match the initial slots, otherwise
        # reset_state_ids_all's split would not line up and old/new offsets
        # would collide.
        self.total_num_processes = total_processes
        self._core_form = core_form
        self._vector_initialized = False
        self._slots = slots
        self._envs = envs
        self.closed = False

    def close(self) -> None:
        """Close every libero subprocess.

        In vector form all slots share the same env, so closing is done by
        iterating ``_envs`` rather than by slot, avoiding repeatedly calling
        ``close`` on the same env.
        """
        for env in self._envs:
            _close_env(env)
        self._slots = []
        self._envs = []
        self.closed = True

    # -------------------------------------------------------- Dynamic slot resizing

    def slot_count(self) -> int:
        """Return the current total number of slots
        (``core.env_execution.DynamicSlotPool``).

        Returns:
            The current slot count, including any slots dynamically appended
            after ``build``.
        """
        return len(self._slots)

    def add_slot(self, seed_offset: int) -> int:
        """Append an independent ``LiberoEnv(num_envs=1)`` as a new slot.

        Only meaningful in ``per_slot`` form: in that form each slot is
        already a fully independent family env sharing no state with others,
        so appending one does not affect existing slots. In
        ``lockstep_vector`` form all slots share the same vectorized env, and
        appending a lane would require rebuilding the whole scene — this
        method rejects it outright, causing ``EnvPool.dynamic`` to detect it
        as false and fall back to the fixed-pool semantics.

        Args:
            seed_offset: **Ignored**. The ``DynamicSlotPool`` protocol defines
                it as a caller-supplied hint (``EnvPool._next_seed_offset()``
                only ever offers generic hints such as "current slot count"),
                but libero's valid offset range is determined internally by
                ``LiberoEnv.__init__`` splitting ``reset_state_ids_all`` by
                ``total_num_processes`` (``reset_state_ids_all[seed_offset]``),
                and must be strictly less than the ``total_num_processes``
                declared at ``build`` time, and must not collide with any live
                slot's offset — a constraint only this core knows about, so
                the offset actually used is picked by
                ``_allocate_process_offset`` from a genuinely free value within
                ``[0, total_num_processes)``, without trusting the caller's
                hint (using the caller's hint directly was reproduced on a
                real multi-GPU host with a real LiberoEnv as
                ``IndexError: index 1 is out of bounds for axis 0 with size 1``
                — ``build`` had only declared ``total_num_processes`` based on
                the initial ``num_envs``, so any offset from later growth was
                bound to go out of range; after the fix, ``build`` declares
                the denominator via ``effective_max_pool_size()`` instead, so
                this method has free offset slots to allocate from).

        Returns:
            The new slot's index (equal to the total slot count before
            appending).

        Raises:
            RuntimeApiError: This core is in ``lockstep_vector`` form, no free
                process-offset slot is available (``INVALID_ARGUMENT``), or
                the family failed to construct (``ENV_FAILURE``).
        """
        del seed_offset  # See the docstring above: the caller's hint does not apply to libero, deliberately unused.
        if self._core_form == LOCKSTEP_VECTOR_FORM:
            raise RuntimeApiError(
                make_error(
                    ErrorCode.INVALID_ARGUMENT,
                    "cannot add a slot to a lockstep_vector pool: all lanes share one "
                    "vector env instance, growing it means rebuilding the whole scene "
                    "with a different num_envs, which is not a resource-management "
                    "operation; use core_form='per_slot' for dynamic pools",
                    core_form=self._core_form,
                )
            )
        with self._slot_mutation_lock:
            process_offset = self._allocate_process_offset()
            env_class = _libero_env_class()
            cfg = self.config.to_rlinf_cfg()
            new_index = len(self._slots)
            try:
                env = env_class(
                    cfg=cfg,
                    num_envs=1,
                    seed_offset=process_offset,
                    total_num_processes=self.total_num_processes,
                    worker_info=None,
                )
                env.is_start = False
            except BaseException as exc:
                raise RuntimeApiError(
                    make_error(
                        ErrorCode.ENV_FAILURE,
                        f"failed to add libero slot {new_index}: "
                        f"{type(exc).__name__}: {exc}",
                        slot_index=new_index,
                    )
                ) from exc
            self._envs.append(env)
            self._slots.append(
                _LiberoSlot(env=env, lane_index=0, process_offset=process_offset)
            )
            return new_index

    def _allocate_process_offset(self) -> int:
        """Pick an offset within ``[0, total_num_processes)`` not occupied by
        any live slot.

        Returns:
            A free process offset.

        Raises:
            RuntimeApiError: No free offset is available
                (``INVALID_ARGUMENT``) — in theory this should never happen:
                ``EnvPool`` only calls ``add_slot`` when
                ``pool_size < effective_max_pool_size()``, and
                ``total_num_processes`` is declared based on exactly that
                value; if it does happen, the two are out of sync, which is
                worth treating as a configuration error rather than silently
                retrying.
        """
        used = {slot.process_offset for slot in self._slots}
        for candidate in range(self.total_num_processes):
            if candidate not in used:
                return candidate
        raise RuntimeApiError(
            make_error(
                ErrorCode.INVALID_ARGUMENT,
                f"no free process offset in [0, {self.total_num_processes}): "
                f"{len(self._slots)} slot(s) already occupy all of them; this means "
                "build() was not given a total_num_processes wide enough for "
                "max_dynamic_pool_size",
                total_num_processes=self.total_num_processes,
                occupied=sorted(used),
            )
        )

    def remove_slot(self, slot_index: int) -> None:
        """Close and remove the trailing independent slot.

        Args:
            slot_index: The slot index to remove; must equal the current
                trailing index (``slot_count() - 1``).

        Raises:
            RuntimeApiError: The index is not the current trailing index
                (``INVALID_ARGUMENT``), or this core is in
                ``lockstep_vector`` form (same as above, a vector pool has no
                such thing as "removing a single lane").
        """
        if self._core_form == LOCKSTEP_VECTOR_FORM:
            raise RuntimeApiError(
                make_error(
                    ErrorCode.INVALID_ARGUMENT,
                    "cannot remove a slot from a lockstep_vector pool: all lanes share "
                    "one vector env instance",
                    core_form=self._core_form,
                )
            )
        with self._slot_mutation_lock:
            last_index = len(self._slots) - 1
            if slot_index != last_index:
                raise RuntimeApiError(
                    make_error(
                        ErrorCode.INVALID_ARGUMENT,
                        f"can only remove the trailing slot (expected {last_index}, "
                        f"got {slot_index}): removing a middle slot would shift every "
                        "later slot's index",
                        requested_slot=slot_index,
                        trailing_slot=last_index,
                    )
                )
            env = self._envs.pop(last_index)
            self._slots.pop(last_index)
        _close_env(env)

    # -------------------------------------------------------------- Operations

    def reset(self, slots: Sequence[int], reset_spec: ResetSpec) -> list[Observation]:
        """Reset the given slots (the family signature is
        ``reset(env_idx, reset_state_ids)``).

        Both forms use the same family signature, differing only in
        ``env_idx``: in ``per_slot`` form, each env resets its own single
        ``[0]``; in ``lockstep_vector`` form, the requested lanes are handed
        to the same vector env in one call (``LiberoEnv.reset`` already
        supports an ``env_idx`` subset — its internal zero-action warm-up is
        itself ``self.env.step(zero_actions, env_idx)``).

        Args:
            slots: The slot indices.
            reset_spec: Episode initialization parameters (``task_id`` /
                ``seed`` / ``reset_state_id``).

        Returns:
            Initial observations, in the same order as ``slots``.
        """
        if self._core_form == LOCKSTEP_VECTOR_FORM:
            return self._reset_lockstep(slots, reset_spec)
        observations: list[Observation] = []
        for slot_index in slots:
            slot = self._require_slot(slot_index)
            reset_state_id = self._reset_state_id(slot, reset_spec)
            obs, _info = slot.env.reset(
                env_idx=np.array([0]),
                reset_state_ids=np.array([reset_state_id]),
            )
            self._after_reset(slot_index, reset_state_id, reset_spec, obs)
            observations.append(self._observation(slot_index, obs))
        return observations

    def _reset_lockstep(
        self, slots: Sequence[int], reset_spec: ResetSpec
    ) -> list[Observation]:
        """Vector form's reset: the requested lanes are handed to the same
        env in one call.

        Args:
            slots: The slot indices.
            reset_spec: Episode initialization parameters.

        Returns:
            Initial observations, in the same order as ``slots``.
        """
        targets = [self._require_slot(index) for index in slots]
        if not targets:
            return []
        env = targets[0].env
        reset_state_ids = [self._reset_state_id(slot, reset_spec) for slot in targets]
        lanes = [slot.lane_index for slot in targets]
        if not self._vector_initialized:
            # The vector env's **first** reset must cover every lane.
            # ``LiberoEnv.reset`` ends by calling
            # ``_wrap_obs(self.current_raw_obs)``, which iterates over
            # **every** lane, and a lane that has never been reset is still
            # ``None`` -- so a subset reset would directly raise
            # ``'NoneType' object is not subscriptable`` (reproduced on GPU;
            # Runtime's reset is per-session, so the normal path is bound to
            # reset only one lane first). Absent lanes are first initialized
            # with the id from the request; when their own session later
            # resets, they will be reset again with their own spec.
            filler = reset_state_ids[0]
            requested = dict(zip(lanes, reset_state_ids, strict=True))
            all_lanes = [slot.lane_index for slot in self._slots]
            lanes = all_lanes
            reset_state_ids = [requested.get(lane, filler) for lane in all_lanes]
        obs, _info = env.reset(
            env_idx=np.array(lanes),
            reset_state_ids=np.array(reset_state_ids),
        )
        # Only set the flag **after success**: setting it on failure would
        # make a retry go through the subset-reset path -- exactly the path
        # that raises ``'NoneType' object is not subscriptable`` on an
        # uninitialized vector env, permanently jamming the pool (found
        # during an independent audit).
        self._vector_initialized = True
        observations: list[Observation] = []
        for slot_index in slots:
            slot = self._slots[slot_index]
            index = lanes.index(slot.lane_index)
            self._after_reset(slot_index, int(reset_state_ids[index]), reset_spec, obs)
            observations.append(self._observation(slot_index, obs))
        return observations

    def _after_reset(
        self,
        slot_index: int,
        reset_state_id: int,
        reset_spec: ResetSpec,
        obs: dict[str, Any],
    ) -> None:
        """Per-lane bookkeeping after reset (shared by both forms).

        Args:
            slot_index: The slot index.
            reset_state_id: This episode's reset state id.
            reset_spec: Episode initialization parameters.
            obs: The family reset observation, used to seed the EEF
                displacement anchor for the first physical step.
        """
        slot = self._slots[slot_index]
        lane = slot.lane_index
        slot.started = True
        slot.terminated = False
        slot.truncated = False
        slot.frozen = False
        slot.step_index = 0
        slot.resets += 1
        slot.reset_state_id = reset_state_id
        slot.task_id = int(slot.env.task_ids[lane])
        slot.trial_id = int(slot.env.trial_ids[lane])
        slot.instruction = str(
            reset_spec.instruction or slot.env.task_descriptions[lane]
        )
        slot.audit_trace = []
        states = _to_numpy(obs["states"])[lane].reshape(-1)
        slot.critic_previous_eef = (
            states[:3].astype(np.float64).copy() if states.size >= 3 else None
        )
        slot.critic_history_pending_reset = True
        self._configure_critic(slot, reset_spec)

    def _configure_critic(self, slot: _LiberoSlot, reset_spec: ResetSpec) -> None:
        """(Re)build this episode's Critic based on
        ``ResetSpec.options["critic_rules"]``.

        Semantics aligned with the legacy
        ``LiberoEnvClient._configure_critic``: an episode is only
        (re)constructed once, at reset time; if ``options`` has no
        ``critic_rules`` key, or its value is an empty list/``None``, that
        means this episode does not enable Critic evaluation
        (``slot.critic`` stays ``None``, so ``_chunk_step_one`` skips the
        evaluation branch entirely, at zero cost).

        This is not the legacy constraint of "rejecting a rule-set change
        within the same episode" (that constraint lived inside
        ``critic_chunk_step``, and this Runtime re-invokes this method on
        every reset anyway, naturally constructing a fresh instance);
        instead, this method also lands the remaining Critic parameters from
        ``options`` (``interrupt_on_proposal``) onto the slot.

        Args:
            slot: A slot that has already completed basic reset bookkeeping.
            reset_spec: Episode initialization parameters; ``options``
                carries ``critic_rules`` / ``critic_interrupt_on_proposal``.

        Raises:
            RuntimeApiError: ``critic_rules`` is not an array, an item is
                missing a required field (``INVALID_ARGUMENT``), or this
                core is in ``lockstep_vector`` form (also
                ``INVALID_ARGUMENT``: that form runs
                ``backends.rlinf_family.run_lockstep_chunk``'s
                family-shared algorithm, which never calls
                ``_evaluate_critic``, so configuring rules there would
                silently never take effect -- better to reject it up front
                at reset time).
        """
        payload = reset_spec.options.get("critic_rules")
        interrupt = bool(reset_spec.options.get("critic_interrupt_on_proposal", True))
        slot.critic_interrupt_on_proposal = interrupt
        if not payload:
            slot.critic = None
            slot.critic_rule_fingerprint = None
            return
        if self._core_form == LOCKSTEP_VECTOR_FORM:
            raise RuntimeApiError(
                make_error(
                    ErrorCode.INVALID_ARGUMENT,
                    "critic_rules is only supported for core_form='per_slot': "
                    "lockstep_vector chunk execution goes through the "
                    "family-shared run_lockstep_chunk loop, which never calls the "
                    "libero Critic evaluation hook, so configuring rules there "
                    "would silently never fire",
                    core_form=self._core_form,
                )
            )
        try:
            rules = critic_rules_from_payload(payload)
            fingerprint = canonical_rules_fingerprint(payload)
        except (ValueError, KeyError, TypeError) as exc:
            raise RuntimeApiError(
                make_error(
                    ErrorCode.INVALID_ARGUMENT,
                    f"invalid critic_rules payload: {exc}",
                )
            ) from exc
        slot.critic = TemporalCritic(rules)
        slot.critic_rule_fingerprint = fingerprint

    def observe(self, slots: Sequence[int]) -> list[Observation]:
        """Read the cached observation without changing environment state.

        Args:
            slots: The slot indices.

        Returns:
            Observations, in the same order as ``slots``.

        Raises:
            RuntimeApiError: A slot has not been reset yet (``SESSION_NOT_READY``).
        """
        observations: list[Observation] = []
        for slot_index in slots:
            slot = self._require_slot(slot_index)
            if slot.last_observation is None:
                raise RuntimeApiError(
                    make_error(
                        ErrorCode.SESSION_NOT_READY,
                        f"libero slot {slot_index} has not been reset yet",
                        slot_index=slot_index,
                    )
                )
            observations.append(slot.last_observation)
        return observations

    def lane_status(self, slots: Sequence[int]) -> list[LaneStatus]:
        """Read a snapshot of lane lifecycle state (the ``LaneStatusReader``
        read-back channel for masking semantics).

        Args:
            slots: The slot indices.

        Returns:
            Snapshots, in the same order as ``slots``.
        """
        return lane_statuses(self._slots, slots)

    def chunk_step(
        self, slots: Sequence[int], chunk_actions: Sequence[np.ndarray]
    ) -> list[ChunkOutcome]:
        """Execute an action chunk on the given slots.

        Args:
            slots: The slot indices.
            chunk_actions: ``[chunk, action_dim]`` actions for each slot.

        Returns:
            Normalized results, in the same order as ``slots``.

        Raises:
            RuntimeApiError: The number of slots does not match the number
                of action blocks (``INVALID_ARGUMENT``).
        """
        if len(slots) != len(chunk_actions):
            raise RuntimeApiError(
                make_error(
                    ErrorCode.INVALID_ARGUMENT,
                    f"chunk_step got {len(slots)} slots but "
                    f"{len(chunk_actions)} action blocks",
                )
            )
        if self._core_form == LOCKSTEP_VECTOR_FORM:
            return self._chunk_step_lockstep(slots, chunk_actions)
        return [
            self._chunk_step_one(slot_index, actions)
            for slot_index, actions in zip(slots, chunk_actions, strict=True)
        ]

    def _chunk_step_lockstep(
        self, slots: Sequence[int], chunk_actions: Sequence[np.ndarray]
    ) -> list[ChunkOutcome]:
        """Vector form: advance the whole pool by one tick together as
        **one** real coalesced batch.

        The algorithm and masking semantics live in
        ``backends/rlinf_family.run_lockstep_chunk``, shared by all three
        families; this method only supplies libero's four callbacks (action
        preprocessing, single step, elapsed steps, obs extraction).

        Args:
            slots: Slot indices participating in this group.
            chunk_actions: ``[chunk, action_dim]`` actions, in the same order as ``slots``.

        Returns:
            Normalized results, in the same order as ``slots``.

        Raises:
            RuntimeApiError: A lane in the pool has never been reset
                (``SESSION_NOT_READY``), or the action shape / chunk length
                is invalid (``INVALID_ARGUMENT``).
        """
        from zetta.compat.actions import prepare_actions

        for slot_index in slots:
            self._require_slot(slot_index)
        self.total_chunk_calls += 1
        self.coalesced_group_count += 1
        env = self._envs[0]

        def _prepare(batch: np.ndarray, chunk_len: int) -> np.ndarray:
            # Divergence 4: the whole batch goes through the family's
            # preprocessing at once, using the same function as the
            # per_slot form.
            return np.asarray(
                prepare_actions(
                    batch,
                    env_type=LIBERO_ENV_FAMILY,
                    model_type=self.config.action_model_type,
                    num_action_chunks=chunk_len,
                    action_dim=int(batch.shape[2]),
                ),
                dtype=np.float32,
            )

        def _step(actions: np.ndarray) -> tuple[Any, Any, Any, Any, Any]:
            self.total_env_steps += 1
            return env.step(actions, auto_reset=False)

        def _elapsed(lane_index: int) -> int:
            return int(_to_numpy(env.elapsed_steps).reshape(-1)[lane_index])

        outcomes, stats = run_lockstep_chunk(
            behavior=self.behavior,
            core_form=self._core_form,
            lanes=self._slots,
            slots=list(slots),
            blocks=list(chunk_actions),
            action_dim=int(self.config.action_dim),
            hold_action=self._hold_action(),
            prepare=_prepare,
            step=_step,
            elapsed_steps=_elapsed,
            observe=self._observation,
            include_step_observations=self.config.return_all_frames,
            chunk_info=lambda slot_index: {
                "chunk_calls": self._slots[slot_index].chunk_calls,
                "task_id": self._slots[slot_index].task_id,
                "reset_state_id": self._slots[slot_index].reset_state_id,
                "masked_steps": self._slots[slot_index].masked_steps,
            },
        )
        self.total_masked_steps = sum(slot.masked_steps for slot in self._slots)
        return outcomes

    def extension(
        self, slot: int, namespace: str, method: str, args: dict[str, Any]
    ) -> dict[str, Any]:
        """Execute one of LIBERO's six privileged methods.

        All six are read-only, so ``EXTENSION_CALL`` neither consumes an
        ``operation_seq`` nor requires the episode to already be reset
        (calling ``get_camera_meta`` / ``cached_image`` before reset is
        legitimate). ``raw_obs`` is the fifth extension added later (a
        dozen or so legacy motion primitives rely on it to keep working),
        and ``critic_state`` is the sixth, added for the Critic-Recovery
        three-way comparison (reads the current Critic privileged state
        independently of ``chunk_step``, for diagnostic scripts / parity comparison).

        Args:
            slot: The slot index.
            namespace: Must be ``"libero"``.
            method: One of the method names in ``LIBERO_EXTENSIONS``.
            args: Method arguments.

        Returns:
            Structured result (msgpack-native + ``PayloadRef``).

        Raises:
            RuntimeApiError: The namespace or method is not declared
                (``UNSUPPORTED_EXTENSION``).
        """
        full_name = f"{namespace}.{method}"
        if full_name not in LIBERO_EXTENSIONS:
            raise RuntimeApiError(
                make_error(
                    ErrorCode.UNSUPPORTED_EXTENSION,
                    f"libero env does not implement extension {full_name!r}",
                    namespace=namespace,
                    method=method,
                    supported=sorted(LIBERO_EXTENSIONS),
                )
            )
        state = self._require_slot(slot)
        if method == "get_camera_meta":
            return dict(
                state.env.get_camera_meta(
                    camera_name=str(args.get("camera_name", "agentview")),
                    height=int(args.get("height", 256)),
                    width=int(args.get("width", 256)),
                )
            )
        if method == "render_camera":
            return self._render_camera(state, args)
        if method == "cached_image":
            return self._cached_image(state)
        if method == "raw_obs":
            return self._raw_obs(state, args)
        if method == "critic_state":
            return self._privileged_critic_state(
                state, reset_tracker_override=args.get("reset_tracker")
            )
        return self._privileged_contacts(state, args)

    # ------------------------------------------------------------------ Internal

    def _require_slot(self, slot_index: int) -> _LiberoSlot:
        """Look up slot state and validate the index.

        Args:
            slot_index: The slot index.

        Returns:
            The slot state.

        Raises:
            RuntimeApiError: The index is out of bounds
                (``INVALID_ARGUMENT``). The pool is pre-allocated and does
                not grow.
        """
        if not 0 <= slot_index < len(self._slots):
            raise RuntimeApiError(
                make_error(
                    ErrorCode.INVALID_ARGUMENT,
                    f"slot {slot_index} is outside the pool "
                    f"(size {len(self._slots)}); pools do not grow (plan D6)",
                    slot_index=slot_index,
                    pool_size=len(self._slots),
                )
            )
        return self._slots[slot_index]

    def _reset_state_id(self, slot: _LiberoSlot, reset_spec: ResetSpec) -> int:
        """Compute the reset state id from ``(task_id, seed)``.

        The formula is **verbatim identical** to legacy
        ``robots/libero/env_server.py::make_env``
        (``first_id + seed % trials``): parity is aligned precisely through it.

        Args:
            slot: The slot state.
            reset_spec: Episode initialization parameters.

        Returns:
            The reset state id.

        Raises:
            RuntimeApiError: ``task_id`` is out of range (``INVALID_ARGUMENT``).
        """
        if reset_spec.reset_state_id is not None:
            return int(reset_spec.reset_state_id)
        suite = slot.env.task_suite
        task_id = (
            int(reset_spec.task_id)
            if reset_spec.task_id is not None
            else int(self.config.task_id)
        )
        num_tasks = int(suite.get_num_tasks())
        if not 0 <= task_id < num_tasks:
            raise RuntimeApiError(
                make_error(
                    ErrorCode.INVALID_ARGUMENT,
                    f"task_id {task_id} is outside [0, {num_tasks - 1}] for suite "
                    f"{self.config.task_suite_name!r}",
                    task_id=task_id,
                    num_tasks=num_tasks,
                )
            )
        first_id = sum(
            len(suite.get_task_init_states(index)) for index in range(task_id)
        )
        trials = len(suite.get_task_init_states(task_id))
        seed = int(reset_spec.seed) if reset_spec.seed is not None else 0
        return first_id + (seed % max(1, trials))

    def _observation(self, slot_index: int, obs: dict[str, Any]) -> Observation:
        """Convert the family's 5-key obs dict into a Runtime
        ``Observation``, and update the cache.

        Args:
            slot_index: The slot index.
            obs: The output of ``LiberoEnv._wrap_obs`` (the 5-key schema).

        Returns:
            An ``Observation``; ``session_id`` / ``episode_id`` are left
            blank for the worker to stamp.
        """
        slot = self._slots[slot_index]
        lane = slot.lane_index
        main = np.ascontiguousarray(_to_numpy(obs["main_images"])[lane])
        wrist_raw = obs.get("wrist_images")
        state_array = _to_numpy(obs["states"])[lane].reshape(-1)
        descriptions = obs.get("task_descriptions") or []
        instruction = (
            slot.instruction
            if slot.instruction
            else (str(descriptions[lane]) if lane < len(descriptions) else "")
        )
        observation = Observation(
            session_id=SessionId(""),
            episode_id=EpisodeId(0),
            step_index=slot.step_index,
            main_image=payload_module.encode_image(main),
            wrist_image=(
                payload_module.encode_image(
                    np.ascontiguousarray(_to_numpy(wrist_raw)[lane])
                )
                if wrist_raw is not None
                else None
            ),
            state=[float(value) for value in state_array],
            instruction=instruction,
            extras={
                "slot_index": slot_index,
                "task_id": slot.task_id,
                "trial_id": slot.trial_id,
                "reset_state_id": slot.reset_state_id,
                "env_family": LIBERO_ENV_FAMILY,
            },
        )
        slot.last_observation = observation
        slot.last_main_image = main
        return observation

    def _chunk_step_one(self, slot_index: int, actions: np.ndarray) -> ChunkOutcome:
        """Execute an action chunk step by step on one slot, stopping at the
        first termination signal.

        Args:
            slot_index: The slot index.
            actions: ``[chunk, action_dim]`` actions.

        Returns:
            The normalized result.

        Raises:
            RuntimeApiError: The slot has not been reset, or the action
                shape is wrong (``SESSION_NOT_READY`` / ``INVALID_ARGUMENT``).
        """
        from zetta.compat.actions import prepare_actions

        slot = self._require_slot(slot_index)
        slot.chunk_calls += 1
        self.total_chunk_calls += 1
        if not slot.started:
            raise RuntimeApiError(
                make_error(
                    ErrorCode.SESSION_NOT_READY,
                    f"libero slot {slot_index} has not been reset yet",
                    slot_index=slot_index,
                )
            )
        block = np.asarray(actions, dtype=np.float32)
        if block.ndim != 2 or block.shape[1] != self.config.action_dim:
            raise RuntimeApiError(
                make_error(
                    ErrorCode.INVALID_ARGUMENT,
                    f"libero expects [chunk, {self.config.action_dim}] actions, got "
                    f"shape {tuple(int(dim) for dim in block.shape)}",
                    action_dim=self.config.action_dim,
                )
            )
        # Divergence 4: action preprocessing reuses rlinf's family branch,
        # not a hand-written gripper transform.
        prepared = np.asarray(
            prepare_actions(
                block[None, ...],
                env_type=LIBERO_ENV_FAMILY,
                model_type=self.config.action_model_type,
                num_action_chunks=int(block.shape[0]),
                action_dim=int(block.shape[1]),
            ),
            dtype=np.float32,
        )

        rewards: list[float] = []
        terminations: list[bool] = []
        truncations: list[bool] = []
        per_step_info: list[dict[str, Any]] = []
        frames: list[Observation] = []
        proposals: list[dict[str, Any]] = []
        obs: dict[str, Any] | None = None
        for index in range(int(prepared.shape[1])):
            if slot.terminated or slot.truncated:
                break
            action = block[index]
            obs, reward, terminated, truncated, _info = slot.env.step(
                prepared[:, index], auto_reset=False
            )
            slot.step_index = int(_to_numpy(slot.env.elapsed_steps).reshape(-1)[0])
            slot.env_steps += 1
            self.total_env_steps += 1
            slot.terminated = bool(_scalar(terminated))
            slot.truncated = bool(_scalar(truncated))
            reward_value = _scalar(reward)
            rewards.append(reward_value)
            terminations.append(slot.terminated)
            truncations.append(slot.truncated)
            step_info: dict[str, Any] = {"step_index": slot.step_index}
            step_proposals = self._evaluate_critic(
                slot, obs, reward=reward_value, action=action
            )
            if step_proposals:
                step_info["critic_proposal_rule_ids"] = [
                    row["rule_id"] for row in step_proposals
                ]
                proposals.extend(step_proposals)
            per_step_info.append(step_info)
            frames.append(self._observation(slot_index, obs))
            if step_proposals and slot.critic_interrupt_on_proposal:
                # Hitting a proposal breaks out of the physical action loop
                # early: matches the legacy ``critic_chunk_step``'s
                # ``interrupt_on_proposal`` semantics, so the remaining
                # action block is not executed.
                break
        final = (
            frames[-1]
            if frames
            else (slot.last_observation or self._observation(slot_index, obs or {}))
        )
        info: dict[str, Any] = {
            "chunk_calls": slot.chunk_calls,
            "task_id": slot.task_id,
            "reset_state_id": slot.reset_state_id,
        }
        if slot.critic is not None:
            info["critic_proposals"] = proposals
            info["critic_rule_count"] = len(slot.critic.rules)
        return normalize_chunk_outcome(
            behavior=self.behavior,
            final_observation=final,
            step_observations=frames,
            rewards=rewards,
            terminations=terminations,
            truncations=truncations,
            requested_horizon=int(block.shape[0]),
            per_step_info=per_step_info,
            include_step_observations=self.config.return_all_frames,
            info=info,
        )

    def _evaluate_critic(
        self,
        slot: _LiberoSlot,
        obs: dict[str, Any],
        *,
        reward: float,
        action: np.ndarray,
    ) -> list[dict[str, Any]]:
        """Run Critic evaluation for a physical action step that has just
        finished (skipped at zero cost if Critic is not configured).

        Semantically aligned with legacy
        ``LiberoEnvClient.critic_chunk_step``: after each physical action,
        extract features from the current obs + privileged state, feed them
        to ``TemporalCritic.evaluate``, append any hits to the
        cross-chunk-persistent ``audit_trace``, and update
        ``critic_previous_eef`` for the next step's genuine-displacement computation.

        Args:
            slot: The slot that has just completed this step's ``env.step``
                (``slot.step_index`` / ``slot.terminated`` / ``slot.truncated``
                already reflect this step's values).
            obs: This step's family 5-key obs dict (the single lane is
                already implicitly indexed by the caller of
                ``_observation``; here it is re-indexed by ``lane``
                directly, to avoid depending on call order).
            reward: This step's reward (a scalar, after ``_scalar``).
            action: The raw 7-dim action executed this step (before
                preprocessing, aligned with the legacy ``action`` parameter
                -- legacy passes the caller's originally supplied single-step
                action, not the version after ``prepare_actions``).

        Returns:
            The list of proposals that fired this step; returns an empty
            list if Critic is not configured or nothing fired this step.
        """
        if slot.critic is None:
            return []
        lane = slot.lane_index
        states = _to_numpy(obs["states"])[lane].reshape(-1)
        privileged_state = self._privileged_critic_state(slot)
        features = extract_libero_critic_features(
            {"states": states},
            step_index=slot.step_index,
            reward=reward,
            terminated=slot.terminated,
            truncated=slot.truncated,
            privileged_state=privileged_state,
            action=action,
            previous_eef=slot.critic_previous_eef,
        )
        slot.critic_previous_eef = states[:3].astype(np.float64).copy()
        step_proposals = slot.critic.evaluate(features, step_index=slot.step_index)
        slot.audit_trace.append(
            {
                "step_index": slot.step_index,
                "reward": reward,
                "terminated": slot.terminated,
                "truncated": slot.truncated,
                "proposal_rule_ids": [row["rule_id"] for row in step_proposals],
            }
        )
        return step_proposals

    def _privileged_critic_state(
        self, slot: _LiberoSlot, *, reset_tracker_override: Any = None
    ) -> dict[str, Any]:
        """Read the Critic-specific privileged state from the subprocess
        holding the sim (without advancing the simulation).

        Goes through the ``render`` forwarding hook that ``libero_privileged``
        installs on the subprocess env, the same path used by
        ``_privileged_contacts``. ``reset_tracker`` has two independent
        driving paths:

        - Called internally from ``_evaluate_critic``
          (``reset_tracker_override`` is ``None``): uses the explicit
          ``slot.critic_history_pending_reset`` flag (rather than
          ``env_steps == 0`` -- the latter accumulates across episodes and
          is never reset, so it cannot identify "this episode's first
          collection"); it is set true on every reset and set false after
          this episode's first collection;
        - Called externally via
          ``extension_call("critic_state", {"reset_tracker": ...})``
          (``reset_tracker_override`` is not ``None``): the caller
          explicitly declares whether to clear, without touching
          ``slot.critic_history_pending_reset``, so it does not disturb the
          next automatic decision made by ``_evaluate_critic``. Both paths
          share the same subprocess-side ``_rr_critic_history``; each
          requests its own clearing independently.

        Args:
            slot: The slot state.
            reset_tracker_override: An explicit externally-specified
                ``reset_tracker``; ``None`` means follow the internal
                automatic semantics (reading
                ``slot.critic_history_pending_reset``).

        Returns:
            The return value of the ``libero.critic_state`` extension (a
            flat dict of ``privileged.*`` keys); returns
            ``{"privileged.available": False, ...}`` if the extension was
            not installed in that subprocess.
        """
        worker = slot.env.env.workers[slot.lane_index]
        if reset_tracker_override is None:
            reset_tracker = slot.critic_history_pending_reset
            advance_pending_reset = True
        else:
            reset_tracker = bool(reset_tracker_override)
            advance_pending_reset = False
        payload = worker.render(
            **{
                libero_privileged.RENDER_EXTENSION_KEY: (
                    libero_privileged.CRITIC_STATE_METHOD
                ),
                "reset_tracker": reset_tracker,
            }
        )
        if advance_pending_reset:
            slot.critic_history_pending_reset = False
        if not isinstance(payload, dict):
            return {
                "privileged.available": False,
                "privileged.task.semantic_available": False,
            }
        return payload

    def _render_camera(self, slot: _LiberoSlot, args: dict[str, Any]) -> dict[str, Any]:
        """Render an arbitrary camera (the image goes through ``PayloadRef``;
        numpy cannot cross msgpack).

        Args:
            slot: The slot state.
            args: ``camera_name`` / ``height`` / ``width`` / ``depth``。

        Returns:
            ``{"image": PayloadRef, "depth": PayloadRef | None, ...}``。
        """
        depth = bool(args.get("depth", False))
        rendered = slot.env.render_camera(
            camera_name=str(args.get("camera_name", "agentview")),
            height=int(args.get("height", 1024)),
            width=int(args.get("width", 1024)),
            depth=depth,
        )
        if depth and isinstance(rendered, tuple):
            rgb, depth_map = rendered
        else:
            rgb, depth_map = rendered, None
        rgb_array = np.ascontiguousarray(_to_numpy(rgb))
        result: dict[str, Any] = {
            "available": True,
            "camera_name": str(args.get("camera_name", "agentview")),
            "height": int(rgb_array.shape[0]),
            "width": int(rgb_array.shape[1]),
            "image": payload_module.encode_image(rgb_array),
            "depth": None,
        }
        if depth_map is not None:
            depth_array = np.ascontiguousarray(
                _to_numpy(depth_map).astype(np.float32, copy=False)
            )
            result["depth"] = payload_module.encode_array(depth_array)
        return result

    def _cached_image(self, slot: _LiberoSlot) -> dict[str, Any]:
        """Return the most recent main-view frame (without touching the environment).

        Args:
            slot: The slot state.

        Returns:
            ``{"available": bool, "image": PayloadRef | None, "shape": [...]}``。
        """
        cached = slot.last_main_image
        if cached is None:
            return {"available": False, "image": None, "shape": []}
        return {
            "available": True,
            "image": payload_module.encode_image(cached),
            "shape": [int(dim) for dim in cached.shape],
            "step_index": slot.step_index,
        }

    def _raw_obs(self, slot: _LiberoSlot, args: dict[str, Any]) -> dict[str, Any]:
        """Return the **raw** obs dict from inside the subprocess (the fifth extension).

        Corresponds to legacy ``robots/libero/env_server.py::raw_obs``
        (``_to_numpy_tree(self._env.current_raw_obs[self._env_idx])``).
        numpy cannot cross msgpack, so arrays are always returned as
        ``PayloadRef``, decoded back into numpy by the seam
        (``adapters/zetta/runtime_env_client.py``); non-array values
        (strings, python scalars) are placed as-is into ``scalars``.

        The encoding deliberately **does not use** ``encode_payload``: it
        would promote a 2D uint8 array into ``[H, W, 1]`` (PNG only has an
        HWC layout), while legacy's ``raw_obs`` preserves shape per key. So
        only "3-dim uint8 with channel count in {1,2,3,4}" goes through PNG
        (image, lossless, and saves an order of magnitude of bytes); the
        rest goes through raw encoding, preserving shape and dtype verbatim.

        Args:
            slot: The slot state.
            args: Optional ``keys`` (only these keys are taken; defaults to
                all, matching legacy).

        Returns:
            ``{"available": bool, "keys": [...], "arrays": {key: PayloadRef},
            "scalars": {key: value}, "step_index": int}``。
        """
        entries = getattr(slot.env, "current_raw_obs", None)
        lane = slot.lane_index
        entry = entries[lane] if entries is not None and lane < len(entries) else None
        if not isinstance(entry, Mapping):
            return {
                "available": False,
                "keys": [],
                "arrays": {},
                "scalars": {},
                "step_index": slot.step_index,
                "reason": (
                    "LiberoEnv.current_raw_obs is empty; raw_obs is only available "
                    "after reset"
                ),
            }
        wanted = args.get("keys")
        selected = (
            [str(key) for key in entry]
            if not wanted
            else [str(key) for key in entry if str(key) in {str(k) for k in wanted}]
        )
        arrays: dict[str, Any] = {}
        scalars: dict[str, Any] = {}
        for key in selected:
            value = _to_numpy(entry[key])
            if not isinstance(value, np.ndarray) or value.dtype == np.dtype("O"):
                scalars[key] = entry[key]
                continue
            contiguous = np.ascontiguousarray(value)
            if (
                contiguous.dtype == np.uint8
                and contiguous.ndim == 3
                and int(contiguous.shape[2]) in (1, 2, 3, 4)
            ):
                arrays[key] = payload_module.encode_image(contiguous)
            else:
                arrays[key] = payload_module.encode_array(contiguous)
        return {
            "available": True,
            "keys": selected,
            "arrays": arrays,
            "scalars": scalars,
            "step_index": slot.step_index,
        }

    def _privileged_contacts(
        self, slot: _LiberoSlot, args: dict[str, Any]
    ) -> dict[str, Any]:
        """Read current contact evidence (executed inside the subprocess holding the sim).

        Goes through the ``render`` forwarding hook that ``libero_privileged``
        installs on the subprocess env: rlinf's libero worker loop has no
        generic ``env_call`` command, and the submodule is read-only.

        Args:
            slot: The slot state.
            args: ``include_all_contacts`` / ``max_contacts``。

        Returns:
            A dict with the same structure as legacy ``privileged_contacts``.
        """
        worker = slot.env.env.workers[slot.lane_index]
        payload = worker.render(
            **{
                libero_privileged.RENDER_EXTENSION_KEY: (
                    libero_privileged.PRIVILEGED_CONTACTS_METHOD
                ),
                "include_all_contacts": bool(args.get("include_all_contacts", False)),
                "max_contacts": int(args.get("max_contacts", 64)),
            }
        )
        if not isinstance(payload, dict):
            return {
                "available": False,
                "status": "unavailable",
                "reason": (
                    "the libero worker returned "
                    f"{type(payload).__name__} instead of a contact report; the "
                    "runtime extension was not installed in that subprocess"
                ),
            }
        return payload


def _close_env(env: Any) -> None:
    """Make a best effort to close a ``LiberoEnv``'s subprocess; never raises.

    Args:
        env: The ``LiberoEnv`` instance.
    """
    inner = getattr(env, "env", None)
    for candidate in (inner, env):
        close = getattr(candidate, "close", None)
        if callable(close):
            try:
                close()
            except BaseException:  # noqa: BLE001 - cleanup must never raise
                pass
            return


def libero_env_capability() -> EnvFamilyCapability:
    """Return the libero family's capability declaration.

    Returns:
        ``EnvFamilyCapability``: per-step obs available, ``reset_state_id``
        supported, no accelerator needed (libero is a CPU subprocess,
        **do not assign a GPU to the EnvWorker**), and the four privileged extensions.
    """
    return capability_from_behavior(
        behavior_for(LIBERO_ENV_FAMILY),
        supports_auto_reset=False,
        supports_reset_state_id=True,
    )


class LiberoEnvFamily:
    """The ``EnvFamilyAdapter`` for the libero family."""

    @property
    def env_family(self) -> str:
        """The family name.

        Returns:
            ``"libero"``.
        """
        return LIBERO_ENV_FAMILY

    @property
    def capability(self) -> EnvFamilyCapability:
        """The family's capability declaration.

        Returns:
            The capability entry.
        """
        return libero_env_capability()

    def create_core(self) -> LiberoEnvCore:
        """Create a not-yet-``build``-ed execution core.

        Returns:
            A ``LiberoEnvCore`` instance.
        """
        return LiberoEnvCore()


def register_libero_env_family(*, replace: bool = True) -> LiberoEnvFamily:
    """Register the libero family into ``ENV_FAMILY_REGISTRY``.

    Defaults to ``replace=True``: a process may build a runtime repeatedly,
    and re-registration should not raise (same shape as
    ``backends/fake/env.py::register_fake_env_family``).

    Args:
        replace: Whether to allow overwriting a same-named registration.

    Returns:
        The registered family adapter.
    """
    adapter = LiberoEnvFamily()
    register_env_family(adapter, replace=replace)
    return adapter
