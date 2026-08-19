"""The robocasa family's ``EnvExecutionCore``.

**Does not reimplement RoboCasaSession**: ``robots.robocasa.session_core.RoboCasaSession``
is reused as-is — it is already a standalone class, its constructor arguments are
scalars, and ``reset``/``execute_chunk`` are already pure dict in/out. This module
does exactly three things: translate Runtime's session/slot semantics into
``RoboCasaSession`` calls, normalize its output through
``core.env_execution.normalize_chunk_outcome``, and encode the images in an
observation into the ``PayloadRef`` needed by ``Observation``.

Key divergences from libero / maniskill (confirmed against
``env_registry.py::ENV_FAMILY_BEHAVIORS["robocasa"]``; see that file's field
comments for details):

1. **``build()`` constructs ``num_envs`` independent ``RoboCasaSession``
   instances**, not one vectorized env taking a ``num_envs`` parameter —
   ``RoboCasaSession`` itself only wraps **one** ``gymnasium.make(...)`` call and
   has no batch dimension. Because of this, this family declares **only
   ``per_slot``** (``core_forms=frozenset({PER_SLOT_FORM})``); ``lockstep_vector``
   has no meaning here.
2. **``RoboCasaSession.reset``/``execute_chunk`` take a single dict payload**
   (``task``/``seed``/``split``/...), not any of the four "env index + keyword"
   signatures used by rlinf families (``reset_signature="task_seed_split_payload"``,
   already updated accordingly in env_registry.py).
3. **``execute_chunk``'s per-step records carry no images** (``session_core.py``'s
   step_record only records ``state``, skipping image encoding for HTTP bandwidth
   reasons) — so the per-step ``Observation`` objects constructed by this adapter
   likewise carry no images (``chunk_obs_layout="per_step"`` still holds: it does
   return an observation per step, just without images for the intermediate
   steps; only the final frame has one). The final frame's image is instead
   assembled by reading ``session.observation`` (the raw gym dict, always
   available, zero extra encoding cost) directly, rather than depending on the
   ``observation`` field in ``execute_chunk``'s return value; state is likewise
   reused from ``session.snapshot(include_images=False)`` / the per-step
   ``state_record``, since they already fold
   ``privileged_state.extract_privileged_state`` into ``state`` and there is no
   need to reimplement that folding.
4. **Needs an accelerator but is not a batched GPU tensor**: rendering forces
   ``MUJOCO_GL=egl`` (a module-level setting in ``session_core.py``), but
   ``RoboCasaSession`` is single-process numpy and does not do the kind of batched
   tensor computation maniskill does — ``needs_accelerator_override=True`` is
   declared independently.

Critic-Recovery and episode wrap-up:

- ``critic_rules`` does **not** go into ``env_config`` (it goes into
  ``EnvSpecMsg.digest()``, and changing the rule set every episode would create a
  new env pool each time), but instead goes through ``ResetSpec.options``: a
  bundle is frozen, the rules are constant within one episode, and
  ``RoboCasaSession._configure_critic`` itself refuses to change rules mid-episode.
  The adapter records it, alongside ``video_dir``/``bundle_sha256``/
  ``interrupt_on_proposal``/``capture_event_images``/``action_scale``, on the slot
  at ``reset`` time and replays it as-is on every ``execute_chunk`` call — an
  earlier version hardcoded ``critic_rules=[]`` here, which meant the Critic could
  never make a proposal and the ``active_bundle`` mode had no meaning on the
  Runtime path.
- ``extensions`` declares two methods (``core.env_registry.ROBOCASA_EXTENSIONS``;
  see that constant's comment for the rationale): ``robocasa.snapshot`` for
  Role1's visual review and Gen0's observation identity, and
  ``robocasa.finalize_episode`` to flush video to disk. Privileged state still
  does not need a separate extension method
  (``privileged_state.extract_privileged_state`` is already folded into every
  observation's ``state`` by ``RoboCasaSession._current_observation``).

Seed semantics differ from the other families: ``reset`` uses ``ResetSpec.seed``'s
**raw value**, with no ``seed_offset``/``slot_index`` offset applied (see
``reset``'s docstring).

Dependency surface: depends only on ``robots.robocasa.session_core`` (numpy +
gymnasium + robocasa + robosuite, all lazily imported by that module itself); does
not introduce rlinf or torch.
"""

from __future__ import annotations

import dataclasses
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
from rollout_runtime.core import payload as payload_module
from rollout_runtime.core.env_execution import (
    PER_SLOT_FORM,
    ChunkOutcome,
    normalize_chunk_outcome,
)
from rollout_runtime.core.env_registry import (
    ROBOCASA_ENV_FAMILY,
    ROBOCASA_EXTENSIONS,
    behavior_for,
    capability_from_behavior,
    register_env_family,
)

__all__ = [
    "ROBOCASA_CURRENT_EXTENSIONS",
    "RobocasaCurrentConfig",
    "RobocasaCurrentCore",
    "RobocasaCurrentFamily",
    "robocasa_current_capability",
    "register_robocasa_current_env_family",
]

ROBOCASA_CURRENT_EXTENSIONS: frozenset[str] = ROBOCASA_EXTENSIONS
"""The robocasa extension methods supported by this build (the single source of
truth lives in ``core.env_registry``).

The family declaration (``ENV_FAMILY_BEHAVIORS["robocasa"].extensions``) and the
execution core's dispatch table must be the same set:
``RuntimeEnvWorker.extension_call`` filters first by family capability, so an
undeclared method never reaches here; the execution core then filters again,
returning ``UNSUPPORTED_EXTENSION`` rather than ``AttributeError`` for a method
that is declared but not implemented.
"""

_CAMERA_SLOTS = (
    "video.robot0_agentview_left",
    "video.robot0_agentview_right",
    "video.robot0_eye_in_hand",
)
"""``RoboCasaSession``'s three camera keys (sharing a source with
``session_core.CAMERA_KEYS``).

The first key goes into ``Observation.main_image``, the third (wrist) goes into
``wrist_image``, and the middle one (right view) goes into
``extra_view_images`` — all three slots can be filled simultaneously only when
all three keys are present; when only some camera keys appear in the
observation, the missing slots stay ``None``/an empty list.
"""


def _flatten_numeric(value: Any) -> list[float]:
    """Recursively flatten numeric leaves (int/float/nested list), skipping
    strings and other non-numeric types.

    After ``RoboCasaSession``'s state fields are converted by ``_json_scalar``
    (session_core.py), vector fields are nested python lists (not numpy
    arrays), while scalar fields are bare int/float — both need to go into the
    flat vector, while strings (e.g. a text field that accidentally ended up
    in state) and ``None`` are always skipped.

    Args:
        value: The value to flatten.

    Returns:
        A list of numeric leaves; returns an empty list for non-numeric input.
    """
    if isinstance(value, bool):
        return []
    if isinstance(value, (int, float)):
        return [float(value)]
    if isinstance(value, (list, tuple)):
        return [leaf for item in value for leaf in _flatten_numeric(item)]
    return []


@dataclasses.dataclass(kw_only=True)
class RobocasaCurrentConfig:
    """The robocasa-current family's private config (corresponds to
    ``EnvSpecMsg.env_config``).

    Fields correspond one-to-one with ``RoboCasaSession.__init__``/``reset``'s
    payload, with defaults matching the CLI defaults in
    ``robots/robocasa/env_server.py::main``.

    Does not include ``operation_gate_*`` fields: ``RoboCasaSession`` used to
    support, via ``GpuOperationGate``, a deployment form where "multiple
    *independent OS processes* share one GPU" (an flock-based semaphore). That
    dependency (``gpu_gate.py`` and ``RoboCasaSession``'s reliance on it) has
    since been removed (along with the multi-process shared-GPU deployment path
    in ``env_server.py``/``robocasa_capacity_worker.py``, which is no longer
    supported). Ray's placement declaration
    (``cluster.component_placement`` + ``env_worker.placement_strategy=packed``)
    already guarantees **one Ray rank exclusively owns one GPU**, so the same
    GPU will never again be contended for by a second *process*; the
    concurrency of multiple sessions (threads) within one rank sharing the same
    card is instead controlled by ``env_worker.max_sessions_per_rank`` (set to 1
    in this preset, see ``config/presets/robocasa_current.yaml``). The field
    itself was already removed from ``RobocasaCurrentConfig`` earlier; this
    paragraph preserves the historical context.

    Attributes:
        task: The RoboCasa task name (e.g. ``"SlideDishwasherRack"``), fed into
            ``gymnasium.make(f"robocasa/{task}", ...)``.
        split: The task split (``"train"``/``"target"``, etc.).
        camera_size: Camera resolution (square edge length).
        max_steps: Maximum steps per episode.
        cold_reset_lock: Path to a cross-process renderer-creation mutex lock;
            ``None`` means no locking (single-process scenario).
        require_isolated_renderer: Whether to require an isolated MuJoCo
            renderer (true by default in production).
        enable_task_program: Whether to enable the task program (e.g. the
            dishwasher rack state machine).
        interrupt_on_proposal: Whether to interrupt a chunk when the
            critic/task program makes a proposal.
        capture_event_images: Whether proposal events include images (via a
            JPEG data URL, at a bandwidth cost; only enable when audit
            screenshots are actually needed).
        process_isolation: Whether to put each slot's ``RoboCasaSession`` in its
            own independent OS subprocess
            (``robots.robocasa.session_process.RemoteRoboCasaSession``), rather
            than constructing it directly within this rank's process. Defaults
            to false (preserving existing behavior/unit tests unchanged; the
            ``pool_size=1`` scenario has zero extra IPC overhead). Only needs
            to be enabled when ``pool_size > 1`` within the same rank (multiple
            sessions sharing one GPU): ``env_worker.py::_call_pool_core`` takes
            no lock at all for ``per_slot`` pools, assuming slots are
            independent — that assumption holds at the Python thread level, but
            two threads' ``reset()`` calls would simultaneously enter native
            robosuite/MuJoCo/EGL initialization paths
            (``_set_mujoco_context_and_buffers``/``from_xml_string``, etc.),
            which in practice randomly produces an ``EGL_BAD_ACCESS`` crash or a
            complete deadlock (two manifestations of the same race window, see
            the process-isolation design notes §0.2). A genuine OS process
            boundary prevents two slots from sharing the same native GL/EGL
            state; a Python lock cannot, because the race is in native code
            outside the Python lock's protection.

    The last three items are **family default values** that can be overridden
    per episode by ``ResetSpec.options`` (see ``_EpisodeOptions``): they do not
    affect the env pool's identity, and are placed in ``env_config`` only to
    give callers without per-episode needs a static default.
    """

    task: str
    split: str = "target"
    camera_size: int = 256
    max_steps: int = 1000
    cold_reset_lock: str | None = None
    require_isolated_renderer: bool = True
    enable_task_program: bool = False
    interrupt_on_proposal: bool = True
    capture_event_images: bool = False
    process_isolation: bool = False

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any] | None) -> RobocasaCurrentConfig:
        """Construct a config from an ``env_config`` dict.

        Unknown keys are always rejected rather than silently ignored:
        ``env_config`` feeds into ``EnvSpecMsg.digest()``, and a typo'd key
        would silently create a new pool, which is far harder to diagnose than
        an error (same reasoning as ``fake/env.py``).

        Args:
            config: The family-private config.

        Returns:
            The structured config.

        Raises:
            RuntimeApiError: Missing the required ``task``, or an unknown key
                is present (``INVALID_ARGUMENT``).
        """
        payload = dict(config or {})
        if "task" not in payload:
            raise RuntimeApiError(
                make_error(
                    ErrorCode.INVALID_ARGUMENT,
                    "robocasa env_config requires a 'task' key",
                )
            )
        known = {field.name for field in dataclasses.fields(cls)}
        unknown = sorted(set(payload) - known)
        if unknown:
            raise RuntimeApiError(
                make_error(
                    ErrorCode.INVALID_ARGUMENT,
                    f"unknown robocasa env config keys: {unknown}",
                    unknown_keys=unknown,
                    known_keys=sorted(known),
                )
            )
        return cls(**payload)


@dataclasses.dataclass(kw_only=True)
class _EpisodeOptions:
    """One episode's ``ResetSpec.options`` payload (does not go into
    ``EnvSpecMsg.digest()``).

    These fields differ per episode (``video_dir`` is the current attempt's
    directory, ``critic_rules`` is the current candidate's frozen rule set),
    so they **must** go through ``ResetSpec.options`` rather than
    ``env_config`` — the latter feeds into the env spec digest, and per-episode
    variation would cause a new env pool to be built for every episode (i.e. a
    cold-started MuJoCo environment every time).

    Attributes:
        video_dir: The episode's video directory; ``None`` means no recording.
        bundle_sha256: The digest of the frozen candidate bundle, fed into
            ``RoboCasaSession``'s attestation fields.
        critic_rules: The frozen Critic rule payload (the ``as_dict()`` form of
            ``CandidateBundle.critic_rules``), constant for the whole episode.
        interrupt_on_proposal: Whether to interrupt a chunk when the
            Critic/task program makes a proposal; ``None`` means infer from
            "interrupt if there are rules" (matching the semantics of
            ``interrupt_on_proposal=bool(critic_rules)`` from the era of direct
            connection via ``run_rollout.py``).
        capture_event_images: Whether proposal events include images; ``None``
            means fall back to the family config default.
        enable_task_program: Whether to enable the task program; ``None`` means
            fall back to the family config default.
        action_scale: Scaling coefficients for normalized commands.
    """

    video_dir: str | None = None
    bundle_sha256: str | None = None
    critic_rules: list[dict[str, Any]] = dataclasses.field(default_factory=list)
    interrupt_on_proposal: bool | None = None
    capture_event_images: bool | None = None
    enable_task_program: bool | None = None
    action_scale: dict[str, float] = dataclasses.field(default_factory=dict)

    @classmethod
    def from_mapping(cls, options: Mapping[str, Any] | None) -> _EpisodeOptions:
        """Construct from ``ResetSpec.options``, always rejecting unknown keys.

        Args:
            options: ``ResetSpec.options``.

        Returns:
            Structured episode options.

        Raises:
            RuntimeApiError: An unknown key is present, or ``critic_rules`` is
                not an array of objects (``INVALID_ARGUMENT``).
        """
        payload = dict(options or {})
        known = {field.name for field in dataclasses.fields(cls)}
        unknown = sorted(set(payload) - known)
        if unknown:
            raise RuntimeApiError(
                make_error(
                    ErrorCode.INVALID_ARGUMENT,
                    f"unknown robocasa reset options: {unknown}",
                    unknown_keys=unknown,
                    known_keys=sorted(known),
                )
            )
        rules = payload.get("critic_rules") or []
        if not isinstance(rules, (list, tuple)) or any(
            not isinstance(rule, Mapping) for rule in rules
        ):
            raise RuntimeApiError(
                make_error(
                    ErrorCode.INVALID_ARGUMENT,
                    "robocasa reset option 'critic_rules' must be an array of objects",
                )
            )
        payload["critic_rules"] = [dict(rule) for rule in rules]
        payload["action_scale"] = dict(payload.get("action_scale") or {})
        return cls(**payload)


@dataclasses.dataclass
class _RobocasaSlot:
    """One slot's ``RoboCasaSession`` and its lifecycle markers.

    Attributes:
        session: The ``RoboCasaSession`` exclusively owned by this slot.
        started: Whether it has already been ``reset``.
        chunk_calls: Number of ``chunk_step`` calls.
        episode: The current episode's ``ResetSpec.options`` (replaced on each
            ``reset``).
    """

    session: Any
    started: bool = False
    chunk_calls: int = 0
    episode: _EpisodeOptions = dataclasses.field(default_factory=_EpisodeOptions)


class RobocasaCurrentCore:
    """The robocasa-current family's ``EnvExecutionCore`` implementation.

    Declares only the ``per_slot`` form: each slot is an independent
    ``RoboCasaSession`` (see point 1 of the module docstring), with no shared
    state between them, so ``per_slot``'s independence requirement is
    naturally satisfied.
    """

    def __init__(self) -> None:
        """Initialize a not-yet-``build``-ed execution core."""
        self.config = RobocasaCurrentConfig(task="")
        self.env_spec: EnvSpecMsg | None = None
        self.seed_offset = 0
        self.closed = False
        self._slots: list[_RobocasaSlot] = []
        # Protects append/pop on ``self._slots``: ``add_slot``/``remove_slot``
        # may be called concurrently by ``EnvPool``'s maintenance loop and the
        # cold-create path (the same posture as the libero version's
        # ``_slot_mutation_lock``, see rlinf_env.py::LiberoEnvCore). Reads of
        # the list itself (``slot_count``/``_require_slot``) do not need this
        # lock: a single ``len``/index read on a CPython list is atomic; what
        # genuinely needs mutual exclusion is the compound operation of
        # "read length, decide the new index, then append".
        self._slot_mutation_lock = threading.Lock()

    @property
    def core_form(self) -> str:
        """This core instance's form.

        Returns:
            Always ``per_slot``: see point 1 of the module docstring.
        """
        return PER_SLOT_FORM

    @property
    def behavior(self):  # noqa: ANN201 - a Protocol property; see env_registry.behavior_for for the type
        """The declaration of the family this core belongs to.

        Returns:
            ``ENV_FAMILY_BEHAVIORS["robocasa"]``.
        """
        return behavior_for(ROBOCASA_ENV_FAMILY)

    # -------------------------------------------------------------- Construction and release

    def build(
        self,
        env_spec: EnvSpecMsg,
        *,
        num_envs: int,
        seed_offset: int = 0,
        total_num_processes: int = 1,
    ) -> None:
        """Construct ``num_envs`` independent ``RoboCasaSession`` instances
        according to the spec (baked in at construction time).

        When ``self.config.process_isolation`` is true, each slot builds not a
        same-process ``RoboCasaSession`` but a
        ``robots.robocasa.session_process.RemoteRoboCasaSession`` — a proxy
        running in its own independent OS subprocess, exposing exactly the
        same public signature as the rest of ``RobocasaCurrentCore``'s methods
        (see this module's docstring). Defaults to false, preserving existing
        behavior and unit tests unchanged; only needs to be enabled when
        ``pool_size > 1`` within a single rank (multiple sessions sharing one
        GPU), see the field description of
        ``RobocasaCurrentConfig.process_isolation``.

        This method still constructs each slot **serially** (regardless of
        mode): even with ``process_isolation`` enabled, subprocess startup
        itself spawns one at a time and waits for its ``ready`` handshake
        before building the next, never spawning concurrently — this is
        isomorphic to the non-isolated path of "constructing
        ``RoboCasaSession`` one at a time", and the construction phase has
        never been the race window this feature addresses (the actual race
        occurs when multiple already-built slots each call
        ``reset``/``chunk_step``, see the full diagnostic record in the
        module-level ``session_process.py``).

        Args:
            env_spec: The environment spec; ``env_config`` must contain
                ``task``.
            num_envs: The number of slots in the pool, i.e. the number of
                ``RoboCasaSession`` instances to construct.
            seed_offset: The seed offset for this rank, added to each slot's
                reset seed.
            total_num_processes: Total number of processes participating in
                the split (this family does not need cross-process asset
                splitting; kept only for signature consistency).

        Raises:
            RuntimeApiError: ``num_envs`` is invalid, or ``env_config`` is
                missing ``task``/contains an unknown key
                (``INVALID_ARGUMENT``).
            robots.robocasa.session_process.RemoteSessionCrashed:
                When ``process_isolation=True``, a subprocess failed to start
                or construct.
        """
        if num_envs < 1:
            raise RuntimeApiError(
                make_error(
                    ErrorCode.INVALID_ARGUMENT, f"num_envs must be >= 1, got {num_envs}"
                )
            )
        self.config = RobocasaCurrentConfig.from_mapping(env_spec.env_config)
        self.env_spec = env_spec
        self.seed_offset = seed_offset
        self._slots = [_RobocasaSlot(session=self._make_session()) for _ in range(num_envs)]
        del total_num_processes

    def _make_session(self) -> Any:
        """Construct a new slot's session according to
        ``self.config.process_isolation``.

        Shared by two call sites: ``build()`` constructs the initial pool, and
        ``add_slot`` appends a dynamic slot — both use exactly the same
        per-slot construction parameters (``camera_size``/``max_steps``/
        ``cold_reset_lock``/``require_isolated_renderer``), differing only in
        "how many to build", so there is no reason to fork into two
        implementations (which would silently drift apart over time).

        Returns:
            When ``process_isolation=True``, a
            ``robots.robocasa.session_process.RemoteRoboCasaSession``;
            otherwise a same-process
            ``robots.robocasa.session_core.RoboCasaSession``. Both expose the
            same public signature for the rest of ``RobocasaCurrentCore``'s
            methods (see the module docstring).

        Raises:
            robots.robocasa.session_process.RemoteSessionCrashed:
                When ``process_isolation=True``, the subprocess failed to
                start or construct.
        """
        if self.config.process_isolation:
            from robots.robocasa.session_process import spawn_robocasa_subprocess

            return spawn_robocasa_subprocess(
                camera_size=self.config.camera_size,
                max_steps=self.config.max_steps,
                cold_reset_lock=self.config.cold_reset_lock,
                require_isolated_renderer=self.config.require_isolated_renderer,
            )
        from robots.robocasa.session_core import RoboCasaSession

        return RoboCasaSession(
            camera_size=self.config.camera_size,
            max_steps=self.config.max_steps,
            cold_reset_lock=self.config.cold_reset_lock,
            require_isolated_renderer=self.config.require_isolated_renderer,
        )

    def close(self) -> None:
        """Release every ``RoboCasaSession`` (close environments, flush video
        writers)."""
        for slot in self._slots:
            slot.session.close_environment()
        self._slots = []
        self.closed = True

    # ---------------------------------------------------- Dynamic slot resizing

    def slot_count(self) -> int:
        """Return the current total number of slots
        (``core.env_execution.DynamicSlotPool``).

        Returns:
            The current slot count, including any slots dynamically appended
            after ``build``.
        """
        return len(self._slots)

    def add_slot(self, seed_offset: int) -> int:
        """Append an independent ``RoboCasaSession`` (or its subprocess proxy)
        as a new slot.

        Each RoboCasa slot is already a fully independent ``RoboCasaSession``
        (``build()`` explicitly does ``del total_num_processes``, and there is
        none of libero's constraint of splitting assets across slots or
        picking a free process offset; see point 1 of the module docstring and
        the comparison with ``rlinf_env.py::LiberoEnvCore.add_slot``), so this
        method does not need the layer of complexity that
        ``_allocate_process_offset`` has in the libero version: a new slot
        directly reuses the single-slot construction logic already in
        ``build()`` (``_make_session``) and simply gets appended to the end of
        ``self._slots``.

        Args:
            seed_offset: **Ignored**. ``reset``'s docstring already explains
                that this family's seed semantics do "no slot offsetting" —
                every ``reset`` carries its own authoritative
                ``ResetSpec.seed``, and there is no need, as with libero, to
                pick a non-conflicting offset within
                ``[0, total_num_processes)`` (that mechanism is only needed
                when "one ResetSpec is spread across the whole pool"; this
                family resets each slot independently, see ``reset``'s
                docstring). So the suggested value from the caller
                (``EnvPool._next_seed_offset``) has no consumable use for this
                family, deliberately unused for the same reason as in libero.

        Returns:
            The new slot's index (equal to the total slot count before
            appending).

        Raises:
            RuntimeApiError: The family failed to construct
                (``ENV_FAILURE``).
            robots.robocasa.session_process.RemoteSessionCrashed:
                When ``process_isolation=True``, the subprocess failed to
                start or construct (not wrapped as ``RuntimeApiError``,
                consistent with ``build()``/``spawn_robocasa_subprocess``'s
                existing exception contract; the caller
                ``EnvPool._cold_create_slot`` re-raises any exception other
                than ``RuntimeApiError``/``MemoryError`` as-is).
        """
        del seed_offset  # See the docstring above: this family's reset does no slot offsetting, so the hint is unused.
        with self._slot_mutation_lock:
            new_index = len(self._slots)
            try:
                session = self._make_session()
            except RuntimeApiError:
                raise
            except BaseException as exc:
                raise RuntimeApiError(
                    make_error(
                        ErrorCode.ENV_FAILURE,
                        f"failed to add robocasa slot {new_index}: "
                        f"{type(exc).__name__}: {exc}",
                        slot_index=new_index,
                    )
                ) from exc
            self._slots.append(_RobocasaSlot(session=session))
            return new_index

    def remove_slot(self, slot_index: int) -> None:
        """Close and remove the trailing independent slot.

        Args:
            slot_index: The slot index to remove; must equal the current
                trailing index (``slot_count() - 1``).

        Raises:
            RuntimeApiError: The index is not the current trailing index
                (``INVALID_ARGUMENT``).
        """
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
            slot = self._slots.pop(last_index)
        slot.session.close_environment()

    # ------------------------------------------------------------------ Operations

    def reset(self, slots: Sequence[int], reset_spec: ResetSpec) -> list[Observation]:
        """Reset each of the given slots' ``RoboCasaSession``.

        **Seeds do not get slot offsetting** (deliberately different from
        libero/maniskill's ``seed_offset + slot_index`` convention):
        ``zetta.evolution``'s paired gate treats "the same seed" as a hard
        contract for comparability between the two arms (``EpisodeRecord.seed``
        is checked item-by-item against the preregistration's seed table), and
        offsetting would let which slot a request lands on determine the
        actual initial state, so the seed in the audit record would no longer
        be authoritative. This family's pool is ``per_slot`` and each rollout
        process only drives its own single session, with each session's
        ``reset`` carrying its own authoritative seed; there is no scenario of
        "one ResetSpec spread across the whole pool" causing episode
        duplication (which is exactly the scenario the offsetting convention
        exists to solve).

        Args:
            slots: The slot indices.
            reset_spec: Episode initialization parameters; ``seed``'s raw
                value is used, and family-private
                ``video_dir``/``bundle_sha256``/``critic_rules``/... go through
                ``options`` (see ``_EpisodeOptions``).

        Returns:
            The initial observations, in the same order as ``slots``.

        Raises:
            RuntimeApiError: ``options`` contains an unknown key
                (``INVALID_ARGUMENT``).
        """
        options = _EpisodeOptions.from_mapping(reset_spec.options)
        enable_task_program = (
            self.config.enable_task_program
            if options.enable_task_program is None
            else bool(options.enable_task_program)
        )
        observations: list[Observation] = []
        for slot_index in slots:
            slot = self._require_slot(slot_index)
            slot.session.reset(
                {
                    "task": self.config.task,
                    "split": self.config.split,
                    "seed": reset_spec.seed if reset_spec.seed is not None else 0,
                    "enable_task_program": enable_task_program,
                    "video_dir": options.video_dir,
                    "bundle_sha256": options.bundle_sha256,
                    "action_scale": options.action_scale,
                }
            )
            slot.started = True
            slot.episode = options
            observations.append(self._observation(slot_index))
        return observations

    def observe(self, slots: Sequence[int]) -> list[Observation]:
        """Read the cached observation without changing environment state.

        Args:
            slots: The slot indices.

        Returns:
            Observations in the same order as ``slots``.
        """
        return [self._observation(slot_index) for slot_index in slots]

    def chunk_step(
        self, slots: Sequence[int], chunk_actions: Sequence[np.ndarray]
    ) -> list[ChunkOutcome]:
        """Execute an action chunk on the given slots.

        Args:
            slots: The slot indices.
            chunk_actions: Each slot's ``[chunk, 12]`` actions (``action_dim``
                is fixed to ``action_contract.FLAT_ACTION_SIZE``).

        Returns:
            Normalized results in the same order as ``slots``.

        Raises:
            RuntimeApiError: The number of slots does not match the number of
                action blocks (``INVALID_ARGUMENT``).
        """
        if len(slots) != len(chunk_actions):
            raise RuntimeApiError(
                make_error(
                    ErrorCode.INVALID_ARGUMENT,
                    f"chunk_step got {len(slots)} slots but "
                    f"{len(chunk_actions)} action blocks",
                )
            )
        return [
            self._chunk_step_one(slot_index, actions)
            for slot_index, actions in zip(slots, chunk_actions, strict=True)
        ]

    def extension(
        self, slot: int, namespace: str, method: str, args: dict[str, Any]
    ) -> dict[str, Any]:
        """Dispatch the two methods in ``ROBOCASA_CURRENT_EXTENSIONS``.

        Both map directly onto existing ``RoboCasaSession`` methods, adding no
        new business logic:

        - ``robocasa.snapshot`` (read-only): ``args["include_images"]``
          defaults to true, returns ``session.snapshot(...)`` — named state +
          data URL images, byte-for-byte identical to the payload from the era
          of direct connection via ``POST /observation``.
        - ``robocasa.finalize_episode`` (has a side effect but does not touch
          the environment): flushes the video writer and returns
          ``video_paths``/``video_manifest``, with the environment staying
          warm for the next episode to reuse.

        Args:
            slot: The slot index.
            namespace: Extension namespace.
            method: Extension method name.
            args: Method arguments.

        Returns:
            A structured result (a plain, msgpack-encodable dict).

        Raises:
            RuntimeApiError: The method is not in the supported set
                (``UNSUPPORTED_EXTENSION``), or the slot has not been reset
                yet (``SESSION_NOT_READY``).
        """
        full_name = f"{namespace}.{method}"
        entry = self._require_slot(slot)
        if full_name not in ROBOCASA_CURRENT_EXTENSIONS:
            raise RuntimeApiError(
                make_error(
                    ErrorCode.UNSUPPORTED_EXTENSION,
                    f"robocasa-current does not implement extension {full_name!r}",
                    namespace=namespace,
                    method=method,
                    supported=sorted(ROBOCASA_CURRENT_EXTENSIONS),
                )
            )
        if not entry.started:
            raise RuntimeApiError(
                make_error(
                    ErrorCode.SESSION_NOT_READY,
                    f"slot {slot} has not been reset yet",
                    slot_index=slot,
                    method=full_name,
                )
            )
        if full_name == "robocasa.snapshot":
            include_images = bool(args.get("include_images", True))
            return dict(entry.session.snapshot(include_images=include_images))
        return dict(entry.session.finalize_episode_artifacts())

    # ------------------------------------------------------------------ Internal

    def _require_slot(self, slot_index: int) -> _RobocasaSlot:
        if not 0 <= slot_index < len(self._slots):
            raise RuntimeApiError(
                make_error(
                    ErrorCode.INVALID_ARGUMENT,
                    f"slot {slot_index} is outside the pool "
                    f"(size {len(self._slots)}); pools may grow via add_slot up to "
                    "max_dynamic_pool_size, but this index is beyond the current "
                    "slot count",
                    slot_index=slot_index,
                    pool_size=len(self._slots),
                )
            )
        return self._slots[slot_index]

    def _observation(self, slot_index: int) -> Observation:
        """Assemble an ``Observation`` from a ``RoboCasaSession`` (or its
        subprocess proxy).

        The numpy frame from ``session.observation`` (the raw gym dict) is
        first run through ``session_core.jpeg_lossy_rgb_frame`` for one JPEG
        quality=80 lossy quantization pass (the same quantization parameters
        as the JPEG data URL branch of
        ``RoboCasaSession._current_observation``/``snapshot``; see that
        function's docstring), then encoded with
        ``payload_module.encode_image`` (PNG): PNG is only a **transport**
        encoding that saves a base64/JSON round trip within the local process,
        and cannot change the pixels GR00T actually sees — policy inference is
        deterministic (per ``groot_core.parse_inference_seed``'s per-request
        seed contract), and skipping the quantization would feed the same seed
        "cleaner" pixels on Runtime v3 than before, silently changing the
        action-chunk values and making same-seed replays no longer match
        pre-migration episode records item-by-item. State is reused from
        ``snapshot(include_images=False)``: it already folds
        ``privileged_state.extract_privileged_state`` into the ``state`` field
        (see the module docstring), and there is no need to reimplement that
        folding logic.

        When ``self.config.process_isolation`` is true, ``slot.session`` is a
        ``session_process.RemoteRoboCasaSession``, with no ``session.observation``
        same-process attribute to read — detected via
        ``getattr(session, "observe_encoded", None)`` (the same posture as
        ``EnvPool.dynamic``'s detection of ``add_slot``/``remove_slot``). On
        that branch, the JPEG quantization above happens inside the
        subprocess (see ``session_process.py``'s ``_observe_encoded``), but
        the PNG **transport** encoding is still done in this method (the
        parent process): ``robots/robocasa/session_process.py`` belongs to
        ``robots/robocasa/**``, and per this repository's layering rules it
        cannot import ``rollout_runtime``, so ``payload_module.encode_image``
        cannot be moved into the subprocess. The ``Observation`` produced by
        both branches is entirely equivalent at the pixel/field level; only
        the PNG encoding happens after the subprocess finishes JPEG
        quantization, going through the same ``payload_module.encode_image``
        call once control returns to the parent process (not two independent
        implementations).

        ``Observation.state`` is a flat vector (this field's declared type),
        but GR00T (``groot_core.py::STATE_FIELDS``) reads state by **named**
        keys (e.g. ``state.end_effector_position_relative``), and flattening
        would lose the key names. So the named dict is preserved as-is here in
        ``extras["raw_state"]``, and ``groot_policy.py`` retrieves values by
        name from there, without depending on the flat vector's ordering
        convention.

        Args:
            slot_index: The slot index.

        Returns:
            The current observation for this slot.

        Raises:
            RuntimeApiError: This slot has not been reset yet
                (``SESSION_NOT_READY``).
        """
        slot = self._require_slot(slot_index)
        session = slot.session
        observe_encoded = getattr(session, "observe_encoded", None)
        if callable(observe_encoded):
            # ``process_isolation=True``: session is a ``RemoteRoboCasaSession``,
            # with no ``session.observation`` same-process attribute to read;
            # the subprocess only does JPEG quantization and passes back the
            # still-uint8 array over IPC (see the docstring paragraph above),
            # while the PNG transport encoding is done uniformly here (the
            # parent process).
            try:
                encoded = observe_encoded(camera_keys=list(_CAMERA_SLOTS))
            except Exception as exc:  # noqa: BLE001 - normalize like any other RPC
                raise RuntimeApiError(
                    make_error(
                        ErrorCode.SESSION_NOT_READY,
                        f"slot {slot_index} has not been reset yet",
                        slot_index=slot_index,
                    )
                ) from exc
            images = {
                key: payload_module.encode_image(array) if array is not None else None
                for key, array in encoded["images"].items()
            }
            main_image = images.get(_CAMERA_SLOTS[0])
            wrist_image = images.get(_CAMERA_SLOTS[2])
            extra_view = images.get(_CAMERA_SLOTS[1])
            attestation = encoded["attestation"]
            step_index = encoded["step_index"]
            description = encoded["task_descriptions"]
        else:
            # ``process_isolation=False`` (default): session is a same-process
            # ``RoboCasaSession``, following the existing path of reading
            # ``session.observation`` directly, with zero extra overhead.
            if session.observation is None:
                raise RuntimeApiError(
                    make_error(
                        ErrorCode.SESSION_NOT_READY,
                        f"slot {slot_index} has not been reset yet",
                        slot_index=slot_index,
                    )
                )
            raw = session.observation
            main_image = self._encode_camera(raw, _CAMERA_SLOTS[0])
            wrist_image = self._encode_camera(raw, _CAMERA_SLOTS[2])
            extra_view = self._encode_camera(raw, _CAMERA_SLOTS[1])
            attestation = session.snapshot(include_images=False)
            step_index = session.step_index
            description = raw.get("task_descriptions") if isinstance(raw, dict) else None
        state_payload = attestation["observation"]["state"]
        # ``_json_scalar`` (session_core.py) converts 1D numpy vectors into
        # python lists; most RoboCasa state fields (e.g.
        # end_effector_position_relative) are 3/4-dimensional vectors rather
        # than scalars — filtering only by
        # ``isinstance(value, (int, float))`` would drop all of them. Here the
        # list is recursively flattened, keeping numeric leaves; the order is
        # fixed by ``sorted(state_payload)``, for generic consumers that don't
        # need named fields (consumers that need named fields read
        # ``extras["raw_state"]``, see below).
        state_vector = [
            float(leaf)
            for key in sorted(state_payload)
            for leaf in _flatten_numeric(state_payload[key])
        ]
        instruction = ""
        if isinstance(description, (list, tuple)) and description:
            instruction = str(description[0])
        return Observation(
            session_id=SessionId(""),
            episode_id=EpisodeId(0),
            step_index=step_index,
            main_image=main_image,
            wrist_image=wrist_image,
            extra_view_images=[extra_view] if extra_view is not None else [],
            state=state_vector,
            instruction=instruction,
            extras={
                "slot_index": slot_index,
                "official_success": attestation["official_success"],
                "success_latched": attestation["success_latched"],
                "authoritative_success": attestation["authoritative_success"],
                # A named state dict (see this method's docstring):
                # groot_policy.py reads STATE_FIELDS by name, without
                # depending on ``state``'s flat-vector ordering convention.
                "raw_state": dict(state_payload),
                # Gen0's strict_pure_vla attestation fields (run_rollout.py
                # checks per episode that "the task program is off and the
                # online Critic rule count is 0"). They only exist in
                # snapshot; ``StepResult.info`` is fixed-shape and generated
                # by the EnvWorker, with no room for family-private fields, so
                # this goes through the family's own ``extras`` channel.
                "task_program_enabled": bool(attestation["task_program_enabled"]),
                "critic_rule_count": int(attestation["critic_rule_count"]),
                "bundle_sha256": attestation["bundle_sha256"],
                "video_paths": dict(attestation["video_paths"]),
            },
        )

    def _encode_camera(self, raw: dict[str, Any], key: str):  # noqa: ANN201
        value = raw.get(key) if isinstance(raw, dict) else None
        if value is None:
            return None
        array = np.asarray(value)
        if array.ndim != 3 or array.shape[-1] != 3:
            return None
        from robots.robocasa.session_core import jpeg_lossy_rgb_frame

        # PNG is lossless as a *transport* codec (payload_module.encode_image),
        # but the pixels it carries must already be JPEG-quantized: the
        # pre-migration direct-HTTP path (groot_client.py -> session_core.py's
        # ``_encode_image``) only ever exposed cameras through a JPEG
        # quality=80 round-trip, and GR00T's request-level seed contract makes
        # inference deterministic given its inputs. Skipping this round-trip
        # would feed the policy exact simulator pixels the pre-migration path
        # never produced, silently diverging the action chunk (and therefore
        # the whole episode) from a same-seed pre-migration replay.
        quantized = jpeg_lossy_rgb_frame(array.astype(np.uint8, copy=False))
        return payload_module.encode_image(quantized)

    def _chunk_step_one(self, slot_index: int, actions: np.ndarray) -> ChunkOutcome:
        slot = self._require_slot(slot_index)
        if not slot.started:
            raise RuntimeApiError(
                make_error(
                    ErrorCode.SESSION_NOT_READY,
                    f"slot {slot_index} has not been reset yet",
                    slot_index=slot_index,
                )
            )
        block = np.asarray(actions, dtype=np.float32)
        if block.ndim != 2:
            raise RuntimeApiError(
                make_error(
                    ErrorCode.INVALID_ARGUMENT,
                    f"robocasa expects [chunk, action_dim] actions, got shape "
                    f"{tuple(int(dim) for dim in block.shape)}",
                )
            )
        slot.chunk_calls += 1
        episode = slot.episode
        interrupt_on_proposal = (
            bool(episode.critic_rules) or self.config.interrupt_on_proposal
            if episode.interrupt_on_proposal is None
            else bool(episode.interrupt_on_proposal)
        )
        capture_event_images = (
            self.config.capture_event_images
            if episode.capture_event_images is None
            else bool(episode.capture_event_images)
        )
        enable_task_program = (
            self.config.enable_task_program
            if episode.enable_task_program is None
            else bool(episode.enable_task_program)
        )
        result = slot.session.execute_chunk(
            {
                "actions": block.tolist(),
                # Episode-level frozen rules (delivered at reset time via
                # ResetSpec.options, see _EpisodeOptions). An earlier version
                # hardcoded an empty list here, which meant the Critic was
                # permanently silent.
                "critic_rules": list(episode.critic_rules),
                "interrupt_on_proposal": interrupt_on_proposal,
                "capture_event_images": capture_event_images,
                "enable_task_program": enable_task_program,
            }
        )
        steps = result.get("steps", [])
        # RoboCasaSession.execute_chunk overwrites self.reward wholesale, every
        # step, with the reward that the env returns directly for that step
        # (session_core.py:485 ``self.reward = float(np.asarray(reward).max())``),
        # not a cumulative value, so step_record["reward"] is already a
        # per-step increment and does not need to be differenced again.
        rewards = [float(step["reward"]) for step in steps]
        terminations = [bool(step["terminated"]) for step in steps]
        truncations = [bool(step["truncated"]) for step in steps]
        # ``PerStepRecord.info`` is the only channel for per-step audit
        # records: ``zetta.evolution``'s trajectory / failure-segment /
        # visual-evidence pipelines need per-step applied_action +
        # action_sha256 + named state (actions.jsonl / states.jsonl), while
        # ``PerStepRecord``'s named fields are only reward/terminated/truncated,
        # and ``observation`` is dropped by
        # ``include_step_observations=False`` (per-step frames carry no
        # images and would only waste bandwidth). So all per-step
        # family-private data is placed here.
        per_step_info = [
            {
                "official_success": bool(step["official_success"]),
                "success_latched": bool(step["success_latched"]),
                "proposal_rule_ids": list(step.get("proposal_rule_ids", ())),
                "applied_action": step["applied_action"],
                "action_sha256": step["action_sha256"],
                "observation_sha256": step["observation_sha256"],
                "raw_state": dict(step["state"]),
            }
            for step in steps
        ]
        # The family declares chunk_obs_layout="per_step" (correctly:
        # execute_chunk really does record per step), so
        # normalize_chunk_outcome requires non-empty step_observations; but
        # step_record itself carries no images (a bandwidth consideration in
        # session_core.py, see point 3 of the module docstring), only state,
        # so what is constructed here is a per-step Observation **without
        # images** — the real image only exists in final_observation
        # (provided by self._observation(slot_index)).
        step_observations = [
            Observation(
                session_id=SessionId(""),
                episode_id=EpisodeId(0),
                step_index=int(step["step_index"]),
                state=[
                    float(leaf)
                    for key in sorted(step["state"])
                    for leaf in _flatten_numeric(step["state"][key])
                ],
                extras={"slot_index": slot_index, "raw_state": dict(step["state"])},
            )
            for step in steps
        ]
        return normalize_chunk_outcome(
            behavior=self.behavior,
            final_observation=self._observation(slot_index),
            step_observations=step_observations,
            rewards=rewards,
            terminations=terminations,
            truncations=truncations,
            requested_horizon=int(block.shape[0]),
            per_step_info=per_step_info,
            include_step_observations=False,
            info={
                "chunk_calls": slot.chunk_calls,
                "environment_write_owner": result.get("environment_write_owner"),
                "critic_proposals": result.get("critic_proposals", []),
                "video_paths": result.get("video_paths", {}),
                # Chunk-level attestation fields: Gen0 checks per chunk that
                # "no online Critic/task program is active"; active_bundle
                # reads critic_proposals per chunk to drive
                # Role1/RecoveryController.
                "task_program_enabled": bool(result.get("task_program_enabled")),
                "critic_rule_count": int(result.get("critic_rule_count", 0)),
                "authoritative_success": bool(result.get("authoritative_success")),
                "official_success": bool(result.get("official_success")),
                "success_latched": bool(result.get("success_latched")),
                "success_first_step": result.get("success_first_step"),
            },
        )


def robocasa_current_capability() -> EnvFamilyCapability:
    """Return the robocasa-current family's capability declaration.

    Returns:
        An ``EnvFamilyCapability``; ``supports_reset_state_id=False``
        (``RoboCasaSession`` has no concept of "selecting the initial state by
        id", deciding only by ``seed``).
    """
    return capability_from_behavior(
        behavior_for(ROBOCASA_ENV_FAMILY), supports_reset_state_id=False
    )


class RobocasaCurrentFamily:
    """The robocasa-current family's ``EnvFamilyAdapter``."""

    @property
    def env_family(self) -> str:
        """Family name.

        Returns:
            ``"robocasa"``.
        """
        return ROBOCASA_ENV_FAMILY

    @property
    def capability(self) -> EnvFamilyCapability:
        """The family's capability declaration.

        Returns:
            The capability table entry.
        """
        return robocasa_current_capability()

    def create_core(self) -> RobocasaCurrentCore:
        """Create a not-yet-``build``-ed execution core.

        Returns:
            A ``RobocasaCurrentCore`` instance.
        """
        return RobocasaCurrentCore()


def register_robocasa_current_env_family(
    *, replace: bool = True
) -> RobocasaCurrentFamily:
    """Register the robocasa-current family into ``ENV_FAMILY_REGISTRY``.

    Args:
        replace: Whether to allow overwriting an existing registration under
            the same name (for tests; a local launcher may build the runtime
            repeatedly).

    Returns:
        The registered family adapter.
    """
    adapter = RobocasaCurrentFamily()
    register_env_family(adapter, replace=replace)
    return adapter
