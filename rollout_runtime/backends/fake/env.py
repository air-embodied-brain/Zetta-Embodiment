"""Fake env backend.

A deterministic state machine: ``state = f(seed, step)``,
``image = f(seed, step, y, x, c)``, becomes ``terminated`` once
``episode_length`` steps are exhausted. The same ``(seed, step)`` always
produces the same observation, so idempotency and late-result assertions can
use byte-level comparison.

Four kinds of fault injection (these alone cover "the three cancellation
states," "backpressure," and "error isolation" among the end-to-end assertions):

| Config | Behavior | Used for |
|---|---|---|
| ``fail_on_step`` | The k-th ``chunk_step`` raises ``RuntimeError`` | Error isolation (D5) |
| ``fail_on_reset`` | ``reset`` raises ``RuntimeError`` | Create/reset failure path |
| ``step_delay_seconds`` | ``time.sleep`` inside ``chunk_step`` | The "env step already started" cancellation state |
| ``hang_on_step`` | The k-th ``chunk_step`` blocks until ``release_hangs()`` or timeout | Hang isolation |

Injection granularity is at the **env spec** level: a different
``EnvSpecMsg.env_config`` -> a different digest -> a different pool -> a
different core, so "only this one session's environment explodes" needs no
special hook -- just give it an ``env_config`` with ``fail_on_step`` set (the
pool semantics get exercised as a side effect).

``chunk_step`` uses **blocking** semantics (``time.sleep`` /
``threading.Event.wait``) rather than asyncio: the EnvWorker calls the
execution core via ``asyncio.to_thread``, which both mirrors real simulation
(the libero CPU subprocess) and naturally satisfies the requirement that "an
env step that has already started cannot be rolled back" -- a step running
inside a thread cannot be cancelled.

This module depends only on stdlib + numpy (the layering constraint for
``backends/fake``).
"""

from __future__ import annotations

import dataclasses
import threading
import time
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
    capability_from_behavior,
    register_env_family,
    requested_core_form,
)

__all__ = [
    "FAKE_ENV_EXTENSIONS",
    "FAKE_ENV_FAMILY",
    "FakeEnvConfig",
    "FakeEnvCore",
    "FakeEnvFamily",
    "fake_env_behavior",
    "fake_env_capability",
    "register_fake_env_family",
]

FAKE_ENV_FAMILY = "fake"
"""The value used for ``EnvSpecMsg.env_family``."""

FAKE_ENV_EXTENSIONS = frozenset({"fake.ping", "fake.stats"})
"""Full names of the ``extension_call`` methods declared supported by the fake family.

``fake.stats`` lets tests read the env-side step counters **through the
Runtime API alone**, without reaching into the worker's internal objects.
"""


def fake_env_behavior(*, per_step_obs: bool = True) -> EnvFamilyBehavior:
    """Return the fake family's declaration across the six behavioral axes.

    ``per_step_obs=False`` lets fake impersonate a robotwin-style family that
    "only returns the final frame after submitting the whole block";
    ``tests/runtime/test_env_family_normalization.py`` relies on this to
    cover both branches of the family-normalization logic under
    **no-simulator, local-only** conditions.

    Args:
        per_step_obs: Whether to return observations step by step.

    Returns:
        The family declaration.
    """
    return EnvFamilyBehavior(
        env_family=FAKE_ENV_FAMILY,
        env_type="fake",
        reset_signature="env_idx_reset_state_ids",
        chunk_obs_layout="per_step" if per_step_obs else "final_only",
        action_layout="numpy_env_chunk_dim",
        device_kind="cpu_subproc",
        extensions=FAKE_ENV_EXTENSIONS,
        # fake also declares both core forms: real lockstep batching only has
        # a real environment on GPU machines, so letting fake support it
        # means ``run_lockstep_chunk`` (the algorithm shared by the three
        # real families) and ``SlotGroupCoalescer`` can each get one local
        # run through both transports.
        core_forms=frozenset({PER_SLOT_FORM, LOCKSTEP_VECTOR_FORM}),
        obs_extraction="FakeEnvCore._observation",
    )


@dataclasses.dataclass(kw_only=True)
class FakeEnvConfig:
    """Family-private configuration for the fake env (corresponds to
    ``EnvSpecMsg.env_config``).

    Attributes:
        action_dim: Per-step action dimensionality.
        chunk_size: Expected action chunk length (declarative only; actual
            execution follows the row count of the supplied action array).
        episode_length: Number of env steps after which ``terminated`` is judged.
        truncate_at: Number of env steps after which ``truncated`` is judged;
            ``None`` means never truncate.
        image_height: Image height.
        image_width: Image width.
        state_dim: State vector dimensionality.
        instruction: Default task instruction (overridable via
            ``ResetSpec.instruction``).
        per_step_obs: Whether to return a ``PerStepRecord`` for each step
            (the two family-behavior forms).
        return_all_frames: Whether to also place per-step observations into
            ``PerStepRecord`` (corresponds to the legacy
            ``chunk_step(return_all_frames=...)``).
        step_delay_seconds: Blocking duration for each ``chunk_step`` (slow-step injection).
        reset_delay_seconds: Blocking duration for each ``reset``.
        fail_on_step: The k-th ``chunk_step`` raises an exception (1-indexed);
            ``None`` means no injection.
        fail_on_reset: Whether ``reset`` raises an exception.
        hang_on_step: The k-th ``chunk_step`` hangs (1-indexed); ``None``
            means no injection.
        hang_timeout_seconds: Fallback upper bound on the hang; raises
            ``TimeoutError`` once elapsed. **Must be finite**: an unbounded
            block would prevent the ``asyncio.to_thread`` thread from
            exiting, and joining the ``ThreadPoolExecutor`` on interpreter
            shutdown would hang pytest.
    """

    action_dim: int = 7
    chunk_size: int = 4
    episode_length: int = 16
    truncate_at: int | None = None
    image_height: int = 32
    image_width: int = 32
    state_dim: int = 8
    instruction: str = "fake: pick the deterministic cube"
    per_step_obs: bool = True
    return_all_frames: bool = False
    step_delay_seconds: float = 0.0
    reset_delay_seconds: float = 0.0
    fail_on_step: int | None = None
    fail_on_reset: bool = False
    hang_on_step: int | None = None
    hang_timeout_seconds: float = 5.0
    core_form: str = PER_SLOT_FORM

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any] | None) -> FakeEnvConfig:
        """Construct configuration from an ``env_config`` dict.

        Unknown keys are always rejected rather than silently ignored:
        ``env_config`` feeds into ``EnvSpecMsg.digest()``, and a misspelled
        key would silently produce a new pool, which is far harder to
        diagnose than an explicit error.

        Args:
            config: Family-private configuration; ``None`` means all defaults.

        Returns:
            Structured configuration.

        Raises:
            RuntimeApiError: An unknown key is present (``INVALID_ARGUMENT``).
        """
        if not config:
            return cls()
        known = {field.name for field in dataclasses.fields(cls)}
        unknown = sorted(set(config) - known)
        if unknown:
            raise RuntimeApiError(
                make_error(
                    ErrorCode.INVALID_ARGUMENT,
                    f"unknown fake env config keys: {unknown}",
                    unknown_keys=unknown,
                    known_keys=sorted(known),
                )
            )
        return cls(**dict(config))


@dataclasses.dataclass
class _FakeSlot:
    """State of a single env slot.

    Attributes:
        seed: The random seed for this episode (determines the observation).
        task_id: Task index.
        instruction: Task instruction.
        step_index: Number of env steps executed so far.
        terminated: Termination flag.
        truncated: Truncation flag.
        started: Whether it has been reset yet.
        chunk_calls: Number of ``chunk_step`` calls (including rejected ones).
        env_steps: Cumulative number of env steps executed (not reset across episodes).
        resets: Number of ``reset`` calls.
        last_action_checksum: Sum of the most recent step's action, used to
            verify the action actually reached the environment.
    """

    seed: int = 0
    task_id: int = 0
    instruction: str = ""
    step_index: int = 0
    terminated: bool = False
    truncated: bool = False
    started: bool = False
    chunk_calls: int = 0
    env_steps: int = 0
    resets: int = 0
    last_action_checksum: float = 0.0
    # The next three fields are only meaningful in the
    # ``core_form="lockstep_vector"`` form: the lane index, "the episode has
    # already ended but has not been reset yet," and the number of steps
    # advanced under a hold action.
    lane_index: int = 0
    frozen: bool = False
    masked_steps: int = 0


def _fake_state(seed: int, step: int, dim: int) -> list[float]:
    """Generate a deterministic state vector.

    Args:
        seed: Random seed.
        step: Current step number.
        dim: Vector dimensionality.

    Returns:
        A list of ``dim`` floats in ``[0, 1)``.
    """
    return [
        round(((seed * 131 + step * 17 + index * 7) % 1000) / 1000.0, 6)
        for index in range(dim)
    ]


def _fake_image(
    seed: int, step: int, height: int, width: int, channels: int
) -> np.ndarray:
    """Generate a deterministic uint8 HWC image.

    Args:
        seed: Random seed.
        step: Current step number.
        height: Image height.
        width: Image width.
        channels: Number of channels.

    Returns:
        A uint8 array of shape ``[height, width, channels]``.
    """
    rows = np.arange(height, dtype=np.int32)[:, None, None]
    cols = np.arange(width, dtype=np.int32)[None, :, None]
    chans = np.arange(channels, dtype=np.int32)[None, None, :]
    raw = rows * 3 + cols * 5 + chans * 37 + seed * 11 + step * 23
    return (raw % 256).astype(np.uint8)


class FakeEnvCore:
    """The ``EnvExecutionCore`` implementation for the fake family.

    Entirely independent of transport / session: it only knows slot indices;
    the ``session_id`` / ``episode_id`` fields in ``Observation`` are left
    blank, to be stamped by the EnvWorker (the execution core does not know
    about sessions).
    """

    def __init__(self) -> None:
        """Initialize a not-yet-``build``-ed execution core."""
        self.config = FakeEnvConfig()
        self.env_spec: EnvSpecMsg | None = None
        self.seed_offset = 0
        self.closed = False
        self.total_chunk_calls = 0
        self.total_env_steps = 0
        self.total_masked_steps = 0
        self.coalesced_group_count = 0
        self._core_form = PER_SLOT_FORM
        self._slots: list[_FakeSlot] = []
        self._slot_mutation_lock = threading.Lock()
        self._hang_release = threading.Event()

    @property
    def core_form(self) -> str:
        """The form of this core instance (``per_slot`` or ``lockstep_vector``).

        Returns:
            The form name.
        """
        return self._core_form

    @property
    def behavior(self) -> EnvFamilyBehavior:
        """The declaration of the family this core belongs to (switches
        between the two behavioral branches based on ``per_step_obs``).

        Returns:
            The family declaration.
        """
        return fake_env_behavior(per_step_obs=self.config.per_step_obs)

    # -------------------------------------------------------------- Construction and release

    def build(
        self,
        env_spec: EnvSpecMsg,
        *,
        num_envs: int,
        seed_offset: int = 0,
        total_num_processes: int = 1,
    ) -> None:
        """Construct the env pool according to the spec (``num_envs`` is
        baked in at construction time, and the pool does not grow).

        Args:
            env_spec: Environment specification.
            num_envs: Number of slots in the pool.
            seed_offset: This rank's seed offset.
            total_num_processes: Total number of processes participating in the split.

        Raises:
            RuntimeApiError: ``num_envs`` is invalid (``INVALID_ARGUMENT``).
        """
        if num_envs < 1:
            raise RuntimeApiError(
                make_error(
                    ErrorCode.INVALID_ARGUMENT, f"num_envs must be >= 1, got {num_envs}"
                )
            )
        self.config = FakeEnvConfig.from_mapping(env_spec.env_config)
        self._core_form = requested_core_form(env_spec, self.behavior)
        self.env_spec = env_spec
        self.seed_offset = seed_offset
        self._slots = [_FakeSlot(lane_index=index) for index in range(num_envs)]
        self.closed = False
        del total_num_processes

    def close(self) -> None:
        """Release all environment resources and unblock any hung step."""
        self._hang_release.set()
        self._slots = []
        self.closed = True

    def release_hangs(self) -> None:
        """Unblock all ``chunk_step`` calls hung by ``hang_on_step`` (for test teardown)."""
        self._hang_release.set()

    # -------------------------------------------------------- Dynamic slot scaling

    def slot_count(self) -> int:
        """Return the current total number of slots
        (``core.env_execution.DynamicSlotPool``).

        Returns:
            The current slot count, including slots dynamically appended after ``build``.
        """
        return len(self._slots)

    def add_slot(self, seed_offset: int) -> int:
        """Append one independent slot (a reference implementation of
        ``DynamicSlotPool`` for tests).

        Same constraint as the real ``per_slot`` families (libero): only
        allowed in the ``per_slot`` form; ``lockstep_vector`` rejects it outright.

        Args:
            seed_offset: Seed offset for the new slot (fake does not
                actually use it for randomness, only records it for
                assertion convenience).

        Returns:
            The new slot's index (equal to the total slot count before appending).

        Raises:
            RuntimeApiError: This core is in the ``lockstep_vector`` form
                (``INVALID_ARGUMENT``).
        """
        if self._core_form == LOCKSTEP_VECTOR_FORM:
            raise RuntimeApiError(
                make_error(
                    ErrorCode.INVALID_ARGUMENT,
                    "cannot add a slot to a lockstep_vector pool: all lanes share one "
                    "vector env instance",
                    core_form=self._core_form,
                )
            )
        with self._slot_mutation_lock:
            new_index = len(self._slots)
            self._slots.append(_FakeSlot(lane_index=0, seed=seed_offset))
            return new_index

    def remove_slot(self, slot_index: int) -> None:
        """Close and remove the trailing independent slot.

        Args:
            slot_index: Index of the slot to remove; must equal the current trailing index.

        Raises:
            RuntimeApiError: The index is not the current trailing index, or
                this core is in the ``lockstep_vector`` form (``INVALID_ARGUMENT``).
        """
        if self._core_form == LOCKSTEP_VECTOR_FORM:
            raise RuntimeApiError(
                make_error(
                    ErrorCode.INVALID_ARGUMENT,
                    "cannot remove a slot from a lockstep_vector pool: all lanes "
                    "share one vector env instance",
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
                        f"got {slot_index})",
                        requested_slot=slot_index,
                        trailing_slot=last_index,
                    )
                )
            self._slots.pop(last_index)

    # ------------------------------------------------------------------ Operations

    def reset(self, slots: Sequence[int], reset_spec: ResetSpec) -> list[Observation]:
        """Reset the given slots.

        Args:
            slots: Slot indices.
            reset_spec: Episode initialization parameters.

        Returns:
            Initial observations, in the same order as ``slots``.

        Raises:
            RuntimeError: Failure injected via ``fail_on_reset``.
        """
        observations: list[Observation] = []
        for slot_index in slots:
            slot = self._require_slot(slot_index)
            if self.config.fail_on_reset:
                raise RuntimeError(
                    f"fake env injected reset failure on slot {slot_index}"
                )
            if self.config.reset_delay_seconds > 0:
                time.sleep(self.config.reset_delay_seconds)
            slot.seed = (
                (reset_spec.seed if reset_spec.seed is not None else 0)
                + self.seed_offset
                + slot_index
            )
            slot.task_id = reset_spec.task_id or 0
            slot.instruction = reset_spec.instruction or self.config.instruction
            slot.step_index = 0
            slot.terminated = False
            slot.truncated = False
            slot.frozen = False
            slot.started = True
            slot.resets += 1
            slot.last_action_checksum = 0.0
            observations.append(self._observation(slot_index))
        return observations

    def observe(self, slots: Sequence[int]) -> list[Observation]:
        """Read the current observation without changing environment state.

        Args:
            slots: Slot indices.

        Returns:
            Observations, in the same order as ``slots``.
        """
        return [self._observation(slot_index) for slot_index in slots]

    def lane_status(self, slots: Sequence[int]) -> list[LaneStatus]:
        """Read a snapshot of lane lifecycle state (the ``LaneStatusReader``
        read-back channel for masking semantics).

        Args:
            slots: Slot indices.

        Returns:
            Snapshots, in the same order as ``slots``.
        """
        return lane_statuses(self._slots, slots)

    def chunk_step(
        self, slots: Sequence[int], chunk_actions: Sequence[np.ndarray]
    ) -> list[ChunkOutcome]:
        """Execute an action chunk on the given slots.

        Args:
            slots: Slot indices.
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
        """The ``lockstep_vector`` form: advance the whole pool by one tick together.

        This runs through ``backends/rlinf_family.run_lockstep_chunk`` --
        **exactly the same algorithm as libero / maniskill / robocasa** --
        so the three masking semantics and within-group early-stopping are
        genuinely exercised locally, without needing a GPU (this is the same
        reason fake also goes through ``normalize_chunk_outcome``).

        Args:
            slots: Slots participating in this group.
            chunk_actions: Action chunks, in the same order as ``slots``.

        Returns:
            Normalized results, in the same order as ``slots``.
        """
        for slot_index in slots:
            self._require_slot(slot_index)
        self.total_chunk_calls += 1
        self.coalesced_group_count += 1

        def _step(actions: np.ndarray) -> tuple[Any, Any, Any, Any, Any]:
            """Advance the whole batch of actions by one step (one row per lane).

            Args:
                actions: ``[pool_size, action_dim]`` actions.

            Returns:
                ``(payload, rewards, terminations, truncations, info)``.
            """
            rewards: list[float] = []
            terminations: list[bool] = []
            truncations: list[bool] = []
            for lane_index, slot in enumerate(self._slots):
                row = np.asarray(actions[lane_index], dtype=np.float32)
                slot.last_action_checksum = float(np.sum(row))
                step_reward = 0.0
                if not (slot.terminated or slot.truncated):
                    slot.step_index += 1
                    self.total_env_steps += 1
                    if slot.step_index >= self.config.episode_length:
                        slot.terminated = True
                        step_reward = 1.0
                    elif (
                        self.config.truncate_at is not None
                        and slot.step_index >= self.config.truncate_at
                    ):
                        slot.truncated = True
                rewards.append(step_reward)
                terminations.append(slot.terminated)
                truncations.append(slot.truncated)
            return (
                {},
                np.asarray(rewards, dtype=np.float32),
                np.asarray(terminations, dtype=bool),
                np.asarray(truncations, dtype=bool),
                {},
            )

        outcomes, _stats = run_lockstep_chunk(
            behavior=self.behavior,
            core_form=self._core_form,
            lanes=self._slots,
            slots=list(slots),
            blocks=list(chunk_actions),
            action_dim=int(self.config.action_dim),
            hold_action=np.zeros(int(self.config.action_dim), dtype=np.float32),
            prepare=lambda batch, chunk_len: batch,
            step=_step,
            elapsed_steps=lambda lane_index: self._slots[lane_index].step_index,
            observe=lambda slot_index, _payload: self._observation(slot_index),
            include_step_observations=self.config.return_all_frames,
            chunk_info=lambda slot_index: {
                "chunk_calls": self._slots[slot_index].chunk_calls,
                "masked_steps": self._slots[slot_index].masked_steps,
            },
        )
        self.total_masked_steps = sum(slot.masked_steps for slot in self._slots)
        return outcomes

    def extension(
        self, slot: int, namespace: str, method: str, args: dict[str, Any]
    ) -> dict[str, Any]:
        """Execute a read-only extension of the fake family.

        Args:
            slot: Slot index.
            namespace: Extension namespace, must be ``"fake"``.
            method: ``"ping"`` or ``"stats"``.
            args: Method arguments (``ping`` echoes them back).

        Returns:
            Structured result.

        Raises:
            RuntimeApiError: The namespace or method is not declared
                (``UNSUPPORTED_EXTENSION``).
        """
        full_name = f"{namespace}.{method}"
        if full_name not in FAKE_ENV_EXTENSIONS:
            raise RuntimeApiError(
                make_error(
                    ErrorCode.UNSUPPORTED_EXTENSION,
                    f"fake env does not implement extension {full_name!r}",
                    namespace=namespace,
                    method=method,
                    supported=sorted(FAKE_ENV_EXTENSIONS),
                )
            )
        state = self._require_slot(slot)
        if method == "ping":
            return {"pong": True, "slot_index": slot, "echo": dict(args)}
        return {
            "slot_index": slot,
            "seed": state.seed,
            "step_index": state.step_index,
            "env_steps": state.env_steps,
            "chunk_calls": state.chunk_calls,
            "resets": state.resets,
            "terminated": state.terminated,
            "truncated": state.truncated,
            "last_action_checksum": state.last_action_checksum,
            "pool_env_steps": self.total_env_steps,
            "pool_chunk_calls": self.total_chunk_calls,
        }

    # ------------------------------------------------------------------ Internal

    def _require_slot(self, slot_index: int) -> _FakeSlot:
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

    def _observation(self, slot_index: int) -> Observation:
        slot = self._require_slot(slot_index)
        config = self.config
        main = payload_module.encode_image(
            _fake_image(
                slot.seed, slot.step_index, config.image_height, config.image_width, 3
            )
        )
        wrist = payload_module.encode_image(
            _fake_image(
                slot.seed + 7,
                slot.step_index,
                config.image_height,
                config.image_width,
                3,
            )
        )
        return Observation(
            session_id=SessionId(""),
            episode_id=EpisodeId(0),
            step_index=slot.step_index,
            main_image=main,
            wrist_image=wrist,
            state=_fake_state(slot.seed, slot.step_index, config.state_dim),
            instruction=slot.instruction,
            extras={
                "slot_index": slot_index,
                "seed": slot.seed,
                "task_id": slot.task_id,
                "last_action_checksum": slot.last_action_checksum,
            },
        )

    def _chunk_step_one(self, slot_index: int, actions: np.ndarray) -> ChunkOutcome:
        slot = self._require_slot(slot_index)
        config = self.config
        slot.chunk_calls += 1
        self.total_chunk_calls += 1
        if not slot.started:
            raise RuntimeApiError(
                make_error(
                    ErrorCode.SESSION_NOT_READY,
                    f"slot {slot_index} has not been reset yet",
                    slot_index=slot_index,
                )
            )
        if config.fail_on_step is not None and slot.chunk_calls == config.fail_on_step:
            raise RuntimeError(
                f"fake env injected failure on chunk_step #{slot.chunk_calls} "
                f"(slot {slot_index})"
            )
        if config.hang_on_step is not None and slot.chunk_calls == config.hang_on_step:
            if not self._hang_release.wait(config.hang_timeout_seconds):
                raise TimeoutError(
                    f"fake env hung on chunk_step #{slot.chunk_calls} for "
                    f"{config.hang_timeout_seconds}s (slot {slot_index})"
                )
        if config.step_delay_seconds > 0:
            time.sleep(config.step_delay_seconds)

        block = np.asarray(actions, dtype=np.float32)
        if block.ndim != 2 or block.shape[1] != config.action_dim:
            raise RuntimeApiError(
                make_error(
                    ErrorCode.INVALID_ARGUMENT,
                    f"fake env expects [chunk, {config.action_dim}] actions, "
                    f"got shape {tuple(int(dim) for dim in block.shape)}",
                    action_dim=config.action_dim,
                )
            )

        reward = 0.0
        executed = 0
        rewards: list[float] = []
        terminations: list[bool] = []
        truncations: list[bool] = []
        per_step_info: list[dict[str, Any]] = []
        frames: list[Observation] = []
        for row in block:
            if slot.terminated or slot.truncated:
                break
            slot.step_index += 1
            slot.env_steps += 1
            self.total_env_steps += 1
            executed += 1
            slot.last_action_checksum = float(np.sum(row))
            step_reward = 0.0
            if slot.step_index >= config.episode_length:
                slot.terminated = True
                step_reward = 1.0
            elif (
                config.truncate_at is not None and slot.step_index >= config.truncate_at
            ):
                slot.truncated = True
            reward += step_reward
            rewards.append(step_reward)
            terminations.append(slot.terminated)
            truncations.append(slot.truncated)
            per_step_info.append({"action_checksum": slot.last_action_checksum})
            if config.per_step_obs:
                frames.append(self._observation(slot_index))
        # Normalization always goes through the core's sole exit point;
        # fake and the real families run the same code.
        return normalize_chunk_outcome(
            behavior=self.behavior,
            final_observation=self._observation(slot_index),
            step_observations=frames if config.per_step_obs else None,
            rewards=rewards,
            terminations=terminations,
            truncations=truncations,
            requested_horizon=int(block.shape[0]),
            per_step_info=per_step_info if config.per_step_obs else None,
            include_step_observations=config.return_all_frames,
            info={"chunk_calls": slot.chunk_calls},
        )


def fake_env_capability() -> EnvFamilyCapability:
    """Return the fake family's capability declaration.

    Returns:
        ``EnvFamilyCapability``; ``per_step_obs_available`` is declared
        true, while the actual value for a single spec is still honestly
        carried by ``ChunkOutcome.per_step_obs_available``.
    """
    return capability_from_behavior(fake_env_behavior(), supports_reset_state_id=True)


class FakeEnvFamily:
    """The ``EnvFamilyAdapter`` for the fake family."""

    @property
    def env_family(self) -> str:
        """The family name.

        Returns:
            ``"fake"``.
        """
        return FAKE_ENV_FAMILY

    @property
    def capability(self) -> EnvFamilyCapability:
        """The family's capability declaration.

        Returns:
            The capability entry.
        """
        return fake_env_capability()

    def create_core(self) -> FakeEnvCore:
        """Create a not-yet-``build``-ed execution core.

        Returns:
            A ``FakeEnvCore`` instance.
        """
        return FakeEnvCore()


def register_fake_env_family(*, replace: bool = True) -> FakeEnvFamily:
    """Register the fake family into ``ENV_FAMILY_REGISTRY``.

    Defaults to ``replace=True``: the local launcher may build a runtime
    repeatedly within one process, and re-registration should not raise.

    Args:
        replace: Whether to allow overwriting a same-named registration.

    Returns:
        The registered family adapter.
    """
    adapter = FakeEnvFamily()
    register_env_family(adapter, replace=replace)
    return adapter
