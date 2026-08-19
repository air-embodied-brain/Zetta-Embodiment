"""Lane state and the lockstep coalescing executor shared by the three rlinf
families (M6).

M4 has only one real family (libero), so the "slot → family env" mapping was
directly hardcoded as "one slot, one ``num_envs=1`` env". M6 adds maniskill and
robocasa and requires ``SlotGroupCoalescer`` to do real coalescing, giving rise
to two forms (``core.env_execution.CORE_FORMS``):

- ``per_slot``: each slot has its own ``num_envs=1`` family env, with slots
  fully independent (M4 semantics);
- ``lockstep_vector``: the whole pool is a single ``num_envs=pool_size`` family
  env, with slots as its lanes, and ``chunk_step`` calls for the same pool and
  tick merged into a single call.

This module holds only the **family-agnostic** half: lane state, observation
assembly (rlinf's 5-key schema is shared across all families), and the lockstep
advancement algorithm. Family differences (reset signature, cfg projection,
action preprocessing, privileged extensions) stay in each family's own
``backends/rlinf_*.py``.

The four masking semantics for lockstep (the first three finalized earlier, the
fourth added 2026-08-07; this is where they are implemented):

1. Before the first ``chunk_step``, **every lane in the pool must have already
   been reset** — a family's ``step`` advances all lanes at once, and a
   never-reset lane cannot be touched even once (an un-``reconfigure``-d libero
   subprocess would break);
2. An absent lane is carried forward using the **hold action declared by the
   family**, and ``masked_steps`` is accumulated per lane; its simulation time
   **really has** advanced, so the executor refreshes its observation cache at
   wrap-up, never pretending nothing moved;
3. **Early stop within the group**: the vector advances until either "the
   chunk is exhausted" or "any lane **newly** terminates". A lane that has
   already terminated (``frozen``) no longer triggers early stop, otherwise one
   finished lane would compress the whole pool's batch length down to 1.
4. Semantics 2 only guarantees that the **core side** honestly records the
   facts, while this function only returns a ``ChunkOutcome`` to participants,
   so the state of an absent lane must be **read back** by the worker
   (``lane_statuses`` / ``LaneStatus``, worker-side ``_sync_lockstep_pool``).
   Without this step, ``RuntimeEnvWorker``'s ``SessionSlot`` would run
   inference on a stale frame, and a masked termination's success would be
   treated as "still running". The readback merely **discovers it earlier**: if
   it fails to happen, a ``frozen`` lane that later enters as a participant
   will still get an outcome from this function with ``executed_horizon=0`` and
   the termination flag set to **true**, rather than an ``Ok`` with
   ``terminated`` false.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from rollout_runtime.api.enums import ErrorCode
from rollout_runtime.api.errors import RuntimeApiError, make_error
from rollout_runtime.api.ids import EpisodeId, SessionId
from rollout_runtime.api.messages import Observation
from rollout_runtime.core import payload as payload_module
from rollout_runtime.core.env_execution import (
    ChunkOutcome,
    EnvFamilyBehavior,
    normalize_chunk_outcome,
)

__all__ = [
    "LaneState",
    "LaneStatus",
    "LockstepStats",
    "lane_statuses",
    "observation_from_payload",
    "run_lockstep_chunk",
    "to_numpy",
    "to_scalar",
]


def to_numpy(value: Any) -> np.ndarray:
    """Normalize a torch tensor / numpy array / sequence into a numpy array.

    Both obs and reward from maniskill are torch tensors on the GPU, while
    libero / robocasa use numpy or CPU tensors, so this step is necessary for
    all three families.

    Args:
        value: The tensor or array returned by the family.

    Returns:
        A numpy array (torch goes through ``detach().cpu()``).
    """
    detach = getattr(value, "detach", None)
    if callable(detach):
        return detach().cpu().numpy()
    return np.asarray(value)


def to_scalar(value: Any, index: int = 0) -> float:
    """Extract one lane's scalar from a batched tensor.

    Args:
        value: A scalar, or a tensor / array of shape ``[num_envs]``.
        index: Lane index; ignored for scalar input.

    Returns:
        A float value; returns ``0.0`` for out-of-range or empty input.
    """
    array = to_numpy(value).reshape(-1)
    if array.size == 0:
        return 0.0
    return float(array[index if index < array.size else 0])


@dataclasses.dataclass
class LaneState:
    """The driver-side state for one slot.

    ``env_index`` / ``lane_index`` are exactly where the two forms differ: in
    ``per_slot`` form, ``env_index == slot`` and ``lane_index == 0``; in
    ``lockstep_vector`` form, ``env_index == 0`` and ``lane_index == slot``.

    Attributes:
        env_index: The index into ``_envs`` of the family env this slot uses.
        lane_index: This slot's lane index within that env.
        started: Whether it has already been reset (checked by lockstep
            precondition 1).
        step_index: Number of env steps already executed in the current
            episode.
        terminated: Termination flag (monotonic).
        truncated: Truncation flag (monotonic).
        frozen: The episode has ended and has not been reset again; no longer
            triggers early stop within the group.
        instruction: The current task instruction.
        last_observation: The most recent observation frame (``observe`` only
            reads it).
        last_main_image: The most recent main-view image
            (``libero.cached_image`` reads it).
        masked_steps: Number of steps carried forward by a hold action.
        chunk_calls: Number of ``chunk_step`` calls.
        env_steps: Cumulative number of env steps (not reset across episodes).
        resets: Number of resets.
        extras: Family-private per-lane information, fed directly into
            ``Observation.extras``.
    """

    env_index: int
    lane_index: int
    started: bool = False
    step_index: int = 0
    terminated: bool = False
    truncated: bool = False
    frozen: bool = False
    instruction: str = ""
    last_observation: Observation | None = None
    last_main_image: np.ndarray | None = None
    masked_steps: int = 0
    chunk_calls: int = 0
    env_steps: int = 0
    resets: int = 0
    extras: dict[str, Any] = dataclasses.field(default_factory=dict)

    def begin_episode(self) -> None:
        """Mark the lane as "a new episode has begun"."""
        self.started = True
        self.step_index = 0
        self.terminated = False
        self.truncated = False
        self.frozen = False
        self.resets += 1


@dataclasses.dataclass
class LockstepStats:
    """Statistics for one lockstep coalesced batch.

    Attributes:
        executed_steps: The number of steps the vector actually advanced (the
            same for all lanes in the group).
        requested_steps: The requested chunk length.
        participating_slots: The slots participating in this group.
        masked_slots: Slots carried forward by a hold action.
        early_stopped: Whether it stopped early due to "any lane newly
            terminating".
    """

    executed_steps: int = 0
    requested_steps: int = 0
    participating_slots: tuple[int, ...] = ()
    masked_slots: tuple[int, ...] = ()
    early_stopped: bool = False


@dataclasses.dataclass(frozen=True, kw_only=True)
class LaneStatus:
    """A lifecycle snapshot of one lane (the return item of
    ``EnvExecutionCore.lane_status``).

    This exists to close a gap in masking semantics 2: ``run_lockstep_chunk``
    only returns a ``ChunkOutcome`` to **participants**, while a lane carried
    forward by a hold action is likewise advanced and may likewise terminate.
    The core side honestly records this (``lane.masked_steps`` /
    ``lane.terminated`` / ``lane.frozen``), but ``RuntimeEnvWorker``'s
    ``SessionSlot`` has no way to obtain it — it only updates the one slot of
    the submitter in ``_step_slot``. Missing this readback channel has three
    consequences: the worker runs inference on a stale frame, a masked
    termination's success gets treated as "still running", and
    ``_lockstep_lane_count`` permanently over-counts one lane.

    Deliberately **does not carry an observation**: it needs to be cheap
    enough to call before every ``policy_step``, while the observation itself
    goes through the existing ``observe`` (a real family's ``observe`` only
    returns the cached object, but the worker still has to take an extra pool
    lock for it, so it decides whether to do so based on ``masked_steps``).

    Attributes:
        slot_index: Slot index.
        step_index: Number of env steps already executed in the current
            episode.
        terminated: Termination flag (monotonic).
        truncated: Truncation flag (monotonic).
        frozen: The episode has ended and has not been reset again.
        masked_steps: Cumulative number of steps carried forward by a hold
            action (not reset across episodes).
        started: Whether it has already been reset.
    """

    slot_index: int
    step_index: int
    terminated: bool
    truncated: bool
    frozen: bool
    masked_steps: int
    started: bool


def lane_statuses(lanes: Sequence[Any], slots: Sequence[int]) -> list[LaneStatus]:
    """Pack the lifecycle state of several lanes into ``LaneStatus`` (family
    agnostic).

    Uses duck typing rather than requiring ``LaneState``: libero's
    ``_LiberoSlot`` and fake's ``_FakeSlot`` are their own dataclasses (for
    historical reasons, with field names matching ``LaneState``), while
    maniskill uses ``LaneState`` directly. Only those six same-named fields are
    read here, so all three families share this one implementation.

    Args:
        lanes: **All** lanes in the pool; the index is the slot index.
        slots: The slot indices to read.

    Returns:
        Snapshots in the same order as ``slots``.

    Raises:
        RuntimeApiError: A slot index is out of range (``INVALID_ARGUMENT``).
            Pools are pre-allocated and do not grow.
    """
    statuses: list[LaneStatus] = []
    for slot_index in slots:
        if not 0 <= slot_index < len(lanes):
            raise RuntimeApiError(
                make_error(
                    ErrorCode.INVALID_ARGUMENT,
                    f"slot {slot_index} is outside the pool (size {len(lanes)}); "
                    "pools do not grow (plan D6)",
                    slot_index=slot_index,
                    pool_size=len(lanes),
                )
            )
        lane = lanes[slot_index]
        statuses.append(
            LaneStatus(
                slot_index=int(slot_index),
                step_index=int(lane.step_index),
                terminated=bool(lane.terminated),
                truncated=bool(lane.truncated),
                frozen=bool(lane.frozen),
                masked_steps=int(lane.masked_steps),
                started=bool(lane.started),
            )
        )
    return statuses


def observation_from_payload(
    *,
    payload: dict[str, Any],
    lane: LaneState,
    slot_index: int,
    env_family: str,
    core_form: str,
    state_dtype: Any = np.float32,
) -> Observation:
    """Convert a family's 5-key obs dict into a Runtime ``Observation``, and
    update the lane cache.

    The 5-key schema (``main_images`` / ``wrist_images`` / ``extra_view_images``
    / ``states`` / ``task_descriptions``) is the unified output of rlinf's
    ``EnvOutput.prepare_observations``, used by all three families — so this
    extraction logic is family agnostic, and the only difference is "which keys
    have values" (maniskill's default ``_wrap_obs`` has no wrist camera,
    robocasa has a third-person view).

    ``session_id`` / ``episode_id`` are left blank, stamped by
    ``RuntimeEnvWorker._stamp``: the execution core is entirely session
    agnostic.

    Args:
        payload: The output of a family's ``_wrap_obs``.
        lane: Lane state (updated in place).
        slot_index: Slot index, fed into ``extras``.
        env_family: Family name, fed into ``extras``.
        core_form: Execution core form, fed into ``extras`` (lets the caller
            know it is in a vector pool).
        state_dtype: The target dtype for the state vector.

    Returns:
        An ``Observation``.
    """
    index = lane.lane_index
    main = np.ascontiguousarray(to_numpy(payload["main_images"])[index])
    state_array = to_numpy(payload["states"])[index].reshape(-1).astype(state_dtype)
    descriptions = payload.get("task_descriptions") or []
    instruction = lane.instruction
    if not instruction and index < len(descriptions):
        instruction = str(descriptions[index])
    wrist_image = None
    wrist_raw = payload.get("wrist_images")
    if wrist_raw is not None:
        wrist_image = payload_module.encode_image(
            np.ascontiguousarray(to_numpy(wrist_raw)[index])
        )
    extra_views: list[Any] = []
    extra_raw = payload.get("extra_view_images")
    if extra_raw is not None:
        frames = np.ascontiguousarray(to_numpy(extra_raw)[index])
        # A family may give [H, W, C] (single image) or [views, H, W, C]
        # (multiple images).
        stack = frames[None, ...] if frames.ndim == 3 else frames
        extra_views = [
            payload_module.encode_image(np.ascontiguousarray(frame)) for frame in stack
        ]
    observation = Observation(
        session_id=SessionId(""),
        episode_id=EpisodeId(0),
        step_index=lane.step_index,
        main_image=payload_module.encode_image(main),
        wrist_image=wrist_image,
        extra_view_images=extra_views,
        state=[float(value) for value in state_array],
        instruction=instruction,
        extras={
            "slot_index": slot_index,
            **dict(lane.extras),
            "env_family": env_family,
            "core_form": core_form,
            "lane_index": index,
        },
    )
    lane.last_observation = observation
    lane.last_main_image = main
    return observation


def run_lockstep_chunk(
    *,
    behavior: EnvFamilyBehavior,
    core_form: str,
    lanes: Sequence[LaneState],
    slots: Sequence[int],
    blocks: Sequence[np.ndarray],
    action_dim: int,
    hold_action: np.ndarray,
    prepare: Callable[[np.ndarray, int], np.ndarray],
    step: Callable[[np.ndarray], tuple[Any, Any, Any, Any, Any]],
    elapsed_steps: Callable[[int], int] | None = None,
    observe: Callable[[int, dict[str, Any]], Observation],
    include_step_observations: bool = False,
    chunk_info: Callable[[int], dict[str, Any]] | None = None,
) -> tuple[list[ChunkOutcome], LockstepStats]:
    """Merge same-pool same-tick action blocks into **one** vector advance step
    (the real coalescing implementation).

    ``EnvExecutionCore.chunk_step(slots, chunk_actions)``'s signature has been
    batched since M1, so the protocol does not need to change at all:
    ``per_slot`` form loops inside the core, while ``lockstep_vector`` form
    calls this function.

    Args:
        behavior: The family declaration (needed by
            ``normalize_chunk_outcome``, the single output point).
        core_form: Execution core form, fed into ``Observation.extras`` and
            ``info``.
        lanes: **All** lanes in the pool; the index is the slot index.
        slots: The slots participating in this group.
        blocks: Actions in the same order as ``slots``, shape
            ``[chunk, action_dim]``.
        action_dim: The family's action dimension.
        hold_action: The ``[action_dim]`` hold action used for absent lanes.
        prepare: ``(batch[pool, chunk, dim], chunk_len) -> prepared``, the
            family's ``prepare_actions`` branch.
        step: ``prepared[:, i] -> (payload, reward, terminated, truncated,
            info)``, i.e. the family's ``step(actions, auto_reset=False)``.
        elapsed_steps: ``lane_index -> steps already executed``; ``None``
            means the driver accumulates it itself.
        observe: ``(slot_index, payload) -> Observation``, the family's obs
            extraction (updates the cache).
        include_step_observations: Whether to include per-step observations in
            the ``PerStepRecord``.
        chunk_info: ``slot_index -> family-private chunk-level info``.

    Returns:
        ``(ChunkOutcome list in the same order as slots, this group's stats)``.

    Raises:
        RuntimeApiError: A lane in the pool was never reset
            (``SESSION_NOT_READY``, semantics 1), the group's chunk lengths are
            inconsistent, or an action shape is wrong (``INVALID_ARGUMENT``).
    """
    pool_size = len(lanes)
    if not slots:
        raise RuntimeApiError(
            make_error(
                ErrorCode.INVALID_ARGUMENT,
                f"{behavior.env_family}: chunk_step needs at least one slot",
            )
        )
    if len(set(slots)) != len(slots):
        # Two sessions landing on the same lane is exactly the symptom of the
        # M6 pool-build race, which took a long time to diagnose. An explicit
        # error here is far better than "the later action silently overwrites
        # the earlier one".
        raise RuntimeApiError(
            make_error(
                ErrorCode.INVALID_ARGUMENT,
                f"{behavior.env_family}: one lockstep group must not repeat a slot, "
                f"got {list(slots)}; two sessions sharing a lane means the pool was "
                "built twice (see the double-build race in RUNTIME_BASELINE §10.7)",
                slots=list(slots),
            )
        )
    unstarted = [index for index, lane in enumerate(lanes) if not lane.started]
    if unstarted:
        raise RuntimeApiError(
            make_error(
                ErrorCode.SESSION_NOT_READY,
                f"{behavior.env_family}: a lockstep_vector pool steps every lane at "
                f"once, so all {pool_size} lane(s) must be reset before the first "
                f"chunk_step; lane(s) {unstarted} never were. Bind one session per "
                "slot and reset them all first.",
                env_family=behavior.env_family,
                core_form=core_form,
                pool_size=pool_size,
                unstarted_slots=unstarted,
            )
        )
    prepared_blocks = [np.asarray(block, dtype=np.float32) for block in blocks]
    for slot_index, block in zip(slots, prepared_blocks, strict=True):
        if block.ndim != 2 or block.shape[1] != action_dim:
            raise RuntimeApiError(
                make_error(
                    ErrorCode.INVALID_ARGUMENT,
                    f"{behavior.env_family} expects [chunk, {action_dim}] actions, got "
                    f"shape {tuple(int(dim) for dim in block.shape)} for slot "
                    f"{slot_index}",
                    action_dim=action_dim,
                    slot_index=slot_index,
                )
            )
    lengths = {int(block.shape[0]) for block in prepared_blocks}
    if len(lengths) != 1:
        raise RuntimeApiError(
            make_error(
                ErrorCode.INVALID_ARGUMENT,
                f"{behavior.env_family}: one lockstep group must share a single chunk "
                f"length, got {sorted(lengths)}; the coalescer groups by "
                "(pool_key, chunk_len) exactly to avoid masking in the time dimension",
                chunk_lengths=sorted(lengths),
            )
        )
    chunk_len = lengths.pop()
    if chunk_len < 1:
        raise RuntimeApiError(
            make_error(
                ErrorCode.INVALID_ARGUMENT,
                f"{behavior.env_family}: chunk_step requires at least one action",
            )
        )

    # Semantics 2: first fill the entire batch with the hold action, then
    # overlay each participant's own actions.
    batch = np.tile(
        np.asarray(hold_action, dtype=np.float32).reshape(1, 1, action_dim),
        (pool_size, chunk_len, 1),
    )
    participant_of_lane: dict[int, int] = {}
    for slot_index, block in zip(slots, prepared_blocks, strict=True):
        lane = lanes[slot_index]
        batch[lane.lane_index] = block
        participant_of_lane[lane.lane_index] = slot_index
        lane.chunk_calls += 1
    masked_slots = tuple(
        index
        for index, lane in enumerate(lanes)
        if lane.lane_index not in participant_of_lane
    )
    # A participant that should have been blocked by the worker's
    # ``_require_running_episode`` (the readback in semantics 4 exists exactly
    # to make that check authoritative). If it genuinely gets in, the whole
    # group **cannot** be aborted for it — the other lanes' actions in the same
    # group are legitimate — so it advances normally, and only at wrap-up does
    # it get an honest outcome: ``executed_horizon=0`` (it itself executed zero
    # steps) but ``terminated`` / ``truncated`` reported per the core's true
    # flags. Never return an ``Ok`` with ``terminated`` false, which would
    # discard a success that has already happened, and "success is determined
    # solely by the environment's termination signal" is a hard rule.
    frozen_participants = frozenset(
        slot_index for slot_index in slots if lanes[slot_index].frozen
    )

    actions = prepare(batch, chunk_len)
    rewards: dict[int, list[float]] = {index: [] for index in slots}
    terminations: dict[int, list[bool]] = {index: [] for index in slots}
    truncations: dict[int, list[bool]] = {index: [] for index in slots}
    per_step_info: dict[int, list[dict[str, Any]]] = {index: [] for index in slots}
    frames: dict[int, list[Observation]] = {index: [] for index in slots}
    stats = LockstepStats(
        requested_steps=chunk_len,
        participating_slots=tuple(slots),
        masked_slots=masked_slots,
    )
    payload: dict[str, Any] | None = None
    for step_index in range(chunk_len):
        payload, reward, terminated, truncated, _info = step(actions[:, step_index])
        stats.executed_steps += 1
        newly_finished = False
        for slot_index, lane in enumerate(lanes):
            lane_terminated = bool(to_scalar(terminated, lane.lane_index))
            lane_truncated = bool(to_scalar(truncated, lane.lane_index))
            if elapsed_steps is not None:
                lane.step_index = int(elapsed_steps(lane.lane_index))
            else:
                lane.step_index += 1
            lane.env_steps += 1
            if lane.lane_index not in participant_of_lane:
                # Semantics 2: state has genuinely moved, so count it honestly
                # (its observation is also refreshed at wrap-up).
                lane.masked_steps += 1
                # **The termination flag must also be honestly recorded** (a
                # real bug found by an independent audit): a lane carried
                # forward by a hold action can equally terminate, and
                # maniskill's success predicate is not monotonic (rlinf itself
                # needs ``success_once`` to remember it), so missing this step
                # would discard a success, and "success is determined solely
                # by the environment's termination signal" is a hard rule of
                # this project. Also freeze it here, otherwise it would be
                # treated as a "new termination" in the **next** group and
                # wrongly trigger an early stop for that whole group.
                lane.terminated = lane.terminated or lane_terminated
                lane.truncated = lane.truncated or lane_truncated
                if lane.terminated or lane.truncated:
                    lane.frozen = True
                continue
            if lane.frozen:
                # A lane that has already finished should not re-enter as a
                # participant (the worker-side ``_require_running_episode``
                # blocks this first). Even if it genuinely gets in, this step
                # must never be counted toward its executed_horizon — the
                # environment was driven by someone else, and it itself
                # "executed" nothing. Its outcome is corrected at wrap-up via
                # ``frozen_participants`` into an honest "0 steps + true
                # termination flags".
                continue
            was_finished = False
            lane.terminated = lane.terminated or lane_terminated
            lane.truncated = lane.truncated or lane_truncated
            rewards[slot_index].append(to_scalar(reward, lane.lane_index))
            terminations[slot_index].append(lane.terminated)
            truncations[slot_index].append(lane.truncated)
            per_step_info[slot_index].append({"step_index": lane.step_index})
            # Per-step observations are always collected: ``PerStepRecord``'s
            # ``step_index`` is derived from them, and skipping this would
            # stamp the last frame's step number onto every record. Whether
            # they are actually **returned** is controlled by
            # ``include_step_observations`` (real images are large, so they
            # are not returned by default).
            frames[slot_index].append(observe(slot_index, payload))
            if (lane.terminated or lane.truncated) and not was_finished:
                lane.frozen = True
                newly_finished = True
        if newly_finished and step_index + 1 < chunk_len:
            # Semantics 3: early stop within the group. The vector cannot stop
            # only one lane, so the whole group stops here.
            stats.early_stopped = True
            break

    # Wrap-up: refresh one real observation frame each for participants and
    # masked lanes (semantics 2's "never pretend nothing moved").
    outcomes: list[ChunkOutcome] = []
    assert payload is not None  # chunk_len >= 1, so the loop runs at least once
    for slot_index in slots:
        step_frames = frames[slot_index]
        final = step_frames[-1] if step_frames else observe(slot_index, payload)
        outcome = normalize_chunk_outcome(
            behavior=behavior,
            final_observation=final,
            step_observations=step_frames if behavior.per_step_obs_available else None,
            rewards=rewards[slot_index],
            terminations=terminations[slot_index],
            truncations=truncations[slot_index],
            requested_horizon=chunk_len,
            per_step_info=per_step_info[slot_index],
            include_step_observations=include_step_observations,
            info={
                **(chunk_info(slot_index) if chunk_info is not None else {}),
                "core_form": core_form,
                "coalesced_slots": list(slots),
                "masked_slots": list(masked_slots),
                "group_early_stopped": stats.early_stopped,
            },
        )
        if slot_index in frozen_participants:
            # Honest accounting: 0 steps executed + the core's true
            # termination flags (``normalize_chunk_outcome`` can only derive
            # flags from the per-step sequence, and this lane executed zero
            # steps, so the sequence must be empty).
            lane = lanes[slot_index]
            outcome = dataclasses.replace(
                outcome,
                terminated=bool(lane.terminated),
                truncated=bool(lane.truncated),
                info={**outcome.info, "frozen_participant": True},
            )
        outcomes.append(outcome)
    for slot_index in masked_slots:
        observe(slot_index, payload)
    return outcomes, stats
