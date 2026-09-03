# Copyright (c) 2026 Zetta Contributors
"""Pure-VLA RoboTwin rollout entrypoint.

This is the **sole** file on the ``robots``/``zetta`` side allowed to import
``rollout_runtime`` (``tests/runtime/test_layering.py``): everything else under
``robots/robotwin/`` is contract-level and simulator-free.

Scope, stated plainly: this drives one frozen episode through the shared
Rollout Runtime and writes an ``EpisodeRecord``. It is the **pure-VLA** path --
observe, infer, execute, repeat -- with no Role1 review, no runtime Critic and
no recovery controller. Those are the next increment; wiring them in before the
bimanual action and tool contracts had been exercised against real hardware
would have meant freezing a tool catalog nobody had run.

Two RoboTwin-specific facts shape the loop:

- The family is ``final_only``. One ``policy_step`` submits a whole chunk and
  returns exactly one observation, so ``StepResult.per_step`` is always
  ``None``; per-step evidence does not exist and the loop must not pretend
  otherwise. The executed horizon comes back in ``info["executed_horizon"]``.
- The seed **is** the scene. RoboTwin's ``env_seeds`` selects the initial
  configuration outright, so the episode seed rides in
  ``ResetSpec.reset_state_id`` rather than in ``seed``, and a paired same-seed
  gate reproduces the scene exactly.
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import hashlib
import io
import json
import time
import uuid
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from robots.robotwin.action_contract import ACTION_DIM
from robots.robotwin.critic_runtime import (
    critic_rules_from_payload,
    describe_dwell_semantics,
    extract_robotwin_critic_features,
    next_stall_counts,
    robotwin_state_features,
)
from robots.robotwin.recovery_controller import RecoveryController
from robots.robotwin.role1_actor import ArmAwareRole1, Role1EpisodeActor
from robots.robotwin.role1_agent import ModelBackedRole1, Role1DecisionStore
from robots.robotwin.tool_bindings import binding_for_task
from robots.robotwin.tool_catalog import DEFAULT_ROBOTWIN_TOOL_CATALOG
from rollout_runtime.api.ids import SessionId
from rollout_runtime.api.messages import (
    CreateSessionRequest,
    EnvSpecMsg,
    PolicyRequest,
    ResetSpec,
    StepResult,
)
from rollout_runtime.api.result import Err, Result
from rollout_runtime.core.payload import decode_array, encode_array
from rollout_runtime.serve.client import RemoteRuntimeClient
from zetta.evolution.critic import TemporalCritic
from zetta.evolution.jsonio import atomic_write_json, canonical_sha256, read_json
from zetta.evolution.models import CandidateBundle, EpisodeRecord
from zetta.evolution.trajectory import TrajectoryArtifacts, index_episode_trajectory
from zetta.evolution.visual_artifacts import build_episode_visual_artifacts

RUNTIME_APPLICATION_ID = "zetta-robotwin"
"""Tenant identity reported to the Gateway; the bearer token is authoritative."""


def _now() -> str:
    """Return an ISO-8601 UTC timestamp.

    Returns:
        The current time in UTC.
    """
    return datetime.now(timezone.utc).isoformat()


def _append_jsonl(path: Path, value: dict[str, Any]) -> None:
    """Append one JSON object to a JSONL file.

    Args:
        path: Target file; parent directories are created.
        value: The record to append.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")


class RuntimeOperationError(RuntimeError):
    """A Runtime operation returned an error result."""


def _single(results: Sequence[Result[Any]], *, operation: str) -> Any:
    """Unwrap a batch-of-one Runtime result.

    Args:
        results: The returned results.
        operation: Operation name, for the error message.

    Returns:
        The single successful value.

    Raises:
        RuntimeOperationError: The batch was not of size one, or carried an
            error.
    """
    if len(results) != 1:
        raise RuntimeOperationError(
            f"{operation} returned {len(results)} results, expected 1"
        )
    result = results[0]
    if isinstance(result, Err):
        raise RuntimeOperationError(f"{operation} failed: {result.error}")
    return getattr(result, "value", result)


class RolloutSession:
    """One rollout's view of the shared Runtime: a single session, unbatched.

    Attributes:
        client: The shared runtime's HTTP client.
        session_id: This rollout's session.
        lease_seconds: Requested lease length; the server may clamp it.
        lease_expiration: Current lease deadline as a unix timestamp.
        episode_started: Whether ``reset`` has completed at least once.
        closed: Whether the session has been closed.
    """

    def __init__(
        self,
        client: Any,
        session_id: SessionId,
        *,
        lease_seconds: float,
        lease_expiration: float,
    ) -> None:
        """Initialize the adapter.

        Args:
            client: A ``RemoteRuntimeClient``-shaped object.
            session_id: The created session.
            lease_seconds: Requested lease length.
            lease_expiration: Lease deadline from ``create_sessions``.
        """
        self.client = client
        self.session_id = session_id
        self.lease_seconds = float(lease_seconds)
        self.lease_expiration = float(lease_expiration)
        self.episode_started = False
        self.closed = False

    @property
    def _ids(self) -> list[SessionId]:
        """The session id as a batch of one.

        Returns:
            A single-element list.
        """
        return [self.session_id]

    async def reset(self, reset_spec: ResetSpec) -> StepResult:
        """Reset the episode.

        Args:
            reset_spec: Episode parameters. For RoboTwin the scene is selected
                by ``reset_state_id``.

        Returns:
            The reset step result.
        """
        step = _single(
            await self.client.reset(self._ids, reset_spec), operation="reset"
        )
        self.episode_started = True
        return step

    async def policy_step(self, policy_request: PolicyRequest) -> StepResult:
        """Atomic observe -> infer -> chunk_step.

        Args:
            policy_request: Inference parameters.

        Returns:
            The step result.
        """
        return _single(
            await self.client.policy_step(self._ids, policy_request),
            operation="policy_step",
        )

    async def policy_infer(
        self, policy_request: PolicyRequest
    ) -> tuple[list[list[float]], dict[str, Any]]:
        """Infer without writing to the environment (the Role1 review path).

        Args:
            policy_request: Inference parameters.

        Returns:
            ``(actions, metadata)``; ``actions`` is the nested float list that
            Role1 and ``action_step`` both expect.

        Raises:
            RuntimeOperationError: The policy returned no usable action chunk.
        """
        result = _single(
            await self.client.policy_infer(self._ids, policy_request),
            operation="policy_infer",
        )
        if result.actions is None:
            raise RuntimeOperationError("policy_infer returned no action chunk")
        block = np.asarray(decode_array(result.actions), dtype=np.float32)
        if block.ndim != 2 or block.shape[1] != ACTION_DIM:
            raise RuntimeOperationError(
                f"policy_infer returned shape {tuple(int(v) for v in block.shape)}, "
                f"expected [chunk, {ACTION_DIM}]"
            )
        metadata = {
            "source": "policy_infer",
            "horizon": int(block.shape[0]),
            "model_version": result.model_version,
        }
        return [[float(value) for value in row] for row in block], metadata

    async def action_step(self, actions: Sequence[Sequence[float]]) -> StepResult:
        """Execute an action chunk Role1 has already reviewed.

        Args:
            actions: ``[chunk, 14]`` actions.

        Returns:
            The step result.
        """
        block = np.asarray(actions, dtype=np.float32)
        return _single(
            await self.client.action_step(self._ids, [encode_array(block)]),
            operation="action_step",
        )

    async def renew_if_needed(self, *, now: float | None = None) -> None:
        """Renew the lease before it can expire mid-episode.

        Args:
            now: Injectable clock reading; defaults to ``time.time()``.
        """
        moment = time.time() if now is None else now
        margin = max(30.0, self.lease_seconds / 4.0)
        if moment < self.lease_expiration - margin:
            return
        status = _single(
            await self.client.renew_sessions(self._ids, self.lease_seconds),
            operation="renew_sessions",
        )
        self.lease_expiration = float(status.lease_expiration)

    async def close(self) -> dict[str, Any]:
        """Close the session, returning its slot to the pool.

        Returns:
            ``{"session_closed": bool, ...}``; a failure is reported rather
            than raised so cleanup cannot mask the episode's own outcome.
        """
        try:
            _single(
                await self.client.close_sessions(self._ids),
                operation="close_sessions",
            )
        except Exception as exc:  # noqa: BLE001 - cleanup must not mask the outcome
            return {
                "session_closed": False,
                "session_id": str(self.session_id),
                "failure_class": type(exc).__name__,
            }
        self.closed = True
        return {"session_closed": True, "session_id": str(self.session_id)}


def _chunk_record(step: StepResult, *, index: int) -> dict[str, Any]:
    """Summarise one chunk into an audit row.

    ``per_step`` is always ``None`` for this family, so the row is chunk-
    granular by construction. It records that explicitly rather than leaving a
    reader to infer it from an absent field.

    Args:
        step: The step result.
        index: The chunk's index within the episode.

    Returns:
        A JSON-friendly row.
    """
    info = dict(step.info or {})
    return {
        "at": _now(),
        "chunk_index": index,
        # `StepResult.executed_horizon` is the canonical count; the adapter also
        # echoes it into `info` alongside the discard bookkeeping.
        "executed_horizon": int(step.executed_horizon) or info.get("executed_horizon"),
        "requested_horizon": info.get("requested_horizon"),
        "discarded_actions": info.get("discarded_actions"),
        "reward": float(step.reward),
        "terminated": bool(step.terminated),
        "truncated": bool(step.truncated),
        "success": info.get("success"),
        "per_step_available": step.per_step is not None,
        "evidence_granularity": "chunk",
    }


def _camera_refs(observation: Any) -> list[tuple[str, Any]]:
    """Return this observation's populated camera payloads, named.

    The family delivers the right wrist as the first extra view rather than as
    a second ``wrist_image``; naming the cameras in one place keeps the video
    recorder and the reset identity from drifting apart.

    Args:
        observation: The observation to read; ``None`` yields no cameras.

    Returns:
        ``(camera name, payload ref)`` for each camera that is present.
    """

    if observation is None:
        return []
    candidates = (
        ("head", observation.main_image),
        ("left_wrist", observation.wrist_image),
        (
            "right_wrist",
            observation.extra_view_images[0]
            if observation.extra_view_images
            else None,
        ),
    )
    return [(name, ref) for name, ref in candidates if ref is not None]


def _reset_observation_identity(observation: Any) -> dict[str, Any]:
    """Bind a paired gate to the physical reset this episode actually got.

    ``gate_runner._source_parent_for_pair`` rejects a same-seed parent whose
    ``artifact_index`` carries no ``initial_observation_identity``, and
    ``gating._same_physical_reset`` treats ``state_sha256`` as the stable
    binding while camera digests are audited for renderer drift without
    deciding the comparison.

    LIBERO and RoboCasa build this from a privileged snapshot; RoboTwin
    declares no extensions, so the identity is the reset state vector the
    policy itself saw.  That is exact rather than approximate here: the seed is
    the scene, and pinning ``reset_state_id`` to it reproduces the
    configuration bit for bit.

    Args:
        observation: The reset observation.

    Returns:
        The identity mapping, or an empty mapping when there is no observation.
    """

    if observation is None:
        return {}
    state = list(observation.state) if observation.state is not None else []
    return {
        "state_sha256": canonical_sha256(state),
        "camera_sha256": {
            name: hashlib.sha256(ref.data).hexdigest()
            for name, ref in _camera_refs(observation)
        },
    }


class FrameRecorder:
    """Accumulate per-camera frames and encode one video per camera.

    RoboTwin records nothing itself -- ``robotwin/envs/vector_env.py`` sets
    ``eval_video_log = False`` unconditionally -- so the only frames that exist
    are the ones already inside each ``Observation``, PNG-encoded by the payload
    layer. This collects those bytes and encodes them at the end; nothing is
    re-rendered, so the video is exactly what the policy saw.

    Videos matter beyond being nice to look at: Stage 1 refuses a diagnosis with
    fewer than three visual evidence items
    (``lifecycle.py`` ``len(diagnosis.visual_evidence) < 3``), and
    ``build_episode_visual_artifacts`` needs at least two synchronized cameras.

    Attributes:
        root: Directory the videos are written into.
        fps: Frame rate of the written videos.
    """

    def __init__(self, root: Path, *, fps: int = 20) -> None:
        """Initialize an empty recorder.

        Args:
            root: Output directory.
            fps: Frame rate for the encoded videos.
        """
        self.root = Path(root)
        self.fps = int(fps)
        self._frames: dict[str, list[bytes]] = {}

    def capture(self, observation: Any) -> None:
        """Record one step's frames from an observation.

        Args:
            observation: The observation to read; ``None`` is ignored.
        """
        if observation is None:
            return
        for name, ref in _camera_refs(observation):
            self._frames.setdefault(name, []).append(ref.data)

    @property
    def frame_count(self) -> int:
        """Number of frames captured for the densest camera.

        Returns:
            The maximum per-camera frame count.
        """
        return max((len(rows) for rows in self._frames.values()), default=0)

    def encode(self) -> dict[str, str]:
        """Write one video per camera.

        Returns:
            Camera name -> written path. Empty when nothing was captured or no
            encoder is available; the caller degrades rather than failing, so a
            missing ffmpeg costs visual evidence but not the episode.
        """
        if not self._frames:
            return {}
        try:
            import imageio.v2 as iio
        except ImportError:
            return {}
        self.root.mkdir(parents=True, exist_ok=True)
        written: dict[str, str] = {}
        for name, payloads in sorted(self._frames.items()):
            target = self.root / f"{name}.mp4"
            try:
                images = [iio.imread(io.BytesIO(data)) for data in payloads]
                iio.mimsave(target, images, fps=self.fps, codec="libx264")
            except Exception:  # noqa: BLE001 - encoder availability varies by host
                continue
            written[name] = str(target)
        return written


def _append_action_rows(
    path: Path,
    actions: Sequence[Sequence[float]] | None,
    *,
    first_step_index: int,
    source: str,
) -> None:
    """Record one row per executed action.

    ``actions`` is ``None`` on the pure-VLA fast path: ``policy_step`` performs
    inference and execution inside the Runtime in one operation, so the client
    genuinely never sees the individual actions. That is not a gap to paper
    over -- ``_make_events`` derives ``action_count`` from the chunk rows'
    ``executed_horizon`` instead, so the trajectory index stays correct with an
    empty actions artifact.

    Args:
        path: The actions JSONL.
        actions: The executed ``[chunk, 14]`` block, or ``None``.
        first_step_index: Env step index of the block's first action.
        source: ``"vla"``, ``"recovery"`` or ``"hold"``.
    """
    if actions is None:
        return
    for offset, action in enumerate(actions):
        values = [float(value) for value in action]
        _append_jsonl(
            path,
            {
                "at": _now(),
                "step_index": first_step_index + offset,
                "source": source,
                "action_dim": len(values),
                "action": values,
            },
        )


def _state_row(
    state: Sequence[float],
    *,
    step_index: int,
    chunk_index: int,
    step: StepResult,
) -> dict[str, Any]:
    """Build one state-timeline row from a chunk-final observation.

    RoboTwin is ``final_only``, so this timeline is **chunk-final**, not
    per-step: there are ``ceil(max_steps / execute_horizon)`` rows for an
    episode, not one per simulator step. ``evidence_granularity`` records that
    in the row itself rather than leaving a reader to infer it from the row
    count.

    The row deliberately carries **no** ``task_progress`` / ``residual_to_success``
    field. ``trajectory._window_no_progress`` keys off exactly those names, and
    RoboTwin exposes no task-grounded progress scalar without a privileged
    simulator feature, which this family does not declare
    (``extensions=frozenset()`` in ``env_registry.py``). Publishing joint travel
    under a name the detector reads would make "the robot stopped moving" look
    like "the task stopped progressing"; those are different claims and the
    second one would be fabricated.

    Args:
        state: The 14-dim joint state.
        step_index: Env step index this observation belongs to.
        chunk_index: The chunk that produced it.
        step: The step result, for reward and flags.

    Returns:
        A JSON-friendly row.
    """
    values = [float(value) for value in state]
    return {
        "at": _now(),
        "step_index": int(step_index),
        "chunk_index": int(chunk_index),
        "evidence_granularity": "chunk",
        "reward": float(step.reward),
        "terminated": bool(step.terminated),
        "truncated": bool(step.truncated),
        "state": {
            # Runtime feature names, from the same function the Critic plane is
            # built with.  An earlier revision published a nested
            # ``{"left": {"gripper": ...}}`` here, which
            # lifecycle._observed_critic_features flattened to "left.gripper";
            # Stage2 bound that name and every gate rollout then died on
            # "critic feature is unavailable: left.gripper", because the Critic
            # evaluates "robotwin.arm.left.gripper".
            **robotwin_state_features(values),
            # Raw evidence.  A list is not a scalar, so this adds no name to the
            # vocabulary Stage2 may bind.
            "joint_positions": values,
        },
    }


def _optional_sha(value: str | None) -> str | None:
    """Normalise the campaign's ``"none"`` sentinel into ``None``.

    ``campaign.py`` substitutes ``{bundle_sha256}`` with ``current_bundle or
    "none"``, so a Gen0 rollout receives the literal string ``"none"``.
    ``EpisodeRecord`` validates the field as a 64-character sha256 and rejects
    it, which surfaces as every Gen0 episode failing as ``infra_invalid``.

    Args:
        value: The supplied digest, the ``"none"`` sentinel, or ``None``.

    Returns:
        The digest, or ``None``.
    """
    if value is None:
        return None
    cleaned = value.strip()
    return None if not cleaned or cleaned == "none" else cleaned


def _visual_evidence(
    *,
    video_paths: dict[str, str],
    states_path: Path,
    output_root: Path,
    segments: Sequence[Any],
) -> dict[str, Any] | None:
    """Build the multimodal evidence Stage 1 requires, when it is possible.

    ``lifecycle`` refuses a provisional diagnosis carrying fewer than three
    visual evidence items, and ``build_episode_visual_artifacts`` needs at least
    two synchronized cameras. Both conditions are met only when the episode ran
    with ``--capture-frames``; a run without it still produces a valid episode,
    it just cannot be diagnosed.

    A failure to build is reported rather than raised: visual evidence is
    downstream of the episode's own outcome, and losing it must not turn a valid
    episode into an infrastructure failure.

    Args:
        video_paths: Camera name -> encoded video path.
        states_path: The state-timeline JSONL.
        output_root: Directory to create the evidence in.
        segments: The episode's failure segments, for divergence windows.

    Returns:
        The evidence manifest, or a dict explaining why there is none.
    """
    if len(video_paths) < 2:
        return {
            "available": False,
            "reason": (
                "fewer than two synchronized cameras were recorded; rerun with "
                "--capture-frames to make this episode diagnosable"
            ),
            "camera_count": len(video_paths),
        }
    try:
        return build_episode_visual_artifacts(
            video_paths=dict(video_paths),
            states_path=states_path,
            output_root=output_root,
            divergence_steps=tuple(
                segment.earliest_divergence_step
                for segment in segments
                if segment.earliest_divergence_step is not None
            ),
        )
    except Exception as exc:  # noqa: BLE001 - evidence must not fail the episode
        return {
            "available": False,
            "reason": f"{type(exc).__name__}: {exc}",
            "camera_count": len(video_paths),
        }


async def _execute_block(
    session: RolloutSession,
    block: Sequence[Sequence[float]],
    *,
    recorder: FrameRecorder | None,
    states_path: Path,
    first_step_index: int,
    chunk_index: int,
) -> tuple[StepResult, int]:
    """Execute one decided action block and record the state timeline.

    Two submission shapes, and the choice is purely about **observation
    density**, not physics: submitting a block of N and submitting its N actions
    one at a time produce bit-identical joint states at every matched step
    (measured on ``adjust_bottle``/aloha-agilex/mplib, max delta 0.0). What
    changes is that stepwise submission yields an observation -- and therefore a
    state row and a frame -- after every simulator step, instead of one per
    chunk.

    That resolution is what the evidence machinery is built for:
    ``index_episode_trajectory``'s ``context_before``/``context_after``/
    ``no_progress_window`` all default to 8 **rows**, which at chunk granularity
    would span an entire episode.

    Args:
        session: The bound rollout session.
        block: The ``[chunk, 14]`` actions to execute.
        recorder: Frame recorder, or ``None`` to submit the block whole.
        states_path: The state-timeline JSONL.
        first_step_index: Env step index before this block.
        chunk_index: The chunk this block belongs to.

    Returns:
        ``(final_step_result, executed_action_count)``.
    """
    if recorder is None:
        step = await session.action_step(block)
        executed = int(step.executed_horizon) or len(block)
        if step.observation is not None:
            _append_jsonl(
                states_path,
                _state_row(
                    list(step.observation.state),
                    step_index=first_step_index + executed,
                    chunk_index=chunk_index,
                    step=step,
                ),
            )
        return step, executed

    step: StepResult | None = None
    executed = 0
    for action in block:
        step = await session.action_step([action])
        executed += int(step.executed_horizon) or 1
        if step.observation is not None:
            recorder.capture(step.observation)
            _append_jsonl(
                states_path,
                _state_row(
                    list(step.observation.state),
                    step_index=first_step_index + executed,
                    chunk_index=chunk_index,
                    step=step,
                ),
            )
        if step.terminated or step.truncated:
            # Stepwise submission sees termination the moment it happens.
            # A chunked submission would keep executing to the end of the block
            # -- up to horizon-1 steps past the terminal state.
            break
    if step is None:
        raise RuntimeOperationError("action block was empty")
    return step, executed


def _env_spec(args: argparse.Namespace) -> EnvSpecMsg:
    """Build the env spec that selects (or creates) this episode's env pool.

    Every field here enters ``EnvSpecMsg.digest()``, which is the pool key, so
    all rollout processes of one campaign must pass identical values or each
    will cold-start its own SAPIEN pool. Per-episode values (the seed) ride in
    ``ResetSpec`` instead.

    Args:
        args: Parsed CLI arguments.

    Returns:
        The env spec for ``create_sessions``.
    """
    return EnvSpecMsg(
        env_family="robotwin",
        env_config={
            "task_name": args.task,
            "assets_path": args.assets_path,
            "embodiment": list(args.embodiment),
            "planner_backend": args.planner_backend,
            "max_episode_steps": int(args.env_max_steps),
            "step_lim": int(args.env_max_steps),
            "execute_horizon": int(args.execute_horizon),
            "collect_wrist_camera": True,
            "collect_head_camera": True,
            "center_crop": bool(args.center_crop),
        },
        pool_size=int(args.env_pool_size),
        resource_hints={"accelerator": True},
    )


def _observation_features(
    step: StepResult,
    *,
    chunk_index: int,
    executed_horizon: int,
    previous_state: list[float] | None,
    stall_counts: dict[str, int],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build the Critic feature plane from a chunk-final step result.

    Args:
        step: The step just executed.
        chunk_index: Index of the chunk.
        executed_horizon: Simulator steps the chunk advanced.
        previous_state: The previous chunk-final joint state.
        stall_counts: Running per-arm stall counters.

    Returns:
        ``(observation, features)``; the observation is the small dict the
        Critic and the Actor both read.
    """
    state = list(step.observation.state) if step.observation is not None else None
    observation: dict[str, Any] = {"state": state}
    info = dict(step.info or {})
    features = extract_robotwin_critic_features(
        observation,
        chunk_index=chunk_index,
        executed_horizon=executed_horizon,
        reward=float(step.reward),
        terminated=bool(step.terminated),
        truncated=bool(step.truncated),
        previous_state=previous_state,
        stall_counts=stall_counts,
    )
    features["robotwin.chunk.success"] = bool(info.get("success", False))
    return observation, features


def _load_rules(path: str | None) -> list[dict[str, Any]]:
    """Read a frozen rule list from JSON, tolerating an absent path.

    Args:
        path: Path to a JSON list, or ``None``.

    Returns:
        The rule list; empty when no path was given.
    """
    if not path:
        return []
    value = read_json(Path(path))
    return list(value) if isinstance(value, list) else list(value.get("rules", []))


def _load_bundle(args: argparse.Namespace) -> CandidateBundle | None:
    """Load the frozen candidate bundle a gate arm runs under.

    A paired gate refuses a rollout command that cannot consume the bundle
    artifact (``gate_runner._command_template`` requires ``{bundle_file}``),
    because both arms must be pinned to one immutable behaviour change rather
    than to whatever rule files happen to sit on disk.

    Args:
        args: Parsed CLI arguments.

    Returns:
        The bundle, or ``None`` for a Gen0 rollout that has none.

    Raises:
        ValueError: If loose rule files are combined with a bundle, or the
            bundle does not match the digest the campaign froze.
    """

    path = getattr(args, "bundle", None)
    if not path or path == "none":
        return None
    if args.critic_rules or args.recovery_rules:
        raise ValueError(
            "--bundle is exclusive with --critic-rules/--recovery-rules: a gate "
            "arm takes its rules from the frozen bundle alone"
        )
    bundle = CandidateBundle.from_dict(read_json(Path(path)))
    expected = _optional_sha(args.bundle_sha256)
    if expected is not None and bundle.sha256 != expected:
        raise ValueError("candidate bundle SHA does not match --bundle-sha256")
    return bundle


async def _run_episode(
    args: argparse.Namespace, session: RolloutSession, *, env_spec_digest: str
) -> EpisodeRecord:
    """Drive one episode to completion.

    Two paths share this function. Without frozen critic rules it is the
    **pure-VLA** path: one atomic ``policy_step`` per chunk. With them it is the
    **reviewed** path: ``policy_infer`` produces a proposal, the Critic reads the
    chunk-final evidence, Role1 rules on it, and only then does ``action_step``
    write to the simulator. The split matters because only the second path can
    execute a recovery, and only the first can be used as a clean baseline.

    Args:
        args: Parsed CLI arguments.
        session: The bound rollout session.
        env_spec_digest: The pool key reported by ``create_sessions``.

    Returns:
        The episode record.
    """
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    chunks_path = output_dir / "chunks.jsonl"
    actions_path = output_dir / "actions.jsonl"
    states_path = output_dir / "states.jsonl"
    tools_path = output_dir / "tools.jsonl"
    decisions_path = output_dir / "role1_decisions.jsonl"
    # `_strict_jsonl` reads all four unconditionally and raises on a missing
    # file, so a run that records nothing of a given kind must still leave an
    # empty artifact rather than no artifact.
    for path in (chunks_path, actions_path, states_path, tools_path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch(exist_ok=True)

    binding = binding_for_task(args.task)
    bundle = _load_bundle(args)
    if bundle is not None:
        critic_rules = critic_rules_from_payload(
            [rule.as_dict() for rule in bundle.critic_rules]
        )
        recovery_rules = [rule.as_dict() for rule in bundle.recovery_rules]
    else:
        critic_rules = critic_rules_from_payload(_load_rules(args.critic_rules))
        recovery_rules = _load_rules(args.recovery_rules)
    reviewed = bool(critic_rules)

    critic = TemporalCritic(critic_rules) if reviewed else None
    controller = (
        RecoveryController(
            bundle_sha256=_optional_sha(args.bundle_sha256) or "0" * 64,
            audit_path=output_dir / "recovery.jsonl",
        )
        if reviewed and recovery_rules
        else None
    )
    decider: Any = ArmAwareRole1()
    if reviewed and args.role1_model:
        # A model-backed Role1 is only meaningful once a bundle is active: Gen0
        # freezes an empty critic rule set, so no proposal ever reaches Role1
        # and a provider call would buy nothing.
        decider = ModelBackedRole1(
            store=Role1DecisionStore(output_dir / "role1"),
            binding=binding,
            output_root=output_dir / "role1-invocations",
            planner_type=args.role1_planner,
            model=args.role1_model,
            reasoning_effort=args.reasoning_effort,
            max_tokens=int(args.role1_max_tokens),
            timeout_s=int(args.role1_timeout_s),
            max_turns=int(args.role1_max_turns),
        )
    actor = (
        Role1EpisodeActor(decider=decider, binding=binding, recovery=controller)
        if reviewed
        else None
    )

    recorder = (
        FrameRecorder(output_dir / "video", fps=int(args.video_fps))
        if args.capture_frames
        else None
    )

    started_at = _now()
    started_monotonic = time.monotonic()

    # RoboTwin's seed *is* the scene: pin it through reset_state_id so a paired
    # same-seed gate reproduces the configuration exactly.
    reset_step = await session.reset(
        ResetSpec(seed=int(args.seed), reset_state_id=int(args.seed))
    )
    previous_state = (
        list(reset_step.observation.state)
        if reset_step.observation is not None
        else None
    )
    initial_observation_identity = _reset_observation_identity(reset_step.observation)
    if recorder is not None:
        recorder.capture(reset_step.observation)

    policy_request = PolicyRequest(
        policy_id=args.policy_id,
        inference_parameters={"mode": "eval"},
        actions_per_chunk=(
            int(args.actions_per_chunk) if args.actions_per_chunk else None
        ),
    )

    success = False
    executed_steps = 0
    chunk_index = 0
    terminated = False
    truncated = False
    stall_counts: dict[str, int] = {}
    recovery_activations = 0
    while executed_steps < int(args.env_max_steps):
        await session.renew_if_needed()

        if not reviewed and recorder is None:
            step = await session.policy_step(policy_request)
            source = "vla"
            selected_tool = None
            # Atomic infer+execute: the actions never cross the wire.
            submitted_actions = None
            block_executed = int(step.executed_horizon) or 1
            if step.observation is not None:
                _append_jsonl(
                    states_path,
                    _state_row(
                        list(step.observation.state),
                        step_index=executed_steps + block_executed,
                        chunk_index=chunk_index,
                        step=step,
                    ),
                )
        elif not reviewed:
            # Capturing without frozen critic rules: still the pure-VLA
            # baseline, but split into infer + submit so every simulator step
            # yields an observation. Experiment A established that this does not
            # change the physics.
            proposal, _metadata = await session.policy_infer(policy_request)
            block = [
                [float(value) for value in row]
                for row in proposal[: int(args.execute_horizon)]
            ]
            step, block_executed = await _execute_block(
                session,
                block,
                recorder=recorder,
                states_path=states_path,
                first_step_index=executed_steps,
                chunk_index=chunk_index,
            )
            source = "vla"
            selected_tool = None
            submitted_actions = block
        else:
            proposal, _metadata = await session.policy_infer(policy_request)
            observation: dict[str, Any] = {"state": previous_state}
            features = extract_robotwin_critic_features(
                observation,
                chunk_index=chunk_index,
                executed_horizon=int(args.execute_horizon),
                reward=0.0,
                terminated=False,
                truncated=False,
                previous_state=previous_state,
                stall_counts=stall_counts,
            )
            proposals = critic.evaluate(features, step_index=chunk_index)
            decided = actor.decide_action(
                task=args.task,
                chunk_index=chunk_index,
                observation=observation,
                vla_actions=proposal,
                features=features,
                critic_proposals=proposals,
                recovery_rules=recovery_rules,
                environment_step=executed_steps,
            )
            if decided.decision is not None:
                _append_jsonl(
                    decisions_path,
                    {
                        "at": _now(),
                        "chunk_index": chunk_index,
                        "source": decided.source,
                        **decided.decision.public_dict(),
                    },
                )
            step, block_executed = await _execute_block(
                session,
                decided.actions,
                recorder=recorder,
                states_path=states_path,
                first_step_index=executed_steps,
                chunk_index=chunk_index,
            )
            source = decided.source
            selected_tool = decided.selected_tool
            submitted_actions = decided.actions
            if controller is not None and controller.active:
                if decided.source == "recovery":
                    recovery_activations += 1
                    controller.complete_current_step(
                        selected_tool=selected_tool,
                        environment_step=executed_steps,
                        executed_horizon=int(step.executed_horizon),
                        executed_arm=(
                            decided.commanded_arms[0]
                            if len(decided.commanded_arms) == 1
                            else None
                        ),
                    )

        row = _chunk_record(step, index=chunk_index)
        row["source"] = source
        row["selected_tool"] = selected_tool
        row["step_index"] = executed_steps
        # Stepwise submission returns `executed_horizon=1` on its last sub-step;
        # the chunk row must carry what the whole block advanced, because
        # `_make_events` derives `action_count` from exactly this field.
        row["executed_horizon"] = block_executed
        _append_jsonl(chunks_path, row)
        _append_action_rows(
            actions_path,
            submitted_actions,
            first_step_index=executed_steps,
            source=source,
        )
        _append_jsonl(
            tools_path,
            {
                "at": _now(),
                "step_index": executed_steps,
                "chunk_index": chunk_index,
                "tool": selected_tool,
                "source": source,
                "proposal_only": source != "vla",
                "environment_write": True,
            },
        )

        if step.observation is not None:
            observed = list(step.observation.state)
            features_after = extract_robotwin_critic_features(
                {"state": observed},
                chunk_index=chunk_index,
                executed_horizon=int(row["executed_horizon"] or 0),
                reward=float(step.reward),
                terminated=bool(step.terminated),
                truncated=bool(step.truncated),
                previous_state=previous_state,
                stall_counts=stall_counts,
            )
            stall_counts = next_stall_counts(features_after)
            previous_state = observed

        executed_steps += int(row["executed_horizon"] or 0) or 1
        chunk_index += 1
        if row["success"]:
            success = True
        terminated = bool(step.terminated)
        truncated = bool(step.truncated)
        if terminated or truncated:
            break

    finished_at = _now()
    artifact_index = {
        "initial_observation_identity": initial_observation_identity,
        "chunks": str(chunks_path),
        "actions": str(actions_path),
        "states": str(states_path),
        "tool_events": str(tools_path),
        "env_spec_digest": env_spec_digest,
        "tool_catalog_digest": DEFAULT_ROBOTWIN_TOOL_CATALOG.digest,
        "tool_binding_digest": binding.digest,
        "arm_scoped_tools": list(binding.arm_scoped_tool_names),
        "chunks_executed": chunk_index,
        "steps_executed": executed_steps,
        "terminated": terminated,
        "truncated": truncated,
        # Recorded on every episode: a RoboTwin diagnosis is chunk-granular and
        # must never be compared like-for-like against a per-step family's.
        "evidence_granularity": "chunk",
        "execute_horizon": int(args.execute_horizon),
        "baseline_mode": "pure_vla" if not reviewed else "critic_reviewed",
        "role1_decider": type(decider).__name__ if reviewed else None,
        "critic_rule_count": len(critic_rules),
        "recovery_rule_count": len(recovery_rules),
        "recovery_steps_executed": recovery_activations,
        # gating._candidate_intervened reads this; without it the fallback looks
        # for a "role1:"-prefixed key, which this family never writes (its key is
        # "role1_decisions"), so every candidate episode would silently attest no
        # intervention.  mechanism_diverged is then always False and the same-seed
        # gate can never pass -- while its rationale claims "candidate never
        # changed the failed parent action trajectory", which would be the wrong
        # reason.  Counting executed recoveries rather than Role1 decisions is the
        # stricter reading: a rejected proposal leaves the VLA chunk intact and
        # changes no action.
        "candidate_intervention": recovery_activations > 0,
        # The dwell unit differs from every per-step family; a reader of this
        # record must not have to infer it.
        "dwell_semantics": describe_dwell_semantics(
            execute_horizon=int(args.execute_horizon)
        ),
    }
    if reviewed:
        artifact_index["role1_decisions"] = str(decisions_path)
    record = EpisodeRecord(
        episode_id=args.episode_id or f"{args.logical_id}-{uuid.uuid4().hex[:8]}",
        logical_id=args.logical_id,
        generation=int(args.generation),
        seed=int(args.seed),
        policy_rng=int(args.policy_rng),
        bundle_sha256=_optional_sha(args.bundle_sha256),
        status="valid",
        success=bool(success),
        started_at=started_at,
        finished_at=finished_at,
        elapsed_s=time.monotonic() - started_monotonic,
        artifact_index=artifact_index,
        attempt_index=int(args.attempt_index),
    )

    # Index the trajectory and attach its failure segments: without them the
    # campaign's Cluster stage has nothing to cluster and stalls silently
    # (`cluster_failure_segments([])` returns `[]` rather than raising).
    video_paths = recorder.encode() if recorder is not None else {}
    analysis = index_episode_trajectory(
        result=record,
        artifacts=TrajectoryArtifacts(
            chunks=chunks_path,
            actions=actions_path,
            states=states_path,
            tools=tools_path,
            videos=tuple(sorted(video_paths.values())),
        ),
    )
    visual_evidence = _visual_evidence(
        video_paths=video_paths,
        states_path=states_path,
        output_root=output_dir / "visual-evidence",
        segments=analysis.segments,
    )
    record = dataclasses.replace(
        record,
        artifact_index={
            **record.artifact_index,
            "trajectory_index": (
                analysis.index.as_dict() if analysis.index is not None else None
            ),
            "failure_segment_count": len(analysis.segments),
            "videos": video_paths,
            "visual_evidence": visual_evidence,
        },
        failure_segment=analysis.segments[0] if analysis.segments else None,
        failure_segments=analysis.segments,
    )

    atomic_write_json(output_dir / "episode.json", record.as_dict())
    if getattr(args, "result_file", None):
        # How the campaign queue harvests the episode: the worker publishes the
        # record at the path the job named, and `SubprocessRolloutExecutor`
        # reads it back even if the worker later crashes.
        atomic_write_json(args.result_file, record.as_dict(), overwrite=False)
    return record


async def run(args: argparse.Namespace, *, client: Any | None = None) -> EpisodeRecord:
    """Run one episode against the shared rollout runtime.

    The session is always closed, which is what returns the env slot to the
    pool for the next rollout process.

    Args:
        args: Parsed CLI arguments.
        client: Injected ``RemoteRuntimeClient``-shaped object; tests pass a
            fake. When ``None`` this function owns the connection pool.

    Returns:
        The episode record.
    """
    owns_client = client is None
    if client is None:
        client = RemoteRuntimeClient(
            args.runtime_url,
            token=args.runtime_token,
            operation_timeout_s=args.operation_timeout_s,
            session_timeout_s=args.session_timeout_s,
        )
    session: RolloutSession | None = None
    try:
        handle = _single(
            await client.create_sessions(
                [
                    CreateSessionRequest(
                        application_id=RUNTIME_APPLICATION_ID,
                        client_session_key=(
                            f"{args.logical_id}-attempt-{args.attempt_index}"
                        ),
                        env_spec=_env_spec(args),
                        default_policy_id=args.policy_id,
                        lease_seconds=float(args.session_lease_s),
                        metadata={
                            "task": args.task,
                            "seed": int(args.seed),
                            "generation": int(args.generation),
                            "logical_id": args.logical_id,
                        },
                    )
                ]
            ),
            operation="create_sessions",
        )
        session = RolloutSession(
            client,
            handle.session_id,
            lease_seconds=float(args.session_lease_s),
            lease_expiration=float(handle.lease_expiration),
        )
        return await _run_episode(args, session, env_spec_digest=handle.env_spec_digest)
    finally:
        if session is not None and not session.closed:
            closed = await session.close()
            if closed.get("session_closed") is not True:
                _append_jsonl(
                    Path(args.output_dir) / "cleanup_errors.jsonl",
                    {"at": _now(), "failure_class": closed.get("failure_class")},
                )
        if owns_client:
            await client.aclose()


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser.

    Returns:
        The configured parser.
    """
    parser = argparse.ArgumentParser(description="Run one frozen RoboTwin rollout")
    parser.add_argument("--runtime-url", required=True)
    parser.add_argument("--runtime-token", default=None)
    parser.add_argument("--policy-id", required=True)
    parser.add_argument("--task", default="adjust_bottle")
    parser.add_argument(
        "--assets-path",
        required=True,
        help="RoboTwin repository root (not its assets/ subdirectory)",
    )
    parser.add_argument("--embodiment", nargs="+", default=["aloha-agilex"])
    parser.add_argument(
        "--planner-backend", default="mplib", choices=["mplib", "curobo"]
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--policy-rng", type=int, default=0)
    parser.add_argument("--generation", type=int, default=0)
    parser.add_argument("--logical-id", required=True)
    parser.add_argument("--episode-id", default=None)
    parser.add_argument("--attempt-index", type=int, default=0)
    parser.add_argument("--bundle-sha256", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--result-file",
        default=None,
        help="Where the campaign queue expects this episode's record",
    )
    parser.add_argument("--env-max-steps", type=int, default=200)
    parser.add_argument("--execute-horizon", type=int, default=25)
    parser.add_argument("--actions-per-chunk", type=int, default=None)
    parser.add_argument("--env-pool-size", type=int, default=4)
    parser.add_argument("--center-crop", action="store_true")
    parser.add_argument(
        "--capture-frames",
        action="store_true",
        help=(
            "Submit each action separately so every simulator step yields an "
            "observation, and encode one video per camera. Required for Stage 1: "
            "a diagnosis needs at least three visual evidence items."
        ),
    )
    parser.add_argument("--video-fps", type=int, default=20)
    parser.add_argument(
        "--role1-model",
        default=None,
        help=(
            "Model identifier for a model-backed Role1; omit to use the "
            "deterministic ArmAwareRole1 reference decider"
        ),
    )
    parser.add_argument("--role1-planner", default="codex", choices=["api", "codex"])
    parser.add_argument("--reasoning-effort", default="high")
    parser.add_argument("--role1-max-tokens", type=int, default=4096)
    parser.add_argument("--role1-timeout-s", type=int, default=600)
    parser.add_argument("--role1-max-turns", type=int, default=2)
    parser.add_argument(
        "--bundle",
        default="none",
        help=(
            "Path to the frozen CandidateBundle JSON a gate arm runs under. "
            "The campaign substitutes {bundle_file}; Gen0 receives 'none'."
        ),
    )
    parser.add_argument(
        "--critic-rules",
        default=None,
        help="JSON list of frozen critic rules; enables the reviewed path",
    )
    parser.add_argument(
        "--recovery-rules",
        default=None,
        help="JSON list of frozen recoveries; requires --critic-rules",
    )
    parser.add_argument("--session-lease-s", type=float, default=1800.0)
    parser.add_argument("--operation-timeout-s", type=float, default=600.0)
    parser.add_argument("--session-timeout-s", type=float, default=60.0)
    return parser


def main() -> int:
    """CLI entrypoint.

    Returns:
        ``0`` when the episode completed, ``1`` when it did not.
    """
    args = build_parser().parse_args()
    if int(args.execute_horizon) < 1:
        raise SystemExit("--execute-horizon must be >= 1")
    record = asyncio.run(run(args))
    print(
        json.dumps(
            {
                "episode_id": record.episode_id,
                "success": record.success,
                "elapsed_s": round(record.elapsed_s, 3),
                "action_dim": ACTION_DIM,
            },
            ensure_ascii=False,
        )
    )
    return 0 if record.status == "valid" else 1


if __name__ == "__main__":
    raise SystemExit(main())
