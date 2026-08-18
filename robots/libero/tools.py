# Copyright (c) 2026 RPent Contributors
"""LIBERO + OpenPI tool implementation."""
from __future__ import annotations

import hashlib
import json
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Mapping

import imageio.v2 as imageio
import numpy as np
from PIL import Image

from robots.libero.env_client import LiberoEnvClient
from robots.libero.graspgen import GraspGenAdapter
from rpent.utils.logging import get_logger, get_output_dir
from rpent.utils.sam3_client import Sam3Client
from rpent.utils.vla_client import VLAClient

logger = get_logger("libero")

ARTIFACT_LAYOUT: dict[tuple[str | None, str | None, str], str] = {
    ("agentview", "low", "policy_image"): "images/image_{step:02d}.png",
    ("agentview", "low", "image"): "images_cam/image_cam_{step:02d}.png",
    ("agentview", "low", "depth"): "depths/depth_{step:02d}.npy",
    ("agentview", "low", "world"): "world/world_{step:02d}.npy",
    ("agentview", "low", "metadata"): "camera_meta.json",
    ("agentview", "high", "image"): "images_cam_hi/image_cam_hi_{step:02d}.png",
    ("agentview", "high", "world"): "world_hi/world_hi_{step:02d}.npy",
    ("wrist", "low", "image"): "images_wrist/image_wrist_{step:02d}.png",
    ("wrist", "low", "depth"): "depths_wrist/depth_wrist_{step:02d}.npy",
    ("wrist", "low", "world"): "world_wrist/world_wrist_{step:02d}.npy",
    ("wrist", "low", "metadata"): "wrist_meta/wrist_meta_{step:02d}.json",
    ("wrist", "high", "image"): "images_wrist_hi/image_wrist_hi_{step:02d}.png",
    ("wrist", "high", "world"): "world_wrist_hi/world_wrist_hi_{step:02d}.npy",
    (None, None, "states"): "states.json",
    (None, None, "episode_video"): "episode.mp4",
    (None, None, "episode_wrist_video"): "episode_wrist.mp4",
    (None, None, "episode_multiview_video"): "episode_multiview.mp4",
    (None, None, "segments"): "segments",
    (None, None, "action_videos"): "action_videos",
}
ARTIFACT_DIRECTORIES: tuple[str, ...] = (
    "images",
    "images_cam",
    "depths",
    "world",
    "images_cam_hi",
    "world_hi",
    "images_wrist",
    "depths_wrist",
    "world_wrist",
    "wrist_meta",
    "images_wrist_hi",
    "world_wrist_hi",
    "segments",
    "action_videos",
)


def artifact_path(
    output_dir: str | os.PathLike[str],
    kind: str,
    *,
    step: int | None = None,
    camera: str | None = None,
    resolution: str | None = None,
) -> Path:
    """Resolve one artifact path from the shared layout.

    ``kind`` identifies the artifact; the remaining fields are optional
    qualifiers and must be passed by keyword to avoid mixing them up.
    """
    return Path(output_dir) / _artifact_relative_path(step, camera, resolution, kind)


def _artifact_relative_path(
    step: int | None,
    camera: str | None,
    resolution: str | None,
    kind: str,
) -> str:
    template = ARTIFACT_LAYOUT[(camera, resolution, kind)]
    if step is None and "{step" in template:
        raise ValueError(f"{kind} artifact requires a step")
    return template.format(step=step)


def _normalize_xyz(xyz):
    """Coerce an LLM-supplied xyz into a length-3 list[float]."""
    if not isinstance(xyz, (list, tuple)) or len(xyz) != 3:
        raise ValueError(
            'xyz must be a JSON array of three numbers, e.g. "xyz":[-0.05,0,0.3]'
        )
    return [float(v) for v in xyz]


def _semantic_grasp_retained(state: Mapping[str, Any]) -> bool:
    """Require the current authoritative grasp and, when present, retention."""

    grasped = bool(
        state.get("privileged.task.manipulated_object.grasped", False)
    )
    retained = state.get("privileged.task.manipulated_object.retained")
    return grasped and (retained is None or bool(retained))


def _side_by_side(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Return a uint8, equal-height two-camera mosaic."""
    from PIL import Image

    lhs = np.asarray(left, dtype=np.uint8)
    rhs = np.asarray(right, dtype=np.uint8)
    if lhs.ndim != 3 or rhs.ndim != 3:
        raise ValueError("video frames must be HxWxC")
    target_h = int(lhs.shape[0])
    if rhs.shape[0] != target_h:
        target_w = max(1, round(rhs.shape[1] * target_h / rhs.shape[0]))
        rhs = np.asarray(
            Image.fromarray(rhs).resize((target_w, target_h), Image.Resampling.BILINEAR)
        )
    return np.ascontiguousarray(np.concatenate([lhs, rhs], axis=1))


def _owned_rgb_frame(value: Any, *, name: str) -> np.ndarray:
    """Detach one camera frame from a renderer-owned observation buffer."""

    frame = np.array(value, dtype=np.uint8, order="C", copy=True)
    if frame.ndim != 3 or frame.shape[-1] != 3:
        raise ValueError(f"{name} video frame must be HxWx3, got {frame.shape}")
    if not frame.flags.owndata or not frame.flags.c_contiguous:
        raise RuntimeError(f"{name} video frame must own C-contiguous memory")
    return frame


def _validate_owned_video_frames(
    agentview: list[np.ndarray], wrist: list[np.ndarray]
) -> dict[str, Any]:
    """Validate alignment and reject synchronized framebuffer corruption."""

    errors: list[dict[str, Any]] = []
    if len(agentview) != len(wrist):
        errors.append(
            {
                "kind": "source_frame_steps_unaligned",
                "agentview": len(agentview),
                "wrist": len(wrist),
            }
        )
    dimensions: dict[str, list[int]] = {}
    for name, frames in (("agentview", agentview), ("wrist", wrist)):
        shapes = {tuple(frame.shape) for frame in frames}
        if len(shapes) > 1:
            errors.append(
                {"kind": "source_frame_dimensions_changed", "stream": name}
            )
        if shapes:
            dimensions[name] = list(next(iter(shapes)))
    synchronized_discontinuities = []
    for step in range(1, min(len(agentview), len(wrist))):
        scores = {
            "agentview": float(
                np.abs(
                    agentview[step].astype(np.int16)
                    - agentview[step - 1].astype(np.int16)
                ).mean()
            ),
            "wrist": float(
                np.abs(
                    wrist[step].astype(np.int16) - wrist[step - 1].astype(np.int16)
                ).mean()
            ),
        }
        if all(score >= 40.0 for score in scores.values()):
            synchronized_discontinuities.append(
                {"step": step, "transition_scores": scores}
            )
    if synchronized_discontinuities:
        errors.append(
            {
                "kind": "source_frame_corruption",
                "first_step": synchronized_discontinuities[0]["step"],
                "count": len(synchronized_discontinuities),
            }
        )
    return {
        "status": "valid" if not errors else "invalid",
        "aligned_steps": min(len(agentview), len(wrist)),
        "dimensions": dimensions,
        "synchronized_discontinuity_threshold": 40.0,
        "errors": errors,
    }


class LiberoPrimitives:
    """Wraps a single-env LIBERO-shaped env + VLA policy with primitive-
    level methods.

    ``pi0_pick`` and ``pi0_doubled`` override ``obs['task_descriptions']``
    with a sub-instruction. ``move_to`` and friends are scripted (no VLM
    call) and drive the underlying OSC controller directly.
    """

    def __init__(
        self,
        env: LiberoEnvClient,
        model: VLAClient,
        sam3_client: Sam3Client,
        *,
        allow_privileged_actions: bool = False,
    ):
        self.env = env
        self.model = model
        self._sam3_client = sam3_client
        self._last_obs = None
        self._last_obs_eef_pos = None
        self._last_obs_eef_z = None
        self._last_obs_gripper = None
        # Per-env-step frame buffer for diagnostic video rendering.
        # Toggled via start_recording() / stop_recording_and_save().
        self._recording = False
        self._frames = []
        self._wrist_frames = []
        self._last_vla_diagnostics: dict[str, Any] = {}
        self._critic_rules: list[dict[str, Any]] = []
        self._critic_frozen = False
        self._critic_configured = False
        self._suppressed_recovery_rule_ids: set[str] = set()
        self._last_critic_proposals: list[dict[str, Any]] = []
        self._last_chunk_info: dict[str, Any] = {}
        self._policy_rng: int | None = None
        self._policy_call_index = 0
        self._allow_privileged_actions = bool(allow_privileged_actions)
        # Motion proposal state is episode-local and never exposed as raw
        # simulator geometry to the VLA/Critic plane.  Action wrappers below
        # use it to enforce candidate freshness across Role1 decisions.
        self._graspgen = GraspGenAdapter()
        self._motion_candidate: dict[str, Any] | None = None
        self._motion_history: list[dict[str, Any]] = []

    def configure_policy_rng(self, policy_rng: int) -> None:
        if self._policy_rng is not None and self._policy_rng != int(policy_rng):
            raise ValueError("LIBERO policy RNG cannot change within an episode")
        self._policy_rng = int(policy_rng)

    def _policy_parameters(
        self, parameters: dict[str, Any] | None
    ) -> dict[str, Any] | None:
        values = dict(parameters or {})
        if self._policy_rng is not None:
            digest = hashlib.sha256(
                f"{self._policy_rng}:{self._policy_call_index}".encode()
            ).digest()
            expected = int.from_bytes(digest[:4], "big") & 0x7FFFFFFF
            declared = values.get("seed")
            if declared is not None and int(declared) != expected:
                raise ValueError("planner cannot override the Harness policy RNG")
            values["seed"] = expected
            self._policy_call_index += 1
        return values or None

    def configure_critic(self, rules: list[dict[str, Any]]) -> None:
        """Freeze the episode Critic without enabling an empty Gen0 set.

        An empty rule set is an attested pure-VLA baseline.  It must use the
        ordinary environment path instead of collecting privileged state on
        every step through ``critic_chunk_step``.
        """

        if self._critic_frozen and rules != self._critic_rules:
            raise ValueError("LIBERO critic rules cannot change within an episode")
        self._critic_rules = json.loads(json.dumps(rules))
        self._critic_frozen = True
        self._critic_configured = bool(self._critic_rules)

    @contextmanager
    def suppress_recovery_rules(self, rule_ids: set[str] | frozenset[str]):
        """Filter the rule that authorized the active recovery step.

        The server still evaluates the frozen Critic on every physical action;
        suppression only prevents the same proposal from recursively
        interrupting the recovery that was authorized to replace it.
        """

        previous = self._suppressed_recovery_rule_ids
        self._suppressed_recovery_rule_ids = previous | {
            str(rule_id) for rule_id in rule_ids
        }
        try:
            yield
        finally:
            self._suppressed_recovery_rule_ids = previous

    def _active_critic_rules(self) -> list[dict[str, Any]]:
        # Keep the server-side critic frozen.  Recovery suppression is a
        # rollout-local filter applied after the audited RPC returns.
        return self._critic_rules

    def _filter_suppressed_proposals(
        self, proposals: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        if not self._suppressed_recovery_rule_ids:
            return proposals
        return [
            proposal
            for proposal in proposals
            if str(proposal.get("rule_id", ""))
            not in self._suppressed_recovery_rule_ids
        ]

    def critic_interrupted(self) -> bool:
        """Whether the latest physical action/chunk raised a Critic proposal.

        Recovery primitives use this signal to yield control back to Role1
        immediately.  Merely evaluating the Critic on every action is not
        sufficient: a multi-step primitive must not silently continue after
        the first detected deviation.
        """

        return bool(self._last_critic_proposals)

    def begin_recovery_step(self) -> None:
        """Acknowledge the proposal that authorized the next recovery action.

        The proposal belongs to the preceding VLA chunk. Recovery primitives
        must be allowed to execute their first bounded action, while each
        subsequent action still repopulates this field if the frozen Critic
        raises a new proposal.
        """

        self._last_critic_proposals = []

    def start_recording(self):
        self._recording = True
        self._frames = []
        self._wrist_frames = []

    def record_frame(self, obs):
        """Append synchronized agentview and wrist frames from ``obs``."""
        # MuJoCo may reuse the camera framebuffer on the next render.  A view
        # that is merely contiguous can therefore still be overwritten before
        # imageio encodes it; force an owned snapshot for every step.
        self._frames.append(_owned_rgb_frame(obs["main_images"], name="agentview"))
        wrist = obs.get("wrist_images")
        if wrist is not None:
            arr = np.asarray(wrist)
            if arr.ndim == 3 and arr.shape[-1] == 3:
                self._wrist_frames.append(_owned_rgb_frame(arr, name="wrist"))

    def recorded_frame_count(self) -> int:
        return len(self._frames)

    def stop_recording_and_save(self, path: str, fps: int = 20):
        """Save owned, aligned camera videos plus a source-frame manifest."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        n = len(self._frames)
        outputs: dict[str, Any] = {"path": path, "n_frames": n, "fps": fps}
        if n > 0:
            frame_root = Path(path).parent / "raw" / "images"
            frame_root.mkdir(parents=True, exist_ok=True)
            wrist_frames = self._wrist_frames[:n]
            validation = _validate_owned_video_frames(self._frames, wrist_frames)
            if validation["status"] != "valid":
                raise RuntimeError(
                    "LIBERO video source validation failed: "
                    f"{validation['errors']}"
                )
            for step, frame in enumerate(self._frames):
                image_path = frame_root / f"frame_step{step:06d}_agentview.jpg"
                Image.fromarray(frame, mode="RGB").save(
                    image_path, format="JPEG", quality=90, subsampling=0
                )
            imageio.mimwrite(path, self._frames, fps=fps, codec="libx264")
            base = Path(path)
            wrist_path = base.with_name(f"{base.stem}_wrist{base.suffix}")
            mosaic_path = base.with_name(f"{base.stem}_multiview{base.suffix}")
            for step, frame in enumerate(wrist_frames):
                image_path = frame_root / f"frame_step{step:06d}_wrist.jpg"
                Image.fromarray(frame, mode="RGB").save(
                    image_path, format="JPEG", quality=90, subsampling=0
                )
            if wrist_frames:
                imageio.mimwrite(wrist_path, wrist_frames, fps=fps, codec="libx264")
                outputs["wrist_path"] = str(wrist_path)
                outputs["wrist_frames"] = len(wrist_frames)
                paired = min(n, len(wrist_frames))
                mosaics = [
                    _side_by_side(self._frames[i], wrist_frames[i])
                    for i in range(paired)
                ]
                imageio.mimwrite(mosaic_path, mosaics, fps=fps)
                outputs["multiview_path"] = str(mosaic_path)
                outputs["multiview_frames"] = paired
            manifest = {
                "schema_version": "rpent-libero-video-artifacts-v2",
                "status": "complete",
                "frame_alignment": "frame index equals post-step index; frame 0 is reset",
                "frame_count": n,
                "frame_rate": int(fps),
                "raw_frame_directory": str(frame_root),
                "streams": {
                    "agentview": {"frame_count": n},
                    "wrist": {"frame_count": len(wrist_frames)},
                },
                "source_validation": validation,
            }
            manifest_path = frame_root.parent / "VIDEO_SOURCE_MANIFEST.json"
            manifest_path.write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            outputs["source_manifest"] = str(manifest_path)
        self._recording = False
        self._frames = []
        self._wrist_frames = []
        return outputs

    def save_frame_slice(self, start: int, path: str, fps: int = 20):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        frames = list(self._frames[int(start):])
        n = len(frames)
        if n > 0:
            imageio.mimwrite(path, frames, fps=fps)
        return {"path": path, "n_frames": n, "fps": fps}

    def set_obs(self, obs):
        self._last_obs = obs
        states_arr = np.asarray(obs["states"])
        self._last_obs_eef_pos = np.asarray(states_arr[:3], dtype=np.float32)
        self._last_obs_eef_z = float(self._last_obs_eef_pos[2])
        # robosuite 2f85: qpos[6] in [~0, ~0.04], qpos[7] in [~-0.04, ~0].
        # Use |qpos[6]| + |qpos[7]| ≈ finger separation proxy.
        # When open ≈ 0.08; when closed ≈ 0.
        gp = np.asarray(states_arr[6:8], dtype=np.float32)
        self._last_obs_gripper = float(abs(gp[0]) + abs(gp[1]))

    def reset(self):
        obs, info = self.env.reset()
        self.set_obs(obs)
        return self._last_obs, info

    def _step_env(self, action) -> None:
        """Execute one action and update the cached observation and video."""
        if self._critic_configured:
            frames, _r, _t, _tr, _i = self.env.critic_chunk_step(
                [action],
                critic_rules=self._active_critic_rules(),
                interrupt_on_proposal=False,
                return_all_frames=True,
            )
            obs = frames[-1]
            self._last_chunk_info = dict(_i) if isinstance(_i, dict) else {}
            self._last_critic_proposals = self._filter_suppressed_proposals(
                list(self._last_chunk_info.get("critic_proposals", ()))
            )
        elif hasattr(self.env, "chunk_step"):
            frames, _r, _t, _tr, _i = self.env.chunk_step(
                [action], return_all_frames=True
            )
            obs = frames[-1]
            self._last_chunk_info = dict(_i) if isinstance(_i, dict) else {}
            self._last_critic_proposals = []
        else:
            obs, _r, _t, _tr, _i = self.env.step(action)
            self._last_chunk_info = dict(_i) if isinstance(_i, dict) else {}
            self._last_critic_proposals = []
        self.set_obs(obs)
        if self._recording:
            self.record_frame(obs)

    def _vlm_chunk(
        self,
        instruction: str,
        *,
        mode: str = "eval",
        actions_per_chunk: int | None = None,
        translation_scale: float = 1.0,
        rotation_scale: float = 1.0,
        gripper_scale: float = 1.0,
        action_clip: float = 1.0,
        inference_parameters: dict[str, Any] | None = None,
    ):
        """Run one VLA inference and a planner-controlled receding horizon."""
        inference_parameters = self._policy_parameters(inference_parameters)
        # Stash & override task_descriptions (one prompt).
        original_td = self._last_obs.get("task_descriptions")
        self._last_obs["task_descriptions"] = instruction
        self._last_obs.setdefault("extra_view_images", None)

        try:
            actions, metadata = self.model.predict_action_batch(
                self._last_obs,
                mode=mode,
                inference_parameters=inference_parameters,
            )
        finally:
            # The prompt is a per-call planner override, not environment state.
            # Restore it even when a backend rejects parameters or inference
            # fails so a later recovery primitive cannot inherit stale text.
            if original_td is None:
                self._last_obs.pop("task_descriptions", None)
            else:
                self._last_obs["task_descriptions"] = original_td
        actions = np.asarray(actions, dtype=np.float32)
        if actions.ndim != 2 or actions.shape[1] != 7:
            raise ValueError(
                f"VLA actions must have shape [horizon, 7]; got {actions.shape}"
            )
        if not np.isfinite(actions).all():
            raise ValueError("VLA actions must contain only finite values")
        predicted_horizon = int(actions.shape[0])
        if actions_per_chunk is not None:
            requested = int(actions_per_chunk)
            if requested < 1:
                raise ValueError("actions_per_chunk must be positive")
            actions = actions[:requested]
        if actions.shape[0] == 0:
            raise ValueError("VLA returned an empty action horizon")
        clip = float(action_clip)
        if not 0.05 <= clip <= 1.0:
            raise ValueError("action_clip must be in [0.05, 1.0]")
        actions = actions.copy()
        actions[:, :3] *= float(translation_scale)
        actions[:, 3:6] *= float(rotation_scale)
        actions[:, 6] *= float(gripper_scale)
        actions = np.clip(actions, -clip, clip)
        # actions: [chunk_size, action_dim] The whole chunk
        # runs in a single env.chunk_step RPC; the env owns the per-step
        # loop server-side.
        if self._critic_configured:
            chunk_obs, _r, _t, _tr, _i = self.env.critic_chunk_step(
                actions,
                critic_rules=self._active_critic_rules(),
                interrupt_on_proposal=True,
                return_all_frames=self._recording or self.env.return_all_frames,
            )
            obs = (
                chunk_obs[-1]
                if self._recording or self.env.return_all_frames
                else chunk_obs
            )
        elif not self._recording:
            chunk_obs,  _r, _t, _tr, _i = self.env.chunk_step(actions)
            obs = chunk_obs[-1] if self.env.return_all_frames else chunk_obs
        else:
            chunk_obs,  _r, _t, _tr, _i = self.env.chunk_step(
                actions, return_all_frames=True
            )
            obs = chunk_obs[-1]
        if self._recording:
            for obs in chunk_obs:
                self.record_frame(obs)
            obs = chunk_obs[-1]
        self.set_obs(obs)
        # Current LiberoEnvFacade returns one termination value per executed
        # action plus an explicit horizon. Older/external clients can still
        # return a scalar bool after executing the complete requested chunk;
        # treating that scalar's size as the horizon would incorrectly report
        # one action. Prefer the explicit value, then the vector length, and
        # finally the requested chunk length for the legacy scalar contract.
        info_horizon = _i.get("executed_horizon") if isinstance(_i, dict) else None
        term_values = np.asarray(_t)
        if info_horizon is not None:
            executed_horizon = int(info_horizon)
        elif term_values.ndim > 0:
            executed_horizon = int(term_values.size)
        else:
            executed_horizon = int(actions.shape[0])
        self._last_vla_diagnostics = {
            "predicted_horizon": predicted_horizon,
            "executed_horizon": executed_horizon,
            "translation_scale": float(translation_scale),
            "rotation_scale": float(rotation_scale),
            "gripper_scale": float(gripper_scale),
            "action_clip": clip,
            "mode": mode,
            "backend_metadata": metadata,
        }
        self._last_chunk_info = dict(_i) if isinstance(_i, dict) else {}
        self._last_critic_proposals = self._filter_suppressed_proposals(
            list(self._last_chunk_info.get("critic_proposals", ()))
        )
        return self._last_obs

    def vla_execute(
        self,
        prompt: str,
        *,
        max_chunks: int = 8,
        actions_per_chunk: int | None = 5,
        mode: str = "eval",
        translation_scale: float = 1.0,
        rotation_scale: float = 1.0,
        gripper_scale: float = 1.0,
        action_clip: float = 1.0,
        stop_on_success: bool = True,
        stop_on_truncation: bool = True,
        inference_parameters: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Execute a planner-authored VLA subtask with explicit control knobs.

        The model is re-queried from the latest observation after every chunk.
        This is intentionally a generic primitive: it does not encode object,
        fixture, or benchmark-task recipes.
        """
        prompt = str(prompt).strip()
        if not prompt:
            raise ValueError("prompt must not be empty")
        max_chunks = int(max_chunks)
        if not 1 <= max_chunks <= 64:
            raise ValueError("max_chunks must be in [1, 64]")
        if actions_per_chunk is not None and not 1 <= int(actions_per_chunk) <= 32:
            raise ValueError("actions_per_chunk must be null or in [1, 32]")
        for label, value in (
            ("translation_scale", translation_scale),
            ("rotation_scale", rotation_scale),
            ("gripper_scale", gripper_scale),
        ):
            if not 0.0 <= float(value) <= 2.0:
                raise ValueError(f"{label} must be in [0, 2]")

        chunks: list[dict[str, Any]] = []
        for idx in range(max_chunks):
            self._vlm_chunk(
                prompt,
                mode=mode,
                actions_per_chunk=actions_per_chunk,
                translation_scale=translation_scale,
                rotation_scale=rotation_scale,
                gripper_scale=gripper_scale,
                action_clip=action_clip,
                inference_parameters=inference_parameters,
            )
            chunks.append({"chunk": idx + 1, **self._last_vla_diagnostics})
            if self.critic_interrupted():
                break
            if stop_on_success and self.env.episode_terminated:
                break
            if stop_on_truncation and self.env.episode_truncated:
                break

        return {
            "name": "vla_execute",
            "instruction": prompt,
            "chunks_used": len(chunks),
            "max_chunks": max_chunks,
            "actions_executed": sum(c["executed_horizon"] for c in chunks),
            "libero_terminated": self.env.episode_terminated,
            "episode_truncated": self.env.episode_truncated,
            "controls": {
                "actions_per_chunk": actions_per_chunk,
                "mode": mode,
                "translation_scale": float(translation_scale),
                "rotation_scale": float(rotation_scale),
                "gripper_scale": float(gripper_scale),
                "action_clip": float(action_clip),
                "inference_parameters": inference_parameters or {},
            },
            "chunk_diagnostics": chunks,
        }

    def pi0_pick(
        self,
        prompt: str,
        *,
        max_chunks: int = 8,
        lift_thresh: float = 0.05,
        gripper_closed_thresh: float = 0.06,
    ) -> dict:
        """Closed-loop Pi0.5 pick driven by ``prompt`` as the VLA instruction.

        Success := eef lifted by >= ``lift_thresh`` AND gripper_opening
        below ``gripper_closed_thresh``. Terminates early on libero
        ``terminated`` (official success) or ``max_chunks``.
        """
        instr = prompt
        max_chunks = int(max_chunks)
        if not 1 <= max_chunks <= 8:
            raise ValueError("pi0_pick max_chunks must be in [1, 8]")
        start_z = self._last_obs_eef_z
        peak_z = start_z
        min_z = start_z
        # Track ascent AFTER min_z has been observed — descent then re-ascent
        # is the actual "lift" signal, distinct from raw |peak - min| which
        # also fires at the BOTTOM of the descent.
        post_min_peak_z = start_z
        min_grip = self._last_obs_gripper
        last_grip = min_grip
        descent_done = False
        success = False
        chunks_used = 0

        for c in range(max_chunks):
            self._vlm_chunk(instr)
            chunks_used = c + 1
            z = self._last_obs_eef_z
            grip = self._last_obs_gripper
            peak_z = max(peak_z, z)
            if z < min_z:
                min_z = z
                post_min_peak_z = z  # reset after a new deeper min
            else:
                post_min_peak_z = max(post_min_peak_z, z)
            if (start_z - min_z) >= 0.10:  # descended ≥ 10 cm — committed to grasp
                descent_done = True
            min_grip = min(min_grip, grip)
            last_grip = grip
            if self.critic_interrupted():
                break
            ascended = (post_min_peak_z - min_z) >= lift_thresh
            closed = grip < gripper_closed_thresh
            if descent_done and ascended and closed:
                success = True
                break
            if self.env.episode_terminated or self.env.episode_truncated:
                success = self.env.episode_terminated
                break

        return {
            "name": "pick",
            "instruction": instr,
            "success": success,
            "chunks_used": chunks_used,
            "max_chunks": max_chunks,
            "peak_lift_m": post_min_peak_z - min_z,  # actual post-descent ascent
            "min_gripper_opening": min_grip,
            "final_gripper_opening": last_grip,
            "libero_terminated": self.env.episode_terminated,
            "diagnostics": {
                "start_eef_z": round(start_z, 4),
                "peak_eef_z": round(peak_z, 4),
                "min_eef_z": round(min_z, 4),
                "post_min_peak_z": round(post_min_peak_z, 4),
                "descent_m": round(start_z - min_z, 4),
                "post_min_ascent_m": round(post_min_peak_z - min_z, 4),
                "descent_done": descent_done,
                "lift_thresh": lift_thresh,
                "gripper_closed_thresh": gripper_closed_thresh,
            },
        }

    def pi0_doubled(
        self,
        prompt: str,
        *,
        max_chunks: int = 20,
    ) -> dict:
        """Closed-loop Pi0.5 contact skill.

        Intended for non-pick contact interactions such as turning knobs,
        toggling stoves, or short pushes. Success is the official LIBERO
        termination predicate, not a private object-pose oracle.
        """
        instr = prompt
        task_success = False
        chunks_used = 0

        for c in range(max_chunks):
            self._vlm_chunk(instr)
            chunks_used = c + 1
            if self.critic_interrupted():
                break
            if self.env.episode_terminated or self.env.episode_truncated:
                task_success = self.env.episode_terminated
                break

        return {
            "name": "pi0_doubled",
            "instruction": instr,
            "success": task_success,
            "task_success": task_success,
            "contact_skill_executed": chunks_used > 0,
            "chunks_used": chunks_used,
            "max_chunks": max_chunks,
            "libero_terminated": self.env.episode_terminated,
            "diagnostics": {
                "mode": "contact_skill_success_by_libero_terminated",
                "success_meaning": (
                    "`success` mirrors official LIBERO task termination only; "
                    "for intermediate contact skills, inspect image/state evidence."
                ),
            },
        }

    def move_to(
        self,
        xyz,
        *,
        max_steps: int = 80,
        gripper: float = -1.0,
        step_clip: float = 0.025,
        tol: float = 0.012,
        action_scale: float = 0.05,
        target_yaw: float | None = None,
        yaw_step_clip: float = 0.10,
    ) -> dict:
        """Scripted EEF servo to a world-frame target xyz.

        Sends 7-D delta actions; the env's underlying OSC_POSE controller
        interprets ``action[:3] ∈ [-1, 1]`` as a per-step desired delta scaled
        by ``action_scale`` (so ``action=1.0`` -> ~5 cm per env step).
        ``gripper``: +1.0 keeps it closed (holding object), -1.0 opens.
        """
        target = np.asarray(_normalize_xyz(xyz), dtype=np.float32)
        traj = []
        for step in range(max_steps):
            cur = self._last_obs_eef_pos
            diff = target - cur
            dist = float(np.linalg.norm(diff))
            traj.append({
                "step": step,
                "eef_pos": [round(float(x), 4) for x in cur],
                "dist_to_target_m": round(dist, 4),
            })
            if dist < tol:
                break
            step_dxyz = np.clip(diff, -step_clip, step_clip)
            action = np.zeros(7, dtype=np.float32)
            action[:3] = step_dxyz / action_scale  # -> roughly [-0.5, 0.5]
            action[:3] = np.clip(action[:3], -1.0, 1.0)
            if target_yaw is not None:
                # add wrist yaw control via action[5] (z-axis axis-angle).
                # NOTE: extract world yaw via atan2(R[1,0], R[0,0]), NOT
                # as_euler('zyx')[0] — the latter returns -world_yaw for
                # gripper-down configs (R[2,2]≈-1) and silently flips the
                # commanded rotation direction. See feedback_rotate_wrist_yaw_sign.
                from scipy.spatial.transform import Rotation as _R
                q = self.env.raw_obs()["robot0_eef_quat"]
                _R_mat = _R.from_quat([q[0], q[1], q[2], q[3]]).as_matrix()
                cur_yaw = float(np.arctan2(_R_mat[1, 0], _R_mat[0, 0]))
                err = (float(target_yaw) - cur_yaw + np.pi) % (2 * np.pi) - np.pi
                step_dyaw = float(np.clip(err, -yaw_step_clip, yaw_step_clip))
                action[5] = float(np.clip(step_dyaw / 0.10, -1.0, 1.0))
            action[6] = gripper
            self._step_env(action)
            if (
                self.critic_interrupted()
                or self.env.episode_terminated
                or self.env.episode_truncated
            ):
                break
        final = self._last_obs_eef_pos
        return {
            "name": "move_to",
            "target_xyz": [float(x) for x in target],
            "final_eef_pos": [round(float(x), 4) for x in final],
            "final_dist_m": round(float(np.linalg.norm(target - final)), 4),
            "steps_used": len(traj),
            "max_steps": max_steps,
            "libero_terminated": self.env.episode_terminated,
        }

    def rotate_wrist(
        self,
        *,
        target_yaw: float | None = None,
        delta_yaw: float | None = None,
        gripper: float = 1.0,
        max_steps: int = 40,
        tol: float = 0.02,
        step_clip: float = 0.10,
    ) -> dict:
        """Rotate wrist around world z-axis. Provide EITHER target_yaw (absolute)
        or delta_yaw (relative, applied as a single rotation goal).

        Uses ``action[5]`` (axis-angle z component) to drive wrist yaw via the
        OSC controller. Holds xyz pose constant during rotation.

        Yaw is the world-frame z-rotation, recovered as
        ``atan2(R[1,0], R[0,0])`` where R is the eef rotation matrix in the
        world frame. (Note: ``as_euler('zyx')[0]`` returns the *negative*
        of this value for gripper-down configurations because the Z-Y-X
        decomposition picks the chart with γ ≈ π, flipping α. Bug fixed
        2026-05-19 — previous implementation rotated the wrist in the
        opposite direction of the commanded yaw.)
        """
        from scipy.spatial.transform import Rotation as _R

        def _yaw_of(quat_xyzw):
            # robot0_eef_quat in libero+robosuite is xyzw (scipy convention).
            q = quat_xyzw
            rot = _R.from_quat([q[0], q[1], q[2], q[3]])
            R = rot.as_matrix()
            # World-frame yaw: angle of the eef x-axis projected onto the
            # world xy plane. Robust to gripper-down (R[2,2]≈-1) which is
            # where the euler 'zyx' chart flips sign.
            return float(np.arctan2(R[1, 0], R[0, 0]))

        raw = self.env.raw_obs()
        cur_quat = raw["robot0_eef_quat"]
        start_yaw = _yaw_of(cur_quat)
        if target_yaw is None and delta_yaw is None:
            return {"name": "rotate_wrist", "error": "need target_yaw or delta_yaw"}
        if target_yaw is None:
            target_yaw = start_yaw + float(delta_yaw)

        traj = []
        for step in range(max_steps):
            raw = self.env.raw_obs()
            cur_yaw = _yaw_of(raw["robot0_eef_quat"])
            err = float(target_yaw - cur_yaw)
            # wrap to [-pi, pi]
            err = (err + np.pi) % (2 * np.pi) - np.pi
            traj.append({"step": step, "yaw": round(cur_yaw, 4), "err": round(err, 4)})
            if abs(err) < tol:
                break
            step_dyaw = float(np.clip(err, -step_clip, step_clip))
            action = np.zeros(7, dtype=np.float32)
            action[5] = step_dyaw / 0.10  # scale to ~[-1,1] action range
            action[5] = float(np.clip(action[5], -1.0, 1.0))
            action[6] = float(gripper)
            self._step_env(action)
            if (
                self.critic_interrupted()
                or self.env.episode_terminated
                or self.env.episode_truncated
            ):
                break
        final_yaw = _yaw_of(self.env.raw_obs()["robot0_eef_quat"])
        return {
            "name": "rotate_wrist",
            "start_yaw": round(start_yaw, 4),
            "target_yaw": round(float(target_yaw), 4),
            "final_yaw": round(final_yaw, 4),
            "final_err": round(float((target_yaw - final_yaw + np.pi) % (2 * np.pi) - np.pi), 4),
            "steps_used": len(traj),
            "libero_terminated": self.env.episode_terminated,
        }

    def rotate_pitch(
        self,
        *,
        target_pitch: float | None = None,
        delta_pitch: float | None = None,
        gripper: float = 1.0,
        max_steps: int = 40,
        tol: float = 0.02,
        step_clip: float = 0.10,
    ) -> dict:
        """Tilt the gripper around the world X-axis ("pitch").

        Pitch is defined as the angle between the eef z-axis and the
        world -z direction, measured in the world yz-plane:

            pitch = atan2(R[1, 2], -R[2, 2])

        - pitch =  0       -> gripper z-axis aligned with world -z (default
                              "gripper down" rest pose).
        - pitch = +pi/2    -> gripper z-axis points in world +y (gripper
                              "looking forward" along world +y).
        - pitch = -pi/2    -> gripper z-axis points in world -y.

        Driven by ``action[3]`` (axis-angle X component) of the OSC_POSE
        controller. Sign verified empirically (probe_pitch.py 2026-05-19):
        action[3]=+1.0 tilts eef z toward world +y, matching this pitch
        definition with no sign flip.

        Holds xyz, yaw, and gripper constant during rotation. Use BEFORE
        threading the gripper into a narrow opening whose front face
        normal is along world ±y (e.g. microwave cavity in libero_10 t9).

        Provide EITHER ``target_pitch`` (absolute) or ``delta_pitch``
        (relative). Both in radians.
        """
        from scipy.spatial.transform import Rotation as _R

        def _pitch_of(quat_xyzw):
            q = quat_xyzw
            R = _R.from_quat([q[0], q[1], q[2], q[3]]).as_matrix()
            return float(np.arctan2(R[1, 2], -R[2, 2]))

        raw = self.env.raw_obs()
        start_pitch = _pitch_of(raw["robot0_eef_quat"])
        if target_pitch is None and delta_pitch is None:
            return {"name": "rotate_pitch",
                    "error": "need target_pitch or delta_pitch"}
        if target_pitch is None:
            target_pitch = start_pitch + float(delta_pitch)

        traj = []
        for step in range(max_steps):
            raw = self.env.raw_obs()
            cur_pitch = _pitch_of(raw["robot0_eef_quat"])
            err = float(target_pitch - cur_pitch)
            err = (err + np.pi) % (2 * np.pi) - np.pi
            traj.append({"step": step,
                         "pitch": round(cur_pitch, 4),
                         "err": round(err, 4)})
            if abs(err) < tol:
                break
            step_dpitch = float(np.clip(err, -step_clip, step_clip))
            action = np.zeros(7, dtype=np.float32)
            action[3] = step_dpitch / 0.10
            action[3] = float(np.clip(action[3], -1.0, 1.0))
            action[6] = float(gripper)
            self._step_env(action)
            if (
                self.critic_interrupted()
                or self.env.episode_terminated
                or self.env.episode_truncated
            ):
                break
        final_pitch = _pitch_of(self.env.raw_obs()["robot0_eef_quat"])
        return {
            "name": "rotate_pitch",
            "start_pitch": round(start_pitch, 4),
            "target_pitch": round(float(target_pitch), 4),
            "final_pitch": round(final_pitch, 4),
            "final_err": round(float(
                (target_pitch - final_pitch + np.pi) % (2 * np.pi) - np.pi), 4),
            "steps_used": len(traj),
            "libero_terminated": self.env.episode_terminated,
        }

    def move_pose(
        self,
        xyz,
        *,
        target_pitch: float | None = None,
        target_yaw: float | None = None,
        gripper: float = -1.0,
        step_clip: float = 0.02,
        pitch_step: float = 0.08,
        yaw_step: float = 0.08,
        tol: float = 0.012,
        ori_tol: float = 0.05,
        action_scale: float = 0.05,
        max_steps: int = 150,
    ) -> dict:
        """Servo position AND orientation (pitch + yaw) SIMULTANEOUSLY.

        Unlike ``move_to`` (holds orientation) + ``rotate_pitch`` (holds
        xyz), this co-varies xyz and wrist tilt every env.step. Co-variation
        lets the OSC controller thread cabinet-front-low poses where a
        decoupled position servo (fixed gripper-down orientation) drives
        the wrist into a singularity and stalls — mimicking pi0's curved
        reach-in.
        """
        from scipy.spatial.transform import Rotation as _R

        def _pitch_of(q):
            R = _R.from_quat([q[0], q[1], q[2], q[3]]).as_matrix()
            return float(np.arctan2(R[1, 2], -R[2, 2]))

        def _yaw_of(q):
            R = _R.from_quat([q[0], q[1], q[2], q[3]]).as_matrix()
            return float(np.arctan2(R[1, 0], R[0, 0]))

        target = np.asarray(_normalize_xyz(xyz), dtype=np.float32)
        traj = []
        step = 0
        for step in range(max_steps):
            cur = self._last_obs_eef_pos
            q = self.env.raw_obs()["robot0_eef_quat"]
            diff = target - cur
            dist = float(np.linalg.norm(diff))
            p_err = 0.0 if target_pitch is None else \
                float((target_pitch - _pitch_of(q) + np.pi) % (2 * np.pi) - np.pi)
            y_err = 0.0 if target_yaw is None else \
                float((target_yaw - _yaw_of(q) + np.pi) % (2 * np.pi) - np.pi)
            traj.append({"step": step, "eef": [round(float(x), 4) for x in cur],
                         "dist": round(dist, 4), "p_err": round(p_err, 3)})
            if dist < tol and abs(p_err) < ori_tol and abs(y_err) < ori_tol:
                break
            action = np.zeros(7, dtype=np.float32)
            sd = np.clip(diff, -step_clip, step_clip)
            action[:3] = np.clip(sd / action_scale, -1.0, 1.0)
            action[3] = float(np.clip(np.clip(p_err, -pitch_step, pitch_step) / 0.10, -1.0, 1.0))
            action[5] = float(np.clip(np.clip(y_err, -yaw_step, yaw_step) / 0.10, -1.0, 1.0))
            action[6] = float(gripper)
            self._step_env(action)
            if (
                self.critic_interrupted()
                or self.env.episode_terminated
                or self.env.episode_truncated
            ):
                break
        final = self._last_obs_eef_pos
        fq = self.env.raw_obs()["robot0_eef_quat"]
        return {
            "name": "move_pose",
            "final_eef_pos": [round(float(x), 4) for x in final],
            "final_dist_m": round(float(np.linalg.norm(target - final)), 4),
            "final_pitch": round(_pitch_of(fq), 4),
            "steps_used": step + 1,
            "libero_terminated": self.env.episode_terminated,
        }

    # ------------------------------------------------------------------
    # Proposal and closed-loop motion tools
    # ------------------------------------------------------------------
    def _motion_record(self, tool: str, before: int, result: Mapping[str, Any]) -> None:
        """Keep bounded public motion evidence for the liveness monitor."""
        row = {
            "tool": str(tool),
            "before_step": int(before),
            "after_step": int(self.env.episode_steps),
            "eef_xyz": [float(value) for value in self._last_obs_eef_pos],
            "result": dict(result),
        }
        reader = getattr(self.env, "privileged_critic_state", None)
        if callable(reader):
            try:
                state = reader()
            except Exception:
                state = None
            if isinstance(state, Mapping):
                progress = state.get("privileged.task.goal.progress")
                if isinstance(progress, (int, float)) and np.isfinite(progress):
                    row["task_progress"] = float(progress)
                success = state.get("privileged.task.success")
                if isinstance(success, bool):
                    row["task_success"] = success
        self._motion_history.append(row)
        del self._motion_history[:-32]

    def _motion_scene_sha256(self) -> str | None:
        observation = getattr(self, "_last_obs", None)
        if not isinstance(observation, Mapping):
            return None
        digest = hashlib.sha256()
        seen = False
        for key in ("main_images", "wrist_images"):
            value = observation.get(key)
            if value is None:
                continue
            array = np.asarray(value)
            digest.update(key.encode("ascii"))
            digest.update(str(array.shape).encode("ascii"))
            digest.update(array.tobytes())
            seen = True
        return digest.hexdigest() if seen else None

    def _terminal_motion_noop(self, name: str) -> dict[str, Any] | None:
        if bool(self.env.episode_terminated) or bool(self.env.episode_truncated):
            return {
                "name": str(name),
                "status": "terminal_noop",
                "no_op_verified": True,
                "environment_advanced": False,
                "libero_terminated": bool(self.env.episode_terminated),
                "episode_truncated": bool(self.env.episode_truncated),
            }
        return None

    def _cached_candidate_fresh(
        self, *, max_age_steps: int = 8, max_eef_delta_m: float = 0.08
    ) -> bool:
        candidate = self._motion_candidate
        if not isinstance(candidate, Mapping) or not candidate.get("candidates"):
            return False
        age = int(self.env.episode_steps) - int(candidate.get("created_step", -1))
        anchor = np.asarray(candidate.get("eef_xyz", ()), dtype=np.float32)
        if anchor.shape != (3,) or not np.isfinite(anchor).all():
            return False
        delta = float(np.linalg.norm(np.asarray(self._last_obs_eef_pos) - anchor))
        expected_scene = candidate.get("scene_sha256")
        current_scene = self._motion_scene_sha256()
        scene_fresh = not (
            isinstance(expected_scene, str)
            and isinstance(current_scene, str)
            and expected_scene != current_scene
        )
        return (
            0 <= age <= int(max_age_steps)
            and delta <= float(max_eef_delta_m)
            and scene_fresh
        )

    def _motion_target(
        self,
        *,
        candidate_index: int = 0,
        target_xyz: list[float] | tuple[float, float, float] | None = None,
    ) -> tuple[np.ndarray | None, float | None, float | None, str | None]:
        """Resolve a target from an explicit point or the latest GraspGen row."""

        if target_xyz is not None:
            return np.asarray(_normalize_xyz(target_xyz), dtype=np.float32), None, None, None
        candidate = self._motion_candidate or {}
        rows = candidate.get("candidates", ())
        if not isinstance(rows, (list, tuple)) or not rows:
            return None, None, None, None
        index = int(candidate_index)
        if not 0 <= index < len(rows) or not isinstance(rows[index], Mapping):
            return None, None, None, None
        row = rows[index]
        pose = np.asarray(row.get("transform_world"), dtype=np.float64)
        if pose.shape != (4, 4) or not np.isfinite(pose).all():
            return None, None, None, None
        # The existing OSC primitives expose pitch/yaw, so retain the 6D
        # proposal while adapting only its orientation representation locally.
        from scipy.spatial.transform import Rotation as _R

        rotation = _R.from_matrix(pose[:3, :3]).as_matrix()
        pitch = float(np.arctan2(rotation[1, 2], -rotation[2, 2]))
        yaw = float(np.arctan2(rotation[1, 0], rotation[0, 0]))
        candidate_id = str(row.get("candidate_id") or candidate.get("candidate_id") or "")
        return pose[:3, 3].astype(np.float32), pitch, yaw, candidate_id or None

    def graspgen(
        self,
        mode: str = "propose",
        camera: str = "agentview",
        step: int | None = None,
        target_world_xyz: list[float] | None = None,
        crop_radius_m: float = 0.12,
        max_candidates: int = 16,
        max_points: int = 2048,
        min_depth_m: float = 0.05,
        max_depth_m: float = 3.0,
        filter_collisions: bool = True,
        remove_outliers: bool = False,
        timeout_s: float = 420.0,
    ) -> dict[str, Any]:
        """Proposal-only GraspGen tool; it never advances the environment."""

        result = self._graspgen.propose(
            mode=mode,
            camera=camera,
            step=step,
            target_world_xyz=target_world_xyz,
            crop_radius_m=crop_radius_m,
            max_candidates=max_candidates,
            max_points=max_points,
            min_depth_m=min_depth_m,
            max_depth_m=max_depth_m,
            filter_collisions=filter_collisions,
            remove_outliers=remove_outliers,
            timeout_s=timeout_s,
        )
        result = dict(result)
        result.update({"name": "graspgen", "no_op_verified": True, "environment_advanced": False})
        if mode == "propose" and result.get("available") and result.get("candidates"):
            candidate_id = "graspgen-" + hashlib.sha256(
                json.dumps(result["candidates"], sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()[:16]
            self._motion_candidate = {
                "candidate_id": candidate_id,
                "created_step": int(self.env.episode_steps),
                "eef_xyz": [float(value) for value in self._last_obs_eef_pos],
                "scene_sha256": self._motion_scene_sha256(),
                "candidates": list(result["candidates"]),
            }
            result["candidate_id"] = candidate_id
        return result

    def candidate_freshness(
        self,
        *,
        max_age_steps: int = 8,
        max_eef_delta_m: float = 0.08,
        invalidate: bool = False,
    ) -> dict[str, Any]:
        """Check whether the cached proposal can be reused without re-GraspGen."""

        candidate = self._motion_candidate
        age = None if candidate is None else int(self.env.episode_steps) - int(candidate["created_step"])
        delta = None
        scene_changed = None
        if candidate is not None:
            delta = float(
                np.linalg.norm(
                    np.asarray(self._last_obs_eef_pos, dtype=np.float32)
                    - np.asarray(candidate["eef_xyz"], dtype=np.float32)
                )
            )
            expected_scene = candidate.get("scene_sha256")
            current_scene = self._motion_scene_sha256()
            if isinstance(expected_scene, str) and isinstance(current_scene, str):
                scene_changed = expected_scene != current_scene
        fresh = bool(
            candidate
            and age is not None
            and 0 <= age <= int(max_age_steps)
            and delta is not None
            and delta <= float(max_eef_delta_m)
            and scene_changed is not True
            and candidate.get("candidates")
        )
        invalidated = bool(invalidate or (candidate is not None and not fresh))
        if invalidated:
            self._motion_candidate = None
        return {
            "name": "candidate_freshness",
            "fresh": fresh and not bool(invalidate),
            "invalidated": invalidated,
            "candidate_id": None if candidate is None else candidate.get("candidate_id"),
            "age_steps": age,
            "eef_delta_m": delta,
            "scene_changed": scene_changed,
            "no_op_verified": True,
            "environment_advanced": False,
        }

    def curobo_reachability(
        self,
        *,
        target_xyz: list[float] | tuple[float, float, float] | None = None,
        candidate_index: int = 0,
        max_distance_m: float = 0.55,
    ) -> dict[str, Any]:
        """CuRobo-compatible reachability preflight with a deterministic fallback.

        A real CuRobo service may replace this backend later; the fallback is
        deliberately labelled workspace-only and is never treated as a path
        collision certificate.
        """

        target, _, _, candidate_id = self._motion_target(
            candidate_index=candidate_index, target_xyz=target_xyz
        )
        if target is None or not np.isfinite(target).all():
            return {
                "name": "curobo_reachability",
                "reachable": False,
                "reason": "missing_or_invalid_target",
                "backend": "analytic_fallback",
                "certificate_level": "none",
                "no_op_verified": True,
                "environment_advanced": False,
            }
        distance = float(np.linalg.norm(target - np.asarray(self._last_obs_eef_pos)))
        workspace = bool(
            -1.2 <= float(target[0]) <= 1.2
            and -1.2 <= float(target[1]) <= 1.2
            and -0.05 <= float(target[2]) <= 1.5
        )
        reachable = workspace and distance <= float(max_distance_m)
        return {
            "name": "curobo_reachability",
            "reachable": reachable,
            "requires_base_staging": False,
            "distance_m": distance,
            "candidate_id": candidate_id,
            "backend": "analytic_fallback",
            "certificate_level": "workspace_only",
            "reason": "ok" if reachable else ("outside_workspace" if not workspace else "distance_limit"),
            "no_op_verified": True,
            "environment_advanced": False,
        }

    def curobo_motiongen_pregrasp(
        self,
        *,
        target_xyz: list[float] | tuple[float, float, float] | None = None,
        candidate_index: int = 0,
        pregrasp_offset_m: float = 0.0,
        target_pitch: float | None = None,
        target_yaw: float | None = None,
        max_steps: int = 80,
    ) -> dict[str, Any]:
        """Execute one bounded pregrasp using the CuRobo-compatible fallback."""

        terminal = self._terminal_motion_noop("curobo_motiongen_pregrasp")
        if terminal is not None:
            return terminal
        if not 1 <= int(max_steps) <= 150:
            return {"name": "curobo_motiongen_pregrasp", "status": "invalid_max_steps", "no_op_verified": True, "environment_advanced": False}
        if target_xyz is None and self._motion_candidate is not None and not self._cached_candidate_fresh():
            self._motion_candidate = None
            return {"name": "curobo_motiongen_pregrasp", "status": "candidate_stale", "candidate_rejected": True, "no_op_verified": True, "environment_advanced": False}

        target, pitch, yaw, candidate_id = self._motion_target(
            candidate_index=candidate_index, target_xyz=target_xyz
        )
        if target is None:
            return {
                "name": "curobo_motiongen_pregrasp",
                "status": "candidate_unavailable",
                "candidate_rejected": True,
                "no_op_verified": True,
                "environment_advanced": False,
            }
        target = target.copy()
        target[2] += float(pregrasp_offset_m)
        reach = self.curobo_reachability(
            target_xyz=target.tolist(), max_distance_m=0.65
        )
        if not reach.get("reachable"):
            return {
                "name": "curobo_motiongen_pregrasp",
                "status": "candidate_rejected",
                "candidate_rejected": True,
                "reachability": reach,
                "candidate_id": candidate_id,
                "no_op_verified": True,
                "environment_advanced": False,
            }
        before = int(self.env.episode_steps)
        result = self.move_pose(
            target.tolist(),
            target_pitch=pitch if target_pitch is None else target_pitch,
            target_yaw=yaw if target_yaw is None else target_yaw,
            gripper=-1.0,
            max_steps=int(max_steps),
        )
        self._motion_record("curobo_motiongen_pregrasp", before, result)
        advanced = int(self.env.episode_steps) > before
        return {
            "name": "curobo_motiongen_pregrasp",
            "status": "executed" if advanced else "already_at_target",
            "planner_backend": "bounded_libero_osc_fallback",
            "certificate_level": "workspace_only",
            "candidate_id": candidate_id,
            "reachability": reach,
            "result": result,
            "no_op_verified": not advanced,
            "environment_advanced": advanced,
        }

    def mink_reach(
        self,
        *,
        target_xyz: list[float] | tuple[float, float, float] | None = None,
        candidate_index: int = 0,
        target_pitch: float | None = None,
        target_yaw: float | None = None,
        max_steps: int = 48,
    ) -> dict[str, Any]:
        """Execute one bounded Mink-compatible local reach fallback.

        LIBERO currently has no Mink service in this checkout. The fallback
        uses the audited OSC pose primitive after a workspace/reachability
        gate and carries no collision certificate.
        """

        terminal = self._terminal_motion_noop("mink_reach")
        if terminal is not None:
            return terminal
        if not 1 <= int(max_steps) <= 120:
            return {"name": "mink_reach", "status": "invalid_max_steps", "no_op_verified": True, "environment_advanced": False}
        if target_xyz is None and self._motion_candidate is not None and not self._cached_candidate_fresh():
            self._motion_candidate = None
            return {"name": "mink_reach", "status": "candidate_stale", "candidate_rejected": True, "no_op_verified": True, "environment_advanced": False}

        target, pitch, yaw, candidate_id = self._motion_target(
            candidate_index=candidate_index, target_xyz=target_xyz
        )
        if target is None:
            return {
                "name": "mink_reach",
                "status": "candidate_unavailable",
                "candidate_rejected": True,
                "no_op_verified": True,
                "environment_advanced": False,
            }
        reach = self.curobo_reachability(
            target_xyz=target.tolist(), max_distance_m=0.65
        )
        if not reach.get("reachable"):
            return {
                "name": "mink_reach",
                "status": "candidate_rejected",
                "candidate_rejected": True,
                "reachability": reach,
                "candidate_id": candidate_id,
                "no_op_verified": True,
                "environment_advanced": False,
            }
        before = int(self.env.episode_steps)
        result = self.move_pose(
            target.tolist(),
            target_pitch=pitch if target_pitch is None else target_pitch,
            target_yaw=yaw if target_yaw is None else target_yaw,
            gripper=-1.0,
            max_steps=int(max_steps),
        )
        self._motion_record("mink_reach", before, result)
        advanced = int(self.env.episode_steps) > before
        return {
            "name": "mink_reach",
            "status": "executed" if advanced else "already_at_target",
            "backend": "bounded_libero_osc_fallback",
            "certificate_level": "workspace_only",
            "candidate_id": candidate_id,
            "reachability": reach,
            "result": result,
            "no_op_verified": not advanced,
            "environment_advanced": advanced,
        }

    def mink_precontact(
        self,
        *,
        target_xyz: list[float] | tuple[float, float, float] | None = None,
        candidate_index: int = 0,
        standoff_m: float = 0.04,
        target_pitch: float | None = None,
        target_yaw: float | None = None,
        max_steps: int = 40,
    ) -> dict[str, Any]:
        """Execute a bounded precontact approach with mandatory re-observation."""

        terminal = self._terminal_motion_noop("mink_precontact")
        if terminal is not None:
            return terminal
        if not 1 <= int(max_steps) <= 100 or not 0.0 <= float(standoff_m) <= 0.15:
            return {"name": "mink_precontact", "status": "invalid_parameters", "no_op_verified": True, "environment_advanced": False}
        if target_xyz is None and self._motion_candidate is not None and not self._cached_candidate_fresh():
            self._motion_candidate = None
            return {"name": "mink_precontact", "status": "candidate_stale", "candidate_rejected": True, "no_op_verified": True, "environment_advanced": False}

        target, pitch, yaw, candidate_id = self._motion_target(
            candidate_index=candidate_index, target_xyz=target_xyz
        )
        if target is None:
            return {"name": "mink_precontact", "status": "candidate_unavailable", "no_op_verified": True, "environment_advanced": False}
        target = target.copy()
        target[2] += abs(float(standoff_m))
        before = int(self.env.episode_steps)
        result = self.move_pose(
            target.tolist(),
            target_pitch=pitch if target_pitch is None else target_pitch,
            target_yaw=yaw if target_yaw is None else target_yaw,
            gripper=-1.0,
            max_steps=int(max_steps),
        )
        self._motion_record("mink_precontact", before, result)
        advanced = int(self.env.episode_steps) > before
        return {"name": "mink_precontact", "status": "executed" if advanced else "already_at_target", "backend": "bounded_libero_osc_fallback", "candidate_id": candidate_id, "result": result, "no_op_verified": not advanced, "environment_advanced": advanced}

    def mink_engage_close(
        self,
        *,
        target_xyz: list[float] | tuple[float, float, float] | None = None,
        candidate_index: int = 0,
        micro_advance_m: float = 0.012,
        close_steps: int = 3,
        max_steps: int = 24,
    ) -> dict[str, Any]:
        """Close, micro-advance along the approach axis, and return evidence."""

        terminal = self._terminal_motion_noop("mink_engage_close")
        if terminal is not None:
            return terminal
        if not 1 <= int(close_steps) <= 8 or not 1 <= int(max_steps) <= 64 or not 0.0 <= float(micro_advance_m) <= 0.04:
            return {"name": "mink_engage_close", "status": "invalid_parameters", "no_op_verified": True, "environment_advanced": False}
        if target_xyz is None and self._motion_candidate is not None and not self._cached_candidate_fresh():
            self._motion_candidate = None
            return {"name": "mink_engage_close", "status": "candidate_stale", "candidate_rejected": True, "no_op_verified": True, "environment_advanced": False}

        target, pitch, yaw, candidate_id = self._motion_target(
            candidate_index=candidate_index, target_xyz=target_xyz
        )
        if target is None:
            return {"name": "mink_engage_close", "status": "candidate_unavailable", "no_op_verified": True, "environment_advanced": False}
        target = target.copy()
        approach_axis = np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
        if target_xyz is None:
            rows = (self._motion_candidate or {}).get("candidates", ())
            index = int(candidate_index)
            if isinstance(rows, (list, tuple)) and 0 <= index < len(rows):
                pose = np.asarray(rows[index].get("transform_world"), dtype=np.float32)
                if pose.shape == (4, 4) and np.isfinite(pose).all():
                    axis = pose[:3, :3] @ np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
                    norm = float(np.linalg.norm(axis))
                    if norm > 1e-6:
                        approach_axis = axis / norm
        direction = target - np.asarray(self._last_obs_eef_pos, dtype=np.float32)
        if target_xyz is not None and float(np.linalg.norm(direction)) > 1e-6:
            approach_axis = direction / float(np.linalg.norm(direction))
        if float(np.dot(approach_axis, direction)) < 0.0:
            approach_axis = -approach_axis
        target += approach_axis * abs(float(micro_advance_m))
        before = int(self.env.episode_steps)
        close = self.set_gripper(gripper=1.0, steps=int(close_steps))
        if not self.env.episode_terminated and not self.env.episode_truncated:
            approach = self.move_pose(
                target.tolist(), target_pitch=pitch, target_yaw=yaw, gripper=1.0, max_steps=int(max_steps)
            )
        else:
            approach = {"status": "terminal_before_advance"}
        result = {"close": close, "micro_advance": approach}
        self._motion_record("mink_engage_close", before, result)
        return {"name": "mink_engage_close", "status": "executed", "backend": "bounded_libero_osc_fallback", "candidate_id": candidate_id, "approach_axis_world": approach_axis.tolist(), "result": result, "environment_advanced": int(self.env.episode_steps) > before}

    def mink_pull(
        self,
        *,
        delta_xyz: list[float] | tuple[float, float, float] = (0.0, 0.02, 0.0),
        max_steps: int = 24,
    ) -> dict[str, Any]:
        """Execute one incremental closed-gripper pull and re-observe."""

        terminal = self._terminal_motion_noop("mink_pull")
        if terminal is not None:
            return terminal
        if not 1 <= int(max_steps) <= 64:
            return {"name": "mink_pull", "status": "invalid_max_steps", "no_op_verified": True, "environment_advanced": False}

        delta = np.asarray(_normalize_xyz(delta_xyz), dtype=np.float32)
        norm = float(np.linalg.norm(delta))
        if not np.isfinite(delta).all() or not 0.001 <= norm <= 0.08:
            return {"name": "mink_pull", "status": "invalid_delta", "no_op_verified": True, "environment_advanced": False}
        before = int(self.env.episode_steps)
        target = np.asarray(self._last_obs_eef_pos, dtype=np.float32) + delta
        result = self.move_to(target.tolist(), gripper=1.0, max_steps=int(max_steps), step_clip=min(0.012, norm))
        self._motion_record("mink_pull", before, result)
        return {"name": "mink_pull", "status": "executed", "backend": "bounded_libero_osc_fallback", "delta_xyz": delta.tolist(), "result": result, "environment_advanced": int(self.env.episode_steps) > before}

    def progress_liveness(
        self,
        *,
        window: int = 4,
        min_eef_progress_m: float = 0.002,
    ) -> dict[str, Any]:
        """Report whether recent motion produced physical progress."""

        rows = self._motion_history[-max(1, int(window)) :]
        if len(rows) < 2:
            progress = None
        else:
            first = np.asarray(rows[0]["eef_xyz"], dtype=np.float32)
            last = np.asarray(rows[-1]["eef_xyz"], dtype=np.float32)
            progress = float(np.linalg.norm(last - first))
        stagnant = progress is not None and progress < float(min_eef_progress_m)
        task_progress = None
        if len(rows) >= 2 and all(
            isinstance(row.get("task_progress"), (int, float)) for row in rows
        ):
            task_progress = float(rows[-1]["task_progress"]) - float(
                rows[0]["task_progress"]
            )
        task_stagnant = task_progress is not None and task_progress <= 0.0
        stalled = bool(stagnant and (task_stagnant or task_progress is None))
        return {
            "name": "progress_liveness",
            "observations": len(rows),
            "eef_progress_m": progress,
            "task_progress_delta": task_progress,
            "stagnant": stalled,
            "proposal": "reobserve_or_switch_candidate" if stalled else None,
            "no_op_verified": True,
            "environment_advanced": False,
        }

    def semantic_joint_interact(
        self,
        entity: str,
        joint: str,
        *,
        direction: str,
        max_sweep_steps: int = 64,
        sweep_step_m: float = 0.015,
        close_steps: int = 3,
    ) -> dict[str, Any]:
        """Servo a named fixture joint using audited geometry and real actions.

        Role1 supplies semantic identity and the desired range endpoint. The
        simulator-derived contact points remain private to this local tool.
        Joint qpos is read only as feedback; this primitive never writes qpos,
        reward, termination, or any task predicate directly.
        """

        if not self._allow_privileged_actions:
            raise PermissionError(
                "semantic_joint_interact requires privileged action authorization"
            )
        entity = str(entity).strip()
        joint = str(joint).strip()
        direction = str(direction).strip().casefold()
        if not entity or not joint:
            raise ValueError("entity and joint must not be empty")
        if direction not in {"lower", "upper"}:
            raise ValueError("direction must be 'lower' or 'upper'")
        max_sweep_steps = int(max_sweep_steps)
        close_steps = int(close_steps)
        sweep_step_m = float(sweep_step_m)
        if not 1 <= max_sweep_steps <= 64:
            raise ValueError("max_sweep_steps must be in [1, 64]")
        if not 1 <= close_steps <= 8:
            raise ValueError("close_steps must be in [1, 8]")
        if not 0.001 <= sweep_step_m <= 0.015:
            raise ValueError("sweep_step_m must be in [0.001, 0.015]")

        def plan() -> dict[str, Any]:
            value = self.env.privileged_semantic_joint_plan(
                entity=entity, joint=joint, direction=direction
            )
            required = {
                "joint",
                "qpos",
                "qvel",
                "range_lower",
                "range_upper",
                "goal_satisfied",
                "approach_position_world",
                "press_position_world",
                "tangent_direction_world",
            }
            if not isinstance(value, dict) or not required.issubset(value):
                raise RuntimeError("semantic joint plan is incomplete")
            return value

        start = plan()
        slide_joint = str(start.get("joint_type", "")).casefold() == "slide"
        actions_before = int(self.env.episode_steps)
        sweep_steps = 0
        direction_reversals = 0
        direct_contact_steps = 0
        recontacted = False
        target_qpos = (
            float(start["range_lower"])
            if direction == "lower"
            else float(start["range_upper"])
        )

        def terminal() -> bool:
            return bool(
                self.critic_interrupted()
                or self.env.episode_terminated
                or self.env.episode_truncated
            )

        def target_distance(value: dict[str, Any]) -> float:
            return abs(float(value["qpos"]) - target_qpos)

        desired_velocity_sign = -1.0 if direction == "lower" else 1.0

        # Keep the reachability mode stable for the entire primitive.  A
        # short correction can cross the 4 cm boundary during the direct
        # contact budget; reclassifying it in the continuation sweep rotates
        # the tangent frame mid-recovery and can reverse the realized motion.
        stabilize_tangent = target_distance(start) <= 0.04

        def desired_velocity(value: dict[str, Any]) -> float:
            return desired_velocity_sign * float(value["qvel"])

        def sweep(*, budget: int) -> tuple[bool, bool, int]:
            """Sweep from the current contact and report realized response."""

            nonlocal direction_reversals, sweep_steps
            direction_sign = 1.0
            # Contact geometry can change as the EEF moves around a fixture,
            # which may flip the realized tangent direction mid-sweep.  Use a
            # small bounded reversal budget rather than locking the primitive
            # to its first correction for the entire horizon.
            local_reversals = 0
            responded = False
            window_start = plan()
            # Recompute contact geometry only when a new sweep starts.  The
            # radial frame can rotate as the EEF moves; following a newly
            # rotated tangent every action can silently reverse the physical
            # joint motion even though the command sign is unchanged.
            tangent_reference = np.asarray(
                window_start["tangent_direction_world"], dtype=np.float32
            ).reshape(-1)
            window_steps = 0
            used = 0
            for _ in range(max(0, int(budget))):
                if terminal():
                    break
                current = plan()
                if bool(current["goal_satisfied"]):
                    break
                tangent = tangent_reference
                if not stabilize_tangent:
                    tangent = np.asarray(
                        current["tangent_direction_world"], dtype=np.float32
                    ).reshape(-1)
                if tangent.size != 3 or not np.isfinite(tangent).all():
                    raise RuntimeError("semantic joint tangent is invalid")
                before_distance = target_distance(current)
                action = np.zeros(7, dtype=np.float32)
                action[:3] = np.clip(
                    tangent * direction_sign * (sweep_step_m / 0.05),
                    -0.4,
                    0.4,
                )
                # Drawers respond to an open-finger downward push; closing on
                # the long handle bar is not a stable grasp for Panda's fixed
                # wrist frame. Hinges retain the original closed downward
                # press behavior.
                # OSC translation is scaled by 0.05 m/action.  Drawer-opening
                # evidence uses a roughly 2 cm downward push per step, while
                # the hinge path retains its historical small brake.
                action[2] -= 0.4 if slide_joint else 0.02
                action[6] = -1.0 if slide_joint else 1.0
                self._step_env(action)
                sweep_steps += 1
                used += 1
                window_steps += 1
                if terminal():
                    break
                after = plan()
                after_distance = target_distance(after)
                if (
                    after_distance < before_distance - 1e-4
                    or desired_velocity(after) > 0.005
                ):
                    responded = True
                if bool(after["goal_satisfied"]):
                    break
                if window_steps < 3:
                    continue

                distance_delta = after_distance - target_distance(window_start)
                velocity_delta = desired_velocity(after) - desired_velocity(window_start)
                moving_away = distance_delta > 2e-4
                severe_divergence = distance_delta > 0.01
                velocity_worsened = velocity_delta < -0.005
                nearly_stationary = max(
                    abs(float(after["qvel"])),
                    abs(float(window_start["qvel"])),
                ) < 0.005
                if (
                    moving_away
                    and (
                        severe_divergence
                        or velocity_worsened
                        or nearly_stationary
                    )
                    and local_reversals < 3
                ):
                    direction_sign *= -1.0
                    local_reversals += 1
                    direction_reversals += 1
                window_start = after
                window_steps = 0
            return bool(plan()["goal_satisfied"]), responded, used

        # A qpos guard normally fires while the VLA is still touching the
        # fixture. Brake at that realized contact before moving away; lifting
        # first lets the existing angular velocity carry a knob away from its
        # requested endpoint.
        direct_budget = min(max_sweep_steps, 12)
        satisfied, direct_response, direct_contact_steps = sweep(
            budget=direct_budget
        )

        if not satisfied and not terminal() and not direct_response:
            recontacted = True
            current = plan()
            approach = np.asarray(
                current["approach_position_world"], dtype=np.float64
            ).reshape(3)
            eef = np.asarray(self._last_obs_eef_pos, dtype=np.float64).reshape(3)
            retreat = eef.copy()
            retreat[2] = max(float(eef[2]) + 0.045, float(approach[2]))
            self.move_pose(
                retreat.tolist(),
                gripper=-1.0,
                step_clip=0.012,
                tol=0.008,
                max_steps=36,
            )
            if not terminal():
                current = plan()
                approach = np.asarray(
                    current["approach_position_world"], dtype=np.float64
                ).reshape(3)
                approach[2] = max(float(approach[2]), float(retreat[2]))
                self.move_pose(
                    approach.tolist(),
                    gripper=-1.0,
                    step_clip=0.02,
                    tol=0.01,
                    max_steps=40,
                )
            if not terminal():
                current = plan()
                self.move_pose(
                    list(current["press_position_world"]),
                    gripper=-1.0,
                    step_clip=0.012,
                    tol=0.005,
                    max_steps=32,
                )
            if not terminal():
                for _ in range(close_steps):
                    action = np.zeros(7, dtype=np.float32)
                    action[6] = -1.0 if slide_joint else 1.0
                    self._step_env(action)
                    if terminal():
                        break

        if not satisfied and not terminal():
            remaining = max_sweep_steps - direct_contact_steps
            satisfied, _, _ = sweep(budget=remaining)

        final = plan()
        return {
            "name": "semantic_joint_interact",
            "entity": entity,
            "joint": str(final["joint"]),
            "direction": direction,
            "start_qpos": float(start["qpos"]),
            "final_qpos": float(final["qpos"]),
            "range_lower": float(final["range_lower"]),
            "range_upper": float(final["range_upper"]),
            "joint_goal_satisfied": bool(final["goal_satisfied"]),
            "sweep_steps": sweep_steps,
            "direct_contact_steps": direct_contact_steps,
            "recontacted": recontacted,
            "direction_reversals": direction_reversals,
            "steps_used": int(self.env.episode_steps) - actions_before,
            "interrupted_by_critic": self.critic_interrupted(),
            "libero_terminated": self.env.episode_terminated,
        }

    def release(
        self,
        *,
        max_steps: int = 20,
    ) -> dict:
        """Open gripper for ``max_steps`` env steps while keeping eef in place.

        Returns once libero terminates (success) or step budget exhausted.
        """
        start_grip = self._last_obs_gripper
        peak_grip = start_grip
        for step in range(max_steps):
            action = np.zeros(7, dtype=np.float32)
            action[6] = -1.0  # open
            self._step_env(action)
            peak_grip = max(peak_grip, self._last_obs_gripper)
            if (
                self.critic_interrupted()
                or self.env.episode_terminated
                or self.env.episode_truncated
            ):
                break
        return {
            "name": "release",
            "steps_used": step + 1,
            "start_gripper_opening": round(start_grip, 4),
            "peak_gripper_opening": round(peak_grip, 4),
            "final_gripper_opening": round(self._last_obs_gripper, 4),
            "libero_terminated": self.env.episode_terminated,
        }

    def set_gripper(
        self,
        *,
        gripper: float = -1.0,
        steps: int = 5,
    ) -> dict:
        """Hold the current EEF pose and drive ``gripper`` for ``steps`` env steps."""
        g = float(gripper)
        n = int(steps)
        for _ in range(n):
            action = np.zeros(7, dtype=np.float32)
            action[6] = g
            self._step_env(action)
            if (
                self.critic_interrupted()
                or self.env.episode_terminated
                or self.env.episode_truncated
            ):
                break
        return {
            "name": "set_gripper",
            "gripper": g,
            "steps": n,
            "libero_terminated": self.env.episode_terminated,
        }

    def privileged_pick_place(
        self,
        *,
        grasp_offset_xyz: list[float] | tuple[float, float, float] = (
            0.0,
            -0.036,
            0.038,
        ),
        retreat_height: float = 0.025,
        approach_height: float = 0.055,
        close_steps: int = 24,
        grasp_confirm_steps: int = 4,
        lift_height: float = 0.13,
        target_height: float = 0.035,
        carry_height: float = 0.10,
        max_steps_per_move: int = 48,
        grasp_pose_max_steps: int | None = None,
        move_step_clip: float = 0.025,
        lift_step_clip: float | None = None,
        transport_step_clip: float | None = None,
        contact_lift_min_m: float = 0.02,
        max_segment_distance_m: float | None = None,
        vertical_first_carry: bool = False,
        approach_pitch: float | None = None,
        approach_yaw: float | None = None,
    ) -> dict:
        """Run a bounded semantic pick/place recovery from the audited sidecar.

        This is intentionally an Actor-only recovery primitive.  It reads the
        current manipulated-object and target poses from the simulator-side
        semantic sensor, then uses the same OSC primitives as ordinary Role1
        actions.  It never writes coordinates into the Critic or VLA plane and
        checks the official LIBERO predicate after each irreversible phase.
        """
        if not self._allow_privileged_actions:
            raise PermissionError(
                "privileged_pick_place requires privileged action authorization"
            )
        sidecar = self.env.privileged_critic_state()
        required = (
            "privileged.task.manipulated_object.position.x",
            "privileged.task.manipulated_object.position.y",
            "privileged.task.manipulated_object.position.z",
            "privileged.task.target.position.x",
            "privileged.task.target.position.y",
            "privileged.task.target.position.z",
        )
        if not all(key in sidecar for key in required):
            return {"name": "privileged_pick_place", "status": "semantic_state_unavailable"}
        if bool(sidecar.get("privileged.task.success", False)):
            return {"name": "privileged_pick_place", "status": "already_success"}

        def xyz(state: Mapping[str, Any], prefix: str) -> np.ndarray:
            return np.asarray(
                [state[f"{prefix}.position.{axis}"] for axis in ("x", "y", "z")],
                dtype=np.float32,
            )

        object_xyz = xyz(sidecar, "privileged.task.manipulated_object")
        target_xyz = xyz(sidecar, "privileged.task.target")
        grasp_offset = np.asarray(_normalize_xyz(grasp_offset_xyz), dtype=np.float32)
        retreat_height = float(retreat_height)
        close_steps = int(close_steps)
        grasp_confirm_steps = int(grasp_confirm_steps)
        max_steps_per_move = int(max_steps_per_move)
        grasp_pose_max_steps = (
            None if grasp_pose_max_steps is None else int(grasp_pose_max_steps)
        )
        move_step_clip = float(move_step_clip)
        lift_step_clip = move_step_clip if lift_step_clip is None else float(lift_step_clip)
        transport_step_clip = (
            move_step_clip if transport_step_clip is None else float(transport_step_clip)
        )
        contact_lift_min_m = float(contact_lift_min_m)
        max_segment_distance_m = (
            None
            if max_segment_distance_m is None
            else float(max_segment_distance_m)
        )
        if not isinstance(vertical_first_carry, (bool, np.bool_)):
            raise ValueError("vertical_first_carry must be boolean")
        vertical_first_carry = bool(vertical_first_carry)
        approach_pitch = (
            None if approach_pitch is None else float(approach_pitch)
        )
        approach_yaw = None if approach_yaw is None else float(approach_yaw)
        if not 0.0 <= retreat_height <= 0.08:
            raise ValueError("retreat_height must be in [0, 0.08]")
        if not 1 <= close_steps <= 32:
            raise ValueError("close_steps must be in [1, 32]")
        if not 1 <= grasp_confirm_steps <= 8:
            raise ValueError("grasp_confirm_steps must be in [1, 8]")
        if grasp_confirm_steps > close_steps:
            raise ValueError("grasp_confirm_steps cannot exceed close_steps")
        if not 1 <= max_steps_per_move <= 140:
            raise ValueError("max_steps_per_move must be in [1, 140]")
        if grasp_pose_max_steps is not None and not 1 <= grasp_pose_max_steps <= 140:
            raise ValueError("grasp_pose_max_steps must be in [1, 140]")
        if not 0.005 <= move_step_clip <= 0.05:
            raise ValueError("move_step_clip must be in [0.005, 0.05]")
        if not 0.005 <= lift_step_clip <= 0.05:
            raise ValueError("lift_step_clip must be in [0.005, 0.05]")
        if not 0.005 <= transport_step_clip <= 0.05:
            raise ValueError("transport_step_clip must be in [0.005, 0.05]")
        if not 0.005 <= contact_lift_min_m <= 0.05:
            raise ValueError("contact_lift_min_m must be in [0.005, 0.05]")
        if max_segment_distance_m is not None and not 0.10 <= max_segment_distance_m <= 0.50:
            raise ValueError("max_segment_distance_m must be in [0.10, 0.50]")
        if approach_pitch is not None and not -np.pi <= approach_pitch <= np.pi:
            raise ValueError("approach_pitch must be in [-pi, pi]")
        if approach_yaw is not None and not -np.pi <= approach_yaw <= np.pi:
            raise ValueError("approach_yaw must be in [-pi, pi]")
        if not all(
            np.isfinite(value).all()
            for value in (object_xyz, target_xyz, grasp_offset)
        ):
            return {"name": "privileged_pick_place", "status": "invalid_semantic_pose"}

        def move_once(
            target: np.ndarray,
            *,
            gripper: float,
            step_clip: float,
            max_steps: int | None = None,
        ) -> dict:
            move_budget = max_steps_per_move if max_steps is None else int(max_steps)
            if approach_pitch is not None or approach_yaw is not None:
                return self.move_pose(
                    xyz=[float(value) for value in target],
                    target_pitch=approach_pitch,
                    target_yaw=approach_yaw,
                    gripper=gripper,
                    max_steps=move_budget,
                    tol=0.012,
                    step_clip=step_clip,
                )
            return self.move_to(
                xyz=[float(value) for value in target],
                gripper=gripper,
                max_steps=move_budget,
                tol=0.012,
                step_clip=step_clip,
            )

        def move(
            target: np.ndarray,
            *,
            gripper: float,
            step_clip: float,
            max_steps: int | None = None,
        ) -> dict:
            """Move directly or through bounded Cartesian waypoints.

            Long closed-gripper carries can cross clutter even when both
            endpoints are reachable.  Waypoints are opt-in so existing
            callers retain their historical trajectory and budgets.
            """

            if max_segment_distance_m is None:
                return move_once(
                    target,
                    gripper=gripper,
                    step_clip=step_clip,
                    max_steps=max_steps,
                )
            current = np.asarray(self._last_obs_eef_pos, dtype=np.float32)
            delta = target - current
            distance = float(np.linalg.norm(delta))
            segment_count = max(
                1, int(np.ceil(distance / max_segment_distance_m))
            )
            results: list[dict[str, Any]] = []
            for index in range(1, segment_count + 1):
                waypoint = current + delta * (index / segment_count)
                result = move_once(
                    waypoint,
                    gripper=gripper,
                    step_clip=step_clip,
                    max_steps=max_steps,
                )
                results.append(result)
                if (
                    self.critic_interrupted()
                    or self.env.episode_terminated
                    or self.env.episode_truncated
                    or not reached(result)
                ):
                    break
            final = dict(results[-1])
            if segment_count > 1:
                final["waypoint_count"] = len(results)
                final["waypoints"] = [
                    result.get("final_eef_pos") for result in results
                ]
            return final

        def reached(result: Mapping[str, Any]) -> bool:
            value = result.get("final_dist_m")
            return value is None or float(value) <= 0.025

        def retained_after_contact(
            state: Mapping[str, Any], *, minimum_lift_m: float | None = None
        ) -> bool:
            """Accept contact-only grasps only after physical lift evidence.

            Some thin LIBERO objects can be retained between the fingers while
            the environment's binary grasp helper remains false.  The fallback
            remains conservative: it requires current gripper contact, bounded
            EEF distance, and after lift a measured rise of the object itself.
            """

            if _semantic_grasp_retained(state):
                return True
            if not bool(
                state.get(
                    "privileged.task.manipulated_object.gripper_contact", False
                )
            ):
                return False
            distance = state.get(
                "privileged.task.manipulated_object.distance_to_eef_m"
            )
            if not isinstance(distance, (int, float)) or float(distance) > 0.12:
                return False
            if minimum_lift_m is None:
                return True
            current_z = state.get("privileged.task.manipulated_object.position.z")
            return (
                isinstance(current_z, (int, float))
                and float(current_z) >= float(object_xyz[2]) + minimum_lift_m
            )

        def contact_aligned_for_release(result: Mapping[str, Any]) -> bool:
            """Accept a contact-limited descent only when XY is still aligned."""

            final_eef = result.get("final_eef_pos")
            target = result.get("target_xyz")
            if not (
                isinstance(final_eef, (list, tuple))
                and isinstance(target, (list, tuple))
                and len(final_eef) == 3
                and len(target) == 3
            ):
                return False
            final_xyz = np.asarray(final_eef, dtype=np.float32)
            target_position = np.asarray(target, dtype=np.float32)
            if not np.isfinite(final_xyz).all() or not np.isfinite(
                target_position
            ).all():
                return False
            xy_error = float(np.linalg.norm(final_xyz[:2] - target_position[:2]))
            height_error = float(final_xyz[2] - target_position[2])
            return xy_error <= 0.025 and -0.01 <= height_error <= 0.075

        def stopped(phase: str, **evidence: Any) -> dict[str, Any] | None:
            if not (self.env.episode_terminated or self.env.episode_truncated):
                return None
            state = self.env.privileged_critic_state()
            return {
                "name": "privileged_pick_place",
                "status": (
                    "success" if self.env.episode_terminated else "episode_truncated"
                ),
                "phase": phase,
                "primary_relation_satisfied": bool(
                    state.get("privileged.task.primary_relation_satisfied", False)
                ),
                "libero_terminated": self.env.episode_terminated,
                **evidence,
            }

        # Preserve an already verified grasp.  Recovery is commonly triggered
        # exactly when the VLA is about to release a still-retained object far
        # from its target.  Re-opening and re-grasping at that point consumes
        # most of the remaining horizon and can drop the object.  Keep the
        # gripper closed and only use the semantic pose oracle for the bounded
        # transport target.  Unheld objects retain the original pick sequence.
        retained_at_entry = _semantic_grasp_retained(sidecar)
        current_eef = np.asarray(self._last_obs_eef_pos, dtype=np.float32).copy()
        if retained_at_entry:
            grasp_pose = current_eef.copy()
            transport_offset = current_eef - object_xyz
            close = {
                "name": "retained_grasp",
                "steps_used": 0,
                "contact_seen": bool(
                    sidecar.get(
                        "privileged.task.manipulated_object.gripper_contact", False
                    )
                ),
                "grasp_verified": True,
                "retention_confirmation_steps": 1,
                "required_retention_confirmation_steps": 1,
            }
            lift = current_eef.copy()
            lift[2] += float(lift_height)
        else:
            transport_offset = grasp_offset
            # A single short upward clearance is enough to break cabinet-edge
            # contact observed in failed rollouts before a fresh semantic pick.
            retreat = current_eef.copy()
            retreat[2] = float(current_eef[2]) + retreat_height
            retreat_result = move(retreat, gripper=-1.0, step_clip=move_step_clip)
            if terminal := stopped("retreat", retreat=retreat_result):
                return terminal
            if not reached(retreat_result):
                return {
                    "name": "privileged_pick_place",
                    "status": "retreat_not_reached",
                    "retreat": retreat_result,
                }

            # Re-stage above the object before closing.  The short approach
            # avoids inheriting a stale VLA pose while keeping fingers open.
            grasp_pose = object_xyz + grasp_offset
            approach = grasp_pose.copy()
            approach[2] += float(approach_height)
            approach_result = move(approach, gripper=-1.0, step_clip=move_step_clip)
            if terminal := stopped("approach", approach=approach_result):
                return terminal
            if not reached(approach_result):
                return {
                    "name": "privileged_pick_place",
                    "status": "approach_not_reached",
                    "approach": approach_result,
                }
            grasp_result = move(
                grasp_pose,
                gripper=-1.0,
                step_clip=move_step_clip,
                max_steps=grasp_pose_max_steps,
            )
            if terminal := stopped("grasp_pose", grasp_pose=grasp_result):
                return terminal
            if not reached(grasp_result):
                return {
                    "name": "privileged_pick_place",
                    "status": "grasp_pose_not_reached",
                    "grasp_pose": grasp_result,
                }

            state_after_close = self.env.privileged_critic_state()
            contact_seen = bool(
                state_after_close.get(
                    "privileged.task.manipulated_object.gripper_contact", False
                )
            )
            stable_grasp_steps = 1 if retained_after_contact(state_after_close) else 0
            close_used = 0
            while close_used < close_steps and stable_grasp_steps < grasp_confirm_steps:
                self.set_gripper(gripper=1.0, steps=1)
                close_used += 1
                state_after_close = self.env.privileged_critic_state()
                contact_seen = contact_seen or bool(
                    state_after_close.get(
                        "privileged.task.manipulated_object.gripper_contact", False
                    )
                )
                stable_grasp_steps = (
                    stable_grasp_steps + 1
                    if retained_after_contact(state_after_close)
                    else 0
                )
                if (
                    self.critic_interrupted()
                    or self.env.episode_terminated
                    or self.env.episode_truncated
                ):
                    break
            close = {
                "name": "adaptive_close",
                "steps_used": close_used,
                "contact_seen": contact_seen,
                "grasp_verified": stable_grasp_steps >= grasp_confirm_steps,
                "retention_confirmation_steps": stable_grasp_steps,
                "required_retention_confirmation_steps": grasp_confirm_steps,
            }
            if terminal := stopped("close", close=close):
                return terminal
            if stable_grasp_steps < grasp_confirm_steps:
                return {
                    "name": "privileged_pick_place",
                    "status": "grasp_not_verified",
                    "close": close,
                    "critic_state": {
                        key: state_after_close.get(key)
                        for key in (
                            "privileged.task.manipulated_object.gripper_contact",
                            "privileged.task.manipulated_object.distance_to_eef_m",
                        )
                    },
                }

            lift = grasp_pose.copy()
            lift[2] += float(lift_height)
        lift_result = move(lift, gripper=1.0, step_clip=lift_step_clip)
        if terminal := stopped("lift", close=close, lift=lift_result):
            return terminal
        if not reached(lift_result):
            return {
                "name": "privileged_pick_place",
                "status": "lift_not_reached",
                "grasped": True,
                "close": close,
                "lift": lift_result,
            }
        state_after_lift = self.env.privileged_critic_state()
        if not retained_after_contact(
            state_after_lift, minimum_lift_m=contact_lift_min_m
        ):
            return {
                "name": "privileged_pick_place",
                "status": "grasp_lost_during_lift",
                "grasped": False,
                "close": close,
                "lift": lift_result,
            }

        # The receptacle can move while the VLA opens a drawer. Refresh its
        # semantic pose after grasp/lift instead of carrying toward a stale
        # pre-recovery snapshot.
        state_before_carry = self.env.privileged_critic_state()
        if all(key in state_before_carry for key in required[3:]):
            target_xyz = xyz(state_before_carry, "privileged.task.target")
        carry = target_xyz + transport_offset
        carry[2] += float(carry_height)
        carry_clearance_result: dict[str, Any] | None = None
        if vertical_first_carry and float(self._last_obs_eef_pos[2]) < float(carry[2]) - 0.012:
            carry_clearance = np.asarray(
                self._last_obs_eef_pos, dtype=np.float32
            ).copy()
            carry_clearance[2] = float(carry[2])
            carry_clearance_result = move(
                carry_clearance,
                gripper=1.0,
                step_clip=transport_step_clip,
            )
            if terminal := stopped(
                "carry_clearance",
                close=close,
                lift=lift_result,
                carry_clearance=carry_clearance_result,
            ):
                return terminal
            if not reached(carry_clearance_result):
                return {
                    "name": "privileged_pick_place",
                    "status": "carry_clearance_not_reached",
                    "grasped": True,
                    "close": close,
                    "lift": lift_result,
                    "carry_clearance": carry_clearance_result,
                }
            state_after_clearance = self.env.privileged_critic_state()
            if not retained_after_contact(
                state_after_clearance, minimum_lift_m=contact_lift_min_m
            ):
                return {
                    "name": "privileged_pick_place",
                    "status": "grasp_lost_during_carry_clearance",
                    "grasped": False,
                    "close": close,
                    "lift": lift_result,
                    "carry_clearance": carry_clearance_result,
                }
        carry_result = move(carry, gripper=1.0, step_clip=transport_step_clip)
        if terminal := stopped("carry", close=close, lift=lift_result):
            return terminal
        if not reached(carry_result):
            return {
                "name": "privileged_pick_place",
                "status": "carry_not_reached",
                "grasped": True,
                "close": close,
                "lift": lift_result,
                "carry_clearance": carry_clearance_result,
                "carry": carry_result,
            }
        state_after_carry = self.env.privileged_critic_state()
        if not retained_after_contact(
            state_after_carry, minimum_lift_m=contact_lift_min_m
        ):
            return {
                "name": "privileged_pick_place",
                "status": "grasp_lost_during_carry",
                "grasped": False,
                "close": close,
                "lift": lift_result,
                "carry_clearance": carry_clearance_result,
                "carry": carry_result,
            }
        place = target_xyz + transport_offset
        place[2] += float(target_height)
        place_result = move(place, gripper=1.0, step_clip=transport_step_clip)
        if self.env.episode_terminated or self.env.episode_truncated:
            final_state = self.env.privileged_critic_state()
            return {
                "name": "privileged_pick_place",
                "status": (
                    "success" if self.env.episode_terminated else "episode_truncated"
                ),
                "grasped": True,
                "close": close,
                "lift": lift_result,
                "carry_clearance": carry_clearance_result,
                "carry": carry_result,
                "place": place_result,
                "primary_relation_satisfied": bool(
                    final_state.get(
                        "privileged.task.primary_relation_satisfied", False
                    )
                ),
                "libero_terminated": self.env.episode_terminated,
            }
        contact_limited_release = not reached(
            place_result
        ) and contact_aligned_for_release(place_result)
        if not reached(place_result) and not contact_limited_release:
            return {
                "name": "privileged_pick_place",
                "status": "place_not_reached",
                "grasped": True,
                "close": close,
                "lift": lift_result,
                "carry_clearance": carry_clearance_result,
                "carry": carry_result,
                "place": place_result,
            }
        release = self.release(max_steps=12)
        final_state = self.env.privileged_critic_state()
        return {
            "name": "privileged_pick_place",
            "status": "success" if self.env.episode_terminated else "placed",
            "grasped": True,
            "close": close,
            "lift": lift_result,
            "carry_clearance": carry_clearance_result,
            "carry": carry_result,
            "place": place_result,
            "release": release,
            "contact_limited_release": contact_limited_release,
            "primary_relation_satisfied": bool(
                final_state.get("privileged.task.primary_relation_satisfied", False)
            ),
            "libero_terminated": self.env.episode_terminated,
        }

    # ---- introspection helpers (for LLM-in-the-loop) ----

    def segment(
        self,
        prompt: str = "",
        camera: str = "agentview",
        step: int | None = None,
        point: list[int] | None = None,
        min_score: float = 0.2,
    ) -> dict:
        """Call SAM3 on an existing image artifact without advancing the env.

        This tool deliberately does not render camera views or create wrist/high-res
        artifacts. Errors are structured so the agent can continue with image
        inspection and ``back_project``.
        """
        nn = _latest_step() if step is None else int(step)
        if nn is None:
            return {"error": "no state entries; cannot select segment image"}

        camera = camera or "agentview"
        prompt = prompt.strip()
        has_prompt = bool(prompt)
        has_point = point is not None
        if has_prompt == has_point:
            return {"error": "segment needs exactly one of prompt or point"}
        try:
            image_path, world_path, artifact_pairs = _select_segment_artifacts(
                nn, camera
            )
        except ValueError as e:
            return {"error": str(e)}
        if image_path is None:
            return {
                "error": "complete segment artifacts not found",
                "step": nn,
                "camera": camera,
                "checked_paths": [
                    str(path)
                    for image, world in artifact_pairs
                    for path in (image, world)
                ],
            }

        try:
            data = self._sam3_client.segment(
                image_path,
                text_prompt=prompt if has_prompt else None,
                point=point,
                min_score=min_score,
            )
        except ValueError as e:
            return {
                "error": str(e),
                "step": nn,
                "camera": camera,
                "image_path": str(image_path),
            }
        except Exception as e:
            return {
                "error": f"segmentation service call failed: {e}",
                "step": nn,
                "camera": camera,
                "image_path": str(image_path),
                "fallback": "Use manual visual localization and back_project.",
            }

        out_dir = get_output_dir()
        segment_path, overlay_candidate_path, segment_index = (
            _next_segment_artifact_paths(out_dir, nn)
        )
        overlay_path = None
        mask = data.mask
        if data.found and isinstance(mask, np.ndarray):
            if world_path is None or not world_path.exists():
                world_result = {
                    "world_xyz": None,
                    "world_error": "world map artifact not found for selected image",
                    "expected_world_path": str(world_path) if world_path else None,
                }
            else:
                world_result = _mask_to_world(mask, np.load(world_path))
                world_result["world_path"] = str(world_path)
            overlay_path = overlay_candidate_path
            if not _write_segment_overlay(image_path, mask, overlay_path):
                overlay_path = None
        else:
            world_result = {
                "world_xyz": None,
                "world_error": data.reason or "segmentation did not find a mask",
            }

        segment_blob = {
            "found": data.found,
            "mode": "text" if has_prompt else "point",
            "camera": camera,
            "source_step": nn,
            "segment_index": segment_index,
            "image_path": str(image_path),
            "min_score": min_score,
            "score": round(float(data.score), 3) if data.score is not None else None,
            "box": data.box,
            "mask_shape": list(data.mask_shape) if data.mask_shape else None,
        }
        if has_prompt:
            segment_blob["prompt"] = prompt
        else:
            segment_blob["point"] = point
        if not data.found:
            segment_blob["error"] = data.reason or "SAM3 found no mask"
        segment_blob.update(world_result)
        segment_path.write_text(json.dumps(segment_blob, indent=2, default=str))

        result = {
            "found": data.found,
            "step": nn,
            "camera": camera,
            "image_path": str(image_path),
            "segment_path": str(segment_path),
            "score": segment_blob["score"],
            "box": segment_blob["box"],
            "world_xyz": segment_blob["world_xyz"],
            "world_error": segment_blob.get("world_error"),
        }
        if "error" in segment_blob:
            result["error"] = segment_blob["error"]
            result["fallback"] = "Use manual visual localization and back_project."
        if overlay_path is not None and overlay_path.exists():
            result["overlay_path"] = str(overlay_path)
        return result


# ---------------------------------------------------------------------------
# State artifacts
# ---------------------------------------------------------------------------


def _append_state(output_dir: str, blob: dict) -> None:
    """Append *blob* to ``<output_dir>/states.json`` atomically."""
    path = artifact_path(output_dir, "states")
    tmp_path = path.parent / f"{path.name}.tmp"
    if path.exists():
        with open(path) as f:
            states = json.load(f)
    else:
        states = []
    states.append(blob)
    with open(tmp_path, "w") as f:
        json.dump(states, f, indent=2)
    os.replace(tmp_path, path)


def write_recipe_from_states(output_dir: str, recipe_tag: str) -> str:
    """Find a command sequence that gets ``libero_terminated=True``.

    Export non-error LIBERO primitive commands from ``states.json`` and
    successful segment calls from ``segments/segment_*.json``.
    """
    states_path = artifact_path(output_dir, "states")
    if states_path.exists():
        with open(states_path) as f:
            states = json.load(f)
    else:
        states = []

    command_events = []
    for step_idx, entry in enumerate(states):
        if not entry:
            continue
        command = entry.get("command")
        if command is None:
            continue
        if command.get("action") not in PRIMITIVE_TOOL_NAMES:
            continue
        result = entry.get("result")
        if isinstance(result, dict) and result.get("error"):
            continue
        command_events.append(((step_idx, -1), command))

    for artifact in artifact_path(output_dir, "segments").glob("segment_*.json"):
        with artifact.open() as f:
            segment = json.load(f)
        if segment.get("error"):
            continue
        if segment["mode"] == "text":
            command = {
                "action": "segment",
                "prompt": segment["prompt"],
                "camera": segment["camera"],
            }
        else:
            command = {
                "action": "segment",
                "point": segment["point"],
                "camera": segment["camera"],
            }
        source_step = int(segment["source_step"])
        event_order = (source_step, int(segment["segment_index"]))
        command_events.append((event_order, command))

    recipe_path = os.path.join(output_dir, f"recipe_{recipe_tag}.jsonl")
    tmp_path = recipe_path + ".tmp"
    command_events.sort(key=lambda event: event[0])
    with open(tmp_path, "w") as f:
        for _, command in command_events:
            f.write(json.dumps(command, separators=(",", ":")) + "\n")
    os.replace(tmp_path, recipe_path)
    return recipe_path


def _metric_depth(depth: Any, camera_meta: dict) -> np.ndarray:
    d = np.asarray(depth, dtype=np.float32)
    if d.ndim == 3:
        d = d[..., 0]
    near = camera_meta.get("depth_near")
    far = camera_meta.get("depth_far")
    if near is not None and far is not None:
        d = near / (1.0 - d * (1.0 - near / far))
    return d


def _world_from_depth(depth_metric: np.ndarray, camera_meta: dict) -> np.ndarray:
    k_matrix = np.array(camera_meta["intrinsic_K"], dtype=np.float64)
    extrinsic = np.array(camera_meta["extrinsic_cam2world"], dtype=np.float64)
    fx, fy = k_matrix[0, 0], k_matrix[1, 1]
    cx, cy = k_matrix[0, 2], k_matrix[1, 2]
    height, width = depth_metric.shape
    rr, cc = np.mgrid[0:height, 0:width]
    z = depth_metric.astype(np.float64)
    camera_points = np.stack(
        [(cc - cx) * z / fx, (rr - cy) * z / fy, z, np.ones_like(z)],
        axis=-1,
    )
    return (camera_points @ extrinsic.T)[..., :3]


def dump_state(primitives: LiberoPrimitives, output_dir: str, step_idx: int,
               log: dict | None = None) -> dict:
    """Dump state snapshot, images, and depth for step *step_idx*.

    Writes:
      - ``<output_dir>/images/image_NN.png``       (Pi0-frame agentview)
      - ``<output_dir>/images_cam/image_cam_NN.png`` (calibration-frame agentview)
      - ``<output_dir>/depths/depth_NN.npy``        (metric depth, meters)
      - ``<output_dir>/world/world_NN.npy``         (agentview world xyz map)
      - ``<output_dir>/images_wrist/image_wrist_NN.png``
      - ``<output_dir>/depths_wrist/depth_wrist_NN.npy``
      - ``<output_dir>/world_wrist/world_wrist_NN.npy``
      - ``<output_dir>/wrist_meta/wrist_meta_NN.json``
      - high-res ``images_cam_hi`` / ``world_hi`` artifacts
      - high-res ``images_wrist_hi`` / ``world_wrist_hi`` artifacts
      - ``<output_dir>/camera_meta.json``           (static, once)
      - appends the step blob to ``<output_dir>/states.json``

    If *log* is provided (the return value of :func:`execute`), its
    ``command``, ``result``, and ``elapsed_s`` fields are merged into the
    step blob so a single entry captures everything.
    """
    for directory in ARTIFACT_DIRECTORIES:
        (Path(output_dir) / directory).mkdir(parents=True, exist_ok=True)

    agent_world_map = None
    wrist_world_map = None
    agent_world_map_hi = None
    wrist_world_map_hi = None
    # Reuse one raw observation snapshot for state and per-step artifacts.
    raw = primitives.env.raw_obs()
    # Expose robot proprioception and object names, but never privileged
    # object coordinates; the agent must localize through visual artifacts.
    state = {
        "robot0_eef_pos": [float(x) for x in raw["robot0_eef_pos"]],
        "robot0_eef_quat": [float(x) for x in raw["robot0_eef_quat"]],
        "robot0_gripper_qpos": [float(x) for x in raw["robot0_gripper_qpos"]],
        "object_names": sorted(
            k[:-4]
            for k in raw
            if k.endswith("_pos") and "robot0" not in k and "to_robot" not in k
        ),
    }
    imageio.imwrite(
        artifact_path(output_dir, "policy_image", step=step_idx, camera="agentview", resolution="low"),
        primitives._last_obs["main_images"],
    )

    # --- camera calibration (static for agentview): fetch metadata as needed ---
    agentview_meta = primitives.env.get_camera_meta(
        camera_name="agentview",
        height=256,
        width=256,
    ) or {}
    camera_meta_path = artifact_path(output_dir, "metadata", camera="agentview", resolution="low")
    if agentview_meta and not camera_meta_path.exists():
        cam_meta_out = dict(agentview_meta)
        cam_meta_out["projection"] = (
            "Prefer the back_project(row, col, step=NN) MCP tool; it "
            "uses the 1024x1024 high-resolution world map by default. "
            "Pass resolution='low' only when row/col came from the "
            "256x256 calibration-frame image."
        )
        cam_meta_out["note"] = (
            "depth_NN.npy is in this camera frame (vertical-flipped raw "
            "buffer). image_NN.png is rotated 180deg (Pi0 convention) and "
            "is NOT in the same frame as depth/K."
        )
        with open(camera_meta_path, "w") as f:
            json.dump(cam_meta_out, f, indent=2)

    # --- per-step RGB in the depth/K frame (vertical-flip of the raw buffer) ---
    # The agent picks object pixels HERE (same frame as depth_NN.npy + K), so
    # pixel -> depth -> back-project is direct. (image_NN.png is the 180°-rotated
    # Pi0-convention frame and must NOT be used for back-projection.)
    try:
        ci = raw.get("agentview_image")
        if ci is not None:
            ci = np.asarray(ci)
            if ci.dtype != np.uint8:
                ci = ci.astype(np.uint8)
            imageio.imwrite(artifact_path(output_dir, "image", step=step_idx, camera="agentview", resolution="low"), ci[::-1])
    except Exception as e:
        logger.warning("image_cam dump failed: %s", e)

    # --- per-step metric depth (agentview), native orientation, in meters ---
    try:
        d = raw.get("agentview_depth")
        if d is not None:
            # Vertical flip to align with the camera matrices: robosuite's
            # camera_utils projection M = K_exp @ inv(extrinsic) expects the
            # depth map in this frame. VERIFIED 5/5: projecting each GT object
            # world pos via M lands on a pixel whose depth_flip[row,col] matches
            # the object's surface depth (plate Δ6mm, cookies Δ14mm). So
            # pixel(row,col) in depth_NN.npy back-projects correctly with
            # camera_meta.json (NOT the same frame as the 180°-rotated
            # image_NN.png — see camera_meta note).
            d = _metric_depth(d, agentview_meta)[::-1]
            np.save(
                artifact_path(output_dir, "depth", step=step_idx, camera="agentview", resolution="low"),
                d.astype(np.float32),
            )
            world = _world_from_depth(d, agentview_meta).astype(np.float32)
            np.save(
                artifact_path(output_dir, "world", step=step_idx, camera="agentview", resolution="low"),
                world,
            )
            agent_world_map = _artifact_relative_path(
                step_idx, "agentview", "low", "world"
            )
    except Exception as e:
        logger.warning("depth dump failed: %s", e)

    # --- per-step wrist camera (robot0_eye_in_hand), calibration frame ---
    try:
        wimg = raw.get("robot0_eye_in_hand_image")
        if wimg is None:
            logger.warning("wrist image missing from raw_obs")
        else:
            wimg = np.asarray(wimg)
            if wimg.dtype != np.uint8:
                wimg = wimg.astype(np.uint8)
            imageio.imwrite(artifact_path(output_dir, "image", step=step_idx, camera="wrist", resolution="low"), wimg[::-1])
    except Exception as e:
        logger.warning("wrist image dump failed: %s", e)

    try:
        wdpt = raw.get("robot0_eye_in_hand_depth")
        if wdpt is None:
            logger.warning("wrist depth missing from raw_obs")
        else:
            wdpt_arr = np.asarray(wdpt, dtype=np.float32)
            height, width = wdpt_arr.shape[:2]
            wmeta = primitives.env.get_camera_meta(
                camera_name="robot0_eye_in_hand",
                height=int(height),
                width=int(width),
            )
            if wmeta is None:
                logger.warning("wrist camera meta missing; skipping wrist depth/world")
            else:
                wdpt_metric = _metric_depth(wdpt_arr, wmeta)[::-1]
                np.save(
                    artifact_path(output_dir, "depth", step=step_idx, camera="wrist", resolution="low"),
                    wdpt_metric.astype(np.float32),
                )
                world_w = _world_from_depth(wdpt_metric, wmeta).astype(np.float32)
                np.save(
                    artifact_path(output_dir, "world", step=step_idx, camera="wrist", resolution="low"),
                    world_w,
                )
                wrist_world_map = _artifact_relative_path(
                    step_idx, "wrist", "low", "world"
                )

                wmeta_out = dict(wmeta)
                wmeta_out["note"] = (
                    "MOVING camera: extrinsic_cam2world is for THIS step "
                    "only. world_wrist_NN.npy[row,col] gives world "
                    "(x,y,z) for that pixel, in the SAME world frame as "
                    "agentview world_NN.npy."
                )
                with open(
                    artifact_path(output_dir, "metadata", step=step_idx, camera="wrist", resolution="low"),
                    "w",
                ) as f:
                    json.dump(wmeta_out, f, indent=2)
    except Exception as e:
        logger.warning("wrist depth/world dump failed: %s", e)

    try:
        rgb_hi, depth_hi = primitives.env.render_camera(
            camera_name="agentview",
            height=1024,
            width=1024,
            depth=True,
        )
        meta_hi = primitives.env.get_camera_meta("agentview", 1024, 1024)
        if meta_hi is None:
            raise RuntimeError("agentview camera metadata missing")
        imageio.imwrite(
            artifact_path(output_dir, "image", step=step_idx, camera="agentview", resolution="high"),
            np.asarray(rgb_hi)[::-1],
        )
        world_hi = _world_from_depth(
            _metric_depth(depth_hi, meta_hi)[::-1],
            meta_hi,
        ).astype(np.float16)
        np.save(
            artifact_path(output_dir, "world", step=step_idx, camera="agentview", resolution="high"),
            world_hi,
        )
        agent_world_map_hi = _artifact_relative_path(
            step_idx, "agentview", "high", "world"
        )
    except Exception as e:
        logger.warning("agentview high-res dump failed: %s", e)

    try:
        rgb_wrist_hi, depth_wrist_hi = primitives.env.render_camera(
            camera_name="robot0_eye_in_hand",
            height=1024,
            width=1024,
            depth=True,
        )
        meta_wrist_hi = primitives.env.get_camera_meta(
            "robot0_eye_in_hand", 1024, 1024
        )
        if meta_wrist_hi is None:
            raise RuntimeError("robot0_eye_in_hand camera metadata missing")
        imageio.imwrite(
            artifact_path(output_dir, "image", step=step_idx, camera="wrist", resolution="high"),
            np.asarray(rgb_wrist_hi)[::-1],
        )
        world_wrist_hi = _world_from_depth(
            _metric_depth(depth_wrist_hi, meta_wrist_hi)[::-1],
            meta_wrist_hi,
        ).astype(np.float16)
        np.save(
            artifact_path(output_dir, "world", step=step_idx, camera="wrist", resolution="high"),
            world_wrist_hi,
        )
        wrist_world_map_hi = _artifact_relative_path(
            step_idx, "wrist", "high", "world"
        )
    except Exception as e:
        logger.warning("wrist high-res dump failed: %s", e)

    for old_step in range(max(0, int(step_idx) - 4)):
        for path in (
            artifact_path(output_dir, "image", step=old_step, camera="agentview", resolution="high"),
            artifact_path(output_dir, "world", step=old_step, camera="agentview", resolution="high"),
            artifact_path(output_dir, "image", step=old_step, camera="wrist", resolution="high"),
            artifact_path(output_dir, "world", step=old_step, camera="wrist", resolution="high"),
        ):
            try:
                os.unlink(path)
            except FileNotFoundError:
                pass

    blob = {
        "step_idx": step_idx,
        "libero_terminated": primitives.env.episode_terminated,
        "episode_truncated": primitives.env.episode_truncated,
        "task_language": primitives.env.get_task_language(),
        "state": state,
        "world_map": agent_world_map,
        "wrist_world_map": wrist_world_map,
        "world_map_hi": agent_world_map_hi,
        "wrist_world_map_hi": wrist_world_map_hi,
    }
    # Merge the execution log (command + result + elapsed_s) into the
    # state blob so a single entry captures everything for the step.
    if log is not None:
        blob["command"] = log.get("command")
        blob["result"] = log.get("result")
        blob["elapsed_s"] = log.get("elapsed_s")
    _append_state(output_dir, blob)
    return blob


# ---------------------------------------------------------------------------
# Tool schema declarations (Anthropic-shaped canonical schema)
# ---------------------------------------------------------------------------

PRIMITIVE_TOOL_NAMES: tuple[str, ...] = (
    "move_to",
    "vla_execute",
    "privileged_pick_place",
    "pi0_pick",
    "pi0_doubled",
    "release",
    "set_gripper",
    "rotate_wrist",
    "rotate_pitch",
    "move_pose",
    "semantic_joint_interact",
)

RECOVERY_MOTION_TOOL_NAMES: tuple[str, ...] = (
    "graspgen",
    "candidate_freshness",
    "curobo_reachability",
    "curobo_motiongen_pregrasp",
    "mink_reach",
    "mink_precontact",
    "mink_engage_close",
    "mink_pull",
    "progress_liveness",
)

RECOVERY_PROPOSAL_TOOL_NAMES: frozenset[str] = frozenset(
    {
        "graspgen",
        "candidate_freshness",
        "curobo_reachability",
        "progress_liveness",
    }
)

TOOLS_SPEC = [
    {
        "name": "view_driver_state",
        "description": (
            "Read step NN from `states.json` + the matching "
            "state images in {{output_dir}}. If step is "
            "null, returns the latest entry. Each entry contains the robot "
            "state, libero_terminated flag, command log, and result. Returns "
            "available PNG paths in this stable "
            "order: 1) `images/image_NN.png` (Pi0-frame agentview), "
            "2) `images_cam/image_cam_NN.png` (calibration-frame agentview), "
            "3) `images_wrist/image_wrist_NN.png` (calibration-frame wrist). "
            "High-resolution calibration-frame images are returned as file "
            "paths, not embedded as image bytes. "
            "Use the calibration-frame images for pixel back-projection; JSON "
            "state alone is not enough. Use agentview for global tabletop "
            "layout and object locations; use wrist for close-range details "
            "near the gripper, occlusions, and container/cabinet interiors."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "step": {
                    "type": ["integer", "null"],
                    "description": "Step number; 0 = initial. Null = latest.",
                },
            },
        },
    },
    {
        "name": "move_to",
        "description": (
            "Scripted EEF servo to a world-frame XYZ target via the OSC "
            "controller. Holds orientation (use rotate_wrist / rotate_pitch "
            "/ move_pose to reorient). gripper: -1 = open, +1 = close. NEVER "
            "command a single move_to with |Δxy| > 0.30 — OSC flips IK and "
            "the run corrupts; split long traversal into 2-3 mid waypoints "
            "at carry z."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "xyz": {
                    "type": "array",
                    "description": "World-frame target [x, y, z] in meters",
                    "items": {"type": "number"},
                    "minItems": 3,
                    "maxItems": 3,
                },
                "gripper": {
                    "type": "number",
                    "description": "Gripper command: -1 open, +1 close (default -1)",
                },
                "tol": {"type": "number", "description": "Position tolerance, m (default 0.012)"},
                "step_clip": {"type": "number", "description": "Per-step Δxyz cap before action_scale, m (default 0.025)"},
                "max_steps": {"type": "integer", "description": "Step budget (default 80)"},
                "action_scale": {"type": "number", "description": "OSC action scale (default 0.05)"},
                "target_yaw": {
                    "type": ["number", "null"],
                    "description": "Optional world-frame yaw target in radians",
                },
                "yaw_step_clip": {"type": "number", "description": "Per-step yaw clip, rad (default 0.10)"},
            },
            "required": ["xyz"],
        },
    },
    {
        "name": "vla_execute",
        "description": (
            "Run a planner-authored subtask through the configured VLA backend. "
            "You control the prompt, receding horizon, per-channel action scales, "
            "clip, stopping policy, and backend-specific inference parameters. "
            "The policy is queried again from a fresh observation after every "
            "chunk. Start with short horizons for contact or irreversible stages."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "prompt": {"type": "string", "description": "VLA subtask instruction."},
                "max_chunks": {"type": "integer", "description": "Replanning budget in [1,64] (default 8)."},
                "actions_per_chunk": {"type": ["integer", "null"], "description": "Execute this many predicted actions before replanning; null uses the full model horizon."},
                "mode": {"type": "string", "description": "Backend inference mode (default eval)."},
                "translation_scale": {"type": "number", "description": "Scale action xyz in [0,2]."},
                "rotation_scale": {"type": "number", "description": "Scale action rotation in [0,2]."},
                "gripper_scale": {"type": "number", "description": "Scale gripper action in [0,2]."},
                "action_clip": {"type": "number", "description": "Absolute action clip in [0.05,1]."},
                "stop_on_success": {"type": "boolean", "description": "Stop when LIBERO terminates successfully."},
                "stop_on_truncation": {"type": "boolean", "description": "Stop when the episode truncates."},
                "inference_parameters": {"type": ["object", "null"], "description": "Backend-specific parameters validated by the selected VLA server."},
            },
            "required": ["prompt"],
        },
    },
    {
        "name": "privileged_pick_place",
        "description": (
            "Actor-only bounded semantic recovery. Reads the current audited "
            "manipulated object and target poses, approaches with an open "
            "gripper, verifies grasp retention, transports at clearance, "
            "releases, and stops only on the official LIBERO predicate."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "grasp_offset_xyz": {
                    "type": "array",
                    "items": {"type": "number"},
                    "minItems": 3,
                    "maxItems": 3,
                    "default": [0.0, -0.036, 0.038],
                    "description": (
                        "Object-relative grasp offset [x, y, z] in meters "
                        "(default [0.0, -0.036, 0.038])."
                    ),
                },
                "approach_height": {
                    "type": "number",
                    "default": 0.055,
                    "description": "Open-gripper approach clearance in meters (default 0.055).",
                },
                "lift_height": {
                    "type": "number",
                    "default": 0.13,
                    "description": "Post-grasp lift distance in meters (default 0.13).",
                },
                "target_height": {
                    "type": "number",
                    "default": 0.035,
                    "description": "Placement height above the semantic target (default 0.035).",
                },
                "carry_height": {
                    "type": "number",
                    "default": 0.10,
                    "description": "Transport clearance above the semantic target (default 0.10).",
                },
                "max_steps_per_move": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 140,
                    "default": 48,
                    "description": "Bounded step budget for each Cartesian move (default 48, maximum 140).",
                },
                "grasp_pose_max_steps": {
                    "type": ["integer", "null"],
                    "minimum": 1,
                    "maximum": 140,
                    "default": None,
                    "description": "Optional cap for the final open-gripper contact move; null follows max_steps_per_move.",
                },
                "move_step_clip": {
                    "type": "number",
                    "minimum": 0.005,
                    "maximum": 0.05,
                    "default": 0.025,
                    "description": "Per-step Cartesian delta cap before controller scaling (default 0.025 m).",
                },
                "lift_step_clip": {
                    "type": ["number", "null"],
                    "minimum": 0.005,
                    "maximum": 0.05,
                    "default": None,
                    "description": "Optional slower cap for the verified lift phase; null follows move_step_clip.",
                },
                "transport_step_clip": {
                    "type": ["number", "null"],
                    "minimum": 0.005,
                    "maximum": 0.05,
                    "default": None,
                    "description": "Optional cap for closed-gripper carry/place phases; null follows move_step_clip.",
                },
                "contact_lift_min_m": {
                    "type": "number",
                    "minimum": 0.005,
                    "maximum": 0.05,
                    "default": 0.02,
                    "description": "Minimum measured object rise required for contact-only retention (default 0.02 m).",
                },
                "max_segment_distance_m": {
                    "type": ["number", "null"],
                    "minimum": 0.10,
                    "maximum": 0.50,
                    "default": None,
                    "description": "Optional maximum Cartesian distance per waypoint for long recovery carries.",
                },
                "vertical_first_carry": {
                    "type": "boolean",
                    "default": False,
                    "description": "Raise to the carry clearance before horizontal transport (default false).",
                },
                "approach_pitch": {
                    "type": ["number", "null"],
                    "minimum": -3.141592653589793,
                    "maximum": 3.141592653589793,
                    "default": None,
                    "description": "Optional world-frame pitch held through approach, grasp, lift, carry, and place.",
                },
                "approach_yaw": {
                    "type": ["number", "null"],
                    "minimum": -3.141592653589793,
                    "maximum": 3.141592653589793,
                    "default": None,
                    "description": "Optional world-frame yaw held through approach, grasp, lift, carry, and place.",
                },
                "grasp_confirm_steps": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 8,
                    "default": 4,
                    "description": "Consecutive retained-grasp confirmations (default 4).",
                },
            },
        },
    },
    {
        "name": "pi0_pick",
        "description": (
            "Pi0.5 closed-loop pick. Use it for the grasp; YOU then do "
            "every move_to and release. Use modest max_chunks and verify "
            "the grasp from EEF lift, gripper closure, and available images."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "Pi0 prompt (e.g. 'pick up the akita black bowl').",
                },
                "max_chunks": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 8,
                    "description": "Hard-bounded grasp budget in [1,8] (default 8).",
                },
                "lift_thresh": {"type": "number", "description": "EEF post-descent ascent threshold for success, m (default 0.05)"},
                "gripper_closed_thresh": {"type": "number", "description": "Finger-separation closed threshold (default 0.06)"},
            },
            "required": ["prompt"],
        },
    },
    {
        "name": "pi0_doubled",
        "description": (
            "Pi0.5 closed-loop contact skill for non-pick interactions "
            "(e.g. stove/knob/button/short push). Returned success/task_success "
            "only mirrors official libero_terminated; for intermediate contact "
            "skills, success=false does not necessarily mean the contact "
            "interaction failed. Inspect image/state evidence. Do not use it "
            "as a general pick/place shortcut."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "Contact-skill prompt, e.g. 'turn on the stove'.",
                },
                "max_chunks": {"type": "integer", "description": "Action-chunk budget (default 20)"},
            },
            "required": ["prompt"],
        },
    },
    {
        "name": "release",
        "description": (
            "Open the gripper for up to max_steps env steps while holding "
            "EEF in place. Triggers libero termination if the matching "
            "On/In predicate is met."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "max_steps": {"type": "integer", "description": "Step budget (default 20)"},
            },
        },
    },
    {
        "name": "set_gripper",
        "description": (
            "Hold the current EEF pose and drive the gripper command for "
            "`steps` env steps. Use to firm up a grip mid-carry."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "gripper": {
                    "type": "number",
                    "description": "Gripper command: -1 open, +1 close (default -1)",
                },
                "steps": {"type": "integer", "description": "Number of env steps (default 5)"},
            },
        },
    },
    {
        "name": "rotate_wrist",
        "description": (
            "Rotate the wrist around the world Z-axis. Provide either "
            "target_yaw (absolute) or delta_yaw (relative). Holds xyz fixed."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "target_yaw": {"type": ["number", "null"], "description": "Absolute world-frame yaw target, rad"},
                "delta_yaw": {"type": ["number", "null"], "description": "Relative yaw delta, rad"},
                "gripper": {"type": "number", "description": "Gripper command held during rotation (default +1)"},
                "max_steps": {"type": "integer", "description": "Step budget (default 40)"},
                "tol": {"type": "number", "description": "Yaw tolerance, rad (default 0.02)"},
                "step_clip": {"type": "number", "description": "Per-step yaw clip, rad (default 0.10)"},
            },
        },
    },
    {
        "name": "rotate_pitch",
        "description": (
            "Tilt the gripper around the world X-axis. Provide either "
            "target_pitch (absolute) or delta_pitch (relative). Holds xyz "
            "and yaw fixed. Use before threading the gripper into a narrow "
            "opening whose front face normal is along world ±y (e.g. "
            "microwave cavity)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "target_pitch": {"type": ["number", "null"], "description": "Absolute world-frame pitch target, rad"},
                "delta_pitch": {"type": ["number", "null"], "description": "Relative pitch delta, rad"},
                "gripper": {"type": "number", "description": "Gripper command held during rotation (default +1)"},
                "max_steps": {"type": "integer", "description": "Step budget (default 40)"},
                "tol": {"type": "number", "description": "Pitch tolerance, rad (default 0.02)"},
                "step_clip": {"type": "number", "description": "Per-step pitch clip, rad (default 0.10)"},
            },
        },
    },
    {
        "name": "move_pose",
        "description": (
            "Servo position AND orientation (pitch + yaw) SIMULTANEOUSLY. "
            "Unlike move_to (holds orientation) + rotate_pitch (holds xyz), "
            "this co-varies xyz and wrist tilt every env.step. Use to thread "
            "cabinet-front / low-shelf poses where a decoupled position "
            "servo drives the wrist into an IK singularity and stalls."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "xyz": {
                    "type": "array",
                    "description": "World-frame target [x, y, z] in meters",
                    "items": {"type": "number"},
                    "minItems": 3,
                    "maxItems": 3,
                },
                "target_pitch": {"type": ["number", "null"], "description": "Absolute pitch target, rad"},
                "target_yaw": {"type": ["number", "null"], "description": "Absolute yaw target, rad"},
                "gripper": {"type": "number", "description": "Gripper command held during the move (default -1)"},
                "step_clip": {"type": "number", "description": "Per-step Δxyz cap, m (default 0.02)"},
                "pitch_step": {"type": "number", "description": "Per-step pitch clip, rad (default 0.08)"},
                "yaw_step": {"type": "number", "description": "Per-step yaw clip, rad (default 0.08)"},
                "tol": {"type": "number", "description": "Position tolerance, m (default 0.012)"},
                "ori_tol": {"type": "number", "description": "Orientation tolerance, rad (default 0.05)"},
                "action_scale": {"type": "number", "description": "OSC action scale (default 0.05)"},
                "max_steps": {"type": "integer", "description": "Step budget (default 150)"},
            },
            "required": ["xyz"],
        },
    },
    {
        "name": "semantic_joint_interact",
        "description": (
            "Privileged, audited fixture-joint contact primitive. Select a "
            "declared semantic entity and joint plus the desired lower/upper "
            "range endpoint. The local tool resolves randomized MuJoCo geometry "
            "privately, approaches the contact surface, and uses only bounded "
            "OSC actions with joint feedback. It never writes qpos, predicates, "
            "reward, or success; official success remains libero_terminated. "
            "Use for hinged knobs, buttons, doors, or drawers only when the "
            "Critic evidence identifies the intended semantic joint."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "entity": {
                    "type": "string",
                    "description": "Declared LIBERO/BDDL entity name, for example flat_stove_1.",
                },
                "joint": {
                    "type": "string",
                    "description": "Semantic or fully-qualified joint name, for example button.",
                },
                "direction": {
                    "type": "string",
                    "enum": ["lower", "upper"],
                    "description": "Desired endpoint of the audited joint range.",
                },
                "max_sweep_steps": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 64,
                    "description": "Bounded feedback sweep budget (default 64).",
                },
                "sweep_step_m": {
                    "type": "number",
                    "minimum": 0.001,
                    "maximum": 0.015,
                    "description": "Tangential Cartesian increment per step (default 0.015 m).",
                },
                "close_steps": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 8,
                    "description": "Gripper-contact settling steps (default 3).",
                },
            },
            "required": ["entity", "joint", "direction"],
        },
    },
    {
        "name": "view_camera_meta",
        "description": (
            "Read camera calibration metadata from the output dir. "
            "camera='agentview' reads static camera_meta.json. "
            "camera='wrist' reads the per-step wrist metadata."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "camera": {
                    "type": "string",
                    "enum": ["agentview", "wrist"],
                    "description": "Camera metadata to read (default agentview).",
                },
                "step": {
                    "type": ["integer", "null"],
                    "description": "Wrist metadata step to use (default latest).",
                },
            },
        },
    },
    {
        "name": "segment",
        "description": (
            "SAM3 visual segmentation over an existing run artifact. It never "
            "renders a new camera view. Provide exactly one text prompt or "
            "single positive point. A successful top-ranked mask is projected "
            "through the matching world map to produce world_xyz."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "Object/text prompt to segment.",
                },
                "camera": {
                    "type": "string",
                    "enum": ["agentview", "wrist"],
                    "description": "Artifact camera to use (default agentview).",
                },
                "step": {
                    "type": ["integer", "null"],
                    "description": "Step NN to segment; null = latest.",
                },
                "point": {
                    "type": ["array", "null"],
                    "description": (
                        "Optional single positive point as [row, col]. "
                        "Mutually exclusive with prompt."
                    ),
                    "items": {"type": "integer"},
                    "minItems": 2,
                    "maxItems": 2,
                },
                "min_score": {
                    "type": "number",
                    "description": "Minimum accepted mask score (default 0.2).",
                },
            },
        },
    },
    {
        "name": "back_project",
        "description": (
            "Back-project a pixel (row, col) to a world XYZ point using the "
            "selected camera's precomputed world map. Row 0 = top of image, "
            "col 0 = left. Returns world_xyz in meters.\n\n"
            "USE THIS to find where an object is in the world — look at "
            "the high-resolution paths returned by view_driver_state "
            "to pick a pixel on the target object, then call back_project. "
            "The default resolution is high (1024x1024). Pass "
            "resolution='low' only for pixels from the embedded/standard "
            "256 image. The pixel coordinates must come "
            "from the same camera and resolution requested here. Use "
            "camera='agentview' for global tabletop layout and object "
            "locations; use camera='wrist' for close-range details near the "
            "gripper, occlusions, and container/cabinet interiors. "
            "Sample several pixels on the object and median their xy for "
            "robustness.\n\n"
            "REGION MODE: pass row_range=[r0,r1] and col_range=[c0,c1] instead "
            "of row/col to get the midpoint of world xy over that pixel window, "
            "with an optional world-z band (z_min, z_max). Use it for the "
            "center of a container cavity or flat region, where a single-pixel "
            "or mask-median estimate is biased toward an edge/rim."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "row": {
                    "type": ["integer", "null"],
                    "description": "Pixel row (0=top) in the selected resolution image.",
                },
                "col": {
                    "type": ["integer", "null"],
                    "description": "Pixel column (0=left) in the selected resolution image.",
                },
                "step": {
                    "type": ["integer", "null"],
                    "description": "Depth/world-map step to use (default latest). 0 for initial.",
                },
                "camera": {
                    "type": "string",
                    "enum": ["agentview", "wrist"],
                    "description": "Camera to back-project from (default agentview).",
                },
                "resolution": {
                    "type": "string",
                    "enum": ["high", "low"],
                    "description": (
                        "Coordinate system for row/col (default high). "
                        "Use low only when row/col came from the "
                        "embedded/standard 256 image."
                    ),
                },
                "row_range": {
                    "type": ["array", "null"],
                    "items": {"type": "integer"},
                    "description": "Region mode: [r0, r1] pixel row window. Requires col_range.",
                },
                "col_range": {
                    "type": ["array", "null"],
                    "items": {"type": "integer"},
                    "description": "Region mode: [c0, c1] pixel col window. Requires row_range.",
                },
                "z_min": {
                    "type": ["number", "null"],
                    "description": "Region mode: keep only pixels with world z >= z_min.",
                },
                "z_max": {
                    "type": ["number", "null"],
                    "description": "Region mode: keep only pixels with world z <= z_max.",
                },
            },
        },
    },
    {
        "name": "verify_transition",
        "description": (
            "Read two ordered state snapshots and verify generic temporal "
            "evidence: EEF motion, gripper change, official termination, and "
            "absence of truncation. This never moves the robot and does not "
            "replace LIBERO's authoritative task predicate."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "before_step": {"type": "integer", "description": "Earlier recorded tool step."},
                "after_step": {"type": ["integer", "null"], "description": "Later step; null uses latest."},
                "min_eef_motion_m": {"type": ["number", "null"], "description": "Optional minimum Euclidean EEF displacement."},
                "min_gripper_change": {"type": ["number", "null"], "description": "Optional minimum gripper-qpos L2 change."},
                "require_terminated": {"type": "boolean", "description": "Require official LIBERO success at after_step."},
                "reject_truncation": {"type": "boolean", "description": "Fail if the episode was truncated (default true)."},
            },
            "required": ["before_step"],
        },
    },
]

TOOLS_SPEC.extend(
    [
        {
            "name": "graspgen",
            "description": "Generate sensor-only 6D grasp proposals and cache their audited candidate lifecycle without stepping the environment.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "mode": {"type": "string", "enum": ["health", "propose"]},
                    "camera": {"type": "string", "enum": ["agentview", "wrist"]},
                    "step": {"type": ["integer", "null"], "minimum": 0},
                    "target_world_xyz": {"type": ["array", "null"], "items": {"type": "number"}, "minItems": 3, "maxItems": 3},
                    "crop_radius_m": {"type": "number", "minimum": 0.01, "maximum": 1.0},
                    "max_candidates": {"type": "integer", "minimum": 1, "maximum": 64},
                    "max_points": {"type": "integer", "minimum": 32, "maximum": 65536},
                    "timeout_s": {"type": "number", "minimum": 1, "maximum": 900},
                },
            },
        },
        {
            "name": "candidate_freshness",
            "description": "Validate or invalidate the episode-local GraspGen candidate using action age and EEF displacement; never steps the environment.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "max_age_steps": {"type": "integer", "minimum": 0, "maximum": 128},
                    "max_eef_delta_m": {"type": "number", "minimum": 0, "maximum": 1.0},
                    "invalidate": {"type": "boolean"},
                },
            },
        },
        {
            "name": "curobo_reachability",
            "description": "Run the CuRobo-compatible proposal-only reachability gate for an explicit or cached target. The local fallback is workspace-only, not a path-collision certificate.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "target_xyz": {"type": ["array", "null"], "items": {"type": "number"}, "minItems": 3, "maxItems": 3},
                    "candidate_index": {"type": "integer", "minimum": 0, "maximum": 63},
                    "max_distance_m": {"type": "number", "minimum": 0.01, "maximum": 2.0},
                },
            },
        },
        {
            "name": "curobo_motiongen_pregrasp",
            "description": "Execute one bounded pregrasp from a GraspGen candidate through the CuRobo-compatible LIBERO OSC fallback, then re-observe.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "target_xyz": {"type": ["array", "null"], "items": {"type": "number"}, "minItems": 3, "maxItems": 3},
                    "candidate_index": {"type": "integer", "minimum": 0, "maximum": 63},
                    "pregrasp_offset_m": {"type": "number", "minimum": -0.15, "maximum": 0.15},
                    "target_pitch": {"type": ["number", "null"]},
                    "target_yaw": {"type": ["number", "null"]},
                    "max_steps": {"type": "integer", "minimum": 1, "maximum": 150},
                },
            },
        },
        {
            "name": "mink_reach",
            "description": "Execute one bounded Mink-compatible local reach fallback from an explicit or cached candidate, then re-observe.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "target_xyz": {"type": ["array", "null"], "items": {"type": "number"}, "minItems": 3, "maxItems": 3},
                    "candidate_index": {"type": "integer", "minimum": 0, "maximum": 63},
                    "target_pitch": {"type": ["number", "null"]},
                    "target_yaw": {"type": ["number", "null"]},
                    "max_steps": {"type": "integer", "minimum": 1, "maximum": 120},
                },
            },
        },
        {
            "name": "mink_precontact",
            "description": "Execute a bounded open-gripper precontact approach from an explicit or cached candidate and return fresh tracking evidence.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "target_xyz": {"type": ["array", "null"], "items": {"type": "number"}, "minItems": 3, "maxItems": 3},
                    "candidate_index": {"type": "integer", "minimum": 0, "maximum": 63},
                    "standoff_m": {"type": "number", "minimum": 0, "maximum": 0.15},
                    "target_pitch": {"type": ["number", "null"]},
                    "target_yaw": {"type": ["number", "null"]},
                    "max_steps": {"type": "integer", "minimum": 1, "maximum": 100},
                },
            },
        },
        {
            "name": "mink_engage_close",
            "description": "Close the gripper and execute one bounded micro-advance while preserving the cached candidate orientation.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "target_xyz": {"type": ["array", "null"], "items": {"type": "number"}, "minItems": 3, "maxItems": 3},
                    "candidate_index": {"type": "integer", "minimum": 0, "maximum": 63},
                    "micro_advance_m": {"type": "number", "minimum": 0, "maximum": 0.04},
                    "close_steps": {"type": "integer", "minimum": 1, "maximum": 8},
                    "max_steps": {"type": "integer", "minimum": 1, "maximum": 64},
                },
            },
        },
        {
            "name": "mink_pull",
            "description": "Execute one incremental closed-gripper Cartesian pull and yield for fresh Critic/Role1 review.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "delta_xyz": {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3},
                    "max_steps": {"type": "integer", "minimum": 1, "maximum": 64},
                },
            },
        },
        {
            "name": "progress_liveness",
            "description": "Measure physical EEF progress over recent motion-tool calls and propose re-observation or candidate switching without stepping.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "window": {"type": "integer", "minimum": 1, "maximum": 32},
                    "min_eef_progress_m": {"type": "number", "minimum": 0, "maximum": 0.2},
                },
            },
        },
    ]
)


# ---------------------------------------------------------------------------
# State trace readers
# ---------------------------------------------------------------------------


def _load_states() -> list:
    """Return the parsed state trace from the local output dir."""
    path = artifact_path(get_output_dir(), "states")
    if not path.exists():
        return []
    with open(path) as f:
        return json.load(f)


def _latest_step() -> int | None:
    states = _load_states()
    if not states:
        return None
    return states[-1]["step_idx"]


def _load_step(nn: int) -> dict:
    """Look up the state blob for step ``nn`` from states.json."""
    for entry in _load_states():
        if entry.get("step_idx") == nn:
            return entry
    raise FileNotFoundError(f"step {nn} not present in states.json")


def verify_transition(
    before_step: int,
    after_step: int | None = None,
    *,
    min_eef_motion_m: float | None = None,
    min_gripper_change: float | None = None,
    require_terminated: bool = False,
    reject_truncation: bool = True,
) -> dict[str, Any]:
    """Verify an ordered state transition using only recorded evidence.

    This temporal check is deliberately task agnostic.  It prevents a planner
    from declaring a stage complete from one image while still leaving the
    authoritative LIBERO predicate to the environment.
    """
    latest = _latest_step()
    if latest is None:
        return {"passed": False, "error": "no state entries"}
    before_idx = int(before_step)
    after_idx = latest if after_step is None else int(after_step)
    if after_idx <= before_idx:
        return {
            "passed": False,
            "error": "after_step must be greater than before_step",
            "before_step": before_idx,
            "after_step": after_idx,
        }
    try:
        before = _load_step(before_idx)
        after = _load_step(after_idx)
    except FileNotFoundError as exc:
        return {"passed": False, "error": str(exc)}

    before_state = before.get("state") or {}
    after_state = after.get("state") or {}
    before_pos = np.asarray(before_state.get("robot0_eef_pos", []), dtype=np.float32)
    after_pos = np.asarray(after_state.get("robot0_eef_pos", []), dtype=np.float32)
    before_grip = np.asarray(
        before_state.get("robot0_gripper_qpos", []), dtype=np.float32
    )
    after_grip = np.asarray(
        after_state.get("robot0_gripper_qpos", []), dtype=np.float32
    )
    if before_pos.shape != (3,) or after_pos.shape != (3,):
        return {"passed": False, "error": "state trace lacks 3-D EEF positions"}

    motion = float(np.linalg.norm(after_pos - before_pos))
    gripper_change = (
        float(np.linalg.norm(after_grip - before_grip))
        if before_grip.shape == after_grip.shape and before_grip.size
        else None
    )
    checks: dict[str, bool] = {}
    if min_eef_motion_m is not None:
        checks["eef_motion"] = motion >= float(min_eef_motion_m)
    if min_gripper_change is not None:
        checks["gripper_change"] = (
            gripper_change is not None
            and gripper_change >= float(min_gripper_change)
        )
    if require_terminated:
        checks["libero_terminated"] = bool(after.get("libero_terminated"))
    if reject_truncation:
        checks["not_truncated"] = not bool(after.get("episode_truncated"))
    return {
        "passed": all(checks.values()) if checks else True,
        "before_step": before_idx,
        "after_step": after_idx,
        "checks": checks,
        "evidence": {
            "eef_motion_m": round(motion, 5),
            "gripper_change": (
                round(gripper_change, 5) if gripper_change is not None else None
            ),
            "libero_terminated": bool(after.get("libero_terminated")),
            "episode_truncated": bool(after.get("episode_truncated")),
            "after_command": after.get("command"),
        },
    }


def _load_image_path(nn: int, kind: str) -> str | None:
    """Return the path to a dumped state image. None if not present."""
    out_dir = get_output_dir()
    if kind == "agent":
        path = artifact_path(out_dir, "policy_image", step=nn, camera="agentview", resolution="low")
    elif kind == "camera":
        path = artifact_path(out_dir, "image", step=nn, camera="agentview", resolution="low")
    elif kind == "wrist":
        path = artifact_path(out_dir, "image", step=nn, camera="wrist", resolution="low")
    else:
        raise ValueError(f"unknown image kind: {kind}")
    if not path.exists():
        return None
    return str(path)


def _load_camera_meta(camera: str = "agentview", nn: int | None = None) -> dict:
    out_dir = get_output_dir()
    if camera == "agentview":
        path = artifact_path(out_dir, "metadata", camera="agentview", resolution="low")
    elif camera == "wrist" and nn is not None:
        path = artifact_path(out_dir, "metadata", step=nn, camera="wrist", resolution="low")
    else:
        raise ValueError("camera must be 'agentview' or 'wrist' with nn")
    if not path.exists():
        raise FileNotFoundError(f"{path.name} not found in {out_dir}")
    with open(path) as f:
        return json.load(f)


def _load_depth(camera: str, nn: int) -> np.ndarray:
    out_dir = get_output_dir()
    if camera not in ("agentview", "wrist"):
        raise ValueError("camera must be 'agentview' or 'wrist'")
    path = artifact_path(out_dir, "depth", step=nn, camera=camera, resolution="low")
    if not path.exists():
        raise FileNotFoundError(f"{path.name} not found in {out_dir}")
    depth = np.load(path)
    if depth.ndim == 3:
        depth = depth[..., 0]
    return depth


def view_driver_state(step: int | None = None) -> dict:
    latest = _latest_step()
    if latest is None:
        return {"error": "no state entries; env not ready"}
    nn = latest if step is None else int(step)
    try:
        data = _load_step(nn)
    except Exception as e:
        return {"error": f"step {nn} not present in state trace: {e}"}

    out: dict = {"step": nn}
    out["task_language"] = data.get("task_language")
    out["state"] = data["state"]
    out["libero_terminated"] = data.get("libero_terminated")
    out["episode_truncated"] = data.get("episode_truncated")
    out["world_map"] = data.get("world_map")
    out["wrist_world_map"] = data.get("wrist_world_map")
    out["world_map_hi"] = data.get("world_map_hi")
    out["wrist_world_map_hi"] = data.get("wrist_world_map_hi")
    out["log"] = {
        "command": data.get("command"),
        "result": data.get("result"),
        "elapsed_s": data.get("elapsed_s"),
    }
    for field, kind in (
        ("image_path", "agent"),
        ("image_cam_path", "camera"),
        ("image_wrist_path", "wrist"),
    ):
        image_path = _load_image_path(nn, kind)
        if image_path:
            out[field] = image_path
    for field, camera in (
        ("image_cam_hi_path", "agentview"),
        ("image_wrist_hi_path", "wrist"),
    ):
        image_path = artifact_path(get_output_dir(), "image", step=nn, camera=camera, resolution="high")
        if image_path.exists():
            out[field] = str(image_path)
    return out


def _select_segment_artifacts(nn: int, camera: str):
    out_dir = get_output_dir()
    if camera not in ("agentview", "wrist"):
        raise ValueError(f"unknown segment camera: {camera}")
    pairs = [
        (
            artifact_path(out_dir, "image", step=nn, camera=camera, resolution=resolution),
            artifact_path(out_dir, "world", step=nn, camera=camera, resolution=resolution),
        )
        for resolution in ("high", "low")
    ]

    for image_path, world_path in pairs:
        if image_path.exists() and world_path.exists():
            return image_path, world_path, pairs
    return None, None, pairs


def _next_segment_artifact_paths(out_dir: Path, nn: int):
    segments_dir = artifact_path(out_dir, "segments")
    segments_dir.mkdir(parents=True, exist_ok=True)
    idx = 0
    while True:
        segment_path = segments_dir / f"segment_{nn:02d}_{idx:02d}.json"
        overlay_path = segments_dir / f"segment_overlay_{nn:02d}_{idx:02d}.png"
        if not segment_path.exists() and not overlay_path.exists():
            return segment_path, overlay_path, idx
        idx += 1


def _mask_to_world(mask: np.ndarray, world_map: np.ndarray,
                   min_valid: int = 10) -> dict:
    if world_map.ndim != 3 or world_map.shape[2] < 3:
        return {
            "world_xyz": None,
            "world_error": f"invalid world map shape: {tuple(world_map.shape)}",
            "n_pixels": int(mask.sum()),
            "n_valid": 0,
            "mask_resized_to_world_shape": False,
        }

    if mask.shape != world_map.shape[:2]:
        return {
            "world_xyz": None,
            "world_error": (
                f"mask/world shape mismatch: mask={tuple(mask.shape)}, "
                f"world={tuple(world_map.shape[:2])}"
            ),
            "n_pixels": int(mask.sum()),
            "n_valid": 0,
            "mask_resized_to_world_shape": False,
        }

    ys, xs = np.where(mask)
    if ys.size == 0:
        return {"world_xyz": None, "world_error": "empty mask"}

    pts = world_map[ys, xs].astype(np.float64)
    valid = np.isfinite(pts).all(axis=1) & (np.abs(pts).sum(axis=1) > 1e-6)
    pts = pts[valid]
    result = {
        "centroid_pixel": [
            int(round(float(np.median(xs)))),
            int(round(float(np.median(ys)))),
        ],
        "n_pixels": int(mask.sum()),
        "n_valid": int(pts.shape[0]),
        "mask_resized_to_world_shape": False,
    }
    if pts.shape[0] < min_valid:
        result.update({
            "world_xyz": None,
            "world_error": f"too few valid depth pixels ({int(pts.shape[0])})",
        })
        return result

    result["world_xyz"] = [
        round(float(np.median(pts[:, 0])), 4),
        round(float(np.median(pts[:, 1])), 4),
        round(float(np.median(pts[:, 2])), 4),
    ]
    return result


def _write_segment_overlay(image_path: Path, mask: np.ndarray,
                           overlay_path: Path) -> bool:
    try:
        image = imageio.imread(image_path)
        if image.ndim != 3 or image.shape[:2] != mask.shape:
            return False
        overlay = image.copy()
        red = np.zeros_like(overlay)
        red[..., 0] = 255
        overlay[mask] = (
            0.55 * overlay[mask].astype(np.float32)
            + 0.45 * red[mask].astype(np.float32)
        ).astype(np.uint8)
        imageio.imwrite(overlay_path, overlay)
        return overlay_path.exists()
    except Exception:
        return False


def view_camera_meta(camera: str = "agentview", step: int | None = None) -> dict:
    """Read camera calibration metadata for localization."""
    if camera not in ("agentview", "wrist"):
        return {"error": f"bad camera '{camera}' (use 'agentview' or 'wrist')"}

    nn = None
    if camera == "wrist":
        nn = _latest_step() if step is None else int(step)
        if nn is None:
            return {"error": "no wrist metadata available"}

    try:
        meta = _load_camera_meta(camera, nn)
    except Exception as e:
        return {"error": f"{camera} camera metadata not found: {e}"}

    if camera == "agentview":
        return {"camera": "agentview", "camera_meta": meta}
    return {"camera": "wrist", "step": nn, "camera_meta": meta}


def back_project(
    row: int | None = None,
    col: int | None = None,
    step: int | None = None,
    camera: str = "agentview",
    resolution: str = "high",
    row_range: list | None = None,
    col_range: list | None = None,
    z_min: float | None = None,
    z_max: float | None = None,
) -> dict:
    """Look up a pixel's world XYZ in the precomputed world map."""
    if camera not in ("agentview", "wrist"):
        return {"error": f"bad camera '{camera}' (use 'agentview' or 'wrist')"}
    if resolution not in ("high", "low"):
        return {"error": f"bad resolution '{resolution}' (use 'high' or 'low')"}

    region_mode = row_range is not None or col_range is not None
    if not region_mode and (row is None or col is None):
        return {
            "error": (
                "provide either (row, col) for a single pixel, or "
                "row_range=[r0,r1] and col_range=[c0,c1] for a region center"
            )
        }

    latest = _latest_step()
    nn = latest if step is None else int(step)
    if nn is None:
        return {"error": "no depth/world-map files available"}

    try:
        data = _load_step(nn)
    except Exception as e:
        return {"error": f"step {nn} not present in state trace: {e}"}

    if camera == "agentview":
        hi_artifact = data.get("world_map_hi")
        low_artifact = data.get("world_map")
    else:
        hi_artifact = data.get("wrist_world_map_hi")
        low_artifact = data.get("wrist_world_map")
    source_artifact = hi_artifact if resolution == "high" else low_artifact
    if not source_artifact:
        return {
            "error": (
                f"{camera} {resolution}-resolution world map not recorded "
                f"for step {nn}"
            )
        }

    try:
        world_path = artifact_path(get_output_dir(), "world", step=nn, camera=camera, resolution=resolution)
        world_map = np.load(world_path)
    except Exception as e:
        return {
            "error": (
                f"{camera} {resolution}-resolution artifact not found "
                f"for step {nn}: {e}"
            )
        }

    height, width = world_map.shape[:2]

    if region_mode:
        if row_range is None or col_range is None:
            return {
                "error": "region mode needs BOTH row_range=[r0,r1] and col_range=[c0,c1]"
            }
        try:
            r0, r1 = int(row_range[0]), int(row_range[1])
            c0, c1 = int(col_range[0]), int(col_range[1])
        except Exception:
            return {"error": "row_range/col_range must each be [min, max] integers"}
        r0, r1 = sorted((max(0, r0), min(height, r1)))
        c0, c1 = sorted((max(0, c0), min(width, c1)))
        if r1 <= r0 or c1 <= c0:
            return {
                "error": (
                    f"empty region after clamping to image {height}x{width}: "
                    f"rows [{r0},{r1}] cols [{c0},{c1}]"
                )
            }
        window = world_map[r0:r1, c0:c1].reshape(-1, world_map.shape[2]).astype(
            np.float64
        )
        finite = np.isfinite(window).all(axis=1) & (
            np.abs(window[:, :3]).sum(axis=1) > 1e-6
        )
        pts = window[finite]
        n_total = int(pts.shape[0])
        if z_min is not None:
            pts = pts[pts[:, 2] >= float(z_min)]
        if z_max is not None:
            pts = pts[pts[:, 2] <= float(z_max)]
        if pts.shape[0] < 8:
            return {
                "error": (
                    f"too few valid pixels in region after z-filter "
                    f"({int(pts.shape[0])}); widen the window or the z band"
                ),
                "n_valid_before_zfilter": n_total,
            }
        xs, ys, zs = pts[:, 0], pts[:, 1], pts[:, 2]
        center = [
            round(float((xs.min() + xs.max()) / 2.0), 4),
            round(float((ys.min() + ys.max()) / 2.0), 4),
            round(float(np.median(zs)), 4),
        ]
        return {
            "camera": camera,
            "resolution": resolution,
            "mode": "region",
            "row_range": [r0, r1],
            "col_range": [c0, c1],
            "z_band": [z_min, z_max],
            "center_xyz": center,
            "median_xyz": [
                round(float(np.median(xs)), 4),
                round(float(np.median(ys)), 4),
                round(float(np.median(zs)), 4),
            ],
            "n_valid": int(pts.shape[0]),
            "step": nn,
            "image_size": [height, width],
            "source_artifact": source_artifact,
        }

    if row < 0 or row >= height or col < 0 or col >= width:
        return {
            "error": (
                f"pixel ({row},{col}) out of bounds; {camera} image is "
                f"{height}x{width}"
            )
        }

    depth_m = None
    if source_artifact == low_artifact:
        try:
            depth = _load_depth(camera, nn)
        except Exception as e:
            return {"error": f"{camera} depth not found for step {nn}: {e}"}
        depth_m = float(depth[row, col])
        if not np.isfinite(depth_m) or depth_m <= 0 or depth_m > 10:
            return {
                "error": (
                    f"invalid {camera} depth {depth_m:.3f}m at pixel "
                    f"({row},{col}); pick a different pixel"
                )
            }
    world_xyz_raw = world_map[row, col]
    if (
        not np.isfinite(world_xyz_raw).all()
        or float(np.abs(world_xyz_raw[:3]).sum()) <= 1e-6
    ):
        return {"error": f"invalid {camera} world xyz at pixel ({row},{col})"}
    world_xyz = [round(float(v), 4) for v in world_xyz_raw[:3]]

    out = {
        "camera": camera,
        "resolution": resolution,
        "pixel": [row, col],
        "world_xyz": world_xyz,
        "step": nn,
        "image_size": [height, width],
        "source_artifact": source_artifact,
    }
    if depth_m is not None:
        out["depth_m"] = round(depth_m, 4)
    return out
