#!/usr/bin/env python3
# Copyright (c) 2026 Zetta Contributors
"""Dump RoboTwin camera frames so an episode can actually be looked at.

RoboTwin records nothing on its own: ``robotwin/envs/vector_env.py`` sets
``args["eval_video_log"] = False`` unconditionally, overriding whatever the task
config asks for, and RLinf's ``robotwin_env`` stores ``video_cfg`` without ever
reading it. So the frames have to be captured on the way past.

This script drives the **real** ``RobotwinEnvCore``, which means the frames it
writes are the ones a policy actually saw, with the D1 camera mapping applied
(head -> ``main_image``, left wrist -> ``wrist_image``, right wrist ->
``extra_view_images[0]``). Images arrive already PNG-encoded inside the
``Observation``, so capturing a frame is writing those bytes out -- no re-encode,
nothing to drift.

Two modes:

``--scene-only``
    Reset at a few seeds and dump the opening frame of each. Cheap: no
    checkpoint, a few seconds. Answers "what does this task look like".

default
    Run one episode under the policy and dump every simulator step. The env is
    built with ``execute_horizon=1`` so each submitted action yields an
    observation, while the policy is still only re-run every ``--replan-every``
    steps -- that is the preset's control behaviour at full frame rate, rather
    than the 8-frame slideshow a chunk-granular capture would give.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import numpy as np  # noqa: E402

from rollout_runtime.api.ids import EpisodeId, RequestId, SessionId  # noqa: E402
from rollout_runtime.api.internal import InferenceRequest  # noqa: E402
from rollout_runtime.api.messages import (  # noqa: E402
    EnvSpecMsg,
    Observation,
    ResetSpec,
)
from rollout_runtime.backends.rlinf_robotwin import RobotwinEnvCore  # noqa: E402

CAMERAS = ("head", "left_wrist", "right_wrist")
"""Output names for the three views, in the order the contract fixes them."""


def _frames(observation: Observation) -> dict[str, bytes]:
    """Pull the PNG bytes for each camera out of an observation.

    Args:
        observation: A chunk-final observation.

    Returns:
        Camera name -> PNG bytes, omitting views the config did not render.
    """
    out: dict[str, bytes] = {}
    if observation.main_image is not None:
        out["head"] = observation.main_image.data
    if observation.wrist_image is not None:
        out["left_wrist"] = observation.wrist_image.data
    if observation.extra_view_images:
        out["right_wrist"] = observation.extra_view_images[0].data
    return out


def _write(frames: dict[str, bytes], root: Path, index: int) -> None:
    """Write one step's frames.

    Args:
        frames: Camera name -> PNG bytes.
        root: Output directory.
        index: Step index, used in the file name.
    """
    for name, data in frames.items():
        target = root / name
        target.mkdir(parents=True, exist_ok=True)
        (target / f"{index:04d}.png").write_bytes(data)


def _encode_videos(root: Path, fps: int) -> dict[str, str]:
    """Assemble one video per camera, when imageio can.

    Args:
        root: Directory holding the per-camera PNG folders.
        fps: Output frame rate.

    Returns:
        Camera name -> written file path. Empty when no encoder is available;
        the PNGs are the deliverable either way, so a missing ffmpeg is a
        downgrade rather than a failure.
    """
    try:
        import imageio.v2 as iio
    except ImportError:
        print("[viz] imageio unavailable; PNG frames only", flush=True)
        return {}
    written: dict[str, str] = {}
    for name in CAMERAS:
        folder = root / name
        paths = sorted(folder.glob("*.png"))
        if not paths:
            continue
        images = [iio.imread(path) for path in paths]
        for suffix, kwargs in ((".mp4", {"fps": fps}), (".gif", {"duration": 1 / fps})):
            target = root / f"{name}{suffix}"
            try:
                iio.mimsave(target, images, **kwargs)
            except Exception as exc:  # noqa: BLE001 - encoder availability varies
                print(f"[viz] {name}{suffix} failed: {type(exc).__name__}", flush=True)
                continue
            written[name] = str(target)
            break
    return written


def _env_config(args: argparse.Namespace, *, execute_horizon: int) -> dict[str, Any]:
    """Build the family config for a capture run.

    Args:
        args: Parsed CLI arguments.
        execute_horizon: Actions submitted per ``chunk_step``.

    Returns:
        The ``env_config`` mapping.
    """
    return {
        "task_name": args.task,
        "assets_path": args.assets_path,
        "embodiment": list(args.embodiment),
        "planner_backend": args.planner_backend,
        "max_episode_steps": int(args.max_steps),
        "step_lim": int(args.max_steps),
        "execute_horizon": execute_horizon,
        "collect_head_camera": True,
        "collect_wrist_camera": True,
        "center_crop": bool(args.center_crop),
    }


def capture_scenes(args: argparse.Namespace) -> dict[str, Any]:
    """Dump the opening frame of several seeds, without loading a policy.

    Args:
        args: Parsed CLI arguments.

    Returns:
        A summary dict.
    """
    core = RobotwinEnvCore()
    core.build(
        EnvSpecMsg(
            env_family="robotwin", env_config=_env_config(args, execute_horizon=1)
        ),
        num_envs=1,
    )
    root = Path(args.output_dir)
    captured: list[int] = []
    try:
        for index, seed in enumerate(args.seeds):
            observation = core.reset([0], ResetSpec(reset_state_id=int(seed)))[0]
            frames = _frames(observation)
            _write(frames, root / f"seed-{seed}", 0)
            captured.append(int(seed))
            print(
                f"[viz] seed={seed} views={sorted(frames)} "
                f"instruction={observation.instruction!r}",
                flush=True,
            )
    finally:
        core.close()
    return {"mode": "scene_only", "seeds": captured, "output_dir": str(root)}


def capture_episode(args: argparse.Namespace) -> dict[str, Any]:
    """Run one episode under the policy and dump every simulator step.

    Args:
        args: Parsed CLI arguments.

    Returns:
        A summary dict.

    Raises:
        RuntimeError: The policy returned an error for a step.
    """
    from rollout_runtime.backends.rlinf_policy import RlinfPolicyConfig, RlinfPolicyCore

    core = RobotwinEnvCore()
    # execute_horizon=1: one observation per submitted action, so the capture is
    # per simulator step rather than per chunk. The policy is still only re-run
    # every --replan-every steps, so the control behaviour matches the preset.
    core.build(
        EnvSpecMsg(
            env_family="robotwin", env_config=_env_config(args, execute_horizon=1)
        ),
        num_envs=1,
    )
    policy = RlinfPolicyCore(
        RlinfPolicyConfig.from_mapping(json.loads(Path(args.policy_config).read_text()))
    )
    policy.load()

    root = Path(args.output_dir) / f"seed-{args.seeds[0]}"
    observation = core.reset([0], ResetSpec(reset_state_id=int(args.seeds[0])))[0]
    _write(_frames(observation), root, 0)

    buffer: list[list[float]] = []
    success = False
    step_index = 0
    try:
        while step_index < int(args.max_steps):
            if not buffer:
                response = policy.infer_batch(
                    [
                        InferenceRequest(
                            request_id=RequestId(f"viz-{step_index}"),
                            session_id=SessionId("viz"),
                            episode_id=EpisodeId(0),
                            policy_id=args.policy_id,
                            observation=observation,
                            routing_token="viz",
                            compat_key="viz",
                        )
                    ]
                )[0]
                if response.error is not None:
                    raise RuntimeError(f"policy failed: {response.error}")
                from rollout_runtime.core.payload import decode_array

                chunk = np.asarray(decode_array(response.actions), dtype=np.float32)
                buffer = [list(row) for row in chunk[: int(args.replan_every)]]

            action = np.asarray([buffer.pop(0)], dtype=np.float32)
            outcome = core.chunk_step([0], [action])[0]
            step_index += 1
            observation = outcome.observation
            if observation is not None:
                _write(_frames(observation), root, step_index)
            if outcome.info.get("success"):
                success = True
            if outcome.terminated or outcome.truncated:
                break
            if step_index % 25 == 0:
                print(f"[viz] step {step_index} success={success}", flush=True)
    finally:
        core.close()

    videos = _encode_videos(root, fps=args.fps)
    print(f"[viz] captured {step_index} steps, success={success}", flush=True)
    return {
        "mode": "episode",
        "seed": int(args.seeds[0]),
        "steps": step_index,
        "success": success,
        "output_dir": str(root),
        "videos": videos,
    }


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser.

    Returns:
        The configured parser.
    """
    parser = argparse.ArgumentParser(
        description="Dump RoboTwin camera frames for one task"
    )
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
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--seeds", nargs="+", type=int, required=True, help="RoboTwin scene seeds"
    )
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--replan-every", type=int, default=25)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--center-crop", action="store_true")
    parser.add_argument("--policy-id", default="pi05_aloha_robotwin")
    parser.add_argument(
        "--policy-config",
        default=None,
        help="JSON file of RlinfPolicyConfig fields; required unless --scene-only",
    )
    parser.add_argument(
        "--scene-only",
        action="store_true",
        help="Only dump the opening frame of each seed; no checkpoint needed",
    )
    return parser


def main() -> int:
    """CLI entrypoint.

    Returns:
        ``0`` on success.
    """
    args = build_parser().parse_args()
    if args.scene_only:
        summary = capture_scenes(args)
    else:
        if not args.policy_config:
            raise SystemExit("--policy-config is required unless --scene-only")
        summary = capture_episode(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
