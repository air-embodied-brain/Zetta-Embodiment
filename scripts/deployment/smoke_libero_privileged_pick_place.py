#!/usr/bin/env python3
# Copyright (c) 2026 Zetta Contributors
"""Run one audited LIBERO semantic pick-place primitive without an LLM call."""

from __future__ import annotations

import argparse
import base64
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from robots.libero.env_client import LiberoEnvClient  # noqa: E402
from robots.libero.tools import LiberoPrimitives  # noqa: E402
from zetta.utils.daemon import pick_free_port  # noqa: E402
from zetta.utils.http_rpc import HttpRpcClient  # noqa: E402
from zetta.utils.logging import init_output_dir  # noqa: E402
from zetta.utils.sam3_client import UnavailableSam3Client  # noqa: E402
from zetta.utils.vla_client import VLAClient  # noqa: E402


def _wait_for_env_server(
    process: subprocess.Popen,
    rpc: HttpRpcClient,
    *,
    log_path: Path,
    timeout_s: float = 300.0,
) -> None:
    deadline = time.monotonic() + timeout_s
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        returncode = process.poll()
        if returncode is not None:
            raise RuntimeError(
                f"LIBERO env server exited with code {returncode}; see {log_path}"
            )
        try:
            rpc.call("healthz", timeout_s=1.0)
            return
        except Exception as exc:
            last_error = exc
            time.sleep(0.5)
    raise TimeoutError(
        f"LIBERO env server was not ready after {timeout_s:.0f}s: {last_error}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--suite", default="libero_goal_task")
    parser.add_argument("--task", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--cuda-device", type=int, required=True)
    parser.add_argument(
        "--grasp-offset-xyz",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        required=True,
    )
    parser.add_argument("--joint-entity")
    parser.add_argument("--joint")
    parser.add_argument("--joint-direction", choices=("lower", "upper"))
    parser.add_argument("--joint-max-sweep-steps", type=int, default=64)
    parser.add_argument("--joint-sweep-step-m", type=float, default=0.015)
    parser.add_argument("--joint-close-steps", type=int, default=3)
    parser.add_argument("--pre-vla-prompt")
    parser.add_argument("--pre-vla-prompt-b64")
    parser.add_argument("--pre-vla-max-chunks", type=int, default=12)
    parser.add_argument("--pre-vla-actions-per-chunk", type=int, default=5)
    parser.add_argument("--pre-vla-action-clip", type=float, default=1.0)
    parser.add_argument("--skip-pick-place", action="store_true")
    parser.add_argument("--joint-plan-only", action="store_true")
    parser.add_argument("--max-episode-steps", type=int, default=310)
    parser.add_argument("--retreat-height", type=float, default=0.025)
    parser.add_argument("--approach-height", type=float, default=0.055)
    parser.add_argument("--close-steps", type=int, default=24)
    parser.add_argument("--grasp-confirm-steps", type=int, default=4)
    parser.add_argument("--lift-height", type=float, default=0.13)
    parser.add_argument("--target-height", type=float, default=0.035)
    parser.add_argument("--carry-height", type=float, default=0.10)
    parser.add_argument("--max-steps-per-move", type=int, default=48)
    parser.add_argument("--vla-endpoint", default="http://127.0.0.1:18810")
    args = parser.parse_args()
    if args.pre_vla_prompt and args.pre_vla_prompt_b64:
        raise ValueError("provide only one of pre-vla-prompt or pre-vla-prompt-b64")
    pre_vla_prompt = args.pre_vla_prompt
    if args.pre_vla_prompt_b64:
        pre_vla_prompt = base64.b64decode(args.pre_vla_prompt_b64).decode("utf-8")

    output = init_output_dir(args.output)
    port = pick_free_port()
    environment = dict(os.environ)
    inherited_pythonpath = environment.get("PYTHONPATH", "")
    environment.update(
        {
            "MUJOCO_GL": "egl",
            "ROBOT_PLATFORM": "LIBERO",
            "LIBERO_TYPE": "pro",
            "PYTHONPATH": os.pathsep.join(
                part for part in (str(REPO_ROOT), inherited_pythonpath) if part
            ),
        }
    )
    log_path = output / "env_server.log"
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            [
                sys.executable,
                "robots/libero/env_server.py",
                "--transport",
                "http",
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
                "--suite",
                args.suite,
                "--task",
                str(args.task),
                "--seed",
                str(args.seed),
                "--max-episode-steps",
                str(args.max_episode_steps),
                "--cuda-device",
                str(args.cuda_device),
                "--parent-watch",
            ],
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
        try:
            rpc = HttpRpcClient(f"http://127.0.0.1:{port}")
            _wait_for_env_server(process, rpc, log_path=log_path)
            env = LiberoEnvClient(
                rpc,
                expected_meta={
                    "suite": args.suite,
                    "task": args.task,
                    "seed": args.seed,
                    "max_episode_steps": args.max_episode_steps,
                },
            )
            primitives = LiberoPrimitives(
                env,
                VLAClient(HttpRpcClient(args.vla_endpoint)),
                UnavailableSam3Client("privileged smoke test"),
                allow_privileged_actions=True,
            )
            obs, _ = primitives.reset()
            primitives.start_recording()
            primitives.record_frame(obs)
            before = env.privileged_critic_state()
            pre_vla_result = None
            if pre_vla_prompt:
                pre_vla_result = primitives.vla_execute(
                    prompt=pre_vla_prompt,
                    max_chunks=args.pre_vla_max_chunks,
                    actions_per_chunk=args.pre_vla_actions_per_chunk,
                    translation_scale=1.0,
                    rotation_scale=1.0,
                    gripper_scale=1.0,
                    action_clip=args.pre_vla_action_clip,
                    stop_on_success=False,
                    stop_on_truncation=True,
                )
            state_after_pre_vla = env.privileged_critic_state()
            joint_result = None
            joint_plan_before = None
            joint_args = (args.joint_entity, args.joint, args.joint_direction)
            if any(value is not None for value in joint_args):
                if not all(value is not None for value in joint_args):
                    raise ValueError(
                        "joint-entity, joint, and joint-direction must be provided together"
                    )
                joint_plan_before = env.privileged_semantic_joint_plan(
                    entity=args.joint_entity,
                    joint=args.joint,
                    direction=args.joint_direction,
                )
                if not args.joint_plan_only:
                    joint_result = primitives.semantic_joint_interact(
                        entity=args.joint_entity,
                        joint=args.joint,
                        direction=args.joint_direction,
                        max_sweep_steps=args.joint_max_sweep_steps,
                        sweep_step_m=args.joint_sweep_step_m,
                        close_steps=args.joint_close_steps,
                    )
            if args.skip_pick_place:
                result = {"name": "privileged_pick_place", "status": "skipped"}
            else:
                result = primitives.privileged_pick_place(
                    grasp_offset_xyz=args.grasp_offset_xyz,
                    retreat_height=args.retreat_height,
                    approach_height=args.approach_height,
                    close_steps=args.close_steps,
                    grasp_confirm_steps=args.grasp_confirm_steps,
                    lift_height=args.lift_height,
                    target_height=args.target_height,
                    carry_height=args.carry_height,
                    max_steps_per_move=args.max_steps_per_move,
                )
            video = primitives.stop_recording_and_save(
                str(output / "privileged_pick_place.mp4")
            )
            after = env.privileged_critic_state()
            joint_plan_after = None
            if all(value is not None for value in joint_args):
                joint_plan_after = env.privileged_semantic_joint_plan(
                    entity=args.joint_entity,
                    joint=args.joint,
                    direction=args.joint_direction,
                )
            summary = {
                "suite": args.suite,
                "task": args.task,
                "seed": args.seed,
                "cuda_device": args.cuda_device,
                "grasp_offset_xyz": args.grasp_offset_xyz,
                "joint": joint_result,
                "joint_plan_before": joint_plan_before,
                "joint_plan_after": joint_plan_after,
                "pre_vla": pre_vla_result,
                "top_drawer_qpos_after_pre_vla": state_after_pre_vla.get(
                    "privileged.joint.wooden_cabinet_1_top_level.position"
                ),
                "retreat_height": args.retreat_height,
                "approach_height": args.approach_height,
                "close_steps": args.close_steps,
                "grasp_confirm_steps": args.grasp_confirm_steps,
                "lift_height": args.lift_height,
                "target_height": args.target_height,
                "carry_height": args.carry_height,
                "max_steps_per_move": args.max_steps_per_move,
                "manipulated_object": before.get(
                    "privileged.task.manipulated_object.name"
                ),
                "target": before.get("privileged.task.target.name"),
                "result": result,
                "official_success": bool(after.get("privileged.task.success", False)),
                "final_stage": after.get("privileged.task.stage.name"),
                "final_distance_to_target_m": after.get(
                    "privileged.task.manipulated_object.distance_to_target_m"
                ),
                "final_distance_to_eef_m": after.get(
                    "privileged.task.manipulated_object.distance_to_eef_m"
                ),
                "ever_grasped": bool(
                    after.get("privileged.task.manipulated_object.ever_grasped", False)
                ),
                "retained": bool(
                    after.get("privileged.task.manipulated_object.retained", False)
                ),
                "episode_terminated": env.episode_terminated,
                "episode_truncated": env.episode_truncated,
                "video": video,
            }
            result_path = output / "result.json"
            result_path.write_text(
                json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
            )
            print(json.dumps(summary, sort_keys=True))
        finally:
            process.terminate()
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)


if __name__ == "__main__":
    main()
