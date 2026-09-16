# Copyright (c) 2026 Zetta Contributors
"""Exercise real Panda reset, RGB, official scoring, control and shutdown.

The scripted controller reads simulator poses only to validate the environment.
Its score is not a learned-policy benchmark.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--environment", type=Path)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--gpu", default="0")
    args = parser.parse_args()
    if args.environment:
        exports = json.loads(args.environment.read_text())
        restart = exports.get("LD_LIBRARY_PATH") != os.environ.get("LD_LIBRARY_PATH")
        os.environ.update(exports)
        if restart:
            # The ELF loader reads LD_LIBRARY_PATH at process startup.
            os.execv(sys.executable, [sys.executable, *sys.argv])
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

    import importlib.metadata

    import imageio.v2 as iio
    import numpy as np
    import torch

    from rollout_runtime.api.messages import EnvSpecMsg, ResetSpec
    from rollout_runtime.backends.rlinf_maniskill import (
        ManiskillEnvConfig,
        ManiskillEnvCore,
    )
    from rollout_runtime.core.payload import decode_payload

    if args.episodes < 2:
        parser.error("at least two episodes are required for reset reproducibility")
    args.output.mkdir(parents=True, exist_ok=False)
    config = ManiskillEnvConfig(
        robot_uids="panda",
        return_all_frames=True,
        goal_visibility="visible",
        extra_init_params={"enhanced_determinism": True, "reconfiguration_freq": 1},
    )
    core = ManiskillEnvCore()
    records = []
    reset_digests = []
    replay_actions = []
    replay_expected = []
    memory_samples = []
    runtime_metadata = None

    def memory_sample(episode):
        status = Path("/proc/self/status").read_text().splitlines()
        rss = next(int(line.split()[1]) for line in status if line.startswith("VmRSS:"))
        query = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_memory",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
        own = [
            line.split(",")
            for line in query.splitlines()
            if line.split(",")[0].strip() == str(os.getpid())
        ]
        return {
            "episode": episode,
            "rss_kib": rss,
            "gpu_mib": sum(int(row[1]) for row in own),
        }

    def numpy(value):
        return value.detach().cpu().numpy()

    def state_values(value):
        if isinstance(value, dict):
            return {k: state_values(v) for k, v in value.items()}
        if hasattr(value, "detach"):
            return numpy(value).tolist()
        if isinstance(value, np.ndarray):
            return value.tolist()
        return value

    try:
        core.build(
            EnvSpecMsg(env_family="maniskill", env_config=dataclasses.asdict(config)),
            num_envs=1,
        )
        env = core._envs[0].env.unwrapped
        action_space = env.single_action_space
        if action_space.shape != (7,):
            raise ValueError(f"unexpected action space: {action_space}")
        for episode in range(args.episodes):
            obs = core.reset([0], ResetSpec(seed=args.seed))[0]
            # The first policy frame must already include the configured goal.
            visible_rgb = env.get_obs()["sensor_data"]["base_camera"]["rgb"][0]
            np.testing.assert_array_equal(
                decode_payload(obs.main_image), numpy(visible_rgb)
            )
            if runtime_metadata is None:
                runtime_metadata = dict(obs.extras)
            reset_state = state_values(env.get_state_dict())
            digest = hashlib.sha256(
                json.dumps(reset_state, sort_keys=True).encode()
            ).hexdigest()
            reset_digests.append(digest)
            if episode == 0:
                (args.output / "initial-state.json").write_text(json.dumps(reset_state))
            script = episode % 2 == 0
            writer = (
                iio.get_writer(
                    str(args.output / f"episode-{episode}.mp4"), fps=env.control_freq
                )
                if episode < 2
                else None
            )
            try:
                if writer:
                    writer.append_data(decode_payload(obs.main_image))
                while True:
                    index = obs.step_index
                    action = np.zeros(7, dtype=np.float32)
                    if script and episode > 0:
                        action = replay_actions[index].copy()
                    elif script:
                        cube = numpy(env.cube.pose.p)[0]
                        tcp = numpy(env.agent.tcp.pose.p)[0]
                        goal = numpy(env.goal_site.pose.p)[0]
                        if index < 8:
                            target, gripper = cube + np.array([0, 0, 0.12]), 1.0
                        elif index < 16:
                            target, gripper = cube, 1.0
                        elif index < 22:
                            target, gripper = cube, -1.0
                        else:
                            target, gripper = goal, -1.0
                        action[:3] = np.clip((target - tcp) / 0.1, -1, 1)
                        action[6] = gripper
                    outcome = core.chunk_step([0], [action[None]])[0]
                    obs = outcome.observation
                    row = {
                        "episode": episode,
                        "step_index": obs.step_index,
                        "controller": "privileged_scripted_smoke"
                        if script
                        else "zero_action",
                        "action": action.tolist(),
                        "state": list(obs.state),
                        "reward": outcome.reward,
                        "terminated": outcome.terminated,
                        "truncated": outcome.truncated,
                        **outcome.info,
                    }
                    with (args.output / "steps.jsonl").open("a") as stream:
                        stream.write(json.dumps(row, allow_nan=False) + "\n")
                    if script and episode == 0:
                        replay_actions.append(action.copy())
                        replay_expected.append(
                            (
                                outcome.reward,
                                row["success"],
                                outcome.terminated,
                                outcome.truncated,
                                obs.state,
                            )
                        )
                    elif script:
                        expected = replay_expected[index]
                        np.testing.assert_allclose(
                            outcome.reward, expected[0], atol=1e-5, rtol=1e-5
                        )
                        assert (
                            row["success"],
                            outcome.terminated,
                            outcome.truncated,
                        ) == expected[1:4]
                        np.testing.assert_allclose(
                            obs.state, expected[4], atol=1e-5, rtol=1e-5
                        )
                    if writer:
                        frame = decode_payload(obs.main_image)
                        assert frame.shape == (128, 128, 3) and frame.dtype == np.uint8
                        writer.append_data(frame)
                    if outcome.terminated or outcome.truncated:
                        records.append(row)
                        break
            finally:
                if writer:
                    writer.close()
            if episode in {0, 1, args.episodes - 1} or (episode + 1) % 10 == 0:
                memory_samples.append(memory_sample(episode))
        if len(set(reset_digests)) != 1:
            raise ValueError("same-seed reset state changed between episodes")
    finally:
        core.close()

    import gymnasium as gym
    from omegaconf import OmegaConf

    raw = gym.make(
        **OmegaConf.to_container(
            config.to_rlinf_cfg(num_envs=1).init_params, resolve=True
        )
    )
    try:
        raw.reset(seed=[args.seed], options={})
        if raw.unwrapped.goal_site in raw.unwrapped._hidden_objects:
            raw.unwrapped._hidden_objects.remove(raw.unwrapped.goal_site)
        raw.unwrapped.goal_site.show_visual()
        raw_digest = hashlib.sha256(
            json.dumps(
                state_values(raw.unwrapped.get_state_dict()), sort_keys=True
            ).encode()
        ).hexdigest()
        if raw_digest != reset_digests[0]:
            raise ValueError("raw/Runtime simulator initial states differ")
        for action, expected in zip(replay_actions, replay_expected, strict=True):
            _, reward, terminated, truncated, info = raw.step(
                torch.tensor(action[None], device=raw.unwrapped.device)
            )
            actual = (
                float(reward.item()),
                bool(info["success"].item()),
                bool(terminated.item()),
                bool(truncated.item()),
            )
            np.testing.assert_allclose(actual[0], expected[0], atol=1e-5, rtol=1e-5)
            if actual[1:] != expected[1:4]:
                raise ValueError(f"raw/Runtime scoring differs: {actual} != {expected}")
            robot = raw.unwrapped.agent.robot
            native_state = torch.cat((robot.get_qpos(), robot.get_qvel()), dim=-1)[0]
            np.testing.assert_allclose(
                numpy(native_state), expected[4], atol=1e-5, rtol=1e-5
            )
    finally:
        raw.close()
    report = {
        "schema_version": 1,
        "env_id": config.env_id,
        "config": dataclasses.asdict(config),
        "seed": args.seed,
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("mani-skill", "sapien", "torch", "gymnasium")
        },
        "gpu": torch.cuda.get_device_name(),
        "episodes": len(records),
        "reset_state_sha256": reset_digests[0],
        "same_seed_reset": True,
        "replayed_success_episodes": sum(i % 2 == 0 for i in range(2, args.episodes)),
        "raw_runtime_parity": True,
        "scripted_success": records[0]["success"],
        "zero_action_success": records[1]["success"],
        "results": records,
        "memory_samples": memory_samples,
        "runtime_metadata": runtime_metadata,
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {k: v for k, v in report.items() if k not in {"results", "config"}},
            indent=2,
        )
    )
    if not report["scripted_success"] or report["zero_action_success"]:
        raise RuntimeError(
            "expected official success for scripted control and failure for zero control"
        )


if __name__ == "__main__":
    main()
