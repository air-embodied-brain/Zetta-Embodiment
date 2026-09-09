# Copyright (c) 2026 Zetta Contributors
"""Exercise the native Genie environment through the Runtime client on one GPU."""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import time
import traceback
import uuid
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

from rollout_runtime.api.messages import CreateSessionRequest, EnvSpecMsg, ResetSpec
from rollout_runtime.api.result import unwrap
from rollout_runtime.core.payload import decode_payload, encode_array
from rollout_runtime.launch.local import build_local_components
from zetta.envs.geniesim.config import CAMERA_NAMES, GenieSimConfig


def capture(observation, directory: Path, label: str) -> dict:
    images = [
        observation.main_image,
        observation.wrist_image,
        *observation.extra_view_images,
    ]
    if len(images) != 3 or len(observation.state) != 21:
        raise RuntimeError("expected three cameras and 21 state values")
    statistics = {}
    for name, reference in zip(CAMERA_NAMES, images, strict=True):
        pixels = decode_payload(reference)
        if (
            pixels.dtype != np.uint8
            or pixels.ndim != 3
            or pixels.shape[2] != 3
            or pixels.std() < 2
        ):
            raise RuntimeError(f"invalid or blank {name} image")
        imageio.imwrite(directory / f"{label}_{name}.png", pixels)
        statistics[name] = {"shape": list(pixels.shape), "std": float(pixels.std())}
    return {
        "state": list(observation.state),
        "images": statistics,
        "instruction": observation.instruction,
    }


async def run_group(
    config: GenieSimConfig, seeds: list[int], directory: Path, transport: str
) -> list[dict]:
    runtime = build_local_components(
        {
            "env_family": "geniesim",
            "env_config": config.to_dict(),
            "transport": {"kind": transport, "command_timeout_seconds": 1800.0},
            "env_worker": {"placement_strategy": "packed", "max_sessions_per_rank": 1},
            "gateway": {
                "default_lease_seconds": 7200.0,
                "heartbeat_timeout_seconds": 1800.0,
            },
        }
    )
    reports = []
    async with runtime:
        client = runtime.gateway
        handle = unwrap(
            (
                await client.create_sessions(
                    [
                        CreateSessionRequest(
                            application_id="geniesim-smoke",
                            client_session_key=uuid.uuid4().hex,
                            env_spec=EnvSpecMsg(
                                env_family="geniesim", env_config=config.to_dict()
                            ),
                            lease_seconds=7200.0,
                        )
                    ]
                )
            )[0]
        )
        ids = [handle.session_id]
        for index, seed in enumerate(seeds):
            output = directory / f"episode-{index:02d}-seed-{seed}"
            output.mkdir(parents=True)
            print(f"GENIESIM_RUNTIME_RESET seed={seed}", flush=True)
            initial = unwrap(
                (await client.reset(ids, ResetSpec(seed=seed)))[0]
            ).observation
            if initial.step_index != 0:
                raise RuntimeError("reset did not clear the control step counter")
            initial_stats = capture(initial, output, "initial")
            cached = unwrap((await client.observe(ids))[0])
            if cached.step_index != 0 or cached.state != initial.state:
                raise RuntimeError("observe changed the cached control observation")
            hold = np.asarray(initial.state[:16], dtype=np.float32)
            frames = [decode_payload(initial.main_image)]
            trajectory = []
            movement = 0.0
            final_step = None
            # A larger final chunk verifies that the time limit stops execution
            # after the first remaining control step, without padding its record.
            for step in range(config.max_steps):
                target = hold.copy()
                if 8 <= step < 24:
                    target[0] += 0.01
                horizon = 4 if step == config.max_steps - 1 else 1
                action = encode_array(np.tile(target, (horizon, 1)))
                final_step = unwrap((await client.action_step(ids, [action]))[0])
                if final_step.error is not None:
                    raise RuntimeError(str(final_step.error))
                if (
                    final_step.executed_horizon != 1
                    or len(final_step.per_step or []) != 1
                ):
                    raise RuntimeError(
                        "Runtime misreported the actually executed horizon"
                    )
                observation = final_step.observation
                frames.append(decode_payload(observation.main_image))
                trajectory.extend(
                    dataclasses.asdict(record) for record in final_step.per_step
                )
                if 8 <= step < 24:
                    movement = max(movement, float(observation.state[0] - hold[0]))
                if final_step.terminated or final_step.truncated:
                    break
            final = capture(final_step.observation, output, "final")
            result = unwrap(
                (await client.extension_call(ids, "geniesim", "result", {}))[0]
            )
            if result["control_steps"] != len(trajectory):
                raise RuntimeError(
                    "official control step count disagrees with Runtime records"
                )
            if not (final_step.terminated or final_step.truncated):
                raise RuntimeError(
                    "episode did not finish at its configured step limit"
                )
            if result["evaluation_step"] < 30:
                raise RuntimeError("native official evaluation was never updated")
            if movement < 0.002:
                raise RuntimeError(
                    "small absolute joint pulse produced no measurable response"
                )
            with imageio.get_writer(
                output / "head.mp4", fps=10, macro_block_size=1
            ) as video:
                for frame in frames:
                    video.append_data(frame)
            (output / "trajectory.json").write_text(
                json.dumps(trajectory, indent=2) + "\n"
            )
            repeated = unwrap(
                (await client.reset(ids, ResetSpec(seed=seed)))[0]
            ).observation
            repeat_stats = capture(repeated, output, "reset")
            delta = np.abs(np.asarray(repeated.state) - np.asarray(initial.state))
            if delta.max() > 0.02:
                raise RuntimeError(f"same-seed reset state drift: {delta.tolist()}")
            report = {
                "seed": seed,
                "initial": initial_stats,
                "final": final,
                "repeated_reset": repeat_stats,
                "reset_max_joint_delta": float(delta.max()),
                "joint_response_radians": movement,
                "result": result,
                "video": str(output / "head.mp4"),
            }
            reports.append(report)
            (output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
            print(
                f"GENIESIM_RUNTIME_EPISODE_DONE seed={seed} steps={result['control_steps']} success={result['success']}",
                flush=True,
            )
        unwrap((await client.close_sessions(ids))[0])
    for report in reports:
        process = json.loads(
            (Path(report["result"]["artifact_directory"]) / "process.json").read_text()
        )
        if (
            process.get("status") != "closed"
            or process.get("returncode") != 0
            or process.get("forced") is not False
        ):
            raise RuntimeError(f"simulator did not close cleanly: {process}")
        report["process"] = process
    return reports


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sim-python", required=True)
    parser.add_argument("--geniesim-root", required=True)
    parser.add_argument("--assets-root", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--library-path", action="append", default=[])
    parser.add_argument("--seeds", type=int, nargs="+", default=[1001, 1002, 1003])
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument(
        "--transport", choices=["inproc", "ray_channel"], default="inproc"
    )
    parser.add_argument("--skip-recreation", action="store_true")
    args = parser.parse_args()
    if args.steps < 30:
        parser.error("--steps must reach at least one official evaluation tick (30)")
    output = args.output_root.resolve() / ("run-" + uuid.uuid4().hex)
    output.mkdir(parents=True)
    config = GenieSimConfig(
        python_executable=args.sim_python,
        geniesim_root=args.geniesim_root,
        assets_root=args.assets_root,
        output_root=str(output / "simulator"),
        gpu_id=args.gpu_id,
        max_steps=args.steps,
        library_paths=args.library_path,
    )
    report = {
        "status": "running",
        "transport": args.transport,
        "config": config.to_dict(),
        "controller": "observed_joint_hold_and_0.01_rad_pulse",
        "output": str(output),
    }
    report_file = output / "result.json"
    report_file.write_text(json.dumps(report, indent=2) + "\n")
    started = time.monotonic()
    print(f"GENIESIM_RUNTIME_OUTPUT {output}", flush=True)
    try:
        report["episodes"] = asyncio.run(
            run_group(config, args.seeds, output / "initial", args.transport)
        )
        if not args.skip_recreation:
            report["recreated"] = asyncio.run(
                run_group(config, [args.seeds[0]], output / "recreated", args.transport)
            )
            first = np.asarray(report["episodes"][0]["initial"]["state"])
            again = np.asarray(report["recreated"][0]["initial"]["state"])
            if np.max(np.abs(first - again)) > 0.02:
                raise RuntimeError(
                    "recreated same-seed joint state drift exceeded tolerance"
                )
        report["status"] = "passed"
    except BaseException:
        report.update(status="failed", error=traceback.format_exc())
        raise
    finally:
        report["elapsed_seconds"] = time.monotonic() - started
        report_file.write_text(json.dumps(report, indent=2) + "\n")
    print(f"GENIESIM_RUNTIME_PASSED {report_file}", flush=True)


if __name__ == "__main__":
    main()
