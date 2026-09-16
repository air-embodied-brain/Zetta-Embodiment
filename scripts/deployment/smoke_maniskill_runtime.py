# Copyright (c) 2026 Zetta Contributors
"""Replay native smoke actions through real Ray workers and the Runtime Gateway."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path


async def exercise(args):
    import numpy as np
    import ray

    from rollout_runtime.api.enums import ErrorCode
    from rollout_runtime.api.messages import (
        CreateSessionRequest,
        EnvSpecMsg,
        PolicyRequest,
        ResetSpec,
    )
    from rollout_runtime.api.result import Err, unwrap
    from rollout_runtime.config.schema import load_config
    from rollout_runtime.core.payload import decode_payload, encode_array
    from rollout_runtime.launch.ray_launch import build_ray_components
    from zetta.envs.maniskill.contracts import contract_digest

    config = load_config("maniskill_smoke")
    rows = [
        json.loads(line)
        for line in (args.native_output / "steps.jsonl").read_text().splitlines()
    ]
    replay = [row for row in rows if row["episode"] == 0]
    native = json.loads((args.native_output / "report.json").read_text())
    native_state = json.loads((args.native_output / "initial-state.json").read_text())

    def lane_zero(value):
        if isinstance(value, dict):
            return {key: lane_zero(item) for key, item in value.items()}
        return value[0]

    expected_initial = contract_digest(lane_zero(native_state))

    def record(name, value):
        with (args.output / name).open("a") as stream:
            stream.write(json.dumps(value, allow_nan=False) + "\n")

    config.env_config = native["config"]
    config.env_worker.group_name += f"-{os.getpid()}"
    config.rollout_worker.group_name += f"-{os.getpid()}"

    async def create_session(target, key):
        created = await target.gateway.create_sessions(
            [
                CreateSessionRequest(
                    application_id="maniskill-smoke",
                    client_session_key=key,
                    env_spec=EnvSpecMsg(
                        env_family="maniskill",
                        env_config=config.env_config,
                        pool_size=1,
                    ),
                    default_policy_id="fake",
                    lease_seconds=1800,
                )
            ]
        )
        return [unwrap(created[0]).session_id]

    async def assert_exited(pid):
        import psutil

        for _ in range(100):
            if not psutil.pid_exists(pid):
                return
            await asyncio.sleep(0.1)
        raise AssertionError(f"worker {pid} remained alive after shutdown")

    runtime = build_ray_components(config)
    ids = []
    results = []
    deviations = []
    try:
        await runtime.start()
        for index in range(2):
            ids = await create_session(runtime, f"episode-{index}")
            reset = unwrap(
                (await runtime.gateway.reset(ids, ResetSpec(seed=native["seed"])))[0]
            )
            observation = reset.observation
            frame = decode_payload(observation.main_image)
            assert frame.shape == (128, 128, 3) and frame.dtype == np.uint8
            initial = observation.extras["initial_state_sha256"]
            record(
                "resets.jsonl",
                {
                    "episode": index,
                    "state": list(observation.state),
                    "expected_initial_state_sha256": expected_initial,
                    "runtime_metadata": observation.extras,
                },
            )
            assert initial == expected_initial, (
                "native/Ray simulator initial states differ"
            )
            assert observation.extras["worker_pid"] != os.getpid()
            assert observation.extras["simulation_device"].startswith("cuda")
            for name in (
                "simulation_pci_id",
                "render_pci_id",
                "control_frequency",
                "simulation_frequency",
            ):
                assert observation.extras[name] == native["runtime_metadata"][name]
            if results:
                assert results[0]["initial_state_sha256"] == initial
            if index == 1:
                # Exercise the separate policy process without using its output as benchmark evidence.
                infer = unwrap(
                    (
                        await runtime.gateway.policy_infer(
                            ids, PolicyRequest(policy_id="fake")
                        )
                    )[0]
                )
                assert infer.actions is not None
                assert infer.observation_step_index == 0
            for expected in replay:
                action = encode_array(np.array([expected["action"]], dtype=np.float32))
                step = unwrap((await runtime.gateway.action_step(ids, [action]))[0])
                actual = {
                    "episode": index,
                    "step_index": step.observation.step_index,
                    "action": expected["action"],
                    "state": list(step.observation.state),
                    "reward": step.reward,
                    "expected_reward": expected["reward"],
                    "reward_absolute_error": abs(step.reward - expected["reward"]),
                    "success": step.info["success"],
                    "terminated": step.terminated,
                    "truncated": step.truncated,
                }
                if "state" in expected:
                    actual["state_max_absolute_error"] = float(
                        np.max(
                            np.abs(
                                np.asarray(step.observation.state) - expected["state"]
                            )
                        )
                    )
                record("steps.jsonl", actual)
                assert step.executed_horizon == 1
                assert step.observation.step_index == expected["step_index"]
                assert step.info["success"] == expected["success"]
                assert step.terminated == expected["terminated"]
                assert step.truncated == expected["truncated"]
                for field, value in (
                    ("reward", step.reward),
                    ("state", step.observation.state),
                ):
                    if field in expected and not np.allclose(
                        value, expected[field], atol=1e-5, rtol=1e-5
                    ):
                        deviations.append(
                            {
                                "episode": index,
                                "step": expected["step_index"],
                                "field": field,
                            }
                        )
            results.append(
                {
                    "initial_state_sha256": initial,
                    "success": step.info["success"],
                    "steps": step.observation.step_index,
                    "worker_pid": observation.extras["worker_pid"],
                    "cuda_visible_devices": observation.extras["cuda_visible_devices"],
                    "runtime_metadata": observation.extras,
                }
            )
            if not (args.fault_test and index == 1):
                unwrap((await runtime.gateway.close_sessions(ids))[0])
                ids = []
        report = {
            "ray_ranks": runtime.observed_ranks,
            "episodes": results,
            "policy_counters": await runtime.rollout_counters(),
            "native_gateway_parity": not deviations,
            "numeric_tolerance": {"atol": 1e-5, "rtol": 1e-5},
            "deviations": deviations,
            "inference_kind": "fake infrastructure smoke only",
        }
        if args.fault_test:
            unwrap(
                (await runtime.gateway.reset(ids, ResetSpec(seed=native["seed"])))[0]
            )
            # The actor belongs to this isolated cluster and this smoke invocation.
            ray.kill(runtime.env_group.actors[0], no_restart=True)
            await assert_exited(results[-1]["worker_pid"])
            runtime.transport._timeout = 2.0
            failed = (
                await runtime.gateway.action_step(
                    ids,
                    [encode_array(np.asarray([replay[0]["action"]], dtype=np.float32))],
                )
            )[0]
            assert isinstance(failed, Err), (
                "worker loss must not produce a valid task result"
            )
            assert failed.error.code in {
                ErrorCode.DEADLINE_EXCEEDED,
                ErrorCode.WORKER_LOST,
            }
            report["worker_failure"] = {
                "error_code": failed.error.code.name,
                "killed_pid": results[-1]["worker_pid"],
            }
            record("lifecycle.jsonl", report["worker_failure"])
    finally:
        if ids:
            await runtime.gateway.close_sessions(ids)
        await runtime.gateway.stop()
        await runtime.aclose()
    await assert_exited(results[-1]["worker_pid"])
    report["environment_worker_exited"] = True
    if args.fault_test:
        runtime = build_ray_components(config)
        ids = []
        try:
            await runtime.start()
            ids = await create_session(runtime, "after-relaunch")
            reset = unwrap(
                (await runtime.gateway.reset(ids, ResetSpec(seed=native["seed"])))[0]
            )
            pid = reset.observation.extras["worker_pid"]
            assert pid != results[-1]["worker_pid"]
            assert reset.observation.extras["initial_state_sha256"] == expected_initial
            step = unwrap(
                (
                    await runtime.gateway.action_step(
                        ids,
                        [
                            encode_array(
                                np.asarray([replay[0]["action"]], dtype=np.float32)
                            )
                        ],
                    )
                )[0]
            )
            np.testing.assert_allclose(
                step.reward, replay[0]["reward"], atol=1e-5, rtol=1e-5
            )
            np.testing.assert_allclose(
                step.observation.state, replay[0]["state"], atol=1e-5, rtol=1e-5
            )
            unwrap((await runtime.gateway.close_sessions(ids))[0])
            ids = []
            report["worker_recovery"] = {
                "kind": "explicit_runtime_relaunch",
                "worker_pid": pid,
                "initial_state_sha256": expected_initial,
                "first_step_parity": True,
            }
            record("lifecycle.jsonl", report["worker_recovery"])
        finally:
            if ids:
                await runtime.gateway.close_sessions(ids)
            await runtime.gateway.stop()
            await runtime.aclose()
        await assert_exited(pid)
        report["replacement_worker_exited"] = True
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    if deviations:
        raise AssertionError(f"native/Gateway numeric parity failed: {deviations}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--environment", type=Path, required=True)
    parser.add_argument("--gpu", default="0")
    parser.add_argument(
        "--fault-test",
        action="store_true",
        help="Kill this smoke's environment actor, check error propagation, and relaunch the runtime",
    )
    args = parser.parse_args()
    exports = json.loads(args.environment.read_text())
    restart = exports.get("LD_LIBRARY_PATH") != os.environ.get("LD_LIBRARY_PATH")
    os.environ.update(exports)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    if restart:
        os.execv(sys.executable, [sys.executable, *sys.argv])
    args.output.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    import ray

    # Always launch an isolated local cluster, never attach to another user's job.
    ray.init(
        address="local",
        namespace="zetta-runtime",
        num_gpus=1,
        num_cpus=4,
        include_dashboard=False,
        object_store_memory=256 * 1024 * 1024,
    )
    try:
        asyncio.run(exercise(args))
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()
