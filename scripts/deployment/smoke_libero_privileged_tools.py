#!/usr/bin/env python3
"""Exercise collision sensing and retain one real LIBERO RGB-D snapshot."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from robots.libero.env_client import LiberoEnvClient
from robots.libero.toolkit import LiberoToolkit
from robots.libero.tools import dump_state
from rpent.utils.daemon import pick_free_port
from rpent.utils.http_rpc import HttpRpcClient
from rpent.utils.logging import init_output_dir
from rpent.utils.sam3_client import UnavailableSam3Client
from rpent.utils.vla_client import VLAClient


def wait_for_env_server(
    process: subprocess.Popen,
    rpc: HttpRpcClient,
    *,
    log_path: Path,
    timeout_s: float = 300.0,
) -> None:
    """Wait for readiness while failing immediately if the child exits."""
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
        except Exception as exc:  # Server is still starting.
            last_error = exc
            time.sleep(0.5)
    raise TimeoutError(
        f"LIBERO env server did not become ready in {timeout_s:.0f}s; "
        f"last error: {last_error}; see {log_path}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--suite", default="libero_10_swap")
    parser.add_argument("--task", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--cuda-device", type=int, default=0)
    parser.add_argument("--libero-type", default="pro")
    parser.add_argument("--vla-endpoint", default="http://127.0.0.1:18091")
    args = parser.parse_args()

    output = init_output_dir(args.output)
    port = pick_free_port()
    environment = dict(os.environ)
    inherited_pythonpath = environment.get("PYTHONPATH", "")
    environment.update(
        {
            "MUJOCO_GL": "egl",
            "ROBOT_PLATFORM": "LIBERO",
            "LIBERO_TYPE": args.libero_type,
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
                "1000",
                "--cuda-device",
                str(args.cuda_device),
            ],
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
        try:
            rpc = HttpRpcClient(f"http://127.0.0.1:{port}")
            wait_for_env_server(process, rpc, log_path=log_path, timeout_s=300)
            metadata = {
                "suite": args.suite,
                "task": args.task,
                "seed": args.seed,
                "max_episode_steps": 1000,
            }
            env = LiberoEnvClient(rpc, expected_meta=metadata)
            toolkit = LiberoToolkit(
                primitives_kwargs={
                    "env": env,
                    "model": VLAClient(HttpRpcClient(args.vla_endpoint)),
                    "sam3_client": UnavailableSam3Client("smoke test"),
                }
            )
            toolkit._primitives.reset()
            snapshot = dump_state(toolkit._primitives, str(output), 0)
            collision = toolkit._collision_check(include_all_contacts=True)
            result = {
                "suite": args.suite,
                "task": args.task,
                "seed": args.seed,
                "collision_check": collision,
                "snapshot": {
                    "step": snapshot.get("step"),
                    "eef_pos": snapshot.get("eef_pos"),
                    "artifact_root": str(output),
                },
                "environment_advanced_by_collision_check": False,
            }
            (output / "smoke_result.json").write_text(
                json.dumps(result, indent=2), encoding="utf-8"
            )
            print(json.dumps(result, sort_keys=True))
        finally:
            process.terminate()
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
            time.sleep(0.2)


if __name__ == "__main__":
    main()
