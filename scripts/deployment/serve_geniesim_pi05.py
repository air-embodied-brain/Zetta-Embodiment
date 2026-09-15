#!/usr/bin/env python3
# Copyright (c) 2026 Zetta Contributors
"""Serve the pinned Genie Sim instruction Pi0.5 baseline in its model environment.

The upstream checkpoint loader and observation/action transforms remain the
authority. This entrypoint adds local WebSocket serving, seeded sampling and
content-addressed model identity for Campaign.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import logging
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import msgpack
import numpy as np

MODEL_SOURCE_REVISION = "9ded4eb4ce8fa3a969ba1d35a45528155381ea86"
MODEL_CONFIG = "pi05_genie_sim_instruction_and_robust_20260526"
CHECKPOINT_REPOSITORY = "agibot_world/GenieSim3.0-Dataset"
CHECKPOINT_REVISION = "a7792c06746330fb4f566dee8787cb2b4da2200d"
CHECKPOINT_PATH = "checkpoints/instruction_and_robust_pi05"


def _digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def model_identity(source: Path, checkpoint: Path) -> dict[str, Any]:
    revision = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != MODEL_SOURCE_REVISION:
        raise ValueError("model source revision does not match the pinned baseline")
    subprocess.run(["git", "-C", str(source), "diff", "--quiet", "HEAD"], check=True)
    required = (checkpoint / "params", checkpoint / "assets")
    if (
        any(not path.is_dir() for path in required)
        or not (checkpoint / "assets" / "norm_stats.json").is_file()
    ):
        raise ValueError("checkpoint requires params and training norm_stats.json")
    paths = [checkpoint / "_CHECKPOINT_METADATA"]
    for directory in required:
        paths.extend(sorted(path for path in directory.rglob("*") if path.is_file()))
    if not any(path.is_file() for path in (checkpoint / "params").rglob("*")):
        raise ValueError("checkpoint params are empty")
    files = {
        str(path.relative_to(checkpoint)): {
            "sha256": _digest(path),
            "bytes": path.stat().st_size,
        }
        for path in paths
    }
    with Path(__file__).with_name("geniesim_pi05_checkpoint.json").open() as stream:
        expected = json.load(stream)
    if files != expected["files"]:
        raise ValueError("checkpoint files do not match the pinned official weights")
    identity = {
        "source_revision": revision,
        "config": MODEL_CONFIG,
        "sampling_contract": "jax_key_from_uint32_per_inference_v1",
        "files": files,
    }
    digest = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {**identity, "model_version": f"geniesim-pi05:{digest}"}


class SeededPolicyService:
    def __init__(
        self,
        *,
        policy: Any,
        adapt_observation: Any,
        build_response: Any,
        random_key: Any,
        model_version: str,
        output_root: Path,
    ) -> None:
        self.policy = policy
        self.adapt_observation = adapt_observation
        self.build_response = build_response
        self.random_key = random_key
        self.model_version = model_version
        self.output_root = output_root
        self._lock = threading.Lock()
        self._inferences = 0

    def infer(self, packet: bytes) -> bytes:
        if not isinstance(packet, bytes):
            raise ValueError("requests must be binary MessagePack")
        request = msgpack.unpackb(packet, raw=False)
        if not isinstance(request, dict) or request.get("method") != "infer":
            raise ValueError("only the infer method is supported")
        params = request.get("params")
        if not isinstance(params, dict):
            raise ValueError("infer params must be an object")
        seed = params.get("policy_rng")
        if type(seed) is not int or not 0 <= seed < 2**32:
            raise ValueError("policy_rng must be an integer in [0, 2**32)")
        if (
            params.get("robot_type") != "G2_omnipicker"
            or params.get("task_name") != "pick_block_color"
        ):
            raise ValueError("this service supports the G2 pick_block_color profile")
        with self._lock:
            observation = self.adapt_observation(request)
            # The pinned upstream Policy.infer passes _rng directly to the
            # compiled action sampler. Keep assignment and inference serialized.
            self.policy._rng = self.random_key(seed)
            started = time.monotonic()
            action = self.policy.infer(observation)
            actions = np.asarray(action["actions"], dtype=np.float32)
            if (
                actions.ndim != 2
                or actions.shape[1] != 16
                or not 1 <= actions.shape[0] <= 64
                or not np.isfinite(actions).all()
            ):
                raise ValueError("model returned invalid G2 joint actions")
            # The official Policy output transform already applies
            # ACOTAbsoluteActions to the first 14 arm joints. Keep the
            # resulting absolute targets unchanged; adding the observation
            # state here would apply the delta conversion twice.
            response = self.build_response(action)
            response["result"].update(policy_rng=seed, model_version=self.model_version)
            if self._inferences == 0:
                with (self.output_root / "first-request.msgpack").open("xb") as stream:
                    stream.write(packet)
            audit = {
                "inference_index": self._inferences,
                "policy_rng": seed,
                "request_sha256": hashlib.sha256(packet).hexdigest(),
                "actions_sha256": hashlib.sha256(actions.tobytes()).hexdigest(),
                "actions_shape": list(actions.shape),
                "actions_min": actions.min(axis=0).tolist(),
                "actions_max": actions.max(axis=0).tolist(),
                "elapsed_s": time.monotonic() - started,
                "model_version": self.model_version,
            }
            with (self.output_root / "inferences.jsonl").open("a") as stream:
                stream.write(json.dumps(audit, sort_keys=True) + "\n")
            self._inferences += 1
            return msgpack.packb(response, use_bin_type=True)

    def handle(self, socket: Any) -> None:
        try:
            for packet in socket:
                try:
                    response = self.infer(packet)
                except Exception as error:
                    logging.exception("policy inference failed")
                    socket.send(
                        msgpack.packb(
                            {"error": {"message": str(error)}}, use_bin_type=True
                        )
                    )
                    return
                socket.send(response)
        except Exception:
            logging.exception("policy connection closed with an error")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-source", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18990)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    source = args.model_source.resolve()
    checkpoint = args.checkpoint_dir.resolve()
    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=False)
    identity = model_identity(source, checkpoint)
    sys.path.insert(0, str(source / "src"))
    spec = importlib.util.spec_from_file_location(
        "geniesim_baseline_tunnel_agent", source / "scripts" / "tunnel_agent.py"
    )
    upstream = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(upstream)
    import jax
    from websockets.sync.server import serve

    handler = upstream.PolicyHandler(MODEL_CONFIG, str(checkpoint))
    handler.load()
    service = SeededPolicyService(
        policy=handler._policy,
        adapt_observation=upstream._adapt_obs,
        build_response=upstream._build_action_response,
        random_key=jax.random.key,
        model_version=identity["model_version"],
        output_root=output,
    )
    with serve(
        service.handle,
        args.host,
        args.port,
        compression=None,
        max_size=16 * 1024 * 1024,
        close_timeout=1,
    ) as server:
        record = {
            **identity,
            "checkpoint_repository": CHECKPOINT_REPOSITORY,
            "checkpoint_revision": CHECKPOINT_REVISION,
            "checkpoint_path": CHECKPOINT_PATH,
            "checkpoint_directory": str(checkpoint),
            "pid": os.getpid(),
            "devices": [str(device) for device in jax.devices()],
            "endpoint": f"ws://{args.host}:{server.socket.getsockname()[1]}",
        }
        with (output / "model.json").open("x") as stream:
            json.dump(record, stream, sort_keys=True, indent=2)
            stream.write("\n")
        print(json.dumps({"event": "ready", **record}), flush=True)
        server.serve_forever()


if __name__ == "__main__":
    main()
