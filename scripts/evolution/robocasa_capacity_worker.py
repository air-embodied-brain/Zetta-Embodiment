# Copyright (c) 2026 RPent Contributors
"""Persistent JSONL worker for the real single-GPU RoboCasa capacity ladder."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any


def _response(**values: Any) -> None:
    print(json.dumps(values, separators=(",", ":"), default=str), flush=True)


def resolve_gpu(*, slot: int, gpu: str | None, gpu_map: str | None) -> str:
    """Resolve one physical GPU without coupling the benchmark to one device.

    ``gpu_map`` is deliberately a plain comma-separated list so it can be
    embedded in the existing secret-free command template.  Slot assignment is
    deterministic and therefore recoverable: ``gpu_map[slot % len(gpu_map)]``.
    """

    if gpu is not None and gpu_map is not None:
        raise ValueError("--gpu and --gpu-map are mutually exclusive")
    if gpu_map is not None:
        devices = tuple(item.strip() for item in gpu_map.split(",") if item.strip())
        if not devices:
            raise ValueError("--gpu-map must contain at least one GPU")
        if any(not item.isdecimal() for item in devices):
            raise ValueError("--gpu-map entries must be non-negative integers")
        return devices[slot % len(devices)]
    if gpu is None or not gpu.strip():
        raise ValueError("one of --gpu or --gpu-map is required")
    if not gpu.strip().isdecimal():
        raise ValueError("--gpu must be a non-negative integer")
    return gpu.strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slot", type=int, required=True)
    parser.add_argument("--slots", type=int, required=True)
    parser.add_argument("--gpu", default=None)
    parser.add_argument(
        "--gpu-map",
        default=None,
        help="Comma-separated physical GPUs; slot N uses map[N %% len(map)].",
    )
    parser.add_argument("--task", default="SlideDishwasherRack")
    parser.add_argument("--split", default="target")
    parser.add_argument("--base-seed", type=int, default=43000)
    parser.add_argument("--camera-size", type=int, default=256)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--gpu-operation-slots", type=int, default=2)
    parser.add_argument("--cache-root", type=Path, required=True)
    args = parser.parse_args()

    gpu = resolve_gpu(slot=args.slot, gpu=args.gpu, gpu_map=args.gpu_map)

    slot_root = args.cache_root / f"slot-{args.slot:03d}"
    slot_root.mkdir(parents=True, exist_ok=True)
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    os.environ["MUJOCO_EGL_DEVICE_ID"] = gpu
    os.environ["MUJOCO_GL"] = "egl"
    os.environ["PYOPENGL_PLATFORM"] = "egl"
    os.environ["MESA_SHADER_CACHE_DIR"] = str(slot_root / "mesa")
    os.environ["ROBOCASA_MJCF_CACHE_DIR"] = str(slot_root / "mjcf")
    Path(os.environ["MESA_SHADER_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["ROBOCASA_MJCF_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)

    # Third-party RoboCasa code prints status lines to stdout.  A command
    # worker's stdout is a strict JSONL control channel, so all third-party
    # chatter must be redirected before imports or environment operations.
    with contextlib.redirect_stdout(sys.stderr):
        from robots.robocasa.action_contract import zero_action
        from robots.robocasa.env_server import RoboCasaSession

        session = RoboCasaSession(
            camera_size=args.camera_size,
            max_steps=args.max_steps,
            cold_reset_lock=str(args.cache_root / f"cold-reset-gpu-{gpu}.lock"),
            operation_gate_root=str(args.cache_root / "operation-gates"),
            operation_gate_gpu=gpu,
            operation_gate_slots=args.gpu_operation_slots,
            require_isolated_renderer=True,
        )
    episode_index = 0
    try:
        for line in sys.stdin:
            try:
                request = json.loads(line)
                operation = request.get("op")
                if operation == "health":
                    _response(
                        ok=True,
                        valid=True,
                        infra_invalid=False,
                        failure_class="none",
                    )
                elif operation == "reset":
                    seed = args.base_seed + args.slot + episode_index * args.slots
                    episode_index += 1
                    with contextlib.redirect_stdout(sys.stderr):
                        session.reset(
                            {
                                "task": args.task,
                                "split": args.split,
                                "seed": seed,
                                "action_scale": {},
                            }
                        )
                    _response(
                        ok=True,
                        valid=True,
                        infra_invalid=False,
                        failure_class="none",
                    )
                elif operation == "step":
                    with contextlib.redirect_stdout(sys.stderr):
                        session.execute_chunk(
                            {
                                "actions": [zero_action()],
                                "critic_rules": [],
                                "interrupt_on_proposal": False,
                                "capture_event_images": False,
                            }
                        )
                    _response(
                        ok=True,
                        valid=True,
                        infra_invalid=False,
                        vla_queue_s=0.0,
                        failure_class="none",
                    )
                elif operation == "close":
                    _response(
                        ok=True,
                        valid=True,
                        infra_invalid=False,
                        failure_class="none",
                    )
                    break
                else:
                    raise ValueError(f"unknown operation: {operation!r}")
            except Exception as exc:
                traceback.print_exc(file=sys.stderr)
                _response(
                    ok=False,
                    valid=False,
                    infra_invalid=True,
                    failure_class=type(exc).__name__,
                )
    finally:
        with contextlib.redirect_stdout(sys.stderr):
            session.close_environment()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
