# Copyright (c) 2026 Zetta Contributors
"""Capture one real LIBERO observation (post-``reset()``) into an ``.npz``
file for FlashRT's FP8 calibration.

``FlashRtPolicyCore.load()`` (``rollout_runtime/backends/flashrt_policy.py``) can
warm up FlashRT's ``calibrate_with_real_data`` path with a real camera
frame instead of letting FlashRT auto-calibrate on the first live request
(or on synthetic noise, as the v1 standalone benchmark did -- see
the FlashRT report). This script drives a
running rollout_runtime gateway through one ``create_session`` + ``reset`` (no
``policy_step``, so no policy backend needs to be loaded on the server for
this to work -- run it against the same ``zetta_openpi`` preset used for
the latency comparison, or any preset with the same ``env_config``/camera
setup) and dumps the resulting observation's images/state/instruction to
an ``.npz`` consumed by ``FlashRtPolicyConfig.calibration_data_path``.

Usage::

    python3 scripts/experiments/capture_flashrt_calibration_frame.py \\
        --url http://localhost:8711 --token libero-e2e-token \\
        --out /workspace/e2e/flashrt_calibration.npz

The dumped images are the *raw* env-resolution frames (e.g. 256x256 for the
LIBERO-PRO presets used in the FlashRT comparison); ``FlashRtPolicyCore``
resizes them to FlashRT's fixed 224x224 buffer with the same
``_resize_with_pad_uint8`` helper used for live requests, so the saved
frame does not need to be pre-resized.
"""

from __future__ import annotations

import argparse
import os
import asyncio

import numpy as np

from rollout_runtime.api.messages import CreateSessionRequest, EnvSpecMsg, ResetSpec
from rollout_runtime.api.result import Err
from rollout_runtime.core import payload as payload_module
from rollout_runtime.serve.client import RemoteRuntimeClient

DEFAULT_ENV_CONFIG = {
    "task_suite_name": os.environ.get("LIBERO_SUITE", "libero_goal_swap"),
    "task_id": 0,
    "max_episode_steps": 310,
    "camera_height": 256,
    "camera_width": 256,
    "camera_depths": True,
    "num_steps_wait": 15,
    "action_dim": 7,
    "chunk_size": 10,
    "action_model_type": "openpi",
    "return_all_frames": True,
    "libero_variant": "pro",
    "gpu_memory_reserve_mib": 800,
}


async def _capture(url: str, token: str, out_path: str, seed: int) -> None:
    client = RemoteRuntimeClient(url, token=token)
    try:
        created = await client.create_sessions(
            [
                CreateSessionRequest(
                    application_id="ignored-by-server",
                    client_session_key="flashrt-calibration-capture",
                    env_spec=EnvSpecMsg(env_family="libero", env_config=DEFAULT_ENV_CONFIG, pool_size=1),
                    lease_seconds=120.0,
                )
            ]
        )
        assert not isinstance(created[0], Err), created[0]
        session_id = created[0].value.session_id

        reset = await client.reset([session_id], ResetSpec(task_id=0, seed=seed))
        assert not isinstance(reset[0], Err), reset[0]
        observation = reset[0].value.observation

        if observation.main_image is None:
            raise RuntimeError("reset observation is missing main_image; cannot calibrate")
        arrays: dict[str, np.ndarray] = {
            "main_image": payload_module.decode_payload(observation.main_image),
            "state": np.asarray(observation.state, dtype=np.float32),
            "instruction": np.asarray(observation.instruction),
        }
        if observation.wrist_image is not None:
            arrays["wrist_image"] = payload_module.decode_payload(observation.wrist_image)
        if observation.extra_view_images:
            arrays["extra_view_images"] = np.stack(
                [payload_module.decode_payload(ref) for ref in observation.extra_view_images]
            )
        np.savez(out_path, **arrays)
        print(f"saved calibration sample to {out_path}: main_image {arrays['main_image'].shape}")

        await client.close_sessions([session_id])
    finally:
        await client.aclose()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://localhost:8711")
    parser.add_argument("--token", default="libero-e2e-token")
    parser.add_argument("--out", default="flashrt_calibration.npz")
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    asyncio.run(_capture(args.url, args.token, args.out, args.seed))


if __name__ == "__main__":
    main()
