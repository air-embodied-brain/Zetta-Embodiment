# Copyright (c) 2026 RPent Contributors
"""RPent RPC adapter for NVIDIA GR00T N1.7 LIBERO.

This server intentionally lives in a separate process/environment from RPent.
Install the official Isaac-GR00T repository there and use the official
``nvidia/GR00T-N1.7-LIBERO`` checkpoint with embodiment ``LIBERO_PANDA``.
The adapter translates RPent's compact two-camera/8-D-state wire protocol to
GR00T's simulation policy API and translates its split action streams back to
LIBERO's 7-D OSC action.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from robots.libero.runtime_devices import vla_runtime_info
from rpent.utils.logging import get_logger
from rpent.utils.rpc import RpcFacade

logger = get_logger("groot_vla_server")
ACTION_KEYS = ("x", "y", "z", "roll", "pitch", "yaw", "gripper")


def _repeat_time(value: np.ndarray, horizon: int) -> np.ndarray:
    arr = np.asarray(value)
    return np.repeat(arr[:, None], int(horizon), axis=1)


def build_groot_observation(
    instruction: str,
    env_obs: dict[str, Any],
    modality_configs: dict[str, Any],
) -> dict[str, Any]:
    """Build GR00T's flat simulation observation without privileged state."""
    main = np.asarray(env_obs["main_images"], dtype=np.uint8)
    wrist = env_obs.get("wrist_images")
    if wrist is None:
        raise ValueError("GR00T LIBERO requires the wrist camera")
    wrist = np.asarray(wrist, dtype=np.uint8)
    state = np.asarray(env_obs["states"], dtype=np.float32)
    if state.ndim != 2 or state.shape[1] < 8:
        raise ValueError(f"GR00T LIBERO state must be [B,>=8], got {state.shape}")

    video_horizon = len(modality_configs["video"].delta_indices)
    state_horizon = len(modality_configs["state"].delta_indices)
    out: dict[str, Any] = {}
    for key in modality_configs["video"].modality_keys:
        if key == "image":
            out["video.image"] = _repeat_time(main, video_horizon)
        elif key == "wrist_image":
            out["video.wrist_image"] = _repeat_time(wrist, video_horizon)
        else:
            raise ValueError(f"unsupported GR00T LIBERO video stream: {key}")

    state_slices = {
        "x": state[:, 0:1],
        "y": state[:, 1:2],
        "z": state[:, 2:3],
        "roll": state[:, 3:4],
        "pitch": state[:, 4:5],
        "yaw": state[:, 5:6],
        "gripper": state[:, 6:8],
    }
    for key in modality_configs["state"].modality_keys:
        if key not in state_slices:
            raise ValueError(f"unsupported GR00T LIBERO state stream: {key}")
        out[f"state.{key}"] = _repeat_time(state_slices[key], state_horizon).astype(
            np.float32
        )
    for key in modality_configs["language"].modality_keys:
        out[key] = [str(instruction)] * state.shape[0]
    return out


def flatten_groot_actions(action: dict[str, Any]) -> np.ndarray:
    """Combine split GR00T streams and apply official LIBERO gripper mapping."""
    streams: list[np.ndarray] = []
    for key in ACTION_KEYS:
        wire_key = f"action.{key}"
        if wire_key not in action:
            raise ValueError(f"GR00T response omitted {wire_key}")
        arr = np.asarray(action[wire_key], dtype=np.float32)
        if arr.ndim != 3 or arr.shape[-1] != 1:
            raise ValueError(f"{wire_key} must be [B,T,1], got {arr.shape}")
        streams.append(arr)
    merged = np.concatenate(streams, axis=-1)
    # GR00T's LIBERO dataset uses 0=close, 1=open.  The environment consumes
    # +1=close, -1=open, matching the official evaluation wrapper.
    merged[..., -1] = 1.0 - 2.0 * merged[..., -1]
    return np.clip(merged, -1.0, 1.0).astype(np.float32)


class GrootVLAFacade(RpcFacade):
    def __init__(self, model_path: str, *, device: str = "cuda:0", strict: bool = True):
        super().__init__()
        from gr00t.data.embodiment_tags import EmbodimentTag
        from gr00t.policy.gr00t_policy import Gr00tPolicy, Gr00tSimPolicyWrapper

        t0 = time.time()
        logger.info("loading GR00T N1.7 LIBERO from %s", model_path)
        policy = Gr00tPolicy(
            model_path=model_path,
            embodiment_tag=EmbodimentTag.LIBERO_PANDA,
            device=device,
            strict=strict,
        )
        self._policy = Gr00tSimPolicyWrapper(policy, strict=strict)
        self._modalities = self._policy.get_modality_config()
        logger.info("GR00T ready in %.1fs", time.time() - t0)

    def _dispatch(self, method: str, args: tuple, kwargs: dict) -> Any:
        if method == "predict":
            return self.predict(*args, **kwargs)
        if method == "runtime_info":
            return vla_runtime_info(backend="groot-n1.7")
        raise ValueError(f"unknown RPC method: {method!r}")

    def predict(
        self,
        instruction: str,
        images: dict[str, Any],
        state: list,
        mode: str = "eval",
        parameters: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if mode != "eval":
            raise ValueError("GR00T adapter only supports mode='eval'")
        parameters = dict(parameters or {})
        unknown = sorted(set(parameters) - {"action_horizon"})
        if unknown:
            raise ValueError(f"unsupported GR00T parameters: {unknown}")
        from robots.libero.vla_server import _build_env_obs

        env_obs = _build_env_obs(instruction, images, state)
        observation = build_groot_observation(
            instruction, env_obs, self._modalities
        )
        action, info = self._policy.get_action(observation)
        merged = flatten_groot_actions(action)
        if "action_horizon" in parameters:
            horizon = int(parameters["action_horizon"])
            if not 1 <= horizon <= merged.shape[1]:
                raise ValueError(
                    f"action_horizon must be in [1,{merged.shape[1]}]"
                )
            merged = merged[:, :horizon]
        return {
            "actions": merged.tolist(),
            "shape": list(merged.shape),
            "dtype": "float32",
            "metadata": {
                "backend": "nvidia/GR00T-N1.7-LIBERO",
                "parameters": parameters,
                "policy_info": info if isinstance(info, dict) else {},
            },
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--groot-root", default=os.environ.get("GROOT_REPO_PATH"))
    parser.add_argument(
        "--model-path", default=os.environ.get("GROOT_LIBERO_CHECKPOINT_PATH")
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--transport", choices=["http", "socket"], default="http")
    parser.add_argument("--parent-watch", action="store_true")
    parser.add_argument("--no-strict", action="store_true")
    args = parser.parse_args()
    if args.groot_root:
        root = str(Path(args.groot_root).expanduser().resolve())
        if root not in sys.path:
            sys.path.insert(0, root)
    if not args.model_path:
        raise RuntimeError("provide --model-path or GROOT_LIBERO_CHECKPOINT_PATH")
    facade = GrootVLAFacade(
        args.model_path, device=args.device, strict=not args.no_strict
    )
    facade.serve(
        transport=args.transport,
        host=args.host,
        port=args.port,
        parent_watch=args.parent_watch,
    )


if __name__ == "__main__":
    main()
