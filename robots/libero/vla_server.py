# Copyright (c) 2026 RPent Contributors
"""RPC server wrapping the Pi0.5 VLA."""
from __future__ import annotations

import argparse
import base64
import io
import os
import sys
import threading
import time
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf

from robots.libero.runtime_devices import vla_runtime_info
from rpent.utils.config import (
    get_pi05_checkpoint_path,
    get_repo_root,
    get_rlinf_repo_path,
)
from rpent.utils.logging import get_logger
from rpent.utils.rpc import RpcFacade

logger = get_logger("vla_server")

RPENT_ROOT = get_repo_root()
RLINF_REPO_PATH = get_rlinf_repo_path() or (RPENT_ROOT.parent / "rlinf").resolve()
if str(RLINF_REPO_PATH) not in sys.path:
    sys.path.insert(0, str(RLINF_REPO_PATH))
os.environ.setdefault("ROBOT_PLATFORM", "LIBERO")

# ---------------------------------------------------------------------------
# Config builders
# ---------------------------------------------------------------------------


def build_model_cfg(model_path: str) -> Any:
    """OmegaConf for ``rlinf.models.embodiment.openpi.get_model``."""
    return OmegaConf.create(
        {
            "model_type": "openpi",
            "model_path": model_path,
            "precision": None,
            "num_action_chunks": 5,
            "action_dim": 7,
            "is_lora": False,
            "lora_rank": 32,
            "use_proprio": True,
            "num_steps": 5,
            "add_value_head": False,
            "openpi": {
                "config_name": "pi05_libero",
                "num_images_in_input": 2,
                "noise_level": 0.5,
                "action_chunk": 5,
                "num_steps": 5,
                "train_expert_only": True,
                "action_env_dim": 7,
                "noise_method": "flow_sde",
                "add_value_head": False,
                "value_after_vlm": False,
                "value_vlm_mode": "mean_token",
                "detach_critic_input": None,
                "use_dsrl": False,
            },
        }
    )


def _decode_image_block(block: dict[str, Any]) -> np.ndarray:
    import imageio.v2 as imageio
    fmt = (block.get("format") or "png").lower()
    if fmt != "png":
        raise ValueError(f"unsupported image format: {fmt!r} (only 'png')")
    data = block.get("data")
    if not isinstance(data, str) or not data:
        raise ValueError("image block missing base64 'data'")
    raw = base64.b64decode(data)
    img = np.asarray(imageio.imread(io.BytesIO(raw)))
    if img.ndim != 3 or img.shape[-1] != 3:
        raise ValueError(f"image must be HxWx3 RGB; got {img.shape}")
    if img.dtype != np.uint8:
        img = img.astype(np.uint8)
    return img


def _build_env_obs(instruction: str, images: dict[str, Any],
                   state: list) -> dict[str, Any]:
    if "main" not in images:
        raise ValueError("'images.main' is required")
    main = _decode_image_block(images["main"])
    main_batch = main[None]
    # ``wrist_images`` / ``extra_view_images`` must be present even when unused;
    # downstream policies (e.g. openpi ``obs_processor``) index them directly
    # rather than ``.get`` and would KeyError on missing keys.
    obs: dict[str, Any] = {
        "main_images": main_batch,
        "task_descriptions": [str(instruction)] * main_batch.shape[0],
        "wrist_images": None,
        "extra_view_images": None,
    }
    if isinstance(images.get("wrist"), dict):
        obs["wrist_images"] = _decode_image_block(images["wrist"])[None]
    if isinstance(images.get("extra"), dict):
        obs["extra_view_images"] = _decode_image_block(images["extra"])[None]
    states = np.asarray(state, dtype=np.float32)
    if states.ndim != 2:
        raise ValueError(f"state must be [B, state_dim]; got shape {states.shape}")
    obs["states"] = states
    return obs


# ---------------------------------------------------------------------------
# Facade implementing the rpent.utils.vla_client protocol
# ---------------------------------------------------------------------------


class VLAFacade(RpcFacade):
    """Implements :class:`rpent.utils.vla_client.VLAClient` over a Pi0.5 model.

    Loads the model once at construction; each ``predict`` call runs one
    single-env inference and returns a JSON-safe dict.
    """

    def __init__(self, model_path: str):
        super().__init__()
        from rlinf.models.embodiment.openpi import get_model as get_openpi_model

        cfg = build_model_cfg(model_path=model_path)
        t0 = time.time()
        logger.info("loading Pi0.5 (model_path=%s) ...", cfg["model_path"])
        self._model = get_openpi_model(cfg, torch_dtype=None).cuda().eval()
        self._inference_lock = threading.Lock()
        logger.info("model ready in %.1fs", time.time() - t0)

    def _dispatch(self, method: str, args: tuple, kwargs: dict) -> Any:
        if method == "predict":
            return self.predict(*args, **kwargs)
        if method == "runtime_info":
            return vla_runtime_info(backend="pi0.5")
        raise ValueError(f"unknown RPC method: {method!r}")

    def predict(self, instruction: str, images: dict[str, Any], state: list,
                mode: str = "eval", parameters: dict[str, Any] | None = None) -> dict[str, Any]:
        env_obs = _build_env_obs(instruction, images, state)
        parameters = dict(parameters or {})
        unknown = sorted(set(parameters) - {"num_steps", "seed"})
        if unknown:
            raise ValueError(
                "Pi0.5 only supports per-call parameters 'num_steps' and 'seed'; "
                f"got unsupported keys: {unknown}"
            )
        predict_kwargs: dict[str, Any] = {"mode": mode}
        if "num_steps" in parameters:
            num_steps = int(parameters["num_steps"])
            if not 1 <= num_steps <= 64:
                raise ValueError("num_steps must be in [1, 64]")
            predict_kwargs["num_steps"] = num_steps
        seed = parameters.get("seed")
        if seed is not None:
            seed = int(seed)
            if not 0 <= seed <= 0x7FFFFFFF:
                raise ValueError("seed must be in [0, 2^31-1]")
        # Pi0.5 samples flow noise with the process-global torch RNG. Serialize
        # inference and fork the RNG so paired parent/candidate calls can bind a
        # per-call policy seed without perturbing concurrent unrelated jobs.
        device = torch.cuda.current_device()
        with self._inference_lock, torch.random.fork_rng(devices=[device]):
            if seed is not None:
                torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
            with torch.no_grad():
                try:
                    actions, _ = self._model.predict_action_batch(
                        env_obs, **predict_kwargs
                    )
                except TypeError as exc:
                    if "num_steps" in predict_kwargs:
                        raise ValueError(
                            "this Pi0.5 checkpoint does not expose per-call num_steps"
                        ) from exc
                    raise
        actions_np = (
            actions.detach().cpu().numpy()
            if isinstance(actions, torch.Tensor)
            else np.asarray(actions)
        ).astype(np.float32)
        return {
            "actions": actions_np.tolist(),
            "shape": list(actions_np.shape),
            "dtype": "float32",
            "metadata": {
                "backend": "pi0.5",
                "parameters": parameters,
            },
        }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--transport", choices=["socket", "http"], default="http")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=0)
    p.add_argument(
        "--model-path",
        default=None,
        help="Pi0.5 checkpoint (defaults to PI05_CHECKPOINT_PATH env)",
    )
    p.add_argument("--parent-watch", action="store_true",
                   help="watch parent process via stdin pipe and exit when it dies")
    p.add_argument("--cuda-device", type=int, default=None,
                   help="GPU device exposed through CUDA_VISIBLE_DEVICES.")
    args = p.parse_args()

    if args.cuda_device is not None:
        target = str(args.cuda_device)
        prev = os.environ.get("CUDA_VISIBLE_DEVICES")
        if prev is not None and prev != target:
            logger.warning(
                "CUDA_VISIBLE_DEVICES=%s is already set; overriding with --cuda-device=%s",
                prev, args.cuda_device,
            )
        os.environ["CUDA_VISIBLE_DEVICES"] = target

    model_path = args.model_path or get_pi05_checkpoint_path()
    if not model_path:
        raise RuntimeError(
            "PI05_CHECKPOINT_PATH is not set; provide the Pi0.5 checkpoint "
            "path via --model-path or the environment."
        )

    facade = VLAFacade(model_path=model_path)
    facade.serve(transport=args.transport, host=args.host, port=args.port,
                 parent_watch=args.parent_watch)


if __name__ == "__main__":
    main()
