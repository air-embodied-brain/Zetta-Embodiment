# Copyright (c) 2026 Zetta Contributors
"""Zetta RPC adapter for the official Fast-WAM LIBERO checkpoint.

Fast-WAM is kept in an isolated environment because its Wan/Torch dependency
stack differs from Zetta.  The adapter uses the official model, processor,
normalization statistics, and LIBERO action convention while exposing Zetta's
standard ``predict`` protocol.  Optional predicted-future videos are persisted
as immutable intermediate evidence on the server.
"""
from __future__ import annotations

import argparse
import inspect
import os
import sys
import time
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np

from robots.libero.runtime_devices import vla_runtime_info
from zetta.utils.logging import get_logger
from zetta.utils.rpc import RpcFacade

logger = get_logger("fastwam_vla_server")


def _center_crop_resize(image: np.ndarray, *, width: int, height: int) -> np.ndarray:
    from PIL import Image

    source = Image.fromarray(np.asarray(image, dtype=np.uint8))
    src_w, src_h = source.size
    scale = max(width / src_w, height / src_h)
    resized = source.resize(
        (round(src_w * scale), round(src_h * scale)),
        resample=Image.Resampling.BILINEAR,
    )
    rw, rh = resized.size
    left = max((rw - width) // 2, 0)
    top = max((rh - height) // 2, 0)
    return np.asarray(
        resized.crop((left, top, left + width, top + height)), dtype=np.uint8
    )


def _prepare_fastwam_rgb(
    main: np.ndarray,
    wrist: np.ndarray | None,
    image_meta: list[dict[str, Any]],
    *,
    concatenation: str,
) -> np.ndarray:
    """Apply the official per-camera crop/resize and concatenation contract."""
    views = [np.asarray(main, dtype=np.uint8)]
    if len(image_meta) == 2:
        if wrist is None:
            raise ValueError("the released 2-camera Fast-WAM policy requires wrist RGB")
        views.append(np.asarray(wrist, dtype=np.uint8))
    if len(image_meta) not in {1, 2}:
        raise ValueError(f"Fast-WAM LIBERO expects one or two cameras, got {len(image_meta)}")
    resized: list[np.ndarray] = []
    for idx, (view, meta) in enumerate(zip(views, image_meta, strict=True)):
        shape = meta["shape"]
        if len(shape) != 3:
            raise ValueError(f"shape_meta.images[{idx}] must be [C,H,W], got {shape}")
        resized.append(
            _center_crop_resize(view, width=int(shape[2]), height=int(shape[1]))
        )
    if len(resized) == 1:
        return resized[0]
    if concatenation == "horizontal":
        return np.concatenate(resized, axis=1)
    if concatenation == "vertical":
        return np.concatenate(resized, axis=0)
    raise ValueError(f"invalid concat_multi_camera: {concatenation}")


class FastWAMVLAFacade(RpcFacade):
    def __init__(
        self,
        *,
        fastwam_root: str,
        checkpoint: str,
        dataset_stats: str,
        task_config: str = "libero_uncond_2cam224_1e-4",
        device: str = "cuda",
        prediction_dir: str | None = None,
        redirect_common_files: bool | None = None,
    ):
        super().__init__()
        import torch
        from hydra import compose, initialize_config_dir
        from hydra.utils import instantiate
        from omegaconf import OmegaConf

        root = Path(fastwam_root).expanduser().resolve()
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        for name, resolver in (
            ("eval", eval),
            ("max", lambda x: max(x)),
            ("split", lambda value, idx: value.split("/")[int(idx)]),
        ):
            if not OmegaConf.has_resolver(name):
                OmegaConf.register_new_resolver(name, resolver)

        from fastwam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT
        from fastwam.datasets.lerobot.utils.normalizer import (
            load_dataset_stats_from_json,
        )

        overrides = [f"task={task_config}", f"ckpt={checkpoint}"]
        if redirect_common_files is not None:
            overrides.append(
                "model.redirect_common_files="
                + ("true" if redirect_common_files else "false")
            )
        with initialize_config_dir(version_base="1.3", config_dir=str(root / "configs")):
            cfg = compose(
                config_name="sim_libero",
                overrides=overrides,
            )
        dtype_name = str(cfg.get("mixed_precision", "bf16")).lower()
        dtype = {"no": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[
            dtype_name
        ]
        t0 = time.time()
        logger.info("loading Fast-WAM checkpoint %s", checkpoint)
        model = instantiate(cfg.model, model_dtype=dtype, device=device)
        model.load_checkpoint(str(checkpoint))
        self._model = model.to(device).eval()
        self._processor = instantiate(cfg.data.train.processor).eval()
        self._processor.set_normalizer_from_stats(
            load_dataset_stats_from_json(str(dataset_stats))
        )
        self._cfg = cfg
        self._torch = torch
        self._device = device
        self._prompt_template = DEFAULT_PROMPT
        self._prediction_dir = (
            Path(prediction_dir).expanduser().resolve() if prediction_dir else None
        )
        if self._prediction_dir:
            self._prediction_dir.mkdir(parents=True, exist_ok=True)
        self._prediction_index = 0
        logger.info("Fast-WAM ready in %.1fs", time.time() - t0)

    def _dispatch(self, method: str, args: tuple, kwargs: dict) -> Any:
        if method == "predict":
            return self.predict(*args, **kwargs)
        if method == "runtime_info":
            return vla_runtime_info(backend="fast-wam")
        raise ValueError(f"unknown RPC method: {method!r}")

    def _normalize_state(self, state: np.ndarray):
        processor = self._processor
        state_meta = processor.shape_meta["state"]
        if len(state_meta) != 1:
            raise ValueError("Fast-WAM LIBERO expects one merged state key")
        key = state_meta[0]["key"]
        batch = {
            "state": {
                key: self._torch.as_tensor(state, dtype=self._torch.float32)
            }
        }
        batch = processor.action_state_transform(batch)
        batch = processor.normalizer.forward(batch)
        return batch["state"][key]

    def _denormalize_action(self, action) -> np.ndarray:
        if action.ndim == 2:
            action = action.unsqueeze(0)
        meta = self._processor.shape_meta["action"]
        if len(meta) != 1:
            raise ValueError("Fast-WAM LIBERO expects one merged action key")
        normalizer = self._processor.normalizer.normalizers["action"][meta[0]["key"]]
        value = normalizer.backward(action.to(dtype=self._torch.float32, device="cpu"))
        result = value.numpy().astype(np.float32)
        # Official checkpoint convention: normalized 0=close, 1=open;
        # LIBERO OSC convention: +1=close, -1=open.
        result[..., -1] = 1.0 - 2.0 * result[..., -1]
        return np.clip(result, -1.0, 1.0)

    def _save_future_video(self, frames: list[Any]) -> str | None:
        if not frames or self._prediction_dir is None:
            return None
        self._prediction_index += 1
        path = self._prediction_dir / f"prediction_{self._prediction_index:06d}.mp4"
        arrays: list[np.ndarray] = []
        for frame in frames:
            if isinstance(frame, dict):
                arrays.append(
                    np.concatenate([np.asarray(value) for value in frame.values()], axis=1)
                )
            else:
                arrays.append(np.asarray(frame))
        imageio.mimwrite(path, arrays, fps=10)
        return str(path)

    def predict(
        self,
        instruction: str,
        images: dict[str, Any],
        state: list,
        mode: str = "eval",
        parameters: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if mode != "eval":
            raise ValueError("Fast-WAM adapter only supports mode='eval'")
        parameters = dict(parameters or {})
        allowed = {
            "action_horizon",
            "num_inference_steps",
            "text_cfg_scale",
            "negative_prompt",
            "sigma_shift",
            "seed",
            "visualize_future_video",
        }
        unknown = sorted(set(parameters) - allowed)
        if unknown:
            raise ValueError(f"unsupported Fast-WAM parameters: {unknown}")
        from robots.libero.vla_server import _build_env_obs

        env_obs = _build_env_obs(instruction, images, state)
        processor = self._processor
        image_meta = processor.shape_meta["images"][: int(processor.num_output_cameras)]
        rgb = _prepare_fastwam_rgb(
            env_obs["main_images"][0],
            None if env_obs.get("wrist_images") is None else env_obs["wrist_images"][0],
            image_meta,
            concatenation=str(self._cfg.data.train.get("concat_multi_camera", "horizontal")),
        )
        input_image = self._torch.as_tensor(rgb).permute(2, 0, 1).unsqueeze(0)
        input_image = input_image.to(
            device=self._device, dtype=self._model.torch_dtype
        )
        input_image = input_image * (2.0 / 255.0) - 1.0
        proprio = self._normalize_state(np.asarray(state, dtype=np.float32))
        default_horizon = int(self._cfg.data.train.num_frames) - 1
        action_horizon = int(parameters.get("action_horizon", default_horizon))
        if action_horizon < 1:
            raise ValueError("action_horizon must be positive")
        infer_kwargs = {
            "prompt": self._prompt_template.format(task=str(instruction)),
            "input_image": input_image,
            "action_horizon": action_horizon,
            "negative_prompt": str(parameters.get("negative_prompt", "")),
            "text_cfg_scale": float(parameters.get("text_cfg_scale", 1.0)),
            "num_inference_steps": int(
                parameters.get(
                    "num_inference_steps",
                    self._cfg.get("eval_num_inference_steps", 20),
                )
            ),
            "proprio": proprio,
            "sigma_shift": (
                None
                if parameters.get("sigma_shift") is None
                else float(parameters["sigma_shift"])
            ),
            "seed": (
                None if parameters.get("seed") is None else int(parameters["seed"])
            ),
            "rand_device": "cpu",
            "tiled": False,
        }
        visualize = bool(parameters.get("visualize_future_video", False))
        ratio = int(self._cfg.data.train.action_video_freq_ratio)
        num_video_frames = default_horizon // ratio + 1
        if visualize:
            action_conditioned = self._cfg.model.video_dit_config.get(
                "action_conditioned", None
            )
            if action_conditioned is not False:
                raise ValueError(
                    "visualize_future_video requires "
                    "model.video_dit_config.action_conditioned=false"
                )
            infer_kwargs["num_video_frames"] = num_video_frames
        elif "num_video_frames" in inspect.signature(
            self._model.infer_action
        ).parameters:
            # The released evaluator supplies this even for action-only
            # inference when the model signature exposes it.
            infer_kwargs["num_video_frames"] = num_video_frames
        with self._torch.no_grad():
            if visualize:
                prediction = self._model.infer_joint(**infer_kwargs)
                future_path = self._save_future_video(list(prediction.get("video") or []))
            else:
                prediction = self._model.infer_action(**infer_kwargs)
                future_path = None
        actions = self._denormalize_action(prediction["action"])
        return {
            "actions": actions.tolist(),
            "shape": list(actions.shape),
            "dtype": "float32",
            "metadata": {
                "backend": "Fast-WAM-LIBERO",
                "parameters": parameters,
                "predicted_future_video": future_path,
            },
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fastwam-root", default=os.environ.get("FASTWAM_REPO_PATH"))
    parser.add_argument("--checkpoint", default=os.environ.get("FASTWAM_CHECKPOINT_PATH"))
    parser.add_argument("--dataset-stats", default=os.environ.get("FASTWAM_DATASET_STATS_PATH"))
    parser.add_argument("--task-config", default="libero_uncond_2cam224_1e-4")
    parser.add_argument("--prediction-dir", default=os.environ.get("FASTWAM_PREDICTION_DIR"))
    parser.add_argument(
        "--no-redirect-common-files",
        action="store_true",
        help="load the original Wan text encoder/VAE files instead of converted mirrors",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--transport", choices=["http", "socket"], default="http")
    parser.add_argument("--parent-watch", action="store_true")
    args = parser.parse_args()
    missing = [
        name
        for name, value in (
            ("fastwam-root", args.fastwam_root),
            ("checkpoint", args.checkpoint),
            ("dataset-stats", args.dataset_stats),
        )
        if not value
    ]
    if missing:
        raise RuntimeError(f"missing required Fast-WAM arguments: {missing}")
    facade = FastWAMVLAFacade(
        fastwam_root=args.fastwam_root,
        checkpoint=args.checkpoint,
        dataset_stats=args.dataset_stats,
        task_config=args.task_config,
        device=args.device,
        prediction_dir=args.prediction_dir,
        redirect_common_files=False if args.no_redirect_common_files else None,
    )
    facade.serve(
        transport=args.transport,
        host=args.host,
        port=args.port,
        parent_watch=args.parent_watch,
    )


if __name__ == "__main__":
    main()
