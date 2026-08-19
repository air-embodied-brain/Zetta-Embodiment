# Copyright (c) 2026 Zetta Contributors
"""Sensor-only, proposal-only GraspGen adapter for Zetta LIBERO."""

from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.request
from typing import Any

import numpy as np

from robots.libero.contact_graspnet import (
    camera_point_cloud,
    transform_grasp_candidates,
)
from zetta.utils.logging import get_output_dir


def _cap_points(points: np.ndarray, max_points: int) -> np.ndarray:
    value = np.asarray(points, dtype=np.float32)
    if value.ndim != 2 or value.shape[1] != 3:
        raise ValueError("point cloud must have shape Nx3")
    if len(value) <= int(max_points):
        return value
    digest = hashlib.sha256(value.tobytes()).digest()
    generator = np.random.default_rng(int.from_bytes(digest[:8], "big"))
    selected = generator.choice(len(value), size=int(max_points), replace=False)
    return value[selected]


def prepare_sensor_object_cloud(
    depth_m: np.ndarray,
    intrinsic_k: np.ndarray,
    camera_to_world: np.ndarray,
    *,
    target_world_xyz: list[float] | None,
    crop_radius_m: float,
    min_depth_m: float,
    max_depth_m: float,
    max_points: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Build an object-centred camera-frame cloud without simulator geometry."""
    extrinsic = np.asarray(camera_to_world, dtype=np.float64)
    if extrinsic.shape != (4, 4) or not np.isfinite(extrinsic).all():
        raise ValueError("camera_to_world must be one finite 4x4 matrix")
    if not 32 <= int(max_points) <= 65536:
        raise ValueError("max_points must be in [32,65536]")
    if not 0.01 <= float(crop_radius_m) <= 1.0:
        raise ValueError("crop_radius_m must be in [0.01,1.0]")

    camera_points = camera_point_cloud(
        depth_m,
        intrinsic_k,
        min_depth_m=float(min_depth_m),
        max_depth_m=float(max_depth_m),
        max_points=200_000,
    )
    selected = camera_points
    targeting: dict[str, Any] = {"mode": "full_scene_sensor_cloud"}
    if target_world_xyz is not None:
        target = np.asarray(target_world_xyz, dtype=np.float64)
        if target.shape != (3,) or not np.isfinite(target).all():
            raise ValueError("target_world_xyz must contain three finite numbers")
        world = (extrinsic @ np.c_[camera_points, np.ones(len(camera_points))].T).T[
            :, :3
        ]
        distances = np.linalg.norm(world - target[None, :], axis=1)
        selected = camera_points[distances <= float(crop_radius_m)]
        targeting = {
            "mode": "planner_sensor_roi",
            "target_world_xyz": target.astype(float).tolist(),
            "crop_radius_m": float(crop_radius_m),
        }
    if len(selected) < 32:
        raise ValueError(f"sensor ROI contains only {len(selected)} depth points")
    selected = _cap_points(selected, int(max_points))
    centroid_camera = selected.mean(axis=0, dtype=np.float64)
    model_points = (selected.astype(np.float64) - centroid_camera).astype(np.float32)
    targeting.update(
        {
            "selected_point_count": len(selected),
            "centroid_camera": centroid_camera.astype(float).tolist(),
        }
    )
    return model_points, centroid_camera, targeting


def transform_graspgen_candidates(
    candidates: list[dict[str, Any]],
    centroid_camera: np.ndarray,
    camera_to_world: np.ndarray,
) -> list[dict[str, Any]]:
    """Undo object centring and transform valid GraspGen poses to world."""
    centroid = np.asarray(centroid_camera, dtype=np.float64)
    if centroid.shape != (3,) or not np.isfinite(centroid).all():
        raise ValueError("centroid_camera must be one finite xyz")
    camera_candidates: list[dict[str, Any]] = []
    for candidate in candidates:
        raw_pose = candidate.get(
            "transform_model",
            candidate.get("transform", candidate.get("grasp_pose")),
        )
        pose = np.asarray(raw_pose, dtype=np.float64)
        if pose.shape != (4, 4) or not np.isfinite(pose).all():
            continue
        pose = pose.copy()
        pose[:3, 3] += centroid
        row = dict(candidate)
        row["transform_camera"] = pose.astype(float).tolist()
        camera_candidates.append(row)
    return transform_grasp_candidates(camera_candidates, camera_to_world)


class GraspGenAdapter:
    """Expose GraspGen while forbidding privileged handle geometry."""

    def __init__(self, endpoint: str | None = None) -> None:
        self.endpoint = (endpoint or os.environ.get("GRASPGEN_URL", "")).rstrip("/")
        self.call_index = 0

    def _health(self, timeout_s: float) -> dict[str, Any]:
        if not self.endpoint:
            return {
                "available": False,
                "status": "unconfigured",
                "required_env": "GRASPGEN_URL",
            }
        request = urllib.request.Request(self.endpoint + "/health", method="GET")
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            result = json.loads(response.read().decode("utf-8"))
        return {"available": bool(result.get("ready", result.get("ok"))), **result}

    def propose(
        self,
        mode: str = "propose",
        camera: str = "agentview",
        step: int | None = None,
        target_world_xyz: list[float] | None = None,
        crop_radius_m: float = 0.12,
        max_candidates: int = 16,
        max_points: int = 2048,
        min_depth_m: float = 0.05,
        max_depth_m: float = 3.0,
        filter_collisions: bool = True,
        remove_outliers: bool = False,
        timeout_s: float = 420.0,
    ) -> dict[str, Any]:
        if mode == "health":
            try:
                return self._health(float(timeout_s))
            except Exception as exc:
                return {
                    "available": False,
                    "status": "health_error",
                    "error": f"{type(exc).__name__}: {exc}",
                }
        if mode != "propose":
            return {"error": "mode must be 'health' or 'propose'"}
        if not self.endpoint:
            return {
                "available": False,
                "status": "unconfigured",
                "required_env": "GRASPGEN_URL",
                "proposal_only": True,
                "privileged_geometry_used": False,
            }
        if camera not in {"agentview", "wrist"}:
            return {"error": "camera must be 'agentview' or 'wrist'"}
        if not 1 <= int(max_candidates) <= 64:
            return {"error": "max_candidates must be in [1,64]"}

        from robots.libero import tools as libero_tools

        latest = libero_tools._latest_step()
        selected_step = latest if step is None else int(step)
        if selected_step is None:
            return {"error": "no LIBERO state/depth artifacts are available"}
        try:
            depth = libero_tools._load_depth(camera, selected_step)
            metadata = libero_tools._load_camera_meta(camera, selected_step)
            model_points, centroid_camera, targeting = prepare_sensor_object_cloud(
                depth,
                np.asarray(metadata["intrinsic_K"]),
                np.asarray(metadata["extrinsic_cam2world"]),
                target_world_xyz=target_world_xyz,
                crop_radius_m=float(crop_radius_m),
                min_depth_m=float(min_depth_m),
                max_depth_m=float(max_depth_m),
                max_points=int(max_points),
            )
        except Exception as exc:
            return {"error": f"sensor point cloud build failed: {type(exc).__name__}: {exc}"}

        self.call_index += 1
        evidence_dir = (
            get_output_dir()
            / "graspgen"
            / f"call-{self.call_index:04d}-step-{selected_step:02d}-{camera}"
        )
        while evidence_dir.exists():
            self.call_index += 1
            evidence_dir = (
                get_output_dir()
                / "graspgen"
                / f"call-{self.call_index:04d}-step-{selected_step:02d}-{camera}"
            )
        evidence_dir.mkdir(parents=True, exist_ok=False)
        point_path = evidence_dir / "point_cloud_model_frame.npz"
        np.savez_compressed(
            point_path,
            point_cloud=model_points,
            centroid_camera=centroid_camera,
            intrinsic_K=np.asarray(metadata["intrinsic_K"], dtype=np.float64),
            extrinsic_cam2world=np.asarray(
                metadata["extrinsic_cam2world"], dtype=np.float64
            ),
        )
        payload = {
            "point_cloud": model_points.astype(float).tolist(),
            "frame": "object_centered_opencv_camera_xyz_m",
            "gripper": "franka_panda",
            "num_grasps": max(64, int(max_candidates)),
            "topk_num_grasps": int(max_candidates),
            "filter_collisions": bool(filter_collisions),
            "remove_outliers": bool(remove_outliers),
        }
        started = time.perf_counter()
        try:
            request = urllib.request.Request(
                self.endpoint + "/generate",
                data=json.dumps(payload, separators=(",", ":")).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=float(timeout_s)) as response:
                raw = json.loads(response.read().decode("utf-8"))
        except Exception as exc:
            failure = {
                "available": False,
                "status": "service_error",
                "error": f"{type(exc).__name__}: {exc}",
                "proposal_only": True,
                "privileged_geometry_used": False,
                "point_cloud_path": str(point_path),
                "point_count": len(model_points),
            }
            (evidence_dir / "result.json").write_text(
                json.dumps(failure, indent=2), encoding="utf-8"
            )
            return failure

        candidates = transform_graspgen_candidates(
            list(raw.get("grasps", raw.get("candidates", []))),
            centroid_camera,
            np.asarray(metadata["extrinsic_cam2world"], dtype=np.float64),
        )[: int(max_candidates)]
        result = {
            "available": True,
            "status": str(raw.get("status", "unknown")),
            "proposal_only": True,
            "environment_advanced": False,
            "privileged_geometry_used": False,
            "camera": camera,
            "step": selected_step,
            "targeting": targeting,
            "point_count": len(model_points),
            "point_cloud_sha256": hashlib.sha256(model_points.tobytes()).hexdigest(),
            "point_cloud_path": str(point_path),
            "candidate_count": len(candidates),
            "candidates": candidates,
            "collision_filter_requested": bool(filter_collisions),
            # A request parameter is not proof of backend execution. Preserve
            # an attested boolean only when the service explicitly returns it.
            "collision_filter_applied": (
                raw.get("collision_filter_applied")
                if isinstance(raw.get("collision_filter_applied"), bool)
                else (
                    raw.get("evidence", {}).get("collision_filter_applied")
                    if isinstance(raw.get("evidence"), dict)
                    and isinstance(
                        raw.get("evidence", {}).get("collision_filter_applied"),
                        bool,
                    )
                    else None
                )
            ),
            "latency_s": round(time.perf_counter() - started, 3),
            "execution_warning": (
                "GraspGen candidates are perception proposals, not Panda commands. "
                "Check current reachability and collision/contact evidence before motion."
            ),
            "service_evidence": raw.get("evidence", {}),
        }
        (evidence_dir / "result.json").write_text(
            json.dumps(result, indent=2), encoding="utf-8"
        )
        return result
