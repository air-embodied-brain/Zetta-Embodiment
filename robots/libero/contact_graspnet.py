"""Proposal-only Contact-GraspNet adapter for Zetta LIBERO observations."""

from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np

from zetta.utils.logging import get_output_dir


def _quaternion_xyzw(rotation: np.ndarray) -> list[float]:
    matrix = np.asarray(rotation, dtype=np.float64)
    if matrix.shape != (3, 3):
        raise ValueError("rotation must be 3x3")
    trace = float(np.trace(matrix))
    if trace > 0:
        scale = np.sqrt(trace + 1.0) * 2
        value = np.array(
            [
                (matrix[2, 1] - matrix[1, 2]) / scale,
                (matrix[0, 2] - matrix[2, 0]) / scale,
                (matrix[1, 0] - matrix[0, 1]) / scale,
                0.25 * scale,
            ]
        )
    else:
        axis = int(np.argmax(np.diag(matrix)))
        if axis == 0:
            scale = np.sqrt(1 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2
            value = np.array(
                [
                    0.25 * scale,
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                    (matrix[2, 1] - matrix[1, 2]) / scale,
                ]
            )
        elif axis == 1:
            scale = np.sqrt(1 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2
            value = np.array(
                [
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    0.25 * scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                    (matrix[0, 2] - matrix[2, 0]) / scale,
                ]
            )
        else:
            scale = np.sqrt(1 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2
            value = np.array(
                [
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                    0.25 * scale,
                    (matrix[1, 0] - matrix[0, 1]) / scale,
                ]
            )
    value /= max(float(np.linalg.norm(value)), 1e-12)
    if value[3] < 0:
        value = -value
    return value.astype(float).tolist()


def camera_point_cloud(
    depth_m: np.ndarray,
    intrinsic_k: np.ndarray,
    *,
    min_depth_m: float,
    max_depth_m: float,
    max_points: int,
) -> np.ndarray:
    """Back-project metric depth into the OpenCV-style camera frame."""
    depth = np.asarray(depth_m, dtype=np.float64)
    intrinsic = np.asarray(intrinsic_k, dtype=np.float64)
    if depth.ndim != 2 or intrinsic.shape != (3, 3):
        raise ValueError("depth must be HxW and intrinsic_k must be 3x3")
    if not 0 < min_depth_m < max_depth_m:
        raise ValueError("depth limits must satisfy 0 < min < max")
    if not 32 <= int(max_points) <= 200_000:
        raise ValueError("max_points must be in [32, 200000]")
    rows, cols = np.mgrid[0 : depth.shape[0], 0 : depth.shape[1]]
    valid = np.isfinite(depth) & (depth >= min_depth_m) & (depth <= max_depth_m)
    z = depth[valid]
    if z.size < 32:
        raise ValueError(f"only {z.size} valid depth points")
    x = (cols[valid] - intrinsic[0, 2]) * z / intrinsic[0, 0]
    y = (rows[valid] - intrinsic[1, 2]) * z / intrinsic[1, 1]
    points = np.column_stack([x, y, z]).astype(np.float32)
    if len(points) > int(max_points):
        digest = hashlib.sha256(points.tobytes()).digest()
        generator = np.random.default_rng(int.from_bytes(digest[:8], "big"))
        selected = generator.choice(len(points), size=int(max_points), replace=False)
        points = points[selected]
    return points


def transform_grasp_candidates(
    candidates: list[dict[str, Any]], camera_to_world: np.ndarray
) -> list[dict[str, Any]]:
    extrinsic = np.asarray(camera_to_world, dtype=np.float64)
    if extrinsic.shape != (4, 4):
        raise ValueError("camera_to_world must be 4x4")
    transformed: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates):
        camera_pose = np.asarray(candidate.get("transform_camera"), dtype=np.float64)
        if camera_pose.shape != (4, 4) or not np.isfinite(camera_pose).all():
            continue
        world_pose = extrinsic @ camera_pose
        row = dict(candidate)
        row.update(
            {
                "candidate_index": index,
                "transform_world": world_pose.astype(float).tolist(),
                "position_world": world_pose[:3, 3].astype(float).tolist(),
                "quaternion_world_xyzw": _quaternion_xyzw(world_pose[:3, :3]),
            }
        )
        contact = np.asarray(candidate.get("contact_point_camera", []), dtype=np.float64)
        if contact.shape == (3,) and np.isfinite(contact).all():
            world_contact = extrinsic @ np.r_[contact, 1.0]
            row["contact_point_world"] = world_contact[:3].astype(float).tolist()
        transformed.append(row)
    return transformed


class ContactGraspNetAdapter:
    """Turn Zetta metric depth into bounded, auditable grasp proposals."""

    def __init__(self, endpoint: str | None = None) -> None:
        self.endpoint = (endpoint or os.environ.get("CONTACT_GRASPNET_URL", "")).rstrip(
            "/"
        )
        self.call_index = 0

    def _health(self, timeout_s: float) -> dict[str, Any]:
        if not self.endpoint:
            return {
                "available": False,
                "status": "unconfigured",
                "required_env": "CONTACT_GRASPNET_URL",
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
        max_candidates: int = 16,
        max_points: int = 65536,
        min_depth_m: float = 0.05,
        max_depth_m: float = 3.0,
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
                "required_env": "CONTACT_GRASPNET_URL",
                "proposal_only": True,
            }
        if camera not in {"agentview", "wrist"}:
            return {"error": "camera must be 'agentview' or 'wrist'"}
        if not 1 <= int(max_candidates) <= 64:
            return {"error": "max_candidates must be in [1,64]"}

        # Import lazily to keep this adapter unit-testable without LIBERO.
        from robots.libero import tools as libero_tools

        latest = libero_tools._latest_step()
        selected_step = latest if step is None else int(step)
        if selected_step is None:
            return {"error": "no LIBERO state/depth artifacts are available"}
        try:
            depth = libero_tools._load_depth(camera, selected_step)
            metadata = libero_tools._load_camera_meta(camera, selected_step)
            points = camera_point_cloud(
                depth,
                np.asarray(metadata["intrinsic_K"]),
                min_depth_m=float(min_depth_m),
                max_depth_m=float(max_depth_m),
                max_points=int(max_points),
            )
        except Exception as exc:
            return {"error": f"point cloud build failed: {type(exc).__name__}: {exc}"}

        self.call_index += 1
        evidence_dir = (
            get_output_dir()
            / "contact_graspnet"
            / f"call-{self.call_index:04d}-step-{selected_step:02d}-{camera}"
        )
        while evidence_dir.exists():
            self.call_index += 1
            evidence_dir = (
                get_output_dir()
                / "contact_graspnet"
                / f"call-{self.call_index:04d}-step-{selected_step:02d}-{camera}"
            )
        evidence_dir.mkdir(parents=True, exist_ok=False)
        point_path = evidence_dir / "point_cloud_camera.npz"
        np.savez_compressed(
            point_path,
            point_cloud=points,
            intrinsic_K=np.asarray(metadata["intrinsic_K"], dtype=np.float64),
            extrinsic_cam2world=np.asarray(
                metadata["extrinsic_cam2world"], dtype=np.float64
            ),
        )
        payload = {
            "point_cloud": points.astype(float).tolist(),
            "max_candidates": int(max_candidates),
            "local_regions": False,
            "filter_grasps": False,
        }
        started = time.perf_counter()
        try:
            request = urllib.request.Request(
                self.endpoint + "/propose",
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
                "point_cloud_path": str(point_path),
                "point_count": len(points),
            }
            (evidence_dir / "result.json").write_text(
                json.dumps(failure, indent=2), encoding="utf-8"
            )
            return failure

        candidates = transform_grasp_candidates(
            list(raw.get("grasps", [])),
            np.asarray(metadata["extrinsic_cam2world"], dtype=np.float64),
        )
        result = {
            "available": True,
            "status": str(raw.get("status", "unknown")),
            "proposal_only": True,
            "environment_advanced": False,
            "camera": camera,
            "step": selected_step,
            "frame": {
                "model_input": "opencv_camera_xyz_m",
                "returned_world": "LIBERO_world",
            },
            "point_count": len(points),
            "point_cloud_sha256": hashlib.sha256(points.tobytes()).hexdigest(),
            "point_cloud_path": str(point_path),
            "candidate_count": len(candidates),
            "candidates": candidates,
            "latency_s": round(time.perf_counter() - started, 3),
            "execution_warning": (
                "A Contact-GraspNet transform is a grasp proposal, not a Panda EEF "
                "command. Validate gripper-frame calibration, reachability, and current "
                "collision/contact state before executing."
            ),
            "service_evidence": raw.get("evidence", {}),
        }
        (evidence_dir / "result.json").write_text(
            json.dumps(result, indent=2), encoding="utf-8"
        )
        return result
