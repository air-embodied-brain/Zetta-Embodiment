# Copyright (c) 2026 Zetta Contributors
"""Proposal-only runtime for the RoboCasa tool surface.

The runtime deliberately has no simulator client and no ``step`` operation.
Learned tools are reached through endpoint URLs read only from environment
variables; local tools are deterministic geometry, verification, or action
proposal functions.  Privileged simulator evidence is allowed only when the
caller opts in and is always declared in the returned audit metadata.
"""

from __future__ import annotations

import fnmatch
import hashlib
import heapq
import json
import math
import os
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from robots.robocasa.action_contract import canonical_action, serializable_action
from robots.robocasa.dishwasher_critic import evaluate_slide_dishwasher_rack
from robots.robocasa.slide_dishwasher_program import (
    BASE_ASSIST_TOOL,
    CONTACT_PUSH_TOOL,
    GUARDED_SUFFIX_TOOL,
    base_assisted_terminal_action,
    guard_terminal_suffix,
)
from robots.robocasa.tool_catalog import (
    DEFAULT_ROBOCASA_TOOL_CATALOG,
    ToolCatalog,
    ToolSpec,
)

HttpTransport = Callable[[str, bytes, float, Mapping[str, str]], Mapping[str, Any]]


def _digest(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class ToolRuntimeError(RuntimeError):
    """Base class for safe, classified tool runtime failures."""

    failure_class = "tool_runtime"
    retryable = False

    def safe_dict(self) -> dict[str, Any]:
        return {
            "status": "error",
            "failure_class": self.failure_class,
            "retryable": self.retryable,
        }


class ToolPolicyError(ToolRuntimeError):
    failure_class = "tool_policy"


class ToolContractError(ToolRuntimeError):
    failure_class = "tool_contract"


class ToolServiceUnavailable(ToolRuntimeError):
    failure_class = "tool_service_unavailable"
    retryable = True


class ToolServiceRejected(ToolRuntimeError):
    failure_class = "tool_service_rejected"


@dataclass(frozen=True, slots=True)
class InvocationPolicy:
    """Per-invocation authorization; privileged input is opt-in and audited."""

    allow: frozenset[str] | None = None
    deny: frozenset[str] = frozenset()
    allow_privileged: bool = False
    timeout_s: float = 60.0

    def __post_init__(self) -> None:
        if self.timeout_s <= 0.0 or not math.isfinite(self.timeout_s):
            raise ValueError("timeout_s must be finite and positive")


def _default_http_transport(
    url: str,
    body: bytes,
    timeout_s: float,
    headers: Mapping[str, str],
) -> Mapping[str, Any]:
    request = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers=dict(headers),
    )
    with urllib.request.urlopen(request, timeout=timeout_s) as response:  # noqa: S310
        value = json.loads(response.read().decode("utf-8"))
    if not isinstance(value, Mapping):
        raise ToolContractError("service response must be an object")
    return value


_SENSITIVE_PARTS = ("api_key", "apikey", "authorization", "password", "secret", "token")
_LOCATION_PARTS = ("endpoint", "service_url", "url")


def _sanitize(value: Any, known_secrets: frozenset[str]) -> Any:
    if isinstance(value, Mapping):
        cleaned: dict[str, Any] = {}
        for key, item in value.items():
            lowered = str(key).lower()
            if any(part in lowered for part in (*_SENSITIVE_PARTS, *_LOCATION_PARTS)):
                continue
            cleaned[str(key)] = _sanitize(item, known_secrets)
        return cleaned
    if isinstance(value, (list, tuple)):
        return [_sanitize(item, known_secrets) for item in value]
    if isinstance(value, str):
        if value in known_secrets or "://" in value:
            return "[redacted]"
        redacted = value
        for secret in known_secrets:
            if secret and secret in redacted:
                redacted = redacted.replace(secret, "[redacted]")
        return redacted
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _get_path(value: Any, path: str) -> tuple[bool, Any]:
    if isinstance(value, Mapping) and path in value:
        return True, value[path]
    current: Any = value
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return False, None
        current = current[part]
    return True, current


def _matching_privileged_fields(
    payload: Mapping[str, Any], patterns: Sequence[str]
) -> tuple[str, ...]:
    paths: set[str] = set()

    def walk(current: Any, prefix: str = "") -> None:
        if prefix:
            paths.add(prefix)
        if not isinstance(current, Mapping):
            return
        for key, item in current.items():
            name = f"{prefix}.{key}" if prefix else str(key)
            walk(item, name)

    walk(payload)
    matches = {
        path
        for path in paths
        for pattern in patterns
        if fnmatch.fnmatchcase(path, pattern)
    }
    roots = {
        path
        for path in matches
        if not any(
            path != other and path.startswith(other + ".") for other in matches
        )
    }
    return tuple(sorted(roots))


def _vector(value: Any, size: int, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (size,) or not np.isfinite(array).all():
        raise ToolContractError(f"{name} must be a finite length-{size} vector")
    return array


def _positive(payload: Mapping[str, Any], name: str, default: float) -> float:
    value = float(payload.get(name, default))
    if not math.isfinite(value) or value <= 0.0:
        raise ToolContractError(f"{name} must be finite and positive")
    return value


def _clip_norm(vector: np.ndarray, maximum: float) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    return vector if norm <= maximum or norm < 1e-12 else vector * maximum / norm


def _canonical_action(
    position: Sequence[float] = (0.0, 0.0, 0.0),
    rotation: Sequence[float] = (0.0, 0.0, 0.0),
    gripper_close: float = 0.0,
    base_motion: Sequence[float] = (0.0, 0.0, 0.0, 0.0),
) -> dict[str, list[float]]:
    return {
        "end_effector_position": np.clip(position, -1.0, 1.0).astype(float).tolist(),
        "end_effector_rotation": np.clip(rotation, -1.0, 1.0).astype(float).tolist(),
        "gripper_close": [float(np.clip(gripper_close, 0.0, 1.0))],
        "base_motion": np.clip(base_motion, -1.0, 1.0).astype(float).tolist(),
        "control_mode": [0.0],
    }


def _action_is_bounded(value: Any) -> bool:
    if isinstance(value, Mapping):
        return all(_action_is_bounded(item) for item in value.values())
    if isinstance(value, (list, tuple, np.ndarray)):
        return all(_action_is_bounded(item) for item in value)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return math.isfinite(float(value)) and -1.0 <= float(value) <= 1.0
    return False


def _quaternion_axis_angle(current: np.ndarray, target: np.ndarray) -> np.ndarray:
    def normalized(value: np.ndarray) -> np.ndarray:
        norm = float(np.linalg.norm(value))
        if norm < 1e-12:
            raise ToolContractError("quaternion norm must be non-zero")
        return value / norm

    x1, y1, z1, w1 = normalized(current)
    x2, y2, z2, w2 = normalized(target)
    inverse = np.array([-x1, -y1, -z1, w1])
    ix, iy, iz, iw = inverse
    error = np.array(
        [
            iw * x2 + ix * w2 + iy * z2 - iz * y2,
            iw * y2 - ix * z2 + iy * w2 + iz * x2,
            iw * z2 + ix * y2 - iy * x2 + iz * w2,
            iw * w2 - ix * x2 - iy * y2 - iz * z2,
        ]
    )
    if error[3] < 0.0:
        error = -error
    xyz_norm = float(np.linalg.norm(error[:3]))
    if xyz_norm < 1e-12:
        return np.zeros(3)
    angle = 2.0 * math.atan2(xyz_norm, float(np.clip(error[3], -1.0, 1.0)))
    return error[:3] * angle / xyz_norm


class ToolRuntime:
    """Invoke the catalog without ever taking ownership of the environment."""

    def __init__(
        self,
        catalog: ToolCatalog = DEFAULT_ROBOCASA_TOOL_CATALOG,
        *,
        environ: Mapping[str, str] | None = None,
        http_transport: HttpTransport | None = None,
    ) -> None:
        self.catalog = catalog
        self._environ = dict(os.environ if environ is None else environ)
        self._transport = http_transport or _default_http_transport
        self._handlers: dict[str, Callable[[Mapping[str, Any]], dict[str, Any]]] = {
            "robocasa.observation.view_driver_state": self._view_driver_state,
            "robocasa.camera.view_meta": self._view_meta,
            "robocasa.geometry.back_project": self._back_project,
            "robocasa.control.move_to": self._move_to,
            "robocasa.base.move": self._base_move,
            "robocasa.control.move_pose": self._move_pose,
            "robocasa.control.rotate_wrist": lambda value: self._rotate(
                value, 2, "delta_yaw_rad"
            ),
            "robocasa.control.rotate_pitch": lambda value: self._rotate(
                value, 1, "delta_pitch_rad"
            ),
            "robocasa.gripper.set": self._gripper,
            "robocasa.gripper.release": lambda value: self._gripper(
                {**value, "gripper_close": 0.0}
            ),
            "robocasa.verify.state": self._verify,
            "robocasa.cap.servo_pose": self._cap_servo,
            "robocasa.cap.contact_retaining_motion": self._cap_contact,
            "robocasa.critic.temporal_engagement": self._temporal_critic,
            "robocasa.motion.base_se2_astar": self._base_astar,
            "robocasa.motion.base_se2_servo": self._base_servo,
            CONTACT_PUSH_TOOL: self._slide_contact_push,
            GUARDED_SUFFIX_TOOL: self._slide_guarded_suffix,
            BASE_ASSIST_TOOL: self._slide_base_assist,
        }

    def invoke(
        self,
        name: str,
        payload: Mapping[str, Any],
        *,
        policy: InvocationPolicy | None = None,
    ) -> dict[str, Any]:
        if not isinstance(payload, Mapping):
            raise ToolContractError("tool payload must be an object")
        policy = policy or InvocationPolicy()
        allow = None if policy.allow is None else set(policy.allow)
        selected = self.catalog.select(allow=allow, deny=policy.deny)
        selected_names = {item.name for item in selected}
        if name not in selected_names:
            self.catalog.get(name)
            raise ToolPolicyError("tool is denied by the invocation policy")
        spec = self.catalog.get(name)
        privileged_fields = _matching_privileged_fields(payload, spec.privileged_fields)
        if privileged_fields and not policy.allow_privileged:
            raise ToolPolicyError(
                "privileged tool input requires explicit authorization"
            )
        privileged_audit = {
            "authorized": bool(policy.allow_privileged),
            "used": bool(privileged_fields),
            "fields": list(privileged_fields),
            "payload_digest": _digest(payload),
        }
        if spec.local:
            try:
                handler = self._handlers[name]
            except KeyError as exc:
                raise ToolContractError("local tool has no runtime handler") from exc
            result = handler(payload)
        else:
            result = self._invoke_service(spec, payload, policy.timeout_s)
        if not isinstance(result, Mapping):
            raise ToolContractError("tool result must be an object")
        result = dict(result)
        if "action" in result and not _action_is_bounded(result["action"]):
            raise ToolContractError("tool returned an out-of-bounds action")
        if "actions" in result and not _action_is_bounded(result["actions"]):
            raise ToolContractError("tool returned out-of-bounds actions")
        result.update(
            {
                "tool": name,
                "proposal_only": spec.proposal_only,
                "environment_write": False,
                "privileged_audit": privileged_audit,
            }
        )
        secrets = frozenset(
            value
            for key, value in self._environ.items()
            if value and any(part in key.lower() for part in _SENSITIVE_PARTS)
        )
        return dict(_sanitize(result, secrets))

    def _invoke_service(
        self, spec: ToolSpec, payload: Mapping[str, Any], timeout_s: float
    ) -> dict[str, Any]:
        endpoint = self._environ.get(str(spec.endpoint_env), "").strip()
        if not endpoint:
            raise ToolServiceUnavailable("service endpoint is not configured")
        body = json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")
        url = endpoint.rstrip("/") + spec.service_path
        try:
            value = self._transport(
                url,
                body,
                timeout_s,
                {"Content-Type": "application/json"},
            )
        except ToolRuntimeError:
            raise
        except urllib.error.HTTPError as exc:
            if exc.code == 429 or 500 <= exc.code < 600:
                raise ToolServiceUnavailable("service request failed") from None
            raise ToolServiceRejected("service rejected the request") from None
        except (OSError, TimeoutError, urllib.error.URLError):
            raise ToolServiceUnavailable("service transport failed") from None
        except Exception:
            raise ToolServiceUnavailable("service invocation failed") from None
        if not isinstance(value, Mapping):
            raise ToolContractError("service response must be an object")
        return dict(value)

    @staticmethod
    def _slide_state(payload: Mapping[str, Any]) -> Mapping[str, Any]:
        observation = payload.get("observation")
        state = observation.get("state") if isinstance(observation, Mapping) else None
        if not isinstance(state, Mapping):
            raise ToolContractError("live observation.state is required")
        return state

    @staticmethod
    def _slide_contact_push(payload: Mapping[str, Any]) -> dict[str, Any]:
        actions = payload.get("pending_vla_actions")
        if not isinstance(actions, Sequence) or isinstance(actions, (str, bytes)):
            raise ToolContractError("contact push requires a pending VLA suffix")
        normalized = [serializable_action(canonical_action(action)) for action in actions]
        if not normalized:
            raise ToolContractError("contact push suffix must not be empty")
        return {"actions": normalized, "preserved": True}

    @classmethod
    def _slide_guarded_suffix(cls, payload: Mapping[str, Any]) -> dict[str, Any]:
        actions = payload.get("pending_vla_actions")
        if not isinstance(actions, Sequence) or isinstance(actions, (str, bytes)):
            raise ToolContractError("guarded suffix requires pending VLA actions")
        return {
            "actions": guard_terminal_suffix(
                actions,
                state=cls._slide_state(payload),
                minimum_projection=float(payload.get("minimum_projection", 0.05)),
            ),
            "reverse_projection_clamped_only": True,
        }

    @classmethod
    def _slide_base_assist(cls, payload: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "action": base_assisted_terminal_action(
                state=cls._slide_state(payload),
                base_command=float(payload.get("base_command", 1.0)),
                arm_retract_command=float(payload.get("arm_retract_command", 0.05)),
            ),
            "positive_world_advance_required": True,
        }

    @staticmethod
    def _view_driver_state(payload: Mapping[str, Any]) -> dict[str, Any]:
        observation = payload.get("observation")
        if not isinstance(observation, Mapping):
            raise ToolContractError("observation must be an object")
        state = observation.get("state", {})
        if not isinstance(state, Mapping):
            raise ToolContractError("observation.state must be an object")
        images = observation.get("image_paths", observation.get("images", {}))
        if not isinstance(images, Mapping):
            images = {}
        return {
            "state": dict(state),
            "images": dict(images),
            "task": str(payload.get("task", "")),
            "step_index": int(payload.get("step_index", 0)),
            "digest": _digest({"state": state, "images": images}),
        }

    @staticmethod
    def _view_meta(payload: Mapping[str, Any]) -> dict[str, Any]:
        metadata = payload.get("camera_meta", {})
        if not isinstance(metadata, Mapping):
            raise ToolContractError("camera_meta must be an object")
        width = payload.get("width", metadata.get("width"))
        height = payload.get("height", metadata.get("height"))
        if width is None or height is None:
            shape = payload.get("image_shape")
            if isinstance(shape, Sequence) and len(shape) >= 2:
                height, width = shape[:2]
        if width is None or height is None:
            raise ToolContractError("width and height or image_shape are required")
        calibrated = "K" in metadata or "intrinsic_K" in metadata
        return {
            "status": "ready" if calibrated else "image_only",
            "camera": str(payload.get("camera", "unknown")),
            "width": int(width),
            "height": int(height),
            "camera_meta": dict(metadata),
            "evidence": {"intrinsics_available": calibrated},
        }

    @staticmethod
    def _back_project(payload: Mapping[str, Any]) -> dict[str, Any]:
        depth = np.asarray(payload.get("depth"), dtype=np.float64)
        if depth.ndim == 3 and depth.shape[-1] == 1:
            depth = depth[..., 0]
        if depth.ndim != 2 or not np.isfinite(depth).all():
            raise ToolContractError("depth must be a finite HxW array")
        intrinsics = payload.get("intrinsics")
        if not isinstance(intrinsics, Mapping):
            raise ToolContractError("intrinsics must be an object")
        matrix = np.asarray(
            intrinsics.get("K", intrinsics.get("intrinsic_K")), dtype=np.float64
        )
        if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
            raise ToolContractError("intrinsics must contain a finite 3x3 matrix")
        mask_value = payload.get("mask")
        mask = (
            np.ones(depth.shape, dtype=bool)
            if mask_value is None
            else np.asarray(mask_value, dtype=bool)
        )
        if mask.shape != depth.shape:
            raise ToolContractError("mask shape must match depth")
        mask &= depth > 0.0
        rows, columns = np.where(mask)
        stride = max(1, int(payload.get("stride", 1)))
        rows, columns = rows[::stride], columns[::stride]
        maximum = max(1, int(payload.get("max_points", 4096)))
        if len(rows) > maximum:
            keep = np.linspace(0, len(rows) - 1, maximum, dtype=np.int64)
            rows, columns = rows[keep], columns[keep]
        z = depth[rows, columns]
        points = np.stack(
            [
                (columns - matrix[0, 2]) * z / matrix[0, 0],
                (rows - matrix[1, 2]) * z / matrix[1, 1],
                z,
            ],
            axis=1,
        )
        frame = "camera"
        extrinsic_value = payload.get("extrinsic_cam2world")
        if extrinsic_value is not None:
            extrinsic = np.asarray(extrinsic_value, dtype=np.float64)
            if extrinsic.shape != (4, 4) or not np.isfinite(extrinsic).all():
                raise ToolContractError("extrinsic_cam2world must be finite 4x4")
            homogeneous = np.c_[points, np.ones(len(points))]
            points = (homogeneous @ extrinsic.T)[:, :3]
            frame = "world"
        centroid = np.median(points, axis=0).tolist() if len(points) else [None] * 3
        return {
            "points": points.astype(np.float32).tolist(),
            "pixels": np.c_[rows, columns].astype(int).tolist(),
            "centroid": centroid,
            "count": int(len(points)),
            "frame": frame,
            "evidence": {
                "input_digest": _digest(
                    {"shape": depth.shape, "pixels": int(mask.sum())}
                )
            },
        }

    @staticmethod
    def _move_to(payload: Mapping[str, Any]) -> dict[str, Any]:
        current = _vector(payload.get("current_position"), 3, "current_position")
        target = _vector(payload.get("target_position"), 3, "target_position")
        scale = _positive(payload, "position_action_scale_m", 0.35)
        maximum = _positive(payload, "max_translation_m", 0.03)
        tolerance = _positive(payload, "position_tolerance_m", 0.01)
        error = target - current
        bounded = _clip_norm(error, maximum)
        action = _canonical_action(
            position=np.clip(bounded / scale, -1.0, 1.0),
            gripper_close=float(payload.get("gripper_close", 0.0)),
        )
        return {
            "action": action,
            "reached": float(np.linalg.norm(error)) <= tolerance,
            "position_error_m": float(np.linalg.norm(error)),
            "evidence": {"action_digest": _digest(action)},
        }

    @staticmethod
    def _base_move(payload: Mapping[str, Any]) -> dict[str, Any]:
        requested = _vector(payload.get("base_motion"), 4, "base_motion")
        applied = np.clip(requested, -1.0, 1.0)
        action = _canonical_action(
            gripper_close=float(payload.get("gripper_close", 0.0)),
            base_motion=applied,
        )
        return {
            "action": action,
            "requested_base_motion": requested.tolist(),
            "applied_base_motion": applied.tolist(),
            "clipped": not np.array_equal(requested, applied),
            "evidence": {"frame": "robot_base", "action_digest": _digest(action)},
        }

    @staticmethod
    def _move_pose(payload: Mapping[str, Any]) -> dict[str, Any]:
        current = payload.get("current_pose")
        target = payload.get("target_pose")
        if not isinstance(current, Mapping) or not isinstance(target, Mapping):
            raise ToolContractError("current_pose and target_pose must be objects")
        current_position = _vector(current.get("position"), 3, "current_pose.position")
        target_position = _vector(target.get("position"), 3, "target_pose.position")
        current_quaternion = _vector(
            current.get("quaternion_xyzw"), 4, "current quaternion"
        )
        target_quaternion = _vector(
            target.get("quaternion_xyzw"), 4, "target quaternion"
        )
        position_error = target_position - current_position
        rotation_error = _quaternion_axis_angle(current_quaternion, target_quaternion)
        bounded_position = _clip_norm(
            position_error, _positive(payload, "max_translation_m", 0.03)
        )
        bounded_rotation = _clip_norm(
            rotation_error, _positive(payload, "max_rotation_rad", 0.12)
        )
        action = _canonical_action(
            position=bounded_position
            / _positive(payload, "position_action_scale_m", 0.35),
            rotation=bounded_rotation
            / _positive(payload, "rotation_action_scale_rad", 0.35),
            gripper_close=float(payload.get("gripper_close", 0.0)),
        )
        return {
            "action": action,
            "reached": (
                float(np.linalg.norm(position_error))
                <= float(payload.get("position_tolerance_m", 0.01))
                and float(np.linalg.norm(rotation_error))
                <= float(payload.get("rotation_tolerance_rad", 0.05))
            ),
            "position_error_m": float(np.linalg.norm(position_error)),
            "rotation_error_rad": float(np.linalg.norm(rotation_error)),
            "evidence": {"action_digest": _digest(action)},
        }

    @staticmethod
    def _rotate(payload: Mapping[str, Any], axis: int, field: str) -> dict[str, Any]:
        requested = float(payload.get(field, 0.0))
        maximum = _positive(payload, "max_rotation_rad", 0.12)
        scale = _positive(payload, "rotation_action_scale_rad", 0.35)
        bounded = float(np.clip(requested, -maximum, maximum))
        rotation = np.zeros(3)
        rotation[axis] = bounded / scale
        action = _canonical_action(
            rotation=rotation,
            gripper_close=float(payload.get("gripper_close", 0.0)),
        )
        return {
            "action": action,
            "requested_rotation_rad": requested,
            "bounded_rotation_rad": bounded,
            "evidence": {"axis": axis, "action_digest": _digest(action)},
        }

    @staticmethod
    def _gripper(payload: Mapping[str, Any]) -> dict[str, Any]:
        close = float(np.clip(float(payload.get("gripper_close", 0.0)), 0.0, 1.0))
        action = _canonical_action(gripper_close=close)
        return {
            "action": action,
            "gripper_close": close,
            "evidence": {"action_digest": _digest(action)},
        }

    @staticmethod
    def _verify(payload: Mapping[str, Any]) -> dict[str, Any]:
        state = payload.get("state")
        criteria = payload.get("criteria")
        if not isinstance(state, Mapping) or not isinstance(criteria, Mapping):
            raise ToolContractError("state and criteria must be objects")
        groups: dict[str, list[str]] = {
            "satisfied": [],
            "unsatisfied": [],
            "unknown": [],
        }
        details: dict[str, Any] = {}
        for raw_key, requirement in criteria.items():
            key = str(raw_key)
            present, value = _get_path(state, key)
            outcome = "unknown"
            if present and value is not None:
                if isinstance(requirement, bool):
                    outcome = "satisfied" if value is requirement else "unsatisfied"
                elif isinstance(requirement, (int, float)) and not isinstance(
                    requirement, bool
                ):
                    if isinstance(value, (int, float)) and not isinstance(value, bool):
                        outcome = (
                            "satisfied"
                            if math.isclose(
                                float(value), float(requirement), abs_tol=1e-9
                            )
                            else "unsatisfied"
                        )
                elif (
                    isinstance(requirement, Mapping)
                    and isinstance(value, (int, float))
                    and not isinstance(value, bool)
                ):
                    checks = []
                    if "min" in requirement:
                        checks.append(float(value) >= float(requirement["min"]))
                    if "max" in requirement:
                        checks.append(float(value) <= float(requirement["max"]))
                    if "target" in requirement:
                        checks.append(
                            abs(float(value) - float(requirement["target"]))
                            <= float(requirement.get("tolerance", 0.0))
                        )
                    if checks:
                        outcome = "satisfied" if all(checks) else "unsatisfied"
            groups[outcome].append(key)
            details[key] = {"status": outcome, "requirement": requirement}
        status = (
            "unsatisfied"
            if groups["unsatisfied"]
            else "unknown"
            if groups["unknown"]
            else "satisfied"
        )
        return {
            "status": status,
            "satisfied": groups["satisfied"],
            "failed": groups["unsatisfied"],
            "unknown": groups["unknown"],
            "evidence": {"details": details, "state_digest": _digest(state)},
        }

    @staticmethod
    def _cap_servo(payload: Mapping[str, Any]) -> dict[str, Any]:
        current_position = _vector(
            payload.get("current_position"), 3, "current_position"
        )
        target_position = _vector(payload.get("target_position"), 3, "target_position")
        current_quaternion = _vector(
            payload.get("current_quaternion"), 4, "current_quaternion"
        )
        target_quaternion = _vector(
            payload.get("target_quaternion"), 4, "target_quaternion"
        )
        position_error = target_position - current_position
        rotation_error = _quaternion_axis_angle(current_quaternion, target_quaternion)
        command_position = _clip_norm(
            position_error * float(payload.get("position_gain", 1.0)),
            _positive(payload, "maximum_position_command", 0.1),
        )
        command_rotation = _clip_norm(
            rotation_error * float(payload.get("orientation_gain", 1.0)),
            _positive(payload, "maximum_rotation_command", 0.1),
        )
        action = _canonical_action(
            command_position, command_rotation, float(bool(payload.get("close", False)))
        )
        return {
            "action": action,
            "target_position": target_position.tolist(),
            "target_quaternion": target_quaternion.tolist(),
            "position_error": position_error.tolist(),
            "orientation_error": rotation_error.tolist(),
            "position_error_norm": float(np.linalg.norm(position_error)),
            "orientation_error_norm": float(np.linalg.norm(rotation_error)),
            "evidence": {"action_digest": _digest(action)},
        }

    @staticmethod
    def _cap_contact(payload: Mapping[str, Any]) -> dict[str, Any]:
        current = _vector(payload.get("current_position"), 3, "current_position")
        anchor = _vector(payload.get("live_anchor_position"), 3, "live_anchor_position")
        direction = _vector(payload.get("motion_direction"), 3, "motion_direction")
        norm = float(np.linalg.norm(direction))
        if norm < 1e-12:
            raise ToolContractError("motion_direction must be non-zero")
        target = anchor + direction / norm * float(
            payload.get("feedforward_speed", 0.01)
        )
        delegated = dict(payload)
        delegated.update(
            {
                "current_position": current,
                "target_position": target,
                "position_gain": float(payload.get("hold_gain", 1.0)),
            }
        )
        return ToolRuntime._cap_servo(delegated)

    @staticmethod
    def _temporal_critic(payload: Mapping[str, Any]) -> dict[str, Any]:
        if payload.get("task") == "SlideDishwasherRack" or payload.get("mode") == (
            "slide_dishwasher_rack"
        ):
            return evaluate_slide_dishwasher_rack(payload)
        history = payload.get("history")
        if not isinstance(history, Sequence) or isinstance(history, (str, bytes)):
            raise ToolContractError("history must be an array")
        window = max(2, int(payload.get("window", 4)))
        recent = [item for item in history[-window:] if isinstance(item, Mapping)]
        progress_key = str(payload.get("progress_key", "progress"))
        contact_key = str(payload.get("contact_key", "contact"))
        progress_values = []
        contacts = []
        for item in recent:
            present, value = _get_path(item, progress_key)
            if present and isinstance(value, (int, float)):
                progress_values.append(float(value))
            present, value = _get_path(item, contact_key)
            if present:
                contacts.append(bool(value))
        minimum_delta = float(payload.get("minimum_progress_delta", 1e-3))
        stalled = (
            len(progress_values) >= 2
            and progress_values[-1] - progress_values[0] < minimum_delta
        )
        contact_lost = bool(contacts) and not contacts[-1]
        triggered = stalled or contact_lost
        return {
            "status": "proposal" if triggered else "clear",
            "triggered": triggered,
            "proposal": "reobserve_and_recover" if triggered else None,
            "reasons": [
                name
                for name, active in (
                    ("contact_lost", contact_lost),
                    ("progress_stalled", stalled),
                )
                if active
            ],
            "evidence": {"history_digest": _digest(history), "window": len(recent)},
        }

    @staticmethod
    def _base_astar(payload: Mapping[str, Any]) -> dict[str, Any]:
        start = _vector(payload.get("start_world"), 2, "start_world")
        goal_value = payload.get("goal")
        if isinstance(goal_value, Mapping):
            goal_value = goal_value.get("position_world")
        goal = _vector(goal_value, 2, "goal")
        obstacles = payload.get("obstacles", [])
        if not isinstance(obstacles, Sequence):
            raise ToolContractError("obstacles must be an array")
        resolution = _positive(payload, "resolution_m", 0.08)
        footprint = _positive(payload, "footprint_radius_m", 0.25)
        clearance = float(payload.get("clearance_m", 0.03))
        max_cells = max(1, int(payload.get("max_cells", 12000)))

        def cell(point: np.ndarray) -> tuple[int, int]:
            return tuple(np.rint(point / resolution).astype(int))  # type: ignore[return-value]

        def world(node: tuple[int, int]) -> np.ndarray:
            return np.asarray(node, dtype=np.float64) * resolution

        def blocked(node: tuple[int, int]) -> bool:
            point = world(node)
            for item in obstacles:
                if not isinstance(item, Mapping):
                    continue
                position = item.get("position_world")
                if not isinstance(position, Sequence) or len(position) < 2:
                    continue
                center = np.asarray(position[:2], dtype=np.float64)
                radius = float(item.get("radius_xy_m", 0.0))
                if (
                    float(np.linalg.norm(point - center))
                    <= radius + footprint + clearance
                ):
                    return True
            return False

        initial, target = cell(start), cell(goal)
        if blocked(target):
            return {"status": "goal_blocked", "waypoints_world": [], "expanded": 0}
        queue: list[tuple[float, tuple[int, int]]] = [(0.0, initial)]
        cost = {initial: 0.0}
        parent: dict[tuple[int, int], tuple[int, int]] = {}
        expanded = 0
        while queue and expanded < max_cells:
            _, current = heapq.heappop(queue)
            expanded += 1
            if current == target:
                nodes = [current]
                while nodes[-1] in parent:
                    nodes.append(parent[nodes[-1]])
                nodes.reverse()
                points = [world(item).tolist() for item in nodes[1:]]
                return {
                    "status": "planned",
                    "waypoints_world": points,
                    "expanded": expanded,
                    "resolution_m": resolution,
                }
            for dx, dy in (
                (-1, -1),
                (-1, 0),
                (-1, 1),
                (0, -1),
                (0, 1),
                (1, -1),
                (1, 0),
                (1, 1),
            ):
                nxt = (current[0] + dx, current[1] + dy)
                if blocked(nxt):
                    continue
                candidate_cost = cost[current] + math.hypot(dx, dy)
                if candidate_cost >= cost.get(nxt, float("inf")):
                    continue
                cost[nxt] = candidate_cost
                parent[nxt] = current
                heuristic = float(np.linalg.norm(world(nxt) - goal)) / resolution
                heapq.heappush(queue, (candidate_cost + heuristic, nxt))
        return {"status": "no_path", "waypoints_world": [], "expanded": expanded}

    @staticmethod
    def _base_servo(payload: Mapping[str, Any]) -> dict[str, Any]:
        handle = _vector(payload.get("handle_xy"), 2, "handle_xy")
        goal_value = payload.get("goal", {})
        if not isinstance(goal_value, Mapping):
            raise ToolContractError("goal must be an object")
        target = _vector(
            goal_value.get("target_handle_xy_m", (0.62, 0.0)), 2, "target_handle_xy_m"
        )
        error = handle - target
        yaw_error = math.atan2(float(handle[1]), float(handle[0])) - float(
            goal_value.get("target_relative_yaw_rad", 0.0)
        )
        yaw_error = (yaw_error + math.pi) % (2.0 * math.pi) - math.pi
        converged = float(np.linalg.norm(error)) <= float(
            payload.get("translation_tolerance_m", 0.06)
        ) and abs(yaw_error) <= float(payload.get("yaw_tolerance_rad", 0.08))
        base_motion = (
            [0.0] * 4
            if converged
            else [
                float(
                    np.clip(
                        float(payload.get("translation_gain", 1.0)) * error[0],
                        -float(payload.get("maximum_translation_command", 0.8)),
                        float(payload.get("maximum_translation_command", 0.8)),
                    )
                ),
                float(
                    np.clip(
                        float(payload.get("translation_gain", 1.0)) * error[1],
                        -float(payload.get("maximum_translation_command", 0.8)),
                        float(payload.get("maximum_translation_command", 0.8)),
                    )
                ),
                float(
                    np.clip(
                        float(payload.get("yaw_gain", 0.8)) * yaw_error,
                        -float(payload.get("maximum_yaw_command", 0.8)),
                        float(payload.get("maximum_yaw_command", 0.8)),
                    )
                ),
                0.0,
            ]
        )
        action = _canonical_action(
            gripper_close=float(payload.get("gripper_close", 0.0)),
            base_motion=base_motion,
        )
        return {
            "action": action,
            "base_motion": base_motion,
            "translation_error_m": float(np.linalg.norm(error)),
            "yaw_error_rad": yaw_error,
            "converged": converged,
            "evidence": {"action_digest": _digest(action)},
        }


__all__ = [
    "InvocationPolicy",
    "ToolContractError",
    "ToolPolicyError",
    "ToolRuntime",
    "ToolRuntimeError",
    "ToolServiceRejected",
    "ToolServiceUnavailable",
]
