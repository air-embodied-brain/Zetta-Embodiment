# Copyright (c) 2026 RPent Contributors
"""Persistent RoboCasa environment with server-side chunk execution.

The implementation is clean-room and uses RoboCasa's public Gym action
mapping.  One process owns one persistent environment and isolated renderer;
many such processes may share a GPU.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import importlib.util
import json
import os
import threading
import traceback
from contextlib import contextmanager, nullcontext
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import numpy as np

from robots.robocasa.action_contract import (
    ActionScale,
    canonical_action,
    serializable_action,
)
from robots.robocasa.gpu_gate import GpuOperationGate
from robots.robocasa.operation_protocol import IdempotentWriteRegistry
from robots.robocasa.privileged_state import extract_privileged_state
from robots.robocasa.slide_dishwasher_program import SlideDishwasherProgramState
from robots.robocasa.stable_renderer import (
    close_persistent_rgb_renderer,
    install_persistent_rgb_renderer,
)
from robots.robocasa.video_artifacts import EpisodeVideoArtifacts
from rpent.evolution.critic import TemporalCritic
from rpent.evolution.models import CriticPredicate, CriticRule

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

CAMERA_KEYS = (
    "video.robot0_agentview_left",
    "video.robot0_agentview_right",
    "video.robot0_eye_in_hand",
)
MAXIMUM_REQUEST_BYTES = 8 * 1024 * 1024


class SimulationHealthError(RuntimeError):
    """The simulator produced non-finite state and must not keep executing."""


class BoundedThreadingHTTPServer(ThreadingHTTPServer):
    """Reject excess requests before creating another handler thread."""

    daemon_threads = True

    def __init__(self, server_address: tuple[str, int], handler: Any, *, limit: int):
        super().__init__(server_address, handler)
        self._admission = threading.BoundedSemaphore(limit)

    def process_request(self, request: Any, client_address: Any) -> None:
        if not self._admission.acquire(blocking=False):
            body = b'{"error":"queue_full","retry_after_s":0.25}'
            response = (
                b"HTTP/1.1 503 Service Unavailable\r\n"
                b"Content-Type: application/json\r\n"
                + f"Content-Length: {len(body)}\r\n".encode()
                + b"Retry-After: 1\r\nConnection: close\r\n\r\n"
                + body
            )
            try:
                request.sendall(response)
            finally:
                self.shutdown_request(request)
            return
        try:
            super().process_request(request, client_address)
        except Exception:
            self._admission.release()
            raise

    def process_request_thread(self, request: Any, client_address: Any) -> None:
        try:
            super().process_request_thread(request, client_address)
        finally:
            self._admission.release()


def isolated_renderer_status() -> dict[str, Any]:
    """Inspect the installed robosuite renderer without creating a context."""

    try:
        package = importlib.util.find_spec("robosuite")
        locations = tuple(package.submodule_search_locations or ()) if package else ()
        candidates = [Path(root) / "utils" / "binding_utils.py" for root in locations]
        path = next(candidate for candidate in candidates if candidate.is_file())
        source = path.read_text(encoding="utf-8")
    except Exception as exc:
        return {
            "ready": False,
            "reason": f"renderer_inspection:{type(exc).__name__}",
            "detail": str(exc)[:240],
        }
    markers = {
        "dedicated_renderer": "mujoco.Renderer(" in source,
        "scene_option_forwarded": "scene_option=" in source and ".vopt" in source,
    }
    return {
        "ready": all(markers.values()),
        "markers": markers,
        "source_name": path.name,
    }


@contextmanager
def _cold_reset_guard(path: str | None):
    """Serialize renderer creation per GPU; POSIX flock is crash-recoverable."""

    if not path:
        yield
        return
    lock_path = Path(path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as stream:
        if os.name == "posix":
            import fcntl

            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            if os.name == "posix":
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _sha_array(value: Any) -> str:
    array = np.asarray(value)
    return hashlib.sha256(array.tobytes()).hexdigest()


def _json_scalar(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        if value.ndim >= 2:
            return None
        return value.astype(float).tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {
            str(key): converted
            for key, item in value.items()
            if (converted := _json_scalar(item)) is not None
        }
    if isinstance(value, (list, tuple)):
        converted = [_json_scalar(item) for item in value]
        return [item for item in converted if item is not None]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _owned_rgb_frame(
    value: Any,
    *,
    expected_size: int | None = None,
) -> np.ndarray:
    """Detach one RGB image from a renderer-owned framebuffer.

    RoboSuite / MuJoCo camera observations may be views backed by a renderer
    buffer that is reused by the next render.  Passing such a view directly to
    imageio lets the encoder observe overwritten bytes.  Always materialize an
    owned C-contiguous snapshot before JPEG or video encoding.
    """

    frame = np.array(value, dtype=np.uint8, order="C", copy=True)
    if frame.ndim != 3 or frame.shape[-1] != 3:
        raise ValueError(f"expected an HxWx3 RGB frame, got {frame.shape}")
    if expected_size is not None and frame.shape[:2] != (
        expected_size,
        expected_size,
    ):
        raise ValueError(
            "unexpected camera frame size: "
            f"{frame.shape[:2]}, expected {(expected_size, expected_size)}"
        )
    if not frame.flags.c_contiguous or not frame.flags.owndata:
        raise RuntimeError("camera frame snapshot must own C-contiguous memory")
    return frame


def _encode_image(value: Any) -> str:
    import imageio.v3 as iio

    image = _owned_rgb_frame(value)
    payload = iio.imwrite("<bytes>", image, extension=".jpg", quality=80)
    return "data:image/jpeg;base64," + base64.b64encode(payload).decode("ascii")


def _observation_payload(
    observation: dict[str, Any], *, include_images: bool
) -> dict[str, Any]:
    scalar = _json_scalar(observation)
    images: dict[str, str] = {}
    hashes: dict[str, str] = {}
    for key, value in observation.items():
        array = np.asarray(value) if not isinstance(value, str) else None
        if array is not None and array.ndim >= 2:
            hashes[key] = _sha_array(array)
            if include_images and key in CAMERA_KEYS:
                images[key] = _encode_image(array)
    return {"state": scalar, "images": images, "image_sha256": hashes}


def _critic_from_payload(payload: list[dict[str, Any]]) -> tuple[CriticRule, ...]:
    rules = []
    for item in payload:
        value = dict(item)
        value["evidence_ids"] = tuple(value.get("evidence_ids", ()))
        value["activation_conditions"] = tuple(
            CriticPredicate(**condition)
            for condition in value.get("activation_conditions", ())
        )
        rules.append(CriticRule(**value))
    return tuple(rules)


def _unwrap_environment(environment: Any) -> Any:
    current = environment
    visited: set[int] = set()
    while id(current) not in visited:
        visited.add(id(current))
        nested = getattr(current, "env", None)
        if nested is None or nested is current:
            break
        current = nested
    return current


def _official_task_success(environment: Any, info: dict[str, Any]) -> bool:
    """Read RoboCasa's official task API, never a geometric proxy.

    The public Gym wrapper computes sparse reward and ``info['success']`` from
    ``env._check_success()`` but its ``terminated`` value is inherited from the
    simulator horizon. Calling the official predicate directly makes that
    distinction explicit and lets the adapter stop at the first true success.
    """

    task = _unwrap_environment(environment)
    check = getattr(task, "_check_success", None)
    if callable(check):
        return bool(np.asarray(check()).any())
    value = info.get("success", False)
    return bool(np.asarray(value).any())


def _is_depth_observation(name: str) -> bool:
    """Return whether an observation is a renderer depth buffer.

    RoboCasa/robosuite represents pixels with no geometry hit as positive
    infinity.  That is a valid depth sentinel, not evidence that MuJoCo's
    dynamic state diverged.
    """

    normalized = name.lower()
    return normalized.endswith("_depth") or normalized.endswith(".depth")


def _assert_simulation_health(environment: Any, observation: dict[str, Any]) -> None:
    """Fail closed on unhealthy observations or MuJoCo dynamic state.

    Positive infinity is permitted only in renderer depth buffers, where it
    denotes a background ray with no geometry hit.  NaN, negative infinity,
    non-depth infinities, and non-finite simulator state remain hard failures.
    """

    for name, value in observation.items():
        try:
            array = np.asarray(value)
        except Exception:
            continue
        if not np.issubdtype(array.dtype, np.number):
            continue
        if _is_depth_observation(name):
            if np.isnan(array).any() or np.isneginf(array).any():
                raise SimulationHealthError(f"invalid depth observation: {name}")
            continue
        if not np.isfinite(array).all():
            raise SimulationHealthError(f"non-finite observation: {name}")
    task = _unwrap_environment(environment)
    simulator = getattr(task, "sim", None)
    data = getattr(simulator, "data", None)
    for name in ("qpos", "qvel", "ctrl"):
        value = getattr(data, name, None)
        if value is not None and not np.isfinite(np.asarray(value)).all():
            raise SimulationHealthError(f"non-finite simulator state: {name}")


class RoboCasaSession:
    def __init__(
        self,
        *,
        camera_size: int,
        max_steps: int,
        cold_reset_lock: str | None = None,
        operation_gate_root: str | None = None,
        operation_gate_gpu: str | None = None,
        operation_gate_slots: int = 1,
        require_isolated_renderer: bool = True,
    ) -> None:
        self.camera_size = camera_size
        self.max_steps = max_steps
        self.env: Any = None
        self.identity: tuple[str, str] | None = None
        self.observation: dict[str, Any] | None = None
        self.info: dict[str, Any] = {}
        self.terminated = False
        self.truncated = False
        self.step_index = 0
        self.reward = 0.0
        self.official_success = False
        self.success_latched = False
        self.success_first_step: int | None = None
        self.action_scale = ActionScale()
        self.bundle_sha256: str | None = None
        self.critic: TemporalCritic | None = None
        self.critic_fingerprint: str | None = None
        self.task_program: SlideDishwasherProgramState | None = None
        self.video_artifacts: EpisodeVideoArtifacts | None = None
        self.video_paths: dict[str, str] = {}
        self.video_manifest: dict[str, Any] | None = None
        self.cold_reset_lock = cold_reset_lock
        self.operation_gate = (
            GpuOperationGate(
                operation_gate_root,
                gpu_id=str(operation_gate_gpu),
                slots=operation_gate_slots,
            )
            if operation_gate_root is not None and operation_gate_gpu is not None
            else None
        )
        self.renderer_status = isolated_renderer_status()
        if require_isolated_renderer and not self.renderer_status.get("ready"):
            raise RuntimeError(
                "installed robosuite does not expose an isolated MuJoCo renderer"
            )
        self.lock = threading.RLock()

    def _operation_guard(self):
        """Bound expensive simulator work without coupling slots to processes."""

        if self.operation_gate is None:
            return nullcontext()
        return self.operation_gate.acquire()

    def _ensure_environment(self, task: str, split: str) -> None:
        identity = (task, split)
        if self.env is not None and identity == self.identity:
            return
        self.close_environment()
        import gymnasium as gym
        import robocasa  # noqa: F401
        import robocasa.wrappers.gym_wrapper  # noqa: F401

        persistent_renderer = install_persistent_rgb_renderer()
        self.renderer_status["rgb_renderer"] = persistent_renderer

        with _cold_reset_guard(self.cold_reset_lock):
            self.env = gym.make(
                f"robocasa/{task}",
                split=split,
                enable_render=True,
                camera_widths=self.camera_size,
                camera_heights=self.camera_size,
                camera_depths=True,
            )
        self.identity = identity

    def reset(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self.lock:
            task = str(payload["task"])
            split = str(payload.get("split", "target"))
            seed = int(payload["seed"])
            self._ensure_environment(task, split)
            with self._operation_guard():
                result = self.env.reset(seed=seed)
            observation, info = result if isinstance(result, tuple) else (result, {})
            self.observation = dict(observation)
            self.info = dict(info)
            self.terminated = False
            self.truncated = False
            self.step_index = 0
            self.reward = 0.0
            self.official_success = _official_task_success(self.env, self.info)
            self.success_latched = self.official_success
            self.success_first_step = 0 if self.success_latched else None
            _assert_simulation_health(self.env, self.observation)
            self.action_scale = ActionScale.from_payload(payload.get("action_scale"))
            self.bundle_sha256 = payload.get("bundle_sha256")
            self.critic = None
            self.critic_fingerprint = None
            self.task_program = (
                SlideDishwasherProgramState()
                if task == "SlideDishwasherRack"
                and bool(payload.get("enable_task_program", False))
                else None
            )
            if self.task_program is not None:
                state = self._current_observation(include_images=False)["state"]
                self.task_program.reset(state)
            self._open_video_writers(payload.get("video_dir"))
            self._append_video_frames()
            return self.snapshot(include_images=True)

    def _open_video_writers(self, video_dir: str | None) -> None:
        self._close_video_writers()
        self.video_paths.clear()
        self.video_manifest = None
        if not video_dir:
            return
        self.video_artifacts = EpisodeVideoArtifacts(
            Path(video_dir), frame_size=self.camera_size, frame_rate=20.0
        )
        self.video_paths.update(self.video_artifacts.video_paths)

    def _append_video_frames(self) -> None:
        if self.observation is None or self.video_artifacts is None:
            return
        self.video_artifacts.append(self.observation, step_index=self.step_index)

    def _close_video_writers(self) -> None:
        if self.video_artifacts is None:
            return
        self.video_manifest = self.video_artifacts.finalize()
        self.video_artifacts = None

    def _configure_critic(self, rules_payload: list[dict[str, Any]]) -> None:
        fingerprint = hashlib.sha256(
            json.dumps(rules_payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        if self.critic is None:
            self.critic = TemporalCritic(_critic_from_payload(rules_payload))
            self.critic_fingerprint = fingerprint
        elif fingerprint != self.critic_fingerprint:
            raise ValueError("critic rules cannot change within an episode")

    def _current_observation(self, *, include_images: bool) -> dict[str, Any]:
        if self.observation is None:
            raise RuntimeError("environment has no observation")
        result = _observation_payload(self.observation, include_images=include_images)
        try:
            privileged = extract_privileged_state(self.env, self.observation)
        except Exception as exc:
            privileged = {
                "privileged.error": type(exc).__name__,
                "privileged.source": "live_mujoco_simulator",
            }
        result["state"].update(privileged)
        return result

    def execute_chunk(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self.lock, self._operation_guard():
            if self.env is None or self.observation is None:
                raise RuntimeError("call /reset before /execute_chunk")
            if self.terminated or self.truncated:
                raise RuntimeError("episode already terminated")
            actions = payload.get("actions")
            if not isinstance(actions, list) or not actions:
                raise ValueError("actions must be a non-empty array")
            rules_payload = payload.get("critic_rules") or []
            if not isinstance(rules_payload, list):
                raise ValueError("critic_rules must be an array")
            self._configure_critic(rules_payload)
            interrupt = bool(payload.get("interrupt_on_proposal", True))
            capture_events = bool(payload.get("capture_event_images", True))
            enable_task_program = bool(payload.get("enable_task_program", False))
            if enable_task_program != (self.task_program is not None):
                raise ValueError("task-program activation cannot change within episode")
            steps: list[dict[str, Any]] = []
            proposals: list[dict[str, Any]] = []
            event_images: list[dict[str, Any]] = []
            for action_index, requested_action in enumerate(actions):
                action = canonical_action(requested_action, scale=self.action_scale)
                applied_action = serializable_action(action)
                state_before = self._current_observation(include_images=False)["state"]
                pre_action_proposals = (
                    self.task_program.before_action(
                        applied_action,
                        state=state_before,
                        step_index=self.step_index + 1,
                    )
                    if self.task_program is not None
                    else []
                )
                if pre_action_proposals and interrupt:
                    proposals.extend(pre_action_proposals)
                    if capture_events:
                        event_images.append(
                            {
                                "step_index": self.step_index,
                                "phase": "pre_action",
                                "observation": self._current_observation(
                                    include_images=True
                                ),
                            }
                        )
                    break
                result = self.env.step(action)
                observation, reward, terminated, truncated, info = result
                self.observation = dict(observation)
                self.info = dict(info)
                self.step_index += 1
                self.reward = float(np.asarray(reward).max())
                self.terminated = bool(np.asarray(terminated).any())
                self.truncated = (
                    bool(np.asarray(truncated).any())
                    or self.step_index >= self.max_steps
                )
                _assert_simulation_health(self.env, self.observation)
                self.official_success = _official_task_success(self.env, self.info)
                if self.official_success and not self.success_latched:
                    self.success_latched = True
                    self.success_first_step = self.step_index
                self._append_video_frames()
                scalar_observation = self._current_observation(include_images=False)[
                    "state"
                ]
                learned_proposals = (
                    self.critic.evaluate(scalar_observation, step_index=self.step_index)
                    if self.critic
                    else []
                )
                task_proposals = (
                    self.task_program.after_action(
                        state=scalar_observation,
                        step_index=self.step_index,
                        at_chunk_boundary=action_index == len(actions) - 1,
                    )
                    if self.task_program is not None
                    else []
                )
                step_proposals = [*learned_proposals, *task_proposals]
                proposals.extend(step_proposals)
                step_record = {
                    "step_index": self.step_index,
                    "action_sha256": hashlib.sha256(
                        json.dumps(
                            applied_action, sort_keys=True, separators=(",", ":")
                        ).encode()
                    ).hexdigest(),
                    "applied_action": applied_action,
                    "observation_sha256": hashlib.sha256(
                        json.dumps(
                            scalar_observation, sort_keys=True, default=str
                        ).encode()
                    ).hexdigest(),
                    "state": scalar_observation,
                    "reward": self.reward,
                    "official_success": self.official_success,
                    "success_latched": self.success_latched,
                    "terminated": self.terminated,
                    "truncated": self.truncated,
                    "proposal_rule_ids": [item["rule_id"] for item in step_proposals],
                }
                steps.append(step_record)
                if step_proposals and capture_events:
                    event_images.append(
                        {
                            "step_index": self.step_index,
                            "observation": self._current_observation(
                                include_images=True
                            ),
                        }
                    )
                if (
                    self.authoritative_success
                    or self.terminated
                    or self.truncated
                    or (interrupt and step_proposals)
                ):
                    break
            return {
                "executed_horizon": len(steps),
                "steps": steps,
                "critic_proposals": proposals,
                "event_images": event_images,
                "observation": self._current_observation(include_images=False),
                "terminated": self.terminated,
                "truncated": self.truncated,
                "official_success": self.official_success,
                "success_latched": self.success_latched,
                "success_first_step": self.success_first_step,
                "authoritative_success": self.authoritative_success,
                "video_paths": dict(self.video_paths),
                "environment_write_owner": "robocasa_session",
                "task_program_enabled": self.task_program is not None,
                "critic_rule_count": len(rules_payload),
            }

    def snapshot(self, *, include_images: bool) -> dict[str, Any]:
        if self.observation is None:
            raise RuntimeError("environment has not been reset")
        observation = self._current_observation(include_images=include_images)
        return {
            "observation": observation,
            "task": self.identity[0] if self.identity else None,
            "split": self.identity[1] if self.identity else None,
            "step_index": self.step_index,
            "terminated": self.terminated,
            "truncated": self.truncated,
            "reward": self.reward,
            "official_success": self.official_success,
            "success_latched": self.success_latched,
            "success_first_step": self.success_first_step,
            "authoritative_success": self.authoritative_success,
            "bundle_sha256": self.bundle_sha256,
            "task_program_enabled": self.task_program is not None,
            "critic_rule_count": (
                len(self.critic.rules) if self.critic is not None else 0
            ),
            "video_paths": dict(self.video_paths),
            "renderer": dict(self.renderer_status),
        }

    def finalize_episode_artifacts(self) -> dict[str, Any]:
        """Flush video encoders while keeping the environment process warm."""

        with self.lock:
            if self.env is None:
                raise RuntimeError("environment has not been reset")
            self._close_video_writers()
            return {
                "finalized": True,
                "video_paths": dict(self.video_paths),
                "video_manifest": self.video_manifest,
                "step_index": self.step_index,
                "authoritative_success": self.authoritative_success,
            }

    @property
    def authoritative_success(self) -> bool:
        """Use sticky RoboCasa official success, not reward or termination alone."""

        return self.success_latched

    def close_environment(self) -> None:
        self._close_video_writers()
        if self.env is not None:
            try:
                simulator = getattr(self.env.unwrapped, "sim", None)
                if simulator is not None:
                    close_persistent_rgb_renderer(simulator)
            except Exception:
                pass
            self.env.close()
        self.env = None
        self.identity = None
        self.observation = None


def serve(
    *,
    host: str,
    port: int,
    camera_size: int,
    max_steps: int,
    cold_reset_lock: str | None,
    operation_gate_root: str | None,
    operation_gate_gpu: str | None,
    operation_gate_slots: int,
    require_isolated_renderer: bool,
    maximum_inflight_requests: int = 2,
) -> None:
    if maximum_inflight_requests < 1:
        raise ValueError("maximum_inflight_requests must be positive")
    session = RoboCasaSession(
        camera_size=camera_size,
        max_steps=max_steps,
        cold_reset_lock=cold_reset_lock,
        operation_gate_root=operation_gate_root,
        operation_gate_gpu=operation_gate_gpu,
        operation_gate_slots=operation_gate_slots,
        require_isolated_renderer=require_isolated_renderer,
    )
    write_registry = IdempotentWriteRegistry()

    class Handler(BaseHTTPRequestHandler):
        def _send(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
            body = json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")
            self.send_response(int(status))
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _payload(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length", "0"))
            if length < 0 or length > MAXIMUM_REQUEST_BYTES:
                raise ValueError("request body exceeds the configured limit")
            value = (
                json.loads(self.rfile.read(length).decode("utf-8")) if length else {}
            )
            if not isinstance(value, dict):
                raise ValueError("request body must be an object")
            return value

        def do_GET(self) -> None:
            if self.path == "/health":
                self._send(
                    HTTPStatus.OK,
                    {
                        "status": "healthy",
                        "persistent": True,
                        "renderer": session.renderer_status,
                        "gpu_visible": os.environ.get("CUDA_VISIBLE_DEVICES"),
                        "egl_device": os.environ.get("MUJOCO_EGL_DEVICE_ID"),
                        "write_protocol": write_registry.state,
                    },
                )
            elif self.path == "/schema":
                self._send(
                    HTTPStatus.OK,
                    {
                        "reset": "POST /reset",
                        "observation": "POST /observation",
                        "execute_chunk": "POST /execute_chunk",
                        "finalize_episode": "POST /finalize_episode",
                        "release": "POST /release",
                        "critic_semantics": "proposal_only",
                        "environment_writer": "session_only",
                        "action_contract": "named_robocasa_gym_mapping",
                        "flat_action_size": 12,
                        "camera_keys": list(CAMERA_KEYS),
                        "write_protocol": "binding+episode+sequence+request_digest",
                        "maximum_inflight_requests": maximum_inflight_requests,
                    },
                )
            else:
                self._send(HTTPStatus.NOT_FOUND, {"error": "not_found"})

        def do_POST(self) -> None:
            try:
                payload = self._payload()

                def marker() -> tuple[Any, ...]:
                    return (
                        id(session.env),
                        session.identity,
                        session.step_index,
                        session.observation is not None,
                    )

                if self.path == "/reset":
                    terminal = write_registry.execute(
                        self.path,
                        payload,
                        lambda: session.reset(payload),
                        side_effect_marker=marker,
                    )
                    self._send(terminal.status, terminal.payload)
                    return
                elif self.path == "/observation":
                    result = session.snapshot(
                        include_images=bool(payload.get("include_images", True))
                    )
                elif self.path == "/execute_chunk":
                    terminal = write_registry.execute(
                        self.path,
                        payload,
                        lambda: session.execute_chunk(payload),
                        side_effect_marker=marker,
                    )
                    self._send(terminal.status, terminal.payload)
                    return
                elif self.path == "/finalize_episode":
                    terminal = write_registry.execute(
                        self.path,
                        payload,
                        session.finalize_episode_artifacts,
                        side_effect_marker=marker,
                    )
                    self._send(terminal.status, terminal.payload)
                    return
                elif self.path == "/release":
                    terminal = write_registry.execute(
                        self.path,
                        payload,
                        session.finalize_episode_artifacts,
                        side_effect_marker=marker,
                        release_binding=True,
                    )
                    self._send(terminal.status, terminal.payload)
                    return
                elif self.path == "/close":
                    terminal = write_registry.execute(
                        self.path,
                        payload,
                        lambda: session.close_environment() or {"closed": True},
                        side_effect_marker=marker,
                    )
                    self._send(terminal.status, terminal.payload)
                    return
                else:
                    self._send(HTTPStatus.NOT_FOUND, {"error": "not_found"})
                    return
                self._send(HTTPStatus.OK, result)
            except ValueError as exc:
                self._send(
                    HTTPStatus.CONFLICT,
                    {"error": type(exc).__name__, "detail": str(exc)},
                )
            except Exception as exc:
                traceback.print_exc()
                self._send(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    {"error": type(exc).__name__, "detail": str(exc)},
                )

        def log_message(self, format: str, *args: Any) -> None:
            print(f"{self.client_address[0]} - {format % args}", flush=True)

    server = BoundedThreadingHTTPServer(
        (host, port), Handler, limit=maximum_inflight_requests
    )
    try:
        server.serve_forever()
    finally:
        session.close_environment()


def main() -> None:
    parser = argparse.ArgumentParser(description="Persistent RoboCasa chunk server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18800)
    parser.add_argument("--camera-size", type=int, default=256)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--cold-reset-lock")
    parser.add_argument("--operation-gate-root")
    parser.add_argument("--operation-gate-gpu")
    parser.add_argument("--operation-gate-slots", type=int, default=1)
    parser.add_argument("--maximum-inflight-requests", type=int, default=2)
    parser.add_argument(
        "--allow-shared-renderer",
        action="store_true",
        help="diagnostic only; production requires an isolated MuJoCo renderer",
    )
    args = parser.parse_args()
    serve(
        host=args.host,
        port=args.port,
        camera_size=args.camera_size,
        max_steps=args.max_steps,
        cold_reset_lock=args.cold_reset_lock,
        operation_gate_root=args.operation_gate_root,
        operation_gate_gpu=args.operation_gate_gpu,
        operation_gate_slots=args.operation_gate_slots,
        require_isolated_renderer=not args.allow_shared_renderer,
        maximum_inflight_requests=args.maximum_inflight_requests,
    )


if __name__ == "__main__":
    main()
