"""Thread-safe in-memory state for dashboard live runs."""
from __future__ import annotations

import threading
from pathlib import Path
from typing import Any


class State:
    """Thread-safe dashboard state for one run."""

    def __init__(
        self,
        *,
        run_id: str,
        name: str,
        suite: str,
        task: int,
        seed: int,
        output_dir: str,
        video_path: str,
    ) -> None:
        self.run_id = run_id
        self.name = name
        self.suite = suite
        self.task = task
        self.seed = seed
        self.output_dir = Path(output_dir)
        self.video_path = Path(video_path)

        self._lock = threading.Lock()
        self._state = "running"
        self._terminated = False
        self._usage = {"in": 0, "out": 0, "tool_calls": 0}
        self._events: list[dict[str, Any]] = []
        self._timeline: list[dict[str, Any]] = []
        self._frame_png: bytes | None = None
        self._frame_cam_png: bytes | None = None
        self._frame_idx = -1

    def on_event(self, event: dict[str, Any]) -> None:
        with self._lock:
            self._events.append(event)

    def on_usage(self, *, inp: int, out: int, tool_calls: int) -> None:
        with self._lock:
            self._usage = {"in": int(inp), "out": int(out), "tool_calls": int(tool_calls)}

    def on_tool_result(self, name: str, result: Any) -> None:
        if not isinstance(result, dict):
            return
        image_path = result.get("overlay_path") or result.get("image_path")
        image_cam_path = result.get("image_cam_path")
        self._update_frame(
            step=result.get("step"),
            image=Path(image_path).read_bytes() if image_path else None,
            image_cam=Path(image_cam_path).read_bytes() if image_cam_path else None,
        )
        log = result.get("log")
        if not isinstance(log, dict):
            return
        command = log.get("command")
        if not isinstance(command, dict) or command.get("action") != name:
            return
        try:
            step = int(result["step"])
        except Exception:
            return
        terminated = bool(result.get("libero_terminated"))
        item = {
            "step": step,
            "action": str(command.get("action", name)),
            "args": {k: v for k, v in command.items() if k != "action"},
            "result": log.get("result"),
            "elapsed_s": log.get("elapsed_s"),
            "terminated": terminated,
            "has_action_video": (
                self.output_dir
                / "action_videos"
                / f"step_{step:02d}_{command.get('action', name)}.mp4"
            ).exists(),
        }
        with self._lock:
            self._timeline.append(item)
            self._terminated = self._terminated or terminated

    def _update_frame(
        self,
        *,
        step: Any,
        image: bytes | None = None,
        image_cam: bytes | None = None,
    ) -> None:
        if image is None and image_cam is None:
            return
        try:
            frame_idx = int(step)
        except Exception:
            frame_idx = None
        with self._lock:
            if frame_idx is not None:
                self._frame_idx = frame_idx
            if image is not None:
                self._frame_png = bytes(image)
            if image_cam is not None:
                self._frame_cam_png = bytes(image_cam)

    def mark_done(self, terminated: bool | None = None) -> None:
        with self._lock:
            self._state = "done"
            if terminated is None:
                terminated = any(item.get("terminated") for item in self._timeline)
            self._terminated = bool(terminated)

    def events_since(self, since: int) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._events[since:])

    def frame(self, kind: str) -> bytes | None:
        with self._lock:
            if kind == "camera":
                return self._frame_cam_png
            return self._frame_png

    def action_video_path(self, step: int) -> Path | None:
        with self._lock:
            for item in self._timeline:
                if int(item.get("step", -1)) != int(step):
                    continue
                video_path = (
                    self.output_dir
                    / "action_videos"
                    / f"step_{int(step):02d}_{item.get('action', '')}.mp4"
                )
                return video_path if video_path.exists() else None
        return None

    def has_video(self) -> bool:
        with self._lock:
            return self._state == "done" and self.video_path.exists()

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "state": self._state,
                "terminated": self._terminated,
                "usage": dict(self._usage),
                "has_video": self._state == "done" and self.video_path.exists(),
                "frame_idx": self._frame_idx,
                "n_steps": len(self._timeline),
            }

    def run_info(self) -> dict[str, Any]:
        with self._lock:
            return {
                "id": self.run_id,
                "name": self.name,
                "suite": self.suite,
                "task": self.task,
                "seed": self.seed,
                "state": self._state,
                "n_steps": len(self._timeline),
            }

    def run_detail(self) -> dict[str, Any]:
        with self._lock:
            return {
                "state": self._state,
                "terminated": self._terminated,
                "suite": self.suite,
                "name": self.name,
                "task": self.task,
                "seed": self.seed,
                "usage": dict(self._usage),
                "timeline": list(self._timeline),
                "has_video": self._state == "done" and self.video_path.exists(),
                "frame_idx": self._frame_idx,
            }
