# Copyright (c) 2026 RPent Contributors
"""Deterministic, synchronized multi-camera evidence for offline agents."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from rpent.evolution.jsonio import atomic_write_json, file_sha256

_PRIVILEGED_SUMMARY_MAX_ROWS = 32
_PRIVILEGED_SUMMARY_EXCLUDED_TOKENS = (".position.", ".target_offset.")
_PRIVILEGED_SUMMARY_PREFIXES = (
    "privileged.",
    "command.",
    "episode.",
    "robot.gripper.opening",
    "robot.eef.motion_m",
)
_PRIVILEGED_SUMMARY_CHANGE_KEYS = (
    "privileged.task.goal.progress",
    "privileged.task.success",
    "privileged.task.primary_relation_satisfied",
    "privileged.task.manipulated_object.grasped",
    "privileged.task.manipulated_object.retained",
    "privileged.task.manipulated_object.in_target",
    "privileged.task.stage.name",
    "privileged.contact.gripper.count",
    "command.realization.stalled",
    "episode.terminated",
    "episode.truncated",
)

_DEFAULT_LONG_OVERVIEW_FRAMES = 25
_DEFAULT_SHORT_OVERVIEW_FRAMES = 17
_DEFAULT_EVENT_WINDOW_RADIUS = 8
_DEFAULT_EVENT_WINDOW_STRIDE = 2
_DEFAULT_EVENT_WINDOW_COUNT = 6


def _read_state_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"state timeline row {line_number} is not an object")
            step = value.get("step_index")
            if not isinstance(step, int) or isinstance(step, bool):
                continue
            rows.append(value)
    return sorted(rows, key=lambda row: int(row["step_index"]))


def _summary_state_row(row: dict[str, Any]) -> dict[str, Any]:
    state = row.get("state")
    state = state if isinstance(state, dict) else {}
    values: dict[str, Any] = {}
    for raw_key, value in state.items():
        key = str(raw_key)
        if not any(key.startswith(prefix) for prefix in _PRIVILEGED_SUMMARY_PREFIXES):
            continue
        if any(token in key for token in _PRIVILEGED_SUMMARY_EXCLUDED_TOKENS):
            continue
        if any(token in key.casefold() for token in ("seed", "rng", "path")):
            continue
        if isinstance(value, bool | int | float | str):
            if isinstance(value, float) and not math.isfinite(value):
                continue
            values[key] = value
    values["libero_terminated"] = bool(
        row.get("libero_terminated", values.get("episode.terminated", False))
    )
    values["episode_truncated"] = bool(
        row.get("truncated", values.get("episode.truncated", False))
    )
    return {"step_index": int(row["step_index"]), **dict(sorted(values.items()))}


def build_privileged_state_summary(
    *,
    states_path: str | Path,
    output_path: str | Path,
    max_rows: int = _PRIVILEGED_SUMMARY_MAX_ROWS,
) -> dict[str, Any]:
    """Write a bounded, prompt-safe LIBERO Critic timeline for diagnosis."""

    if not 4 <= int(max_rows) <= 128:
        raise ValueError("privileged state summary max_rows must be in [4, 128]")
    rows = [_summary_state_row(row) for row in _read_state_rows(Path(states_path))]
    if not rows:
        raise ValueError("state timeline is empty")
    selected: set[int] = {0, len(rows) - 1}
    stride = max(1, math.ceil(len(rows) / int(max_rows)))
    selected.update(range(0, len(rows), stride))
    changes: list[list[str]] = [[] for _ in rows]
    changes[0] = ["reset"]
    for index in range(1, len(rows)):
        before = rows[index - 1]
        current = rows[index]
        changes[index] = [
            key
            for key in _PRIVILEGED_SUMMARY_CHANGE_KEYS
            if before.get(key) != current.get(key)
        ]
        if changes[index]:
            selected.add(index)
    if len(selected) > int(max_rows):
        ordered = sorted(selected)
        keep = {
            ordered[int(round(position))]
            for position in np.linspace(0, len(ordered) - 1, int(max_rows))
        }
        keep.update((0, len(rows) - 1))
        selected = keep
    sampled = []
    for index in sorted(selected):
        sampled.append({**rows[index], "changes": changes[index]})
    field_names = sorted(
        {
            key
            for row in rows
            for key in row
            if key != "step_index"
        }
    )
    summary = {
        "schema_version": 1,
        "evidence_kind": "libero_privileged_critic_state_summary",
        "purpose": "bounded simulator-truth comparison for offline diagnosis",
        "privacy": {
            "seed_and_rng": "excluded",
            "private_paths": "excluded",
            "absolute_motion_oracle_coordinates": "excluded",
            "raw_actor_vla_observation": "excluded",
        },
        "step_count": len(rows),
        "sampled_step_count": len(sampled),
        "field_names": field_names,
        "sampling": {
            "max_rows": int(max_rows),
            "includes": ["reset", "terminal", "change_points", "uniform_stride"],
        },
        "steps": sampled,
    }
    path = Path(output_path)
    atomic_write_json(path, summary, overwrite=False)
    return summary


def write_video_metadata(
    *,
    video_dir: str | Path,
    video_paths: dict[str, str],
    visual_evidence: dict[str, Any],
    suite: str,
    task: str,
    task_id: int,
    generation: int,
    logical_id: str,
    attempt_index: int,
    episode_id: str,
    outcome: str,
    status: str,
    seed: int | None = None,
    policy_rng: int | None = None,
) -> dict[str, Any]:
    """Write human-readable per-attempt video mapping without changing names."""

    root = Path(video_dir)
    root.mkdir(parents=True, exist_ok=True)
    cameras = []
    for camera, raw_path in sorted(video_paths.items()):
        path = Path(raw_path)
        cameras.append(
            {
                "camera": _camera_label(camera),
                "file": path.name,
                "path": str(path),
                "exists": path.is_file(),
            }
        )
    payload = {
        "schema_version": 1,
        "episode": {
            "suite": suite,
            "task": task,
            "task_id": int(task_id),
            "generation": int(generation),
            "logical_id": logical_id,
            "attempt_index": int(attempt_index),
            "episode_id": episode_id,
            "outcome": outcome,
            "status": status,
            "seed": seed,
            "policy_rng": policy_rng,
        },
        "frame_alignment": "frame index equals post-step index; frame 0 is reset",
        "videos": cameras,
        "related_artifacts": {
            name: str(value)
            for name, value in visual_evidence.get("artifacts", {}).items()
            if isinstance(value, str)
        },
    }
    index_path = root / "VIDEO_INDEX.json"
    atomic_write_json(index_path, payload, overwrite=False)
    lines = [
        "LIBERO rollout video index",
        "===========================",
        f"Suite: {suite}",
        f"Task: {task} (task_id={int(task_id)})",
        f"Generation: {int(generation)}",
        f"Logical rollout: {logical_id}",
        f"Attempt: {int(attempt_index):03d}",
        f"Episode: {episode_id}",
        f"Outcome: {outcome}",
        f"Status: {status}",
        f"Seed: {seed if seed is not None else 'unavailable'}",
        f"Policy RNG: {policy_rng if policy_rng is not None else 'unavailable'}",
        "",
        "Camera files:",
    ]
    for item in cameras:
        lines.append(f"- {item['camera']}: {item['file']}")
    lines.extend(
        [
            "",
            "Related evidence:",
            "- See ../visual-evidence/visual-evidence-manifest.json for frame alignment.",
            "- See ../visual-evidence/privileged-state-summary.json for bounded Critic state.",
        ]
    )
    readme_path = root / "README.md"
    readme_path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return {
        "index": str(index_path),
        "readme": str(readme_path),
        "artifact_sha256": {
            "video_index": file_sha256(index_path),
            "video_readme": file_sha256(readme_path),
        },
    }


def _camera_label(value: str) -> str:
    return value.rsplit(".", 1)[-1].replace("robot0_", "")


def _read_state_steps(path: Path) -> list[int]:
    values = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            step = row.get("step_index")
            if isinstance(step, int):
                values.append(step)
    return sorted(set(values))


def _detect_event_steps(rows: list[dict[str, Any]]) -> list[int]:
    """Find state transitions worth showing at action-level visual density."""

    summaries = [_summary_state_row(row) for row in rows]
    events: list[int] = []
    for previous, current in zip(summaries, summaries[1:]):
        changed = [
            key
            for key in _PRIVILEGED_SUMMARY_CHANGE_KEYS
            if previous.get(key) != current.get(key)
        ]
        if changed:
            events.append(int(current["step_index"]))
            continue
        for key, threshold in (
            ("robot.gripper.opening", 0.01),
            ("privileged.contact.gripper.count", 1.0),
            ("privileged.task.goal.progress", 0.05),
        ):
            before = previous.get(key)
            after = current.get(key)
            if isinstance(before, (int, float)) and isinstance(after, (int, float)):
                if abs(float(after) - float(before)) >= threshold:
                    events.append(int(current["step_index"]))
                    break
    return sorted(set(events))


def _frame(reader: Any, index: int) -> np.ndarray:
    try:
        value = reader.get_data(index)
    except (IndexError, RuntimeError):
        value = reader.get_data(max(0, reader.count_frames() - 1))
    image = np.asarray(value, dtype=np.uint8)
    if image.ndim != 3 or image.shape[2] not in {3, 4}:
        raise ValueError("camera video frame must be RGB/RGBA")
    return image[:, :, :3]


def _montage(readers: list[tuple[str, Any]], frame_index: int) -> np.ndarray:
    from PIL import Image, ImageDraw

    cells = []
    for camera, reader in readers:
        image = Image.fromarray(_frame(reader, frame_index))
        canvas = Image.new("RGB", (image.width, image.height + 20), "black")
        canvas.paste(image, (0, 20))
        ImageDraw.Draw(canvas).text(
            (4, 3), f"{_camera_label(camera)} step={frame_index}", fill="white"
        )
        cells.append(np.asarray(canvas))
    height = max(cell.shape[0] for cell in cells)
    padded = []
    for cell in cells:
        if cell.shape[0] < height:
            pad = np.zeros((height - cell.shape[0], cell.shape[1], 3), dtype=np.uint8)
            cell = np.concatenate((cell, pad), axis=0)
        padded.append(cell)
    return np.concatenate(padded, axis=1)


def _contact_sheet(
    readers: list[tuple[str, Any]], frame_indexes: list[int], path: Path
) -> None:
    import imageio.v3 as iio

    rows = [_montage(readers, frame_index) for frame_index in frame_indexes]
    iio.imwrite(path, np.concatenate(rows, axis=0), extension=".png")


def build_episode_visual_artifacts(
    *,
    video_paths: dict[str, str],
    states_path: str | Path,
    output_root: str | Path,
    divergence_steps: tuple[int, ...] = (),
    source_fps: int = 20,
    include_privileged_state_summary: bool = False,
    overview_frame_count: int | None = None,
    event_window_radius_steps: int = _DEFAULT_EVENT_WINDOW_RADIUS,
    event_window_stride_steps: int = _DEFAULT_EVENT_WINDOW_STRIDE,
    maximum_event_windows: int = _DEFAULT_EVENT_WINDOW_COUNT,
) -> dict[str, Any]:
    """Create overview, divergence windows and one synchronized short clip."""

    import imageio.v2 as iio

    if len(video_paths) < 2:
        raise ValueError("visual evidence requires at least two synchronized cameras")
    camera_token = "three-camera" if len(video_paths) == 3 else "multi-camera"
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=False)
    states = Path(states_path)
    state_rows = _read_state_rows(states)
    steps = [int(row["step_index"]) for row in state_rows]
    if not steps:
        raise ValueError("state timeline is empty")
    if source_fps < 1:
        raise ValueError("source_fps must be positive")
    if overview_frame_count is None:
        overview_frame_count = (
            _DEFAULT_LONG_OVERVIEW_FRAMES
            if max(steps) >= 400
            else _DEFAULT_SHORT_OVERVIEW_FRAMES
        )
    if not 5 <= int(overview_frame_count) <= 64:
        raise ValueError("overview_frame_count must be in [5, 64]")
    if not 1 <= int(event_window_radius_steps) <= 64:
        raise ValueError("event_window_radius_steps must be in [1, 64]")
    if not 1 <= int(event_window_stride_steps) <= int(event_window_radius_steps):
        raise ValueError("event_window_stride_steps must not exceed radius")
    if not 1 <= int(maximum_event_windows) <= 16:
        raise ValueError("maximum_event_windows must be in [1, 16]")
    readers = [(name, iio.get_reader(path)) for name, path in sorted(video_paths.items())]
    try:
        frame_counts = [reader.count_frames() for _, reader in readers]
        maximum_frame = min(max(0, count - 1) for count in frame_counts)
        maximum_step = min(max(steps), maximum_frame)
        overview_indexes = sorted(
            {
                int(round(value))
                for value in np.linspace(
                    0, maximum_step, int(overview_frame_count)
                )
            }
        )
        overview = root / f"overview-{camera_token}-contact-sheet.png"
        _contact_sheet(readers, overview_indexes, overview)

        centers = sorted(
            {max(0, min(int(step), maximum_step)) for step in divergence_steps}
        )
        if not centers:
            centers = [maximum_step]
        divergence_files = []
        for index, center in enumerate(centers[:3]):
            window = sorted(
                {
                    max(0, min(maximum_step, center + offset))
                    for offset in (-40, -20, 0, 20, 40)
                }
            )
            path = root / f"divergence-{index:02d}-contact-sheet.png"
            _contact_sheet(readers, window, path)
            divergence_files.append(
                {"path": str(path), "center_step": center, "sample_steps": window}
            )

        event_candidates = sorted(
            {int(step) for step in divergence_steps}
            | set(_detect_event_steps(state_rows))
        )
        prioritized = list(dict.fromkeys(int(step) for step in divergence_steps))
        remaining = [step for step in event_candidates if step not in prioritized]
        remaining_slots = max(0, int(maximum_event_windows) - len(prioritized))
        if len(remaining) > remaining_slots > 0:
            remaining = [
                remaining[int(round(index))]
                for index in np.linspace(0, len(remaining) - 1, remaining_slots)
            ]
        prioritized.extend(remaining[:remaining_slots])
        event_centers = [
            max(0, min(int(step), maximum_step))
            for step in prioritized[: int(maximum_event_windows)]
        ]
        event_files = []
        event_offsets = range(
            -int(event_window_radius_steps),
            int(event_window_radius_steps) + 1,
            int(event_window_stride_steps),
        )
        for index, center in enumerate(event_centers):
            window = sorted(
                {
                    max(0, min(maximum_step, center + int(offset)))
                    for offset in event_offsets
                }
            )
            path = root / f"event-{index:02d}-contact-sheet.png"
            _contact_sheet(readers, window, path)
            event_files.append(
                {"path": str(path), "center_step": center, "sample_steps": window}
            )

        clip_center = centers[0]
        clip_start = max(0, clip_center - source_fps * 3)
        clip_end = min(maximum_step, clip_center + source_fps * 3)
        clip_stride = max(1, source_fps // 4)
        clip_steps = list(range(clip_start, clip_end + 1, clip_stride))
        clip = root / f"divergence-{camera_token}-4fps.mp4"
        writer = iio.get_writer(clip, fps=4, codec="libx264")
        try:
            for step in clip_steps:
                writer.append_data(_montage(readers, step))
        finally:
            writer.close()
    finally:
        for _, reader in readers:
            reader.close()

    privileged_summary: Path | None = None
    if include_privileged_state_summary:
        privileged_summary = root / "privileged-state-summary.json"
        build_privileged_state_summary(
            states_path=states,
            output_path=privileged_summary,
        )
    manifest = {
        "schema_version": 1,
        "alignment": "video frame index equals post-step index; frame 0 is reset",
        "cameras": [_camera_label(name) for name in sorted(video_paths)],
        "overview": {
            "path": str(overview),
            "sample_steps": overview_indexes,
        },
        "divergence_contact_sheets": divergence_files,
        "event_contact_sheets": event_files,
        "divergence_clip": {
            "path": str(clip),
            "sample_steps": clip_steps,
            "fps": 4,
        },
        "state_timeline": str(states.resolve()),
    }
    if privileged_summary is not None:
        manifest["privileged_state_summary"] = str(privileged_summary)
    manifest_path = root / "visual-evidence-manifest.json"
    atomic_write_json(manifest_path, manifest, overwrite=False)
    artifacts = {
        "overview_contact_sheet": str(overview),
        "divergence_clip": str(clip),
        "manifest": str(manifest_path),
    }
    if privileged_summary is not None:
        artifacts["privileged_state_summary"] = str(privileged_summary)
    artifacts.update(
        {
            f"divergence_contact_sheet_{index:02d}": value["path"]
            for index, value in enumerate(divergence_files)
        }
    )
    artifacts.update(
        {
            f"event_contact_sheet_{index:02d}": value["path"]
            for index, value in enumerate(event_files)
        }
    )
    return {
        "artifacts": artifacts,
        "artifact_sha256": {name: file_sha256(path) for name, path in artifacts.items()},
        "cameras": manifest["cameras"],
        "overview_steps": overview_indexes,
        "divergence_steps": centers,
        "event_steps": [item["center_step"] for item in event_files],
    }
