# Copyright (c) 2026 Zetta Contributors
"""Durable RoboCasa rollout video artifacts.

The live renderer may reuse its framebuffer, and piping long-lived raw frames
directly into an imageio / FFmpeg writer has produced valid MP4 containers with
corrupted pixels.  This module follows the validated harness path instead:

1. persist every camera observation as an independent JPEG;
2. validate decodeability, dimensions, step alignment, and synchronized jumps;
3. encode MP4 files from the immutable JPEG sequence only after validation.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image, ImageChops, ImageStat


STREAM_SPECS = {
    "video.robot0_agentview_left": {
        "pattern": "*_video.robot0_agentview_left.jpg",
        "filename": "robot0_agentview_left.mp4",
    },
    "video.robot0_agentview_right": {
        "pattern": "*_video.robot0_agentview_right.jpg",
        "filename": "robot0_agentview_right.mp4",
    },
    "video.robot0_eye_in_hand": {
        "pattern": "*_video.robot0_eye_in_hand.jpg",
        "filename": "robot0_eye_in_hand.mp4",
    },
}
MULTIVIEW_FILENAME = "multiview_diagnostic.mp4"
STEP_PATTERN = re.compile(r"(?:^|[_-])step(\d+)(?:[_-]|\.|$)", re.IGNORECASE)
SOURCE_DISCONTINUITY_THRESHOLD = 40.0
SOURCE_DISCONTINUITY_STREAMS = 3
SOURCE_SINGLE_STREAM_DISCONTINUITY_THRESHOLD = 55.0


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(dict(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _run_command(command: Sequence[str]) -> str:
    completed = subprocess.run(
        list(command), capture_output=True, check=False, text=True
    )
    if completed.returncode != 0:
        stderr = completed.stderr.strip()
        if len(stderr) > 4000:
            stderr = stderr[-4000:]
        raise RuntimeError(stderr or f"command exited with {completed.returncode}")
    return completed.stdout


def _probe_video(ffprobe: str, path: Path) -> dict[str, Any]:
    payload = json.loads(
        _run_command(
            [
                ffprobe,
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=codec_name,width,height,nb_frames:format=duration,size",
                "-of",
                "json",
                str(path),
            ]
        )
    )
    streams = payload.get("streams", [])
    if not streams:
        raise RuntimeError("ffprobe found no video stream")
    stream = streams[0]
    container = payload.get("format", {})
    return {
        "path": str(path),
        "codec": str(stream.get("codec_name", "unknown")),
        "width": int(stream.get("width", 0)),
        "height": int(stream.get("height", 0)),
        "frames": int(stream.get("nb_frames", 0)),
        "duration_seconds": float(container.get("duration", 0.0)),
        "size_bytes": int(container.get("size", path.stat().st_size)),
    }


def source_frame_validation(image_dir: Path) -> dict[str, Any]:
    """Reject damaged or misaligned persisted camera frames before encoding."""

    stream_frames: dict[str, dict[int, Path]] = {}
    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    dimensions: dict[str, list[int]] = {}
    for name, spec in STREAM_SPECS.items():
        indexed: dict[int, Path] = {}
        for path in sorted(image_dir.glob(str(spec["pattern"]))):
            match = STEP_PATTERN.search(path.name)
            if match is None:
                errors.append(
                    {"kind": "source_frame_step_missing", "stream": name, "path": str(path)}
                )
                continue
            step = int(match.group(1))
            if step in indexed:
                errors.append(
                    {"kind": "source_frame_step_duplicate", "stream": name, "step": step}
                )
            indexed[step] = path
        stream_frames[name] = indexed

    step_sets = [set(frames) for frames in stream_frames.values()]
    aligned_steps = sorted(set.intersection(*step_sets)) if step_sets else []
    if not aligned_steps:
        errors.append(
            {"kind": "source_frames_missing", "message": "no aligned persisted frames"}
        )
    if any(steps != set(aligned_steps) for steps in step_sets):
        errors.append(
            {
                "kind": "source_frame_steps_unaligned",
                "stream_steps": {
                    name: len(frames) for name, frames in stream_frames.items()
                },
                "aligned_steps": len(aligned_steps),
            }
        )
    if aligned_steps and aligned_steps != list(range(aligned_steps[-1] + 1)):
        errors.append(
            {
                "kind": "source_frame_step_sequence_incomplete",
                "first_step": aligned_steps[0],
                "last_step": aligned_steps[-1],
                "count": len(aligned_steps),
            }
        )

    previous: dict[str, Image.Image] = {}
    synchronized_discontinuities: list[dict[str, Any]] = []
    single_stream_discontinuities: list[dict[str, Any]] = []
    for step in aligned_steps:
        transition_scores: dict[str, float] = {}
        current: dict[str, Image.Image] = {}
        for name, frames in stream_frames.items():
            path = frames[step]
            try:
                with Image.open(path) as image:
                    image.load()
                    dimensions.setdefault(name, [image.width, image.height])
                    if dimensions[name] != [image.width, image.height]:
                        errors.append(
                            {
                                "kind": "source_frame_dimensions_changed",
                                "stream": name,
                                "step": step,
                                "expected": dimensions[name],
                                "observed": [image.width, image.height],
                            }
                        )
                    frame = image.convert("RGB").resize((64, 64)).copy()
            except Exception as exc:
                errors.append(
                    {
                        "kind": "source_frame_decode_failed",
                        "stream": name,
                        "step": step,
                        "message": f"{type(exc).__name__}: {exc}",
                    }
                )
                continue
            current[name] = frame
            if name in previous:
                channel_means = ImageStat.Stat(
                    ImageChops.difference(frame, previous[name])
                ).mean
                transition_scores[name] = round(
                    sum(channel_means) / len(channel_means), 3
                )
        if transition_scores:
            for name, score in transition_scores.items():
                if score >= SOURCE_SINGLE_STREAM_DISCONTINUITY_THRESHOLD:
                    single_stream_discontinuities.append(
                        {"step": step, "stream": name, "transition_score": score}
                    )
            discontinuous_streams = sorted(
                name
                for name, score in transition_scores.items()
                if score >= SOURCE_DISCONTINUITY_THRESHOLD
            )
            if len(discontinuous_streams) >= SOURCE_DISCONTINUITY_STREAMS:
                synchronized_discontinuities.append(
                    {
                        "step": step,
                        "streams": discontinuous_streams,
                        "transition_scores": transition_scores,
                    }
                )
        previous.update(current)

    if synchronized_discontinuities:
        errors.append(
            {
                "kind": "source_frame_corruption",
                "message": "synchronized full-frame discontinuities detected across all persisted views",
                "first_step": synchronized_discontinuities[0]["step"],
                "count": len(synchronized_discontinuities),
            }
        )
    if single_stream_discontinuities:
        warnings.append(
            {
                "kind": "source_frame_single_stream_discontinuity_warning",
                "message": "large change detected in one view without synchronized multi-view corruption",
                "first_step": single_stream_discontinuities[0]["step"],
                "count": len(single_stream_discontinuities),
                "streams": sorted(
                    {item["stream"] for item in single_stream_discontinuities}
                ),
            }
        )
    return {
        "status": "valid" if not errors else "invalid",
        "aligned_steps": len(aligned_steps),
        "dimensions": dimensions,
        "discontinuity_threshold": SOURCE_DISCONTINUITY_THRESHOLD,
        "required_synchronized_streams": SOURCE_DISCONTINUITY_STREAMS,
        "single_stream_discontinuity_threshold": SOURCE_SINGLE_STREAM_DISCONTINUITY_THRESHOLD,
        "synchronized_discontinuities": synchronized_discontinuities,
        "single_stream_discontinuities": single_stream_discontinuities,
        "warnings": warnings,
        "errors": errors,
    }


class EpisodeVideoArtifacts:
    """Persist, validate, and encode one episode's three camera streams."""

    def __init__(self, root: Path, *, frame_size: int, frame_rate: float = 20.0):
        self.root = root.expanduser().resolve()
        self.image_dir = self.root / "raw" / "images"
        self.frame_size = int(frame_size)
        self.frame_rate = float(frame_rate)
        self.manifest_path = self.root / "manifest.json"
        self.image_dir.mkdir(parents=True, exist_ok=True)
        existing = [
            path
            for spec in STREAM_SPECS.values()
            for path in self.image_dir.glob(str(spec["pattern"]))
        ]
        if existing:
            raise FileExistsError(
                f"refusing to overwrite {len(existing)} persisted video frames under {self.image_dir}"
            )
        self.frame_count = 0
        self._report: dict[str, Any] | None = None
        self.video_paths = {
            key: str(self.root / str(spec["filename"]))
            for key, spec in STREAM_SPECS.items()
        }

    def append(self, observation: Mapping[str, Any], *, step_index: int) -> None:
        if self._report is not None:
            raise RuntimeError("cannot append frames after video artifacts were finalized")
        if step_index != self.frame_count:
            raise ValueError(
                f"video frame step mismatch: expected {self.frame_count}, got {step_index}"
            )
        snapshots: dict[str, np.ndarray] = {}
        for key in STREAM_SPECS:
            if key not in observation:
                raise KeyError(f"camera observation missing: {key}")
            frame = np.array(observation[key], dtype=np.uint8, order="C", copy=True)
            if frame.shape != (self.frame_size, self.frame_size, 3):
                raise ValueError(
                    f"unexpected {key} frame shape {frame.shape}; expected "
                    f"{(self.frame_size, self.frame_size, 3)}"
                )
            if not frame.flags.c_contiguous or not frame.flags.owndata:
                raise RuntimeError(f"{key} snapshot must own C-contiguous memory")
            snapshots[key] = frame
        for key, frame in snapshots.items():
            suffix = key.removeprefix("video.")
            path = self.image_dir / f"frame_step{step_index:06d}_video.{suffix}.jpg"
            if path.exists():
                raise FileExistsError(f"refusing to overwrite persisted frame {path}")
            temporary = path.with_suffix(path.suffix + ".tmp")
            Image.fromarray(frame, mode="RGB").save(
                temporary, format="JPEG", quality=90, subsampling=0
            )
            temporary.replace(path)
        self.frame_count += 1

    def finalize(self) -> dict[str, Any]:
        if self._report is not None:
            return self._report
        frame_counts = {
            name: len(list(self.image_dir.glob(str(spec["pattern"]))))
            for name, spec in STREAM_SPECS.items()
        }
        report: dict[str, Any] = {
            "schema_version": "zetta-robocasa-video-artifacts-v2",
            "status": "unavailable",
            "frame_rate": self.frame_rate,
            "source_image_directory": str(self.image_dir),
            "source_frame_counts": frame_counts,
            "videos": {},
            "errors": [],
        }
        source_validation = source_frame_validation(self.image_dir)
        report["source_validation"] = source_validation
        if source_validation["status"] != "valid":
            report["status"] = "invalid_source_frames"
            report["errors"].extend(source_validation["errors"])
            _write_json(self.manifest_path, report)
            self._report = report
            return report

        ffmpeg = shutil.which("ffmpeg")
        ffprobe = shutil.which("ffprobe")
        if ffmpeg is None or ffprobe is None:
            report["errors"].append(
                {
                    "kind": "video_tool_missing",
                    "message": "ffmpeg and ffprobe are required for episode video export",
                }
            )
            _write_json(self.manifest_path, report)
            self._report = report
            return report

        frame_rate_text = format(self.frame_rate, "g")
        source_outputs: dict[str, Path] = {}
        for name, spec in STREAM_SPECS.items():
            output_path = self.root / str(spec["filename"])
            temporary_path = output_path.with_suffix(".tmp.mp4")
            if output_path.exists() or temporary_path.exists():
                report["errors"].append(
                    {
                        "kind": "video_output_exists",
                        "stream": name,
                        "path": str(output_path),
                    }
                )
                continue
            try:
                _run_command(
                    [
                        ffmpeg,
                        "-nostdin",
                        "-hide_banner",
                        "-loglevel",
                        "error",
                        "-framerate",
                        frame_rate_text,
                        "-pattern_type",
                        "glob",
                        "-i",
                        str(self.image_dir / str(spec["pattern"])),
                        "-c:v",
                        "libx264",
                        "-preset",
                        "veryfast",
                        "-crf",
                        "18",
                        "-threads",
                        "2",
                        "-pix_fmt",
                        "yuv420p",
                        "-movflags",
                        "+faststart",
                        str(temporary_path),
                    ]
                )
                temporary_path.replace(output_path)
                source_outputs[name] = output_path
            except Exception as exc:
                temporary_path.unlink(missing_ok=True)
                report["errors"].append(
                    {
                        "kind": "stream_encode_failed",
                        "stream": name,
                        "message": f"{type(exc).__name__}: {exc}",
                    }
                )

        expected_names = list(STREAM_SPECS)
        aligned_counts = {frame_counts[name] for name in expected_names}
        multiview_path = self.root / MULTIVIEW_FILENAME
        if set(source_outputs) == set(expected_names) and len(aligned_counts) == 1:
            try:
                command = [
                    ffmpeg,
                    "-nostdin",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                ]
                for name in expected_names:
                    command.extend(["-i", str(source_outputs[name])])
                command.extend(
                    [
                        "-filter_complex",
                        "[0:v][1:v][2:v]hstack=inputs=3[out]",
                        "-map",
                        "[out]",
                        "-c:v",
                        "libx264",
                        "-preset",
                        "veryfast",
                        "-crf",
                        "20",
                        "-threads",
                        "4",
                        "-pix_fmt",
                        "yuv420p",
                        "-movflags",
                        "+faststart",
                        str(multiview_path),
                    ]
                )
                _run_command(command)
            except Exception as exc:
                multiview_path.unlink(missing_ok=True)
                report["errors"].append(
                    {
                        "kind": "multiview_encode_failed",
                        "message": f"{type(exc).__name__}: {exc}",
                    }
                )
        else:
            report["errors"].append(
                {
                    "kind": "multiview_sources_unaligned",
                    "message": "all three camera streams must exist with equal frame counts",
                }
            )

        expected_outputs = dict(source_outputs)
        if multiview_path.is_file():
            expected_outputs["multiview_diagnostic"] = multiview_path
        for name, path in expected_outputs.items():
            try:
                video = _probe_video(ffprobe, path)
                expected_frames = (
                    next(iter(aligned_counts))
                    if name == "multiview_diagnostic"
                    else frame_counts[name]
                )
                if video["frames"] != expected_frames:
                    raise RuntimeError(
                        f"expected {expected_frames} frames, found {video['frames']}"
                    )
                report["videos"][name] = video
            except Exception as exc:
                report["errors"].append(
                    {
                        "kind": "video_validation_failed",
                        "stream": name,
                        "message": f"{type(exc).__name__}: {exc}",
                    }
                )

        report["status"] = (
            "complete"
            if len(report["videos"]) == len(STREAM_SPECS) + 1
            and not report["errors"]
            else "partial"
        )
        _write_json(self.manifest_path, report)
        self._report = report
        return report
