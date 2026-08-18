from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from robots.robocasa.video_artifacts import (
    STREAM_SPECS,
    EpisodeVideoArtifacts,
    source_frame_validation,
)


def _frame(value: int, *, size: int = 32) -> np.ndarray:
    return np.full((size, size, 3), value, dtype=np.uint8)


def _observation(step: int) -> dict[str, np.ndarray]:
    return {
        key: _frame(20 + index * 30 + step)
        for index, key in enumerate(STREAM_SPECS)
    }


def test_persisted_frames_are_owned_and_aligned(tmp_path: Path) -> None:
    writer = EpisodeVideoArtifacts(tmp_path, frame_size=32)
    observation = _observation(0)
    writer.append(observation, step_index=0)
    for value in observation.values():
        value[:] = 255
    writer.append(_observation(1), step_index=1)

    report = source_frame_validation(tmp_path / "raw" / "images")

    assert report["status"] == "valid"
    assert report["aligned_steps"] == 2
    first = next((tmp_path / "raw" / "images").glob("*step000000*left.jpg"))
    with Image.open(first) as image:
        assert np.asarray(image).mean() < 30


def test_source_validation_rejects_synchronized_corruption(tmp_path: Path) -> None:
    writer = EpisodeVideoArtifacts(tmp_path, frame_size=32)
    writer.append(_observation(0), step_index=0)
    writer.append({key: _frame(255) for key in STREAM_SPECS}, step_index=1)

    report = source_frame_validation(tmp_path / "raw" / "images")

    assert report["status"] == "invalid"
    assert report["errors"][-1]["kind"] == "source_frame_corruption"


def test_finalize_encodes_readable_videos_when_ffmpeg_is_available(
    tmp_path: Path,
) -> None:
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        pytest.skip("system ffmpeg and ffprobe are not installed")
    writer = EpisodeVideoArtifacts(tmp_path, frame_size=32, frame_rate=10.0)
    for step in range(3):
        writer.append(_observation(step), step_index=step)

    report = writer.finalize()

    assert report["status"] == "complete"
    assert report["source_validation"]["aligned_steps"] == 3
    assert len(report["videos"]) == len(STREAM_SPECS) + 1
    assert all(Path(path).is_file() for path in writer.video_paths.values())
