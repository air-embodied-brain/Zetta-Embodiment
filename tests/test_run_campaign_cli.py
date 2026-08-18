# Copyright (c) 2026 RPent Contributors
from __future__ import annotations

from pathlib import Path

import pytest

from scripts.evolution.run_campaign import _parse_args


def _required_args() -> list[str]:
    return [
        "--manifest",
        "manifest.json",
        "--root",
        "campaign",
        "--queue-root",
        "queue",
        "--tool-catalog",
        "tool-catalog.json",
        "--workers",
        "libero-gpu1",
    ]


def test_worker_command_consumes_nested_worker_options() -> None:
    args = _parse_args(
        [
            *_required_args(),
            "--worker-command",
            "python",
            "-m",
            "rpent.evolution.cli",
            "worker",
            "--queue-root",
            "{queue_root}",
            "--host",
            "{host}",
            "--poll-s",
            "2",
            "--concurrency",
            "1",
        ]
    )

    assert args.queue_root == Path("queue")
    assert args.worker_command == [
        "python",
        "-m",
        "rpent.evolution.cli",
        "worker",
        "--queue-root",
        "{queue_root}",
        "--host",
        "{host}",
        "--poll-s",
        "2",
        "--concurrency",
        "1",
    ]


def test_worker_command_must_not_be_empty() -> None:
    with pytest.raises(SystemExit):
        _parse_args([*_required_args(), "--worker-command"])
