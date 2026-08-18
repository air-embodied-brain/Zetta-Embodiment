#!/usr/bin/env python3
"""Run base-runtime pytest while appending an isolated dependency overlay.

Appending (rather than prepending) avoids incomplete overlay packages shadowing
the simulator runtime's pinned OmegaConf, imageio, prompt-toolkit, and pytest.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pytest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overlay", type=Path, required=True)
    args, pytest_args = parser.parse_known_args()
    overlay = args.overlay.resolve()
    if not overlay.is_dir():
        raise SystemExit(f"dependency overlay does not exist: {overlay}")
    sys.path.append(str(overlay))
    return int(pytest.main(pytest_args or ["-q"]))


if __name__ == "__main__":
    raise SystemExit(main())
