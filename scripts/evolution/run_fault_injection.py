#!/usr/bin/env python3
# Copyright (c) 2026 Zetta Contributors
"""Run campaign-stage fault injection without modifying the campaign runtime."""

from __future__ import annotations

import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from zetta.evolution.fault_injection import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
