# Copyright (c) 2026 Zetta Contributors
"""Test worker using the production framing code without importing Isaac."""

from pathlib import Path

__path__.append(str(Path(__file__).resolve().parents[3] / "zetta/envs/geniesim"))
