"""Env registry: maps env name to its ``get_env_spec`` / ``get_toolkit`` factories.

Env implementations live in the top-level ``robots/`` directory (a sibling of
the ``rpent`` package); an env is resolved by importing ``robots.<name>``. The
``EnvSpec`` / ``PromptBundle`` / ``RunConfig`` dataclasses themselves live in
:mod:`rpent.envs` so planners and envs share the same contract types without
crossing module layers. ``EnvSpec`` also carries the three runner hooks
(``add_cli_args`` / ``parse_config`` / ``init_runtime``) that keep
``rpent/cli/main.py`` env-agnostic.
"""
from __future__ import annotations

import importlib
import sys
from typing import Any

from rpent.envs.env_spec import EnvSpec
from rpent.tools.toolkit import Toolkit
from rpent.utils.config import get_repo_root

# Env packages live under ``<repo>/robots/``, which is not part of the installed
# ``rpent`` distribution. Ensure the repo root is importable so ``robots.<name>``
# resolves regardless of the process's current working directory.
_REPO_ROOT = str(get_repo_root())
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def _resolve_env(name: str) -> Any:
    """Import ``robots.<name>`` lazily and return the module."""
    if not name:
        raise ValueError("env name must be non-empty")
    env_name = name.lower()
    try:
        return importlib.import_module(f"robots.{env_name}")
    except ModuleNotFoundError as e:
        raise ValueError(f"unknown env: {env_name!r}") from e


def get_env_spec(name: str) -> EnvSpec:
    return _resolve_env(name).get_env_spec()


def get_toolkit(name: str, **kwargs) -> Toolkit:
    """Build the env toolkit (common tools + env-specific tools)."""
    return _resolve_env(name).get_toolkit(**kwargs)
