"""Static env-extension descriptor.

Lives in :mod:`rpent.envs` alongside
:class:`~rpent.envs.prompt_bundle.PromptBundle` so envs
and planners can both import it without pulling in
:mod:`rpent.tools` or the RPC transport layer. Tool schemas,
handlers, server lifecycle, and the MCP allowlist live on
:class:`rpent.tools.toolkit.Toolkit` and its env subclasses —
``EnvSpec`` carries the env identity, the prompt bundle, and
the three runner hooks (``add_cli_args`` / ``parse_config`` /
``init_runtime``) that keep ``rpent/cli/main.py`` env-agnostic.
"""
from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from rpent.envs.prompt_bundle import PromptBundle

if TYPE_CHECKING:
    from rpent.dashboard.state import State
    from rpent.utils.daemon import ProcessDaemon


@dataclass(frozen=True)
class RunConfig:
    """Derived per-run identifiers produced by :attr:`EnvSpec.parse_config`."""

    recipe_tag: str
    output_dir: Path
    prompt_vars: dict[str, Any]
    dashboard_state: "State | None"
    task_desc: dict[str, Any]


@dataclass(frozen=True)
class EnvSpec:
    """Environment-level (non-tool) extension points for RPent."""

    name: str
    prompts: PromptBundle
    add_cli_args: Callable[[argparse.ArgumentParser, bool], None]
    parse_config: Callable[[argparse.Namespace], RunConfig]
    init_runtime: Callable[
        [argparse.Namespace, Path],
        tuple[list["ProcessDaemon"], dict[str, Any]],
    ]

