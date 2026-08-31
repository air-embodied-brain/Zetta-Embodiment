# Copyright (c) 2026 Zetta Contributors
"""Shared protocol for high-level reasoning backends."""

from __future__ import annotations

import os
import queue
from pathlib import Path
from typing import Any, Protocol

from zetta.tools.toolkit import Toolkit
from zetta.utils.config import (
    get_memory_dir,
    get_repo_root,
)

#: MCP namespace prefix for Zetta tools (``mcp__<server>__<tool>``).
#: Toolkits expose plain tool names; planners add/strip this prefix.
MCP_TOOL_PREFIX = "mcp__zetta__"


def add_mcp_prefix(name: str) -> str:
    """Return the namespaced MCP tool name for a bare tool name."""
    if name.startswith(MCP_TOOL_PREFIX):
        return name
    return f"{MCP_TOOL_PREFIX}{name}"


def strip_mcp_prefix(name: str) -> str:
    """Return the bare tool name, dropping the MCP namespace if present."""
    return name.removeprefix(MCP_TOOL_PREFIX)


class PlannerResult:
    """Result returned by a planner invocation."""

    __slots__ = ("finish_result", "messages", "stats", "error")

    def __init__(
        self,
        *,
        finish_result: dict | None = None,
        messages: list[dict] | None = None,
        stats: dict | None = None,
        error: str | None = None,
    ):
        """Initialize a serializable planner result."""
        self.finish_result = (
            finish_result  # {"status": "success"/"failure"/"stuck", "summary": "..."}
        )
        self.messages = messages or []  # serialisable conversation transcript
        self.stats = (
            stats or {}
        )  # {"total_input_tokens", "total_output_tokens", "turns_used", "tool_calls"}
        self.error = error  # str | None  — set when the planner raises


class Planner(Protocol):
    """A planner solves a task by conversing with an LLM/VLM backend.

    It is given one system prompt, one initial user message, and a set of
    tool definitions.  It returns a ``PlannerResult`` after the task is
    finished or the turn budget is exhausted.
    """

    def solve(
        self,
        *,
        system_prompt: str,
        user_message: str,
        toolkit: Toolkit,
        max_turns: int,
        input_queue: queue.Queue[str | None] | None = None,
    ) -> PlannerResult:
        """Run the multi-turn agent loop until completion or budget.

        Args:
            system_prompt: System-level instructions (role, rules, workflow).
            user_message: Initial user message (task description, first steps).
            toolkit: The full :class:`~zetta.tools.toolkit.Toolkit`
                (common + env tools). Backends derive ``tools_spec`` via
                ``toolkit.get_tools_spec()`` and dispatch calls via
                ``toolkit.execute_tool()``.
            max_turns: Maximum LLM turns before giving up.
            input_queue: Optional queue of user-typed lines for interactive steering.

        Returns:
            ``PlannerResult`` with finish status, conversation transcript,
            token-usage stats, and optional error string.
        """
        ...


# ---------------------------------------------------------------------------
# Planner construction
# ---------------------------------------------------------------------------


def build_planner(
    planner_type: str,
    *,
    output_dir: str | Path,
    recipe_tag: str,
    env_name: str,
    base_url: str | None = None,
    model: str | None = None,
    max_tokens: int = 8192,
    planner_timeout_s: int | None = None,
    claude_code_max_budget_usd: float | None = None,
    dashboard: Any = None,
    no_images: bool = False,
    codex_thread_id: str | None = None,
    reasoning_effort: str | None = None,
):
    """Build a planner for the given backend, resolving credentials from env vars."""
    # Imports are deferred to avoid a circular import: api_loop / claude_code /
    # codex all import from this module (PlannerResult).

    if planner_type == "api":
        if not model:
            raise ValueError(
                "the 'api' planner requires a model id; pass --model with a "
                "provider prefix (e.g. 'anthropic:claude-opus-4-8', "
                "'openai:gpt-5.5', 'openai-chat:glm-5.2')."
            )

        # Campaign manifests intentionally keep stable deployment aliases such
        # as ``gpt-5.6-sol`` instead of coupling the frozen protocol to
        # pydantic-ai's provider syntax.  The API planner's default transport
        # is OpenAI Responses, so qualify every bare alias at this boundary.
        # Explicit provider-qualified names remain unchanged.
        api_model_name = model if ":" in model else f"openai-responses:{model}"

        import inspect

        from pydantic_ai.models import infer_model
        from pydantic_ai.providers import infer_provider, infer_provider_class

        from zetta.planner.api_loop import ApiAgentLoop
        from zetta.planner.provider_pool import (
            build_provider_broker_model,
            build_provider_pool_model,
            load_provider_pool_config,
        )
        from zetta.planner.provider_proxy import load_provider_broker_connection

        def _provider_factory(provider_name: str):
            """Build the provider for ``provider_name``.

            The API key is always read from the provider's own env vars
            (e.g. ``ANTHROPIC_API_KEY``, ``OPENAI_API_KEY``). When
            ``base_url`` is given it overrides the provider's base URL env
            var (e.g. ``ANTHROPIC_BASE_URL`` / ``OPENAI_BASE_URL``).
            """
            if not base_url:
                return infer_provider(provider_name)
            provider_cls = infer_provider_class(provider_name)
            params = inspect.signature(provider_cls.__init__).parameters
            kwargs = {}
            if "base_url" in params:
                kwargs["base_url"] = base_url
            return provider_cls(**kwargs)

        provider_pool = load_provider_pool_config(default_model=api_model_name)
        provider_broker = load_provider_broker_connection()
        if provider_broker is not None:
            if provider_pool is None:
                raise ValueError(
                    "central provider broker requires ZETTA_API_PROVIDERS"
                )
            broker_url, broker_api_key = provider_broker
            api_model = build_provider_broker_model(
                provider_pool,
                base_url=broker_url,
                api_key=broker_api_key,
            )
        elif provider_pool is None:
            api_model = infer_model(
                api_model_name, provider_factory=_provider_factory
            )
        else:
            api_model = build_provider_pool_model(provider_pool)
        return ApiAgentLoop(
            model=api_model,
            model_name=model,
            reasoning_effort=reasoning_effort or "high",
            max_tokens=max_tokens,
            dashboard=dashboard,
            no_images=no_images,
        )
    if planner_type == "claude_code":
        from zetta.planner.claude_code import ClaudeCodePlanner

        cc_timeout_s = planner_timeout_s
        if cc_timeout_s is None:
            cc_timeout_s = int(os.environ.get("CELL_TIMEOUT_S", "1200"))
        cc_budget = claude_code_max_budget_usd
        if cc_budget is None:
            cc_budget = float(os.environ.get("MAX_BUDGET_USD", "10"))
        return ClaudeCodePlanner(
            output_dir=output_dir,
            repo_root=get_repo_root(),
            model=model or "sonnet",
            timeout_s=cc_timeout_s,
            max_budget_usd=cc_budget,
            extra_dirs=[str(get_memory_dir(env_name))],
            output_path=Path(output_dir) / f"claude_{recipe_tag}.txt",
            dashboard=dashboard,
        )
    if planner_type == "codex":
        from zetta.planner.codex import CodexPlanner

        cx_timeout_s = planner_timeout_s
        if cx_timeout_s is None:
            cx_timeout_s = int(
                os.environ.get(
                    "CODEX_TIMEOUT_S",
                    os.environ.get("CELL_TIMEOUT_S", "1200"),
                )
            )
        return CodexPlanner(
            output_dir=output_dir,
            repo_root=get_repo_root(),
            model=model,
            reasoning_effort=reasoning_effort,
            timeout_s=cx_timeout_s,
            extra_dirs=[str(get_memory_dir(env_name))],
            output_path=Path(output_dir) / f"codex_{recipe_tag}.txt",
            dashboard=dashboard,
            resume_thread_id=codex_thread_id,
        )
    raise ValueError(f"unknown planner_type: {planner_type}")
