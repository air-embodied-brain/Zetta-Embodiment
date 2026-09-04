# Copyright (c) 2026 Zetta Contributors
"""Codex SDK planner.

Mirror of ``claude_code.py``: a thin, SDK-first backend. ``solve()`` prepares
artifacts, drives one Codex SDK turn, and assembles a ``PlannerResult``.
Zetta tools are exposed via the stdio MCP bridge configured through
``_codex_mcp_config_overrides``; this backend does not register tools in
process. Event rendering and stats live in a single ``_Recorder``.
"""

from __future__ import annotations

import json
import os
import queue
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import openai_codex

from zetta.cli.tui import next_user_line
from zetta.planner.base import PlannerResult, strip_mcp_prefix
from zetta.planner.codex_artifacts import export_codex_stream_artifacts
from zetta.planner.provider_pool import load_provider_pool_config
from zetta.planner.provider_proxy import (
    ProviderPoolProxy,
    load_provider_broker_connection,
)
from zetta.planner.utils.http_mcp_server import HttpMcpServer
from zetta.tools.toolkit import Toolkit
from zetta.utils.config import get_repo_root
from zetta.utils.logging import get_logger

logger = get_logger("codex")

PROVIDER_ID = "zetta_proxy"
PROVIDER_ENV_KEY = "ZETTA_CODEX_PROVIDER_KEY"
PHYSICAL_TOOL_NAMES = frozenset(
    {
        "move_to",
        "pi0_pick",
        "pi0_doubled",
        "release",
        "set_gripper",
        "rotate_wrist",
        "rotate_pitch",
        "move_pose",
        "vla_execute",
    }
)

# ---------------------------------------------------------------------------
# Public backend
# ---------------------------------------------------------------------------


class CodexPlanner:
    """Planner backed by the OpenAI Codex Python SDK."""

    def __init__(
        self,
        *,
        output_dir: str,
        repo_root: str | Path | None = None,
        timeout_s: int = 600,
        extra_dirs: list[str] | None = None,
        output_path: str | Path | None = None,
        model: str | None = None,
        reasoning_effort: str | None = None,
        dashboard: Any = None,
        resume_thread_id: str | None = None,
        sandbox: str | None = None,
        native_tools: bool = True,
    ):
        """Initialize the Codex SDK backend."""
        self._output_dir = str(output_dir)
        self._repo_root = str(repo_root) if repo_root else str(get_repo_root())
        self._timeout_s = timeout_s
        self._extra_dirs = extra_dirs or []
        self._output_path = Path(output_path) if output_path else None
        self._model = model or os.environ.get("CODEX_MODEL", None)
        self._reasoning_effort = reasoning_effort or os.environ.get(
            "CODEX_REASONING_EFFORT", None
        )
        self._reasoning_summary = os.environ.get("CODEX_REASONING_SUMMARY", "detailed")
        self._base_url = os.environ.get("CODEX_BASE_URL", None)
        self._api_key = os.environ.get("CODEX_API_KEY", None)
        default_pool_model = f"openai-responses:{self._model or 'gpt-5.6-sol'}"
        self._provider_pool_config = load_provider_pool_config(
            default_model=default_pool_model
        )
        self._external_provider_broker = load_provider_broker_connection()
        self._provider_proxy_base_url: str | None = None
        self._provider_proxy_api_key: str | None = None
        self._sandbox = (
            _sandbox_from_env() if sandbox is None else _sandbox_from_name(sandbox)
        )
        self._native_tools = bool(native_tools)
        self._dashboard = dashboard
        if resume_thread_id is not None and not str(resume_thread_id).strip():
            raise ValueError("resume_thread_id must be non-empty when provided")
        self._resume_thread_id = (
            None if resume_thread_id is None else str(resume_thread_id).strip()
        )

    def solve(
        self,
        *,
        system_prompt: str,
        user_message: str,
        toolkit: Toolkit,
        max_turns: int,
        input_queue=None,
    ) -> PlannerResult:
        """Run one or more Codex SDK turns for the given prompt."""
        prompt = f"{system_prompt}\n\n{user_message}" if system_prompt else user_message
        if self._output_path is None:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".out", prefix="codex_sdk_task_", delete=False
            ) as f:
                output_path = Path(f.name)
        else:
            output_path = self._output_path
            output_path.parent.mkdir(parents=True, exist_ok=True)
        raw_stream_path = output_path.with_suffix(output_path.suffix + ".stream.jsonl")
        last_message_path = output_path.with_suffix(output_path.suffix + ".last")
        recorder = _Recorder(max_turns=max_turns, dashboard=self._dashboard)
        has_physical_tools = any(
            str(spec.get("name", "")) in PHYSICAL_TOOL_NAMES
            for spec in toolkit.get_tools_spec()
        )
        state: dict[str, Any] = {}

        # Start the in-thread MCP HTTP server so Codex can reach the
        # shared toolkit without spawning a subprocess.
        mcp_server = HttpMcpServer(toolkit)
        mcp_url = mcp_server.start()
        logger.info("mcp http endpoint: %s", mcp_url)
        provider_proxy: ProviderPoolProxy | None = None
        if (
            self._provider_pool_config is not None
            and self._external_provider_broker is None
        ):
            try:
                provider_proxy = ProviderPoolProxy(
                    self._provider_pool_config,
                    timeout_seconds=float(self._timeout_s),
                )
                self._provider_proxy_base_url = provider_proxy.start()
                self._provider_proxy_api_key = provider_proxy.api_key
            except Exception:
                mcp_server.stop()
                raise
            logger.info(
                "Codex provider pool enabled with %d ordered route(s)",
                len(self._provider_pool_config.routes),
            )

        model_desc = self._model or "(configured default)"
        logger.info("prompt: %d chars", len(prompt))
        logger.info("output_dir: %s", self._output_dir)
        logger.info(
            "invoking Codex SDK model %s (timeout=%ds)",
            model_desc,
            self._timeout_s,
        )

        started = time.time()
        worker = threading.Thread(
            target=self._run_session,
            args=(
                prompt,
                output_path,
                raw_stream_path,
                last_message_path,
                recorder,
                state,
                mcp_url,
                input_queue,
                has_physical_tools,
            ),
            name="codex-sdk",
            daemon=True,
        )
        worker.start()
        try:
            worker.join(timeout=self._timeout_s)

            error: str | None = None
            if worker.is_alive():
                error = f"Codex SDK timed out after {self._timeout_s}s"
                _interrupt(state)
                rendered = f"\n[codex-planner] {error}\n"
                with open(output_path, "a") as out_f:
                    out_f.write(rendered)
                with open(raw_stream_path, "a") as raw_f:
                    _write_jsonl(raw_f, {"type": "timeout", "message": error})
                logger.info(rendered.rstrip())
                worker.join(timeout=15)
            elif "error" in state:
                exc = state["error"]
                error = f"{type(exc).__name__}: {exc}"
                rendered = f"\n[codex-planner] {error}\n"
                with open(output_path, "a") as out_f:
                    out_f.write(rendered)
                with open(raw_stream_path, "a") as raw_f:
                    _write_jsonl(raw_f, {"type": "error", "message": error})
                logger.info(rendered.rstrip())
        finally:
            mcp_server.stop()
            if provider_proxy is not None:
                provider_proxy.stop()
            self._provider_proxy_base_url = None
            self._provider_proxy_api_key = None

        elapsed = time.time() - started
        text = state.get("text", "") or output_path.read_text(errors="replace")
        error = error or recorder.error
        artifact_stats: dict[str, Any] = {}
        try:
            artifact_manifest = export_codex_stream_artifacts(
                raw_stream_path,
                output_path,
            )
            artifact_stats = {
                "artifact_manifest_path": artifact_manifest["manifest_path"],
                "reasoning_events_path": artifact_manifest["derived_artifacts"][
                    "reasoning_events"
                ],
                "reasoning_visible_path": artifact_manifest["derived_artifacts"][
                    "reasoning_visible"
                ],
                "planner_messages_path": artifact_manifest["derived_artifacts"][
                    "planner_messages"
                ],
                "tool_events_path": artifact_manifest["derived_artifacts"][
                    "tool_events"
                ],
                "reasoning_events_preserved": artifact_manifest["reasoning"][
                    "events_preserved"
                ],
                "reasoning_visible_text_chars": artifact_manifest["reasoning"][
                    "visible_text_chars"
                ],
                "raw_stream_parse_complete": artifact_manifest["completeness"][
                    "raw_stream_parse_complete"
                ],
                "terminal_event_present": artifact_manifest["completeness"][
                    "terminal_event_present"
                ],
            }
        except Exception as exc:
            artifact_stats = {"artifact_export_error": f"{type(exc).__name__}: {exc}"}
            logger.exception("failed to export Codex stream artifacts")

        logger.info("Codex SDK finished in %.1fs", elapsed)
        logger.info("output: %s", output_path)
        logger.info("raw stream: %s", raw_stream_path)

        return PlannerResult(
            finish_result=recorder.finish_result,
            messages=[{"role": "codex_sdk", "content": text}],
            stats={
                "backend": "codex_sdk",
                "model": self._model,
                "reasoning_effort": self._reasoning_effort,
                "reasoning_summary": self._reasoning_summary,
                "provider": (
                    "central_broker"
                    if self._external_provider_broker is not None
                    else "failover_pool"
                    if self._provider_pool_config is not None
                    else PROVIDER_ID
                    if self._base_url
                    else "default"
                ),
                "sandbox": self._sandbox.value,
                "native_tools": self._native_tools,
                "elapsed_s": round(elapsed, 1),
                "output_chars": len(text),
                "output_path": str(output_path),
                "raw_stream_path": str(raw_stream_path),
                "last_message_path": str(last_message_path),
                "last_message_chars": len(recorder.final_response or ""),
                "thread_id": state.get("thread_id"),
                "thread_resumed": bool(state.get("thread_resumed", False)),
                **artifact_stats,
                **recorder.stats(),
            },
            error=error,
        )

    # -- internal session --------------------------------------------------

    def _run_session(
        self,
        prompt: str,
        output_path: Path,
        raw_stream_path: Path,
        last_message_path: Path,
        recorder: "_Recorder",
        state: dict[str, Any],
        mcp_url: str,
        input_queue: "queue.Queue[str | None] | None" = None,
        has_physical_tools: bool = False,
    ) -> None:
        try:
            approval = openai_codex.ApprovalMode.deny_all
            sandbox = self._sandbox
            chunks: list[str] = []
            with openai_codex.Codex(config=self._build_config(mcp_url)) as codex:
                state["codex"] = codex
                if self._resume_thread_id is None:
                    thread = codex.thread_start(
                        approval_mode=approval,
                        cwd=self._repo_root,
                        model=self._model,
                        sandbox=sandbox,
                    )
                    state["thread_resumed"] = False
                else:
                    thread = codex.thread_resume(
                        self._resume_thread_id,
                        approval_mode=approval,
                        cwd=self._repo_root,
                        model=self._model,
                        sandbox=sandbox,
                    )
                    state["thread_resumed"] = True
                state["thread"] = thread
                state["thread_id"] = str(thread.id)

                with (
                    open(output_path, "w") as out_f,
                    open(raw_stream_path, "w") as raw_f,
                ):
                    write_lock = threading.Lock()

                    turn = thread.turn(
                        prompt,
                        approval_mode=approval,
                        cwd=self._repo_root,
                        model=self._model,
                        sandbox=sandbox,
                    )
                    state["turn"] = turn

                    fast_path_stop = threading.Event()
                    first_action_deadline_s = max(
                        0,
                        int(os.environ.get("ZETTA_FIRST_ACTION_DEADLINE_S", "240")),
                    )
                    if has_physical_tools and first_action_deadline_s:

                        def _enforce_first_action() -> None:
                            reminders = (
                                "FAST-PATH DEADLINE: stop reading now. Use the task-memory "
                                "already in the prompt, inspect only the immediate target and "
                                "destination if still needed, then call your first safe physical "
                                "tool. If no safe action is plausible, write the failure audit "
                                "and call finish.",
                                "SECOND FAST-PATH DEADLINE: no physical tool has started. Do not "
                                "read more files. Act now with the best safe physics-only step, "
                                "or finish with an honest failure audit.",
                            )
                            for reminder in reminders:
                                if fast_path_stop.wait(first_action_deadline_s):
                                    return
                                if recorder.physical_action_started:
                                    return
                                try:
                                    turn.steer(reminder)
                                    logger.info(
                                        "[codex-planner] sent first-action reminder"
                                    )
                                except Exception as exc:
                                    logger.info(
                                        "[codex-planner] first-action reminder failed: %s",
                                        exc,
                                    )
                                    return

                        threading.Thread(
                            target=_enforce_first_action,
                            name="codex-first-action",
                            daemon=True,
                        ).start()

                    stop_steer: threading.Event | None = None
                    if input_queue is not None:
                        stop_steer = threading.Event()

                        def _steer() -> None:
                            while True:
                                nxt = next_user_line(input_queue)
                                if stop_steer.is_set():
                                    return
                                if nxt is None:
                                    try:
                                        turn.interrupt()
                                    except Exception:
                                        pass
                                    return
                                rendered = f"\n[user] {nxt}\n"
                                with write_lock:
                                    chunks.append(rendered)
                                    out_f.write(rendered)
                                    out_f.flush()
                                logger.info(rendered.strip())
                                try:
                                    turn.steer(nxt)
                                except Exception as e:
                                    rendered = f"\n[codex-planner] steer failed: {e}\n"
                                    with write_lock:
                                        chunks.append(rendered)
                                        out_f.write(rendered)
                                        out_f.flush()
                                    logger.info(rendered.strip())
                                    return

                        threading.Thread(
                            target=_steer,
                            name="codex-steer",
                            daemon=True,
                        ).start()

                    try:
                        for event in turn.stream():
                            _write_jsonl(raw_f, _message_to_json(event))
                            if rendered := recorder.observe(event):
                                with write_lock:
                                    chunks.append(rendered)
                                    out_f.write(rendered)
                                    out_f.flush()
                                logger.info(rendered.strip())
                    finally:
                        fast_path_stop.set()
                        if stop_steer is not None:
                            stop_steer.set()

            state["text"] = "".join(chunks)
            if recorder.final_response is not None:
                last_message_path.write_text(recorder.final_response)
        except Exception as e:
            state["error"] = e

    # -- config builder ----------------------------------------------------

    def _build_config(self, mcp_url: str) -> Any:
        env = {**os.environ}
        for key in ("NO_PROXY", "no_proxy"):
            entries = [
                item.strip() for item in env.get(key, "").split(",") if item.strip()
            ]
            for loopback in ("127.0.0.1", "localhost"):
                if loopback not in entries:
                    entries.append(loopback)
            env[key] = ",".join(entries)
        external_url = (
            self._external_provider_broker[0]
            if self._external_provider_broker is not None
            else None
        )
        external_key = (
            self._external_provider_broker[1]
            if self._external_provider_broker is not None
            else None
        )
        effective_base_url = (
            external_url or self._provider_proxy_base_url or self._base_url
        )
        effective_api_key = (
            external_key
            or (
                self._provider_proxy_api_key
                if self._provider_proxy_base_url
                else self._api_key
            )
        )
        if effective_api_key:
            env[PROVIDER_ENV_KEY] = effective_api_key
        kwargs: dict[str, Any] = {
            "config_overrides": tuple(
                _codex_mcp_config_overrides(
                    mcp_url=mcp_url,
                    base_url=effective_base_url,
                    reasoning_effort=self._reasoning_effort,
                    reasoning_summary=self._reasoning_summary,
                    native_tools=self._native_tools,
                )
            ),
            "cwd": self._repo_root,
            "env": env,
            # MCP is available on the stable app-server surface. Keep the
            # experimental API off by default for OpenAI-compatible gateways;
            # it remains opt-in for diagnostics and future namespace tools.
            "experimental_api": os.environ.get("CODEX_EXPERIMENTAL_API", "false")
            .strip()
            .lower()
            not in {"0", "false", "no", "off"},
        }
        if codex_bin := os.environ.get("CODEX_BIN"):
            kwargs["codex_bin"] = codex_bin
        return openai_codex.CodexConfig(**kwargs)


def _sandbox_from_env() -> Any:
    """Resolve the planner sandbox without weakening the historical default."""
    return _sandbox_from_name(os.environ.get("ZETTA_CODEX_SANDBOX", "full-access"))


def _sandbox_from_name(requested: str) -> Any:
    """Resolve one explicit Codex sandbox name."""
    normalized = requested.strip().lower()
    aliases = {
        "read-only": openai_codex.Sandbox.read_only,
        "readonly": openai_codex.Sandbox.read_only,
        "workspace-write": openai_codex.Sandbox.workspace_write,
        "full-access": openai_codex.Sandbox.full_access,
        "full": openai_codex.Sandbox.full_access,
    }
    try:
        return aliases[normalized]
    except KeyError as exc:
        raise ValueError(
            "ZETTA_CODEX_SANDBOX must be read-only, workspace-write, or full-access; "
            f"got {normalized!r}"
        ) from exc


# ---------------------------------------------------------------------------
# Observation layer
# ---------------------------------------------------------------------------


@dataclass
class _Recorder:
    """Pure adapter: consume Codex SDK events, emit text + accumulate stats."""

    max_turns: int
    dashboard: Any = None
    turns: int = 0
    tool_calls: int = 0
    physical_action_started: bool = False
    physical_actions: int = 0
    usage: dict[str, int] = field(
        default_factory=lambda: {
            "total_input_tokens": 0,
            "total_cached_input_tokens": 0,
            "total_output_tokens": 0,
            "total_reasoning_output_tokens": 0,
        }
    )
    final_response: str | None = None
    finish_result: dict[str, Any] | None = None
    error: str | None = None

    def stats(self) -> dict[str, int]:
        return {
            "turns_used": self.turns,
            "tool_calls": self.tool_calls,
            "physical_actions": self.physical_actions,
            **self.usage,
        }

    def observe(self, event: Any) -> str:
        method = str(_get(event, "method", ""))
        payload = _get(event, "payload")

        if method in {"thread/started", "turn/started"}:
            return f"[codex-system] {method}\n"
        if method == "item/started":
            self._observe_item_started(_get(payload, "item"))
            return ""
        if method == "item/completed":
            return self._render_item(_get(payload, "item"))
        if method == "thread/tokenUsage/updated":
            self._set_usage(_get(payload, "token_usage"))
            return ""
        if method == "turn/completed":
            return self._render_turn_completed(_get(payload, "turn"))
        if "requestApproval" in method:
            return f"[codex-approval] {method}\n"
        if method in {"error", "fatal"}:
            return f"[codex-error] {_short_json(_jsonable(payload), limit=500)}\n"
        return ""

    # -- per-item handlers -------------------------------------------------

    def _observe_item_started(self, item: Any) -> None:
        item = _unwrap(item)
        item_type = str(_get(item, "type", ""))
        if item_type not in {"mcpToolCall", "dynamicToolCall"}:
            return
        name = strip_mcp_prefix(str(_get(item, "tool", item_type)))
        if name in PHYSICAL_TOOL_NAMES:
            self.physical_action_started = True

    def _render_item(self, item: Any) -> str:
        item = _unwrap(item)
        item_type = str(_get(item, "type", ""))

        if item_type == "userMessage":
            text = _extract_text(_get(item, "content"))
            return f"\n[codex][user] {text}\n" if text else ""

        if item_type in {"hookPrompt", "plan"}:
            return ""

        if item_type == "agentMessage":
            text = str(_get(item, "text", "")).strip()
            if not text:
                return ""
            self.final_response = text
            self.turns += 1
            if self.dashboard is not None:
                self.dashboard.on_event({"type": "text", "text": text})
            return (
                f"\n[agent] === turn {self.turns}/{self.max_turns} ===\n"
                f"[codex] {text}\n"
            )

        if item_type == "reasoning":
            text = _extract_text(_get(item, "summary") or _get(item, "content"))
            if text and self.dashboard is not None:
                self.dashboard.on_event({"type": "thinking", "text": text})
            return f"[codex-reasoning] {text}\n" if text else ""

        if item_type in {
            "mcpToolCall",
            "dynamicToolCall",
            "commandExecution",
            "fileChange",
        }:
            self.tool_calls += 1
            if item_type in {"mcpToolCall", "dynamicToolCall"}:
                name = strip_mcp_prefix(str(_get(item, "tool", item_type)))
                if name in PHYSICAL_TOOL_NAMES:
                    self.physical_action_started = True
                    self.physical_actions += 1
                self._maybe_capture_finish(name, item)
            elif item_type == "commandExecution":
                name = str(_get(item, "command", item_type))
            else:
                name = "fileChange"
            payload = _summarise_item(item)
            if self.dashboard is not None:
                data = _jsonable(item)
                args = data.get("arguments", {}) if isinstance(data, dict) else {}
                self.dashboard.on_event(
                    {"type": "tool_call", "tool": name, "args": args}
                )
                self.dashboard.on_event(
                    {"type": "tool_result", "tool": name, "result": payload}
                )
            return f"[tool<-] {name}: {json.dumps(payload, ensure_ascii=False)}\n"

        return ""

    def _render_turn_completed(self, turn: Any) -> str:
        self.turns = max(self.turns, 1)
        status = str(_get(_get(turn, "status"), "value", _get(turn, "status", "")))
        duration_ms = _get(turn, "duration_ms")
        if error := _get(turn, "error"):
            self.error = str(_get(error, "message", str(error)))

        parts = ["[codex-result]", status]
        if duration_ms is not None:
            parts.append(f"duration={float(duration_ms) / 1000:.1f}s")
        usage_line = (
            f"\n[usage] in={self.usage['total_input_tokens']} "
            f"cached={self.usage['total_cached_input_tokens']} "
            f"out={self.usage['total_output_tokens']} "
            f"reasoning={self.usage['total_reasoning_output_tokens']} "
            f"tool_calls={self.tool_calls}"
        )
        return " ".join(p for p in parts if p) + usage_line + "\n"

    # -- helpers -----------------------------------------------------------

    def _set_usage(self, usage: Any) -> None:
        if usage is None:
            return
        total = _get(usage, "total") or usage
        self.usage = {
            "total_input_tokens": _int_attr(total, "input_tokens"),
            "total_cached_input_tokens": _int_attr(total, "cached_input_tokens"),
            "total_output_tokens": _int_attr(total, "output_tokens"),
            "total_reasoning_output_tokens": _int_attr(
                total, "reasoning_output_tokens"
            ),
        }
        if self.dashboard is not None:
            self.dashboard.on_usage(
                inp=self.usage["total_input_tokens"],
                out=self.usage["total_output_tokens"],
                tool_calls=self.tool_calls,
            )

    def _maybe_capture_finish(self, name: str, item: Any) -> None:
        if self.finish_result is not None:
            return
        if name.lower() != "finish":
            return
        data = _jsonable(item)
        args = data.get("arguments") if isinstance(data, dict) else None
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except Exception:
                args = None
        if isinstance(args, dict):
            self.finish_result = {"_finish": True, **args}


# ---------------------------------------------------------------------------
# Codex config overrides
# ---------------------------------------------------------------------------


def _codex_mcp_config_overrides(
    *,
    mcp_url: str,
    base_url: str | None,
    reasoning_effort: str | None = None,
    reasoning_summary: str | None = None,
    native_tools: bool = True,
) -> list[str]:
    config: list[tuple[str, Any]] = [
        ("mcp_servers.zetta.url", mcp_url),
        ("mcp_servers.zetta.required", True),
    ]
    if not native_tools:
        config.extend(
            [
                ("features.shell_tool", False),
                ("features.shell_snapshot", False),
                ("tools.view_image", False),
                ("tools.web_search", False),
            ]
        )
    if base_url:
        normalized = base_url.rstrip("/")
        if not normalized.endswith("/v1"):
            normalized = normalized + "/v1"
        config.extend(
            [
                ("model_provider", PROVIDER_ID),
                (f"model_providers.{PROVIDER_ID}.name", PROVIDER_ID),
                (f"model_providers.{PROVIDER_ID}.base_url", normalized),
                (f"model_providers.{PROVIDER_ID}.wire_api", "responses"),
                (f"model_providers.{PROVIDER_ID}.env_key", PROVIDER_ENV_KEY),
            ]
        )
    if reasoning_effort:
        normalized_effort = reasoning_effort.strip().lower()
        allowed = {"minimal", "low", "medium", "high", "xhigh", "max", "ultra"}
        if normalized_effort not in allowed:
            raise ValueError(
                "CODEX_REASONING_EFFORT must be one of "
                f"{sorted(allowed)}; got {reasoning_effort!r}"
            )
        config.append(("model_reasoning_effort", normalized_effort))
    if reasoning_summary:
        normalized_summary = reasoning_summary.strip().lower()
        allowed_summaries = {"auto", "concise", "detailed", "none"}
        if normalized_summary not in allowed_summaries:
            raise ValueError(
                "CODEX_REASONING_SUMMARY must be one of "
                f"{sorted(allowed_summaries)}; got {reasoning_summary!r}"
            )
        config.append(("model_reasoning_summary", normalized_summary))
    return [f"{key}={json.dumps(value)}" for key, value in config]


# ---------------------------------------------------------------------------
# SDK utilities
# ---------------------------------------------------------------------------


def _interrupt(state: dict[str, Any]) -> None:
    if (turn := state.get("turn")) is not None:
        try:
            turn.interrupt()
        except Exception:
            pass
    if (codex := state.get("codex")) is not None:
        try:
            codex.close()
        except Exception:
            pass


def _write_jsonl(file_obj, value: dict[str, Any]) -> None:
    file_obj.write(json.dumps(value, ensure_ascii=False, default=str) + "\n")
    file_obj.flush()


def _message_to_json(message: Any) -> dict[str, Any]:
    return {
        "method": _get(message, "method", ""),
        "payload": _jsonable(_get(message, "payload")),
    }


def _jsonable(value: Any) -> Any:
    value = _unwrap(value)
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json", exclude_none=True)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(v) for v in value]
    if isinstance(value, bytes):
        return {"type": "bytes", "size": len(value)}
    return value


def _unwrap(value: Any) -> Any:
    return getattr(value, "root", value)


def _get(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(key, default)
    return getattr(value, key, default)


def _kind(value: Any) -> str:
    value = _unwrap(value)
    if isinstance(value, dict):
        return str(value.get("type") or value.get("kind") or "")
    return value.__class__.__name__


def _summarise_item(item: Any) -> dict[str, Any]:
    data = _jsonable(item)
    if not isinstance(data, dict):
        return {"size": _payload_size(data)}

    summary: dict[str, Any] = {}
    for key in ("path", "file_path", "filename", "status", "state", "exit_code"):
        value = data.get(key)
        if value not in (None, ""):
            summary[key] = value
    if command := (data.get("command") or data.get("cmd")):
        command_text = str(command)
        if len(command_text) > 200:
            command_text = command_text[:200] + f"...(+{len(command_text) - 200})"
        summary["command"] = command_text
    for key in ("content", "text", "output", "stdout", "stderr", "result"):
        if key in data and data[key] not in (None, ""):
            summary[f"{key}_size"] = _payload_size(data[key])

    if not summary:
        summary["keys"] = sorted(
            key for key in data if key not in {"content", "text", "output"}
        )
    return summary


def _extract_text(value: Any) -> str:
    value = _unwrap(value)
    if isinstance(value, str):
        text = value.strip()
        if "data:image" in text or (
            "base64" in text and ("image" in text or "iVBOR" in text)
        ):
            return "<image omitted>"
        return text
    if isinstance(value, list):
        parts = [_extract_text(item) for item in value]
        return "\n".join(part for part in parts if part)
    return ""


def _payload_size(value: Any) -> int:
    return len(str(value or ""))


def _short_json(value: Any, *, limit: int) -> str:
    text = json.dumps(value, ensure_ascii=False, default=str)
    if len(text) <= limit:
        return text
    return text[:limit] + f"...(+{len(text) - limit})"


def _int_attr(value: Any, key: str) -> int:
    return int(_get(value, key, 0) or 0)
