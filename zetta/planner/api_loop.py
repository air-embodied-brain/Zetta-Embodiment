"""Provider-independent tool-use agent loop built on pydantic-ai.

The loop wraps the agent's :class:`~zetta.tools.toolkit.Toolkit` as
pydantic-ai function tools and drives a single :class:`pydantic_ai.Agent` run,
streaming each turn so progress is logged in real time. Task completion is
signalled by the env-provided ``finish`` tool, whose result carries ``_finish``.
"""

from __future__ import annotations

import asyncio
import base64
import dataclasses
import json
import queue
from typing import Any

from pydantic_ai import Agent, BinaryContent, ModelSettings, Tool, ToolReturn
from pydantic_ai.capabilities import ProcessHistory, Thinking
from pydantic_ai.exceptions import ModelHTTPError, UsageLimitExceeded
from pydantic_ai.messages import (
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    ModelMessage,
    ModelResponse,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    UserPromptPart,
)
from pydantic_ai.models import Model
from pydantic_ai.usage import RunUsage, UsageLimits

from zetta.cli.tui import QUIT_TOKENS
from zetta.planner.base import PlannerResult
from zetta.tools.toolkit import Toolkit
from zetta.utils.logging import get_logger

logger = get_logger("api_loop")

#: Console-log truncation limits (characters).
_TEXT_LOG_LIMIT = 500
_ARGS_LOG_LIMIT = 250
_TOOL_LOG_LIMIT = 350

#: Cap on cumulative decoded image bytes kept in the resent request history.
_MAX_HISTORY_IMAGE_BYTES = 4 * 1024 * 1024

#: Always retain at least this many of the most recent images, even if a single
#: frame exceeds the byte budget, so the model never loses its current view.
_MIN_RECENT_IMAGES = 2


class ApiAgentLoop:
    """Planner that runs the tool-calling loop via a pydantic-ai ``Agent``."""

    def __init__(
        self,
        model: Model,
        model_name: str | None = None,
        reasoning_effort: str = "high",
        max_tokens: int = 8192,
        dashboard: Any = None,
        no_images: bool = False,
    ):
        """Store the pydantic-ai model and the output-token cap."""
        if reasoning_effort not in {"low", "medium", "high", "xhigh"}:
            raise ValueError("unsupported API reasoning effort")
        self._model = model
        self._model_name = model_name
        self._reasoning_effort = reasoning_effort
        self._max_tokens = max_tokens
        self._dashboard = dashboard
        self._no_images = no_images

    def solve(
        self,
        *,
        system_prompt: str,
        user_message: str,
        toolkit: Toolkit,
        max_turns: int,
        input_queue: queue.Queue[str | None] | None = None,
    ) -> PlannerResult:
        """Run the tool-calling loop until finish, normal stop, or budget."""
        return asyncio.run(
            self._solve(
                system_prompt=system_prompt,
                user_message=user_message,
                toolkit=toolkit,
                max_turns=max_turns,
                input_queue=input_queue,
            )
        )

    async def _solve(
        self,
        *,
        system_prompt: str,
        user_message: str,
        toolkit: Toolkit,
        max_turns: int,
        input_queue: queue.Queue[str | None] | None = None,
    ) -> PlannerResult:
        agent = Agent(
            self._model,
            instructions=system_prompt or None,
            tools=_build_tools(toolkit, no_images=self._no_images),
            model_settings=_build_model_settings(self._model, self._max_tokens),
            capabilities=[
                Thinking(effort=self._reasoning_effort),
                ProcessHistory(processor=_prune_history_images),
            ],
        )

        interactive = input_queue is not None
        messages: list[dict[str, Any]] = [{"role": "user", "content": user_message}]
        finish_result: dict[str, Any] | None = None
        n_tool_calls = 0
        turns = 0
        last_error: str | None = None
        usage: RunUsage | None = None
        quit_requested = False

        def _inject_pending(run: Any) -> bool:
            """Drain queued user lines into the live run; True => end session.

            Each line is enqueued ``asap`` so it lands in the next model request
            (the next turn boundary). This runs on the event-loop thread, so
            mutating the run's pending-message queue here is race-free.
            """
            while True:
                try:
                    line = input_queue.get_nowait()  # type: ignore[union-attr]
                except queue.Empty:
                    return False
                if line is None:
                    return True
                line = line.strip()
                if line.lower() in QUIT_TOKENS:
                    return True
                if not line:
                    continue
                run.enqueue(line, priority="asap")
                messages.append({"role": "user", "content": line})
                logger.info("[user] %s", _clip(line, _ARGS_LOG_LIMIT))

        async def _await_next() -> str | None:
            """Block off-loop for the next user line between runs (None => end)."""
            logger.info("awaiting input — type a message to continue, /quit to end")
            while True:
                line = await asyncio.to_thread(input_queue.get)  # type: ignore[union-attr]
                if line is None:
                    return None
                line = line.strip()
                if line.lower() in QUIT_TOKENS:
                    return None
                if line:
                    logger.info("[user] %s", _clip(line, _ARGS_LOG_LIMIT))
                    return line

        seed = user_message
        history: list[ModelMessage] | None = None
        try:
            while True:
                run_turns = 0
                # request_limit overrides pydantic-ai's default (50) so the
                # manual max_turns break below is what bounds each run.
                async with agent.iter(
                    seed,
                    message_history=history,
                    usage_limits=UsageLimits(request_limit=max_turns + 1),
                ) as run:
                    async for node in run:
                        if interactive and _inject_pending(run):
                            quit_requested = True
                            break
                        if Agent.is_call_tools_node(node):
                            turns += 1
                            run_turns += 1
                            response = node.model_response
                            response_message = _serialize_response(response)
                            messages.append(response_message)
                            _log_response(response, run.usage, run_turns, max_turns)
                            if self._dashboard is not None:
                                for block in response_message["content"]:
                                    if block["type"] == "text":
                                        dashboard_event = {
                                            "type": "text",
                                            "text": block["text"],
                                        }
                                    elif block["type"] == "thinking":
                                        dashboard_event = {
                                            "type": "thinking",
                                            "text": block["thinking"],
                                        }
                                    else:
                                        continue
                                    self._dashboard.on_event(dashboard_event)
                                self._dashboard.on_usage(
                                    inp=int(run.usage.input_tokens or 0),
                                    out=int(run.usage.output_tokens or 0),
                                    tool_calls=n_tool_calls,
                                )

                            async with node.stream(run.ctx) as stream:
                                async for event in stream:
                                    if isinstance(event, FunctionToolCallEvent):
                                        n_tool_calls += 1
                                        if self._dashboard is not None:
                                            self._dashboard.on_event(
                                                {
                                                    "type": "tool_call",
                                                    "tool": event.part.tool_name,
                                                    "args": event.part.args_as_dict(),
                                                }
                                            )
                                        if event.part.tool_name == "finish":
                                            finish_result = {
                                                "_finish": True,
                                                **event.part.args_as_dict(),
                                            }
                                    elif isinstance(event, FunctionToolResultEvent):
                                        message = _serialize_tool_result(event)
                                        messages.append(message)
                                        _log_tool_result(message)
                                        if self._dashboard is not None:
                                            dashboard_result = {
                                                "is_error": bool(
                                                    getattr(
                                                        event.part, "is_error", False
                                                    )
                                                ),
                                                "size": len(message["content"]),
                                            }
                                            self._dashboard.on_event(
                                                {
                                                    "type": "tool_result",
                                                    "tool": message.get("name")
                                                    or "tool_result",
                                                    "result": dashboard_result,
                                                }
                                            )
                                    if self._dashboard is not None:
                                        self._dashboard.on_usage(
                                            inp=int(run.usage.input_tokens or 0),
                                            out=int(run.usage.output_tokens or 0),
                                            tool_calls=n_tool_calls,
                                        )

                            if finish_result is not None:
                                logger.info("FINISH called: %s", finish_result)
                                break
                            if turns >= max_turns:
                                logger.info(
                                    "reached max_turns=%d. Stopping.", max_turns
                                )
                                break
                        elif Agent.is_end_node(node):
                            if interactive:
                                logger.info(
                                    "model ended turn without a tool call "
                                    "— awaiting your input."
                                )
                            else:
                                logger.info(
                                    "model ended turn without a tool call. Stopping."
                                )
                            break

                    usage = run.usage
                    if interactive:
                        history = run.all_messages()

                # finish, quit, non-interactive, or the cumulative turn budget is
                # spent => end the whole session so max_turns is enforced across
                # every run, not per run.
                if (
                    finish_result is not None
                    or quit_requested
                    or not interactive
                    or turns >= max_turns
                ):
                    break
                nxt = await _await_next()
                if nxt is None:
                    break
                seed = nxt
                messages.append({"role": "user", "content": seed})
        except UsageLimitExceeded as e:
            logger.info("usage limit reached: %s", e)
        except Exception as e:  # noqa: BLE001 - surfaced via PlannerResult.error
            last_error = f"{type(e).__name__}: {e}"
            if _is_image_rejection(e) and not self._no_images:
                last_error += (
                    "\n\nThe model rejected image input — it is likely a "
                    "text-only model (no vision support). Re-run with "
                    "--no-images: Zetta will then keep every visual "
                    "observation as a file-path text notice instead of "
                    "sending image bytes."
                )
            logger.error("agent run failed: %s", last_error)

        return PlannerResult(
            finish_result=finish_result,
            messages=messages,
            stats={
                **_build_stats(usage, turns, n_tool_calls),
                "model": self._model_name,
                "reasoning_effort": self._reasoning_effort,
            },
            error=last_error,
        )


def _build_model_settings(model: Model, max_tokens: int) -> ModelSettings:
    """Build model settings, enabling prompt caching for Anthropic models."""
    # ``pydantic-ai-slim`` deliberately keeps provider SDKs optional.  Importing
    # the Anthropic model module unconditionally therefore breaks an OpenAI-only
    # deployment when the unrelated ``anthropic`` package is absent.  Inspect
    # the MRO without importing that optional module and only load its settings
    # for an actual Anthropic model (including subclasses).
    anthropic_model = any(
        base.__module__ == "pydantic_ai.models.anthropic"
        and base.__name__ == "AnthropicModel"
        for base in type(model).__mro__
    )
    if not anthropic_model:
        return ModelSettings(max_tokens=max_tokens)

    from pydantic_ai.models.anthropic import AnthropicModelSettings

    return AnthropicModelSettings(
        max_tokens=max_tokens,
        anthropic_cache_instructions=True,
        anthropic_cache_tool_definitions=True,
        anthropic_cache_messages=True,
    )


def _prune_history_images(messages: list[ModelMessage]) -> list[ModelMessage]:
    """Drop old camera images so the resent request body stays bounded."""
    # Every image in history, oldest -> newest: (msg_idx, part_idx, item_idx, nbytes).
    located: list[tuple[int, int, int, int]] = []
    for mi, message in enumerate(messages):
        for pi, part in enumerate(getattr(message, "parts", ()) or ()):
            if not isinstance(part, UserPromptPart) or not isinstance(
                part.content, list
            ):
                continue
            for ii, item in enumerate(part.content):
                if isinstance(item, BinaryContent) and item.media_type.startswith(
                    "image/"
                ):
                    located.append((mi, pi, ii, len(item.data)))

    if not located:
        return messages

    # Walk newest -> oldest, keeping images while under the byte budget.
    keep: set[tuple[int, int, int]] = set()
    total = 0
    for rank, (mi, pi, ii, nbytes) in enumerate(reversed(located)):
        if rank < _MIN_RECENT_IMAGES or total + nbytes <= _MAX_HISTORY_IMAGE_BYTES:
            keep.add((mi, pi, ii))
            total += nbytes

    if len(keep) == len(located):
        return messages

    drop_items_by_part: dict[tuple[int, int], set[int]] = {}
    for mi, pi, ii, _ in located:
        if (mi, pi, ii) not in keep:
            drop_items_by_part.setdefault((mi, pi), set()).add(ii)

    new_messages = list(messages)
    for (mi, pi), drop_items in drop_items_by_part.items():
        message = new_messages[mi]
        part = message.parts[pi]
        new_content = [
            "[earlier camera image omitted to bound request size]"
            if ci in drop_items
            else item
            for ci, item in enumerate(part.content)
        ]
        new_parts = list(message.parts)
        new_parts[pi] = dataclasses.replace(part, content=new_content)
        new_messages[mi] = dataclasses.replace(message, parts=new_parts)

    return new_messages


def _is_image_rejection(e: Exception) -> bool:
    """True when the provider returned a 4xx complaining about image input.

    Matches errors like ``400 {'code': 10007, 'msg': "Bad Request: [message
    type 'image_url' is not supported]"}`` from OpenAI-compatible endpoints
    serving text-only models.
    """
    if not isinstance(e, ModelHTTPError):
        return False
    if not 400 <= e.status_code < 500:
        return False
    return "image" in str(e).lower()


def _build_tools(toolkit: Toolkit, *, no_images: bool = False) -> list[Tool]:
    """Build the API-only image reader plus pydantic-ai toolkit wrappers."""
    image_reader = read_image_text_only if no_images else read_image
    tools: list[Tool] = [Tool(image_reader, name="read_image")]
    for spec in toolkit.get_tools_spec():
        name = spec["name"]
        tools.append(
            Tool.from_schema(
                function=_make_tool_function(toolkit, name, no_images=no_images),
                name=name,
                description=spec.get("description", ""),
                json_schema=spec.get("input_schema")
                or {"type": "object", "properties": {}},
                takes_ctx=False,
            )
        )
    return tools


def read_image(path: str) -> ToolReturn:
    """Read a local image path returned by an Zetta tool as visual input."""
    return ToolReturn(
        return_value=path,
        content=[BinaryContent.from_path(path)],
    )


def read_image_text_only(path: str) -> str:
    """``read_image`` stub for ``--no-images``: acknowledge, send no bytes."""
    return (
        f"{path} exists, but image input is disabled (--no-images, text-only "
        "model). Reason from textual state instead: view_driver_state, "
        "back_project, and the numeric fields in tool results."
    )


def _make_tool_function(toolkit: Toolkit, name: str, *, no_images: bool = False):
    """Return a callable that dispatches one tool call to the toolkit."""

    def _call(**kwargs: Any) -> Any:
        result = toolkit.execute_tool(name, kwargs)
        text, images = _content_blocks_to_pydantic(result.content_blocks)
        if images and not no_images:
            return ToolReturn(return_value=text, content=images)
        return text

    _call.__name__ = name
    return _call


def _content_blocks_to_pydantic(
    blocks: list[dict[str, Any]],
) -> tuple[str, list[BinaryContent]]:
    """Split Anthropic-shaped content blocks into text and image content."""
    text_parts: list[str] = []
    images: list[BinaryContent] = []
    for block in blocks:
        block_type = block.get("type")
        if block_type == "text":
            text_parts.append(block.get("text", ""))
        elif block_type == "image":
            source = block.get("source") or {}
            data = source.get("data")
            if source.get("type") == "base64" and data:
                images.append(
                    BinaryContent(
                        data=base64.b64decode(data),
                        media_type=source.get("media_type", "image/png"),
                    )
                )
    text = "\n\n".join(part for part in text_parts if part) or "{}"
    return text, images


def _serialize_response(response: ModelResponse) -> dict[str, Any]:
    """Render one assistant turn as a serialisable transcript message."""
    content: list[dict[str, Any]] = []
    for part in response.parts:
        if isinstance(part, TextPart):
            if part.content:
                content.append({"type": "text", "text": part.content})
        elif isinstance(part, ThinkingPart):
            if part.content:
                content.append({"type": "thinking", "thinking": part.content})
        elif isinstance(part, ToolCallPart):
            content.append(
                {
                    "type": "tool_use",
                    "id": part.tool_call_id,
                    "name": part.tool_name,
                    "input": part.args_as_dict(),
                }
            )
    return {"role": "assistant", "content": content}


def _serialize_tool_result(event: FunctionToolResultEvent) -> dict[str, Any]:
    """Render one tool result as a serialisable transcript message (no images)."""
    part = event.part
    content = getattr(part, "content", None)
    if not isinstance(content, str):
        content = json.dumps(content, default=str)
    return {
        "role": "tool",
        "name": getattr(part, "tool_name", None),
        "tool_call_id": getattr(part, "tool_call_id", None),
        "content": content,
    }


def _build_stats(
    usage: RunUsage | None, turns: int, n_tool_calls: int
) -> dict[str, Any]:
    """Assemble the run stats dict from accumulated usage and counters."""
    stats: dict[str, Any] = {"turns_used": turns, "tool_calls": n_tool_calls}
    if usage is not None:
        stats.update(
            {
                "total_input_tokens": int(usage.input_tokens or 0),
                "total_output_tokens": int(usage.output_tokens or 0),
                "cache_read_tokens": int(usage.cache_read_tokens or 0),
                "cache_write_tokens": int(usage.cache_write_tokens or 0),
                "requests": int(usage.requests or 0),
            }
        )
    return stats


def _log_response(
    response: ModelResponse, usage: RunUsage, turn: int, max_turns: int
) -> None:
    """Log model text, thinking, tool calls, and cumulative usage for a turn."""
    logger.info("=== turn %d/%d ===", turn, max_turns)
    for part in response.parts:
        if isinstance(part, TextPart):
            text = (part.content or "").strip()
            if text:
                logger.info("[model] %s", text)
        elif isinstance(part, ThinkingPart):
            text = (part.content or "").strip()
            if text:
                logger.info("[think] %s", _clip(text, _TEXT_LOG_LIMIT))
        elif isinstance(part, ToolCallPart):
            args = json.dumps(part.args_as_dict(), default=str)
            logger.info("[tool>] %s(%s)", part.tool_name, _clip(args, _ARGS_LOG_LIMIT))
    logger.info(
        "[usage] in=%s out=%s cache_read=%s cache_write=%s requests=%s",
        usage.input_tokens,
        usage.output_tokens,
        usage.cache_read_tokens,
        usage.cache_write_tokens,
        usage.requests,
    )


def _log_tool_result(message: dict[str, Any]) -> None:
    """Log a one-line summary of a tool result."""
    content = " ".join((message.get("content") or "").split())
    logger.info("[tool<] %s: %s", message.get("name"), _clip(content, _TOOL_LOG_LIMIT))


def _clip(text: str, limit: int) -> str:
    """Truncate ``text`` to ``limit`` characters with an overflow marker."""
    if len(text) <= limit:
        return text
    return text[:limit] + "...(+%d)" % (len(text) - limit)
