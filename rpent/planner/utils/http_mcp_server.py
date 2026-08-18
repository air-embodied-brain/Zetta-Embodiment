"""In-process streamable-HTTP MCP server that wraps a :class:`Toolkit`.

The Codex CLI accepts streamable-HTTP MCP servers via
``mcp_servers.<name>.url``. This module builds an MCP
:class:`~mcp.server.lowlevel.Server` and serves it through a
:class:`~mcp.server.streamable_http_manager.StreamableHTTPSessionManager`
on a background uvicorn thread — no separate subprocess, no second
``Toolkit`` instance.

Usage::

    server = HttpMcpServer(toolkit)
    server.start()  # binds 127.0.0.1 on a free port
    codex_url = server.url   # e.g. "http://127.0.0.1:54321/mcp/"
    ...
    server.stop()
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import socket
import threading
from typing import Any

import httpx

import uvicorn
from mcp import types
from mcp.server.lowlevel import Server
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from starlette.responses import Response

from rpent.tools.toolkit import Toolkit

logger = logging.getLogger("mcp_http")

SERVER_NAME = "rpent"


def _toolkit_to_mcp_content(
    tr: Any,
) -> tuple[list[types.TextContent | types.ImageContent], bool]:
    """Translate a :class:`ToolResult` into MCP content blocks + isError."""
    blocks = getattr(tr, "content_blocks", None)
    if blocks is None:
        return [types.TextContent(type="text", text=str(tr))], False

    out: list[types.TextContent | types.ImageContent] = []
    for block in blocks:
        block_type = block.get("type")
        if block_type == "text":
            out.append(types.TextContent(type="text", text=block.get("text", "")))
        elif block_type == "image":
            src = block.get("source", {})
            out.append(
                types.ImageContent(
                    type="image",
                    data=src.get("data", ""),
                    mimeType=src.get("media_type", "image/png"),
                )
            )
    result_dict = getattr(tr, "result", None)
    is_error = isinstance(result_dict, dict) and bool(result_dict.get("error"))
    return out, is_error


def _strip_mcp_prefix(name: str) -> str:
    """``mcp__rpent__mcp_list_dir`` -> ``mcp_list_dir`` ; passthrough."""
    prefix = f"mcp__{SERVER_NAME}__"
    if name.startswith(prefix):
        return name[len(prefix):]
    return name


def _build_asgi_app(toolkit: Toolkit, *, path: str = "/mcp") -> Any:
    """Build a raw ASGI3 app wrapping an MCP ``Server`` + streamable HTTP."""
    mcp_app: Server = Server(SERVER_NAME, version="0.1.0")

    @mcp_app.list_tools()
    async def _list_tools() -> list[types.Tool]:
        tools: list[types.Tool] = []
        for spec in toolkit.get_tools_spec():
            tools.append(
                types.Tool(
                    name=str(spec["name"]),
                    description=str(spec.get("description", "")),
                    inputSchema=spec.get("input_schema", {"type": "object"}),
                )
            )
        return tools

    @mcp_app.call_tool()
    async def _call_tool(
        name: str, arguments: dict[str, Any]
    ) -> types.CallToolResult:
        lookup = _strip_mcp_prefix(name)
        tr = await asyncio.get_running_loop().run_in_executor(
            None, toolkit.execute_tool, lookup, arguments or {}
        )
        content, is_error = _toolkit_to_mcp_content(tr)
        return types.CallToolResult(content=content, isError=is_error)

    session_manager = StreamableHTTPSessionManager(
        app=mcp_app,
        stateless=True,
        json_response=True,
    )

    # minimal ASGI wrapper
    async def asgi_app(
        scope: dict[str, Any], receive: Any, send: Any
    ) -> None:
        if scope["type"] == "lifespan":
            while True:
                event = await receive()
                if event["type"] == "lifespan.startup":
                    async with session_manager.run():
                        await send({"type": "lifespan.startup.complete"})
                        shutdown_event = await receive()
                        assert shutdown_event["type"] == "lifespan.shutdown"
                        await send({"type": "lifespan.shutdown.complete"})
                    break
                elif event["type"] == "lifespan.shutdown":
                    await send({"type": "lifespan.shutdown.complete"})
                    break
            return
        if scope["type"] == "http":
            request_path = str(scope.get("path", "")).rstrip("/") or "/"
            endpoint_path = path.rstrip("/") or "/"
            if request_path != endpoint_path:
                await Response(status_code=404)(scope, receive, send)
                return
            if scope.get("method") == "GET":
                headers = {
                    key.decode("latin-1").lower(): value.decode("latin-1")
                    for key, value in scope.get("headers", [])
                }
                if "text/event-stream" not in headers.get("accept", ""):
                    await Response(
                        status_code=405, headers={"Allow": "POST, GET"}
                    )(scope, receive, send)
                    return
            await session_manager.handle_request(scope, receive, send)

    return asgi_app


def _pick_free_port(host: str) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, 0))
        return int(s.getsockname()[1])


def _wait_for_ready(url: str, *, timeout_s: float) -> None:
    """POST an MCP ``initialize`` request, retrying connection failures."""
    body = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "hc", "version": "0"},
        },
    }
    transport = httpx.HTTPTransport(retries=10)
    with httpx.Client(transport=transport, timeout=httpx.Timeout(timeout_s, connect=2)) as c:
        resp = c.post(url, json=body, headers={"Accept": "application/json"})
        if not (resp.is_success and "result" in resp.json()):
            raise RuntimeError(f"HttpMcpServer not ready: {resp.status_code}")


class HttpMcpServer:
    """Run an in-process streamable-HTTP MCP server over a Toolkit.

    The server runs on a background daemon thread with its own asyncio loop;
    callers must invoke :meth:`start` before reading :attr:`url` and
    :meth:`stop` to release the port before the process exits.
    """

    def __init__(
        self,
        toolkit: Toolkit,
        *,
        host: str = "127.0.0.1",
        port: int = 0,
        path: str = "/mcp",
    ) -> None:
        self._toolkit = toolkit
        self._host = host
        self._port = port or _pick_free_port(host)
        self._path = path if path.startswith("/") else f"/{path}"
        self._server: uvicorn.Server | None = None
        self._thread: threading.Thread | None = None

    @property
    def url(self) -> str:
        return f"http://{self._host}:{self._port}{self._path}/"

    def start(self, *, ready_timeout_s: float = 30.0) -> str:
        """Launch uvicorn in a background thread and block until it's serving."""
        if self._thread is not None:
            return self.url

        app = _build_asgi_app(self._toolkit, path=self._path)

        access_log = os.environ.get("RPENT_MCP_ACCESS_LOG", "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        config = uvicorn.Config(
            app,
            host=self._host,
            port=self._port,
            log_level="info" if access_log else "warning",
            access_log=access_log,
            lifespan="on",
        )
        self._server = uvicorn.Server(config)

        self._thread = threading.Thread(
            target=self._server.run, name="mcp-http-server", daemon=True
        )
        self._thread.start()

        _wait_for_ready(self.url, timeout_s=ready_timeout_s)
        logger.info("HttpMcpServer ready at %s", self.url)
        return self.url

    def stop(self, *, timeout_s: float = 5.0) -> None:
        if self._server is not None:
            self._server.should_exit = True
        if self._thread is not None:
            self._thread.join(timeout=timeout_s)
        self._server = None
        self._thread = None
