"""Dashboard HTTP server for the live monitor.

The FastAPI app runs on an OS-picked free port inside a daemon thread, so it can
sit alongside the agent run loop and keep serving after the run finishes, until
the process is stopped.

Routes mirror the fixed frontend contract in ``rpent/dashboard/index.html``.
"""
from __future__ import annotations

import asyncio
import json
import socket
import threading
import time
from pathlib import Path
from typing import Any

from fastapi import Body, FastAPI
from fastapi.responses import (
    FileResponse,
    JSONResponse,
    Response,
    StreamingResponse,
)
import uvicorn

from rpent.dashboard.state import State


class DashboardServer:
    """Threaded FastAPI server exposing the dashboard API for registered runs."""

    def __init__(
        self,
        *,
        host: str = "127.0.0.1",
        port: int = 0,
        runs_dir: str = "",
        language: str = "en",
    ) -> None:
        self.host = host
        self.port = int(port)
        self.runs_dir = runs_dir
        self._index_file = Path(__file__).parent / ("index.zh-cn.html" if language == "zh-cn" else "index.html")
        self._runs: dict[str, State] = {}
        self._app = self._build_app()
        self._server: uvicorn.Server | None = None

        # -- launcher state: when armed, the frontend shows a start screen and
        # the run loop blocks in wait_for_launch() until the user clicks Run.
        self._launch_enabled = False
        self._launch_defaults: dict[str, Any] = {}
        self._launch_config: dict[str, Any] | None = None
        self._launch_event = threading.Event()

    def register(self, run: State) -> None:
        self._runs[run.run_id] = run
        if not self.runs_dir:
            self.runs_dir = str(run.output_dir.parent)

    def wait_for_launch(self, defaults: dict[str, Any]) -> dict[str, Any]:
        """Arm the launcher and block until the user submits the start form.

        ``defaults`` pre-fills the launcher form (typically ``vars(args)``).
        Returns the config the user confirmed via ``POST /api/launch/run``.
        """
        self._launch_defaults = dict(defaults)
        self._launch_enabled = True
        self._launch_event.wait()
        return self._launch_config or {}

    def start(self) -> str:
        """Launch uvicorn in a daemon thread; return the base URL once serving."""
        port = self.port or _free_port(self.host)
        config = uvicorn.Config(
            self._app, host=self.host, port=port, log_level="warning"
        )
        self._server = uvicorn.Server(config)
        threading.Thread(target=self._server.run, daemon=True).start()

        t0 = time.time()
        while not self._server.started and time.time() - t0 < 10:
            time.sleep(0.05)
        if not self._server.started:
            raise RuntimeError(f"dashboard server did not start on {self.host}:{port}")
        self.port = port
        return f"http://{self.host}:{port}"

    # -- routes ------------------------------------------------------------

    def _build_app(self) -> FastAPI:
        app = FastAPI(title="RPent dashboard")

        @app.get("/")
        def index() -> Response:
            return FileResponse(self._index_file, media_type="text/html")

        @app.get("/healthz")
        def healthz() -> JSONResponse:
            return JSONResponse({"ok": True})

        @app.get("/api/launch/state")
        def api_launch_state() -> JSONResponse:
            return JSONResponse(
                {
                    "enabled": self._launch_enabled,
                    "pending": self._launch_enabled and not self._launch_event.is_set(),
                    "defaults": self._launch_defaults,
                }
            )

        @app.post("/api/launch/run")
        def api_launch_run(payload: dict[str, Any] = Body(default={})) -> JSONResponse:
            if not self._launch_enabled:
                return JSONResponse({"error": "launcher not armed"}, status_code=409)
            if self._launch_event.is_set():
                return JSONResponse({"error": "already launched"}, status_code=409)
            self._launch_config = dict(payload or {})
            self._launch_event.set()
            return JSONResponse({"ok": True})

        @app.get("/api/runs")
        def api_runs() -> JSONResponse:
            return JSONResponse(
                {
                    "runs_dir": self.runs_dir,
                    "runs": [r.run_info() for r in self._runs.values()],
                }
            )

        @app.get("/api/run")
        def api_run(run: str) -> JSONResponse:
            live = self._runs.get(run)
            if live is None:
                return JSONResponse({"error": "unknown run"}, status_code=404)
            return JSONResponse(live.run_detail())

        @app.get("/api/run/transcript")
        def api_transcript(run: str, since: int = 0) -> JSONResponse:
            live = self._runs.get(run)
            events = live.events_since(since) if live else []
            return JSONResponse({"events": events})

        @app.get("/api/run/frame")
        def api_frame(run: str, kind: str = "agent", t: str = "") -> Response:
            live = self._runs.get(run)
            png = live.frame(kind) if live else None
            if png is None:
                return Response(status_code=404)
            return Response(png, media_type="image/png")

        @app.get("/api/run/video")
        def api_video(run: str) -> Response:
            live = self._runs.get(run)
            if live is None or not live.has_video():
                return Response(status_code=404)
            return FileResponse(
                live.video_path,
                media_type="video/mp4",
                headers={"Cache-Control": "no-store, max-age=0"},
            )

        @app.get("/api/run/action-video")
        def api_action_video(run: str, step: int) -> Response:
            live = self._runs.get(run)
            path = live.action_video_path(step) if live else None
            if path is None:
                return Response(status_code=404)
            return FileResponse(
                path,
                media_type="video/mp4",
                headers={"Cache-Control": "no-store, max-age=0"},
            )

        @app.get("/api/stream")
        def api_stream(run: str) -> StreamingResponse:
            async def gen():
                while True:
                    live = self._runs.get(run)
                    if live is not None:
                        yield f"data: {json.dumps(live.snapshot())}\n\n"
                    else:
                        yield ": keepalive\n\n"
                    await asyncio.sleep(0.1)

            return StreamingResponse(gen(), media_type="text/event-stream")

        return app


def _free_port(host: str) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, 0))
        return int(s.getsockname()[1])
