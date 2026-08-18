"""Subprocess RPC servers: spawn (parent side), death-watch (child side)."""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import threading
from typing import Callable

from rpent.utils.config import get_repo_root
from rpent.utils.logging import get_logger

logger = get_logger("daemon")


# ---------------------------------------------------------------------------
# Child side
# ---------------------------------------------------------------------------


def watch_parent_death(on_death: Callable[[], None]) -> None:
    """Call ``on_death()`` once, from a background thread, when stdin hits EOF.

    Under :class:`ProcessDaemon` this fires exactly when the parent dies. If
    invoked from a terminal, ``read()`` blocks on user input and never fires;
    if stdin is redirected from ``/dev/null`` or already closed, it fires
    immediately.
    """
    def _watch() -> None:
        try:
            sys.stdin.buffer.read()
        except Exception:
            pass
        on_death()

    threading.Thread(target=_watch, daemon=True).start()


# ---------------------------------------------------------------------------
# Parent side
# ---------------------------------------------------------------------------


def pick_free_port(host: str = "127.0.0.1") -> int:
    """Return an unused TCP port on ``host``.

    There's a small race between the socket closing and the child binding;
    on 127.0.0.1 it's near-zero in practice.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, 0))
        return int(s.getsockname()[1])


class ProcessDaemon:
    """Wraps a subprocess server with ready-detection and lifecycle."""

    def __init__(
        self,
        name: str,
        cmd: list[str],
        *,
        env: dict[str, str] | None = None,
        log_path: str | None = None,
        cwd: str | None = None,
    ) -> None:
        self.name = name
        self.cmd = cmd
        self.subprocess_env = env or os.environ.copy()
        self.log_path = log_path
        self.cwd = cwd
        self._proc: subprocess.Popen | None = None
        self._log_f = None

    def poll(self) -> int | None:
        """Return the subprocess exit code, or None if it's still running.

        None also before ``start()``. A non-None value means the child has
        exited — used by ``wait_for_ready`` to fail fast on early crashes.
        """
        return self._proc.poll() if self._proc is not None else None

    def start(self) -> None:
        """Spawn the subprocess. Returns immediately; does not wait for readiness."""
        self._log_f = (
            open(self.log_path, "a") if self.log_path else open(os.devnull, "w")
        )
        self._proc = subprocess.Popen(
            self.cmd,
            # stdin pipe lets the child detect our death via EOF; see
            # watch_parent_death above.
            stdin=subprocess.PIPE,
            stdout=self._log_f,
            stderr=subprocess.STDOUT,
            env=self.subprocess_env,
            cwd=self.cwd or get_repo_root(),
        )
        logger.info("%s spawned (pid=%s)", self.name, self._proc.pid)

    def stop(self, timeout: float = 15.0) -> None:
        if self._proc is not None and self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        if self._log_f is not None:
            try:
                self._log_f.close()
            except Exception:
                pass
            self._log_f = None
