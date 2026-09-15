# Copyright (c) 2026 Zetta Contributors
"""Launch one Isaac interpreter and supervise its native environment requests."""

from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from .config import GenieSimConfig
from .wire import PROTOCOL_VERSION, receive_packet, send_packet

_BOOTSTRAP = """import importlib
import importlib.util
import sys
from pathlib import Path

package = Path(sys.argv.pop(1))
name = "_zetta_geniesim_worker"
spec = importlib.util.spec_from_file_location(
    name, package / "__init__.py", submodule_search_locations=[str(package)]
)
module = importlib.util.module_from_spec(spec)
sys.modules[name] = module
spec.loader.exec_module(module)
importlib.import_module(name + ".worker").main()
"""


class GenieSimProcessError(RuntimeError):
    """An environment operation failed or its outcome is no longer known."""


class GenieSimProcess:
    def __init__(
        self, config: GenieSimConfig, *, _worker_package: Path | None = None
    ) -> None:
        self.config = config
        self._lock = threading.RLock()
        self._closed = False
        self._sequence = 0
        self._process: subprocess.Popen | None = None
        self._channel: socket.socket | None = None
        self.directory = Path(config.output_root).expanduser().resolve() / (
            "session-" + uuid.uuid4().hex
        )
        self.directory.mkdir(parents=True)
        self.log_path = self.directory / "simulator.log"
        self.metadata: dict[str, Any] = {}
        self._record("starting")
        child_channel = None
        try:
            # Preserve the venv symlink: resolving it selects the base interpreter.
            python = Path(config.python_executable).expanduser().absolute()
            if not python.is_file() or not os.access(python, os.X_OK):
                raise ValueError(f"simulator interpreter is not executable: {python}")
            app_root = (
                Path(config.geniesim_root).expanduser().resolve()
                / "source/geniesim_benchmark/src/geniesim_benchmark/app"
            )
            if not (app_root / "robot_cfg").is_dir():
                raise ValueError(
                    f"Genie robot configuration directory missing: {app_root}"
                )
            # Upstream Lula resolves robot_cfg beside __main__.__file__.
            (self.directory / "robot_cfg").symlink_to(
                app_root / "robot_cfg", target_is_directory=True
            )
            bootstrap = self.directory / "bootstrap.py"
            bootstrap.write_text(_BOOTSTRAP)
            package = (_worker_package or Path(__file__).parent).resolve()
            self._channel, child_channel = socket.socketpair()
            environment = self._environment(python)
            with self.log_path.open("wb") as log:
                self._process = subprocess.Popen(
                    [
                        str(python),
                        str(bootstrap),
                        str(package),
                        "--connection-fd",
                        str(child_channel.fileno()),
                        "--run-directory",
                        str(self.directory),
                    ],
                    pass_fds=(child_channel.fileno(),),
                    start_new_session=True,
                    cwd=self.directory,
                    env=environment,
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
            child_channel.close()
            child_channel = None
            deadline = time.monotonic() + config.startup_timeout_seconds
            send_packet(
                self._channel,
                {"protocol": PROTOCOL_VERSION, "config": config.to_dict()},
                deadline,
            )
            ready = receive_packet(self._channel, deadline)
            if (
                ready.get("protocol") != PROTOCOL_VERSION
                or ready.get("kind") != "ready"
                or ready.get("pid") != self.pid
            ):
                raise GenieSimProcessError(f"invalid Genie startup response: {ready}")
            self.metadata = ready
            self._record("ready")
        except BaseException as error:
            self._abort(str(error))
            if not isinstance(error, Exception):
                raise
            raise GenieSimProcessError(
                f"Genie startup failed: {error}; log: {self.log_path}"
            ) from error
        finally:
            if child_channel is not None:
                child_channel.close()

    @property
    def pid(self) -> int | None:
        return self._process.pid if self._process is not None else None

    def _environment(self, python: Path) -> dict[str, str]:
        environment = dict(os.environ)
        visible = environment.get("CUDA_VISIBLE_DEVICES")
        if visible is not None and str(self.config.gpu_id) not in [
            part.strip() for part in visible.split(",")
        ]:
            raise ValueError("gpu_id must belong to the inherited CUDA_VISIBLE_DEVICES")
        for name in ("PYTHONPATH", "PYTHONHOME", "CUDA_VISIBLE_DEVICES"):
            environment.pop(name, None)
        simulator = python.parent.parent / "lib/python3.11/site-packages/isaacsim"
        libraries = list(self.config.library_paths)
        libraries.append(str(simulator / "exts/isaacsim.ros2.bridge/humble/lib"))
        if environment.get("LD_LIBRARY_PATH"):
            libraries.append(environment["LD_LIBRARY_PATH"])
        cache = Path(self.config.output_root).expanduser().resolve() / "cache"
        environment.update(
            PYTHONNOUSERSITE="1",
            PYTHONUNBUFFERED="1",
            OMNI_KIT_ACCEPT_EULA="YES",
            CUDA_DEVICE_ORDER="PCI_BUS_ID",
            LD_LIBRARY_PATH=os.pathsep.join(libraries),
            ROS_DISTRO="humble",
            RMW_IMPLEMENTATION="rmw_fastrtps_cpp",
            XDG_CACHE_HOME=str(cache / "xdg"),
            XDG_CONFIG_HOME=str(cache / "config"),
            CUDA_CACHE_PATH=str(cache / "cuda"),
            TORCH_HOME=str(cache / "torch"),
        )
        return environment

    def request(
        self, method: str, payload: dict[str, Any], *, timeout: float | None = None
    ) -> dict[str, Any]:
        with self._lock:
            if self._closed or self._channel is None:
                raise GenieSimProcessError("Genie subprocess is closed")
            self._sequence += 1
            deadline = time.monotonic() + (
                self.config.request_timeout_seconds if timeout is None else timeout
            )
            try:
                send_packet(
                    self._channel,
                    {"id": self._sequence, "method": method, "payload": payload},
                    deadline,
                )
                response = receive_packet(self._channel, deadline)
                if response.get("id") != self._sequence:
                    raise GenieSimProcessError("Genie response sequence mismatch")
                if response.get("ok") is not True:
                    raise GenieSimProcessError(str(response.get("error", response)))
                result = response.get("result")
                if not isinstance(result, dict):
                    raise GenieSimProcessError("invalid Genie operation result")
                return result
            except BaseException as error:
                # The action may already have executed. Never replay or reuse this process.
                self._abort(f"{method}: {error}")
                if not isinstance(error, Exception):
                    raise
                raise GenieSimProcessError(
                    f"Genie {method} failed: {error}; log: {self.log_path}"
                ) from error

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            try:
                response = self.request(
                    "close", {}, timeout=self.config.close_timeout_seconds
                )
                if response.get("closing") is not True:
                    raise GenieSimProcessError("Genie did not acknowledge close")
                code = self._process.wait(timeout=self.config.close_timeout_seconds)
                if code != 0:
                    raise GenieSimProcessError(f"Genie exited with code {code}")
                self._closed = True
                self._channel.close()
                self._record("closed", forced=False)
            except BaseException as error:
                if not self._closed:
                    self._abort(f"close: {error}")
                if not isinstance(error, Exception):
                    raise
                raise GenieSimProcessError(
                    f"Genie close failed: {error}; log: {self.log_path}"
                ) from error

    def _abort(self, reason: str) -> None:
        process = self._process
        forced = process is not None and process.poll() is None
        if forced:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                # macOS can report EPERM for an exiting process group. The
                # bounded wait below must still prove that the child exited.
                pass
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass
                process.wait(timeout=5)
        if self._channel is not None:
            self._channel.close()
        self._closed = True
        self._record("failed", forced=forced, error=reason)

    def abort(self, reason: str) -> None:
        """Invalidate a process after a malformed native observation or result."""
        with self._lock:
            if not self._closed:
                self._abort(reason)

    def _record(self, status: str, **details: Any) -> None:
        record = {
            "status": status,
            "pid": self.pid,
            "returncode": self._process.poll() if self._process is not None else None,
            **details,
        }
        (self.directory / "process.json").write_text(
            json.dumps(record, indent=2) + "\n"
        )
