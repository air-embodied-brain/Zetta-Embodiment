# Copyright (c) 2026 Zetta Contributors
"""Isaac main thread and the native task request loop in an isolated interpreter."""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import os
import socket
import sys
import threading
import time
import traceback
from pathlib import Path

from .config import ASSET_REVISION, GENIESIM_REVISION, ISAACSIM_VERSION, GenieSimConfig
from .wire import PROTOCOL_VERSION, receive_packet, send_packet


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--connection-fd", type=int, required=True)
    parser.add_argument("--run-directory", type=Path, required=True)
    args = parser.parse_args()
    channel = socket.socket(fileno=args.connection_fd)
    hello = receive_packet(channel, time.monotonic() + 60)
    if hello.get("protocol") != PROTOCOL_VERSION:
        raise ValueError("unsupported Genie IPC version")
    config = GenieSimConfig.from_mapping(hello["config"])
    if sys.version_info[:2] != (3, 11):
        raise RuntimeError("the Genie profile requires Python 3.11")
    version = importlib.metadata.version("isaacsim")
    if version != ISAACSIM_VERSION:
        raise RuntimeError(f"expected Isaac {ISAACSIM_VERSION}, found {version}")
    os.environ["SIM_REPO_ROOT"] = config.geniesim_root
    os.environ["SIM_ASSETS"] = config.assets_root
    sys.path.insert(
        0, str(Path(config.geniesim_root) / "source/geniesim_benchmark/src")
    )
    spec = importlib.util.spec_from_file_location(
        "geniesim_assets",
        Path(config.assets_root) / "__init__.py",
        submodule_search_locations=[config.assets_root],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["geniesim_assets"] = module
    spec.loader.exec_module(module)

    tokens = {}
    for name in ("data", "cache", "logs", "documents"):
        path = args.run_directory / "kit" / name
        path.mkdir(parents=True, exist_ok=True)
        tokens[name] = path
        tokens["omni_" + name] = path
    tokens["shared_documents"] = tokens["documents"]
    from isaacsim import SimulationApp

    app = SimulationApp(
        {
            "headless": True,
            "active_gpu": config.gpu_id,
            "physics_gpu": config.gpu_id,
            "multi_gpu": False,
            "width": 640,
            "height": 480,
            "limit_cpu_threads": 8,
            "extra_args": [
                f"--/app/tokens/{name}={path}" for name, path in tokens.items()
            ],
        }
    )
    done = threading.Event()
    failure = []
    close_request = []
    api = None
    session = None

    def record_failure() -> None:
        error = traceback.format_exc()
        failure.append(error)
        print(error, flush=True)
        (args.run_directory / "failure.json").write_text(
            json.dumps({"error": error}) + "\n"
        )

    try:
        import carb
        from isaacsim.core.api import World
        from isaacsim.core.utils import extensions

        extensions.enable_extension("isaacsim.ros2.bridge")
        from geniesim_benchmark.app.controllers.api_core import APICore
        from geniesim_benchmark.app.workflow.ui_builder import UIBuilder
        from geniesim_benchmark.config.params import Config

        from .session import GenieSimSession

        settings = carb.settings.get_settings()
        settings.set("/rtx/dataWindow/fitOutputToDataWindow", True)
        for index, value in enumerate((0.0, 0.0, 1.0, 1.0)):
            settings.set(f"/rtx/dataWindowNDC/{index}", value)
        cfg = Config()
        cfg.app.headless = True
        cfg.app.enable_ros = False
        cfg.benchmark.task_name = "table_task_1_g2_op"
        cfg.benchmark.sub_task_name = "pick_block_color"
        cfg.benchmark.model_arc = "corobot"
        world = World(
            stage_units_in_meters=1.0, physics_dt=1 / 120, rendering_dt=1 / 60
        )
        api = APICore(UIBuilder(world), cfg)

        def physics_tick(step_size):
            api.physics_step()
            api.on_ros_tick(step_size)

        world.add_physics_callback("zetta_geniesim", callback_fn=physics_tick)
        session = GenieSimSession(config, cfg, api, args.run_directory)

        def requests():
            request = {}
            try:
                send_packet(
                    channel,
                    {
                        "protocol": PROTOCOL_VERSION,
                        "kind": "ready",
                        "pid": os.getpid(),
                        "isaacsim_version": version,
                        "expected_geniesim_revision": GENIESIM_REVISION,
                        "expected_asset_revision": ASSET_REVISION,
                    },
                    time.monotonic() + 60,
                )
                while True:
                    request = receive_packet(channel, time.monotonic() + 86400)
                    method, payload = request["method"], request["payload"]
                    if method == "close":
                        session.close()
                        close_request.append(request["id"])
                        break
                    if method == "reset":
                        result = session.reset(payload["seed"])
                    elif method == "step":
                        result = session.step(payload["actions"])
                    elif method == "result":
                        result = session.result()
                    else:
                        raise ValueError(f"unknown Genie method: {method}")
                    send_packet(
                        channel,
                        {"id": request["id"], "ok": True, "result": result},
                        time.monotonic() + 60,
                    )
            except BaseException:
                record_failure()
                try:
                    send_packet(
                        channel,
                        {"id": request.get("id"), "ok": False, "error": failure[-1]},
                        time.monotonic() + 5,
                    )
                except (OSError, TimeoutError):
                    pass
            finally:
                done.set()

        api.request_render()
        api._startup_render_held = True
        thread = threading.Thread(
            target=requests, name="geniesim-requests", daemon=True
        )
        thread.start()
        while app.is_running() and not done.is_set():
            world.step(render=api.frame_render_enabled)
            api.render_step()
        if not done.is_set():
            raise RuntimeError("Isaac stopped before the request loop completed")
        thread.join(timeout=5)
    except BaseException:
        record_failure()
    finally:
        try:
            if api is not None:
                api.stop_all_recording()
                api.shutdown_ros()
            if close_request and not failure:
                send_packet(
                    channel,
                    {
                        "id": close_request[0],
                        "ok": True,
                        "result": {"closing": True},
                    },
                    time.monotonic() + 5,
                )
        except BaseException:
            record_failure()
        # Isaac's default fast shutdown exits the interpreter. The supervisor
        # requires both the close acknowledgement and exit 0 before success.
        app.close()


if __name__ == "__main__":
    main()
