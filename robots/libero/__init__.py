"""LIBERO environment extension."""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from robots.libero.prompt_bundle import (
    system_prompt,
    user_prompt,
)
from zetta.envs.env_spec import EnvSpec, RunConfig
from zetta.envs.prompt_bundle import PromptBundle
from zetta.utils.config import get_repo_root

if TYPE_CHECKING:
    from zetta.utils.daemon import ProcessDaemon
    from zetta.utils.rpc import RpcClient


def get_env_spec() -> EnvSpec:
    """Return the LIBERO env identity, prompt bundle, and runner hooks.

    Tool schemas, handlers, server lifecycle, and the MCP allowlist live on
    the LIBERO toolkit (see :func:`get_toolkit`).
    """
    return EnvSpec(
        name="libero",
        prompts=PromptBundle(
            system=system_prompt,
            user=user_prompt,
        ),
        add_cli_args=_add_cli_args,
        parse_config=_parse_config,
        init_runtime=_init_runtime,
    )


def get_toolkit(
    *,
    primitives_kwargs: dict[str, Any],
    video_path: str | None = None,
    dashboard: Any = None,
):
    """Return the LIBERO toolkit (common tools + LIBERO primitives)."""
    from robots.libero.toolkit import LiberoToolkit

    tool_profile = str(primitives_kwargs.pop("_tool_profile", "all"))
    return LiberoToolkit(
        primitives_kwargs=primitives_kwargs,
        video_path=video_path,
        dashboard=dashboard,
        tool_profile=tool_profile,
    )


def _add_cli_args(parser: argparse.ArgumentParser, use_dashboard: bool) -> None:
    """Register LIBERO CLI flags on the shared ``parser``.

    When ``use_dashboard`` is True, ``--suite`` / ``--task`` are made optional
    because the dashboard launcher will fill them in before ``_parse_config``
    validates. Under CLI-only, they are required — argparse errors out early
    with the usual usage message.
    """
    required = not use_dashboard
    parser.add_argument("--max-episode-steps", type=int, default=10000)
    parser.add_argument("--libero-type", default=None,
                        choices=["standard", "pro", "plus"],
                        help="LIBERO variant (auto-routed from suite suffix if not set).")
    parser.add_argument("--suite", default=None, required=required,
                        help="e.g. libero_object_task, libero_spatial_swap")
    parser.add_argument("--task", type=int, default=None, required=required)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--env-endpoint", default=None,
                        help="[protocol://]host:port of an existing env_server "
                             "(protocol=http|socket, defaults to http). "
                             "If unset, a local env_server is spawned.")
    parser.add_argument("--vla-endpoint", default=os.environ.get("ZETTA_VLA_ENDPOINT"),
                        help="[protocol://]host:port of an existing vla_server "
                             "(protocol=http|socket, defaults to http). "
                             "Defaults to ZETTA_VLA_ENDPOINT; if both are unset, "
                             "a local vla_server is spawned.")
    parser.add_argument("--sam3-endpoint", default=None,
                        help="[protocol://]host:port of an existing SAM3 server "
                             "(protocol=http|socket, defaults to http). "
                             "If unset, a local SAM3 server is spawned.")
    parser.add_argument(
        "--disable-sam3",
        action="store_true",
        help=(
            "Do not launch SAM3; segment returns a structured unavailable result. "
            "Use only when gated SAM3 weights are unavailable."
        ),
    )
    parser.add_argument("--cuda-device", type=int, default=None,
                        help="GPU device to expose via CUDA_VISIBLE_DEVICES.")
    parser.add_argument(
        "--tool-profile",
        choices=("all", "pi05_only"),
        default="all",
        help="Frozen planner-visible robotic tool profile for controlled studies.",
    )


def _parse_config(args: argparse.Namespace) -> RunConfig:
    """Validate final ``args`` and derive per-run identifiers.

    Under ``--dashboard``, ``_add_cli_args`` left ``--suite`` / ``--task``
    optional so the dashboard could fill them; this is where we enforce
    they're set now that any overrides have been applied.
    """
    if not args.suite:
        raise ValueError("--suite is required")
    if args.task is None:
        raise ValueError("--task is required")
    if args.disable_sam3 and args.sam3_endpoint is not None:
        raise ValueError("--disable-sam3 and --sam3-endpoint are mutually exclusive")

    recipe_tag = f"{args.suite.replace('libero_', '')}_t{args.task}_s{args.seed}"
    prompt_vars = {
        "suite": args.suite,
        "task": args.task,
        "seed": args.seed,
        "recipe_tag": recipe_tag,
    }

    output_dir = args.output_dir
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d-%H:%M:%S")
        output_dir = get_repo_root() / "logs" / f"{timestamp}_{args.suite}_t{args.task}_s{args.seed}"
    output_dir = Path(output_dir)

    dashboard_state = None
    if getattr(args, "dashboard", False):
        from robots.libero.tools import artifact_path
        from zetta.dashboard.state import State

        dashboard_state = State(
            run_id=f"{args.suite}/{output_dir.name}",
            name=recipe_tag,
            suite=args.suite,
            task=args.task,
            seed=args.seed,
            output_dir=str(output_dir),
            video_path=str(
                artifact_path(output_dir, "episode_video")
            ),
        )
    return RunConfig(
        recipe_tag=recipe_tag,
        output_dir=output_dir,
        prompt_vars=prompt_vars,
        dashboard_state=dashboard_state,
        task_desc={"suite": args.suite, "task": args.task, "seed": args.seed},
    )


def _subprocess_env(**extra: str) -> dict[str, str]:
    """Build the env dict for a subprocess: inherit from parent, layer extras on top.

    CUDA device selection is passed via ``--cuda-device`` on the server command
    line — the server itself handles ``CUDA_VISIBLE_DEVICES`` and EGL alignment.
    """
    env = os.environ.copy()
    env.update(extra)
    return env


def _init_runtime(
    args: argparse.Namespace,
    output_dir: Path,
) -> tuple[list[ProcessDaemon], dict[str, Any]]:
    """Spawn env + vla + SAM3 daemons and build clients for LIBERO.

    Each server can be spawned or attached-to independently: pass an
    endpoint to attach, or leave it unset to spawn a local subprocess.

    Heavy deps (rpc / vla / daemon / env_client) are imported lazily so
    that a bare ``import robots.libero`` (for ``get_env_spec`` /
    ``get_toolkit``) doesn't drag them in.
    """
    from robots.libero.env_client import LiberoEnvClient
    from zetta.utils.config import get_libero_type
    from zetta.utils.daemon import ProcessDaemon, pick_free_port
    from zetta.utils.http_rpc import HttpRpcClient
    from zetta.utils.rpc import parse_endpoint, wait_for_ready
    from zetta.utils.sam3_client import Sam3Client, UnavailableSam3Client
    from zetta.utils.socket_rpc import SocketRpcClient
    from zetta.utils.vla_client import VLAClient

    daemons: list[ProcessDaemon] = []
    libero_type = args.libero_type or get_libero_type()
    _cuda_args = ["--cuda-device", str(args.cuda_device)] if args.cuda_device is not None else []

    # --- env_server --------------------------------------------------------
    if args.env_endpoint is None:
        host, port = "127.0.0.1", pick_free_port()
        env_daemon = ProcessDaemon(
            name="env_server",
            cmd=[
                sys.executable,
                str(get_repo_root() / "robots" / "libero" / "env_server.py"),
                "--suite", args.suite,
                "--task", str(args.task),
                "--seed", str(args.seed),
                "--max-episode-steps", str(args.max_episode_steps),
                "--transport", "http",
                "--host", host,
                "--port", str(port),
                "--parent-watch",
                *_cuda_args,
            ],
            env=_subprocess_env(
                LIBERO_TYPE=libero_type,
                MUJOCO_GL="egl",
                ROBOT_PLATFORM="LIBERO",
            ),
            log_path=str(Path(output_dir) / "env_server.log"),
        )
        env_daemon.start()
        daemons.append(env_daemon)
        env_rpc: RpcClient = HttpRpcClient(f"http://{host}:{port}")
        wait_for_ready(env_rpc, daemon=env_daemon)
    else:
        protocol, host, port = parse_endpoint(args.env_endpoint)
        if protocol == "socket":
            env_rpc = SocketRpcClient(host, port)
        elif protocol == "http":
            env_rpc = HttpRpcClient(f"http://{host}:{port}")
        else:
            raise ValueError(
                f"--env-endpoint protocol must be socket or http, got {protocol!r}"
            )
        wait_for_ready(env_rpc)

    # --- vla_server --------------------------------------------------------
    if args.vla_endpoint is None:
        host, port = "127.0.0.1", pick_free_port()
        vla_daemon = ProcessDaemon(
            name="vla_server",
            cmd=[
                sys.executable,
                str(get_repo_root() / "robots" / "libero" / "vla_server.py"),
                "--transport", "http",
                "--host", host,
                "--port", str(port),
                "--parent-watch",
                *_cuda_args,
            ],
            env=_subprocess_env(),
            log_path=str(Path(output_dir) / "vla_server.log"),
        )
        vla_daemon.start()
        daemons.append(vla_daemon)
        vla_rpc: RpcClient = HttpRpcClient(f"http://{host}:{port}")
        wait_for_ready(vla_rpc, daemon=vla_daemon)
    else:
        protocol, host, port = parse_endpoint(args.vla_endpoint)
        if protocol == "socket":
            vla_rpc = SocketRpcClient(host, port)
        elif protocol == "http":
            vla_rpc = HttpRpcClient(f"http://{host}:{port}")
        else:
            raise ValueError(
                f"--vla-endpoint protocol must be socket or http, got {protocol!r}"
            )
        wait_for_ready(vla_rpc)

    # --- sam3_server -------------------------------------------------------
    sam3_client: Sam3Client | UnavailableSam3Client
    if args.disable_sam3:
        sam3_client = UnavailableSam3Client(
            "SAM3 disabled by --disable-sam3; use calibrated images/back_project"
        )
    elif args.sam3_endpoint is None:
        host, port = "127.0.0.1", pick_free_port()
        sam3_daemon = ProcessDaemon(
            name="sam3_server",
            cmd=[
                sys.executable,
                str(get_repo_root() / "robots" / "libero" / "sam3_server.py"),
                "--transport", "http",
                "--host", host,
                "--port", str(port),
                "--parent-watch",
                *_cuda_args,
            ],
            env=_subprocess_env(),
            log_path=str(Path(output_dir) / "sam3_server.log"),
        )
        sam3_daemon.start()
        daemons.append(sam3_daemon)
        sam3_rpc: RpcClient = HttpRpcClient(f"http://{host}:{port}")
        wait_for_ready(sam3_rpc, daemon=sam3_daemon)
        sam3_client = Sam3Client(sam3_rpc)
    else:
        protocol, host, port = parse_endpoint(args.sam3_endpoint)
        if protocol == "socket":
            sam3_rpc = SocketRpcClient(host, port)
        elif protocol == "http":
            sam3_rpc = HttpRpcClient(f"http://{host}:{port}")
        else:
            raise ValueError(
                f"--sam3-endpoint protocol must be socket or http, got {protocol!r}"
            )
        wait_for_ready(sam3_rpc)
        sam3_client = Sam3Client(sam3_rpc)

    primitives_kwargs = {
        "env": LiberoEnvClient(
            env_rpc,
            expected_meta={
                "suite": args.suite,
                "task": args.task,
                "seed": args.seed,
                "max_episode_steps": args.max_episode_steps,
            },
        ),
        "model": VLAClient(vla_rpc),
        "sam3_client": sam3_client,
        "_tool_profile": args.tool_profile,
    }
    return daemons, primitives_kwargs
