"""Physical agent main CLI entrypoint."""
# `rpent/cli/`
#
# CLI entrypoints for RPent (currently just `main.py`).
#
# ## Run
#
# `main()` is exposed as the `rpent` console script (see `[project.scripts]`
# in `pyproject.toml`):
#
# ```bash
# rpent --env libero --suite libero_object_task --task 0 --seed 0 [...]
# ```
#
# ## Note
#
# Do not import `rpent.cli` from other `rpent` modules. `main.py` pulls in
# `rpent.planner`, `rpent.envs`, `rpent.utils`, `rpent.dashboard`, and
# `rpent.tools`, so importing the CLI back into any of them would create an
# import cycle. Nothing else should depend on this package.
from __future__ import annotations

import argparse
import json
import os
import queue
import shlex
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path

from rpent.cli.tui import (
    start_first_prompt_resolver,
    start_interactive_reader,
)
from rpent.envs import get_env_spec, get_toolkit
from rpent.planner.base import build_planner
from rpent.utils.logging import get_logger, init_output_dir
from rpent.utils.resources import ensure_resources

logger = get_logger("agent")


# ---------------------------------------------------------------------------
# API agent transcript serialization
# ---------------------------------------------------------------------------


def _strip_images(value):
    """Return a copy of ``value`` with inline image payloads omitted.

    SDK objects are left untouched; ``json.dump(..., default=str)`` handles
    them at write time. Only the bulky base64 image blocks are replaced.
    """
    if isinstance(value, list):
        return [_strip_images(v) for v in value]
    if isinstance(value, dict):
        if value.get("type") == "image":
            return {"type": "image", "source": {"_omitted_for_transcript": True}}
        if value.get("type") == "image_url":
            return {"type": "image_url", "image_url": {"_omitted_for_transcript": True}}
        return {k: _strip_images(v) for k, v in value.items()}
    return value


def _serialize_messages(messages: list[dict]) -> list[dict]:
    """Strip inline image payloads from messages before writing the transcript."""
    return [
        {**{k: v for k, v in m.items() if k != "content"},
         "content": _strip_images(m.get("content"))}
        for m in messages
    ]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Standalone hybrid LLM-in-the-loop agent for LIBERO PRO",
    )

    ap.add_argument("--env", dest="env_name", required=True, choices=["libero"],
                    help="Environment backend: libero.")

    # models
    ap.add_argument("--planner", default="api",
                    choices=["api", "claude_code", "codex"],
                    help="LLM backend: api | claude_code | codex.")
    ap.add_argument("--model", default=None,
                    help="Model id. For the 'api' planner, prefix the provider "
                         "(e.g. anthropic:claude-opus-4-8, openai:gpt-5.5, "
                         "openai-chat:glm-5.2). For claude_code/codex this "
                         "overrides the backend default model.")
    ap.add_argument("--base-url", default=None,
                    help="API base URL. Defaults to the selected backend's base URL env var.")
    ap.add_argument("--max-turns", type=int, default=100)
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--no-images", action="store_true",
                    help="Never send image bytes to the model (api planner only). "
                         "Use for text-only models that reject image input "
                         "(e.g. 400 \"message type 'image_url' is not supported\"); "
                         "read_image then returns the file path with a notice.")
    ap.add_argument("--planner-timeout-s", type=int, default=None,
                    help="Wall-clock cap for the claude_code/codex planner "
                         "subprocess. Defaults to CODEX_TIMEOUT_S (codex only), "
                         "CELL_TIMEOUT_S, or 1200.")
    ap.add_argument("--claude-code-max-budget-usd", type=float, default=None,
                    help="Budget passed to claude -p --max-budget-usd. "
                         "Defaults to MAX_BUDGET_USD env or 10.")

    # other config
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--dashboard", action="store_true",
                    help="Start a local dashboard server for this single run.")
    ap.add_argument("--dashboard-host", default="127.0.0.1",
                    help="Dashboard bind host. Defaults to 127.0.0.1.")
    ap.add_argument("--dashboard-port", type=int, default=0,
                    help="Dashboard port. 0 asks the OS for a free port.")
    ap.add_argument("--dashboard-language", choices=["en", "zh-cn"], default="en",
                    help="Dashboard UI language. 'zh-cn' serves the Chinese "
                         "variant (index.zh-cn.html); defaults to English.")
    ap.add_argument("--verbose", action="store_true",
                    help="Enable DEBUG-level logging for stdout and the run.log "
                         "file. Defaults to INFO when not set.")
    ap.add_argument(
        "--task-memory-snapshot",
        default=None,
        help=(
            "Immutable task-memory JSON snapshot injected into the initial prompt. "
            "The mutable store is never exposed to the episode planner."
        ),
    )
    ap.add_argument(
        "--episode-memory-frozen",
        action="store_true",
        help=(
            "Enforce episode-level memory immutability: Codex runs read-only and "
            "write_text_file is limited to this run's output_dir."
        ),
    )
    ap.add_argument("--interactive", "-i", action="store_true",
                    help="Interactive mode: opens an interactive cli session.")

    return ap


def main() -> int:
    parser = _build_argparser()
    # Two-phase argparse: first grab --env / --dashboard so we know which
    # env's flags to add and whether to make its required flags optional.
    early, _ = parser.parse_known_args()

    env_spec = get_env_spec(early.env_name)
    env_spec.add_cli_args(parser, use_dashboard=early.dashboard)
    args = parser.parse_args()

    # With --dashboard, open the launcher first: serve the start screen, then
    # block until the user clicks Run and overlay their choices onto args.
    # parse_config runs afterwards so validation + derivation see the final
    # config.
    dashboard_server = None
    dashboard_url = None
    launch_config = None
    if args.dashboard:
        from rpent.dashboard import DashboardServer
        from rpent.dashboard.launcher import apply_to_args, defaults_from_args

        dashboard_server = DashboardServer(
            host=args.dashboard_host, port=args.dashboard_port,
            language=args.dashboard_language,
        )
        dashboard_url = dashboard_server.start()
        # The run directory is not final until the launcher form is submitted, so
        # print the pre-launch URL without initializing the run.log file handler.
        print(
            f"Dashboard: {dashboard_url}. "
            "Open it, adjust the run config, and click Run to start.",
            flush=True,
        )
        launch_config = dashboard_server.wait_for_launch(
            defaults=defaults_from_args(args)
        )
        apply_to_args(args, launch_config)

    run_config = env_spec.parse_config(args)
    recipe_tag = run_config.recipe_tag
    output_dir = run_config.output_dir
    prompt_vars = run_config.prompt_vars
    dashboard_state = run_config.dashboard_state
    task_desc = run_config.task_desc

    env_name = args.env_name

    if args.episode_memory_frozen and not args.task_memory_snapshot:
        raise ValueError("--episode-memory-frozen requires --task-memory-snapshot")
    memory_snapshot: dict | None = None
    memory_snapshot_meta: dict | None = None
    if args.task_memory_snapshot:
        from rpent.memory.task_memory import (
            canonical_sha256,
            render_episode_memory,
            validate_task_memory,
        )

        memory_path = Path(args.task_memory_snapshot).expanduser().resolve()
        memory_snapshot = validate_task_memory(
            json.loads(memory_path.read_text(encoding="utf-8"))
        )
        if memory_snapshot["suite"] != str(args.suite):
            raise ValueError(
                "task-memory suite mismatch: "
                f"expected {args.suite!r}, got {memory_snapshot['suite']!r}"
            )
        if int(memory_snapshot["task"]) != int(args.task):
            raise ValueError(
                "task-memory task mismatch: "
                f"expected {args.task}, got {memory_snapshot['task']}"
            )
        memory_snapshot_meta = {
            "path": str(memory_path),
            "task_key": memory_snapshot["task_key"],
            "revision": memory_snapshot["revision"],
            "sha256": canonical_sha256(memory_snapshot),
            "frozen": bool(args.episode_memory_frozen),
        }
        if args.episode_memory_frozen:
            os.environ["RPENT_EPISODE_MEMORY_FROZEN"] = "1"

    # mkdir + logging wiring (env-side already picked the path).
    output_dir = init_output_dir(output_dir, verbose=args.verbose)
    # Now that output_dir is fixed, repeat launcher details into this run's log.
    if dashboard_url is not None:
        logger.info("Dashboard: %s", dashboard_url)
    if launch_config is not None:
        logger.info("launcher config applied: %s", launch_config)
    logger.info("physical agent cmd: %s", shlex.join([sys.executable, *sys.argv]))

    ensure_resources(env_name)

    # --- dashboard state ---------------------------------------------------
    if dashboard_state is not None and dashboard_server is not None:
        # Server is already serving the launcher; register the run so the
        # frontend can switch from the start screen to the live monitor.
        dashboard_server.register(dashboard_state)

    planner = build_planner(
        args.planner,
        output_dir=output_dir,
        recipe_tag=recipe_tag,
        env_name=env_name,
        base_url=args.base_url,
        model=args.model,
        max_tokens=args.max_tokens,
        planner_timeout_s=args.planner_timeout_s,
        claude_code_max_budget_usd=args.claude_code_max_budget_usd,
        dashboard=dashboard_state,
        no_images=args.no_images,
    )
    prompt_bundle = env_spec.prompts

    prompt_vars = {**prompt_vars, "output_dir": output_dir}
    system_prompt = prompt_bundle.render(
        "system",
        variables=prompt_vars,
    )
    user_msg = prompt_bundle.render(
        "user",
        variables=prompt_vars,
    )
    if memory_snapshot is not None:
        system_prompt = system_prompt.rstrip() + "\n\n" + render_episode_memory(
            memory_snapshot
        )

    input_queue: "queue.Queue[str | None] | None" = None
    await_first_prompt: "Callable[[], str | None] | None" = None
    if args.interactive:
        input_queue = queue.Queue()
        # Pre-fill the first prompt with the rendered default task (editable
        # preset);
        start_interactive_reader(input_queue, first_prompt_default=user_msg)
        logger.info(
            "interactive mode on: the built-in task is pre-filled — "
            "edit it and press Enter, submit it as-is, or clear it to "
            "type your own. Once running, type to steer the agent. "
            "/help for commands."
        )
        # Resolve the opening prompt on a background thread so the user can type
        # it while the (slow) env/VLA servers boot below.
        await_first_prompt = start_first_prompt_resolver(input_queue)

    # --- initialise environment --------------------------------------------
    daemons, primitives_kwargs = env_spec.init_runtime(args, output_dir)

    # --- toolkit -----------------------------------------------------------
    toolkit = get_toolkit(
        env_name,
        primitives_kwargs=primitives_kwargs,
        video_path=str(Path(output_dir) / "episode.mp4"),
        dashboard=dashboard_state,
    )

    # --- agent loop --------------------------------------------------------
    t0 = time.time()
    finish_result, messages, agent_error = None, [], None
    stats: dict = {}
    first_user_msg: str | None = user_msg
    if await_first_prompt is not None:
        # Block until the opening prompt typed during startup is ready.
        first_user_msg = await_first_prompt()
        if first_user_msg is None:
            logger.info("no task entered; ending session before start.")
    try:
        if first_user_msg is not None:
            result = planner.solve(
                system_prompt=system_prompt,
                user_message=first_user_msg,
                toolkit=toolkit,
                max_turns=args.max_turns,
                input_queue=input_queue,
            )
            finish_result = result.finish_result
            messages = result.messages
            stats = result.stats
            agent_error = result.error
    except Exception as e:
        logger.error("EXCEPTION in agent loop: %s", e)
    finally:
        # Agent-side: flush the episode video before the env+model
        recipe_path = toolkit.write_recipe(recipe_tag)
        logger.info("recipe: %s", recipe_path)

        toolkit.close()
        for d in daemons:
            d.stop()

    elapsed = time.time() - t0

    transcript_path = Path(output_dir) / f"transcript_{recipe_tag}.json"
    record = {
        **task_desc,
        "model": args.model,
        "elapsed_s": round(elapsed, 1),
        "finish": finish_result,
        "stats": stats,
        "messages": _serialize_messages(messages),
        "episode_memory": memory_snapshot_meta,
    }
    with open(transcript_path, "a") as f:
        json.dump(record, f, indent=2, default=str)

    logger.info("elapsed: %.1fs", elapsed)
    logger.info("usage: in=%s out=%s tool_calls=%s",
                 stats.get('total_input_tokens', '?'),
                 stats.get('total_output_tokens', '?'),
                 stats.get('tool_calls', '?'))
    logger.info("transcript: %s", transcript_path)
    if agent_error:
        logger.error("error: %s", agent_error)

    if args.dashboard and dashboard_state is not None:
        dashboard_state.mark_done()
        logger.info(
            "Run finished. Dashboard still serving at %s. Press Ctrl+C to stop.",
            dashboard_url,
        )
        try:
            threading.Event().wait()
        except KeyboardInterrupt:
            pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
