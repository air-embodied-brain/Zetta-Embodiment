"""Smoke-test Zetta's MCP bridge and Codex planner without a simulator episode."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

import openai_codex

from zetta.planner.codex import CodexPlanner
from zetta.planner.utils.http_mcp_server import HttpMcpServer
from zetta.tools.toolkit import Toolkit
from zetta.utils.logging import init_output_dir


def _write_result(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--model", default=os.environ.get("CODEX_MODEL", "gpt-5.6-terra"))
    parser.add_argument("--reasoning-effort", default=os.environ.get("CODEX_REASONING_EFFORT", "medium"))
    parser.add_argument("--timeout-seconds", type=int, default=300)
    parser.add_argument(
        "--mcp-only",
        action="store_true",
        help="Check the in-process MCP lifecycle without making a model request.",
    )
    parser.add_argument(
        "--codex-mcp-list",
        action="store_true",
        help="Ask the configured Codex binary to list the temporary MCP server.",
    )
    parser.add_argument(
        "--codex-app-mcp-status",
        action="store_true",
        help="Inspect app-server's initialized MCP tools without a model request.",
    )
    args = parser.parse_args()

    output_dir = init_output_dir(args.output_dir.resolve())
    toolkit = Toolkit()
    result_path = output_dir / "smoke_result.json"

    if args.mcp_only or args.codex_mcp_list or args.codex_app_mcp_status:
        server = HttpMcpServer(toolkit)
        try:
            server.start()
            if args.codex_app_mcp_status:
                diagnostic_planner = CodexPlanner(
                    output_dir=str(output_dir),
                    repo_root=args.repo_root.resolve(),
                    timeout_s=args.timeout_seconds,
                    model=args.model,
                )
                with openai_codex.Codex(
                    config=diagnostic_planner._build_config(server.url)
                ) as codex:
                    thread = codex.thread_start(
                        approval_mode=openai_codex.ApprovalMode.deny_all,
                        cwd=str(args.repo_root.resolve()),
                        model=args.model,
                        sandbox=openai_codex.Sandbox.full_access,
                    )
                    status = codex._client._request_raw(
                        "mcpServerStatus/list",
                        {
                            "threadId": thread.id,
                            "detail": "toolsAndAuthOnly",
                            "limit": 100,
                        },
                    )
                    app_server_stderr = list(codex._client._stderr_lines)
                _write_result(
                    result_path,
                    {
                        "mcp": "ok",
                        "app_mcp_status": status,
                        "app_server_stderr": app_server_stderr[-100:],
                        "simulator_episodes": 0,
                    },
                )
                print(result_path)
                return 0
            if args.codex_mcp_list:
                codex_bin = os.environ.get("CODEX_BIN")
                if not codex_bin:
                    raise RuntimeError("CODEX_BIN is required with --codex-mcp-list")
                completed = subprocess.run(
                    [
                        codex_bin,
                        "--config",
                        f"mcp_servers.zetta.url={json.dumps(server.url)}",
                        "mcp",
                        "list",
                        "--json",
                    ],
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                listed = json.loads(completed.stdout)
                payload = {
                    "mcp": "ok",
                    "codex_mcp_list_exit_code": completed.returncode,
                    "codex_mcp_servers": listed,
                    "codex_mcp_stderr": completed.stderr[-2000:],
                    "simulator_episodes": 0,
                }
                _write_result(result_path, payload)
                print(result_path)
                return 0 if completed.returncode == 0 else 2
        finally:
            server.stop()
        _write_result(result_path, {"mcp": "ok", "simulator_episodes": 0})
        print(result_path)
        return 0

    if not os.environ.get("CODEX_API_KEY"):
        raise RuntimeError("CODEX_API_KEY is required")
    if args.model != "gpt-5.6-terra":
        raise ValueError("this acceptance smoke must use gpt-5.6-terra")
    if args.reasoning_effort != "medium":
        raise ValueError("this acceptance smoke must use reasoning effort medium")
    os.environ["CODEX_REASONING_EFFORT"] = args.reasoning_effort

    planner = CodexPlanner(
        output_dir=str(output_dir),
        repo_root=args.repo_root.resolve(),
        timeout_s=args.timeout_seconds,
        output_path=output_dir / "codex_smoke.txt",
        model=args.model,
    )
    result = planner.solve(
        system_prompt=(
            "You are validating Zetta's planner-to-tool bridge. Your only valid action "
            "is a real MCP tool call to the Zetta finish tool "
            "(mcp__zetta__finish/finish) exactly once with status='success' and a short "
            "summary stating that the MCP bridge works. Do not answer in text, do not "
            "claim success without the tool result, do not read or write files, and do "
            "not run shell commands."
        ),
        user_message="Perform the requested bridge validation now.",
        toolkit=toolkit,
        max_turns=2,
    )
    finish = result.finish_result or {}
    payload: dict[str, object] = {
        "backend": result.stats.get("backend"),
        "model": result.stats.get("model"),
        "reasoning_effort": result.stats.get("reasoning_effort"),
        "provider": result.stats.get("provider"),
        "finish_status": finish.get("status"),
        "finish_summary": finish.get("summary"),
        "tool_calls": result.stats.get("tool_calls"),
        "turns_used": result.stats.get("turns_used"),
        "total_input_tokens": result.stats.get("total_input_tokens"),
        "total_cached_input_tokens": result.stats.get("total_cached_input_tokens"),
        "total_output_tokens": result.stats.get("total_output_tokens"),
        "elapsed_s": result.stats.get("elapsed_s"),
        "error": result.error,
        "simulator_episodes": 0,
    }
    _write_result(result_path, payload)
    print(result_path)
    return 0 if finish.get("status") == "success" and result.error is None else 2


if __name__ == "__main__":
    raise SystemExit(main())
