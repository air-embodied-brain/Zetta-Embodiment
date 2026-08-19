# Copyright (c) 2026 Zetta Contributors
"""Probe model channel availability without printing credentials or endpoints."""

from __future__ import annotations

import argparse
import json
import os
import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


def _post(url: str, key: str, payload: dict[str, Any]) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "User-Agent": "OpenAI/Python channel-probe",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=90) as response:
            body = json.loads(response.read())
            return {
                "http_status": response.status,
                "status": body.get("status"),
                "model": body.get("model"),
            }
    except urllib.error.HTTPError as exc:
        try:
            body = json.loads(exc.read())
        except Exception:
            body = {}
        error = body.get("error") if isinstance(body.get("error"), dict) else {}
        return {
            "http_status": exc.code,
            "error_code": error.get("code"),
            "error_type": error.get("type"),
        }
    except Exception as exc:
        # Exception messages can contain the complete endpoint. Persist only the
        # class so this probe remains safe to attach to an experiment artifact.
        return {"transport_error": type(exc).__name__}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-file", type=Path, required=True)
    parser.add_argument(
        "--base-url", default=os.environ.get("ZETTA_CHANNEL_PROBE_BASE_URL")
    )
    parser.add_argument("--model", default="gpt-5.6-sol")
    parser.add_argument("--reasoning-effort", default="high")
    parser.add_argument("--select-key-index", type=int)
    parser.add_argument("--env-file", type=Path)
    args = parser.parse_args()
    if not args.base_url:
        raise ValueError("--base-url or ZETTA_CHANNEL_PROBE_BASE_URL is required")

    text = args.api_file.read_text(encoding="utf-8", errors="replace")
    keys = list(dict.fromkeys(re.findall(r"sk-[A-Za-z0-9_-]{20,}", text)))
    if args.select_key_index is not None:
        if args.env_file is None:
            raise ValueError("--env-file is required with --select-key-index")
        offset = args.select_key_index - 1
        if not 0 <= offset < len(keys):
            raise ValueError("selected key index is out of range")
        env_text = args.env_file.read_text(encoding="utf-8")
        updated, count = re.subn(
            r"^export CODEX_API_KEY=.*$",
            "export CODEX_API_KEY=" + keys[offset],
            env_text,
            flags=re.MULTILINE,
        )
        if count != 1:
            raise RuntimeError("expected exactly one CODEX_API_KEY assignment")
        args.env_file.write_text(updated, encoding="utf-8")
        os.chmod(args.env_file, 0o600)
        print(json.dumps({"selected_key_index": args.select_key_index}))
        return 0

    results = []
    for index, key in enumerate(keys, 1):
        results.append(
            {
                "key_index": index,
                "responses": _post(
                    args.base_url.rstrip("/") + "/responses",
                    key,
                    {
                        "model": args.model,
                        "input": "Reply exactly OK.",
                        "reasoning": {"effort": args.reasoning_effort},
                        "max_output_tokens": 16,
                    },
                ),
                "chat_completions": _post(
                    args.base_url.rstrip("/") + "/chat/completions",
                    key,
                    {
                        "model": args.model,
                        "messages": [
                            {"role": "user", "content": "Reply exactly OK."}
                        ],
                        "reasoning_effort": args.reasoning_effort,
                        "max_tokens": 16,
                    },
                ),
            }
        )
    print(json.dumps({"candidate_keys": len(keys), "results": results}, indent=2))
    return 0 if any(row["responses"].get("http_status") == 200 for row in results) else 2


if __name__ == "__main__":
    raise SystemExit(main())
