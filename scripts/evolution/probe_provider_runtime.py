#!/usr/bin/env python3
# Copyright (c) 2026 Zetta Contributors
"""Issue secret-free health probes for every configured model route."""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from typing import Any

DEFAULT_MODEL = "gpt-5.6-sol"
DEFAULT_REASONING_EFFORT = "high"
MAXIMUM_PROBE_CONCURRENCY = 8
_ALLOWED_REASONING_EFFORTS = ("low", "medium", "high", "xhigh")


def _probe(
    route: dict[str, Any],
    timeout_s: float,
    *,
    model: str,
    reasoning_effort: str,
    wire_api: str = "responses",
) -> dict[str, Any]:
    name = str(route["name"])
    key = os.environ.get(str(route["api_key_env"]), "")
    if not key:
        return {"name": name, "ok": False, "status": "missing_key"}
    provider = str(route["provider"])
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        # Match the production OpenAI client. Some compatible endpoints put
        # Cloudflare in front and reject urllib's default browser signature
        # with error 1010 even though the same route works through the SDK.
        "User-Agent": "OpenAI/Python 2.11.0",
    }
    if provider == "azure-responses":
        target = str(route["base_url"])
        headers["api-key"] = key
        request_body = {
            "model": model,
            "input": "Reply only with OK.",
            "max_output_tokens": 16,
            "reasoning": {"effort": reasoning_effort},
            "stream": False,
        }
    elif provider == "openai-responses" or (
        provider == "openai-chat" and wire_api == "responses"
    ):
        target = str(route["base_url"]).rstrip("/") + "/responses"
        request_body = {
            "model": model,
            "input": "Reply only with OK.",
            "max_output_tokens": 16,
            "reasoning": {"effort": reasoning_effort},
            "stream": False,
        }
    elif provider == "openai-chat" and wire_api == "native":
        target = str(route["base_url"]).rstrip("/") + "/chat/completions"
        request_body = {
            "model": model,
            "messages": [
                {"role": "user", "content": "Reply with exactly: API_TEST_OK"}
            ],
            "reasoning_effort": reasoning_effort,
            "stream": False,
        }
    else:
        return {"name": name, "ok": False, "status": "unsupported_provider"}
    payload = json.dumps(request_body, separators=(",", ":")).encode("utf-8")
    request = urllib.request.Request(
        target,
        data=payload,
        headers=headers,
        method="POST",
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            status = int(response.status)
            body = response.read()
        parsed = json.loads(body)
        response_shape = isinstance(parsed, dict) and (
            isinstance(parsed.get("choices"), list)
            or isinstance(parsed.get("output"), list)
        )
        return {
            "name": name,
            "ok": status == 200 and response_shape,
            "status": status,
            "latency_s": round(time.perf_counter() - started, 3),
        }
    except urllib.error.HTTPError as exc:
        exc.close()
        return {
            "name": name,
            "ok": False,
            "status": int(exc.code),
            "latency_s": round(time.perf_counter() - started, 3),
        }
    except Exception as exc:  # Only the class is persisted; messages can leak URLs.
        return {
            "name": name,
            "ok": False,
            "status": type(exc).__name__,
            "latency_s": round(time.perf_counter() - started, 3),
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--concurrency", type=int, default=MAXIMUM_PROBE_CONCURRENCY)
    parser.add_argument("--timeout-s", type=float, default=90.0)
    parser.add_argument(
        "--wire-api",
        choices=("responses", "native"),
        default="responses",
        help="probe the production broker wire by default; native is diagnostic only",
    )
    parser.add_argument(
        "--model", default=os.environ.get("ZETTA_PROVIDER_MODEL", DEFAULT_MODEL)
    )
    parser.add_argument(
        "--reasoning-effort",
        choices=_ALLOWED_REASONING_EFFORTS,
        default=os.environ.get(
            "ZETTA_PROVIDER_REASONING_EFFORT", DEFAULT_REASONING_EFFORT
        ),
    )
    args = parser.parse_args()
    if not 1 <= args.concurrency <= MAXIMUM_PROBE_CONCURRENCY:
        raise ValueError(
            f"probe concurrency must be between 1 and {MAXIMUM_PROBE_CONCURRENCY}"
        )
    if args.timeout_s <= 0:
        raise ValueError("probe concurrency and timeout must be positive")
    raw = os.environ.get("ZETTA_API_PROVIDERS", "")
    config = json.loads(raw)
    providers = config.get("providers") if isinstance(config, dict) else config
    if not isinstance(providers, list) or not providers:
        raise ValueError("ZETTA_API_PROVIDERS has no routes")
    with ThreadPoolExecutor(max_workers=min(args.concurrency, len(providers))) as pool:
        results = list(
            pool.map(
                lambda route: _probe(
                    route,
                    args.timeout_s,
                    model=args.model,
                    reasoning_effort=args.reasoning_effort,
                    wire_api=args.wire_api,
                ),
                providers,
            )
        )
    results.sort(key=lambda result: str(result["name"]))
    print(
        json.dumps(
            {
                "concurrency": min(args.concurrency, len(providers)),
                "model": args.model,
                "reasoning_effort": args.reasoning_effort,
                "wire_api": args.wire_api,
                "routes": results,
            },
            separators=(",", ":"),
            sort_keys=True,
        )
    )
    return 0 if all(bool(result["ok"]) for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
