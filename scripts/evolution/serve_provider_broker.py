#!/usr/bin/env python3
# Copyright (c) 2026 Zetta Contributors
"""Serve the single bounded provider queue used by every planner process."""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import threading
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from zetta.planner.provider_pool import load_provider_pool_config  # noqa: E402
from zetta.planner.provider_proxy import (  # noqa: E402
    BROKER_API_KEY_ENV,
    BROKER_MAX_CONCURRENCY_ENV,
    BROKER_QUEUE_CAPACITY_ENV,
    DEFAULT_BROKER_QUEUE_CAPACITY,
    MAXIMUM_BROKER_CONCURRENCY,
    ProviderPoolProxy,
)


def _positive_int(name: str, default: int) -> int:
    raw = os.environ.get(name, str(default))
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if value < 1:
        raise ValueError(f"{name} must be positive")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--host", choices=("127.0.0.1", "localhost"), default="127.0.0.1"
    )
    parser.add_argument("--port", type=int, default=4110)
    parser.add_argument(
        "--model", default=os.environ.get("ZETTA_PROVIDER_MODEL", "gpt-5.6-sol")
    )
    parser.add_argument("--timeout-s", type=float, default=900.0)
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=_positive_int(
            BROKER_MAX_CONCURRENCY_ENV, MAXIMUM_BROKER_CONCURRENCY
        ),
    )
    parser.add_argument(
        "--queue-capacity",
        type=int,
        default=_positive_int(BROKER_QUEUE_CAPACITY_ENV, DEFAULT_BROKER_QUEUE_CAPACITY),
    )
    args = parser.parse_args()

    if not 1 <= args.max_concurrency <= MAXIMUM_BROKER_CONCURRENCY:
        parser.error(
            "--max-concurrency must be between 1 and "
            f"{MAXIMUM_BROKER_CONCURRENCY}"
        )
    api_key = os.environ.get(BROKER_API_KEY_ENV, "").strip()
    if not api_key:
        parser.error(f"{BROKER_API_KEY_ENV} is required")
    config = load_provider_pool_config(default_model=f"openai-responses:{args.model}")
    if config is None:
        parser.error("ZETTA_API_PROVIDERS is required")

    broker = ProviderPoolProxy(
        config,
        timeout_seconds=args.timeout_s,
        host=args.host,
        port=args.port,
        api_key=api_key,
        max_concurrency=args.max_concurrency,
        queue_capacity=args.queue_capacity,
    )
    url = broker.start()
    print(
        json.dumps(
            {
                "event": "provider_broker_ready",
                "url": url,
                "model": args.model,
                "route_count": len(config.routes),
                "max_concurrency": args.max_concurrency,
                "queue_capacity": args.queue_capacity,
            },
            sort_keys=True,
        ),
        flush=True,
    )

    stopping = threading.Event()

    def _stop(_signum: int, _frame: object) -> None:
        stopping.set()

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)
    try:
        stopping.wait()
    finally:
        broker.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
