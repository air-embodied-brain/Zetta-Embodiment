#!/usr/bin/env python3
"""Run two warm Pi0.5 RPC inferences and retain non-image test evidence."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from zetta.utils.http_rpc import HttpRpcClient
from zetta.utils.vla_client import VLAClient


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default="http://127.0.0.1:18091")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows, columns = np.mgrid[:256, :256]
    main_image = np.stack(
        [rows.astype(np.uint8), columns.astype(np.uint8), ((rows + columns) // 2).astype(np.uint8)],
        axis=-1,
    )
    wrist_image = np.flip(main_image, axis=1).copy()
    observation = {
        "main_images": main_image,
        "wrist_images": wrist_image,
        "states": np.zeros(8, dtype=np.float32),
        "task_descriptions": "pick up the black bowl",
    }
    client = VLAClient(HttpRpcClient(args.endpoint))
    health = client.healthz(timeout_s=3)
    calls = []
    for index in range(2):
        started = time.perf_counter()
        actions, metadata = client.predict_action_batch(observation, mode="eval")
        elapsed = time.perf_counter() - started
        if actions.ndim != 2 or actions.shape[1] != 7 or not np.isfinite(actions).all():
            raise RuntimeError(f"invalid Pi0.5 action tensor: {actions.shape}")
        calls.append(
            {
                "index": index,
                "latency_s": round(elapsed, 3),
                "shape": list(actions.shape),
                "dtype": str(actions.dtype),
                "finite": True,
                "metadata": metadata,
            }
        )
    result = {"endpoint": args.endpoint, "health": health, "calls": calls}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
