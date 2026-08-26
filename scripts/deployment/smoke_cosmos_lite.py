# Copyright (c) 2026 Zetta Contributors
"""Send deterministic Zetta requests to a running Cosmos-Lite service."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from rollout_runtime.api.ids import EpisodeId, OperationSeq, RequestId, SessionId
from rollout_runtime.api.internal import InferenceRequest
from rollout_runtime.api.messages import Observation
from rollout_runtime.backends.cosmos_lite import (
    COSMOS_LITE_V030_REVISION,
    CosmosLitePolicyConfig,
    CosmosLitePolicyCore,
)
from rollout_runtime.core import payload as payload_module


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default="ws://127.0.0.1:8000")
    parser.add_argument("--resolved-config", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--repository-revision", default=COSMOS_LITE_V030_REVISION)
    parser.add_argument("--instruction", default="move the robot arm to the target")
    parser.add_argument("--requests", type=int, default=2)
    parser.add_argument("--request-timeout-s", type=float, default=900.0)
    parser.add_argument("--action-atol", type=float, default=1e-6)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--include-actions",
        action="store_true",
        help="Include every returned action chunk in the JSON replay artifact.",
    )
    parser.add_argument("--allow-insecure-remote", action="store_true")
    parser.add_argument("--allow-runtime-fallbacks", action="store_true")
    return parser


def _observation() -> Observation:
    """Build a deterministic upstream-compatible single-image observation."""
    rows = np.arange(540, dtype=np.uint16)[:, None, None]
    columns = np.arange(640, dtype=np.uint16)[None, :, None]
    channels = np.arange(3, dtype=np.uint16)[None, None, :]
    image = ((rows + columns + channels * 31) % 256).astype(np.uint8)
    return Observation(
        session_id=SessionId("cosmos-lite-smoke"),
        episode_id=EpisodeId(1),
        step_index=0,
        main_image=payload_module.encode_image(image),
        state=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        instruction="",
    )


def main() -> int:
    args = _parser().parse_args()
    if args.requests < 1:
        raise ValueError("--requests must be positive")
    if args.action_atol < 0:
        raise ValueError("--action-atol must be non-negative")
    config = CosmosLitePolicyConfig(
        endpoint=args.endpoint,
        resolved_config_path=str(args.resolved_config),
        expected_manifest_sha256=args.manifest_sha256,
        expected_repository_revision=args.repository_revision,
        image_layout="single",
        request_timeout_s=args.request_timeout_s,
        allow_insecure_remote=args.allow_insecure_remote,
        allow_runtime_fallbacks=args.allow_runtime_fallbacks,
    )
    core = CosmosLitePolicyCore(config)
    reports: list[dict[str, object]] = []
    action_arrays: list[np.ndarray] = []
    observation = _observation()
    try:
        core.load()
        for index in range(args.requests):
            request = InferenceRequest(
                request_id=RequestId(f"cosmos-lite-smoke-{index}"),
                session_id=observation.session_id,
                episode_id=observation.episode_id,
                operation_seq=OperationSeq(index + 1),
                policy_id="cosmos_lite",
                observation=observation,
                instruction_override=args.instruction,
                routing_token="smoke:0",
                compat_key="cosmos-lite-smoke",
                deadline=time.time() + args.request_timeout_s,
            )
            started = time.perf_counter()
            response = core.infer_batch([request])[0]
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            if response.error is not None:
                raise RuntimeError(
                    f"{response.error.code.value}: {response.error.message}"
                )
            actions = payload_module.decode_payload(response.actions)
            action_arrays.append(actions)
            reports.append(
                {
                    "index": index,
                    "latency_ms": elapsed_ms,
                    "shape": list(actions.shape),
                    "dtype": str(actions.dtype),
                    "action_sha256": hashlib.sha256(actions.tobytes()).hexdigest(),
                    "model_version": response.model_version,
                    "auxiliary_outputs": response.auxiliary_outputs,
                }
            )
            if args.include_actions:
                reports[-1]["actions"] = actions.tolist()
    finally:
        core.close()

    reference = action_arrays[0]
    max_abs_diff = max(
        (float(np.max(np.abs(actions - reference))) for actions in action_arrays[1:]),
        default=0.0,
    )
    deterministic = max_abs_diff <= args.action_atol
    report = {
        "schema_version": 1,
        "endpoint": args.endpoint,
        "request_count": len(reports),
        "deterministic": deterministic,
        "action_atol": args.action_atol,
        "max_action_abs_diff": max_abs_diff,
        "input": {
            "instruction": args.instruction,
            "image_shape": [540, 640, 3],
            "state": list(observation.state),
        },
        "requests": reports,
    }
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    if not deterministic:
        raise RuntimeError(
            "fixed Cosmos-Lite seed exceeded action tolerance: "
            f"{max_abs_diff} > {args.action_atol}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
