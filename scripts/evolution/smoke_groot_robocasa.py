# Copyright (c) 2026 Zetta Contributors
"""Real RoboCasa observation -> GR00T deterministic action smoke."""

from __future__ import annotations

import os
import argparse
from pathlib import Path

from robots.robocasa.env_client import RoboCasaEnvClient
from robots.robocasa.groot_client import Gr00tClient
from zetta.evolution.jsonio import atomic_write_json, canonical_sha256


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-endpoint", required=True)
    parser.add_argument("--vla-endpoint", required=True)
    parser.add_argument("--task", default="SlideDishwasherRack")
    parser.add_argument("--split", default="target")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--inference-seed", type=int, default=20260807)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    for name in (
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
    ):
        os.environ.pop(name, None)
    environment = RoboCasaEnvClient(args.env_endpoint, timeout_s=180)
    vla = Gr00tClient(args.vla_endpoint, timeout_s=180)
    try:
        reset = environment.reset(
            task=args.task,
            seed=args.seed,
            split=args.split,
            video_dir=str(args.output.parent / "videos"),
        )
        instruction = str(
            reset.get("observation", {})
            .get("state", {})
            .get("annotation.human.task_description", args.task)
        )
        observation = environment.observation(include_images=True)
        first_actions, first = vla.act(
            observation,
            instruction=instruction,
            inference_seed=args.inference_seed,
        )
        second_actions, second = vla.act(
            observation,
            instruction=instruction,
            inference_seed=args.inference_seed,
        )
        if first["action_chunk_sha256"] != second["action_chunk_sha256"]:
            raise RuntimeError("fixed inference seed produced different action chunks")
        if first_actions != second_actions:
            raise RuntimeError("fixed inference seed produced byte-different actions")
        schema = vla.schema()
        report = {
            "schema_version": 1,
            "task": args.task,
            "split": args.split,
            "deterministic_replay": True,
            "action_chunk_sha256": first["action_chunk_sha256"],
            "action_horizon": first["horizon"],
            "first_latency_s": first["latency_s"],
            "second_latency_s": second["latency_s"],
            "observation_field_sha256": first["observation_field_sha256"],
            "observation_identity_sha256": canonical_sha256(
                first["observation_field_sha256"]
            ),
            "checkpoint_sha256": schema["checkpoint_sha256"],
            "schema_sha256": first["schema_sha256"],
        }
        atomic_write_json(args.output, report, overwrite=False)
        return 0
    finally:
        if environment.episode_id is not None and not environment.outcome_unknown:
            environment.finalize_episode()
            environment.release()


if __name__ == "__main__":
    raise SystemExit(main())
