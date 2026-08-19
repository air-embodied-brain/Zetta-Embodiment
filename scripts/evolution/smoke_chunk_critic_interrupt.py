# Copyright (c) 2026 Zetta Contributors
"""Verify that a frozen Critic interrupts a real server-side action chunk."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from robots.robocasa.env_client import RoboCasaEnvClient
from robots.robocasa.groot_client import Gr00tClient
from zetta.evolution.jsonio import atomic_write_json
from zetta.evolution.models import CriticRule


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-endpoint", required=True)
    parser.add_argument("--vla-endpoint", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--inference-seed", type=int, default=202608079)
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

    env = RoboCasaEnvClient(args.env_endpoint, timeout_s=240)
    vla = Gr00tClient(args.vla_endpoint, timeout_s=180)
    try:
        env.reset(
            task="SlideDishwasherRack",
            seed=args.seed,
            split="target",
            video_dir=str(args.output.parent / "videos"),
        )
        observation = env.observation(include_images=True)
        state = observation["observation"]["state"]
        instruction = str(state["annotation.human.task_description"])
        actions, inference = vla.act(
            observation,
            instruction=instruction,
            inference_seed=args.inference_seed,
        )
        rule = CriticRule(
            rule_id="real-rack-residual-interrupt",
            title="real rack residual remains above diagnostic bound",
            feature="privileged.dishwasher.rack.residual_to_success",
            operator="gt",
            threshold=0.1,
            dwell_steps=1,
            cooldown_steps=0,
            proposal="pause and inspect rack engagement",
            evidence_ids=("real-smoke-preregistered-rule",),
        )
        result = env.execute_chunk(
            actions[:8],
            critic_rules=[rule.as_dict()],
            interrupt_on_proposal=True,
            capture_event_images=True,
        )
        proposals = result["critic_proposals"]
        passed = (
            result["executed_horizon"] == 1
            and len(proposals) == 1
            and proposals[0]["rule_id"] == rule.rule_id
            and proposals[0]["environment_write"] is False
            and len(result["event_images"]) == 1
        )
        report = {
            "schema_version": 1,
            "passed": passed,
            "task": "SlideDishwasherRack",
            "seed": args.seed,
            "policy_rng": args.inference_seed,
            "requested_horizon": min(8, len(actions)),
            "executed_horizon": result["executed_horizon"],
            "proposal_count": len(proposals),
            "proposal_rule_id": proposals[0]["rule_id"] if proposals else None,
            "proposal_environment_write": (
                proposals[0]["environment_write"] if proposals else None
            ),
            "event_image_count": len(result["event_images"]),
            "action_chunk_sha256": inference["action_chunk_sha256"],
            "authoritative_success": result["authoritative_success"],
        }
        atomic_write_json(args.output, report, overwrite=False)
        return 0 if passed else 2
    finally:
        if env.episode_id is not None and not env.outcome_unknown:
            env.finalize_episode()
            env.release()


if __name__ == "__main__":
    raise SystemExit(main())
