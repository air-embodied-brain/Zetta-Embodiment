#!/usr/bin/env python3
# Copyright (c) 2026 Zetta Contributors
"""Run a small BEHAVIOR import/reset/step/close lifecycle smoke test."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rollout_runtime.api.messages import EnvSpecMsg, ResetSpec
from rollout_runtime.backends.behavior import BehaviorEnvConfig, BehaviorEnvCore


def _parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--activity-name", default="put_bowl_on_table")
    parser.add_argument("--scene-model", default="Rs_int")
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--check-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse()
    available = {
        name: importlib.util.find_spec(name) is not None
        for name in ("omnigibson", "bddl", "PIL")
    }
    if args.check_only:
        print(json.dumps({"ok": available["omnigibson"] and available["bddl"], "modules": available}, sort_keys=True))
        return 0
    if not (available["omnigibson"] and available["bddl"]):
        print(json.dumps({"ok": False, "reason": "missing BEHAVIOR prerequisites", "modules": available}, sort_keys=True))
        return 2
    config = BehaviorEnvConfig(activity_name=args.activity_name, scene_model=args.scene_model)
    core = BehaviorEnvCore()
    spec = EnvSpecMsg(env_family="behavior", env_config={field.name: getattr(config, field.name) for field in __import__("dataclasses").fields(config)})
    try:
        core.build(spec, num_envs=1)
        core.reset([0], ResetSpec(seed=0))
        actions = np.zeros((max(1, args.steps), config.action_dim), dtype=np.float32)
        result = core.chunk_step([0], [actions])[0]
        print(json.dumps({"ok": True, "executed_horizon": result.executed_horizon, "terminated": result.terminated, "truncated": result.truncated}, sort_keys=True))
    finally:
        core.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
