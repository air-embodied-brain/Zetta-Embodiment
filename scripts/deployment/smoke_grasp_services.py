#!/usr/bin/env python3
"""Run both grasp proposal adapters against a retained LIBERO RGB-D snapshot."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from robots.libero.contact_graspnet import ContactGraspNetAdapter
from robots.libero.graspgen import GraspGenAdapter
from rpent.utils.logging import init_output_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--contact-url", default="http://127.0.0.1:18092")
    parser.add_argument("--graspgen-url", default="http://127.0.0.1:18093")
    parser.add_argument("--camera", choices=("agentview", "wrist"), default="agentview")
    parser.add_argument(
        "--tool", choices=("all", "contact_graspnet", "graspgen"), default="all"
    )
    args = parser.parse_args()

    if not args.snapshot.is_dir():
        raise FileNotFoundError(args.snapshot)
    args.output.mkdir(parents=True, exist_ok=True)
    init_output_dir(args.snapshot)
    results: dict[str, object] = {
        "snapshot": str(args.snapshot.resolve()),
        "environment_advanced": False,
        "proposal_only": True,
    }
    if args.tool in {"all", "contact_graspnet"}:
        adapter = ContactGraspNetAdapter(args.contact_url)
        results["contact_graspnet_health"] = adapter.propose(mode="health")
        results["contact_graspnet"] = adapter.propose(
            camera=args.camera,
            max_candidates=16,
            max_points=65_536,
        )
    if args.tool in {"all", "graspgen"}:
        adapter = GraspGenAdapter(args.graspgen_url)
        results["graspgen_health"] = adapter.propose(mode="health")
        results["graspgen"] = adapter.propose(
            camera=args.camera,
            max_candidates=16,
            max_points=2048,
        )

    output_path = args.output / "grasp_services_smoke.json"
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    summary = {
        "output": str(output_path),
        "contact_graspnet_status": (
            results.get("contact_graspnet", {}).get("status")
            if isinstance(results.get("contact_graspnet"), dict)
            else None
        ),
        "contact_graspnet_candidates": (
            results.get("contact_graspnet", {}).get("candidate_count")
            if isinstance(results.get("contact_graspnet"), dict)
            else None
        ),
        "graspgen_status": (
            results.get("graspgen", {}).get("status")
            if isinstance(results.get("graspgen"), dict)
            else None
        ),
        "graspgen_candidates": (
            results.get("graspgen", {}).get("candidate_count")
            if isinstance(results.get("graspgen"), dict)
            else None
        ),
        "environment_advanced": False,
    }
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
