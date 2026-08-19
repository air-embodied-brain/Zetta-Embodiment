#!/usr/bin/env python3
# Copyright (c) 2026 Zetta Contributors
"""Create an explicitly non-formal bundle for the LIBERO Role1 runtime smoke.

The step predicate is intentionally synthetic.  It validates the online
Critic/Actor/Recovery wiring and must never be registered or promoted by a
campaign.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from zetta.evolution.jsonio import (  # noqa: E402
    atomic_write_json,
    canonical_sha256,
    read_json,
)
from zetta.evolution.models import (  # noqa: E402
    CandidateBundle,
    CriticRule,
    RecoveryRule,
    RecoveryStep,
)


def prepare(output: Path, *, trigger_step: int = 12) -> dict[str, object]:
    if trigger_step < 1:
        raise ValueError("trigger_step must be positive")
    evidence_id = "development-role1-wiring-smoke"
    diagnosis_sha256 = canonical_sha256(
        {
            "kind": "development_only_runtime_smoke",
            "claim": "the online Critic can interrupt one physical step",
            "formal_evidence": False,
        }
    )
    bundle = CandidateBundle(
        candidate_id="development-libero-role1-wiring-smoke",
        generation=1,
        parent_sha256=None,
        diagnosis_sha256=diagnosis_sha256,
        causal_hypothesis=(
            "Synthetic step activation can exercise the frozen Role1 recovery "
            "path without claiming task improvement."
        ),
        mechanism_change="interrupt once and request a bounded gripper command",
        validation_plan=(
            "audit-only real episode; require per-step proposal, one persisted "
            "Role1 decision, Actor-owned execution, and continued episode"
        ),
        critic_rules=(
            CriticRule(
                rule_id="development-step-interrupt",
                title="development-only step interrupt",
                feature="episode.step_index",
                operator="ge",
                threshold=trigger_step,
                dwell_steps=1,
                cooldown_steps=10_000,
                proposal="review and execute the frozen bounded gripper recovery",
                evidence_ids=(evidence_id,),
            ),
        ),
        recovery_rules=(
            RecoveryRule(
                recovery_id="development-bounded-gripper",
                title="development-only bounded gripper command",
                trigger_rule_ids=("development-step-interrupt",),
                precondition="the synthetic development critic is active",
                steps=(
                    RecoveryStep(
                        tool="set_gripper",
                        parameters={"gripper": -1.0, "steps": 2},
                        stop_when="two bounded gripper actions have executed",
                    ),
                ),
                safety_constraints=(
                    "retain the fixed action contract",
                    "do not use task collision heuristics",
                ),
                stop_condition="the frozen recovery step is complete",
                fallback="return control to the authoritative full-task VLA",
                evidence_ids=(evidence_id,),
            ),
        ),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output, bundle.as_dict(), overwrite=False)
    published = CandidateBundle.from_dict(read_json(output))
    if published.sha256 != bundle.sha256:
        raise RuntimeError("published smoke bundle failed semantic hash verification")
    report = {
        "development_only": True,
        "bundle_sha256": bundle.sha256,
        "critic_rule_count": len(bundle.critic_rules),
        "recovery_rule_count": len(bundle.recovery_rules),
        "trigger_step": trigger_step,
    }
    atomic_write_json(output.with_suffix(".attestation.json"), report, overwrite=False)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trigger-step", type=int, default=12)
    args = parser.parse_args()
    print(json.dumps(prepare(args.output, trigger_step=args.trigger_step), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
