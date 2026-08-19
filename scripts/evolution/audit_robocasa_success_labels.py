#!/usr/bin/env python3
# Copyright (c) 2026 Zetta Contributors
"""Create an append-only audit report for legacy RoboCasa success labels.

The RoboCasa Gym wrapper's sparse reward is computed directly from the
official ``_check_success()`` API. Legacy rollout adapters incorrectly also
required ``terminated=True``. This tool never mutates source artifacts; it
emits a content-addressed report declaring affected evidence audit-only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from zetta.evolution.jsonio import atomic_write_json, canonical_sha256, file_sha256


def _rows(path: Path) -> list[dict[str, Any]]:
    result = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            value["_line_number"] = line_number
            result.append(value)
    return result


def audit(source_root: Path) -> dict[str, Any]:
    affected = []
    scanned = 0
    for record_path in sorted(source_root.rglob("episode_record.json")):
        scanned += 1
        record = json.loads(record_path.read_text(encoding="utf-8"))
        states_value = (record.get("artifact_index") or {}).get("states")
        states_path = Path(states_value) if isinstance(states_value, str) else None
        if states_path is None or not states_path.is_file():
            continue
        first_official = None
        for row in _rows(states_path):
            state = row.get("state") or {}
            # In the frozen RoboCasa Gym wrapper reward=1 is assigned iff
            # env._check_success() is true. remaining==0 is reported only as
            # corroborating privileged telemetry, never as the authority.
            if row.get("reward") == 1 or row.get("official_success") is True:
                first_official = {
                    "line_number": row["_line_number"],
                    "step_index": row.get("step_index"),
                    "reward": row.get("reward"),
                    "terminated": row.get("terminated"),
                    "truncated": row.get("truncated"),
                    "remaining_to_success_m": state.get(
                        "privileged.dishwasher.rack.remaining_to_success_m"
                    ),
                }
                break
        if first_official is not None and record.get("success") is not True:
            affected.append(
                {
                    "episode_id": record.get("episode_id"),
                    "logical_id": record.get("logical_id"),
                    "record_path": str(record_path.resolve()),
                    "record_sha256": file_sha256(record_path),
                    "states_path": str(states_path.resolve()),
                    "states_sha256": file_sha256(states_path),
                    "first_official_success": first_official,
                    "legacy_recorded_success": record.get("success"),
                    "disposition": "invalid_for_scoring_and_learning_audit_only",
                }
            )
    report = {
        "schema_version": 1,
        "source_root": str(source_root.resolve()),
        "source_semantics": (
            "RoboCasa Gym sparse reward equals official _check_success; legacy "
            "adapter additionally required terminated and could miss success"
        ),
        "records_scanned": scanned,
        "affected_count": len(affected),
        "affected_episodes": affected,
        "downstream_disposition": (
            "Any cluster, diagnosis, candidate, or gate derived from an affected "
            "episode is audit-only and must be recomputed from corrected rollouts."
        ),
    }
    report["audit_sha256"] = canonical_sha256(report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.source_root)
    atomic_write_json(args.output, report, overwrite=False)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
