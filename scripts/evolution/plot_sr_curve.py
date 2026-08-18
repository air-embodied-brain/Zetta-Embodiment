#!/usr/bin/env python3
# Copyright (c) 2026 RPent Contributors
"""Plot paired success-rate evidence across immutable campaign generations.

The script never infers a promotion from missing data.  A point is emitted for
the generation's own valid rollout ledger and, when present, for a completed
paired heldout decision.  This keeps development smoke and formal paired
evidence visibly separate while supporting the branch's 10/20/50-seed
contracts.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

from rpent.evolution.store import CampaignStore


def _wilson(successes: int, count: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if count <= 0:
        return (math.nan, math.nan)
    p = successes / count
    denominator = 1.0 + z * z / count
    centre = (p + z * z / (2.0 * count)) / denominator
    radius = z * math.sqrt(p * (1.0 - p) / count + z * z / (4.0 * count * count)) / denominator
    return centre - radius, centre + radius


def _interval_error_lengths(row: dict[str, Any]) -> tuple[float, float]:
    rate = float(row["success_rate"])
    # Wilson endpoints at exactly 0% or 100% can cross the observed rate by a
    # few ulps. Matplotlib requires nonnegative error-bar lengths.
    return (
        max(0.0, rate - float(row["ci95_low"])),
        max(0.0, float(row["ci95_high"]) - rate),
    )


def _point(root: Path) -> list[dict[str, Any]]:
    store = CampaignStore(root)
    manifest = store.manifest()
    valid = [row for row in store.episodes.records() if row.get("status") == "valid"]
    rows: list[dict[str, Any]] = []
    if valid:
        successes = sum(bool(row.get("success")) for row in valid)
        low, high = _wilson(successes, len(valid))
        rows.append(
            {
                "campaign_root": str(root),
                "task": manifest.task,
                "generation": manifest.generation,
                "arm": "active_rollout",
                "gate_kind": None,
                "successes": successes,
                "count": len(valid),
                "success_rate": successes / len(valid),
                "ci95_low": low,
                "ci95_high": high,
                "phase": store.state()["phase"],
            }
        )
    for decision in store.gates.records():
        # Statistical conclusiveness governs promotion, not whether a
        # completed paired observation may appear in the evidence curve.
        # Keep this open to the branch's paired heldout contracts.  The
        # original fixed block is ``heldout_20``; independent higher-power
        # blocks use explicit ``heldout_10``/``heldout_50`` decisions (older
        # campaigns may use the generic ``heldout`` kind).
        gate_kind = decision.get("kind")
        if gate_kind not in {"heldout_10", "heldout_20", "heldout_50", "heldout"}:
            continue
        for arm, successes in (
            ("parent", int(decision["parent_successes"])),
            ("candidate", int(decision["candidate_successes"])),
        ):
            count = int(decision["paired_count"])
            low, high = _wilson(successes, count)
            rows.append(
                {
                    "campaign_root": str(root),
                    "task": manifest.task,
                    "generation": manifest.generation,
                    "arm": arm,
                    "gate_kind": gate_kind,
                    "successes": successes,
                    "count": count,
                    "success_rate": successes / count,
                    "ci95_low": low,
                    "ci95_high": high,
                    "p_value": decision.get("p_value"),
                    "conclusive": bool(decision.get("conclusive")),
                    "passed": bool(decision.get("passed")),
                    "decision_id": decision.get("decision_id"),
                    "candidate_sha256": decision.get("candidate_sha256"),
                    "phase": store.state()["phase"],
                }
            )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="PNG output path")
    parser.add_argument("--title", default="Libero-Pro success rate by generation")
    args = parser.parse_args()

    rows = [row for root in args.campaign_root for row in _point(root.resolve())]
    if not rows:
        raise SystemExit("no valid rollout or completed paired heldout evidence found")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    stem = args.output.with_suffix("")
    json_path = stem.with_suffix(".json")
    csv_path = stem.with_suffix(".csv")
    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=sorted({key for row in rows for key in row}))
        writer.writeheader()
        writer.writerows(rows)

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - deployment dependency
        raise SystemExit(f"matplotlib is required to draw the curve: {exc}") from exc

    labels = {"active_rollout": "active rollout", "parent": "parent", "candidate": "candidate"}
    markers = {"active_rollout": "o", "parent": "s", "candidate": "^"}
    colors = {"active_rollout": "#6b7280", "parent": "#2563eb", "candidate": "#dc2626"}
    figure, axis = plt.subplots(figsize=(8.5, 5.0), constrained_layout=True)
    for arm in ("active_rollout", "parent", "candidate"):
        subset = sorted((row for row in rows if row["arm"] == arm), key=lambda row: row["generation"])
        if not subset:
            continue
        x = [row["generation"] for row in subset]
        y = [row["success_rate"] for row in subset]
        errors = [_interval_error_lengths(row) for row in subset]
        lower = [error[0] for error in errors]
        upper = [error[1] for error in errors]
        axis.errorbar(
            x,
            y,
            yerr=[lower, upper],
            marker=markers[arm],
            color=colors[arm],
            linewidth=1.8,
            capsize=3,
            label=labels[arm],
        )
    axis.set_xlabel("generation")
    axis.set_ylabel("success rate")
    axis.set_ylim(0.0, 1.0)
    axis.set_title(args.title)
    axis.grid(alpha=0.25)
    axis.legend()
    figure.savefig(args.output, dpi=160)
    print(json.dumps({"plot": str(args.output), "json": str(json_path), "csv": str(csv_path), "points": len(rows)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
