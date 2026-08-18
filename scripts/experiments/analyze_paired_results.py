"""Analyze valid baseline/integrated RPent result pairs."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validated_correction(
    original_path: Path, correction_path: Path
) -> tuple[dict[str, Any] | None, str | None]:
    row = _load(correction_path)
    if not row:
        return None, "invalid_json"
    audit = row.get("correction")
    if not isinstance(audit, dict):
        return None, "missing_correction_audit"
    expected = audit.get("original_result_sha256")
    if not isinstance(expected, str) or expected != _sha256(original_path):
        return None, "original_result_sha256_mismatch"
    return row, None


def _bootstrap_delta(
    pairs: list[tuple[bool, bool]], *, samples: int = 20000, seed: int = 20260801
) -> list[float] | None:
    if not pairs:
        return None
    rng = random.Random(seed)
    deltas: list[float] = []
    for _ in range(samples):
        drawn = [pairs[rng.randrange(len(pairs))] for _ in pairs]
        deltas.append(
            100.0
            * sum(int(integrated) - int(baseline) for baseline, integrated in drawn)
            / len(drawn)
        )
    deltas.sort()
    return [
        round(deltas[int(0.025 * (samples - 1))], 3),
        round(deltas[int(0.975 * (samples - 1))], 3),
    ]


def _mcnemar_exact(improved: int, regressed: int) -> float:
    discordant = improved + regressed
    if discordant == 0:
        return 1.0
    tail = sum(
        math.comb(discordant, value)
        for value in range(0, min(improved, regressed) + 1)
    ) / (2**discordant)
    return min(1.0, 2.0 * tail)


def analyze(root: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    invalid: list[dict[str, Any]] = []
    applied_corrections: list[dict[str, str]] = []
    ignored_corrections: list[dict[str, str]] = []
    for path in sorted(root.rglob("result.json")):
        row = _load(path)
        selected_path = path
        correction_candidates = sorted(path.parent.glob("result.corrected*.json"))
        valid_corrections: list[tuple[Path, dict[str, Any]]] = []
        for correction_path in correction_candidates:
            corrected, reason = _validated_correction(path, correction_path)
            if corrected is None:
                ignored_corrections.append(
                    {"path": str(correction_path), "reason": str(reason)}
                )
            else:
                valid_corrections.append((correction_path, corrected))
        if len(valid_corrections) == 1:
            selected_path, row = valid_corrections[0]
            applied_corrections.append(
                {"original": str(path), "selected": str(selected_path)}
            )
        elif len(valid_corrections) > 1:
            ignored_corrections.extend(
                {
                    "path": str(correction_path),
                    "reason": "multiple_valid_corrections",
                }
                for correction_path, _ in valid_corrections
            )
        if not row:
            invalid.append({"path": str(path), "reasons": ["invalid_json"]})
            continue
        row["_path"] = str(selected_path)
        if row.get("valid") is not True:
            invalid.append(
                {"path": str(path), "reasons": row.get("invalid_reasons", [])}
            )
            continue
        rows.append(row)

    grouped: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    conflicts: list[dict[str, Any]] = []
    for row in rows:
        pair_id = str(row.get("pair_id") or "")
        variant = str(row.get("variant") or "")
        if not pair_id or variant not in {"baseline", "integrated"}:
            conflicts.append(
                {"path": row["_path"], "reason": "missing pair_id/known variant"}
            )
            continue
        if variant in grouped[pair_id]:
            conflicts.append(
                {
                    "path": row["_path"],
                    "reason": f"duplicate {variant} for {pair_id}",
                }
            )
            continue
        grouped[pair_id][variant] = row

    complete: list[tuple[dict[str, Any], dict[str, Any]]] = []
    unpaired: list[dict[str, Any]] = []
    for pair_id, variants in sorted(grouped.items()):
        if set(variants) != {"baseline", "integrated"}:
            unpaired.append({"pair_id": pair_id, "variants": sorted(variants)})
            continue
        baseline = variants["baseline"]
        integrated = variants["integrated"]
        identity = ("protocol_id", "suite", "task", "seed")
        mismatches = [key for key in identity if baseline.get(key) != integrated.get(key)]
        if not baseline.get("protocol_id"):
            mismatches.insert(0, "protocol_id_missing")
        if mismatches:
            conflicts.append(
                {"pair_id": pair_id, "reason": f"identity mismatch: {mismatches}"}
            )
            continue
        complete.append((baseline, integrated))

    outcomes = [
        (bool(baseline.get("success")), bool(integrated.get("success")))
        for baseline, integrated in complete
    ]
    baseline_success = sum(int(value[0]) for value in outcomes)
    integrated_success = sum(int(value[1]) for value in outcomes)
    improved = sum(not baseline and integrated for baseline, integrated in outcomes)
    regressed = sum(baseline and not integrated for baseline, integrated in outcomes)
    both_success = sum(baseline and integrated for baseline, integrated in outcomes)
    both_failure = sum(not baseline and not integrated for baseline, integrated in outcomes)
    count = len(outcomes)
    token_totals: dict[str, int] = defaultdict(int)
    for row in rows:
        usage = row.get("planner", {}).get("usage", {})
        if isinstance(usage, dict):
            for key, value in usage.items():
                if isinstance(value, int):
                    token_totals[key] += value
    return {
        "valid_results": len(rows),
        "invalid_results": invalid,
        "applied_corrections": applied_corrections,
        "ignored_corrections": ignored_corrections,
        "complete_pairs": count,
        "unpaired": unpaired,
        "conflicts": conflicts,
        "baseline": {
            "successes": baseline_success,
            "rate_percent": round(100 * baseline_success / count, 3) if count else None,
        },
        "integrated": {
            "successes": integrated_success,
            "rate_percent": round(100 * integrated_success / count, 3)
            if count
            else None,
        },
        "paired_delta_pp": round(
            100 * (integrated_success - baseline_success) / count, 3
        )
        if count
        else None,
        "paired_bootstrap_95_percent": _bootstrap_delta(outcomes),
        "discordant": {
            "improved": improved,
            "regressed": regressed,
            "mcnemar_exact_two_sided_p": round(
                _mcnemar_exact(improved, regressed), 6
            ),
        },
        "concordant": {"both_success": both_success, "both_failure": both_failure},
        "token_totals": dict(token_totals),
        "interpretation": (
            "Targeted paired engineering validation; not a replacement for the "
            "paper's full 200-episode L10 benchmark."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = analyze(args.result_root.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if not report["conflicts"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
