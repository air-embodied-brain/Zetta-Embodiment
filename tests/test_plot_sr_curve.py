# Copyright (c) 2026 RPent Contributors
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from scripts.evolution import plot_sr_curve


class _Ledger:
    def __init__(self, rows: list[dict[str, object]]) -> None:
        self._rows = rows

    def records(self) -> list[dict[str, object]]:
        return self._rows


class _Store:
    def __init__(self, _root: Path) -> None:
        self.episodes = _Ledger([])
        self.gates = _Ledger(
            [
                {
                    "kind": "heldout_20",
                    "decision_id": "gate-observed",
                    "candidate_sha256": "a" * 64,
                    "parent_successes": 16,
                    "candidate_successes": 19,
                    "paired_count": 20,
                    "p_value": 0.125,
                    "conclusive": False,
                    "passed": False,
                },
                {
                    "kind": "heldout",
                    "decision_id": "gate-independent",
                    "candidate_sha256": "b" * 64,
                    "parent_successes": 38,
                    "candidate_successes": 44,
                    "paired_count": 50,
                    "p_value": 0.03125,
                    "conclusive": False,
                    "passed": False,
                },
                {
                    "kind": "heldout_10",
                    "decision_id": "gate-10",
                    "candidate_sha256": "c" * 64,
                    "parent_successes": 7,
                    "candidate_successes": 8,
                    "paired_count": 10,
                    "p_value": 0.5,
                    "conclusive": False,
                    "passed": False,
                },
                {
                    "kind": "heldout_50",
                    "decision_id": "gate-50",
                    "candidate_sha256": "d" * 64,
                    "parent_successes": 39,
                    "candidate_successes": 45,
                    "paired_count": 50,
                    "p_value": 0.0625,
                    "conclusive": False,
                    "passed": False,
                },
            ]
        )

    def manifest(self):
        return SimpleNamespace(task="libero/task7", generation=1)

    def state(self) -> dict[str, object]:
        return {"phase": "heldout_gate"}


def test_curve_keeps_complete_but_inconclusive_paired_evidence(
    monkeypatch,
) -> None:
    monkeypatch.setattr(plot_sr_curve, "CampaignStore", _Store)

    rows = plot_sr_curve._point(Path("campaign"))

    assert [(row["arm"], row["successes"]) for row in rows] == [
        ("parent", 16),
        ("candidate", 19),
        ("parent", 38),
        ("candidate", 44),
        ("parent", 7),
        ("candidate", 8),
        ("parent", 39),
        ("candidate", 45),
    ]
    assert [row["gate_kind"] for row in rows] == [
        "heldout_20",
        "heldout_20",
        "heldout",
        "heldout",
        "heldout_10",
        "heldout_10",
        "heldout_50",
        "heldout_50",
    ]
    assert all(row["conclusive"] is False for row in rows)
    assert all(row["passed"] is False for row in rows)


def test_interval_error_lengths_clamp_floating_point_boundary_crossing() -> None:
    low, high = plot_sr_curve._wilson(20, 20)

    lower, upper = plot_sr_curve._interval_error_lengths(
        {"success_rate": 1.0, "ci95_low": low, "ci95_high": high}
    )

    assert lower >= 0.0
    assert upper == 0.0
