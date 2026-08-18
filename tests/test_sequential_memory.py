import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from rpent.memory.task_memory import TaskMemoryStore
from scripts.experiments.run_sequential_memory import (
    _seed_for_step,
    _validate_manifest,
    analyze,
)


def _manifest(root: Path, *, arm: str = "cycling") -> dict:
    seeds = list(range(1, 9)) if arm == "cycling" else [1]
    return {
        "schema_version": 2,
        "protocol_id": f"test-{arm}",
        "model": "gpt-5.6-terra",
        "reasoning_effort": "medium",
        "arm": arm,
        "seed_cycle": seeds,
        "fixed_seed": 1,
        "global_reservation_cap": 2000,
        "arm_reservation_cap": 80,
        "max_valid_episodes": 80,
        "ceiling_confirmation_blocks": 3,
        "round_offset": 10,
        "repo": str(root),
        "python": "python",
        "output_root": str(root / "outputs"),
        "memory_root": str(root / "memory"),
        "ledger_path": str(root / "ledger.jsonl"),
        "commit": "a" * 40,
        "suite": "libero_10",
        "task": 3,
        "task_key": "libero_10/task-3",
        "initial_memory_path": str(root / "initial.json"),
        "initial_memory_sha256": "b" * 64,
        "token_budget_path": str(root / "token-budget.json"),
        "frozen_resources": {},
        "common": {},
    }


class SequentialMemoryTests(unittest.TestCase):
    def test_seed_schedules(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cycling = _manifest(root)
            fixed = _manifest(root, arm="fixed")
            self.assertEqual([_seed_for_step(cycling, i) for i in range(10)], [1,2,3,4,5,6,7,8,1,2])
            self.assertEqual([_seed_for_step(fixed, i) for i in range(10)], [1] * 10)

    def test_manifest_accepts_both_arms_and_rejects_duplicate_cycle(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _validate_manifest(_manifest(root))
            _validate_manifest(_manifest(root, arm="fixed"))
            invalid = _manifest(root)
            invalid["seed_cycle"] = [1] * 8
            with self.assertRaisesRegex(ValueError, "window_size unique"):
                _validate_manifest(invalid)

    def test_ten_seed_window_is_supported(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = _manifest(root)
            manifest["seed_cycle"] = list(range(10))
            manifest["window_size"] = 10
            _validate_manifest(manifest)
            self.assertEqual(
                [_seed_for_step(manifest, i) for i in range(12)],
                list(range(10)) + [0, 1],
            )

    def test_analysis_uses_non_overlapping_blocks_and_reports_rolling(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = _manifest(root)
            store = TaskMemoryStore(
                root / "memory",
                task_key="libero_10/task-3",
                suite="libero_10",
                task=3,
            )
            store.initialize()
            # Baseline block: 2/8.  Later block: 3/8.
            for step, success in enumerate(
                [True, True, False, False, False, False, False, False]
                + [True, True, True, False, False, False, False, False]
            ):
                target = root / "outputs" / "steps" / f"step-{step:03d}"
                target.mkdir(parents=True)
                (target / "summary.json").write_text(
                    json.dumps(
                        {
                            "step_index": step,
                            "seed": (step % 8) + 1,
                            "episode_id": f"e{step}",
                            "success": success,
                        }
                    ),
                    encoding="utf-8",
                )
            with mock.patch(
                "scripts.experiments.run_sequential_memory.token_budget.measure",
                return_value={"token_equivalents": 0.0},
            ):
                report = analyze(manifest, store)
            self.assertEqual(report["baseline_rate"], 0.25)
            self.assertEqual(report["improved_block"], 1)
            self.assertTrue(report["improvement_observed"])
            self.assertEqual(report["terminal_reason"], "strict_block_improvement")
            self.assertEqual(len(report["rolling_last_8"]), 9)
            self.assertEqual(report["rolling_last_n"], report["rolling_last_8"])

    def test_full_budget_mode_records_but_does_not_stop_on_improvement(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = _manifest(root)
            manifest["seed_cycle"] = list(range(10))
            manifest["window_size"] = 10
            manifest["early_stop_on_metric"] = False
            manifest["max_valid_episodes"] = 100
            store = TaskMemoryStore(
                root / "memory",
                task_key="libero_10/task-3",
                suite="libero_10",
                task=3,
            )
            store.initialize()
            outcomes = [True] + [False] * 9 + [True, True] + [False] * 8
            for step, success in enumerate(outcomes):
                target = root / "outputs" / "steps" / f"step-{step:03d}"
                target.mkdir(parents=True)
                (target / "summary.json").write_text(
                    json.dumps(
                        {
                            "step_index": step,
                            "seed": step % 10,
                            "episode_id": f"e{step}",
                            "success": success,
                        }
                    ),
                    encoding="utf-8",
                )
            with mock.patch(
                "scripts.experiments.run_sequential_memory.token_budget.measure",
                return_value={"token_equivalents": 0.0},
            ):
                report = analyze(manifest, store)
            self.assertEqual(report["baseline_rate"], 0.1)
            self.assertEqual(report["improved_block"], 1)
            self.assertTrue(report["improvement_observed"])
            self.assertIsNone(report["terminal_reason"])
            self.assertEqual(len(report["rolling_last_10"]), 11)

    def test_three_perfect_blocks_confirm_a_ceiling(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = _manifest(root)
            store = TaskMemoryStore(
                root / "memory",
                task_key="libero_10/task-3",
                suite="libero_10",
                task=3,
            )
            store.initialize()
            for step in range(24):
                target = root / "outputs" / "steps" / f"step-{step:03d}"
                target.mkdir(parents=True)
                (target / "summary.json").write_text(
                    json.dumps(
                        {
                            "step_index": step,
                            "seed": (step % 8) + 1,
                            "episode_id": f"e{step}",
                            "success": True,
                        }
                    ),
                    encoding="utf-8",
                )
            with mock.patch(
                "scripts.experiments.run_sequential_memory.token_budget.measure",
                return_value={"token_equivalents": 0.0},
            ):
                report = analyze(manifest, store)
            self.assertEqual(report["baseline_rate"], 1.0)
            self.assertEqual(report["terminal_reason"], "ceiling_confirmed")


if __name__ == "__main__":
    unittest.main()
