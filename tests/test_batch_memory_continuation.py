import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from rpent.memory.task_memory import TaskMemoryStore, build_initial_memory, canonical_sha256
from scripts.experiments.run_batch_memory_continuation import (
    _store,
    _validate_manifest,
    analyze,
)


class BatchContinuationTests(unittest.TestCase):
    def test_fresh_batch_uses_exact_frozen_revision_zero(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            initial = build_initial_memory(
                task_key="libero_10_swap/task-3",
                suite="libero_10_swap",
                task=3,
            )
            initial_path = root / "initial.json"
            initial_path.write_text(json.dumps(initial), encoding="utf-8")
            manifest = {
                "memory_root": str(root / "memory"),
                "task_key": "libero_10_swap/task-3",
                "suite": "libero_10_swap",
                "task": 3,
                "initial_memory_path": str(initial_path),
                "initial_memory_sha256": canonical_sha256(initial),
            }

            store = _store(manifest)

            self.assertEqual(canonical_sha256(store.load()), canonical_sha256(initial))

    def test_fresh_manifest_requires_frozen_initial_memory(self):
        manifest = {
            "schema_version": 2,
            "model": "gpt-5.6-terra",
            "reasoning_effort": "medium",
            "seeds": list(range(1, 9)),
            "start_round": 0,
            "max_rounds": 4,
            "global_reservation_cap": 500,
            **{
                key: "unused"
                for key in (
                    "protocol_id",
                    "repo",
                    "commit",
                    "python",
                    "output_root",
                    "memory_root",
                    "ledger_path",
                    "token_budget_path",
                    "task_key",
                    "suite",
                    "task",
                    "common",
                    "frozen_resources",
                )
            },
        }

        with self.assertRaisesRegex(ValueError, "initial_memory_path"):
            _validate_manifest(manifest)

    def test_confirmation_requires_two_improved_post_baseline_rounds(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "output"
            for round_index, successes in enumerate([2, 3, 2, 4]):
                for seed in range(1, 9):
                    episode_id = f"r{round_index:02d}-s{seed:02d}-a00"
                    target = output / f"round-{round_index:02d}" / episode_id
                    target.mkdir(parents=True)
                    (target / "result.json").write_text(
                        json.dumps(
                            {
                                "episode_id": episode_id,
                                "seed": seed,
                                "attempt_index": 0,
                                "valid": True,
                                "success": seed <= successes,
                                "memory_snapshot_revision": round_index * 8,
                            }
                        ),
                        encoding="utf-8",
                    )
            store = TaskMemoryStore(
                root / "memory",
                task_key="libero_10/task-3",
                suite="libero_10",
                task=3,
            )
            store.initialize()
            manifest = {
                "protocol_id": "test",
                "seeds": list(range(1, 9)),
                "max_rounds": 4,
                "output_root": str(output),
                "task_key": "libero_10/task-3",
                "token_budget_path": str(root / "token-budget.json"),
            }
            with mock.patch(
                "scripts.experiments.run_batch_memory_continuation.token_budget.measure",
                return_value={"token_equivalents": 0.0},
            ):
                report = analyze(manifest, store)
            self.assertEqual(report["baseline_rate"], 0.25)
            self.assertEqual(report["improved_rounds"], [1, 3])
            self.assertTrue(report["confirmation_observed"])


if __name__ == "__main__":
    unittest.main()
