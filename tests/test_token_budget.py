import json
import tempfile
import unittest
from pathlib import Path

from scripts.experiments.token_budget import (
    _planner_usage,
    acquire,
    freeze_config,
    measure,
    release,
)


def _write_episode(
    root: Path,
    episode_id: str,
    *,
    valid: bool,
    planner_tokens: int,
    updater_tokens: int,
) -> None:
    episode = root / "round-00" / episode_id
    episode.mkdir(parents=True)
    (episode / "result.json").write_text(
        json.dumps(
            {
                "episode_id": episode_id,
                "valid": valid,
                "success": False,
            }
        ),
        encoding="utf-8",
    )
    (episode / "transcript_test.json").write_text(
        json.dumps(
            {
                "stats": {
                    "total_input_tokens": planner_tokens - 10,
                    "total_output_tokens": 10,
                    "total_cached_input_tokens": 0,
                }
            }
        ),
        encoding="utf-8",
    )
    update = root / "memory-updates" / episode_id / "attempt-01"
    update.mkdir(parents=True)
    (update / "update_result.json").write_text(
        json.dumps(
            {
                "accepted": True,
                "updater": {
                    "usage": {
                        "total_input_tokens": updater_tokens - 10,
                        "total_output_tokens": 10,
                    }
                },
            }
        ),
        encoding="utf-8",
    )


class TokenBudgetTests(unittest.TestCase):
    def test_interrupted_stream_usage_is_counted_without_transcript(self):
        with tempfile.TemporaryDirectory() as directory:
            episode = Path(directory)
            stream = episode / "planner.stream.jsonl"
            stream.write_text(
                json.dumps(
                    {
                        "method": "thread/tokenUsage/updated",
                        "payload": {
                            "token_usage": {
                                "total": {
                                    "input_tokens": 120,
                                    "cached_input_tokens": 80,
                                    "output_tokens": 30,
                                    "reasoning_output_tokens": 10,
                                }
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
            self.assertEqual(_planner_usage(episode)["api_tokens"], 150)

    def test_measure_uses_actual_tokens_and_reports_invalid_fractionally(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            experiments = root / "episodes"
            _write_episode(
                experiments,
                "r00-s01-a00",
                valid=True,
                planner_tokens=800,
                updater_tokens=200,
            )
            _write_episode(
                experiments,
                "r00-s02-a00",
                valid=False,
                planner_tokens=200,
                updater_tokens=0,
            )
            config_path = root / "token-budget.json"
            freeze_config(
                config_path,
                {
                    "schema_version": 1,
                    "created_at": "test",
                    "cap_equivalents": 5.0,
                    "reference_tokens": 1000,
                    "reference_definition": {"statistic": "test"},
                    "lease_reserve_equivalents": 2.0,
                    "experiment_roots": [
                        {"name": "batch", "path": str(experiments)}
                    ],
                },
            )
            report = measure(config_path)
            self.assertEqual(report["token_equivalents"], 1.2)
            self.assertEqual(report["valid_token_equivalents"], 1.0)
            self.assertEqual(report["invalid_token_equivalents"], 0.2)

    def test_leases_prevent_overcommit_and_are_idempotent(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            experiments = root / "episodes"
            experiments.mkdir()
            config_path = root / "token-budget.json"
            freeze_config(
                config_path,
                {
                    "schema_version": 1,
                    "created_at": "test",
                    "cap_equivalents": 3.0,
                    "reference_tokens": 1000,
                    "reference_definition": {"statistic": "test"},
                    "lease_reserve_equivalents": 2.0,
                    "experiment_roots": [
                        {"name": "batch", "path": str(experiments)}
                    ],
                },
            )
            first = acquire(config_path, lease_id="episode-1", arm="batch")
            self.assertIsNotNone(first)
            self.assertEqual(
                acquire(config_path, lease_id="episode-1", arm="batch"), first
            )
            self.assertIsNone(
                acquire(config_path, lease_id="episode-2", arm="batch")
            )
            release(config_path, lease_id="episode-1", outcome="valid_failure")
            self.assertIsNotNone(
                acquire(config_path, lease_id="episode-2", arm="batch")
            )


if __name__ == "__main__":
    unittest.main()
