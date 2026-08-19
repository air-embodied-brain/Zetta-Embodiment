import json
import os
import tempfile
import unittest
from pathlib import Path

from scripts.experiments.run_memory_loop import _append_ledger
from unittest import mock

from zetta.memory.task_memory import (
    TaskMemoryStore,
    canonical_sha256,
    render_episode_memory,
    validate_parameter_block,
)
from zetta.tools import common
from scripts.experiments.run_memory_loop import (
    MAX_ALLOWED_EPISODES,
    _decode_json_submission,
    _generic_memory_state,
    _planner_environment,
    _tool_endpoint_environment,
    _validate_manifest,
    analyze,
)


def _valid_parameters() -> dict:
    return {
        "strategy_summary": "Use current-scene perception and a short grasp-only prompt.",
        "decision_rules": ["If the grasp is visually uncertain, re-observe before carrying."],
        "tool_parameters": [
            {
                "tool": "pi0_pick",
                "when": "the target and destination have been localized",
                "parameters": {"max_chunks": 8, "lift_thresh": 0.05},
                "why": "cap the VLA before it begins an uncontrolled placement",
                "confidence": 0.5,
            }
        ],
        "prompt_ladder": ["grasp the target bottle"],
        "failure_modes": [
            {
                "symptom": "the gripper closes without lifting the target",
                "likely_cause": "the grasp missed",
                "response": "re-observe and retry from a safer pre-position",
                "evidence_ids": ["r00-s01-a00"],
            }
        ],
        "success_checks": ["the benchmark termination flag is true"],
        "open_questions": ["Does the same short prompt work across all seeds?"],
    }


class EpisodeMemoryTests(unittest.TestCase):
    def test_local_adapter_routing_replaces_inherited_provider_secret(self):
        self.assertEqual(
            _planner_environment(
                {
                    "base_url": "http://127.0.0.1:4101",
                    "use_local_adapter_key_placeholder": True,
                }
            ),
            {
                "CODEX_BASE_URL": "http://127.0.0.1:4101",
                "CODEX_API_KEY": "zetta-local-adapter",
            },
        )

    def test_manifest_frozen_grasp_endpoints_override_parent_environment(self):
        self.assertEqual(
            _tool_endpoint_environment(
                {
                    "contact_graspnet_endpoint": "http://127.0.0.1:18092",
                    "graspgen_endpoint": "http://127.0.0.1:18093",
                }
            ),
            {
                "CONTACT_GRASPNET_URL": "http://127.0.0.1:18092",
                "GRASPGEN_URL": "http://127.0.0.1:18093",
            },
        )

    def test_disabled_generic_memory_has_an_auditable_empty_state(self):
        manifest = {
            "frozen_resources": {
                "generic_memory": {
                    "enabled": False,
                    "path": None,
                    "sha256": None,
                    "file_count": 0,
                }
            }
        }
        self.assertEqual(_generic_memory_state(manifest), (None, 0))

    def test_legacy_generic_memory_descriptor_remains_supported(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "memory.txt").write_text("frozen", encoding="utf-8")
            manifest = {
                "frozen_resources": {
                    "generic_memory": {
                        "path": str(root),
                        "sha256": "checked elsewhere",
                        "file_count": 1,
                    }
                }
            }
            digest, count = _generic_memory_state(manifest)
            self.assertEqual(len(digest), 64)
            self.assertEqual(count, 1)

    def test_shared_ledger_scopes_episode_identity_by_protocol(self):
        with tempfile.TemporaryDirectory() as directory:
            ledger = Path(directory) / "ledger.jsonl"
            base = {
                "event": "reserved",
                "episode_id": "r00-s01-a00",
                "budget_cap": 10,
            }
            _append_ledger(ledger, {**base, "protocol_id": "batch"}, reserve=True)
            _append_ledger(ledger, {**base, "protocol_id": "cycling"}, reserve=True)

            with self.assertRaisesRegex(RuntimeError, "batch:r00-s01-a00"):
                _append_ledger(
                    ledger,
                    {**base, "protocol_id": "batch"},
                    reserve=True,
                )

    def test_store_is_append_only_and_hash_chained(self):
        with tempfile.TemporaryDirectory() as directory:
            store = TaskMemoryStore(
                directory,
                task_key="libero_10/task-3",
                suite="libero_10",
                task=3,
            )
            initial = store.initialize()
            updated = store.update(
                parameter_block=_valid_parameters(),
                episode={
                    "episode_id": "r00-s01-a00",
                    "round": 0,
                    "seed": 1,
                    "valid": True,
                    "success": False,
                    "result_sha256": "a" * 64,
                    "memory_snapshot_sha256": canonical_sha256(initial),
                },
                updater={"model": "gpt-5.6-terra", "reasoning_effort": "medium"},
            )
            self.assertEqual(updated["revision"], 1)
            self.assertEqual(updated["parent_sha256"], canonical_sha256(initial))
            self.assertTrue((store.versions_dir / "rev-0000.json").is_file())
            self.assertTrue((store.versions_dir / "rev-0001.json").is_file())
            self.assertEqual(store.load(), updated)

    def test_parameter_validator_rejects_seed_coordinates_and_paths(self):
        parameters = _valid_parameters()
        parameters["tool_parameters"][0]["parameters"]["xyz"] = [0.1, 0.2, 0.3]
        with self.assertRaisesRegex(ValueError, "coordinate key"):
            validate_parameter_block(parameters)
        parameters = _valid_parameters()
        parameters["open_questions"] = ["inspect C:/zetta-fixture/episode.png"]
        with self.assertRaisesRegex(ValueError, "absolute artifact path"):
            validate_parameter_block(parameters)

    def test_rendered_snapshot_declares_hash_and_immutability(self):
        with tempfile.TemporaryDirectory() as directory:
            memory = TaskMemoryStore(
                directory,
                task_key="libero_10/task-3",
                suite="libero_10",
                task=3,
            ).initialize()
            rendered = render_episode_memory(memory)
            self.assertIn(canonical_sha256(memory), rendered)
            self.assertIn("IMMUTABLE PARAMETER SNAPSHOT", rendered)
            self.assertIn("MUST NOT", rendered)

    def test_frozen_episode_write_tool_is_output_scoped(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "episode"
            output.mkdir()
            with (
                mock.patch.dict(os.environ, {"ZETTA_EPISODE_MEMORY_FROZEN": "1"}),
                mock.patch.object(common, "get_output_dir", return_value=output),
            ):
                allowed = common.write_text_file(str(output / "audit.json"), "ok")
                denied = common.write_text_file(str(root / "memory.json"), "bad")
            self.assertNotIn("error", allowed)
            self.assertIn("error", denied)
            self.assertFalse((root / "memory.json").exists())

    def test_manifest_has_a_hard_200_episode_ceiling(self):
        manifest = {
            "schema_version": 1,
            "protocol_id": "test",
            "model": "gpt-5.6-terra",
            "reasoning_effort": "medium",
            "budget_cap": MAX_ALLOWED_EPISODES + 1,
            "max_rounds": 1,
            "seeds": list(range(1, 9)),
        }
        with self.assertRaisesRegex(ValueError, "budget_cap"):
            _validate_manifest(manifest)

    def test_final_response_json_fallback_is_strict(self):
        payload = _valid_parameters()
        self.assertEqual(_decode_json_submission(json.dumps(payload)), payload)
        self.assertEqual(
            _decode_json_submission("```json\n" + json.dumps(payload) + "\n```"),
            payload,
        )
        with self.assertRaises(json.JSONDecodeError):
            _decode_json_submission("analysis before {}")

    def test_analysis_detects_later_batch_improvement(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "outputs"
            store = TaskMemoryStore(
                root / "memory",
                task_key="libero_10/task-3",
                suite="libero_10",
                task=3,
            )
            store.initialize()
            for round_index, successes in ((0, 2), (1, 3)):
                for seed in range(1, 9):
                    target = output / f"round-{round_index:02d}" / f"r{round_index:02d}-s{seed:02d}-a00"
                    target.mkdir(parents=True)
                    (target / "result.json").write_text(
                        json.dumps(
                            {
                                "episode_id": target.name,
                                "seed": seed,
                                "attempt_index": 0,
                                "valid": True,
                                "success": seed <= successes,
                                "memory_snapshot_revision": round_index * 8,
                            }
                        ),
                        encoding="utf-8",
                    )
            manifest = {
                "protocol_id": "test",
                "seeds": list(range(1, 9)),
                "max_rounds": 2,
                "output_root": str(output),
                "ledger_path": str(root / "ledger.jsonl"),
                "budget_cap": 200,
                "task_key": "libero_10/task-3",
            }
            report = analyze(manifest, store)
            self.assertEqual(report["baseline_rate"], 0.25)
            self.assertEqual(report["improved_round"], 1)
            self.assertTrue(report["improvement_observed"])
            self.assertEqual(report["rounds"][1]["success_rate"], 0.375)


if __name__ == "__main__":
    unittest.main()
