import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from scripts.experiments import run_paired_episode
from scripts.experiments.analyze_paired_results import analyze
from scripts.experiments.build_retry_manifest import build_retry
from scripts.experiments.run_paired_episode import (
    MAX_ALLOWED_EPISODES,
    _frozen_runtime_environment,
    _sha256_file,
    _sha256_tree,
    _planner_stream_health,
    _summarize,
    _validate_manifest,
    _verify_file_snapshot,
)


class ExperimentProtocolTests(unittest.TestCase):
    def test_runtime_routing_uses_adapter_placeholder_and_frozen_tools(self):
        self.assertEqual(
            _frozen_runtime_environment(
                {
                    "base_url": "http://127.0.0.1:4101",
                    "use_local_adapter_key_placeholder": True,
                    "contact_graspnet_endpoint": "http://127.0.0.1:18092",
                    "graspgen_endpoint": "http://127.0.0.1:18093",
                }
            ),
            {
                "CODEX_BASE_URL": "http://127.0.0.1:4101",
                "CODEX_API_KEY": "zetta-local-adapter",
                "CONTACT_GRASPNET_URL": "http://127.0.0.1:18092",
                "GRASPGEN_URL": "http://127.0.0.1:18093",
            },
        )

    @staticmethod
    def _write_minimal_finished_failure(output: Path, *, truncated: bool = False) -> None:
        (output / "states.json").write_text(
            json.dumps(
                [
                    {
                        "step_idx": 0,
                        "libero_terminated": False,
                        "episode_truncated": truncated,
                    }
                ]
            ),
            encoding="utf-8",
        )
        (output / "transcript_case.json").write_text(
            json.dumps(
                {
                    "finish": {"_finish": True, "status": "failure"},
                    "stats": {
                        "backend": "codex_sdk",
                        "model": "gpt-5.6-terra",
                        "reasoning_effort": "medium",
                        "provider": "zetta_proxy",
                    },
                }
            ),
            encoding="utf-8",
        )
        for filename in ("episode.mp4", "episode_wrist.mp4", "episode_multiview.mp4"):
            (output / filename).write_bytes(b"video-evidence")

    def test_stream_health_distinguishes_budget_from_transport_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stream = root / "planner.stream.jsonl"
            stream.write_text(
                json.dumps({"type": "timeout", "message": "planner budget"}) + "\n",
                encoding="utf-8",
            )
            health = _planner_stream_health(root)
            self.assertTrue(health["budget_exhausted"])
            self.assertEqual(health["transport_error_count"], 0)
            with stream.open("a", encoding="utf-8") as file_obj:
                file_obj.write(
                    json.dumps(
                        {
                            "method": "error",
                            "payload": {"message": "stream disconnected before completion"},
                        }
                    )
                    + "\n"
                )
            health = _planner_stream_health(root)
            self.assertEqual(health["transport_error_count"], 1)

    def test_retry_manifest_rewrites_identity_and_preserves_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_path = root / "formal.json"
            source = {
                "protocol_id": "formal-v1",
                "episodes": [
                    {
                        "id": f"original-{variant}",
                        "pair_id": "pair-1",
                        "variant": variant,
                        "commit": "a" * 40,
                        "tool_manifest_sha256": "c" * 40,
                        "output_dir": f"/formal/{variant}",
                        "resource_snapshot": {"files": [{"sha256": "a" * 64}]},
                    }
                    for variant in ("baseline", "integrated")
                ],
            }
            source_path.write_text(json.dumps(source), encoding="utf-8")
            retry = build_retry(
                source,
                source_path=source_path,
                pair_ids=["pair-1"],
                retry_tag="retry-1",
                output_root=root / "results",
                reason="proxy overload",
                purpose="infrastructure-invalid retry; excluded from the frozen first attempt",
                commit_overrides={"integrated": "b" * 40},
                tool_sha_overrides={"integrated": "d" * 40},
            )
            self.assertEqual(retry["selection"], {
                "pairs": 1,
                "episodes": 2,
                "purpose": "infrastructure-invalid retry; excluded from the frozen first attempt",
            })
            self.assertEqual(retry["retry"]["source_manifest_sha256"], _sha256_file(source_path))
            self.assertEqual(
                {episode["pair_id"] for episode in retry["episodes"]},
                {"retry-1-pair-1"},
            )
            self.assertEqual(
                {episode["original_pair_id"] for episode in retry["episodes"]},
                {"pair-1"},
            )
            self.assertEqual(
                {episode["resource_snapshot"]["files"][0]["sha256"] for episode in retry["episodes"]},
                {"a" * 64},
            )
            integrated = next(
                episode for episode in retry["episodes"] if episode["variant"] == "integrated"
            )
            self.assertEqual(integrated["commit"], "b" * 40)
            self.assertEqual(integrated["original_commit"], "a" * 40)
            self.assertEqual(integrated["tool_manifest_sha256"], "d" * 40)
            self.assertEqual(integrated["original_tool_manifest_sha256"], "c" * 40)

    def test_resource_tree_snapshot_detects_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "a.txt").write_text("alpha", encoding="utf-8")
            (root / "nested").mkdir()
            (root / "nested" / "b.txt").write_text("beta", encoding="utf-8")
            digest, count = _sha256_tree(root)
            snapshot = {
                "trees": [
                    {"path": str(root), "sha256": digest, "file_count": count}
                ]
            }
            _verify_file_snapshot(snapshot)
            (root / "nested" / "b.txt").write_text("changed", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "tree hash mismatch"):
                _verify_file_snapshot(snapshot)

    def test_runner_preserves_virtualenv_python_path(self):
        source = Path(run_paired_episode.__file__).read_text(encoding="utf-8")
        self.assertIn('Path(manifest["python"]).expanduser().absolute()', source)
        self.assertNotIn('Path(manifest["python"]).expanduser().resolve()', source)

    def test_manifest_rejects_more_than_hard_budget(self):
        manifest = {
            "schema_version": 1,
            "protocol_id": "test-protocol",
            "budget_cap": MAX_ALLOWED_EPISODES + 1,
            "common": {
                "model": "gpt-5.6-terra",
                "reasoning_effort": "medium",
            },
            "episodes": [],
        }
        with self.assertRaisesRegex(ValueError, "budget_cap"):
            _validate_manifest(manifest)

    def test_clean_planner_budget_exhaustion_is_a_valid_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output / "states.json").write_text(
                json.dumps(
                    [
                        {
                            "step_idx": 1,
                            "libero_terminated": False,
                            "episode_truncated": False,
                            "command": {"action": "move_to"},
                        }
                    ]
                ),
                encoding="utf-8",
            )
            (output / "transcript_case.json").write_text(
                json.dumps(
                    {
                        "finish": None,
                        "stats": {
                            "backend": "codex_sdk",
                            "model": "gpt-5.6-terra",
                            "reasoning_effort": "medium",
                            "provider": "zetta_proxy",
                        },
                    }
                ),
                encoding="utf-8",
            )
            (output / "planner.stream.jsonl").write_text(
                json.dumps({"type": "timeout", "message": "budget"}) + "\n",
                encoding="utf-8",
            )
            for filename in ("episode.mp4", "episode_wrist.mp4", "episode_multiview.mp4"):
                (output / filename).write_bytes(b"video-evidence")
            result = _summarize(
                episode={
                    "id": "timeout-0",
                    "suite": "libero_10",
                    "task": 3,
                    "seed": 0,
                },
                common={
                    "protocol_id": "test-protocol",
                    "model": "gpt-5.6-terra",
                    "reasoning_effort": "medium",
                },
                output_dir=output,
                commit="a" * 40,
                manifest_digest="b" * 64,
                attempt_id="attempt",
                exit_code=0,
                timed_out=False,
                elapsed_s=1800.0,
            )
            self.assertTrue(result["valid"])
            self.assertFalse(result["success"])
            self.assertTrue(result["planner"]["stream_health"]["budget_exhausted"])

    def test_no_action_and_environment_truncation_are_valid_task_failures(self):
        for truncated in (False, True):
            with self.subTest(truncated=truncated), tempfile.TemporaryDirectory() as directory:
                output = Path(directory)
                self._write_minimal_finished_failure(output, truncated=truncated)
                result = _summarize(
                    episode={"id": "failure", "suite": "libero_10", "task": 3, "seed": 0},
                    common={
                        "protocol_id": "test-protocol",
                        "model": "gpt-5.6-terra",
                        "reasoning_effort": "medium",
                    },
                    output_dir=output,
                    commit="a" * 40,
                    manifest_digest="b" * 64,
                    attempt_id="attempt",
                    exit_code=0,
                    timed_out=False,
                    elapsed_s=12.0,
                )
                self.assertTrue(result["valid"])
                self.assertFalse(result["success"])
                self.assertEqual(result["physical_actions"], [])
                self.assertEqual(result["episode_truncated"], truncated)

    def test_required_camera_gate_rejects_black_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            self._write_minimal_finished_failure(output)
            pattern = np.arange(48, dtype=np.uint8).reshape(4, 4, 3)
            for dirname, filename, image in (
                ("images_cam", "image_cam_00.png", np.zeros((4, 4, 3), dtype=np.uint8)),
                ("images_wrist", "image_wrist_00.png", pattern),
                ("images_cam_hi", "image_cam_hi_00.png", pattern[::-1]),
                ("images_wrist_hi", "image_wrist_hi_00.png", np.roll(pattern, 1, axis=0)),
            ):
                target = output / dirname
                target.mkdir()
                Image.fromarray(image).save(target / filename)
            result = _summarize(
                episode={"id": "bad-camera", "suite": "libero_10", "task": 3, "seed": 0},
                common={
                    "protocol_id": "test-protocol",
                    "model": "gpt-5.6-terra",
                    "reasoning_effort": "medium",
                    "require_camera_health": True,
                },
                output_dir=output,
                commit="a" * 40,
                manifest_digest="b" * 64,
                attempt_id="attempt",
                exit_code=0,
                timed_out=False,
                elapsed_s=12.0,
            )
            self.assertFalse(result["valid"])
            self.assertFalse(result["camera_health"]["healthy"])
            self.assertTrue(
                any(reason.startswith("camera_black_or_constant=") for reason in result["invalid_reasons"])
            )

    def test_complete_evidence_is_valid_and_uses_authoritative_success(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output / "states.json").write_text(
                json.dumps(
                    [
                        {"step_idx": 0, "libero_terminated": False},
                        {
                            "step_idx": 1,
                            "libero_terminated": True,
                            "episode_truncated": False,
                            "command": {"action": "vla_execute"},
                        },
                    ]
                ),
                encoding="utf-8",
            )
            (output / "transcript_case.json").write_text(
                json.dumps(
                    {
                        "finish": {"_finish": True, "status": "success"},
                        "stats": {
                            "backend": "codex_sdk",
                            "model": "gpt-5.6-terra",
                            "reasoning_effort": "medium",
                            "provider": "zetta_custom",
                        },
                    }
                ),
                encoding="utf-8",
            )
            for filename in (
                "episode.mp4",
                "episode_wrist.mp4",
                "episode_multiview.mp4",
            ):
                (output / filename).write_bytes(b"video-evidence")

            result = _summarize(
                episode={
                    "id": "integrated-0",
                    "pair_id": "pair-0",
                    "phase": "integrated",
                    "variant": "integrated",
                    "suite": "libero_10",
                    "task": 3,
                    "seed": 0,
                },
                common={
                    "protocol_id": "test-protocol",
                    "model": "gpt-5.6-terra",
                    "reasoning_effort": "medium",
                },
                output_dir=output,
                commit="a" * 40,
                manifest_digest="b" * 64,
                attempt_id="attempt",
                exit_code=0,
                timed_out=False,
                elapsed_s=12.5,
            )

            self.assertTrue(result["valid"])
            self.assertTrue(result["success"])
            self.assertEqual(result["physical_actions"], ["vla_execute"])

    def test_analysis_excludes_invalid_and_requires_identity_match(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = [
                ("a", "p0", "baseline", False, 0, True),
                ("b", "p0", "integrated", True, 0, True),
                ("c", "p1", "baseline", True, 1, True),
                ("d", "p1", "integrated", False, 2, True),
                ("e", "p2", "baseline", True, 0, False),
            ]
            for name, pair, variant, success, seed, valid in rows:
                target = root / name
                target.mkdir()
                (target / "result.json").write_text(
                    json.dumps(
                        {
                            "valid": valid,
                            "invalid_reasons": [] if valid else ["no_video"],
                            "protocol_id": "test-protocol",
                            "pair_id": pair,
                            "variant": variant,
                            "suite": "libero_10",
                            "task": 3,
                            "seed": seed,
                            "success": success,
                            "planner": {"usage": {}},
                        }
                    ),
                    encoding="utf-8",
                )

            report = analyze(root)

            self.assertEqual(report["valid_results"], 4)
            self.assertEqual(report["complete_pairs"], 1)
            self.assertEqual(report["discordant"]["improved"], 1)
            self.assertEqual(len(report["conflicts"]), 1)
            self.assertEqual(len(report["invalid_results"]), 1)

    def test_analysis_uses_only_hash_bound_correction(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "episode"
            target.mkdir()
            original_path = target / "result.json"
            original_path.write_text(
                json.dumps(
                    {
                        "valid": False,
                        "invalid_reasons": ["missing_finish"],
                        "protocol_id": "test-protocol",
                        "pair_id": "pair-0",
                        "variant": "integrated",
                        "suite": "libero_10",
                        "task": 3,
                        "seed": 1,
                        "success": False,
                        "planner": {"usage": {}},
                    }
                ),
                encoding="utf-8",
            )
            import hashlib

            original_sha256 = hashlib.sha256(original_path.read_bytes()).hexdigest()
            corrected = json.loads(original_path.read_text(encoding="utf-8"))
            corrected.update(
                {
                    "valid": True,
                    "invalid_reasons": [],
                    "correction": {"original_result_sha256": original_sha256},
                }
            )
            (target / "result.corrected-v1.json").write_text(
                json.dumps(corrected), encoding="utf-8"
            )

            report = analyze(root)

            self.assertEqual(report["valid_results"], 1)
            self.assertEqual(len(report["applied_corrections"]), 1)
            self.assertEqual(report["invalid_results"], [])

            corrected["correction"]["original_result_sha256"] = "0" * 64
            (target / "result.corrected-v1.json").write_text(
                json.dumps(corrected), encoding="utf-8"
            )
            report = analyze(root)

            self.assertEqual(report["valid_results"], 0)
            self.assertEqual(len(report["ignored_corrections"]), 1)
            self.assertEqual(len(report["invalid_results"]), 1)


if __name__ == "__main__":
    unittest.main()
