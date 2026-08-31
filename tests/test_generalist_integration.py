# Copyright (c) 2026 Zetta Contributors
from __future__ import annotations

import unittest

import numpy as np

from robots.libero.fastwam_vla_server import _prepare_fastwam_rgb
from robots.libero.groot_vla_server import (
    build_groot_observation,
    flatten_groot_actions,
)
from robots.libero.tools import LiberoPrimitives, _side_by_side
from zetta.planner.codex import _codex_mcp_config_overrides, _Recorder
from zetta.tools.contracts import ToolContract
from zetta.tools.toolkit import Toolkit
from zetta.utils.sam3_client import UnavailableSam3Client


def _observation(z: float = 0.5) -> dict:
    return {
        "main_images": np.zeros((16, 16, 3), dtype=np.uint8),
        "wrist_images": np.ones((8, 8, 3), dtype=np.uint8),
        "extra_view_images": None,
        "states": np.asarray([0, 0, z, 0, 0, 0, 0.04, -0.04], dtype=np.float32),
        "task_descriptions": "original task",
    }


class _FakeEnv:
    return_all_frames = True
    episode_terminated = False
    episode_truncated = False

    def __init__(self):
        self.obs = _observation()

    def reset(self):
        self.episode_terminated = False
        self.episode_truncated = False
        return self.obs, {}

    def chunk_step(self, actions, *, return_all_frames=False):
        frames = []
        for _ in actions:
            self.obs = _observation(float(self.obs["states"][2] + 0.001))
            frames.append(self.obs)
        return (frames if return_all_frames else frames[-1], 0, False, False, {})


class _FakeModel:
    def __init__(self):
        self.calls = 0

    def predict_action_batch(self, _obs, mode="eval", **kwargs):
        self.calls += 1
        return np.full((6, 7), 0.8, dtype=np.float32), {
            "mode": mode,
            "parameters": kwargs.get("inference_parameters"),
        }


class _FailingModel:
    def predict_action_batch(self, _obs, mode="eval", **kwargs):
        raise RuntimeError("backend unavailable")


class _Modality:
    def __init__(self, keys, horizon=1):
        self.modality_keys = keys
        self.delta_indices = list(range(horizon))


class GeneralistIntegrationTests(unittest.TestCase):
    def test_codex_requests_detailed_reasoning_summary(self):
        overrides = _codex_mcp_config_overrides(
            mcp_url="http://127.0.0.1:1/mcp",
            base_url="https://example.invalid/v1",
            reasoning_effort="medium",
            reasoning_summary="detailed",
        )
        self.assertIn('model_reasoning_summary="detailed"', overrides)

    def test_codex_restricted_profile_disables_native_file_tools(self):
        overrides = _codex_mcp_config_overrides(
            mcp_url="http://127.0.0.1:1/mcp",
            base_url=None,
            native_tools=False,
        )
        self.assertIn("features.shell_tool=false", overrides)
        self.assertIn("features.shell_snapshot=false", overrides)
        self.assertIn("tools.view_image=false", overrides)
        self.assertIn("tools.web_search=false", overrides)

    def test_codex_recorder_reads_nested_total_usage(self):
        recorder = _Recorder(max_turns=2)
        recorder.observe(
            {
                "method": "thread/tokenUsage/updated",
                "payload": {
                    "token_usage": {
                        "last": {"input_tokens": 2},
                        "total": {
                            "input_tokens": 26717,
                            "cached_input_tokens": 24064,
                            "output_tokens": 262,
                            "reasoning_output_tokens": 163,
                        },
                    }
                },
            }
        )
        recorder.observe(
            {
                "method": "turn/completed",
                "payload": {"turn": {"status": "completed", "duration_ms": 1}},
            }
        )
        self.assertEqual(recorder.usage["total_input_tokens"], 26717)
        self.assertEqual(recorder.usage["total_cached_input_tokens"], 24064)
        self.assertEqual(recorder.turns, 1)

    def test_codex_recorder_tracks_started_and_completed_physical_actions(self):
        recorder = _Recorder(max_turns=2)
        item = {
            "type": "mcpToolCall",
            "tool": "mcp__zetta__move_to",
            "arguments": {"xyz": [0.0, 0.0, 0.5]},
        }
        recorder.observe({"method": "item/started", "payload": {"item": item}})
        self.assertTrue(recorder.physical_action_started)
        self.assertEqual(recorder.physical_actions, 0)
        recorder.observe({"method": "item/completed", "payload": {"item": item}})
        self.assertEqual(recorder.physical_actions, 1)
        self.assertEqual(recorder.stats()["physical_actions"], 1)

    def test_tool_contract_manifest(self):
        toolkit = Toolkit()
        toolkit.add_tool(
            "proposal",
            {"name": "proposal", "description": "test", "input_schema": {"type": "object"}},
            lambda: {"ok": True},
            contract=ToolContract(
                capabilities=("grasp",), proposal_only=True, risk_level="read_only"
            ),
        )
        manifest = toolkit.describe_tools(["proposal"])["tools"]["proposal"]
        self.assertTrue(manifest["proposal_only"])
        self.assertEqual(manifest["capabilities"], ["grasp"])

    def test_vla_execute_replans_and_limits_horizon(self):
        env = _FakeEnv()
        model = _FakeModel()
        primitives = LiberoPrimitives(env=env, model=model, sam3_client=None)
        primitives.reset()
        primitives.start_recording()
        result = primitives.vla_execute(
            "move carefully",
            max_chunks=2,
            actions_per_chunk=3,
            translation_scale=0.5,
            action_clip=0.25,
            inference_parameters={"action_horizon": 6},
        )
        self.assertEqual(model.calls, 2)
        self.assertEqual(result["actions_executed"], 6)
        self.assertEqual(primitives.recorded_frame_count(), 6)
        self.assertEqual(result["chunk_diagnostics"][0]["executed_horizon"], 3)

    def test_vla_prompt_override_is_restored_after_backend_error(self):
        env = _FakeEnv()
        primitives = LiberoPrimitives(
            env=env, model=_FailingModel(), sam3_client=None
        )
        primitives.reset()
        with self.assertRaisesRegex(RuntimeError, "backend unavailable"):
            primitives.vla_execute("temporary subtask", max_chunks=1)
        self.assertEqual(
            primitives._last_obs["task_descriptions"], "original task"
        )

    def test_pi0_pick_rejects_an_unbounded_chunk_budget(self):
        primitives = LiberoPrimitives(env=_FakeEnv(), model=_FakeModel(), sam3_client=None)
        primitives.reset()
        with self.assertRaisesRegex(ValueError, r"\[1, 8\]"):
            primitives.pi0_pick("pick up the bottle", max_chunks=9)

    def test_multiview_mosaic_resizes_wrist(self):
        mosaic = _side_by_side(
            np.zeros((16, 20, 3), dtype=np.uint8),
            np.zeros((8, 10, 3), dtype=np.uint8),
        )
        self.assertEqual(mosaic.shape, (16, 40, 3))

    def test_groot_observation_and_action_mapping(self):
        configs = {
            "video": _Modality(["image", "wrist_image"], horizon=2),
            "state": _Modality(
                ["x", "y", "z", "roll", "pitch", "yaw", "gripper"], horizon=1
            ),
            "language": _Modality(["annotation.human.action.task_description"]),
        }
        env_obs = {
            "main_images": np.zeros((1, 16, 16, 3), dtype=np.uint8),
            "wrist_images": np.zeros((1, 16, 16, 3), dtype=np.uint8),
            "states": np.zeros((1, 8), dtype=np.float32),
        }
        obs = build_groot_observation("task", env_obs, configs)
        self.assertEqual(obs["video.image"].shape, (1, 2, 16, 16, 3))
        action = {
            f"action.{key}": np.zeros((1, 3, 1), dtype=np.float32)
            for key in ("x", "y", "z", "roll", "pitch", "yaw", "gripper")
        }
        merged = flatten_groot_actions(action)
        self.assertEqual(merged.shape, (1, 3, 7))
        self.assertTrue(np.all(merged[..., -1] == 1.0))
        action["action.gripper"][:] = 1.0
        self.assertTrue(np.all(flatten_groot_actions(action)[..., -1] == -1.0))

    def test_fastwam_camera_contract(self):
        image_meta = [
            {"shape": [3, 224, 112]},
            {"shape": [3, 224, 112]},
        ]
        rgb = _prepare_fastwam_rgb(
            np.zeros((256, 256, 3), dtype=np.uint8),
            np.zeros((256, 256, 3), dtype=np.uint8),
            image_meta,
            concatenation="horizontal",
        )
        self.assertEqual(rgb.shape, (224, 224, 3))

    def test_unavailable_sam_is_structured(self):
        client = UnavailableSam3Client("gated checkpoint unavailable")
        result = client.segment("unused.png", text_prompt="bowl")
        self.assertFalse(result.found)
        self.assertIn("gated", result.reason)


if __name__ == "__main__":
    unittest.main()
