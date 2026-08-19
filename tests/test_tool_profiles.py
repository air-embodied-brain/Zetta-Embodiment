import tempfile
import unittest
from pathlib import Path

from zetta.tools.toolkit import Toolkit
from scripts.experiments.run_paired_episode import _episode_command


class ToolProfileTests(unittest.TestCase):
    def test_registry_filter_hides_and_blocks_unlisted_tools(self):
        toolkit = Toolkit()
        toolkit.retain_tools({"finish", "describe_tools"})
        self.assertEqual(
            [spec["name"] for spec in toolkit.get_tools_spec()],
            ["finish", "describe_tools"],
        )
        result = toolkit.execute_tool("list_dir", {"path": "."})
        self.assertEqual(result.result, {"error": "unknown tool: list_dir"})

    def test_episode_command_freezes_endpoints_and_tool_profile(self):
        with tempfile.TemporaryDirectory() as directory:
            command = _episode_command(
                {
                    "suite": "libero_goal_task",
                    "task": 6,
                    "seed": 0,
                    "cuda_device": 1,
                },
                {
                    "model": "gpt-5.6-terra",
                    "max_turns": 100,
                    "max_episode_steps": 10000,
                    "planner_timeout_s": 1800,
                    "disable_sam3": False,
                    "sam3_endpoint": "http://127.0.0.1:18094",
                    "vla_endpoint": "http://127.0.0.1:18100",
                    "tool_profile": "pi05_only",
                },
                "python",
                Path(directory),
            )
        rendered = " ".join(command)
        self.assertIn("--sam3-endpoint http://127.0.0.1:18094", rendered)
        self.assertIn("--vla-endpoint http://127.0.0.1:18100", rendered)
        self.assertIn("--tool-profile pi05_only", rendered)
        self.assertNotIn("--disable-sam3", command)


if __name__ == "__main__":
    unittest.main()
