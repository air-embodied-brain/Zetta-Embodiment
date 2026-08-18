"""Regression checks for the clean LIBERO experiment prompt."""

from __future__ import annotations

import inspect
import unittest

from robots.libero import prompt_bundle
from rpent.context.prompt_utils import format_prompt


class CleanLiberoPromptTest(unittest.TestCase):
    def setUp(self) -> None:
        variables = {
            "suite": "libero_goal_swap",
            "task": 3,
            "seed": 7,
            "output_dir": "/tmp/episode",
            "recipe_tag": "goal_swap_t3_s7",
        }
        self.system = format_prompt(
            prompt_bundle.system_prompt(), variables=variables
        )
        self.user = format_prompt(prompt_bundle.user_prompt(), variables=variables)

    def test_active_prompt_has_no_static_task_solution_retrieval(self) -> None:
        combined = f"{self.system}\n{self.user}".lower()
        forbidden = (
            "proven levers",
            "memory.md",
            "results_",
            "seed-0",
            "seed 0",
            "solved 9/10",
            "per-task recipes",
        )
        for fragment in forbidden:
            self.assertNotIn(fragment, combined)

    def test_active_bundle_does_not_import_legacy_prompt(self) -> None:
        source = inspect.getsource(prompt_bundle)
        self.assertNotIn("from robots.libero.prompts import system_legacy", source)
        self.assertNotIn("from robots.libero.prompts import user_legacy", source)

    def test_prompt_keeps_clean_protocol_contract(self) -> None:
        combined = f"{self.system}\n{self.user}"
        self.assertIn("libero_terminated", combined)
        self.assertIn("perception-isolated", combined)
        self.assertIn("TASK-SCOPED EPISODE MEMORY", combined)
        self.assertIn('view_driver_state({"step": 0})', combined)
        self.assertLess(len(self.system), 12_000)


if __name__ == "__main__":
    unittest.main()
