"""Clean user prompt fragments for one LIBERO evaluation cell."""

from __future__ import annotations


CELL = """- suite:      {{suite}}
- task:       {{task}}
- seed:       {{seed}}
- output_dir: {{output_dir}}
- audit:      {{output_dir}}/{{recipe_tag}}.json
- recipe:     {{output_dir}}/recipe_{{recipe_tag}}.jsonl"""


MODE = """Use the live task language and current RGB-D observations. If an
immutable task-memory snapshot is present in the system prompt, use it only as
a fallible task-level prior and re-ground everything in this seed."""


BEGIN = """Call ``view_driver_state({\"step\": 0})`` first. Localize the
immediate target and destination, then begin the best safe physical action.
Do not read Global Memory, generic guides, reference-seed results, or static
recipes."""
