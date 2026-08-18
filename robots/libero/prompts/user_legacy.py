"""ARCHIVED legacy RPent LIBERO user prompt; intentionally unused at runtime.

It is retained only to reproduce historical RPent prompt behavior. The active
prompt bundle imports ``user.py``.
"""

from __future__ import annotations

CELL = """- suite:      {{suite}}
- task:       {{task}}
- seed:       {{seed}}
- output_dir: {{output_dir}}
- audit:      {{output_dir}}/{{recipe_tag}}.json
- recipe:     {{output_dir}}/recipe_{{recipe_tag}}.jsonl"""


MODE = """Use the high-resolution image paths returned by view_driver_state and
back_project to localize objects before motion."""


BEGIN = """read MEMORY.md, the guides, then `view_driver_state({"step":0})` and the
returned high-resolution images. Localize the target, then plan and execute."""
