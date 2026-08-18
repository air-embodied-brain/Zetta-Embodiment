"""Global prompt definitions shared by environments."""
from __future__ import annotations

from rpent.context.prompt_utils import BulletList

OUTPUT = BulletList([
    "Brief reasoning before each tool call (1-2 sentences): observation -> decision.",
    "Don't re-read files already in this session. Don't view_driver_state right after a primitive tool already returned the state.",
    "Numerical coords in 3 decimals are enough.",
    "Save artifacts BEFORE calling finish. Stop immediately after writing the audit; do not chat further.",
])

USER = {
    "Task": """
    - suite:   {{suite}}
    - task:    {{task}}
    - seed:    {{seed}}
    - output_dir: {{output_dir}}
    - output:  {{output_dir}}/
      - audit filename:  {{recipe_tag}}.json
    """,
}
