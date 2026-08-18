"""Single ``{{name}}`` placeholder substitution utility for prompts and tool specs."""
from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

from rpent.utils.logging import get_output_dir

#: Matches ``{{name}}`` placeholders, tolerating surrounding whitespace.
PLACEHOLDER = re.compile(r"\{\{\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\}\}")


def default_variables() -> dict[str, str]:
    """Built-in variables available to every template (e.g. ``output_dir``)."""
    return {"output_dir": str(get_output_dir())}


def substitute_text(
    text: str,
    variables: Mapping[str, Any],
    *,
    strict: bool = False,
) -> str:
    """Replace ``{{name}}`` placeholders in ``text`` using ``variables``.

    Unknown placeholders are left untouched, unless ``strict`` is set — in
    which case a ``KeyError`` is raised. Prompt rendering uses ``strict=True``
    to catch authoring typos; tool-spec binding stays lenient because it only
    supplies the built-in variables.
    """
    def _sub(match: re.Match[str]) -> str:
        name = match.group(1)
        if name in variables:
            return str(variables[name])
        if strict:
            raise KeyError(name)
        return match.group(0)

    return PLACEHOLDER.sub(_sub, text)


def substitute(
    obj: Any,
    variables: Mapping[str, Any] | None = None,
    *,
    strict: bool = False,
) -> Any:
    """Recursively substitute ``{{name}}`` placeholders within ``obj``.

    Walks dicts (keys preserved, string values substituted) and lists;
    non-string leaves are returned unchanged. New containers are returned so
    the input is never mutated. Defaults to :func:`default_variables` when
    ``variables`` is omitted.
    """
    if variables is None:
        variables = default_variables()
    if isinstance(obj, str):
        return substitute_text(obj, variables, strict=strict)
    if isinstance(obj, list):
        return [substitute(item, variables, strict=strict) for item in obj]
    if isinstance(obj, dict):
        return {key: substitute(value, variables, strict=strict) for key, value in obj.items()}
    return obj
