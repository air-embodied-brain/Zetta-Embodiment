"""Small adapter between the dashboard launch form and CLI args."""
from __future__ import annotations

from typing import Any


FIELDS = (
    "suite",
    "task",
    "seed",
    "model",
    "planner",
    "cuda-device",
    "max-turns",
    "max-tokens",
    "max-episode-steps",
    "planner-timeout-s",
    "claude-code-max-budget-usd",
)

DEFAULTS = {
    "suite": "libero_object_task",
    "task": 6,
    "seed": 0,
    "planner": "claude_code",
    "max-turns": 100,
    "max-tokens": 8192,
    "max-episode-steps": 10000,
}

INT_FIELDS = {
    "task",
    "seed",
    "max-turns",
    "max-tokens",
    "max-episode-steps",
    "planner-timeout-s",
}
FLOAT_FIELDS = {"claude-code-max-budget-usd"}
BOOL_FIELDS: set[str] = set()
OPTIONAL_STR_FIELDS = {"model", "cuda-device"}


def defaults_from_args(args: Any) -> dict[str, Any]:
    """Build form defaults from parsed CLI args, falling back to UI defaults."""
    defaults: dict[str, Any] = {}
    for key in FIELDS:
        value = getattr(args, key.replace("-", "_"), None)
        defaults[key] = DEFAULTS.get(key) if value is None else value
    return defaults


def apply_to_args(args: Any, payload: dict[str, Any]) -> None:
    """Overlay launch form values onto parsed CLI args."""
    for key in FIELDS:
        if key not in payload:
            continue
        value = payload[key]
        if key in OPTIONAL_STR_FIELDS and value in ("", None):
            value = None
        elif key in INT_FIELDS:
            value = DEFAULTS.get(key) if value in ("", None) else int(value)
        elif key in FLOAT_FIELDS:
            value = DEFAULTS.get(key) if value in ("", None) else float(value)
        elif key in BOOL_FIELDS:
            value = _as_bool(value)
        setattr(args, key.replace("-", "_"), value)


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)
