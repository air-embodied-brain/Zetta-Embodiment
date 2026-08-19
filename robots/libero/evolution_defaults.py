# Copyright (c) 2026 Zetta Contributors
"""Shared defaults for every LIBERO-Pro evolution task."""

from __future__ import annotations

import argparse

# A campaign may explicitly disable the privileged Critic plane, but an
# individual task must not silently diverge from the branch-wide defaults.
LIBERO_PRIVILEGED_EVIDENCE_DEFAULT = True
LIBERO_PROVISIONAL_MIN_DIAGNOSIS_CONFIDENCE = 0.0
# A full LIBERO episode can take well over 15 minutes when every
# VLA chunk is persisted and the local Pi0.5 endpoint is cold.  The previous
# 900-second default converted normal horizon failures into infrastructure
# invalids, so all new LIBERO campaigns reserve a 45-minute wall clock.
LIBERO_EPISODE_TIMEOUT_S_DEFAULT = 2700
LIBERO_NO_PROGRESS_TIMEOUT_S_DEFAULT = 240

# OpenPI's public LIBERO evaluation protocol assigns the action budget by base
# task family. LIBERO-PRO perturbation suffixes (_task, _swap, _lan, _object)
# change the scene distribution, not the task-family horizon.
LIBERO_POLICY_ACTION_HORIZONS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}
LIBERO_WAIT_STEPS_DEFAULT = 10
LIBERO_HORIZON_PROTOCOL_SOURCE = (
    "https://github.com/Physical-Intelligence/openpi/blob/main/examples/libero/main.py"
)


def libero_base_suite(suite: str) -> str:
    """Map a LIBERO/LIBERO-Pro suite to its horizon-defining base family."""

    normalized = str(suite).strip()
    for base_suite in LIBERO_POLICY_ACTION_HORIZONS:
        if normalized == base_suite or normalized.startswith(f"{base_suite}_"):
            return base_suite
    raise ValueError(f"no official OpenPI action horizon for LIBERO suite {suite!r}")


def libero_horizon_contract(
    suite: str,
    *,
    max_actions: int | None = None,
    wait_steps: int = LIBERO_WAIT_STEPS_DEFAULT,
) -> dict[str, object]:
    """Resolve and describe the complete, auditable episode horizon contract."""

    base_suite = libero_base_suite(suite)
    standard_max_actions = LIBERO_POLICY_ACTION_HORIZONS[base_suite]
    policy_action_horizon = (
        standard_max_actions if max_actions is None else int(max_actions)
    )
    wait_steps = int(wait_steps)
    if policy_action_horizon <= 0:
        raise ValueError("LIBERO policy action horizon must be positive")
    if wait_steps < 0:
        raise ValueError("LIBERO wait steps must be non-negative")
    return {
        "schema_version": 1,
        "protocol": "openpi_libero_suite_horizon_v1",
        "source": LIBERO_HORIZON_PROTOCOL_SOURCE,
        "suite": str(suite).strip(),
        "base_suite": base_suite,
        "standard_policy_action_horizon": standard_max_actions,
        "policy_action_horizon": policy_action_horizon,
        "wait_steps": wait_steps,
        "max_episode_steps": policy_action_horizon + wait_steps,
        "standard_wait_steps": LIBERO_WAIT_STEPS_DEFAULT,
        "is_standard": (
            policy_action_horizon == standard_max_actions
            and wait_steps == LIBERO_WAIT_STEPS_DEFAULT
        ),
    }


def add_privileged_evidence_argument(
    parser: argparse.ArgumentParser, *, help_text: str
) -> None:
    """Add the common LIBERO privileged-evidence switch to a CLI parser."""

    parser.add_argument(
        "--allow-privileged-evidence",
        action=argparse.BooleanOptionalAction,
        default=LIBERO_PRIVILEGED_EVIDENCE_DEFAULT,
        help=help_text,
    )


def privileged_evidence_enabled(args: object) -> bool:
    """Read the shared default while allowing legacy Namespace callers."""

    return bool(
        getattr(args, "allow_privileged_evidence", LIBERO_PRIVILEGED_EVIDENCE_DEFAULT)
    )
