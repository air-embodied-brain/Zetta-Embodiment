# Copyright (c) 2026 RPent Contributors
"""Frozen Role1 contracts for the audited LIBERO recovery primitives.

Recovery bundles intentionally keep short tool names because those names bind
directly to :class:`LiberoPrimitives`. Role1 sees namespaced public
contracts so its persisted decisions cannot accidentally validate against the
RoboCasa catalog.
"""

from __future__ import annotations

from robots.robocasa.tool_catalog import ToolCatalog, ToolSpec

LIBERO_RECOVERY_TOOL_NAMES = (
    "vla_execute",
    "privileged_pick_place",
    "pi0_pick",
    "pi0_doubled",
    "move_to",
    "move_pose",
    "rotate_wrist",
    "rotate_pitch",
    "release",
    "set_gripper",
    "semantic_joint_interact",
    # Proposal and bounded motion additions. Legacy names above
    # remain valid for existing frozen bundles.
    "graspgen",
    "candidate_freshness",
    "curobo_reachability",
    "curobo_motiongen_pregrasp",
    "mink_reach",
    "mink_precontact",
    "mink_engage_close",
    "mink_pull",
    "progress_liveness",
)


def role1_tool_name(short_name: str) -> str:
    short_name = str(short_name).strip()
    if short_name not in LIBERO_RECOVERY_TOOL_NAMES:
        raise KeyError(f"unknown frozen LIBERO recovery tool: {short_name}")
    return f"libero.{short_name}"


def short_tool_name(role1_name: str) -> str:
    value = str(role1_name).strip()
    if not value.startswith("libero."):
        raise KeyError(f"invalid LIBERO Role1 tool name: {value}")
    short = value.removeprefix("libero.")
    if short not in LIBERO_RECOVERY_TOOL_NAMES:
        raise KeyError(f"unknown frozen LIBERO recovery tool: {short}")
    return short


def build_libero_role1_tool_catalog() -> ToolCatalog:
    privileged_tools = {"privileged_pick_place", "semantic_joint_interact"}
    descriptions = {
        "vla_execute": "Propose a bounded Pi0.5 action chunk for the full task.",
        "privileged_pick_place": (
            "Execute a bounded simulator-side semantic pick/place recovery using "
            "the audited task object and target state."
        ),
        "pi0_pick": "Propose a bounded Pi0.5 local pick action chunk.",
        "pi0_doubled": "Propose two bounded Pi0.5 chunks with an observation refresh.",
        "move_to": "Propose a bounded Cartesian translation primitive.",
        "move_pose": "Propose a bounded Cartesian pose primitive.",
        "rotate_wrist": "Propose a bounded wrist-yaw primitive.",
        "rotate_pitch": "Propose a bounded wrist-pitch primitive.",
        "release": "Propose a bounded gripper-open primitive.",
        "set_gripper": "Propose a bounded gripper command.",
        "semantic_joint_interact": (
            "Propose a bounded privileged semantic fixture-joint contact "
            "primitive; geometry remains inside the local audited tool."
        ),
        "graspgen": (
            "Generate sensor-only 6D grasp proposals and cache an audited "
            "candidate without stepping the environment."
        ),
        "candidate_freshness": (
            "Check whether the cached grasp candidate remains valid after "
            "recent physical motion."
        ),
        "curobo_reachability": (
            "Run a proposal-only CuRobo-compatible reachability preflight; "
            "the local fallback is not a path-collision certificate."
        ),
        "curobo_motiongen_pregrasp": (
            "Execute one bounded CuRobo-compatible pregrasp and return fresh "
            "tracking evidence."
        ),
        "mink_reach": (
            "Execute one bounded Mink-compatible local reach fallback and "
            "return fresh tracking evidence."
        ),
        "mink_precontact": (
            "Execute one bounded open-gripper Mink-style precontact approach."
        ),
        "mink_engage_close": (
            "Close and micro-advance along the candidate pose axis using the "
            "bounded LIBERO fallback."
        ),
        "mink_pull": (
            "Execute one incremental closed-gripper Mink-style pull and yield "
            "to Critic/Role1 review."
        ),
        "progress_liveness": (
            "Measure recent physical progress and propose re-observation or "
            "candidate switching without stepping the environment."
        ),
    }
    return ToolCatalog(
        ToolSpec(
            name=role1_tool_name(name),
            description=descriptions[name],
            capabilities=(
                f"libero.recovery.{name}",
                *(("libero.privileged_geometry",) if name in privileged_tools else ()),
            ),
            risk=(
                "high"
                if name.startswith(("vla_", "pi0_"))
                or name in privileged_tools
                else "medium"
            ),
            proposal_only=True,
            service=False,
            local=True,
        )
        for name in LIBERO_RECOVERY_TOOL_NAMES
    )


DEFAULT_LIBERO_ROLE1_TOOL_CATALOG = build_libero_role1_tool_catalog()


__all__ = [
    "DEFAULT_LIBERO_ROLE1_TOOL_CATALOG",
    "LIBERO_RECOVERY_TOOL_NAMES",
    "build_libero_role1_tool_catalog",
    "role1_tool_name",
    "short_tool_name",
]
