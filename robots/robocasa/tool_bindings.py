# Copyright (c) 2026 Zetta Contributors
"""Task bindings for the audited RoboCasa tool programs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from robots.robocasa.slide_dishwasher_program import (
    BASE_ASSIST_TOOL,
    CONTACT_PUSH_TOOL,
    GUARDED_SUFFIX_TOOL,
    PROGRAM_COMPONENTS,
)
from robots.robocasa.tool_catalog import (
    DEFAULT_ROBOCASA_TOOL_CATALOG,
    ToolCatalog,
    ToolSpec,
)
from zetta.evolution.jsonio import canonical_sha256

@dataclass(frozen=True, slots=True)
class TaskToolBinding:
    task: str
    tool_names: tuple[str, ...]
    vla_tool_name: str = "robocasa.vla.groot"
    candidate_tool_names: tuple[str, ...] = ()
    program_components: tuple[str, ...] = ()
    privilege_mode: str = "simulator_audited"
    schema_version: int = 1

    def __post_init__(self) -> None:
        if not self.task or not self.tool_names:
            raise ValueError("task binding requires a task and at least one tool")
        if len(set(self.tool_names)) != len(self.tool_names):
            raise ValueError("task binding contains duplicate tools")
        if self.vla_tool_name not in self.tool_names:
            raise ValueError("task binding VLA tool must be directly selectable")
        if not set(self.candidate_tool_names).issubset(self.tool_names):
            raise ValueError("candidate-only tools must belong to the task binding")
        if len(set(self.program_components)) != len(self.program_components):
            raise ValueError("task program contains duplicate components")
        unknown = set(self.tool_names) - set(DEFAULT_ROBOCASA_TOOL_CATALOG.names())
        if unknown:
            raise ValueError(
                f"task binding references unknown tools: {sorted(unknown)}"
            )

    def public_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "task": self.task,
            "tool_names": list(self.tool_names),
            "vla_tool_name": self.vla_tool_name,
            "candidate_tool_names": list(self.candidate_tool_names),
            "program_components": list(self.program_components),
            "privilege_mode": self.privilege_mode,
        }

    @property
    def digest(self) -> str:
        return canonical_sha256(self.public_dict())

    def select(
        self, catalog: ToolCatalog = DEFAULT_ROBOCASA_TOOL_CATALOG
    ) -> tuple[ToolSpec, ...]:
        return catalog.select(allow=self.tool_names)


_COMMON = (
    "robocasa.observation.view_driver_state",
    "robocasa.camera.view_meta",
    "robocasa.vla.groot",
    "robocasa.verify.state",
)


TASK_BINDINGS = {
    "SlideDishwasherRack": TaskToolBinding(
        task="SlideDishwasherRack",
        tool_names=(CONTACT_PUSH_TOOL, GUARDED_SUFFIX_TOOL, BASE_ASSIST_TOOL),
        vla_tool_name=CONTACT_PUSH_TOOL,
        candidate_tool_names=(GUARDED_SUFFIX_TOOL, BASE_ASSIST_TOOL),
        program_components=PROGRAM_COMPONENTS,
    ),
    "OpenDrawer": TaskToolBinding(
        task="OpenDrawer",
        tool_names=tuple(
            sorted(
                {
                    *_COMMON,
                    "robocasa.geometry.back_project",
                    "robocasa.grasp.graspgen",
                    "robocasa.motion.curobo_reachability",
                    "robocasa.motion.curobo_motiongen_pregrasp",
                    "robocasa.motion.mink_reach",
                    "robocasa.motion.base_se2_astar",
                    "robocasa.gripper.set",
                    "robocasa.gripper.release",
                    "robocasa.critic.temporal_engagement",
                }
            )
        ),
    ),
    "TurnOffStove": TaskToolBinding(
        task="TurnOffStove",
        tool_names=tuple(
            sorted(
                {
                    *_COMMON,
                    "robocasa.perception.grounded_sam2",
                    "robocasa.perception.depth_anything_v2",
                    "robocasa.geometry.back_project",
                    "robocasa.control.move_pose",
                    "robocasa.control.rotate_wrist",
                    "robocasa.motion.mink_reach",
                    "robocasa.cap.servo_pose",
                    "robocasa.cap.contact_retaining_motion",
                    "robocasa.critic.temporal_engagement",
                }
            )
        ),
    ),
}


def binding_for_task(task: str) -> TaskToolBinding:
    """Resolve an explicit task binding; unknown tasks fail closed."""

    try:
        return TASK_BINDINGS[task]
    except KeyError as exc:
        raise KeyError(
            f"no audited tool binding exists for RoboCasa task {task!r}"
        ) from exc
