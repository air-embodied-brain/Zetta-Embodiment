# Copyright (c) 2026 Zetta Contributors
"""Task bindings for the audited RoboTwin tool programs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from robots.robotwin.tool_catalog import (
    DEFAULT_ROBOTWIN_TOOL_CATALOG,
    ToolCatalog,
    ToolSpec,
)
from zetta.evolution.jsonio import canonical_sha256


@dataclass(frozen=True, slots=True)
class TaskToolBinding:
    """The frozen tool set one RoboTwin task may draw from.

    Attributes:
        task: RoboTwin task name.
        tool_names: Every tool the task may use.
        vla_tool_name: The action source; must be directly selectable.
        candidate_tool_names: Tools a Stage-2 candidate may additionally bind.
        privilege_mode: How privileged reads are authorised.
        schema_version: Binding schema version.
    """

    task: str
    tool_names: tuple[str, ...]
    vla_tool_name: str = "robotwin.vla.pi05"
    candidate_tool_names: tuple[str, ...] = ()
    privilege_mode: str = "simulator_audited"
    schema_version: int = 1

    def __post_init__(self) -> None:
        """Validate the binding against the catalog.

        Raises:
            ValueError: The binding is empty, self-inconsistent, or references
                a tool the catalog does not declare.
        """
        if not self.task or not self.tool_names:
            raise ValueError("task binding requires a task and at least one tool")
        if len(set(self.tool_names)) != len(self.tool_names):
            raise ValueError("task binding contains duplicate tools")
        if self.vla_tool_name not in self.tool_names:
            raise ValueError("task binding VLA tool must be directly selectable")
        if not set(self.candidate_tool_names).issubset(self.tool_names):
            raise ValueError("candidate-only tools must belong to the task binding")
        unknown = set(self.tool_names) - set(DEFAULT_ROBOTWIN_TOOL_CATALOG.names())
        if unknown:
            raise ValueError(
                f"task binding references unknown tools: {sorted(unknown)}"
            )

    def public_dict(self) -> dict[str, Any]:
        """Return the stable representation hashed into the manifest.

        Returns:
            A JSON-friendly dict.
        """
        return {
            "schema_version": self.schema_version,
            "task": self.task,
            "tool_names": list(self.tool_names),
            "vla_tool_name": self.vla_tool_name,
            "candidate_tool_names": list(self.candidate_tool_names),
            "privilege_mode": self.privilege_mode,
        }

    @property
    def digest(self) -> str:
        """The binding's content hash.

        Returns:
            A hex sha256 over :meth:`public_dict`.
        """
        return canonical_sha256(self.public_dict())

    @property
    def arm_scoped_tool_names(self) -> tuple[str, ...]:
        """The bound tools that require an ``arm`` argument.

        Returns:
            Names in catalog order.
        """
        selected = set(self.tool_names)
        return tuple(
            name
            for name in DEFAULT_ROBOTWIN_TOOL_CATALOG.arm_scoped_names()
            if name in selected
        )

    def select(
        self, catalog: ToolCatalog = DEFAULT_ROBOTWIN_TOOL_CATALOG
    ) -> tuple[ToolSpec, ...]:
        """Resolve the binding into declarations.

        Args:
            catalog: The catalog to resolve against.

        Returns:
            The bound declarations.
        """
        return catalog.select(allow=self.tool_names)


_COMMON = (
    "robotwin.observation.view_driver_state",
    "robotwin.camera.view_meta",
    "robotwin.vla.pi05",
    "robotwin.verify.state",
)
"""Tools every RoboTwin task binding includes."""


TASK_BINDINGS = {
    "adjust_bottle": TaskToolBinding(
        task="adjust_bottle",
        # The task is defined as "lift the bottle with the *correct* arm and
        # keep it upright", so arm selection is not incidental here -- it is the
        # thing being judged. The binding therefore carries the whole arm-scoped
        # group rather than a single-arm subset.
        tool_names=tuple(
            sorted(
                {
                    *_COMMON,
                    "robotwin.arm.select",
                    "robotwin.arm.read_joint_state",
                    "robotwin.control.hold_arm",
                    "robotwin.gripper.set",
                    "robotwin.critic.temporal_engagement",
                }
            )
        ),
        candidate_tool_names=(
            "robotwin.control.hold_arm",
            "robotwin.critic.temporal_engagement",
            "robotwin.gripper.set",
        ),
    ),
}
"""Audited bindings, keyed by RoboTwin task name."""


def binding_for_task(task: str) -> TaskToolBinding:
    """Resolve an explicit task binding; unknown tasks fail closed.

    Args:
        task: The RoboTwin task name.

    Returns:
        The audited binding.

    Raises:
        KeyError: No binding has been audited for this task.
    """
    try:
        return TASK_BINDINGS[task]
    except KeyError as exc:
        raise KeyError(
            f"no audited tool binding exists for RoboTwin task {task!r}"
        ) from exc
