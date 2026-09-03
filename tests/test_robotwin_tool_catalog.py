# Copyright (c) 2026 Zetta Contributors
"""The RoboTwin tool catalog and task bindings.

The catalog is content-hashed into a campaign manifest and cannot be edited once
a campaign is running, so the tests here are less about behaviour than about
pinning the things that would be expensive to get wrong: that the arm
requirement is a **declared, hashed** property rather than a documentation
convention, and that the digest actually moves when a declaration does.
"""

from __future__ import annotations

import pytest

from robots.robotwin.action_contract import ArmSelectionError
from robots.robotwin.tool_bindings import TaskToolBinding, binding_for_task
from robots.robotwin.tool_catalog import (
    DEFAULT_ROBOTWIN_TOOL_CATALOG,
    TOOL_NAMESPACE,
    ToolCatalog,
    ToolSpec,
    build_robotwin_tool_catalog,
)


def _spec(name: str = "robotwin.test.tool", **overrides) -> ToolSpec:
    """Build a minimal valid declaration.

    Args:
        name: Tool name.
        **overrides: Field overrides.

    Returns:
        The declaration.
    """
    payload = {
        "name": name,
        "description": "A test tool.",
        "capabilities": ("test.capability",),
    }
    payload.update(overrides)
    return ToolSpec(**payload)


def test_every_tool_uses_the_robotwin_namespace() -> None:
    """A stray namespace would collide with another robot's catalog."""
    for name in DEFAULT_ROBOTWIN_TOOL_CATALOG.names():
        assert name.startswith(TOOL_NAMESPACE)
    with pytest.raises(ValueError, match="must start with"):
        _spec("robocasa.borrowed.tool")


def test_catalog_digest_is_stable_across_rebuilds() -> None:
    """Two builds of the same declarations must hash identically."""
    assert build_robotwin_tool_catalog().digest == DEFAULT_ROBOTWIN_TOOL_CATALOG.digest


def test_arm_requirement_is_part_of_the_hash() -> None:
    """Flipping ``arm_scoped`` must change the catalog digest.

    If it did not, a campaign's frozen manifest could not distinguish a catalog
    where a tool required an arm from one where it did not -- which is exactly
    the mistake this field exists to prevent.
    """
    base = ToolCatalog([_spec(arm_scoped=False)])
    scoped = ToolCatalog([_spec(arm_scoped=True)])
    assert base.digest != scoped.digest
    assert "arm_scoped" in base.public_dict()["tools"][0]


def test_arm_scoped_tools_reject_an_invocation_without_an_arm() -> None:
    """The arm is enforced by the schema, not by the implementation's choice."""
    spec = _spec(arm_scoped=True)
    with pytest.raises(ValueError, match="must name an arm"):
        spec.validate_arguments({})
    assert spec.validate_arguments({"arm": "LEFT"}) == "left"
    with pytest.raises(ArmSelectionError):
        spec.validate_arguments({"arm": "port"})


def test_non_arm_scoped_tools_reject_a_stray_arm_argument() -> None:
    """An arm on a whole-robot tool means the caller misunderstood it."""
    spec = _spec(arm_scoped=False)
    assert spec.validate_arguments({}) is None
    with pytest.raises(ValueError, match="not arm-scoped"):
        spec.validate_arguments({"arm": "left"})


def test_the_shipped_catalog_marks_the_right_tools_arm_scoped() -> None:
    """Anything that moves or reads a specific hand must be arm-scoped."""
    scoped = set(DEFAULT_ROBOTWIN_TOOL_CATALOG.arm_scoped_names())
    assert "robotwin.control.hold_arm" in scoped
    assert "robotwin.gripper.set" in scoped
    assert "robotwin.arm.read_joint_state" in scoped
    # Whole-robot tools must not be: the VLA emits both arms at once, and arm
    # *selection* is precisely the thing that has no arm yet.
    assert "robotwin.vla.pi05" not in scoped
    assert "robotwin.arm.select" not in scoped
    assert "robotwin.observation.view_driver_state" not in scoped


def test_hold_arm_is_declared_because_zero_is_not_a_hold() -> None:
    """The declaration carries the reason, so a candidate writer sees it."""
    spec = DEFAULT_ROBOTWIN_TOOL_CATALOG.get("robotwin.control.hold_arm")
    assert "absolute targets" in spec.description
    assert spec.proposal_only is True


def test_catalog_selection_fails_closed_on_unknown_names() -> None:
    """A policy naming a tool that does not exist must not silently shrink."""
    with pytest.raises(KeyError, match="unknown RoboTwin tools"):
        DEFAULT_ROBOTWIN_TOOL_CATALOG.select(allow=["robotwin.not.a.tool"])


def test_duplicate_declarations_are_rejected() -> None:
    """A duplicate would make the digest depend on iteration order."""
    with pytest.raises(ValueError, match="duplicate"):
        ToolCatalog([_spec(), _spec()])


def test_adjust_bottle_binding_carries_the_arm_group() -> None:
    """The task is judged on using the *correct* arm, so the group is bound."""
    binding = binding_for_task("adjust_bottle")
    assert binding.vla_tool_name in binding.tool_names
    assert "robotwin.arm.select" in binding.tool_names
    assert set(binding.arm_scoped_tool_names) <= set(binding.tool_names)
    assert binding.arm_scoped_tool_names, "a bimanual task must bind arm-scoped tools"
    assert len(binding.digest) == 64


def test_binding_rejects_tools_outside_the_catalog() -> None:
    """A binding is only as trustworthy as the catalog it resolves against."""
    with pytest.raises(ValueError, match="unknown tools"):
        TaskToolBinding(task="x", tool_names=("robotwin.vla.pi05", "robotwin.ghost"))


def test_binding_requires_a_selectable_vla_tool() -> None:
    """An unreachable action source would leave the episode with no policy."""
    with pytest.raises(ValueError, match="VLA tool"):
        TaskToolBinding(task="x", tool_names=("robotwin.verify.state",))


def test_unknown_task_fails_closed() -> None:
    """No binding means no audited tool set; that must not be improvised."""
    with pytest.raises(KeyError, match="no audited tool binding"):
        binding_for_task("place_empty_cup")
