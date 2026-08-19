from __future__ import annotations

from pathlib import Path

import pytest

from robots.robocasa.harness_adapter import (
    HarnessRuntimeUnavailable,
    HarnessToolRuntimeAdapter,
)
from robots.robocasa.tool_runtime import InvocationPolicy, ToolPolicyError, ToolRuntime


class _FakeExecutionPolicy:
    def __init__(self, *, allowed_tools, max_risk_level):
        self.allowed_tools = allowed_tools
        self.max_risk_level = max_risk_level


class _FakeRegistry:
    def __init__(self, names):
        self._names = set(names)
        self.last_policy = None

    def names(self):
        return set(self._names)

    def manifest(self):
        return {name: {"name": name} for name in sorted(self._names)}

    def invoke(self, name, inputs, *, policy):
        self.last_policy = policy
        return {"action": {"gripper_close": [0.0]}, "inputs": dict(inputs)}


def _adapter(*names: str) -> HarnessToolRuntimeAdapter:
    return HarnessToolRuntimeAdapter(
        _FakeRegistry(names),
        _FakeExecutionPolicy,
        root=Path("/tmp/frozen-harness"),
    )


def test_adapter_translates_policy_and_preserves_proposal_only_boundary():
    adapter = _adapter("robocasa.gripper.release")
    result = adapter.invoke(
        "robocasa.gripper.release",
        {},
        policy=InvocationPolicy(
            allow=frozenset({"robocasa.gripper.release"}),
            allow_privileged=False,
        ),
    )
    assert result["tool"] == "robocasa.gripper.release"
    assert result["proposal_only"] is True
    assert result["environment_write"] is False
    assert result["runtime_backend"] == "harness_registry"
    assert adapter.registry.last_policy.max_risk_level == "critical"


def test_adapter_rejects_tool_outside_zetta_allowlist():
    adapter = _adapter("robocasa.gripper.release")
    with pytest.raises(ToolPolicyError, match="denied"):
        adapter.invoke(
            "robocasa.gripper.release",
            {},
            policy=InvocationPolicy(allow=frozenset()),
        )


def test_adapter_requires_explicit_privileged_authorization():
    adapter = _adapter("robocasa.motion.mink_reach")
    with pytest.raises(ToolPolicyError, match="privileged"):
        adapter.invoke(
            "robocasa.motion.mink_reach",
            {},
            policy=InvocationPolicy(
                allow=frozenset({"robocasa.motion.mink_reach"}),
                allow_privileged=False,
            ),
        )


def test_adapter_fails_closed_when_binding_tool_is_missing():
    adapter = _adapter("robocasa.gripper.release")
    with pytest.raises(HarnessRuntimeUnavailable, match="missing required"):
        adapter.require_tools(
            {"robocasa.gripper.release", "robocasa.motion.mink_reach"}
        )


def test_adapter_uses_an_audited_builtin_fallback_for_incremental_deployment():
    adapter = HarnessToolRuntimeAdapter(
        _FakeRegistry({"robocasa.gripper.release"}),
        _FakeExecutionPolicy,
        root=Path("/tmp/frozen-harness"),
        fallback_runtime=ToolRuntime(),
    )
    adapter.require_tools(
        {"robocasa.gripper.release", "robocasa.motion.base_se2_astar"}
    )
    result = adapter.invoke(
        "robocasa.motion.base_se2_astar",
        {
            "start_world": [0.0, 0.0],
            "goal": [0.16, 0.0],
            "obstacles": [],
            "resolution_m": 0.08,
        },
        policy=InvocationPolicy(
            allow=frozenset({"robocasa.motion.base_se2_astar"}),
            allow_privileged=True,
        ),
    )
    assert result["runtime_backend"] == "builtin_fallback"
    assert result["harness_registry_available"] is False
    assert adapter.describe()["builtin_fallback_tool_names"] == [
        "robocasa.motion.base_se2_astar"
    ]


def test_adapter_requires_an_explicit_complete_snapshot(tmp_path, monkeypatch):
    monkeypatch.delenv("ZETTA_ROBOCASA_HARNESS_ROOT", raising=False)
    with pytest.raises(HarnessRuntimeUnavailable, match="requires"):
        HarnessToolRuntimeAdapter.from_root()
    with pytest.raises(HarnessRuntimeUnavailable, match="incomplete"):
        HarnessToolRuntimeAdapter.from_root(tmp_path)
