# Copyright (c) 2026 Zetta Contributors
from __future__ import annotations

import json
from pathlib import Path

import pytest

from robots.robocasa.recovery_controller import RecoveryController


def _rule() -> dict:
    return {
        "recovery_id": "recover-contact",
        "title": "recover contact",
        "trigger_rule_ids": ["critic-stall"],
        "precondition": "stall is observed",
        "steps": [
            {"tool": "tool.align", "parameters": {"gain": 0.2}, "stop_when": "aligned"},
            {"tool": "tool.push", "parameters": {"gain": 0.1}, "stop_when": "moving"},
        ],
        "safety_constraints": ["finite actions"],
        "stop_condition": "progress resumes",
        "fallback": "return to VLA",
        "evidence_ids": ["artifact-1"],
    }


def test_recovery_stays_active_across_ordered_steps_and_does_not_retrigger(
    tmp_path: Path,
) -> None:
    audit = tmp_path / "recovery.jsonl"
    controller = RecoveryController(bundle_sha256="a" * 64, audit_path=audit)
    proposal = [{"rule_id": "critic-stall"}]
    assert controller.activate(
        critic_proposals=proposal, recovery_rules=[_rule()], environment_step=7
    )
    execution_id = controller.context()["execution"]["execution_id"]
    assert controller.context()["current_step"]["tool"] == "tool.align"
    assert not controller.activate(
        critic_proposals=proposal, recovery_rules=[_rule()], environment_step=8
    )
    assert controller.context()["execution"]["execution_id"] == execution_id

    state = controller.complete_current_step(
        selected_tool="tool.align", environment_step=9, executed_horizon=2
    )
    assert state.status == "active"
    assert controller.context()["current_step"]["tool"] == "tool.push"
    state = controller.complete_current_step(
        selected_tool="tool.push", environment_step=11, executed_horizon=2
    )
    assert state.status == "completed"
    assert not controller.active
    rows = [json.loads(line) for line in audit.read_text(encoding="utf-8").splitlines()]
    assert [row["event"] for row in rows] == [
        "activated",
        "step_advanced",
        "completed",
    ]
    assert len({row["event_id"] for row in rows}) == 3


def test_recovery_fails_closed_on_wrong_tool_or_zero_action(tmp_path: Path) -> None:
    controller = RecoveryController(
        bundle_sha256="b" * 64, audit_path=tmp_path / "recovery.jsonl"
    )
    controller.activate(
        critic_proposals=[{"rule_id": "critic-stall"}],
        recovery_rules=[_rule()],
        environment_step=0,
    )
    with pytest.raises(ValueError, match="frozen recovery step tool"):
        controller.complete_current_step(
            selected_tool="tool.push", environment_step=1, executed_horizon=1
        )
    with pytest.raises(ValueError, match="no environment action"):
        controller.complete_current_step(
            selected_tool="tool.align", environment_step=1, executed_horizon=0
        )


def test_recovery_accepts_explicitly_verified_proposal_only_step(tmp_path: Path) -> None:
    controller = RecoveryController(
        bundle_sha256="d" * 64, audit_path=tmp_path / "recovery.jsonl"
    )
    rule = _rule()
    rule["steps"] = [
        {"tool": "tool.inspect", "parameters": {}, "stop_when": "proposal recorded"}
    ]
    controller.activate(
        critic_proposals=[{"rule_id": "critic-stall"}],
        recovery_rules=[rule],
        environment_step=0,
    )
    state = controller.complete_current_step(
        selected_tool="tool.inspect",
        environment_step=0,
        executed_horizon=0,
        no_op_verified=True,
    )
    assert state.status == "completed"
    row = json.loads(
        (tmp_path / "recovery.jsonl").read_text(encoding="utf-8").splitlines()[-1]
    )
    assert row["no_op_verified"] is True


def test_recovery_accepts_frozen_tuple_steps(tmp_path: Path) -> None:
    controller = RecoveryController(
        bundle_sha256="c" * 64, audit_path=tmp_path / "recovery.jsonl"
    )
    rule = _rule()
    rule["steps"] = tuple(rule["steps"])
    assert controller.activate(
        critic_proposals=[{"rule_id": "critic-stall"}],
        recovery_rules=[rule],
        environment_step=3,
    )
    assert controller.context()["current_step"]["tool"] == "tool.align"
