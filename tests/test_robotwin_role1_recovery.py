# Copyright (c) 2026 Zetta Contributors
"""RoboTwin recovery execution and the Role1 Actor.

The protocol fixes the role boundaries: the Critic proposes, Role1 rules, and
only the Actor writes actions. What RoboTwin adds is that a recovery is not
executable unless it names an arm, and that "the other arm holds" has to be
composed explicitly because the robot takes absolute joint targets.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from robots.robotwin.action_contract import ACTION_DIM, ARM_SLICES, GRIPPER_OFFSET
from robots.robotwin.recovery_controller import (
    RecoveryContractError,
    RecoveryController,
    validate_arm_program,
)
from robots.robotwin.role1_actor import (
    GRIPPER_TOOL,
    HOLD_TOOL,
    ArmAwareRole1,
    Role1ActorError,
    Role1Decision,
    Role1EpisodeActor,
    action_for_recovery_step,
)
from robots.robotwin.tool_bindings import binding_for_task

BUNDLE = "a" * 64


def _state(left: float = 0.2, right: float = -0.3, grip: float = 0.8) -> list[float]:
    """Build a distinguishable joint state.

    Args:
        left: Left joint value.
        right: Right joint value.
        grip: Gripper opening for both arms.

    Returns:
        The 14-dim state.
    """
    state = np.zeros(ACTION_DIM, dtype=np.float64)
    state[ARM_SLICES["left"]] = left
    state[ARM_SLICES["right"]] = right
    state[GRIPPER_OFFSET] = grip
    state[7 + GRIPPER_OFFSET] = grip
    return list(state)


def _recovery(arm: str = "left", tool: str = HOLD_TOOL, **arguments) -> dict:
    """Build a frozen recovery with one step.

    Args:
        arm: The step's arm.
        tool: The step's tool.
        **arguments: Extra step arguments.

    Returns:
        The recovery rule.
    """
    step: dict = {"tool": tool, "arm": arm}
    if arguments:
        step["arguments"] = dict(arguments)
    return {
        "recovery_id": "rec-1",
        "trigger_rule_ids": ["left-arm-stalled"],
        "title": "Hold the stalled arm",
        "steps": [step],
    }


def _controller(tmp_path: Path) -> RecoveryController:
    """Build a controller writing into a temporary audit file.

    Args:
        tmp_path: Test directory.

    Returns:
        The controller.
    """
    return RecoveryController(
        bundle_sha256=BUNDLE, audit_path=tmp_path / "recovery.jsonl"
    )


# --------------------------------------------------------------- arm program


def test_arm_scoped_step_without_an_arm_is_rejected() -> None:
    """An arm-less step is not executable; discovering that later is too late."""
    with pytest.raises(RecoveryContractError, match="must name an arm"):
        validate_arm_program([{"tool": HOLD_TOOL}])


def test_non_arm_scoped_step_with_an_arm_is_rejected() -> None:
    """A stray arm means the candidate misunderstood the tool."""
    with pytest.raises(RecoveryContractError, match="not arm-scoped"):
        validate_arm_program([{"tool": "robotwin.verify.state", "arm": "left"}])


def test_unknown_step_tool_is_rejected() -> None:
    """A step is only as trustworthy as the catalog that declares its tool."""
    with pytest.raises(RecoveryContractError, match="does not declare"):
        validate_arm_program([{"tool": "robotwin.ghost", "arm": "left"}])


def test_valid_program_resolves_one_arm_per_step() -> None:
    """The resolved program is what execution is later verified against."""
    program = validate_arm_program(
        [
            {"tool": HOLD_TOOL, "arm": "LEFT"},
            {"tool": "robotwin.verify.state"},
        ]
    )
    assert program == ("left", None)


# ------------------------------------------------------------- controller


def test_activation_validates_the_whole_program_up_front(tmp_path: Path) -> None:
    """A bad step must not be found halfway, after the episode was perturbed."""
    controller = _controller(tmp_path)
    broken = _recovery()
    broken["steps"].append({"tool": HOLD_TOOL})  # no arm
    with pytest.raises(RecoveryContractError):
        controller.activate(
            critic_proposals=[{"rule_id": "left-arm-stalled"}],
            recovery_rules=[broken],
            environment_step=0,
        )
    assert controller.active is False


def test_activation_exposes_the_step_arm(tmp_path: Path) -> None:
    """The Actor needs the arm to compose joint targets."""
    controller = _controller(tmp_path)
    assert controller.activate(
        critic_proposals=[{"rule_id": "left-arm-stalled"}],
        recovery_rules=[_recovery(arm="right")],
        environment_step=3,
    )
    context = controller.context()
    assert context["current_step_arm"] == "right"
    assert context["remaining_steps"] == 1


def test_executing_the_wrong_arm_is_refused(tmp_path: Path) -> None:
    """Otherwise a left-arm recovery could be 'satisfied' by the right arm."""
    controller = _controller(tmp_path)
    controller.activate(
        critic_proposals=[{"rule_id": "left-arm-stalled"}],
        recovery_rules=[_recovery(arm="left")],
        environment_step=0,
    )
    with pytest.raises(RecoveryContractError, match="drove"):
        controller.complete_current_step(
            selected_tool=HOLD_TOOL,
            environment_step=1,
            executed_horizon=25,
            executed_arm="right",
        )
    with pytest.raises(RecoveryContractError, match="reported no arm"):
        controller.complete_current_step(
            selected_tool=HOLD_TOOL, environment_step=1, executed_horizon=25
        )
    # Still active: a refused step must not silently advance the program.
    assert controller.active is True


def test_matching_arm_completes_the_step(tmp_path: Path) -> None:
    """The happy path advances and writes a durable audit row."""
    controller = _controller(tmp_path)
    controller.activate(
        critic_proposals=[{"rule_id": "left-arm-stalled"}],
        recovery_rules=[_recovery(arm="left")],
        environment_step=0,
    )
    state = controller.complete_current_step(
        selected_tool=HOLD_TOOL,
        environment_step=25,
        executed_horizon=25,
        executed_arm="left",
    )
    assert state.status == "completed"
    assert controller.active is False

    rows = [
        json.loads(line)
        for line in (tmp_path / "recovery.jsonl").read_text().splitlines()
    ]
    assert [row["event"] for row in rows] == ["activated", "completed"]
    assert rows[-1]["executed_arm"] == "left"


def test_an_active_recovery_is_never_restarted(tmp_path: Path) -> None:
    """A fresh proposal must not erase a bounded program mid-flight."""
    controller = _controller(tmp_path)
    controller.activate(
        critic_proposals=[{"rule_id": "left-arm-stalled"}],
        recovery_rules=[_recovery()],
        environment_step=0,
    )
    assert (
        controller.activate(
            critic_proposals=[{"rule_id": "left-arm-stalled"}],
            recovery_rules=[_recovery()],
            environment_step=1,
        )
        is False
    )


# ------------------------------------------------------------------ actions


def test_holding_one_arm_still_sends_the_other_arm_its_measured_pose() -> None:
    """This is the whole reason the Actor composes rather than slices.

    RoboTwin takes absolute targets, so a 'hold the left arm' step must still
    say what the right arm does -- and the only correct answer is where it is.
    """
    state = _state(left=0.2, right=-0.3)
    action = action_for_recovery_step(
        {"tool": HOLD_TOOL, "arm": "left"}, arm="left", observation={"state": state}
    )
    assert action.arm_half("left") == tuple(state[ARM_SLICES["left"]])
    assert action.arm_half("right") == tuple(state[ARM_SLICES["right"]])
    assert not np.allclose(action.as_array()[ARM_SLICES["right"]], 0.0)


def test_gripper_step_moves_only_the_gripper() -> None:
    """The six joints repeat, so the arm does not drift while the hand closes."""
    state = _state(grip=0.9)
    action = action_for_recovery_step(
        {"tool": GRIPPER_TOOL, "arm": "right", "arguments": {"opening": 0.0}},
        arm="right",
        observation={"state": state},
    )
    right = action.arm_half("right")
    assert right[GRIPPER_OFFSET] == pytest.approx(0.0)
    assert right[:GRIPPER_OFFSET] == tuple(state[ARM_SLICES["right"]][:GRIPPER_OFFSET])
    assert action.arm_half("left")[GRIPPER_OFFSET] == pytest.approx(0.9)


def test_unexecutable_step_tool_is_refused_not_approximated() -> None:
    """Substituting a different motion would make the audit trail lie."""
    with pytest.raises(Role1ActorError, match="cannot turn"):
        action_for_recovery_step(
            {"tool": "robotwin.arm.select", "arm": "left"},
            arm="left",
            observation={"state": _state()},
        )


def test_missing_state_refuses_rather_than_zero_filling() -> None:
    """Guessing the idle arm's pose is exactly the mistake to avoid."""
    with pytest.raises(Role1ActorError, match="no joint state"):
        action_for_recovery_step(
            {"tool": HOLD_TOOL, "arm": "left"}, arm="left", observation={}
        )


def test_gripper_step_requires_an_opening() -> None:
    """An under-specified step is a candidate defect, not a default."""
    with pytest.raises(Role1ActorError, match="opening"):
        action_for_recovery_step(
            {"tool": GRIPPER_TOOL, "arm": "left"},
            arm="left",
            observation={"state": _state()},
        )


# -------------------------------------------------------------------- Role1


def _actor(tmp_path: Path, **overrides) -> Role1EpisodeActor:
    """Build an Actor bound to ``adjust_bottle``.

    Args:
        tmp_path: Test directory.
        **overrides: Constructor overrides.

    Returns:
        The Actor.
    """
    kwargs = {
        "decider": ArmAwareRole1(),
        "binding": binding_for_task("adjust_bottle"),
        "recovery": _controller(tmp_path),
    }
    kwargs.update(overrides)
    return Role1EpisodeActor(**kwargs)


def _vla(rows: int = 2) -> list[list[float]]:
    """Build a plausible policy chunk.

    Args:
        rows: Number of actions.

    Returns:
        A ``[rows, 14]`` chunk.
    """
    return [list(np.full(ACTION_DIM, 0.05 * (index + 1))) for index in range(rows)]


def test_reference_role1_rejects_a_proposal_that_names_no_arm(tmp_path: Path) -> None:
    """A RoboTwin recovery cannot run without an arm, however good the evidence."""
    actor = _actor(tmp_path)
    result = actor.decide_action(
        task="adjust_bottle",
        chunk_index=0,
        observation={"state": _state()},
        vla_actions=_vla(),
        critic_proposals=[{"rule_id": "stalled", "proposal": "do something"}],
        recovery_rules=[_recovery()],
    )
    assert result.source == "vla"
    assert result.decision is not None
    assert result.decision.accepted is False
    assert "names no arm" in result.decision.reason


def test_accepted_proposal_activates_and_executes_the_recovery(
    tmp_path: Path,
) -> None:
    """An arm-named proposal reaches the frozen program."""
    actor = _actor(tmp_path)
    result = actor.decide_action(
        task="adjust_bottle",
        chunk_index=1,
        observation={"state": _state()},
        vla_actions=_vla(),
        critic_proposals=[
            {"rule_id": "left-arm-stalled", "arm": "left", "proposal": "hold it"}
        ],
        recovery_rules=[_recovery(arm="left")],
    )
    assert result.source == "recovery"
    assert result.selected_tool == HOLD_TOOL
    assert result.commanded_arms == ("left",)
    assert result.decision is not None and result.decision.accepted


def test_an_unexecutable_frozen_program_falls_back_to_the_policy(
    tmp_path: Path,
) -> None:
    """A candidate defect must not stall the episode; the audit says why."""
    broken = _recovery()
    broken["steps"] = [{"tool": HOLD_TOOL}]  # arm-scoped, no arm
    actor = _actor(tmp_path)
    result = actor.decide_action(
        task="adjust_bottle",
        chunk_index=0,
        observation={"state": _state()},
        vla_actions=_vla(),
        critic_proposals=[{"rule_id": "left-arm-stalled", "arm": "left"}],
        recovery_rules=[broken],
    )
    assert result.source == "vla"
    assert result.decision is not None
    assert "recovery rejected" in result.decision.reason


def test_a_running_recovery_owns_the_boundary(tmp_path: Path) -> None:
    """A fresh policy chunk must not erase a bounded program mid-flight."""
    controller = _controller(tmp_path)
    controller.activate(
        critic_proposals=[{"rule_id": "left-arm-stalled"}],
        recovery_rules=[_recovery(arm="right")],
        environment_step=0,
    )
    actor = _actor(tmp_path, recovery=controller)
    result = actor.decide_action(
        task="adjust_bottle",
        chunk_index=5,
        observation={"state": _state()},
        vla_actions=_vla(),
    )
    assert result.source == "recovery"
    assert result.commanded_arms == ("right",)


def test_policy_chunk_passes_through_when_nothing_is_proposed(
    tmp_path: Path,
) -> None:
    """The common path stays the policy's own chunk."""
    actor = _actor(tmp_path)
    result = actor.decide_action(
        task="adjust_bottle",
        chunk_index=0,
        observation={"state": _state()},
        vla_actions=_vla(rows=3),
    )
    assert result.source == "vla"
    assert len(result.actions) == 3
    assert result.commanded_arms == ("left", "right")
    assert result.decision is None


def test_task_must_match_the_frozen_binding(tmp_path: Path) -> None:
    """An Actor bound to one task must not silently serve another."""
    actor = _actor(tmp_path)
    with pytest.raises(Role1ActorError, match="does not match"):
        actor.decide_action(
            task="lift_pot",
            chunk_index=0,
            observation={"state": _state()},
            vla_actions=_vla(),
        )


def test_malformed_policy_chunk_is_refused(tmp_path: Path) -> None:
    """A half-width chunk reaching a bimanual robot must fail loudly."""
    actor = _actor(tmp_path)
    with pytest.raises(Role1ActorError, match="not a valid action"):
        actor.decide_action(
            task="adjust_bottle",
            chunk_index=0,
            observation={"state": _state()},
            vla_actions=[[0.0] * 7],
        )
    with pytest.raises(Role1ActorError, match="empty action chunk"):
        actor.decide_action(
            task="adjust_bottle",
            chunk_index=0,
            observation={"state": _state()},
            vla_actions=[],
        )


def test_repeated_holds_terminate_the_episode(tmp_path: Path) -> None:
    """A boundary that never produces motion must be bounded, not infinite."""
    actor = _actor(tmp_path, maximum_rejections_without_action=2)
    observation = {"state": _state()}
    first = actor.hold(observation)
    assert first.terminate is False
    second = actor.hold(observation)
    assert second.terminate is True
    assert "2 consecutive boundaries" in second.termination_reason


def test_a_custom_decider_can_veto(tmp_path: Path) -> None:
    """Role1 is the decision authority; the Critic only proposes."""

    class _AlwaysReject:
        def decide(self, **_kwargs) -> Role1Decision:
            return Role1Decision(accepted=False, reason="vetoed")

    actor = _actor(tmp_path, decider=_AlwaysReject())
    result = actor.decide_action(
        task="adjust_bottle",
        chunk_index=0,
        observation={"state": _state()},
        vla_actions=_vla(),
        critic_proposals=[{"rule_id": "left-arm-stalled", "arm": "left"}],
        recovery_rules=[_recovery(arm="left")],
    )
    assert result.source == "vla"
    assert result.decision.reason == "vetoed"
