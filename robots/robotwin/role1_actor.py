# Copyright (c) 2026 Zetta Contributors
"""Single-writer Actor that turns a persisted Role1 decision into an action chunk.

The role boundaries the campaign protocol fixes are load-bearing here:

- the **Critic** reads temporal evidence and may only *propose* a recovery;
- **Role1** accepts or rejects that proposal and is the sole high-level decision
  authority;
- the **recovery actor** executes only an accepted, bounded recovery program;
- only the environment actor may write simulator actions.

This module is that last one. Every tool that can influence it is declared
``proposal_only`` in the catalog: the frozen recovery step says *what* should
happen to *which arm*, and the Actor is what composes the actual joint targets.
It deliberately holds no environment client, so a failed decision cannot
partially mutate the simulator.

The bimanual contract does real work at this boundary. A recovery step like
"close the right gripper" is not a 7-number command -- RoboTwin consumes 14
absolute joint targets, so the Actor must also decide what the *left* arm does,
and the only correct answer is "hold its measured pose". That is why every
action here is built through :func:`~robots.robotwin.action_contract.compose_action`
with the observed state, and never by zero-filling.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from robots.robotwin.action_contract import (
    ACTION_DIM,
    ARM_SLICES,
    GRIPPER_OFFSET,
    GRIPPER_RANGE,
    ArmSelectionError,
    RoboTwinAction,
    action_from_flat,
    compose_action,
    hold_action,
    normalize_arm,
)
from robots.robotwin.critic_runtime import arm_from_proposal
from robots.robotwin.recovery_controller import (
    RecoveryContractError,
    RecoveryController,
)
from robots.robotwin.tool_bindings import TaskToolBinding

VLA_TOOL = "robotwin.vla.pi05"
"""The action source when no recovery is running."""

HOLD_TOOL = "robotwin.control.hold_arm"
"""Frozen step that pins one arm at its measured pose."""

GRIPPER_TOOL = "robotwin.gripper.set"
"""Frozen step that drives one arm's gripper."""

ROLE1_SYSTEM_CONTRACT = """You are the sole high-level Role1 decision authority
for one event in a bimanual robot episode. Tool outputs are proposals only. A
critic may only recommend rejecting the current action; it cannot execute,
replace, recover, switch, or terminate. Select no default tool and make no
implicit fallback. Use only the event's allowed stages and tools. Do not
request, infer, or mention experiment identity, random generators, or future
rollout order. Return exactly one JSON object and no Markdown or surrounding
text.

Two facts about this robot constrain every decision you make.

The robot has two arms. Every action is 14 absolute joint targets, laid out as
left arm (6 joints, 1 gripper) then right arm (6 joints, 1 gripper). A decision
that drives an arm must name it: "left" or "right". A proposal that names no arm
is not executable and must be rejected rather than resolved to a default. There
is no default hand.

The targets are absolute, not deltas. Commanding an arm to hold position means
repeating its measured joint values; a zero vector commands every joint to angle
zero and throws the arm across the workspace. When you accept a recovery that
drives one arm, the other arm holds -- that is a decision you are making, not a
side effect.

Evidence is chunk-granular. This environment submits a whole action chunk and
returns only the final frame, so you never see intermediate steps and a critic's
dwell counts chunk boundaries, not simulator steps. Do not reason about
per-step motion you were not shown.
"""
"""The system contract for a model-backed Role1 on this family.

Frozen into the campaign's prompt contract, so its hash is part of what a
preregistration commits to.
"""


EXECUTABLE_STEP_TOOLS = frozenset({HOLD_TOOL, GRIPPER_TOOL})
"""The frozen step tools this Actor knows how to turn into joint targets.

Deliberately small. A step naming anything else is refused rather than
approximated: silently substituting a different motion for an unimplemented one
would make the audit trail claim a recovery ran that never did.
"""


class Role1ActorError(RuntimeError):
    """The Actor cannot honour a decision or a frozen step."""


@dataclass(frozen=True, slots=True)
class Role1Decision:
    """Role1's verdict on one Critic proposal.

    Attributes:
        accepted: Whether the proposal is accepted.
        reason: Why, recorded verbatim in the audit trail.
        proposal_id: The proposal this verdict answers.
        arm: The arm the verdict concerns, when the proposal named one.
    """

    accepted: bool
    reason: str
    proposal_id: str | None = None
    arm: str | None = None

    def public_dict(self) -> dict[str, Any]:
        """Return the JSON-friendly form.

        Returns:
            A plain dict.
        """
        return {
            "accepted": bool(self.accepted),
            "reason": self.reason,
            "proposal_id": self.proposal_id,
            "arm": self.arm,
        }


class Role1Decider(Protocol):
    """The seam where a model-backed Role1 plugs in.

    Kept narrow on purpose: the Actor needs a verdict, not a conversation, and
    an implementation that reaches the simulator would break the single-writer
    property this module exists to hold.
    """

    def decide(
        self,
        *,
        task: str,
        chunk_index: int,
        features: Mapping[str, Any],
        proposals: Sequence[Mapping[str, Any]],
    ) -> Role1Decision:
        """Accept or reject the leading Critic proposal.

        Args:
            task: The task under evaluation.
            chunk_index: Index of the chunk boundary being decided.
            features: The Critic feature plane for this boundary.
            proposals: The Critic's proposals, most significant first.

        Returns:
            The verdict.
        """
        ...


class ArmAwareRole1(Role1Decider):
    """Reference Role1: accepts only proposals that are actually executable.

    This is not a stand-in for a model-backed Role1 -- it is the floor beneath
    one. A RoboTwin recovery cannot run without an arm, so a proposal that names
    none is rejected here regardless of how convincing its evidence is, and a
    model-backed decider inherits the same requirement by being asked for the
    same verdict shape.
    """

    def __init__(self, *, require_arm: bool = True) -> None:
        """Configure the reference decider.

        Args:
            require_arm: Whether a proposal must name an arm to be accepted.
        """
        self.require_arm = require_arm

    def decide(
        self,
        *,
        task: str,
        chunk_index: int,
        features: Mapping[str, Any],
        proposals: Sequence[Mapping[str, Any]],
    ) -> Role1Decision:
        """Accept the first proposal that names an arm.

        Args:
            task: The task under evaluation.
            chunk_index: The chunk boundary index.
            features: The Critic feature plane.
            proposals: The Critic's proposals.

        Returns:
            The verdict.
        """
        if not proposals:
            return Role1Decision(accepted=False, reason="no critic proposal")
        leading = proposals[0]
        proposal_id = str(leading.get("rule_id") or leading.get("proposal_id") or "")
        arm = arm_from_proposal(leading)
        if self.require_arm and arm is None:
            return Role1Decision(
                accepted=False,
                reason=(
                    "proposal names no arm; a RoboTwin recovery cannot be "
                    "executed without one"
                ),
                proposal_id=proposal_id or None,
            )
        return Role1Decision(
            accepted=True,
            reason=str(leading.get("proposal") or leading.get("reason") or "accepted"),
            proposal_id=proposal_id or None,
            arm=arm,
        )


@dataclass(frozen=True, slots=True)
class Role1ActorResult:
    """What the Actor decided to execute at one chunk boundary.

    Attributes:
        actions: The ``[chunk, 14]`` action block to submit.
        source: ``"vla"``, ``"recovery"`` or ``"hold"``.
        commanded_arms: The arms the action actually commands.
        selected_tool: The frozen step tool executed, when in a recovery.
        decision: Role1's verdict, when one was taken.
        terminate: Whether the episode should stop.
        termination_reason: Why, when terminating.
    """

    actions: tuple[tuple[float, ...], ...]
    source: str
    commanded_arms: tuple[str, ...]
    selected_tool: str | None = None
    decision: Role1Decision | None = None
    terminate: bool = False
    termination_reason: str = ""

    def as_array(self) -> np.ndarray:
        """Return the action block as float32.

        Returns:
            A ``[chunk, 14]`` array.
        """
        return np.asarray(self.actions, dtype=np.float32)


def _state_from_observation(observation: Mapping[str, Any]) -> np.ndarray:
    """Read the 14-dim joint state out of an observation.

    Args:
        observation: The chunk-final observation.

    Returns:
        The state as a float64 array.

    Raises:
        Role1ActorError: The observation carries no usable state. The Actor
            cannot compose a hold without it, and guessing would be exactly the
            zero-fill mistake the action contract exists to prevent.
    """
    state = observation.get("state")
    if state is None:
        raise Role1ActorError("observation carries no joint state")
    array = np.asarray(state, dtype=np.float64).reshape(-1)
    if array.shape != (ACTION_DIM,) or not np.isfinite(array).all():
        raise Role1ActorError(
            f"observation state must be {ACTION_DIM} finite values, got shape "
            f"{array.shape}"
        )
    return array


def _gripper_half(state: np.ndarray, arm: str, opening: float) -> list[float]:
    """Build one arm's command that only moves the gripper.

    The six joints repeat their measured values, so the arm does not drift while
    the gripper closes.

    Args:
        state: The observed 14-dim state.
        arm: The arm to command.
        opening: Normalised gripper opening.

    Returns:
        The arm's 7-slot command.

    Raises:
        Role1ActorError: The opening is outside the contract's range.
    """
    low, high = GRIPPER_RANGE
    if not low <= float(opening) <= high:
        raise Role1ActorError(f"gripper opening {opening} is outside [{low}, {high}]")
    half = list(state[ARM_SLICES[arm]])
    half[GRIPPER_OFFSET] = float(opening)
    return half


def action_for_recovery_step(
    step: Mapping[str, Any],
    *,
    arm: str | None,
    observation: Mapping[str, Any],
) -> RoboTwinAction:
    """Compose the joint targets one frozen recovery step calls for.

    Args:
        step: The frozen step.
        arm: The step's validated arm, or ``None`` when it is not arm-scoped.
        observation: The chunk-final observation.

    Returns:
        The composed action.

    Raises:
        Role1ActorError: The step names a tool this Actor cannot execute, or its
            arguments are unusable.
    """
    tool = str(step.get("tool", ""))
    if tool not in EXECUTABLE_STEP_TOOLS:
        raise Role1ActorError(
            f"frozen recovery step names {tool!r}, which this Actor cannot turn "
            f"into joint targets; executable steps are {sorted(EXECUTABLE_STEP_TOOLS)}"
        )
    state = _state_from_observation(observation)
    if arm is None:
        raise Role1ActorError(f"{tool} is arm-scoped but the step resolved no arm")
    selector = normalize_arm(arm)
    if selector == "both":
        raise Role1ActorError(f"{tool} must name a single arm, not 'both'")

    if tool == HOLD_TOOL:
        # Holding one arm still means sending a full 14-vector: the other arm
        # keeps its measured pose too, which is exactly what compose_action does.
        return compose_action(state, **{selector: list(state[ARM_SLICES[selector]])})

    arguments = step.get("arguments")
    arguments = dict(arguments) if isinstance(arguments, Mapping) else {}
    if "opening" not in arguments:
        raise Role1ActorError(f"{GRIPPER_TOOL} step requires an 'opening' argument")
    half = _gripper_half(state, selector, float(arguments["opening"]))
    return compose_action(state, **{selector: half})


class Role1EpisodeActor:
    """Resolve one action boundary through proposal-only tools and Role1."""

    def __init__(
        self,
        *,
        decider: Role1Decider,
        binding: TaskToolBinding,
        recovery: RecoveryController | None = None,
        maximum_rejections_without_action: int = 4,
    ) -> None:
        """Initialize the Actor.

        Args:
            decider: Role1's verdict source.
            binding: The task's frozen tool binding.
            recovery: The recovery controller, when a bundle is active.
            maximum_rejections_without_action: Bound on consecutive boundaries
                resolved without executing anything.

        Raises:
            ValueError: The bound is not positive.
        """
        if maximum_rejections_without_action < 1:
            raise ValueError("maximum_rejections_without_action must be positive")
        self.decider = decider
        self.binding = binding
        self.recovery = recovery
        self.maximum_rejections_without_action = maximum_rejections_without_action
        self._consecutive_holds = 0

    def decide_action(
        self,
        *,
        task: str,
        chunk_index: int,
        observation: Mapping[str, Any],
        vla_actions: Sequence[Sequence[float]],
        features: Mapping[str, Any] | None = None,
        critic_proposals: Sequence[Mapping[str, Any]] = (),
        recovery_rules: Sequence[Mapping[str, Any]] = (),
        environment_step: int = 0,
    ) -> Role1ActorResult:
        """Resolve one chunk boundary into an action block.

        Args:
            task: The task; must match the frozen binding.
            chunk_index: Index of this chunk boundary.
            observation: The chunk-final observation.
            vla_actions: The policy's ``[chunk, 14]`` proposal.
            features: The Critic feature plane for this boundary.
            critic_proposals: The Critic's proposals.
            recovery_rules: The frozen recoveries in the active bundle.
            environment_step: The current env step, for the recovery audit.

        Returns:
            The resolved action block and its provenance.

        Raises:
            Role1ActorError: The task does not match the binding, or the VLA
                proposal is unusable and no recovery covers the boundary.
        """
        if task != self.binding.task:
            raise Role1ActorError("Actor task does not match its frozen tool binding")

        # 1. An already-running recovery owns the boundary: a fresh VLA proposal
        #    must not be able to erase a bounded program mid-flight.
        if self.recovery is not None and self.recovery.active:
            return self._execute_recovery_step(observation)

        # 2. Otherwise Role1 rules on the Critic's proposals.
        decision: Role1Decision | None = None
        if critic_proposals:
            decision = self.decider.decide(
                task=task,
                chunk_index=chunk_index,
                features=dict(features or {}),
                proposals=list(critic_proposals),
            )
            if decision.accepted and self.recovery is not None:
                try:
                    activated = self.recovery.activate(
                        critic_proposals=critic_proposals,
                        recovery_rules=recovery_rules,
                        environment_step=environment_step,
                    )
                except RecoveryContractError as exc:
                    # An unexecutable frozen program is a candidate defect, not
                    # a reason to stall the episode: fall back to the policy and
                    # let the audit record why.
                    return self._vla_result(
                        vla_actions,
                        decision=Role1Decision(
                            accepted=False,
                            reason=f"recovery rejected: {exc}",
                            proposal_id=decision.proposal_id,
                            arm=decision.arm,
                        ),
                    )
                if activated:
                    return self._execute_recovery_step(observation, decision=decision)

        # 3. No accepted, executable recovery: run the policy's own chunk.
        return self._vla_result(vla_actions, decision=decision)

    def _execute_recovery_step(
        self,
        observation: Mapping[str, Any],
        *,
        decision: Role1Decision | None = None,
    ) -> Role1ActorResult:
        """Compose the action for the recovery step awaiting execution.

        Args:
            observation: The chunk-final observation.
            decision: The verdict that activated the recovery, if this is its
                first step.

        Returns:
            The action block for the current step.

        Raises:
            Role1ActorError: No recovery context is available.
        """
        if self.recovery is None:
            raise Role1ActorError("no recovery controller is attached")
        context = self.recovery.context()
        if context is None:
            raise Role1ActorError("recovery is active but exposed no context")
        step = context["current_step"]
        arm = context.get("current_step_arm")
        action = action_for_recovery_step(step, arm=arm, observation=observation)
        self._consecutive_holds = 0
        return Role1ActorResult(
            actions=(tuple(action.values),),
            source="recovery",
            commanded_arms=action.commanded,
            selected_tool=str(step.get("tool", "")),
            decision=decision,
        )

    def _vla_result(
        self,
        vla_actions: Sequence[Sequence[float]],
        *,
        decision: Role1Decision | None,
    ) -> Role1ActorResult:
        """Pass the policy's chunk through, recording its arm provenance.

        Args:
            vla_actions: The policy's ``[chunk, 14]`` proposal.
            decision: Role1's verdict, when one was taken.

        Returns:
            The action block.

        Raises:
            Role1ActorError: The proposal is empty or malformed.
        """
        if not len(vla_actions):
            raise Role1ActorError("policy returned an empty action chunk")
        try:
            rows = [action_from_flat(row) for row in vla_actions]
        except (ValueError, ArmSelectionError) as exc:
            raise Role1ActorError(f"policy chunk is not a valid action: {exc}") from exc
        self._consecutive_holds = 0
        return Role1ActorResult(
            actions=tuple(tuple(row.values) for row in rows),
            source="vla",
            commanded_arms=rows[0].commanded,
            selected_tool=VLA_TOOL,
            decision=decision,
        )

    def hold(self, observation: Mapping[str, Any]) -> Role1ActorResult:
        """Emit an explicit both-arm hold, bounded by the no-action limit.

        Args:
            observation: The chunk-final observation.

        Returns:
            The hold action, terminating once the bound is reached.
        """
        action = hold_action(_state_from_observation(observation))
        self._consecutive_holds += 1
        exhausted = self._consecutive_holds >= self.maximum_rejections_without_action
        return Role1ActorResult(
            actions=(tuple(action.values),),
            source="hold",
            commanded_arms=action.commanded,
            terminate=exhausted,
            termination_reason=(
                f"no action taken for {self._consecutive_holds} consecutive boundaries"
                if exhausted
                else ""
            ),
        )


__all__ = [
    "ROLE1_SYSTEM_CONTRACT",
    "ArmAwareRole1",
    "Role1ActorError",
    "Role1ActorResult",
    "Role1Decider",
    "Role1Decision",
    "Role1EpisodeActor",
    "action_for_recovery_step",
]
