# Copyright (c) 2026 Zetta Contributors
"""Episode-local, append-only execution state for frozen RoboTwin recoveries.

The controller does not restore simulator state after a process crash. It keeps
a selected recovery active across multiple online chunks so a fresh VLA proposal
cannot silently erase it after one tool call.

Two things differ from a single-arm controller, and both come from the same
place -- RoboTwin has two hands and absolute joint targets:

**A recovery step that drives an arm must name it, and the name is checked at
activation.** A step whose tool is arm-scoped (:attr:`ToolSpec.arm_scoped`) but
carries no ``arm`` is not executable, and discovering that halfway through a
recovery means the episode has already been perturbed. So the whole program is
validated before the first step runs.

**The arm actually driven is verified against the frozen step.** The Actor
reports which arm it drove; if it does not match the frozen program, the step is
rejected rather than counted. Without this, a recovery frozen against the left
arm could be satisfied by moving the right one and the audit trail would still
read as compliant.
"""

from __future__ import annotations

import json
import os
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from robots.robotwin.action_contract import ArmSelectionError, normalize_arm
from robots.robotwin.tool_catalog import DEFAULT_ROBOTWIN_TOOL_CATALOG, ToolCatalog
from zetta.evolution.jsonio import canonical_sha256


@dataclass(frozen=True, slots=True)
class RecoveryExecutionState:
    """One recovery's episode-local progress.

    Attributes:
        execution_id: Unique id for this activation.
        bundle_sha256: The candidate bundle this recovery was frozen in.
        recovery_id: The frozen recovery's id.
        trigger_rule_ids: The critic rules that activated it.
        current_step_index: Index of the step awaiting execution.
        status: ``"active"`` or ``"completed"``.
        started_at_environment_step: Env step at activation.
        last_environment_step: Env step of the most recent advance.
        completed_tool_invocations: How many steps have completed.
        arm_program: The frozen per-step arm sequence, ``None`` where a step is
            not arm-scoped.
    """

    execution_id: str
    bundle_sha256: str
    recovery_id: str
    trigger_rule_ids: tuple[str, ...]
    current_step_index: int
    status: str
    started_at_environment_step: int
    last_environment_step: int
    completed_tool_invocations: int
    arm_program: tuple[str | None, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        """Return the JSON-friendly form.

        Returns:
            A plain dict.
        """
        return asdict(self)


class RecoveryContractError(ValueError):
    """A frozen recovery cannot be executed as written."""


def validate_arm_program(
    steps: Sequence[Mapping[str, Any]],
    *,
    catalog: ToolCatalog = DEFAULT_ROBOTWIN_TOOL_CATALOG,
) -> tuple[str | None, ...]:
    """Resolve and validate the per-step arm sequence of a recovery.

    Args:
        steps: The frozen recovery steps.
        catalog: Catalog the step tools are resolved against.

    Returns:
        One entry per step: the canonical arm name, or ``None`` when the step's
        tool is not arm-scoped.

    Raises:
        RecoveryContractError: A step names an unknown tool, an arm-scoped step
            omits its arm, a non-arm-scoped step carries one, or the arm is not
            a valid selector.
    """
    program: list[str | None] = []
    for index, step in enumerate(steps):
        tool = str(step.get("tool", "")).strip()
        if not tool:
            raise RecoveryContractError(f"recovery step {index} names no tool")
        try:
            spec = catalog.get(tool)
        except KeyError as exc:
            raise RecoveryContractError(
                f"recovery step {index} names tool {tool!r}, which this catalog "
                "does not declare"
            ) from exc
        arguments = step.get("arguments")
        arguments = dict(arguments) if isinstance(arguments, Mapping) else {}
        if "arm" not in arguments and step.get("arm") is not None:
            arguments["arm"] = step["arm"]
        try:
            program.append(spec.validate_arguments(arguments))
        except (ValueError, ArmSelectionError) as exc:
            raise RecoveryContractError(
                f"recovery step {index} ({tool}): {exc}"
            ) from exc
    return tuple(program)


class RecoveryController:
    """Select one frozen recovery and advance its ordered steps exactly once."""

    def __init__(
        self,
        *,
        bundle_sha256: str,
        audit_path: str | Path,
        catalog: ToolCatalog = DEFAULT_ROBOTWIN_TOOL_CATALOG,
    ) -> None:
        """Initialize the controller.

        Args:
            bundle_sha256: The active candidate bundle.
            audit_path: JSONL file the controller appends events to.
            catalog: Catalog used to resolve step tools.
        """
        self.bundle_sha256 = bundle_sha256
        self.audit_path = Path(audit_path)
        self.audit_path.parent.mkdir(parents=True, exist_ok=True)
        self.catalog = catalog
        self._rule: dict[str, Any] | None = None
        self.state: RecoveryExecutionState | None = None

    @property
    def active(self) -> bool:
        """Whether a recovery is mid-execution.

        Returns:
            ``True`` while steps remain.
        """
        return self.state is not None and self.state.status == "active"

    def _append(self, event: str, **payload: Any) -> None:
        """Append one durable audit row.

        Args:
            event: Event name.
            **payload: Event fields.
        """
        row = {
            "event_id": f"recovery-event-{uuid.uuid4().hex}",
            "event": event,
            "bundle_sha256": self.bundle_sha256,
            **payload,
        }
        with self.audit_path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    def activate(
        self,
        *,
        critic_proposals: Sequence[Mapping[str, Any]],
        recovery_rules: Sequence[Mapping[str, Any]],
        environment_step: int,
    ) -> bool:
        """Activate the first matching frozen recovery; never reset active work.

        Args:
            critic_proposals: The Critic's proposals for this boundary.
            recovery_rules: The frozen recoveries in the active bundle.
            environment_step: The current env step.

        Returns:
            ``True`` when a recovery was activated.

        Raises:
            RecoveryContractError: The selected recovery has no steps, no id, or
                an unexecutable arm program.
        """
        if self.active:
            return False
        triggers = {
            str(row.get("rule_id", ""))
            for row in critic_proposals
            if row.get("rule_id")
        }
        matches = [
            dict(rule)
            for rule in recovery_rules
            if triggers.intersection(
                str(value) for value in rule.get("trigger_rule_ids", ())
            )
        ]
        if not matches:
            return False
        # Candidate validation guarantees stable unique ids; sorting makes the
        # Actor-visible choice deterministic when one Critic is linked to more
        # than one retained parent recovery.
        matches.sort(key=lambda row: str(row.get("recovery_id", "")))
        selected = matches[0]
        steps = selected.get("steps")
        if (
            not isinstance(steps, Sequence)
            or isinstance(steps, (str, bytes, bytearray))
            or not steps
        ):
            raise RecoveryContractError("activated recovery has no executable steps")
        recovery_id = str(selected.get("recovery_id", ""))
        if not recovery_id:
            raise RecoveryContractError("activated recovery has no recovery_id")
        selected["steps"] = [dict(step) for step in steps]
        # Validate the whole arm program up front: finding an unexecutable step
        # halfway through means the episode has already been perturbed by the
        # steps that did run.
        arm_program = validate_arm_program(selected["steps"], catalog=self.catalog)
        self._rule = selected
        self.state = RecoveryExecutionState(
            execution_id=f"recovery-{uuid.uuid4().hex}",
            bundle_sha256=self.bundle_sha256,
            recovery_id=recovery_id,
            trigger_rule_ids=tuple(sorted(triggers)),
            current_step_index=0,
            status="active",
            started_at_environment_step=environment_step,
            last_environment_step=environment_step,
            completed_tool_invocations=0,
            arm_program=arm_program,
        )
        self._append(
            "activated",
            state=self.state.as_dict(),
            recovery_rule_sha256=canonical_sha256(selected),
        )
        return True

    def context(self) -> dict[str, Any] | None:
        """Describe the step awaiting execution.

        Returns:
            The Actor-visible context, or ``None`` when no recovery is active.
        """
        if not self.active or self._rule is None or self.state is None:
            return None
        steps = self._rule["steps"]
        index = self.state.current_step_index
        return {
            "execution": self.state.as_dict(),
            "recovery_id": self.state.recovery_id,
            "title": self._rule.get("title"),
            "precondition": self._rule.get("precondition"),
            "current_step": dict(steps[index]),
            "current_step_arm": self.state.arm_program[index]
            if index < len(self.state.arm_program)
            else None,
            "remaining_steps": len(steps) - index,
            "safety_constraints": list(self._rule.get("safety_constraints", ())),
            "stop_condition": self._rule.get("stop_condition"),
            "fallback": self._rule.get("fallback"),
        }

    def complete_current_step(
        self,
        *,
        selected_tool: str | None,
        environment_step: int,
        executed_horizon: int,
        executed_arm: str | None = None,
        no_op_verified: bool = False,
    ) -> RecoveryExecutionState:
        """Record that the current step executed, and advance.

        Args:
            selected_tool: The tool the Actor actually invoked.
            environment_step: The env step after execution.
            executed_horizon: Simulator steps the action advanced.
            executed_arm: The arm the Actor actually drove, when the step is
                arm-scoped.
            no_op_verified: Whether a zero-horizon step was a verified no-op.

        Returns:
            The updated execution state.

        Raises:
            ValueError: No recovery is active, or the step produced no action.
            RecoveryContractError: The executed tool or arm does not match the
                frozen program.
        """
        if not self.active or self._rule is None or self.state is None:
            raise ValueError("cannot advance an inactive recovery")
        index = self.state.current_step_index
        step = self._rule["steps"][index]
        required_tool = str(step.get("tool", ""))
        if selected_tool != required_tool:
            raise RecoveryContractError(
                "Actor did not execute the frozen recovery step tool"
            )
        required_arm = (
            self.state.arm_program[index]
            if index < len(self.state.arm_program)
            else None
        )
        if required_arm is not None:
            if executed_arm is None:
                raise RecoveryContractError(
                    f"frozen recovery step {index} drives the {required_arm} arm, "
                    "but the Actor reported no arm"
                )
            if normalize_arm(executed_arm) != required_arm:
                raise RecoveryContractError(
                    f"frozen recovery step {index} drives the {required_arm} arm, "
                    f"but the Actor drove {executed_arm!r}"
                )
        elif executed_arm is not None:
            raise RecoveryContractError(
                f"frozen recovery step {index} is not arm-scoped, but the Actor "
                f"reported driving {executed_arm!r}"
            )
        if executed_horizon < 1 and not bool(no_op_verified):
            raise ValueError("recovery step produced no environment action")
        next_index = index + 1
        complete = next_index >= len(self._rule["steps"])
        prior = self.state
        self.state = RecoveryExecutionState(
            execution_id=prior.execution_id,
            bundle_sha256=prior.bundle_sha256,
            recovery_id=prior.recovery_id,
            trigger_rule_ids=prior.trigger_rule_ids,
            current_step_index=(index if complete else next_index),
            status="completed" if complete else "active",
            started_at_environment_step=prior.started_at_environment_step,
            last_environment_step=environment_step,
            completed_tool_invocations=prior.completed_tool_invocations + 1,
            arm_program=prior.arm_program,
        )
        self._append(
            "completed" if complete else "step_advanced",
            prior_state=prior.as_dict(),
            state=self.state.as_dict(),
            selected_tool=selected_tool,
            executed_arm=executed_arm,
            executed_horizon=executed_horizon,
            no_op_verified=bool(no_op_verified),
        )
        if complete:
            self._rule = None
        return self.state


__all__ = [
    "RecoveryContractError",
    "RecoveryController",
    "RecoveryExecutionState",
    "validate_arm_program",
]
