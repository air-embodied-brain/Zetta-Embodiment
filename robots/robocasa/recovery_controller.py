# Copyright (c) 2026 Zetta Contributors
"""Episode-local, append-only execution state for frozen recovery policies.

The controller does not restore simulator state after a process crash.  It only
keeps a selected RecoveryRule active across multiple online action chunks so a
fresh VLA proposal cannot silently erase the recovery after one tool call.
"""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from zetta.evolution.jsonio import canonical_sha256


@dataclass(frozen=True, slots=True)
class RecoveryExecutionState:
    execution_id: str
    bundle_sha256: str
    recovery_id: str
    trigger_rule_ids: tuple[str, ...]
    current_step_index: int
    status: str
    started_at_environment_step: int
    last_environment_step: int
    completed_tool_invocations: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class RecoveryController:
    """Select one frozen recovery and advance its ordered steps exactly once."""

    def __init__(self, *, bundle_sha256: str, audit_path: str | Path) -> None:
        self.bundle_sha256 = bundle_sha256
        self.audit_path = Path(audit_path)
        self.audit_path.parent.mkdir(parents=True, exist_ok=True)
        self._rule: dict[str, Any] | None = None
        self.state: RecoveryExecutionState | None = None

    @property
    def active(self) -> bool:
        return self.state is not None and self.state.status == "active"

    def _append(self, event: str, **payload: Any) -> None:
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
        """Activate the unique first matching frozen recovery; never reset active work."""

        if self.active:
            return False
        triggers = {
            str(row.get("rule_id", "")) for row in critic_proposals if row.get("rule_id")
        }
        matches = [
            dict(rule)
            for rule in recovery_rules
            if triggers.intersection(str(value) for value in rule.get("trigger_rule_ids", ()))
        ]
        if not matches:
            return False
        # Candidate validation guarantees stable unique IDs.  Sorting makes the
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
            raise ValueError("activated recovery has no executable steps")
        recovery_id = str(selected.get("recovery_id", ""))
        if not recovery_id:
            raise ValueError("activated recovery has no recovery_id")
        # Dataclass serialization may preserve tuple-valued fields. Normalize
        # only this episode-local execution copy; the frozen candidate and its
        # digest remain untouched.
        selected["steps"] = [dict(step) for step in steps]
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
        )
        self._append(
            "activated",
            state=self.state.as_dict(),
            recovery_rule_sha256=canonical_sha256(selected),
        )
        return True

    def context(self) -> dict[str, Any] | None:
        if not self.active or self._rule is None or self.state is None:
            return None
        steps = self._rule["steps"]
        step = dict(steps[self.state.current_step_index])
        return {
            "execution": self.state.as_dict(),
            "recovery_id": self.state.recovery_id,
            "title": self._rule.get("title"),
            "precondition": self._rule.get("precondition"),
            "current_step": step,
            "remaining_steps": len(steps) - self.state.current_step_index,
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
        no_op_verified: bool = False,
    ) -> RecoveryExecutionState:
        if not self.active or self._rule is None or self.state is None:
            raise ValueError("cannot advance an inactive recovery")
        step = self._rule["steps"][self.state.current_step_index]
        required_tool = str(step.get("tool", ""))
        if selected_tool != required_tool:
            raise ValueError("Actor did not execute the frozen recovery step tool")
        if executed_horizon < 1 and not bool(no_op_verified):
            raise ValueError("recovery step produced no environment action")
        next_index = self.state.current_step_index + 1
        complete = next_index >= len(self._rule["steps"])
        prior = self.state
        self.state = RecoveryExecutionState(
            execution_id=prior.execution_id,
            bundle_sha256=prior.bundle_sha256,
            recovery_id=prior.recovery_id,
            trigger_rule_ids=prior.trigger_rule_ids,
            current_step_index=(prior.current_step_index if complete else next_index),
            status="completed" if complete else "active",
            started_at_environment_step=prior.started_at_environment_step,
            last_environment_step=environment_step,
            completed_tool_invocations=prior.completed_tool_invocations + 1,
        )
        self._append(
            "completed" if complete else "step_advanced",
            prior_state=prior.as_dict(),
            state=self.state.as_dict(),
            selected_tool=selected_tool,
            executed_horizon=executed_horizon,
            no_op_verified=bool(no_op_verified),
        )
        if complete:
            self._rule = None
        return self.state


__all__ = ["RecoveryController", "RecoveryExecutionState"]
