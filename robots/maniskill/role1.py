# Copyright (c) 2026 Zetta Contributors
"""Deterministic Role1 review of frozen, bounded recovery proposals."""

from __future__ import annotations

from typing import Any

from robots.maniskill.contracts import validate_step
from zetta.evolution.models import RecoveryRule


class BoundedRole1:
    """The sole recovery decision authority for this preregistered profile."""

    def review(
        self,
        proposals: list[dict[str, Any]],
        recoveries: tuple[RecoveryRule, ...],
        *,
        remaining_steps: int,
    ) -> dict[str, Any]:
        triggers = {proposal["rule_id"] for proposal in proposals}
        for recovery in sorted(recoveries, key=lambda value: value.recovery_id):
            if not triggers.intersection(recovery.trigger_rule_ids):
                continue
            for step in recovery.steps:
                validate_step(step)
            budget = sum(step.parameters["max_steps"] for step in recovery.steps)
            if budget > 0 and budget <= remaining_steps:
                return {
                    "accepted": True,
                    "recovery_id": recovery.recovery_id,
                    "reason": "frozen tools and total recovery budget satisfy the task contract",
                    "environment_write": False,
                }
        return {
            "accepted": False,
            "recovery_id": None,
            "reason": "no matching recovery fits the remaining episode budget",
            "environment_write": False,
        }
