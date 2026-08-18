"""Planner-visible operational contracts for RPent tools.

The JSON schema says *how* to call a tool.  A :class:`ToolContract` says
whether the tool observes, proposes, or changes the world, and what must be
checked around the call.  Keeping this metadata separate from task recipes
makes it reusable across robots and prevents service adapters from silently
becoming environment owners.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal


RiskLevel = Literal["read_only", "low", "medium", "high"]


@dataclass(frozen=True)
class ToolContract:
    """Operational metadata used by the planner and safety gates."""

    capabilities: tuple[str, ...] = ()
    risk_level: RiskLevel = "low"
    proposal_only: bool = False
    irreversible: bool = False
    requires_reobservation: bool = False
    requirements: tuple[str, ...] = ()
    notes: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "capabilities": list(self.capabilities),
            "risk_level": self.risk_level,
            "proposal_only": self.proposal_only,
            "irreversible": self.irreversible,
            "requires_reobservation": self.requires_reobservation,
            "requirements": list(self.requirements),
            "notes": self.notes,
        }


READ_ONLY_CONTRACT = ToolContract(
    capabilities=("observation",),
    risk_level="read_only",
)
