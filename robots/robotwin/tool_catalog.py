# Copyright (c) 2026 Zetta Contributors
"""RoboTwin tool declarations implemented in this repository.

A tool declaration never grants ownership of the simulator: control tools only
propose actions or plans for the single environment actor to review and execute.

The one structural difference from the single-arm catalogs is
:attr:`ToolSpec.arm_scoped`. RoboTwin is bimanual, so a tool that moves an arm
has to say *which* arm, and "which arm" cannot be a documentation convention:
this catalog is content-hashed into a campaign manifest
(:meth:`ToolCatalog.digest`) and cannot be edited once a campaign is running.
Making it a declared, hashed field means a candidate writer that omits the
``arm`` argument fails against the schema instead of silently driving whichever
hand the implementation happened to pick.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal

from robots.robotwin.action_contract import ARMS

RiskLevel = Literal["low", "medium", "high"]

TOOL_NAMESPACE = "robotwin."
"""Every tool in this catalog lives under one namespace."""


def _canonical_json(value: Any) -> bytes:
    """Serialize deterministically for hashing.

    Args:
        value: A JSON-compatible value.

    Returns:
        Canonical UTF-8 bytes.
    """
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


@dataclass(frozen=True, slots=True)
class ToolSpec:
    """Immutable public contract for one clean-room RoboTwin tool.

    Attributes:
        name: Fully qualified tool name under :data:`TOOL_NAMESPACE`.
        description: One-line human-readable contract.
        capabilities: Capability tags used for selection.
        risk: Declared risk level.
        privileged: Whether the tool reads privileged simulator state.
        privileged_fields: The privileged fields it may read.
        proposal_only: Whether it may only propose, never act.
        arm_scoped: Whether an invocation must carry an ``arm`` argument. True
            for anything that moves or inspects a specific hand.
        service: Whether it is served over HTTP.
        local: Whether it runs in-process.
        endpoint_env: Environment variable naming the service endpoint.
        service_path: HTTP path for a service tool.
    """

    name: str
    description: str
    capabilities: tuple[str, ...]
    risk: RiskLevel = "low"
    privileged: bool = False
    privileged_fields: tuple[str, ...] = ()
    proposal_only: bool = False
    arm_scoped: bool = False
    service: bool = False
    local: bool = True
    endpoint_env: str | None = None
    service_path: str = "/infer"

    def __post_init__(self) -> None:
        """Validate the declaration.

        Raises:
            ValueError: Any field violates the catalog contract.
        """
        if not self.name.startswith(TOOL_NAMESPACE):
            raise ValueError(f"tool names must start with {TOOL_NAMESPACE!r}")
        if not self.description.strip():
            raise ValueError("tool description must not be empty")
        if not self.capabilities:
            raise ValueError("tool capabilities must not be empty")
        if self.service == self.local:
            raise ValueError("a tool must be exactly one of service or local")
        if self.service and not self.endpoint_env:
            raise ValueError("service tools require an endpoint environment name")
        if self.local and self.endpoint_env is not None:
            raise ValueError("local tools cannot declare a service endpoint")
        if self.privileged_fields and not self.privileged:
            raise ValueError("privileged_fields require privileged=True")
        if self.risk not in {"low", "medium", "high"}:
            raise ValueError("invalid risk level")
        object.__setattr__(self, "capabilities", tuple(sorted(set(self.capabilities))))
        object.__setattr__(
            self, "privileged_fields", tuple(sorted(set(self.privileged_fields)))
        )

    def public_dict(self) -> dict[str, Any]:
        """Return the stable, secret-free representation used for hashing.

        Returns:
            A JSON-friendly dict; ``arm_scoped`` is part of the hash, so
            changing a tool's arm requirement changes the catalog digest.
        """
        return {
            "name": self.name,
            "description": self.description,
            "capabilities": list(self.capabilities),
            "risk": self.risk,
            "privileged": self.privileged,
            "privileged_fields": list(self.privileged_fields),
            "proposal_only": self.proposal_only,
            "arm_scoped": self.arm_scoped,
            "service": self.service,
            "local": self.local,
            "endpoint_env": self.endpoint_env,
            "service_path": self.service_path,
        }

    def validate_arguments(self, arguments: Mapping[str, Any]) -> str | None:
        """Check an invocation's arm argument against the declaration.

        Args:
            arguments: The proposed tool arguments.

        Returns:
            The canonical arm selector for an arm-scoped tool, else ``None``.

        Raises:
            ValueError: An arm-scoped tool was invoked without a usable ``arm``,
                or a non-arm-scoped tool was given one.
        """
        from robots.robotwin.action_contract import normalize_arm

        supplied = arguments.get("arm")
        if not self.arm_scoped:
            if supplied is not None:
                raise ValueError(
                    f"{self.name} is not arm-scoped but was given arm={supplied!r}"
                )
            return None
        if supplied is None:
            raise ValueError(
                f"{self.name} is arm-scoped: the invocation must name an arm "
                f"({', '.join(ARMS)}, or 'both')"
            )
        return normalize_arm(str(supplied))


class ToolCatalog:
    """An immutable, content-hashed set of tool declarations."""

    def __init__(self, specs: Iterable[ToolSpec]) -> None:
        """Index and hash the declarations.

        Args:
            specs: The tool declarations.

        Raises:
            ValueError: The catalog is empty or contains duplicate names.
        """
        indexed: dict[str, ToolSpec] = {}
        for spec in specs:
            if spec.name in indexed:
                raise ValueError(f"duplicate tool declaration: {spec.name}")
            indexed[spec.name] = spec
        if not indexed:
            raise ValueError("tool catalog must not be empty")
        self._specs: Mapping[str, ToolSpec] = MappingProxyType(
            dict(sorted(indexed.items()))
        )
        self._digest = hashlib.sha256(
            _canonical_json([item.public_dict() for item in self._specs.values()])
        ).hexdigest()

    @property
    def digest(self) -> str:
        """The catalog's content hash.

        Returns:
            A hex sha256 over every declaration.
        """
        return self._digest

    def names(self) -> tuple[str, ...]:
        """List the declared tool names.

        Returns:
            Names in sorted order.
        """
        return tuple(self._specs)

    def get(self, name: str) -> ToolSpec:
        """Look up one declaration.

        Args:
            name: The tool name.

        Returns:
            The declaration.

        Raises:
            KeyError: The tool is not declared.
        """
        try:
            return self._specs[name]
        except KeyError as exc:
            raise KeyError(f"unknown robot tool: {name}") from exc

    def select(
        self,
        *,
        allow: Iterable[str] | None = None,
        deny: Iterable[str] = (),
        capabilities: Iterable[str] = (),
    ) -> tuple[ToolSpec, ...]:
        """Select tools deterministically; unknown policy names fail closed.

        Args:
            allow: Names to allow; ``None`` allows everything declared.
            deny: Names to exclude.
            capabilities: When non-empty, keep only tools sharing a capability.

        Returns:
            The selected declarations, in catalog order.

        Raises:
            KeyError: A policy names a tool this catalog does not declare.
        """
        allowed = set(self._specs) if allow is None else set(allow)
        denied = set(deny)
        unknown = (allowed | denied) - set(self._specs)
        if unknown:
            raise KeyError(f"unknown RoboTwin tools in policy: {sorted(unknown)}")
        required = set(capabilities)
        return tuple(
            spec
            for name, spec in self._specs.items()
            if name in allowed
            and name not in denied
            and (not required or required.intersection(spec.capabilities))
        )

    def arm_scoped_names(self) -> tuple[str, ...]:
        """List the tools that require an ``arm`` argument.

        Returns:
            Names in catalog order.
        """
        return tuple(name for name, spec in self._specs.items() if spec.arm_scoped)

    def public_dict(self) -> dict[str, Any]:
        """Return the catalog's frozen public form.

        Returns:
            A JSON-friendly dict carrying the digest and every declaration.
        """
        return {
            "schema_version": 1,
            "digest": self.digest,
            "tools": [spec.public_dict() for spec in self._specs.values()],
        }


def _local(
    name: str,
    description: str,
    *capabilities: str,
    risk: RiskLevel = "low",
    privileged: bool = False,
    privileged_fields: tuple[str, ...] = (),
    proposal_only: bool = False,
    arm_scoped: bool = False,
) -> ToolSpec:
    """Declare an in-process tool.

    Args:
        name: Tool name.
        description: One-line contract.
        *capabilities: Capability tags.
        risk: Risk level.
        privileged: Whether it reads privileged state.
        privileged_fields: The privileged fields it may read.
        proposal_only: Whether it may only propose.
        arm_scoped: Whether it requires an ``arm`` argument.

    Returns:
        The declaration.
    """
    return ToolSpec(
        name=name,
        description=description,
        capabilities=capabilities,
        risk=risk,
        privileged=privileged,
        privileged_fields=privileged_fields,
        proposal_only=proposal_only,
        arm_scoped=arm_scoped,
        local=True,
        service=False,
    )


def _service(
    name: str,
    description: str,
    endpoint_env: str,
    *capabilities: str,
    risk: RiskLevel = "medium",
    proposal_only: bool = True,
    arm_scoped: bool = False,
    service_path: str = "/infer",
) -> ToolSpec:
    """Declare an HTTP-served tool.

    Args:
        name: Tool name.
        description: One-line contract.
        endpoint_env: Environment variable naming the endpoint.
        *capabilities: Capability tags.
        risk: Risk level.
        proposal_only: Whether it may only propose.
        arm_scoped: Whether it requires an ``arm`` argument.
        service_path: HTTP path.

    Returns:
        The declaration.
    """
    return ToolSpec(
        name=name,
        description=description,
        capabilities=capabilities,
        risk=risk,
        proposal_only=proposal_only,
        arm_scoped=arm_scoped,
        local=False,
        service=True,
        endpoint_env=endpoint_env,
        service_path=service_path,
    )


def build_robotwin_tool_catalog() -> ToolCatalog:
    """Build the RoboTwin tool catalog.

    Deliberately small. This is the set an ``adjust_bottle`` campaign needs, not
    a port of the RoboCasa catalog: roughly a third of that one is specific to a
    dishwasher, and a declaration that no implementation backs is worse than a
    missing one -- it is hashed into the manifest either way.

    Returns:
        The frozen catalog.
    """
    specs = (
        _local(
            "robotwin.observation.view_driver_state",
            "Read the 14-dim joint state and image references without advancing "
            "the simulator.",
            "perception.observe",
        ),
        _local(
            "robotwin.camera.view_meta",
            "Inspect image dimensions for the head and both wrist cameras.",
            "perception.camera_metadata",
        ),
        _local(
            "robotwin.arm.select",
            "Choose which arm should act next, given the target's position "
            "relative to the two grippers.",
            "planning.arm_selection",
            proposal_only=True,
        ),
        _local(
            "robotwin.arm.read_joint_state",
            "Read one arm's 6 joint angles and gripper opening.",
            "perception.joint_state",
            arm_scoped=True,
        ),
        _local(
            "robotwin.control.hold_arm",
            "Repeat one arm's measured joint targets so it holds position while "
            "the other arm acts. Required because RoboTwin consumes absolute "
            "targets: a zero vector commands every joint to angle zero.",
            "control.hold",
            arm_scoped=True,
            proposal_only=True,
        ),
        _local(
            "robotwin.gripper.set",
            "Propose a normalized gripper opening for one arm.",
            "control.gripper",
            arm_scoped=True,
            proposal_only=True,
        ),
        _local(
            "robotwin.vla.pi05",
            "Sample a bimanual action chunk from the Pi0.5 RoboTwin policy.",
            "control.vla_chunk",
            risk="medium",
            proposal_only=True,
        ),
        _local(
            "robotwin.verify.state",
            "Compare the observed joint state against an expected pose within a "
            "tolerance.",
            "verification.state",
        ),
        _local(
            "robotwin.critic.temporal_engagement",
            "Report whether the commanded arm has made progress over a bounded "
            "window of chunk-final frames.",
            "critic.temporal",
            arm_scoped=True,
            proposal_only=True,
        ),
    )
    return ToolCatalog(specs)


DEFAULT_ROBOTWIN_TOOL_CATALOG = build_robotwin_tool_catalog()
"""The catalog every RoboTwin campaign freezes into its manifest."""
