# Copyright (c) 2026 Zetta Contributors
"""RoboCasa tool declarations implemented in this repository.

A tool declaration never grants ownership of the simulator: control tools only
propose actions or plans for the single environment actor to review and
execute.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Iterable, Literal, Mapping

RiskLevel = Literal["low", "medium", "high"]


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


@dataclass(frozen=True, slots=True)
class ToolSpec:
    """Immutable public contract for one clean-room RoboCasa tool."""

    name: str
    description: str
    capabilities: tuple[str, ...]
    risk: RiskLevel = "low"
    privileged: bool = False
    privileged_fields: tuple[str, ...] = ()
    proposal_only: bool = False
    service: bool = False
    local: bool = True
    endpoint_env: str | None = None
    service_path: str = "/infer"

    def __post_init__(self) -> None:
        if not self.name.startswith(("robocasa.", "libero.")):
            raise ValueError("tool names must use a supported robot namespace")
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
            self,
            "privileged_fields",
            tuple(sorted(set(self.privileged_fields))),
        )

    def public_dict(self) -> dict[str, Any]:
        """Return the stable, secret-free representation used for hashing."""
        return {
            "name": self.name,
            "description": self.description,
            "capabilities": list(self.capabilities),
            "risk": self.risk,
            "privileged": self.privileged,
            "privileged_fields": list(self.privileged_fields),
            "proposal_only": self.proposal_only,
            "service": self.service,
            "local": self.local,
            "endpoint_env": self.endpoint_env,
            "service_path": self.service_path,
        }


class ToolCatalog:
    """Immutable catalog with deterministic allow/deny selection and digest."""

    def __init__(self, specs: Iterable[ToolSpec]) -> None:
        indexed: dict[str, ToolSpec] = {}
        for spec in specs:
            if spec.name in indexed:
                raise ValueError(f"duplicate tool: {spec.name}")
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
        return self._digest

    def names(self) -> tuple[str, ...]:
        return tuple(self._specs)

    def get(self, name: str) -> ToolSpec:
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
        """Select tools deterministically; unknown policy names fail closed."""
        allowed = set(self._specs) if allow is None else set(allow)
        denied = set(deny)
        unknown = (allowed | denied) - set(self._specs)
        if unknown:
            raise KeyError(f"unknown RoboCasa tools in policy: {sorted(unknown)}")
        required = set(capabilities)
        return tuple(
            spec
            for name, spec in self._specs.items()
            if name in allowed
            and name not in denied
            and (not required or required.intersection(spec.capabilities))
        )

    def public_dict(self) -> dict[str, Any]:
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
) -> ToolSpec:
    return ToolSpec(
        name=name,
        description=description,
        capabilities=capabilities,
        risk=risk,
        privileged=privileged,
        privileged_fields=privileged_fields,
        proposal_only=proposal_only,
        local=True,
        service=False,
    )


def _service(
    name: str,
    description: str,
    endpoint_env: str,
    *capabilities: str,
    risk: RiskLevel = "medium",
    privileged: bool = False,
    privileged_fields: tuple[str, ...] = (),
    proposal_only: bool = True,
    service_path: str = "/infer",
) -> ToolSpec:
    return ToolSpec(
        name=name,
        description=description,
        capabilities=capabilities,
        risk=risk,
        privileged=privileged,
        privileged_fields=privileged_fields,
        proposal_only=proposal_only,
        local=False,
        service=True,
        endpoint_env=endpoint_env,
        service_path=service_path,
    )


def build_robocasa_tool_catalog() -> ToolCatalog:
    """Build 23 general tools plus the audited OpenDrawer MotionGen tool."""
    specs = (
        _local(
            "robocasa.observation.view_driver_state",
            "Read state and image references without advancing the simulator.",
            "perception.observe",
        ),
        _local(
            "robocasa.camera.view_meta",
            "Inspect image dimensions and optional camera calibration metadata.",
            "perception.camera_metadata",
        ),
        _local(
            "robocasa.geometry.back_project",
            "Back-project metric depth through camera calibration into a point cloud.",
            "geometry.back_project",
            "geometry.point_cloud",
        ),
        _service(
            "robocasa.perception.grounded_sam2",
            "Propose open-vocabulary detections and masks from an RGB image.",
            "ZETTA_ROBOCASA_GROUNDED_SAM2_URL",
            "perception.open_vocabulary_segmentation",
        ),
        _service(
            "robocasa.perception.depth_anything_v2",
            "Predict scale-ambiguous relative depth for diagnosis, not metric control.",
            "ZETTA_ROBOCASA_DEPTH_ANYTHING_V2_URL",
            "perception.relative_depth",
        ),
        _service(
            "robocasa.grasp.contact_graspnet",
            "Propose 6D grasps from a camera-frame point cloud.",
            "ZETTA_ROBOCASA_CONTACT_GRASPNET_URL",
            "grasp.propose_6d",
            "perception.grasp_proposal",
            risk="high",
        ),
        _service(
            "robocasa.grasp.graspgen",
            "Generate and filter 6D grasp proposals from privileged object geometry.",
            "ZETTA_ROBOCASA_GRASPGEN_URL",
            "perception.grasp_proposal",
            "geometry.point_cloud",
            risk="high",
            privileged=True,
            privileged_fields=(
                "observation.privileged",
                "observation.privileged.*",
                "observation.state.privileged.*",
                "privileged",
                "privileged.*",
            ),
        ),
        _local(
            "robocasa.control.move_to",
            "Propose one bounded end-effector translation action.",
            "motion.relative",
            risk="medium",
            proposal_only=True,
        ),
        _local(
            "robocasa.base.move",
            "Propose one bounded robot-frame base and torso action.",
            "base.command",
            "motion.base",
            risk="high",
            proposal_only=True,
        ),
        _local(
            "robocasa.control.move_pose",
            "Propose one bounded 6D end-effector pose-servo action.",
            "motion.pose_servo",
            risk="medium",
            proposal_only=True,
        ),
        _local(
            "robocasa.control.rotate_wrist",
            "Propose one bounded wrist-yaw action.",
            "motion.orientation",
            risk="medium",
            proposal_only=True,
        ),
        _local(
            "robocasa.control.rotate_pitch",
            "Propose one bounded wrist-pitch action.",
            "motion.orientation",
            risk="medium",
            proposal_only=True,
        ),
        _local(
            "robocasa.gripper.set",
            "Propose one bounded gripper command while holding motion channels still.",
            "gripper.command",
            risk="medium",
            proposal_only=True,
        ),
        _local(
            "robocasa.gripper.release",
            "Propose one fully open gripper command while holding motion still.",
            "gripper.command",
            risk="medium",
            proposal_only=True,
        ),
        _local(
            "robocasa.verify.state",
            "Evaluate state predicates with satisfied, unsatisfied, or unknown output.",
            "verification.state_predicates",
            privileged=True,
            privileged_fields=(
                "state.privileged",
                "state.privileged.*",
                "observation.state.privileged.*",
                "privileged",
                "privileged.*",
            ),
        ),
        _service(
            "robocasa.vla.groot",
            "Propose a language-conditioned GR00T action chunk from three camera views.",
            "ZETTA_ROBOCASA_GROOT_URL",
            "vision",
            "language_conditioning",
            "continuous_control",
            risk="high",
            service_path="/act",
        ),
        _local(
            "robocasa.slide_dishwasher.vla.contact_push",
            "Expose the frozen GR00T contact-push chunk as a low-level proposal.",
            "task.slide_dishwasher.contact_push",
            risk="high",
            proposal_only=True,
        ),
        _local(
            "robocasa.slide_dishwasher.guarded_vla.terminal_suffix",
            "Clamp only reverse success-axis translation in an approved suffix.",
            "task.slide_dishwasher.guarded_suffix",
            risk="high",
            privileged=True,
            privileged_fields=("observation.state.privileged.*",),
            proposal_only=True,
        ),
        _local(
            "robocasa.slide_dishwasher.base_assisted.terminal_push",
            "Propose a positive-world-advance base-assisted terminal action.",
            "task.slide_dishwasher.base_assist",
            risk="high",
            privileged=True,
            privileged_fields=("observation.state.privileged.*",),
            proposal_only=True,
        ),
        _service(
            "robocasa.motion.mink_reach",
            "Propose a constrained local IK reach action with collision evidence.",
            "ZETTA_ROBOCASA_MINK_URL",
            "motion.pose_reach",
            "motion.constrained_ik",
            "motion.collision_aware_local",
            risk="high",
            privileged=True,
            privileged_fields=(
                "collision_world",
                "observation.privileged",
                "observation.privileged.*",
                "observation.state.privileged.*",
            ),
        ),
        _service(
            "robocasa.motion.curobo_reachability",
            "Evaluate collision-aware IK reachability without executing motion.",
            "ZETTA_ROBOCASA_CUROBO_REACHABILITY_URL",
            "motion.pose_reachability",
            "motion.constrained_ik",
            "motion.collision_aware_local",
            privileged=True,
            privileged_fields=("collision_world", "world_config"),
        ),
        _service(
            "robocasa.motion.curobo_motiongen_pregrasp",
            "Propose an isolated collision-aware pregrasp trajectory; never execute it.",
            "ZETTA_ROBOCASA_CUROBO_MOTIONGEN_URL",
            "motion.pregrasp_trajectory",
            "motion.collision_aware_global",
            risk="high",
            privileged=True,
            privileged_fields=("collision_world", "world_config"),
        ),
        _local(
            "robocasa.cap.servo_pose",
            "Propose one bounded compliant pose-servo action.",
            "cap.primitive",
            "motion.pose_servo",
            risk="medium",
            proposal_only=True,
        ),
        _local(
            "robocasa.cap.contact_retaining_motion",
            "Propose one bounded motion that retains a live contact anchor.",
            "cap.primitive",
            "motion.contact_retaining",
            risk="high",
            privileged=True,
            privileged_fields=(
                "contact",
                "observation.privileged",
                "observation.privileged.*",
                "observation.state.privileged.*",
            ),
            proposal_only=True,
        ),
        _local(
            "robocasa.critic.temporal_engagement",
            "Assess temporal contact, progress, and liveness and emit a proposal only.",
            "critic.temporal_engagement",
            "verification.contact_load",
            risk="medium",
            privileged=True,
            privileged_fields=("history.*.privileged", "privileged"),
            proposal_only=True,
        ),
        _local(
            "robocasa.motion.base_se2_astar",
            "Propose collision-inflated SE(2) A* waypoints from live geometry.",
            "motion.base_planning",
            "motion.collision_aware_global",
            risk="high",
            privileged=True,
            privileged_fields=("obstacles", "collision_world"),
            proposal_only=True,
        ),
        _local(
            "robocasa.motion.base_se2_servo",
            "Propose one closed-loop base command toward a live relative target.",
            "motion.base_servo",
            risk="high",
            proposal_only=True,
        ),
    )
    return ToolCatalog(specs)


DEFAULT_ROBOCASA_TOOL_CATALOG = build_robocasa_tool_catalog()
