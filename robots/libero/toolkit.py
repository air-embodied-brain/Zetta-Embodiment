# Copyright (c) 2026 Zetta Contributors
"""LIBERO toolkit: common tools + LIBERO primitives.

Inherits the common file/IO tools from :class:`Toolkit` and registers the
LIBERO primitives (``move_to``, ``pi0_pick``, ``release``, ...) on top.
"""
from __future__ import annotations

import fnmatch
import shutil
import time
from functools import partial
from typing import Any

from robots.libero import tools as libero_tools
from robots.libero.contact_graspnet import ContactGraspNetAdapter
from robots.libero.graspgen import GraspGenAdapter
from zetta.tools.contracts import READ_ONLY_CONTRACT, ToolContract
from zetta.tools.toolkit import Toolkit
from zetta.utils.logging import get_logger, get_output_dir


class LiberoToolkit(Toolkit):
    """Toolkit for the LIBERO environment."""

    TOOL_PROFILES = {
        "all": None,
        "pi05_only": {
            "read_text_file",
            "write_text_file",
            "list_dir",
            "finish",
            "describe_tools",
            "view_driver_state",
            "view_camera_meta",
            "verify_transition",
            "vla_execute",
            "pi0_pick",
            "pi0_doubled",
        },
    }

    # Tool schemas keyed by name (built once from the canonical ordered list
    # in libero_tools.TOOLS_SPEC) so each tool registers with its own spec.
    _SPECS = {spec["name"]: spec for spec in libero_tools.TOOLS_SPEC}

    def __init__(
        self,
        *,
        primitives_kwargs: dict[str, Any],
        video_path: str | None = None,
        dashboard: Any = None,
        tool_profile: str = "all",
    ) -> None:
        super().__init__(dashboard=dashboard)
        self._next_step: int = 0
        self._video_path: str | None = video_path
        self.init_primitives_clean(primitives_kwargs=primitives_kwargs)
        self._contact_graspnet = ContactGraspNetAdapter()
        self._graspgen = GraspGenAdapter()
        self._register_libero_tools()
        if tool_profile not in self.TOOL_PROFILES:
            raise ValueError(
                f"unknown LIBERO tool profile {tool_profile!r}; "
                f"expected one of {sorted(self.TOOL_PROFILES)}"
            )
        allowed = self.TOOL_PROFILES[tool_profile]
        if allowed is not None:
            self.retain_tools(set(allowed))
        self.tool_profile = tool_profile

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------
    def _register_libero_tools(self) -> None:
        specs = self._SPECS
        # Inspection tools do not advance environment state. Most are stateless
        # module functions; segment is bound to the primitives-owned SAM3 client.
        inspection_handlers = {
            "view_driver_state": libero_tools.view_driver_state,
            "view_camera_meta": libero_tools.view_camera_meta,
            "back_project": libero_tools.back_project,
            "segment": self._primitives.segment,
            "verify_transition": libero_tools.verify_transition,
        }
        for name, handler in inspection_handlers.items():
            contract = READ_ONLY_CONTRACT
            if name == "segment":
                contract = ToolContract(
                    capabilities=("observation", "segmentation", "proposal"),
                    risk_level="read_only",
                    proposal_only=True,
                    requirements=("SAM3 service", "matching image/world-map artifacts"),
                    notes="A mask/world point is a proposal; inspect evidence before motion.",
                )
            self.add_tool(name, specs[name], handler, contract=contract)
        self.add_tool(
            "collision_check",
            self._collision_spec(),
            self._collision_check,
            contract=ToolContract(
                capabilities=("observation", "contact", "collision", "force"),
                risk_level="read_only",
                requirements=("LIBERO privileged MuJoCo contact sensor",),
                notes=(
                    "Current-state sensor with real wrist-F/T, torque, or tactile "
                    "analogues; it is not a future-path collision certificate."
                ),
            ),
        )
        self.add_tool(
            "contact_graspnet",
            self._contact_graspnet_spec(),
            self._contact_graspnet.propose,
            contract=ToolContract(
                capabilities=("observation", "point_cloud", "grasp_proposal"),
                risk_level="read_only",
                proposal_only=True,
                requirements=(
                    "configured Contact-GraspNet service",
                    "calibrated current LIBERO metric depth",
                ),
                notes=(
                    "Returns 6D proposals only. Validate frame calibration, "
                    "reachability, and collision state before motion."
                ),
            ),
        )
        self.add_tool(
            "graspgen",
            self._graspgen_spec(),
            self._graspgen.propose,
            contract=ToolContract(
                capabilities=("observation", "point_cloud", "grasp_proposal"),
                risk_level="read_only",
                proposal_only=True,
                requirements=(
                    "configured GraspGen service",
                    "calibrated current LIBERO metric depth",
                ),
                notes=(
                    "Sensor point clouds only: the RoboCasa privileged handle geometry "
                    "adapter is deliberately not exposed."
                ),
            ),
        )
        # Primitive tools: each goes through _step, which looks up the
        # matching primitive method via getattr at call time.
        for name in libero_tools.PRIMITIVE_TOOL_NAMES:
            self.add_tool(
                name,
                specs[name],
                partial(self._step, name),
                contract=self._primitive_contract(name),
            )
        for name in libero_tools.RECOVERY_MOTION_TOOL_NAMES:
            self.add_tool(
                name,
                specs[name],
                partial(self._motion_tool, name),
                contract=self._motion_contract(name),
            )
        self.add_tool(
            "assess_tool_applicability",
            self._applicability_spec(),
            self._assess_tool_applicability,
            contract=READ_ONLY_CONTRACT,
        )

    @staticmethod
    def _primitive_contract(name: str) -> ToolContract:
        if name == "vla_execute":
            return ToolContract(
                capabilities=("vla", "contact", "manipulation", "receding_horizon"),
                risk_level="high",
                irreversible=True,
                requires_reobservation=True,
                requirements=("compatible 7-D LIBERO VLA service",),
                notes="Use short horizons near contact and verify every stage handoff.",
            )
        if name in {"pi0_pick", "pi0_doubled"}:
            return ToolContract(
                capabilities=("vla", "manipulation"),
                risk_level="high",
                irreversible=True,
                requires_reobservation=True,
                requirements=("Pi0.5-compatible VLA service",),
            )
        if name in {"release", "set_gripper"}:
            return ToolContract(
                capabilities=("gripper", "contact"),
                risk_level="high" if name == "release" else "medium",
                irreversible=name == "release",
                requires_reobservation=True,
            )
        if name == "semantic_joint_interact":
            return ToolContract(
                capabilities=("motion", "contact", "privileged_geometry"),
                risk_level="high",
                irreversible=True,
                requires_reobservation=True,
                requirements=(
                    "explicit privileged action authorization",
                    "declared semantic fixture joint",
                ),
                notes=(
                    "Simulator geometry is consumed inside the local primitive; "
                    "the Actor and VLA do not receive absolute coordinates."
                ),
            )
        return ToolContract(
            capabilities=("motion", "manipulation"),
            risk_level="medium",
            requires_reobservation=True,
            requirements=("reachable target", "collision-aware waypoint selection"),
            notes=(
                "Zetta does not treat the RoboCasa Mink/curobo adapters as valid "
                "LIBERO collision or IK certificates."
            ),
        )

    @staticmethod
    def _motion_contract(name: str) -> ToolContract:
        if name in libero_tools.RECOVERY_PROPOSAL_TOOL_NAMES:
            return ToolContract(
                capabilities=("observation", "proposal", "motion_planning"),
                risk_level="read_only",
                proposal_only=True,
                requires_reobservation=False,
                notes="Proposal-only evidence; this tool never advances LIBERO.",
            )
        return ToolContract(
            capabilities=("motion", "contact", "receding_horizon", "motion_planning"),
            risk_level="high",
            irreversible=True,
            requires_reobservation=True,
            notes=(
                "One bounded action wrapper; inspect the returned state before "
                "selecting another motion tool."
            ),
        )

    def _motion_tool(self, name: str, **kwargs: Any) -> dict[str, Any]:
        handler = getattr(self._primitives, str(name), None)
        if handler is None or not callable(handler):
            return {"name": str(name), "status": "unavailable", "error": "primitive handler missing"}
        # Read-only proposals are intentionally returned directly. Mutating
        # motion wrappers use the same dump/video/re-observation path as every
        # legacy primitive so Toolkit calls cannot create unindexed actions.
        if str(name) in libero_tools.RECOVERY_PROPOSAL_TOOL_NAMES:
            return dict(handler(**kwargs))
        return dict(self._step(str(name), **kwargs))

    @staticmethod
    def _applicability_spec() -> dict[str, Any]:
        return {
            "name": "assess_tool_applicability",
            "description": (
                "Advisory preflight for one tool at the current state. It reports "
                "the operational contract, episode status, optional target "
                "distance/workspace checks, and irreversible-action acknowledgement. "
                "It is not an IK or collision certificate."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "tool": {"type": "string"},
                    "target_xyz": {
                        "type": ["array", "null"],
                        "items": {"type": "number"},
                        "minItems": 3,
                        "maxItems": 3,
                    },
                    "acknowledge_irreversible": {"type": "boolean"},
                },
                "required": ["tool"],
            },
        }

    @staticmethod
    def _collision_spec() -> dict[str, Any]:
        return {
            "name": "collision_check",
            "description": (
                "Read current privileged MuJoCo contact pairs and available contact "
                "forces. allowed_geom_patterns marks intentional target contacts; all "
                "other robot contacts are reported as unexpected. This models a real "
                "force/torque or tactile sensor and does not predict future paths."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "allowed_geom_patterns": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "Optional glob patterns for current intentional contact "
                            "geometries, for example '*bowl*'."
                        ),
                    },
                    "minimum_force_n": {
                        "type": "number",
                        "minimum": 0,
                        "description": (
                            "Ignore contacts below this normal force only when force "
                            "evidence is available (default 0)."
                        ),
                    },
                    "include_all_contacts": {
                        "type": "boolean",
                        "description": "Also return non-robot scene contacts.",
                    },
                    "max_contacts": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 256,
                    },
                },
            },
        }

    @staticmethod
    def _contact_graspnet_spec() -> dict[str, Any]:
        return {
            "name": "contact_graspnet",
            "description": (
                "Health-check or request proposal-only Contact-GraspNet 6D grasps "
                "from the current calibrated LIBERO camera depth. The tool never "
                "steps the environment."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "mode": {"type": "string", "enum": ["health", "propose"]},
                    "camera": {
                        "type": "string",
                        "enum": ["agentview", "wrist"],
                    },
                    "step": {"type": ["integer", "null"], "minimum": 0},
                    "max_candidates": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 64,
                    },
                    "max_points": {
                        "type": "integer",
                        "minimum": 32,
                        "maximum": 200000,
                    },
                    "min_depth_m": {"type": "number", "exclusiveMinimum": 0},
                    "max_depth_m": {"type": "number", "exclusiveMinimum": 0},
                    "timeout_s": {
                        "type": "number",
                        "minimum": 1,
                        "maximum": 900,
                    },
                },
            },
        }

    @staticmethod
    def _graspgen_spec() -> dict[str, Any]:
        return {
            "name": "graspgen",
            "description": (
                "Health-check or request proposal-only GraspGen 6D grasps from a "
                "current sensor point cloud. target_world_xyz may come from current "
                "SAM/back-projection evidence; no MuJoCo object geometry is used."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "mode": {"type": "string", "enum": ["health", "propose"]},
                    "camera": {"type": "string", "enum": ["agentview", "wrist"]},
                    "step": {"type": ["integer", "null"], "minimum": 0},
                    "target_world_xyz": {
                        "type": ["array", "null"],
                        "items": {"type": "number"},
                        "minItems": 3,
                        "maxItems": 3,
                    },
                    "crop_radius_m": {
                        "type": "number",
                        "minimum": 0.01,
                        "maximum": 1.0,
                    },
                    "max_candidates": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 64,
                    },
                    "max_points": {
                        "type": "integer",
                        "minimum": 32,
                        "maximum": 65536,
                    },
                    "min_depth_m": {"type": "number", "exclusiveMinimum": 0},
                    "max_depth_m": {"type": "number", "exclusiveMinimum": 0},
                    "filter_collisions": {"type": "boolean"},
                    "remove_outliers": {"type": "boolean"},
                    "timeout_s": {"type": "number", "minimum": 1, "maximum": 900},
                },
            },
        }

    def _collision_check(
        self,
        allowed_geom_patterns: list[str] | None = None,
        minimum_force_n: float = 0.0,
        include_all_contacts: bool = False,
        max_contacts: int = 64,
    ) -> dict[str, Any]:
        patterns = [
            str(pattern).strip()
            for pattern in (allowed_geom_patterns or [])
            if str(pattern).strip()
        ]
        if len(patterns) > 32:
            return {"error": "allowed_geom_patterns accepts at most 32 entries"}
        threshold = float(minimum_force_n)
        if threshold < 0:
            return {"error": "minimum_force_n must be non-negative"}
        try:
            raw = self._primitives.env.privileged_contacts(
                include_all_contacts=bool(include_all_contacts),
                max_contacts=int(max_contacts),
            )
        except Exception as exc:
            return {
                "available": False,
                "status": "unavailable",
                "error": f"{type(exc).__name__}: {exc}",
                "certificate": False,
            }
        if raw.get("available") is not True:
            return {**raw, "certificate": False}

        expected: list[dict[str, Any]] = []
        unexpected: list[dict[str, Any]] = []
        below_threshold: list[dict[str, Any]] = []
        for contact in raw.get("contacts", []):
            if not contact.get("involves_robot"):
                continue
            force = contact.get("normal_force_n")
            if force is not None and float(force) < threshold:
                below_threshold.append(contact)
                continue
            names = (str(contact.get("geom1", "")), str(contact.get("geom2", "")))
            allowed = any(
                fnmatch.fnmatchcase(name, pattern)
                for name in names
                for pattern in patterns
            )
            (expected if allowed else unexpected).append(contact)
        status = (
            "collision"
            if unexpected
            else "expected_contact"
            if expected
            else "clear"
        )
        return {
            "available": True,
            "status": status,
            "safe_under_allowlist": not unexpected,
            "hard_rejection": bool(unexpected),
            "certificate": False,
            "current_state_only": True,
            "allowed_geom_patterns": patterns,
            "minimum_force_n": threshold,
            "unexpected_contacts": unexpected,
            "expected_contacts": expected,
            "below_force_threshold": below_threshold,
            "raw_summary": {
                key: value
                for key, value in raw.items()
                if key != "contacts"
            },
        }

    def _assess_tool_applicability(
        self,
        tool: str,
        target_xyz: list[float] | None = None,
        acknowledge_irreversible: bool = False,
    ) -> dict[str, Any]:
        contract = self.get_tool_contract(str(tool))
        if contract is None:
            return {"applicable": False, "error": f"unknown tool: {tool}"}
        blockers: list[str] = []
        warnings: list[str] = []
        env = self._primitives.env
        if env.episode_terminated or env.episode_truncated:
            blockers.append("episode already terminated or truncated")
        if contract.irreversible and not acknowledge_irreversible:
            blockers.append("irreversible action not acknowledged")

        evidence: dict[str, Any] = {
            "current_eef_xyz": [
                round(float(v), 5) for v in self._primitives._last_obs_eef_pos
            ],
            "libero_terminated": env.episode_terminated,
            "episode_truncated": env.episode_truncated,
        }
        if target_xyz is not None:
            import numpy as np

            target = np.asarray(target_xyz, dtype=np.float32)
            if target.shape != (3,) or not np.isfinite(target).all():
                blockers.append("target_xyz must contain three finite numbers")
            else:
                current = self._primitives._last_obs_eef_pos
                distance = float(np.linalg.norm(target - current))
                evidence["target_xyz"] = [round(float(v), 5) for v in target]
                evidence["straight_line_distance_m"] = round(distance, 5)
                if not (-1.2 <= target[0] <= 1.2 and -1.2 <= target[1] <= 1.2 and -0.05 <= target[2] <= 1.5):
                    blockers.append("target is outside the broad LIBERO world sanity bounds")
                if distance > 0.30:
                    warnings.append(
                        "long direct move: split into re-observed waypoints; this is not an IK certificate"
                    )
        return {
            "applicable": not blockers,
            "tool": str(tool),
            "contract": contract.as_dict(),
            "blockers": blockers,
            "warnings": warnings,
            "evidence": evidence,
            "certificate": False,
        }

    def _step(self, name: str, **kwargs) -> dict:
        """Run ``self._primitives.<name>(**kwargs)``, dump the new step, and
        return the rendered state view + log.
        """
        command = {"action": name, **kwargs}
        t0 = time.time()
        start_frame = self._primitives.recorded_frame_count()
        result = getattr(self._primitives, name)(**kwargs)
        elapsed = round(time.time() - t0, 2)

        if isinstance(result, dict):
            result_dict = result
        else:
            result_dict = {"value": result}

        self._next_step += 1
        step_idx = self._next_step
        if self._dashboard is not None:
            video_dir = libero_tools.artifact_path(get_output_dir(), "action_videos")
            video_path = video_dir / f"step_{step_idx:02d}_{name}.mp4"
            try:
                self._primitives.save_frame_slice(start_frame, str(video_path), fps=20)
            except Exception as e:
                get_logger("libero_toolkit").warning(
                    f"failed to save action clip to {video_path}: {e}"
                )
        libero_tools.dump_state(
            self._primitives,
            str(get_output_dir()),
            step_idx=step_idx,
            log={"command": command, "result": result_dict, "elapsed_s": elapsed},
        )
        out = libero_tools.view_driver_state(step_idx)
        out["agent_elapsed_s"] = elapsed
        return out

    def init_primitives_clean(
        self,
        *,
        primitives_kwargs: dict[str, Any],
    ) -> None:
        """Wipe stale run artifacts, build the LiberoPrimitives, dump step 0."""
        out_dir = get_output_dir()
        out_dir.mkdir(parents=True, exist_ok=True)
        for sub in libero_tools.ARTIFACT_DIRECTORIES:
            target = out_dir / sub
            if target.exists():
                shutil.rmtree(target)
        for target in (
            libero_tools.artifact_path(out_dir, "states"),
            libero_tools.artifact_path(out_dir, "metadata", camera="agentview", resolution="low"),
            libero_tools.artifact_path(out_dir, "episode_video"),
            libero_tools.artifact_path(out_dir, "episode_wrist_video"),
            libero_tools.artifact_path(out_dir, "episode_multiview_video"),
        ):
            if target.exists():
                target.unlink()

        primitives = libero_tools.LiberoPrimitives(**primitives_kwargs)
        primitives.reset()
        primitives.start_recording()
        # Preserve the reset observation so the synchronized evidence videos
        # cover the full episode, including the pre-action scene.
        primitives.record_frame(primitives._last_obs)
        libero_tools.dump_state(primitives, str(out_dir), step_idx=0, log=None)
        if self._dashboard is not None:
            self._dashboard.on_tool_result("view_driver_state", libero_tools.view_driver_state(0))

        self._primitives = primitives

    def close(self) -> None:
        """Flush the agent-side video buffer to disk (end-of-run).
        """
        if self._video_path is None:
            return
        try:
            self._primitives.stop_recording_and_save(self._video_path)
        except Exception as e:
            # The runner is in the cleanup path; never let a video save
            # abort it.
            get_logger("libero_toolkit").warning(
                f"failed to save video to {self._video_path}: {e}"
            )

    def write_recipe(self, recipe_tag: str) -> str:
        """Write the LIBERO recipe JSONL from the dumped state trace."""
        return libero_tools.write_recipe_from_states(str(get_output_dir()), recipe_tag)
