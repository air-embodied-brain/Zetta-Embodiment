# Copyright (c) 2026 Zetta Contributors
from __future__ import annotations

import json
import urllib.error
from dataclasses import FrozenInstanceError
from typing import Any, Mapping

import pytest

from robots.robocasa.slide_dishwasher_program import (
    BASE_ASSIST_TOOL,
    CONTACT_PUSH_TOOL,
    GUARDED_SUFFIX_TOOL,
    PROGRAM_COMPONENTS,
    SlideDishwasherProgramState,
    base_assisted_terminal_action,
    guard_terminal_suffix,
)
from robots.robocasa.tool_bindings import TASK_BINDINGS, binding_for_task
from robots.robocasa.tool_catalog import (
    DEFAULT_ROBOCASA_TOOL_CATALOG,
    build_robocasa_tool_catalog,
)
from robots.robocasa.tool_runtime import (
    InvocationPolicy,
    ToolContractError,
    ToolPolicyError,
    ToolRuntime,
    ToolServiceRejected,
    ToolServiceUnavailable,
)

EXPECTED_TOOLS = {
    "robocasa.observation.view_driver_state",
    "robocasa.camera.view_meta",
    "robocasa.geometry.back_project",
    "robocasa.perception.grounded_sam2",
    "robocasa.perception.depth_anything_v2",
    "robocasa.grasp.contact_graspnet",
    "robocasa.grasp.graspgen",
    "robocasa.control.move_to",
    "robocasa.base.move",
    "robocasa.control.move_pose",
    "robocasa.control.rotate_wrist",
    "robocasa.control.rotate_pitch",
    "robocasa.gripper.set",
    "robocasa.gripper.release",
    "robocasa.verify.state",
    "robocasa.vla.groot",
    "robocasa.motion.mink_reach",
    "robocasa.motion.curobo_reachability",
    "robocasa.motion.curobo_motiongen_pregrasp",
    "robocasa.cap.servo_pose",
    "robocasa.cap.contact_retaining_motion",
    "robocasa.critic.temporal_engagement",
    "robocasa.motion.base_se2_astar",
    "robocasa.motion.base_se2_servo",
    CONTACT_PUSH_TOOL,
    GUARDED_SUFFIX_TOOL,
    BASE_ASSIST_TOOL,
}


def _all_numbers(value: Any) -> list[float]:
    if isinstance(value, Mapping):
        return [number for item in value.values() for number in _all_numbers(item)]
    if isinstance(value, (list, tuple)):
        return [number for item in value for number in _all_numbers(item)]
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return [float(value)]
    return []


def _identity_pose(position: list[float]) -> dict[str, Any]:
    return {"position": position, "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0]}


def test_catalog_contains_complete_surface_and_is_immutable() -> None:
    catalog = build_robocasa_tool_catalog()
    assert set(catalog.names()) == EXPECTED_TOOLS
    assert catalog.get("robocasa.vla.groot").service_path == "/act"
    assert len(catalog.names()) == 27
    assert catalog.digest == build_robocasa_tool_catalog().digest
    assert len(catalog.digest) == 64
    with pytest.raises(FrozenInstanceError):
        catalog.get("robocasa.base.move").name = "changed"  # type: ignore[misc]


def test_task_bindings_are_audited_subsets_and_unknown_tasks_fail_closed() -> None:
    catalog_names = set(DEFAULT_ROBOCASA_TOOL_CATALOG.names())
    assert set(TASK_BINDINGS) == {"SlideDishwasherRack", "OpenDrawer", "TurnOffStove"}
    for task, binding in TASK_BINDINGS.items():
        assert binding.task == task
        assert set(binding.tool_names) < catalog_names
        assert {item.name for item in binding.select()} == set(binding.tool_names)
        assert len(binding.digest) == 64
    assert (
        "robocasa.grasp.graspgen"
        not in binding_for_task("SlideDishwasherRack").tool_names
    )
    assert "robocasa.grasp.graspgen" in binding_for_task("OpenDrawer").tool_names
    slide = binding_for_task("SlideDishwasherRack")
    assert slide.vla_tool_name == CONTACT_PUSH_TOOL
    assert slide.program_components == PROGRAM_COMPONENTS
    assert set(slide.tool_names) == {
        CONTACT_PUSH_TOOL,
        GUARDED_SUFFIX_TOOL,
        BASE_ASSIST_TOOL,
    }
    with pytest.raises(KeyError, match="no audited"):
        binding_for_task("UnknownTask")


def test_catalog_allow_deny_and_unknown_names_fail_closed() -> None:
    catalog = DEFAULT_ROBOCASA_TOOL_CATALOG
    selected = catalog.select(
        allow={"robocasa.control.move_to", "robocasa.verify.state"},
        deny={"robocasa.verify.state"},
    )
    assert [spec.name for spec in selected] == ["robocasa.control.move_to"]
    assert {
        spec.name
        for spec in catalog.select(capabilities={"motion.collision_aware_global"})
    } == {
        "robocasa.motion.base_se2_astar",
        "robocasa.motion.curobo_motiongen_pregrasp",
    }
    with pytest.raises(KeyError, match="unknown"):
        catalog.get("robocasa.missing")
    with pytest.raises(KeyError, match="unknown"):
        catalog.select(deny={"robocasa.missing"})


def test_every_control_surface_is_proposal_only_and_runtime_cannot_step() -> None:
    proposal_tools = {
        spec.name
        for spec in DEFAULT_ROBOCASA_TOOL_CATALOG.select()
        if spec.proposal_only
    }
    assert {
        "robocasa.control.move_to",
        "robocasa.base.move",
        "robocasa.control.move_pose",
        "robocasa.control.rotate_wrist",
        "robocasa.control.rotate_pitch",
        "robocasa.gripper.set",
        "robocasa.gripper.release",
        "robocasa.vla.groot",
        "robocasa.motion.mink_reach",
        "robocasa.cap.servo_pose",
        "robocasa.cap.contact_retaining_motion",
        "robocasa.critic.temporal_engagement",
        "robocasa.motion.base_se2_astar",
        "robocasa.motion.base_se2_servo",
    }.issubset(proposal_tools)
    runtime = ToolRuntime(environ={})
    assert not hasattr(runtime, "step")
    result = runtime.invoke(
        "robocasa.control.move_to",
        {"current_position": [0, 0, 0], "target_position": [10, -10, 5]},
    )
    assert result["proposal_only"] is True
    assert result["environment_write"] is False


def test_privileged_input_requires_opt_in_and_records_only_safe_audit() -> None:
    seen: list[str] = []

    def transport(
        url: str, body: bytes, timeout: float, headers: Mapping[str, str]
    ) -> Mapping[str, Any]:
        del body, timeout, headers
        seen.append(url)
        return {"candidates": [], "selected_candidate": None}

    secret_url = "http://127.0.0.1:9911/private"
    runtime = ToolRuntime(
        environ={"ZETTA_ROBOCASA_GRASPGEN_URL": secret_url},
        http_transport=transport,
    )
    payload = {
        "observation": {
            "privileged": {
                "drawer_handle_pose": [0.1, 0.2, 0.3],
                "ground_truth_marker": "must-not-be-copied-to-audit",
            }
        }
    }
    with pytest.raises(ToolPolicyError, match="privileged"):
        runtime.invoke("robocasa.grasp.graspgen", payload)
    result = runtime.invoke(
        "robocasa.grasp.graspgen",
        payload,
        policy=InvocationPolicy(allow_privileged=True),
    )
    audit = result["privileged_audit"]
    assert audit["authorized"] is True
    assert audit["used"] is True
    assert audit["fields"] == ["observation.privileged"]
    encoded = json.dumps(result, sort_keys=True)
    assert "ground_truth_marker" not in encoded
    assert secret_url not in encoded
    assert seen == [secret_url + "/infer"]


def test_flattened_live_privileged_state_is_recorded_in_audit() -> None:
    runtime = ToolRuntime(
        environ={"ZETTA_ROBOCASA_GRASPGEN_URL": "http://127.0.0.1:9911"},
        http_transport=lambda *_args, **_kwargs: {"candidates": []},
    )
    result = runtime.invoke(
        "robocasa.grasp.graspgen",
        {
            "observation": {
                "state": {
                    "privileged.source": "live_mujoco_simulator",
                    "privileged.dishwasher.rack.position": 0.25,
                }
            }
        },
        policy=InvocationPolicy(allow_privileged=True),
    )
    audit = result["privileged_audit"]
    assert audit["authorized"] is True
    assert audit["used"] is True
    assert audit["fields"] == [
        "observation.state.privileged.dishwasher.rack.position",
        "observation.state.privileged.source",
    ]


def _slide_state(
    *, residual: float, contact: bool = True, collision: bool = False
) -> dict[str, Any]:
    return {
        "privileged.dishwasher.rack.residual_to_success": residual,
        "privileged.dishwasher.rack.remaining_to_success_m": residual * 0.1,
        "privileged.dishwasher.rack.success_direction_base": [1.0, 0.0, 0.0],
        "privileged.dishwasher.rack.target_contact": contact,
        "privileged.dishwasher.collision.detected": collision,
    }


def test_slide_task_program_reviews_reverse_action_before_execution() -> None:
    program = SlideDishwasherProgramState()
    program.reset(_slide_state(residual=0.8))
    program.after_action(
        state=_slide_state(residual=0.7), step_index=1, at_chunk_boundary=False
    )
    program.after_action(
        state=_slide_state(residual=0.7), step_index=2, at_chunk_boundary=False
    )
    proposals = program.before_action(
        {"end_effector_position": [-0.2, 0.1, 0.0]},
        state=_slide_state(residual=0.7),
        step_index=3,
    )
    assert [item["rule_id"] for item in proposals] == [
        "slide_dishwasher.premature_disengagement"
    ]
    assert proposals[0]["environment_write"] is False


def test_slide_collision_telemetry_is_diagnostic_only() -> None:
    program = SlideDishwasherProgramState()
    program.reset(_slide_state(residual=0.8, contact=False))
    proposals = program.before_action(
        {"end_effector_position": [0.2, 0.0, 0.0]},
        state=_slide_state(residual=0.8, contact=False, collision=True),
        step_index=1,
    )
    assert proposals == []


def test_slide_terminal_tools_preserve_suffix_fields_and_propose_only() -> None:
    state = _slide_state(residual=0.2)
    original = {
        "end_effector_position": [-0.4, 0.3, 0.2],
        "end_effector_rotation": [0.1, -0.2, 0.3],
        "gripper_close": [1.0],
        "base_motion": [0.1, 0.2, 0.3, 0.4],
    }
    guarded = guard_terminal_suffix([original], state=state)[0]
    assert guarded["action.end_effector_position"][0] == pytest.approx(0.05)
    assert guarded["action.end_effector_position"][1:] == pytest.approx([0.3, 0.2])
    assert guarded["action.end_effector_rotation"] == pytest.approx([0.1, -0.2, 0.3])
    assert guarded["action.gripper_close"] == [1.0]
    assert guarded["action.base_motion"] == pytest.approx([0.1, 0.2, 0.3, 0.4])
    assisted = base_assisted_terminal_action(state=state)
    assert assisted["action.base_motion"][0] > 0
    assert assisted["action.end_effector_position"][0] < 0


@pytest.mark.parametrize(
    ("name", "payload"),
    [
        (
            "robocasa.control.move_to",
            {"current_position": [0, 0, 0], "target_position": [10, -5, 2]},
        ),
        (
            "robocasa.base.move",
            {"base_motion": [4, -3, 2, -2], "gripper_close": 4},
        ),
        (
            "robocasa.control.move_pose",
            {
                "current_pose": _identity_pose([0, 0, 0]),
                "target_pose": _identity_pose([5, -5, 3]),
            },
        ),
        ("robocasa.control.rotate_wrist", {"delta_yaw_rad": 9}),
        ("robocasa.control.rotate_pitch", {"delta_pitch_rad": -9}),
        ("robocasa.gripper.set", {"gripper_close": 9}),
        ("robocasa.gripper.release", {}),
        (
            "robocasa.cap.servo_pose",
            {
                "current_position": [0, 0, 0],
                "target_position": [5, -5, 5],
                "current_quaternion": [0, 0, 0, 1],
                "target_quaternion": [0, 0, 1, 0],
                "maximum_position_command": 0.1,
                "maximum_rotation_command": 0.1,
                "close": True,
            },
        ),
        (
            "robocasa.cap.contact_retaining_motion",
            {
                "current_position": [0, 0, 0],
                "live_anchor_position": [0, 0, 0],
                "motion_direction": [1, 0, 0],
                "current_quaternion": [0, 0, 0, 1],
                "target_quaternion": [0, 0, 0, 1],
                "maximum_position_command": 0.1,
                "maximum_rotation_command": 0.1,
                "close": True,
            },
        ),
        (
            "robocasa.motion.base_se2_servo",
            {
                "handle_xy": [4, -3],
                "goal": {"target_handle_xy_m": [0.62, 0], "target_relative_yaw_rad": 0},
            },
        ),
    ],
)
def test_local_action_proposals_are_finite_and_bounded(
    name: str, payload: Mapping[str, Any]
) -> None:
    result = ToolRuntime(environ={}).invoke(name, payload)
    numbers = _all_numbers(result["action"])
    assert numbers
    assert all(-1.0 <= number <= 1.0 for number in numbers)
    assert result["environment_write"] is False


def test_back_project_geometry_and_privileged_astar_proposal() -> None:
    runtime = ToolRuntime(environ={})
    cloud = runtime.invoke(
        "robocasa.geometry.back_project",
        {
            "depth": [[1.0, 1.0], [1.0, 1.0]],
            "mask": [[True, False], [False, False]],
            "intrinsics": {"K": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]},
            "extrinsic_cam2world": [
                [1, 0, 0, 1],
                [0, 1, 0, 2],
                [0, 0, 1, 3],
                [0, 0, 0, 1],
            ],
        },
    )
    assert cloud["count"] == 1
    assert cloud["frame"] == "world"
    assert cloud["points"] == [[1.0, 2.0, 4.0]]
    with pytest.raises(ToolPolicyError):
        runtime.invoke(
            "robocasa.motion.base_se2_astar",
            {"start_world": [0, 0], "goal": [0.3, 0], "obstacles": []},
        )
    plan = runtime.invoke(
        "robocasa.motion.base_se2_astar",
        {"start_world": [0, 0], "goal": [0.3, 0], "obstacles": []},
        policy=InvocationPolicy(allow_privileged=True),
    )
    assert plan["status"] == "planned"
    assert plan["waypoints_world"]
    assert plan["environment_write"] is False


@pytest.mark.parametrize(
    ("state", "criteria", "expected"),
    [
        ({"success": True}, {"success": True}, "satisfied"),
        ({"progress": 0.1}, {"progress": {"min": 0.5}}, "unsatisfied"),
        ({}, {"contact": True}, "unknown"),
    ],
)
def test_verify_is_strictly_tri_state(
    state: Mapping[str, Any], criteria: Mapping[str, Any], expected: str
) -> None:
    result = ToolRuntime(environ={}).invoke(
        "robocasa.verify.state", {"state": state, "criteria": criteria}
    )
    assert result["status"] == expected
    assert result["status"] in {"satisfied", "unsatisfied", "unknown"}


def test_temporal_critic_is_proposal_only_and_does_not_claim_execution() -> None:
    result = ToolRuntime(environ={}).invoke(
        "robocasa.critic.temporal_engagement",
        {
            "history": [
                {"progress": 0.2, "contact": True},
                {"progress": 0.2, "contact": False},
            ]
        },
    )
    assert result["triggered"] is True
    assert result["status"] == "proposal"
    assert result["proposal_only"] is True
    assert result["environment_write"] is False


def test_service_failures_are_classified_without_leaking_endpoint_or_error() -> None:
    endpoint = "https://private.invalid/tool?api_key=do-not-leak"

    def unavailable(
        url: str, body: bytes, timeout: float, headers: Mapping[str, str]
    ) -> Mapping[str, Any]:
        del body, timeout, headers
        raise urllib.error.URLError(f"secret upstream at {url}")

    runtime = ToolRuntime(
        environ={"ZETTA_ROBOCASA_GROOT_URL": endpoint},
        http_transport=unavailable,
    )
    with pytest.raises(ToolServiceUnavailable) as caught:
        runtime.invoke("robocasa.vla.groot", {"observation": {}, "request": {}})
    safe = json.dumps(caught.value.safe_dict(), sort_keys=True)
    assert caught.value.retryable is True
    assert endpoint not in safe
    assert "do-not-leak" not in safe
    assert "private.invalid" not in str(caught.value)

    def rejected(
        url: str, body: bytes, timeout: float, headers: Mapping[str, str]
    ) -> Mapping[str, Any]:
        del body, timeout, headers
        raise urllib.error.HTTPError(url, 400, "token=secret", {}, None)

    runtime = ToolRuntime(
        environ={"ZETTA_ROBOCASA_GROOT_URL": endpoint},
        http_transport=rejected,
    )
    with pytest.raises(ToolServiceRejected) as rejected_error:
        runtime.invoke("robocasa.vla.groot", {"observation": {}, "request": {}})
    assert rejected_error.value.retryable is False
    assert endpoint not in str(rejected_error.value)


def test_service_action_contract_rejects_out_of_bounds_model_output() -> None:
    def transport(
        url: str, body: bytes, timeout: float, headers: Mapping[str, str]
    ) -> Mapping[str, Any]:
        del url, body, timeout, headers
        return {"action": [0.0, 1.01]}

    runtime = ToolRuntime(
        environ={"ZETTA_ROBOCASA_GROOT_URL": "http://127.0.0.1:9000"},
        http_transport=transport,
    )
    with pytest.raises(ToolContractError, match="out-of-bounds"):
        runtime.invoke("robocasa.vla.groot", {"observation": {}, "request": {}})


def test_catalog_and_results_never_contain_configured_secrets_or_service_urls() -> None:
    secret = "sk-test-extremely-secret"
    endpoint = "https://service.example.invalid/private"

    def transport(
        url: str, body: bytes, timeout: float, headers: Mapping[str, str]
    ) -> Mapping[str, Any]:
        del url, body, timeout, headers
        return {
            "found": True,
            "annotations": [],
            "service_url": endpoint,
            "api_key": secret,
            "note": f"model used {secret}",
        }

    runtime = ToolRuntime(
        environ={
            "ZETTA_ROBOCASA_GROUNDED_SAM2_URL": endpoint,
            "GROUNDING_API_KEY": secret,
        },
        http_transport=transport,
    )
    result = runtime.invoke(
        "robocasa.perception.grounded_sam2",
        {"image": "digest-only", "prompt": "dishwasher rack"},
    )
    serialized = json.dumps(
        {
            "catalog": runtime.catalog.public_dict(),
            "result": result,
        },
        sort_keys=True,
    )
    assert secret not in serialized
    assert endpoint not in serialized
    assert "service_url" not in result
    assert "api_key" not in result
    assert "[redacted]" in serialized
