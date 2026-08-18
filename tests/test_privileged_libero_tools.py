# Copyright (c) 2026 RPent Contributors
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import numpy as np
import pytest

from robots.libero import contact_graspnet, graspgen
from robots.libero.contact_graspnet import (
    ContactGraspNetAdapter,
    camera_point_cloud,
    transform_grasp_candidates,
)
from robots.libero.graspgen import (
    GraspGenAdapter,
    prepare_sensor_object_cloud,
    transform_graspgen_candidates,
)
from robots.libero.privileged_sensors import (
    collect_privileged_contacts,
    install_libero_contact_extension,
)
from robots.libero.role1_recovery import LiberoRole1RecoveryActor
from robots.libero.tool_catalog import LIBERO_RECOVERY_TOOL_NAMES
from robots.libero.toolkit import LiberoToolkit
from robots.libero.tools import (
    TOOLS_SPEC,
    RECOVERY_MOTION_TOOL_NAMES,
    LiberoPrimitives,
)


class _FakeModel:
    names = ["robot0_gripper_collision", "target_bowl", "table"]
    ngeom = len(names)

    def geom_id2name(self, index: int) -> str:
        return self.names[index]


class _FakeContact:
    def __init__(self, geom1: int, geom2: int, distance: float = -0.001):
        self.geom1 = geom1
        self.geom2 = geom2
        self.dist = distance
        self.pos = np.array([0.1, 0.2, 0.3])
        self.frame = np.eye(3).reshape(-1)


def _fake_outer_env():
    data = SimpleNamespace(
        ncon=2,
        contact=[_FakeContact(0, 1), _FakeContact(1, 2)],
    )
    raw = SimpleNamespace(sim=SimpleNamespace(model=_FakeModel(), data=data), robots=[])
    return SimpleNamespace(env=raw)


def test_privileged_pick_place_schema_exposes_grasp_offset_and_defaults():
    spec = next(row for row in TOOLS_SPEC if row["name"] == "privileged_pick_place")
    properties = spec["input_schema"]["properties"]

    assert properties["grasp_offset_xyz"] == {
        "type": "array",
        "items": {"type": "number"},
        "minItems": 3,
        "maxItems": 3,
        "default": [0.0, -0.036, 0.038],
        "description": (
            "Object-relative grasp offset [x, y, z] in meters "
            "(default [0.0, -0.036, 0.038])."
        ),
    }
    assert properties["grasp_confirm_steps"]["default"] == 4


def test_recovery_motion_tools_are_additive_and_have_specs():
    spec_names = {row["name"] for row in TOOLS_SPEC}
    assert set(RECOVERY_MOTION_TOOL_NAMES).issubset(spec_names)
    assert set(RECOVERY_MOTION_TOOL_NAMES).issubset(set(LIBERO_RECOVERY_TOOL_NAMES))
    assert {"move_to", "move_pose", "semantic_joint_interact"}.issubset(
        set(LIBERO_RECOVERY_TOOL_NAMES)
    )


def test_motion_proposal_tools_do_not_step_or_require_graspgen_service():
    class _Env:
        episode_steps = 0
        episode_terminated = False
        episode_truncated = False

    primitives = object.__new__(LiberoPrimitives)
    primitives.env = _Env()
    primitives._last_obs_eef_pos = np.array([0.0, 0.0, 0.4], dtype=np.float32)
    primitives._motion_candidate = None
    primitives._motion_history = []
    primitives._graspgen = GraspGenAdapter(endpoint=None)

    health = primitives.graspgen(mode="health")
    freshness = primitives.candidate_freshness()
    reach = primitives.curobo_reachability(target_xyz=[0.1, 0.0, 0.4])
    liveness = primitives.progress_liveness()

    assert health["no_op_verified"] is True
    assert freshness["fresh"] is False
    assert reach["reachable"] is True
    assert liveness["environment_advanced"] is False
    assert primitives.env.episode_steps == 0


@pytest.mark.parametrize(
    "tool_name,result",
    [
        ("graspgen", {"no_op_verified": True, "environment_advanced": False}),
        ("candidate_freshness", {"no_op_verified": True, "environment_advanced": False}),
        ("curobo_reachability", {"no_op_verified": True, "environment_advanced": False}),
        ("progress_liveness", {"no_op_verified": True, "environment_advanced": False}),
    ],
)
def test_role1_actor_accepts_verified_proposal_only_recovery(tmp_path, tool_name, result):
    class _Env:
        episode_steps = 0
        episode_terminated = False
        episode_truncated = False

    class _Primitives:
        env = _Env()

        def begin_recovery_step(self):
            return None

        def __getattr__(self, name):
            if name == tool_name:
                return lambda **_kwargs: dict(result)
            raise AttributeError(name)

    class _Adapter:
        def decide(self, event, *, image_payloads=None):
            return SimpleNamespace(
                decision_id="decision-proposal-only",
                selected_tool=event.current_tool,
                proposal_disposition="accept",
                modifications={},
                termination=SimpleNamespace(approved=False),
                direct_action=None,
            )

    actor = LiberoRole1RecoveryActor(
        adapter=_Adapter(),
        audit_root=tmp_path / "actor",
        allowed_tools=(tool_name,),
        allow_privileged_evidence=False,
    )
    outcome = actor.decide_and_execute(
        task="test",
        step_index=0,
        observation={},
        critic_values=({"rule_id": "r1", "proposal": "inspect"},),
        recovery_context={
            "recovery_id": "recovery-1",
            "current_step": {"tool": tool_name, "parameters": {}},
        },
        primitives=_Primitives(),
    )
    assert outcome.selected_tool == tool_name
    assert outcome.executed_horizon == 0
    audit = next((tmp_path / "actor").glob("*.json"))
    assert json.loads(audit.read_text())["no_op_verified"] is True


def _motion_stub(*, terminated=False, truncated=False):
    class _Env:
        episode_steps = 0
        episode_terminated = terminated
        episode_truncated = truncated

    primitives = object.__new__(LiberoPrimitives)
    primitives.env = _Env()
    primitives._last_obs_eef_pos = np.asarray([0.0, 0.0, 0.4], dtype=np.float32)
    primitives._motion_candidate = None
    primitives._motion_history = []
    return primitives


def test_mutating_motion_tools_fail_closed_after_terminal_state():
    primitives = _motion_stub(terminated=True)

    def fail_if_called(**_kwargs):
        raise AssertionError("terminal motion must not call gripper or OSC")

    primitives.set_gripper = fail_if_called
    primitives.move_pose = fail_if_called
    result = primitives.mink_engage_close(target_xyz=[0.1, 0.0, 0.4])
    assert result["status"] == "terminal_noop"
    assert result["no_op_verified"] is True


def test_motion_tools_reject_stale_cached_candidate():
    primitives = _motion_stub()
    primitives.env.episode_steps = 20
    primitives._motion_candidate = {
        "created_step": 0,
        "eef_xyz": [0.0, 0.0, 0.4],
        "candidates": [{"transform_world": np.eye(4).tolist()}],
    }
    primitives.move_pose = lambda **_kwargs: (_ for _ in ()).throw(
        AssertionError("stale candidate must not execute")
    )
    result = primitives.mink_reach()
    assert result["status"] == "candidate_stale"
    assert result["candidate_rejected"] is True


def test_mink_reach_executes_explicit_json_target(monkeypatch):
    primitives = _motion_stub()

    def move_pose(target, **_kwargs):
        primitives.env.episode_steps += 1
        primitives._last_obs_eef_pos = np.asarray(target, dtype=np.float32)
        return {"steps_used": 1}

    monkeypatch.setattr(primitives, "move_pose", move_pose)
    result = primitives.mink_reach(target_xyz=[0.01, 0.0, 0.4], max_steps=3)
    assert result["status"] == "executed"
    assert result["environment_advanced"] is True
    assert result["reachability"]["certificate_level"] == "workspace_only"


def test_mink_reach_reports_already_at_target_as_verified_noop(monkeypatch):
    primitives = _motion_stub()
    monkeypatch.setattr(
        primitives,
        "move_pose",
        lambda _target, **_kwargs: {"steps_used": 0, "final_dist_m": 0.0},
    )
    result = primitives.mink_reach(target_xyz=[0.0, 0.0, 0.4], max_steps=3)
    assert result["status"] == "already_at_target"
    assert result["no_op_verified"] is True
    assert result["environment_advanced"] is False


def test_candidate_freshness_rejects_changed_scene_observation():
    primitives = _motion_stub()
    primitives._last_obs = {
        "main_images": np.zeros((4, 4, 3), dtype=np.uint8),
        "wrist_images": np.zeros((4, 4, 3), dtype=np.uint8),
    }
    primitives._motion_candidate = {
        "candidate_id": "candidate-1",
        "created_step": 0,
        "eef_xyz": [0.0, 0.0, 0.4],
        "scene_sha256": primitives._motion_scene_sha256(),
        "candidates": [{"transform_world": np.eye(4).tolist()}],
    }
    primitives._last_obs["main_images"][0, 0, 0] = 255
    result = primitives.candidate_freshness()
    assert result["fresh"] is False
    assert result["scene_changed"] is True
    assert result["invalidated"] is True


def test_mink_engage_close_uses_candidate_approach_axis(monkeypatch):
    primitives = _motion_stub()
    rotation = np.asarray(
        [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]],
        dtype=np.float32,
    )
    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = rotation
    pose[:3, 3] = [0.1, 0.0, 0.4]
    primitives._motion_candidate = {
        "created_step": 0,
        "eef_xyz": [0.0, 0.0, 0.4],
        "candidates": [{"transform_world": pose.tolist()}],
    }
    calls = []

    def set_gripper(*, gripper, steps):
        calls.append(("gripper", gripper, steps))
        primitives.env.episode_steps += steps
        return {"steps": steps}

    def move_pose(target, **_kwargs):
        calls.append(("pose", target))
        primitives.env.episode_steps += 1
        return {"steps_used": 1}

    monkeypatch.setattr(primitives, "set_gripper", set_gripper)
    monkeypatch.setattr(primitives, "move_pose", move_pose)
    result = primitives.mink_engage_close(micro_advance_m=0.01)
    assert result["status"] == "executed"
    assert result["approach_axis_world"] == pytest.approx([1.0, 0.0, 0.0])
    assert calls[1][1][0] == pytest.approx(0.11)
    assert calls[1][1][2] == pytest.approx(0.4)


def test_progress_liveness_includes_semantic_goal_progress():
    primitives = _motion_stub()
    progress_values = iter([0.2, 0.2])

    def critic_state():
        return {"privileged.task.goal.progress": next(progress_values)}

    primitives.env.privileged_critic_state = critic_state
    primitives._motion_record("mink_pull", 0, {"status": "executed"})
    primitives.env.episode_steps = 1
    primitives._last_obs_eef_pos = np.asarray([0.0001, 0.0, 0.4], dtype=np.float32)
    primitives._motion_record("mink_pull", 1, {"status": "executed"})
    result = primitives.progress_liveness()
    assert result["task_progress_delta"] == pytest.approx(0.0)
    assert result["stagnant"] is True


def test_toolkit_motion_dispatch_preserves_state_and_video_dump_path():
    toolkit = object.__new__(LiberoToolkit)
    toolkit._primitives = SimpleNamespace(
        mink_pull=lambda **_kwargs: {"unexpected": True},
        curobo_reachability=lambda **_kwargs: {
            "no_op_verified": True,
            "environment_advanced": False,
        },
    )
    calls = []

    def step(name, **kwargs):
        calls.append((name, kwargs))
        return {"dumped": True}

    toolkit._step = step
    assert toolkit._motion_tool("mink_pull", delta_xyz=[0.0, 0.01, 0.0]) == {
        "dumped": True
    }
    assert calls == [("mink_pull", {"delta_xyz": [0.0, 0.01, 0.0]})]
    proposal = toolkit._motion_tool("curobo_reachability", target_xyz=[0, 0, 0])
    assert proposal["no_op_verified"] is True
    assert len(calls) == 1


def test_privileged_contact_sensor_filters_scene_only_contacts():
    result = collect_privileged_contacts(_fake_outer_env())
    assert result["available"] is True
    assert result["total_contact_count"] == 2
    assert result["robot_contact_count"] == 1
    assert result["returned_contact_count"] == 1
    assert result["contacts"][0]["geom2"] == "target_bowl"
    assert result["trajectory_collision_certificate"] is False


def test_collision_tool_separates_expected_and_unexpected_contacts():
    raw = collect_privileged_contacts(_fake_outer_env())

    class _Env:
        def privileged_contacts(self, **_kwargs):
            return raw

    toolkit = object.__new__(LiberoToolkit)
    toolkit._primitives = SimpleNamespace(env=_Env())
    expected = toolkit._collision_check(allowed_geom_patterns=["*bowl*"])
    assert expected["status"] == "expected_contact"
    assert expected["safe_under_allowlist"] is True
    rejected = toolkit._collision_check()
    assert rejected["status"] == "collision"
    assert rejected["hard_rejection"] is True
    assert rejected["certificate"] is False


def test_contact_extension_attaches_bound_sensor_to_subprocess_factory():
    class _LiberoEnv:
        def get_env_fns(self):
            return [lambda: _fake_outer_env()]

    install_libero_contact_extension(_LiberoEnv)
    instance = _LiberoEnv()
    env = instance.get_env_fns()[0]()
    result = env.rpent_privileged_contacts()
    assert result["available"] is True
    # Installation is idempotent.
    install_libero_contact_extension(_LiberoEnv)
    assert len(instance.get_env_fns()) == 1


def test_camera_point_cloud_and_world_transform():
    depth = np.ones((8, 8), dtype=np.float32)
    intrinsic = np.array([[4.0, 0.0, 3.5], [0.0, 4.0, 3.5], [0.0, 0.0, 1.0]])
    points = camera_point_cloud(
        depth,
        intrinsic,
        min_depth_m=0.1,
        max_depth_m=2.0,
        max_points=64,
    )
    assert points.shape == (64, 3)
    assert np.allclose(points[:, 2], 1.0)

    camera_pose = np.eye(4)
    camera_pose[:3, 3] = [0.1, 0.2, 0.3]
    extrinsic = np.eye(4)
    extrinsic[:3, 3] = [1.0, 2.0, 3.0]
    result = transform_grasp_candidates(
        [
            {
                "transform_camera": camera_pose.tolist(),
                "contact_point_camera": [0.0, 0.0, 1.0],
                "score": 0.9,
            }
        ],
        extrinsic,
    )
    assert np.allclose(result[0]["position_world"], [1.1, 2.2, 3.3])
    assert np.allclose(result[0]["contact_point_world"], [1.0, 2.0, 4.0])


def test_contact_graspnet_adapter_calls_service_and_never_steps(tmp_path, monkeypatch):
    class _Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            return

        def do_GET(self):
            body = json.dumps({"ok": True, "ready": True}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            length = int(self.headers["Content-Length"])
            payload = json.loads(self.rfile.read(length))
            assert len(payload["point_cloud"]) == 64
            body = json.dumps(
                {
                    "status": "ok",
                    "grasps": [
                        {
                            "transform_camera": np.eye(4).tolist(),
                            "contact_point_camera": [0.0, 0.0, 1.0],
                            "score": 0.8,
                            "gripper_opening_m": 0.04,
                        }
                    ],
                    "evidence": {"test": True},
                }
            ).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        metadata = {
            "intrinsic_K": [[4.0, 0.0, 3.5], [0.0, 4.0, 3.5], [0.0, 0.0, 1.0]],
            "extrinsic_cam2world": np.eye(4).tolist(),
        }
        monkeypatch.setattr(contact_graspnet, "get_output_dir", lambda: tmp_path)
        from robots.libero import tools as libero_tools

        monkeypatch.setattr(libero_tools, "_latest_step", lambda: 0)
        monkeypatch.setattr(
            libero_tools, "_load_depth", lambda _camera, _step: np.ones((8, 8))
        )
        monkeypatch.setattr(
            libero_tools, "_load_camera_meta", lambda _camera, _step: metadata
        )
        adapter = ContactGraspNetAdapter(
            f"http://127.0.0.1:{server.server_address[1]}"
        )
        health = adapter.propose(mode="health", timeout_s=2)
        assert health["available"] is True
        result = adapter.propose(max_points=64, timeout_s=2)
        assert result["status"] == "ok"
        assert result["proposal_only"] is True
        assert result["environment_advanced"] is False
        assert result["candidate_count"] == 1
        assert (tmp_path / "contact_graspnet").is_dir()
    finally:
        server.shutdown()
        server.server_close()


def test_graspgen_sensor_cloud_is_centered_and_world_transform_is_restored():
    depth = np.ones((8, 8), dtype=np.float32)
    intrinsic = np.array([[4.0, 0.0, 3.5], [0.0, 4.0, 3.5], [0.0, 0.0, 1.0]])
    model_points, centroid, targeting = prepare_sensor_object_cloud(
        depth,
        intrinsic,
        np.eye(4),
        target_world_xyz=[0.0, 0.0, 1.0],
        crop_radius_m=1.0,
        min_depth_m=0.1,
        max_depth_m=2.0,
        max_points=32,
    )
    assert model_points.shape == (32, 3)
    assert np.allclose(model_points.mean(axis=0), 0.0, atol=1e-6)
    assert targeting["mode"] == "planner_sensor_roi"

    pose = np.eye(4)
    pose[:3, 3] = [0.1, 0.0, 0.0]
    extrinsic = np.eye(4)
    extrinsic[:3, 3] = [1.0, 2.0, 3.0]
    result = transform_graspgen_candidates(
        [{"transform_model": pose.tolist(), "score": 0.9}],
        centroid,
        extrinsic,
    )
    expected = extrinsic[:3, 3] + centroid + np.array([0.1, 0.0, 0.0])
    assert np.allclose(result[0]["position_world"], expected)


def test_graspgen_adapter_uses_sensor_cloud_and_never_steps(tmp_path, monkeypatch):
    class _Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            return

        def do_GET(self):
            body = json.dumps({"ok": True, "ready": True}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            assert self.path == "/generate"
            length = int(self.headers["Content-Length"])
            payload = json.loads(self.rfile.read(length))
            assert payload["frame"] == "object_centered_opencv_camera_xyz_m"
            assert payload["gripper"] == "franka_panda"
            assert len(payload["point_cloud"]) == 64
            body = json.dumps(
                {
                    "status": "ok",
                    "grasps": [
                        {
                            "transform_model": np.eye(4).tolist(),
                            "score": 0.85,
                        }
                    ],
                    "evidence": {"test": True},
                }
            ).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        metadata = {
            "intrinsic_K": [[4.0, 0.0, 3.5], [0.0, 4.0, 3.5], [0.0, 0.0, 1.0]],
            "extrinsic_cam2world": np.eye(4).tolist(),
        }
        monkeypatch.setattr(graspgen, "get_output_dir", lambda: tmp_path)
        from robots.libero import tools as libero_tools

        monkeypatch.setattr(libero_tools, "_latest_step", lambda: 0)
        monkeypatch.setattr(
            libero_tools, "_load_depth", lambda _camera, _step: np.ones((8, 8))
        )
        monkeypatch.setattr(
            libero_tools, "_load_camera_meta", lambda _camera, _step: metadata
        )
        adapter = GraspGenAdapter(f"http://127.0.0.1:{server.server_address[1]}")
        health = adapter.propose(mode="health", timeout_s=2)
        assert health["available"] is True
        result = adapter.propose(max_points=64, crop_radius_m=1.0, timeout_s=2)
        assert result["status"] == "ok"
        assert result["proposal_only"] is True
        assert result["environment_advanced"] is False
        assert result["privileged_geometry_used"] is False
        assert result["collision_filter_requested"] is True
        assert result["collision_filter_applied"] is None
        assert result["candidate_count"] == 1
        assert (tmp_path / "graspgen").is_dir()
    finally:
        server.shutdown()
        server.server_close()


def _semantic_sidecar(
    *, grasped: bool = False, grasp_on_close: bool = False, success: bool = False
) -> dict:
    return {
        "privileged.available": True,
        "privileged.task.manipulated_object.position.x": 0.10,
        "privileged.task.manipulated_object.position.y": -0.20,
        "privileged.task.manipulated_object.position.z": 0.04,
        "privileged.task.target.position.x": -0.15,
        "privileged.task.target.position.y": 0.25,
        "privileged.task.target.position.z": 0.03,
        "privileged.task.manipulated_object.grasped": grasped,
        "privileged.task.manipulated_object.retained": grasped,
        "privileged.task.manipulated_object.gripper_contact": grasped,
        "privileged.task.manipulated_object.distance_to_eef_m": 0.0,
        "privileged.task.primary_relation_satisfied": success,
        "privileged.task.success": success,
        "_test.grasp_on_close": grasp_on_close,
    }


def _mock_primitives(monkeypatch, sidecar: dict) -> tuple[LiberoPrimitives, list]:
    class _Env:
        episode_terminated = False
        episode_truncated = False
        episode_steps = 0

        def privileged_critic_state(self):
            return dict(sidecar)

    primitives = object.__new__(LiberoPrimitives)
    primitives.env = _Env()
    primitives._allow_privileged_actions = True
    primitives._last_critic_proposals = []
    primitives._last_obs_eef_pos = np.asarray([0.0, 0.0, 0.5], dtype=np.float32)
    calls: list[tuple[str, dict]] = []

    def move_to(*, xyz, gripper, max_steps, tol, step_clip):
        calls.append(("move_to", {"xyz": xyz, "gripper": gripper,
                                   "max_steps": max_steps, "tol": tol,
                                   "step_clip": step_clip}))
        primitives.env.episode_steps += 3
        move_count = sum(name == "move_to" for name, _ in calls)
        if sidecar.get("_test.lose_on_move_call") == move_count:
            sidecar["privileged.task.manipulated_object.grasped"] = False
            sidecar["privileged.task.manipulated_object.retained"] = False
            sidecar["privileged.task.manipulated_object.gripper_contact"] = False
        if (
            sidecar.get("_test.contact_lift_on_move_call") == move_count
            and sidecar.get("privileged.task.manipulated_object.gripper_contact")
        ):
            sidecar["privileged.task.manipulated_object.position.z"] += float(
                sidecar.get("_test.contact_lift_m", 0.04)
            )
        if sidecar.get("_test.contact_limited_move_call") == move_count:
            xy_error = float(sidecar.get("_test.contact_limited_xy_error", 0.01))
            return {
                "name": "move_to",
                "steps_used": 3,
                "final_dist_m": 0.05,
                "final_eef_pos": [xyz[0] + xy_error, xyz[1], xyz[2] + 0.049],
                "target_xyz": list(xyz),
            }
        return {"name": "move_to", "steps_used": 3}

    def move_pose(
        *, xyz, target_pitch, target_yaw, gripper, max_steps, tol, step_clip
    ):
        calls.append(("move_pose", {
            "xyz": xyz,
            "target_pitch": target_pitch,
            "target_yaw": target_yaw,
            "gripper": gripper,
            "max_steps": max_steps,
            "tol": tol,
            "step_clip": step_clip,
        }))
        primitives.env.episode_steps += 3
        return {"name": "move_pose", "steps_used": 3}

    def set_gripper(*, gripper, steps):
        calls.append(("set_gripper", {"gripper": gripper, "steps": steps}))
        primitives.env.episode_steps += steps
        if sidecar.get("_test.grasp_on_close"):
            sidecar["privileged.task.manipulated_object.grasped"] = True
            sidecar["privileged.task.manipulated_object.retained"] = True
            sidecar["privileged.task.manipulated_object.gripper_contact"] = True
        if sidecar.get("_test.contact_on_close"):
            sidecar["privileged.task.manipulated_object.gripper_contact"] = True
        return {"name": "set_gripper", "steps": steps}

    def release(*, max_steps):
        calls.append(("release", {"max_steps": max_steps}))
        primitives.env.episode_steps += 2
        if sidecar.get("privileged.task.manipulated_object.grasped"):
            sidecar["privileged.task.primary_relation_satisfied"] = True
            sidecar["privileged.task.success"] = True
            primitives.env.episode_terminated = True
        return {"name": "release", "steps_used": 2}

    monkeypatch.setattr(primitives, "move_to", move_to)
    monkeypatch.setattr(primitives, "move_pose", move_pose)
    monkeypatch.setattr(primitives, "set_gripper", set_gripper)
    monkeypatch.setattr(primitives, "release", release)
    return primitives, calls


def test_privileged_pick_place_fails_closed_without_semantic_pose(monkeypatch):
    primitives, calls = _mock_primitives(monkeypatch, {"privileged.available": True})

    result = primitives.privileged_pick_place()

    assert result == {
        "name": "privileged_pick_place",
        "status": "semantic_state_unavailable",
    }
    assert calls == []


def test_privileged_pick_place_requires_privileged_authorization(monkeypatch):
    primitives, _calls = _mock_primitives(monkeypatch, _semantic_sidecar())
    primitives._allow_privileged_actions = False

    with pytest.raises(PermissionError, match="privileged action authorization"):
        primitives.privileged_pick_place()


def test_privileged_pick_place_does_not_carry_unverified_grasp(monkeypatch):
    sidecar = _semantic_sidecar(grasped=False)
    primitives, calls = _mock_primitives(monkeypatch, sidecar)

    result = primitives.privileged_pick_place()

    assert result["status"] == "grasp_not_verified"
    assert [name for name, _ in calls[:3]] == ["move_to", "move_to", "move_to"]
    assert [name for name, _ in calls[3:]] == ["set_gripper"] * 24
    assert not any(name == "release" for name, _ in calls)


def test_privileged_pick_place_uses_verified_closed_loop_sequence(monkeypatch):
    sidecar = _semantic_sidecar(grasp_on_close=True)
    primitives, calls = _mock_primitives(monkeypatch, sidecar)

    result = primitives.privileged_pick_place(
        max_steps_per_move=42,
        move_step_clip=0.05,
        lift_step_clip=0.01,
        transport_step_clip=0.02,
    )

    assert result["status"] == "success"
    assert result["primary_relation_satisfied"] is True
    assert result["libero_terminated"] is True
    assert [name for name, _ in calls] == [
        "move_to", "move_to", "move_to",
        "set_gripper", "set_gripper", "set_gripper", "set_gripper",
        "move_to", "move_to", "move_to", "release",
    ]
    assert all(
        call[1].get("max_steps") == 42
        for call in calls
        if call[0] == "move_to"
    )
    assert [call[1]["step_clip"] for call in calls if call[0] == "move_to"] == [
        0.05,
        0.05,
        0.05,
        0.01,
        0.02,
        0.02,
    ]
    assert calls[3][1] == {"gripper": 1.0, "steps": 1}
    assert calls[2][1]["xyz"] == pytest.approx([0.10, -0.236, 0.078])
    assert calls[8][1]["xyz"] == pytest.approx([-0.15, 0.214, 0.168])
    assert calls[9][1]["xyz"] == pytest.approx([-0.15, 0.214, 0.103])


def test_privileged_pick_place_accepts_extended_transport_budget(monkeypatch):
    sidecar = _semantic_sidecar(grasp_on_close=True)
    primitives, calls = _mock_primitives(monkeypatch, sidecar)

    result = primitives.privileged_pick_place(
        max_steps_per_move=120,
        transport_step_clip=0.01,
    )

    assert result["status"] == "success"
    assert all(
        call[1].get("max_steps") == 120
        for call in calls
        if call[0] == "move_to"
    )


def test_privileged_pick_place_caps_only_final_grasp_pose_move(monkeypatch):
    sidecar = _semantic_sidecar(grasp_on_close=True)
    primitives, calls = _mock_primitives(monkeypatch, sidecar)

    result = primitives.privileged_pick_place(
        max_steps_per_move=120,
        grasp_pose_max_steps=24,
    )

    assert result["status"] == "success"
    move_calls = [call for call in calls if call[0] == "move_to"]
    assert move_calls[2][1]["max_steps"] == 24
    assert all(
        call[1]["max_steps"] == 120
        for index, call in enumerate(move_calls)
        if index != 2
    )


def test_privileged_pick_place_holds_requested_orientation(monkeypatch):
    sidecar = _semantic_sidecar(grasp_on_close=True)
    primitives, calls = _mock_primitives(monkeypatch, sidecar)

    result = primitives.privileged_pick_place(
        approach_pitch=-0.11,
        approach_yaw=0.0,
    )

    assert result["status"] == "success"
    pose_calls = [call for call in calls if call[0] == "move_pose"]
    assert len(pose_calls) == 6
    assert all(call[1]["target_pitch"] == -0.11 for call in pose_calls)
    assert all(call[1]["target_yaw"] == 0.0 for call in pose_calls)
    assert not any(name == "move_to" for name, _ in calls)


def test_privileged_pick_place_accepts_contact_only_grasp_after_object_lift(
    monkeypatch,
):
    sidecar = _semantic_sidecar(grasped=False)
    sidecar["_test.contact_on_close"] = True
    sidecar["_test.contact_lift_on_move_call"] = 4
    primitives, _calls = _mock_primitives(monkeypatch, sidecar)

    result = primitives.privileged_pick_place()

    assert result["status"] == "placed"
    assert result["close"]["grasp_verified"] is True


def test_privileged_pick_place_configures_contact_lift_threshold(monkeypatch):
    sidecar = _semantic_sidecar(grasped=False)
    sidecar["_test.contact_on_close"] = True
    sidecar["_test.contact_lift_on_move_call"] = 4
    sidecar["_test.contact_lift_m"] = 0.012
    primitives, _calls = _mock_primitives(monkeypatch, sidecar)

    result = primitives.privileged_pick_place(contact_lift_min_m=0.01)

    assert result["status"] == "placed"
    assert result["close"]["grasp_verified"] is True

    sidecar["privileged.task.success"] = False
    primitives, _calls = _mock_primitives(monkeypatch, sidecar)
    result = primitives.privileged_pick_place(contact_lift_min_m=0.02)
    assert result["status"] == "grasp_lost_during_lift"


def test_privileged_pick_place_can_split_long_moves(monkeypatch):
    sidecar = _semantic_sidecar(grasp_on_close=True)
    primitives, calls = _mock_primitives(monkeypatch, sidecar)

    result = primitives.privileged_pick_place(max_segment_distance_m=0.10)

    assert result["status"] == "success"
    assert result["carry"]["waypoint_count"] >= 2
    assert all(
        call[0] == "move_to" or call[0] in {"set_gripper", "release"}
        for call in calls
    )


def test_privileged_pick_place_can_raise_before_horizontal_carry(monkeypatch):
    sidecar = _semantic_sidecar(grasp_on_close=True)
    primitives, calls = _mock_primitives(monkeypatch, sidecar)
    primitives._last_obs_eef_pos = np.asarray([0.0, 0.0, 0.10], dtype=np.float32)

    result = primitives.privileged_pick_place(vertical_first_carry=True)

    assert result["status"] == "success"
    assert result["carry_clearance"] is not None
    move_calls = [call for call in calls if call[0] == "move_to"]
    clearance = move_calls[4][1]["xyz"]
    carry = move_calls[5][1]["xyz"]
    assert clearance[:2] == pytest.approx([0.0, 0.0])
    assert carry[2] == pytest.approx(clearance[2])


def test_privileged_pick_place_preserves_retained_grasp(monkeypatch):
    sidecar = _semantic_sidecar(grasped=True)
    primitives, calls = _mock_primitives(monkeypatch, sidecar)

    result = primitives.privileged_pick_place(max_steps_per_move=42)

    assert result["status"] == "success"
    assert result["primary_relation_satisfied"] is True
    assert result["close"]["name"] == "retained_grasp"
    assert [name for name, _ in calls] == ["move_to", "move_to", "move_to", "release"]
    assert calls[0][1]["gripper"] == 1.0
    assert calls[0][1]["xyz"] == pytest.approx([0.0, 0.0, 0.63])
    assert calls[1][1]["xyz"] == pytest.approx([-0.25, 0.45, 0.59])
    assert calls[2][1]["xyz"] == pytest.approx([-0.25, 0.45, 0.525])


def test_privileged_pick_place_releases_after_aligned_contact_limited_descent(
    monkeypatch,
):
    sidecar = _semantic_sidecar(grasped=True)
    sidecar["_test.contact_limited_move_call"] = 3
    primitives, calls = _mock_primitives(monkeypatch, sidecar)

    result = primitives.privileged_pick_place()

    assert result["status"] == "success"
    assert result["contact_limited_release"] is True
    assert calls[-1][0] == "release"


def test_privileged_pick_place_rejects_xy_misaligned_contact_descent(monkeypatch):
    sidecar = _semantic_sidecar(grasped=True)
    sidecar["_test.contact_limited_move_call"] = 3
    sidecar["_test.contact_limited_xy_error"] = 0.04
    primitives, calls = _mock_primitives(monkeypatch, sidecar)

    result = primitives.privileged_pick_place()

    assert result["status"] == "place_not_reached"
    assert not any(name == "release" for name, _ in calls)


def test_privileged_pick_place_stops_when_grasp_is_lost_during_lift(monkeypatch):
    sidecar = _semantic_sidecar(grasp_on_close=True)
    sidecar["_test.lose_on_move_call"] = 4
    primitives, calls = _mock_primitives(monkeypatch, sidecar)

    result = primitives.privileged_pick_place()

    assert result["status"] == "grasp_lost_during_lift"
    assert result["grasped"] is False
    assert sum(name == "move_to" for name, _ in calls) == 4
    assert not any(name == "release" for name, _ in calls)
