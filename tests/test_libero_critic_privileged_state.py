# Copyright (c) 2026 RPent Contributors
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from robots.libero.critic_runtime import (
    LIBERO_CRITIC_FEATURES,
    extract_libero_critic_features,
)
from robots.libero.env_server import LiberoEnvFacade
from robots.libero.privileged_sensors import (
    collect_privileged_critic_state,
    collect_privileged_semantic_joint_plan,
    wrap_libero_env_factories,
)


class _Model:
    body_names = ["world", "robot0_hand", "red_cup", "target_bowl"]
    geom_names = [
        "robot0_left_finger",
        "robot0_right_finger",
        "red_cup_geom",
        "target_bowl_geom",
    ]
    site_names = ["gripper0_grip_site"]
    joint_names = ["cabinet_slide", "stove_knob_hinge"]
    nbody = len(body_names)
    ngeom = len(geom_names)
    nsite = len(site_names)
    njnt = len(joint_names)
    body_parentid = np.asarray([0, 0, 0, 0])
    geom_bodyid = np.asarray([1, 1, 2, 3])
    jnt_type = np.asarray([2, 3])
    jnt_qposadr = np.asarray([0, 1])
    jnt_range = np.asarray([[0.0, 0.4], [-1.0, 1.0]])

    def body_name2id(self, name: str) -> int:
        return self.body_names.index(name)

    def body_id2name(self, identifier: int) -> str:
        return self.body_names[identifier]

    def geom_name2id(self, name: str) -> int:
        return self.geom_names.index(name)

    def geom_id2name(self, identifier: int) -> str:
        return self.geom_names[identifier]

    def site_name2id(self, name: str) -> int:
        return self.site_names.index(name)

    def site_id2name(self, identifier: int) -> str:
        return self.site_names[identifier]

    def joint_name2id(self, name: str) -> int:
        return self.joint_names.index(name)

    def joint_id2name(self, identifier: int) -> str:
        return self.joint_names[identifier]


class _Contact:
    def __init__(self, first: int, second: int) -> None:
        self.geom1 = first
        self.geom2 = second
        self.dist = -0.001
        self.pos = np.zeros(3)
        self.frame = np.eye(3).reshape(-1)


def _task_env() -> Any:
    model = _Model()
    data = SimpleNamespace(
        body_xpos=np.asarray(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.2],
                [0.0, 0.0, 0.1],
                [0.3, 0.0, 0.1],
            ]
        ),
        body_xquat=np.tile(np.asarray([1.0, 0.0, 0.0, 0.0]), (4, 1)),
        site_xpos=np.asarray([[0.0, 0.0, 0.2]]),
        qpos=np.asarray([0.1, 0.5]),
        ncon=2,
        contact=[_Contact(0, 2), _Contact(1, 2)],
    )
    gripper = SimpleNamespace(
        contact_geoms=["robot0_left_finger", "robot0_right_finger"]
    )
    robot_model = SimpleNamespace(
        contact_geoms=["robot0_left_finger", "robot0_right_finger"]
    )
    red_cup = SimpleNamespace(
        name="red_cup",
        root_body="red_cup",
        contact_geoms=["red_cup_geom"],
    )
    target_bowl = SimpleNamespace(
        name="target_bowl",
        root_body="target_bowl",
        contact_geoms=["target_bowl_geom"],
    )
    raw = SimpleNamespace(
        sim=SimpleNamespace(model=model, data=data),
        robots=[SimpleNamespace(gripper=gripper, robot_model=robot_model)],
        objects_dict={"red_cup": red_cup, "target_bowl": target_bowl},
        parsed_problem={"goal_state": [("In", "red_cup", "target_bowl")]},
        relation_satisfied=False,
    )
    raw._eval_predicate = lambda atom: raw.relation_satisfied and atom[0] == "In"
    raw._check_success = lambda: raw.relation_satisfied
    return raw


def test_privileged_state_exposes_bddl_goal_pose_contact_and_grasp() -> None:
    raw = _task_env()
    state = collect_privileged_critic_state(SimpleNamespace(env=raw))

    assert state["privileged.available"] is True
    assert state["privileged.task.semantic_available"] is True
    assert state["privileged.task.primary_relation"] == "In"
    assert state["privileged.task.manipulated_object.name"] == "red_cup"
    assert state["privileged.task.target.name"] == "target_bowl"
    assert state["privileged.task.manipulated_object.grasped"] is True
    assert state["privileged.task.manipulated_object.in_target"] is False
    assert state["privileged.task.manipulated_object.distance_to_target_m"] == pytest.approx(
        0.3
    )
    assert state["privileged.entity.red_cup.gripper_contact"] is True
    assert state["privileged.entity.target_bowl.position.x"] == pytest.approx(0.3)
    assert state["privileged.joint.cabinet_slide.normalized"] == pytest.approx(0.25)
    assert state["privileged.joint.stove_knob_hinge.distance_to_upper"] == pytest.approx(
        0.5
    )


def test_privileged_state_selects_unsatisfied_object_in_multi_object_goal() -> None:
    class MultiObjectModel(_Model):
        body_names = [
            "world",
            "robot0_hand",
            "alphabet_soup_1",
            "cream_cheese_1",
            "basket_1",
        ]
        geom_names = [
            "robot0_left_finger",
            "robot0_right_finger",
            "alphabet_soup_1_geom",
            "cream_cheese_1_geom",
            "basket_1_geom",
        ]
        nbody = len(body_names)
        ngeom = len(geom_names)
        body_parentid = np.asarray([0, 0, 0, 0, 0])
        geom_bodyid = np.asarray([1, 1, 2, 3, 4])

    model = MultiObjectModel()
    data = SimpleNamespace(
        body_xpos=np.asarray(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.2],
                [0.3, 0.0, 0.1],
                [0.1, 0.0, 0.1],
                [0.3, 0.0, 0.1],
            ]
        ),
        body_xquat=np.tile(np.asarray([1.0, 0.0, 0.0, 0.0]), (5, 1)),
        site_xpos=np.asarray([[0.0, 0.0, 0.2]]),
        qpos=np.asarray([0.1, 0.5]),
        ncon=0,
        contact=[],
    )

    def entity(name: str) -> Any:
        return SimpleNamespace(
            name=name,
            root_body=name,
            contact_geoms=[f"{name}_geom"],
        )

    raw = SimpleNamespace(
        sim=SimpleNamespace(model=model, data=data),
        robots=[],
        objects_dict={
            name: entity(name)
            for name in ("alphabet_soup_1", "cream_cheese_1", "basket_1")
        },
        parsed_problem={
            "goal_state": [
                ("In", "alphabet_soup_1", "basket_1"),
                ("In", "cream_cheese_1", "basket_1"),
            ]
        },
    )
    raw._eval_predicate = lambda atom: atom[1] == "alphabet_soup_1"
    raw._check_success = lambda: False

    state = collect_privileged_critic_state(SimpleNamespace(env=raw))

    assert state["privileged.task.goal.satisfied_count"] == 1
    assert state["privileged.task.manipulated_object.name"] == "cream_cheese_1"
    assert state["privileged.task.target.name"] == "basket_1"
    assert state["privileged.task.primary_relation_satisfied"] is False


def test_privileged_state_falls_back_to_first_goal_when_all_are_satisfied() -> None:
    raw = _task_env()
    raw.parsed_problem["goal_state"] = [
        ("In", "red_cup", "target_bowl"),
        ("On", "red_cup", "target_bowl"),
    ]
    raw._eval_predicate = lambda _atom: True

    state = collect_privileged_critic_state(SimpleNamespace(env=raw))

    assert state["privileged.task.primary_relation"] == "In"
    assert state["privileged.task.primary_relation_satisfied"] is True


def test_semantic_joint_plan_is_range_aware_and_does_not_mutate_qpos() -> None:
    class JointModel(_Model):
        body_names = ["world", "flat_stove_1_button"]
        geom_names = ["flat_stove_1_button_visual", "flat_stove_1_button_collision"]
        nbody = 2
        ngeom = 2
        nsite = 0
        njnt = 1
        body_parentid = np.asarray([0, 0])
        geom_bodyid = np.asarray([1])
        jnt_type = np.asarray([3])
        jnt_qposadr = np.asarray([0])
        jnt_dofadr = np.asarray([0])
        jnt_bodyid = np.asarray([1])
        jnt_axis = np.asarray([[0.0, 0.0, 1.0]])
        jnt_pos = np.asarray([[0.01, 0.02, 0.03]])
        jnt_range = np.asarray([[-0.005, 2.1]])
        geom_bodyid = np.asarray([1, 1])
        geom_type = np.asarray([7, 6])  # visual mesh, collision box
        geom_contype = np.asarray([0, 1])
        geom_conaffinity = np.asarray([0, 1])
        geom_size = np.asarray([[1.0, 1.0, 1.0], [0.01, 0.04, 0.02]])

        def body_name2id(self, name: str) -> int:
            return self.body_names.index(name)

        def body_id2name(self, identifier: int) -> str:
            return self.body_names[identifier]

        def geom_id2name(self, identifier: int) -> str:
            return self.geom_names[identifier]

        def joint_name2id(self, name: str) -> int:
            return ["flat_stove_1_button"].index(name)

        def joint_id2name(self, identifier: int) -> str:
            return ["flat_stove_1_button"][identifier]

    model = JointModel()
    qpos = np.asarray([0.6])
    raw = SimpleNamespace(
            sim=SimpleNamespace(
                model=model,
                data=SimpleNamespace(
                    qpos=qpos,
                    qvel=np.asarray([0.125]),
                    xpos=np.asarray([[0.0, 0.0, 0.0], [0.2, 0.1, 0.9]]),
                    xmat=np.tile(np.eye(3).reshape(1, 9), (2, 1)),
                    geom_xpos=np.asarray(
                        [[0.2, 0.1, 2.0], [0.23, 0.1, 0.92]]
                    ),
                    geom_xmat=np.tile(np.eye(3).reshape(1, 9), (2, 1)),
                ),
            ),
        _eef_xpos=np.asarray([0.2, 0.2, 1.1]),
        objects_dict={},
        fixtures_dict={},
        robots=[],
    )

    plan = collect_privileged_semantic_joint_plan(
        SimpleNamespace(env=raw),
        entity="flat_stove_1",
        joint="button",
        direction="lower",
    )

    assert plan["joint"] == "flat_stove_1_button"
    assert plan["qpos"] == pytest.approx(0.6)
    assert plan["qvel"] == pytest.approx(0.125)
    assert plan["range_lower"] == pytest.approx(-0.005)
    assert plan["range_upper"] == pytest.approx(2.1)
    assert plan["goal_satisfied"] is False
    assert np.isfinite(np.asarray(plan["approach_position_world"])).all()
    assert np.isfinite(np.asarray(plan["tangent_direction_world"])).all()
    assert qpos.tolist() == [0.6]

    # The visual mesh has a deliberately absurd size.  It must not lift the
    # press pose above the collision box that the controller can touch.
    assert plan["press_position_world"][2] == pytest.approx(0.948)

    qpos[0] = -0.002
    assert collect_privileged_semantic_joint_plan(
        SimpleNamespace(env=raw),
        entity="flat_stove_1",
        joint="button",
        direction="lower",
    )["goal_satisfied"] is False
    qpos[0] = -0.004
    assert collect_privileged_semantic_joint_plan(
        SimpleNamespace(env=raw),
        entity="flat_stove_1",
        joint="button",
        direction="lower",
    )["goal_satisfied"] is True


def test_semantic_joint_plan_uses_slide_axis_and_outer_handle() -> None:
    class DrawerModel(_Model):
        body_names = ["world", "wooden_cabinet_1_top"]
        geom_names = [
            "drawer_front_collision",
            "top_handle_collision",
            "robot0_finger_collision",
        ]
        nbody = 2
        ngeom = 3
        nsite = 0
        njnt = 1
        body_parentid = np.asarray([0, 0])
        geom_bodyid = np.asarray([1, 1, 0])
        geom_type = np.asarray([6, 6, 6])
        geom_contype = np.asarray([1, 1, 1])
        geom_conaffinity = np.asarray([1, 1, 1])
        geom_size = np.asarray(
            [[0.1, 0.003, 0.04], [0.04, 0.008, 0.008], [0.01, 0.04, 0.01]]
        )
        jnt_type = np.asarray([2])
        jnt_qposadr = np.asarray([0])
        jnt_dofadr = np.asarray([0])
        jnt_bodyid = np.asarray([1])
        jnt_axis = np.asarray([[0.0, 1.0, 0.0]])
        jnt_pos = np.asarray([[0.0, 0.0, 0.0]])
        jnt_range = np.asarray([[-0.16, 0.01]])

        def joint_name2id(self, name: str) -> int:
            return ["top_level"].index(name)

        def joint_id2name(self, identifier: int) -> str:
            return ["top_level"][identifier]

    model = DrawerModel()
    raw = SimpleNamespace(
        sim=SimpleNamespace(
            model=model,
            data=SimpleNamespace(
                qpos=np.asarray([0.0]),
                qvel=np.asarray([0.0]),
                xpos=np.asarray([[0.0, 0.0, 0.0], [0.5, 0.2, 0.8]]),
                xmat=np.tile(np.eye(3).reshape(1, 9), (2, 1)),
                geom_xpos=np.asarray(
                    [[0.5, 0.12, 0.98], [0.5, 0.09, 0.94], [0.5, 0.05, 1.12]]
                ),
                geom_xmat=np.tile(np.eye(3).reshape(1, 9), (3, 1)),
            ),
        ),
        _eef_xpos=np.asarray([0.5, 0.0, 1.1]),
        objects_dict={},
        fixtures_dict={},
        robots=[
            SimpleNamespace(
                gripper=SimpleNamespace(contact_geoms=["robot0_finger_collision"])
            )
        ],
    )

    plan = collect_privileged_semantic_joint_plan(
        SimpleNamespace(env=raw),
        entity="wooden_cabinet_1",
        joint="top_level",
        direction="lower",
    )

    assert plan["joint_type"] == "slide"
    assert plan["tangent_direction_world"] == pytest.approx([0.0, -1.0, 0.0])
    assert plan["press_position_world"][1] == pytest.approx(-0.008)
    assert plan["press_position_world"][2] == pytest.approx(0.92)
    assert plan["approach_position_world"][2] == pytest.approx(0.985)


def test_privileged_state_tracks_target_progress_retention_and_release() -> None:
    raw = _task_env()
    wrapped = wrap_libero_env_factories([lambda: SimpleNamespace(env=raw)])[0]()

    initial = wrapped.rpent_privileged_critic_state(reset_tracker=True)
    assert initial["privileged.task.manipulated_object.ever_grasped"] is True
    assert initial[
        "privileged.task.manipulated_object.target_progress_available"
    ] is False

    raw.sim.data.body_xpos[2, 0] = 0.1
    progressed = wrapped.rpent_privileged_critic_state()
    assert progressed[
        "privileged.task.manipulated_object.target_progress_m"
    ] == pytest.approx(0.1)
    assert progressed["privileged.task.manipulated_object.retained"] is True

    raw.sim.data.ncon = 0
    raw.sim.data.contact = []
    released = wrapped.rpent_privileged_critic_state()
    assert released["privileged.task.manipulated_object.released_now"] is True
    assert released["privileged.task.manipulated_object.ever_released"] is True
    assert released["privileged.task.manipulated_object.retained"] is False


def test_factory_rebinds_asset_override_inside_spawned_worker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("robosuite")
    import robosuite.models
    import robosuite.utils.mjcf_utils as mjcf_utils

    monkeypatch.setenv("LIBERO_ASSETS_ROOT_OVERRIDE", str(tmp_path))
    observed: list[tuple[str, str]] = []

    def factory() -> Any:
        # RLinF's nested LIBERO import can reset this module global after the
        # outer worker bound it. Path completion must remain on the overlay.
        robosuite.models.assets_root = "/package/default/assets"
        observed.append(
            (
                str(robosuite.models.assets_root),
                mjcf_utils.xml_path_completion("scenes/example.xml"),
            )
        )
        return SimpleNamespace()

    wrapped = wrap_libero_env_factories([factory])[0]()

    assert observed == [
        (
            "/package/default/assets",
            str(tmp_path.resolve() / "scenes/example.xml"),
        )
    ]
    assert callable(wrapped.rpent_privileged_contacts)


def test_feature_extractor_merges_sidecar_without_mutating_actor_observation() -> None:
    observation = {"states": np.asarray([0.01, 0.0, 0.2, 0, 0, 0, 0.01, -0.01])}
    features = extract_libero_critic_features(
        observation,
        step_index=4,
        reward=0.0,
        terminated=False,
        truncated=False,
        privileged_state={
            "privileged.task.manipulated_object.grasped": True,
            "privileged.task.manipulated_object.distance_to_target_m": 0.2,
        },
        action=[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0],
        previous_eef=[0.0, 0.0, 0.2],
    )

    assert features["robot.eef.motion_m"] == pytest.approx(0.01)
    assert features["command.realization.direction_cosine"] == pytest.approx(1.0)
    assert features["privileged.task.manipulated_object.grasped"] is True
    assert set(observation) == {"states"}
    assert "privileged.task.manipulated_object.grasped" in LIBERO_CRITIC_FEATURES

    with pytest.raises(ValueError, match="invalid LIBERO privileged feature"):
        extract_libero_critic_features(
            observation,
            step_index=4,
            reward=0.0,
            terminated=False,
            truncated=False,
            privileged_state={"actor.object_pose": 0.2},
        )


class _Worker:
    def env_call(self, method: str, *, kwargs: dict[str, Any], target: str) -> Any:
        assert method == "rpent_privileged_critic_state"
        assert target == "self"
        assert kwargs == {"reset_tracker": False}
        return {
            "privileged.available": True,
            "privileged.task.manipulated_object.grasped": True,
        }


class _FacadeEnv:
    def __init__(self) -> None:
        self.env = SimpleNamespace(workers=[_Worker()])
        self.steps = 0

    def step(self, _action: Any) -> tuple[Any, ...]:
        self.steps += 1
        states = np.zeros((1, 8), dtype=np.float32)
        return (
            {"states": states},
            np.asarray([0.0]),
            np.asarray([False]),
            np.asarray([False]),
            {},
        )


def test_online_critic_receives_privilege_but_actor_observation_does_not() -> None:
    facade = LiberoEnvFacade(_FacadeEnv(), meta={})
    observations, _rewards, _terms, _truncations, info = facade.critic_chunk_step(
        np.zeros((1, 7), dtype=np.float32),
        critic_rules=[
            {
                "rule_id": "critic-grasped",
                "title": "grasp is active",
                "feature": "privileged.task.manipulated_object.grasped",
                "operator": "eq",
                "threshold": True,
                "dwell_steps": 1,
                "cooldown_steps": 0,
                "proposal": "replan toward the BDDL target",
                "evidence_ids": ["test"],
                "safety_only": False,
                "activation_conditions": [],
            }
        ],
    )

    assert "privileged.task.manipulated_object.grasped" not in observations[0]
    assert info["step_records"][0]["state"][
        "privileged.task.manipulated_object.grasped"
    ] is True
    assert [row["rule_id"] for row in info["critic_proposals"]] == [
        "critic-grasped"
    ]
