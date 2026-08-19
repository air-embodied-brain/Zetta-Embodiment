# Copyright (c) 2026 Zetta Contributors
"""Audited privileged RoboCasa state used by local proposal tools."""

from __future__ import annotations

from collections.abc import Mapping
from math import cos, sin
from typing import Any

import numpy as np


def _task_environment(environment: Any) -> Any:
    queue = [environment]
    seen: set[int] = set()
    while queue:
        candidate = queue.pop(0)
        if candidate is None or id(candidate) in seen:
            continue
        seen.add(id(candidate))
        if hasattr(candidate, "sim") and (
            hasattr(candidate, "dishwasher") or hasattr(candidate, "robots")
        ):
            return candidate
        for attribute in ("unwrapped", "env"):
            child = getattr(candidate, attribute, None)
            if child is not candidate:
                queue.append(child)
    raise RuntimeError("RoboCasa task environment and simulator are unavailable")


def _name_to_id(model: Any, kind: str, name: str) -> int:
    modern = getattr(model, kind, None)
    if callable(modern):
        try:
            return int(modern(name).id)
        except Exception:
            pass
    legacy = getattr(model, f"{kind}_name2id", None)
    if callable(legacy):
        return int(legacy(name))
    raise KeyError(f"cannot resolve MuJoCo {kind} {name!r}")


def _id_to_name(model: Any, kind: str, identifier: int) -> str:
    modern = getattr(model, kind, None)
    if callable(modern):
        try:
            return str(modern(identifier).name)
        except Exception:
            pass
    legacy = getattr(model, f"{kind}_id2name", None)
    if callable(legacy):
        value = legacy(identifier)
        return str(value) if value is not None else f"{kind}-{identifier}"
    return f"{kind}-{identifier}"


def _descends_from(model: Any, body_id: int, ancestor: int) -> bool:
    current = int(body_id)
    while current >= 0:
        if current == ancestor:
            return True
        parent = int(model.body_parentid[current])
        if parent == current:
            break
        current = parent
    return False


def _instruction(observation: Mapping[str, Any]) -> str:
    return str(observation.get("annotation.human.task_description", ""))


def _rack_direction(instruction: str) -> str:
    normalized = " ".join(instruction.lower().strip().rstrip(".").split())
    if normalized.endswith(" in") or "rack in" in normalized:
        return "in"
    if normalized.endswith(" out") or "rack out" in normalized:
        return "out"
    return "unknown"


def _robot_collision_geoms(task_env: Any, model: Any) -> set[int]:
    """Resolve the robot's declared contact geometry without name guessing."""

    robots = tuple(getattr(task_env, "robots", ()) or ())
    if not robots:
        return set()
    robot = robots[0]
    robot_model = getattr(robot, "robot_model", None)
    gripper = getattr(robot, "gripper", None)
    declared = set(getattr(robot_model, "contact_geoms", ()) or ())
    declared.update(getattr(gripper, "contact_geoms", ()) or ())
    available = {
        _id_to_name(model, "geom", geom_id): geom_id
        for geom_id in range(int(model.ngeom))
    }
    return {int(available[name]) for name in declared if name in available}


def _robot_base_rotation(task_env: Any, model: Any, data: Any) -> np.ndarray | None:
    """Return the world-from-base rotation across known robosuite variants."""

    robots = tuple(getattr(task_env, "robots", ()) or ())
    if not robots:
        return None
    robot_model = getattr(robots[0], "robot_model", None)
    base_model = getattr(robot_model, "base", None)
    candidates: list[str] = []
    for owner in (base_model, robot_model):
        correct_naming = getattr(owner, "correct_naming", None)
        if callable(correct_naming):
            try:
                candidates.append(str(correct_naming("center")))
            except Exception:
                continue
    candidates.extend(("mobilebase0_center", "robot0_center", "robot0_right_center"))
    for name in dict.fromkeys(candidates):
        try:
            site_id = _name_to_id(model, "site", name)
        except (KeyError, ValueError):
            continue
        return np.asarray(data.site_xmat[site_id], dtype=np.float64).reshape(3, 3)
    return None


def _slide_dishwasher_state(
    task_env: Any, observation: Mapping[str, Any]
) -> dict[str, Any]:
    dishwasher = getattr(task_env, "dishwasher", None)
    simulator = getattr(task_env, "sim", None)
    if dishwasher is None or simulator is None:
        return {}
    update = getattr(dishwasher, "update_state", None)
    if callable(update):
        update(task_env)
    get_state = getattr(dishwasher, "get_state", None)
    appliance_state = get_state(task_env) if callable(get_state) else {}
    if not isinstance(appliance_state, Mapping) or "rack" not in appliance_state:
        return {}
    rack_position = float(appliance_state["rack"])
    joint_names = getattr(dishwasher, "_joint_names", {})
    joint_name = str(joint_names.get("rack", ""))
    if not joint_name:
        return {}
    model = simulator.model
    joint_id = _name_to_id(model, "joint", joint_name)
    rack_body_id = int(model.jnt_bodyid[joint_id])
    joint_range = np.asarray(model.jnt_range[joint_id], dtype=np.float64)
    joint_span = float(abs(joint_range[1] - joint_range[0]))
    raw_qpos = float(simulator.data.qpos[int(model.jnt_qposadr[joint_id])])
    local_axis = np.asarray(model.jnt_axis[joint_id], dtype=np.float64)
    body_rotation = np.asarray(
        simulator.data.body_xmat[rack_body_id], dtype=np.float64
    ).reshape(3, 3)
    rail_axis_world = body_rotation @ local_axis
    if float(joint_range[0]) < 0.0:
        rail_axis_world *= -1.0
    norm = float(np.linalg.norm(rail_axis_world))
    if norm > 1e-12:
        rail_axis_world /= norm
    should_pull = getattr(task_env, "should_pull", None)
    direction = (
        "out"
        if should_pull is True
        else "in"
        if should_pull is False
        else _rack_direction(_instruction(observation))
    )
    success_axis_world = rail_axis_world * (-1.0 if direction == "in" else 1.0)
    base_rotation = _robot_base_rotation(task_env, model, simulator.data)
    success_axis_base = (
        base_rotation.T @ success_axis_world
        if base_rotation is not None
        else np.full(3, np.nan, dtype=np.float64)
    )
    base_norm = float(np.linalg.norm(success_axis_base))
    if np.isfinite(base_norm) and base_norm > 1e-12:
        success_axis_base /= base_norm
    threshold = 0.05 if direction == "in" else 0.95
    if direction == "in":
        residual = max(0.0, rack_position - threshold)
    elif direction == "out":
        residual = max(0.0, threshold - rack_position)
    else:
        residual = float("nan")

    rack_geom_ids = {
        geom_id
        for geom_id in range(int(model.ngeom))
        if _descends_from(model, int(model.geom_bodyid[geom_id]), rack_body_id)
    }
    rack_geom_names = sorted(_id_to_name(model, "geom", item) for item in rack_geom_ids)
    robot_geom_ids = _robot_collision_geoms(task_env, model)
    contact_pairs: list[list[str]] = []
    target_contact = False
    non_target_collision = False
    for index in range(int(simulator.data.ncon)):
        contact = simulator.data.contact[index]
        first, second = int(contact.geom1), int(contact.geom2)
        if first in robot_geom_ids and second in rack_geom_ids:
            target_contact = True
        if second in robot_geom_ids and first in rack_geom_ids:
            target_contact = True
        if first in robot_geom_ids or second in robot_geom_ids:
            contact_pairs.append(
                [_id_to_name(model, "geom", first), _id_to_name(model, "geom", second)]
            )
            if first not in rack_geom_ids and second not in rack_geom_ids:
                non_target_collision = True
    return {
        "privileged.source": "live_mujoco_simulator",
        "privileged.class": "simulator_ground_truth",
        "privileged.dishwasher.rack.position": rack_position,
        "privileged.dishwasher.rack.raw_qpos": raw_qpos,
        "privileged.dishwasher.rack.joint_name": joint_name,
        "privileged.dishwasher.rack.joint_range_m": joint_range.astype(float).tolist(),
        "privileged.dishwasher.rack.joint_span_m": joint_span,
        "privileged.dishwasher.rack.commanded_direction": direction,
        "privileged.dishwasher.rack.success_threshold": threshold,
        "privileged.dishwasher.rack.residual_to_success": residual,
        "privileged.dishwasher.rack.remaining_to_success_m": residual * joint_span,
        "privileged.dishwasher.rack.success_direction_world": success_axis_world.astype(
            float
        ).tolist(),
        "privileged.dishwasher.rack.success_direction_base": success_axis_base.astype(
            float
        ).tolist(),
        "privileged.dishwasher.rack.geom_names": rack_geom_names,
        "privileged.dishwasher.rack.target_contact": target_contact,
        "privileged.dishwasher.collision.detected": non_target_collision,
        "privileged.dishwasher.collision.pairs": contact_pairs,
    }


def _relative_to_fixture(point: np.ndarray, fixture: Any) -> list[float]:
    """Express a world point in the fixture's planar frame."""

    origin = np.asarray(getattr(fixture, "pos"), dtype=np.float64)
    yaw = float(getattr(fixture, "rot", 0.0))
    delta = np.asarray(point, dtype=np.float64) - origin
    c, s = cos(yaw), sin(yaw)
    return [
        float(c * delta[0] + s * delta[1]),
        float(-s * delta[0] + c * delta[1]),
        float(delta[2]),
    ]


def _fixture_rotation(fixture: Any) -> np.ndarray:
    yaw = float(getattr(fixture, "rot", 0.0))
    c, s = cos(yaw), sin(yaw)
    return np.asarray(
        [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64
    )


def _fixture_geom_ids(model: Any, fixture: Any) -> set[int]:
    """Resolve all fixture geoms without relying on generated name prefixes."""

    root_body = getattr(fixture, "root_body", None)
    if root_body:
        try:
            root_id = _name_to_id(model, "body", str(root_body))
            return {
                geom_id
                for geom_id in range(int(model.ngeom))
                if _descends_from(model, int(model.geom_bodyid[geom_id]), root_id)
            }
        except (KeyError, ValueError, TypeError):
            pass
    declared = set(getattr(fixture, "contact_geoms", ()) or ())
    names = {
        _id_to_name(model, "geom", geom_id): geom_id
        for geom_id in range(int(model.ngeom))
    }
    return {int(names[name]) for name in declared if name in names}


def _pick_place_state(task_env: Any, observation: Mapping[str, Any]) -> dict[str, Any]:
    """Record immutable simulator truth for PickPlaceDrawerToCounter."""

    simulator = getattr(task_env, "sim", None)
    objects = getattr(task_env, "objects", None)
    drawer = getattr(task_env, "drawer", None)
    counter = getattr(task_env, "counter", None)
    if simulator is None or not isinstance(objects, Mapping) or drawer is None:
        return {}
    obj = objects.get("obj")
    if obj is None or not hasattr(task_env, "obj_body_id"):
        return {}

    import robocasa.utils.object_utils as OU

    model = simulator.model
    data = simulator.data
    obj_body_id = int(task_env.obj_body_id["obj"])
    obj_pos = np.asarray(data.body_xpos[obj_body_id], dtype=np.float64)
    obj_quat = np.asarray(data.body_xquat[obj_body_id], dtype=np.float64)
    obj_mat = np.asarray(data.body_xmat[obj_body_id], dtype=np.float64).reshape(3, 3)
    robot = tuple(getattr(task_env, "robots", ()) or ())
    eef_pos = None
    if robot:
        try:
            eef_id = int(robot[0].eef_site_id["right"])
            eef_pos = np.asarray(data.site_xpos[eef_id], dtype=np.float64)
        except (KeyError, IndexError, TypeError):
            eef_pos = None

    inside_drawer = bool(OU.obj_inside_of(task_env, "obj", drawer))
    on_any_counter = bool(OU.check_obj_any_counter_contact(task_env, "obj"))
    gripper_far = bool(OU.gripper_obj_far(task_env, "obj"))
    grasped = bool(OU.check_obj_grasped(task_env, "obj"))
    object_drawer_contact = bool(task_env.check_contact(obj, drawer))
    object_counter_contact = bool(
        counter is not None and task_env.check_contact(obj, counter)
    )
    gripper_object_contact = False
    if robot:
        try:
            gripper = robot[0].gripper["right"]
            gripper_object_contact = bool(task_env.check_contact(gripper, obj))
        except (KeyError, TypeError):
            pass

    object_geom_ids = {
        geom_id
        for geom_id in range(int(model.ngeom))
        if _descends_from(model, int(model.geom_bodyid[geom_id]), obj_body_id)
    }
    drawer_geom_ids = _fixture_geom_ids(model, drawer)
    counter_geom_ids = _fixture_geom_ids(model, counter) if counter is not None else set()
    robot_geom_ids = _robot_collision_geoms(task_env, model)
    contact_pairs: list[list[str]] = []
    object_contact_pairs: list[list[str]] = []
    robot_contact_pairs: list[list[str]] = []
    robot_non_target_contact = False
    for index in range(int(data.ncon)):
        contact = data.contact[index]
        first, second = int(contact.geom1), int(contact.geom2)
        pair = [_id_to_name(model, "geom", first), _id_to_name(model, "geom", second)]
        contact_pairs.append(pair)
        if first in object_geom_ids or second in object_geom_ids:
            object_contact_pairs.append(pair)
        if first in robot_geom_ids or second in robot_geom_ids:
            robot_contact_pairs.append(pair)
            other = second if first in robot_geom_ids else first
            if (
                other not in object_geom_ids
                and other not in drawer_geom_ids
                and other not in counter_geom_ids
            ):
                robot_non_target_contact = True

    drawer_rel = _relative_to_fixture(obj_pos, drawer)
    counter_rel = _relative_to_fixture(obj_pos, counter) if counter is not None else None
    drawer_rot = _fixture_rotation(drawer)
    counter_rot = _fixture_rotation(counter) if counter is not None else None
    obj_rot_drawer = drawer_rot.T @ obj_mat
    obj_rot_counter = counter_rot.T @ obj_mat if counter_rot is not None else None
    drawer_regions = {}
    try:
        drawer_regions = {
            str(name): [
                np.asarray(point, dtype=np.float64).astype(float).tolist()
                for point in points
            ]
            for name, points in drawer.get_int_sites(relative=False).items()
        }
    except Exception:
        drawer_regions = {}

    state: dict[str, Any] = {
        "privileged.source": "live_mujoco_simulator",
        "privileged.class": "simulator_ground_truth",
        "privileged.pick_place.task": type(task_env).__name__,
        "privileged.pick_place.instruction": _instruction(observation),
        "privileged.pick_place.object_position_world": obj_pos.astype(float).tolist(),
        "privileged.pick_place.object_quaternion_wxyz": obj_quat.astype(float).tolist(),
        "privileged.pick_place.object_rotation_relative_to_drawer": obj_rot_drawer.astype(
            float
        )
        .reshape(-1)
        .tolist(),
        "privileged.pick_place.object_position_relative_to_drawer": drawer_rel,
        "privileged.pick_place.drawer_interior_regions_world": drawer_regions,
        "privileged.pick_place.object_inside_drawer": inside_drawer,
        "privileged.pick_place.object_drawer_contact": object_drawer_contact,
        "privileged.pick_place.object_position_relative_to_counter": counter_rel,
        "privileged.pick_place.object_rotation_relative_to_counter": (
            obj_rot_counter.astype(float).reshape(-1).tolist()
            if obj_rot_counter is not None
            else None
        ),
        "privileged.pick_place.object_counter_contact": object_counter_contact,
        "privileged.pick_place.object_any_counter_contact": on_any_counter,
        "privileged.pick_place.gripper_object_distance": (
            float(np.linalg.norm(eef_pos - obj_pos)) if eef_pos is not None else None
        ),
        "privileged.pick_place.gripper_object_far": gripper_far,
        "privileged.pick_place.gripper_object_contact": gripper_object_contact,
        "privileged.pick_place.object_grasped": grasped,
        "privileged.pick_place.contact_pairs": contact_pairs,
        "privileged.pick_place.object_contact_pairs": object_contact_pairs,
        "privileged.pick_place.robot_contact_pairs": robot_contact_pairs,
        "privileged.pick_place.robot_non_target_contact": robot_non_target_contact,
        "privileged.pick_place.success_predicate": bool(on_any_counter and gripper_far),
    }
    return state


def _pick_place_toaster_state(
    task_env: Any, observation: Mapping[str, Any]
) -> dict[str, Any]:
    """Record simulator truth for PickPlaceToasterToCounter.

    The authoritative task predicate remains RoboCasa's own sparse reward.  This
    function only exposes read-only task-relative state for audit, milestones,
    contact/collision summaries, and later nominal-trajectory indexing.
    """

    simulator = getattr(task_env, "sim", None)
    objects = getattr(task_env, "objects", None)
    toaster = getattr(task_env, "toaster", None)
    counter = getattr(task_env, "counter", None)
    if (
        simulator is None
        or not isinstance(objects, Mapping)
        or toaster is None
        or counter is None
    ):
        return {}
    obj = objects.get("obj")
    plate = objects.get("plate")
    body_ids = getattr(task_env, "obj_body_id", None)
    if obj is None or plate is None or not isinstance(body_ids, Mapping):
        return {}

    import robocasa.utils.object_utils as OU

    model = simulator.model
    data = simulator.data
    obj_body_id = int(body_ids["obj"])
    plate_body_id = int(body_ids["plate"])
    obj_pos = np.asarray(data.body_xpos[obj_body_id], dtype=np.float64)
    obj_quat = np.asarray(data.body_xquat[obj_body_id], dtype=np.float64)
    obj_mat = np.asarray(data.body_xmat[obj_body_id], dtype=np.float64).reshape(3, 3)
    plate_pos = np.asarray(data.body_xpos[plate_body_id], dtype=np.float64)
    plate_quat = np.asarray(data.body_xquat[plate_body_id], dtype=np.float64)
    plate_mat = np.asarray(data.body_xmat[plate_body_id], dtype=np.float64).reshape(
        3, 3
    )
    robot = tuple(getattr(task_env, "robots", ()) or ())
    eef_pos = None
    if robot:
        try:
            eef_id = int(robot[0].eef_site_id["right"])
            eef_pos = np.asarray(data.site_xpos[eef_id], dtype=np.float64)
        except (KeyError, IndexError, TypeError):
            eef_pos = None

    toaster_slot_contact = bool(toaster.check_slot_contact(task_env, "obj"))
    object_toaster_contact = bool(task_env.check_contact(obj, toaster))
    object_plate_contact = bool(task_env.check_contact(obj, plate))
    object_on_plate = bool(OU.check_obj_in_receptacle(task_env, "obj", "plate"))
    on_any_counter = bool(OU.check_obj_any_counter_contact(task_env, "obj"))
    gripper_far = bool(OU.gripper_obj_far(task_env, "obj"))
    grasped = bool(OU.check_obj_grasped(task_env, "obj"))
    gripper_object_contact = False
    if robot:
        try:
            gripper = robot[0].gripper["right"]
            gripper_object_contact = bool(task_env.check_contact(gripper, obj))
        except (KeyError, TypeError):
            pass

    object_geom_ids = {
        geom_id
        for geom_id in range(int(model.ngeom))
        if _descends_from(model, int(model.geom_bodyid[geom_id]), obj_body_id)
    }
    plate_geom_ids = {
        geom_id
        for geom_id in range(int(model.ngeom))
        if _descends_from(model, int(model.geom_bodyid[geom_id]), plate_body_id)
    }
    toaster_geom_ids = _fixture_geom_ids(model, toaster)
    counter_geom_ids = _fixture_geom_ids(model, counter)
    robot_geom_ids = _robot_collision_geoms(task_env, model)
    target_geom_ids = (
        object_geom_ids | plate_geom_ids | toaster_geom_ids | counter_geom_ids
    )
    contact_pairs: list[list[str]] = []
    object_contact_pairs: list[list[str]] = []
    robot_contact_pairs: list[list[str]] = []
    robot_non_target_contact = False
    for index in range(int(data.ncon)):
        contact = data.contact[index]
        first, second = int(contact.geom1), int(contact.geom2)
        pair = [_id_to_name(model, "geom", first), _id_to_name(model, "geom", second)]
        contact_pairs.append(pair)
        if first in object_geom_ids or second in object_geom_ids:
            object_contact_pairs.append(pair)
        if first in robot_geom_ids or second in robot_geom_ids:
            robot_contact_pairs.append(pair)
            other = second if first in robot_geom_ids else first
            if other not in target_geom_ids:
                robot_non_target_contact = True

    toaster_rel = _relative_to_fixture(obj_pos, toaster)
    counter_rel = _relative_to_fixture(obj_pos, counter)
    toaster_rot = _fixture_rotation(toaster)
    counter_rot = _fixture_rotation(counter)
    obj_rot_toaster = toaster_rot.T @ obj_mat
    obj_rot_counter = counter_rot.T @ obj_mat
    object_rel_plate = plate_mat.T @ (obj_pos - plate_pos)
    obj_rot_plate = plate_mat.T @ obj_mat
    toaster_regions = {}
    try:
        toaster_regions = {
            str(name): {
                str(key): np.asarray(value, dtype=np.float64).astype(float).tolist()
                if isinstance(value, (tuple, list, np.ndarray))
                else value
                for key, value in region.items()
            }
            for name, region in toaster.get_reset_regions(env=task_env).items()
        }
    except Exception:
        toaster_regions = {}

    return {
        "privileged.source": "live_mujoco_simulator",
        "privileged.class": "simulator_ground_truth",
        "privileged.pick_place.task": type(
            getattr(task_env, "unwrapped", task_env)
        ).__name__,
        "privileged.pick_place.instruction": _instruction(observation),
        "privileged.pick_place.object_position_world": obj_pos.astype(float).tolist(),
        "privileged.pick_place.object_quaternion_wxyz": obj_quat.astype(float).tolist(),
        "privileged.pick_place.object_position_relative_to_toaster": toaster_rel,
        "privileged.pick_place.object_rotation_relative_to_toaster": (
            obj_rot_toaster.astype(float).reshape(-1).tolist()
        ),
        "privileged.pick_place.toaster_reset_regions": toaster_regions,
        "privileged.pick_place.object_toaster_contact": object_toaster_contact,
        "privileged.pick_place.object_toaster_slot_contact": toaster_slot_contact,
        "privileged.pick_place.plate_position_world": plate_pos.astype(float).tolist(),
        "privileged.pick_place.plate_quaternion_wxyz": plate_quat.astype(float).tolist(),
        "privileged.pick_place.object_position_relative_to_plate": (
            object_rel_plate.astype(float).tolist()
        ),
        "privileged.pick_place.object_rotation_relative_to_plate": (
            obj_rot_plate.astype(float).reshape(-1).tolist()
        ),
        "privileged.pick_place.object_plate_contact": object_plate_contact,
        "privileged.pick_place.object_on_plate": object_on_plate,
        "privileged.pick_place.object_position_relative_to_counter": counter_rel,
        "privileged.pick_place.object_rotation_relative_to_counter": (
            obj_rot_counter.astype(float).reshape(-1).tolist()
        ),
        "privileged.pick_place.object_any_counter_contact": on_any_counter,
        "privileged.pick_place.gripper_object_distance": (
            float(np.linalg.norm(eef_pos - obj_pos)) if eef_pos is not None else None
        ),
        "privileged.pick_place.gripper_object_far": gripper_far,
        "privileged.pick_place.gripper_object_contact": gripper_object_contact,
        "privileged.pick_place.object_grasped": grasped,
        "privileged.pick_place.contact_pairs": contact_pairs,
        "privileged.pick_place.object_contact_pairs": object_contact_pairs,
        "privileged.pick_place.robot_contact_pairs": robot_contact_pairs,
        "privileged.pick_place.robot_non_target_contact": robot_non_target_contact,
        "privileged.pick_place.success_predicate": bool(
            object_on_plate and gripper_far
        ),
    }


def extract_privileged_state(
    environment: Any, observation: Mapping[str, Any]
) -> dict[str, Any]:
    """Return task-relevant simulator truth without changing the environment."""

    task_env = _task_environment(environment)
    task_name = type(task_env).__name__.lower()
    if "slidedishwasherrack" in task_name or hasattr(task_env, "dishwasher"):
        return _slide_dishwasher_state(task_env, observation)
    instruction = " ".join(_instruction(observation).lower().split())
    if (
        "pickplacetoastertocounter" in task_name
        or (
            hasattr(task_env, "toaster")
            and hasattr(task_env, "counter")
            and "toasted item" in instruction
            and "plate" in instruction
        )
    ):
        return _pick_place_toaster_state(task_env, observation)
    if (
        "pickplacedrawertocounter" in task_name
        or (
            hasattr(task_env, "drawer")
            and hasattr(task_env, "counter")
            and "from the drawer" in instruction
            and "on the counter" in instruction
        )
    ):
        return _pick_place_state(task_env, observation)
    return {
        "privileged.source": "live_mujoco_simulator",
        "privileged.class": "simulator_ground_truth",
    }
