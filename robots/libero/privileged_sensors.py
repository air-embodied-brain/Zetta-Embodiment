# Copyright (c) 2026 RPent Contributors
"""Privileged LIBERO contact sensing with a real-robot analogue.

The simulator implementation reads MuJoCo contact pairs and, when available,
their contact-frame wrench.  This intentionally uses simulator privilege: on a
real robot the same tool contract can be backed by wrist force/torque, joint
torque, or tactile sensors.  It observes the current state only and is not a
trajectory-level collision certificate.
"""

from __future__ import annotations

import os
import re
import types
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np

_FEATURE_TOKEN = re.compile(r"[^a-z0-9]+")
_LOGICAL_PREDICATES = {"and", "or", "not"}


def _feature_token(value: str) -> str:
    token = _FEATURE_TOKEN.sub("_", value.casefold()).strip("_")
    return token or "unnamed"


def _unwrap_robosuite_env(env: Any) -> Any:
    current = env
    seen: set[int] = set()
    while hasattr(current, "env") and id(current) not in seen:
        seen.add(id(current))
        current = current.env
    return current


def _id_to_name(model: Any, kind: str, index: int) -> str:
    method = getattr(model, f"{kind}_id2name", None)
    if callable(method):
        value = method(int(index))
        if value is not None:
            return str(value)
    native_model = getattr(model, "_model", model)
    try:
        import mujoco

        object_type = {
            "geom": mujoco.mjtObj.mjOBJ_GEOM,
            "body": mujoco.mjtObj.mjOBJ_BODY,
            "joint": mujoco.mjtObj.mjOBJ_JOINT,
            "site": mujoco.mjtObj.mjOBJ_SITE,
            "sensor": mujoco.mjtObj.mjOBJ_SENSOR,
        }[kind]
        value = mujoco.mj_id2name(native_model, object_type, int(index))
        if value is not None:
            return str(value)
    except Exception:
        pass
    return f"{kind}_{int(index)}"


def _flatten_names(value: Any) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, str):
        return {value}
    if isinstance(value, dict):
        result: set[str] = set()
        for item in value.values():
            result.update(_flatten_names(item))
        return result
    try:
        result = set()
        for item in value:
            result.update(_flatten_names(item))
        return result
    except TypeError:
        return set()


def _robot_geom_names(raw_env: Any, model: Any) -> set[str]:
    names: set[str] = set()
    for robot in getattr(raw_env, "robots", []) or []:
        for owner in (
            robot,
            getattr(robot, "robot_model", None),
            getattr(robot, "gripper", None),
        ):
            if owner is None:
                continue
            for attribute in ("contact_geoms", "visual_geoms", "collision_geoms"):
                names.update(_flatten_names(getattr(owner, attribute, None)))

    # Robosuite versions expose slightly different contact-geom attributes.
    # Prefix fallback is restricted to canonical Panda/robosuite robot names.
    ngeom = int(getattr(model, "ngeom", 0))
    for geom_id in range(ngeom):
        name = _id_to_name(model, "geom", geom_id)
        lowered = name.lower()
        if (
            lowered.startswith(("robot0_", "gripper0_", "panda_"))
            or "finger" in lowered
            or "hand_collision" in lowered
        ):
            names.add(name)
    return names


def _contact_wrench(simulator: Any, contact_index: int) -> list[float] | None:
    native_model = getattr(simulator.model, "_model", simulator.model)
    native_data = getattr(simulator.data, "_data", simulator.data)
    wrench = np.zeros(6, dtype=np.float64)
    try:
        import mujoco

        mujoco.mj_contactForce(native_model, native_data, int(contact_index), wrench)
        return wrench.astype(float).tolist()
    except Exception:
        pass
    try:
        from mujoco_py import functions

        functions.mj_contactForce(native_model, native_data, int(contact_index), wrench)
        return wrench.astype(float).tolist()
    except Exception:
        return None


def collect_privileged_contacts(
    outer_env: Any,
    *,
    include_all_contacts: bool = False,
    max_contacts: int = 64,
) -> dict[str, Any]:
    """Return current MuJoCo contact evidence without advancing the simulator."""
    raw_env = _unwrap_robosuite_env(outer_env)
    simulator = getattr(raw_env, "sim", None)
    if simulator is None:
        return {
            "available": False,
            "status": "unavailable",
            "reason": "LIBERO robosuite environment exposes no MuJoCo simulator",
        }
    model = simulator.model
    data = simulator.data
    robot_names = _robot_geom_names(raw_env, model)
    limit = max(1, min(int(max_contacts), 256))
    contacts: list[dict[str, Any]] = []
    robot_contact_count = 0
    force_available = False
    total = int(getattr(data, "ncon", 0))
    for contact_index in range(total):
        contact = data.contact[contact_index]
        geom1_id = int(contact.geom1)
        geom2_id = int(contact.geom2)
        geom1 = _id_to_name(model, "geom", geom1_id)
        geom2 = _id_to_name(model, "geom", geom2_id)
        geom1_robot = geom1 in robot_names
        geom2_robot = geom2 in robot_names
        involves_robot = geom1_robot or geom2_robot
        if involves_robot:
            robot_contact_count += 1
        if not include_all_contacts and not involves_robot:
            continue
        wrench = _contact_wrench(simulator, contact_index)
        force_available |= wrench is not None
        frame = np.asarray(getattr(contact, "frame", []), dtype=np.float64).reshape(-1)
        normal = frame[:3].astype(float).tolist() if frame.size >= 3 else None
        contacts.append(
            {
                "contact_index": contact_index,
                "geom1": geom1,
                "geom2": geom2,
                "geom1_id": geom1_id,
                "geom2_id": geom2_id,
                "geom1_robot": geom1_robot,
                "geom2_robot": geom2_robot,
                "involves_robot": involves_robot,
                "robot_self_contact": geom1_robot and geom2_robot,
                "distance_m": float(getattr(contact, "dist", 0.0)),
                "position_world": np.asarray(contact.pos, dtype=np.float64)
                .astype(float)
                .tolist(),
                "normal_world": normal,
                "wrench_contact_frame": wrench,
                "normal_force_n": float(abs(wrench[0])) if wrench is not None else None,
            }
        )
        if len(contacts) >= limit:
            break
    return {
        "available": True,
        "status": "ok",
        "source": "privileged_mujoco_contact_proxy",
        "real_world_analogue": [
            "wrist_force_torque",
            "joint_torque_residual",
            "tactile_contact",
        ],
        "current_state_only": True,
        "trajectory_collision_certificate": False,
        "total_contact_count": total,
        "robot_contact_count": robot_contact_count,
        "returned_contact_count": len(contacts),
        "truncated": len(contacts) >= limit and robot_contact_count > len(contacts),
        "robot_geom_count": len(robot_names),
        "force_available": force_available,
        "contacts": contacts,
    }


def _joint_identifier(model: Any, entity: str, joint: str) -> tuple[int, str]:
    """Resolve a semantic joint name without exposing model coordinates."""

    names = _model_name_ids(model, "joint")
    requested = str(joint).strip()
    entity_name = str(entity).strip()
    candidates = [requested]
    if entity_name:
        candidates.extend((f"{entity_name}_{requested}", f"{entity_name}{requested}"))
    for candidate in candidates:
        if candidate in names:
            return int(names[candidate]), candidate
    token = _feature_token(requested)
    entity_token = _feature_token(entity_name)
    matches = [
        (name, identifier)
        for name, identifier in names.items()
        if _feature_token(name).endswith(token)
        and (not entity_token or entity_token in _feature_token(name))
    ]
    if len(matches) == 1:
        name, identifier = matches[0]
        return int(identifier), name
    raise KeyError(f"semantic joint is not uniquely resolvable: {entity}/{joint}")


def _geom_vertical_extent(
    model: Any,
    data: Any,
    geom_id: int,
    size: np.ndarray,
) -> float:
    """Return a collision geom's world-z half extent.

    ``geom_size`` is expressed in the geom frame. Taking its Euclidean norm,
    as the old planner did, substantially overestimates rotated boxes and can
    leave the gripper several centimetres above a small fixture control.
    """

    rotations = getattr(data, "geom_xmat", None)
    if rotations is None:
        rotation = np.eye(3, dtype=np.float64)
    else:
        try:
            rotation = np.asarray(rotations[geom_id], dtype=np.float64).reshape(3, 3)
        except (IndexError, ValueError):
            rotation = np.eye(3, dtype=np.float64)
    geom_types = np.asarray(getattr(model, "geom_type", ())).reshape(-1)
    geom_type = int(geom_types[geom_id]) if geom_id < geom_types.size else 6
    values = np.pad(np.asarray(size, dtype=np.float64).reshape(-1), (0, 3))[:3]
    if geom_type == 2:  # sphere
        return float(abs(values[0]))
    if geom_type in {3, 5}:  # capsule / cylinder, local axis is z
        radial = float(abs(values[0]))
        half_length = float(abs(values[1]))
        axis_z = float(abs(rotation[2, 2]))
        radial_z = float(np.linalg.norm(rotation[2, :2]))
        return radial * radial_z + half_length * axis_z + (
            radial * axis_z if geom_type == 3 else 0.0
        )
    if geom_type == 4:  # ellipsoid
        return float(np.linalg.norm(values * rotation[2, :]))
    if geom_type == 6:  # box
        return float(np.dot(np.abs(rotation[2, :]), np.abs(values)))
    bounds = np.asarray(getattr(model, "geom_rbound", ())).reshape(-1)
    if geom_id < bounds.size and np.isfinite(bounds[geom_id]):
        return float(max(0.0, bounds[geom_id]))
    return float(np.linalg.norm(values))


def _geom_axis_extent(
    model: Any,
    data: Any,
    geom_id: int,
    size: np.ndarray,
    axis: np.ndarray,
) -> float:
    """Return a conservative half extent of a geom along a world axis."""

    rotations = getattr(data, "geom_xmat", None)
    if rotations is None:
        rotation = np.eye(3, dtype=np.float64)
    else:
        try:
            rotation = np.asarray(rotations[geom_id], dtype=np.float64).reshape(3, 3)
        except (IndexError, ValueError):
            rotation = np.eye(3, dtype=np.float64)
    axis = np.asarray(axis, dtype=np.float64).reshape(3)
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm < 1e-8 or not np.isfinite(axis_norm):
        return 0.0
    axis = axis / axis_norm
    values = np.pad(
        np.abs(np.asarray(size, dtype=np.float64).reshape(-1)), (0, 3)
    )[:3]
    geom_types = np.asarray(getattr(model, "geom_type", ())).reshape(-1)
    geom_type = int(geom_types[geom_id]) if geom_id < geom_types.size else 6
    if geom_type == 2:  # sphere
        return float(values[0])
    if geom_type == 6:  # box
        return float(np.dot(np.abs(rotation.T @ axis), values))
    if geom_type in {3, 5}:  # capsule / cylinder, local axis is z
        radial = float(values[0])
        half_length = float(values[1])
        axial = abs(float(np.dot(rotation[:, 2], axis)))
        return radial + half_length * axial
    if geom_type == 4:  # ellipsoid
        return float(np.linalg.norm(values * (rotation.T @ axis)))
    bounds = np.asarray(getattr(model, "geom_rbound", ())).reshape(-1)
    if geom_id < bounds.size and np.isfinite(bounds[geom_id]):
        return float(max(0.0, bounds[geom_id]))
    return float(np.linalg.norm(values))


def collect_privileged_semantic_joint_plan(
    outer_env: Any,
    *,
    entity: str,
    joint: str,
    direction: str,
) -> dict[str, Any]:
    """Build a bounded contact plan from audited simulator geometry.

    The returned world-frame points are an internal primitive input. They must
    never be copied into VLA observations, Role1 evidence, or tool results.
    This function only reads MuJoCo state; it does not mutate qpos or advance
    the environment.
    """

    direction = str(direction).strip().casefold()
    if direction not in {"lower", "upper"}:
        raise ValueError("direction must be 'lower' or 'upper'")
    raw_env = _unwrap_robosuite_env(outer_env)
    simulator = getattr(raw_env, "sim", None)
    if simulator is None:
        raise RuntimeError("privileged simulator is unavailable")
    model = simulator.model
    data = simulator.data
    entity_name = str(entity).strip()
    joint_id, joint_name = _joint_identifier(model, entity_name, joint)
    qpos_address = int(np.asarray(model.jnt_qposadr).reshape(-1)[joint_id])
    qpos = float(np.asarray(data.qpos).reshape(-1)[qpos_address])
    dof_addresses = np.asarray(getattr(model, "jnt_dofadr", ())).reshape(-1)
    qvel_values = np.asarray(getattr(data, "qvel", ()), dtype=np.float64).reshape(-1)
    qvel = (
        float(qvel_values[int(dof_addresses[joint_id])])
        if joint_id < dof_addresses.size
        and int(dof_addresses[joint_id]) < qvel_values.size
        else 0.0
    )
    joint_range = np.asarray(model.jnt_range[joint_id], dtype=np.float64).reshape(-1)
    if joint_range.size != 2 or not np.isfinite(joint_range).all():
        raise ValueError(f"joint {joint_name} has no finite range")
    lower, upper = float(joint_range[0]), float(joint_range[1])
    tolerance = max(0.0005, min(0.002, abs(upper - lower) * 0.005))
    goal_satisfied = qpos <= lower + tolerance if direction == "lower" else qpos >= upper - tolerance

    body_id = int(np.asarray(model.jnt_bodyid).reshape(-1)[joint_id])
    body_positions = getattr(data, "xpos", getattr(data, "body_xpos", None))
    body_rotations = getattr(data, "xmat", None)
    if body_positions is None or body_rotations is None:
        raise RuntimeError("simulator does not expose joint body pose")
    body_position = np.asarray(body_positions[body_id], dtype=np.float64).reshape(3)
    body_rotation = np.asarray(body_rotations[body_id], dtype=np.float64).reshape(3, 3)
    joint_positions = np.asarray(
        getattr(model, "jnt_pos", np.zeros((joint_id + 1, 3))),
        dtype=np.float64,
    )
    local_joint_position = (
        joint_positions[joint_id].reshape(3)
        if joint_id < len(joint_positions)
        else np.zeros(3, dtype=np.float64)
    )
    joint_position = body_position + body_rotation @ local_joint_position
    local_axis = np.asarray(model.jnt_axis[joint_id], dtype=np.float64).reshape(3)
    axis = body_rotation @ local_axis
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm < 1e-8 or not np.isfinite(axis_norm):
        raise RuntimeError("joint axis is invalid")
    axis /= axis_norm

    # Use the joint body's collision geometry to estimate a stable contact
    # radius and the top surface. This remains valid across randomized poses.
    geom_body_ids = np.asarray(getattr(model, "geom_bodyid", ())).reshape(-1)
    geom_positions = np.asarray(getattr(data, "geom_xpos", ()), dtype=np.float64)
    geom_sizes = np.asarray(getattr(model, "geom_size", ()), dtype=np.float64)
    geom_contype = np.asarray(getattr(model, "geom_contype", ())).reshape(-1)
    geom_conaffinity = np.asarray(getattr(model, "geom_conaffinity", ())).reshape(-1)
    points: list[np.ndarray] = []
    collision_geoms: list[tuple[np.ndarray, float, str]] = []
    top = float(joint_position[2])
    for geom_id, owner_body in enumerate(geom_body_ids):
        if not _descends_from(model, int(owner_body), body_id):
            continue
        if (
            geom_id < geom_contype.size
            and geom_id < geom_conaffinity.size
            and int(geom_contype[geom_id]) == 0
            and int(geom_conaffinity[geom_id]) == 0
        ):
            continue
        try:
            point = np.asarray(geom_positions[geom_id], dtype=np.float64).reshape(3)
            size = np.asarray(geom_sizes[geom_id], dtype=np.float64).reshape(-1)
        except (IndexError, ValueError):
            continue
        if not np.isfinite(point).all():
            continue
        points.append(point)
        collision_geoms.append(
            (
                point,
                _geom_axis_extent(model, data, geom_id, size, axis),
                _id_to_name(model, "geom", geom_id),
            )
        )
        extent = _geom_vertical_extent(model, data, geom_id, size)
        top = max(top, float(point[2]) + extent)
    if not points:
        points = [joint_position.copy()]

    joint_types = np.asarray(getattr(model, "jnt_type", ())).reshape(-1)
    joint_type = int(joint_types[joint_id]) if joint_id < joint_types.size else 3
    is_slide = joint_type == 2
    eef_position = _eef_position(raw_env, simulator)
    gripper_contact_surfaces: list[tuple[float, np.ndarray]] = []
    motion_axis = axis * (-1.0 if direction == "lower" else 1.0)
    if eef_position is not None:
        for robot in getattr(raw_env, "robots", ()):
            gripper = getattr(robot, "gripper", None)
            for geom_name in getattr(gripper, "contact_geoms", ()):
                try:
                    geom_id = _name_to_id(model, "geom", str(geom_name))
                    geom_position = np.asarray(
                        geom_positions[geom_id], dtype=np.float64
                    ).reshape(3)
                    geom_size = np.asarray(
                        geom_sizes[geom_id], dtype=np.float64
                    ).reshape(-1)
                except (KeyError, IndexError, TypeError, ValueError):
                    continue
                offset = geom_position - eef_position
                if np.isfinite(offset).all():
                    if is_slide:
                        extent = _geom_axis_extent(
                            model, data, geom_id, geom_size, motion_axis
                        )
                        surface_offset = offset - motion_axis * extent
                        gripper_contact_surfaces.append(
                            (float(np.dot(surface_offset, motion_axis)), surface_offset)
                        )
    if is_slide and collision_geoms:
        # A slide joint moves along its axis. Select the outermost collision
        # geometry in that direction and prefer a named handle when surfaces
        # are nearly tied. The radial construction below is hinge-specific.
        scored = [
            (
                float(np.dot(point - joint_position, motion_axis) + extent),
                "handle" in _feature_token(name),
                point,
                extent,
            )
            for point, extent, name in collision_geoms
        ]
        best_score = max(item[0] for item in scored)
        near = [item for item in scored if item[0] >= best_score - 0.015]
        _, _, selected, selected_extent = max(
            near, key=lambda item: (item[1], item[0])
        )
        handle_surface = selected.copy() + motion_axis * selected_extent
        if gripper_contact_surfaces:
            fingertip_surfaces = [
                item
                for item in gripper_contact_surfaces
                if abs(float(item[1][2])) <= 0.05
            ]
            _, gripper_surface_offset = min(
                fingertip_surfaces or gripper_contact_surfaces,
                key=lambda item: item[0],
            )
            # Return the EEF pose that places the rear finger surface on the
            # drawer handle.  The EEF site is behind the finger collision
            # geometry by roughly the finger length in this frame; targeting
            # the handle centre directly therefore stalls before contact.
            contact = handle_surface - gripper_surface_offset
        else:
            contact = selected.copy() + motion_axis * min(0.004, selected_extent * 0.25)
        press_z = float(contact[2])
        radius = float(selected_extent)
        tangent = motion_axis
    else:
        eef = _eef_position(raw_env, simulator)
        up = np.asarray((0.0, 0.0, 1.0), dtype=np.float64)
        radial = (
            (eef - joint_position)
            if eef is not None
            else np.asarray((1.0, 0.0, 0.0))
        )
        radial -= axis * float(np.dot(radial, axis))
        radial_norm = float(np.linalg.norm(radial))
        if radial_norm < 1e-8:
            radial = up - axis * float(np.dot(up, axis))
            radial_norm = float(np.linalg.norm(radial))
        if radial_norm < 1e-8:
            radial = np.asarray((1.0, 0.0, 0.0))
            radial_norm = 1.0
        radial /= radial_norm
        radii = []
        for point in points:
            offset = point - joint_position
            radii.append(
                float(np.linalg.norm(offset - axis * float(np.dot(offset, axis))))
            )
        radius = float(np.clip(max(radii or [0.025]) * 0.65, 0.015, 0.06))
        contact = joint_position + radial * radius
        press_z = max(float(joint_position[2]) + 0.018, top - 0.012)
        tangent = np.cross(axis, radial)
        if direction == "lower":
            tangent *= -1.0
    approach = contact.copy()
    approach[2] = press_z + 0.065
    press = contact.copy()
    press[2] = press_z
    tangent_norm = float(np.linalg.norm(tangent))
    if tangent_norm < 1e-8:
        tangent = np.asarray((1.0, 0.0, 0.0))
        tangent_norm = 1.0
    tangent /= tangent_norm
    return {
        "available": True,
        "entity": entity_name,
        "joint": joint_name,
        "qpos": qpos,
        "qvel": qvel,
        "range_lower": lower,
        "range_upper": upper,
        "goal_satisfied": bool(goal_satisfied),
        "joint_position_world": joint_position.astype(float).tolist(),
        "approach_position_world": approach.astype(float).tolist(),
        "press_position_world": press.astype(float).tolist(),
        "tangent_direction_world": tangent.astype(float).tolist(),
        "contact_radius_m": radius,
        "joint_type": "slide" if is_slide else "hinge",
    }


def _name_to_id(model: Any, kind: str, name: str) -> int:
    modern = getattr(model, kind, None)
    if callable(modern):
        try:
            return int(modern(name).id)
        except Exception:
            pass
    legacy = getattr(model, f"{kind}_name2id", None)
    if callable(legacy):
        try:
            return int(legacy(name))
        except Exception:
            pass
    raise KeyError(f"cannot resolve MuJoCo {kind} {name!r}")


def _descends_from(model: Any, body_id: int, ancestor_id: int) -> bool:
    parents = getattr(model, "body_parentid", None)
    if parents is None:
        return body_id == ancestor_id
    current = int(body_id)
    seen: set[int] = set()
    while current >= 0 and current not in seen:
        if current == ancestor_id:
            return True
        seen.add(current)
        parent = int(parents[current])
        if parent == current:
            break
        current = parent
    return False


def _position_and_quaternion(
    simulator: Any,
    *,
    kind: str,
    identifier: int,
) -> tuple[np.ndarray, np.ndarray | None]:
    data = simulator.data
    if kind == "body":
        positions = getattr(data, "body_xpos", getattr(data, "xpos", None))
        quaternions = getattr(data, "body_xquat", getattr(data, "xquat", None))
    elif kind == "site":
        positions = getattr(data, "site_xpos", None)
        quaternions = None
    else:
        positions = getattr(data, "geom_xpos", None)
        quaternions = None
    if positions is None:
        raise KeyError(f"MuJoCo data has no {kind} positions")
    position = np.asarray(positions[int(identifier)], dtype=np.float64).reshape(-1)
    if position.size != 3 or not np.isfinite(position).all():
        raise ValueError(f"invalid {kind} position")
    quaternion = None
    if quaternions is not None:
        candidate = np.asarray(
            quaternions[int(identifier)], dtype=np.float64
        ).reshape(-1)
        if candidate.size == 4 and np.isfinite(candidate).all():
            quaternion = candidate
    return position, quaternion


def _owner_name(owner: Any, fallback: str | None = None) -> str | None:
    for value in (fallback, getattr(owner, "name", None)):
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _owner_root_body(owner: Any) -> str | None:
    for attribute in ("root_body", "root_body_name"):
        value = getattr(owner, attribute, None)
        if isinstance(value, str) and value:
            return value
    return None


def _owner_contact_geoms(owner: Any) -> set[str]:
    names: set[str] = set()
    for attribute in ("contact_geoms", "collision_geoms"):
        names.update(_flatten_names(getattr(owner, attribute, None)))
    return names


def _entity_owners(raw_env: Any) -> dict[str, Any]:
    owners: dict[str, Any] = {}
    for attribute in ("objects_dict", "fixtures_dict"):
        value = getattr(raw_env, attribute, None)
        if isinstance(value, Mapping):
            for name, owner in value.items():
                resolved = _owner_name(owner, str(name))
                if resolved:
                    owners.setdefault(resolved, owner)
    for attribute in ("objects", "fixtures", "object_sites_dict"):
        value = getattr(raw_env, attribute, None)
        if isinstance(value, Mapping):
            items = value.items()
        else:
            try:
                items = ((None, owner) for owner in (value or ()))
            except TypeError:
                continue
        for fallback, owner in items:
            resolved = _owner_name(owner, str(fallback) if fallback else None)
            if resolved:
                owners.setdefault(resolved, owner)
    return owners


def _model_name_ids(model: Any, kind: str) -> dict[str, int]:
    count_attribute = "njnt" if kind == "joint" else f"n{kind}"
    count = int(getattr(model, count_attribute, 0))
    return {
        name: identifier
        for identifier in range(count)
        if (name := _id_to_name(model, kind, identifier))
    }


def _resolve_entity_record(
    simulator: Any,
    *,
    name: str,
    owner: Any | None,
) -> dict[str, Any] | None:
    model = simulator.model
    candidates = []
    root_body = _owner_root_body(owner) if owner is not None else None
    if root_body:
        candidates.append(("body", root_body))
    candidates.extend((kind, name) for kind in ("body", "site", "geom"))
    for kind, candidate in candidates:
        try:
            identifier = _name_to_id(model, kind, candidate)
            position, quaternion = _position_and_quaternion(
                simulator, kind=kind, identifier=identifier
            )
            break
        except (KeyError, ValueError, IndexError):
            continue
    else:
        normalized = _feature_token(name)
        identifier = -1
        kind = ""
        position = np.asarray([], dtype=np.float64)
        quaternion = None
        for candidate_kind in ("body", "site", "geom"):
            for candidate_name, candidate_id in _model_name_ids(
                model, candidate_kind
            ).items():
                candidate_token = _feature_token(candidate_name)
                if candidate_token == normalized or candidate_token.startswith(
                    f"{normalized}_"
                ):
                    try:
                        position, quaternion = _position_and_quaternion(
                            simulator,
                            kind=candidate_kind,
                            identifier=candidate_id,
                        )
                    except (KeyError, ValueError, IndexError):
                        continue
                    identifier = candidate_id
                    kind = candidate_kind
                    break
            if identifier >= 0:
                break
        if identifier < 0:
            return None

    geom_ids: set[int] = set()
    explicit_geoms = _owner_contact_geoms(owner) if owner is not None else set()
    for geom_name in explicit_geoms:
        try:
            geom_ids.add(_name_to_id(model, "geom", geom_name))
        except KeyError:
            continue
    if kind == "body":
        body_id = int(identifier)
        geom_body_ids = getattr(model, "geom_bodyid", None)
        if geom_body_ids is not None:
            for geom_id in range(int(getattr(model, "ngeom", 0))):
                if _descends_from(model, int(geom_body_ids[geom_id]), body_id):
                    geom_ids.add(geom_id)
    elif kind == "geom":
        geom_ids.add(int(identifier))
    return {
        "name": name,
        "owner": owner,
        "kind": kind,
        "identifier": int(identifier),
        "position": position,
        "quaternion": quaternion,
        "geom_ids": geom_ids,
        "geom_names": {
            _id_to_name(model, "geom", geom_id) for geom_id in geom_ids
        },
    }


def _goal_atoms(value: Any) -> list[tuple[str, ...]]:
    atoms: list[tuple[str, ...]] = []
    if isinstance(value, Mapping):
        predicate = value.get("predicate") or value.get("name")
        arguments = value.get("arguments") or value.get("args")
        if isinstance(predicate, str) and isinstance(arguments, (list, tuple)):
            atoms.append((predicate, *(str(item) for item in arguments)))
        else:
            for item in value.values():
                atoms.extend(_goal_atoms(item))
        return atoms
    if not isinstance(value, (list, tuple)) or not value:
        return atoms
    first = value[0]
    if isinstance(first, str):
        predicate = first.casefold()
        if predicate in _LOGICAL_PREDICATES:
            for item in value[1:]:
                atoms.extend(_goal_atoms(item))
        elif all(not isinstance(item, (list, tuple, Mapping)) for item in value[1:]):
            atoms.append(tuple(str(item) for item in value))
        else:
            for item in value[1:]:
                atoms.extend(_goal_atoms(item))
        return atoms
    for item in value:
        atoms.extend(_goal_atoms(item))
    return atoms


def _goal_state(raw_env: Any) -> Any:
    for owner in (raw_env, getattr(raw_env, "unwrapped", None)):
        if owner is None:
            continue
        for attribute in ("parsed_problem", "_parsed_problem"):
            problem = getattr(owner, attribute, None)
            if isinstance(problem, Mapping):
                for key in ("goal_state", "goal", "goals"):
                    if key in problem:
                        return problem[key]
    return None


def _predicate_value(raw_env: Any, atom: tuple[str, ...]) -> bool | None:
    evaluator = getattr(raw_env, "_eval_predicate", None)
    if not callable(evaluator):
        return None
    for value in (list(atom), tuple(atom)):
        try:
            return bool(evaluator(value))
        except Exception:
            continue
    return None


def _eef_position(raw_env: Any, simulator: Any) -> np.ndarray | None:
    for owner in (raw_env, *(getattr(raw_env, "robots", ()) or ())):
        for attribute in ("_eef_xpos", "eef_xpos"):
            value = getattr(owner, attribute, None)
            if value is None:
                continue
            candidate = np.asarray(value, dtype=np.float64).reshape(-1)
            if candidate.size == 3 and np.isfinite(candidate).all():
                return candidate
    model = simulator.model
    for name in ("gripper0_grip_site", "robot0_grip_site", "eef_site"):
        try:
            site_id = _name_to_id(model, "site", name)
            position, _ = _position_and_quaternion(
                simulator, kind="site", identifier=site_id
            )
            return position
        except (KeyError, ValueError, IndexError):
            continue
    return None


def _gripper_geom_names(raw_env: Any, model: Any) -> set[str]:
    names: set[str] = set()
    for robot in getattr(raw_env, "robots", ()) or ():
        gripper = getattr(robot, "gripper", None)
        for attribute in ("contact_geoms", "collision_geoms"):
            names.update(_flatten_names(getattr(gripper, attribute, None)))
    for identifier in range(int(getattr(model, "ngeom", 0))):
        name = _id_to_name(model, "geom", identifier)
        if "finger" in name.casefold() or "gripper" in name.casefold():
            names.add(name)
    return names


def _contact_summary(raw_env: Any, entities: Mapping[str, dict[str, Any]]) -> dict[str, Any]:
    simulator = raw_env.sim
    model = simulator.model
    robot_names = _robot_geom_names(raw_env, model)
    gripper_names = _gripper_geom_names(raw_env, model)
    robot_ids = {
        identifier
        for identifier in range(int(getattr(model, "ngeom", 0)))
        if _id_to_name(model, "geom", identifier) in robot_names
    }
    gripper_ids = {
        identifier
        for identifier in range(int(getattr(model, "ngeom", 0)))
        if _id_to_name(model, "geom", identifier) in gripper_names
    }
    pairs: list[tuple[int, int]] = []
    robot_contacts = 0
    gripper_contacts = 0
    max_normal_force = 0.0
    force_available = False
    for contact_index in range(int(getattr(simulator.data, "ncon", 0))):
        contact = simulator.data.contact[contact_index]
        first, second = int(contact.geom1), int(contact.geom2)
        pairs.append((first, second))
        robot_contacts += int(first in robot_ids or second in robot_ids)
        gripper_contacts += int(first in gripper_ids or second in gripper_ids)
        wrench = _contact_wrench(simulator, contact_index)
        if wrench is not None:
            force_available = True
            max_normal_force = max(max_normal_force, float(abs(wrench[0])))

    per_entity: dict[str, dict[str, Any]] = {}
    for name, record in entities.items():
        entity_ids = set(record["geom_ids"])
        robot_contact = False
        gripper_contact_ids: set[int] = set()
        for first, second in pairs:
            if first in entity_ids and second in robot_ids:
                robot_contact = True
            if second in entity_ids and first in robot_ids:
                robot_contact = True
            if first in entity_ids and second in gripper_ids:
                gripper_contact_ids.add(second)
            if second in entity_ids and first in gripper_ids:
                gripper_contact_ids.add(first)
        gripper_contact = bool(gripper_contact_ids)
        grasped: bool | None = None
        checker = getattr(raw_env, "_check_grasp", None)
        robots = tuple(getattr(raw_env, "robots", ()) or ())
        if callable(checker) and robots and record["geom_names"]:
            gripper = getattr(robots[0], "gripper", None)
            try:
                grasped = bool(checker(gripper, sorted(record["geom_names"])))
            except Exception:
                grasped = None
        if grasped is None:
            grasped = len(gripper_contact_ids) >= 2
        per_entity[name] = {
            "robot_contact": robot_contact,
            "gripper_contact": gripper_contact,
            "grasped": grasped,
        }
    return {
        "robot_count": robot_contacts,
        "gripper_count": gripper_contacts,
        "force_available": force_available,
        "max_normal_force_n": max_normal_force,
        "per_entity": per_entity,
    }


def _task_success(raw_env: Any) -> bool | None:
    checker = getattr(raw_env, "_check_success", None)
    if not callable(checker):
        return None
    try:
        return bool(checker())
    except Exception:
        return None


def _write_pose_features(
    result: dict[str, Any], prefix: str, record: Mapping[str, Any]
) -> None:
    position = np.asarray(record["position"], dtype=np.float64)
    for axis, value in zip(("x", "y", "z"), position, strict=True):
        result[f"{prefix}.position.{axis}"] = float(value)
    quaternion = record.get("quaternion")
    if quaternion is not None:
        values = np.asarray(quaternion, dtype=np.float64)
        for axis, value in zip(("w", "x", "y", "z"), values, strict=True):
            result[f"{prefix}.orientation.{axis}"] = float(value)


def _write_joint_features(result: dict[str, Any], simulator: Any) -> None:
    model = simulator.model
    data = simulator.data
    joint_types = getattr(model, "jnt_type", ())
    qpos_addresses = getattr(model, "jnt_qposadr", ())
    ranges = getattr(model, "jnt_range", ())
    qpos = getattr(data, "qpos", ())
    count = 0
    for identifier in range(int(getattr(model, "njnt", 0))):
        # MuJoCo: 2=slide, 3=hinge. Free and ball joints are object poses,
        # already represented by the entity feature plane.
        if int(joint_types[identifier]) not in (2, 3):
            continue
        try:
            position = float(qpos[int(qpos_addresses[identifier])])
            joint_range = np.asarray(ranges[identifier], dtype=np.float64).reshape(-1)
        except (IndexError, TypeError, ValueError):
            continue
        if not np.isfinite(position) or joint_range.size != 2:
            continue
        name = _id_to_name(model, "joint", identifier)
        prefix = f"privileged.joint.{_feature_token(name)}"
        lower, upper = float(joint_range[0]), float(joint_range[1])
        result[f"{prefix}.name"] = name
        result[f"{prefix}.position"] = position
        result[f"{prefix}.range.lower"] = lower
        result[f"{prefix}.range.upper"] = upper
        result[f"{prefix}.distance_to_lower"] = abs(position - lower)
        result[f"{prefix}.distance_to_upper"] = abs(upper - position)
        span = upper - lower
        if np.isfinite(span) and abs(span) > 1e-12:
            result[f"{prefix}.normalized"] = (position - lower) / span
        count += 1
    result["privileged.joint.count"] = count


def collect_privileged_critic_state(outer_env: Any) -> dict[str, Any]:
    """Return audited simulator truth exclusively for the online Critic.

    The returned mapping is read-only and scalar-oriented.  It is not inserted
    into the Actor or VLA observation.  Dynamic ``privileged.entity.*`` names
    preserve the task's declared LIBERO/BDDL identities so a candidate can
    distinguish the intended receptacle from a visually similar distractor.
    """

    raw_env = _unwrap_robosuite_env(outer_env)
    simulator = getattr(raw_env, "sim", None)
    if simulator is None:
        return {
            "privileged.available": False,
            "privileged.task.semantic_available": False,
        }

    owners = _entity_owners(raw_env)
    atoms = _goal_atoms(_goal_state(raw_env))
    goal_names = {
        argument
        for atom in atoms
        for argument in atom[1:]
        if argument and argument.casefold() not in {"true", "false"}
    }
    entities: dict[str, dict[str, Any]] = {}
    for name in dict.fromkeys((*owners.keys(), *sorted(goal_names))):
        record = _resolve_entity_record(
            simulator, name=name, owner=owners.get(name)
        )
        if record is not None:
            entities[name] = record

    eef = _eef_position(raw_env, simulator)
    contacts = _contact_summary(raw_env, entities)
    success = _task_success(raw_env)
    result: dict[str, Any] = {
        "privileged.available": True,
        "privileged.task.semantic_available": bool(atoms and entities),
        "privileged.task.goal.predicate_count": len(atoms),
        "privileged.contact.robot.count": int(contacts["robot_count"]),
        "privileged.contact.gripper.count": int(contacts["gripper_count"]),
        "privileged.contact.force_available": bool(contacts["force_available"]),
        "privileged.contact.max_normal_force_n": float(
            contacts["max_normal_force_n"]
        ),
    }
    _write_joint_features(result, simulator)
    if success is not None:
        result["privileged.task.success"] = success

    predicate_values = [_predicate_value(raw_env, atom) for atom in atoms]
    known_values = [value for value in predicate_values if value is not None]
    result["privileged.task.goal.evaluable_count"] = len(known_values)
    result["privileged.task.goal.satisfied_count"] = sum(known_values)
    result["privileged.task.goal.progress_available"] = bool(known_values)
    if known_values:
        result["privileged.task.goal.progress"] = float(
            sum(known_values) / len(known_values)
        )
    for index, (atom, value) in enumerate(zip(atoms, predicate_values, strict=True)):
        result[f"privileged.task.goal.predicate.{index}.name"] = atom[0]
        if value is not None:
            result[f"privileged.task.goal.predicate.{index}.satisfied"] = value

    entity_contact = contacts["per_entity"]
    for name, record in entities.items():
        prefix = f"privileged.entity.{_feature_token(name)}"
        result[f"{prefix}.name"] = name
        _write_pose_features(result, prefix, record)
        if eef is not None:
            result[f"{prefix}.distance_to_eef_m"] = float(
                np.linalg.norm(np.asarray(record["position"]) - eef)
            )
        for key, value in entity_contact[name].items():
            result[f"{prefix}.{key}"] = bool(value)

    resolved_atoms: list[tuple[int, tuple[str, ...]]] = []
    for index, atom in enumerate(atoms):
        resolved = [argument for argument in atom[1:] if argument in entities]
        if len(resolved) >= 2:
            resolved_atoms.append((index, (atom[0], *resolved[:2])))
    if not resolved_atoms:
        return result

    # Multi-goal LIBERO tasks need recovery to operate on unfinished work.
    # Keep BDDL order deterministic, but skip already-satisfied predicates when
    # an evaluable unsatisfied relation is available.
    primary_index, primary_atom = next(
        (
            (index, atom)
            for index, atom in resolved_atoms
            if predicate_values[index] is False
        ),
        resolved_atoms[0],
    )

    relation, manipulated_name, target_name = primary_atom
    manipulated = entities[manipulated_name]
    target = entities[target_name]
    manipulated_position = np.asarray(manipulated["position"], dtype=np.float64)
    target_position = np.asarray(target["position"], dtype=np.float64)
    offset = target_position - manipulated_position
    target_distance = float(np.linalg.norm(offset))
    result.update(
        {
            "privileged.task.primary_relation": relation,
            "privileged.task.manipulated_object.name": manipulated_name,
            "privileged.task.target.name": target_name,
            "privileged.task.target_identity_confidence": 1.0,
            "privileged.task.manipulated_object.distance_to_target_m": target_distance,
            "privileged.task.manipulated_object.target_offset.x": float(offset[0]),
            "privileged.task.manipulated_object.target_offset.y": float(offset[1]),
            "privileged.task.manipulated_object.target_offset.z": float(offset[2]),
        }
    )
    _write_pose_features(result, "privileged.task.manipulated_object", manipulated)
    _write_pose_features(result, "privileged.task.target", target)
    for key, value in entity_contact[manipulated_name].items():
        result[f"privileged.task.manipulated_object.{key}"] = bool(value)
    for key, value in entity_contact[target_name].items():
        result[f"privileged.task.target.{key}"] = bool(value)
    if eef is not None:
        result["privileged.task.manipulated_object.distance_to_eef_m"] = float(
            np.linalg.norm(manipulated_position - eef)
        )
        result["privileged.task.target.distance_to_eef_m"] = float(
            np.linalg.norm(target_position - eef)
        )
    relation_value = predicate_values[primary_index]
    if relation_value is not None:
        result["privileged.task.primary_relation_satisfied"] = relation_value
        if relation.casefold() in {"in", "inside", "on", "ontop", "on_top"}:
            result["privileged.task.manipulated_object.in_target"] = relation_value

    movable_names = set(
        getattr(raw_env, "objects_dict", {}).keys()
        if isinstance(getattr(raw_env, "objects_dict", None), Mapping)
        else ()
    )
    comparison_names = movable_names | {target_name}
    peer_distances = sorted(
        (
            float(np.linalg.norm(np.asarray(record["position"]) - manipulated_position)),
            name,
        )
        for name, record in entities.items()
        if name != manipulated_name and name in comparison_names
    )
    if peer_distances:
        result["privileged.task.nearest_entity.name"] = peer_distances[0][1]
        result["privileged.task.nearest_entity.distance_m"] = peer_distances[0][0]
        result["privileged.task.target.is_nearest_entity"] = (
            peer_distances[0][1] == target_name
        )
        result["privileged.task.target.distance_rank"] = 1 + next(
            index
            for index, (_, name) in enumerate(peer_distances)
            if name == target_name
        )
    return result


def _enrich_critic_history(
    state: dict[str, Any], history: dict[str, Any]
) -> dict[str, Any]:
    grasp_key = "privileged.task.manipulated_object.grasped"
    if grasp_key not in state:
        return state
    grasped = bool(state.get(grasp_key, False))
    previously_grasped = bool(history.get("grasped", False))
    ever_grasped = bool(history.get("ever_grasped", False) or grasped)
    released_now = previously_grasped and not grasped
    ever_released = bool(history.get("ever_released", False) or released_now)
    state["privileged.task.manipulated_object.ever_grasped"] = ever_grasped
    state["privileged.task.manipulated_object.retained"] = ever_grasped and grasped
    state["privileged.task.manipulated_object.mechanical_engagement"] = grasped
    state["privileged.task.manipulated_object.coupled"] = (
        ever_grasped and grasped
    )
    state["privileged.task.manipulated_object.released_now"] = released_now
    state["privileged.task.manipulated_object.ever_released"] = ever_released
    task_success = bool(state.get("privileged.task.success", False))
    state["privileged.task.stage.index"] = (
        3 if task_success else 2 if ever_released else 1 if grasped else 0
    )
    state["privileged.task.stage.name"] = (
        "complete"
        if task_success
        else "released"
        if ever_released
        else "transport"
        if grasped
        else "pregrasp"
    )

    distance_key = "privileged.task.manipulated_object.distance_to_target_m"
    distance = state.get(distance_key)
    previous_distance = history.get("target_distance_m")
    progress_available = isinstance(distance, (int, float)) and isinstance(
        previous_distance, (int, float)
    )
    state["privileged.task.manipulated_object.target_progress_available"] = (
        progress_available
    )
    if progress_available:
        delta = float(distance) - float(previous_distance)
        state["privileged.task.manipulated_object.target_distance_delta_m"] = delta
        state["privileged.task.manipulated_object.target_progress_m"] = -delta

    history.update(
        {
            "grasped": grasped,
            "ever_grasped": ever_grasped,
            "ever_released": ever_released,
        }
    )
    if isinstance(distance, (int, float)):
        history["target_distance_m"] = float(distance)
    return state


def _bound_collect_contacts(self: Any, **kwargs: Any) -> dict[str, Any]:
    return collect_privileged_contacts(self, **kwargs)


def _bound_semantic_joint_plan(self: Any, **kwargs: Any) -> dict[str, Any]:
    return collect_privileged_semantic_joint_plan(self, **kwargs)


def _bound_collect_critic_state(
    self: Any, *, reset_tracker: bool = False
) -> dict[str, Any]:
    if reset_tracker or not hasattr(self, "_rpent_critic_history"):
        self._rpent_critic_history = {}
    state = collect_privileged_critic_state(self)
    return _enrich_critic_history(state, self._rpent_critic_history)


def wrap_libero_env_factories(
    factories: list[Callable[[], Any]],
) -> list[Callable[[], Any]]:
    """Attach the sensor method inside each LIBERO subprocess factory."""

    wrapped: list[Callable[[], Any]] = []
    for factory in factories:

        def sensor_factory(factory: Callable[[], Any] = factory) -> Any:
            assets_override = os.environ.get("LIBERO_ASSETS_ROOT_OVERRIDE")
            if assets_override:
                # RLinF uses multiprocessing spawn. Bind again immediately
                # before its nested factory imports the concrete LIBERO env.
                from robots.libero.assets import bind_libero_assets_root

                bind_libero_assets_root(assets_override)
            # RLinF creates environments with multiprocessing spawn, so the
            # persistent renderer patch must be installed inside this nested
            # factory rather than only in the parent RPC process. This isolates
            # every simulator's RGB context when several environments share a
            # physical GPU.
            try:
                from robots.robocasa.stable_renderer import (
                    install_persistent_rgb_renderer,
                )

                install_persistent_rgb_renderer()
            except ModuleNotFoundError as exc:
                # Lightweight protocol tests use a fake factory without the
                # RoboSuite runtime. A real LIBERO worker imports RoboSuite
                # before this factory and therefore never takes this branch.
                if exc.name != "robosuite":
                    raise
            env = factory()
            env.rpent_privileged_contacts = types.MethodType(
                _bound_collect_contacts, env
            )
            env.rpent_privileged_semantic_joint_plan = types.MethodType(
                _bound_semantic_joint_plan, env
            )
            env.rpent_privileged_critic_state = types.MethodType(
                _bound_collect_critic_state, env
            )
            return env

        wrapped.append(sensor_factory)
    return wrapped


def install_libero_contact_extension(libero_env_class: type) -> None:
    """Patch one RLinF ``LiberoEnv`` class before it creates subprocesses."""
    if getattr(libero_env_class, "_rpent_contact_extension", False):
        return
    original = libero_env_class.get_env_fns

    def get_env_fns_with_contact(self: Any) -> list[Callable[[], Any]]:
        return wrap_libero_env_factories(original(self))

    libero_env_class.get_env_fns = get_env_fns_with_contact
    libero_env_class._rpent_contact_extension = True
