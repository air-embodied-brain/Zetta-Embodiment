from __future__ import annotations

from scripts.evolution.run_pure_vla_loop1 import (
    FROZEN_EVALUATION_HORIZON,
    _config_hash,
    _milestone_steps,
    _object_relative_state,
    _parser,
)


def _row(step: int, **values: object) -> dict[str, object]:
    state = {
        "privileged.pick_place.object_toaster_slot_contact": step < 2,
        "privileged.pick_place.object_grasped": False,
        "privileged.pick_place.object_plate_contact": False,
        "privileged.pick_place.object_on_plate": False,
        "privileged.pick_place.gripper_object_far": True,
    }
    state.update(values)
    return {"step_index": step, "state": state}


def test_milestone_prefix_and_first_missing_are_frozen():
    states = [
        _row(0),
        _row(1, **{"privileged.pick_place.object_grasped": True}),
        _row(2, **{"privileged.pick_place.object_toaster_slot_contact": False}),
        _row(3, **{"privileged.pick_place.object_plate_contact": True}),
        _row(
            4,
            **{
                "privileged.pick_place.object_plate_contact": True,
                "privileged.pick_place.object_on_plate": True,
                "privileged.pick_place.gripper_object_far": True,
            },
        ),
    ]
    steps, completed, first_missing, progress = _milestone_steps(states, True)
    assert steps["initial_object_in_toaster"] == 0
    assert steps["object_grasped"] == 1
    assert steps["object_exited_toaster"] == 2
    assert steps["object_contacted_plate"] == 3
    assert steps["released_far_on_plate"] == 4
    assert steps["authoritative_task_success"] == 4
    assert first_missing is None
    assert len(completed) == 6
    assert progress == 1.0


def test_milestone_failure_preserves_first_missing_and_object_relative_fields():
    states = [_row(0), _row(1)]
    steps, completed, first_missing, progress = _milestone_steps(states, False)
    assert steps["initial_object_in_toaster"] == 0
    assert completed == ["initial_object_in_toaster"]
    assert first_missing == "object_grasped"
    assert progress == 1 / 6
    state = {
        "privileged.pick_place.object_position_relative_to_toaster": [1, 2, 3],
        "state.gripper_qpos": [0, 0],
    }
    assert _object_relative_state(state) == {
        "object_position_relative_to_toaster": [1, 2, 3]
    }


def test_config_hash_ignores_only_its_own_digest():
    config = {"schema_version": 1, "task": "PickPlaceToasterToCounter"}
    config["config_sha256"] = _config_hash(config)
    assert _config_hash(config) == config["config_sha256"]


def test_frozen_loop1_uses_full_robocasa_horizon():
    parser = _parser()
    assert FROZEN_EVALUATION_HORIZON == 1000
    assert parser.get_default("sim_max_steps") == FROZEN_EVALUATION_HORIZON
    assert parser.get_default("max_actions") == FROZEN_EVALUATION_HORIZON
    assert parser.get_default("actions_per_chunk") == 16
