# Copyright (c) 2026 Zetta Contributors
"""The robotwin policy path: camera contract and the shipped preset (S3).

Two classes of silent breakage are pinned here.

The **camera contract**: RoboTwin gives three views (head, left wrist, right
wrist) and ``pi05_aloha_robotwin`` is built with ``num_images_in_input: 3``.
The Runtime carries them as ``main_image`` + ``wrist_image`` +
``extra_view_images[0]``, and that structure enters ``obs_schema_digest``,
which in turn enters the inference ``compat_key``. Losing a view does not
raise anywhere -- it silently changes the batching bucket and feeds the model
a different observation than it was trained on -- so the digest is asserted to
*change* when the layout changes and to *hold* when only pixels do.

The **preset**: ``load_config`` validates the runtime skeleton but never looks
inside ``env_config``/``policy_config``, so a preset can name a field that no
backend accepts and only fail on a GPU host minutes into a deploy. These tests
parse both blocks with the real backend configs and cross-check the handful of
values that have to agree between the env and the policy.
"""

from __future__ import annotations

import numpy as np
import pytest

from rollout_runtime.api.messages import ResetSpec
from rollout_runtime.config.schema import load_config
from rollout_runtime.core.obs_schema import (
    obs_schema_digest,
    observations_to_env_output,
)
from tests.runtime.test_robotwin_family import (
    ACTION_DIM,
    LEFT_FILL,
    RIGHT_FILL,
    _StubRoboTwinEnv,
    _build,
    _frame,
    robotwin_backend,  # noqa: F401 - fixture re-export
)

PRESET = "robotwin_pi05"


# ------------------------------------------------------------ camera contract


def test_three_views_survive_into_the_five_key_env_output(robotwin_backend) -> None:  # noqa: F811
    """All three RoboTwin cameras reach the policy-facing batch dict.

    ``observations_to_env_output`` is what the policy backend actually reads.
    If the right wrist were dropped on the way, ``num_images_in_input: 3``
    would still be configured and the model would be fed a short stack.
    """
    core = _build(robotwin_backend)
    observation = core.reset([0], ResetSpec(seed=1))[0]
    output = observations_to_env_output([observation])

    assert output["main_images"] is not None
    assert output["wrist_images"] is not None
    assert output["extra_view_images"] is not None
    assert output["task_descriptions"] == ["pick the bottle with the correct arm"]
    assert np.asarray(output["states"]).shape == (1, ACTION_DIM)


def test_schema_digest_changes_when_a_view_disappears(robotwin_backend) -> None:  # noqa: F811
    """Dropping the right wrist must change the compat bucket, not pass silently."""
    core = _build(robotwin_backend)
    both = obs_schema_digest(core.reset([0], ResetSpec(seed=1))[0])

    _StubRoboTwinEnv.instances[0].wrist_names = ["left_wrist_image"]
    left_only = obs_schema_digest(core.reset([0], ResetSpec(seed=1))[0])

    _StubRoboTwinEnv.instances[0].wrist_names = []
    none_at_all = obs_schema_digest(core.reset([0], ResetSpec(seed=1))[0])

    assert len({both, left_only, none_at_all}) == 3


def test_schema_digest_is_stable_across_pixel_changes(robotwin_backend) -> None:  # noqa: F811
    """The digest is structural: same shapes, same bucket, different content.

    Without this the digest would be content-sensitive and every frame would
    open its own batching bucket, quietly destroying inference batching.
    """
    core = _build(robotwin_backend)
    first = obs_schema_digest(core.reset([0], ResetSpec(seed=1))[0])
    core.chunk_step([0], [np.zeros((4, ACTION_DIM), dtype=np.float32)])
    second = obs_schema_digest(core.observe([0])[0])
    assert first == second


def test_left_and_right_wrists_are_not_interchangeable(robotwin_backend) -> None:  # noqa: F811
    """A mirrored wrist mapping is invisible to the schema, so pin the bytes.

    Both orderings produce the *same* digest -- identical shapes in identical
    fields -- which is exactly why the digest cannot be the only guard here.
    """
    from rollout_runtime.core import payload as payload_module

    core = _build(robotwin_backend)
    straight = core.reset([0], ResetSpec(seed=1))[0]

    _StubRoboTwinEnv.instances[0].wrist_names = [
        "right_wrist_image",
        "left_wrist_image",
    ]
    swapped = core.reset([0], ResetSpec(seed=1))[0]

    assert obs_schema_digest(straight) == obs_schema_digest(swapped)
    # ... and the adapter still routes by name, so the frames do not move.
    for observation in (straight, swapped):
        assert observation.wrist_image == payload_module.encode_image(_frame(LEFT_FILL))
        assert observation.extra_view_images[0] == payload_module.encode_image(
            _frame(RIGHT_FILL)
        )


# -------------------------------------------------------------------- preset


def test_preset_env_config_parses_with_the_real_backend() -> None:
    """``load_config`` never validates ``env_config``; this does."""
    from rollout_runtime.backends.rlinf_robotwin import RobotwinEnvConfig

    config = load_config(PRESET)
    assert config.env_family == "robotwin"
    env_config = RobotwinEnvConfig.from_mapping(config.env_config)
    assert env_config.task_name == "adjust_bottle"
    assert env_config.planner_backend == "mplib"
    assert env_config.action_dim == ACTION_DIM


def test_preset_policy_config_parses_with_the_real_backend() -> None:
    """A preset naming a field no backend accepts must fail here, not on a GPU host."""
    from rollout_runtime.backends.rlinf_policy import RlinfPolicyConfig

    config = load_config(PRESET)
    policy = RlinfPolicyConfig.from_mapping(config.rollout_worker.policy_config)
    assert policy.model_type == "openpi"
    assert policy.action_dim == ACTION_DIM
    assert policy.family_params["config_name"] == "pi05_aloha_robotwin"


def test_preset_matches_the_rlinf_baseline_configuration() -> None:
    """The values that produced the S0 baseline must not drift silently.

    ``plan/robotwin_s0_findings.md`` records success_once=0.9375 for RLinf's
    own eval at these settings. S3 compares against that number, so a change
    to any of them invalidates the comparison and should be a deliberate edit.
    """
    from rollout_runtime.backends.rlinf_policy import RlinfPolicyConfig

    policy = RlinfPolicyConfig.from_mapping(
        load_config(PRESET).rollout_worker.policy_config
    )
    assert policy.num_action_chunks == 50
    assert policy.num_steps == 5
    assert policy.family_params["num_images_in_input"] == 3
    assert policy.family_params["noise_level"] == 0.3

    model_cfg = policy.to_model_cfg()
    # `to_model_cfg` derives these; the model is built from them, not from the
    # checkpoint's own config.json (which carries a stale action_horizon: 10).
    assert model_cfg.openpi.action_chunk == 50
    assert model_cfg.openpi.action_env_dim == ACTION_DIM


def test_preset_env_and_policy_agree_on_the_shared_values() -> None:
    """Env and policy are configured separately but must describe one robot."""
    from rollout_runtime.backends.rlinf_policy import RlinfPolicyConfig
    from rollout_runtime.backends.rlinf_robotwin import RobotwinEnvConfig

    config = load_config(PRESET)
    env_config = RobotwinEnvConfig.from_mapping(config.env_config)
    policy = RlinfPolicyConfig.from_mapping(config.rollout_worker.policy_config)

    assert env_config.action_dim == policy.action_dim, "one robot, one action width"
    # Three model inputs require the wrist cameras to actually be rendered.
    assert env_config.collect_wrist_camera is True
    assert env_config.collect_head_camera is True
    assert policy.family_params["num_images_in_input"] == 3
    # D3: the executed horizon is an open-loop truncation of the model's chunk,
    # so it can never exceed it.
    assert env_config.execute_horizon is not None
    assert env_config.execute_horizon <= policy.num_action_chunks


def test_preset_pool_sizes_respect_the_measured_sapien_ceiling() -> None:
    """24 GB cards cannot host more than 16 concurrent RoboTwin envs, machine-wide."""
    from rollout_runtime.core.env_registry import behavior_for

    config = load_config(PRESET)
    ceiling = behavior_for("robotwin").max_pool_size
    assert ceiling is not None
    assert config.env_worker.default_pool_size <= ceiling
    assert config.env_worker.max_sessions_per_rank <= ceiling
    # SAPIEN needs a GPU, so the env group must not use the `node` strategy --
    # that infers has_accelerator=False and the registry refuses to schedule.
    assert config.env_worker.placement_strategy == "packed"


def test_preset_carries_no_real_machine_paths() -> None:
    """Presets ship placeholders; a real checkpoint path would leak a host layout."""
    config = load_config(PRESET)
    assert config.env_config["assets_path"].startswith("/path/to/")
    assert config.rollout_worker.policy_config["model_path"].startswith("/path/to/")


# ------------------------------------------------------- wrist re-stacking


def test_stack_wrist_views_rebuilds_the_aloha_pair() -> None:
    """The split in the env adapter and this re-stack are a matched pair.

    ``aloha_policy._decode_aloha`` reads the left and right wrist as
    ``wrist_images[0]`` / ``wrist_images[1]`` and never touches
    ``extra_view_image``. The Runtime's ``Observation`` cannot carry a stacked
    pair, so the pair is reassembled here, right before the forward pass.
    """
    from rollout_runtime.backends.rlinf_policy import _stack_wrist_views

    left = np.zeros((2, 4, 5, 3), dtype=np.uint8)
    right = np.ones((2, 1, 4, 5, 3), dtype=np.uint8)
    stacked = _stack_wrist_views({"wrist_images": left, "extra_view_images": right})

    assert stacked["wrist_images"].shape == (2, 2, 4, 5, 3)
    assert stacked["extra_view_images"] is None
    # Order matters: index 0 must stay the left wrist.
    assert stacked["wrist_images"][0, 0].max() == 0
    assert stacked["wrist_images"][0, 1].min() == 1


def test_stack_wrist_views_is_a_noop_without_both_views() -> None:
    """Safe to leave enabled for a single-wrist configuration."""
    from rollout_runtime.backends.rlinf_policy import _stack_wrist_views

    only_wrist = {"wrist_images": np.zeros((1, 4, 5, 3)), "extra_view_images": None}
    assert _stack_wrist_views(only_wrist) is only_wrist

    neither = {"wrist_images": None, "extra_view_images": None}
    assert _stack_wrist_views(neither) is neither


def test_preset_enables_wrist_restacking() -> None:
    """RoboTwin + ALOHA needs it; forgetting the flag fails silently at inference.

    Without it the model receives a single ``[H, W, C]`` wrist frame and indexes
    row 0 and row 1 out of its *height*, which surfaces only as a channel-count
    mismatch inside the vision tower.
    """
    from rollout_runtime.backends.rlinf_policy import RlinfPolicyConfig
    from rollout_runtime.backends.rlinf_robotwin import RobotwinEnvConfig

    config = load_config(PRESET)
    policy = RlinfPolicyConfig.from_mapping(config.rollout_worker.policy_config)
    env_config = RobotwinEnvConfig.from_mapping(config.env_config)

    assert policy.stack_wrist_views is True
    # Only meaningful when the env actually renders both wrists.
    assert env_config.collect_wrist_camera is True


def test_stacked_wrists_round_trip_from_the_env_adapter(robotwin_backend) -> None:  # noqa: F811
    """End to end: adapter splits the pair, the policy helper puts it back.

    This is the invariant that the real hardware run broke, so it is asserted
    on the actual observation the adapter produces rather than on a synthetic
    tensor.
    """
    from rollout_runtime.backends.rlinf_policy import _stack_wrist_views

    core = _build(robotwin_backend)
    observation = core.reset([0], ResetSpec(seed=1))[0]
    env_obs = observations_to_env_output([observation])

    stacked = _stack_wrist_views(env_obs)["wrist_images"]
    assert stacked.shape[:2] == (1, 2)
    assert int(np.asarray(stacked)[0, 0].flat[0]) == LEFT_FILL
    assert int(np.asarray(stacked)[0, 1].flat[0]) == RIGHT_FILL
