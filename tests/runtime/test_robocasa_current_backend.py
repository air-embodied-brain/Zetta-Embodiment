# Copyright (c) 2026 Zetta Contributors
"""``rollout_runtime/backends/robocasa_current.py`` — runtime v3 design
§3.4/Stage 4, ``EnvExecutionCore`` for the current-branch ``RoboCasaSession``.

The real RoboCasa/robosuite simulator is not installed in this environment (nor is
it expected to be for unit tests, see ``tests/test_robocasa_env_runtime.py``'s same
constraint), so these tests monkeypatch ``RoboCasaSession._ensure_environment`` with
a fake gym-like environment, exactly like
``tests/test_robocasa_runtime_split.py::test_robocasa_session_reset_and_execute_chunk_work_without_any_server``
does for the plain ``RoboCasaSession`` class. The point here is the *adapter*
boundary: build() constructing one RoboCasaSession per slot, reset()/chunk_step()
translating Runtime <-> RoboCasaSession, and Observation carrying both a flattened
state vector and the named ``extras["raw_state"]`` dict GR00T needs.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from robots.robocasa import session_core
from rollout_runtime.api.enums import ErrorCode
from rollout_runtime.api.errors import RuntimeApiError
from rollout_runtime.api.messages import EnvSpecMsg, ResetSpec
from rollout_runtime.backends.robocasa_current import (
    RobocasaCurrentConfig,
    RobocasaCurrentCore,
    RobocasaCurrentFamily,
    robocasa_current_capability,
)
from rollout_runtime.core.env_execution import PER_SLOT_FORM


class _FakeEnv:
    """Minimal fake gym env: 3 cameras + a couple of named vector state fields."""

    def __init__(self) -> None:
        self.actions: list[Any] = []
        self._step_index = 0

    def reset(self, seed: int) -> tuple[dict[str, Any], dict]:
        del seed
        self._step_index = 0
        return self._observation(), {"success": False}

    def step(self, action: dict[str, np.ndarray]) -> tuple[Any, ...]:
        self.actions.append(action)
        self._step_index += 1
        terminated = self._step_index >= 3
        return (
            self._observation(),
            1.0 if terminated else 0.0,
            terminated,
            False,
            {"success": terminated},
        )

    def close(self) -> None:
        pass

    def _observation(self) -> dict[str, Any]:
        return {
            "video.robot0_agentview_left": np.zeros((4, 4, 3), dtype=np.uint8),
            "video.robot0_agentview_right": np.full((4, 4, 3), 9, dtype=np.uint8),
            "video.robot0_eye_in_hand": np.full((4, 4, 3), 7, dtype=np.uint8),
            "state.end_effector_position_relative": np.array(
                [0.1, 0.2, 0.3], dtype=np.float32
            ),
            "state.gripper_qpos": np.array([0.0, 1.0], dtype=np.float32),
            "task_descriptions": ["move the pan"],
        }


@pytest.fixture
def fake_ensure_environment(monkeypatch: pytest.MonkeyPatch):
    """Stub ``RoboCasaSession._ensure_environment`` to inject ``_FakeEnv`` instances.

    Args:
        monkeypatch: pytest fixture.

    Returns:
        A dict of ``{RoboCasaSession id: _FakeEnv}`` populated as sessions build
        their environment.
    """
    envs: dict[int, _FakeEnv] = {}

    def _fake_ensure_environment(self, task: str, split: str) -> None:
        env = _FakeEnv()
        envs[id(self)] = env
        self.env = env
        self.identity = (task, split)

    monkeypatch.setattr(
        session_core.RoboCasaSession,
        "_ensure_environment",
        _fake_ensure_environment,
    )
    return envs


def _spec(**overrides: Any) -> EnvSpecMsg:
    config: dict[str, Any] = {
        "task": "SlideDishwasherRack",
        "require_isolated_renderer": False,
    }
    config.update(overrides)
    return EnvSpecMsg(env_family="robocasa", env_config=config, pool_size=1)


def test_config_requires_task_and_rejects_unknown_keys() -> None:
    with pytest.raises(RuntimeApiError) as excinfo:
        RobocasaCurrentConfig.from_mapping({})
    assert excinfo.value.info.code is ErrorCode.INVALID_ARGUMENT

    with pytest.raises(RuntimeApiError) as excinfo:
        RobocasaCurrentConfig.from_mapping({"task": "X", "bogus": 1})
    assert excinfo.value.info.code is ErrorCode.INVALID_ARGUMENT

    config = RobocasaCurrentConfig.from_mapping({"task": "SlideDishwasherRack"})
    assert config.task == "SlideDishwasherRack"
    assert config.require_isolated_renderer is True


def test_config_no_longer_accepts_operation_gate_fields() -> None:
    """GPU gating has moved to Ray's placement declarations; the
    ``operation_gate_*`` fields have been removed from
    ``RobocasaCurrentConfig``. Passing them must now be rejected as unknown
    keys, rather than being silently accepted and doing nothing.
    """
    with pytest.raises(RuntimeApiError) as excinfo:
        RobocasaCurrentConfig.from_mapping(
            {"task": "X", "operation_gate_root": "/tmp/gates", "operation_gate_gpu": "0"}
        )
    assert excinfo.value.info.code is ErrorCode.INVALID_ARGUMENT
    assert "operation_gate_root" in str(excinfo.value.info.detail)


def test_build_creates_one_independent_session_per_slot(fake_ensure_environment) -> None:
    core = RobocasaCurrentCore()
    core.build(_spec(), num_envs=3, seed_offset=10)
    assert core.core_form == PER_SLOT_FORM
    assert len(core._slots) == 3
    sessions = [slot.session for slot in core._slots]
    assert len({id(session) for session in sessions}) == 3
    core.close()


def test_reset_and_chunk_step_round_trip(fake_ensure_environment) -> None:
    core = RobocasaCurrentCore()
    core.build(_spec(), num_envs=1, seed_offset=0)

    observations = core.reset([0], ResetSpec(seed=5))
    assert len(observations) == 1
    obs = observations[0]
    assert obs.main_image is not None
    assert obs.wrist_image is not None
    assert len(obs.extra_view_images) == 1
    assert obs.instruction == "move the pan"
    # named state survives in extras, not just the flattened vector
    assert obs.extras["raw_state"]["state.end_effector_position_relative"] == pytest.approx(
        [0.1, 0.2, 0.3]
    )
    assert obs.state  # flattened vector is non-empty (vectors weren't dropped)
    assert len(obs.state) >= 5  # 3 (eef pos) + 2 (gripper) at minimum

    outcome = core.chunk_step(
        [0], [np.zeros((3, 12), dtype=np.float32)]
    )[0]
    assert outcome.executed_horizon == 3
    assert outcome.terminated is True
    assert outcome.observation is not None
    assert outcome.observation.main_image is not None
    fake_env = core._slots[0].session.env
    assert len(fake_env.actions) == 3
    core.close()


def test_chunk_step_before_reset_is_session_not_ready(fake_ensure_environment) -> None:
    core = RobocasaCurrentCore()
    core.build(_spec(), num_envs=1, seed_offset=0)
    with pytest.raises(RuntimeApiError) as excinfo:
        core.chunk_step([0], [np.zeros((1, 12), dtype=np.float32)])
    assert excinfo.value.info.code is ErrorCode.SESSION_NOT_READY
    core.close()


def test_slot_out_of_range_is_invalid_argument(fake_ensure_environment) -> None:
    core = RobocasaCurrentCore()
    core.build(_spec(), num_envs=1, seed_offset=0)
    core.reset([0], ResetSpec(seed=1))
    with pytest.raises(RuntimeApiError) as excinfo:
        core.observe([5])
    assert excinfo.value.info.code is ErrorCode.INVALID_ARGUMENT
    core.close()


def test_reset_uses_the_authoritative_seed_without_slot_offset(
    fake_ensure_environment,
) -> None:
    """RoboCasa's seed does **not** get a ``seed_offset + slot_index`` offset
    applied.

    The paired gate treats "the same seed" as a hard contract for comparing
    the two arms (``EpisodeRecord.seed`` is cross-checked item-by-item against
    the preregistration seed table); an offset would let "which slot it lands
    on" determine the real initial state, so the seed in the audit record
    would no longer be authoritative.
    """
    seeds: list[int] = []

    def _record_seed(self, payload):  # type: ignore[no-untyped-def]
        seeds.append(int(payload["seed"]))
        return {"observation": {"state": {}}}

    core = RobocasaCurrentCore()
    core.build(_spec(), num_envs=2, seed_offset=100)
    monkey = pytest.MonkeyPatch()
    try:
        monkey.setattr(session_core.RoboCasaSession, "reset", _record_seed)
        monkey.setattr(RobocasaCurrentCore, "_observation", lambda self, index: index)
        core.reset([0, 1], ResetSpec(seed=7))
    finally:
        monkey.undo()
    assert seeds == [7, 7]
    core.close()


def test_reset_forwards_episode_options_to_the_session(fake_ensure_environment) -> None:
    """``video_dir``/``bundle_sha256``/``action_scale`` must actually reach
    ``RoboCasaSession``.

    If ``ResetSpec.options`` were dropped entirely, an episode would never
    record video (``zetta.evolution.trajectory`` hard-requires video to exist)
    and would also never carry the bundle proof field.
    """
    core = RobocasaCurrentCore()
    core.build(_spec(), num_envs=1, seed_offset=0)
    core.reset(
        [0],
        ResetSpec(
            seed=3,
            options={
                "video_dir": None,
                "bundle_sha256": "b" * 64,
                "action_scale": {"end_effector_position": 0.5},
            },
        ),
    )
    session = core._slots[0].session
    assert session.bundle_sha256 == "b" * 64
    assert session.action_scale.end_effector_position == pytest.approx(0.5)
    core.close()


def test_reset_rejects_unknown_options(fake_ensure_environment) -> None:
    core = RobocasaCurrentCore()
    core.build(_spec(), num_envs=1, seed_offset=0)
    with pytest.raises(RuntimeApiError) as excinfo:
        core.reset([0], ResetSpec(seed=1, options={"bogus": 1}))
    assert excinfo.value.info.code is ErrorCode.INVALID_ARGUMENT
    core.close()


def test_episode_critic_rules_reach_execute_chunk(fake_ensure_environment) -> None:
    """Core fix: ``critic_rules`` is backfilled per chunk, no longer a
    hardcoded empty list.

    Without this, the critic could never produce a proposal, and the
    ``active_bundle`` mode has no effect on the Runtime path
    (``run_rollout.py``'s ``last_proposals`` would always be empty, so
    Role1/RecoveryController would never trigger).
    """
    rule = {
        "rule_id": "critic-1",
        "title": "privileged state came from the live simulator",
        # Use a string field guaranteed to exist in the fake env so the rule
        # **actually** fires: what this test checks is "the rule can still
        # interrupt a chunk after crossing the runtime boundary", not just
        # that the field is forwarded.
        "feature": "privileged.source",
        "operator": "eq",
        "threshold": "live_mujoco_simulator",
        "dwell_steps": 1,
        "cooldown_steps": 0,
        "proposal": "stop and re-approach",
        "evidence_ids": ["evidence-1"],
        "safety_only": False,
        "activation_conditions": [],
    }
    payloads: list[dict[str, Any]] = []
    core = RobocasaCurrentCore()
    core.build(_spec(), num_envs=1, seed_offset=0)
    core.reset(
        [0],
        ResetSpec(
            seed=1,
            options={"critic_rules": [rule], "capture_event_images": False},
        ),
    )
    original = session_core.RoboCasaSession.execute_chunk

    def _capture(self, payload):  # type: ignore[no-untyped-def]
        payloads.append(payload)
        return original(self, payload)

    monkey = pytest.MonkeyPatch()
    try:
        monkey.setattr(session_core.RoboCasaSession, "execute_chunk", _capture)
        outcome = core.chunk_step([0], [np.zeros((2, 12), dtype=np.float32)])[0]
    finally:
        monkey.undo()
    assert payloads[0]["critic_rules"] == [rule]
    # A rule interrupts the chunk (same semantics as the pre-migration
    # interrupt_on_proposal=bool(critic_rules)).
    assert payloads[0]["interrupt_on_proposal"] is True
    assert payloads[0]["capture_event_images"] is False
    assert outcome.info["critic_rule_count"] == 1
    assert [item["rule_id"] for item in outcome.info["critic_proposals"]] == ["critic-1"]
    # The rule fires on step 1, the chunk is interrupted, and the 2nd action never lands.
    assert outcome.executed_horizon == 1
    assert outcome.per_step is not None
    assert outcome.per_step[0].info["proposal_rule_ids"] == ["critic-1"]
    core.close()


def test_chunk_outcome_carries_the_per_step_audit_trail(fake_ensure_environment) -> None:
    """Per-step ``applied_action``/``action_sha256``/named state must cross the
    runtime boundary.

    ``run_rollout.py``'s ``actions.jsonl``/``states.jsonl`` (as well as the
    trajectory index, failure slicing, and visual evidence) are all built on
    these per-step fields, and ``PerStepRecord``'s named fields only cover
    reward/terminated/truncated -- so they can only travel through
    ``PerStepRecord.info``.
    """
    core = RobocasaCurrentCore()
    core.build(_spec(), num_envs=1, seed_offset=0)
    core.reset([0], ResetSpec(seed=1))
    outcome = core.chunk_step([0], [np.zeros((2, 12), dtype=np.float32)])[0]
    assert outcome.per_step is not None
    first = outcome.per_step[0]
    assert set(first.info) >= {
        "applied_action",
        "action_sha256",
        "observation_sha256",
        "raw_state",
        "official_success",
        "success_latched",
        "proposal_rule_ids",
    }
    assert len(first.info["action_sha256"]) == 64
    assert "action.end_effector_position" in first.info["applied_action"]
    assert outcome.info["task_program_enabled"] is False
    assert outcome.info["authoritative_success"] is False
    core.close()


def test_observation_extras_carry_the_gen0_attestation(fake_ensure_environment) -> None:
    """``task_program_enabled``/``critic_rule_count`` must travel alongside the
    observation.

    ``StepResult.info`` is generated by a fixed shape in EnvWorker
    (``{"reset": True, ...}``), which has no room for family-private fields,
    so the Gen0 strict_pure_vla attestation can only travel through
    ``Observation.extras``.
    """
    core = RobocasaCurrentCore()
    core.build(_spec(), num_envs=1, seed_offset=0)
    observation = core.reset([0], ResetSpec(seed=1))[0]
    assert observation.extras["task_program_enabled"] is False
    assert observation.extras["critic_rule_count"] == 0
    assert observation.extras["bundle_sha256"] is None
    assert observation.extras["video_paths"] == {}
    core.close()


def test_declared_extensions_dispatch_and_unknown_ones_do_not(
    fake_ensure_environment,
) -> None:
    """The two declared extension methods actually reach
    ``RoboCasaSession``, while everything else remains UNSUPPORTED."""
    core = RobocasaCurrentCore()
    core.build(_spec(), num_envs=1, seed_offset=0)
    core.reset([0], ResetSpec(seed=1))

    snapshot = core.extension(0, "robocasa", "snapshot", {"include_images": False})
    assert snapshot["task_program_enabled"] is False
    assert "state" in snapshot["observation"]
    # include_images=False only skips data-URL encoding; per-camera hashes are
    # still present (session_core's ``_observation_payload``); the Role1 path
    # uses include_images=True to fetch the real image.
    assert snapshot["observation"]["images"] == {}
    assert snapshot["observation"]["image_sha256"]

    with_images = core.extension(0, "robocasa", "snapshot", {})
    assert set(with_images["observation"]["images"]) == {
        "video.robot0_agentview_left",
        "video.robot0_agentview_right",
        "video.robot0_eye_in_hand",
    }
    assert all(
        str(value).startswith("data:")
        for value in with_images["observation"]["images"].values()
    )

    finalized = core.extension(0, "robocasa", "finalize_episode", {})
    assert finalized["finalized"] is True
    assert finalized["video_paths"] == {}

    with pytest.raises(RuntimeApiError) as excinfo:
        core.extension(0, "robocasa", "anything", {})
    assert excinfo.value.info.code is ErrorCode.UNSUPPORTED_EXTENSION
    core.close()


def test_extension_before_reset_is_session_not_ready(fake_ensure_environment) -> None:
    core = RobocasaCurrentCore()
    core.build(_spec(), num_envs=1, seed_offset=0)
    with pytest.raises(RuntimeApiError) as excinfo:
        core.extension(0, "robocasa", "snapshot", {})
    assert excinfo.value.info.code is ErrorCode.SESSION_NOT_READY
    core.close()


def test_jpeg_lossy_rgb_frame_matches_the_legacy_snapshot_encoding() -> None:
    """``jpeg_lossy_rgb_frame`` and the debug-HTTP ``_encode_image`` data URL must
    decode to identical pixels: both are the same JPEG quality=80 quantization
    of the same frame, just packaged for two different transports (raw ndarray
    vs. base64 JPEG data URL).
    """
    import base64
    import io

    import imageio.v3 as iio

    frame = np.arange(4 * 4 * 3, dtype=np.uint8).reshape(4, 4, 3)

    from_helper = session_core.jpeg_lossy_rgb_frame(frame)

    data_url = session_core._encode_image(frame)
    encoded = base64.b64decode(data_url.split(",", 1)[1])
    from_data_url = np.asarray(iio.imread(io.BytesIO(encoded)))

    np.testing.assert_array_equal(from_helper, from_data_url)


def test_observation_images_match_the_pre_migration_jpeg_quantization(
    fake_ensure_environment,
) -> None:
    """Bug fix regression: policy-facing images must be JPEG-80 lossy, not raw pixels.

    The pre-migration direct-HTTP path (``groot_client.py::act`` reading
    ``RoboCasaEnvClient.observation()``) only ever exposed camera frames through
    ``session_core.py``'s JPEG quality=80 round-trip
    (``session_core.jpeg_lossy_rgb_frame`` / ``_encode_image``). If
    ``RobocasaCurrentCore._encode_camera`` fed GR00T the exact simulator pixels
    instead, the same seed would produce a different (numerically "cleaner")
    policy input on Runtime v3 than it did pre-migration, silently diverging the
    action chunk and the whole episode from a same-seed pre-migration replay
    (runtime v3 design Stage 9 step 26: episode records must be
    byte-comparable across the migration).
    """
    from rollout_runtime.core import payload as payload_module

    core = RobocasaCurrentCore()
    core.build(_spec(), num_envs=1, seed_offset=0)
    observation = core.reset([0], ResetSpec(seed=1))[0]

    raw_frame = np.zeros((4, 4, 3), dtype=np.uint8)
    expected = session_core.jpeg_lossy_rgb_frame(raw_frame)

    decoded_main = payload_module.decode_image(observation.main_image)
    np.testing.assert_array_equal(decoded_main, expected)
    # The raw simulator frame for this camera is exactly zero; if the adapter
    # had skipped quantization, decoded_main would equal raw_frame byte-for-byte
    # instead of (possibly) differing after the JPEG round-trip.
    assert decoded_main.dtype == np.uint8
    core.close()


def test_family_declares_the_same_extension_set_as_the_core(
    fake_ensure_environment,
) -> None:
    """The family capability and the execution core's dispatch table must
    share the same source.

    ``RuntimeEnvWorker.extension_call`` filters first by capability: if the
    capability declares one fewer method than the execution core implements,
    that method can never be dispatched to (returns UNSUPPORTED_EXTENSION).
    """
    from rollout_runtime.backends.robocasa_current import (
        ROBOCASA_CURRENT_EXTENSIONS,
    )
    from rollout_runtime.core.env_registry import ROBOCASA_ENV_FAMILY, behavior_for

    assert ROBOCASA_CURRENT_EXTENSIONS == frozenset(
        {"robocasa.snapshot", "robocasa.finalize_episode"}
    )
    assert behavior_for(ROBOCASA_ENV_FAMILY).extensions == ROBOCASA_CURRENT_EXTENSIONS
    assert robocasa_current_capability().extensions == ROBOCASA_CURRENT_EXTENSIONS


def test_capability_matches_the_corrected_behavior_declaration() -> None:
    capability = robocasa_current_capability()
    assert capability.needs_accelerator is True
    assert capability.supports_reset_state_id is False
    assert capability.core_forms == frozenset({PER_SLOT_FORM})
    assert capability.supports_coalescing is False


def test_family_adapter_creates_a_fresh_core_each_time() -> None:
    family = RobocasaCurrentFamily()
    assert family.env_family == "robocasa"
    first = family.create_core()
    second = family.create_core()
    assert first is not second


def test_register_env_family_for_robocasa_uses_this_backend() -> None:
    from rollout_runtime.backends import register_env_family_for

    adapter = register_env_family_for("robocasa")
    assert isinstance(adapter, RobocasaCurrentFamily)


def test_robocasa_current_preset_loads_and_declares_an_accelerator_rank() -> None:
    """The ``robocasa_current.yaml`` preset must load, and the env_worker must
    declare an accelerator rank (RoboCasaSession needs GPU rendering, see
    env_registry.py's ``needs_accelerator_override``).
    """
    from rollout_runtime.config.schema import load_config

    config = load_config("robocasa_current")
    assert config.env_family == "robocasa"
    assert config.rollout_worker.policy_backend == "groot"
    assert config.env_worker.accelerator_present() is True


# --------------------------------------------------------- DynamicSlotPool


def test_slot_count_reflects_the_initial_pool_size(fake_ensure_environment) -> None:
    core = RobocasaCurrentCore()
    assert core.slot_count() == 0
    core.build(_spec(), num_envs=3, seed_offset=0)
    assert core.slot_count() == 3
    core.close()
    assert core.slot_count() == 0


def test_add_slot_appends_an_independent_session_and_returns_the_new_index(
    fake_ensure_environment,
) -> None:
    core = RobocasaCurrentCore()
    core.build(_spec(), num_envs=1, seed_offset=0)
    assert core.slot_count() == 1

    new_index = core.add_slot(seed_offset=999)
    assert new_index == 1
    assert core.slot_count() == 2

    # The new slot is immediately usable: reset/chunk_step work with no extra
    # "activation" step (the protocol requires it to be usable for
    # reset/chunk_step/observe as soon as it's returned).
    core.reset([0, 1], ResetSpec(seed=1))
    outcome = core.chunk_step(
        [0, 1], [np.zeros((1, 12), dtype=np.float32), np.zeros((1, 12), dtype=np.float32)]
    )
    assert len(outcome) == 2

    # The new slot is an independent session instance, not a reuse/alias of an
    # existing slot.
    sessions = [slot.session for slot in core._slots]
    assert len({id(session) for session in sessions}) == 2
    core.close()


def test_add_slot_ignores_the_seed_offset_argument(fake_ensure_environment) -> None:
    """This family's reset does not apply a slot offset (see reset's
    docstring): the ``seed_offset`` argument suggested to ``add_slot`` has no
    consumable use for this family, and must be ignored rather than silently
    changing behavior (same reasoning as the libero version, see
    ``rlinf_env.py::LiberoEnvCore.add_slot``).
    """
    core = RobocasaCurrentCore()
    core.build(_spec(), num_envs=1, seed_offset=0)
    # Pass an obviously invalid/out-of-range value: if it were actually used,
    # something would fail or produce an observable side effect somewhere;
    # when ignored, it should succeed as usual.
    new_index = core.add_slot(seed_offset=10_000_000)
    assert new_index == 1
    core.close()


def test_remove_slot_closes_and_pops_the_trailing_slot(fake_ensure_environment) -> None:
    core = RobocasaCurrentCore()
    core.build(_spec(), num_envs=1, seed_offset=0)
    core.add_slot(seed_offset=1)
    assert core.slot_count() == 2
    core.reset([0, 1], ResetSpec(seed=1))
    removed_session = core._slots[1].session
    assert removed_session.env is not None  # sanity: session had a real fake env

    core.remove_slot(1)
    assert core.slot_count() == 1
    # close_environment() must have actually run against the removed slot (not
    # just the ones left in the pool): RoboCasaSession.close_environment() tears
    # down self.env.
    assert removed_session.env is None
    core.close()


def test_remove_slot_rejects_a_non_trailing_index(fake_ensure_environment) -> None:
    core = RobocasaCurrentCore()
    core.build(_spec(), num_envs=1, seed_offset=0)
    core.add_slot(seed_offset=1)
    core.add_slot(seed_offset=2)
    assert core.slot_count() == 3

    with pytest.raises(RuntimeApiError) as excinfo:
        core.remove_slot(0)
    assert excinfo.value.info.code is ErrorCode.INVALID_ARGUMENT
    assert core.slot_count() == 3, "a rejected removal must not mutate the pool"

    with pytest.raises(RuntimeApiError) as excinfo:
        core.remove_slot(1)
    assert excinfo.value.info.code is ErrorCode.INVALID_ARGUMENT
    assert core.slot_count() == 3

    # Only the true trailing index (2) is accepted.
    core.remove_slot(2)
    assert core.slot_count() == 2
    core.close()


def test_grow_then_shrink_then_grow_again_does_not_corrupt_slot_indices(
    fake_ensure_environment,
) -> None:
    """A general-purpose version of a LIBERO lesson: grow to the ceiling,
    shrink one at a time, then grow again -- indices must stay contiguous
    and usable throughout, with no out-of-range or misalignment (RoboCasa
    has no ``total_num_processes``-style denominator, but this sequence
    itself still needs to be verified against similar pitfalls).
    """
    core = RobocasaCurrentCore()
    core.build(_spec(), num_envs=1, seed_offset=0)

    # Grow to 4.
    for offset in range(1, 4):
        new_index = core.add_slot(seed_offset=offset)
        assert new_index == offset
    assert core.slot_count() == 4

    # Reset all of them, confirming every index is genuinely usable (not
    # "appears to exist but is actually a hole").
    core.reset([0, 1, 2, 3], ResetSpec(seed=1))
    outcomes = core.chunk_step([0, 1, 2, 3], [np.zeros((1, 12), dtype=np.float32)] * 4)
    assert len(outcomes) == 4

    # Shrink back to 1, one at a time (must start from the tail).
    core.remove_slot(3)
    core.remove_slot(2)
    core.remove_slot(1)
    assert core.slot_count() == 1

    # Grow again: the new index must immediately follow the current tail
    # (1), not reuse a previously-used index of 2/3.
    new_index = core.add_slot(seed_offset=0)
    assert new_index == 1
    assert core.slot_count() == 2

    # The new slot is immediately usable.
    core.reset([0, 1], ResetSpec(seed=2))
    outcomes = core.chunk_step([0, 1], [np.zeros((1, 12), dtype=np.float32)] * 2)
    assert len(outcomes) == 2
    assert all(outcome.observation is not None for outcome in outcomes)
    core.close()


def test_add_slot_respects_process_isolation_and_builds_a_remote_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When ``process_isolation=True``, a slot newly created by ``add_slot``
    must also go through ``spawn_robocasa_subprocess``
    (``RemoteRoboCasaSession``), and must not bypass isolation protection
    to directly construct an in-process ``RoboCasaSession`` -- otherwise a
    dynamically grown slot would reintroduce the native EGL race risk that
    Path B was fixed to avoid (see the module docstring /
    ``RobocasaCurrentConfig.process_isolation`` field description).
    """
    from robots.robocasa import session_process as session_process_module
    from robots.robocasa.session_process import RemoteRoboCasaSession

    calls: list[dict[str, Any]] = []
    fake_remote = object.__new__(RemoteRoboCasaSession)

    def _fake_spawn(**kwargs: Any) -> Any:
        calls.append(kwargs)
        # Construct an independent sentinel object each time, used only to
        # assert "this branch was actually invoked" and "each slot is a
        # different instance," without needing to actually run a subprocess.
        return object()

    monkeypatch.setattr(
        session_process_module, "spawn_robocasa_subprocess", _fake_spawn
    )
    del fake_remote  # Unused, only illustrates the expected return-type shape

    core = RobocasaCurrentCore()
    core.build(_spec(process_isolation=True), num_envs=1, seed_offset=0)
    assert len(calls) == 1

    new_index = core.add_slot(seed_offset=0)
    assert new_index == 1
    assert len(calls) == 2, "add_slot must also go through spawn_robocasa_subprocess"

    sessions = [slot.session for slot in core._slots]
    assert len({id(session) for session in sessions}) == 2
    # Not an in-process RoboCasaSession: it never went through the
    # fake_ensure_environment branch.
    for kwargs in calls:
        assert kwargs["camera_size"] == core.config.camera_size
        assert kwargs["require_isolated_renderer"] == core.config.require_isolated_renderer
    core._slots = []  # Avoid close() trying to call close_environment on the sentinel objects
    core.closed = True


def test_add_slot_wraps_unexpected_construction_failures_as_env_failure(
    fake_ensure_environment, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A family construction failure must be normalized into
    ``RuntimeApiError(ENV_FAILURE)``, rather than letting the raw exception
    (e.g. ``RemoteSessionCrashed``/any ``BaseException``) pass through
    directly -- the caller ``EnvPool._cold_create_slot`` branches on
    ``RuntimeApiError``/``MemoryError``, and an unnormalized exception type
    would bypass this branching logic.
    """
    core = RobocasaCurrentCore()
    core.build(_spec(), num_envs=1, seed_offset=0)

    def _boom() -> Any:
        raise RuntimeError("simulated construction failure")

    monkeypatch.setattr(core, "_make_session", _boom)
    with pytest.raises(RuntimeApiError) as excinfo:
        core.add_slot(seed_offset=0)
    assert excinfo.value.info.code is ErrorCode.ENV_FAILURE
    assert core.slot_count() == 1, "a failed add_slot must not leave a partial slot"
    core.close()


def test_remove_slot_out_of_range_index_is_invalid_argument(
    fake_ensure_environment,
) -> None:
    core = RobocasaCurrentCore()
    core.build(_spec(), num_envs=1, seed_offset=0)
    with pytest.raises(RuntimeApiError) as excinfo:
        core.remove_slot(5)
    assert excinfo.value.info.code is ErrorCode.INVALID_ARGUMENT
    core.close()
