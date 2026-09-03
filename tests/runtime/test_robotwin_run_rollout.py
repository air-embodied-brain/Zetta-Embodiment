# Copyright (c) 2026 Zetta Contributors
"""The RoboTwin rollout entrypoint, driven against a fake Runtime client.

No simulator and no HTTP: the point is the wiring the campaign depends on --
that the episode seed reaches the family as a **reset-state id** (RoboTwin's
seed *is* its scene), that the chunk loop tolerates a ``final_only`` family
returning no per-step records, and that the resulting ``EpisodeRecord`` carries
the digests a manifest is checked against.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from robots.robotwin.run_rollout import build_parser, run
from robots.robotwin.tool_bindings import binding_for_task
from robots.robotwin.tool_catalog import DEFAULT_ROBOTWIN_TOOL_CATALOG
from rollout_runtime.api.ids import EpisodeId, RequestId, SessionId
from rollout_runtime.api.messages import Observation, StepResult
from rollout_runtime.api.result import Ok
from rollout_runtime.core.payload import decode_array, encode_array, encode_image

ENV_SPEC_DIGEST = "digest-robotwin-test"


class _FakeClient:
    """A Runtime client that ends the episode after a fixed number of chunks."""

    def __init__(
        self,
        *,
        chunks_until_done: int = 3,
        succeed: bool = True,
        with_images: bool = False,
    ) -> None:
        """Configure the fake.

        Args:
            chunks_until_done: How many chunks before termination is reported.
            succeed: Whether the final chunk reports success.
            with_images: Whether observations carry the three camera views.
        """
        self.chunks_until_done = chunks_until_done
        self.succeed = succeed
        self.with_images = with_images
        self.created: list[Any] = []
        self.reset_specs: list[Any] = []
        self.policy_requests: list[Any] = []
        self.chunk_calls = 0
        self.infer_calls = 0
        self.action_steps: list[Any] = []
        self.closed = False

    async def create_sessions(self, requests):
        """Record the request and hand back a session handle."""
        self.created.extend(requests)
        return [
            Ok(
                value=SimpleNamespace(
                    session_id=SessionId("session-1"),
                    env_spec_digest=ENV_SPEC_DIGEST,
                    lease_expiration=1e12,
                )
            )
        ]

    def _observation(self) -> Observation:
        """Build a chunk-final observation carrying a state and three views."""
        state = [0.01 * self.chunk_calls] * 14
        frame = np.full((4, 6, 3), self.chunk_calls % 256, dtype=np.uint8)
        images = (
            {
                "main_image": encode_image(frame),
                "wrist_image": encode_image(frame),
                "extra_view_images": [encode_image(frame)],
            }
            if self.with_images
            else {}
        )
        return Observation(
            session_id=SessionId("session-1"),
            episode_id=EpisodeId(0),
            step_index=self.chunk_calls,
            state=state,
            instruction="lift the bottle with the correct arm",
            **images,
        )

    async def policy_infer(self, ids, policy_request):
        """Return a [chunk, 14] proposal without touching the environment."""
        self.policy_requests.append(policy_request)
        self.infer_calls += 1
        block = np.full((2, 14), 0.05, dtype=np.float32)
        return [
            Ok(
                value=SimpleNamespace(
                    actions=encode_array(block), model_version="pi05-test"
                )
            )
        ]

    async def action_step(self, ids, actions):
        """Execute a chunk Role1 already reviewed.

        ``executed_horizon`` reports the number of actions actually submitted,
        which is what the real adapter does: the env executes what it is given,
        capped by ``execute_horizon``. Reporting a fixed number here instead
        would let the rollout's action step indices drift from its chunk step
        indices without any test noticing.
        """
        self.action_steps.append(actions)
        self.chunk_calls += 1
        submitted = int(np.asarray(decode_array(actions[0])).shape[0])
        done = self.chunk_calls >= self.chunks_until_done
        return [
            Ok(
                value=StepResult(
                    request_id=RequestId(f"act-{self.chunk_calls}"),
                    session_id=ids[0],
                    observation=self._observation(),
                    reward=0.0,
                    terminated=done,
                    executed_horizon=submitted,
                    per_step=None,
                    info={
                        "executed_horizon": submitted,
                        "requested_horizon": 50,
                        "success": bool(done and self.succeed),
                    },
                )
            )
        ]

    async def reset(self, ids, reset_spec):
        """Record the reset spec and return the initial frame."""
        self.reset_specs.append(reset_spec)
        return [
            Ok(
                value=StepResult(
                    request_id=RequestId("reset-1"),
                    session_id=ids[0],
                    observation=self._observation(),
                    info={},
                )
            )
        ]

    async def policy_step(self, ids, policy_request):
        """Return a chunk-granular step result, never a per-step one."""
        self.policy_requests.append(policy_request)
        self.chunk_calls += 1
        done = self.chunk_calls >= self.chunks_until_done
        return [
            Ok(
                value=StepResult(
                    request_id=RequestId(f"chunk-{self.chunk_calls}"),
                    session_id=ids[0],
                    observation=self._observation(),
                    reward=1.0 if done and self.succeed else 0.0,
                    terminated=done,
                    executed_horizon=25,
                    # final_only: the family produces no intermediate frames.
                    per_step=None,
                    info={
                        "executed_horizon": 25,
                        "requested_horizon": 50,
                        "discarded_actions": 25,
                        "success": bool(done and self.succeed),
                    },
                )
            )
        ]

    async def renew_sessions(self, ids, lease_seconds):
        """Renew the lease."""
        return [Ok(value=SimpleNamespace(lease_expiration=1e12))]

    async def close_sessions(self, ids):
        """Close the session."""
        self.closed = True
        return [Ok(value={"session_closed": True})]

    async def aclose(self) -> None:
        """No-op; the fake owns no connection pool."""


def _args(tmp_path: Path, **overrides: Any) -> argparse.Namespace:
    """Parse a minimal valid argument set.

    Args:
        tmp_path: Output directory.
        **overrides: Attribute overrides applied after parsing.

    Returns:
        The namespace.
    """
    argv = [
        "--runtime-url",
        "http://runtime.invalid",
        "--policy-id",
        "pi05_aloha_robotwin",
        "--assets-path",
        "/workspace/RoboTwin",
        "--seed",
        "100100000",
        "--logical-id",
        "adjust-bottle-0",
        "--output-dir",
        str(tmp_path),
    ]
    args = build_parser().parse_args(argv)
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def test_episode_seed_reaches_the_family_as_a_reset_state_id(tmp_path: Path) -> None:
    """RoboTwin's seed *is* the scene, so it must be pinned, not merely seeded.

    A paired same-seed gate only reproduces if the scene id is explicit; passing
    the value as ``seed`` alone would let the family fall back to its own
    success-seed rotation.
    """
    client = _FakeClient()
    asyncio.run(run(_args(tmp_path), client=client))

    spec = client.reset_specs[0]
    assert spec.reset_state_id == 100100000
    assert spec.seed == 100100000


def test_chunk_loop_tolerates_a_family_with_no_per_step_records(
    tmp_path: Path,
) -> None:
    """``final_only`` means ``per_step`` is None; the audit says so explicitly."""
    client = _FakeClient(chunks_until_done=3)
    record = asyncio.run(run(_args(tmp_path), client=client))

    rows = [
        json.loads(line)
        for line in (tmp_path / "chunks.jsonl").read_text().splitlines()
    ]
    assert len(rows) == 3
    assert all(row["per_step_available"] is False for row in rows)
    assert all(row["evidence_granularity"] == "chunk" for row in rows)
    assert rows[0]["executed_horizon"] == 25
    assert rows[0]["discarded_actions"] == 25
    assert record.artifact_index["chunks_executed"] == 3
    assert record.artifact_index["steps_executed"] == 75


def test_success_is_taken_from_the_family_info(tmp_path: Path) -> None:
    """Success is the family's authoritative flag, not a reward threshold."""
    ok = asyncio.run(run(_args(tmp_path / "ok"), client=_FakeClient(succeed=True)))
    assert ok.success is True
    assert ok.status == "valid"

    bad = asyncio.run(run(_args(tmp_path / "bad"), client=_FakeClient(succeed=False)))
    assert bad.success is False


def test_record_carries_the_frozen_digests(tmp_path: Path) -> None:
    """A campaign checks the manifest against exactly these values."""
    record = asyncio.run(run(_args(tmp_path), client=_FakeClient()))
    index = record.artifact_index
    assert index["env_spec_digest"] == ENV_SPEC_DIGEST
    assert index["tool_catalog_digest"] == DEFAULT_ROBOTWIN_TOOL_CATALOG.digest
    assert index["tool_binding_digest"] == binding_for_task("adjust_bottle").digest
    assert index["arm_scoped_tools"] == list(
        binding_for_task("adjust_bottle").arm_scoped_tool_names
    )
    # Recorded on every episode so a chunk-granular diagnosis is never compared
    # like-for-like with a per-step family's.
    assert index["evidence_granularity"] == "chunk"


def test_env_spec_carries_the_pool_key_fields(tmp_path: Path) -> None:
    """Every field here enters the pool digest; a mismatch cold-starts SAPIEN."""
    client = _FakeClient()
    asyncio.run(run(_args(tmp_path, execute_horizon=25), client=client))

    env_spec = client.created[0].env_spec
    assert env_spec.env_family == "robotwin"
    assert env_spec.env_config["task_name"] == "adjust_bottle"
    assert env_spec.env_config["execute_horizon"] == 25
    assert env_spec.env_config["collect_wrist_camera"] is True
    assert env_spec.resource_hints == {"accelerator": True}
    # The per-episode seed must NOT be in the spec: it would split the pool.
    assert "seed" not in env_spec.env_config


def test_session_is_always_closed(tmp_path: Path) -> None:
    """Closing is what returns the env slot to the pool for the next rollout."""
    client = _FakeClient()
    asyncio.run(run(_args(tmp_path), client=client))
    assert client.closed is True


def test_horizon_cap_stops_a_runaway_episode(tmp_path: Path) -> None:
    """A family that never terminates must still be bounded by max steps."""
    client = _FakeClient(chunks_until_done=10_000)
    record = asyncio.run(run(_args(tmp_path, env_max_steps=100), client=client))
    # 100 steps at 25 per chunk.
    assert client.chunk_calls == 4
    assert record.success is False


def test_cli_rejects_a_zero_execute_horizon() -> None:
    """A zero horizon would submit an empty chunk forever."""
    args = build_parser().parse_args(
        [
            "--runtime-url",
            "u",
            "--policy-id",
            "p",
            "--assets-path",
            "a",
            "--seed",
            "1",
            "--logical-id",
            "l",
            "--output-dir",
            "o",
            "--execute-horizon",
            "0",
        ]
    )
    assert args.execute_horizon == 0  # parsed; rejected by main()


# ------------------------------------------------------- the reviewed path


def _rules_file(tmp_path: Path, rules: list[dict]) -> str:
    """Write a frozen rule list to disk.

    Args:
        tmp_path: Test directory.
        rules: The rules.

    Returns:
        The path as a string.
    """
    path = tmp_path / "rules.json"
    path.write_text(json.dumps(rules), encoding="utf-8")
    return str(path)


STALL_RULE = {
    "rule_id": "left-arm-stalled",
    "title": "Left arm has not moved",
    "feature": "robotwin.arm.left.stalled_chunks",
    "operator": "ge",
    "threshold": 1,
    "dwell_steps": 1,
    "cooldown_steps": 0,
    "proposal": "hold the left arm",
    "evidence_ids": ["robotwin.arm.left.joint_motion"],
}


def test_without_critic_rules_the_path_stays_pure_vla(tmp_path: Path) -> None:
    """The baseline must remain one atomic policy_step per chunk."""
    client = _FakeClient()
    record = asyncio.run(run(_args(tmp_path), client=client))
    assert client.infer_calls == 0
    assert client.action_steps == []
    assert record.artifact_index["baseline_mode"] == "pure_vla"
    assert record.artifact_index["critic_rule_count"] == 0


def test_critic_rules_switch_to_the_reviewed_path(tmp_path: Path) -> None:
    """With rules, nothing reaches the simulator before Role1 has ruled."""
    client = _FakeClient(chunks_until_done=2)
    record = asyncio.run(
        run(
            _args(tmp_path, critic_rules=_rules_file(tmp_path, [STALL_RULE])),
            client=client,
        ),
    )
    assert client.infer_calls == 2
    assert len(client.action_steps) == 2
    assert record.artifact_index["baseline_mode"] == "critic_reviewed"
    assert record.artifact_index["critic_rule_count"] == 1

    rows = [
        json.loads(line)
        for line in (tmp_path / "chunks.jsonl").read_text().splitlines()
    ]
    assert all(row["source"] in {"vla", "recovery"} for row in rows)


def test_role1_decisions_are_persisted(tmp_path: Path) -> None:
    """A verdict that never reaches the audit trail cannot be reviewed later."""
    client = _FakeClient(chunks_until_done=3)
    asyncio.run(
        run(
            _args(tmp_path, critic_rules=_rules_file(tmp_path, [STALL_RULE])),
            client=client,
        ),
    )
    decisions = tmp_path / "role1_decisions.jsonl"
    assert decisions.exists()
    rows = [json.loads(line) for line in decisions.read_text().splitlines()]
    assert rows, "a firing critic rule must produce a recorded verdict"
    # The reference Role1 rejects an arm-less proposal, and the stall rule does
    # not name one -- so the recorded verdict must say exactly that.
    assert rows[0]["accepted"] is False
    assert "names no arm" in rows[0]["reason"]


def test_record_states_the_dwell_unit(tmp_path: Path) -> None:
    """``dwell_steps`` counts chunks here and simulator steps elsewhere.

    A reader of the record must not have to infer which one a frozen rule meant.
    """
    record = asyncio.run(
        run(_args(tmp_path, execute_horizon=25), client=_FakeClient()),
    )
    semantics = record.artifact_index["dwell_semantics"]
    assert semantics["dwell_unit"] == "chunk"
    assert semantics["sim_steps_per_chunk"] == 25


# --------------------------------------------------- trajectory artifacts


def _rows(path: Path) -> list[dict]:
    """Read a JSONL artifact.

    Args:
        path: The file.

    Returns:
        Its rows.
    """
    text = path.read_text()
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def test_all_four_trajectory_artifacts_exist(tmp_path: Path) -> None:
    """``_strict_jsonl`` reads all four and raises on a missing file.

    A run that records nothing of a given kind must still leave an empty
    artifact, or indexing fails before it can look at anything.
    """
    asyncio.run(run(_args(tmp_path), client=_FakeClient()))
    for name in ("chunks", "actions", "states", "tools"):
        assert (tmp_path / f"{name}.jsonl").exists(), f"{name}.jsonl missing"


def test_failed_episode_carries_a_failure_segment(tmp_path: Path) -> None:
    """Without segments the Cluster stage stalls silently rather than failing.

    ``cluster_failure_segments([])`` returns ``[]``, so an episode that reaches
    the campaign with no segments produces no cluster and no error -- the
    hardest kind of gap to notice.
    """
    record = asyncio.run(
        run(_args(tmp_path), client=_FakeClient(succeed=False)),
    )
    assert record.success is False
    assert record.failure_segments, "a failed episode must localise something"
    assert record.failure_segment is not None
    assert record.artifact_index["failure_segment_count"] == len(
        record.failure_segments
    )
    assert record.artifact_index["trajectory_index"] is not None


def test_gen0_failure_falls_back_to_horizon_incomplete(tmp_path: Path) -> None:
    """The honest current limit: no progress scalar means no causal localisation.

    A pure-VLA RoboTwin failure has no critic reject, no tool error and no Role1
    contract failure, and ``_window_no_progress`` keys off feature names
    (``task_progress`` / ``residual_to_success`` / ...) that this family cannot
    supply without a privileged simulator feature it does not declare. So every
    such failure lands on ``_fallback_incomplete``.

    That is a real limitation, not a passing detail: it produces one cluster at
    100% prevalence, which is exactly what ``trajectory._segments`` warns hides
    the actionable failure modes. This test pins the current behaviour so the
    day a progress feature is added, it fails and has to be re-read.
    """
    record = asyncio.run(
        run(_args(tmp_path), client=_FakeClient(succeed=False)),
    )
    classes = {segment.failure_class for segment in record.failure_segments}
    assert classes == {"horizon_incomplete"}
    assert record.failure_segments[0].earliest_divergence_step is None


def test_successful_episode_carries_no_segments(tmp_path: Path) -> None:
    """``EpisodeRecord`` rejects a success that also claims a failure."""
    record = asyncio.run(run(_args(tmp_path), client=_FakeClient(succeed=True)))
    assert record.success is True
    assert record.failure_segments == ()
    assert record.failure_segment is None


def test_action_count_survives_an_empty_actions_artifact(tmp_path: Path) -> None:
    """The fast path never sees individual actions; the count must still be right.

    ``policy_step`` does inference and execution inside the Runtime in one
    operation, so ``actions.jsonl`` is empty on the pure-VLA baseline.
    ``_make_events`` derives the count from the chunk rows' ``executed_horizon``
    instead.
    """
    client = _FakeClient(chunks_until_done=3)
    record = asyncio.run(run(_args(tmp_path), client=client))
    assert _rows(tmp_path / "actions.jsonl") == []
    index = record.artifact_index["trajectory_index"]
    assert index["action_count"] == 75  # 3 chunks x executed_horizon 25


def test_reviewed_path_records_every_executed_action(tmp_path: Path) -> None:
    """With Role1 in the loop the client owns the chunk, so it can record it."""
    client = _FakeClient(chunks_until_done=2)
    asyncio.run(
        run(
            _args(tmp_path, critic_rules=_rules_file(tmp_path, [STALL_RULE])),
            client=client,
        ),
    )
    actions = _rows(tmp_path / "actions.jsonl")
    assert actions, "the reviewed path must record its actions"
    assert all(row["action_dim"] == 14 for row in actions)
    # Step indices are contiguous across chunks, not restarted per chunk.
    assert [row["step_index"] for row in actions] == list(range(len(actions)))


def test_state_rows_are_chunk_final_and_say_so(tmp_path: Path) -> None:
    """``final_only`` means the state timeline is chunk-granular, not per-step.

    A reader counting rows must not conclude the episode was 3 steps long.
    """
    client = _FakeClient(chunks_until_done=3)
    asyncio.run(run(_args(tmp_path), client=client))
    states = _rows(tmp_path / "states.jsonl")
    assert len(states) == 3
    assert all(row["evidence_granularity"] == "chunk" for row in states)
    assert [row["step_index"] for row in states] == [25, 50, 75]
    assert len(states[0]["state"]["joint_positions"]) == 14
    assert set(states[0]["state"]) == {
        "joint_positions",
        "robotwin.state_available",
        "robotwin.arm.left.gripper",
        "robotwin.arm.right.gripper",
    }


def test_state_rows_publish_no_fabricated_progress_scalar(tmp_path: Path) -> None:
    """Joint travel is not task progress, and must not be labelled as it.

    ``_window_no_progress`` keys off these exact names. Publishing proprioception
    under one of them would make "the robot stopped moving" indistinguishable
    from "the task stopped progressing" in every downstream diagnosis.
    """
    asyncio.run(run(_args(tmp_path), client=_FakeClient()))
    forbidden = ("task_progress", "progress", "completion")
    for row in _rows(tmp_path / "states.jsonl"):
        flat = json.dumps(row)
        assert not any(f'"{name}"' in flat for name in forbidden)
        assert "residual_to_success" not in flat
        assert "distance_to_goal" not in flat


def test_tool_events_record_the_action_source(tmp_path: Path) -> None:
    """The tool timeline is what tells a diagnosis which actor drove a step."""
    asyncio.run(run(_args(tmp_path), client=_FakeClient(chunks_until_done=2)))
    tools = _rows(tmp_path / "tools.jsonl")
    assert len(tools) == 2
    assert all(row["source"] == "vla" for row in tools)
    assert all(row["environment_write"] is True for row in tools)


# ------------------------------------------------- per-step frame capture


def test_capture_switches_the_baseline_off_the_atomic_path(tmp_path: Path) -> None:
    """Capturing requires seeing every step, which ``policy_step`` cannot give.

    The atomic operation does inference and execution inside the Runtime, so
    there is no per-step boundary to observe at. Capturing therefore splits it
    into ``policy_infer`` + one ``action_step`` per action -- which Experiment A
    established is physically identical.
    """
    plain = _FakeClient(chunks_until_done=2)
    asyncio.run(run(_args(tmp_path / "plain"), client=plain))
    assert plain.infer_calls == 0
    assert plain.action_steps == []

    captured = _FakeClient(chunks_until_done=1_000, with_images=True)
    asyncio.run(
        run(
            _args(tmp_path / "cap", capture_frames=True, env_max_steps=4),
            client=captured,
        ),
    )
    assert captured.infer_calls > 0
    # One action_step per action, not per chunk.
    assert len(captured.action_steps) == 4


def test_captured_states_are_per_step_not_per_chunk(tmp_path: Path) -> None:
    """The evidence machinery's windows are counted in rows, so rows must be steps.

    ``index_episode_trajectory`` defaults ``context_before``/``context_after``/
    ``no_progress_window`` to 8 rows; at chunk granularity that spans a whole
    episode and localisation degenerates.
    """
    client = _FakeClient(chunks_until_done=1_000, with_images=True)
    asyncio.run(
        run(
            _args(tmp_path, capture_frames=True, env_max_steps=6),
            client=client,
        ),
    )
    states = _rows(tmp_path / "states.jsonl")
    assert len(states) == 6
    assert [row["step_index"] for row in states] == [1, 2, 3, 4, 5, 6]


def test_chunk_row_reports_the_whole_block_not_the_last_sub_step(
    tmp_path: Path,
) -> None:
    """``_make_events`` derives ``action_count`` from this field.

    Stepwise submission makes the final sub-step report ``executed_horizon=1``;
    recording that as the chunk's horizon would undercount the episode by the
    block length.
    """
    client = _FakeClient(chunks_until_done=1_000, with_images=True)
    record = asyncio.run(
        run(
            _args(tmp_path, capture_frames=True, env_max_steps=4, execute_horizon=2),
            client=client,
        ),
    )
    chunks = _rows(tmp_path / "chunks.jsonl")
    assert [row["executed_horizon"] for row in chunks] == [2, 2]
    assert record.artifact_index["trajectory_index"]["action_count"] == 4


def test_videos_are_encoded_from_the_observation_frames(tmp_path: Path) -> None:
    """RoboTwin records nothing, so the only frames are the ones the policy saw."""
    pytest.importorskip("imageio")
    client = _FakeClient(chunks_until_done=1_000, with_images=True)
    record = asyncio.run(
        run(
            _args(tmp_path, capture_frames=True, env_max_steps=6),
            client=client,
        ),
    )
    videos = record.artifact_index["videos"]
    assert set(videos) == {"head", "left_wrist", "right_wrist"}
    for path in videos.values():
        assert Path(path).is_file()
        assert Path(path).stat().st_size > 0


def test_visual_evidence_is_built_when_three_cameras_were_recorded(
    tmp_path: Path,
) -> None:
    """Stage 1 refuses a diagnosis with fewer than three evidence items."""
    pytest.importorskip("imageio")
    client = _FakeClient(chunks_until_done=1_000, with_images=True, succeed=False)
    record = asyncio.run(
        run(
            _args(tmp_path, capture_frames=True, env_max_steps=12),
            client=client,
        ),
    )
    evidence = record.artifact_index["visual_evidence"]
    assert evidence is not None
    assert evidence.get("available") is not False, evidence.get("reason")


def test_missing_cameras_degrade_instead_of_failing_the_episode(
    tmp_path: Path,
) -> None:
    """Losing evidence must not turn a valid episode into an infra failure.

    The record still has to say *why* it cannot be diagnosed, or the gap only
    surfaces much later inside Stage 1.
    """
    client = _FakeClient(chunks_until_done=1_000, with_images=False)
    record = asyncio.run(
        run(
            _args(tmp_path, capture_frames=True, env_max_steps=4),
            client=client,
        ),
    )
    assert record.status == "valid"
    evidence = record.artifact_index["visual_evidence"]
    assert evidence["available"] is False
    assert "two synchronized cameras" in evidence["reason"]


def test_no_capture_reports_why_there_is_no_evidence(tmp_path: Path) -> None:
    """A throughput baseline is still a valid episode, just not a diagnosable one."""
    record = asyncio.run(run(_args(tmp_path), client=_FakeClient()))
    evidence = record.artifact_index["visual_evidence"]
    assert evidence["available"] is False
    assert "--capture-frames" in evidence["reason"]


def test_the_campaign_none_sentinel_is_not_a_digest(tmp_path: Path) -> None:
    """``campaign.py`` substitutes ``{bundle_sha256}`` with ``"none"`` in Gen0.

    Passing it straight through makes ``EpisodeRecord`` reject the episode as
    malformed, which the campaign then counts as ``infra_invalid`` -- so every
    Gen0 rollout fails and the campaign blocks on infrastructure. Observed on a
    real campaign run before this was fixed.
    """
    record = asyncio.run(
        run(_args(tmp_path, bundle_sha256="none"), client=_FakeClient()),
    )
    assert record.bundle_sha256 is None

    digest = "a" * 64
    kept = asyncio.run(
        run(
            _args(tmp_path / "kept", bundle_sha256=digest),
            client=_FakeClient(),
        ),
    )
    assert kept.bundle_sha256 == digest


# --- paired-gate reset identity ---------------------------------------------
#
# Observed on a real campaign: with this absent, Diagnose and Stage2 both
# completed and the run then died at the same-seed gate with
# "gate parent source lacks reset observation identity"
# (gate_runner._source_parent_for_pair), after the rollout budget was spent.


def test_record_carries_the_reset_observation_identity(tmp_path: Path) -> None:
    """A same-seed gate refuses a parent that cannot prove its reset."""
    record = asyncio.run(
        run(_args(tmp_path), client=_FakeClient(with_images=True))
    )

    identity = record.artifact_index["initial_observation_identity"]
    # gating._same_physical_reset raises unless the state digest is a non-empty
    # string; camera digests are audited for drift but never decide the match.
    assert isinstance(identity["state_sha256"], str)
    assert identity["state_sha256"]
    assert set(identity["camera_sha256"]) == {"head", "left_wrist", "right_wrist"}


def test_reset_identity_does_not_depend_on_cameras(tmp_path: Path) -> None:
    """The binding is the physical state; cameras are evidence, not identity.

    A reset observation that carries no frames must still produce a usable
    identity, because ``_same_physical_reset`` decides on ``state_sha256``
    alone and only counts camera drift.
    """
    record = asyncio.run(
        run(_args(tmp_path), client=_FakeClient(with_images=False))
    )

    identity = record.artifact_index["initial_observation_identity"]
    assert identity["state_sha256"]
    assert identity["camera_sha256"] == {}


def test_reset_identity_is_stable_for_the_same_seed(tmp_path: Path) -> None:
    """The seed is the scene, so two rollouts of it must bind identically."""
    first = asyncio.run(run(_args(tmp_path / "a"), client=_FakeClient()))
    second = asyncio.run(run(_args(tmp_path / "b"), client=_FakeClient()))

    assert (
        first.artifact_index["initial_observation_identity"]["state_sha256"]
        == second.artifact_index["initial_observation_identity"]["state_sha256"]
    )


# --- frozen candidate bundle -------------------------------------------------


def _bundle_payload() -> dict[str, Any]:
    """A minimal bundle whose single critic rule enables the reviewed path."""
    return {
        "candidate_id": "candidate-1",
        "generation": 1,
        "parent_sha256": None,
        "diagnosis_sha256": "0" * 64,
        "causal_hypothesis": "the gripper never closes at contact",
        "mechanism_change": "close the active gripper once contact is confirmed",
        "validation_plan": "paired same-seed trial",
        "critic_rules": [
            {
                "rule_id": "gripper-open",
                "title": "gripper stays open at contact",
                # A real feature name: extract_robotwin_critic_features mirrors
                # the plane per arm under "robotwin.arm.<arm>.".
                "feature": "robotwin.arm.left.gripper",
                "operator": "gt",
                "threshold": 0.9,
                "dwell_steps": 1,
                "cooldown_steps": 0,
                "proposal": "close the left gripper",
                "evidence_ids": ["artifact-1"],
            }
        ],
        # CandidateBundle.__post_init__ rejects a critic rule with no
        # executable recovery, so the pair is the smallest legal bundle.
        "recovery_rules": [
            {
                "recovery_id": "close-gripper",
                "title": "close the active gripper",
                "trigger_rule_ids": ["gripper-open"],
                "precondition": "contact is confirmed",
                "steps": [
                    {
                        "tool": "robotwin.gripper.set",
                        "parameters": {"arm": "left", "opening": 0.0},
                        "stop_when": "the gripper reports closed",
                    }
                ],
                "safety_constraints": ["never move the idle arm"],
                "stop_condition": "the bottle is retained",
                "fallback": "hold both arms",
                "evidence_ids": ["artifact-1"],
            }
        ],
    }


def test_bundle_supplies_the_rules_for_a_gate_arm(tmp_path: Path) -> None:
    """A gate arm is pinned to the frozen bundle, not to loose rule files."""
    from zetta.evolution.models import CandidateBundle

    # The campaign writes CandidateBundle.as_dict() and digests that, not the
    # raw authored JSON; the two differ by the model's defaults.
    bundle = CandidateBundle.from_dict(_bundle_payload())
    path = tmp_path / "bundle.json"
    path.write_text(json.dumps(bundle.as_dict()))
    args = _args(tmp_path / "run")
    args.bundle = str(path)
    args.bundle_sha256 = bundle.sha256

    record = asyncio.run(run(args, client=_FakeClient()))

    assert record.artifact_index["critic_rule_count"] == 1
    assert record.artifact_index["baseline_mode"] == "critic_reviewed"


def test_bundle_digest_is_verified(tmp_path: Path) -> None:
    """A bundle that is not the one the campaign froze must not run."""
    path = tmp_path / "bundle.json"
    path.write_text(json.dumps(_bundle_payload()))
    args = _args(tmp_path / "run")
    args.bundle = str(path)
    args.bundle_sha256 = "1" * 64

    with pytest.raises(ValueError, match="does not match --bundle-sha256"):
        asyncio.run(run(args, client=_FakeClient()))


def test_bundle_and_loose_rule_files_are_mutually_exclusive(tmp_path: Path) -> None:
    """Two rule provenances would make the gate's frozen behaviour ambiguous."""
    path = tmp_path / "bundle.json"
    path.write_text(json.dumps(_bundle_payload()))
    rules = tmp_path / "rules.json"
    rules.write_text("[]")
    args = _args(tmp_path / "run")
    args.bundle = str(path)
    args.critic_rules = str(rules)

    with pytest.raises(ValueError, match="exclusive"):
        asyncio.run(run(args, client=_FakeClient()))


# --- published vocabulary vs the evaluated plane -----------------------------
#
# Observed on a real campaign: states.jsonl published a nested
# {"left": {"gripper": ...}}, lifecycle._observed_critic_features flattened it
# to "left.gripper", Stage2 bound that name, and every same-seed gate rollout
# then died with "critic feature is unavailable: left.gripper" -- the Critic
# evaluates "robotwin.arm.left.gripper".  Two names for one quantity.


def test_published_vocabulary_is_resolvable_by_the_runtime_critic(
    tmp_path: Path,
) -> None:
    """Every name Stage2 can bind must be one the Critic can evaluate."""
    from robots.robotwin.critic_runtime import extract_robotwin_critic_features
    from zetta.evolution.lifecycle import _scalar_feature_names

    asyncio.run(run(_args(tmp_path), client=_FakeClient()))
    rows = _rows(tmp_path / "states.jsonl")

    published: set[str] = set()
    for row in rows:
        published.update(_scalar_feature_names(row["state"]))
    assert published, "states.jsonl offered Stage2 no feature at all"

    evaluable = set(
        extract_robotwin_critic_features(
            {"state": rows[0]["state"]["joint_positions"]},
            chunk_index=0,
            executed_horizon=25,
            reward=0.0,
            terminated=False,
            truncated=False,
        )
    )
    assert published <= evaluable, sorted(published - evaluable)


def test_record_attests_whether_the_learned_path_intervened(tmp_path: Path) -> None:
    """Without this the same-seed gate can never pass, and says so wrongly.

    ``gating._candidate_intervened`` falls back to a ``"role1:"`` key prefix that
    this family does not use, so an absent attestation reads as "no
    intervention"; ``mechanism_diverged`` is then always False and the gate
    rejects with "candidate never changed the failed parent action trajectory".
    """
    record = asyncio.run(run(_args(tmp_path), client=_FakeClient()))

    # A pure-VLA Gen0 episode genuinely did not intervene.
    assert record.artifact_index["candidate_intervention"] is False
    # The key must exist rather than be inferred, which is what the fallback
    # would otherwise do from a key this family never writes.
    assert not any(
        str(key).startswith("role1:") for key in record.artifact_index
    )
