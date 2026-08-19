"""Critic-Recovery integration tests for ``rlinf_env.LiberoEnvCore`` (the
evaluation and interruption semantics of ``ResetSpec.options
["critic_rules"]`` → ``_chunk_step_one``).

Coverage:

1. Zero behavior difference when ``critic_rules`` is not supplied (no
   ``critic_proposals`` key in ``ChunkOutcome.info``, ``_evaluate_critic`` is
   skipped entirely);
2. When a rule hits, ``ChunkOutcome.info["critic_proposals"]`` carries the
   proposal, and ``interrupt_on_proposal`` (default true) breaks out of the
   physical action loop early;
3. When ``critic_interrupt_on_proposal=False``, a hit does not stop early
   either (it keeps running the full chunk, unless the environment itself
   terminates);
4. ``critic_previous_eef`` persists across physical action steps, and
   ``audit_trace`` accumulates step-by-step records;
5. Across episodes (reset), Critic state and history are rebuilt/cleared;
6. An invalid ``critic_rules`` payload reports ``INVALID_ARGUMENT`` at reset;
7. Configuring ``critic_rules`` on a ``lockstep_vector`` form reports
   ``INVALID_ARGUMENT`` (that form goes through the family-shared
   ``run_lockstep_chunk``, which does not go through the Critic evaluation
   hook).

No simulator is run: reuses the minimal ``rlinf`` stub from
``tests/runtime/libero_stub.py``. Field-by-field parity of real-hardware
behavior is covered by verification against the LIBERO-Pro + rlinf
environment on a remote Linux GPU host, which is not the responsibility of
this file.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from rollout_runtime.api.enums import ErrorCode
from rollout_runtime.api.errors import RuntimeApiError
from rollout_runtime.api.messages import ResetSpec
from rollout_runtime.backends.rlinf_env import LiberoEnvCore
from tests.runtime.libero_stub import (
    ACTION_DIM,
    IMAGE_SIZE,
    _StubLiberoEnv,
    stub_rlinf,
)

__all__ = ["stub_rlinf"]


def _build_core(env_config: dict[str, Any] | None = None, *, pool_size: int = 1) -> Any:
    """Build an already-``build``-ed ``LiberoEnvCore`` (mirrors
    ``test_env_family_normalization._build_core``; kept separate to avoid
    cross-file coupling).

    Args:
        env_config: Family-private configuration overrides.
        pool_size: Number of slots in the pool.

    Returns:
        The constructed ``LiberoEnvCore``.
    """
    from rollout_runtime.api.messages import EnvSpecMsg

    config = {
        "task_suite_name": "libero_10",
        "task_id": 0,
        "camera_height": IMAGE_SIZE,
        "camera_width": IMAGE_SIZE,
        "action_dim": ACTION_DIM,
        "chunk_size": 8,
        **(env_config or {}),
    }
    spec = EnvSpecMsg(env_family="libero", env_config=config, pool_size=pool_size)
    core = LiberoEnvCore()
    core.build(spec, num_envs=pool_size, seed_offset=0, total_num_processes=1)
    return core


_STAGNANT_RULE = {
    "rule_id": "eef-stagnant",
    "feature": "robot.eef.motion_m",
    "operator": "le",
    "threshold": 0.0,
    "dwell_steps": 1,
    "proposal": "eef did not move this step",
}
"""A rule the stub naturally satisfies since ``states`` has zero displacement
every step: the first step where real displacement can be computed is a
hit."""


def _action_block(chunk: int) -> np.ndarray:
    """Construct an action block of valid shape.

    Args:
        chunk: Chunk length.

    Returns:
        A float action block of shape ``[chunk, ACTION_DIM]``.
    """
    return np.tile(np.arange(ACTION_DIM, dtype=np.float32) + 1.0, (chunk, 1))


# ------------------------------------------------------------- without critic_rules


def test_no_critic_rules_means_no_critic_proposals_key(
    stub_rlinf: type[_StubLiberoEnv],
) -> None:
    """When ``critic_rules`` is not supplied, ``info`` has no
    ``critic_proposals`` key at all.

    Args:
        stub_rlinf: rlinf stub fixture.
    """
    core = _build_core()
    core.reset([0], ResetSpec(task_id=0, seed=0))
    outcome = core.chunk_step([0], [_action_block(3)])[0]
    assert "critic_proposals" not in outcome.info
    assert outcome.executed_horizon == 3
    core.close()


def test_empty_critic_rules_list_is_equivalent_to_absent(
    stub_rlinf: type[_StubLiberoEnv],
) -> None:
    """An empty list is equivalent to not passing it at all: Critic is not
    enabled either way.

    Args:
        stub_rlinf: rlinf stub fixture.
    """
    core = _build_core()
    core.reset([0], ResetSpec(task_id=0, seed=0, options={"critic_rules": []}))
    outcome = core.chunk_step([0], [_action_block(2)])[0]
    assert "critic_proposals" not in outcome.info
    core.close()


# --------------------------------------------------------------- hits and interruption


def test_hit_rule_populates_proposals_and_interrupts_by_default(
    stub_rlinf: type[_StubLiberoEnv],
) -> None:
    """When a rule hits, the proposal goes into ``info``, and by default the
    physical action loop breaks out early.

    The stub's ``states`` is a constant vector (every step equals the step
    index), so EEF displacement is identically 0 before the first step that
    "has a previous step to compare against" (the 2nd step within the
    chunk)—so ``_STAGNANT_RULE`` should in principle hit at step 1 (the first
    step has no ``previous_eef``, so ``robot.eef.delta_available`` is false,
    but the rule doesn't set it as an activation condition; directly
    evaluating the main operator against a missing ``motion_m`` would raise
    ``KeyError``; hence a stable field not dependent on ``previous_eef`` is
    used instead).

    Args:
        stub_rlinf: rlinf stub fixture.
    """
    core = _build_core()
    core.reset(
        [0],
        ResetSpec(
            task_id=0,
            seed=0,
            options={
                "critic_rules": [
                    {
                        "rule_id": "step-reached",
                        "feature": "episode.step_index",
                        "operator": "ge",
                        "threshold": 2,
                        "dwell_steps": 1,
                        "proposal": "reached step 2",
                    }
                ]
            },
        ),
    )
    outcome = core.chunk_step([0], [_action_block(5)])[0]
    # The rule hits at step 2 (step_index==2 is the second increment of
    # elapsed_steps in the stub), so the physical action loop should stop
    # early after step 2, rather than running the full 5 requested steps.
    assert outcome.executed_horizon == 2
    assert "critic_proposals" in outcome.info
    proposals = outcome.info["critic_proposals"]
    assert len(proposals) == 1
    assert proposals[0]["rule_id"] == "step-reached"
    assert proposals[0]["step_index"] == 2
    assert proposals[0]["environment_write"] is False
    assert outcome.info["critic_rule_count"] == 1
    core.close()


def test_interrupt_on_proposal_false_keeps_running_the_chunk(
    stub_rlinf: type[_StubLiberoEnv],
) -> None:
    """With ``critic_interrupt_on_proposal=False``, a hit does not stop
    execution early.

    Args:
        stub_rlinf: rlinf stub fixture.
    """
    core = _build_core()
    core.reset(
        [0],
        ResetSpec(
            task_id=0,
            seed=0,
            options={
                "critic_rules": [
                    {
                        "rule_id": "step-reached",
                        "feature": "episode.step_index",
                        "operator": "ge",
                        "threshold": 2,
                        "dwell_steps": 1,
                        "proposal": "reached step 2",
                    }
                ],
                "critic_interrupt_on_proposal": False,
            },
        ),
    )
    outcome = core.chunk_step([0], [_action_block(4)])[0]
    assert outcome.executed_horizon == 4
    proposals = outcome.info["critic_proposals"]
    # The rule keeps satisfying ge 2 from step 2 onward, but cooldown_steps
    # defaults to 0 and dwell_steps=1, so it hits again every step (steps 2,
    # 3, 4).
    assert [row["step_index"] for row in proposals] == [2, 3, 4]
    core.close()


def test_cooldown_suppresses_repeated_hits(
    stub_rlinf: type[_StubLiberoEnv],
) -> None:
    """``cooldown_steps`` suppresses repeated hits for the declared number of
    steps after a hit.

    Args:
        stub_rlinf: rlinf stub fixture.
    """
    core = _build_core()
    core.reset(
        [0],
        ResetSpec(
            task_id=0,
            seed=0,
            options={
                "critic_rules": [
                    {
                        "rule_id": "step-reached",
                        "feature": "episode.step_index",
                        "operator": "ge",
                        "threshold": 1,
                        "dwell_steps": 1,
                        "cooldown_steps": 2,
                        "proposal": "reached step 1",
                    }
                ],
                "critic_interrupt_on_proposal": False,
            },
        ),
    )
    outcome = core.chunk_step([0], [_action_block(5)])[0]
    proposals = outcome.info["critic_proposals"]
    # Hits at step 1 → cools down for 2 steps (steps 2, 3 don't count) →
    # hits again at step 4 → cools down → step 5 doesn't count.
    assert [row["step_index"] for row in proposals] == [1, 4]
    core.close()


# --------------------------------------------------------- persistence across chunks


def test_critic_state_persists_across_chunk_calls(
    stub_rlinf: type[_StubLiberoEnv],
) -> None:
    """``TemporalCritic`` and ``critic_previous_eef`` persist across multiple
    ``chunk_step`` calls.

    Args:
        stub_rlinf: rlinf stub fixture.
    """
    core = _build_core()
    core.reset(
        [0],
        ResetSpec(
            task_id=0,
            seed=0,
            options={
                "critic_rules": [
                    {
                        "rule_id": "step-reached",
                        "feature": "episode.step_index",
                        "operator": "ge",
                        "threshold": 3,
                        "dwell_steps": 1,
                        "proposal": "reached step 3",
                    }
                ]
            },
        ),
    )
    slot = core._slots[0]  # noqa: SLF001
    first = core.chunk_step([0], [_action_block(2)])[0]
    assert first.executed_horizon == 2
    assert "critic_proposals" in first.info
    assert first.info["critic_proposals"] == []
    assert slot.critic_previous_eef is not None
    assert len(slot.audit_trace) == 2

    second = core.chunk_step([0], [_action_block(2)])[0]
    # The second chunk starts hitting from step_index=3 and stops early.
    assert second.executed_horizon == 1
    assert [row["rule_id"] for row in second.info["critic_proposals"]] == [
        "step-reached"
    ]
    assert len(slot.audit_trace) == 3
    core.close()


def test_reset_rebuilds_critic_and_clears_history(
    stub_rlinf: type[_StubLiberoEnv],
) -> None:
    """A new episode (reset) rebuilds the Critic and clears
    ``audit_trace`` and ``critic_previous_eef``.

    Args:
        stub_rlinf: rlinf stub fixture.
    """
    core = _build_core()
    rules_payload = [
        {
            "rule_id": "step-reached",
            "feature": "episode.step_index",
            "operator": "ge",
            "threshold": 1,
            "dwell_steps": 1,
            "proposal": "reached step 1",
        }
    ]
    core.reset([0], ResetSpec(task_id=0, seed=0, options={"critic_rules": rules_payload}))
    core.chunk_step([0], [_action_block(2)])
    slot = core._slots[0]  # noqa: SLF001
    assert slot.audit_trace
    first_critic = slot.critic

    core.reset([0], ResetSpec(task_id=0, seed=1, options={"critic_rules": rules_payload}))
    assert slot.audit_trace == []
    assert slot.critic_previous_eef is None
    assert slot.critic is not first_critic
    assert slot.critic_history_pending_reset is True
    core.close()


def test_reset_without_critic_rules_disables_critic_after_previous_episode(
    stub_rlinf: type[_StubLiberoEnv],
) -> None:
    """If the previous episode configured a Critic but the next episode does
    not pass ``critic_rules``, it should be disabled.

    Args:
        stub_rlinf: rlinf stub fixture.
    """
    core = _build_core()
    rules_payload = [
        {
            "rule_id": "step-reached",
            "feature": "episode.step_index",
            "operator": "ge",
            "threshold": 1,
            "dwell_steps": 1,
            "proposal": "reached step 1",
        }
    ]
    core.reset([0], ResetSpec(task_id=0, seed=0, options={"critic_rules": rules_payload}))
    core.reset([0], ResetSpec(task_id=0, seed=1))
    outcome = core.chunk_step([0], [_action_block(2)])[0]
    assert "critic_proposals" not in outcome.info
    core.close()


# ------------------------------------------------------------------- validation


def test_invalid_critic_rules_payload_is_rejected_at_reset(
    stub_rlinf: type[_StubLiberoEnv],
) -> None:
    """A non-array ``critic_rules`` reports ``INVALID_ARGUMENT`` at reset.

    Args:
        stub_rlinf: rlinf stub fixture.
    """
    core = _build_core()
    with pytest.raises(RuntimeApiError) as excinfo:
        core.reset(
            [0],
            ResetSpec(task_id=0, seed=0, options={"critic_rules": {"not": "a list"}}),
        )
    assert excinfo.value.info.code is ErrorCode.INVALID_ARGUMENT
    core.close()


def test_critic_rule_missing_required_field_is_rejected(
    stub_rlinf: type[_StubLiberoEnv],
) -> None:
    """A rule missing a required field (e.g. ``feature``) reports
    ``INVALID_ARGUMENT`` at reset.

    Args:
        stub_rlinf: rlinf stub fixture.
    """
    core = _build_core()
    with pytest.raises(RuntimeApiError) as excinfo:
        core.reset(
            [0],
            ResetSpec(
                task_id=0,
                seed=0,
                options={"critic_rules": [{"rule_id": "no-feature"}]},
            ),
        )
    assert excinfo.value.info.code is ErrorCode.INVALID_ARGUMENT
    core.close()


def test_critic_rules_rejected_for_lockstep_vector_core_form(
    stub_rlinf: type[_StubLiberoEnv],
) -> None:
    """Configuring ``critic_rules`` on the ``lockstep_vector`` form must
    error at reset.

    That form goes through the three families' shared ``run_lockstep_chunk``,
    which does not go through ``_evaluate_critic``; configuring rules that
    never take effect is a silent misuse that must be rejected rather than
    quietly ignored.

    Args:
        stub_rlinf: rlinf stub fixture.
    """
    core = _build_core({"core_form": "lockstep_vector", "libero_variant": "standard"})
    with pytest.raises(RuntimeApiError) as excinfo:
        core.reset(
            [0],
            ResetSpec(
                task_id=0,
                seed=0,
                options={
                    "critic_rules": [
                        {
                            "rule_id": "step-reached",
                            "feature": "episode.step_index",
                            "operator": "ge",
                            "threshold": 1,
                        }
                    ]
                },
            ),
        )
    assert excinfo.value.info.code is ErrorCode.INVALID_ARGUMENT
    core.close()
