"""Batch evaluation semantics for ``adapters/eval_adapter.py``.

Assertion focus:

- It is a **thin wrapper** around ``run_episode``: termination conditions
  all come from the worker, and the adapter itself never counts steps;
- A temporary session is **closed as soon as it finishes**, so ``pool_size``
  slots can be reused by later batches (the pool never grows);
- ``pool_size`` is automatically raised to the concurrency level, otherwise
  the ``pool_size + 1``-th session gets ``QUOTA_EXCEEDED``;
- The success rate is judged only by the environment's termination signal,
  with **valid / invalid episodes tallied separately**;
- The transition genuinely reaches the sink.
"""

from __future__ import annotations

from typing import Any

from rollout_runtime.adapters.eval_adapter import (
    EvaluationAdapter,
    EvaluationTask,
)
from rollout_runtime.api.messages import EnvSpecMsg
from rollout_runtime.launch.local import build_local_components
from tests.runtime.conftest import local_runtime_config

CONCURRENCY = 4


def eval_env_spec(pool_size: int = 1, **overrides: Any) -> EnvSpecMsg:
    """Build a fake env spec (``pool_size`` is deliberately small, to see
    whether the adapter raises it itself).

    Args:
        pool_size: The declared pool capacity.
        **overrides: Overrides for ``env_config``.

    Returns:
        The env spec.
    """
    config: dict[str, Any] = {
        "action_dim": 7,
        "chunk_size": 2,
        "episode_length": 4,
        "image_height": 8,
        "image_width": 8,
        "state_dim": 8,
    }
    config.update(overrides)
    return EnvSpecMsg(env_family="fake", env_config=config, pool_size=pool_size)


def tasks(count: int) -> list[EvaluationTask]:
    """Build an N-cell task list (same task, different seeds).

    Args:
        count: Number of cells.

    Returns:
        The task list.
    """
    return [EvaluationTask(task_id=0, seed=seed) for seed in range(count)]


def test_pool_size_is_raised_to_the_concurrency() -> None:
    """Corollary: running N concurrent sessions requires declaring
    ``pool_size >= N``; the adapter fills this in itself."""
    adapter = EvaluationAdapter(
        client=None,  # type: ignore[arg-type] - this case only checks the construction-time spec projection
        env_spec=eval_env_spec(pool_size=1),
        concurrency=CONCURRENCY,
    )
    assert adapter.env_spec.pool_size == CONCURRENCY
    # Does not shrink when declared larger than the concurrency.
    wider = EvaluationAdapter(
        client=None,  # type: ignore[arg-type]
        env_spec=eval_env_spec(pool_size=8),
        concurrency=2,
    )
    assert wider.env_spec.pool_size == 8


async def test_run_episodes_reports_success_only_from_the_env_signal(
    transport_kind: str,
) -> None:
    """All 8 cells run to completion: the success rate is judged only by
    the environment's termination signal, and the transition reaches the sink."""
    config = local_runtime_config(
        transport_kind, env_worker={"max_sessions_per_rank": CONCURRENCY}
    )
    runtime = build_local_components(config)
    await runtime.start()
    try:
        adapter = EvaluationAdapter(
            runtime.gateway,
            eval_env_spec(),
            concurrency=CONCURRENCY,
            max_steps=8,
            sink_id="mem:eval",
        )
        report = await adapter.run_episodes(tasks(8))
        assert report.attempted == 8
        assert report.invalid == 0, report.error_counts
        assert report.valid == 8
        # The fake env terminates after episode_length=4 steps, and the
        # fake policy gives 4 steps each time
        # (``rollout_worker.actions_per_chunk``), so a single policy_step
        # reaches the end.
        assert report.successes == 8
        assert report.success_rate == 1.0
        assert report.episodes_per_hour > 0.0
        for outcome in report.outcomes:
            assert outcome.valid is True
            assert outcome.success is True
            assert outcome.stop_reason == "terminated"
            assert outcome.num_policy_steps == 1
            assert outcome.executed_horizon == 4
            assert outcome.error is None
        # Per-cell order matches the input.
        assert [outcome.task.seed for outcome in report.outcomes] == list(range(8))
        # The transition goes through the sink, not back to the Gateway.
        records = runtime.env_workers[0].sinks.memory("mem:eval")
        assert len(records) == 8, len(records)
        summary = report.summary()
        assert summary["success_rate"] == 1.0
        assert "invalid_episode" in summary["definitions"]
    finally:
        await runtime.gateway.stop()
        await runtime.aclose()


async def test_sessions_are_closed_after_every_batch(transport_kind: str) -> None:
    """"Closed as soon as it finishes": after 3 batches complete, no
    session remains on the worker, and all pool slots are returned."""
    config = local_runtime_config(
        transport_kind, env_worker={"max_sessions_per_rank": 2}
    )
    runtime = build_local_components(config)
    await runtime.start()
    try:
        adapter = EvaluationAdapter(
            runtime.gateway, eval_env_spec(), concurrency=2, max_steps=4
        )
        report = await adapter.run_episodes(tasks(6))
        assert report.valid == 6
        worker = runtime.env_workers[0]
        assert worker.sessions == {}
        pool = next(iter(worker.pools.pools.values()))
        # The pool was only built once (same digest); 6 cells reuse the
        # same 2 slots.
        assert len(worker.pools.pools) == 1
        assert pool.in_use == 0
        assert pool.core.total_chunk_calls >= 6
    finally:
        await runtime.gateway.stop()
        await runtime.aclose()


async def test_infrastructure_failures_are_invalid_not_zero_scores(
    transport_kind: str,
) -> None:
    """An environment fault is an **invalid episode**: it enters neither
    the numerator nor the denominator, and error codes are counted individually."""
    config = local_runtime_config(
        transport_kind, env_worker={"max_sessions_per_rank": 2}
    )
    runtime = build_local_components(config)
    await runtime.start()
    try:
        adapter = EvaluationAdapter(
            runtime.gateway,
            # fail_on_reset makes every cell explode at the reset stage
            # (an infrastructure fault).
            eval_env_spec(fail_on_reset=True),
            concurrency=2,
            max_steps=4,
        )
        report = await adapter.run_episodes(tasks(2))
        assert report.attempted == 2
        assert report.valid == 0
        assert report.invalid == 2
        assert report.successes == 0
        # The denominator is valid, so the success rate is not the kind of
        # "0/2" accounting that counts a fault as a failure.
        assert report.success_rate == 0.0
        assert report.error_counts == {"ENV_FAILURE": 2}
        for outcome in report.outcomes:
            assert outcome.valid is False
            assert outcome.error is not None
            assert outcome.stop_reason == "error:ENV_FAILURE"
    finally:
        await runtime.gateway.stop()
        await runtime.aclose()


async def test_max_steps_stop_is_a_valid_failure(transport_kind: str) -> None:
    """Running out ``max_steps`` is a **valid failure** (enters the
    denominator, not the numerator)."""
    config = local_runtime_config(
        transport_kind, env_worker={"max_sessions_per_rank": 2}
    )
    runtime = build_local_components(config)
    await runtime.start()
    try:
        adapter = EvaluationAdapter(
            runtime.gateway,
            # The episode needs 100 steps to terminate, but only 1
            # policy_step is given.
            eval_env_spec(episode_length=100),
            concurrency=2,
            max_steps=1,
        )
        report = await adapter.run_episodes(tasks(2))
        assert report.valid == 2
        assert report.invalid == 0
        assert report.successes == 0
        assert report.success_rate == 0.0
        for outcome in report.outcomes:
            assert outcome.valid is True
            assert outcome.success is False
            assert outcome.stop_reason == "max_steps"
            assert outcome.num_policy_steps == 1
    finally:
        await runtime.gateway.stop()
        await runtime.aclose()


async def test_eval_adapter_drives_a_lockstep_pool(transport_kind: str) -> None:
    """Batch eval is a natural user of the vector pool: a batch of 4 cells
    -> one coalesced ``chunk_step`` per tick."""
    from rollout_runtime.core.env_execution import LOCKSTEP_VECTOR_FORM

    config = local_runtime_config(
        transport_kind, env_worker={"max_sessions_per_rank": CONCURRENCY}
    )
    runtime = build_local_components(config)
    await runtime.start()
    try:
        adapter = EvaluationAdapter(
            runtime.gateway,
            eval_env_spec(core_form=LOCKSTEP_VECTOR_FORM),
            concurrency=CONCURRENCY,
            max_steps=8,
            sink_id="mem:eval-lockstep",
        )
        report = await adapter.run_episodes(tasks(CONCURRENCY))
        assert report.valid == CONCURRENCY
        assert report.successes == CONCURRENCY
        worker = runtime.env_workers[0]
        pool = next(iter(worker.pools.pools.values()))
        assert pool.lockstep is True
        stats = worker.coalescer.stats()
        assert stats["max_group_size"] == CONCURRENCY
        # 4 cells x 1 policy_step = 4 commands, but only **1** chunk_step
        # group was used.
        assert stats["coalesced_commands"] == CONCURRENCY
        assert stats["groups_executed"] == 1
        assert pool.core.total_masked_steps == 0
    finally:
        await runtime.gateway.stop()
        await runtime.aclose()


async def test_duplicate_cells_are_not_collapsed(transport_kind: str) -> None:
    """A real defect found during an independent audit: two same-named
    cells used to overwrite each other, silently corrupting the success rate.

    The original implementation keyed the result table by ``task.name()``,
    while "repeated experiments with a fixed random seed" is a formal
    comparison pattern (running the same task/seed multiple times) -- in
    that case two cells share a name: the later cell overwrites the
    earlier one, so one result gets counted twice in the report and the
    other is dropped. It is now indexed by **position within the batch** instead.
    """
    config = local_runtime_config(
        transport_kind, env_worker={"max_sessions_per_rank": 2}
    )
    runtime = build_local_components(config)
    await runtime.start()
    try:
        adapter = EvaluationAdapter(
            runtime.gateway, eval_env_spec(), concurrency=2, max_steps=4
        )
        # Two cells with the exact same name (same task, same seed, blank label).
        repeated = [EvaluationTask(task_id=0, seed=7) for _ in range(2)]
        assert repeated[0].name() == repeated[1].name()
        report = await adapter.run_episodes(repeated)
        assert report.attempted == 2
        assert report.valid == 2
        # Key: the two results must be **two distinct measurements**, not
        # the same object reported twice.
        assert len({id(outcome) for outcome in report.outcomes}) == 2
        assert all(outcome.valid for outcome in report.outcomes)
        # The environment was genuinely run twice (two sessions, two episodes).
        pool = next(iter(runtime.env_workers[0].pools.pools.values()))
        assert pool.core.total_chunk_calls >= 2
    finally:
        await runtime.gateway.stop()
        await runtime.aclose()
