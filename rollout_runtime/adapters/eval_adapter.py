"""Evaluation Adapter.

``run_episodes(tasks, seeds, policy)``: N tasks x M seeds -> batch-create
**temporary** sessions -> ``run_episode`` -> ``close`` on completion ->
results into a sink.

Three deliberate boundaries:

1. **Do not reimplement ``run_episode``**. ``RuntimeEnvWorker.run_episode``
   already has four termination conditions (``max_steps`` / ``deadline`` /
   cancel / terminated·truncated) and a ``TransitionSinkHub``; writing
   another loop would create two competing termination semantics. This
   adapter is only responsible for orchestrating session lifecycle and
   bookkeeping.
2. **``pool_size >= concurrency``**. This follows directly from design
   decision D6: the pool is reused keyed by ``EnvSpecMsg.digest()``, and
   ``pool_size`` is part of that digest, so the same spec only has
   ``pool_size`` slots — the ``pool_size + 1``-th ``create_session`` gets
   ``QUOTA_EXCEEDED``. This adapter therefore raises ``pool_size`` to match
   the concurrency itself. **But ``pool_size`` is per-rank**: every
   EnvWorker rank that receives a session builds its own pool according to
   it. When running across multiple env ranks, either pass ``concurrency``
   as "the slot count for a single rank," or set
   ``env_spec.pool_size`` to ``ceil(total_concurrency / env_ranks)``
   yourself — otherwise the number of environment instances will multiply
   by the rank count.
3. **Success is judged solely by the environment's termination signal,
   valid and invalid episodes are booked separately** (a hard requirement
   from the README). ``success`` = ``EpisodeResult.terminated``;
   ``max_steps`` / ``deadline`` / truncation are all **valid failures**
   (counted in the denominator); infrastructure faults (``Err`` /
   ``ENV_FAILURE`` / ``WORKER_LOST``) are **invalid episodes**, counted in
   neither the numerator nor the denominator, and listed separately under
   ``invalid``.

This is the second piece of evidence (the first being the Gym Adapter) that
"the Runtime is decoupled from Agent semantics": the entire eval path only
uses ``RuntimeClient``'s public methods, knows nothing about LIBERO or the
planner, and cannot reach the transport or the workers.
"""

from __future__ import annotations

import dataclasses
import time
from collections.abc import Sequence
from typing import Any

from rollout_runtime.api.client import RuntimeClient
from rollout_runtime.api.enums import ErrorCode
from rollout_runtime.api.errors import RuntimeErrorInfo, make_error
from rollout_runtime.api.ids import SessionId
from rollout_runtime.api.messages import (
    CreateSessionRequest,
    EnvSpecMsg,
    EpisodeRequest,
    EpisodeResult,
    PolicyRequest,
    ResetSpec,
)
from rollout_runtime.api.result import Err, Result

__all__ = [
    "EpisodeOutcome",
    "EvaluationAdapter",
    "EvaluationReport",
    "EvaluationTask",
]

_INVALID_EPISODE_NOTE = (
    "Any failure (infrastructure faults ENV_FAILURE / POLICY_FAILURE / "
    "WORKER_LOST, as well as manifest-level INVALID_ARGUMENT / "
    "UNSUPPORTED_ENV_SPEC / QUOTA_EXCEEDED) is booked as an **invalid "
    "episode**: it means this cell did not run to completion, and counts "
    "toward neither the numerator nor the denominator of the success rate. "
    "The original error code is preserved via EpisodeOutcome.error, and the "
    "report counts it per code, so the two categories of cause remain "
    "distinguishable."
)
"""The criterion for classifying an episode as invalid (per the README's requirement to book normal failures and infrastructure faults separately)."""


@dataclasses.dataclass(frozen=True, kw_only=True)
class EvaluationTask:
    """A single (task, seed) cell awaiting evaluation.

    Attributes:
        task_id: The task index within the family; ``None`` means use the
            default task in ``env_config``.
        seed: The random seed.
        instruction: An instruction override; ``None`` means use the
            family's own task description.
        reset_state_id: Directly specify the reset state; ``None`` means
            derive it via the family's formula.
        label: A human-readable label, carried into the result and sink
            records.
    """

    task_id: int | None = None
    seed: int = 0
    instruction: str | None = None
    reset_state_id: int | None = None
    label: str = ""

    def reset_spec(self) -> ResetSpec:
        """Project into a ``ResetSpec``.

        Returns:
            The ``ResetSpec``.
        """
        return ResetSpec(
            task_id=self.task_id,
            seed=self.seed,
            instruction=self.instruction,
            reset_state_id=self.reset_state_id,
            options={},
        )

    def name(self) -> str:
        """A human-readable label.

        Returns:
            ``label``, or ``task<task_id>-seed<seed>`` if not set.
        """
        return self.label or f"task{self.task_id}-seed{self.seed}"


@dataclasses.dataclass(frozen=True, kw_only=True)
class EpisodeOutcome:
    """The result of one evaluation cell.

    Attributes:
        task: This cell's task and seed.
        valid: Whether this is a valid episode (counted in the success rate denominator).
        success: Whether it succeeded; determined **solely** by the environment's termination signal.
        stop_reason: The termination reason reported by ``run_episode``.
        num_policy_steps: The number of ``policy_step`` calls executed.
        executed_horizon: The number of env steps actually executed.
        total_reward: The cumulative reward.
        wall_clock_seconds: Wall-clock time for this cell (including session creation and reset).
        error: The normalized error on failure; ``None`` for valid episodes.
        episode_id: The environment-side episode number.
    """

    task: EvaluationTask
    valid: bool
    success: bool = False
    stop_reason: str = ""
    num_policy_steps: int = 0
    executed_horizon: int = 0
    total_reward: float = 0.0
    wall_clock_seconds: float = 0.0
    error: RuntimeErrorInfo | None = None
    episode_id: int | None = None


@dataclasses.dataclass(frozen=True, kw_only=True)
class EvaluationReport:
    """The summary of one ``run_episodes`` call.

    Attributes:
        outcomes: Per-cell results, in the same order as the input ``tasks``.
        attempted: The number of cells attempted.
        valid: The number of valid episodes (the success rate denominator).
        invalid: The number of invalid episodes (infrastructure / manifest
            errors, **not** counted in the denominator).
        successes: The number of successes.
        wall_clock_seconds: Wall-clock time for the whole batch.
        error_counts: Error code -> count.
        sink_id: The transition sink.
    """

    outcomes: tuple[EpisodeOutcome, ...]
    attempted: int
    valid: int
    invalid: int
    successes: int
    wall_clock_seconds: float
    error_counts: dict[str, int]
    sink_id: str | None = None

    @property
    def success_rate(self) -> float:
        """The success rate (the denominator is the **valid** episode count).

        Returns:
            The success rate; ``0.0`` when there are no valid episodes.
        """
        return self.successes / self.valid if self.valid else 0.0

    @property
    def episodes_per_hour(self) -> float:
        """Throughput in valid episodes (used to compare against rlinf's native eval scripts).

        Returns:
            Valid episodes per hour; ``0.0`` if elapsed time is 0.
        """
        if self.wall_clock_seconds <= 0.0:
            return 0.0
        return self.valid * 3600.0 / self.wall_clock_seconds

    def summary(self) -> dict[str, Any]:
        """Project into a dict suitable for direct dumping / printing.

        Returns:
            The summary dictionary.
        """
        return {
            "attempted": self.attempted,
            "valid": self.valid,
            "invalid": self.invalid,
            "successes": self.successes,
            "success_rate": round(self.success_rate, 6),
            "wall_clock_seconds": round(self.wall_clock_seconds, 3),
            "episodes_per_hour": round(self.episodes_per_hour, 3),
            "error_counts": dict(sorted(self.error_counts.items())),
            "sink_id": self.sink_id,
            "definitions": {
                "success": (
                    "Determined solely by the environment's termination signal "
                    "(EpisodeResult.terminated); running out max_steps, exceeding "
                    "the deadline, or being truncated are all valid failures"
                ),
                "invalid_episode": _INVALID_EPISODE_NOTE,
                "success_rate_denominator": "valid (the valid episode count), not attempted",
            },
        }


class EvaluationAdapter:
    """Client-side Adapter for batch evaluation."""

    def __init__(
        self,
        client: RuntimeClient,
        env_spec: EnvSpecMsg,
        *,
        application_id: str = "eval",
        concurrency: int = 1,
        max_steps: int = 64,
        lease_seconds: float = 600.0,
        sink_id: str | None = None,
        deadline_seconds: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Initialize.

        Args:
            client: The Runtime API client (usually a ``RuntimeGateway``).
            env_spec: The environment spec. ``pool_size`` is raised to
                ``max(declared value, concurrency)`` — a consequence of D6,
                see point 2 in the module docstring. **Note that
                ``pool_size`` is per-rank**: every EnvWorker rank that
                receives a session builds a pool sized accordingly, so when
                running across multiple env ranks, ``concurrency`` should be
                passed as "the number of cells a single rank should run" (or
                ``env_spec.pool_size`` should be set to
                ``ceil(total_concurrency / env_ranks)`` yourself); otherwise
                each of the N ranks builds a pool "as large as full
                concurrency," multiplying the number of environment
                instances by N (flagged by an independent audit).
            application_id: The owning application, used for quota and audit.
            concurrency: The number of sessions running simultaneously (i.e.
                how many cells are open at once within a batch).
            max_steps: The ``policy_step`` cap per episode.
            lease_seconds: The lease duration for temporary sessions.
            sink_id: The transition sink (``"mem:<name>"`` / ``"jsonl:<path>"``).
            deadline_seconds: The wall-clock budget per cell; ``None`` means
                unlimited.
            metadata: Passthrough fields.
        """
        self._client = client
        self._concurrency = max(1, int(concurrency))
        self._env_spec = dataclasses.replace(
            env_spec, pool_size=max(int(env_spec.pool_size), self._concurrency)
        )
        self._application_id = application_id
        self._max_steps = max(1, int(max_steps))
        self._lease_seconds = float(lease_seconds)
        self._sink_id = sink_id
        self._deadline_seconds = deadline_seconds
        self._metadata = dict(metadata or {})
        self._sequence = 0

    @property
    def env_spec(self) -> EnvSpecMsg:
        """The environment spec actually used (``pool_size`` already raised for concurrency).

        Returns:
            The environment spec.
        """
        return self._env_spec

    async def run_episodes(
        self,
        tasks: Sequence[EvaluationTask],
        policy: PolicyRequest | None = None,
    ) -> EvaluationReport:
        """Run an entire evaluation manifest.

        The manifest is sliced into batches of ``concurrency``; each batch
        **batch-creates** sessions (a single ``create_sessions`` call),
        **batch**-resets, **batch**-executes ``run_episode``, then
        immediately calls ``close_sessions``. The batch entry point is the
        Runtime API's native shape (every method is batched, §2.3), and
        vector pools (``core_form="lockstep_vector"``) can only actually
        batch because of "same batch, same tick."

        Args:
            tasks: The list of (task, seed) cells to evaluate.
            policy: The inference parameters; ``None`` means use the
                session's default policy.

        Returns:
            The summary report; ``outcomes`` is in the same order as ``tasks``.
        """
        request = policy or PolicyRequest()
        started = time.perf_counter()
        outcomes: list[EpisodeOutcome] = []
        for index in range(0, len(tasks), self._concurrency):
            batch = list(tasks[index : index + self._concurrency])
            outcomes.extend(await self._run_batch(batch, request))
        elapsed = time.perf_counter() - started
        error_counts: dict[str, int] = {}
        for outcome in outcomes:
            if outcome.error is not None:
                key = outcome.error.code.name
                error_counts[key] = error_counts.get(key, 0) + 1
        valid = sum(1 for outcome in outcomes if outcome.valid)
        return EvaluationReport(
            outcomes=tuple(outcomes),
            attempted=len(outcomes),
            valid=valid,
            invalid=len(outcomes) - valid,
            successes=sum(1 for outcome in outcomes if outcome.success),
            wall_clock_seconds=elapsed,
            error_counts=error_counts,
            sink_id=self._sink_id,
        )

    async def _run_batch(
        self, batch: Sequence[EvaluationTask], policy: PolicyRequest
    ) -> list[EpisodeOutcome]:
        """Run one batch (up to ``concurrency`` cells), closing immediately when done.

        Args:
            batch: This batch's tasks.
            policy: The inference parameters.

        Returns:
            Results in the same order as ``batch``.
        """
        started = time.perf_counter()
        requests = []
        for task in batch:
            self._sequence += 1
            requests.append(
                CreateSessionRequest(
                    application_id=self._application_id,
                    client_session_key=f"eval-{self._sequence}-{task.name()}",
                    env_spec=self._env_spec,
                    default_policy_id=policy.policy_id,
                    lease_seconds=self._lease_seconds,
                    metadata={**self._metadata, "eval_task": task.name()},
                )
            )
        created = await self._client.create_sessions(requests)
        session_ids: list[SessionId] = []
        live: list[int] = []
        # **Index by in-batch position, not by task.name()**: two cells in
        # the same batch can legitimately share the same name (repeated
        # experiments with a fixed seed are a formal comparison pattern per
        # the README), and keying by name would let a later cell overwrite
        # an earlier one — one result gets counted twice in the report while
        # another is silently dropped, corrupting the success rate.
        outcomes: list[EpisodeOutcome | None] = [None] * len(batch)
        for index, (task, result) in enumerate(zip(batch, created, strict=True)):
            if isinstance(result, Err):
                outcomes[index] = self._failed(task, result.error, started)
                continue
            session_ids.append(result.value.session_id)
            live.append(index)
        if not session_ids:
            return [outcome for outcome in outcomes if outcome is not None]
        try:
            await self._reset_and_run(
                batch, live, session_ids, policy, started, outcomes
            )
        finally:
            # "Close on completion": temporary sessions do not keep a lease,
            # so the slot returns to the pool immediately for the next batch.
            await self._client.close_sessions(session_ids)
        return [
            outcome
            if outcome is not None
            # Defensive fallback: every cell must have a result; otherwise
            # report an explicit invalid episode rather than nothing.
            else self._failed(
                batch[index],
                make_error(
                    ErrorCode.INTERNAL,
                    "evaluation cell produced no outcome; this is a runtime bug",
                ),
                started,
            )
            for index, outcome in enumerate(outcomes)
        ]

    async def _reset_and_run(
        self,
        batch: Sequence[EvaluationTask],
        live: Sequence[int],
        session_ids: Sequence[SessionId],
        policy: PolicyRequest,
        started: float,
        outcomes: list[EpisodeOutcome | None],
    ) -> None:
        """Reset each cell individually (each with its own task/seed), then batch-run the episode.

        ``reset`` has the shape "one ResetSpec applied to a batch of
        sessions" (§2.3), but each cell has a different task/seed, so calls
        here are made per cell; ``run_episode`` is the step that genuinely
        needs "same batch, same tick," so it is issued as a single batch
        call.

        Args:
            batch: All of this batch's tasks (indexed by in-batch position).
            live: The in-batch indices for which session creation succeeded,
                in the same order as ``session_ids``.
            session_ids: The sessions, in the same order as ``live``.
            policy: The inference parameters.
            started: The ``perf_counter`` value when this batch started.
            outcomes: The results table (written in place by in-batch index).
        """
        runnable: list[tuple[int, SessionId]] = []
        for index, session_id in zip(live, session_ids, strict=True):
            task = batch[index]
            reset = (await self._client.reset([session_id], task.reset_spec()))[0]
            if isinstance(reset, Err):
                outcomes[index] = self._failed(task, reset.error, started)
                continue
            runnable.append((index, session_id))
        if not runnable:
            return
        episode = EpisodeRequest(
            max_steps=self._max_steps,
            policy=policy,
            deadline=(
                time.time() + self._deadline_seconds
                if self._deadline_seconds is not None
                else None
            ),
            sink_id=self._sink_id,
            stop_on_truncation=True,
        )
        results: Sequence[Result[EpisodeResult]] = await self._client.run_episode(
            [session_id for _index, session_id in runnable], episode
        )
        for (index, _session_id), result in zip(runnable, results, strict=True):
            task = batch[index]
            if isinstance(result, Err):
                outcomes[index] = self._failed(task, result.error, started)
                continue
            outcomes[index] = self._succeeded(task, result.value, started)

    def _failed(
        self, task: EvaluationTask, error: RuntimeErrorInfo, started: float
    ) -> EpisodeOutcome:
        """Book a failure as an **invalid episode**.

        Per the criterion in ``_INVALID_EPISODE_NOTE``: only valid episodes
        count toward the success rate denominator, so both infrastructure
        faults and manifest errors are excluded from it; ``error.code`` is
        preserved as-is, and the report counts it per code.

        Args:
            task: This cell.
            error: The normalized error.
            started: The ``perf_counter`` value when this batch started.

        Returns:
            A result with ``valid=False``.
        """
        return EpisodeOutcome(
            task=task,
            valid=False,
            success=False,
            stop_reason=f"error:{error.code.name}",
            wall_clock_seconds=time.perf_counter() - started,
            error=error,
        )

    def _succeeded(
        self, task: EvaluationTask, result: EpisodeResult, started: float
    ) -> EpisodeOutcome:
        """Book a completed episode as a **valid episode**.

        ``success`` looks **solely** at ``EpisodeResult.terminated`` (the
        environment's termination signal). Running out ``max_steps``,
        exceeding the deadline, or being truncated are all valid failures.

        Args:
            task: This cell.
            result: The result of ``run_episode``.
            started: The ``perf_counter`` value when this batch started.

        Returns:
            A result with ``valid=True``.
        """
        return EpisodeOutcome(
            task=task,
            valid=True,
            success=bool(result.terminated),
            stop_reason=result.stop_reason,
            num_policy_steps=int(result.num_policy_steps),
            executed_horizon=int(result.executed_horizon),
            total_reward=float(result.total_reward),
            wall_clock_seconds=time.perf_counter() - started,
            error=None,
            episode_id=int(result.episode_id)
            if result.episode_id is not None
            else None,
        )
