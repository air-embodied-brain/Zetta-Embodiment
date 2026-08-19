"""``launch/ray_launch.py``: workers actually run in separate Ray processes.

Division of labor with ``test_e2e_fake.py --transport=ray_channel``: there, the worker
object lives in the test process (convenient for injecting fake backend latency and
faults); here, the worker is a separate actor process launched as an ``rlinf.Worker``
subclass, verifying the "shell + placement + cross-process channel" path:

- Two groups are launched via ``NodePlacementStrategy`` (the only strategy usable with
  0 local accelerators);
- Sessions are spread across multiple env ranks, and the command / control / result
  channels all work across process boundaries;
- Multiple rollout ranks compete for the same ``rr_infer_req["pending"]`` key, i.e.
  work-stealing;
- Remote methods are total functions: the shell converts exceptions into return values
  and never triggers a ``SIGUSR1``-based job kill.

Marked with ``ray_launch``: it is slower than in-process cases (it launches 4 actor
processes), and can be excluded with ``-m "not ray_launch"`` when needed.
"""

from __future__ import annotations

import contextlib
from typing import Any

import pytest

from rollout_runtime.api.messages import (
    CreateSessionRequest,
    EnvSpecMsg,
    EpisodeRequest,
    PolicyRequest,
    ResetSpec,
)
from rollout_runtime.api.result import Err, unwrap
from rollout_runtime.config.schema import load_config

pytestmark = [pytest.mark.ray, pytest.mark.ray_launch]

POLICY = PolicyRequest(policy_id="fake")
"""Shared inference parameters."""


def _unique_groups(config: Any) -> Any:
    """Give both worker groups a name unique to this call.

    rlinf's ``WorkerGroup`` has no destroy API, and actor names
    (``{group_name}:{rank}``) are globally unique; the ``ray.kill`` inside
    ``RayRuntime.aclose`` only takes effect asynchronously, so launching again
    immediately under the same name can intermittently raise
    ``ActorAlreadyExistsError`` (observed roughly 1/6 of the time). Production naming
    stays stable (the Gateway needs to find an already-started group via
    ``WorkerGroup.from_group_name``), so the name is only changed within test cases.

    Args:
        config: The runtime configuration (modified in place).

    Returns:
        The same configuration object.
    """
    import uuid

    suffix = uuid.uuid4().hex[:8]
    config.env_worker.group_name = f"env-{suffix}"
    config.rollout_worker.group_name = f"rollout-{suffix}"
    return config


async def test_ray_launch_runs_workers_in_separate_processes() -> None:
    """Both groups launch in separate processes, the e2e path works end-to-end, and
    both rollout ranks actually perform work."""
    from rollout_runtime.launch.ray_launch import build_ray_components

    config = _unique_groups(load_config("local_ray_fake"))
    # 4 sessions spread across 2 env ranks: capacity per rank is bounded by
    # max_sessions_per_rank (``EnvWorkerRegistry.select_rank`` prioritizes "already
    # serving the same digest" ahead of load).
    config.env_worker.max_sessions_per_rank = 2
    config.env_config = dict(config.env_config)
    config.env_config["episode_length"] = 64

    runtime = build_ray_components(config)
    await runtime.start()
    try:
        gateway = runtime.gateway
        assert sorted(runtime.transport.worker_ranks()) == [0, 1]

        spec = EnvSpecMsg(
            env_family="fake", env_config=dict(config.env_config), pool_size=2
        )
        created = await gateway.create_sessions(
            [
                CreateSessionRequest(
                    application_id="ray-launch",
                    client_session_key=f"rl-{index}",
                    env_spec=spec,
                    default_policy_id="fake",
                    lease_seconds=120.0,
                )
                for index in range(4)
            ]
        )
        failures = [item.error for item in created if isinstance(item, Err)]
        assert not failures, f"create_sessions failed: {failures}"
        session_ids = [unwrap(item).session_id for item in created]
        ranks = {gateway.sessions.get(sid).worker_rank for sid in session_ids}
        assert ranks == {0, 1}, f"sessions did not spread across env ranks: {ranks}"

        resets = await gateway.reset(session_ids, ResetSpec(seed=3))
        assert not [item for item in resets if isinstance(item, Err)]
        assert all(unwrap(item).episode_id == 1 for item in resets)

        for _ in range(3):
            results = await gateway.policy_step(session_ids, POLICY)
            errors = [item.error for item in results if isinstance(item, Err)]
            assert not errors, f"policy_step failed: {errors}"
            assert [unwrap(item).session_id for item in results] == session_ids
            assert all(unwrap(item).executed_horizon == 4 for item in results)
            assert all(
                unwrap(item).info["model_version"] == "fake-v1" for item in results
            )

        episodes = await gateway.run_episode(
            session_ids,
            EpisodeRequest(max_steps=2, policy=POLICY, sink_id="mem:ray-launch"),
        )
        assert not [item for item in episodes if isinstance(item, Err)]
        assert all(unwrap(item).num_policy_steps == 2 for item in episodes)

        # The request side shares a single key: both rollout ranks should pick up
        # work (work-stealing).
        counters = await runtime.rollout_counters()
        assert len(counters) == 2
        assert sum(item["received"] for item in counters) == 4 * (3 + 2)
        assert all(item["received"] > 0 for item in counters), counters
        assert all(item["dropped_routes"] == 0 for item in counters), counters

        # Results have all flowed back and there are no orphaned results.
        assert runtime.transport.bytes_received > 0
        assert runtime.transport.command_timeout_count == 0
        assert runtime.transport.orphan_result_count == 0

        closed = await gateway.close_sessions(session_ids)
        assert not [item for item in closed if isinstance(item, Err)]
    finally:
        with contextlib.suppress(BaseException):
            await runtime.gateway.stop()
        await runtime.aclose()


async def test_ray_launch_shell_methods_are_total_functions() -> None:
    """The shell's remote methods never raise: repeated ``init_worker`` /
    ``stop_server`` calls only return structured results.

    This directly guards against remote exceptions causing
    ``WorkerGroupFuncResult`` to trigger ``os.kill(pid, SIGUSR1)``, where a single
    failed request would kill the entire job.
    """
    import asyncio

    from rollout_runtime.launch.ray_launch import build_ray_components

    config = _unique_groups(load_config("local_ray_fake"))
    config.env_worker.num_ranks = 1
    config.rollout_worker.num_ranks = 1
    runtime = build_ray_components(config)
    await runtime.start()
    try:
        # Init again: this rebuilds the worker object and channel view, but must
        # never raise.
        outcomes = await asyncio.to_thread(
            lambda: runtime.env_group.init_worker().wait()
        )
        assert all(item["ok"] for item in outcomes), outcomes
        stopped = await asyncio.to_thread(
            lambda: runtime.rollout_group.stop_server().wait()
        )
        assert all(item["ok"] for item in stopped), stopped
        again = await asyncio.to_thread(
            lambda: runtime.rollout_group.stop_server().wait()
        )
        assert all(item["ok"] for item in again), again
    finally:
        with contextlib.suppress(BaseException):
            await runtime.gateway.stop()
        await runtime.aclose()


async def test_ray_launch_requires_ray_channel_transport() -> None:
    """``ray_launch`` only accepts ``ray_channel``: an inproc configuration must raise
    an explicit error rather than silently degrading."""
    from rollout_runtime.launch.ray_launch import build_ray_components

    config = load_config("local_fake")
    with pytest.raises(ValueError, match="ray_channel"):
        build_ray_components(config)


def test_ray_launch_placement_falls_back_to_node_strategy() -> None:
    """When ``cluster.component_placement`` is not declared, falls back to
    ``NodePlacementStrategy``.

    With 0 local accelerators, ``PackedPlacementStrategy`` is unavailable, so this is
    the precondition for running locally at all; ``a100_libero.yaml`` takes a
    different branch (``HybridComponentPlacement``).
    """
    from rollout_runtime.launch.ray_launch import _placement_for
    from zetta.runtime.ray.placement import PlacementSlot

    config = load_config("local_ray_fake")
    strategy = _placement_for("env", config=config, cluster=None, num_ranks=3)
    assert strategy == [PlacementSlot(node_rank=0)] * 3

    # Declaring "packed" without providing cluster.component_placement is a
    # configuration error and must raise explicitly: silently falling back to a
    # single node would cram all 8 ranks on a GPU machine into the default
    # placement.
    config.env_worker.placement_strategy = "packed"
    with pytest.raises(ValueError, match="component_placement"):
        _placement_for("env", config=config, cluster=None, num_ranks=8)


def test_the_two_a100_presets_differ_only_in_the_policy_backend() -> None:
    """The two a100 presets must differ **only** in the policy backend, or they are
    not a valid A/B pair.

    The only reason `a100_libero.yaml` (real env + fake policy) exists is to serve as
    a same-topology control for `a100_libero_pi05.yaml` (real env + real pi0.5):
    subtracting the two throughputs isolates the GPU forward cost. An earlier
    regression happened because the two presets had `chunk_size` values of 8 and 10
    respectively, folding 2 steps of extra simulation rendering into the model cost,
    so this invariant must be enforced by a test rather than relied on by memory.

    `env_config` is compared as the **raw dict** rather than after default resolution:
    it feeds into ``EnvSpecMsg.digest()``, and any literal difference between the two
    would hit a different env pool.
    """
    import dataclasses

    baseline: Any = load_config("a100_libero")
    real: Any = load_config("a100_libero_pi05")
    for field in dataclasses.fields(baseline):
        if field.name == "rollout_worker":
            continue
        assert getattr(baseline, field.name) == getattr(real, field.name), (
            f"a100 presets disagree on {field.name!r}; the pair is only usable as an "
            "A/B if everything except the policy backend matches"
        )
    differing = sorted(
        field.name
        for field in dataclasses.fields(baseline.rollout_worker)
        if getattr(baseline.rollout_worker, field.name)
        != getattr(real.rollout_worker, field.name)
    )
    assert differing == ["policy_backend", "policy_config"], differing
    assert baseline.rollout_worker.policy_backend == "fake"
    assert real.rollout_worker.policy_backend == "zetta_openpi"


def test_a100_presets_declare_a_self_consistent_placement() -> None:
    """The topology of both a100 presets: 8 env ranks / 8 rollout ranks, with
    placement and rank counts self-consistent.

    A constraint observed on multi-GPU hosts: the ``placement`` string in
    ``cluster.component_placement`` enumerates **hardware ranks**, which determine
    the process count, and ``num_ranks`` is silently ignored. Therefore, the CPU
    EnvWorker **must not** be included in ``component_placement`` (otherwise only 1
    process launches per node, and the 2nd ``create_session`` immediately hits
    ``QUOTA_EXCEEDED``); it must instead rely on
    ``placement_strategy="node"`` via ``NodePlacementStrategy([0] * num_ranks)``.
    Only the GPU-occupying rollout declares placement.
    """
    for name, policy_backend in (
        ("a100_libero", "fake"),
        ("a100_libero_pi05", "zetta_openpi"),
    ):
        config: Any = load_config(name)
        assert config.transport.kind == "ray_channel", name
        assert config.env_family == "libero", name
        assert config.env_worker.num_ranks == 8, name
        assert config.rollout_worker.num_ranks == 8, name
        assert config.rollout_worker.tensor_parallel_size == 1, name
        assert config.rollout_worker.policy_backend == policy_backend, name
        # env is excluded from component_placement; rollout occupies all 8 GPUs.
        assert set(config.cluster.component_placement) == {"rollout"}, name
        assert config.cluster.component_placement["rollout"]["placement"] == "0-7", name
        assert config.env_worker.placement_strategy == "node", name
        assert config.rollout_worker.placement_strategy == "packed", name
        # A real VLA inference call can take on the order of seconds: on timeout,
        # only DEADLINE_EXCEEDED is returned, without canceling the operation.
        assert config.transport.command_timeout_seconds >= 60.0, name


# ---------------------------------------------------------------------------
# Rank-count guard (does not require Ray / rlinf / torch: the guard takes effect
# before touching the transport)
# ---------------------------------------------------------------------------


class _FakeCall:
    """The return value of ``group.method()``; ``wait()`` yields one result per rank."""

    def __init__(self, outcomes: list[dict[str, Any]]) -> None:
        self._outcomes = outcomes

    def wait(self) -> list[dict[str, Any]]:
        """Returns: one result dict per rank."""
        return self._outcomes


class _FakeGroup:
    """A worker group stand-in implementing only ``init_worker`` / ``start_server``."""

    def __init__(self, ranks: int) -> None:
        self._outcomes = [{"ok": True} for _ in range(ranks)]

    def init_worker(self) -> _FakeCall:
        """Returns: the init result for each rank."""
        return _FakeCall(self._outcomes)

    def start_server(self) -> _FakeCall:
        """Returns: the start result for each rank."""
        return _FakeCall(self._outcomes)


class _TransportReached(RuntimeError):
    """Sentinel: reaching ``start()``'s transport call means both rank guards passed."""


class _FakeTransport:
    async def start(self) -> None:
        """Raises: _TransportReached: always."""
        raise _TransportReached


def _runtime_with(config: Any, *, env_ranks: int, rollout_ranks: int) -> Any:
    """Build a ``RayRuntime`` that only runs far enough to hit the rank guard."""
    from rollout_runtime.launch.ray_launch import RayRuntime

    return RayRuntime(
        config=config,
        gateway=None,  # type: ignore[arg-type]
        cluster=None,
        env_group=_FakeGroup(env_ranks),
        rollout_group=_FakeGroup(rollout_ranks),
        transport=_FakeTransport(),  # type: ignore[arg-type]
        channel=None,  # type: ignore[arg-type]
        names=None,  # type: ignore[arg-type]
    )


async def test_extra_rollout_ranks_fail_fast_instead_of_loading_extra_weights() -> None:
    """A mismatch between the actually launched rollout rank count and ``num_ranks``
    must fail immediately.

    Why this guard is needed: the placement string in
    ``cluster.component_placement['rollout']`` enumerates **hardware ranks**, which
    determine the process count, and ``rollout_worker.num_ranks`` is silently
    ignored. The env side already has a guard (via the registration count), but the
    Gateway never touches the RolloutWorker, so the rollout side previously had no
    check at all. The consequence is not a crash but **loading extra copies of the
    weights** (each pi0.5 copy is about 7.5 GiB), which invalidates the assumption
    that "N weight copies serve M sessions" — when `a100_libero_pi05` declares
    placement `"0-7"`, changing ``--rollout-ranks`` to 4 does not actually launch
    only 4 ranks, yet a report generated from the config would still claim "4 weight
    copies".
    """
    config = load_config("a100_libero")
    config.rollout_worker.num_ranks = 4
    runtime = _runtime_with(config, env_ranks=8, rollout_ranks=8)
    with pytest.raises(RuntimeError, match="rollout group started 8 rank") as excinfo:
        await runtime.start()
    assert "num_ranks=4" in str(excinfo.value)
    # The error message must point to the real fix (narrowing placement), not just
    # state that "the count is wrong".
    assert "placement" in str(excinfo.value)
    assert runtime.observed_ranks == {"env": 8, "rollout": 8}


async def test_matching_rank_counts_pass_the_guard() -> None:
    """When the counts match, the guard passes and leaves the observed values in
    ``observed_ranks`` for reports to reference."""
    config = load_config("a100_libero")
    config.rollout_worker.num_ranks = 8
    runtime = _runtime_with(config, env_ranks=8, rollout_ranks=8)
    with pytest.raises(_TransportReached):
        await runtime.start()
    assert runtime.observed_ranks == {"env": 8, "rollout": 8}
