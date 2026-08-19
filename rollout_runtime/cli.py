"""``rollout-runtime`` command-line entry point.

Three subcommands:

- ``serve``: expose the Runtime API via FastAPI + uvicorn. **Security
  requirement**: binds only to ``127.0.0.1`` by default; binding to a
  non-loopback address mandates ``RR_AUTH_TOKEN`` and validates
  ``Authorization: Bearer``, refusing to start if missing (exit code 2, and
  refused **before** loading any backend); ``application_id`` is resolved
  from the token, never self-declared by the client. Implemented under
  ``rollout_runtime/serve/``.
- ``smoke``: run the complete flow with one command and print the timeline.
- ``bench``: a throughput matrix. The measurement method matches
  ``runtime validation notes`` §6.1; ``--launch local`` keeps workers in
  this process, ``--launch ray`` launches a separate Ray worker group.

Both ``smoke`` and ``bench`` execution bodies only depend on ``launch/``:
``transport.kind`` decides whether the data plane is inproc or a real rlinf
``Channel``.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import time
from collections.abc import Sequence
from typing import Any

__all__ = ["build_parser", "main"]


def build_parser() -> argparse.ArgumentParser:
    """Construct the command-line parser.

    Returns:
        A parser with the ``serve`` / ``smoke`` / ``bench`` subcommands.
    """
    parser = argparse.ArgumentParser(
        prog="rollout-runtime", description="Rollout Runtime control plane CLI"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    serve = subparsers.add_parser("serve", help="expose the Runtime API over HTTP (M7)")
    serve.add_argument(
        "--config", default="local_fake", help="preset name or yaml path"
    )
    serve.add_argument(
        "--host",
        default="127.0.0.1",
        help="bind address; a non-loopback bind requires RR_AUTH_TOKEN",
    )
    serve.add_argument("--port", type=int, default=8710, help="bind port")
    serve.add_argument(
        "--launch",
        default="local",
        choices=["local", "ray"],
        help="'local' keeps workers in this process, 'ray' launches worker groups",
    )
    serve.add_argument(
        "--gateway-epoch",
        type=int,
        default=None,
        help="epoch published in SessionHandle (default: process start time)",
    )
    serve.add_argument(
        "--max-lease-seconds",
        type=float,
        default=None,
        help="clamp for lease_seconds (default: gateway.default_lease_seconds)",
    )
    serve.add_argument(
        "--max-pool-size",
        type=int,
        default=None,
        help="clamp for env_spec.pool_size (default: env_worker.max_sessions_per_rank)",
    )
    serve.add_argument(
        "--max-episode-steps",
        type=int,
        default=1000,
        help="clamp for EpisodeRequest.max_steps",
    )
    serve.add_argument(
        "--max-body-bytes",
        type=int,
        default=16 * 1024 * 1024,
        help="reject larger HTTP bodies (streamed, aborted at the limit, before any msgpack decoding); the runtime error is INVALID_ARGUMENT with detail.reason='body_too_large'",
    )
    serve.add_argument(
        "--include-traceback",
        action="store_true",
        help="return detail.traceback to callers (off by default: it leaks paths)",
    )
    serve.add_argument("--log-level", default="info", help="uvicorn log level")

    smoke = subparsers.add_parser("smoke", help="run one end-to-end flow (M2)")
    smoke.add_argument(
        "--config", default="local_fake", help="preset name or yaml path"
    )
    smoke.add_argument("--sessions", type=int, default=1, help="session count")
    smoke.add_argument("--steps", type=int, default=8, help="policy steps per session")

    bench = subparsers.add_parser("bench", help="throughput benchmark (M3)")
    bench.add_argument(
        "--config", default="local_fake", help="preset name or yaml path"
    )
    bench.add_argument("--env-ranks", type=int, default=1, help="env worker ranks")
    bench.add_argument("--rollout-ranks", type=int, default=1, help="rollout ranks")
    bench.add_argument(
        "--sessions",
        default="1,4,8,16",
        help="concurrent session counts (comma separated levels)",
    )
    bench.add_argument(
        "--rounds", type=int, default=40, help="policy_step batches per level"
    )
    bench.add_argument(
        "--transport",
        default=None,
        choices=["inproc", "ray_channel"],
        help="override transport.kind (default: whatever the preset says)",
    )
    bench.add_argument(
        "--max-wait-ms",
        type=float,
        default=None,
        help="override scheduler.max_wait_ms (0 = fire every batch immediately)",
    )
    bench.add_argument(
        "--max-batch-size",
        type=int,
        default=None,
        help="override scheduler.max_batch_size",
    )
    bench.add_argument(
        "--launch",
        default="local",
        choices=["local", "ray"],
        help="'local' keeps workers in this process, 'ray' launches worker groups",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """The command-line entry point.

    Args:
        argv: The argument list; ``None`` uses ``sys.argv``.

    Returns:
        The process exit code; 2 if ``serve``'s security precondition is not
        satisfied (in which case **no** port is bound and **no** backend is
        loaded).
    """
    args = build_parser().parse_args(argv)
    if args.command == "smoke":
        return asyncio.run(_run_smoke(args))
    if args.command == "bench":
        return asyncio.run(_run_bench(args))
    return _run_serve(args)


def _run_serve(args: argparse.Namespace) -> int:
    """Run ``serve``.

    Args:
        args: The ``serve`` subcommand's arguments.

    Returns:
        The process exit code; 2 if the security precondition is not satisfied.
    """
    from rollout_runtime.serve.app import ServeLimits
    from rollout_runtime.serve.auth import ServeSecurityError
    from rollout_runtime.serve.server import ServeOptions, run_serve

    options = ServeOptions(
        config=args.config,
        host=args.host,
        port=args.port,
        launch=args.launch,
        gateway_epoch=args.gateway_epoch,
        log_level=args.log_level,
        limits=ServeLimits(
            # 0.0 / 0 is the sentinel for "derive from the effective
            # configuration" (when the flag is not given, this is None).
            # Passing an explicit 0 would be treated the same as "not given,"
            # so the CLI layer rejects an explicit 0 outright, to avoid an
            # operator believing they turned the limit off.
            max_lease_seconds=args.max_lease_seconds if args.max_lease_seconds else 0.0,
            max_pool_size=args.max_pool_size if args.max_pool_size else 0,
            max_episode_steps=args.max_episode_steps,
            max_body_bytes=args.max_body_bytes,
            include_traceback=args.include_traceback,
        ),
    )
    if args.max_lease_seconds is not None and args.max_lease_seconds <= 0:
        print("!! --max-lease-seconds must be positive")
        return 2
    if args.max_pool_size is not None and args.max_pool_size <= 0:
        print("!! --max-pool-size must be positive")
        return 2
    try:
        return asyncio.run(run_serve(options))
    except ServeSecurityError as exc:
        print(f"!! {exc}")
        return 2
    except KeyboardInterrupt:  # pragma: no cover - interactive exit
        return 0


async def _run_smoke(args: argparse.Namespace) -> int:
    """Run through the complete flow once and print the timeline.

    Args:
        args: The ``smoke`` subcommand's arguments.

    Returns:
        The process exit code: 0 on full success, 1 on any failure.
    """
    # Lazy import: ``rollout-runtime --help`` should not drag in numpy and backends.
    from rollout_runtime.api.messages import (
        CreateSessionRequest,
        EnvSpecMsg,
        EpisodeRequest,
        PolicyRequest,
        ResetSpec,
    )
    from rollout_runtime.api.result import Err
    from rollout_runtime.config.schema import load_config
    from rollout_runtime.launch.local import build_local_components

    config = load_config(args.config)
    runtime = build_local_components(config)
    started = time.perf_counter()
    failures = 0
    timeline: list[tuple[str, float, str]] = []

    def mark(label: str, since: float, note: str = "") -> None:
        timeline.append((label, (time.perf_counter() - since) * 1000.0, note))

    def check(label: str, results: Sequence[object]) -> None:
        nonlocal failures
        for result in results:
            if isinstance(result, Err):
                failures += 1
                print(
                    f"    !! {label}: {result.error.code.name}: {result.error.message}"
                )

    async with runtime:
        gateway = runtime.gateway
        env_spec = EnvSpecMsg(
            env_family=config.env_family,
            env_config=dict(config.env_config),
            # Design decision D6: the pool is pre-allocated, and one slot
            # serves only one session. Running N concurrent sessions
            # therefore requires requesting N slots in the spec, otherwise
            # the 2nd create_session would get QUOTA_EXCEEDED (this is
            # exactly the "explicit rejection when session count exceeds
            # pool capacity" item in the risk table).
            pool_size=max(config.env_worker.default_pool_size, args.sessions),
        )
        phase = time.perf_counter()
        created = await gateway.create_sessions(
            [
                CreateSessionRequest(
                    application_id="smoke",
                    client_session_key=f"smoke-{index}",
                    env_spec=env_spec,
                    default_policy_id=config.rollout_worker.policy_id,
                    lease_seconds=config.gateway.default_lease_seconds,
                )
                for index in range(args.sessions)
            ]
        )
        check("create_sessions", created)
        mark("create_sessions", phase, f"{args.sessions} session(s)")
        session_ids = [
            result.value.session_id for result in created if not isinstance(result, Err)
        ]

        phase = time.perf_counter()
        check("reset", await gateway.reset(session_ids, ResetSpec(seed=1)))
        mark("reset", phase)

        phase = time.perf_counter()
        check("observe", await gateway.observe(session_ids))
        mark("observe", phase)

        policy = PolicyRequest(policy_id=config.rollout_worker.policy_id)
        for step in range(args.steps):
            phase = time.perf_counter()
            results = await gateway.policy_step(session_ids, policy)
            check("policy_step", results)
            horizons = [
                result.value.executed_horizon
                for result in results
                if not isinstance(result, Err)
            ]
            mark("policy_step", phase, f"#{step + 1} horizon={horizons}")

        phase = time.perf_counter()
        check(
            "run_episode",
            await gateway.run_episode(
                session_ids,
                EpisodeRequest(
                    max_steps=args.steps, policy=policy, sink_id="mem:smoke"
                ),
            ),
        )
        mark("run_episode", phase, f"max_steps={args.steps}")

        phase = time.perf_counter()
        check("close_sessions", await gateway.close_sessions(session_ids))
        mark("close_sessions", phase)

        print(f"==> rollout-runtime smoke ({args.config})")
        print(f"    transport={config.transport.kind} env_family={config.env_family}")
        for label, elapsed_ms, note in timeline:
            suffix = f"  {note}" if note else ""
            print(f"    {elapsed_ms:8.2f} ms  {label}{suffix}")
        channel = runtime.channel
        print(
            f"    inference requests={channel.requests_put} "
            f"responses={channel.responses_put} "
            f"rejected={channel.requests_rejected} "
            f"late={sum(w.inference.late_response_count for w in runtime.env_workers)}"
        )
        print(f"    total {(time.perf_counter() - started) * 1000.0:.2f} ms")
        print("==> OK" if failures == 0 else f"==> FAILED ({failures} error(s))")
    return 1 if failures else 0


async def _run_bench(args: argparse.Namespace) -> int:
    """Run the throughput matrix (``cli.py bench``).

    The measurement method matches ``runtime validation notes`` §6.1, so the
    numbers for inproc and ray_channel can be directly compared: the
    ``policy_step`` batch entry point (batch = concurrent session count), a
    fake env with ``chunk_size=4``, a 16x16x3 PNG observation x2, ``--rounds``
    rounds per level, skipping the first round as warm-up.

    Session spreading relies on ``max_sessions_per_rank``:
    ``EnvWorkerRegistry.select_rank`` ranks "already serving the same env
    digest" ahead of load (pool reuse takes priority), so making multiple
    env ranks actually do work requires explicitly limiting the number of
    sessions per rank.

    Args:
        args: The ``bench`` subcommand's arguments.

    Returns:
        The process exit code: 0 on full success, 1 on any failure.
    """
    import math
    import statistics

    from rollout_runtime.api.messages import (
        CreateSessionRequest,
        EnvSpecMsg,
        PolicyRequest,
        ResetSpec,
    )
    from rollout_runtime.api.result import Err
    from rollout_runtime.config.schema import load_config

    levels = [int(item) for item in str(args.sessions).split(",") if item.strip()]
    if not levels:
        print("!! --sessions must list at least one level")
        return 1
    peak = max(levels)
    rounds = max(2, args.rounds)

    base = load_config(args.config)
    if args.transport is not None:
        base.transport.kind = args.transport
    base.env_worker.num_ranks = max(1, args.env_ranks)
    base.rollout_worker.num_ranks = max(1, args.rollout_ranks)
    if args.max_wait_ms is not None:
        base.rollout_worker.scheduler.max_wait_ms = args.max_wait_ms
    if args.max_batch_size is not None:
        base.rollout_worker.scheduler.max_batch_size = args.max_batch_size
    # Same observation shape and episode length as §6.1 (so an episode
    # never terminates midway). **Only applies to the fake family**: these
    # keys are ``FakeEnvConfig`` fields; real families (like libero) decide
    # observation size and episode length via their preset, and forcing
    # these would be rejected by the family config's unknown-key validation.
    base.env_config = dict(base.env_config)
    if base.env_family == "fake":
        base.env_config.update(
            {
                "action_dim": 7,
                "chunk_size": 4,
                "image_height": 16,
                "image_width": 16,
                "state_dim": 8,
                "episode_length": (rounds + 4) * 4,
            }
        )
    per_rank = math.ceil(peak / base.env_worker.num_ranks)
    base.env_worker.max_sessions_per_rank = per_rank
    base.admission.max_sessions_per_application = max(
        base.admission.max_sessions_per_application, peak
    )
    base.admission.max_total_inflight_operations = max(
        base.admission.max_total_inflight_operations, peak * 2
    )
    base.admission.max_inflight_operations_per_application = max(
        base.admission.max_inflight_operations_per_application, peak * 2
    )

    chunk = int(base.env_config.get("chunk_size", 1) or 1)
    policy = PolicyRequest(policy_id=base.rollout_worker.policy_id)
    print(f"==> rollout-runtime bench ({args.config})")
    print(
        f"    transport={base.transport.kind} launch={args.launch} "
        f"env_ranks={base.env_worker.num_ranks} "
        f"rollout_ranks={base.rollout_worker.num_ranks} "
        f"max_batch_size={base.rollout_worker.scheduler.max_batch_size} "
        f"max_wait_ms={base.rollout_worker.scheduler.max_wait_ms} "
        f"rounds={rounds} (first round dropped as warm-up)"
    )
    header = (
        f"    {'sessions':>8}  {'policy_step/s':>13}  {'env-step/s':>11}  "
        f"{'batch p50':>9}  {'batch p99':>9}  {'env ranks':>9}"
    )
    print(header)
    failures = 0
    # Build the runtime only once: launching the same group name again via
    # ``WorkerGroup.launch`` fails immediately with ``ValueError: already
    # exists``, and starting actors is far more expensive than creating a
    # session.
    runtime = _build_bench_runtime(base, peak, args.launch)
    await runtime.start()
    try:
        gateway = runtime.gateway
        for level in levels:
            env_spec = EnvSpecMsg(
                env_family=base.env_family,
                env_config=dict(base.env_config),
                pool_size=per_rank,
            )
            created = await gateway.create_sessions(
                [
                    CreateSessionRequest(
                        application_id="bench",
                        client_session_key=f"bench-{level}-{index}",
                        env_spec=env_spec,
                        default_policy_id=base.rollout_worker.policy_id,
                        lease_seconds=base.gateway.default_lease_seconds,
                    )
                    for index in range(level)
                ]
            )
            errors = [item.error for item in created if isinstance(item, Err)]
            if errors:
                failures += len(errors)
                print(
                    f"    !! create_sessions: {errors[0].code.name}: "
                    f"{errors[0].message}"
                )
                continue
            session_ids = [item.value.session_id for item in created]
            ranks = sorted(
                {
                    gateway.sessions.get(session_id).worker_rank
                    for session_id in session_ids
                }
            )
            resets = await gateway.reset(session_ids, ResetSpec(seed=1))
            reset_errors = [item.error for item in resets if isinstance(item, Err)]
            if reset_errors:
                failures += len(reset_errors)
                print(f"    !! reset: {reset_errors[0].code.name}")
                await gateway.close_sessions(session_ids)
                continue

            latencies: list[float] = []
            started = 0.0
            for round_index in range(rounds):
                phase = time.perf_counter()
                results = await gateway.policy_step(session_ids, policy)
                elapsed = time.perf_counter() - phase
                bad = [item.error for item in results if isinstance(item, Err)]
                if bad:
                    failures += len(bad)
                    print(f"    !! policy_step: {bad[0].code.name}: {bad[0].message}")
                    break
                if round_index == 0:
                    started = time.perf_counter()
                    continue
                latencies.append(elapsed * 1000.0)
            if latencies:
                total = time.perf_counter() - started
                # Book by **actually completed** rounds: when a mid-batch
                # error causes a break, it must not be counted as a full
                # set of rounds.
                steps = level * len(latencies)
                ordered = sorted(latencies)
                p50 = statistics.median(ordered)
                p99 = ordered[min(len(ordered) - 1, int(len(ordered) * 0.99))]
                print(
                    f"    {level:>8}  {steps / total:>13.1f}  "
                    f"{steps * chunk / total:>11.1f}  {p50:>7.2f} ms  "
                    f"{p99:>7.2f} ms  {str(ranks):>9}"
                )
            await gateway.close_sessions(session_ids)
    finally:
        with contextlib.suppress(BaseException):
            await runtime.gateway.stop()
        with contextlib.suppress(BaseException):
            await runtime.aclose()
    print("==> OK" if failures == 0 else f"==> FAILED ({failures} error(s))")
    return 1 if failures else 0


def _build_bench_runtime(base: Any, sessions: int, launch: str) -> Any:
    """Build a runtime for one concurrency level.

    Args:
        base: The base configuration (shallow-copied and adjusted per level).
        sessions: The concurrent session count for this level.
        launch: ``"local"`` (workers in this process) or ``"ray"`` (workers
            launched as separate Ray processes).

    Returns:
        A runtime that has not been ``start()``-ed yet (``LocalRuntime`` or
        ``RayRuntime``).

    Raises:
        ValueError: ``--launch ray`` but the transport is not ``ray_channel``.
    """
    import copy

    config = copy.deepcopy(base)
    config.admission.max_sessions_per_application = max(
        config.admission.max_sessions_per_application, sessions
    )
    if launch == "ray":
        if config.transport.kind != "ray_channel":
            raise ValueError("--launch ray requires transport.kind='ray_channel'")
        from rollout_runtime.launch.ray_launch import build_ray_components

        return build_ray_components(config)
    from rollout_runtime.launch.local import build_local_components

    return build_local_components(config)


if __name__ == "__main__":  # pragma: no cover - convenience for `python -m rollout_runtime.cli`
    raise SystemExit(main())
