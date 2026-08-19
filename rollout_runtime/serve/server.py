"""Execution body for ``rollout-runtime serve``.

The startup order is intentional: **run security checks before building
anything**.

```text
TokenAuthority.from_environment(host=...)   # non-loopback + no RR_AUTH_TOKEN -> refuse to start immediately
        |
load_config -> derive gateway_epoch
        |
build_local_components / build_ray_components -> runtime.start()   (real env / policy backends)
        |
build_app(gateway, ...) -> uvicorn.Server(...).serve()             (single process, single event loop)
        |
finally: gateway.stop() -> runtime.aclose()
```

In other words, "refuse to start when the token is missing" happens
**before** loading weights, starting Ray, or building the simulation pool —
otherwise a configuration error would burn several minutes of GPU loading
time before failing.

For the concurrency model: ``workers`` is deliberately not exposed as a
parameter; the Gateway and both worker groups live in this single process and
event loop (the single-writer design, D2).
"""

from __future__ import annotations

import asyncio
import dataclasses
import time
from typing import Any

from rollout_runtime.serve.app import ServeLimits, build_app
from rollout_runtime.serve.auth import TokenAuthority

__all__ = ["ServeOptions", "ServedRuntime", "run_serve"]


@dataclasses.dataclass(kw_only=True)
class ServeOptions:
    """Effective parameters for ``serve``.

    Attributes:
        config: Preset name / yaml path.
        host: Bind address; a non-loopback address requires ``RR_AUTH_TOKEN``.
        port: Bind port.
        launch: ``"local"`` (workers stay in this process) or ``"ray"``
            (workers are launched as separate Ray processes).
        gateway_epoch: Explicit epoch; ``None`` uses the process start time.
        limits: Clamps on untrusted input.
        log_level: uvicorn log level.
    """

    config: str = "local_fake"
    host: str = "127.0.0.1"
    port: int = 8710
    launch: str = "local"
    gateway_epoch: int | None = None
    limits: ServeLimits = dataclasses.field(default_factory=ServeLimits)
    log_level: str = "info"


@dataclasses.dataclass
class ServedRuntime:
    """An already-started served runtime (lets tests reach the app and gateway directly).

    Attributes:
        runtime: A ``LocalRuntime`` or ``RayRuntime``.
        app: The FastAPI application.
        authority: The authenticator.
        epoch: The ``gateway_epoch`` in effect for this run.
    """

    runtime: Any
    app: Any
    authority: TokenAuthority
    epoch: int

    @property
    def gateway(self) -> Any:
        """Return the Gateway.

        Returns:
            The ``RuntimeGateway``.
        """
        return self.runtime.gateway

    async def aclose(self) -> None:
        """Shut down in order: Gateway then runtime (idempotent).

        Each step is individually wrapped in ``try/except``, but
        **cancellation is still re-raised**: swallowing ``CancelledError``
        during shutdown would break upstream timeout/cancellation handling
        (flagged by an independent audit). Real shutdown errors are printed,
        not silently dropped.
        """
        for label, step in (
            ("gateway.stop", self.runtime.gateway.stop),
            ("runtime.aclose", self.runtime.aclose),
        ):
            try:
                await step()
            except (asyncio.CancelledError, GeneratorExit):
                raise
            except BaseException as exc:  # noqa: BLE001 - shutdown failure must not be hidden, but also must not leak
                print(f"!! serve shutdown: {label} failed: {exc!r}", flush=True)


async def build_served_runtime(
    options: ServeOptions, *, environ: dict[str, str] | None = None
) -> ServedRuntime:
    """Run the security check, build the runtime, and construct the app.

    Args:
        options: Effective parameters.
        environ: Environment variable table; ``None`` uses ``os.environ``
            (tests can inject their own).

    Returns:
        The started ``ServedRuntime``.

    Raises:
        ServeSecurityError: Binding to a non-loopback address without
            ``RR_AUTH_TOKEN`` (in which case **nothing has been built yet**).
        ValueError: ``--launch ray`` but the transport is not ``ray_channel``.
    """
    from rollout_runtime.config.schema import load_config

    # 1) The security check must run before anything is built.
    authority = TokenAuthority.from_environment(host=options.host, environ=environ)

    # 2) Configuration and epoch.
    config = load_config(options.config)
    epoch = (
        int(options.gateway_epoch)
        if options.gateway_epoch is not None
        else int(time.time())
    )
    config.gateway.gateway_epoch = epoch

    # 3) Build the runtime.
    if options.launch == "ray":
        if config.transport.kind != "ray_channel":
            raise ValueError("--launch ray requires transport.kind='ray_channel'")
        from rollout_runtime.launch.ray_launch import build_ray_components

        runtime = build_ray_components(config)
    else:
        from rollout_runtime.launch.local import build_local_components

        runtime = build_local_components(config)
    await runtime.start()

    # 4) Clamps: items not explicitly provided are derived from the effective
    # configuration, so the "server-side limit" stays consistent with the
    # deployment shape.
    limits = dataclasses.replace(
        options.limits,
        max_lease_seconds=options.limits.max_lease_seconds
        or config.gateway.default_lease_seconds,
        max_pool_size=options.limits.max_pool_size
        or config.env_worker.max_sessions_per_rank,
        allowed_env_families=options.limits.allowed_env_families
        or frozenset({config.env_family}),
    )
    app = build_app(runtime.gateway, authority=authority, limits=limits)
    return ServedRuntime(runtime=runtime, app=app, authority=authority, epoch=epoch)


async def run_serve(
    options: ServeOptions, *, environ: dict[str, str] | None = None
) -> int:
    """Start the HTTP service and block until a stop signal is received.

    Args:
        options: Effective parameters.
        environ: Environment variable table; ``None`` uses ``os.environ``.

    Returns:
        The process exit code.
    """
    import uvicorn

    served = await build_served_runtime(options, environ=environ)
    config = served.runtime.config
    print(f"==> rollout-runtime serve ({options.config})", flush=True)
    print(
        f"    bind=http://{options.host}:{options.port} "
        f"transport={config.transport.kind} launch={options.launch} "
        f"env_family={config.env_family}",
        flush=True,
    )
    print(
        f"    auth={'bearer' if served.authority.enabled else 'disabled (loopback)'} "
        f"applications={served.authority.application_ids or ['<none>']} "
        f"gateway_epoch={served.epoch}",
        flush=True,
    )
    print(
        "    single process by design: the Gateway is the "
        "asyncio single writer (D2)",
        flush=True,
    )
    server = uvicorn.Server(
        uvicorn.Config(
            app=served.app,
            host=options.host,
            port=options.port,
            log_level=options.log_level,
            # Deliberately not exposing workers: multiple processes would
            # mean multiple Gateways each writing their own session table (D2).
            workers=1,
            lifespan="off",
            access_log=False,
            # Don't return `server: uvicorn`: the served mode may be exposed
            # externally, and there's no need to leak the version fingerprint.
            server_header=False,
        )
    )
    try:
        await server.serve()
    finally:
        await served.aclose()
    return 0
