"""Shared fixtures and pytest configuration for ``tests/runtime``.

- ``asyncio_mode=auto``: no need to add ``@pytest.mark.asyncio`` to every coroutine test.
- ``remote`` marker: for cases that require a configured GPU host; skipped locally by default.
- Time source fixture: all Gateway-side components support injecting a ``time_source``,
  which tests use in place of sleep.
- e2e fixture: ``local_runtime`` starts a full in-process runtime (Gateway + EnvWorker +
  RolloutWorker + fake backend); ``fake_env_spec`` generates an env spec.

The ``--transport`` option is defined in ``tests/conftest.py`` (must be the initial conftest).
"""

from __future__ import annotations

import contextlib
from collections.abc import AsyncIterator, Callable
from typing import Any

import pytest

from rollout_runtime.api.messages import CreateSessionRequest, EnvSpecMsg
from rollout_runtime.api.result import Err, unwrap
from rollout_runtime.launch.local import LocalRuntime, build_local_components


def pytest_configure(config: pytest.Config) -> None:
    """Register markers.

    The authoritative declaration of ``asyncio_mode = "auto"`` and the ``remote`` marker
    lives in ``pyproject.toml::[tool.pytest.ini_options]``; this is just a fallback
    registration to avoid unknown-marker warnings when running this directory alone
    from a different rootdir.

    Args:
        config: pytest config object.
    """
    config.addinivalue_line(
        "markers", "remote: requires a configured GPU host (real env / VLA backends)"
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Automatically skip ``remote`` cases in the local environment.

    Only yield when the ``-m`` expression **explicitly mentions** ``remote`` (``-m
    remote`` runs on a GPU machine, ``-m "not remote"`` is used by CI). The previous
    implementation skipped nothing whenever any ``-m`` was given, so an unrelated
    filter like ``-m "not ray"`` would actually run the remote cases locally.

    Args:
        config: pytest config object.
        items: The collected test cases.
    """
    expression = config.getoption("-m") or ""
    if "remote" in expression:
        return
    skip_remote = pytest.mark.skip(
        reason="remote-only test; run on a configured GPU host with `-m remote`"
    )
    for item in items:
        if "remote" in item.keywords:
            item.add_marker(skip_remote)


class FakeClock:
    """A manually advanceable time source.

    Attributes:
        now: Current timestamp.
    """

    def __init__(self, start: float = 1_000_000.0) -> None:
        """Initialize.

        Args:
            start: Starting timestamp.
        """
        self.now = start

    def __call__(self) -> float:
        """Return the current time.

        Returns:
            The current timestamp.
        """
        return self.now

    def advance(self, seconds: float) -> float:
        """Advance the time.

        Args:
            seconds: Number of seconds to advance.

        Returns:
            The timestamp after advancing.
        """
        self.now += seconds
        return self.now


@pytest.fixture
def clock() -> FakeClock:
    """Provide a controllable time source.

    Returns:
        A ``FakeClock`` instance.
    """
    return FakeClock()


@pytest.fixture
def transport_kind(request: pytest.FixtureRequest) -> str:
    """The transport kind currently under test.

    Both kinds are actually run: ``ray_channel`` starts a local Ray cluster and a real
    rlinf ``Channel`` actor (see the topology notes in ``launch/local.py``).

    Args:
        request: pytest fixture request object.

    Returns:
        ``"inproc"`` or ``"ray_channel"``.
    """
    return str(request.config.getoption("--transport"))


def local_runtime_config(transport_kind: str = "inproc", **overrides: Any) -> Any:
    """Build a config based on the ``local_fake`` preset, overriding fields by section.

    The config dataclass is deliberately not frozen (omegaconf raises
    ``ReadonlyConfigError`` for frozen dataclasses), so ``setattr`` is used directly here.

    Args:
        transport_kind: Written into ``transport.kind``, determines which transport is assembled.
        **overrides: Per-section overrides shaped like ``transport={"infer_request_queue_size": 2}``.

    Returns:
        A ``RuntimeConfig`` that can be passed directly to ``build_local_components``.
    """
    from rollout_runtime.config.schema import load_config

    config = load_config("local_fake")
    config.transport.kind = transport_kind
    for section_name, fields in overrides.items():
        section = getattr(config, section_name)
        for field, value in fields.items():
            setattr(section, field, value)
    return config


@pytest.fixture
async def local_runtime(transport_kind: str) -> AsyncIterator[LocalRuntime]:
    """Start a full in-process runtime (default ``local_fake`` preset).

    Args:
        transport_kind: The transport under test.

    Yields:
        A ``LocalRuntime`` that has already been ``start()``-ed.
    """
    runtime = build_local_components(local_runtime_config(transport_kind))
    await runtime.start()
    try:
        yield runtime
    finally:
        with contextlib.suppress(BaseException):
            await runtime.gateway.stop()
        await runtime.aclose()


@pytest.fixture
def fake_env_spec() -> Callable[..., EnvSpecMsg]:
    """Return a factory that builds a fake env spec.

    Returns:
        ``factory(pool_size=1, **env_config)``; differences in ``env_config`` change the
        digest, and therefore switch to an independent env pool (used for fault isolation).
    """

    def factory(
        pool_size: int = 1,
        *,
        max_dynamic_pool_size: int | None = None,
        **env_config: Any,
    ) -> EnvSpecMsg:
        merged: dict[str, Any] = {
            "action_dim": 7,
            "chunk_size": 4,
            "episode_length": 16,
            "image_height": 16,
            "image_width": 16,
            "state_dim": 8,
        }
        merged.update(env_config)
        return EnvSpecMsg(
            env_family="fake",
            env_config=merged,
            pool_size=pool_size,
            max_dynamic_pool_size=max_dynamic_pool_size,
        )

    return factory


async def open_sessions(
    runtime: LocalRuntime,
    env_spec: EnvSpecMsg,
    count: int = 1,
    *,
    application_id: str = "test",
    key_prefix: str = "s",
    policy_id: str = "fake",
) -> list[Any]:
    """Create sessions in bulk and assert they all succeed.

    Args:
        runtime: In-process runtime.
        env_spec: Environment spec; ``pool_size`` must be ``>= count`` (the pool does not grow).
        count: Number of sessions.
        application_id: Owning application.
        key_prefix: Prefix for ``client_session_key``.
        policy_id: Default policy.

    Returns:
        List of session ids, in creation order.
    """
    requests = [
        CreateSessionRequest(
            application_id=application_id,
            client_session_key=f"{key_prefix}-{index}",
            env_spec=env_spec,
            default_policy_id=policy_id,
            lease_seconds=60.0,
        )
        for index in range(count)
    ]
    results = await runtime.gateway.create_sessions(requests)
    failures = [result.error for result in results if isinstance(result, Err)]
    assert not failures, f"create_sessions failed: {failures}"
    return [unwrap(result).session_id for result in results]
