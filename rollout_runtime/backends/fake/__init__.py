"""Fake backend.

``FakeEnvCore``: a deterministic state machine ``state = f(seed, step)``,
terminated after N steps, with configurable fault injection / slow steps /
hangs. ``FakePolicyCore``: returns predictable ``[chunk, 7]`` output, with
configurable latency, failure, and hangs. The two are the sole dependency of
the 8 end-to-end assertions (idempotency, out-of-order, late-result
rejection, the three cancellation states, backpressure, error isolation) and
can run on a local CPU alone.

The env side has **blocking** semantics (called by EnvWorker via
``asyncio.to_thread``; once an env step starts it cannot be cancelled), while
the policy side has **asyncio** semantics (waits are cancellable) — this
corresponds exactly to the staged cancellation semantics described for the
control plane.
"""

from __future__ import annotations

from rollout_runtime.backends.fake.env import (
    FAKE_ENV_EXTENSIONS,
    FAKE_ENV_FAMILY,
    FakeEnvConfig,
    FakeEnvCore,
    FakeEnvFamily,
    fake_env_capability,
    register_fake_env_family,
)
from rollout_runtime.backends.fake.policy import (
    FAKE_POLICY_FAMILY,
    FakePolicyConfig,
    FakePolicyCore,
)

__all__ = [
    "FAKE_ENV_EXTENSIONS",
    "FAKE_ENV_FAMILY",
    "FAKE_POLICY_FAMILY",
    "FakeEnvConfig",
    "FakeEnvCore",
    "FakeEnvFamily",
    "FakePolicyConfig",
    "FakePolicyCore",
    "fake_env_capability",
    "register_fake_env_family",
]
