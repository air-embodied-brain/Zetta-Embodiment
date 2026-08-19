"""Served mode (``rollout-runtime serve``).

Design decision D2: the embedded and served modes **share the same
``RuntimeGateway`` class**; served just wraps it with HTTP. This subpackage
therefore follows the same rules as ``gateway/``: it only touches the Runtime
through ``RuntimeClient`` and does not import ``ray`` / ``rlinf`` / ``torch``
(enforced by ``tests/runtime/test_layering.py``).
"""

from rollout_runtime.serve.app import ServeLimits, build_app, http_status_for
from rollout_runtime.serve.auth import (
    AUTH_APPLICATION_ENV,
    AUTH_TOKEN_ENV,
    ServeSecurityError,
    TokenAuthority,
    is_loopback_host,
)

__all__ = [
    "AUTH_APPLICATION_ENV",
    "AUTH_TOKEN_ENV",
    "ServeLimits",
    "ServeSecurityError",
    "TokenAuthority",
    "build_app",
    "http_status_for",
    "is_loopback_host",
]
