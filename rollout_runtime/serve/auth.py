"""Authentication and identity resolution for the served mode.

Three hard requirements are implemented here:

1. By default, bind only to ``127.0.0.1``;
2. When binding to a **non-loopback** address, ``RR_AUTH_TOKEN`` is
   **mandatory** and ``Authorization: Bearer`` is validated; if missing,
   startup is **refused** (``ServeSecurityError``, not "start now and worry
   later");
3. The ``application_id`` used by ``AdmissionController`` is resolved from the
   token, **never self-declared by the client**.

Authentication failures use ``INVALID_ARGUMENT`` + ``detail.reason=
"authentication"`` (no new ``ErrorCode`` is added). The HTTP status code is
only a projection of this error code, not a second error surface.

Two accepted forms for ``RR_AUTH_TOKEN``:

```
RR_AUTH_TOKEN=s3cr3t                       # single tenant; application_id comes from RR_AUTH_APPLICATION_ID
RR_AUTH_TOKEN=teamA:tokenA,teamB:tokenB    # multi-tenant; application_id is the part before the colon
```

Token comparison uses ``hmac.compare_digest`` to avoid leaking length/prefix
information via early byte-wise returns.
"""

from __future__ import annotations

import dataclasses
import hmac
import ipaddress
import os

from rollout_runtime.api.enums import ErrorCode
from rollout_runtime.api.errors import RuntimeApiError, make_error

__all__ = [
    "AUTH_APPLICATION_ENV",
    "AUTH_TOKEN_ENV",
    "DEFAULT_LOCAL_APPLICATION_ID",
    "ServeSecurityError",
    "TokenAuthority",
    "is_loopback_host",
]

AUTH_TOKEN_ENV = "RR_AUTH_TOKEN"
"""Environment variable name for the token table."""

AUTH_APPLICATION_ENV = "RR_AUTH_APPLICATION_ID"
"""Environment variable name for ``application_id`` in the single-tenant form."""

DEFAULT_LOCAL_APPLICATION_ID = "local"
"""``application_id`` used by the server in the loopback + no-token development mode.

This is likewise **not** self-declared by the client: in served mode, any
``application_id`` in the request body is always overridden, so quota
attribution rules are identical whether a token is configured or not.
"""


class ServeSecurityError(RuntimeError):
    """A security precondition is not satisfied; the service is **not allowed** to start."""


def is_loopback_host(host: str) -> bool:
    """Determine whether the bind address is a loopback address.

    Args:
        host: The ``--host`` value.

    Returns:
        True for loopback. Names that cannot be resolved to an IP are treated
        as loopback only if they belong to the ``localhost`` family; every
        other name (including ``""`` and ``"*"``) is treated as
        **non-loopback**, i.e. a token is required.
    """
    name = (host or "").strip()
    if not name:
        return False
    lowered = name.lower()
    if lowered in {"localhost", "localhost.", "ip6-localhost", "ip6-loopback"}:
        return True
    candidate = lowered
    if candidate.startswith("[") and candidate.endswith("]"):
        candidate = candidate[1:-1]
    candidate = candidate.split("%", 1)[0]
    try:
        address = ipaddress.ip_address(candidate)
    except ValueError:
        return False
    return address.is_loopback


def _parse_tokens(raw: str, environ: dict[str, str]) -> dict[str, str]:
    """Parse ``RR_AUTH_TOKEN``.

    Args:
        raw: Raw environment variable text.
        environ: The environment variable table — **must** be the one
            injected by the caller. Previously this read ``os.environ``
            directly, which meant the process environment could silently
            override an explicitly injected configuration, and in the
            single-tenant form (``RR_AUTH_TOKEN=<tok>``) ``application_id``
            attribution could become inconsistent with
            ``fallback_application_id`` (confirmed by an independent audit).

    Returns:
        A mapping from ``application_id`` to token.

    Raises:
        ServeSecurityError: On an empty token, a duplicate ``application_id``,
            or when ``RR_AUTH_TOKEN`` is set but no entry can be parsed out of
            it (``","`` / pure whitespace — silently disabling authentication
            is far more dangerous than failing outright).
    """
    default_application = environ.get(AUTH_APPLICATION_ENV, "").strip() or "default"
    tokens: dict[str, str] = {}
    for item in raw.split(","):
        entry = item.strip()
        if not entry:
            continue
        if ":" in entry:
            application_id, _, token = entry.partition(":")
            application_id = application_id.strip()
            token = token.strip()
        else:
            application_id, token = default_application, entry
        if not application_id or not token:
            raise ServeSecurityError(
                f"{AUTH_TOKEN_ENV} entries must be 'token' or 'application_id:token'; "
                "an empty application_id or token is not accepted"
            )
        if application_id in tokens:
            raise ServeSecurityError(
                f"{AUTH_TOKEN_ENV} declares application_id {application_id!r} twice"
            )
        tokens[application_id] = token
    if not tokens:
        raise ServeSecurityError(
            f"{AUTH_TOKEN_ENV} is set but declares no usable token; refusing to start "
            "with authentication silently disabled"
        )
    return tokens


@dataclasses.dataclass(frozen=True)
class TokenAuthority:
    """Resolves token → ``application_id``.

    Attributes:
        tokens: Mapping from ``application_id`` to expected token; empty
            means no authentication (only loopback binds are allowed).
        fallback_application_id: The ``application_id`` used by the server
            when authentication is disabled.
    """

    tokens: dict[str, str] = dataclasses.field(default_factory=dict)
    fallback_application_id: str = DEFAULT_LOCAL_APPLICATION_ID

    @property
    def enabled(self) -> bool:
        """Whether the ``Authorization`` header is enforced.

        Returns:
            True if at least one token is configured.
        """
        return bool(self.tokens)

    @property
    def application_ids(self) -> list[str]:
        """The configured list of tenants (for startup logs and ``/healthz``).

        Returns:
            Sorted ``application_id`` values.
        """
        return sorted(self.tokens)

    @classmethod
    def from_environment(
        cls, *, host: str, environ: dict[str, str] | None = None
    ) -> TokenAuthority:
        """Construct from the bind address and environment variables, and
        enforce the "refuse to start" decision here.

        Args:
            host: The bind address.
            environ: Environment variable table; ``None`` uses ``os.environ``.

        Returns:
            The authority.

        Raises:
            ServeSecurityError: Binding to a non-loopback address without
                ``RR_AUTH_TOKEN``.
        """
        env = os.environ if environ is None else environ
        declared = env.get(AUTH_TOKEN_ENV)
        raw = (declared or "").strip()
        if declared is not None and declared != "" and not raw:
            # Set but blank: indistinguishable from "unset" would make an
            # operator believe auth is on, so fail loudly instead.
            raise ServeSecurityError(
                f"{AUTH_TOKEN_ENV} is set but blank; refusing to start with "
                "authentication silently disabled"
            )
        tokens = _parse_tokens(raw, dict(env)) if raw else {}
        if not tokens and not is_loopback_host(host):
            raise ServeSecurityError(
                f"refusing to bind {host!r}: a non-loopback bind requires "
                f"{AUTH_TOKEN_ENV} (checked against 'Authorization: Bearer <token>'). "
                "Either set it, or bind 127.0.0.1 for local-only use."
            )
        fallback = (
            env.get(AUTH_APPLICATION_ENV, "").strip() or DEFAULT_LOCAL_APPLICATION_ID
        )
        return cls(tokens=tokens, fallback_application_id=fallback)

    def resolve(self, authorization: str | None) -> str:
        """Resolve ``application_id`` from the ``Authorization`` header.

        Args:
            authorization: The raw HTTP ``Authorization`` header value.

        Returns:
            The resolved ``application_id``; ``fallback_application_id`` when
            authentication is disabled.

        Raises:
            RuntimeApiError: Header missing, scheme is not ``Bearer``, or the
                token does not match any tenant (``INVALID_ARGUMENT`` +
                ``detail.reason="authentication"``).
        """
        if not self.enabled:
            return self.fallback_application_id
        if not authorization:
            raise self._reject("missing Authorization header")
        scheme, _, token = authorization.partition(" ")
        if scheme.strip().lower() != "bearer" or not token.strip():
            raise self._reject("Authorization must be 'Bearer <token>'")
        # Compare as **bytes**: ``hmac.compare_digest`` requires ASCII-only
        # str, and a non-ASCII token would raise TypeError, which gets
        # normalized into 400 + `detail.exception=TypeError` (leaking a
        # CPython internal error back to an unauthenticated caller, and
        # losing `reason="authentication"`).
        candidate = token.strip().encode("utf-8", "surrogatepass")
        for application_id, expected in self.tokens.items():
            if hmac.compare_digest(
                candidate, expected.encode("utf-8", "surrogatepass")
            ):
                return application_id
        raise self._reject("bearer token is not recognized")

    @staticmethod
    def _reject(message: str) -> RuntimeApiError:
        """Construct an authentication failure error.

        Args:
            message: Human-readable reason (**never echoes the token**).

        Returns:
            A ``RuntimeApiError`` with ``reason="authentication"``.
        """
        return RuntimeApiError(
            make_error(
                ErrorCode.INVALID_ARGUMENT,
                f"authentication failed: {message}",
                reason="authentication",
            )
        )
