# Copyright (c) 2026 Zetta Contributors
"""Ordered, process-safe failover for API planner model providers.

The pool is configured through ``ZETTA_API_PROVIDERS``.  The value is JSON and
may be either a list of routes, a mapping of route names to route objects, or an
object containing a ``providers``/``routes`` list.  A route is one credentialed
endpoint: two API keys for the same URL are deliberately two distinct routes.

Only opaque route fingerprints and cooldown timestamps are persisted.  URLs
and API keys never enter the shared health file or failover log messages.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import inspect
import json
import os
import re
import tempfile
import threading
import time
from collections.abc import AsyncIterator, Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

from pydantic_ai.exceptions import ModelAPIError, ModelHTTPError
from pydantic_ai.messages import ModelMessage, ModelResponse
from pydantic_ai.models import Model, ModelRequestParameters, infer_model
from pydantic_ai.providers import infer_provider_class

from zetta.evolution.provider_admission import ProviderFailure
from zetta.planner.provider_request_admission import (
    ProviderRequestAdmission,
    ProviderRequestLease,
)
from zetta.utils.logging import get_logger

logger = get_logger("provider_pool")

PROVIDERS_ENV = "ZETTA_API_PROVIDERS"
COOLDOWN_ENV = "ZETTA_API_PROVIDER_COOLDOWN_S"
STATE_FILE_ENV = "ZETTA_API_PROVIDER_STATE_FILE"
DEFAULT_COOLDOWN_SECONDS = 30.0

_SAFE_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
_RETRYABLE_HTTP_STATUSES = {401, 403, 404, 408, 409, 425, 429, 500, 502, 503, 504}
_RETRYABLE_EXCEPTION_NAMES = {
    "APIConnectionError",
    "APITimeoutError",
    "ConnectError",
    "ConnectTimeout",
    "ConnectionError",
    "PoolTimeout",
    "RateLimitError",
    "ReadError",
    "ReadTimeout",
    "RemoteProtocolError",
    "TimeoutError",
    "WriteError",
    "WriteTimeout",
}

_PATH_LOCKS: dict[str, threading.Lock] = {}
_PATH_LOCKS_GUARD = threading.Lock()


class ProviderPoolConfigError(ValueError):
    """Raised for invalid provider-pool configuration without echoing secrets."""


@dataclass(frozen=True)
class ProviderRouteSpec:
    """One ordered, credentialed model endpoint."""

    name: str
    model: str
    provider: str
    base_url: str
    api_key: str = field(repr=False)
    price: float | None = None
    cooldown_seconds: float = DEFAULT_COOLDOWN_SECONDS
    api_version: str | None = None
    max_concurrency: int | None = None
    rpm_limit: int | None = None
    tpm_limit: int | None = None
    source_index: int = 0

    @property
    def route_id(self) -> str:
        """Return a non-reversible identifier used by the shared health file."""
        raw = "\0".join((self.name, self.provider, self.base_url, self.api_key))
        return hashlib.sha256(raw.encode()).hexdigest()[:24]


@dataclass(frozen=True)
class ProviderPoolConfig:
    """Validated and price-ordered provider routes."""

    routes: tuple[ProviderRouteSpec, ...]
    state_file: Path


@dataclass(frozen=True)
class ProviderModelRoute:
    """A validated route paired with its concrete pydantic-ai model."""

    spec: ProviderRouteSpec
    model: Model = field(repr=False)


def load_provider_pool_config(
    *,
    default_model: str,
    env: Mapping[str, str] | None = None,
) -> ProviderPoolConfig | None:
    """Parse ``ZETTA_API_PROVIDERS`` without logging its value.

    Routes with ``price`` are sorted cheapest first.  Routes without a price
    retain their relative input order after priced routes.  A route may store
    ``api_key`` directly in the JSON or refer to a second environment variable
    with ``api_key_env``.
    """
    environ = os.environ if env is None else env
    raw = environ.get(PROVIDERS_ENV, "").strip()
    if not raw:
        return None
    try:
        document = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ProviderPoolConfigError(
            f"{PROVIDERS_ENV} must contain valid JSON"
        ) from exc

    default_cooldown = _positive_float(
        environ.get(COOLDOWN_ENV, DEFAULT_COOLDOWN_SECONDS),
        field_name=COOLDOWN_ENV,
    )
    state_file_value = environ.get(STATE_FILE_ENV)
    route_items, document_options = _route_items(document)
    if "cooldown_seconds" in document_options:
        default_cooldown = _positive_float(
            document_options["cooldown_seconds"],
            field_name="cooldown_seconds",
        )
    if document_options.get("state_file"):
        state_file_value = str(document_options["state_file"])

    routes: list[ProviderRouteSpec] = []
    for index, (mapping_name, value) in enumerate(route_items):
        if not isinstance(value, Mapping):
            raise ProviderPoolConfigError(f"provider route {index} must be an object")
        name = str(
            value.get("name")
            or value.get("site")
            or mapping_name
            or f"route-{index + 1}"
        )
        if not _SAFE_NAME_RE.fullmatch(name):
            raise ProviderPoolConfigError(
                f"provider route {index} has an unsafe name; use letters, digits, '.', '_' or '-'"
            )

        model = str(value.get("model") or default_model).strip()
        if not model:
            raise ProviderPoolConfigError(f"provider route {name!r} has no model")
        provider = str(value.get("provider") or _model_prefix(model)).strip()
        if not provider:
            raise ProviderPoolConfigError(f"provider route {name!r} has no provider")
        model = _model_for_provider(model, provider)

        base_url = str(value.get("base_url") or value.get("url") or "").strip()
        if not base_url.startswith(("http://", "https://")):
            raise ProviderPoolConfigError(
                f"provider route {name!r} must have an http(s) base_url"
            )

        api_key = value.get("api_key") or value.get("key")
        key_env = value.get("api_key_env")
        if api_key is not None and key_env is not None:
            raise ProviderPoolConfigError(
                f"provider route {name!r} must use only one of api_key or api_key_env"
            )
        if key_env is not None:
            key_env = str(key_env).strip()
            api_key = environ.get(key_env)
            if not api_key:
                raise ProviderPoolConfigError(
                    f"provider route {name!r} references an unset API-key environment variable"
                )
        api_key = str(api_key or "").strip()
        if not api_key:
            raise ProviderPoolConfigError(f"provider route {name!r} has no API key")

        price_value = value.get("price")
        price = None
        if price_value is not None:
            price = _nonnegative_float(price_value, field_name=f"{name}.price")
        cooldown = _positive_float(
            value.get("cooldown_seconds", default_cooldown),
            field_name=f"{name}.cooldown_seconds",
        )
        api_version = value.get("api_version")
        max_concurrency = _optional_positive_int(
            value.get("max_concurrency"), field_name=f"{name}.max_concurrency"
        )
        rpm_limit = _optional_positive_int(
            value.get("rpm_limit", value.get("rpm")),
            field_name=f"{name}.rpm_limit",
        )
        tpm_limit = _optional_positive_int(
            value.get("tpm_limit", value.get("tpm")),
            field_name=f"{name}.tpm_limit",
        )
        base_url, derived_api_version = _normalize_provider_endpoint(base_url, provider)
        if not api_version:
            api_version = derived_api_version
        routes.append(
            ProviderRouteSpec(
                name=name,
                model=model,
                provider=provider,
                base_url=base_url,
                api_key=api_key,
                price=price,
                cooldown_seconds=cooldown,
                api_version=str(api_version).strip() if api_version else None,
                max_concurrency=max_concurrency,
                rpm_limit=rpm_limit,
                tpm_limit=tpm_limit,
                source_index=index,
            )
        )

    if not routes:
        raise ProviderPoolConfigError(f"{PROVIDERS_ENV} contains no provider routes")
    routes.sort(
        key=lambda route: (
            route.price is None,
            route.price if route.price is not None else 0.0,
            route.source_index,
        )
    )

    if state_file_value:
        state_file = Path(state_file_value).expanduser().resolve()
    else:
        fingerprint = hashlib.sha256(
            "\0".join(route.route_id for route in routes).encode()
        ).hexdigest()[:20]
        state_file = (
            Path(tempfile.gettempdir()) / f"zetta-provider-pool-{fingerprint}.json"
        )
    return ProviderPoolConfig(routes=tuple(routes), state_file=state_file)


def build_provider_pool_model(config: ProviderPoolConfig) -> "FailoverModel":
    """Construct all concrete provider models and wrap them in a failover model."""
    model_routes: list[ProviderModelRoute] = []
    for spec in config.routes:
        provider_cls = infer_provider_class(spec.provider)
        parameters = inspect.signature(provider_cls.__init__).parameters
        kwargs: dict[str, Any] = {}
        if "api_key" in parameters:
            kwargs["api_key"] = spec.api_key
        else:
            raise ProviderPoolConfigError(
                f"provider route {spec.name!r} does not accept an explicit API key"
            )
        if "base_url" in parameters:
            kwargs["base_url"] = spec.base_url
        elif "azure_endpoint" in parameters:
            kwargs["azure_endpoint"] = spec.base_url
        else:
            raise ProviderPoolConfigError(
                f"provider route {spec.name!r} does not accept a base URL"
            )
        if spec.api_version and "api_version" in parameters:
            kwargs["api_version"] = spec.api_version
        provider = provider_cls(**kwargs)
        model = infer_model(spec.model, provider_factory=lambda _name, p=provider: p)
        model_routes.append(ProviderModelRoute(spec=spec, model=model))
    return FailoverModel(
        model_routes,
        state_file=config.state_file,
        request_admission=ProviderRequestAdmission.from_environment(),
    )


def build_provider_broker_model(
    config: ProviderPoolConfig,
    *,
    base_url: str,
    api_key: str,
) -> Model:
    """Build one Responses client pointed at the standalone central broker.

    The broker owns route selection, queuing, concurrency and fallback.  Client
    processes therefore receive no local ``FailoverModel`` and cannot bypass
    the central dispatch queue.
    """
    normalized = base_url.rstrip("/")
    if not normalized.endswith("/v1"):
        normalized += "/v1"
    provider_cls = infer_provider_class("openai-responses")
    parameters = inspect.signature(provider_cls.__init__).parameters
    if "api_key" not in parameters or "base_url" not in parameters:
        raise ProviderPoolConfigError(
            "openai-responses provider cannot connect to the central broker"
        )
    provider = provider_cls(api_key=api_key, base_url=normalized)
    configured = config.routes[0].model
    model_name = configured.partition(":")[2] or configured
    return infer_model(
        f"openai-responses:{model_name}",
        provider_factory=lambda _name: provider,
    )


class FailoverModel(Model):
    """A pydantic-ai model that retries each request on ordered healthy routes."""

    def __init__(
        self,
        routes: Sequence[ProviderModelRoute],
        *,
        state_file: Path,
        clock: Callable[[], float] = time.time,
        sleeper: Callable[[float], Any] = asyncio.sleep,
        request_admission: ProviderRequestAdmission | None = None,
    ) -> None:
        if not routes:
            raise ValueError("FailoverModel requires at least one route")
        self._routes = tuple(routes)
        self._clock = clock
        self._sleeper = sleeper
        self._cooldowns = _SharedCooldowns(state_file, clock=clock)
        self._request_admission = request_admission
        super().__init__(profile=self._routes[0].model.profile)

    @property
    def model_name(self) -> str:
        """Expose a stable, endpoint-free model name."""
        return f"failover:{self._routes[0].model.model_name}"

    @property
    def system(self) -> str:
        """Use the first (preferred) route's model system."""
        return self._routes[0].model.system

    @property
    def base_url(self) -> str | None:
        """Do not expose a credentialed route's endpoint through the wrapper."""
        return None

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: Any,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        """Attempt one model request on each eligible route, cheapest first."""
        attempted: set[str] = set()
        last_error: Exception | None = None
        while len(attempted) < len(self._routes):
            route = await self._next_route(attempted)
            attempted.add(route.spec.route_id)
            lease = await self._acquire(
                route,
                estimated_tokens=_estimated_request_tokens(
                    messages, model_settings, model_request_parameters
                ),
            )
            started = time.monotonic()
            try:
                response = await route.model.request(
                    messages, model_settings, model_request_parameters
                )
            except Exception as exc:
                if not is_retryable_provider_error(exc):
                    await self._release(lease)
                    raise
                last_error = exc
                await self._fail(
                    lease,
                    _provider_failure(exc),
                    latency_s=time.monotonic() - started,
                )
                cooldown = max(
                    route.spec.cooldown_seconds,
                    _retry_after_seconds(exc) or 0.0,
                )
                self._cooldowns.mark_failure(route.spec.route_id, cooldown)
                logger.warning(
                    "provider route %s unavailable (%s); cooling for %.1fs and falling back",
                    route.spec.name,
                    _safe_error_label(exc),
                    cooldown,
                )
                continue
            await self._succeed(
                lease,
                tokens=_usage_tokens(response),
                latency_s=time.monotonic() - started,
            )
            self._cooldowns.mark_success(route.spec.route_id)
            return response
        assert last_error is not None
        raise last_error

    @contextlib.asynccontextmanager
    async def request_stream(
        self,
        messages: list[ModelMessage],
        model_settings: Any,
        model_request_parameters: ModelRequestParameters,
        run_context: Any = None,
    ) -> AsyncIterator[Any]:
        """Fail over before a stream starts; never replay a partially read stream."""
        attempted: set[str] = set()
        last_error: Exception | None = None
        while len(attempted) < len(self._routes):
            route = await self._next_route(attempted)
            attempted.add(route.spec.route_id)
            lease = await self._acquire(
                route,
                estimated_tokens=_estimated_request_tokens(
                    messages, model_settings, model_request_parameters
                ),
            )
            started = time.monotonic()
            manager = route.model.request_stream(
                messages,
                model_settings,
                model_request_parameters,
                run_context=run_context,
            )
            try:
                stream = await manager.__aenter__()
            except Exception as exc:
                if not is_retryable_provider_error(exc):
                    await self._release(lease)
                    raise
                last_error = exc
                await self._fail(
                    lease,
                    _provider_failure(exc),
                    latency_s=time.monotonic() - started,
                )
                cooldown = max(
                    route.spec.cooldown_seconds,
                    _retry_after_seconds(exc) or 0.0,
                )
                self._cooldowns.mark_failure(route.spec.route_id, cooldown)
                logger.warning(
                    "provider route %s unavailable before streaming (%s); cooling for %.1fs and falling back",
                    route.spec.name,
                    _safe_error_label(exc),
                    cooldown,
                )
                continue
            exit_args: tuple[Any, Any, Any] = (None, None, None)
            try:
                yield stream
            except BaseException as exc:
                exit_args = (type(exc), exc, exc.__traceback__)
                await self._fail(
                    lease,
                    ProviderFailure.STREAM,
                    latency_s=time.monotonic() - started,
                )
                self._cooldowns.mark_failure(
                    route.spec.route_id, route.spec.cooldown_seconds
                )
                raise
            else:
                await self._succeed(
                    lease,
                    tokens=_usage_tokens(stream),
                    latency_s=time.monotonic() - started,
                )
                self._cooldowns.mark_success(route.spec.route_id)
            finally:
                await manager.__aexit__(*exit_args)
            return
        assert last_error is not None
        raise last_error

    async def _next_route(self, attempted: set[str]) -> ProviderModelRoute:
        remaining = [
            route for route in self._routes if route.spec.route_id not in attempted
        ]
        while True:
            cooldowns = self._cooldowns.cooldown_until(
                [route.spec.route_id for route in remaining]
            )
            now = self._clock()
            for route in remaining:
                if cooldowns.get(route.spec.route_id, 0.0) <= now:
                    return route
            delay = max(0.0, min(cooldowns.values()) - now)
            if delay:
                logger.info(
                    "all remaining provider routes are cooling; waiting %.1fs", delay
                )
                await self._sleeper(delay)

    async def _acquire(
        self, route: ProviderModelRoute, *, estimated_tokens: int
    ) -> ProviderRequestLease | None:
        if self._request_admission is None:
            return None
        return await self._request_admission.acquire(
            route.spec.route_id,
            estimated_tokens=estimated_tokens,
            max_concurrency=route.spec.max_concurrency,
            rpm_limit=route.spec.rpm_limit,
            tpm_limit=route.spec.tpm_limit,
        )

    @staticmethod
    async def _succeed(
        lease: ProviderRequestLease | None, *, tokens: int, latency_s: float
    ) -> None:
        if lease is not None:
            await lease.succeed(tokens=tokens, latency_s=latency_s)

    @staticmethod
    async def _fail(
        lease: ProviderRequestLease | None,
        failure: ProviderFailure,
        *,
        latency_s: float,
    ) -> None:
        if lease is not None:
            await lease.fail(failure, tokens=0, latency_s=latency_s)

    @staticmethod
    async def _release(lease: ProviderRequestLease | None) -> None:
        if lease is not None:
            await lease.release()


def _estimated_request_tokens(
    messages: Sequence[ModelMessage],
    model_settings: Any,
    model_request_parameters: ModelRequestParameters,
) -> int:
    """Reserve a conservative, configurable TPM share before dispatch."""

    del messages, model_request_parameters
    raw = os.environ.get("ZETTA_PROVIDER_ESTIMATED_TOKENS", "25000")
    try:
        estimate = int(raw)
    except ValueError as exc:
        raise ValueError("ZETTA_PROVIDER_ESTIMATED_TOKENS must be an integer") from exc
    if estimate < 0:
        raise ValueError("ZETTA_PROVIDER_ESTIMATED_TOKENS must be non-negative")
    if isinstance(model_settings, Mapping):
        maximum = model_settings.get("max_tokens")
        if isinstance(maximum, int) and not isinstance(maximum, bool) and maximum > 0:
            estimate = max(estimate, maximum)
    return estimate


def _usage_tokens(value: Any) -> int:
    """Extract total tokens without depending on one pydantic-ai version."""

    usage = getattr(value, "usage", None)
    if callable(usage):
        try:
            usage = usage()
        except Exception:
            return 0
    if usage is None:
        return 0
    total = getattr(usage, "total_tokens", None)
    if isinstance(total, int) and not isinstance(total, bool) and total >= 0:
        return total
    values = []
    for name in ("request_tokens", "input_tokens", "response_tokens", "output_tokens"):
        item = getattr(usage, name, None)
        if isinstance(item, int) and not isinstance(item, bool) and item >= 0:
            values.append(item)
    return sum(values)


def _provider_failure(exc: BaseException) -> ProviderFailure:
    statuses = [
        getattr(current, "status_code", None) for current in _exception_chain(exc)
    ]
    if any(status == 429 for status in statuses):
        return ProviderFailure.RATE_LIMIT
    if any(isinstance(status, int) and status >= 500 for status in statuses):
        return ProviderFailure.OTHER_RETRYABLE
    return ProviderFailure.TRANSPORT


def is_retryable_provider_error(exc: BaseException) -> bool:
    """Return whether ``exc`` is attributable to a route and safe to fail over."""
    for current in _exception_chain(exc):
        if isinstance(current, ModelHTTPError):
            return current.status_code in _RETRYABLE_HTTP_STATUSES
        if isinstance(current, ModelAPIError):
            return True
        status = getattr(current, "status_code", None)
        if isinstance(status, int):
            return status in _RETRYABLE_HTTP_STATUSES
        if type(current).__name__ in _RETRYABLE_EXCEPTION_NAMES:
            return True
    return False


class _SharedCooldowns:
    """Tiny cross-process JSON health store containing no provider secrets."""

    def __init__(self, path: Path, *, clock: Callable[[], float]) -> None:
        self.path = path
        self.lock_path = path.with_suffix(path.suffix + ".lock")
        self._clock = clock

    def cooldown_until(self, route_ids: Sequence[str]) -> dict[str, float]:
        with _exclusive_file_lock(self.lock_path):
            state = self._read()
        routes = state.get("routes", {})
        return {
            route_id: float(routes.get(route_id, {}).get("cooldown_until", 0.0))
            for route_id in route_ids
        }

    def mark_failure(self, route_id: str, seconds: float) -> None:
        with _exclusive_file_lock(self.lock_path):
            state = self._read()
            routes = state.setdefault("routes", {})
            record = routes.setdefault(route_id, {})
            record["cooldown_until"] = max(
                float(record.get("cooldown_until", 0.0)), self._clock() + seconds
            )
            record["failure_count"] = int(record.get("failure_count", 0)) + 1
            record["last_failure_at"] = self._clock()
            self._write(state)

    def mark_success(self, route_id: str) -> None:
        with _exclusive_file_lock(self.lock_path):
            state = self._read()
            routes = state.setdefault("routes", {})
            record = routes.setdefault(route_id, {})
            record["cooldown_until"] = 0.0
            record["last_success_at"] = self._clock()
            self._write(state)

    def _read(self) -> dict[str, Any]:
        try:
            value = json.loads(self.path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return {"version": 1, "routes": {}}
        if not isinstance(value, dict) or value.get("version") != 1:
            return {"version": 1, "routes": {}}
        return value

    def _write(self, state: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(
            f".{self.path.name}.{os.getpid()}.{threading.get_ident()}.tmp"
        )
        temporary.write_text(
            json.dumps(state, sort_keys=True, separators=(",", ":")),
            encoding="utf-8",
        )
        try:
            os.chmod(temporary, 0o600)
        except OSError:
            pass
        os.replace(temporary, self.path)


@contextlib.contextmanager
def _exclusive_file_lock(path: Path) -> Any:
    """Acquire a thread- and process-wide advisory lock for ``path``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    key = str(path.resolve())
    with _PATH_LOCKS_GUARD:
        thread_lock = _PATH_LOCKS.setdefault(key, threading.Lock())
    with thread_lock:
        with path.open("a+b") as handle:
            try:
                os.chmod(path, 0o600)
            except OSError:
                pass
            if os.name == "nt":
                import msvcrt

                handle.seek(0, os.SEEK_END)
                if handle.tell() == 0:
                    handle.write(b"0")
                    handle.flush()
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
                try:
                    yield
                finally:
                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
                try:
                    yield
                finally:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _route_items(
    document: Any,
) -> tuple[list[tuple[str | None, Any]], Mapping[str, Any]]:
    if isinstance(document, list):
        return [(None, value) for value in document], {}
    if not isinstance(document, Mapping):
        raise ProviderPoolConfigError(f"{PROVIDERS_ENV} must be a JSON list or object")
    if "providers" in document or "routes" in document:
        values = document.get("providers", document.get("routes"))
        if not isinstance(values, list):
            raise ProviderPoolConfigError("providers/routes must be a JSON list")
        return [(None, value) for value in values], document
    if any(key in document for key in ("base_url", "url", "api_key", "key")):
        return [(None, document)], {}
    reserved = {"cooldown_seconds", "state_file"}
    return [
        (str(key), value) for key, value in document.items() if key not in reserved
    ], document


def sanitize_provider_config_for_broker_client(raw: str) -> str:
    """Retain route metadata while removing every upstream credential."""

    try:
        document = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ProviderPoolConfigError(
            f"{PROVIDERS_ENV} must contain valid JSON"
        ) from exc
    route_items, _ = _route_items(document)
    for index, (_, route) in enumerate(route_items):
        if not isinstance(route, dict):
            raise ProviderPoolConfigError(f"provider route {index} must be an object")
        route.pop("api_key_env", None)
        route.pop("api_key", None)
        route.pop("key", None)
        route["api_key"] = "broker-managed-placeholder"
    return json.dumps(document, separators=(",", ":"))


def _model_prefix(model: str) -> str:
    return model.partition(":")[0] if ":" in model else ""


def _model_for_provider(model: str, provider: str) -> str:
    suffix = model.partition(":")[2] if ":" in model else model
    return f"{provider}:{suffix}"


def _normalize_provider_endpoint(value: str, provider: str) -> tuple[str, str | None]:
    """Accept base URLs plus full OpenAI or Azure Responses endpoint URLs."""
    parsed = urlsplit(value)
    if provider.startswith("azure"):
        version = parse_qs(parsed.query).get("api-version", [None])[0]
        endpoint = f"{parsed.scheme}://{parsed.netloc}"
        return endpoint.rstrip("/"), version
    normalized = value.rstrip("/")
    for suffix in ("/chat/completions", "/responses"):
        if normalized.endswith(suffix):
            return normalized[: -len(suffix)], None
    return normalized, None


def _positive_float(value: Any, *, field_name: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ProviderPoolConfigError(f"{field_name} must be a number") from exc
    if parsed <= 0:
        raise ProviderPoolConfigError(f"{field_name} must be greater than zero")
    return parsed


def _nonnegative_float(value: Any, *, field_name: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ProviderPoolConfigError(f"{field_name} must be a number") from exc
    if parsed < 0:
        raise ProviderPoolConfigError(f"{field_name} must not be negative")
    return parsed


def _optional_positive_int(value: Any, *, field_name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ProviderPoolConfigError(f"{field_name} must be a positive integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ProviderPoolConfigError(
            f"{field_name} must be a positive integer"
        ) from exc
    if parsed < 1 or str(parsed) != str(value).strip():
        raise ProviderPoolConfigError(f"{field_name} must be a positive integer")
    return parsed


def _exception_chain(exc: BaseException) -> Iterator[BaseException]:
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        yield current
        current = current.__cause__ or current.__context__


def _retry_after_seconds(exc: BaseException) -> float | None:
    for current in _exception_chain(exc):
        headers = getattr(current, "headers", None)
        if not isinstance(headers, Mapping):
            continue
        value = headers.get("retry-after") or headers.get("Retry-After")
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            continue
        if parsed >= 0:
            return parsed
    return None


def _safe_error_label(exc: BaseException) -> str:
    for current in _exception_chain(exc):
        status = getattr(current, "status_code", None)
        if isinstance(status, int):
            return f"HTTP {status}"
    return type(exc).__name__
