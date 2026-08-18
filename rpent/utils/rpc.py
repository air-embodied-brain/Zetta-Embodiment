"""RPC client protocol and Facade base for subprocess RPC servers."""
from __future__ import annotations

import threading
import time
from typing import TYPE_CHECKING, Any, Literal, Protocol

from rpent.utils.logging import get_logger

if TYPE_CHECKING:
    from rpent.utils.daemon import ProcessDaemon

logger = get_logger("rpc")


class RpcError(RuntimeError):
    """Raised when a remote method call returns an error."""

    def __init__(self, method: str, message: str, *, traceback: str | None = None):
        super().__init__(f"{method}: {message}")
        self.method = method
        self.server_traceback = traceback


class RpcClient(Protocol):
    """Generic method-call RPC transport (agent → out-of-process server)."""

    def call(
        self,
        method: str,
        args: tuple = (),
        kwargs: dict | None = None,
        *,
        timeout_s: float | None = None,
    ) -> Any:
        """Invoke a remote method and return its result."""


def make_error_response(exc: Exception) -> dict:
    """Build the error envelope for a caught exception."""
    import traceback as _tb
    return {"ok": False, "error": str(exc), "traceback": _tb.format_exc()}


def check_response(response: Any, method: str) -> Any:
    """Validate RPC response envelope; raise ``RpcError`` on failure, return result."""
    if not isinstance(response, dict):
        raise RpcError(method, f"bad response type: {type(response).__name__}")
    if not response.get("ok"):
        raise RpcError(
            method,
            str(response.get("error", "<no error message>")),
            traceback=response.get("traceback"),
        )
    return response.get("result")


def wait_for_ready(
    client: RpcClient,
    *,
    timeout_s: float = 300.0,
    poll_interval_s: float = 0.5,
    daemon: "ProcessDaemon | None" = None,
) -> None:
    """Poll ``client.call("healthz")`` until it succeeds or ``timeout_s`` elapses.

    If ``daemon`` is given and its subprocess exits before becoming ready,
    fail fast with ``RuntimeError`` (carrying the exit code) instead of
    blocking until ``timeout_s``.
    """
    deadline = time.time() + timeout_s
    last_err: Exception | None = None
    while time.time() < deadline:
        if daemon is not None:
            rc = daemon.poll()
            if rc is not None:
                detail = last_err if last_err is not None else "no healthz attempt yet"
                raise RuntimeError(
                    f"{daemon.name} exited with code {rc} before becoming "
                    f"ready; check its log. last healthz error: {detail}"
                )
        try:
            client.call("healthz", timeout_s=1.0)
            return
        except Exception as exc:
            last_err = exc
            time.sleep(poll_interval_s)
    raise TimeoutError(
        f"server did not become ready within {timeout_s:.0f}s: {last_err}"
    )


class RpcFacade:
    """Base class for subprocess RPC servers.

    Subclasses implement :meth:`_dispatch`; the base owns the shutdown
    event, the ``shutdown`` / ``healthz`` RPC methods, transport binding,
    parent-watch, and clean teardown.

    Usage::

        class MyFacade(RpcFacade):
            def _dispatch(self, method, args, kwargs):
                if method == "hello":
                    return "world"
                raise ValueError(f"unknown RPC method: {method!r}")

        MyFacade().serve(transport="http", host="127.0.0.1", port=0)
    """

    def __init__(self) -> None:
        self._shutdown_event = threading.Event()

    def _dispatch(self, method: str, args: tuple, kwargs: dict) -> Any:
        """Business RPC dispatch. Override in subclasses.

        Do not handle ``shutdown`` or ``healthz`` here — the base takes care
        of them.
        """
        raise NotImplementedError

    def serve(
        self,
        *,
        transport: Literal["socket", "http"],
        host: str,
        port: int,
        parent_watch: bool = False,
    ) -> None:
        """Bind, announce, watch-parent, serve-forever, shut down cleanly.

        When *parent_watch* is True, a background thread reads stdin (a pipe
        from :class:`ProcessDaemon`) and triggers shutdown when the pipe
        closes — i.e., when the parent process dies.
        """
        from rpent.utils.daemon import watch_parent_death
        from rpent.utils.http_rpc import HttpRpcServer
        from rpent.utils.socket_rpc import SocketRpcServer

        _lock = threading.Lock()

        def dispatch(method: str, args: tuple, kwargs: dict) -> Any:
            if method == "healthz":
                return {"status": "ok"}
            if method == "shutdown":
                with _lock:
                    self._shutdown_event.set()
                return {"ok": True}
            with _lock:
                return self._dispatch(method, args, kwargs)

        server_cls = HttpRpcServer if transport == "http" else SocketRpcServer
        server = server_cls((host, port), dispatch)
        bound_host, bound_port = server.server_address
        client_host = "127.0.0.1" if bound_host == "0.0.0.0" else bound_host
        url = f"{transport}://{client_host}:{bound_port}"
        print(f"RPC server listening on {url}", flush=True)
        logger.info("RPC server listening on %s", url)

        if parent_watch:
            watch_parent_death(self._shutdown_event.set)
        try:
            threading.Thread(target=server.serve_forever, daemon=True).start()
            self._shutdown_event.wait()
        finally:
            server.shutdown()
            server.server_close()


def parse_endpoint(endpoint: str) -> tuple[str, str, int]:
    """Parse ``[protocol://]host:port`` into ``(protocol, host, port)``.

    Protocol defaults to ``http`` when the prefix is omitted.
    """
    if "://" in endpoint:
        protocol, _, rest = endpoint.partition("://")
    else:
        protocol, rest = "http", endpoint
    host, _, port = rest.partition(":")
    if not host or not port:
        raise ValueError(f"endpoint must be [protocol://]host:port, got {endpoint!r}")
    return protocol, host, int(port)


__all__ = [
    "RpcClient",
    "RpcError",
    "RpcFacade",
    "check_response",
    "make_error_response",
    "parse_endpoint",
    "wait_for_ready",
]
