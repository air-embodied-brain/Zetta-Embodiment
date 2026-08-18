"""HTTP-transport RPC for the env + model RPC boundary.

Uses HTTP POST with JSON payloads instead of pickle-framed TCP.
Numpy arrays cross the wire tagged as
``{"__ndarray__": <base64>, "dtype": ..., "shape": [...]}`` — the raw
bytes stay compact (vs ``tolist()``, which stringifies every element)
and the decode is explicit.
"""

from __future__ import annotations

import base64
import json
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from typing import Any, Callable

import numpy as np

from rpent.utils.rpc import RpcError, check_response, make_error_response

DEFAULT_TIMEOUT_S = 30.0


def _from_json(obj: Any) -> Any:
    """Rehydrate ``{"__ndarray__": <b64>, "dtype": ..., "shape": [...]}``
    back into ndarrays. Everything else is passed through unchanged.
    """
    if isinstance(obj, dict):
        if "__ndarray__" in obj and set(obj) <= {"__ndarray__", "dtype", "shape"}:
            raw = base64.b64decode(obj["__ndarray__"])
            arr = np.frombuffer(raw, dtype=obj.get("dtype"))
            # frombuffer returns a read-only view of the base64 bytes; copy
            # so callers can mutate the returned array like they would with
            # a pickle round-tripped one.
            return arr.reshape(obj.get("shape", (-1,))).copy()
        return {k: _from_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_from_json(v) for v in obj]
    return obj


class HttpRpcClient:
    """RPC client that talks to a driver server via HTTP POST.

    Parameters
    ----------
    base_url : str
        Server address, e.g. ``"http://127.0.0.1:8080"``.
    """

    def __init__(self, base_url: str) -> None:
        """Initialize with a base URL, e.g. ``"http://127.0.0.1:8080"``."""
        self._base_url = base_url.rstrip("/")

    def call(
        self,
        method: str,
        args: tuple = (),
        kwargs: dict | None = None,
        *,
        timeout_s: float | None = None,
    ) -> Any:
        """Invoke a remote method via HTTP POST and return the result."""
        payload = {
            "method": method,
            "args": list(args),
            "kwargs": kwargs or {},
        }
        body = json.dumps(payload, cls=_NumpyEncoder).encode("utf-8")
        url = f"{self._base_url}/call"
        request_timeout = timeout_s if timeout_s is not None else DEFAULT_TIMEOUT_S

        req = urllib.request.Request(
            url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=request_timeout) as resp:
                raw = resp.read()
        except urllib.error.HTTPError as exc:
            # HTTPError is an OSError subclass; catch first so we can parse
            # the ok=False body the server sent alongside the status code.
            raw = exc.read()
        except OSError as exc:
            raise RpcError(method, f"HTTP request failed: {exc}") from exc

        try:
            response = _from_json(json.loads(raw))
        except json.JSONDecodeError as exc:
            raise RpcError(method, f"invalid JSON response: {exc}") from exc

        return check_response(response, method)


# ---------------------------------------------------------------------------
#  Server
# ---------------------------------------------------------------------------


class _NumpyEncoder(json.JSONEncoder):
    """JSON encoder that tags numpy arrays and normalizes numpy scalars."""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            return {
                "__ndarray__": base64.b64encode(obj.tobytes()).decode("ascii"),
                "dtype": str(obj.dtype),
                "shape": list(obj.shape),
            }
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        return super().default(obj)


class _HttpRpcHandler(BaseHTTPRequestHandler):
    """Handles POST /call with JSON-RPC body, dispatches to server.dispatch()."""

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        return

    def do_POST(self) -> None:
        if self.path != "/call":
            self.send_response(404)
            self.end_headers()
            return

        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length) if content_length > 0 else b""
        try:
            request = json.loads(body)
            method = request["method"]
            args = tuple(_from_json(v) for v in request.get("args", []))
            kwargs = {k: _from_json(v) for k, v in request.get("kwargs", {}).items()}
            result = self.server.dispatch(method, args, kwargs)  # type: ignore[attr-defined]
            response: dict = {"ok": True, "result": result}
        except Exception as exc:
            response = make_error_response(exc)

        # Always 200; failures are described inside the body via ok=False.
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(
            json.dumps(response, cls=_NumpyEncoder).encode("utf-8")
        )


class HttpRpcServer(ThreadingHTTPServer):
    """HTTP server that dispatches JSON-RPC calls.

    Same interface as ``SocketRpcServer`` — drop-in replacement at dispatch
    sites that use ``server_address``, ``serve_forever``, ``shutdown``, and
    ``server_close``.
    """

    allow_reuse_address = True
    daemon_threads = True

    def __init__(
        self,
        server_address: tuple[str, int],
        dispatch: Callable[[str, tuple, dict], Any],
    ) -> None:
        super().__init__(server_address, _HttpRpcHandler)
        self.dispatch = dispatch
