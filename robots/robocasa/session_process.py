# Copyright (c) 2026 Zetta Contributors
"""Process-level isolation for ``RoboCasaSession`` (Path B, RUNTIME_V3 §0.2 fix).

``rollout_runtime/backends/robocasa_current.py`` builds one ``RoboCasaSession`` per
pool slot; when a single Ray env-worker rank holds more than one slot
(``env_worker.max_sessions_per_rank`` / ``default_pool_size`` > 1), those sessions
share one GPU and are dispatched onto the worker's shared Python thread pool
(``env_worker.py::_call_core`` -> ``asyncio.to_thread``). ``EnvPool._call_pool_core``
deliberately does **not** serialize ``per_slot`` pools ("slot 之间是独立的 env"),
which is true at the Python level but false at the native robosuite/MuJoCo/EGL
level: two sessions' first ``reset()`` (which lazily calls ``gym.make()`` and does
an internal validation render through ``persistent_render`` /
``_set_mujoco_context_and_buffers``) can run genuinely concurrently on two OS
threads inside the same process, racing on unmanaged native GL/EGL context state.

Real py-spy evidence (2026-08-17, ``PickPlaceCounterToStove``, 5 concurrent pairs):
0/10 deterministic deadlocks this run, but 4/10 sessions crashed with
``EGLError(err=EGL_BAD_ACCESS, baseOperation=eglMakeCurrent)`` during
``chunk_step`` (not just during ``reset``), and a live dump caught one thread
inside ``_set_mujoco_context_and_buffers`` (via ``render``) while another was
simultaneously inside ``from_xml_string``/``_initialize_sim`` -- both under
``reset()``, on two different ``RoboCasaSession`` instances. A prior investigation
on ``OpenDrawer`` observed a full deadlock at the same contention point
(``the process-isolation design notes`` §0.2). The two outcomes are the
same underlying race manifesting nondeterministically, not two different bugs.

Neither of the two previously tried mitigations closes this: a Python
``threading.RLock`` on ``RoboCasaSession`` itself only protects *that instance*
(pool_size>1 means N separate instances with N separate locks), and a POSIX
``fcntl.flock`` around ``gym.make()`` (``_cold_reset_guard``) does not help either,
because the actual contention is inside native code invoked by ``reset()``/
``render()``, and the flock's Python-level critical section does not extend into
whatever internal threads or GL/EGL state MuJoCo/robosuite touch there.

**Fix**: give each slot's ``RoboCasaSession`` its own OS process (not just an OS
thread) when ``RobocasaCurrentConfig.process_isolation`` is enabled. Two separate
processes cannot race on the same in-process native GL/EGL state; the operating
system's own process isolation is the boundary, not a Python-level lock. This
module provides:

- ``spawn_robocasa_subprocess``: starts one child process running
  ``_subprocess_worker_main`` and returns a connected ``RemoteRoboCasaSession``.
- ``RemoteRoboCasaSession``: a parent-side proxy exposing the exact same public
  method signatures as ``robots.robocasa.session_core.RoboCasaSession``
  (``reset``/``execute_chunk``/``snapshot``/``finalize_episode_artifacts``/
  ``close_environment``), plus ``observe_encoded()`` (new: moves the JPEG
  quality-80 quantization that ``RobocasaCurrentCore._encode_camera`` normally
  does in the *parent* process into the *child* process instead, using the same
  ``session_core.jpeg_lossy_rgb_frame`` helper the non-isolated path uses, so
  pixels stay identical across isolation modes). This module deliberately does
  **not** also do the PNG *transport* encoding
  (``rollout_runtime.core.payload.encode_image``) here: it lives under
  ``robots/robocasa/`` and the repository's layering rules (D9, architecture
  rule 5, enforced by ``tests/runtime/test_layering.py``) forbid any
  ``robots/robocasa/**`` file other than ``run_rollout.py`` from importing
  ``rollout_runtime``. The still-uint8-array (but already JPEG-quantized) image
  crosses the pipe instead; ``RobocasaCurrentCore._encode_camera`` -- which
  already owns PNG transport encoding for the non-isolated path -- applies the
  same PNG step to it in the parent, keeping that concern on the runtime side of
  the boundary for both isolation modes.
  Every ``RoboCasaSession`` public method already returns plain dict/JSON-safe
  payloads (a deliberate historical design decision recorded in
  ``robocasa_current.py``'s module docstring point 2), so this proxy needs no new
  serialization logic beyond what ``multiprocessing.connection.Connection.send``/
  ``recv`` (pickle) already handles for dicts, lists, strings, floats, bools, and
  numpy arrays.

IPC shape follows the existing ``robots/libero/rlinf_worker_compat.py`` precedent:
a ``[command, payload]`` two-element list sent over a
``multiprocessing.connection.Connection``, with a blocking ``.send()``/``.recv()``
call per RPC (matching ``EnvExecutionCore``'s "must be blocking synchronous"
contract, since the caller already wraps every core method in
``asyncio.to_thread``; see ``core/env_execution.py``'s ``EnvExecutionCore``
docstring).

The child process is started with ``multiprocessing.get_context("spawn")``: CUDA
and EGL contexts are not fork-safe, and this module's whole reason for existing
is to stop sharing exactly that kind of native GPU/GL state across OS threads --
sharing it across a forked child's copied file descriptors and driver handles
would reintroduce a related hazard from a different angle.

**Deliberately not reused**: the removed ``GpuOperationGate`` (flock-based,
cross-*process* GPU admission for deployments where several independent OS
processes shared one GPU, see ``RobocasaCurrentConfig``'s docstring on Stage 6).
That mechanism solved "N independent processes, no scheduler-level placement
guarantee, must not double-book one GPU". This module solves a different problem:
"N sessions inside *one* Ray-rank process, which already exclusively owns its GPU
via Ray placement, must not share unmanaged native GL/EGL state." Reintroducing a
flock here would only re-serialize the symptom Path A already covers (see
``robocasa_current.py``'s ``RobocasaCurrentConfig.cold_reset_lock``); it would not
give the isolation this module provides, and the historical record already shows
the flock does not even reach the actual contention point.
"""

from __future__ import annotations

import multiprocessing as _mp
import traceback
from multiprocessing import connection
from typing import Any

__all__ = [
    "RemoteRoboCasaSession",
    "RemoteSessionCrashed",
    "spawn_robocasa_subprocess",
]

_SPAWN_CONTEXT = _mp.get_context("spawn")
"""CUDA/EGL are not fork-safe; every child in this module starts clean via spawn."""

_SHUTDOWN_COMMAND = "__shutdown__"
"""Sentinel command telling the child loop to exit after closing its environment."""


class RemoteSessionCrashed(RuntimeError):
    """Raised when the subprocess died or the pipe broke while awaiting a reply.

    ``env_worker.py::_call_core`` normalizes any ``BaseException`` raised by the
    execution core into ``ErrorCode.ENV_FAILURE`` (see its docstring), so raising a
    plain ``RuntimeError`` subclass here is enough for the existing normalization
    path to turn a subprocess crash into the same error shape a same-process
    ``RoboCasaSession`` failure would have produced -- callers do not need to know
    which isolation mode built the slot.
    """


def _subprocess_worker_main(
    conn: connection.Connection,
    *,
    camera_size: int,
    max_steps: int,
    cold_reset_lock: str | None,
    require_isolated_renderer: bool,
    session_factory: Any = None,
) -> None:
    """Child process entry point: build one ``RoboCasaSession`` and serve RPCs.

    Runs until it receives ``["__shutdown__", {}]`` or the pipe closes. Every
    request/reply is a two-element list, mirroring
    ``robots/libero/rlinf_worker_compat.py``'s ``env_call`` protocol: the child
    replies ``["ok", result]`` on success or ``["error", {"type": ..., "message":
    ..., "traceback": ...}]`` on failure, and keeps serving further requests after
    an error (a failed ``execute_chunk`` on episode N must not prevent the next
    ``reset`` from reaching the same warm session).

    Args:
        conn: the child's end of the pipe; the parent's end must already be
            closed by the caller of ``get_context("spawn").Process`` (spawn
            duplicates the whole pair into the child, so the child closing its
            copy of the parent's end is not needed the way fork's inherited-fd
            model requires, but the pipe itself becomes the only channel here).
        camera_size: forwarded to ``RoboCasaSession.__init__``.
        max_steps: forwarded to ``RoboCasaSession.__init__``.
        cold_reset_lock: forwarded to ``RoboCasaSession.__init__``.
        require_isolated_renderer: forwarded to ``RoboCasaSession.__init__``.
        session_factory: constructs the session; must accept the same four
            keyword arguments as ``RoboCasaSession.__init__`` and return an
            object exposing its public method surface. Defaults to
            ``RoboCasaSession`` itself; overridable so tests can spawn a real
            child process against a fake gym environment without a robosuite
            install (``spawn`` starts a fresh interpreter, so a same-process
            ``pytest.MonkeyPatch`` on ``RoboCasaSession._ensure_environment``
            never reaches the child -- the factory must be an importable
            module-level callable, not a closure, since ``multiprocessing``
            pickles it by reference).
    """
    if session_factory is None:
        from robots.robocasa.session_core import RoboCasaSession as session_factory

    try:
        session = session_factory(
            camera_size=camera_size,
            max_steps=max_steps,
            cold_reset_lock=cold_reset_lock,
            require_isolated_renderer=require_isolated_renderer,
        )
    except BaseException as exc:  # noqa: BLE001 - report construction failure, then exit
        conn.send(["error", _error_payload(exc)])
        conn.close()
        return
    conn.send(["ready", {}])

    while True:
        try:
            command, payload = conn.recv()
        except EOFError:
            break
        try:
            if command == _SHUTDOWN_COMMAND:
                session.close_environment()
                conn.send(["ok", {}])
                break
            result = _dispatch(session, command, payload)
            conn.send(["ok", result])
        except BaseException as exc:  # noqa: BLE001 - must not kill the RPC loop
            conn.send(["error", _error_payload(exc)])
    conn.close()


def _dispatch(session: Any, command: str, payload: dict[str, Any]) -> Any:
    """Route one RPC command to the matching ``RoboCasaSession`` method.

    Args:
        session: the child's live ``RoboCasaSession``.
        command: RPC method name.
        payload: method kwargs.

    Returns:
        The method's plain dict/JSON-safe return value.

    Raises:
        ValueError: unknown command (a parent/child protocol version mismatch;
            surfaced as ``RemoteSessionCrashed`` on the parent side like any other
            child-side exception).
    """
    if command == "reset":
        return session.reset(payload["payload"])
    if command == "execute_chunk":
        return session.execute_chunk(payload["payload"])
    if command == "snapshot":
        return session.snapshot(include_images=bool(payload["include_images"]))
    if command == "finalize_episode_artifacts":
        return session.finalize_episode_artifacts()
    if command == "close_environment":
        session.close_environment()
        return {}
    if command == "observe_encoded":
        return _observe_encoded(session, camera_keys=payload["camera_keys"])
    raise ValueError(f"unknown RemoteRoboCasaSession command {command!r}")


def _observe_encoded(session: Any, *, camera_keys: list[str]) -> dict[str, Any]:
    """Build the JPEG-quantized-but-not-yet-transport-encoded observation payload.

    Runs the same JPEG quality-80 quantization that
    ``RobocasaCurrentCore._encode_camera`` applies for the non-isolated path
    (``session_core.jpeg_lossy_rgb_frame``), so the pixels a policy conditions on
    are identical regardless of isolation mode (runtime v3 design
    Stage 9 step 26: same-seed replays must stay byte-comparable). The PNG
    *transport* encoding (``rollout_runtime.core.payload.encode_image``) stays
    in the parent process (``RobocasaCurrentCore._encode_camera``) rather than
    running here: this module lives under ``robots/robocasa/`` and must not
    import ``rollout_runtime`` (D9/architecture rule 5 -- ``robots/robocasa``'s
    only sanctioned ``rollout_runtime`` entry point is ``run_rollout.py``, not
    this module, see ``tests/runtime/test_layering.py``). Returning the
    JPEG-quantized-but-still-raw uint8 HxWx3 array (rather than the full
    unquantized simulator frame) is the isolation boundary this module owns;
    packaging it into the wire-transport ``InlineBytes`` PNG payload is the
    runtime-side concern ``robocasa_current.py`` already owns for the
    non-isolated path, and keeps owning here.

    Args:
        session: the child's live ``RoboCasaSession``.
        camera_keys: the ``session.observation`` keys to encode, in order.

    Returns:
        ``{"images": {key: np.ndarray | None}, "step_index": int,
        "attestation": dict, "task_descriptions": list | None}`` -- ``images``
        values are JPEG-quantized uint8 HxWx3 arrays, not yet PNG-encoded.

    Raises:
        RuntimeError: ``session.observation`` is ``None`` (slot not reset yet);
            surfaced to the parent as a normal RPC error, same as any other
            command called before ``reset``.
    """
    import numpy as np

    from robots.robocasa.session_core import jpeg_lossy_rgb_frame

    if session.observation is None:
        raise RuntimeError("environment has not been reset")
    raw = session.observation
    images: dict[str, Any] = {}
    for key in camera_keys:
        value = raw.get(key) if isinstance(raw, dict) else None
        array = np.asarray(value) if value is not None else None
        if array is None or array.ndim != 3 or array.shape[-1] != 3:
            images[key] = None
            continue
        images[key] = jpeg_lossy_rgb_frame(array.astype(np.uint8, copy=False))
    description = raw.get("task_descriptions") if isinstance(raw, dict) else None
    return {
        "images": images,
        "step_index": session.step_index,
        "attestation": session.snapshot(include_images=False),
        "task_descriptions": list(description) if description else None,
    }


def _error_payload(exc: BaseException) -> dict[str, str]:
    """Serialize an exception into a plain dict safe to pickle across the pipe.

    The original exception object is not sent: arbitrary exception types from
    robosuite/MuJoCo/gymnasium may hold unpicklable native handles (the same
    class of object this module exists to keep out of the parent process), so
    only its type name, message, and formatted traceback cross the boundary.

    Args:
        exc: the exception raised in the child.

    Returns:
        A dict with ``type``, ``message``, and ``traceback`` string keys.
    """
    return {
        "type": type(exc).__name__,
        "message": str(exc),
        "traceback": "".join(
            traceback.format_exception(type(exc), exc, exc.__traceback__)
        ),
    }


class RemoteRoboCasaSession:
    """Parent-side proxy for a ``RoboCasaSession`` running in a child process.

    Exposes the exact same public method signatures as
    ``robots.robocasa.session_core.RoboCasaSession`` (``reset``/
    ``execute_chunk``/``snapshot``/``finalize_episode_artifacts``/
    ``close_environment``) so ``RobocasaCurrentCore`` can hold either a real
    ``RoboCasaSession`` (``process_isolation=False``, default, unchanged
    behavior) or this proxy (``process_isolation=True``) behind the same
    ``_RobocasaSlot.session`` attribute without branching on which one it has,
    except for observation assembly (see ``observe_encoded``, which has no
    ``RoboCasaSession`` equivalent because the non-isolated path reads
    ``session.observation`` directly instead).

    Every call blocks the calling OS thread until the child replies or the pipe
    breaks, matching ``EnvExecutionCore``'s "must be blocking synchronous"
    contract -- callers already run these through ``asyncio.to_thread``
    (``env_worker.py::_call_core``), so blocking here is correct, not a bug.
    """

    def __init__(
        self,
        process: _mp.process.BaseProcess,
        conn: connection.Connection,
        *,
        rpc_timeout_s: float = 120.0,
    ) -> None:
        """Wrap an already-started child process and its parent-side pipe end.

        Args:
            process: the child process (for liveness checks and ``join``/
                ``terminate`` during shutdown).
            conn: the parent's end of the pipe.
            rpc_timeout_s: per-RPC wait budget; a slot construction/reset can be
                slow (cold ``gym.make()``), so this is generous rather than
                tuned to steady-state ``chunk_step`` latency. Exceeding it raises
                ``RemoteSessionCrashed`` rather than hanging the caller's
                ``asyncio.to_thread`` slot forever.
        """
        self._process = process
        self._conn = conn
        self._rpc_timeout_s = rpc_timeout_s
        self._closed = False

    def _call(self, command: str, payload: dict[str, Any]) -> Any:
        """Send one RPC and block for its reply.

        Args:
            command: RPC method name.
            payload: method kwargs.

        Returns:
            The child's plain dict/JSON-safe return value.

        Raises:
            RemoteSessionCrashed: the child died, the pipe broke, the reply
                timed out, or the child reported an exception.
        """
        if self._closed:
            raise RemoteSessionCrashed(
                "cannot call a RemoteRoboCasaSession after close_environment()"
            )
        if not self._process.is_alive():
            raise RemoteSessionCrashed(
                f"robocasa subprocess (pid={self._process.pid}) is not alive "
                f"before sending {command!r}; exitcode={self._process.exitcode}"
            )
        try:
            self._conn.send([command, payload])
            if not self._conn.poll(self._rpc_timeout_s):
                self._mark_transport_unusable()
                raise RemoteSessionCrashed(
                    f"robocasa subprocess (pid={self._process.pid}) did not "
                    f"reply to {command!r} within {self._rpc_timeout_s}s"
                )
            status, result = self._conn.recv()
        except (OSError, EOFError, ValueError) as exc:
            self._mark_transport_unusable()
            raise RemoteSessionCrashed(
                f"robocasa subprocess (pid={self._process.pid}) pipe failed "
                f"during {command!r}: {exc}"
            ) from exc
        if status == "error":
            raise RemoteSessionCrashed(
                f"robocasa subprocess (pid={self._process.pid}) raised "
                f"{result['type']} during {command!r}: {result['message']}\n"
                f"{result['traceback']}"
            )
        return result

    def reset(self, payload: dict[str, Any]) -> dict[str, Any]:
        """See ``RoboCasaSession.reset``."""
        return self._call("reset", {"payload": payload})

    def execute_chunk(self, payload: dict[str, Any]) -> dict[str, Any]:
        """See ``RoboCasaSession.execute_chunk``."""
        return self._call("execute_chunk", {"payload": payload})

    def snapshot(self, *, include_images: bool) -> dict[str, Any]:
        """See ``RoboCasaSession.snapshot``."""
        return self._call("snapshot", {"include_images": include_images})

    def finalize_episode_artifacts(self) -> dict[str, Any]:
        """See ``RoboCasaSession.finalize_episode_artifacts``."""
        return self._call("finalize_episode_artifacts", {})

    def observe_encoded(self, *, camera_keys: list[str]) -> dict[str, Any]:
        """Fetch the current observation with cameras JPEG-quantized in the child.

        Args:
            camera_keys: ``session.observation`` keys to encode, in order.

        Returns:
            ``{"images": {key: np.ndarray | None}, "step_index": int,
            "attestation": dict, "task_descriptions": list | None}`` -- ``images``
            values are JPEG-quantized uint8 HxWx3 arrays (see
            ``_observe_encoded``), not yet PNG transport-encoded; the caller
            (``RobocasaCurrentCore._encode_camera``) applies that step, exactly
            like it does for the non-isolated path.
        """
        return self._call("observe_encoded", {"camera_keys": list(camera_keys)})

    def close_environment(self) -> None:
        """Ask the child to close its environment, then stop and join it.

        Unlike ``RoboCasaSession.close_environment`` (which only tears down the
        gym env and keeps the process alive for the next ``reset``), this always
        terminates the child process: ``RobocasaCurrentCore.close()`` calls it
        once per slot when the whole pool is torn down, and a fresh ``reset()``
        after that always goes through ``spawn_robocasa_subprocess`` again to
        build a brand new child (there is currently no "keep the child, just
        close its gym env" call path from ``RobocasaCurrentCore``, matching how
        ``RoboCasaSession`` itself is only ever discarded, not reused, once a
        pool closes).

        Safe to call more than once; the second call is a no-op.
        """
        if self._closed:
            return
        self._closed = True
        try:
            if self._process.is_alive():
                self._conn.send([_SHUTDOWN_COMMAND, {}])
                if self._conn.poll(5.0):
                    with contextlib_suppress():
                        self._conn.recv()
        except OSError:
            pass
        finally:
            self._terminate_and_join()

    def _terminate_and_join(self) -> None:
        """Best-effort shutdown: join briefly, then terminate/kill if needed."""
        try:
            self._process.join(timeout=5.0)
        except (OSError, ValueError):
            pass
        if self._process.is_alive():
            self._process.terminate()
            try:
                self._process.join(timeout=5.0)
            except (OSError, ValueError):
                pass
        if self._process.is_alive():
            # SIGTERM did not land in time (e.g. stuck in a native call this
            # whole module exists to isolate); SIGKILL cannot be ignored.
            self._process.kill()
            try:
                self._process.join(timeout=5.0)
            except (OSError, ValueError):
                pass
        try:
            self._conn.close()
        except OSError:
            pass

    def _mark_transport_unusable(self) -> None:
        """Close and reap a child whose RPC transport can no longer be used."""
        self._closed = True
        self._terminate_and_join()

    @property
    def is_alive(self) -> bool:
        """Whether the child process is still running.

        Returns:
            ``True`` if the child has not exited or been closed.
        """
        return (not self._closed) and self._process.is_alive()


class contextlib_suppress:
    """Minimal ``contextlib.suppress(Exception)`` inline to avoid an extra import
    line for a single best-effort drain in ``close_environment``."""

    def __enter__(self) -> "contextlib_suppress":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        return exc_type is not None and issubclass(exc_type, Exception)


def spawn_robocasa_subprocess(
    *,
    camera_size: int,
    max_steps: int,
    cold_reset_lock: str | None,
    require_isolated_renderer: bool,
    startup_timeout_s: float = 300.0,
    session_factory: Any = None,
) -> RemoteRoboCasaSession:
    """Start one child process running a fresh ``RoboCasaSession`` and connect it.

    The child does not import ``robots.robocasa.session_core`` (and therefore
    does not import ``gymnasium``/``robocasa``/``robosuite``) until it has
    already been spawned, matching ``RoboCasaSession.__init__``'s own behavior
    of doing the heavy simulator import lazily; this function's own construction
    cost is dominated by process startup, not simulator loading (that still
    happens lazily inside the child's own first ``reset()``, exactly like the
    non-isolated path).

    Args:
        camera_size: forwarded to ``RoboCasaSession.__init__``.
        max_steps: forwarded to ``RoboCasaSession.__init__``.
        cold_reset_lock: forwarded to ``RoboCasaSession.__init__``.
        require_isolated_renderer: forwarded to ``RoboCasaSession.__init__``;
            checked inside the child, so a renderer-patch mismatch surfaces as a
            normal RPC error on the first call rather than at spawn time (the
            check itself is cheap and side-effect-free, see
            ``session_core.isolated_renderer_status``, but running it inside the
            child keeps this function from needing to import
            ``robots.robocasa.session_core`` in the parent).
        startup_timeout_s: how long to wait for the child to either construct
            its ``RoboCasaSession`` successfully or report a construction
            failure, before giving up and raising ``RemoteSessionCrashed``.
            Generous because ``RoboCasaSession.__init__`` itself is cheap (it
            does not call ``gym.make()``), so this mostly bounds Python
            interpreter + import startup time in the child, not simulator cold
            start.
        session_factory: forwarded to ``_subprocess_worker_main``; test-only
            override, see its docstring. Must be an importable module-level
            callable (a class works), not a closure or lambda, because
            ``multiprocessing``'s ``spawn`` start method pickles the process
            target and its keyword arguments by reference.

    Returns:
        A connected, ready-to-use ``RemoteRoboCasaSession``.

    Raises:
        RemoteSessionCrashed: the child failed to start, failed to construct
            its ``RoboCasaSession``, or did not respond within
            ``startup_timeout_s``.
    """
    parent_conn, child_conn = _SPAWN_CONTEXT.Pipe(duplex=True)
    process = _SPAWN_CONTEXT.Process(
        target=_subprocess_worker_main,
        kwargs={
            "conn": child_conn,
            "camera_size": camera_size,
            "max_steps": max_steps,
            "cold_reset_lock": cold_reset_lock,
            "require_isolated_renderer": require_isolated_renderer,
            "session_factory": session_factory,
        },
        daemon=True,
    )
    process.start()
    child_conn.close()
    remote = RemoteRoboCasaSession(process, parent_conn)
    # ``RoboCasaSession.__init__`` runs before the child enters its RPC loop; a
    # construction failure (e.g. ``require_isolated_renderer=True`` and no
    # patched robosuite installed) sends an unsolicited ``["error", ...]`` with
    # no request having been sent yet. A cheap round-trip flushes that error (if
    # any) before the caller starts issuing real RPCs, so ``build()`` fails
    # immediately instead of on the first ``reset()``.
    try:
        if not parent_conn.poll(startup_timeout_s):
            raise RemoteSessionCrashed(
                f"robocasa subprocess (pid={process.pid}) did not start within "
                f"{startup_timeout_s}s"
            )
        status, result = parent_conn.recv()
    except (OSError, EOFError, ValueError) as exc:
        remote._terminate_and_join()
        raise RemoteSessionCrashed(
            f"robocasa subprocess (pid={process.pid}) failed to start: {exc}"
        ) from exc
    if status == "error":
        remote._terminate_and_join()
        raise RemoteSessionCrashed(
            f"robocasa subprocess (pid={process.pid}) failed to construct its "
            f"RoboCasaSession: {result['type']}: {result['message']}\n"
            f"{result['traceback']}"
        )
    if status != "ready":
        remote._terminate_and_join()
        raise RemoteSessionCrashed(
            f"robocasa subprocess (pid={process.pid}) sent an unexpected "
            f"startup message {status!r}"
        )
    return remote
