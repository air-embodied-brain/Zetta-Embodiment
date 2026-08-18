# Copyright (c) 2026 RPent Contributors
"""Launch many isolated persistent RoboCasa environment servers on one host."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from rpent.evolution.jsonio import atomic_write_json  # noqa: E402


def _health(endpoint: str, timeout_s: float = 2.0) -> dict[str, Any]:
    with urllib.request.urlopen(endpoint + "/health", timeout=timeout_s) as response:
        value = json.loads(response.read().decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError("health response is not an object")
    return value


def _public_health(value: dict[str, Any]) -> dict[str, Any]:
    """Remove write capabilities before persisting a shared ready manifest."""

    protocol = value.get("write_protocol")
    public_protocol = (
        {
            key: protocol.get(key)
            for key in ("generation", "phase", "lost")
            if key in protocol
        }
        if isinstance(protocol, dict)
        else {}
    )
    return {
        "status": value.get("status"),
        "persistent": value.get("persistent"),
        "renderer": value.get("renderer"),
        "gpu_visible": value.get("gpu_visible"),
        "egl_device": value.get("egl_device"),
        "write_protocol": public_protocol,
    }


def _gpu_list(value: str) -> tuple[str, ...]:
    result = tuple(item.strip() for item in value.split(",") if item.strip())
    if not result:
        raise argparse.ArgumentTypeError("at least one GPU is required")
    return result


def _lease_is_live(path: Path, *, now: datetime | None = None) -> bool:
    """Return true only for a complete, unexpired broker lease."""

    if not path.exists():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        expires = datetime.fromisoformat(str(payload["expires_at"]))
    except (KeyError, OSError, ValueError, json.JSONDecodeError):
        return False
    if expires.tzinfo is None:
        return False
    return expires.astimezone(timezone.utc) > (now or datetime.now(timezone.utc))


def _health_restart_due(
    *, first_failure_at: float, now: float, grace_s: float, lease_live: bool
) -> bool:
    """Never kill a busy reset merely because health shares its request lane."""

    return not lease_live and now - first_failure_at >= grace_s


def _bound_session_is_orphaned(
    *, phase: Any, lease_observed: bool, lease_live: bool
) -> bool:
    """Distinguish abandoned worker leases from supported direct clients."""

    return (
        phase not in (None, "FREE", "LOST")
        and lease_observed
        and not lease_live
    )


def env_server_command(
    *,
    python: str,
    port: int,
    camera_size: int,
    max_steps: int,
    root: Path,
    gpu: str,
    gpu_operation_slots: int,
    maximum_inflight_requests: int,
) -> list[str]:
    """Build the secret-free per-slot server command used by the supervisor."""

    return [
        python,
        "-m",
        "robots.robocasa.env_server",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--camera-size",
        str(camera_size),
        "--max-steps",
        str(max_steps),
        "--cold-reset-lock",
        str(root / f"cold-reset-gpu-{gpu}.lock"),
        "--operation-gate-root",
        str(root / "operation-gates"),
        "--operation-gate-gpu",
        gpu,
        "--operation-gate-slots",
        str(gpu_operation_slots),
        "--maximum-inflight-requests",
        str(maximum_inflight_requests),
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slots", type=int, required=True)
    parser.add_argument("--gpus", type=_gpu_list, default=("0",))
    parser.add_argument("--base-port", type=int, default=18800)
    parser.add_argument("--camera-size", type=int, default=256)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--ready-manifest", type=Path, required=True)
    parser.add_argument("--startup-timeout-s", type=float, default=300.0)
    parser.add_argument("--gpu-operation-slots", type=int, default=1)
    parser.add_argument("--maximum-inflight-requests", type=int, default=2)
    parser.add_argument("--restart-limit", type=int, default=5)
    parser.add_argument("--restart-window-s", type=float, default=300.0)
    parser.add_argument(
        "--health-restart-grace-s",
        type=float,
        default=300.0,
        help="continuous health outage required before restarting an unleased slot",
    )
    parser.add_argument(
        "--slot-broker-root",
        type=Path,
        default=None,
        help="host-local broker root used to detect abandoned bound slots",
    )
    args = parser.parse_args()
    if args.slots < 1:
        raise SystemExit("--slots must be positive")
    if args.gpu_operation_slots < 1 or args.maximum_inflight_requests < 1:
        raise SystemExit("operation and HTTP limits must be positive")
    if (
        args.restart_limit < 1
        or args.restart_window_s <= 0
        or args.health_restart_grace_s <= 0
    ):
        raise SystemExit("restart policy must be positive")

    root = args.runtime_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    processes: list[
        tuple[subprocess.Popen[bytes] | None, Any | None, dict[str, Any]]
    ] = []
    restart_history: dict[int, list[float]] = {}
    health_failure_started: dict[int, float] = {}
    stopping = False

    def stop(_signal: int | None = None, _frame: Any = None) -> None:
        nonlocal stopping
        stopping = True
        for process, _log, _record in processes:
            if process is not None and process.poll() is None:
                process.terminate()

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)

    def launch_slot(
        slot: int, *, generation: int
    ) -> tuple[subprocess.Popen[bytes], Any, dict[str, Any]]:
        gpu = args.gpus[slot % len(args.gpus)]
        slot_root = root / f"slot-{slot:03d}"
        slot_root.mkdir(parents=True, exist_ok=True)
        environment = os.environ.copy()
        environment.update(
            {
                "CUDA_VISIBLE_DEVICES": gpu,
                "MUJOCO_EGL_DEVICE_ID": gpu,
                "MUJOCO_GL": "egl",
                "PYOPENGL_PLATFORM": "egl",
                "MESA_SHADER_CACHE_DIR": str(slot_root / "mesa"),
                "ROBOCASA_MJCF_CACHE_DIR": str(slot_root / "mjcf"),
                "PYTHONUNBUFFERED": "1",
            }
        )
        Path(environment["MESA_SHADER_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)
        Path(environment["ROBOCASA_MJCF_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)
        port = args.base_port + slot
        command = env_server_command(
            python=sys.executable,
            port=port,
            camera_size=args.camera_size,
            max_steps=args.max_steps,
            root=root,
            gpu=gpu,
            gpu_operation_slots=args.gpu_operation_slots,
            maximum_inflight_requests=args.maximum_inflight_requests,
        )
        log_path = slot_root / "server.log"
        log = log_path.open("ab", buffering=0)
        if os.name == "posix":
            os.chmod(log_path, 0o600)
        process = subprocess.Popen(
            command,
            cwd=REPOSITORY_ROOT,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        record = {
            "slot": slot,
            "gpu": gpu,
            "pid": process.pid,
            "endpoint": f"http://127.0.0.1:{port}",
            "slot_root": str(slot_root),
            "generation": generation,
            "status": "starting" if generation == 0 else "restarting",
            "restart_count": len(restart_history.get(slot, ())),
        }
        return process, log, record

    def public_manifest() -> dict[str, Any]:
        records = [dict(record) for _process, _log, record in processes]
        return {
            "schema_version": 2,
            "slots": records,
            "slot_count": len(records),
            "healthy_slot_count": sum(
                record.get("status") == "ready" for record in records
            ),
            "gpus": list(args.gpus),
            "isolated_renderer_required": True,
            "gpu_operation_slots": args.gpu_operation_slots,
            "maximum_inflight_requests": args.maximum_inflight_requests,
        }

    def persist_manifest() -> None:
        atomic_write_json(args.ready_manifest, public_manifest(), overwrite=True)

    try:
        for slot in range(args.slots):
            process, log, record = launch_slot(slot, generation=0)
            processes.append((process, log, record))

        deadline = time.monotonic() + args.startup_timeout_s
        pending = {item[2]["slot"] for item in processes}
        while pending and time.monotonic() < deadline and not stopping:
            for process, _log, record in processes:
                slot = record["slot"]
                if slot not in pending:
                    continue
                if process.poll() is not None:
                    raise RuntimeError(
                        f"environment slot {slot} exited with {process.returncode}"
                    )
                try:
                    health = _health(record["endpoint"])
                except Exception:
                    continue
                if health.get("status") == "healthy" and health.get("renderer", {}).get(
                    "ready"
                ):
                    record["health"] = _public_health(health)
                    record["status"] = "ready"
                    pending.remove(slot)
            if pending:
                time.sleep(0.2)
        if pending:
            raise TimeoutError(
                f"environment slots did not become ready: {sorted(pending)}"
            )
        manifest = public_manifest()
        persist_manifest()
        print(json.dumps(manifest, ensure_ascii=False, sort_keys=True), flush=True)
        while not stopping:
            changed = False
            now = time.monotonic()
            for index, (process, log, record) in enumerate(processes):
                if process is None:
                    continue
                if process.poll() is not None:
                    if log is not None:
                        log.close()
                    slot = int(record["slot"])
                    history = [
                        stamp
                        for stamp in restart_history.get(slot, [])
                        if now - stamp <= args.restart_window_s
                    ]
                    history.append(now)
                    restart_history[slot] = history
                    record["last_exit_code"] = process.returncode
                    record.pop("health", None)
                    if len(history) > args.restart_limit:
                        record["status"] = "quarantined"
                        record["restart_count"] = len(history)
                        processes[index] = (None, None, record)
                    else:
                        replacement = launch_slot(
                            slot, generation=int(record.get("generation", 0)) + 1
                        )
                        processes[index] = replacement
                    changed = True
                    continue
                if record.get("status") == "restarting":
                    try:
                        health = _health(record["endpoint"])
                    except Exception:
                        continue
                    if health.get("status") == "healthy" and health.get(
                        "renderer", {}
                    ).get("ready"):
                        record["health"] = _public_health(health)
                        record["status"] = "ready"
                        changed = True
                    continue
                if record.get("status") == "ready":
                    try:
                        health = _health(record["endpoint"])
                    except Exception:
                        slot = int(record["slot"])
                        failures = int(record.get("health_failures", 0)) + 1
                        record["health_failures"] = failures
                        first_failure = health_failure_started.setdefault(slot, now)
                        unreachable_s = max(0.0, now - first_failure)
                        record["health_unreachable_s"] = round(unreachable_s, 3)
                        lease_live = False
                        if args.slot_broker_root is not None:
                            lease_live = _lease_is_live(
                                args.slot_broker_root
                                / "leases"
                                / f"slot-{slot:03d}.json"
                            )
                        record["health_restart_deferred_for_live_lease"] = lease_live
                        if _health_restart_due(
                            first_failure_at=first_failure,
                            now=now,
                            grace_s=args.health_restart_grace_s,
                            lease_live=lease_live,
                        ):
                            record["status"] = "unhealthy_restarting"
                            process.terminate()
                            changed = True
                        continue
                    health_failure_started.pop(int(record["slot"]), None)
                    record.pop("health_failures", None)
                    record.pop("health_unreachable_s", None)
                    record.pop("health_restart_deferred_for_live_lease", None)
                    protocol = health.get("write_protocol")
                    phase = (
                        protocol.get("phase") if isinstance(protocol, dict) else None
                    )
                    lease_live = False
                    if args.slot_broker_root is not None:
                        lease_live = _lease_is_live(
                            args.slot_broker_root
                            / "leases"
                            / f"slot-{int(record['slot']):03d}.json"
                        )
                    if phase in (None, "FREE"):
                        record.pop("lease_observed_for_binding", None)
                    elif lease_live:
                        record["lease_observed_for_binding"] = True
                    if phase == "LOST" or _bound_session_is_orphaned(
                        phase=phase,
                        lease_observed=bool(
                            record.get("lease_observed_for_binding", False)
                        ),
                        lease_live=lease_live,
                    ):
                        record["status"] = (
                            "lost_restarting"
                            if phase == "LOST"
                            else "orphan_restarting"
                        )
                        record["last_write_phase"] = phase
                        process.terminate()
                        changed = True
            if changed:
                persist_manifest()
            time.sleep(1.0)
    finally:
        stop()
        for process, log, _record in processes:
            if process is None:
                continue
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
            if log is not None:
                log.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
