# Copyright (c) 2026 Zetta Contributors
"""Shared-filesystem host queues and persistent rollout workers."""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

from zetta.evolution.jsonio import atomic_write_json, canonical_sha256, read_json
from zetta.evolution.provider_admission import (
    ProviderAdmission,
    ProviderAdmissionConfig,
    ProviderFailure,
    ProviderLease,
)
from zetta.evolution.slot_broker import EnvironmentSlotBroker, EnvironmentSlotLease

_CLAIM_LOCKS_GUARD = threading.Lock()
_CLAIM_LOCKS: dict[str, threading.Lock] = {}
_QUEUE_SCHEMA_VERSION = 2
_PENDING_KIND = "rollout_pending"
_CLAIM_KIND = "rollout_claim"
_TERMINAL_KIND = "rollout_terminal"
_DEFAULT_CLAIM_LEASE_S = 7200.0


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _timestamp(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat()


def _parse_timestamp(value: object, *, name: str) -> datetime:
    if not isinstance(value, str) or not value:
        raise ValueError(f"queue envelope requires non-empty {name}")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"queue envelope has invalid {name}") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"queue envelope {name} must include a timezone")
    return parsed.astimezone(timezone.utc)


def _job_from_envelope(payload: object) -> "RolloutJob":
    """Decode both v2 queue envelopes and legacy raw job JSON."""

    if not isinstance(payload, dict):
        raise ValueError("queue envelope must be an object")
    nested = payload.get("job")
    job_payload = nested if isinstance(nested, dict) else payload
    job = RolloutJob.from_dict(job_payload)
    declared = payload.get("job_id", job.job_id)
    if declared != job.job_id:
        raise ValueError("queue envelope job_id does not match nested job")
    return job


def _job_id_from_result(result: dict[str, Any]) -> str | None:
    declared = result.get("job_id")
    if isinstance(declared, str) and declared:
        return declared
    nested = result.get("job")
    if isinstance(nested, dict):
        declared = nested.get("job_id")
        if isinstance(declared, str) and declared:
            return declared
    return None


@contextmanager
def _queue_claim_lock(root: Path, host: str):
    """Portable same-host process/thread serialization for atomic claims."""

    identity = str((root.resolve(), host))
    with _CLAIM_LOCKS_GUARD:
        local = _CLAIM_LOCKS.setdefault(identity, threading.Lock())
    with local:
        locks = root / ".claim-locks"
        locks.mkdir(parents=True, exist_ok=True)
        lock = locks / f"{host}.lock"
        deadline = time.monotonic() + 30.0
        while True:
            try:
                lock.mkdir()
                break
            except FileExistsError:
                try:
                    stale = time.time() - lock.stat().st_mtime > 120.0
                except FileNotFoundError:
                    continue
                if stale:
                    quarantine = locks / f"{host}.stale-{uuid.uuid4().hex}"
                    try:
                        lock.rename(quarantine)
                    except OSError:
                        continue
                    shutil.rmtree(quarantine, ignore_errors=True)
                    continue
                if time.monotonic() >= deadline:
                    raise TimeoutError("timed out acquiring queue claim lock")
                time.sleep(0.01)
        try:
            yield
        finally:
            try:
                lock.rmdir()
            except FileNotFoundError:
                pass


@dataclass(frozen=True, slots=True)
class RolloutJob:
    job_id: str
    campaign_root: str
    logical_id: str
    attempt_index: int
    task: str
    seed: int
    policy_rng: int
    bundle_sha256: str | None
    command: tuple[str, ...]
    output_dir: str
    result_file: str
    heartbeat_file: str
    timeout_s: int = 2700
    no_progress_timeout_s: int = 180
    requires_api: bool = False
    requires_environment_slot: bool = False

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "RolloutJob":
        value = dict(payload)
        value["command"] = tuple(str(part) for part in value["command"])
        return cls(**value)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def scheduling_key(self) -> str:
        """Fairness domain: a task and one frozen candidate lineage."""

        return f"{self.task}:{self.bundle_sha256 or 'parent'}"


class SharedHostQueue:
    """Shared-filesystem queue with leased claims and one terminal commit point."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        for name in ("pending", "running", "completed", "failed"):
            (self.root / name).mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _job_filename(job_id: str) -> str:
        if not isinstance(job_id, str) or not job_id:
            raise ValueError("queue job_id must be a non-empty string")
        return f"job-{canonical_sha256(job_id)[:32]}.json"

    def _terminal_path(self, host: str, job_id: str, *, success: bool) -> Path:
        bucket = "completed" if success else "failed"
        return self.root / bucket / host / self._job_filename(job_id)

    def _terminal_candidates(self, host: str, job_id: str) -> list[Path]:
        result: list[Path] = []
        for success in (True, False):
            path = self._terminal_path(host, job_id, success=success)
            if path.exists():
                result.append(path)
        return result

    @staticmethod
    def _pending_envelope(job: RolloutJob) -> dict[str, Any]:
        return {
            "schema_version": _QUEUE_SCHEMA_VERSION,
            "kind": _PENDING_KIND,
            "job_id": job.job_id,
            "job": job.as_dict(),
        }

    @staticmethod
    def _claim_envelope(
        job: RolloutJob,
        *,
        claim_token: str,
        worker_id: str,
        acquired_at: datetime,
        lease_s: float,
    ) -> dict[str, Any]:
        heartbeat = _timestamp(acquired_at)
        return {
            "schema_version": _QUEUE_SCHEMA_VERSION,
            "kind": _CLAIM_KIND,
            "job_id": job.job_id,
            "claim_token": claim_token,
            "worker_id": worker_id,
            "acquired_at": heartbeat,
            "heartbeat": heartbeat,
            "expires_at": _timestamp(acquired_at + timedelta(seconds=lease_s)),
            "job": job.as_dict(),
        }

    @staticmethod
    def _legacy_claim_envelope(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
        """Give an old raw running job a stable token during rolling upgrade."""

        job = _job_from_envelope(payload)
        try:
            acquired = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
        except FileNotFoundError:
            acquired = _utc_now()
        token = (
            "legacy-"
            + canonical_sha256({"job_id": job.job_id, "claim_file": path.name})[:32]
        )
        return SharedHostQueue._claim_envelope(
            job,
            claim_token=token,
            worker_id="legacy-worker",
            acquired_at=acquired,
            lease_s=_DEFAULT_CLAIM_LEASE_S,
        )

    @staticmethod
    def _claim_fields(payload: dict[str, Any]) -> tuple[str, str]:
        token = payload.get("claim_token")
        worker_id = payload.get("worker_id")
        if not isinstance(token, str) or not token:
            raise ValueError("running claim requires a non-empty claim_token")
        if not isinstance(worker_id, str) or not worker_id:
            raise ValueError("running claim requires a non-empty worker_id")
        _parse_timestamp(payload.get("acquired_at"), name="acquired_at")
        _parse_timestamp(payload.get("heartbeat"), name="heartbeat")
        _parse_timestamp(payload.get("expires_at"), name="expires_at")
        return token, worker_id

    def _read_claim(self, claimed: Path) -> tuple[dict[str, Any], RolloutJob]:
        payload = read_json(claimed)
        if not isinstance(payload, dict):
            raise ValueError("running claim must be an object")
        if payload.get("kind") != _CLAIM_KIND:
            payload = self._legacy_claim_envelope(claimed, payload)
            atomic_write_json(claimed, payload, overwrite=True)
        job = _job_from_envelope(payload)
        self._claim_fields(payload)
        return payload, job

    @staticmethod
    def _result_digest(*, success: bool, result: dict[str, Any]) -> str:
        return canonical_sha256({"success": bool(success), "result": result})

    @staticmethod
    def _terminal_envelope(
        *,
        claim: dict[str, Any],
        job: RolloutJob,
        success: bool,
        result: dict[str, Any],
        commit_source: str,
    ) -> dict[str, Any]:
        return {
            "schema_version": _QUEUE_SCHEMA_VERSION,
            "kind": _TERMINAL_KIND,
            "job_id": job.job_id,
            "claim_token": claim["claim_token"],
            "worker_id": claim["worker_id"],
            "acquired_at": claim["acquired_at"],
            "last_heartbeat": claim["heartbeat"],
            "finished_at": _timestamp(_utc_now()),
            "success": bool(success),
            "commit_source": commit_source,
            "result_digest": SharedHostQueue._result_digest(
                success=success, result=result
            ),
            "result": result,
            "job": job.as_dict(),
        }

    @staticmethod
    def _assert_same_terminal(
        existing: dict[str, Any],
        *,
        job_id: str,
        claim_token: str,
        success: bool,
        result: dict[str, Any],
        commit_source: str,
    ) -> None:
        if existing.get("kind") != _TERMINAL_KIND:
            raise ValueError("terminal path contains a non-terminal envelope")
        expected_digest = SharedHostQueue._result_digest(success=success, result=result)
        if existing.get("job_id") != job_id:
            raise ValueError("terminal envelope job_id mismatch")
        if existing.get("claim_token") != claim_token:
            raise ValueError("stale claim token cannot finish this job")
        if existing.get("commit_source") != commit_source:
            raise ValueError("claim was already closed by a different authority")
        if existing.get("result_digest") != expected_digest:
            raise ValueError("terminal result digest differs from immutable commit")

    def _find_terminal(
        self, host: str, job_id: str
    ) -> tuple[Path, dict[str, Any]] | None:
        candidates = self._terminal_candidates(host, job_id)
        if len(candidates) > 1:
            raise ValueError("job has conflicting completed and failed terminals")
        if not candidates:
            return None
        path = candidates[0]
        payload = read_json(path)
        if not isinstance(payload, dict):
            raise ValueError("terminal envelope must be an object")
        if _job_from_envelope(payload).job_id != job_id:
            raise ValueError("terminal filename and envelope identity differ")
        return path, payload

    def enqueue(self, host: str, job: RolloutJob) -> Path:
        target_dir = self.root / "pending" / host
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / self._job_filename(job.job_id)
        atomic_write_json(target, self._pending_envelope(job), overwrite=False)
        return target

    def claim(
        self,
        host: str,
        *,
        worker_id: str | None = None,
        lease_s: float = _DEFAULT_CLAIM_LEASE_S,
    ) -> tuple[Path, RolloutJob] | None:
        pending = self.root / "pending" / host
        running = self.root / "running" / host
        pending.mkdir(parents=True, exist_ok=True)
        running.mkdir(parents=True, exist_ok=True)
        if lease_s <= 0:
            raise ValueError("claim lease_s must be positive")
        owner = worker_id or f"{host}:{os.getpid()}:{threading.get_ident()}"
        with _queue_claim_lock(self.root, host):
            available: dict[str, list[tuple[Path, RolloutJob]]] = {}
            for source in sorted(pending.glob("*.json")):
                payload = read_json(source)
                job = _job_from_envelope(payload)
                terminal = self._find_terminal(host, job.job_id)
                if terminal is not None:
                    source.unlink(missing_ok=True)
                    continue
                available.setdefault(job.scheduling_key, []).append((source, job))
            if not available:
                return None
            cursor_path = self.root / ".fair-cursors" / f"{host}.json"
            last_key: str | None = None
            if cursor_path.exists():
                cursor = read_json(cursor_path)
                if isinstance(cursor, dict) and isinstance(
                    cursor.get("last_scheduling_key"), str
                ):
                    last_key = cursor["last_scheduling_key"]
            keys = sorted(available)
            if last_key in keys:
                start = (keys.index(last_key) + 1) % len(keys)
                keys = keys[start:] + keys[:start]
            scheduling_key = keys[0]
            source, job = min(
                available[scheduling_key],
                key=lambda item: (item[0].stat().st_mtime_ns, item[0].name),
            )
            try:
                claim_token = uuid.uuid4().hex
                acquired = _utc_now()
                claim = self._claim_envelope(
                    job,
                    claim_token=claim_token,
                    worker_id=owner,
                    acquired_at=acquired,
                    lease_s=lease_s,
                )
                # Publish claim metadata before the ownership rename.  A crash
                # here leaves a self-describing pending envelope that the next
                # claimer safely replaces with a fresh token.
                atomic_write_json(source, claim, overwrite=True)
                target = running / f"claim-{claim_token}.json"
                try:
                    source.rename(target)
                except OSError:
                    if source.exists():
                        raise
                    return None
                atomic_write_json(
                    cursor_path,
                    {
                        "schema_version": 1,
                        "host": host,
                        "last_scheduling_key": scheduling_key,
                        "claimed_at": _timestamp(acquired),
                    },
                    overwrite=True,
                )
                return target, job
            except FileNotFoundError:
                return None
        return None

    def claim_token(self, claimed: str | Path) -> str:
        payload, _ = self._read_claim(Path(claimed))
        token, _ = self._claim_fields(payload)
        return token

    def heartbeat(
        self,
        claimed: str | Path,
        *,
        claim_token: str,
        lease_s: float = _DEFAULT_CLAIM_LEASE_S,
    ) -> None:
        """Renew a live claim only when its persisted fencing token matches."""

        if lease_s <= 0:
            raise ValueError("claim heartbeat lease_s must be positive")
        path = Path(claimed)
        host = path.parent.name
        with _queue_claim_lock(self.root, host):
            if not path.exists():
                raise ValueError("cannot heartbeat a non-active claim")
            payload, _ = self._read_claim(path)
            persisted, _ = self._claim_fields(payload)
            if persisted != claim_token:
                raise ValueError("stale claim token cannot renew this job")
            now = _utc_now()
            expires = _parse_timestamp(payload["expires_at"], name="expires_at")
            if now > expires:
                raise ValueError("expired claim token cannot renew this job")
            payload["heartbeat"] = _timestamp(now)
            payload["expires_at"] = _timestamp(now + timedelta(seconds=lease_s))
            atomic_write_json(path, payload, overwrite=True)

    def _finish_locked(
        self,
        claimed: Path,
        *,
        success: bool,
        result: dict[str, Any],
        claim_token: str | None,
        commit_source: str,
    ) -> Path:
        host = claimed.parent.name
        result_job_id = _job_id_from_result(result)
        claim: dict[str, Any] | None = None
        job: RolloutJob | None = None
        if claimed.exists():
            claim, job = self._read_claim(claimed)
            persisted_token, _ = self._claim_fields(claim)
            if claim_token is not None and persisted_token != claim_token:
                raise ValueError("stale claim token cannot finish this job")
            claim_token = persisted_token
            expires = _parse_timestamp(claim["expires_at"], name="expires_at")
            if commit_source == "worker" and _utc_now() > expires:
                raise ValueError("expired claim token cannot finish this job")
            if result_job_id is not None and result_job_id != job.job_id:
                raise ValueError("result job_id does not match running claim")
        elif result_job_id is None or claim_token is None:
            raise ValueError(
                "finishing a missing claim requires result job_id and claim_token"
            )

        job_id = job.job_id if job is not None else result_job_id
        assert job_id is not None
        assert claim_token is not None
        existing = self._find_terminal(host, job_id)
        if existing is not None:
            path, payload = existing
            self._assert_same_terminal(
                payload,
                job_id=job_id,
                claim_token=claim_token,
                success=success,
                result=result,
                commit_source=commit_source,
            )
            if claimed.exists():
                current, _ = self._read_claim(claimed)
                if current.get("claim_token") != claim_token:
                    raise ValueError("stale running claim cannot be cleaned")
                claimed.unlink()
            return path

        if claim is None or job is None:
            raise ValueError("terminal commit is missing its active running claim")
        target = self._terminal_path(host, job.job_id, success=success)
        target.parent.mkdir(parents=True, exist_ok=True)
        terminal = self._terminal_envelope(
            claim=claim,
            job=job,
            success=success,
            result=result,
            commit_source=commit_source,
        )
        # This immutable publish is the sole terminal commit point.  Removing
        # the running claim is cleanup and can be retried after a crash.
        atomic_write_json(target, terminal, overwrite=False)
        if claimed.exists():
            current, _ = self._read_claim(claimed)
            if current.get("claim_token") != claim_token:
                raise ValueError("claim token changed during terminal commit")
            claimed.unlink()
        return target

    def finish(
        self,
        claimed: str | Path,
        *,
        success: bool,
        result: dict[str, Any],
        claim_token: str | None = None,
    ) -> Path:
        """Commit one immutable terminal envelope; exact retries are idempotent."""

        if not isinstance(result, dict):
            raise ValueError("queue result must be an object")
        path = Path(claimed)
        host = path.parent.name
        with _queue_claim_lock(self.root, host):
            return self._finish_locked(
                path,
                success=success,
                result=result,
                claim_token=claim_token,
                commit_source="worker",
            )

    def counts(self) -> dict[str, int]:
        return {
            name: sum(
                not path.name.endswith(".result.json")
                for path in (self.root / name).glob("*/*.json")
            )
            for name in ("pending", "running", "completed", "failed")
        }

    def known_job_ids(self) -> set[str]:
        """Return identities decoded from JSON, never from mutable filenames."""

        identities: set[str] = set()
        for bucket in ("pending", "running", "completed", "failed"):
            for path in sorted((self.root / bucket).glob("*/*.json")):
                try:
                    payload = read_json(path)
                except FileNotFoundError:
                    # Workers atomically rename claims between buckets.  A
                    # path discovered by glob may therefore disappear before
                    # it is opened; the same job is visible in its new bucket
                    # (or will be on the next idempotent supervisor pass).
                    continue
                if not isinstance(payload, dict):
                    raise ValueError(f"queue envelope is not an object: {path}")
                if isinstance(payload.get("job"), dict) or "command" in payload:
                    identities.add(_job_from_envelope(payload).job_id)
                else:
                    declared = payload.get("job_id")
                    if isinstance(declared, str) and declared:
                        identities.add(declared)
                    else:
                        raise ValueError(f"queue envelope has no job identity: {path}")
        return identities

    def terminal_envelope_paths(self) -> list[Path]:
        """List v2 terminals plus readable legacy result sidecars."""

        paths: list[Path] = []
        for bucket in ("completed", "failed"):
            for path in sorted((self.root / bucket).glob("*/*.json")):
                try:
                    payload = read_json(path)
                except FileNotFoundError:
                    # A concurrent terminal cleanup/upgrade may move the
                    # envelope after globbing.  Never turn that benign race
                    # into a supervisor crash.
                    continue
                if path.name.endswith(".result.json"):
                    paths.append(path)
                elif (
                    isinstance(payload, dict) and payload.get("kind") == _TERMINAL_KIND
                ):
                    paths.append(path)
        return paths

    @staticmethod
    def _valid_published_result(job: RolloutJob) -> dict[str, Any] | None:
        path = Path(job.result_file)
        if not path.exists():
            return None
        payload = read_json(path)
        if not isinstance(payload, dict):
            return None
        if payload.get("logical_id") != job.logical_id:
            return None
        attempt = payload.get("attempt_index", job.attempt_index)
        if attempt != job.attempt_index:
            return None
        declared_job = payload.get("job_id")
        if declared_job is not None and declared_job != job.job_id:
            return None
        if payload.get("status") not in ("valid", "infra_invalid"):
            return None
        return {
            "job_id": job.job_id,
            "logical_id": job.logical_id,
            "attempt_index": job.attempt_index,
            "success": True,
            "recovery_reason": "published_result_file_after_worker_crash",
            "result": payload,
            "job": job.as_dict(),
        }

    def _legacy_sidecar_result(
        self, claimed: Path, job: RolloutJob
    ) -> tuple[bool, dict[str, Any]] | None:
        host = claimed.parent.name
        for bucket, success in (("completed", True), ("failed", False)):
            sidecar = (
                self.root / bucket / host / claimed.with_suffix(".result.json").name
            )
            if not sidecar.exists():
                continue
            payload = read_json(sidecar)
            if not isinstance(payload, dict):
                raise ValueError("legacy result sidecar must be an object")
            declared_job = _job_id_from_result(payload)
            if declared_job is not None and declared_job != job.job_id:
                raise ValueError("legacy sidecar job_id does not match running claim")
            return success, payload
        return None

    def recover_abandoned(self, host: str, *, stale_after_s: float) -> int:
        """Finish or close stale claims without replaying their side effects."""

        if stale_after_s < 0:
            raise ValueError("stale_after_s must be non-negative")
        recovered = 0
        for claimed in sorted((self.root / "running" / host).glob("*.json")):
            with _queue_claim_lock(self.root, host):
                if not claimed.exists():
                    continue
                claim, job = self._read_claim(claimed)
                claim_token, _ = self._claim_fields(claim)
                terminal = self._find_terminal(host, job.job_id)
                if terminal is not None:
                    _, terminal_payload = terminal
                    if terminal_payload.get("claim_token") != claim_token:
                        raise ValueError("terminal belongs to a different claim token")
                    claimed.unlink()
                    recovered += 1
                    continue
                heartbeat = _parse_timestamp(claim["heartbeat"], name="heartbeat")
                now = _utc_now()
                heartbeat_age = max(0.0, (now - heartbeat).total_seconds())
                expired = now > _parse_timestamp(claim["expires_at"], name="expires_at")
                try:
                    mtime_age = max(0.0, time.time() - claimed.stat().st_mtime)
                except FileNotFoundError:
                    continue
                # mtime remains a rolling-upgrade compatibility heartbeat for
                # workers that predate the persisted heartbeat field.
                age = max(heartbeat_age, mtime_age)
                if not expired and age < stale_after_s:
                    continue
                legacy = self._legacy_sidecar_result(claimed, job)
                published = self._valid_published_result(job)
                if legacy is not None:
                    success, result = legacy
                elif published is not None:
                    success, result = True, published
                else:
                    success = False
                    result = {
                        "job_id": job.job_id,
                        "logical_id": job.logical_id,
                        "attempt_index": job.attempt_index,
                        "success": False,
                        "watchdog_reason": "abandoned_worker_process",
                        "elapsed_s": age,
                        "job": job.as_dict(),
                    }
                self._finish_locked(
                    claimed,
                    success=success,
                    result=result,
                    claim_token=claim_token,
                    commit_source="recovery",
                )
                recovered += 1
        return recovered


class SubprocessRolloutExecutor:
    """Run one immutable command with wall/no-progress watchdogs."""

    def run(
        self,
        job: RolloutJob,
        *,
        progress_callback: Callable[[], None] | None = None,
        environment_endpoint: str | None = None,
    ) -> dict[str, Any]:
        output_dir = Path(job.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        stdout_path = output_dir / "worker.stdout.log"
        started = time.time()
        last_progress = started
        heartbeat = Path(job.heartbeat_file)
        last_mtime = heartbeat.stat().st_mtime if heartbeat.exists() else None
        reason: str | None = None
        command = list(job.command)
        endpoint_token = "__ZETTA_ENV_ENDPOINT__"
        token_count = sum(part.count(endpoint_token) for part in command)
        if job.requires_environment_slot:
            if environment_endpoint is None:
                raise RuntimeError("rollout requires an environment slot lease")
            if token_count != 1:
                raise ValueError(
                    "slot-backed rollout command requires exactly one environment endpoint token"
                )
            command = [
                part.replace(endpoint_token, environment_endpoint) for part in command
            ]
        elif token_count:
            raise ValueError(
                "rollout command contains an environment endpoint token without slot admission"
            )
        process_environment = {
            **os.environ,
            "ZETTA_POLICY_RNG_SEED": str(job.policy_rng),
        }
        if environment_endpoint is not None:
            process_environment["ZETTA_ROBOCASA_ENV_ENDPOINT"] = environment_endpoint
        with stdout_path.open("ab") as stream:
            process = subprocess.Popen(
                command,
                cwd=output_dir,
                stdout=stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                env=process_environment,
            )
            while process.poll() is None:
                now = time.time()
                if progress_callback is not None:
                    try:
                        progress_callback()
                    except Exception:
                        reason = "provider_admission_heartbeat_failed"
                        break
                if heartbeat.exists():
                    mtime = heartbeat.stat().st_mtime
                    if last_mtime is None or mtime > last_mtime:
                        last_progress = now
                        last_mtime = mtime
                if now - started > job.timeout_s:
                    reason = "episode_wall_timeout"
                    break
                if now - last_progress > job.no_progress_timeout_s:
                    reason = "episode_no_progress_timeout"
                    break
                time.sleep(1.0)
            if reason is not None and process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=10)
            return_code = process.wait()
        result_path = Path(job.result_file)
        result_payload = read_json(result_path) if result_path.exists() else None
        return {
            "job_id": job.job_id,
            "logical_id": job.logical_id,
            "attempt_index": job.attempt_index,
            "return_code": return_code,
            "watchdog_reason": reason,
            "elapsed_s": time.time() - started,
            "result_file": str(result_path),
            "result": result_payload,
            "success": reason is None
            and return_code == 0
            and result_payload is not None,
            "job": job.as_dict(),
        }


@dataclass(frozen=True, slots=True)
class _ProviderRuntime:
    admission: ProviderAdmission
    route_id: str
    acquire_timeout_s: float
    heartbeat_s: float


class _ProviderLeaseHeartbeat:
    """Renew a long-running episode lease without exposing provider details."""

    def __init__(self, lease: ProviderLease, *, interval_s: float) -> None:
        if interval_s <= 0:
            raise ValueError("provider heartbeat interval must be positive")
        self._lease = lease
        self._interval_s = interval_s
        self._next_heartbeat = time.monotonic() + interval_s

    def __call__(self) -> None:
        now = time.monotonic()
        if now < self._next_heartbeat:
            return
        self._lease.heartbeat()
        self._next_heartbeat = now + self._interval_s


class _QueueClaimHeartbeat:
    """Persist liveness for a fenced queue claim during long execution."""

    def __init__(
        self,
        queue: SharedHostQueue,
        claimed: Path,
        claim_token: str,
        *,
        interval_s: float = 30.0,
        lease_s: float = _DEFAULT_CLAIM_LEASE_S,
    ) -> None:
        self._queue = queue
        self._claimed = claimed
        self._claim_token = claim_token
        self._interval_s = interval_s
        self._lease_s = lease_s
        self._next_heartbeat = time.monotonic() + interval_s

    def __call__(self) -> None:
        now = time.monotonic()
        if now < self._next_heartbeat:
            return
        self._queue.heartbeat(
            self._claimed,
            claim_token=self._claim_token,
            lease_s=self._lease_s,
        )
        self._next_heartbeat = now + self._interval_s


class _EnvironmentLeaseHeartbeat:
    """Renew a host-local environment lease while its episode is alive."""

    def __init__(
        self, lease: EnvironmentSlotLease, *, interval_s: float = 30.0
    ) -> None:
        if interval_s <= 0:
            raise ValueError("environment lease heartbeat interval must be positive")
        self._lease = lease
        self._interval_s = interval_s
        self._next_heartbeat = time.monotonic() + interval_s

    def __call__(self) -> None:
        now = time.monotonic()
        if now < self._next_heartbeat:
            return
        self._lease.heartbeat()
        self._next_heartbeat = now + self._interval_s


class _CombinedHeartbeat:
    def __init__(self, *callbacks: Callable[[], None]) -> None:
        self._callbacks = callbacks

    def __call__(self) -> None:
        for callback in self._callbacks:
            callback()


@dataclass(frozen=True, slots=True)
class _ProviderOutcome:
    outcome: str
    tokens: int
    latency_s: float


def _optional_positive_float(value: str | float | None, *, name: str) -> float:
    try:
        parsed = float(value) if value is not None else 0.0
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive number") from exc
    if parsed <= 0:
        raise ValueError(f"{name} must be a positive number")
    return parsed


def _optional_positive_int(value: str | int | None, *, name: str) -> int | None:
    if value is None:
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if parsed < 1:
        raise ValueError(f"{name} must be a positive integer")
    return parsed


def _build_provider_runtime(
    *,
    root: str | Path,
    route_id: str,
    initial_limit: int | None,
    max_limit: int | None,
    acquire_timeout_s: float,
    heartbeat_s: float,
) -> _ProviderRuntime:
    defaults = ProviderAdmissionConfig()
    maximum = defaults.max_limit if max_limit is None else max_limit
    initial = (
        min(defaults.initial_limit, maximum) if initial_limit is None else initial_limit
    )
    config = ProviderAdmissionConfig(initial_limit=initial, max_limit=maximum)
    admission = ProviderAdmission(root, config=config)
    admission.register_route(
        route_id,
        initial_limit=initial,
        max_limit=maximum,
    )
    return _ProviderRuntime(
        admission=admission,
        route_id=route_id,
        acquire_timeout_s=acquire_timeout_s,
        heartbeat_s=heartbeat_s,
    )


def _provider_outcome(result: dict[str, Any]) -> _ProviderOutcome | None:
    """Read an explicit, secret-free provider metric envelope if complete."""

    candidates: list[dict[str, Any]] = [result]
    nested_result = result.get("result")
    if isinstance(nested_result, dict):
        candidates.append(nested_result)

    for candidate in candidates:
        nested = candidate.get("provider")
        if not isinstance(nested, dict):
            nested = candidate.get("provider_metrics")
        if isinstance(nested, dict):
            raw_outcome = nested.get("outcome")
            raw_tokens = nested.get("tokens")
            raw_latency = nested.get("latency_s")
            raw_failure = nested.get("failure")
        else:
            raw_outcome = candidate.get("provider_outcome")
            raw_tokens = candidate.get("provider_tokens")
            raw_latency = candidate.get("provider_latency_s")
            raw_failure = candidate.get("provider_failure")

        values = (raw_outcome, raw_tokens, raw_latency, raw_failure)
        if all(value is None for value in values):
            continue
        if raw_outcome in ("failure", "failed"):
            raw_outcome = raw_failure
        if raw_outcome is None or raw_tokens is None or raw_latency is None:
            return None
        if (
            not isinstance(raw_tokens, int)
            or isinstance(raw_tokens, bool)
            or raw_tokens < 0
            or not isinstance(raw_latency, (int, float))
            or isinstance(raw_latency, bool)
            or raw_latency < 0
        ):
            raise ValueError("provider metrics have invalid numeric fields")
        outcome = str(raw_outcome)
        if outcome != "success":
            try:
                outcome = ProviderFailure(outcome).value
            except ValueError as exc:
                raise ValueError("provider metrics have an invalid outcome") from exc
        return _ProviderOutcome(
            outcome=outcome,
            tokens=raw_tokens,
            latency_s=float(raw_latency),
        )
    return None


def _report_provider_outcome(lease: ProviderLease, result: dict[str, Any]) -> None:
    outcome = _provider_outcome(result)
    if outcome is None:
        return
    if outcome.outcome == "success":
        lease.succeed(tokens=outcome.tokens, latency_s=outcome.latency_s)
    else:
        lease.fail(
            ProviderFailure(outcome.outcome),
            tokens=outcome.tokens,
            latency_s=outcome.latency_s,
        )


def run_worker(
    *,
    queue_root: str | Path,
    host: str,
    poll_s: float = 2.0,
    once: bool = False,
    provider_admission_root: str | Path | None = None,
    provider_route_id: str | None = None,
    provider_initial_limit: int | None = None,
    provider_max_limit: int | None = None,
    provider_acquire_timeout_s: float | None = None,
    provider_heartbeat_s: float | None = None,
    environment_slot_broker_root: str | Path | None = None,
    environment_ready_manifest: str | Path | None = None,
    maximum_active_environment_slots: int | None = None,
    environment_slot_acquire_timeout_s: float = 1800.0,
    worker_id: str | None = None,
    stop_event: threading.Event | None = None,
) -> int:
    queue = SharedHostQueue(queue_root)
    executor = SubprocessRolloutExecutor()
    provider_runtime: _ProviderRuntime | None = None
    slot_broker: EnvironmentSlotBroker | None = None

    def get_slot_broker() -> EnvironmentSlotBroker | None:
        nonlocal slot_broker
        if slot_broker is not None:
            return slot_broker
        root = environment_slot_broker_root or os.environ.get(
            "ZETTA_ENVIRONMENT_SLOT_BROKER_ROOT"
        )
        manifest = environment_ready_manifest or os.environ.get(
            "ZETTA_ENVIRONMENT_READY_MANIFEST"
        )
        if root is None and manifest is None:
            return None
        if root is None or manifest is None:
            raise ValueError(
                "environment slot broker root and ready manifest must be configured together"
            )
        active = maximum_active_environment_slots
        if active is None:
            active = _optional_positive_int(
                os.environ.get("ZETTA_MAX_ACTIVE_ENVIRONMENT_SLOTS"),
                name="maximum active environment slots",
            )
        slot_broker = EnvironmentSlotBroker(
            root,
            ready_manifest=manifest,
            maximum_active_slots=active,
        )
        return slot_broker

    def get_provider_runtime() -> _ProviderRuntime | None:
        nonlocal provider_runtime
        if provider_runtime is not None:
            return provider_runtime
        root = (
            provider_admission_root
            or os.environ.get("ZETTA_PROVIDER_ADMISSION_ROOT")
            or os.environ.get("ZETTA_API_ADMISSION_ROOT")
        )
        if root is None:
            return None
        route_id = (
            provider_route_id or os.environ.get("ZETTA_PROVIDER_ROUTE_ID") or "default"
        )
        legacy_limit = os.environ.get("ZETTA_API_MAX_CONCURRENCY")
        initial = provider_initial_limit
        if initial is None:
            initial = _optional_positive_int(
                os.environ.get("ZETTA_PROVIDER_INITIAL_CONCURRENCY") or legacy_limit,
                name="provider initial concurrency",
            )
        maximum = provider_max_limit
        if maximum is None:
            maximum = _optional_positive_int(
                os.environ.get("ZETTA_PROVIDER_MAX_CONCURRENCY") or legacy_limit,
                name="provider max concurrency",
            )
        acquire_timeout = provider_acquire_timeout_s
        if acquire_timeout is None:
            acquire_timeout = _optional_positive_float(
                os.environ.get("ZETTA_PROVIDER_ACQUIRE_TIMEOUT_S") or 1800.0,
                name="provider acquire timeout",
            )
        heartbeat = provider_heartbeat_s
        if heartbeat is None:
            heartbeat = _optional_positive_float(
                os.environ.get("ZETTA_PROVIDER_HEARTBEAT_S") or 60.0,
                name="provider heartbeat interval",
            )
        provider_runtime = _build_provider_runtime(
            root=root,
            route_id=route_id,
            initial_limit=initial,
            max_limit=maximum,
            acquire_timeout_s=acquire_timeout,
            heartbeat_s=heartbeat,
        )
        return provider_runtime

    while True:
        if stop_event is not None and stop_event.is_set():
            return 0
        active_worker_id = (
            worker_id
            or os.environ.get("ZETTA_WORKER_ID")
            or (f"{host}:{os.getpid()}:{threading.get_ident()}")
        )
        try:
            claimed = queue.claim(host, worker_id=active_worker_id)
        except TimeoutError:
            # The claim lock lives on shared storage. A slow metadata scan or
            # another same-host worker may legitimately hold it longer than
            # one acquisition window. This is not an episode failure and must
            # not kill a persistent worker; retry without claiming or mutating
            # any job envelope.
            if stop_event is None:
                time.sleep(poll_s)
            else:
                stop_event.wait(poll_s)
            continue
        if claimed is None:
            if once:
                return 0
            if stop_event is None:
                time.sleep(poll_s)
            else:
                stop_event.wait(poll_s)
            continue
        claimed_path, job = claimed
        claim_token = queue.claim_token(claimed_path)
        claim_heartbeat = _QueueClaimHeartbeat(
            queue,
            claimed_path,
            claim_token,
        )
        environment_lease: EnvironmentSlotLease | None = None
        result: dict[str, Any] | None = None
        try:
            if job.requires_environment_slot:
                broker = get_slot_broker()
                if broker is None:
                    raise RuntimeError(
                        "rollout requires an environment slot but no broker is configured"
                    )
                environment_lease = broker.acquire(
                    owner=active_worker_id,
                    job_id=job.job_id,
                    timeout_s=environment_slot_acquire_timeout_s,
                    progress_callback=claim_heartbeat,
                )
            callbacks: list[Callable[[], None]] = [claim_heartbeat]
            if environment_lease is not None:
                callbacks.append(_EnvironmentLeaseHeartbeat(environment_lease))

            def execute_episode(callback: Callable[[], None]) -> dict[str, Any]:
                if environment_lease is None:
                    return executor.run(job, progress_callback=callback)
                return executor.run(
                    job,
                    progress_callback=callback,
                    environment_endpoint=environment_lease.endpoint,
                )

            runtime = get_provider_runtime() if job.requires_api else None
            if runtime is not None:
                claim_heartbeat()
                with runtime.admission.acquire(
                    runtime.route_id,
                    owner=f"{host}:{job.logical_id}",
                    timeout_s=runtime.acquire_timeout_s,
                ) as lease:
                    claim_heartbeat()
                    heartbeat = _CombinedHeartbeat(
                        *callbacks,
                        _ProviderLeaseHeartbeat(
                            lease,
                            interval_s=runtime.heartbeat_s,
                        ),
                    )
                    result = execute_episode(heartbeat)
                    if (
                        result.get("watchdog_reason")
                        != "provider_admission_heartbeat_failed"
                    ):
                        _report_provider_outcome(lease, result)
            else:
                result = execute_episode(_CombinedHeartbeat(*callbacks))
        except Exception as exc:
            result = {
                "job_id": job.job_id,
                "logical_id": job.logical_id,
                "attempt_index": job.attempt_index,
                "success": False,
                "worker_exception": type(exc).__name__,
                "job": job.as_dict(),
            }
        finally:
            if environment_lease is not None:
                environment_lease.release(
                    outcome=(
                        "completed"
                        if result is not None and result.get("success")
                        else "worker_failed"
                    )
                )
        assert result is not None
        queue.finish(
            claimed_path,
            success=bool(result.get("success")),
            result=result,
            claim_token=claim_token,
        )
        if once:
            return 0 if result.get("success") else 1


def run_worker_pool(
    *,
    queue_root: str | Path,
    host: str,
    concurrency: int,
    poll_s: float = 2.0,
    stop_event: threading.Event | None = None,
    environment_slot_broker_root: str | Path | None = None,
    environment_ready_manifest: str | Path | None = None,
    maximum_active_environment_slots: int | None = None,
) -> int:
    """Run a fair, work-conserving host dispatcher with independent workers."""

    if concurrency < 1:
        raise ValueError("worker pool concurrency must be positive")
    stopping = stop_event or threading.Event()
    failures: list[BaseException] = []
    failure_lock = threading.Lock()

    def worker(index: int) -> None:
        try:
            run_worker(
                queue_root=queue_root,
                host=host,
                poll_s=poll_s,
                once=False,
                environment_slot_broker_root=environment_slot_broker_root,
                environment_ready_manifest=environment_ready_manifest,
                maximum_active_environment_slots=maximum_active_environment_slots,
                worker_id=f"{host}:pool:{os.getpid()}:{index}",
                stop_event=stopping,
            )
        except BaseException as exc:  # pragma: no cover - surfaced to controller
            with failure_lock:
                failures.append(exc)
            stopping.set()

    threads = [
        threading.Thread(
            target=worker,
            args=(index,),
            name=f"rollout-worker-{index:02d}",
        )
        for index in range(concurrency)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if failures:
        raise RuntimeError(
            f"worker pool stopped after {len(failures)} dispatcher failure(s)"
        ) from failures[0]
    return 0


def round_robin_assign(
    jobs_by_task: dict[str, list[RolloutJob]], hosts: tuple[str, ...]
) -> list[tuple[str, RolloutJob]]:
    """Fairly interleave tasks, then spread jobs over host capacities."""

    if not hosts:
        raise ValueError("at least one worker host is required")
    queues = {task: list(jobs) for task, jobs in sorted(jobs_by_task.items())}
    assignments: list[tuple[str, RolloutJob]] = []
    host_index = 0
    while any(queues.values()):
        for task in sorted(queues):
            if not queues[task]:
                continue
            job = queues[task].pop(0)
            assignments.append((hosts[host_index % len(hosts)], job))
            host_index += 1
    return assignments
