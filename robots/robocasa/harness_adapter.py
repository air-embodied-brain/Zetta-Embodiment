# Copyright (c) 2026 Zetta Contributors
"""Fail-closed adapter for an external RoboCasa harness tool registry.

The harness remains a separately versioned runtime.  Zetta imports it only
when an explicit root is supplied, verifies that the imported modules came
from that root, and exposes only tools already present in Zetta's audited
RoboCasa catalog.  The adapter never owns or steps the simulator.
"""

from __future__ import annotations

import hashlib
import importlib
import os
import sys
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from robots.robocasa.tool_catalog import DEFAULT_ROBOCASA_TOOL_CATALOG
from robots.robocasa.tool_runtime import (
    InvocationPolicy,
    ToolContractError,
    ToolPolicyError,
    ToolRuntime,
)
from zetta.evolution.jsonio import canonical_sha256


class HarnessRuntimeUnavailable(RuntimeError):
    """The requested harness snapshot cannot be loaded safely."""


def _file_sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _module_belongs_to(module: Any, root: Path) -> bool:
    source = getattr(module, "__file__", None)
    if not source:
        return False
    try:
        Path(source).resolve().relative_to(root)
    except (OSError, ValueError):
        return False
    return True


class HarnessToolRuntimeAdapter:
    """Translate Zetta invocation policy into the harness registry policy."""

    def __init__(
        self,
        registry: Any,
        execution_policy_type: type[Any],
        *,
        root: str | Path,
        fallback_runtime: ToolRuntime | None = None,
    ) -> None:
        self.registry = registry
        self.execution_policy_type = execution_policy_type
        self.root = Path(root).expanduser().resolve()
        self.fallback_runtime = fallback_runtime
        names = registry.names()
        self._names = frozenset(str(name) for name in names)
        self._required_names: frozenset[str] = frozenset()
        self._fallback_names: frozenset[str] = frozenset()
        self._manifest_digest = canonical_sha256(registry.manifest())

    @classmethod
    def from_root(cls, root: str | Path | None = None) -> "HarnessToolRuntimeAdapter":
        configured = root or os.environ.get("ZETTA_ROBOCASA_HARNESS_ROOT")
        if not configured:
            raise HarnessRuntimeUnavailable(
                "harness backend requires --harness-root or "
                "ZETTA_ROBOCASA_HARNESS_ROOT"
            )
        resolved = Path(configured).expanduser().resolve()
        scripts = resolved / "skills" / "evolving-skill-library" / "scripts"
        required = (
            scripts / "evolving_skills" / "robocasa_tool_library.py",
            scripts / "evolving_skills" / "tool_registry.py",
            resolved / "robot_tools" / "runtime_services.json",
        )
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise HarnessRuntimeUnavailable(
                f"harness snapshot is incomplete: {missing}"
            )

        for path in (resolved, scripts):
            value = str(path)
            if value not in sys.path:
                sys.path.insert(0, value)

        try:
            library_module = importlib.import_module(
                "evolving_skills.robocasa_tool_library"
            )
            registry_module = importlib.import_module("evolving_skills.tool_registry")
        except Exception as exc:
            raise HarnessRuntimeUnavailable(
                f"cannot import harness registry: {type(exc).__name__}: {exc}"
            ) from exc
        if not _module_belongs_to(library_module, resolved) or not _module_belongs_to(
            registry_module, resolved
        ):
            raise HarnessRuntimeUnavailable(
                "evolving_skills was already imported from a different harness root"
            )

        try:
            config = library_module.RoboCasaToolLibraryConfig(project_root=resolved)
            registry = library_module.build_robocasa_tool_library(config)
        except Exception as exc:
            raise HarnessRuntimeUnavailable(
                f"cannot build harness registry: {type(exc).__name__}: {exc}"
            ) from exc
        return cls(
            registry,
            registry_module.ToolExecutionPolicy,
            root=resolved,
            fallback_runtime=ToolRuntime(),
        )

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._names))

    def require_tools(self, names: Iterable[str]) -> None:
        required = {str(name) for name in names}
        missing = sorted(required - self._names)
        if missing and self.fallback_runtime is None:
            raise HarnessRuntimeUnavailable(
                f"harness registry is missing required Zetta tools: {missing}"
            )
        unknown = sorted(
            name
            for name in missing
            if name not in DEFAULT_ROBOCASA_TOOL_CATALOG.names()
        )
        if unknown:
            raise HarnessRuntimeUnavailable(
                f"required tools have no audited builtin fallback: {unknown}"
            )
        self._required_names = frozenset(required)
        self._fallback_names = frozenset(missing)

    def describe(self) -> dict[str, Any]:
        services = self.root / "robot_tools" / "runtime_services.json"
        return {
            "backend": "harness_registry",
            "root": str(self.root),
            "tool_count": len(self._names),
            "tool_names": list(self.names),
            "required_tool_names": sorted(self._required_names),
            "builtin_fallback_tool_names": sorted(self._fallback_names),
            "manifest_sha256": self._manifest_digest,
            "runtime_services_sha256": _file_sha256(services),
        }

    def invoke(
        self,
        name: str,
        payload: Mapping[str, Any],
        *,
        policy: InvocationPolicy | None = None,
    ) -> dict[str, Any]:
        if not isinstance(payload, Mapping):
            raise ToolContractError("tool payload must be an object")
        policy = policy or InvocationPolicy()
        if policy.allow is not None and name not in policy.allow:
            raise ToolPolicyError("tool is denied by the invocation policy")
        try:
            audited_spec = DEFAULT_ROBOCASA_TOOL_CATALOG.get(name)
        except KeyError as exc:
            raise ToolContractError(
                "harness tool is not present in the audited Zetta catalog"
            ) from exc
        if name not in self._names:
            if self.fallback_runtime is None:
                raise ToolContractError("tool is unavailable in the harness snapshot")
            result = self.fallback_runtime.invoke(name, payload, policy=policy)
            return {
                **result,
                "runtime_backend": "builtin_fallback",
                "harness_registry_available": False,
            }
        if audited_spec.privileged and not policy.allow_privileged:
            raise ToolPolicyError(
                "privileged harness tool requires explicit authorization"
            )

        allowed = frozenset(policy.allow) if policy.allow is not None else frozenset({name})
        harness_policy = self.execution_policy_type(
            allowed_tools=allowed,
            max_risk_level="critical",
        )
        result = self.registry.invoke(name, dict(payload), policy=harness_policy)
        if not isinstance(result, Mapping):
            raise ToolContractError("harness tool returned a non-object result")
        return {
            **dict(result),
            "tool": name,
            "proposal_only": audited_spec.proposal_only,
            "environment_write": False,
            "runtime_backend": "harness_registry",
            "privileged_audit": {
                "authorized": bool(policy.allow_privileged),
                "tool_declared_privileged": bool(audited_spec.privileged),
                "source": "harness_registry_policy",
            },
        }


__all__ = [
    "HarnessRuntimeUnavailable",
    "HarnessToolRuntimeAdapter",
]
