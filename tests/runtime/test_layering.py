# Copyright (c) 2026 Zetta Contributors
"""Layered import guard (five rules governing module boundaries).

Uses AST scanning of import nodes without executing the module — this way the
guard still applies even when a module is missing a local dependency (rlinf /
mujoco).

Five rules:

1. No file under ``rollout_runtime/**`` may have a top-level import of
   ``zetta`` / ``robots``;
2. ``api/**`` only allows stdlib (the sole exception: ``api/wire.py`` may
   import ``msgpack``, since the design requires msgpack serialization);
3. ``gateway/**`` must not import ``ray`` / ``rlinf`` / ``torch``;
4. ``core/**`` must not import ``ray`` / ``rlinf``;
5. Reverse direction: ``rollout_runtime`` being imported by ``zetta`` /
   ``robots`` is only allowed in ``robots/libero/__init__.py`` and
   ``robots/robocasa/__init__.py`` (the single injection point for each
   family).
"""

from __future__ import annotations

import ast
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNTIME_ROOT = REPO_ROOT / "rollout_runtime"

API_THIRD_PARTY_ALLOWLIST: dict[str, frozenset[str]] = {
    "wire.py": frozenset({"msgpack"}),
}
"""Third-party import allowlist for ``api/**``: filename → allowed top-level modules.

Only ``api/wire.py`` has an exception, because the serialization design uses
msgpack for the first version. All other api modules stay pure stdlib, so the
control-plane logic and digest computation do not depend on any third-party
library.
"""

RUNTIME_INJECTION_POINTS = frozenset(
    {
        "robots/libero/__init__.py",
        "robots/libero/run_evolution_rollout.py",
        "robots/robocasa/run_rollout.py",
    }
)
"""The only files on the zetta / robots side allowed to ``import rollout_runtime``.

- ``robots/libero/__init__.py``: the libero family's ``--runtime rollout`` CLI
  switch and ``_init_rollout_runtime``, which does not change the
  ``--runtime legacy`` default path.
- ``robots/robocasa/run_rollout.py``: the sole integration point for runtime
  v3. The robocasa side has **no** legacy/rollout dual-path option: direct
  calls to ``RoboCasaEnvClient``/``Gr00tClient`` have been fully replaced, and
  ``RemoteRuntimeClient`` -> Gateway -> Ray Worker is the only path. The
  injection point therefore moved from ``robots/robocasa/__init__.py`` (an
  earlier CLI switch that the current branch no longer contains any
  rollout_runtime import for) to ``run_rollout.py`` itself.

New entries must also explain why they are a deliberate, application-initiated
injection point rather than accidental coupling."""

RUNTIME_TOOLING_FILES = frozenset(
    {
        "scripts/deployment/runtime_parity_trace.py",
        "scripts/deployment/runtime_transport_spike.py",
        "scripts/deployment/m6_acceptance/rr_eval_bench.py",
        "scripts/deployment/m6_acceptance/rr_probe_libero_vec.py",
        "scripts/deployment/m6_acceptance/rr_probe_robocasa.py",
        "scripts/deployment/m7_acceptance/rr_serve_probe.py",
        "scripts/deployment/m7_acceptance/rr_lost_probe.py",
        "scripts/deployment/m7_acceptance/rr_eval_batching.py",
        "scripts/deployment/m7_acceptance/rr_serve_overhead.py",
        "scripts/deployment/smoke_cosmos_lite.py",
        "scripts/experiments/run_ab_runtime.py",
        "scripts/experiments/run_concurrency_ab.py",
        "scripts/experiments/libero_critic_recovery_latency_v3.py",
    }
)
"""Tooling scripts under ``scripts/`` that belong to Rollout Runtime itself and are
not subject to rule 5.

Rule 5 exists to block "the legacy zetta / robots path silently depending on
runtime" (the default path must have zero changes). Every entry in this
allowlist is a tool belonging to runtime itself (in the same category as
``runtime_ci.sh``), and of course they import ``rollout_runtime``:

- ``runtime_transport_spike.py``: a quantitative baseline tool;
- ``runtime_parity_trace.py``: a single-arm data-collection tool for parity
  checks (each arm runs its own process; see the module docstring of
  ``test_legacy_parity.py``);
- ``run_ab_runtime.py``: an A/B driver that needs to freeze the identity of
  the runtime preset (topology / policy backend / ``chunk_size`` / digest)
  into the report, otherwise the report would only say "rollout arm" without
  specifying which one. It is **not** on the legacy execution path: both arms
  are ``zetta`` subprocesses it forks, and the legacy subprocess's
  ``--runtime legacy`` path never executes a single line of runtime code.
- ``m6_acceptance/rr_eval_bench.py`` / ``rr_probe_libero_vec.py`` /
  ``rr_probe_robocasa.py``: scripts used to collect numbers on a GPU host
  (moved into the repository per an independent audit requirement, since
  otherwise those numbers would not be reproducible). All three only import
  ``rollout_runtime`` and are unrelated to the legacy path; the ``README.md``
  in the same directory explains how to run them.
- ``m7_acceptance/rr_serve_probe.py``: an acceptance probe for the served
  form (auth negative cases + ``application_id`` attribution + ``/metrics``
  scraping), using only ``api.wire`` to encode/decode HTTP bodies.
- ``m7_acceptance/rr_lost_probe.py``: a live driver for the scenario of
  killing an EnvWorker rank. It must be able to ``ray.kill`` an env actor, so
  it is a runtime tool unrelated to the legacy path.
- ``m7_acceptance/rr_eval_batching.py``: batched re-sampling evaluation, a
  same-format counterpart to ``rr_eval_bench.py`` plus batching parameters and
  a forward-pass counter.
- ``m7_acceptance/rr_serve_overhead.py``: a ``policy_step`` micro-benchmark
  across the embedded / ASGI / TCP arms, providing the per-round-trip cost
  (required in the repository per an independent audit requirement).
- ``smoke_cosmos_lite.py``: a deployment probe for the optional remote
  Cosmos-Lite policy backend; it exercises Runtime protocol conversion but
  does not enter any legacy robot path.
- ``run_concurrency_ab.py``: a **concurrent** A/B driver for legacy vs.
  rollout, where both arms share the same scripted workload definition (no
  planner). The rollout arm installs its own runtime (``launch/`` +
  ``config/``) and drives ``RuntimeClient`` directly, while the legacy arm
  only forks ``env_server`` / ``vla_server`` subprocesses and never executes a
  single line of runtime code — so it is in the same category as
  ``run_ab_runtime.py``: a runtime tool that is not on the legacy execution
  path.
- ``zetta_robocasa_seam_latency.py``: an independent latency probe for the
  robocasa seam (``RuntimeSeam`` + ``RuntimeEnvClient`` + ``RuntimePolicyClient``),
  connecting directly to ``rollout_runtime.adapters.zetta`` without going
  through ``zetta.cli.main`` or the ``robots/robocasa/__init__.py`` injection
  path, and therefore also not on the legacy execution path. It uses
  ``transport.kind=inproc`` to avoid the intermittent GCS heartbeat timeouts
  that occur with the ``ray_channel`` form on shared GPU machines (unrelated
  to the seam code itself).
- ``libero_critic_recovery_latency_v3.py``: a single-arm latency probe that
  semantically mirrors ``examples/libero_pi05_critic_recovery/run.py`` on the
  main side, driving ``build_local_components`` directly with
  ``asyncio.run()`` (``transport.kind=ray_channel``, real rlinf policy
  backend + libero family), and not going through any legacy/zetta execution
  path — each arm runs its own process, and the legacy subprocess likewise
  never executes a single line of runtime code (in the same category as
  ``run_ab_runtime.py``/``run_concurrency_ab.py``). It imports directly
  instead of going through the ``run_ab_runtime.py`` framework because this
  probe only measures the online latency of the rollout_runtime single arm
  under real LIBERO-Pro + Pi0.5 + Critic-Recovery, and does not need the
  fork/manifest mechanism shared by the A/B arms.

New entries must also explain why they belong to runtime tooling rather than
the legacy path.
"""

FORBIDDEN_IN_GATEWAY = frozenset({"ray", "rlinf", "torch"})
FORBIDDEN_IN_CORE = frozenset({"ray", "rlinf"})

RULE1_ALLOWED_ROBOTS_IMPORTS = frozenset(
    {
        "rollout_runtime/backends/robocasa_current.py",
        "rollout_runtime/backends/groot_policy.py",
    }
)
"""The only two files allowed to import ``robots`` under rule 1 (C1 independence).

Rule 1 exists to block accidental coupling caused by "legacy code silently
depending on runtime" (see the similar comment for rule 5), not to prevent
"runtime from having an official backend that reuses application business
logic." RoboCasaSession/Gr00tModelCore (``robots/robocasa/session_core.py``/
``groot_core.py``) are the **only** RoboCasa/GR00T business logic retained
under the hard constraints — unlike the libero/maniskill backend packages
which wrap third-party ``rlinf.envs.*``, these two backends can only wrap the
application's own implementation, otherwise it would either duplicate
business logic (violating the "keep only one copy" constraint) or simply be
unable to integrate. These two files are therefore the runtime's **sole,
deliberate** point of integration with application code — a mirror image of
``RUNTIME_INJECTION_POINTS`` (the application deliberately importing
runtime), not the kind of accidental reverse coupling rule 1 is meant to
prevent; the rest of ``rollout_runtime/**`` remains unaffected and must not
contain a single ``robots``/``zetta`` import.

New entries must also explain why this backend must wrap the application's
own business code rather than a third-party implementation.
"""

_STDLIB = frozenset(sys.stdlib_module_names)


def _python_files(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("*.py")
        if "__pycache__" not in path.parts and ".venv" not in str(path)
    )


def _top_level_imports(path: Path) -> Iterator[tuple[str, int]]:
    """Yield the top-level module name and line number for every import in a file.

    Args:
        path: The Python file to scan.

    Yields:
        ``(top-level module name, line number)``; relative imports
        (``from . import x``) are skipped.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name.split(".")[0], node.lineno
        elif isinstance(node, ast.ImportFrom):
            if node.level:  # relative import
                continue
            if node.module:
                yield node.module.split(".")[0], node.lineno


def _runtime_files(subpackage: str = "") -> list[Path]:
    root = RUNTIME_ROOT / subpackage if subpackage else RUNTIME_ROOT
    files = _python_files(root)
    assert files, f"no python files found under {root}"
    return files


def _violations(files: list[Path], forbidden: frozenset[str]) -> list[str]:
    found: list[str] = []
    for path in files:
        for module, lineno in _top_level_imports(path):
            if module in forbidden:
                found.append(
                    f"{path.relative_to(REPO_ROOT)}:{lineno} imports {module!r}"
                )
    return found


def test_runtime_package_exists() -> None:
    """``rollout_runtime`` must be a top-level package with all subpackages present."""
    assert (RUNTIME_ROOT / "__init__.py").is_file()
    expected = {
        "api",
        "core",
        "gateway",
        "workers",
        "transport",
        "backends",
        "adapters",
        "config",
        "launch",
        "serve",
    }
    actual = {
        path.name
        for path in RUNTIME_ROOT.iterdir()
        if path.is_dir() and (path / "__init__.py").is_file()
    }
    assert expected <= actual, f"missing subpackages: {sorted(expected - actual)}"
    assert (RUNTIME_ROOT / "cli.py").is_file()


def test_rule1_runtime_only_imports_robots_at_explicit_boundaries() -> None:
    """Rule 1: the runtime may reuse Zetta's own infrastructure; ``robots``
    imports are still restricted to the business boundaries listed in
    ``RULE1_ALLOWED_ROBOTS_IMPORTS`` (see that constant's comment).
    """
    violations: list[str] = []
    for path in _runtime_files():
        relative = path.relative_to(REPO_ROOT).as_posix()
        for module, lineno in _top_level_imports(path):
            if module == "robots" and relative not in RULE1_ALLOWED_ROBOTS_IMPORTS:
                violations.append(f"{relative}:{lineno} imports {module!r}")
    assert not violations, "unexpected robots dependency:\n" + "\n".join(violations)


def test_rule2_api_is_stdlib_only() -> None:
    """Rule 2: ``api/**`` only allows stdlib; any third-party import outside the
    allowlist fails."""
    violations: list[str] = []
    for path in _runtime_files("api"):
        allowed = API_THIRD_PARTY_ALLOWLIST.get(path.name, frozenset())
        for module, lineno in _top_level_imports(path):
            if module == "rollout_runtime" or module in _STDLIB or module in allowed:
                continue
            violations.append(
                f"{path.relative_to(REPO_ROOT)}:{lineno} imports non-stdlib {module!r}"
            )
    assert not violations, "api/ must stay stdlib-only:\n" + "\n".join(violations)


def test_rule3_gateway_has_no_ray_rlinf_torch() -> None:
    """Rule 3: ``gateway/**`` must not import ``ray`` / ``rlinf`` / ``torch``."""
    violations = _violations(_runtime_files("gateway"), FORBIDDEN_IN_GATEWAY)
    assert not violations, "gateway/ must not depend on the data plane:\n" + "\n".join(
        violations
    )


def test_rule3b_serve_has_no_ray_rlinf_torch() -> None:
    """Rule 3b (serve addendum): ``serve/**`` follows the same spec as ``gateway/**``.

    The served form is just the same ``RuntimeGateway`` wrapped in an HTTP
    layer; it only touches Runtime through ``RuntimeClient`` and never
    steps into the data plane. The allowed third-party packages are
    ``fastapi`` / ``uvicorn`` (they are already in the main
    ``dependencies``; this addendum also lists them under
    ``[project.optional-dependencies].runtime`` so the runtime extra is
    self-describing).
    """
    violations = _violations(_runtime_files("serve"), FORBIDDEN_IN_GATEWAY)
    assert not violations, "serve/ must not depend on the data plane:\n" + "\n".join(
        violations
    )


def test_rule4_core_has_no_ray_rlinf() -> None:
    """Rule 4: ``core/**`` must not import ``ray`` / ``rlinf``."""
    violations = _violations(_runtime_files("core"), FORBIDDEN_IN_CORE)
    assert not violations, "core/ must stay transport-agnostic:\n" + "\n".join(
        violations
    )


def test_rule5_single_injection_point_from_zetta_side() -> None:
    """Rule 5: only the single injection point on the zetta / robots side may
    import ``rollout_runtime``."""
    offenders: list[str] = []
    for package in ("zetta", "robots", "scripts"):
        root = REPO_ROOT / package
        if not root.is_dir():
            continue
        for path in _python_files(root):
            relative = path.relative_to(REPO_ROOT).as_posix()
            if relative in RUNTIME_TOOLING_FILES:
                continue
            for module, lineno in _top_level_imports(path):
                if module == "rollout_runtime" and relative not in (
                    RUNTIME_INJECTION_POINTS
                ):
                    offenders.append(f"{relative}:{lineno}")
    assert not offenders, (
        "rollout_runtime may only be imported from "
        f"{sorted(RUNTIME_INJECTION_POINTS)}; offenders: {offenders}"
    )


def test_stage7_robocasa_injection_point_is_run_rollout_and_nothing_else() -> None:
    """Guard: the robocasa-side runtime integration point is exactly ``run_rollout.py``.

    Rule 5 only blocks "extra injection points." This test pins down two
    equally important facts in the opposite direction:

    1. ``run_rollout.py`` **must** actually import ``rollout_runtime`` —
       otherwise it would mean someone reverted it to the old path that
       calls ``RoboCasaEnvClient``/``Gr00tClient`` directly (the hard
       constraint of a single execution path), and rule 5 alone cannot
       detect a "missing" injection point;
    2. The rest of ``robots/robocasa/**`` (including ``__init__.py``,
       ``env_server.py``, and other retained debugging shells) must not
       contain a single line of runtime code — the application side's
       runtime dependency surface is exactly one file wide.
    """
    robocasa_root = REPO_ROOT / "robots" / "robocasa"
    entrypoint = robocasa_root / "run_rollout.py"
    assert entrypoint.is_file()
    importers = {
        path.relative_to(REPO_ROOT).as_posix()
        for path in _python_files(robocasa_root)
        if any(module == "rollout_runtime" for module, _ in _top_level_imports(path))
    }
    assert importers == {"robots/robocasa/run_rollout.py"}, (
        "robocasa's only rollout_runtime entrypoint must be run_rollout.py; "
        f"found {sorted(importers)}"
    )


@pytest.mark.parametrize(
    "subpackage",
    ["api", "core", "gateway", "transport", "config"],
)
def test_no_wildcard_imports(subpackage: str) -> None:
    """Forbid ``from x import *``: it would degrade the precision of the AST guards
    above.

    Args:
        subpackage: The subpackage to check.
    """
    offenders: list[str] = []
    for path in _runtime_files(subpackage):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and any(
                alias.name == "*" for alias in node.names
            ):
                offenders.append(f"{path.relative_to(REPO_ROOT)}:{node.lineno}")
    assert not offenders, f"wildcard imports found: {offenders}"


def test_rule6_workers_never_touch_rlinf() -> None:
    """Rule 6 (workers addendum): ``workers/**`` neither imports rlinf / ray nor
    subclasses the rlinf ``Worker``.

    Design decision: ``RuntimeEnvWorker`` / ``RuntimeRolloutWorker`` are
    **plain classes** in v1; subclassing the rlinf ``Worker`` is only
    allowed inside the ``launch/ray_launch.py`` shell. The reason is that
    ``.venv-runtime`` does not have rlinf installed, so a top-level subclass
    would prevent even local inproc tests from importing the module. This was
    previously enforced only by convention; it is now enforced by this guard.
    """
    import_offenders = _violations(
        _runtime_files("workers"), frozenset({"ray", "rlinf"})
    )
    assert not import_offenders, (
        "workers/ must not import ray / rlinf (M3 keeps the rlinf Worker subclass in "
        "launch/ray_launch.py):\n" + "\n".join(import_offenders)
    )

    base_offenders: list[str] = []
    for path in _runtime_files("workers"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            for base in node.bases:
                name = (
                    base.attr
                    if isinstance(base, ast.Attribute)
                    else getattr(base, "id", "")
                )
                if name == "Worker":
                    base_offenders.append(
                        f"{path.relative_to(REPO_ROOT)}:{node.lineno} "
                        f"class {node.name} subclasses {name}"
                    )
    assert not base_offenders, (
        "rlinf Worker subclassing belongs in launch/ray_launch.py only:\n"
        + "\n".join(base_offenders)
    )


def test_rule6_runtime_never_imports_main_rlinf_package() -> None:
    """The runtime must not depend on the main ``rlinf`` package; OpenPI is
    accessed only through the standalone ``openpi`` package."""
    offenders: list[str] = []
    for path in _runtime_files():
        relative = path.relative_to(REPO_ROOT).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [alias.name.split(".")[0] for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module.split(".")[0]]
            if "rlinf" not in names:
                continue
            offenders.append(f"{relative}:{node.lineno}")
    assert not offenders, f"main rlinf package imports are forbidden: {offenders}"
