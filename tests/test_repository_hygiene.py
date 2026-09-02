# Copyright (c) 2026 Zetta Contributors
from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
SCAN_ROOTS = (
    "deployment",
    "docs",
    "ops",
    "robots",
    "zetta",
    "scripts",
)
DOCUMENTATION_FILES = (
    ROOT / "README.md",
    ROOT / "scripts" / "deployment" / "VLA_ENV_SETUP.md",
    *(ROOT / "robots" / "libero" / "guides").glob("*.md"),
)
TEXT_SUFFIXES = {".json", ".md", ".py", ".service", ".sh", ".toml", ".yaml", ".yml"}
LEGACY_IDENTITY_TEXT_ALLOWLIST = {"THIRD_PARTY_NOTICES.md"}
ALLOWED_LEGACY_IDENTITY_LITERALS = {"rpent-liberopro", "rpent_liberopro"}
FORBIDDEN_LEGACY_TOKEN = re.compile(
    "|".join(
        (
            "".join(("r", "p", "ent")),
            "".join(("r", "-", "p", "ent")),
            "".join(("@", "r", "p", "ent")),
        )
    ),
    re.IGNORECASE,
)
MARKDOWN_LINK = re.compile(r"!?\[[^]]*\]\((?P<target>[^)\s]+)(?:\s+[^)]*)?\)")

PRIVATE_PATH_PATTERNS = (
    re.compile(r"[A-Za-z]:[\\/]+Users[\\/]+[^\\/\s]+", re.IGNORECASE),
    re.compile(r"/home/[A-Za-z0-9._-]+"),
    re.compile(r"/data[0-9]*/[A-Za-z0-9._-]+"),
    # Generic personal-username segment under /mnt, e.g. /mnt/ssd_data/<user>/...
    # or /mnt/<user>/..., without hardcoding any specific real employee names.
    re.compile(r"/mnt/(?:[A-Za-z0-9._-]+/)?[a-z][a-z0-9]{2,}(?:/|\b)", re.IGNORECASE),
    re.compile(r"\ba100_[A-Za-z0-9]+\b", re.IGNORECASE),
)
SECRET_PATTERNS = (
    re.compile(r"\bsk-[A-Za-z0-9_-]{16,}\b"),
    re.compile(
        r"\b(?:api[_-]?key|authorization|token)\s*[:=]\s*"
        r"['\"](?![<$])[A-Za-z0-9._-]{12,}['\"]",
        re.IGNORECASE,
    ),
)


def _repository_text_files() -> list[Path]:
    files = [ROOT / "README.md", ROOT / "pyproject.toml"]
    for relative_root in SCAN_ROOTS:
        root = ROOT / relative_root
        if not root.is_dir():
            continue
        files.extend(path for path in root.rglob("*") if _is_text_file(path))
    return sorted(set(files))


def _is_text_file(path: Path) -> bool:
    return path.is_file() and path.suffix in TEXT_SUFFIXES


def _repository_paths() -> list[Path]:
    ignored = {".git", "__pycache__", ".pytest_cache", ".ruff_cache"}
    paths: list[Path] = []
    for path in ROOT.rglob("*"):
        relative = path.relative_to(ROOT)
        if any(part in ignored for part in relative.parts):
            continue
        paths.append(path)
    return sorted(paths)


def _matches(patterns: tuple[re.Pattern[str], ...]) -> list[str]:
    matches: list[str] = []
    for path in _repository_text_files():
        text = path.read_text(encoding="utf-8")
        for pattern in patterns:
            if pattern.search(text):
                matches.append(f"{path.relative_to(ROOT)}: {pattern.pattern}")
    return matches


def _contains_forbidden_legacy_identity(text: str) -> bool:
    for literal in ALLOWED_LEGACY_IDENTITY_LITERALS:
        text = re.sub(re.escape(literal), "", text, flags=re.IGNORECASE)
    return FORBIDDEN_LEGACY_TOKEN.search(text) is not None


def test_tracked_runtime_and_docs_have_no_private_machine_paths() -> None:
    assert _matches(PRIVATE_PATH_PATTERNS) == []


def test_tracked_runtime_and_docs_have_no_literal_credentials() -> None:
    assert _matches(SECRET_PATTERNS) == []


def test_repository_paths_do_not_expose_legacy_identity() -> None:
    offenders = [
        path.relative_to(ROOT).as_posix()
        for path in _repository_paths()
        if FORBIDDEN_LEGACY_TOKEN.search(path.relative_to(ROOT).as_posix())
    ]
    assert offenders == []


def test_repository_text_does_not_expose_legacy_identity() -> None:
    offenders: list[str] = []
    for path in _repository_paths():
        if not _is_text_file(path):
            continue
        relative = path.relative_to(ROOT).as_posix()
        if relative in LEGACY_IDENTITY_TEXT_ALLOWLIST:
            continue
        text = path.read_text(encoding="utf-8")
        if _contains_forbidden_legacy_identity(text):
            offenders.append(relative)
    assert offenders == []


def test_third_party_notice_preserves_required_legacy_attribution() -> None:
    notice = (ROOT / "THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8")
    legacy_name = "".join(("R", "Pent"))

    assert legacy_name in notice


def test_required_external_distribution_names_are_the_only_inline_exceptions() -> None:
    assert not _contains_forbidden_legacy_identity("rpent-liberopro==0.1.1")
    assert not _contains_forbidden_legacy_identity(
        "rpent_liberopro-0.1.1-py3-none-any.whl"
    )
    unrelated_legacy_name = "-".join(("r", "pent", "unrelated"))
    assert _contains_forbidden_legacy_identity(unrelated_legacy_name)


def test_readme_is_complete_and_includes_a_real_campaign() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    required_contracts = (
        'python -m pip install -e ".[test]"',
        "tests/test_evolution_protocol.py",
        "scripts/evolution/prepare_libero_campaign.py",
        "scripts/evolution/run_campaign.py",
        "--worker-command",
        "Open the top layer of the drawer and put the bowl inside",
    )
    missing = [contract for contract in required_contracts if contract not in readme]

    assert missing == []
    assert "<authoritative task language>" not in readme
    assert "LIBERO_CONFIG_PATH" not in readme


def test_documentation_local_links_exist() -> None:
    missing: list[str] = []
    repository_root = ROOT.resolve()
    for source in DOCUMENTATION_FILES:
        text = source.read_text(encoding="utf-8")
        for match in MARKDOWN_LINK.finditer(text):
            raw_target = match.group("target").strip("<>")
            parsed = urlsplit(raw_target)
            if parsed.scheme or raw_target.startswith("#"):
                continue
            target = (source.parent / unquote(parsed.path)).resolve()
            if not target.is_relative_to(repository_root) or not target.exists():
                missing.append(f"{source.relative_to(ROOT).as_posix()}: {raw_target}")

    assert missing == []


def test_liberopro_version_is_consistent_across_setup_surfaces() -> None:
    installer = (ROOT / "scripts/deployment/install_vla_env.sh").read_text(
        encoding="utf-8"
    )
    dockerfile = (ROOT / "scripts/deployment/Dockerfile.vla-env").read_text(
        encoding="utf-8"
    )
    setup_guide = (ROOT / "scripts/deployment/VLA_ENV_SETUP.md").read_text(
        encoding="utf-8"
    )
    pro_guide = (ROOT / "robots/libero/guides/pro_hybrid_guide.md").read_text(
        encoding="utf-8"
    )

    patterns = {
        "install_vla_env.sh": (
            r'LIBEROPRO_PACKAGE="\$\{LIBEROPRO_PACKAGE:-rpent-liberopro==([^}"]+)\}"',
            installer,
        ),
        "Dockerfile.vla-env": (
            r"^ARG LIBEROPRO_VERSION=([^\s]+)$",
            dockerfile,
        ),
        "VLA_ENV_SETUP.md": (
            r"rpent-liberopro==([0-9][A-Za-z0-9.+-]*)",
            setup_guide,
        ),
        "pro_hybrid_guide.md": (
            r"rpent-liberopro==([0-9][A-Za-z0-9.+-]*)",
            pro_guide,
        ),
    }
    versions: dict[str, str] = {}
    for name, (pattern, text) in patterns.items():
        match = re.search(pattern, text, re.MULTILINE)
        assert match is not None, f"could not find the LIBERO-Pro version in {name}"
        versions[name] = match.group(1)

    assert len(set(versions.values())) == 1, versions


def test_openpi_rlinf_guard_allows_only_required_distributions() -> None:
    expected = {"rlinf-openpi", "rlinf-transformer-openpi"}
    surfaces = (
        ROOT / "scripts/deployment/install_vla_env.sh",
        ROOT / "scripts/deployment/Dockerfile.vla-env",
    )

    for surface in surfaces:
        text = surface.read_text(encoding="utf-8")
        match = re.search(r"allowed\s*=\s*\{(?P<items>[^}]+)\}", text)
        assert match is not None, (
            f"could not find the RLinf allowlist in {surface.name}"
        )
        actual = set(re.findall(r"['\"]([^'\"]+)['\"]", match.group("items")))
        assert actual == expected, surface.name
