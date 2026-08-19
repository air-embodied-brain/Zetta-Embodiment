# Copyright (c) 2026 Zetta Contributors
from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCAN_ROOTS = (
    "deployment",
    "docs",
    "ops",
    "robots",
    "zetta",
    "scripts",
)
TEXT_SUFFIXES = {".json", ".md", ".py", ".service", ".sh", ".toml", ".yaml", ".yml"}
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
    files = [ROOT / "README.md", ROOT / "README.zh-CN.md", ROOT / "pyproject.toml"]
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
        text = path.read_text(encoding="utf-8")
        if FORBIDDEN_LEGACY_TOKEN.search(text):
            offenders.append(path.relative_to(ROOT).as_posix())
    assert offenders == []


def test_readmes_are_bilingual_complete_and_include_a_real_campaign() -> None:
    english = (ROOT / "README.md").read_text(encoding="utf-8")
    chinese = (ROOT / "README.zh-CN.md").read_text(encoding="utf-8")
    for readme in (english, chinese):
        assert "[English](README.md)" in readme
        assert "[简体中文](README.zh-CN.md)" in readme
        assert 'python -m pip install -e ".[test]"' in readme
        assert "tests/test_evolution_protocol.py" in readme
        assert "| Long-T | `libero_10_task` | 520 | 10 | 530 |" in readme
        assert "| Goal-S | `libero_goal_swap` | 300 | 10 | 310 |" in readme
        assert "scripts/evolution/prepare_libero_campaign.py" in readme
        assert "scripts/evolution/run_campaign.py" in readme
        assert "--worker-command" in readme
        assert "Open the top layer of the drawer and put the bowl inside" in readme
        assert "<authoritative task language>" not in readme
        assert "LIBERO_CONFIG_PATH" not in readme
