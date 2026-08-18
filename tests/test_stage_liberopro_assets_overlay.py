# Copyright (c) 2026 RPent Contributors
from pathlib import Path

import pytest

from scripts.evolution.stage_liberopro_assets_overlay import stage_overlay


def _fixture(tmp_path: Path) -> tuple[Path, Path]:
    package = tmp_path / "site-packages" / "liberopro"
    (package / "liberopro").mkdir(parents=True)
    (package / "liberopro" / "__init__.py").write_text("", encoding="utf-8")
    assets = tmp_path / "assets"
    scene = assets / "scenes" / "libero_tabletop_base_style.xml"
    scene.parent.mkdir(parents=True)
    scene.write_text("<mujoco/>\n", encoding="utf-8")
    return package, assets


def test_stage_overlay_leaves_shared_package_unchanged(tmp_path: Path) -> None:
    package, assets = _fixture(tmp_path)
    output = tmp_path / "overlay"

    report = stage_overlay(
        installed_package=package,
        assets_root=assets,
        output_root=output,
    )

    linked_scene = (
        output
        / "liberopro"
        / "liberopro"
        / "assets"
        / "scenes"
        / "libero_tabletop_base_style.xml"
    )
    assert linked_scene.read_text(encoding="utf-8") == "<mujoco/>\n"
    assert not (package / "liberopro" / "assets").exists()
    assert Path(report["manifest"]).is_file()


def test_stage_overlay_requires_complete_assets(tmp_path: Path) -> None:
    package, assets = _fixture(tmp_path)
    (assets / "scenes" / "libero_tabletop_base_style.xml").unlink()

    with pytest.raises(ValueError, match="missing scenes/libero_tabletop"):
        stage_overlay(
            installed_package=package,
            assets_root=assets,
            output_root=tmp_path / "overlay",
        )
