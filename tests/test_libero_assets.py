# Copyright (c) 2026 RPent Contributors
from __future__ import annotations

import os
import sys
import types
from pathlib import Path

import pytest

from robots.libero.assets import bind_libero_assets_root


@pytest.mark.parametrize(
    "direct_composite",
    [
        False,
        pytest.param(
            True,
            marks=pytest.mark.skipif(
                os.name == "nt", reason="Windows CI cannot create symlinks"
            ),
        ),
    ],
)
def test_bind_assets_rewrites_preimported_path_completion_alias(
    tmp_path: Path, monkeypatch, direct_composite: bool
) -> None:
    assets_path = tmp_path if direct_composite else tmp_path / "assets"
    assets_path.mkdir(exist_ok=True)
    robosuite = types.ModuleType("robosuite")
    models = types.ModuleType("robosuite.models")
    utils = types.ModuleType("robosuite.utils")
    mjcf_utils = types.ModuleType("robosuite.utils.mjcf_utils")

    models.assets_root = "/package/default/assets"

    def original_completion(xml_path: str) -> str:
        return f"{models.assets_root}/{xml_path}"

    mjcf_utils.xml_path_completion = original_completion
    robosuite.models = models
    robosuite.utils = utils
    utils.mjcf_utils = mjcf_utils
    imported_alias = types.ModuleType("liberopro.fake_arena")
    imported_alias.xml_path_completion = original_completion
    liberopro = types.ModuleType("liberopro")
    liberopro_package = types.ModuleType("liberopro.liberopro")
    liberopro_envs = types.ModuleType("liberopro.liberopro.envs")
    bddl_base_domain = types.ModuleType("liberopro.liberopro.envs.bddl_base_domain")
    bddl_base_domain.DIR_PATH = "/package/default/liberopro/envs"
    articulated_objects = types.ModuleType(
        "liberopro.liberopro.envs.objects.articulated_objects"
    )
    articulated_objects.absolute_path = Path("/package/default/liberopro")
    liberopro.liberopro = liberopro_package
    liberopro_package.envs = liberopro_envs
    liberopro_envs.bddl_base_domain = bddl_base_domain

    for name, module in (
        ("robosuite", robosuite),
        ("robosuite.models", models),
        ("robosuite.utils", utils),
        ("robosuite.utils.mjcf_utils", mjcf_utils),
        ("liberopro.fake_arena", imported_alias),
        ("liberopro", liberopro),
        ("liberopro.liberopro", liberopro_package),
        ("liberopro.liberopro.envs", liberopro_envs),
        ("liberopro.liberopro.envs.bddl_base_domain", bddl_base_domain),
        (
            "liberopro.liberopro.envs.objects.articulated_objects",
            articulated_objects,
        ),
    ):
        monkeypatch.setitem(sys.modules, name, module)

    bound = bind_libero_assets_root(assets_path)
    models.assets_root = "/package/reset/after-import"

    expected = str(assets_path.absolute() / "scenes/example.xml")
    legacy_root = Path(bddl_base_domain.DIR_PATH).parent
    assert bound == assets_path.absolute()
    assert mjcf_utils.xml_path_completion("scenes/example.xml") == expected
    assert imported_alias.xml_path_completion("scenes/example.xml") == expected
    assert (legacy_root / "assets").resolve() == assets_path.resolve()
    assert articulated_objects.absolute_path == legacy_root
