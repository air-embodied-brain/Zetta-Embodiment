# Copyright (c) 2026 Zetta Contributors
"""Stage a task-local LIBERO-Pro package overlay with external assets.

Some runtime wheels contain the ``liberopro`` Python code but omit the
``liberopro/liberopro/assets`` tree. This helper copies only the installed
package into an isolated output root and attaches a caller-supplied assets
directory. It never edits the shared site-packages installation.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

REQUIRED_SCENE = Path("scenes/libero_tabletop_base_style.xml")


def stage_overlay(
    *,
    installed_package: Path,
    assets_root: Path,
    output_root: Path,
) -> dict[str, str]:
    installed_package = installed_package.expanduser().resolve()
    assets_root = assets_root.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    if installed_package.name != "liberopro":
        raise ValueError("installed package path must end in 'liberopro'")
    if not (installed_package / "liberopro" / "__init__.py").is_file():
        raise ValueError("installed package is missing liberopro/__init__.py")
    required_scene = assets_root / REQUIRED_SCENE
    if not required_scene.is_file():
        raise ValueError(f"assets root is missing {REQUIRED_SCENE.as_posix()}")
    if output_root.exists():
        raise FileExistsError(f"overlay output already exists: {output_root}")

    package_output = output_root / "liberopro"
    shutil.copytree(installed_package, package_output, symlinks=True)
    nested_assets = package_output / "liberopro" / "assets"
    if nested_assets.exists() or nested_assets.is_symlink():
        raise ValueError("installed package unexpectedly contains an assets tree")
    try:
        nested_assets.symlink_to(assets_root, target_is_directory=True)
        link_mode = "symlink"
    except OSError:
        shutil.copytree(assets_root, nested_assets, symlinks=True)
        link_mode = "copy"

    manifest = {
        "schema_version": 1,
        "installed_package": str(installed_package),
        "assets_root": str(assets_root),
        "package_output": str(package_output),
        "required_scene": str(required_scene),
        "link_mode": link_mode,
    }
    manifest_path = output_root / "overlay-manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    return {**manifest, "manifest": str(manifest_path)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--site-packages",
        type=Path,
        required=True,
        help="site-packages directory containing the installed liberopro package",
    )
    parser.add_argument("--assets-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    report = stage_overlay(
        installed_package=args.site_packages / "liberopro",
        assets_root=args.assets_root,
        output_root=args.output_root,
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
