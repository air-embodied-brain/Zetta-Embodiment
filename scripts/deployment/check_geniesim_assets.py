# Copyright (c) 2026 Zetta Contributors
"""Resolve the USD dependency closure of the pinned Genie pick-block profile.

Run with usd-core in the separate asset-preparation environment. Isaac's built-in
MDL modules are resolved by Kit at runtime; other unresolved assets fail.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ENTRY_POINTS = (
    "robot/G2_omnipicker/robot_fix.usda",
    "background/home/home_b_aligned/background.usda",
    "zetta_tasks/pick_block_color/0/scene.usda",
)
KIT_MODULES = frozenset({"OmniPBR.mdl", "OmniGlass.mdl"})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets-root", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    root = args.assets_root.resolve()

    from pxr import UsdUtils

    report = {}
    for name in ENTRY_POINTS:
        entry = root / name
        if not entry.is_file():
            parser.error(f"required entry point missing: {entry}")
        layers, assets, unresolved = UsdUtils.ComputeAllDependencies(str(entry))
        missing = sorted(set(unresolved) - KIT_MODULES)
        report[name] = {
            "layer_count": len(layers),
            "asset_count": len(assets),
            "kit_modules": sorted(set(unresolved) & KIT_MODULES),
            "missing": missing,
        }
        print(json.dumps({"entry": name, **report[name]}), flush=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if any(item["missing"] for item in report.values()):
        raise SystemExit("USD dependency validation failed; see the report")


if __name__ == "__main__":
    main()
