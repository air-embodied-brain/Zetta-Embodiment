#!/usr/bin/env python3
# Copyright (c) 2026 Zetta Contributors
"""Validate external BEHAVIOR runtime checkouts and write a local manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--omnigibson-root", type=Path, required=True)
    parser.add_argument("--bddl-root", type=Path, required=True)
    parser.add_argument("--assets-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("behavior-runtime.json"))
    return parser.parse_args()


def main() -> int:
    args = _parse()
    roots = {
        "omnigibson_root": args.omnigibson_root,
        "bddl_root": args.bddl_root,
        "assets_root": args.assets_root,
    }
    missing = [name for name, path in roots.items() if not path.is_dir()]
    if missing:
        print(json.dumps({"ok": False, "missing": missing}, sort_keys=True))
        return 2
    manifest = {
        "schema_version": 1,
        "omnigibson_root": str(args.omnigibson_root.resolve()),
        "bddl_root": str(args.bddl_root.resolve()),
        "assets_root": str(args.assets_root.resolve()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"ok": True, "manifest": str(args.output.resolve())}, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
