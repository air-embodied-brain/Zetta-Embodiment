# Copyright (c) 2026 Zetta Contributors
"""Prepare the assets for the initial Genie Sim pick-block-color profile.

Run with huggingface-hub and usd-core in a separate preparation environment.
The immutable download directory is retained alongside the relocatable copy.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import shutil
from pathlib import Path

ASSET_REPOSITORY = "agibot-world/GenieSimAssets"
ASSET_REVISION = "a0813aad7c16165daffbe6c2737754e0344809ee"
GENIESIM_REVISION = "6ca11c7593ecf6b7dae58c28fd19f4a789a0dd46"
ROOT_FILES = ("__init__.py", "pyproject.toml", "setup.py", "VERSION", "LICENSE")
ASSET_PREFIXES = (
    "background/common",
    "background/home/home_b_aligned",
    "robot/G2_omnipicker",
    "objects/benchmark/table/benchmark_table_019",
    "objects/benchmark/building_blocks/benchmark_building_blocks_073",
    "objects/benchmark/building_blocks/benchmark_building_blocks_085",
    "objects/benchmark/building_blocks/benchmark_building_blocks_077",
    "objects/benchmark/building_blocks/benchmark_building_blocks_081",
    "objects/benchmark/building_blocks/benchmark_building_blocks_089",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geniesim-root", required=True, type=Path)
    parser.add_argument("--download-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    source = args.download_dir.resolve()
    output = args.output_dir.resolve()
    if source.is_relative_to(output) or output.is_relative_to(source):
        parser.error("download-dir and output-dir must be separate directories")
    if args.workers < 1:
        parser.error("workers must be positive")
    scene = (
        args.geniesim_root.resolve()
        / "source/geniesim_benchmark/src/geniesim_benchmark"
        / "benchmark/config/llm_task/pick_block_color/0/scene.usda"
    )
    if not scene.is_file():
        parser.error(f"the pinned task scene is missing: {scene}")

    from huggingface_hub import HfApi, hf_hub_download
    from huggingface_hub.hf_api import RepoFile
    from pxr import Sdf, UsdUtils

    api = HfApi()
    filenames = set(ROOT_FILES)
    expected_hashes: dict[str, str] = {}
    expected_bytes = 0
    for prefix in ASSET_PREFIXES:
        for item in api.list_repo_tree(
            ASSET_REPOSITORY,
            path_in_repo=prefix,
            recursive=True,
            revision=ASSET_REVISION,
            repo_type="dataset",
        ):
            if isinstance(item, RepoFile):
                filenames.add(item.path)
                expected_bytes += item.size
                if item.lfs:
                    expected_hashes[item.path] = item.lfs.sha256
    print(f"Downloading {len(filenames)} files ({expected_bytes} bytes)", flush=True)

    def download(filename: str) -> tuple[str, str]:
        path = Path(
            hf_hub_download(
                ASSET_REPOSITORY,
                filename,
                repo_type="dataset",
                revision=ASSET_REVISION,
                local_dir=source,
            )
        )
        digest = sha256_file(path)
        expected = expected_hashes.get(filename)
        if expected and digest != expected:
            raise RuntimeError(f"asset checksum mismatch: {filename}")
        return filename, digest

    source_hashes = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(download, filename) for filename in sorted(filenames)]
        for index, future in enumerate(concurrent.futures.as_completed(futures), 1):
            filename, digest = future.result()
            source_hashes[filename] = digest
            if index % 25 == 0 or index == len(filenames):
                print(f"Downloaded {index}/{len(filenames)}", flush=True)

    for filename in sorted(filenames):
        target = output / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / filename, target)
    task_scene = output / "zetta_tasks/pick_block_color/0/scene.usda"
    task_scene.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(scene, task_scene)
    usd_files = [
        output / name for name in filenames if name.endswith((".usd", ".usda", ".usdc"))
    ]
    usd_files.append(task_scene)

    # Preserve USD reference/payload/list-op semantics while relocating the
    # upstream absolute asset prefix; keep the downloaded originals intact.
    for path in usd_files:
        layer = Sdf.Layer.FindOrOpen(str(path))
        if layer is None:
            raise RuntimeError(f"could not parse USD asset: {path}")

        def relocate(asset_path: str) -> str:
            prefix = "/geniesim_assets/"
            if asset_path.startswith(prefix):
                return os.path.relpath(output / asset_path[len(prefix) :], path.parent)
            return asset_path

        UsdUtils.ModifyAssetPaths(layer, relocate)
        layer.Save()

    manifest = {
        "repository": ASSET_REPOSITORY,
        "revision": ASSET_REVISION,
        "geniesim_revision": GENIESIM_REVISION,
        "task_name": "g2op_if_pick_block_color",
        "scene_instance_id": 0,
        "source_sha256": dict(sorted(source_hashes.items())),
        "task_source_sha256": sha256_file(scene),
        "relocated_usd_sha256": {
            path.relative_to(output).as_posix(): sha256_file(path)
            for path in sorted(usd_files)
        },
    }
    (output / "zetta_assets_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(f"Prepared task assets at {output}", flush=True)


if __name__ == "__main__":
    main()
