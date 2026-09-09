# Copyright (c) 2026 Zetta Contributors
"""Prepare isolated Genie/Isaac, Runtime and asset tools on Ubuntu 22.04 x86_64.

Requires Python 3.11.4+, uv, curl, apt-get and dpkg-deb. Packages are extracted
under --root, never installed into the host OS. Assets are about 9.4 GB.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import tarfile
from pathlib import Path

REVISION = "6ca11c7593ecf6b7dae58c28fd19f4a789a0dd46"
SOURCE_SHA256 = "a2ddb63c1f830dbcad0f6ab686dc8d56df50b39a0590dac1c4663ca2d8db58ff"
SDK_SHA256 = "0f7ab593d668b6ac70863d8d00339063f4e23d4b49500bd8a10c4f5a77b6bd97"
NATIVE_PACKAGES = (
    (
        "libtinyxml2.6.2v5=2.6.2-6ubuntu0.22.04.1",
        "libtinyxml2.6.2v5_2.6.2-6ubuntu0.22.04.1_amd64.deb",
        "f337b4687e87ea6fdfee7b95288fb4fb80247cf4d4cdea4354bfccd0d0054371",
    ),
    (
        "libminizip1=1.1-8build1",
        "libminizip1_1.1-8build1_amd64.deb",
        "3e962dbdd1c3ee9ca2418eda3f57b857bdbec1f97c9f807a1612b366cb5cd69c",
    ),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def check_hash(path: Path, expected: str) -> None:
    if sha256(path) != expected:
        raise RuntimeError(f"SHA256 mismatch: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source-archive", type=Path)
    parser.add_argument(
        "--lock",
        type=Path,
        help="Reuse a previously generated hash lock at the same deployment root",
    )
    parser.add_argument("--index", default="https://pypi.org/simple")
    parser.add_argument("--skip-assets", action="store_true")
    parser.add_argument("--skip-runtime", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        parser.error("this profile requires Linux x86_64 (validated on Ubuntu 22.04)")
    if not hasattr(tarfile, "data_filter"):
        parser.error(
            "safe source extraction requires Python 3.11.4+ or a patched Python"
        )
    root = args.root.expanduser().resolve()
    scripts = Path(__file__).resolve().parent
    repo = scripts.parents[1]
    artifacts = root / "artifacts"
    source = root / "src" / f"genie_sim-{REVISION}"
    archive = args.source_archive or artifacts / f"genie_sim-{REVISION}.tar.gz"
    python = root / "envs/isaac51/bin/python"
    runtime = root / "envs/runtime/bin/python"
    asset_python = root / "envs/asset-tools/bin/python"
    sdk = source / "3rdparty/ik_solver-0.4.3-cp311-cp311-linux_x86_64.whl"
    environment = dict(
        os.environ,
        UV_CACHE_DIR=str(root / "cache/uv"),
        UV_PYTHON_INSTALL_DIR=str(root / "envs/python"),
        UV_DEFAULT_INDEX=args.index,
        UV_EXTRA_INDEX_URL="https://pypi.nvidia.com",
        UV_HTTP_TIMEOUT="120",
        OMNI_KIT_ACCEPT_EULA="YES",
    )

    def run(*command: str | Path, cwd: Path = repo):
        return subprocess.run(
            [str(part) for part in command], cwd=cwd, env=environment, check=True
        )

    def create_venv(interpreter: Path) -> None:
        if not interpreter.is_file():
            run(
                "uv",
                "venv",
                "--python",
                "3.11.13",
                "--python-preference",
                "only-managed",
                interpreter.parent.parent,
            )

    if not args.verify_only:
        artifacts.mkdir(parents=True, exist_ok=True)
        (root / "src").mkdir(exist_ok=True)
        if not archive.is_file():
            run(
                "curl",
                "-fL",
                "--retry",
                "3",
                "--output",
                archive,
                f"https://codeload.github.com/AgibotTech/genie_sim/tar.gz/{REVISION}",
            )
        check_hash(archive, SOURCE_SHA256)
        if not source.is_dir():
            with tarfile.open(archive) as bundle:
                bundle.extractall(root / "src", filter="data")
        check_hash(sdk, SDK_SHA256)
        for package, filename, digest in NATIVE_PACKAGES:
            path = artifacts / filename
            if not path.is_file():
                run("apt-get", "download", package, cwd=artifacts)
            check_hash(path, digest)
            run("dpkg-deb", "--extract", path, root / "envs/native-libs")
        run("uv", "python", "install", "3.11.13")
        create_venv(python)
        lock = args.lock or artifacts / "geniesim-isaac51.lock"
        if not args.lock:
            requirements = artifacts / "geniesim-isaac51.in"
            requirements.write_text(
                (scripts / "geniesim_isaac51.in").read_text()
                + f"\nik-solver @ {sdk.as_uri()}\n"
            )
            run(
                "uv",
                "pip",
                "compile",
                requirements,
                "--python",
                python,
                "--generate-hashes",
                "--output-file",
                lock,
            )
        run("uv", "pip", "install", "--python", python, "--require-hashes", "-r", lock)
        if not args.skip_runtime:
            create_venv(runtime)
            run(
                "uv",
                "pip",
                "install",
                "--python",
                runtime,
                "--index",
                "https://download.pytorch.org/whl/cpu",
                "torch==2.7.0+cpu",
            )
            run(
                "uv",
                "pip",
                "install",
                "--python",
                runtime,
                "-e",
                str(repo) + "[test,ray]",
            )
        if not args.skip_assets:
            create_venv(asset_python)
            run(
                "uv",
                "pip",
                "install",
                "--python",
                asset_python,
                "huggingface-hub==0.34.4",
                "usd-core==25.5.1",
            )
            run(
                asset_python,
                scripts / "prepare_geniesim_assets.py",
                "--geniesim-root",
                source,
                "--download-dir",
                root / "assets/geniesim-source",
                "--output-dir",
                root / "assets/geniesim",
            )
            run(
                asset_python,
                scripts / "check_geniesim_assets.py",
                "--assets-root",
                root / "assets/geniesim",
                "--report",
                artifacts / "geniesim-asset-closure.json",
            )
    check_hash(archive, SOURCE_SHA256)
    check_hash(sdk, SDK_SHA256)
    run("uv", "pip", "check", "--python", python)
    if not args.skip_runtime:
        run("uv", "pip", "check", "--python", runtime)
    if (
        not args.skip_assets
        and not (root / "assets/geniesim/zetta_assets_manifest.json").is_file()
    ):
        raise RuntimeError("prepared task asset manifest is missing")
    print(
        json.dumps(
            {
                "status": "dependencies_checked",
                "root": str(root),
                "geniesim_revision": REVISION,
                "source_sha256": SOURCE_SHA256,
                "sdk_sha256": SDK_SHA256,
                "simulator_python": str(python),
                "runtime_python": str(runtime),
                "hardware_smoke": "run smoke_geniesim.py separately",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
