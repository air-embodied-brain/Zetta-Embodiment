# Copyright (c) 2026 Zetta Contributors
"""Install the native ManiSkill profile under an isolated Linux deployment root."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shlex
import subprocess
import urllib.request
import zipfile
from pathlib import Path

VULKAN_PACKAGE = "libvulkan1_1.3.204.1-2_amd64.deb"
VULKAN_SHA256 = "192adcff489996b3398e7e7c0012b98e9586b46fe9a9eb13fb02c0feba88548b"
PHYSX_VERSION = "105.1-physx-5.3.1.patch0"
PHYSX_ARCHIVE = "physx-linux-so.zip"
PHYSX_SHA256 = "167a01aad7381afef963b89169968c289e7b653880a7a823c116d87ee5c00fc6"
PHYSX_URL = f"https://github.com/sapien-sim/physx-precompiled/releases/download/{PHYSX_VERSION}/linux-so.zip"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--python", default="3.11")
    parser.add_argument("--index", default="https://pypi.org/simple")
    parser.add_argument("--lock", type=Path)
    parser.add_argument(
        "--wheelhouse",
        type=Path,
        help="install pre-downloaded artifacts without index access",
    )
    parser.add_argument(
        "--installer",
        choices=("uv", "pip"),
        default="uv",
        help="installer for the complete hash-locked dependency closure",
    )
    parser.add_argument("--with-vulkan-loader", action="store_true")
    parser.add_argument(
        "--cuda-driver-dir",
        type=Path,
        help="directory containing host-driver libcuda.so.1 (avoids incompatible container compat libraries)",
    )
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument(
        "--ray-temp-alias",
        type=Path,
        help="short symlink to this root's Ray cache for AF_UNIX socket paths",
    )
    args = parser.parse_args()
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        parser.error("this deployment profile requires Linux x86_64")
    root = args.root.expanduser().resolve()
    repo = Path(__file__).resolve().parents[2]
    for name in ("envs", "cache", "assets", "models", "outputs", "artifacts"):
        (root / name).mkdir(parents=True, exist_ok=True)
    (root / "cache/tmp").mkdir(exist_ok=True)
    ray_cache = root / "cache/ray"
    ray_cache.mkdir(exist_ok=True)
    ray_temp = ray_cache
    if args.ray_temp_alias:
        alias = args.ray_temp_alias.expanduser().absolute()
        if alias.exists() or alias.is_symlink():
            if not alias.is_symlink() or alias.resolve() != ray_cache.resolve():
                parser.error("Ray alias already exists and points elsewhere")
        else:
            alias.symlink_to(ray_cache, target_is_directory=True)
        ray_temp = alias
    elif (root / "environment.json").is_file():
        previous = json.loads((root / "environment.json").read_text()).get("RAY_TMPDIR")
        if previous and Path(previous).resolve() == ray_cache.resolve():
            ray_temp = Path(previous)
    python = root / "envs/runtime/bin/python"
    lock = args.lock or root / "artifacts/maniskill-runtime.lock"
    environment = {
        **os.environ,
        "UV_CACHE_DIR": str(root / "cache/uv"),
        "UV_PYTHON_INSTALL_DIR": str(root / "envs/python"),
        "UV_HTTP_TIMEOUT": "120",
        "UV_DEFAULT_INDEX": args.index,
        "PIP_CACHE_DIR": str(root / "cache/pip"),
        "PIP_INDEX_URL": args.index,
        "XDG_CACHE_HOME": str(root / "cache/xdg"),
        "MS_ASSET_DIR": str(root / "assets"),
        "HF_HOME": str(root / "cache/huggingface"),
        "TORCH_HOME": str(root / "cache/torch"),
        "MPLCONFIGDIR": str(root / "cache/matplotlib"),
        "TMPDIR": str(root / "cache/tmp"),
        "RAY_TMPDIR": str(ray_temp),
        "PYTHONUNBUFFERED": "1",
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "SAPIEN_PHYSX_GPU_LIBRARY": str(
            root / "envs/physx" / PHYSX_VERSION / "libPhysXGpu_64.so"
        ),
    }

    def run(*command: str | Path, cwd: Path = repo) -> None:
        subprocess.run([str(v) for v in command], cwd=cwd, env=environment, check=True)

    if not args.verify_only:
        install_source = []
        if args.wheelhouse:
            if not args.wheelhouse.is_dir() or not lock.is_file():
                parser.error(
                    "offline installation requires an existing wheelhouse and lock"
                )
            install_source = [
                "--no-index",
                "--find-links",
                str(args.wheelhouse.resolve()),
            ]
        if not python.is_file():
            run("uv", "venv", "--python", args.python, python.parent.parent)
        if not lock.is_file():
            if args.lock:
                parser.error("the supplied lock file does not exist")
            run(
                "uv",
                "pip",
                "compile",
                repo / "pyproject.toml",
                "--extra",
                "test",
                "--extra",
                "ray",
                "--extra",
                "maniskill",
                "--constraint",
                repo / "scripts/deployment/maniskill_runtime.in",
                "--python",
                python,
                "--generate-hashes",
                "--quiet",
                "--output-file",
                lock,
            )
        # The compiled lock contains the entire closure. Avoid re-resolving
        # unconstrained transitive requirements from an installed CUDA wheel;
        # uv 0.8.22 otherwise rejects these in hash mode. pip check runs below.
        if args.installer == "pip":
            run(
                "uv",
                "pip",
                "install",
                "--python",
                python,
                "--no-deps",
                "pip==25.2",
                "setuptools==80.9.0",
                "wheel==0.45.1",
                *install_source,
            )
            run(
                python,
                "-m",
                "pip",
                "install",
                "--require-hashes",
                "--no-deps",
                "--no-build-isolation",
                *install_source,
                "-r",
                lock,
            )
        else:
            run(
                "uv",
                "pip",
                "install",
                "--python",
                python,
                "--require-hashes",
                "--no-deps",
                *install_source,
                "-r",
                lock,
            )
        run(
            "uv",
            "pip",
            "install",
            "--python",
            python,
            "--no-deps",
            *(["--no-build-isolation"] if args.installer == "pip" else []),
            *install_source,
            "-e",
            repo,
        )
        if args.with_vulkan_loader:
            archive = root / "artifacts" / VULKAN_PACKAGE
            if not archive.is_file():
                run("apt-get", "download", "libvulkan1=1.3.204.1-2", cwd=archive.parent)
            if hashlib.sha256(archive.read_bytes()).hexdigest() != VULKAN_SHA256:
                raise ValueError("Vulkan loader archive SHA256 mismatch")
            run("dpkg-deb", "--extract", archive, root / "envs/native")
        archive = root / "artifacts" / PHYSX_ARCHIVE
        if not archive.is_file():
            if args.wheelhouse:
                parser.error(f"offline preparation also requires {archive}")
            with urllib.request.urlopen(PHYSX_URL, timeout=120) as response:
                archive.write_bytes(response.read())
        if hashlib.sha256(archive.read_bytes()).hexdigest() != PHYSX_SHA256:
            raise ValueError("PhysX GPU archive SHA256 mismatch")
        library = Path(environment["SAPIEN_PHYSX_GPU_LIBRARY"])
        library.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive) as bundle:
            library.write_bytes(bundle.read("libPhysXGpu_64.so"))
    if not Path(environment["SAPIEN_PHYSX_GPU_LIBRARY"]).is_file():
        parser.error("prepared PhysX GPU library is missing; rerun preparation")
    native = root / "envs/native/usr/lib/x86_64-linux-gnu"
    driver = root / "envs/driver"
    if args.cuda_driver_dir:
        driver.mkdir(exist_ok=True)
        source = args.cuda_driver_dir.resolve()
        if not (source / "libcuda.so.1").is_file():
            parser.error("CUDA driver directory must contain libcuda.so.1")
        for name in (
            "libcuda.so",
            "libcuda.so.1",
            "libnvidia-ptxjitcompiler.so.1",
            "libnvidia-nvvm.so.4",
        ):
            library = source / name
            target = driver / name
            if not library.is_file():
                continue
            if target.exists() or target.is_symlink():
                if not target.is_symlink() or target.resolve() != library.resolve():
                    parser.error(f"driver link already points elsewhere: {target}")
            else:
                target.symlink_to(library.resolve())
    exports = {
        key: environment[key]
        for key in (
            "XDG_CACHE_HOME",
            "MS_ASSET_DIR",
            "HF_HOME",
            "TORCH_HOME",
            "MPLCONFIGDIR",
            "TMPDIR",
            "RAY_TMPDIR",
            "PYTHONUNBUFFERED",
            "CUDA_DEVICE_ORDER",
            "SAPIEN_PHYSX_GPU_LIBRARY",
        )
    }
    if native.is_dir():
        icd = root / "envs/nvidia_icd.json"
        icd.write_text(
            json.dumps(
                {
                    "file_format_version": "1.0.0",
                    "ICD": {
                        "library_path": "libGLX_nvidia.so.0",
                        "api_version": "1.2.155",
                    },
                },
                indent=2,
            )
            + "\n"
        )
        exports["VK_ICD_FILENAMES"] = str(icd)
        exports["LD_LIBRARY_PATH"] = str(native) + (
            ":" + os.environ["LD_LIBRARY_PATH"]
            if os.environ.get("LD_LIBRARY_PATH")
            else ""
        )
    if driver.is_dir():
        exports["LD_LIBRARY_PATH"] = (
            str(driver)
            + ":"
            + exports.get("LD_LIBRARY_PATH", os.environ.get("LD_LIBRARY_PATH", ""))
        )
    (root / "environment.json").write_text(json.dumps(exports, indent=2) + "\n")
    (root / "activate.sh").write_text(
        "# Source this file before launching Runtime or GPU smoke tests.\n"
        + "\n".join(
            f"export {key}={shlex.quote(value)}" for key, value in exports.items()
        )
        + f"\n. {shlex.quote(str(python.parent / 'activate'))}\n"
    )
    environment.update(exports)
    run("uv", "pip", "check", "--python", python)
    report = subprocess.check_output(
        [
            str(python),
            "-c",
            "import importlib.metadata,json,sys; "
            "print(json.dumps({'python':sys.version,'packages':{k:importlib.metadata.version(k) "
            "for k in ['mani-skill','sapien','torch','gymnasium','numpy','ray']}}))",
        ],
        env=environment,
        text=True,
    )
    payload = json.loads(report)
    payload["lock_sha256"] = hashlib.sha256(lock.read_bytes()).hexdigest()
    payload["source_root"] = str(repo)
    payload["physx_archive_sha256"] = PHYSX_SHA256
    (root / "outputs/installation.json").write_text(
        json.dumps(payload, indent=2) + "\n"
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
