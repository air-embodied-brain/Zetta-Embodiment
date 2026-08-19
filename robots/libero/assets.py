# Copyright (c) 2026 Zetta Contributors
"""Runtime binding for a composite robosuite / LIBERO asset tree."""

from __future__ import annotations

import hashlib
import os
import sys
import tempfile
from pathlib import Path
from typing import Any


def bind_libero_assets_root(value: str | os.PathLike[str]) -> Path:
    """Bind both robosuite globals and imported path-completion aliases."""

    # Keep symlink components intact. LIBERO-Pro's absolute asset helpers
    # derive ``<parent>/assets`` from a module path; resolving a prepared
    # ``.../libero-overlay/assets`` link would turn that into
    # ``.../assets/libero-composite-*/assets``.
    assets_path = Path(os.path.abspath(os.path.expanduser(os.fspath(value))))
    if not assets_path.is_dir():
        raise FileNotFoundError(
            f"LIBERO_ASSETS_ROOT_OVERRIDE is not a directory: {assets_path}"
        )

    import robosuite.models
    import robosuite.utils.mjcf_utils as mjcf_utils

    previous_completion = mjcf_utils.xml_path_completion

    def overlay_xml_path_completion(xml_path: str) -> str:
        if os.path.isabs(xml_path):
            return xml_path
        return str(assets_path / xml_path)

    robosuite.models.assets_root = str(assets_path)
    mjcf_utils.xml_path_completion = overlay_xml_path_completion

    # LIBERO arena / robot modules import this function directly. Update
    # aliases already loaded before RLinF invokes its nested worker factory;
    # modules imported later receive the patched function from mjcf_utils.
    for module in tuple(sys.modules.values()):
        if module is None:
            continue
        try:
            completion: Any = getattr(module, "xml_path_completion", None)
        except Exception:
            continue
        if completion is previous_completion:
            setattr(module, "xml_path_completion", overlay_xml_path_completion)

    # LIBERO-Pro's BDDL base keeps its scene root in a module-level ``DIR_PATH``
    # and constructs ``../assets/<scene>`` as an absolute path. Existing
    # package overlays already have a lexical ``assets`` directory. For a
    # direct composite root with another name, create a small isolated
    # compatibility root instead of assuming its basename is ``assets``.
    legacy_root = assets_path.parent
    if assets_path.name != "assets":
        key = hashlib.sha256(str(assets_path).encode("utf-8")).hexdigest()[:16]
        legacy_root = Path(tempfile.gettempdir()) / f"zetta-libero-assets-{key}"
        legacy_root.mkdir(parents=True, exist_ok=True)
        link = legacy_root / "assets"
        if not link.exists() and not link.is_symlink():
            try:
                link.symlink_to(assets_path, target_is_directory=True)
            except FileExistsError:
                pass  # Another environment process created the same link.
        if (
            not link.is_symlink()
            or os.path.realpath(link) != os.path.realpath(assets_path)
        ):
            raise RuntimeError(f"asset compatibility path is occupied: {link}")

    # The synthetic parent makes ``DIR_PATH/../assets`` resolve to the caller's
    # asset tree without modifying shared site-packages.
    try:
        from liberopro.liberopro.envs import bddl_base_domain
    except Exception:
        bddl_base_domain = None
    if bddl_base_domain is not None:
        bddl_base_domain.DIR_PATH = str(legacy_root / "runtime")
    for name, module in tuple(sys.modules.items()):
        if not name.startswith("liberopro.liberopro.envs") or module is None:
            continue
        if hasattr(module, "absolute_path"):
            module.absolute_path = legacy_root
    return assets_path
