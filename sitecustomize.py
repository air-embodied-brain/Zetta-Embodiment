"""Optional runtime overlays for isolated Zetta experiment worktrees.

Python imports ``sitecustomize`` during interpreter startup when this worktree
is on ``PYTHONPATH``.  The overlay is deliberately inert unless an explicit
package root is supplied, so normal Zetta processes keep their installed
LIBERO package unchanged.
"""

from __future__ import annotations

import os
from pathlib import Path


def _activate_libero_package_overlay() -> None:
    package_root = os.environ.get("ZETTA_LIBERO_PACKAGE_ROOT")
    if not package_root:
        return

    root = Path(package_root).expanduser().resolve()
    if not (root / "libero" / "__init__.py").is_file():
        raise RuntimeError(
            "ZETTA_LIBERO_PACKAGE_ROOT must contain libero/__init__.py; "
            f"got {root}"
        )

    # The wheel installs an outer ``libero`` package containing the actual
    # ``libero.libero`` implementation.  LIBERO-PRO's source checkout uses an
    # implicit outer namespace, which loses to that regular installed package
    # under ordinary PYTHONPATH precedence.  Extending the installed package's
    # search path gives the isolated process the audited source implementation
    # and assets without writing into the shared virtual environment.
    import libero

    source = str(root)
    paths = list(libero.__path__)
    if source in paths:
        paths.remove(source)
    libero.__path__[:] = [source, *paths]


_activate_libero_package_overlay()
