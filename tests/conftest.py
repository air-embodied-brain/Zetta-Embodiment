"""Repository-level pytest configuration.

Only defines ``pytest_addoption`` here: it must be defined in the **initial
conftest** (an ancestor of rootdir or the command-line argument directory);
placing it in ``tests/runtime/conftest.py`` would make ``pytest tests`` fail
immediately.

``--transport`` is used by ``tests/runtime/test_e2e_fake.py`` for its test
matrix: an early milestone only supports ``inproc``, and once
``RayChannelTransport`` lands the same assertions are rerun with
``--transport=ray_channel``.
"""

from __future__ import annotations

import pytest

TRANSPORT_CHOICES = ("inproc", "ray_channel")
"""Valid values for ``--transport``; ``ray_channel`` becomes available later."""


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register command-line options for runtime tests.

    Args:
        parser: pytest argument parser.
    """
    group = parser.getgroup("rollout_runtime")
    group.addoption(
        "--transport",
        action="store",
        default="inproc",
        choices=list(TRANSPORT_CHOICES),
        help="rollout runtime transport under test (ray_channel lands in M3)",
    )
