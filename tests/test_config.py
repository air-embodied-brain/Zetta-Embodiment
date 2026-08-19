# Copyright (c) 2026 Zetta Contributors
from __future__ import annotations

from zetta.utils import config


def test_rlinf_source_path_configuration_has_been_removed() -> None:
    assert not hasattr(config, "get_rlinf_repo_path")
