from pathlib import Path
from types import SimpleNamespace

import pytest

from zetta.policies.openpi.factory import build_openpi_model


def test_factory_rejects_non_openpi_model_without_loading_torch() -> None:
    config = SimpleNamespace(model_type="openvla")
    with pytest.raises(ValueError, match="openpi"):
        build_openpi_model(config)


def test_openpi_sources_do_not_import_main_rlinf_package() -> None:
    root = Path("zetta/policies/openpi")
    offenders = []
    for path in root.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        if "from rlinf" in text or "import rlinf" in text:
            offenders.append(str(path))
    assert offenders == []
