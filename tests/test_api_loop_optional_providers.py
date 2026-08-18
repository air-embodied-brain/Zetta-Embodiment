from __future__ import annotations

import builtins

import pytest

pydantic_ai = pytest.importorskip("pydantic_ai")
ModelSettings = pydantic_ai.ModelSettings

from rpent.planner.api_loop import _build_model_settings


class _OpenAIOnlyModel:
    pass


def test_openai_model_settings_do_not_import_anthropic(monkeypatch) -> None:
    real_import = builtins.__import__

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name.startswith("pydantic_ai.models.anthropic"):
            raise AssertionError("OpenAI-compatible models must not import Anthropic")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    settings = _build_model_settings(_OpenAIOnlyModel(), max_tokens=4096)

    # ModelSettings is a TypedDict and deliberately has no runtime
    # ``isinstance`` support; compare the returned structure instead.
    assert settings == ModelSettings(max_tokens=4096)
