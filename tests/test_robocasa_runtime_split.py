# Copyright (c) 2026 Zetta Contributors
"""runtime v3 design §4 / Stage 3: session_core.py and groot_core.py
must be independently usable with zero ``http.server`` dependency.

``test_robocasa_env_runtime.py`` and ``test_robocasa_groot_server.py`` already
exercise ``RoboCasaSession.execute_chunk`` and ``Gr00tRuntime.act`` without any
HTTP server; this module pins the actual module-boundary property Stage 3
introduced: the two pure-logic modules import cleanly on their own and never
pull in ``http.server``, while ``env_server.py``/``groot_server.py`` keep
re-exporting the same public objects (so existing callers such as
``scripts/evolution/robocasa_capacity_worker.py`` and the standalone debugging
servers keep working unchanged).
"""

from __future__ import annotations

import ast
import inspect
import sys
from types import SimpleNamespace

import numpy as np

from robots.robocasa import env_server, groot_server, session_core
from robots.robocasa.groot_core import Gr00tModelCore, Gr00tRuntime


def _imports_http_server(module: object) -> bool:
    tree = ast.parse(inspect.getsource(module))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(alias.name.split(".")[0] == "http" for alias in node.names):
                return True
        elif isinstance(node, ast.ImportFrom) and node.module:
            if node.module.split(".")[0] == "http":
                return True
    return False


def test_session_core_has_no_http_server_dependency() -> None:
    """``session_core.py`` must not import ``http`` at all."""
    assert not _imports_http_server(sys.modules["robots.robocasa.session_core"])


def test_groot_core_has_no_http_server_dependency() -> None:
    """``groot_core.py`` must not import ``http`` at all."""
    assert not _imports_http_server(sys.modules["robots.robocasa.groot_core"])


def test_env_server_reexports_the_same_session_class() -> None:
    """The HTTP shell must not fork a second ``RoboCasaSession`` definition."""
    assert env_server.RoboCasaSession is session_core.RoboCasaSession


def test_groot_server_reexports_the_same_runtime_class() -> None:
    """The HTTP shell must not fork a second GR00T runtime definition."""
    assert groot_server.Gr00tRuntime is Gr00tRuntime is Gr00tModelCore


def test_robocasa_session_reset_and_execute_chunk_work_without_any_server(
    monkeypatch,
) -> None:
    """``RoboCasaSession`` drives a fake env end-to-end with no HTTP involved.

    ``reset`` normally calls ``gymnasium.make("robocasa/...")``, which needs
    the real RoboCasa simulator package; this test stubs ``_ensure_environment``
    so the pure control flow (episode bookkeeping, video writer lifecycle,
    critic wiring) runs the same way it would with a real environment.
    """
    session = session_core.RoboCasaSession(
        camera_size=4,
        max_steps=10,
        require_isolated_renderer=False,
    )

    class _FakeEnv:
        def __init__(self) -> None:
            self.actions: list[dict[str, np.ndarray]] = []

        def reset(self, seed: int) -> tuple[dict[str, np.ndarray], dict]:
            del seed
            return {
                "state.end_effector_position_relative": np.zeros(3),
                "frame": np.zeros((4, 4, 3), dtype=np.uint8),
            }, {"success": False}

        def step(self, action: dict[str, np.ndarray]) -> tuple[Any, ...]:  # type: ignore[name-defined]
            self.actions.append(action)
            observation = {
                "state.end_effector_position_relative": np.zeros(3),
                "frame": np.zeros((4, 4, 3), dtype=np.uint8),
            }
            return observation, 1.0, False, False, {"success": True}

        def close(self) -> None:
            pass

    fake_env = _FakeEnv()

    def _fake_ensure_environment(self, task: str, split: str) -> None:
        self.env = fake_env
        self.identity = (task, split)

    monkeypatch.setattr(
        session_core.RoboCasaSession,
        "_ensure_environment",
        _fake_ensure_environment,
    )

    reset_result = session.reset({"task": "SlideDishwasherRack", "seed": 1})
    assert reset_result["step_index"] == 0
    assert reset_result["terminated"] is False

    chunk_result = session.execute_chunk(
        {
            "actions": [[0.0] * 12],
            "critic_rules": [],
            "interrupt_on_proposal": False,
            "capture_event_images": False,
        }
    )
    assert chunk_result["executed_horizon"] == 1
    assert chunk_result["authoritative_success"] is True
    assert len(fake_env.actions) == 1


def test_groot_model_core_runs_one_inference_without_any_server() -> None:
    """``Gr00tModelCore.act`` runs end-to-end with a fake policy, no HTTP."""

    class _FakePolicy:
        def get_action(self, observation: dict[str, np.ndarray]) -> dict[str, Any]:  # type: ignore[name-defined]
            assert observation["video.left"].dtype == np.uint8
            return {"action.x": np.asarray([[0.5]], dtype=np.float32)}

    data_config = SimpleNamespace(
        video_keys=("video.left",),
        state_keys=("state.pose",),
        language_keys=("annotation.human.task_description",),
        observation_indices=(0,),
        action_keys=("action.x",),
        action_indices=(0,),
    )
    core = Gr00tModelCore(
        policy=_FakePolicy(),
        data_config=data_config,
        checkpoint_sha256="b" * 64,
        denoising_steps=4,
    )
    actions, metadata = core.act(
        {
            "seed": 7,
            "observation": {
                "video.left": np.zeros((1, 4, 5, 3), dtype=np.uint8).tolist(),
                "state.pose": [[0.0, 1.0]],
                "annotation.human.task_description": ["Slide the rack."],
            },
        }
    )
    assert actions == {"action.x": [[0.5]]}
    assert metadata["checkpoint_sha256"] == "b" * 64
