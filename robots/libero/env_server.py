# Copyright (c) 2026 RPent Contributors
"""RPC server wrapping a single-env LIBERO environment."""
from __future__ import annotations

import argparse
import os
import sys
from typing import TYPE_CHECKING, Any

import numpy as np
from omegaconf import OmegaConf

from robots.libero.assets import bind_libero_assets_root
from robots.libero.critic_runtime import (
    critic_rules_from_payload,
    extract_libero_critic_features,
)
from rpent.evolution.critic import TemporalCritic
from rpent.evolution.jsonio import canonical_sha256
from rpent.utils.config import (
    get_repo_root,
    get_rlinf_repo_path,
)
from rpent.utils.logging import get_logger
from rpent.utils.rpc import RpcFacade

# MuJoCo env vars must be set BEFORE importing anything that touches MuJoCo.
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
assert "mujoco" not in sys.modules, \
    "mujoco must not be imported before MUJOCO_GL/PYOPENGL_PLATFORM are set"

logger = get_logger("env_server")

RPENT_ROOT = get_repo_root()
RLINF_REPO_PATH = get_rlinf_repo_path() or (RPENT_ROOT.parent / "rlinf").resolve()
if str(RLINF_REPO_PATH) not in sys.path:
    sys.path.insert(0, str(RLINF_REPO_PATH))
os.environ.setdefault("ROBOT_PLATFORM", "LIBERO")

# torch and LiberoEnv are only imported at call time (after --cuda-device
# sets CUDA_VISIBLE_DEVICES in main()); LiberoEnv transitively imports torch.
if TYPE_CHECKING:
    import torch  # noqa: F401  (referenced at runtime in _to_numpy_tree)
    from rlinf.envs.libero.libero_env import LiberoEnv


# ---------------------------------------------------------------------------
# Config builders
# ---------------------------------------------------------------------------


def build_env_cfg(
    *,
    task_suite_name: str = "libero_spatial",
    specific_reset_id: int = 0,
    seed: int = 0,
    max_episode_steps: int = 10000,
) -> Any:
    cfg = OmegaConf.create(
        {
            "env_type": "libero",
            "task_suite_name": task_suite_name,
            "auto_reset": False,
            "ignore_terminations": False,
            "max_steps_per_rollout_epoch": max_episode_steps,
            "max_episode_steps": max_episode_steps,
            "use_rel_reward": False,
            "use_step_penalty": False,
            "reward_coef": 1.0,
            "reset_gripper_open": True,
            "is_eval": True,
            "seed": seed,
            "group_size": 1,
            "use_fixed_reset_state_ids": True,
            "use_ordered_reset_state_ids": True,
            "specific_reset_id": specific_reset_id,
            "video_cfg": {
                "save_video": True,
                "info_on_video": True,
                "video_base_dir": "/tmp/primitive_videos",
            },
            "init_params": {
                "camera_heights": 256,
                "camera_widths": 256,
                # Render depth too, so we can back-project pixels to world
                # from depth + camera calibration
                "camera_depths": True,
                # RLinF owns evaluation truncation. Keep robosuite's internal
                # horizon beyond that boundary because LIBERO-PRO overwrites
                # robosuite's horizon-done with task success; matching limits
                # makes the next action raise instead of returning truncation.
                "horizon": max_episode_steps + 1000,
                # Keep robosuite from locking its internal episode before
                # RLinF emits the authoritative outer truncation. LIBERO's
                # task wrapper still returns _check_success(), so this does
                # not suppress task success or change libero_terminated.
                "ignore_done": True,

                **({"robots": [os.environ["LIBERO_ROBOT_BASE"]]}
                   if os.environ.get("LIBERO_ROBOT_BASE") else {}),
            },
        }
    )
    return cfg


def make_env(task_id: int, seed: int, suite_name: str = "libero_spatial",
             max_episode_steps: int = 10000) -> LiberoEnv:
    """Build a single-env LiberoEnv pinned to ``task_id`` / ``seed``."""
    from robots.libero.rlinf_worker_compat import install_rlinf_env_call_compat

    install_rlinf_env_call_compat()
    from rlinf.envs.libero.libero_env import LiberoEnv
    from rlinf.envs.libero.utils import benchmark as _bench_mod

    from robots.libero.privileged_sensors import install_libero_contact_extension

    assets_override = os.environ.get("LIBERO_ASSETS_ROOT_OVERRIDE")
    if assets_override:
        bind_libero_assets_root(assets_override)

    # Attach the privileged current-contact sensor before RLinF starts its
    # robosuite subprocess. It has a force/tactile analogue on real hardware.
    install_libero_contact_extension(LiberoEnv)
    suite = _bench_mod.get_benchmark(suite_name)()
    first_id = sum(len(suite.get_task_init_states(t)) for t in range(task_id))
    trials = len(suite.get_task_init_states(task_id))
    rid = first_id + (seed % trials)
    cfg = build_env_cfg(
        task_suite_name=suite_name,
        specific_reset_id=rid,
        seed=seed,
        max_episode_steps=max_episode_steps,
    )
    return LiberoEnv(cfg=cfg, num_envs=1, seed_offset=0,
                     total_num_processes=1, worker_info=None)


# ---------------------------------------------------------------------------
# Facade implementing the robots.libero.env_client protocol
# ---------------------------------------------------------------------------


def _to_numpy_tree(x):
    """Recursively convert torch tensors to CPU numpy arrays so the result
    pickles cleanly across the agent/env_server wire."""
    torch = sys.modules.get("torch")
    if torch is not None and isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    if isinstance(x, dict):
        return {k: _to_numpy_tree(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_to_numpy_tree(v) for v in x]
    if isinstance(x, tuple):
        return tuple(_to_numpy_tree(v) for v in x)
    return x


class LiberoEnvFacade(RpcFacade):
    """Implements :class:`robots.libero.env_client.LiberoEnvClient`
    over :class:`rlinf.envs.libero.libero_env.LiberoEnv`.

    All return values are converted to CPU numpy so the agent process
    (which does not import torch) can consume them after the pickle round
    trip.
    """

    def __init__(self, env: LiberoEnv, *, meta: dict):
        super().__init__()
        self._env = env
        self._env_idx = 0
        self._done = False
        # Identifies what task/seed this server was launched with — the
        # client compares against its own expected values at construction
        # and refuses to talk to a stale or mis-configured server.
        self._meta = dict(meta)
        self._critic: TemporalCritic | None = None
        self._critic_fingerprint: str | None = None
        self._step_index = 0
        self._audit_trace: list[dict[str, Any]] = []
        self._critic_previous_eef: np.ndarray | None = None

    def _dispatch(self, method: str, args: tuple, kwargs: dict) -> Any:
        if method.startswith("env."):
            attr = method[len("env."):]
            try:
                return getattr(self, attr)(*args, **kwargs)
            except Exception:
                logger.exception("run method %s failed", method)
                raise
        raise ValueError(f"unknown RPC method: {method!r}")

    # ---- shape helpers ----

    def _strip(self, v):
        """Drop the leading env dim. ``v`` is either a batched numpy array
        (shape ``[B, ...]``), a length-B list (e.g. ``task_descriptions``),
        or ``None`` (optional images). LiberoEnv runs ``num_envs=1`` so
        index ``self._env_idx`` is always present."""
        if v is None:
            return None
        return v[self._env_idx]

    def _strip_obs(self, obs: dict) -> dict:
        """Strip the leading env dim from every value of a LIBERO obs dict."""
        return {k: self._strip(v) for k, v in obs.items()}

    def _expand_action(self, action) -> np.ndarray:
        """Inject the env dim onto a single-env action shaped ``[action_dim]``."""
        return np.asarray(action)[None]

    def _expand_chunk(self, actions) -> np.ndarray:
        """Inject the env dim onto a single-env chunk shaped
        ``[chunk_size, action_dim]``."""
        return np.asarray(actions)[None]

    def _record_done(self, *signals: Any) -> None:
        """OR the truthiness of every termination/truncation signal into
        ``self._done`` so subsequent step() calls short-circuit."""
        for s in signals:
            if np.asarray(s).any():
                self._done = True
                return

    # ---- gym-like surface ----

    def reset(self):
        obs, info = self._env.reset()
        obs = self._strip_obs(_to_numpy_tree(obs))
        self._done = False
        self._critic = None
        self._critic_fingerprint = None
        self._step_index = 0
        self._audit_trace = []
        states = np.asarray(obs.get("states", ()), dtype=np.float64).reshape(-1)
        self._critic_previous_eef = states[:3].copy() if states.size >= 3 else None
        return obs, _to_numpy_tree(info)

    def step(self, action):
        assert not self._done, "step called after episode done"
        try:
            obs, rew, term, trunc, info = self._env.step(
                self._expand_action(action)
            )
        except EOFError as exc:
            workers = getattr(getattr(self._env, "env", None), "workers", ())
            worker = workers[self._env_idx] if len(workers) > self._env_idx else None
            process = getattr(worker, "process", None)
            if process is not None:
                try:
                    process.join(timeout=0.25)
                except (AssertionError, RuntimeError):
                    pass
            pid = getattr(process, "pid", None)
            exitcode = getattr(process, "exitcode", None)
            try:
                alive = process.is_alive() if process is not None else None
            except (AssertionError, RuntimeError):
                alive = None
            raise RuntimeError(
                "LIBERO environment worker pipe closed during step "
                f"{self._step_index + 1}: worker_index={self._env_idx}, "
                f"pid={pid}, exitcode={exitcode}, alive={alive}"
            ) from exc
        obs = self._strip_obs(_to_numpy_tree(obs))
        term = self._strip(_to_numpy_tree(term))
        trunc = self._strip(_to_numpy_tree(trunc))
        self._record_done(term, trunc)
        self._step_index += 1
        return (
            obs,
            self._strip(_to_numpy_tree(rew)),
            term,
            trunc,
            _to_numpy_tree(info),
        )

    def _configure_critic(self, payload: list[dict[str, Any]]) -> None:
        fingerprint = canonical_sha256(payload)
        if self._critic is None:
            self._critic = TemporalCritic(critic_rules_from_payload(payload))
            self._critic_fingerprint = fingerprint
        elif fingerprint != self._critic_fingerprint:
            raise ValueError("critic rules cannot change within one LIBERO episode")

    def critic_chunk_step(
        self,
        actions,
        *,
        critic_rules: list[dict[str, Any]],
        interrupt_on_proposal: bool = True,
        return_all_frames: bool = True,
    ):
        if not isinstance(critic_rules, list):
            raise ValueError("critic_rules must be an array")
        self._configure_critic(critic_rules)
        obs_list = []
        rewards = []
        terms = []
        truncations = []
        infos = []
        proposals: list[dict[str, Any]] = []
        step_records: list[dict[str, Any]] = []
        for action in np.asarray(actions):
            obs, reward, terminated, truncated, info = self.step(action)
            reward_value = float(np.asarray(reward).max())
            features = extract_libero_critic_features(
                obs,
                step_index=self._step_index,
                reward=reward_value,
                terminated=bool(np.asarray(terminated).any()),
                truncated=bool(np.asarray(truncated).any()),
                privileged_state=self._privileged_critic_state(),
                action=action,
                previous_eef=self._critic_previous_eef,
            )
            self._critic_previous_eef = np.asarray(
                obs["states"], dtype=np.float64
            ).reshape(-1)[:3].copy()
            step_proposals = (
                self._critic.evaluate(features, step_index=self._step_index)
                if self._critic is not None
                else []
            )
            action_value = np.asarray(action, dtype=np.float64).tolist()
            record = {
                "step_index": self._step_index,
                "action": action_value,
                "action_sha256": canonical_sha256(action_value),
                "state": features,
                "observation_sha256": canonical_sha256(features),
                "reward": reward_value,
                "libero_terminated": bool(np.asarray(terminated).any()),
                "truncated": bool(np.asarray(truncated).any()),
                "proposal_rule_ids": [row["rule_id"] for row in step_proposals],
            }
            self._audit_trace.append(record)
            step_records.append(record)
            obs_list.append(obs)
            rewards.append(reward)
            terms.append(terminated)
            truncations.append(truncated)
            infos.append(info)
            proposals.extend(step_proposals)
            if self._done or (interrupt_on_proposal and step_proposals):
                break
        if not obs_list:
            raise ValueError("critic_chunk_step requires at least one action")
        info = {
            "per_step": infos,
            "step_records": step_records,
            "critic_proposals": proposals,
            "critic_rule_count": len(critic_rules),
            "executed_horizon": len(obs_list),
        }
        return (
            obs_list if return_all_frames else obs_list[-1],
            np.asarray(rewards),
            np.asarray(terms, dtype=bool),
            np.asarray(truncations, dtype=bool),
            info,
        )

    def audit_trace(self, *, since_step: int = 0) -> list[dict[str, Any]]:
        if since_step < 0:
            raise ValueError("since_step must be non-negative")
        return [dict(row) for row in self._audit_trace if row["step_index"] > since_step]

    def chunk_step(self, actions, *, return_all_frames: bool = False):
        """Run a full action chunk in one RPC. ``actions`` shape
        ``[chunk_size, action_dim]`` (single env).

        Returns the 5-positional tuple
        ``(obs_or_list, reward, terminated, truncated, info)``. ``obs`` is
        ``list[Obs]`` when ``return_all_frames=True`` (full per-step
        trajectory), or just the final ``Obs`` dict when False (default).
        ``terminated`` / ``truncated`` carry shape ``[chunk_size]`` after
        the leading env dim is stripped — the agent reduces across the
        chunk itself.
        """
        assert not self._done, "chunk_step called after episode done"
        obs_list = []
        rewards = []
        terms = []
        truncations = []
        infos = []

        # RLinF's vectorized chunk_step always executes the entire requested
        # chunk.  LIBERO-PRO rejects any step after the task has terminated,
        # so a success in the middle of a chunk used to surface as an RPC
        # infrastructure error.  Step locally and stop at the first terminal
        # signal; this preserves the environment's authoritative termination
        # event and still keeps the whole loop inside one RPC.
        for action in np.asarray(actions):
            obs, reward, terminated, truncated, info = self.step(action)
            reward_value = float(np.asarray(reward).max())
            features = extract_libero_critic_features(
                obs,
                step_index=self._step_index,
                reward=reward_value,
                terminated=bool(np.asarray(terminated).any()),
                truncated=bool(np.asarray(truncated).any()),
                # This stays in the Critic/audit trace and is never returned
                # to the VLA observation path.
                privileged_state=self._privileged_critic_state(),
                action=action,
                previous_eef=self._critic_previous_eef,
            )
            self._critic_previous_eef = np.asarray(
                obs["states"], dtype=np.float64
            ).reshape(-1)[:3].copy()
            action_value = np.asarray(action, dtype=np.float64).tolist()
            self._audit_trace.append(
                {
                    "step_index": self._step_index,
                    "action": action_value,
                    "action_sha256": canonical_sha256(action_value),
                    "state": features,
                    "observation_sha256": canonical_sha256(features),
                    "reward": reward_value,
                    "libero_terminated": bool(np.asarray(terminated).any()),
                    "truncated": bool(np.asarray(truncated).any()),
                    "proposal_rule_ids": [],
                }
            )
            obs_list.append(obs)
            rewards.append(reward)
            terms.append(terminated)
            truncations.append(truncated)
            infos.append(info)
            if self._done:
                break

        if not obs_list:
            raise ValueError("chunk_step requires at least one action")
        rew = np.asarray(rewards)
        term = np.asarray(terms, dtype=bool)
        trunc = np.asarray(truncations, dtype=bool)
        info = {
            "per_step": infos,
            "executed_horizon": len(obs_list),
        }
        obs_field = obs_list if return_all_frames else obs_list[-1]
        return (
            obs_field,
            rew,
            term,
            trunc,
            info,
        )

    def raw_obs(self) -> dict:
        return _to_numpy_tree(self._env.current_raw_obs[self._env_idx])

    def privileged_contacts(
        self,
        *,
        include_all_contacts: bool = False,
        max_contacts: int = 64,
    ) -> dict:
        worker = self._env.env.workers[self._env_idx]
        return _to_numpy_tree(
            worker.env_call(
                "rpent_privileged_contacts",
                kwargs={
                    "include_all_contacts": bool(include_all_contacts),
                    "max_contacts": int(max_contacts),
                },
                target="self",
            )
        )

    def privileged_semantic_joint_plan(
        self,
        *,
        entity: str,
        joint: str,
        direction: str,
    ) -> dict[str, Any]:
        """Return an internal semantic fixture plan from MuJoCo geometry.

        This endpoint is intentionally separate from the Critic state plane:
        it is consumed only by an audited local recovery primitive and never
        serialized into Actor/VLA observations or Role1 context.
        """

        worker = self._env.env.workers[self._env_idx]
        return _to_numpy_tree(
            worker.env_call(
                "rpent_privileged_semantic_joint_plan",
                kwargs={
                    "entity": str(entity),
                    "joint": str(joint),
                    "direction": str(direction),
                },
                target="self",
            )
        )

    def _privileged_critic_state(
        self, *, reset_tracker: bool = False
    ) -> dict[str, Any]:
        workers = getattr(getattr(self._env, "env", None), "workers", None)
        if workers is None:
            return {
                "privileged.available": False,
                "privileged.task.semantic_available": False,
            }
        worker = workers[self._env_idx]
        return _to_numpy_tree(
            worker.env_call(
                "rpent_privileged_critic_state",
                kwargs={"reset_tracker": bool(reset_tracker)},
                target="self",
            )
        )

    def privileged_critic_state(
        self, *, reset_tracker: bool = False
    ) -> dict[str, Any]:
        """Read the Critic-only semantic sidecar without advancing the env."""

        return self._privileged_critic_state(reset_tracker=reset_tracker)

    def get_env_meta(self) -> dict:
        """Return the meta info this server was launched with. """
        return dict(self._meta)

    def render_camera(
        self,
        camera_name: str = "agentview",
        height: int = 1024,
        width: int = 1024,
        depth: bool = False,
    ):
        return _to_numpy_tree(
            self._env.render_camera(
                camera_name=camera_name,
                height=height,
                width=width,
                depth=depth,
            )
        )

    def get_camera_meta(
        self,
        camera_name: str = "agentview",
        height: int = 256,
        width: int = 256,
    ) -> dict | None:
        return _to_numpy_tree(
            self._env.get_camera_meta(
                camera_name=camera_name, height=height, width=width
            )
        )

    def get_task_language(self) -> str | None:
        return self._env.task_descriptions[self._env_idx]

    def cached_image(self) -> np.ndarray | None:
        cached = getattr(self._env, "_cached_full_image", None)
        if cached is None:
            return None
        return cached.cpu().numpy() if hasattr(cached, "cpu") else np.asarray(cached)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--transport", choices=["socket", "http"], default="http")
    p.add_argument("--host", type=str, default="127.0.0.1")
    p.add_argument("--port", type=int, default=0)
    p.add_argument("--suite", type=str, default="libero_spatial")
    p.add_argument("--task", type=int, default=9)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-episode-steps", type=int, default=10000)
    p.add_argument("--parent-watch", action="store_true",
                   help="watch parent process via stdin pipe and exit when it dies")
    p.add_argument("--cuda-device", type=int, default=None,
                   help="GPU device to pin MuJoCo EGL rendering and the torch "
                        "default device to (physical CUDA ordinal).")
    args = p.parse_args()

    if args.cuda_device is not None:
        # Deliberately do NOT set CUDA_VISIBLE_DEVICES. robosuite (imported
        # transitively via libero) asserts at import time that
        # ``MUJOCO_EGL_DEVICE_ID in CUDA_VISIBLE_DEVICES`` (substring check),
        # which assumes the EGL index equals the CUDA ordinal and crashes on
        # multi-GPU boxes where the EGL order differs. That assertion is gated
        # on ``CUDA_VISIBLE_DEVICES != ""``, so leaving it unset skips it in
        # both this process and the multiprocessing-spawned render workers
        # (which inherit the env). Pin the two backends directly instead:
        #   - MuJoCo render device <- MUJOCO_EGL_DEVICE_ID (configure_egl_device)
        #   - torch default device  <- torch.cuda.set_device(N)
        prev = os.environ.get("CUDA_VISIBLE_DEVICES")
        if prev is not None:
            logger.warning(
                "CUDA_VISIBLE_DEVICES=%s is set; clearing it and pinning via "
                "MUJOCO_EGL_DEVICE_ID + torch.cuda.set_device(--cuda-device=%s) "
                "instead (robosuite's CVD assertion is incompatible with EGL<->CUDA mapping)",
                prev, args.cuda_device,
            )
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        from rpent.utils.egl import configure_egl_device
        configure_egl_device(args.cuda_device)
        import torch
        torch.cuda.set_device(args.cuda_device)

    raw_env = make_env(args.task, args.seed, suite_name=args.suite,
                       max_episode_steps=args.max_episode_steps)
    facade = LiberoEnvFacade(
        raw_env,
        meta={
            "suite": args.suite,
            "task": args.task,
            "seed": args.seed,
            "max_episode_steps": args.max_episode_steps,
        },
    )
    facade.serve(transport=args.transport, host=args.host, port=args.port,
                 parent_watch=args.parent_watch)


if __name__ == "__main__":
    main()
