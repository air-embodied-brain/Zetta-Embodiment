"""RoboTwin 2.0 environment, adapted from RLinf's ``envs/robotwin/robotwin_env.py``.

The class shape is kept verbatim so it stays diffable against upstream: the same
constructor signature every rlinf env family shares
(``cfg, num_envs, seed_offset, total_num_processes, worker_info``), the same
``reset``/``step``/``chunk_step``/``update_reset_state_ids``/``is_start`` surface.

Five deliberate deviations from upstream, all of them forced:

1. RLinf package imports are replaced with Zetta-local ones
   (``rlinf.envs.utils`` -> :mod:`zetta.compat.tensors` and
   :mod:`zetta.envs.robotwin.utils`; ``rlinf.envs.robotwin.seed_utils`` ->
   the same local utils module).
2. ``center_crop_image`` no longer goes through TensorFlow -- see
   :mod:`zetta.envs.robotwin.utils`.
3. The process-global ``mp.set_start_method("spawn", force=True)`` becomes
   :func:`~zetta.envs.robotwin.utils.ensure_spawn_start_method`, which is
   idempotent and warns when it actually overrides something.
4. ``omegaconf`` is imported lazily and ``task_config`` may be a plain dict, so
   the class is constructible (and testable) without OmegaConf in the way.
5. Upstream's ``sample_action_space`` is **dropped**: it reads ``self.horizon``,
   which is never assigned anywhere in the class, so calling it always raises
   ``AttributeError``. Nothing in the Rollout Runtime path uses it.
6. ``_extract_obs_image`` additionally returns ``wrist_image_names``. Upstream
   stacks whichever wrist frames happen to be present, so index 0 means "left
   wrist" only when a left wrist exists -- with a right-wrist-only config it
   silently means the right one. The Runtime maps left and right to *different*
   ``Observation`` fields (D1), and getting that backwards is a silent
   wrong-data bug, so the stack is made self-describing. Purely additive: the
   existing four keys are unchanged.

**Observation shapes**, as measured against RoboTwin
``0008ae6800df9f75fc8de7098bacb01735fd8fd2`` (``adjust_bottle``, aloha-agilex):
``full_image`` / ``left_wrist_image`` / ``right_wrist_image`` are
``(240, 320, 3)`` uint8, ``state`` is ``(14,)`` **float64**, ``instruction`` is a
string. ``center_crop=True`` is what brings frames to ``(224, 224, 3)``.
``states`` is left float64 here to stay faithful to upstream; the float32
normalisation the Runtime's ``Observation``/``obs_schema_digest`` requires
happens in ``rollout_runtime/backends/rlinf_robotwin.py``, which is where the
schema contract lives.

``chunk_step`` submits the **whole chunk** in one ``venv.step`` call and returns
an ``obs_list`` of length 1 regardless of the chunk length -- this is the sole
``final_only`` family in the codebase and the reason
``core/env_execution.py::normalize_chunk_outcome`` exists.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any, Optional, Union

import gymnasium as gym
import numpy as np
import torch

from zetta.compat.tensors import list_of_dict_to_dict_of_list
from zetta.envs.robotwin.utils import (
    center_crop_image,
    ensure_spawn_start_method,
    partition_success_seeds,
)

__all__ = ["RoboTwinEnv"]


def _as_container(value: Any) -> Any:
    """Return a plain Python container for an OmegaConf node or a dict.

    Args:
        value: An OmegaConf ``DictConfig``/``ListConfig``, or an already-plain
            value.

    Returns:
        The plain-Python equivalent; ``value`` unchanged when it is not an
        OmegaConf node.
    """
    try:
        from omegaconf import OmegaConf
    except ImportError:
        return value
    if OmegaConf.is_config(value):
        return OmegaConf.to_container(value, resolve=True)
    return value


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    """Read ``key`` from an OmegaConf node, a mapping, or an object.

    Args:
        cfg: The configuration holder.
        key: Attribute or item name.
        default: Value to return when the key is absent.

    Returns:
        The configured value, or ``default``.
    """
    getter = getattr(cfg, "get", None)
    if callable(getter):
        try:
            return getter(key, default)
        except TypeError:
            pass
    return getattr(cfg, key, default)


def _ensure_robotwin_importable(repository_root: str) -> None:
    """Put the RoboTwin repository root on ``sys.path`` if it is not already.

    RoboTwin ships two importable trees at its root: the ``robotwin`` package
    (``robotwin.envs.vector_env``) and a bare top-level ``envs`` package that
    ``vector_env`` itself imports with ``from envs import *``. Both need the
    repository root on the path, so adding it once covers the pair.

    A no-op when ``robotwin`` already imports, so an operator who has exported
    ``PYTHONPATH`` the upstream way keeps exactly the behaviour they set up.

    Args:
        repository_root: The RoboTwin checkout root (the configured
            ``assets_path``).
    """
    import importlib.util

    if importlib.util.find_spec("robotwin") is not None:
        return
    root = os.path.abspath(repository_root)
    if os.path.isdir(os.path.join(root, "robotwin")) and root not in sys.path:
        sys.path.insert(0, root)


class RoboTwinEnv(gym.Env):
    """RoboTwin 2.0 bimanual environment wrapper."""

    def __init__(
        self,
        cfg,
        num_envs,
        seed_offset,
        total_num_processes,
        worker_info,
        record_metrics=True,
    ):
        """Build the vectorised RoboTwin environment.

        Args:
            cfg: Env configuration (``seed``, ``auto_reset``, ``group_size``,
                ``assets_path``, ``seeds_path``, ``task_config``, ...).
            num_envs: Number of lanes this instance owns.
            seed_offset: This worker's index, used for both the seed and the
                success-seed partition.
            total_num_processes: Number of workers sharing the seed pool.
            worker_info: Opaque worker metadata, carried for parity with the
                other rlinf families.
            record_metrics: Whether to accumulate success/return metrics.
        """
        env_seed = cfg.seed
        self.seed = env_seed + seed_offset
        self.base_seed = env_seed
        self.num_envs = num_envs
        self.seed_offset = seed_offset
        self.total_num_processes = total_num_processes
        self.worker_info = worker_info
        self.auto_reset = cfg.auto_reset
        self.use_rel_reward = cfg.use_rel_reward
        self.ignore_terminations = cfg.ignore_terminations

        self.group_size = cfg.group_size
        self.num_group = self.num_envs // self.group_size
        self.use_fixed_reset_state_ids = cfg.use_fixed_reset_state_ids
        self.use_custom_reward = cfg.use_custom_reward

        self.video_cfg = _cfg_get(cfg, "video_cfg")

        self.cfg = cfg
        self.record_metrics = record_metrics
        self._is_start = True

        self.task_name = cfg.task_config.task_name

        self.center_crop = _cfg_get(cfg, "center_crop", False)
        self._init_reset_state_ids()

        self._init_env()

        self.prev_step_reward = torch.zeros(
            self.num_envs, dtype=torch.float32, device=self.device
        )
        # Upstream only creates `_elapsed_steps` under `record_metrics`, yet
        # `step`/`chunk_step` read it unconditionally. Always create it: the
        # truncation check is not a metric.
        self._elapsed_steps = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        if self.record_metrics:
            self._init_metrics()

    def _init_env(self):
        """Start the RoboTwin subprocess vector env.

        ``ASSETS_PATH`` must point at the **RoboTwin repository root**, not at
        its ``assets/`` subdirectory: RoboTwin joins ``assets/...`` onto it
        internally (``envs/utils/rand_create_cluttered_actor.py``). This is
        undocumented upstream and gives a confusing ``.../assets/assets/...``
        ``FileNotFoundError`` when set to the subdirectory.

        That same root is also where the importable ``robotwin`` package lives,
        so it is put on ``sys.path`` when the package is not already importable.
        Upstream leaves this to the operator (RLinf's run scripts export
        ``ROBOTWIN_PATH`` into ``PYTHONPATH``), which means one path has to be
        configured twice and getting it wrong surfaces as a bare
        ``ModuleNotFoundError`` several frames inside pool construction. The
        config already knows the root; there is no reason to ask for it again.
        """
        ensure_spawn_start_method()
        os.environ["ASSETS_PATH"] = self.cfg.assets_path
        _ensure_robotwin_importable(self.cfg.assets_path)

        from robotwin.envs.vector_env import VectorEnv

        env_seeds = self.reset_state_ids.tolist()

        self.venv = VectorEnv(
            task_config=_as_container(self.cfg.task_config),
            n_envs=self.num_envs,
            env_seeds=env_seeds,
        )

    @property
    def device(self):
        """Device the metric/reward tensors live on.

        Returns:
            ``cuda`` when available, else ``cpu``. RoboTwin renders through
            SAPIEN and therefore needs a GPU in practice, which is why the
            family declares ``needs_accelerator_override=True``.
        """
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @property
    def elapsed_steps(self):
        """Per-lane step counter for the current episode.

        Returns:
            A ``[num_envs]`` long tensor.
        """
        return self._elapsed_steps

    @property
    def is_start(self):
        """Whether no reset has happened yet.

        Returns:
            ``True`` until the first ``reset``.
        """
        return self._is_start

    @is_start.setter
    def is_start(self, value):
        """Set the pre-first-reset flag.

        Args:
            value: The new flag value.
        """
        self._is_start = value

    def _init_metrics(self):
        """Allocate the success/return metric accumulators."""
        self.success_once = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool
        )
        self.fail_once = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool
        )
        self.returns = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.float32
        )

    def _reset_metrics(self, env_idx=None):
        """Clear metric state for some or all lanes.

        Args:
            env_idx: Lane indices to clear; ``None`` clears every lane.
        """
        if env_idx is not None:
            mask = torch.zeros(self.num_envs, dtype=bool, device=self.device)
            mask[env_idx] = True
            self.prev_step_reward[mask] = 0.0
            self._elapsed_steps[env_idx] = 0
            if self.record_metrics:
                self.success_once[mask] = False
                self.fail_once[mask] = False
                self.returns[mask] = 0
        else:
            self.prev_step_reward[:] = 0
            self._elapsed_steps[:] = 0
            if self.record_metrics:
                self.success_once[:] = False
                self.fail_once[:] = False
                self.returns[:] = 0.0

    def _record_metrics(self, step_reward, infos):
        """Fold this step's reward and success flag into the episode metrics.

        Args:
            step_reward: Per-lane reward for the step just executed.
            infos: The step's info dict; mutated to carry ``episode``.

        Returns:
            The same ``infos`` dict, with an ``episode`` sub-dict added.
        """
        episode_info = {}
        self.returns += step_reward
        if "success" in infos:
            if isinstance(infos["success"], list):
                infos["success"] = torch.as_tensor(
                    np.array(infos["success"]).reshape(-1), device=self.device
                )
            self.success_once = self.success_once | infos["success"]
            episode_info["success_once"] = self.success_once.clone()
        episode_info["return"] = self.returns.clone()
        episode_info["episode_len"] = self.elapsed_steps.clone()
        episode_info["reward"] = episode_info["return"] / episode_info["episode_len"]
        infos["episode"] = episode_info
        return infos

    def center_and_crop(self, image, center_crop=False):
        """Normalise one raw camera frame to an RGB uint8 array.

        Args:
            image: Raw camera frame.
            center_crop: Whether to apply the centre crop + resize to 224x224.

        Returns:
            An ``HxWx3`` uint8 array; ``(224, 224, 3)`` when ``center_crop``.
        """
        array = np.asarray(image)
        if center_crop:
            return center_crop_image(array)
        from PIL import Image

        return np.asarray(Image.fromarray(array).convert("RGB"))

    def _extract_obs_image(self, raw_obs):
        """Project RoboTwin's per-lane dicts onto the shared 4-key layout.

        RoboTwin exposes three **separate** camera keys (``full_image``,
        ``left_wrist_image``, ``right_wrist_image``); upstream stacks the two
        wrists into a single ``[B, n, H, W, C]`` ``wrist_images`` tensor, with
        ``n`` in ``{1, 2}``, or ``None`` when ``collect_wrist_camera`` is off.
        That stacking is preserved here so this file stays diffable; the
        Runtime adapter unstacks it again to fill ``Observation.wrist_image``
        (left) and ``Observation.extra_view_images[0]`` (right).

        Args:
            raw_obs: List of per-lane observation dicts from ``venv``.

        Returns:
            Dict with ``main_images``, ``wrist_images``, ``states`` and
            ``task_descriptions``.
        """
        batch_images = []
        batch_wrist_images = []
        batch_states = []
        batch_instructions = []
        wrist_names: list[str] = []
        for obs in raw_obs:
            batch_images.append(
                self.center_and_crop(obs["full_image"], center_crop=self.center_crop)
            )
            wrist_images = []
            wrist_names = []
            if "left_wrist_image" in obs and obs["left_wrist_image"] is not None:
                wrist_images.append(
                    self.center_and_crop(
                        obs["left_wrist_image"], center_crop=self.center_crop
                    )
                )
                wrist_names.append("left_wrist_image")
            if "right_wrist_image" in obs and obs["right_wrist_image"] is not None:
                wrist_images.append(
                    self.center_and_crop(
                        obs["right_wrist_image"], center_crop=self.center_crop
                    )
                )
                wrist_names.append("right_wrist_image")
            if len(wrist_images) > 0:
                batch_wrist_images.append(
                    torch.stack([torch.from_numpy(img) for img in wrist_images])
                )
            batch_states.append(obs["state"])
            batch_instructions.append(obs["instruction"])

        batch_images = torch.stack([torch.from_numpy(img) for img in batch_images])
        if len(batch_wrist_images) > 0:
            batch_wrist_images = torch.stack(batch_wrist_images)
        else:
            batch_wrist_images = None
        batch_states = torch.stack(
            [torch.from_numpy(np.asarray(state)) for state in batch_states]
        )

        return {
            "main_images": batch_images,
            "wrist_images": batch_wrist_images,
            "states": batch_states,
            "task_descriptions": batch_instructions,
            # Additive, Zetta-only: names the stacked wrist frames in order, so
            # the Runtime adapter never has to guess that index 0 is the left
            # wrist. Empty when `wrist_images` is None.
            "wrist_image_names": list(wrist_names)
            if batch_wrist_images is not None
            else [],
        }

    def _calc_step_reward(self, terminations):
        """Derive the custom reward from the termination flag.

        Args:
            terminations: Per-lane termination flags.

        Returns:
            Absolute reward, or its difference from the previous step when
            ``use_rel_reward``.
        """
        reward = self.cfg.reward_coef * terminations

        reward_diff = reward - self.prev_step_reward
        self.prev_step_reward = reward

        if self.use_rel_reward:
            return reward_diff
        else:
            return reward

    def _cal_chunk_rewards(self, step_reward, chunk_step, terminations, infos):
        """Spread a chunk's single reward across the chunk's step slots.

        RoboTwin only scores once per submitted chunk, so the reward lands on
        the chunk's final slot. ``n_steps_to_run`` is pinned to zero upstream
        (the real source is commented out there), which makes ``start_idx``
        the last index; that behaviour is preserved.

        Args:
            step_reward: Per-lane reward for the chunk.
            chunk_step: Number of actions submitted in the chunk.
            terminations: Per-lane termination flags.
            infos: The chunk's info dict (unused; kept for signature parity).

        Returns:
            A ``[num_envs, chunk_step]`` reward tensor.
        """
        n_steps_to_run = np.array([[0] for _ in range(self.num_envs)])

        n_steps_to_run = torch.as_tensor(
            np.array(n_steps_to_run).reshape(-1), device=self.device
        )
        chunk_rewards = torch.zeros(self.num_envs, chunk_step, device=self.device)
        for env_id in range(self.num_envs):
            steps_left = n_steps_to_run[env_id]
            reward = step_reward[env_id]
            start_idx = chunk_step - steps_left - 1

            if terminations[env_id] and start_idx > 0:
                if self.use_rel_reward:
                    chunk_rewards[env_id, start_idx] = reward
                else:
                    chunk_rewards[env_id, start_idx:] = reward

        return chunk_rewards

    def reset(
        self,
        env_idx: Optional[Union[int, list[int]]] = None,
        env_seeds=None,
    ):
        """Reset some or all lanes.

        This is the ``env_idx_env_seeds`` reset signature declared for the
        family in ``rollout_runtime/core/env_registry.py``.

        Args:
            env_idx: Lane indices to reset; ``None`` resets every lane.
            env_seeds: Explicit seeds; ``None`` uses the current
                ``reset_state_ids``.

        Returns:
            ``(extracted_obs, infos)``.
        """
        if self._is_start:
            self._is_start = False

        env_seeds = self.reset_state_ids.tolist() if env_seeds is None else env_seeds

        self.venv.reset(env_idx=env_idx, env_seeds=env_seeds)
        raw_obs = self.venv.get_obs()
        infos = {}

        self._reset_metrics(env_idx)

        extracted_obs = self._extract_obs_image(raw_obs)

        return extracted_obs, infos

    def step(
        self, actions: Union[torch.Tensor, np.ndarray, dict] = None, auto_reset=True
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """Advance every lane by the supplied actions.

        Args:
            actions: ``[num_envs, horizon, action_dim]`` (or
                ``[num_envs, action_dim]``, which is promoted to horizon 1).
            auto_reset: Whether to honour ``cfg.auto_reset`` for finished lanes.

        Returns:
            ``(extracted_obs, step_reward, terminations, truncations, infos)``.
        """
        if actions is None:
            assert self._is_start, "Actions must be provided after the first reset."

        if isinstance(actions, torch.Tensor):
            actions = actions.cpu().numpy()
        elif isinstance(actions, dict):
            actions = actions.get("actions", actions)

        # [n_envs, horizon, action_dim]
        if len(actions.shape) == 2:
            # [n_envs, action_dim] -> [n_envs, 1, action_dim]
            actions = actions[:, None, :]

        raw_obs, step_reward, terminations, truncations, info_list = self.venv.step(
            actions
        )
        extracted_obs = self._extract_obs_image(raw_obs)
        infos = list_of_dict_to_dict_of_list(info_list)

        terminations, truncations = self._as_flag_tensors(terminations, truncations)
        step_reward = self._resolve_step_reward(step_reward, terminations)

        self._elapsed_steps += actions.shape[1]
        truncations = self._apply_horizon_truncation(truncations)

        infos = self._record_metrics(step_reward, infos)
        terminations = self._apply_ignore_terminations(terminations, infos)

        dones = torch.logical_or(terminations, truncations)

        _auto_reset = auto_reset and self.auto_reset
        if dones.any() and _auto_reset:
            extracted_obs, infos = self._handle_auto_reset(dones, extracted_obs, infos)

        return extracted_obs, step_reward, terminations, truncations, infos

    def chunk_step(self, chunk_actions):
        """Submit a whole action chunk and observe **only the final frame**.

        This is the ``final_only`` behaviour: one ``venv.step`` call for the
        entire chunk, so ``obs_list`` has length 1 no matter how long the chunk
        is. Reward/termination tensors are chunk-shaped but only their last
        column is populated.

        Args:
            chunk_actions: ``[num_envs, chunk_step, action_dim]``.

        Returns:
            ``(obs_list, chunk_rewards, chunk_terminations, chunk_truncations,
            infos_list)``, where ``obs_list`` and ``infos_list`` both have
            length 1.
        """
        if isinstance(chunk_actions, torch.Tensor):
            chunk_actions = chunk_actions.cpu().numpy()

        # chunk_actions: [num_envs, chunk_step, action_dim]
        num_envs = chunk_actions.shape[0]
        chunk_step = chunk_actions.shape[1]
        obs_list = []
        infos_list = []

        raw_obs, step_reward, terminations, truncations, info_list = self.venv.step(
            chunk_actions
        )
        extracted_obs = self._extract_obs_image(raw_obs)
        infos = list_of_dict_to_dict_of_list(info_list)
        obs_list.append(extracted_obs)
        infos_list.append(infos)

        terminations, truncations = self._as_flag_tensors(terminations, truncations)
        step_reward = self._resolve_step_reward(step_reward, terminations)

        chunk_rewards = self._cal_chunk_rewards(
            step_reward, chunk_step, terminations, infos
        )

        self._elapsed_steps += chunk_actions.shape[1]
        truncations = self._apply_horizon_truncation(truncations)

        infos = self._record_metrics(step_reward, infos)
        terminations = self._apply_ignore_terminations(terminations, infos)

        past_dones = torch.logical_or(terminations, truncations)
        if past_dones.any() and self.auto_reset:
            obs_list[-1], infos_list[-1] = self._handle_auto_reset(
                past_dones, obs_list[-1], infos_list[-1]
            )

        chunk_terminations = torch.zeros((num_envs, chunk_step), dtype=bool)
        chunk_terminations[:, -1] = terminations

        chunk_truncations = torch.zeros((num_envs, chunk_step), dtype=bool)
        chunk_truncations[:, -1] = truncations

        return (
            obs_list,
            chunk_rewards,
            chunk_terminations,
            chunk_truncations,
            infos_list,
        )

    def _as_flag_tensors(self, terminations, truncations):
        """Normalise the venv's list-or-tensor flags into device tensors.

        Args:
            terminations: Per-lane termination flags.
            truncations: Per-lane truncation flags.

        Returns:
            The pair, both as tensors on ``self.device``.
        """
        if isinstance(terminations, list):
            terminations = torch.as_tensor(
                np.array(terminations).reshape(-1), device=self.device
            )
        if isinstance(truncations, list):
            truncations = torch.as_tensor(
                np.array(truncations).reshape(-1), device=self.device
            )
        return terminations, truncations

    def _resolve_step_reward(self, step_reward, terminations):
        """Pick between the custom reward and the environment's own.

        Args:
            step_reward: The venv's reward.
            terminations: Per-lane termination flags.

        Returns:
            The reward tensor to record.
        """
        if self.use_custom_reward:
            return self._calc_step_reward(terminations)
        if isinstance(step_reward, list):
            return torch.as_tensor(
                np.array(step_reward, dtype=np.float32).reshape(-1),
                device=self.device,
            )
        return step_reward

    def _apply_horizon_truncation(self, truncations):
        """OR in the truncation implied by ``max_episode_steps``.

        Args:
            truncations: Per-lane truncation flags.

        Returns:
            The updated truncation flags.
        """
        truncated = self._elapsed_steps >= self.cfg.max_episode_steps
        if truncated.any():
            truncations = torch.logical_or(truncated, truncations)
        return truncations

    def _apply_ignore_terminations(self, terminations, infos):
        """Zero the termination flags when the config asks for fixed horizons.

        Args:
            terminations: Per-lane termination flags.
            infos: Info dict; gains ``episode.success_at_end`` when metrics are
                recorded.

        Returns:
            The updated termination flags.
        """
        if self.ignore_terminations:
            terminations[:] = False
            if self.record_metrics and "success" in infos:
                infos["episode"]["success_at_end"] = infos["success"].clone()
        return terminations

    def _handle_auto_reset(self, dones, extracted_obs, infos):
        """Reset finished lanes and stash their final frame in ``infos``.

        Args:
            dones: Per-lane done flags.
            extracted_obs: The observation about to be replaced.
            infos: The current info dict.

        Returns:
            ``(post_reset_obs, infos)`` where ``infos`` carries
            ``final_observation``/``final_info``.
        """
        final_obs = extracted_obs.copy()
        env_idx = torch.arange(0, self.num_envs, device=self.device)[dones]
        final_info = infos.copy()
        if self.cfg.is_eval:
            self.update_reset_state_ids(env_idx=env_idx)

        extracted_obs, infos = self.reset(env_idx=env_idx.tolist())
        # gymnasium calls it final observation but it really is just o_{t+1} or
        # the true next observation
        infos["final_observation"] = final_obs
        infos["final_info"] = final_info
        infos["_final_info"] = dones
        infos["_final_observation"] = dones
        infos["_elapsed_steps"] = dones
        return extracted_obs, infos

    def offload(self, clear_cache=True):
        """Close the subprocess vector env, if one was built.

        Args:
            clear_cache: Forwarded to ``VectorEnv.close``.
        """
        if hasattr(self, "venv"):
            self.venv.close(clear_cache)

    def _load_success_seeds(self):
        """Read the curated success-seed list for this task, if configured.

        Returns:
            The seed tensor for this worker, or ``None`` when no usable
            ``seeds_path`` is configured.
        """
        seeds_path = _cfg_get(self.cfg, "seeds_path", None)
        if seeds_path is None or not os.path.exists(seeds_path):
            return None
        with open(seeds_path, "r") as handle:
            data = json.load(handle)
        success_seeds = data[self.task_name].get("success_seeds", None)
        if success_seeds is None:
            return None
        return partition_success_seeds(
            torch.as_tensor(success_seeds, dtype=torch.long),
            base_seed=self.base_seed,
            seed_offset=self.seed_offset,
            total_num_processes=self.total_num_processes,
            num_group=self.num_group,
        )

    def _init_reset_state_ids(self):
        """Seed the reset-state schedule from the success-seed list or the RNG."""
        self.success_seeds = self._load_success_seeds()
        self._current_seed_index = 0

        if not hasattr(self, "_generator"):
            self._generator = torch.Generator()
            self._generator.manual_seed(self.seed)
        self.update_reset_state_ids()

    def _next_reset_state_ids(self):
        """Draw the next round of per-group reset seeds.

        Returns:
            A ``[num_envs]`` seed tensor (each group's seed repeated
            ``group_size`` times).
        """
        if self.success_seeds is not None:
            total_seeds = self.success_seeds.numel()
            indices = (
                torch.arange(self.num_group, device=self.success_seeds.device)
                + self._current_seed_index
            ) % total_seeds
            reset_state_ids = self.success_seeds[indices]
            self._current_seed_index = (
                self._current_seed_index + self.num_group
            ) % total_seeds
        else:
            reset_state_ids = torch.randint(
                low=10000,
                high=200000,
                size=(self.num_group,),
                generator=self._generator,
            )
        return reset_state_ids.repeat_interleave(repeats=self.group_size)

    def update_reset_state_ids(self, env_idx=None):
        """Advance the reset-state schedule.

        A no-op once ``use_fixed_reset_state_ids`` is set and a schedule
        already exists -- that is how an eval run keeps every episode on the
        same seed set.

        Args:
            env_idx: Lane indices to refresh; ``None`` refreshes all.
        """
        if self.use_fixed_reset_state_ids and hasattr(self, "reset_state_ids"):
            return

        reset_state_ids = self._next_reset_state_ids()

        if env_idx is not None and hasattr(self, "reset_state_ids"):
            for idx in env_idx:
                self.reset_state_ids[idx] = reset_state_ids[idx]
        else:
            self.reset_state_ids = reset_state_ids

    def check_seeds(self, seeds):
        """Ask the vector env which of ``seeds`` are usable.

        Args:
            seeds: Candidate seeds.

        Returns:
            The vector env's per-seed result.
        """
        return self.venv.check_seeds(seeds)
