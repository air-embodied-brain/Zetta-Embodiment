"""Action adapters derived from RLinf's ``envs/action_utils.py``.

The adapters live here so Zetta does not need to import the main ``rlinf``
package.  The upstream source is kept in ``third_party/RLinf`` for provenance.
"""

from __future__ import annotations

import numpy as np
import torch

_LIBERO_GRIPPER_MODELS = {
    "openvla",
    "openvla_oft",
    "gr00t_n1d6",
    "gr00t_n1d7",
}


def prepare_actions_for_libero(raw_chunk_actions, model_type: str) -> np.ndarray:
    """Convert a policy action chunk to LIBERO's gripper convention."""
    chunk_actions = raw_chunk_actions
    if str(model_type).lower() in _LIBERO_GRIPPER_MODELS:
        chunk_actions[..., -1] = 2 * chunk_actions[..., -1] - 1
        chunk_actions[..., -1] = np.sign(chunk_actions[..., -1]) * -1.0
    return chunk_actions


def prepare_actions_for_maniskill(
    raw_chunk_actions,
    num_action_chunks: int,
    action_dim: int,
    action_scale: float,
    policy: str,
) -> torch.Tensor:
    """Convert a policy action chunk to ManiSkill's seven-DoF action."""
    if "panda" in policy:
        return torch.as_tensor(raw_chunk_actions)

    reshaped_actions = raw_chunk_actions.reshape(-1, action_dim)
    if action_dim != 7:
        raise ValueError(
            f"ManiSkill action conversion requires action_dim=7, got {action_dim}"
        )

    world_vector = np.asarray(reshaped_actions[:, :3]) * action_scale
    rotation_delta = np.asarray(reshaped_actions[:, 3:6]) * action_scale
    open_gripper = np.asarray(reshaped_actions[:, 6:7])

    if policy == "google_robot":
        raise NotImplementedError("google_robot action conversion is not implemented")
    if policy not in {"widowx_bridge", "panda_wristcam"}:
        raise ValueError(f"Unsupported ManiSkill policy: {policy}")

    gripper = 2.0 * (open_gripper > 0.5) - 1.0
    actions = torch.cat(
        [
            torch.tensor(world_vector, dtype=torch.float32),
            torch.tensor(rotation_delta, dtype=torch.float32),
            torch.tensor(gripper, dtype=torch.float32),
        ],
        dim=1,
    )
    return actions.reshape(-1, num_action_chunks, action_dim)


def prepare_actions(
    raw_chunk_actions,
    env_type: str,
    model_type: str,
    num_action_chunks: int,
    action_dim: int,
    action_scale: float = 1.0,
    policy: str = "widowx_bridge",
    **_unused,
) -> torch.Tensor | np.ndarray:
    """Prepare actions for the Zetta embodied backends currently in scope."""
    if isinstance(raw_chunk_actions, torch.Tensor):
        raw_chunk_actions = raw_chunk_actions.detach().cpu().contiguous()
        if raw_chunk_actions.dtype == torch.bfloat16:
            raw_chunk_actions = raw_chunk_actions.float()
        raw_chunk_actions = raw_chunk_actions.numpy()

    normalized_env_type = str(env_type).lower()
    if normalized_env_type == "libero":
        return prepare_actions_for_libero(raw_chunk_actions, model_type)
    if normalized_env_type in {"maniskill", "maniskill_rlt"}:
        return prepare_actions_for_maniskill(
            raw_chunk_actions,
            num_action_chunks,
            action_dim,
            action_scale,
            policy,
        )
    return raw_chunk_actions
