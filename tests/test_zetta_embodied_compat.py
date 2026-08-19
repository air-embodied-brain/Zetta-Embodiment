import numpy as np
import torch

from zetta.compat.actions import prepare_actions
from zetta.compat.tensors import list_of_dict_to_dict_of_list, to_tensor


def test_libero_openpi_actions_are_unchanged():
    raw = np.array(
        [[[0.1, -0.2, 0.3, 0.4, -0.5, 0.6, -1.0]]],
        dtype=np.float32,
    )

    actual = prepare_actions(
        raw,
        env_type="libero",
        model_type="openpi",
        num_action_chunks=1,
        action_dim=7,
    )

    np.testing.assert_array_equal(actual, raw)


def test_maniskill_widowx_scales_pose_and_binarizes_gripper():
    raw = torch.tensor(
        [[[1.0, -1.0, 0.5, 0.2, -0.2, 0.1, 0.75]]],
        dtype=torch.float32,
    )

    actual = prepare_actions(
        raw,
        env_type="maniskill",
        model_type="openpi",
        num_action_chunks=1,
        action_dim=7,
        action_scale=0.5,
        policy="widowx_bridge",
    )

    expected = torch.tensor(
        [[[0.5, -0.5, 0.25, 0.1, -0.1, 0.05, 1.0]]],
        dtype=torch.float32,
    )
    torch.testing.assert_close(actual, expected)


def test_tensor_helpers_preserve_nested_shape():
    merged = list_of_dict_to_dict_of_list([{"x": 1}, {"x": 2}])

    assert merged == {"x": [1, 2]}
    assert to_tensor([1.0, 2.0]).shape == (2,)
