import torch

from zetta.envs.maniskill.environment import extract_termination_from_info


def test_extract_termination_uses_success_and_fail_flags() -> None:
    info = {
        "success": torch.tensor([True, False]),
        "fail": torch.tensor([False, True]),
    }
    actual = extract_termination_from_info(info, num_envs=2, device="cpu")
    torch.testing.assert_close(actual, torch.tensor([True, True]))
