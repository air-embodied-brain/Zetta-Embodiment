"""Tensor helpers derived from RLinf's ``envs/utils.py``."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch


def to_tensor(array: Any, device: str = "cpu"):
    """Recursively map common Python and NumPy values to torch tensors."""
    if array is None:
        return None
    if isinstance(array, dict):
        return {key: to_tensor(value, device=device) for key, value in array.items()}
    if isinstance(array, torch.Tensor):
        result = array.to(device)
    elif isinstance(array, np.ndarray):
        if array.dtype == object:
            return [to_tensor(value, device=device) for value in array]
        if array.dtype == np.uint16:
            array = array.astype(np.int32)
        elif array.dtype == np.uint32:
            array = array.astype(np.int64)
        result = torch.tensor(array).to(device)
    else:
        if isinstance(array, list) and any(value is None for value in array):
            return [to_tensor(value, device=device) for value in array]
        if array and isinstance(array, list) and isinstance(array[0], np.ndarray):
            array = np.array(array)
            if array.dtype == object:
                return [to_tensor(value, device=device) for value in array]
        result = torch.tensor(array, device=device)

    if result.dtype == torch.float64:
        result = result.to(torch.float32)
    return result


def list_of_dict_to_dict_of_list(
    list_of_dict: list[dict[str, Any]],
) -> dict[str, list[Any]]:
    """Convert dictionaries with common keys into lists grouped by key."""
    if not list_of_dict:
        return {}
    output = {key: [] for key in list_of_dict[0]}
    for data in list_of_dict:
        if data.keys() != output.keys():
            raise ValueError("All dictionaries must have the same keys")
        for key, item in data.items():
            output[key].append(item)
    return output
