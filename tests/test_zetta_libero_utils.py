import numpy as np

from zetta.envs.libero.utils import distribute_reset_state_ids_round_robin
from zetta.envs.libero.vector_env import ShArray


def test_reset_states_are_distributed_without_duplication() -> None:
    states = np.arange(7, dtype=np.int64)
    actual = distribute_reset_state_ids_round_robin(states, total_num_processes=3)
    assert sorted(actual[actual >= 0].tolist()) == states.tolist()


def test_shared_array_round_trip() -> None:
    shared = ShArray(np.dtype("float32"), (2, 2))
    value = np.arange(4, dtype=np.float32).reshape(2, 2)
    shared.save(value)
    np.testing.assert_array_equal(shared.get(), value)
