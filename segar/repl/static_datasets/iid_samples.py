__copyright__ = "Copyright (c) Microsoft Corporation and Mila - Quebec AI Institute"
__license__ = "MIT"
"""Datasets for drawing i.i.d. samples from trajectories.

"""

import numpy as np
from torch.utils.data import Dataset

from segar import get_sim
from segar.mdps import Initialization, Observation


class IIDFromInit(Dataset):
    def __init__(self, inputs: list[np.ndarray], targets: list[np.ndarray]):
        super().__init__()
        self._inputs = inputs
        self._targets = targets

    def __len__(self) -> int:
        return len(self._inputs)

    def __getitem__(self, idx) -> tuple[np.ndarray, np.ndarray]:
        return self._inputs[idx], self._targets[idx]


def create_iid_from_init(
    initializer: Initialization,
    input_observation: Observation,
    target_observation: Observation,
    n_observations: int = 10000,
) -> IIDFromInit:
    sim = get_sim()
    inputs = []
    targets = []

    for i in range(n_observations):
        initializer.sample()
        initializer()
        input_observation.reset()
        inp = input_observation(sim.state)
        target = target_observation(sim.state)
        inp = inp.copy()
        inp = inp.transpose((2, 0, 1))
        inputs.append(inp)
        targets.append(target)

    return IIDFromInit(inputs, targets)
