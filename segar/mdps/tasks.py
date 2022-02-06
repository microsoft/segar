"""Task component of the MDP

The task encapsulates actions, rewards, stopping conditions and
initializations. The main purpose of the task is to define and manage the
semantics of the MDP.

"""

__all__ = ('Task',)

from typing import Optional, Type

from gym.spaces import Box
import numpy as np

from segar import get_sim
from segar.factors import ID, Factor, Noise
from segar.sim import Simulator
from segar.things import Entity
from .initializations import Initialization


class Task:
    """Task class.

    Embodies actions, rewards, and stopping conditions.

    """
    def __init__(self, action_range: tuple[float, float],
                 action_shape: tuple, action_type: type,
                 baseline_action: np.ndarray, initialization: Initialization):
        """

        :param action_range: Range of legal actions.
        :param action_shape: Shape of the actions.
        :param action_type: Type of the actions.
        :param baseline_action: Action to add as baseline to all actions.
        :param initialization: Initialization object for the task.
        """

        self._sim = None
        self._action_space = Box(
            action_range[0],
            action_range[1],
            shape=action_shape,
            dtype=action_type)

        self._initialization = initialization
        self._baseline_action = baseline_action

    @property
    def sim(self) -> Simulator:
        if self._sim is None:
            return get_sim()
        return self._sim

    def set_sim(self, sim: Simulator) -> None:
        self._sim = sim
        self._initialization.set_sim(self._sim)

    @property
    def action_space(self) -> Box:
        return self._action_space

    def check_action(self, action: np.ndarray) -> bool:
        """Checks if an action is valid.

        This method should be overridden for custom tasks to be useful. For
        this abstract class, just checks if the action is not None.

        :param action: Action to check.
        :return: True if the action is valid.
        """
        return action is not None

    def demo_action(self) -> np.ndarray:
        """Generate an action used for demos

        :return: np.array action
        """
        raise NotImplementedError

    def apply_baseline(self, action: np.ndarray) -> np.ndarray:
        """Applies the baseline action to the action.

        Should be overridden for custom behavior, e.g., when actions do not
        add.

        :param action: Action to apply baseline to.
        :return: New action.
        """
        self.check_action(action)
        action += self._baseline_action
        return action

    def reward(self, state: dict) -> float:
        raise NotImplementedError

    def apply_action(self, action: np.ndarray) -> None:
        """Applies the action to the simulator.

        Must be overridden.

        :param action: Action to apply.
        """
        raise NotImplementedError

    def done(self, state: dict) -> bool:
        """Checks stopping condition.

        Must be overridden.

        :param state: State dictionary.
        :return: True if stopping conditions have been met.
        """
        raise NotImplementedError

    def sample(self) -> None:
        """Samples the task from a random distribution.

        """
        self._initialization.sample()

    def initialize(self, init_things: Optional[list[Entity]] = None) -> None:
        """Initializes the environment and task.

        """
        self._initialization(init_things=init_things)

    @property
    def initial_state(self) -> list[Entity]:
        return self._initialization.initial_state

    def get_dists_from_init(self) -> dict[ID, dict[Type[Factor], Noise]]:
        return self._initialization.get_dists_from_init()

    def results(self, state: dict) -> dict:
        """Returns results to be processed by the MDP.

        :param state: State dictionary to pull results from.
        :return: Dictionary of results.
        """
        return {}
