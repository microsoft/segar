__copyright__ = (
    "Copyright (c) Microsoft Corporation and Mila - Quebec AI Institute"
)
__license__ = "MIT"
"""Module for reward functions

"""

from segar.factors import Alive


def dead_reward_fn(object_state: dict, reward: float = -100.0) -> float:
    """Reward for when an object is dead.

    :param object_state: state dictionary for an object.
    :param reward: reward used when object is `dead`.
    :return: reward
    """
    is_dead = not object_state[Alive]
    if is_dead:
        rew = reward
    else:
        rew = 0.0

    return rew


def l2_distance_reward_fn(distance: float) -> float:
    # Distance reward is tricky: can't do it directly from states
    # because sim owns scaling
    return -(distance ** 2)
