import numpy as np


def check_action(action):
    assert (isinstance(action, np.ndarray)
            and action.shape == (2,)
            and not np.isnan(action))
