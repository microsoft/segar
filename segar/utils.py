import numpy as np


def check_action(action):
    assert isinstance(action, np.ndarray) and action.shape == (2,) and not np.isnan(action)

def append_dict(d, update_d):
    """Appends new entries from a dictionary to an existing one.
    :param d: Dictionary to update.
    :param update_d: Dictionary to update with.
    :return: None
    """
    for k, v in update_d.items():
        if k in d.keys():
            d[k].append(v)
        else:
            d[k] = [v]


def average_dict(d):
    d_ = {}
    for k, v in d.items():
        d_[k] = sum(d[k]) / float(len(d[k]))
    return d_