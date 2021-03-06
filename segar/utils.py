__copyright__ = "Copyright (c) Microsoft Corporation and Mila - Quebec AI Institute"
__license__ = "MIT"
"""Utilities for SEGAR

"""
from typing import Any


def append_dict(d: dict[str, list], update_d: dict[str, Any]):
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


def average_dict(d: dict[str, list]) -> dict[str, float]:
    """Averages a dictionary of lists.

    :param d: Dictionary of lists.

    """
    d_ = {}
    for k, v in d.items():
        d_[k] = sum(d[k]) / float(len(d[k]))
    return d_
