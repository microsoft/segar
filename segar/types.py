__author__ = "R Devon Hjelm"
__copyright__ = "Copyright (c) Microsoft Corporation and Mila - Quebec AI " \
                "Institute"
__license__ = "MIT"
"""Reusable types.

"""

__all__ = ('ThingID', 'Time')

from typing import Union


ThingID = Union[int, str]


class Time(float):
    pass
