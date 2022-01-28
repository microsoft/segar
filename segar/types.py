"""Reusable types.

"""

__all__ = ('ThingID', 'Time')

from typing import Union


ThingID = Union[int, str]


class Time(float):
    pass
