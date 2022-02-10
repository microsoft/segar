__copyright__ = (
    "Copyright (c) Microsoft Corporation and Mila - Quebec AI Institute"
)
__license__ = "MIT"
"""Additional factors corresponding to global parameters used by the
    environment.

"""

__all__ = (
    "Parameter",
    "MinMass",
    "MaxVelocity",
    "Gravity",
    "FloorFriction",
    "WallDamping",
    "Framerate",
    "DensityScale",
    "Resolution",
    "DoesNotHaveFactor",
)

from typing import TypeVar

import cv2

from segar.factors import NumericFactor


T = TypeVar("T")


class Parameter(NumericFactor[T]):
    pass


class MinMass(Parameter[float], default=1e-1):
    pass


class MaxVelocity(Parameter[float], default=10.0):
    pass


class MinVelocity(Parameter[float], default=1e-4):
    pass


class Gravity(Parameter[float], default=10.0):
    pass


class FloorFriction(Parameter[float], default=0.05):
    pass


class WallDamping(Parameter[float], default=0.025):
    pass


class Framerate(Parameter[int], default=100):
    pass


class DensityScale(Parameter[int], default=1):
    pass


class Resolution(Parameter[int], default=256):
    pass


class DoesNotHaveFactor(Parameter[float], default=-1000):
    pass


_TEXT_FACE = cv2.FONT_HERSHEY_DUPLEX
_TEXT_SCALE = 0.5
_TEXT_THICKNESS = 1
