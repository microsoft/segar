from __future__ import annotations

__copyright__ = (
    "Copyright (c) Microsoft Corporation and Mila - Quebec AI Institute"
)
__license__ = "MIT"
"""Numeric factors (ints and floats).

"""

__all__ = (
    "NumericFactor",
    "Order",
    "Size",
    "Mass",
    "Density",
    "Charge",
    "Magnetism",
    "StoredEnergy",
    "Heat",
    "Friction",
)

import math
from typing import TypeVar, Union

from segar.factors.factors import Factor


T = TypeVar("T")


# Integer and float factors
class NumericFactor(Factor[T], default=0):
    _protected_in_place = (
        "set",
        "__iadd__",
        "__imul__",
        "__isub__",
        "__idiv__",
    )

    __array_priority__ = 1  # For numpy array functionality

    def __neg__(self) -> NumericFactor:
        return self.__class__(-self.value)

    def __add__(
        self, other: Union[NumericFactor, T]
    ) -> Union[NumericFactor, T]:
        val = self.value + Factor._get_value(other)
        if isinstance(other, self.__class__):
            return self.__class__(val)
        else:
            return val

    def __radd__(self, other: Union[NumericFactor, T]) -> NumericFactor:
        return self.__class__(self.value + Factor._get_value(other))

    def __mul__(
        self, other: Union[NumericFactor, T]
    ) -> Union[NumericFactor, T]:
        val = self.value * Factor._get_value(other)
        if isinstance(other, self.__class__):
            return self.__class__(val)
        else:
            return val

    def __rmul__(
        self, other: Union[NumericFactor, T]
    ) -> Union[NumericFactor, T]:
        val = self.value * Factor._get_value(other)
        if isinstance(other, self.__class__):
            return self.__class__(val)
        else:
            return val

    def __sub__(
        self, other: Union[NumericFactor, T]
    ) -> Union[NumericFactor, T]:
        val = self.value - Factor._get_value(other)
        if isinstance(other, self.__class__):
            return self.__class__(val)
        else:
            return val

    def __rsub__(self, other: Union[NumericFactor, T]) -> NumericFactor:
        return self.__class__(Factor._get_value(other) - self.value)

    def __truediv__(
        self, other: Union[NumericFactor, T]
    ) -> Union[NumericFactor, T]:
        val = self.value / Factor._get_value(other)
        if isinstance(other, self.__class__):
            return self.__class__(val)
        else:
            return val

    def __iadd__(self, other: Union[NumericFactor, T]) -> NumericFactor:
        self.value += Factor._get_value(other)
        return self

    def __isub__(self, other: Union[NumericFactor, T]) -> NumericFactor:
        self.value -= Factor._get_value(other)
        return self

    def __imul__(self, other: Union[NumericFactor, T]) -> NumericFactor:
        self.value *= Factor._get_value(other)
        return self

    def __itruediv__(self, other: Union[NumericFactor, T]) -> NumericFactor:
        self.value /= Factor._get_value(other)
        return self

    def __rtruediv__(
        self, other: Union[NumericFactor, T]
    ) -> Union[NumericFactor, T]:
        val = Factor._get_value(other) / self.value
        if isinstance(other, self.__class__):
            return self.__class__(val)
        else:
            return val

    def __gt__(self, other: Union[NumericFactor, T]) -> bool:
        return self.value > Factor._get_value(other)

    def __lt__(self, other: Union[NumericFactor, T]) -> bool:
        return self.value < Factor._get_value(other)

    def __ge__(self, other: Union[NumericFactor, T]) -> bool:
        return self.value >= Factor._get_value(other)

    def __le__(self, other: Union[NumericFactor, T]) -> bool:
        return self.value <= Factor._get_value(other)

    def sqrt(self) -> float:
        return math.sqrt(self.value)

    def rint(self) -> int:
        return round(self.value)

    def abs(self) -> T:
        return abs(self.value)


class Order(NumericFactor[int], default=None):
    pass


class Size(NumericFactor[float], default=1.0):
    pass


class Mass(NumericFactor[float], default=1.0):
    pass


class Density(NumericFactor[float], default=1.0):
    pass


class Charge(NumericFactor[float], default=0.0):
    pass


class Magnetism(NumericFactor[float], default=0.0):
    pass


class StoredEnergy(NumericFactor[float], default=0.0):
    pass


class Heat(NumericFactor[float], default=0.0):
    pass


class Friction(NumericFactor[float], default=0.0):
    pass
