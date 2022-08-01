from __future__ import annotations

__copyright__ = "Copyright (c) Microsoft Corporation and Mila - Quebec AI Institute"
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
from typing import TypeVar, Union, Tuple

import numpy as np

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

    def __init_subclass__(cls, /, default=None, range: Tuple = None, **kwargs):
        if range is not None and not(isinstance(range, (list, tuple))) and len(range) != 2:
            raise TypeError(f'Range must be of form [low, high], got {range}.')
        cls._range = range
        super().__init_subclass__(default=default, **kwargs)

    @Factor.value.setter
    def value(self, value):
        if self._allow_in_place or not self._protected_in_place:
            if hasattr(value, 't'):
                alias = value
                if alias.t == self.t:
                    self._value = alias.value
                else:
                    raise ValueError(
                        'Cannot instantiate factor alias directly from a '
                        'different type of factor unless the factor '
                        'value is of the same type.')
            else:
                self._value = self.new_value(value)
        else:
            raise ValueError('Factor in-place operations are protected.')

        if self._range is not None:
            self._value = self._t(np.clip(self._value, self._range[0], self._range[1]))

    def __neg__(self) -> NumericFactor:
        return self.__class__(-self.value)

    def __add__(self, other: Union[NumericFactor, T]) -> Union[NumericFactor, T]:
        val = self.value + Factor._get_value(other)
        if isinstance(other, self.__class__):
            return self.__class__(val)
        else:
            return val

    def __radd__(self, other: Union[NumericFactor, T]) -> NumericFactor:
        return self.__class__(self.value + Factor._get_value(other))

    def __mul__(self, other: Union[NumericFactor, T]) -> Union[NumericFactor, T]:
        val = self.value * Factor._get_value(other)
        if isinstance(other, self.__class__):
            return self.__class__(val)
        else:
            return val

    def __rmul__(self, other: Union[NumericFactor, T]) -> Union[NumericFactor, T]:
        val = self.value * Factor._get_value(other)
        if isinstance(other, self.__class__):
            return self.__class__(val)
        else:
            return val

    def __sub__(self, other: Union[NumericFactor, T]) -> Union[NumericFactor, T]:
        val = self.value - Factor._get_value(other)
        if isinstance(other, self.__class__):
            return self.__class__(val)
        else:
            return val

    def __rsub__(self, other: Union[NumericFactor, T]) -> NumericFactor:
        return self.__class__(Factor._get_value(other) - self.value)

    def __truediv__(self, other: Union[NumericFactor, T]) -> Union[NumericFactor, T]:
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

    def __rtruediv__(self, other: Union[NumericFactor, T]) -> Union[NumericFactor, T]:
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

    def __pow__(self, exp, mod=None) -> T:
        return pow(self.value, exp, mod)

    def __abs__(self):
        return abs(self.value)

    def sqrt(self) -> float:
        return math.sqrt(self.value)

    def rint(self) -> int:
        return round(self.value)

    def abs(self) -> T:
        return abs(self.value)

    def cos(self) -> float:
        return float(np.cos(self.value))

    def sin(self) -> float:
        return float(np.sin(self.value))


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
