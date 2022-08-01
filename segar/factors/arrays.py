from __future__ import annotations

__copyright__ = "Copyright (c) Microsoft Corporation and Mila - Quebec AI Institute"
__license__ = "MIT"

"""Vector-like factors.

"""

__all__ = ("VectorFactor", "Position", "Velocity", "Acceleration", "Force")

from numbers import Number
from typing import TypeVar, Union

import numpy as np
from scipy.linalg import norm

from segar.factors.factors import Factor


T = TypeVar("T")


class VectorFactor(Factor[np.ndarray], default=[0.0, 0.0]):

    __array_priority__ = 1  # For numpy array functionality

    def __eq__(self, other: Union[VectorFactor, T]) -> bool:
        other = self._get_value(other)
        return (self.value == other).all()

    def __ne__(self, other: Union[VectorFactor, T]) -> bool:
        other = self._get_value(other)
        return (self.value != other).any()

    def __neg__(self) -> VectorFactor:
        return self.__class__(-self.value)

    def __add__(self, other: Union[VectorFactor, T, Number]) -> VectorFactor:
        return self.__class__(self.value + Factor._get_value(other))

    def __radd__(self, other: Union[VectorFactor, T, Number]) -> VectorFactor:
        return self.__add__(other)

    def __mul__(self, other: Union[VectorFactor, T, Number]) -> VectorFactor:
        return self.__class__(self.value * Factor._get_value(other))

    def __rmul__(self, other: Union[VectorFactor, T, Number]) -> VectorFactor:
        return self.__class__(self.value * Factor._get_value(other))

    def __sub__(self, other: Union[VectorFactor, T, Number]) -> VectorFactor:
        return self.__class__(self.value - Factor._get_value(other))

    def __truediv__(self, other: Union[VectorFactor, T, Number]) -> VectorFactor:
        other_value = Factor._get_value(other)
        if norm(other_value) == 0:
            raise ValueError(f"Dividing by zero: {self} / {other}.")
        return self.__class__(self.value / other_value)

    def __iadd__(self, other: Union[VectorFactor, T, Number]) -> VectorFactor:
        self.set(self.value + Factor._get_value(other))
        return self

    def __isub__(self, other: Union[VectorFactor, T, Number]) -> VectorFactor:
        self.set(self.value - Factor._get_value(other))
        return self

    def __imul__(self, other: Union[VectorFactor, T, Number]) -> VectorFactor:
        self.set(self.value * Factor._get_value(other))
        return self

    def __idiv__(self, other: Union[VectorFactor, T, Number]) -> VectorFactor:
        self.set(self.value / Factor._get_value(other))
        return self

    def set(self, value: Union[T, Factor], allow_in_place: bool = False):
        # Numpy can do some undesirable things with binary operations and
        # Factors, which are objects. Check dtype here.
        dtype = self.value.dtype
        value_ = Factor._get_value(value)
        if np.isinf(value_).any():
            raise ValueError(f"Attempting to set factor {self} to infinite " f"value {value}.")
        if np.isnan(value_).any():
            raise ValueError(f"Attempting to set factor {self} to NaN " f"value {value}.")
        super().set(value, allow_in_place=allow_in_place)
        if self.value.dtype != dtype:
            raise TypeError(
                f"dtype changed from {dtype} to "
                f"{self.value.dtype}. This is likely due to "
                f"binary operations on Factors and numpy arrays. "
                f"Use the `value` property when performing such "
                f"operations."
            )

    def __getitem__(self, item: int):
        return self.value[item]

    def norm(self) -> float:
        return norm(self.value)

    def unit_vector(self) -> VectorFactor:
        return self.__class__(self.value / self.norm())

    def tangent_vector(self) -> VectorFactor:
        return self.__class__(np.array([-self.value[1], self.value[0]]))

    def dot(self, other: Union[VectorFactor, T, np.ndarray]) -> float:
        other_value = self._get_value(other)
        return np.dot(self.value, other_value)

    def sign(self) -> VectorFactor:
        return self.__class__(np.sign(self.value))

    def abs(self) -> VectorFactor:
        return self.__class__(np.abs(self.value))

    def __iter__(self):
        return iter(self.value.tolist())

    def __hash__(self):
        return hash(id(self))

    def sqrt(self):
        return np.sqrt(self.value.sum())


class Position(VectorFactor, default=[0.0, 0.0]):
    pass


class Velocity(VectorFactor, default=[0.0, 0.0]):
    pass


class Acceleration(VectorFactor, default=[0.0, 0.0]):
    pass


class Force(VectorFactor, default=[0.0, 0.0]):
    pass
