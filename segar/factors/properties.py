__copyright__ = "Copyright (c) Microsoft Corporation and Mila - Quebec AI Institute"
__license__ = "MIT"
"""Special property factors.

"""

__all__ = ("Floor", "Collides", "Consumes", "ID", "Label", "Text")

from typing import TypeVar
from numbers import Number

from .factors import Factor


T = TypeVar("T")


class Floor(Factor[None], default=None):
    pass


class Collides(Factor[None], default=None):
    pass


class Consumes(Factor[None], default=None):
    pass


class ID(Factor[str]):
    @property
    def value(self) -> T:
        val = super().value
        if isinstance(val, str):
            if val.isdigit():
                return int(val)
        return val

    @value.setter
    def value(self, value):
        if self._allow_in_place or not self._protected_in_place:
            if hasattr(value, "t"):
                alias = value
                if alias.t == self.t:
                    self._value = alias.value
                else:
                    raise ValueError(
                        "Cannot instantiate factor alias directly from a "
                        "different type of factor unless the factor "
                        "value is of the same type."
                    )
            else:
                self._value = self.new_value(value)
        else:
            raise ValueError("Factor in-place operations are protected.")

    def __hash__(self):
        return hash(self.value)


class Label(Factor[str]):
    pass


class Text(Factor[str], default=None):
    pass


class Color(Factor[list], default=(0, 0, 0)):
    @Factor.value.setter
    def value(self, value):
        if isinstance(value, tuple):
            value = list(value)
        super(Color, type(self)).value.fset(self, value)
        error = False
        if len(self._value) != 3:
            error = True
        for i, v in enumerate(self._value):
            if isinstance(v, Number):
                if not(0. <= v <= 255.):
                    error = True
                else:
                    self._value[i] = int(v)
            else:
                error = True
        if error:
            raise ValueError(f'Incorrect value for color, got ({self._value})')
