__copyright__ = (
    "Copyright (c) Microsoft Corporation and Mila - Quebec AI Institute"
)
__license__ = "MIT"
"""Special property factors.

"""

__all__ = ("Floor", "Collides", "Consumes", "ID", "Label", "Text")

from typing import TypeVar

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
