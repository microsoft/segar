__author__ = "R Devon Hjelm"
__copyright__ = "Copyright (c) Microsoft Corporation and Mila - Quebec AI " \
                "Institute"
__license__ = "MIT"

"""Boolean factors.

"""

__all__ = ('Visible', 'Alive', 'Done', 'InfiniteEnergy', 'Mobile',
           'BooleanFactor')

from segar.factors.factors import Factor


class BooleanFactor(Factor[bool]):
    def __bool__(self) -> bool:
        return self.value

    def __float__(self) -> float:
        return float(self.value)

    def __int__(self) -> int:
        return int(self.value)


class Visible(BooleanFactor, default=True):
    pass


class Alive(BooleanFactor, default=True):
    pass


class Done(BooleanFactor, default=False):
    pass


class InfiniteEnergy(BooleanFactor, default=False):
    pass


class Mobile(BooleanFactor, default=True):
    pass
