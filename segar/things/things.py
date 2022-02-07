__author__ = "R Devon Hjelm"
__copyright__ = "Copyright (c) Microsoft Corporation and Mila: The Quebec " \
                "AI Company"
__license__ = "MIT"
"""A thing is the base object in SEGAR.

"""

__all__ = ('Entity', 'Thing')

from copy import copy
from typing import Any, Dict, Optional, Type, TypeVar

import numpy as np

from segar.factors import (BaseShape, Circle, Factor, FactorContainer, ID,
                           Label, Order, Position, Shape, Size, Text, Visible)
from segar.types import ThingID


T = TypeVar('T', bound=Type[Factor])


class Entity(FactorContainer, default={}):
    """Abstract collection of factors that live in the simulator.

    """
    _factor_types = None

    def __init__(self, initial_factors: Dict[Type[Factor], Any],
                 unique_id: Optional[ThingID] = None, sim=None):
        """

        :param initial_factors: Dictionary of initial factor values.
        :param unique_id: Provides an optional unique id for reference.
        :param sim: Optional sim, if passed auto-adopt
        """

        if not isinstance(initial_factors, dict):
            raise ValueError('Argument `initial_factors` must be a '
                             'dictionary.')

        if ID not in initial_factors.keys() or initial_factors[ID] is None:
            initial_factors[ID] = unique_id

        thing_factors = {}
        for k, v in initial_factors.items():
            if not issubclass(k, Factor):
                raise TypeError(f'Initial factors must have {Factor} subclass '
                                f'keys.')

            if k == ID and v is None:
                if ID in self.default:
                    thing_factors[ID] = copy(self.default[ID])
                else:
                    thing_factors[ID] = -1
            elif isinstance(v, BaseShape) and k == Shape:
                thing_factors[k] = k(v)
            elif isinstance(v, k):
                thing_factors[k] = v
            elif v is None and k in self.default:
                thing_factors[k] = copy(self.default[k])
            else:
                thing_factors[k] = k(v)

        super().__init__(thing_factors)
        if sim is not None:
            sim.adopt(self)

    @property
    def factors(self) -> Dict[Type[Factor], Factor]:
        return self.value

    @property
    def state(self) -> Dict[Type[Factor], Factor]:
        state = self.value.copy()
        state['cl'] = self.__class__.__name__

        return state

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(id={self[ID]})'


class Thing(Entity):
    """A collection of base factors along with shape and size.

    """

    _factor_types = (Position, Visible, Order, Shape, Size, Label, Text, ID)

    def __init__(self,
                 initial_factors: Dict[Type[Factor], Any] = None,
                 unique_id: Optional[ThingID] = None, sim=None):
        """

        :param initial_factors: Dictionary of initial factor values.
        :param unique_id: Provides an optional unique id to the tile.
        """
        initial_factors = initial_factors or {}

        #  Lock Shape's size with Size
        if Shape in initial_factors.keys():
            shape = initial_factors[Shape]
            if not isinstance(shape, Shape):
                shape = Shape(shape)
            if Size in initial_factors.keys():
                size = initial_factors[Size]
                if shape.size is not size:
                    raise KeyError('If Shape is provided, it must have the '
                                   'same Size factor as that provided to the '
                                   'Thing.')
            else:
                size = shape.size
            assert size is shape.size

        elif Shape in self._default:
            shape = copy(self._default[Shape])
            if Size in initial_factors.keys():
                size = initial_factors[Size]
                if not isinstance(size, Size):
                    size = Size(size)
                shape.set_size(size)
            else:
                size = shape.size
        else:
            if Size in initial_factors.keys():
                size = initial_factors[Size]
                shape = Shape(Circle(size))
                size = shape.size
            else:
                shape = Shape(Circle())
                size = shape.size

        assert size is shape.size
        initial_factors[Shape] = shape
        initial_factors[Size] = size

        # Things check factors as __class__._factors defines everything
        # subclassed by Thing
        for k, v in initial_factors.items():
            if k not in self._factor_types:
                raise KeyError(f'Class {self.__class__.__name__} does not '
                               f'support factor type {k}. Allowed: '
                               f'{self._factor_types}. For un-constrained '
                               f'factor containers, use `Entity`.')

        for factor_type in self._factor_types:
            if factor_type not in initial_factors.keys():
                initial_factors[factor_type] = None

        super().__init__(initial_factors, unique_id=unique_id, sim=sim)
        self[Shape].set_size(self[Size])  # align size factor with one in shape

    def copy(self):
        """Copies the thing.

        :return: Returns a new copy of the same thing with same-valued factors.
        """
        factors = self.value
        new_factors = {}
        for k, v in factors.items():
            new_factors[k] = copy(v)
        if Size in new_factors and Shape in new_factors:
            new_factors[Size] = new_factors[Shape].size
        new = self.__class__(new_factors)
        return new

    def to_numpy(self, factor_types: list[Type[Factor]]) -> np.ndarray:
        """Returns a numpy array over a set of provided factor types.

        :param factor_types: Types of factors to make into array.
        :return: Array of factor values.
        """
        arr = []
        for factor_type in factor_types:
            if factor_type in self.value:
                value = self[factor_type].value
                if isinstance(value, np.ndarray):
                    arr += value.tolist()
                else:
                    arr.append(float(value))
            else:
                raise KeyError(factor_type)
        return np.array(arr)

    def __setitem__(self, factor_type: Type[Factor], value):
        if factor_type in self._factor_types:
            return super().__setitem__(factor_type, value)
        raise KeyError(f'{self.__class__.__name__} does not have factor '
                       f'`{factor_type}`. Available: {self._factor_types}')
