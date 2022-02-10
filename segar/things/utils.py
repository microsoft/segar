__copyright__ = "Copyright (c) Microsoft Corporation and Mila - Quebec AI Institute"
__license__ = "MIT"
"""Utilities for things.

"""
__all__ = ('ThingFactory',)

from typing import Type, TypeVar, Union

from segar.factors import Choice
from .things import Entity, Thing


E = TypeVar('E', bound=Type[Entity])


class ThingFactory:
    """Creates a class that creates random types of Things.

    Note: This is probably an abuse of the __new__ method, but it's
    convenient for NumberOf.

    """

    def __init__(self, choices: Union[dict[Type[Entity], float],
                                      list[Type[Entity]]]):
        """

        :param choices: Either a dictionary of type / probability pairs or a
            list of types that have equal probability.
        """
        if isinstance(choices, dict):
            probs = choices
        elif isinstance(choices, list):
            p = 1. / len(choices)
            probs = dict((k, p) for k in choices)
        else:
            raise ValueError(type(choices))

        self.distribution = Choice(list(probs.keys()), p=list(probs.values()))
        for k in self.distribution.keys:
            if not issubclass(k, Thing):
                raise ValueError('All probability keys must be a Thing.')

    def __call__(self) -> Type[Entity]:
        """Samples from the types and returns a type.

        :return: Type of thing to instantiate.
        """
        cls = self.distribution.sample()
        return cls
