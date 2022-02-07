__author__ = "R Devon Hjelm"
__copyright__ = "Copyright (c) Microsoft Corporation and Mila - Quebec AI " \
                "Institute"
__license__ = "MIT"
"""Priors

"""
__all__ = ('Prior',)

from typing import Optional, Type, TypeVar, Union
from segar.factors import Factor, Noise, Deterministic
from segar.things import Entity
from .transitions import TransitionFunction, SetFactor
from .relations import Relation


T = TypeVar('T')


class Prior(TransitionFunction):
    """A prior function.

    Used for initialization, for instance sampling a particular factor type
        from a noise distribution.

    """

    def __init__(self, target_factor: Type[Factor], value: Union[T, Factor],
                 factor_type: Optional[Type[Factor]] = None,
                 entity_type: Optional[Type[Entity]] = None,
                 relation: Optional[Relation] = None):
        """

        :param target_factor: Type of factor this prior targets.
        :param value: The value the factor will take. Can be fixed,
            a Factor, or noise.
        :param factor_type: Optional condition on other factor types held by
            Entity that holds target factor.
        :param entity_type: Optional condition on type of Entity that holds
            this factor type.
        :param relation: Optional relation that must be true to apply this
            prior on target factor.
        """

        self.target = target_factor
        self.source = value

        def prior(f: target_factor) -> SetFactor[target_factor]:
            if isinstance(self.source, Noise):
                source_ = self.source.sample()
            else:
                source_ = self.source
            return SetFactor[target_factor](f, source_)

        super().__init__(prior,
                         factor_type=factor_type,
                         entity_type=entity_type,
                         relation=relation)

    def copy_for_sim(self, sim):
        rule_copy = type(self)(self.target, self.source)
        rule_copy.__dict__.update(self.__dict__)
        rule_copy._sim = sim
        return rule_copy

    @property
    def distribution(self):
        if isinstance(self.source, Noise):
            return self.source
        else:
            return Deterministic(self.source)

    def __repr__(self) -> str:
        r = f'{self.target} <- {self.source}'
        if self.factor_type:
            r += f' (if has {self.target})'
        if self.entity_type:
            r += f' (if is {self.entity_type})'
        if self._relation:
            r += f' (if {self._relation})'
        return r
