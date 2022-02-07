from __future__ import annotations
__author__ = "R Devon Hjelm"
__copyright__ = "Copyright (c) Microsoft Corporation and Mila: The Quebec " \
                "AI Company"
__license__ = "MIT"
"""Transition functions and applications.

"""

__all__ = ('Differential', 'SetFactor', 'Aggregate', 'DidNotMatch',
           'DidNotPass', 'Transition', 'TransitionFunction',
           'conditional_transition')

from typing import Any, Callable, Generic, Optional, Type, TypeVar, Union

import numpy as np

from segar.factors import Factor, ID, Label
from segar.parameters import Parameter
from segar.things import Entity, Object, Thing, Tile
from segar.types import Time

from .relations import And, IsEqual, Relation, Or
from .rules import match_pattern, Rule


F = TypeVar('F', bound=Factor)
T = TypeVar('T', bound=Type[Factor])


class TransitionFunction(Rule):
    """Transition function is the main type of rule used by the simulator to
        change states between subsequent time steps.

    """

    def __init__(self,
                 rule_fn: Callable,
                 relation: Optional[Relation] = None,
                 factor_type: Optional[Type[Factor]] = None,
                 entity_type: Optional[Type[Entity]] = None):
        """Wraps a function into a Rule.

        :param rule_fn: Function to wrap.
        :param relation: Optional relation that must be true on at least one
            input to apply transition function.
        :param factor_type: Optional factor type that must be present in
            entity when pattern matching.
        :param entity_type: Optional entity type that must be present in
            pattern matching.
        """

        self._relation = relation
        super().__init__(rule_fn, factor_type=factor_type,
                         entity_type=entity_type)

    @property
    def priority(self) -> int:
        """Priority of the transition function.

        Some functions apply to the same factor. This priority helps solve
        conflicts.

        """
        factor_priority = get_factor_type_priority(self.factor_type)
        entity_priority = get_entity_type_priority(self.entity_type)
        relation_priority = get_relation_priority(self._relation)
        return max(factor_priority, entity_priority, relation_priority)

    def check_condition(self, *args) -> bool:
        """Checks if a relation holds.

        """
        if self._relation is None:
            return True
        else:
            return self._relation(*args)

    def __call__(self,
                 *inputs: Union[Factor, tuple[Factor, ...], Parameter]
                 ) -> Union[Transition, DidNotMatch, DidNotPass]:
        """Apply a transition function.

        :param inputs: Inputs to apply on.
        :return: Either an application of the transition function or
            failure object.
        """

        args = match_pattern(self, *inputs)
        if args is None:
            outcome = DidNotMatch(self, inputs)
        elif self.check_condition(*inputs):
            outcome = self.rule_fn(*args)
            if isinstance(outcome, SetFactor):
                outcome.set_priority(self.priority)
            if isinstance(outcome, Transition):
                outcome.set_metadata(self, args)
        else:
            outcome = DidNotPass(self._relation, inputs)

        return outcome


def get_entity_type_priority(entity_type: Type[Entity]) -> int:
    """Returns priority level given entity type of condition.

    The more abstract the type, the less priority.

    :param entity_type: Entity type.
    :return: Priority level.
    """
    if entity_type is None:
        return 0
    elif entity_type == Thing:
        return 4
    elif entity_type in (Object, Tile):
        return 5
    elif issubclass(entity_type, Object) or issubclass(entity_type, Tile):
        return 6
    else:
        return 0


def get_factor_type_priority(factor_type: Type[Factor]) -> int:
    """Returns priority level given factor type of condition.

    Fixed at 3 if factor type is given. Else 0.

    :param factor_type: Factor type.
    :return: Priority level.
    """
    if factor_type is None:
        return 0
    else:
        return 3


def get_relation_priority(relation: Relation) -> int:
    """This function helps resolve conflicts when multiple conditions apply
        to the same factor transition.

    """
    if isinstance(relation, And):
        priorities = [get_relation_priority(rel) for rel in relation.relations]
        priority = sum(priorities)

    elif isinstance(relation, Or):
        priorities = [get_relation_priority(rel) for rel in relation.relations]
        priority = max(priorities)

    elif isinstance(relation, IsEqual):
        # Identifying factors have high priority.
        if relation.target == ID:
            priority = 8
        elif relation.target == Label:
            priority = 7
        else:
            priority = 2

    else:
        priority = 1

    return priority


def conditional_transition(relation: Relation = None,
                           factor_type: Optional[Type[Factor]] = None,
                           entity_type: Optional[Type[Entity]] = None):
    """A wrapper function to conveniently add conditions to rules.

    """
    class ConditionalTransitionFunction(TransitionFunction):
        def __init__(self, rule_fn: Callable):
            super().__init__(rule_fn, factor_type=factor_type,
                             entity_type=entity_type, relation=relation)

    return ConditionalTransitionFunction


class Transition(Generic[F]):
    """A transition is an application of a transition function.

    Creating this object alone does not change the factor: calling it will.

    """
    def __init__(self, factor: F, value: Union[F.t, F]):
        """

        :param factor: Factor that applies to this transition.
        :param value: Value to change the factor to.
        """
        self.factor = factor
        if isinstance(value, Factor):
            value = value.value
        self.value = value
        self.applied = False
        self.rule = None
        self.args = None

    def set_metadata(self, rule: Rule, args: list):
        """Metadata for allowing the sim to decide priority.

        :param rule: Source rule for this application.
        :param args: Arguments that were applied to this rule.
        :return:
        """
        self.rule = rule
        self.args = args

    def __eq__(self, other: Any):
        if not isinstance(other, self.__class__):
            return False
        if not isinstance(other.factor, Factor):
            return False

        factor_eq = self.factor == other.factor
        if isinstance(self.value, np.ndarray):
            value_eq = all(self.value == other.value)
        else:
            value_eq = self.value == other.value
        return factor_eq and value_eq

    def __add__(self, other: Transition) -> Transition:
        assert self.factor is other.factor
        if isinstance(self, SetFactor):
            if isinstance(other, SetFactor):
                if self == other:
                    return self
                elif self.priority > other.priority:
                    return self
                elif other.priority > self.priority:
                    return other
                else:
                    msg = (f'Conflict resolution not possible withfactor'
                           f' {self.factor} with transitions {self} ('
                           f'priority {self.priority}) and {other} (priority'
                           f' {other.priority}). This is a fail because '
                           f'otherwise transition function may not be '
                           f'deterministic. Change rules if this happens.')
                    raise ValueError(msg)
            else:
                return self
        elif isinstance(other, SetFactor):
            return other
        elif isinstance(self, Aggregate):
            if isinstance(other, Aggregate):
                return Aggregate(self.factor, self.value + other.value)
            else:
                return self
        elif isinstance(self, Differential):
            if isinstance(other, Aggregate):
                return other
            else:
                return Differential(self.factor, self.value + other.value)

    # sum() requires this.
    def __radd__(self, other: Union[int, Transition]) -> Transition:
        if other == 0:
            return self
        else:
            return self.__add__(other)

    def __call__(self) -> F:
        raise NotImplementedError


class Differential(Transition, Generic[F]):
    """Adds to previous value scaled by time.

    """
    def __repr__(self) -> str:
        return f'{self.factor} += Δt {self.value}'

    def __call__(self, dt: Time) -> F:
        if not self.applied:
            self.factor.set(self.factor.value + self.value * dt,
                            allow_in_place=True)
            return self.factor
        else:
            raise ValueError('Transition can only be applied once.')


class Aggregate(Transition, Generic[F]):
    """Aggregates but doesn't use previous value.

    """
    def __repr__(self) -> str:
        return f'{self.factor} += {self.value}'

    def __call__(self) -> F:
        if not self.applied:
            try:
                self.factor.set(self.value, allow_in_place=True)
            except (TypeError, ValueError) as e:
                raise RuntimeError(f'Transition setting on factor '
                                   f'{self.factor} failed with rule '
                                   f'{self.rule}.') from e
            return self.factor
        else:
            raise ValueError('Transition can only be applied once.')


class SetFactor(Transition, Generic[F]):
    """Sets value of factor.

    """
    def __init__(self, factor: F, value: Union[F.t, F]):
        self._priority = None
        super().__init__(factor, value)

    def set_priority(self, priority: int):
        self._priority = priority

    @property
    def priority(self) -> int:
        return self._priority

    def __repr__(self) -> str:
        return f'{self.factor.__class__.__name__} <- {self.value}'

    def __call__(self) -> F:
        if not self.applied:
            try:
                self.factor.set(self.value, allow_in_place=True)
            except (TypeError, ValueError) as e:
                raise RuntimeError(f'Transition setting on factor '
                                   f'{self.factor} failed with rule '
                                   f'{self.rule}.') from e
            return self.factor
        else:
            raise ValueError('Transition can only be applied once.')


class DidNotPass:
    """Placeholder failure of application that means the conditions did not
        pass.

    """
    def __init__(self, rule_condition: Rule, inputs: tuple):
        self.rule_condition = rule_condition
        self.inputs = inputs

    def __repr__(self):
        return f'DidNotPass({self.rule_condition}, {self.inputs})'


class DidNotMatch:
    """Placeholder failure of application that means the inputs did not match.

    """
    def __init__(self, rule: Rule, inputs: tuple):
        self.rule = rule
        self.inputs = inputs

    def __repr__(self):
        return f'DidNotMatch({self.rule}, {self.inputs})'
