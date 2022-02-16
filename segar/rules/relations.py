__copyright__ = "Copyright (c) Microsoft Corporation and Mila - Quebec AI Institute"
__license__ = "MIT"
"""A relation is treated like a special type of Rule that returns bool.

"""

__all__ = (
    "Relation",
    "overlaps",
    "Or",
    "And",
    "IsEqual",
    "IsOn",
    "Contains",
    "colliding",
)

from typing import Callable, Tuple, Type, TypeVar, Union

from segar.factors import Position, Shape, Velocity, Factor
from segar.things import Object, Entity, Tile

from .rules import match_pattern, Rule


T = TypeVar("T")
F = TypeVar("F", bound=Factor)


class Relation(Rule):
    """A relation is a rule that returns a boolean.

    This is a wrapper used on functions that take factors as input and
    return bools.

    """

    def __init__(self, rule_fn: Callable):
        super().__init__(rule_fn)

    def __call__(self, *inputs: Union[Factor, Tuple[Factor, ...]]) -> bool:
        """

        :param inputs: Inputs to relation.
        :return: Whether the relation holds for inputs.
        """
        args = match_pattern(self, *inputs, loose_match=True)
        if args is None:
            raise TypeError(inputs)
        return self.rule_fn(*args)


@Relation
def overlaps(o1_factors: Tuple[Position, Shape], o2_factors: Tuple[Position, Shape]):
    """Whether there is an overlap given position and shape.

    :param o1_factors: First set of factors.
    :param o2_factors: Second set of factors.
    :return: Whether there is an overlap.
    """
    x1, shape1 = o1_factors
    x2, shape2 = o2_factors
    normal_vector = x2 - x1

    return shape2.overlaps(shape1, normal_vector)


@Relation
def colliding(
    o1_factors: Tuple[Position, Shape, Velocity], o2_factors: Tuple[Position, Shape, Velocity],
):
    """Whether there is a collision, given positions, shapes, and velocity.

    :param o1_factors: First set of factors.
    :param o2_factors: Second set of factors.
    :return: Whether there is a collision.
    """
    x1, shape1, _ = o1_factors
    x2, shape2, _ = o2_factors
    normal_vector = x2 - x1
    return shape2.overlaps(shape1, normal_vector)


class Or:
    """Generic OR relation. Checks if any relation holds in list.

    """

    def __init__(self, *relations):
        self.relations = relations

    def __call__(self, *factor_list):
        for relation in self.relations:
            if relation(*factor_list):
                return True
        return False


class And:
    """Generic AND relation. Checks if all relations hold in list.

    """

    def __init__(self, *relations):
        self.relations = relations

    def __call__(self, *factor_list):
        for relation in self.relations:
            if not relation(*factor_list):
                return False
        return True


class IsEqual(Relation):
    """Checks if factor as a particular value.

    """

    def __init__(self, target_factor: Type[Factor], value: T):
        self.target = target_factor
        self.value = value

        def is_equal(f: target_factor) -> bool:
            return f == value

        super().__init__(is_equal)

    def __repr__(self) -> str:
        return f"{self.target} == {self.value}"


class IsOn(Relation):
    """Checks if one object is on top of another entity.

    In order to determine whether query thing is on top of a key thing,
    we have to ask the sim if there's another key thing that overlaps and
    has higher order. This relation is not binary, but must involve all
    things in the sim.

    """

    def __init__(self):
        def ison(obj: Object, tile: Entity) -> bool:
            return self.sim.is_on(obj, tile)

        super().__init__(ison)


class Contains(Relation):
    """Checks if one object is contained in another tile.

    """

    def __init__(self):
        def contains(thing1: Object, thing2: Tile) -> bool:
            is_on = self.sim.is_on(thing1, thing2)
            normal_vec = thing2[Position] - thing1[Position]
            shape1 = thing1[Shape]
            shape2 = thing2[Shape]
            area_overlap = shape1.area_overlap(shape2, normal_vector=normal_vec)
            obj_area = shape1.area()
            return (area_overlap == obj_area) and is_on

        super().__init__(contains)
