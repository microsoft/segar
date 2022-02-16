__copyright__ = "Copyright (c) Microsoft Corporation and Mila - Quebec AI Institute"
__license__ = "MIT"
"""Rules for collisions

"""

__all__ = (
    "overlap_time",
    "object_collision",
    "overlaps_wall",
    "overlap_time_wall",
    "fix_overlap_wall",
    "fix_overlap_objects",
    "wall_collision",
)

from typing import Tuple
import warnings

import numpy as np

from segar.factors import (
    Position,
    Velocity,
    Shape,
    Done,
    Alive,
    StoredEnergy,
    Mobile,
    Mass,
    InfiniteEnergy,
    Circle,
    ConvexHullShape,
)
from segar.things.boundaries import Wall, SquareWall
from segar.things import Object, Thing
from segar.types import Time

from .relations import Relation
from .transitions import SetFactor, TransitionFunction


def overlap_time(obj1: Object, obj2: Object) -> Time:
    """Time needed to reverse overlap, given positions, shapes, and velocities.

    :param obj1: First object.
    :param obj2: Second object.
    :return: Time to reverse overlap.
    """
    normal_vec = obj2[Position] - obj1[Position]
    relative_vel = obj1[Velocity] - obj2[Velocity]
    shape1 = obj1[Shape]
    shape2 = obj2[Shape]

    if obj1[Done] or not obj1[Alive] or obj2[Done] or not obj2[Alive]:
        return Time(0.0)

    if relative_vel.norm() == 0:
        return Time(0.0)

    unit_vel = relative_vel.unit_vector()
    dist = shape1.fix_overlap(shape2, normal_vec.value, unit_vector=unit_vel.value)
    return Time(dist / relative_vel.norm())


@TransitionFunction
def object_collision(
    o1: Object, o2: Object
) -> Tuple[
    SetFactor[Velocity], SetFactor[Velocity], SetFactor[StoredEnergy], SetFactor[StoredEnergy],
]:
    """Computes collision rules.

    :param o1: First object.
    :param o2: Second object.
    :return: Changes in velocity and energy of both objects.
    """

    atol: float = 1e-6
    warn_on_fail: bool = False

    x1, v1, m1, mobile1, e1, ieng1 = o1.get_factors(
        Position, Velocity, Mass, Mobile, StoredEnergy, InfiniteEnergy
    )
    x2, v2, m2, mobile2, e2, ieng2 = o2.get_factors(
        Position, Velocity, Mass, Mobile, StoredEnergy, InfiniteEnergy
    )

    v1_: Velocity = v1 * (1 + e2)
    v2_: Velocity = v2 * (1 + e1)

    e1f = e1 * float(not ieng1)
    e2f = e2 * float(not ieng1)

    normal_vec: Position = x2 - x1
    unit_norm = normal_vec.unit_vector()
    unit_tang = unit_norm.tangent_vector()

    # project onto unit normal and tangent vectors
    v1n = v1_.dot(unit_norm)
    v1t = v1_.dot(unit_tang)
    v2n = v2_.dot(unit_norm)
    v2t = v2_.dot(unit_tang)

    if not mobile1:
        m1 = 100000000.0

    if not mobile2:
        m2 = 100000000.0

    v1nf = (v1n * (m1 - m2) + 2 * m2 * v2n) / (m1 + m2)
    v2nf = (v2n * (m2 - m1) + 2 * m1 * v1n) / (m1 + m2)

    v1f = unit_norm * v1nf + unit_tang * v1t
    v2f = unit_norm * v2nf + unit_tang * v2t

    # Check conservation of momentum
    check = np.allclose((v1f * m1 + v2f * m2).value, (v1_ * m1 + v2_ * m2).value, atol=atol)

    if not check:
        fail_str = (
            f"Failed check on conservation of momentum: "
            f"{(v1f * m1 + v2f * m2 - v1_ * m1 - v2_) * m2}"
        )
        if warn_on_fail:
            warnings.warn(fail_str, RuntimeWarning)
        else:
            raise RuntimeError(fail_str)

    # Check conservation of energy
    check = np.allclose(
        (m1 * v1f.norm() ** 2 + m2 * v2f.norm() ** 2),
        (m1 * v1_.norm() ** 2 + m2 * v2_.norm() ** 2),
        atol=atol,
    )

    if not check:
        fail_str = "Failed check on conservation of energy: {}".format(
            m1 * v1f.norm() ** 2
            + m2 * v2f.norm() ** 2
            - m1 * v1_.norm() ** 2
            + m2 * v2_.norm() ** 2
        )

        if warn_on_fail:
            warnings.warn(fail_str, RuntimeWarning)
        else:
            raise RuntimeError(fail_str)

    return (
        SetFactor[Velocity](v1, v1f),
        SetFactor[Velocity](v2, v2f),
        SetFactor[StoredEnergy](e1, e1f),
        SetFactor[StoredEnergy](e2, e2f),
    )


@Relation
def overlaps_wall(thing: Thing, wall: Wall) -> bool:
    """Checks if a thing overlaps with the wall.

    :param thing: Thing to check.
    :param wall: The wall.
    :return: True if they overlap.
    """
    thresh: float = 1e-7

    if isinstance(wall, SquareWall):
        return (
            wall.overlaps_right_boundary(thing, thresh=thresh)
            or wall.overlaps_left_boundary(thing, thresh=thresh)
            or wall.overlaps_top_boundary(thing, thresh=thresh)
            or wall.overlaps_bottom_boundary(thing, thresh=thresh)
        )
    raise NotImplementedError(type(wall))


def overlap_time_wall(obj: Object, wall: SquareWall, thresh: float = 1e-7) -> Time:
    """Time to reverse an overlap with wall, given shape, position,
        and velocity of object.

    :param obj: Object to reverse time on.
    :param wall: The wall.
    :param thresh: Threshold on overlap detection.
    :return: Time to reverse overlaps.
    """

    shape = obj.Shape.value
    v = obj[Velocity]
    x = obj[Position]
    b = wall.boundaries

    d = 0.0
    if isinstance(shape, Circle):
        # find hypotenuse of the right angle defined by the unit
        # vector and overlap, p, with wall.
        r = shape.radius
        u = v.unit_vector()

        dist_top = x[1] + r - b[1]
        dist_bottom = b[0] - x[1] + r
        dist_left = b[0] - x[0] + r
        dist_right = x[0] + r - b[1]

        if wall.overlaps_left_boundary(obj):
            d = max(d, dist_left / abs(u[0]))

        if wall.overlaps_right_boundary(obj):
            d = max(d, dist_right / abs(u[0]))

        if wall.overlaps_bottom_boundary(obj):
            d = max(d, dist_bottom / abs(u[1]))

        if wall.overlaps_top_boundary(obj):
            d = max(d, dist_top / abs(u[1]))
    else:
        raise NotImplementedError(shape)

    if d > 0:
        d += thresh

    assert d >= 0.0
    return Time(d / v.norm())


@TransitionFunction
def wall_collision(obj: Object, wall: SquareWall) -> SetFactor[Velocity]:
    """Gives the overlap fix distance for an object in the opposite
        direction of a given unit vector.

    :param obj: Object to calculate collisions on.
    :return: Change of velocity of object.
    """

    shape = obj.Shape.value
    v = obj[Velocity]
    x = obj[Position]
    b = wall.boundaries

    if wall.damping:
        v_ = v * (1 - wall.damping)
    else:
        v_ = v

    if isinstance(shape, Circle):
        # find hypotenuse of the right angle defined by the unit
        # vector and overlap, p, with wall.
        r = shape.radius

        dist_top = x[1] + r - b[1]
        dist_bottom = b[0] - x[1] + r
        dist_left = b[0] - x[0] + r
        dist_right = x[0] + r - b[1]

        min_rl = min(-dist_right, -dist_left)
        min_tb = min(-dist_top, -dist_bottom)

        if min_rl < min_tb:
            vx = -v_[0]
            vy = v_[1]
        elif min_rl > min_tb:
            vx = v_[0]
            vy = -v_[1]
        else:
            vx = -v_[0]
            vy = -v_[1]

        vf = [vx, vy]

    else:
        raise NotImplementedError(obj.shape)

    return SetFactor[Velocity](v, vf)


def fix_overlap_wall(obj: Thing, wall: SquareWall, thresh: float = 1e-5) -> None:
    """Fixes any overlaps with wall.

    Useful in initialization when objects may overlap just due to position
        sampling.

    :param obj: Object to fix overlap with.
    :param wall: The wall.
    :param thresh: Threshold on overlaps.
    """
    shape = obj.Shape.value
    x = obj[Position]
    b = wall.boundaries

    # Stationary object
    if isinstance(shape, Circle):
        r = shape.radius
        xf = np.array(
            (
                np.clip(x[0], b[0] + r + thresh, b[1] - r - thresh),
                np.clip(x[1], b[0] + r + thresh, b[1] - r - thresh),
            )
        )
    elif isinstance(shape, ConvexHullShape):
        points = shape.points
        min_x = points[0][0]
        min_y = points[0][1]
        max_x = points[0][0]
        max_y = points[0][1]
        for p in points:
            if p[1] > max_y:
                max_y = p[1]
            if p[1] < min_y:
                min_y = p[1]
            if p[0] < min_x:
                min_x = p[0]
            if p[0] > max_x:
                max_x = p[0]

        xf = np.array(
            (
                np.clip(x[0], b[0] - min_x + thresh, b[1] - max_x - thresh),
                np.clip(x[1], b[0] - min_y + thresh, b[1] - max_y - thresh),
            )
        )

    else:
        raise NotImplementedError(shape)

    with x.in_place():
        x.set(xf)


def fix_overlap_objects(obj1: Thing, obj2: Thing) -> None:
    """Fixes any overlaps between two things.

    Useful in initialization when objects may overlap just due to position
        sampling.

    :param obj1: First object.
    :param obj2: Second object.
    """
    x1 = obj1[Position]
    x2 = obj2[Position]
    normal_vec = x2 - x1
    shape1 = obj1[Shape]
    shape2 = obj2[Shape]

    mag = shape2.fix_overlap(shape1, normal_vec.value)
    xf = x1 - normal_vec * mag / normal_vec.norm()

    with x1.in_place():
        x1.set(xf)
