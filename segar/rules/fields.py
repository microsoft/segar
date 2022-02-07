__author__ = "R Devon Hjelm"
__copyright__ = "Copyright (c) Microsoft Corporation and Mila: The Quebec " \
                "AI Company"
__license__ = "MIT"
"""Interactions from field equations.

"""

__all__ = ('lorentz_law',)

from typing import Tuple

from segar.factors import (Position, Velocity, Acceleration, Charge,
                           Magnetism, Mass)

from .transitions import TransitionFunction, Aggregate


@TransitionFunction
def lorentz_law(o1_factors: Tuple[Position, Velocity, Charge, Magnetism],
                o2_factors: Tuple[Position, Velocity, Charge, Mass,
                                  Acceleration]
                ) -> Aggregate[Acceleration]:
    """Applies electromagetic forces.

    :param o1_factors: Factors of thing that is applying force.
    :param o2_factors: Factors of thing that is receiving force.
    :returns: Aggregate of acceleration on the second thing.
    """

    x1, v1, q1, b1 = o1_factors
    x2, v2, q2, m2, a = o2_factors
    normal_vec = x2 - x1
    unit_norm = normal_vec.unit_vector()

    if m2 == 0:
        # No mass, no acceleration.
        return Aggregate[Acceleration](a, 0. * a)

    if q1:
        f_q = q1 * q2 * unit_norm / normal_vec.norm() ** 2
    else:
        f_q = 0.0

    # assume magnetic field with r^2 decaying strength coming up out of
    # the map
    rel_vel = Velocity(v2 - v1)
    if b1 and rel_vel.norm() != 0.:
        unit_vel = Velocity(rel_vel.unit_vector())
        tang_vel = unit_vel.tangent_vector()
        f_b = (q2 * rel_vel.norm() * b1 *
               tang_vel / rel_vel.norm() ** 2)
    else:
        f_b = 0.0

    da = (f_b + f_q) / m2

    return Aggregate[Acceleration](a, da)
