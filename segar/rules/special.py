__copyright__ = (
    "Copyright (c) Microsoft Corporation and Mila - Quebec AI Institute"
)
__license__ = "MIT"
"""Special case rules.

"""

__all__ = (
    "move",
    "stop_condition",
    "kill_condition",
    "apply_burn",
    "apply_friction",
    "consume",
    "accelerate",
)

from typing import Tuple

from segar.parameters import MinMass, Gravity, MinVelocity
from segar.factors import (
    Position,
    Velocity,
    Acceleration,
    Mobile,
    Alive,
    Mass,
    Visible,
    Friction,
    Heat,
    Done,
    Consumes,
)
from .relations import IsOn, Contains
from .transitions import (
    Aggregate,
    SetFactor,
    Differential,
    TransitionFunction,
    conditional_transition,
)


@TransitionFunction
def move(
    x: Position, v: Velocity, min_vel: MinVelocity
) -> Differential[Position]:
    """Moves a thing that has velocity.

    :param x: Position to change.
    :param v: Velocity to change with.
    :param min_vel: Minimum velocity under which considered stopped.
    :return: Change of position.
    """
    if v.norm() < min_vel:
        dx = 0 * v
    else:
        dx = v
    return Differential[Position](x, dx)


@TransitionFunction
def accelerate(v: Velocity, a: Acceleration) -> Differential[Velocity]:
    """Accelerate given velocity.

    :param v: Velocity to change.
    :param a: Acceleration to change with.
    :return: Change in velocity, to be integrated over time.
    """
    dv = a
    return Differential[Velocity](v, dv)


@TransitionFunction
def stop_condition(
    o_factors: Tuple[Mobile, Alive, Velocity, Acceleration]
) -> Tuple[SetFactor[Velocity], SetFactor[Acceleration]]:
    """Halts an object conditioned on other factors.

    :param o_factors: Whether the object is mobile or alive, it's velocity
        and acceleration.
    :return: Sets velocity and acceleration to 0 if not alive or not mobile.
    """
    mobile, alive, velocity, acceleration = o_factors
    if (not mobile) or (not alive):
        return (
            SetFactor[Velocity](velocity, [0.0, 0.0]),
            SetFactor[Acceleration](acceleration, [0.0, 0.0]),
        )


@TransitionFunction
def kill_condition(
    m: Mass,
    v: Velocity,
    vis: Visible,
    a: Acceleration,
    alive: Alive,
    min_mass: MinMass,
) -> Tuple[
    SetFactor[Mass],
    SetFactor[Velocity],
    SetFactor[Alive],
    SetFactor[Acceleration],
    SetFactor[Visible],
]:
    """Kills and stops object if mass is too small

    :param m: The mass.
    :param v: The velocity.
    :param vis: Whether the object is has visual features.
    :param a: The acceleration.
    :param alive: Whether the object is alive.
    :param min_mass: The minimum mass under which to consider not(alive).
    :return: Set mass, velocity, alive, acceleration, visible to 0 / False.
    """

    if m < min_mass:
        return (
            SetFactor[Mass](m, 0.0),
            SetFactor[Velocity](v, [0.0, 0.0]),
            SetFactor[Alive](alive, False),
            SetFactor[Acceleration](a, [0.0, 0.0]),
            SetFactor[Visible](vis, False),
        )


@conditional_transition(relation=IsOn())
def apply_friction(
    o1_factors: Tuple[Mass, Velocity, Acceleration],
    o2_factors: Tuple[Friction],
    gravity: Gravity,
) -> Aggregate[Acceleration]:
    """Applies friction to an object conditioned on it is on something with
        friction.

    :param o1_factors: First set of factors.
    :param o2_factors: Second set of factors.
    :param gravity: Gravity parameter.
    :return: Change of velocity due to friction.
    """

    mass, velocity, acceleration = o1_factors
    (mu,) = o2_factors
    if velocity.norm() >= 1e-6:

        vel_sign = velocity.sign()
        vel_norm = velocity.norm()

        f_mag = mu * gravity
        norm_abs_vel = velocity.abs() / vel_norm
        da = -vel_sign * f_mag * norm_abs_vel / mass
        return Aggregate[Acceleration](acceleration, da)


@conditional_transition(relation=IsOn())
def apply_burn(
    o1_factors: Tuple[Mass, Mobile], o2_factors: Tuple[Heat],
) -> SetFactor[Mass]:
    """Reduces the mass of something conditioned it is on something with heat.

    :param o1_factors: First set of factors.
    :param o2_factors: Second set of factors.
    :return: Change of mass due to heat.
    """
    mass, mobile = o1_factors
    (heat,) = o2_factors
    if mobile:
        new_mass = Mass(mass * -(heat * 0.1 - 1.0))
        return SetFactor[Mass](mass, new_mass)


@conditional_transition(relation=Contains())
def consume(
    o1_factors: Tuple[Mobile, Done, Visible], o2_factors: Tuple[Consumes]
) -> Tuple[SetFactor[Done], SetFactor[Visible], SetFactor[Mobile]]:
    """Consumes an object, conditioned on it is contained inside something
        with Consumes factor.

    This implements Holes.

    :param o1_factors: First set of factors.
    :param o2_factors: Second set of factors.
    :return: Sets consumed thing to done, invisible, and immobile.
    """

    mobile, done, visible = o1_factors

    if mobile:
        return (
            SetFactor[Done](done, True),
            SetFactor[Visible](visible, False),
            SetFactor[Velocity](mobile, False),
        )
