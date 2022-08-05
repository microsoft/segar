"""Some classic control RL problems implemented in SEGAR

Note it is recommended for mountaincar to change the precision for collision checks using

```
segar.sim import change_precision
change_precision(0.01)
```

"""

import math
from typing import Optional, Tuple, Union

from gym.spaces import Box, Discrete
import numpy as np

from segar.rules.transitions import TransitionFunction, Aggregate, SetFactor, Differential
from segar.factors import (
    Position,
    Velocity,
    Acceleration,
    Size,
    Mass,
    Shape,
    Circle,
    NumericFactor,
    Mobile,
    Label,
    Text,
    Charge,
    ID
)
from segar.mdps.initializations import Initialization
from segar.mdps.observations import StateObservation
from segar.mdps.tasks import Task
from segar.parameters import Gravity
from segar.rendering.rgb_rendering import register_rule
from segar.things import Entity, Object, Thing, Ball, SquareWall


# Classic control doesn't have collisions, so threshold positions if we don't want to bounce, otherwise set to None
_POS_THRESH = .05


# Mountaincar default parameters, matched to gym. Everything is in the original basis (see basis functions below).
_MOUNTAINCAR_POS_RANGE = (-1.2, 0.6)
_MOUNTAINCAR_VEL_RANGE = (-0.07, 0.07)
_MOUNTAINCAR_GOAL_POSITION = 0.5
_MOUNTAINCAR_GOAL_VELOCITY = 0.
_MOUNTAINCAR_TIME_SCALE = 1.  # 0.25 / 9.8  # dt=1 in the gym code, while by default it is 1/100 in segar
_MOUNTAINCAR_FORCE_SCALE = 0.001
_MOUNTAINCAR_ACTIONS = (-1. * _MOUNTAINCAR_FORCE_SCALE, 0, 1. * _MOUNTAINCAR_FORCE_SCALE)
_MOUNTAINCAR_INIT_POS_RANGE = (-0.6, 0.4)


# Cartpole default parameters, matched to gym
_CARTPOLE_ACTIONS = (-10., 10.)
_CARTPOLE_POS_RANGE = (-2.4, 2.4)
_CARTPOLE_ANGLE_RANGE = (-12 * 2 * math.pi / 360., 12 * 2 * math.pi / 360.)


# Simulator parameters. If you use a different boundary size, this must change.
_arena_range = (-1. + _POS_THRESH, 1. - _POS_THRESH)  # If you have different boundaries, this needs to change
_arena_length = _arena_range[1] - _arena_range[0]


# New factors for Cartpole
class Angle(NumericFactor[float], default=0., range=[-float(np.pi) / 2., float(np.pi) / 2.]):
    pass


class AngularVelocity(NumericFactor[float], default=0., range=[-float(np.pi), float(np.pi)]):
    pass


class AngularAcceleration(NumericFactor[float], default=0.):
    pass


class PoleMassProp(Mass, range=[0., 1.], default=0.1):
    pass


class PoleLength(NumericFactor[float], default=0.5):
    pass


#first_factors = [ft for ft in FACTORS if ft not in (Velocity, AngularVelocity, Position, Angle, Force)]
#sim = Simulator(factor_update_order=(first_factors[:], [Force, Velocity, AngularVelocity], [Angle]))


# New Mountaincar rules
def to_mountaincar_basis(x, recenter=True):
    mountaincar_length = _MOUNTAINCAR_POS_RANGE[1] - _MOUNTAINCAR_POS_RANGE[0]
    # center and rescale
    if recenter:
        x -= _arena_range[0]
    # We want a buffer so we don't have to deal with object size.
    x *= mountaincar_length / _arena_length
    # shift
    if recenter:
        x += _MOUNTAINCAR_POS_RANGE[0]
    return x


def from_mountaincar_basis(x, recenter=True):
    mountaincar_length = _MOUNTAINCAR_POS_RANGE[1] - _MOUNTAINCAR_POS_RANGE[0]
    if recenter:
        x -= _MOUNTAINCAR_POS_RANGE[0]

    x *= _arena_length / mountaincar_length

    if recenter:
        x += _arena_range[0]
    return x


@TransitionFunction
def mountaincar_hill(o_factors: Tuple[Size, Position, Acceleration, Mass], gravity: Gravity
                                ) -> Aggregate[Acceleration]:
    size, pos, acc, mass = o_factors
    pos_x = pos[0]
    # mountaincar area is just the x axis minus the size / 2 of the object (hack because assuming circle shape).
    # But the mountaincar range is defined by -1.2 to .5 and our arena is -1. to 1., so change basis
    pos_x = to_mountaincar_basis(pos_x)
    slope = np.cos(3 * pos_x)
    da_x = -gravity * mass * slope * _MOUNTAINCAR_TIME_SCALE
    da_x = from_mountaincar_basis(da_x, recenter=False)

    da = Acceleration([da_x, 0.])
    return Aggregate[Acceleration](acc, da)


@TransitionFunction
def mountaincar_wall_collision(obj: Object, wall: SquareWall) -> SetFactor[Velocity]:
    # Mountaincar stops if it reaches the edge
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
            vx = 0.
            vy = v_[1]
        elif min_rl > min_tb:
            vx = v_[0]
            vy = -v_[1]
        else:
            vx = 0.
            vy = -v_[1]

        vf = [vx, vy]

    else:
        raise NotImplementedError(shape)

    return SetFactor[Velocity](v, vf)


# Cartpole rules
@TransitionFunction
def change_angle(theta: Angle, v: AngularVelocity) -> Differential[Angle]:
    dtheta = v
    return Differential[Angle](theta, dtheta)


@TransitionFunction
def change_angular_velocity(thetavel: AngularVelocity, thetaacc: AngularAcceleration) -> Differential[AngularVelocity]:
    dthetavel = thetaacc
    return Differential[AngularVelocity](thetavel, dthetavel)


@TransitionFunction
def change_angular_acceleration(o_factors: Tuple[
    Mass, Velocity, PoleMassProp, PoleLength, Acceleration, Angle, AngularVelocity, AngularAcceleration],
                                gravity: Gravity
                                ) -> Tuple[Aggregate[AngularAcceleration], Aggregate[Acceleration]]:
    total_mass, vel, pole_prop_mass, length, acceleration, theta, thetavel, thetaacc = o_factors

    if abs(theta) >= np.pi / 2.:
        # This is pretty much game over for carpole, but if we want cartpole physics that work past when the episode normally ends, we need to stop theta
        return None
    a_y = acceleration[1]
    pole_mass = total_mass * pole_prop_mass
    force = a_y * total_mass
    sintheta = np.sin(theta)
    costheta = np.cos(theta)

    # Copied more or less from gym
    temp = (force + length * thetavel ** 2 * sintheta) / total_mass
    dthetavel = (gravity * sintheta - costheta * temp) / (length * (4.0 / 3.0 - pole_mass * costheta ** 2 / total_mass))

    # Here we need to remove the external force because it's already being applied to the ball
    da_y = temp - length * dthetavel * costheta / total_mass - force / total_mass

    return Aggregate[AngularAcceleration](thetaacc, dthetavel), Aggregate[Acceleration](acceleration, [0., da_y])


# For pixel-based observations of Cartpole in 2D
def color_map(thing: Thing):
    if not thing.has_factor(Angle):
        return None

    theta = thing[Angle]
    c = round((theta + float(np.pi / 2.)) * 255 / float(np.pi))
    color = [c, 0, 255 - c]
    return color


register_rule(color_map)


class CartPole(Ball, default={Shape: Circle(0.2), Mobile: True, Label: "cartpole", Text: "C", Charge: .1}):
    _factor_types = Object._factor_types + (Angle, AngularVelocity, AngularAcceleration, PoleMassProp, PoleLength)


# Mountaincar task
class MountainCarInitialization(Initialization):

    def __call__(self, init_things: Optional[list[Entity]] = None) -> None:
        x = float(np.random.uniform(*_MOUNTAINCAR_INIT_POS_RANGE))
        self.sim.add_ball(position=np.array([x, 0.]), unique_id='mountaincar', initial_factors={Mass: 1.0})


class MountainCarTask(Task):
    def __init__(
            self,
            initialization: Initialization,
            x_actions: tuple[float] = _MOUNTAINCAR_ACTIONS,
            reward_shape: bool = False
    ):
        action_space = Discrete(len(x_actions))
        self._actions = x_actions
        self._reward_shape = reward_shape
        super().__init__(initialization, action_space)
        self.sim.add_rule(mountaincar_hill)
        self.sim._wall_collision_rule = mountaincar_wall_collision
        self.terminated: bool = False

    def check_action(self, action: int) -> bool:
        return action in list(range(len(self._actions)))

    def demo_action(self) -> int:
        return 1  # Do nothing

    def get_height(self, x: Union[float, Position]) -> float:
        x = to_mountaincar_basis(x)
        height = np.sin(3 * x) * 0.45 + 0.55
        return height

    def reward(self, state: dict) -> float:
        pos_x = state['things']['mountaincar'][Position][0]
        height = self.get_height(pos_x)
        if not self.terminated:
            reward = -1.
            if self._reward_shape:
                reward += 13 * np.abs(height)
        else:
            reward = 0.

        return reward

    def apply_action(self, action: int) -> None:
        self.sim.add_force('mountaincar', np.array([self._actions[action], 0.]))

    def done(self, state: dict) -> bool:
        pos_x = state['things']['mountaincar'][Position][0]
        vel_x = state['things']['mountaincar'][Velocity][0]
        pos_x = to_mountaincar_basis(pos_x)
        vel_x = to_mountaincar_basis(vel_x, recenter=False)
        if pos_x >= _MOUNTAINCAR_GOAL_POSITION and vel_x >= _MOUNTAINCAR_GOAL_VELOCITY:
            self.terminated = True
        return self.terminated

    def results(self, state: dict) -> dict:
        """Returns results to be processed by the MDP.
        :param state: State dictionary to pull results from.
        :return: Dictionary of results.
        """
        pos_x = state['things']['mountaincar'][Position][0]
        vel_x = state['things']['mountaincar'][Velocity][0]
        acc_x = state['things']['mountaincar'][Acceleration][0]
        height = self.get_height(pos_x)
        return {'x': pos_x, 'h': height, 'x\'': vel_x, 'x\'\'': acc_x}


class MountainCarObservation(StateObservation):
    """Gym state-based observation space for mountaincar.

    """
    def __init__(self):
        super().__init__('mountaincar')

    def _make_observation_space(self) -> None:
        self._observation_space = Box(-100, 100, shape=(2,), dtype=np.float32)

    def __call__(self, states: dict) -> np.ndarray:
        mountaincar_state = states['things'][self.unique_id]
        pos_x = to_mountaincar_basis(mountaincar_state[Position][0])
        vel_x = from_mountaincar_basis(mountaincar_state[Velocity][0], recenter=False)
        return np.array(pos_x, vel_x)


# Cartpole task
class CartPoleInitialization(Initialization):

    def __call__(self, init_things: Optional[list[Entity]] = None) -> None:
        cartpole = CartPole(initial_factors={Angle: 0.01, ID: 'cartpole'})
        self.sim.adopt(cartpole)


class CartPoleTask(Task):

    def __init__(
            self,
            initialization: Initialization,
            x_actions: tuple[float] = _CARTPOLE_ACTIONS
    ):
        action_space = Discrete(len(x_actions))
        self._actions = x_actions
        super().__init__(initialization, action_space)
        self.sim.add_rule(change_angle)
        self.sim.add_rule(change_angular_velocity)
        self.sim.add_rule(change_angular_acceleration)

    def check_action(self, action: int) -> bool:
        return action in list(range(len(self._actions)))

    def demo_action(self) -> int:
        return 1  # Do nothing

    def reward(self, state: dict) -> float:
        theta = state['things']['cartpole'][Angle]
        return float(np.abs(theta) < np.pi / 2.)

    def apply_action(self, action: int) -> None:
        self.sim.add_force(0, np.array([0., self._actions[action]]))

    def done(self, state: dict) -> bool:
        theta = state['things']['cartpole'][Angle]
        return np.abs(theta) >= np.pi / 2.

    def results(self, state: dict) -> dict:
        """Returns results to be processed by the MDP.
        :param state: State dictionary to pull results from.
        :return: Dictionary of results.
        """
        theta = state['things']['cartpole'][Angle]
        thetadot = state['things']['cartpole'][AngularVelocity]
        thetadotdot = state['things']['cartpole'][AngularAcceleration]
        return {'\u03B8': theta, '\u03B8\'': thetadot, '\u03B8\'\'': thetadotdot}


class CartPoleObservation(StateObservation):
    """Gym state-based observation space for cartpole.

    """
    def _make_observation_space(self) -> None:
        self._observation_space = Box(-100, 100, shape=(4,), dtype=np.float32)

    def __call__(self, states: dict) -> np.ndarray:
        cartpole_state = states['things']['cartpole']
        pos_x = change_basis(cartpole_state[Position][0])
        vel_x = change_basis(cartpole_state[Velocity][0])
        theta = change_basis(cartpole_state[Angle][0])
        thetadot = change_basis(cartpole_state[AngularVelocity][0])
        return np.array(pos_x, vel_x, theta, thetadot)
