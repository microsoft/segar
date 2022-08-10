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
    ID,
    FACTORS
)
from segar.mdps.initializations import Initialization
from segar.mdps.observations import StateObservation
from segar.mdps.tasks import Task
from segar.parameters import Gravity
from segar.rendering.rgb_rendering import register_rule
from segar.things import Entity, Object, Thing, Ball, SquareWall


# Classic control doesn't have collisions, so threshold positions if we don't want to bounce, otherwise set to None
_POS_THRESH = .0


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
_CARTPOLE_ANGLE_RANGE = (-24 * math.pi / 180., 24 * math.pi / 180.)
_CARTPOLE_INIT_ANGLE_RANGE = (-0.05, 0.05)

# Simulator parameters. If you use a different boundary size, this must change.
_arena_range = (-1. + _POS_THRESH, 1. - _POS_THRESH)  # If you have different boundaries, this needs to change
_arena_length = _arena_range[1] - _arena_range[0]


# New factors for Cartpole
class Angle(NumericFactor[float], default=0., range=_CARTPOLE_ANGLE_RANGE):
    pass


class AngularVelocity(NumericFactor[float], default=0.):
    pass


class AngularAcceleration(NumericFactor[float], default=0.):
    pass


class PoleMassProp(Mass, range=[0., 1.], default=.1/1.1):
    pass


class PoleLength(NumericFactor[float], default=0.5):
    pass


# New Mountaincar rules
def to_mountaincar_basis(x, recenter=True):
    mountaincar_length = _MOUNTAINCAR_POS_RANGE[1] - _MOUNTAINCAR_POS_RANGE[0]
    # center and rescale
    if recenter:
        x = x - _arena_range[0]
    # We want a buffer so we don't have to deal with object size.
    x = x * mountaincar_length / _arena_length
    # shift
    if recenter:
        x = x + _MOUNTAINCAR_POS_RANGE[0]
    return x


def from_mountaincar_basis(x, recenter=True):
    mountaincar_length = _MOUNTAINCAR_POS_RANGE[1] - _MOUNTAINCAR_POS_RANGE[0]
    if recenter:
        x = x - _MOUNTAINCAR_POS_RANGE[0]

    x = x * _arena_length / mountaincar_length

    if recenter:
        x = x + _arena_range[0]
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


# Mountaincar task
class MountainCarInitialization(Initialization):

    def __call__(self, init_things: Optional[list[Entity]] = None) -> None:
        x = from_mountaincar_basis(float(np.random.uniform(*_MOUNTAINCAR_INIT_POS_RANGE)))
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
        force = from_mountaincar_basis(self._actions[action], recenter=False)
        # Cartpole force is instantaneous as far as its effect on velocity.
        self.sim.add_force('mountaincar', np.array([force, 0.]), continuous=False)

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
        vel_x = to_mountaincar_basis(mountaincar_state[Velocity][0], recenter=False)
        return np.array(pos_x, vel_x)


# Cartpole rules
def to_cartpole_basis(x, recenter=True):
    cartpole_length = _CARTPOLE_POS_RANGE[1] - _CARTPOLE_POS_RANGE[0]
    # center and rescale
    if recenter:
        x = x - _arena_range[0]
    # We want a buffer so we don't have to deal with object size.
    x = x * cartpole_length / _arena_length
    # shift
    if recenter:
        x = x + _CARTPOLE_POS_RANGE[0]
    return x


def from_cartpole_basis(x, recenter=True):
    cartpole_length = _CARTPOLE_POS_RANGE[1] - _CARTPOLE_POS_RANGE[0]
    if recenter:
        x = x - _CARTPOLE_POS_RANGE[0]

    x = x * _arena_length / cartpole_length

    if recenter:
        x = x + _arena_range[0]
    return x


@TransitionFunction
def change_angle(theta: Angle, v: AngularVelocity) -> Differential[Angle]:
    dtheta = v
    return Differential[Angle](theta, dtheta)


@TransitionFunction
def change_angular_velocity(thetavel: AngularVelocity, thetaacc: AngularAcceleration) -> Differential[AngularVelocity]:
    dthetavel = thetaacc
    return Differential[AngularVelocity](thetavel, dthetavel)


@TransitionFunction
def pole_acceleration(o_factors: Tuple[Mass, Velocity, PoleMassProp, PoleLength, Acceleration, Angle, AngularVelocity,
                                       AngularAcceleration], gravity: Gravity
                      ) -> Tuple[Aggregate[AngularAcceleration], Aggregate[Acceleration]]:

    total_mass, vel, pole_prop_mass, length, acceleration, theta, thetavel, thetaacc = o_factors
    a_y = acceleration[1]
    pole_mass = total_mass * pole_prop_mass
    force_total = a_y * total_mass
    force_cp = to_cartpole_basis(force_total, recenter=False)
    sintheta = np.sin(theta)
    costheta = np.cos(theta)
    polemass_length = pole_mass * length

    # Copied more or less from gym
    temp = (force_cp + length * thetavel ** 2 * sintheta) / total_mass
    dthetavel = (gravity * sintheta - costheta * temp) / (
            length * (4.0 / 3.0 - pole_mass * costheta ** 2 / total_mass))

    # Here we need to remove the acceleration because it's already being applied to the object
    da_y = from_cartpole_basis(temp - polemass_length * dthetavel * costheta / total_mass) - a_y
    return Aggregate[AngularAcceleration](thetaacc, dthetavel), Aggregate[Acceleration](acceleration, [0., da_y])


@TransitionFunction
def pole_fell(theta: Angle, thetavel: AngularVelocity, thetaacc: AngularAcceleration
              ) -> Tuple[SetFactor[Angle], SetFactor[AngularVelocity], SetFactor[AngularAcceleration]]:
    if _CARTPOLE_ANGLE_RANGE[0] < theta < _CARTPOLE_ANGLE_RANGE[1]:
        return None
    new_theta = Angle(np.clip(theta.value, *_CARTPOLE_ANGLE_RANGE))

    return (SetFactor[Angle](theta, new_theta), SetFactor[AngularVelocity](thetavel, 0.),
            SetFactor[AngularAcceleration](thetaacc, 0.))


# For pixel-based observations of Cartpole in 2D
def color_map(thing: Thing):
    if not thing.has_factor(Angle):
        return None

    total_angle = (_CARTPOLE_ANGLE_RANGE[1] - _CARTPOLE_ANGLE_RANGE[0])
    theta = thing[Angle]
    c = round((theta - _CARTPOLE_ANGLE_RANGE[0]) * 255 / total_angle)
    color = [c, 0, 255 - c]
    return color


register_rule(color_map)


class CartPole(Ball, default={Shape: Circle(0.2), Mobile: True, Label: "cartpole", Text: "C", Charge: .1}):
    _factor_types = Object._factor_types + (Angle, AngularVelocity, AngularAcceleration, PoleMassProp, PoleLength)


# Cartpole task
class CartPoleInitialization(Initialization):

    def __call__(self, init_things: Optional[list[Entity]] = None) -> None:
        theta = float(np.random.uniform(*_CARTPOLE_INIT_ANGLE_RANGE))
        cartpole = CartPole(initial_factors={Angle: theta, ID: 'cartpole', Mass: 1.1})
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
        self.sim.add_rule(pole_acceleration)
        self.sim.add_rule(pole_fell)
        first_factors = [ft for ft in FACTORS if ft not in (Velocity, AngularVelocity, Position, Angle)]
        self.sim._factor_update_order = (first_factors[:], [Velocity, AngularVelocity], [Angle])
        self.terminated = False
        self.steps_terminated = 0

    def initialize(self, init_things=None):
        self.terminated = False
        self.steps_terminated = 0
        return super().initialize(init_things=init_things)

    def check_action(self, action: int) -> bool:
        return action in list(range(len(self._actions)))

    def demo_action(self) -> int:
        return 1

    def reward(self, state: dict) -> float:
        if self.terminated and self.steps_terminated > 0:
            return 0.
        else:
            return 1.

    def apply_action(self, action: int) -> None:
        force = from_cartpole_basis(self._actions[action], recenter=False)
        # As opposed to mountaincar, which uses force to instantaneously change the velocity,
        # the cartpole force is used directly in the equations of motion.
        self.sim.add_force('cartpole', np.array([0., force]), continuous=True)
        # Small hack due to MDP issues
        mass = self.sim.things['cartpole'][Mass]
        self.sim.add_velocity('cartpole', np.array([0., force / mass * 0.02]))

    def done(self, state: dict) -> bool:
        theta = state['things']['cartpole'][Angle]
        if not(_CARTPOLE_ANGLE_RANGE[0] / 2. < theta < _CARTPOLE_ANGLE_RANGE[1] / 2.):
            self.terminated = True
            self.steps_terminated += 1
        return self.terminated

    def results(self, state: dict) -> dict:
        """Returns results to be processed by the MDP.
        :param state: State dictionary to pull results from.
        :return: Dictionary of results.
        """
        thetastr = 'th'
        theta = state['things']['cartpole'][Angle].value
        thetadot = state['things']['cartpole'][AngularVelocity].value
        thetadotdot = state['things']['cartpole'][AngularAcceleration].value
        return {f'{thetastr}': theta, f'{thetastr}\'': thetadot, f'{thetastr}\'\'': thetadotdot}


class CartPoleObservation(StateObservation):
    """Gym state-based observation space for cartpole.

    """
    def __init__(self):
        super().__init__('cartpole')

    def _make_observation_space(self) -> None:
        self._observation_space = Box(-100, 100, shape=(4,), dtype=np.float32)

    def __call__(self, states: dict) -> np.ndarray:
        cartpole_state = states['things']['cartpole']
        pos_x = to_cartpole_basis(cartpole_state[Position][1])
        vel_x = to_cartpole_basis(cartpole_state[Velocity][1])
        theta = cartpole_state[Angle].value
        thetadot = cartpole_state[AngularVelocity].value
        return np.array([pos_x, vel_x, theta, thetadot])
