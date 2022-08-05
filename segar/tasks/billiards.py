__copyright__ = "Copyright (c) Microsoft Corporation and Mila - Quebec AI Institute"
__license__ = "MIT"
"""Billiards game

"""

__all__ = ("billiards_default_config", "Billiards", "BilliardsInitialization")

import math
from typing import Optional

from gym.spaces import Box
import numpy as np

from segar.mdps.initializations import ArenaInitialization
from segar.mdps.rewards import dead_reward_fn, l2_distance_reward_fn
from segar.mdps.tasks import Task
from segar.rendering.rgb_rendering import register_color
from segar.factors import (
    Label,
    Mass,
    Charge,
    Shape,
    Text,
    Circle,
    GaussianNoise,
    Size,
    Position,
    ID,
    Done,
    Alive,
    Visible,
    Velocity,
)
from segar.rules import Prior
from segar.things import Ball, Hole, Entity, Object
from segar.sim.location_priors import RandomBottomLocation


_DEFAULT_CUEBALL_MASS = 1.0
_DEFAULT_CUEBALL_CHARGE = 1.0
_DEFAULT_BALL_MASS = 1.0
_DEFAULT_BALL_SIZE = 0.2
_DEFAULT_BALL_CHARGE = 1.0
_DEFAULT_HOLE_SIZE = 0.3
_DEFAULT_DEAD_REWARD = -100.0
_HOLE_DISTANCE_THRESH = 1e-4
_MAX_BALL_AT_GOAL_VEL = None
_ACTION_RANGE = (-100, 100)


def billiard_ball_positions(
    start: list[float, float], r: float = _DEFAULT_BALL_SIZE / 2 + 1e-3, n: int = 10
) -> list[list[float, float]]:
    x, y = start
    sq2r = math.sqrt(2.0) * r
    positions = [start]
    positions += [[x - sq2r, y + sq2r], [x + sq2r, y + sq2r]]
    positions += [
        [x - 2 * sq2r, y + 2 * sq2r],
        [x, y + 2 * sq2r],
        [x + 2 * sq2r, y + 2 * sq2r],
    ]
    positions += [
        [x - 3 * sq2r, y + 3 * sq2r],
        [x - sq2r, y + 3 * sq2r],
        [x + sq2r, y + 3 * sq2r],
        [x + 3 * sq2r, y + 3 * sq2r],
    ]
    positions = positions[:n]
    return positions


class CueBall(
    Object,
    default={
        Label: "cueball",
        Mass: _DEFAULT_CUEBALL_MASS,
        Charge: _DEFAULT_CUEBALL_CHARGE,
        Shape: Circle(0.2),
        Text: "X",
        ID: "cueball",
    },
):
    pass


billiards_default_config = {
    "numbers": [(CueBall, 1)],
    "priors": [
        Prior(
            Size,
            GaussianNoise(
                _DEFAULT_BALL_SIZE,
                0.01,
                clip=(_DEFAULT_BALL_SIZE / 2.0, 3 * _DEFAULT_BALL_SIZE / 2.0),
            ),
            entity_type=CueBall,
        ),
        Prior(Size, _DEFAULT_BALL_SIZE, entity_type=Ball),
        Prior(Mass, _DEFAULT_BALL_MASS, entity_type=Ball),
        Prior(Size, _DEFAULT_HOLE_SIZE, entity_type=Hole),
        Prior(Position, RandomBottomLocation(), entity_type=CueBall),
    ],
}


class BilliardsInitialization(ArenaInitialization):
    """Initialization of billiards derived from arena initialization.

    Adds a cueball, holes, and other billiard balls.

    """

    def __init__(self, config=None):

        self.cueball_id = None
        self.ball_ids = []
        self.hole_ids = []

        super().__init__(config=config)

        register_color("cueball", (255, 255, 255))

    def sample(self, max_iterations: int = 100) -> list[Entity]:
        self.ball_ids.clear()
        self.hole_ids.clear()
        sampled_things = super().sample(max_iterations=max_iterations)

        ball_positions = billiard_ball_positions([0.0, 0.0])

        for i, pos in enumerate(ball_positions):
            ball = Ball({Position: pos, Text: f"{i + 1}", ID: f"{i + 1}_ball"})
            sampled_things.append(ball)

        hole_positions = [[-0.9, -0.9], [-0.9, 0.9], [0.9, -0.9], [0.9, 0.9]]
        for i, pos in enumerate(hole_positions):
            hole = Hole({Position: pos, ID: f"{i}_hole", Size: _DEFAULT_HOLE_SIZE})
            sampled_things.append(hole)

        has_cueball = False
        has_balls = False
        has_holes = False

        for thing in sampled_things:
            if isinstance(thing, CueBall):
                has_cueball = True
                self.cueball_id = thing[ID]
            if isinstance(thing, Ball):
                has_balls = True
                self.ball_ids.append(thing[ID])
            if isinstance(thing, Hole):
                has_holes = True
                self.hole_ids.append(thing[ID])

        if not has_cueball:
            raise ValueError("cueball wasn't created.")
        if not has_balls:
            raise ValueError("balls weren't created.")
        if not has_holes:
            raise ValueError("holes weren't created.")

        return sampled_things

    def set_arena(self, init_things: Optional[list[Entity]] = None) -> None:
        super().set_arena(init_things)
        if self.cueball_id is None:
            raise RuntimeError("Cueball was not set in arena.")

        if len(self.ball_ids) == 0:
            raise RuntimeError("Balls not set in arena.")

        if len(self.hole_ids) == 0:
            raise RuntimeError("Holes not set in arena.")


class Billiards(Task):
    """Billiards game.

    Agent controls the cue ball. Hit the cue ball into billiard balls and
    get them into holes. Avoid getting the cue ball into the holes.

    """

    def __init__(
        self,
        initialization: BilliardsInitialization,
        action_range: tuple[float, float] = _ACTION_RANGE,
        action_shape: tuple[int, ...] = (2,),
        dead_reward: float = _DEFAULT_DEAD_REWARD,
        hole_distance_threshold: float = _HOLE_DISTANCE_THRESH,
        max_ball_at_hole_velocity: float = _MAX_BALL_AT_GOAL_VEL,
    ):
        """

        :param initialization: Initialization object used for initializing
        the arena.
        :param action_range: Range of actions used by the agent.
        :param action_shape: Shape of actions.
        :param dead_reward: Reward when cue ball is `dead`.
        :param hole_distance_threshold: Distance between billiard ball and hole
        under which to stop.
        :param max_ball_at_hole_velocity: Max billiard ball velocity under
        which to stop.
        """

        action_type = np.float16
        action_space = Box(
            action_range[0], action_range[1], shape=action_shape, dtype=action_type,
        )
        super().__init__(
            action_space=action_space,
            initialization=initialization,
        )

        self._dead_reward = dead_reward
        self._hole_distance_threshold = hole_distance_threshold
        self._max_ball_at_hole_velocity = max_ball_at_hole_velocity

    @property
    def cueball_id(self) -> ID:
        if not hasattr(self._initialization, "cueball_id"):
            raise AttributeError(
                "Initialization must define `cueball_id` to " "be compatible with task."
            )
        cueball_id = self._initialization.cueball_id
        if cueball_id is None:
            raise ValueError("`cueball_id` is not set yet.")
        return cueball_id

    @property
    def hole_ids(self) -> list[ID]:
        if not hasattr(self._initialization, "hole_ids"):
            raise AttributeError(
                "Initialization must define `hole_ids` to " "be compatible with task."
            )
        hole_ids = self._initialization.hole_ids
        return hole_ids

    @property
    def ball_ids(self) -> list[ID]:
        if not hasattr(self._initialization, "ball_ids"):
            raise AttributeError(
                "Initialization must define `ball_ids` to " "be compatible with task."
            )
        ball_ids = self._initialization.ball_ids
        return ball_ids

    def reward(self, state: dict) -> float:
        """Reward determined by the distance of the billiard balls to the
        nearest hold and whether the cue ball is in a hole (dead).

        :param state: States
        :return: (float) the reward.
        """
        ball_state = state["things"][self.cueball_id]
        dead_reward = dead_reward_fn(ball_state, self._dead_reward)

        # Distance reward is tricky: can't do it directly from states
        # because sim owns scaling
        distance_reward = 0.0
        for ball_id in self.ball_ids:
            distance = min([self.sim.l2_distance(ball_id, hole_id) for hole_id in self.hole_ids])
            if distance <= self._hole_distance_threshold:
                self.sim.change_thing_state(ball_id, Alive, False)
                self.sim.change_thing_state(ball_id, Visible, False)
            distance_reward += l2_distance_reward_fn(distance)
        return dead_reward + distance_reward

    def done(self, state: dict) -> bool:
        """Episode is done if the cue ball is dead or if all of the billiard
        balls are in the holes.

        :param state: The states.
        :return: True if the state indicates the environment is done.
        """

        ball_state = state["things"][self.cueball_id]
        is_finished = ball_state[Done] or not ball_state[Alive]

        balls_are_finished = True
        for ball_id in self.ball_ids:
            ball_state = state["things"][ball_id]
            ball_is_finished = ball_state[Done] or not ball_state[Alive]
            balls_are_finished = balls_are_finished and ball_is_finished

        return is_finished or balls_are_finished

    def apply_action(self, force: np.ndarray) -> None:
        """Applies force to the cue ball.

        :param force: (np.array) Force to apply
        """
        self.sim.add_force(self.cueball_id, force)

    def results(self, state: dict) -> dict:
        """Results for monitoring task.

        :param state: States
        :return: Dictionary of results.
        """

        distance = min(
            [self.sim.l2_distance(self.cueball_id, hole_id) for hole_id in self.hole_ids]
        )
        ball_state = state["things"][self.cueball_id]
        return dict(
            dist_to_goal=distance,
            velocity=ball_state[Velocity].norm(),
            mass=ball_state[Mass].value,
            alive=ball_state[Alive].value,
        )

    def demo_action(self):
        """Generate an action used for demos

        :return: np.array action
        """
        return np.random.normal() + np.array((4, 3))
