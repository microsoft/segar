__copyright__ = "Copyright (c) Microsoft Corporation and Mila - Quebec AI Institute"
__license__ = "MIT"
"""Puttputt game. Get ball as close as possible to the goal.

"""

__all__ = (
    "PuttPuttInitialization",
    "PuttPutt",
    "Invisiball",
    "puttputt_default_config",
    "puttputt_random_middle_config",
    "invisiball_config",
)

from typing import Optional

from gym.spaces import Box
import numpy as np

from segar.mdps.initializations import ArenaInitialization
from segar.mdps.rewards import dead_reward_fn, l2_distance_reward_fn
from segar.mdps.tasks import Task
from segar.rendering.rgb_rendering import register_color
from segar.factors import (
    Position,
    Label,
    Mass,
    Charge,
    Shape,
    Text,
    ID,
    Order,
    Circle,
    DiscreteRangeNoise,
    RandomConvexHull,
    GaussianNoise,
    Size,
    GaussianMixtureNoise,
    Mobile,
    Magnetism,
    UniformNoise,
    Friction,
    Alive,
    Done,
    Velocity,
    Acceleration,
    Visible,
)
from segar.rules import Prior
from segar.things import (
    Object,
    Tile,
    ThingFactory,
    Charger,
    Magnet,
    Bumper,
    Damper,
    Ball,
    SandTile,
    MagmaTile,
    Hole,
    FireTile,
    Entity,
)
from segar.sim.location_priors import (
    RandomEdgeLocation,
    RandomBottomLocation,
    RandomTopLocation,
    RandomMiddleLocation,
)


_DEFAULT_GOLFBALL_MASS = 1.0
_DEFAULT_GOLFBALL_CHARGE = 1.0
_DEFAULT_DEAD_REWARD = -100.0
_GOAL_DISTANCE_THRESH = 1e-2
_MAX_BALL_AT_GOAL_VEL = None
_ACTION_RANGE = (-100, 100)


class GolfBall(
    Object,
    default={
        Label: "golfball",
        Mass: _DEFAULT_GOLFBALL_MASS,
        Charge: _DEFAULT_GOLFBALL_CHARGE,
        Shape: Circle(0.2),
        Text: "X",
        ID: "golfball",
    },
):
    pass


class GoalTile(
    Tile,
    default={Label: "goal", Order: -2, Shape: Circle(0.3), Text: "G", ID: "goal"},
):
    pass


puttputt_default_config = {
    "numbers": [
        (ThingFactory([Charger, Magnet, Bumper, Damper, Ball]), DiscreteRangeNoise(1, 2)),
        (
            ThingFactory({SandTile: 2 / 5.0, MagmaTile: 1 / 5.0, Hole: 1 / 5.0, FireTile: 1 / 5.0}),
            DiscreteRangeNoise(1, 2),
        ),
        (GoalTile, 1),
        (GolfBall, 1),
    ],
    "priors": [
        Prior(Position, RandomEdgeLocation(), entity_type=Object),
        Prior(Position, RandomMiddleLocation(), entity_type=Tile),
        Prior(Position, RandomBottomLocation(), entity_type=GolfBall),
        Prior(Position, RandomTopLocation(), entity_type=GoalTile),
        Prior(Shape, RandomConvexHull(0.3), entity_type=Tile),
        Prior(Shape, Circle(0.3), entity_type=GoalTile),
        Prior(Shape, Circle(0.4), entity_type=Hole),
        Prior(Size, GaussianNoise(0.2, 0.01, clip=(0.1, 0.3)), entity_type=Object),
        Prior(Size, GaussianNoise(1.0, 0.01, clip=(0.5, 1.1)), entity_type=Tile),
        Prior(Mass, 1.0),
        Prior(Mobile, False),
        Prior(Mobile, True, entity_type=Ball),
        Prior(Mobile, True, entity_type=GolfBall),
        Prior(
            Charge, GaussianMixtureNoise(means=[-1.0, 1.0], stds=[0.1, 0.1]), entity_type=Charger
        ),
        Prior(
            Magnetism, GaussianMixtureNoise(means=[-1.0, 1.0], stds=[0.1, 0.1]), entity_type=Magnet
        ),
        Prior(Friction, UniformNoise(0.2, 1.0), entity_type=SandTile),
    ],
}


puttputt_random_middle_config = {
    "numbers": [
        (
            ThingFactory(
                [Charger, Magnet, Bumper, Damper, Object, SandTile, MagmaTile, Hole, FireTile]
            ),
            1,
        ),
        (GoalTile, 1),
        (GolfBall, 1),
    ],
    "priors": [
        Prior(Position, RandomMiddleLocation()),
        Prior(Position, RandomBottomLocation(), entity_type=GolfBall),
        Prior(Position, RandomTopLocation(), entity_type=GoalTile),
        Prior(Shape, RandomConvexHull(0.3), entity_type=Tile),
        Prior(Shape, Circle(0.3), entity_type=GoalTile),
        Prior(Size, GaussianNoise(0.2, 0.01, clip=(0.1, 0.3)), entity_type=Object),
        Prior(Size, GaussianNoise(1.0, 0.01, clip=(0.5, 1.5)), entity_type=Tile),
        Prior(Mass, 1.0),
        Prior(Mobile, True),
        Prior(
            Charge, GaussianMixtureNoise(means=[-1.0, 1.0], stds=[0.1, 0.1]), entity_type=Charger
        ),
        Prior(
            Magnetism, GaussianMixtureNoise(means=[-1.0, 1.0], stds=[0.1, 0.1]), entity_type=Magnet
        ),
        Prior(Friction, UniformNoise(0.2, 1.0), entity_type=SandTile),
    ],
}


invisiball_config = {
    "numbers": [(Charger, DiscreteRangeNoise(1, 3)), (GoalTile, 1), (GolfBall, 1)],
    "priors": [
        Prior(Position, RandomMiddleLocation(), entity_type=Charger),
        Prior(Position, RandomBottomLocation(), entity_type=GolfBall),
        Prior(Position, RandomTopLocation(), entity_type=GoalTile),
        Prior(Size, GaussianNoise(0.2, 0.01, clip=(0.1, 0.25)), entity_type=Object),
        Prior(Mass, 1.0),
        Prior(Mobile, True),
        Prior(Charge, 1.0, entity_type=Charger),
    ],
}


class PuttPuttInitialization(ArenaInitialization):
    """Basic putt-putt initialization based on the arena initialization.

    Adds a golf ball and goal. Both ball and goal are circles.

    """

    def __init__(self, config: dict = None):

        has_golfball = False
        has_goal = False

        for entity_type, _ in config.get("numbers", []):
            if entity_type == GolfBall:
                has_golfball = True
            if entity_type == GoalTile:
                has_goal = True

        if not has_golfball:
            raise KeyError(f"`{GolfBall}` number must be defined in this " "initialization.")
        if not has_goal:
            raise KeyError(f"`{GoalTile}` number must be defined in this " "initialization.")

        super().__init__(config=config)

        self.golfball_id = None
        self.goal_id = None

        # These are new things so we need to define their color for some
        # rendering.
        register_color("goal", (0, 0, 255))
        register_color("golfball", (255, 255, 255))

    def sample(self, max_iterations: int = 100) -> list[Entity]:
        sampled_things = super().sample(max_iterations=max_iterations)
        has_golfball = False
        has_goal = False

        for thing in sampled_things:
            if isinstance(thing, GolfBall):
                has_golfball = True
                self.golfball_id = thing[ID]
            if isinstance(thing, GoalTile):
                has_goal = True
                self.goal_id = thing[ID]

        if not has_golfball:
            raise ValueError("`golfball` params weren't created.")
        if not has_goal:
            raise ValueError("`goal` params weren't created.")

        return sampled_things

    def set_arena(self, init_things: Optional[list[Entity]] = None) -> None:
        super().set_arena(init_things)
        if self.goal_id is None:
            raise RuntimeError(
                f"Goal was not set in arena. Config is "
                f"{self._config} and params are "
                f"{self._numbers}"
            )

        if self.golfball_id is None:
            raise RuntimeError(
                f"Golfball was not set in arena. Config is "
                f"{self._config} and params are "
                f"{self._numbers}."
            )


class PuttPutt(Task):
    """The putt-putt task."""

    def __init__(
        self,
        initialization: PuttPuttInitialization,
        action_range: tuple[float, float] = _ACTION_RANGE,
        action_shape: tuple[int, ...] = (2,),
        dead_reward: float = _DEFAULT_DEAD_REWARD,
        goal_distance_threshold: float = _GOAL_DISTANCE_THRESH,
        max_ball_at_goal_velocity: float = _MAX_BALL_AT_GOAL_VEL,
    ):
        """

        :param initialization: Initialization object used for initializing
        the arena.
        :param action_range: Range of actions used by the agent.
        :param action_shape: Shape of actions.
        :param dead_reward: Reward when golf ball is `dead`.
        :param goal_distance_threshold: Distance between golf ball and goal
        under which to stop.
        :param max_ball_at_goal_velocity: Max golf ball velocity under which to
         stop.
        """

        action_type = np.float32
        action_space = Box(
            action_range[0], action_range[1], shape=action_shape, dtype=action_type,
        )
        super().__init__(
            action_space=action_space,
            initialization=initialization,
        )

        self._dead_reward = dead_reward
        self._goal_distance_threshold = goal_distance_threshold
        self._max_ball_at_goal_velocity = max_ball_at_goal_velocity

    @property
    def golfball_id(self) -> ID:
        if not hasattr(self._initialization, "golfball_id"):
            raise AttributeError(
                "Initialization must define `golfball_id` to " "be compatible with task."
            )
        ball_id = self._initialization.golfball_id
        if ball_id is None:
            raise ValueError("`ball_id` is not set yet.")
        return ball_id

    @property
    def goal_id(self) -> ID:
        if not hasattr(self._initialization, "goal_id"):
            raise AttributeError(
                "Initialization must define `goal_id` to " "be compatible with task."
            )
        goal_id = self._initialization.goal_id
        if goal_id is None:
            raise ValueError("`goal_id` is not set yet.")
        return goal_id

    def distance(self) -> float:
        """Calculate euclidean distance to goal, normalized by arena diagonal.

        :return: float, Distance of ball to goal
        """

        return self.sim.l2_distance(self.golfball_id, self.goal_id)

    def reward(self, state: dict) -> float:
        """Reward function.

        Determined by distance of golf ball from goal and whether the golf
        ball is dead.

        :param state: States.
        :return: (float) reward.
        """
        ball_state = state["things"][self.golfball_id]
        dead_reward = dead_reward_fn(ball_state, self._dead_reward)

        # Distance reward is tricky: can't do it directly from states
        # because sim owns scaling
        distance = self.sim.l2_distance(self.golfball_id, self.goal_id)
        distance_reward = l2_distance_reward_fn(distance)
        return dead_reward + distance_reward

    def done(self, state: dict) -> bool:
        """Stopping condition.

        :param state: States.
        :return: (bool) whether to stop.
        """
        ball_state = state["things"][self.golfball_id.value]
        is_finished = ball_state[Done] or not ball_state[Alive]
        at_goal = bool(self.distance() <= self._goal_distance_threshold)
        under_vel = bool(
            (self._max_ball_at_goal_velocity is None)
            or (ball_state[Velocity].norm() <= self._max_ball_at_goal_velocity)
        )
        return is_finished or (at_goal and under_vel)

    def results(self, state: dict) -> dict:
        """Results for monitoring task.

        :param state: States
        :return: Dictionary of results.
        """
        distance = self.sim.l2_distance(self.golfball_id, self.goal_id)
        ball_state = state["things"][self.golfball_id]
        return dict(
            dist_to_goal=distance,
            velocity=ball_state[Velocity].norm(),
            acceleration=ball_state[Acceleration].norm(),
            mass=ball_state[Mass].value,
            alive=ball_state[Alive].value,
        )

    def apply_action(self, force: np.ndarray) -> None:
        """Applies force to the golf ball.

        :param force: (np.array) magnitude and velocity of force.
        """
        self.sim.add_force(self.golfball_id, force)

    def demo_action(self) -> np.ndarray:
        """Generate an action used for demos

        :return: np.array action
        """
        return np.random.normal() + np.array((2, 1.5))


class Invisiball(PuttPutt):
    def apply_action(self, force):
        """Applies force to the golf ball.

        :param force: (np.array) magnitude and velocity of force.
        """
        self.sim.add_force(self.golfball_id, force)
        self.sim.change_thing_state(self.golfball_id, Visible, False)


class PuttPuttNegDist(PuttPutt):
    def reward(self, state: dict) -> float:
        """same as original but with neg dist, not neg squared dist"""
        ball_state = state["things"][self.golfball_id]
        dead_reward = dead_reward_fn(ball_state, self._dead_reward)
        distance = self.sim.l2_distance(self.golfball_id, self.goal_id)
        distance_reward = -distance
        return dead_reward + distance_reward
