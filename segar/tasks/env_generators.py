__copyright__ = "Copyright (c) Microsoft Corporation and Mila - Quebec AI Institute"
__license__ = "MIT"

import logging
from typing import Callable, Tuple

from segar.configs.handler import get_env_config
from segar.factors import Position, Shape, Size, RandomConvexHull, Circle, GaussianNoise, GaussianMixtureNoise, \
        StoredEnergy, Charge, Magnetism, Mass, Mobile, Friction, UniformNoise, Heat
from segar.mdps import MDP, RGBObservation
from segar.rules import Prior
from segar.sim import Simulator
from segar.sim.location_priors import CenterLocation, RandomTopRightLocation, RandomTopLocation, \
    RandomBottomLocation, \
    RandomUniformLocation, RandomMiddleLocation
from segar.tasks.classic_control import cartpole_init_config_a, cartpole_init_config_b, cartpole_init_config_c, \
    mountaincar_init_config_a, mountaincar_init_config_b, mountaincar_init_config_c, MountainCarTask, \
    MountainCarObservation, CustomMountainCarInitialization, CartPoleObservation, CartPoleTask, \
    CustomCartPoleInitialization, from_mountaincar_basis
from segar.tasks.puttputt import GoalTile, GolfBall, PuttPutt, PuttPuttInitialization, \
    _ACTION_RANGE as _PUTTPUTT_ACTION_RANGE
from segar.things import Tile, ThingFactory, Charger, Magnet, Bumper, Damper, Ball, Object, SandTile, FireTile, \
    MagmaTile, Hole


logger = logging.getLogger(__name__)

_ENV_GENERATORS = {}


def register_generator(name: str, generator):
    global _ENV_GENERATORS

    if name not in _ENV_GENERATORS:
        _ENV_GENERATORS[name] = generator
    else:
        raise KeyError(f'Environment generator with name {name} already in use.')


def get_env_generator(string_to_parse: str) -> Callable:
    logger.info(f'Parsing environment from {string_to_parse}.')
    try:
        env_name = string_to_parse.split('-')[0]
        rest = '-'.join(string_to_parse.split('-')[1:])
    except ValueError:
        raise ValueError('Environment specification must start as `<env_name>-XXXX`')

    if env_name in _ENV_GENERATORS:
        gen = _ENV_GENERATORS[env_name]
        try:
            generator = gen(rest)
        except ValueError:
            raise ValueError(f'Environment generator {env_name} failed to parse string {string_to_parse}.')
        return generator
    else:
        raise ValueError(f'Environment generator {env_name} not found, found {list(_ENV_GENERATORS.keys())}.')


def puttputt_env_generator(string_to_parse: str) -> Callable:
    try:
        task_name, task_distr, obs_type = string_to_parse.split("-")
    except ValueError:
        return None

    task_name, k = task_name.split("x")
    k = int(k)

    init_config = None

    if task_name == "empty":
        if task_distr == "easy":
            init_config = {
                "numbers": [(GoalTile, 1), (GolfBall, 1)],
                "priors": [
                    Prior(Position, CenterLocation(), entity_type=GolfBall),
                    Prior(Position, RandomTopRightLocation(), entity_type=GoalTile),
                    Prior(Shape, RandomConvexHull(0.3), entity_type=Tile),
                    Prior(Shape, Circle(0.4), entity_type=GoalTile),
                    Prior(Size, GaussianNoise(1.0, 0.01, clip=(0.5, 1.5)), entity_type=Tile),
                ],
            }
        elif task_distr == "medium":
            init_config = {
                "numbers": [(GoalTile, 1), (GolfBall, 1)],
                "priors": [
                    Prior(Position, RandomBottomLocation(), entity_type=GolfBall),
                    Prior(Position, RandomTopLocation(), entity_type=GoalTile),
                    Prior(Shape, RandomConvexHull(0.3), entity_type=Tile),
                    Prior(Shape, Circle(0.4), entity_type=GoalTile),
                    Prior(Size, GaussianNoise(1.0, 0.01, clip=(0.5, 1.5)), entity_type=Tile),
                ],
            }
        elif task_distr == "hard":
            init_config = {
                "numbers": [(GoalTile, 1), (GolfBall, 1)],
                "priors": [
                    Prior(Position, RandomUniformLocation(), entity_type=GolfBall),
                    Prior(Position, RandomUniformLocation(), entity_type=GoalTile),
                    Prior(Shape, RandomConvexHull(0.3), entity_type=Tile),
                    Prior(Shape, Circle(0.4), entity_type=GoalTile),
                    Prior(Size, GaussianNoise(1.0, 0.01, clip=(0.5, 1.5)), entity_type=Tile),
                ],
            }
    elif task_name == "objects":
        if task_distr == "easy":
            init_config = {
                "numbers": [
                    (ThingFactory([Charger, Magnet, Bumper, Damper, Ball]), k),
                    (GoalTile, 1),
                    (GolfBall, 1),
                ],
                "priors": [
                    Prior(Position, CenterLocation(), entity_type=GolfBall),
                    Prior(Position, RandomTopLocation(), entity_type=Object),
                    Prior(Position, RandomTopRightLocation(), entity_type=GoalTile),
                    Prior(Shape, RandomConvexHull(0.3), entity_type=Tile),
                    Prior(Shape, Circle(0.4), entity_type=GoalTile),
                    Prior(Size, GaussianNoise(1.0, 0.01, clip=(0.5, 1.5)), entity_type=Tile),
                ],
            }
        elif task_distr == "medium":
            init_config = {
                "numbers": [
                    (ThingFactory([Charger, Magnet, Bumper, Damper, Ball]), k),
                    (GoalTile, 1),
                    (GolfBall, 1),
                ],
                "priors": [
                    Prior(Position, RandomBottomLocation(), entity_type=GolfBall),
                    Prior(Position, RandomMiddleLocation(), entity_type=Object),
                    Prior(Position, RandomTopLocation(), entity_type=GoalTile),
                    Prior(Shape, RandomConvexHull(0.3), entity_type=Tile),
                    Prior(Shape, Circle(0.4), entity_type=GoalTile),
                    Prior(Size, GaussianNoise(1.0, 0.01, clip=(0.5, 1.5)), entity_type=Tile),
                    Prior(
                        Charge,
                        GaussianMixtureNoise(means=[-1.0, 1.0], stds=[0.1, 0.1]),
                        entity_type=Charger,
                    ),
                    Prior(
                        Magnetism,
                        GaussianMixtureNoise(means=[-1.0, 1.0], stds=[0.1, 0.1]),
                        entity_type=Magnet,
                    ),
                    Prior(
                        StoredEnergy,
                        GaussianNoise(1.0, 0.1, clip=(0.5, 1.0)),
                        entity_type=Bumper,
                    ),
                    Prior(
                        StoredEnergy,
                        GaussianNoise(-1.0, 0.1, clip=(-0.5, -1.0)),
                        entity_type=Damper,
                    ),
                ],
            }
        elif task_distr == "hard":
            init_config = {
                "numbers": [
                    (ThingFactory([Charger, Magnet, Bumper, Damper, Ball]), k),
                    (GoalTile, 1),
                    (GolfBall, 1),
                ],
                "priors": [
                    Prior(Position, RandomUniformLocation(), entity_type=GolfBall),
                    Prior(Position, RandomUniformLocation(), entity_type=Object),
                    Prior(Position, RandomUniformLocation(), entity_type=GoalTile),
                    Prior(Shape, RandomConvexHull(0.3), entity_type=Tile),
                    Prior(Shape, Circle(0.4), entity_type=GoalTile),
                    Prior(Size, GaussianNoise(0.2, 0.01, clip=(0.1, 0.3)), entity_type=Object),
                    Prior(Mass, 1.0),
                    Prior(Mobile, True),
                    Prior(
                        Charge,
                        GaussianMixtureNoise(means=[-1.0, 1.0], stds=[0.4, 0.4]),
                        entity_type=Charger,
                    ),
                    Prior(
                        Magnetism,
                        GaussianMixtureNoise(means=[-1.0, 1.0], stds=[0.4, 0.4]),
                        entity_type=Magnet,
                    ),
                    Prior(
                        StoredEnergy,
                        GaussianNoise(1.0, 0.4, clip=(0.0, 1.0)),
                        entity_type=Bumper,
                    ),
                    Prior(
                        StoredEnergy,
                        GaussianNoise(-1.0, 0.4, clip=(-0.5, -1.0)),
                        entity_type=Damper,
                    ),
                ],
            }
    elif task_name == "tiles":
        if task_distr == "easy":
            init_config = {
                "numbers": [
                    (
                        ThingFactory(
                            {
                                SandTile: 1 / 4.0,
                                MagmaTile: 1 / 4.0,
                                Hole: 1 / 4.0,
                                FireTile: 1 / 4.0,
                            }
                        ),
                        k,
                    ),
                    (GoalTile, 1),
                    (GolfBall, 1),
                ],
                "priors": [
                    Prior(Position, CenterLocation(), entity_type=GolfBall),
                    Prior(Position, RandomTopLocation(), entity_type=Tile),
                    Prior(Position, RandomTopRightLocation(), entity_type=GoalTile),
                    Prior(Shape, RandomConvexHull(0.3), entity_type=Tile),
                    Prior(Shape, Circle(0.4), entity_type=GoalTile),
                    Prior(Size, GaussianNoise(1.0, 0.01, clip=(0.5, 1.5)), entity_type=Tile),
                    Prior(Mass, 1.0),
                    Prior(Mobile, True),
                    Prior(Friction, UniformNoise(0.5, 0.55), entity_type=Tile),
                    Prior(Heat, UniformNoise(0.5, 0.55), entity_type=Tile),
                ],
            }
        elif task_distr == "medium":
            init_config = {
                "numbers": [
                    (
                        ThingFactory(
                            {
                                SandTile: 1 / 4.0,
                                MagmaTile: 1 / 4.0,
                                Hole: 1 / 4.0,
                                FireTile: 1 / 4.0,
                            }
                        ),
                        k,
                    ),
                    (GoalTile, 1),
                    (GolfBall, 1),
                ],
                "priors": [
                    Prior(Position, RandomBottomLocation(), entity_type=GolfBall),
                    Prior(Position, RandomMiddleLocation(), entity_type=Tile),
                    Prior(Position, RandomTopLocation(), entity_type=GoalTile),
                    Prior(Shape, RandomConvexHull(0.3), entity_type=Tile),
                    Prior(Shape, Circle(0.4), entity_type=GoalTile),
                    Prior(Size, GaussianNoise(1.0, 0.01, clip=(0.5, 1.5)), entity_type=Tile),
                    Prior(Mass, 1.0),
                    Prior(Mobile, True),
                    Prior(Friction, UniformNoise(0.5, 0.55), entity_type=Tile),
                    Prior(Heat, UniformNoise(0.5, 0.55), entity_type=Tile),
                ],
            }
        elif task_distr == "hard":
            init_config = {
                "numbers": [
                    (
                        ThingFactory(
                            {
                                SandTile: 1 / 4.0,
                                MagmaTile: 1 / 4.0,
                                Hole: 1 / 4.0,
                                FireTile: 1 / 4.0,
                            }
                        ),
                        k,
                    ),
                    (GoalTile, 1),
                    (GolfBall, 1),
                ],
                "priors": [
                    Prior(Position, RandomUniformLocation(), entity_type=GolfBall),
                    Prior(Position, RandomUniformLocation(), entity_type=Tile),
                    Prior(Position, RandomUniformLocation(), entity_type=GoalTile),
                    Prior(Shape, RandomConvexHull(0.3), entity_type=Tile),
                    Prior(Shape, Circle(0.4), entity_type=GoalTile),
                    Prior(Size, GaussianNoise(0.2, 0.01, clip=(0.1, 0.3)), entity_type=Tile),
                    Prior(Mass, 1.0),
                    Prior(Mobile, True),
                    Prior(Friction, UniformNoise(0.4, 1.0), entity_type=Tile),
                    Prior(Heat, UniformNoise(0.4, 1.0), entity_type=Tile),
                ],
            }
    else:
        raise ValueError(f"Puttputt task {task_name}-{task_distr} not yet implemented")

    if init_config is None:
        raise ValueError('Failed to parse initialization from string.')

    logger.info(f'Initialization configuration: {init_config}')

    def generator(
            i: int,
            wall_damping: float = 0.025,
            friction: float = 0.05,
            max_steps: int = 50,
            resolution: int = 64,
            action_range: Tuple[float, float] = _PUTTPUTT_ACTION_RANGE,
            save_path: str = "sim.state"):
        initialization = PuttPuttInitialization(config=init_config.copy())
        task = PuttPutt(action_range=action_range, initialization=initialization)

        sim = Simulator(
            state_buffer_length=50,
            wall_damping=wall_damping,
            friction=friction,
            safe_mode=False,
            save_path=save_path + str(i),
        )

        if obs_type == "rgb":
            visual_config = get_env_config("visual", "linear_ae", dist_name="baseline")
            obs = RGBObservation(resolution=resolution, config=visual_config)
        else:
            raise ValueError

        task.set_sim(sim)
        task.sample()
        mdp = MDP(obs, task, max_steps_per_episode=max_steps, sim=sim,
                  episodes_per_arena=float("inf"), sub_steps=5)

        return mdp

    return generator


def classic_control_env_generator(string_to_parse: str) -> Callable:

    task_name, task_id = string_to_parse.split('-')

    if task_name == 'cartpole':
        task_cls = CartPoleTask
        task_obs = CartPoleObservation
        task_init = CustomCartPoleInitialization
        max_steps_ = 500
        framerate_ = 50
        friction_ = 0.
        max_velocity_ = 1000.
        gravity_ = 9.8
        if task_id == 'a':
            init_config = cartpole_init_config_a.copy()
        elif task_id == 'b':
            init_config = cartpole_init_config_b.copy()
        elif task_id == 'c':
            init_config = cartpole_init_config_c.copy()
        else:
            raise ValueError(f'No cartpole env with dist {task_id}')
    elif task_name == 'mountaincar':
        friction_ = 0.
        max_velocity_ = from_mountaincar_basis(0.7, recenter=False)
        gravity_ = 0.0025
        framerate_ = 1
        task_cls = MountainCarTask
        task_obs = MountainCarObservation
        task_init = CustomMountainCarInitialization
        max_steps_ = 200
        if task_id == 'a':
            init_config = mountaincar_init_config_a.copy()
        elif task_id == 'b':
            init_config = mountaincar_init_config_b.copy()
        elif task_id == 'c':
            init_config = mountaincar_init_config_c.copy()
        else:
            raise ValueError(f'No mountaincar env with dist {task_id}')
    else:
        raise NotImplementedError("Env not yet implemented")

    def generator(
            i: int,
            wall_damping: float = 0.,
            gravity: float = gravity_,
            friction: float = friction_,
            max_velocity: float = max_velocity_,
            framerate: int = framerate_,
            max_steps: int = max_steps_,
            save_path: str = "sim.state"):

        sim = Simulator(
            state_buffer_length=50,
            wall_damping=wall_damping,
            friction=friction,
            gravity=gravity,
            framerate=framerate,
            max_velocity=max_velocity,
            safe_mode=False,
            save_path=save_path + str(i),
        )

        initialization = task_init(config=init_config.copy())
        obs = task_obs()
        task = task_cls(initialization=initialization)

        task.set_sim(sim)
        task.sample()
        mdp = MDP(obs, task, max_steps_per_episode=max_steps, sim=sim,
                  episodes_per_arena=float("inf"), sub_steps=5)

        return mdp

    return generator


register_generator('puttputt', puttputt_env_generator)
register_generator('classic_control', classic_control_env_generator)
