from pprint import pprint

from segar.sim import Simulator
import gym
import numpy as np
from gym.spaces import Box
from segar.configs.handler import get_env_config
from segar.envs.wrappers import SequentialTaskWrapper
from segar.factors import (
    ID,
    Acceleration,
    Alive,
    Charge,
    Circle,
    DiscreteRangeNoise,
    Done,
    Friction,
    Heat,
    StoredEnergy,
    GaussianMixtureNoise,
    GaussianNoise,
    Label,
    Magnetism,
    Mass,
    Mobile,
    Order,
    Position,
    RandomConvexHull,
    Shape,
    Size,
    Text,
    UniformNoise,
    Velocity,
    Visible,
)

from segar.mdps.observations import RGBObservation

from segar.rules import Prior
from segar.sim.location_priors import (
    RandomBottomLocation,
    RandomEdgeLocation,
    RandomUniformLocation,
    RandomMiddleLocation,
    RandomTopLocation,
    CenterLocation,
    RandomTopRightLocation,
)

from segar.tasks.puttputt import GoalTile, GolfBall, PuttPutt, PuttPuttInitialization
from segar.things import (
    Ball,
    Bumper,
    Charger,
    Damper,
    Entity,
    FireTile,
    Hole,
    MagmaTile,
    Magnet,
    Object,
    SandTile,
    ThingFactory,
    Tile,
)


class SEGAREnv(gym.Env):
    def __init__(
        self,
        env_name,
        start_level=0,  # TODO: is this needed?
        num_levels=100,
        num_envs=1,
        resolution=64,
        framestack=1,
        max_steps=50,
        _async=False,
        wall_damping=0.025,
        friction=0.05,
        save_path="sim.state",
        action_max=2,
        seed=123,
    ):
        self.resolution = resolution
        self.action_max = action_max
        self.action_range = [-self.action_max, self.action_max]

        self.env_name = env_name

        task_name, task_distr, obs_type = env_name.split("-")
        task_name, k = task_name.split("x")
        k = int(k)

        if obs_type == "rgb":
            visual_config = get_env_config("visual", "linear_ae", dist_name="baseline")
            obs = RGBObservation(resolution=self.resolution, config=visual_config)

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
                        Prior(Charge, GaussianMixtureNoise(means=[-1.0, 1.0], stds=[0.1, 0.1]), entity_type=Charger),
                        Prior(Magnetism, GaussianMixtureNoise(means=[-1.0, 1.0], stds=[0.1, 0.1]), entity_type=Magnet),
                        Prior(StoredEnergy, GaussianNoise(1.0, 0.1, clip=(0.5, 1.0)), entity_type=Bumper),
                        Prior(StoredEnergy, GaussianNoise(-1.0, 0.1, clip=(-0.5, -1.0)), entity_type=Damper),
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
                        Prior(Charge, GaussianMixtureNoise(means=[-1.0, 1.0], stds=[0.4, 0.4]), entity_type=Charger),
                        Prior(Magnetism, GaussianMixtureNoise(means=[-1.0, 1.0], stds=[0.4, 0.4]), entity_type=Magnet),
                        Prior(StoredEnergy, GaussianNoise(1.0, 0.4, clip=(0.0, 1.0)), entity_type=Bumper),
                        Prior(StoredEnergy, GaussianNoise(-1.0, 0.4, clip=(-0.5, -1.0)), entity_type=Damper),
                    ],
                }
        elif task_name == "tiles":
            if task_distr == "easy":
                init_config = {
                    "numbers": [
                        (ThingFactory({SandTile: 1 / 4.0, MagmaTile: 1 / 4.0, Hole: 1 / 4.0, FireTile: 1 / 4.0}), k),
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
                        (ThingFactory({SandTile: 1 / 4.0, MagmaTile: 1 / 4.0, Hole: 1 / 4.0, FireTile: 1 / 4.0}), k),
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
                        (ThingFactory({SandTile: 1 / 4.0, MagmaTile: 1 / 4.0, Hole: 1 / 4.0, FireTile: 1 / 4.0}), k),
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
                        Prior(Friction, UniformNoise(0.4, 1.0), entity_type=Tile),
                        Prior(Heat, UniformNoise(0.4, 1.0), entity_type=Tile),
                    ],
                }
        else:
            raise NotImplementedError("Env not yet implemented")

        config = dict(max_steps_per_episode=max_steps, episodes_per_arena=float('inf'), sub_steps=5)
        print("==Distribution config==")
        pprint(init_config)
        make_env = lambda: SequentialTaskWrapper(
            obs, init_config, config, self.action_range, num_levels, max_steps, framestack, seed, wall_damping, friction, save_path
        )

        if not _async:
            print("Making sync envs.")
            self.env = gym.vector.SyncVectorEnv(
                [make_env for _ in range(num_envs)],
                observation_space=self.observation_space,
                action_space=self.action_space,
            )
        else:
            print("Making async envs.")
            self.env = gym.vector.AsyncVectorEnv(
                [make_env for _ in range(num_envs)],
                observation_space=self.observation_space,
                action_space=self.action_space,
                shared_memory=True,
                daemon=True,
            )
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.reset()

    def step(self, action):
        # THIS EXPECTS ACTIONS TO BE A VECTOR WITH ONE ACTION FOR EACH ENV

        action_scaled = np.array(action) * self.action_max
        self.env.step_async(action_scaled)
        obs, rew, done, info = self.env.step_wait()
        return obs, rew, done, info

    def reset(self):
        obs = self.env.reset()
        return obs

    def render(self, mode="human"):
        raise NotImplementedError("TODO")


class SEGARSingleEnv(SEGAREnv):
    # TODO yes I'm acutely aware of how shitty of a solution this is but for now it must do
    def __init__(
        self,
        env_name,
        start_level=0,
        num_levels=100,
        num_envs=1,
        resolution=64,
        framestack=1,
        max_steps=50,
        _async=False,
        wall_damping=0.025,
        friction=0.05,
        save_path="sim.state",
        action_max=2,
        seed=123,
    ):
        super().__init__(
            env_name=env_name,
            start_level=start_level,
            num_levels=num_levels,
            num_envs=num_envs,
            resolution=resolution,
            framestack=framestack,
            max_steps=max_steps,
            _async=_async,
            wall_damping=wall_damping,
            friction=friction,
            save_path=save_path,
            action_max=action_max,
            seed=seed,
        )
        self.observation_space = Box(shape=(resolution, resolution, 3 * framestack), low=0, high=255, dtype=np.uint8)
        self.action_space = Box(-self.action_max, self.action_max, shape=(2,), dtype=np.float32)

    def step(self, action):
        obs, rew, done, misc = super().step([action])
        self.last_obs = obs[0]
        return self.last_obs, rew[0], done[0], misc[0]

    def reset(self, **kwargs):
        obs = super().reset(**kwargs)
        self.last_obs = obs[0]
        return self.last_obs

    def render(self, mode="human"):
        if mode == "human":
            raise NotImplementedError("TODO")
        else:
            return self.last_obs


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    def check_same_env(env1, env2):
        objs = ["golfball", "goal", "global_friction"]
        x1 = env1.env.envs[0].reset(0)
        x2 = env2.env.envs[0].reset(0)
        for obj in objs:
            factors_level1 = [
                x.value
                for x in env1.env.envs[0].mdp_list[0].env.env.sim.things[obj].factors.values()
                if type(x.value) == float or type(x.value) == np.ndarray
            ]
            factors_level2 = [
                x.value
                for x in env2.env.envs[0].mdp_list[0].env.env.sim.things[obj].factors.values()
                if type(x.value) == float or type(x.value) == np.ndarray
            ]
            names = np.array(
                [
                    y
                    for y, x in env1.env.envs[0].mdp_list[0].env.env.sim.things[obj].factors.items()
                    if type(x.value) == float or type(x.value) == np.ndarray
                ]
            )
            is_different = False
            for f1, f2, name in zip(factors_level1, factors_level2, names):
                if not np.all(f1 == f2):
                    print("[%s] Factor `%s` is different" % (obj, name))
                    is_different = True
            if not is_different:
                print("[%s] All factors are identical" % (obj))

    def check_same_process(env1):
        objs = ["golfball", "goal", "global_friction"]
        x1 = env1.env.envs[0].reset(0)
        x2 = env1.env.envs[1].reset(0)
        for obj in objs:
            factors_level1 = [
                x.value
                for x in env1.env.envs[0].mdp_list[0].env.env.sim.things[obj].factors.values()
                if type(x.value) == float or type(x.value) == np.ndarray
            ]
            factors_level2 = [
                x.value
                for x in env1.env.envs[1].mdp_list[0].env.env.sim.things[obj].factors.values()
                if type(x.value) == float or type(x.value) == np.ndarray
            ]
            names = np.array(
                [
                    y
                    for y, x in env1.env.envs[0].mdp_list[0].env.env.sim.things[obj].factors.items()
                    if type(x.value) == float or type(x.value) == np.ndarray
                ]
            )
            is_different = False
            for f1, f2, name in zip(factors_level1, factors_level2, names):
                if not np.all(f1 == f2):
                    print("[%s] Factor `%s` is different" % (obj, name))
                    is_different = True
            if not is_different:
                print("[%s] All factors are identical" % (obj))

    def check_same_level(env1):
        objs = ["golfball", "goal", "global_friction"]
        for obj in objs:
            x1 = env1.env.envs[0].reset(0)
            factors_level1 = [
                x.value
                for x in env1.env.envs[0].mdp_list[0].env.env.sim.things[obj].factors.values()
                if type(x.value) == float or type(x.value) == np.ndarray
            ]
            x2 = env1.env.envs[0].reset(1)
            factors_level2 = [
                x.value
                for x in env1.env.envs[0].mdp_list[1].env.env.sim.things[obj].factors.values()
                if type(x.value) == float or type(x.value) == np.ndarray
            ]
            x1_2 = env1.env.envs[0].reset(0)
            factors_level1_2 = [
                x.value
                for x in env1.env.envs[0].mdp_list[0].env.env.sim.things[obj].factors.values()
                if type(x.value) == float or type(x.value) == np.ndarray
            ]
            names = np.array(
                [
                    y
                    for y, x in env1.env.envs[0].mdp_list[1].env.env.sim.things[obj].factors.items()
                    if type(x.value) == float or type(x.value) == np.ndarray
                ]
            )
            is_different = is_different_lvl1 = False
            for f1, f2, f1_2, name in zip(factors_level1, factors_level2, factors_level1_2, names):
                if not np.all(f1 == f2):
                    print("[%s] Factor `%s` is different" % (obj, name))
                    is_different = True
                if not np.all(f1 == f1_2):
                    print("[%s] Factor `%s` is different for level 0" % (obj, name))
                    is_different_lvl1 = True
            if not is_different:
                print("[%s] All factors are identical" % (obj))
            if not is_different_lvl1:
                print("[%s] All factors are identical for level 1" % (obj))
        # plt.imshow(x1)
        # plt.show()
        # plt.imshow(x2)
        # plt.show()

    env1 = SEGAREnv(
        "emptyx0-hard-rgb", num_envs=2, num_levels=5, framestack=1, resolution=64, max_steps=200, seed=123
    )
    env2 = SEGAREnv(
        "emptyx0-hard-rgb", num_envs=1, num_levels=5, framestack=1, resolution=64, max_steps=200, seed=456
    )
    print('')
    print("===Checking whether 2 env objects are identical")
    check_same_env(env1, env2)
    print('')
    print("===Checking whether 2 levels are identical")
    check_same_level(env1)
    print('')
    print("===Checking whether 2 processes are identical")
    check_same_process(env1)

    exit()
    obs = env1.reset()
    print("Obs shape:", obs.shape)

    ep_lens = []
    e = 0
    for i in range(10_000):
        action = env1.action_space.sample()
        next_obs, reward, done, info = env1.step(action)
        if done[0]:
            ep_lens.append(i - e)
            e = i
            env1.reset()
