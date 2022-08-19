__copyright__ = "Copyright (c) Microsoft Corporation and Mila - Quebec AI Institute"
__license__ = "MIT"

import logging
import gym
import numpy as np

from segar.envs.wrappers import SequentialTaskWrapper
from segar.tasks.env_generators import get_env_generator


logger = logging.getLogger(__name__)


class SEGAREnv(gym.Env):
    def __init__(
        self,
        env_name: str,
        num_levels: int = 100,
        num_envs: int = 2,
        framestack: int = 1,
        _async: bool = False,
        seed: int = 123,
        env_args: dict = None,
    ):

        mdp_generator = get_env_generator(env_name)
        if mdp_generator is None:
            raise ValueError(f'Could not parse environment string {env_name}.')

        def make_env():
            return SequentialTaskWrapper(
                mdp_generator,
                num_levels,
                framestack,
                seed,
                env_args
            )

        self.dummy_env = make_env()

        if not _async:
            logger.info("Making sync envs.")
            self.env = gym.vector.SyncVectorEnv(
                [make_env for _ in range(num_envs)],
                observation_space=self.dummy_env.observation_space,
                action_space=self.dummy_env.action_space,
            )
        else:
            logger.info("Making async envs.")
            self.env = gym.vector.AsyncVectorEnv(
                [make_env for _ in range(num_envs)],
                observation_space=self.dummy_env.observation_space,
                action_space=self.dummy_env.action_space,
                shared_memory=True,
                daemon=True,
            )

    def step(self, action):
        self.env.step_async(np.array(action)[None, :])
        return self.env.step_wait()

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def render(self, mode="human"):
        return self.env.render(mode)

    @property
    def action_space(self):
        return self.env.action_space

    @property
    def observation_space(self):
        return self.env.observation_space
