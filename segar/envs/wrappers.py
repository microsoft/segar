__copyright__ = "Copyright (c) Microsoft Corporation and Mila - Quebec AI Institute"
__license__ = "MIT"

from collections import deque
import logging
import random

from gym.spaces import Box
import numpy as np

from segar.mdps import MDP, Observation


logger = logging.getLogger(__name__)


class SequentialTaskWrapper:
    def __init__(
        self,
        mdp_generator,
        num_levels,
        framestack=1,
        seed=None,
        mdp_config=None
    ):
        if seed is not None:
            logger.info(f"Setting env seed to {seed}")
            np.random.seed(seed)
            random.seed(seed)

        if mdp_config is None:
            mdp_config = {}

        self.env_list: list[FrameStackWrapper] = []
        for i in range(num_levels):
            mdp = mdp_generator(i, **mdp_config.copy())
            env = FrameStackWrapper(ReturnMonitor(mdp), framestack)
            self.env_list.append(env)

        self.n_envs = len(self.env_list)
        self.current_env = self._pick_env()

    @property
    def current_step(self) -> int:
        return self.env.num_steps

    @property
    def _max_steps(self) -> int:
        return self.env.max_steps_per_episode

    @property
    def action_space(self):
        return self.env.action_space

    @property
    def sim(self):
        return self.env.sim

    @property
    def observation_space(self):
        return self.env.observation_space

    @property
    def metadata(self):
        return None

    @property
    def env(self):
        return self.current_env

    def reset(self, task_id=None):
        self.current_env = self._pick_env(task_id)
        obs = self.env.reset()
        return obs

    def step(self, action):
        try:
            next_obs, rew, done, info = self.env.step(action)
        except Exception as e:
            # repeat again in hopes the crash doesn't get registered
            next_obs, rew, done, info = self.env.step(action)
            logger.error("Ignoring simulator exception:")
            logger.error(e)
        success = self.env.success
        info["success"] = success
        info["task_id"] = self.task_id
        if self.latent_obs is not None:
            info["latent_features"] = self.latent_obs(self.sim.state)
        return next_obs.copy(), rew, done, info

    def _pick_env(self, task_id=None):
        if task_id is None:
            self.task_id = np.random.randint(low=0, high=self.n_envs, size=(1,)).item()
        else:
            self.task_id = task_id
        return self.env_list[self.task_id]

    def seed(self, seed):
        for mdp in self.env_list:
            mdp.seed(seed)

    def close(self):
        for mdp in self.env_list:
            del mdp

    @property
    def latent_obs(self) -> Observation:
        return self.env.latent_obs


class ReturnMonitor:
    def __init__(self, env):
        self.env: MDP = env
        self.returns = 0

        self.observation_space = env.observation_space
        self.action_space = env.action_space

    def reset(self):
        self.returns = 0
        obs = self.env.reset()
        return obs

    def step(self, action):
        next_obs, rew, done, info = self.env.step(action)
        if done:
            info["returns"] = self.returns
            self.returns = 0
        self.returns += rew
        return next_obs, rew, done, info

    def seed(self, seed):
        self.env.seed(seed)

    def close(self):
        self.env.close()

    @property
    def success(self) -> int:
        return self.env.success

    @property
    def metadata(self):
        return None

    @property
    def num_steps(self) -> int:
        return self.env.num_steps

    @property
    def max_steps_per_episode(self) -> int:
        return self.env.max_steps_per_episode

    @property
    def sim(self):
        return self.env.sim

    @property
    def latent_obs(self) -> Observation:
        return self.env.latent_obs


class FrameStackWrapper:
    def __init__(self, env, n_frames):
        self.env: ReturnMonitor = env
        self.n_frames = n_frames
        self.frames = deque([], maxlen=n_frames)

        obs_space = env.observation_space
        obs_shape = obs_space.shape
        if isinstance(obs_space, Box):
            n = len(obs_shape)
            new_shape = np.concatenate([obs_shape[:(n - 1)], [obs_shape[(n - 1)] * n_frames]], axis=0).astype('int32')
            self.observation_space = Box(
                low=obs_space.low,
                high=obs_space.high,
                shape=new_shape,
                dtype=obs_space.dtype,
            )
        else:
            raise NotImplementedError

        self.action_space = env.action_space

    def reset(self):
        obs = self.env.reset()
        for _ in range(self.n_frames):
            self.frames.append(obs)
        return self._transform_observation(self.frames)

    def step(self, action):
        next_obs, rew, done, info = self.env.step(action)
        self.frames.append(next_obs)
        return self._transform_observation(self.frames), rew, done, info

    def _transform_observation(self, obs):
        obs = np.concatenate(list(obs), axis=-1)
        return obs

    def seed(self, seed):
        self.env.seed(seed)

    def close(self):
        self.env.close()

    @property
    def success(self) -> int:
        return self.env.success

    @property
    def metadata(self):
        return None

    @property
    def num_steps(self) -> int:
        return self.env.num_steps

    @property
    def max_steps_per_episode(self) -> int:
        return self.env.max_steps_per_episode

    @property
    def sim(self):
        return self.env.sim

    @property
    def latent_obs(self) -> Observation:
        return self.env.latent_obs
