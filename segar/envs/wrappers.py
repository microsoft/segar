__author__ = "Bogdan Mazoure, Florian Golemo"
__copyright__ = "Copyright (c) Microsoft Corporation and Mila: The Quebec " \
                "AI Company"
__license__ = "MIT"

from collections import deque

import random
import numpy as np
from gym.spaces import Box

from segar.factors import Charge, Magnetism, Position, Friction
from segar.mdps import MDP
from segar.mdps.observations import AllStateObservation
from segar.tasks import PuttPuttInitialization, PuttPutt
from segar import get_sim

from segar.sim import Simulator


class SequentialTaskWrapper:
    def __init__(
        self,
        obs,
        init_config,
        config,
        action_range,
        num_levels,
        max_steps,
        framestack=1,
        seed=None,
        wall_damping=0.025,
        friction=0.05,
        save_path="sim.state",
    ):
        if seed is not None:
            print('Setting env seed to %d' % seed)
            np.random.seed(seed)
            random.seed(seed)
        self._max_steps = max_steps

        self.task_list = []
        self.mdp_list = []
        for _ in range(num_levels):
            initialization = PuttPuttInitialization(config=init_config)
            task = PuttPutt(action_range=action_range,
                            initialization=initialization)
            sim = Simulator(state_buffer_length=50,
                            wall_damping=wall_damping,
                            friction=friction,
                            safe_mode=False,
                            save_path=save_path)
            task.set_sim(sim)
            task.sample()
            mdp = FrameStackWrapper(
                ReturnMonitor(MDP(obs, task, **config, sim=sim)), framestack)
            self.task_list.append(task)
            self.mdp_list.append(mdp)

        self.n_envs = len(self.task_list)
        self.current_step = 0
        self.current_env = self._pick_env()
        self.sobs = AllStateObservation(
            n_things=20,
            unique_ids=["golfball", "goal"],
            factors=[Charge, Magnetism, Position, Friction])

    @property
    def action_space(self):
        return self.current_env.action_space

    @property
    def sim(self):
        return get_sim()

    @property
    def observation_space(self):
        return self.current_env.observation_space

    @property
    def metadata(self):
        return None

    def reset(self, task_id=None):
        self.current_env = self._pick_env(task_id)
        obs = self.current_env.reset()
        self.current_step = 0
        return obs

    def step(self, action):
        try:
            next_obs, rew, done, info = self.current_env.step(action)
        except Exception as e:
            # repeat again in hopes the crash doesn't get registered
            next_obs, rew, done, info = self.current_env.step(action)
            print('Ignoring simulator exception:')
            print(e)
        self.current_step += 1
        success = int(done and (self.current_step < self._max_steps)
                      and self.sim.things["golfball"].Alive.value)
        done = done or (self.current_step > self._max_steps)
        if done:
            next_obs = self.reset()
            info["success"] = success
        info["task_id"] = self.task_id
        info["latent_features"] = self.sobs(self.current_env.env.env.sim.state)
        return next_obs.copy(), rew, done, info

    def _pick_env(self, task_id=None):
        if task_id is None:
            self.task_id = np.random.randint(low=0,
                                             high=self.n_envs,
                                             size=(1, )).item()
        else:
            self.task_id = task_id
        return self.mdp_list[self.task_id]

    def seed(self, seed):
        for mdp in self.mdp_list:
            mdp.seed(seed)

    def close(self):
        for mdp in self.mdp_list:
            del mdp


class ReturnMonitor:
    def __init__(self, env):
        self.env = env
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
    def metadata(self):
        return None


class FrameStackWrapper:
    def __init__(self, env, n_frames):
        self.env = env
        self.n_frames = n_frames
        self.frames = deque([], maxlen=n_frames)

        wrapped_obs_shape = env.observation_space.shape

        self.observation_space = Box(
            low=0,
            high=255,
            shape=np.concatenate(
                [wrapped_obs_shape[:2], [wrapped_obs_shape[2] * n_frames]],
                axis=0),
            dtype=np.uint8,
        )

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
    def metadata(self):
        return None
