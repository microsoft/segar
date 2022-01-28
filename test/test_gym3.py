
from rpp.mdps.mdps import MDP
from rpp.mdps.observations import RGBObservation
from rpp.tasks.puttputt import PuttPuttRandomCenterInitialization, PuttPutt
from rpp.mdps.states import StateDict
from rpp.sim.sim import Simulator

import gym3
import numpy as np
import tqdm


_N_STEPS = 100


sim = Simulator()
initialization = PuttPuttRandomCenterInitialization()
observations = RGBObservation(resolution=256)
states = StateDict()
task = PuttPutt(initialization)

config = dict(
    max_steps_per_episode=10,
    episodes_per_arena=1,
    sub_steps=100
)

if __name__ == "__main__":
    # Make multiple parallel envs
    n_envs = 4
    env = gym3.vectorize_gym(
        num=n_envs, env_fn=lambda: MDP(states, observations, task,
                                       **config), use_subproc=True)

    def step(action):
        env.act(action)
        rew, obs, done = env.observe()
        info = env.get_info()
        return obs, rew, done, info

    env.step = step
    n_action = env.ac_space.shape[0]

    returns = np.zeros(shape=(n_envs,))
    for _ in tqdm.tqdm(range(_N_STEPS)):
        action = np.random.uniform(size=(n_envs, n_action))
        obs, rew, done, info = env.step(action)
        returns += rew

    print(returns)
