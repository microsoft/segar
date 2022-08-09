from typing import Dict

import flax.linen as nn
import gym
import numpy as np


"""Adapted from JAXRL: https://github.com/ikostrikov/jaxrl"""


def evaluate(agent, env: gym.Env, num_episodes: int) -> Dict[str, float]:
    stats = {'returns': [], 'success': []}
    for _ in range(num_episodes):
        observation, done = env.reset(), False
        while not done:
            action = agent.sample_actions(observation, temperature=0.0)
            observation, _, done, info = env.step(action)
        for k in stats.keys():
            if k in info:
                stats[k].append(info[k])

    for k, v in stats.items():
        stats[k] = np.mean(v)

    return stats
