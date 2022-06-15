from typing import Dict

import flax.linen as nn
import gym
import numpy as np


"""Adapted from JAXRL: https://github.com/ikostrikov/jaxrl"""


def evaluate(agent, env: gym.Env, num_episodes: int) -> Dict[str, float]:
    stats = {'return': [], 'length': []}
    successes = None
    for _ in range(num_episodes):
        observation, done = env.reset(), False
        while not done:
            action = agent.sample_actions(observation, temperature=0.0)
            observation, _, done, info = env.step(action)
        
        for i in range(len(info)):
            for k in stats.keys():
                maybe_episode = info[i].get("episode")
                if maybe_episode:
                    stats[k].append(maybe_episode[k])

            if 'is_success' in info[i]:
                if successes is None:
                    successes = 0.0
                successes += info[i]['is_success']

    for k, v in stats.items():
        stats[k] = np.mean(v)

    if successes is not None:
        stats['success'] = successes / num_episodes

    return stats
