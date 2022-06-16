from typing import Dict

import flax.linen as nn
import gym
import numpy as np


"""Adapted from JAXRL: https://github.com/ikostrikov/jaxrl"""


def evaluate(agent, env: gym.Env, num_episodes: int) -> Dict[str, float]:
    stats = {'returns': []}
    successes = None
    for _ in range(num_episodes):
        observation, done = env.reset(), False
        while not done:
            action = agent.sample_actions(observation, temperature=0.0)
            observation, _, done, info = env.step(action)
        
        for i in range(len(info)):
            maybe_success = info[i].get("success")
            if maybe_success:
                if successes is None:
                    successes = 0.0
                successes += maybe_success

            maybe_returns = info[i].get("returns")
            if maybe_returns:
                stats['returns'].append(maybe_returns)

    for k, v in stats.items():
        stats[k] = np.mean(v)

    if successes is not None:
        stats['success'] = successes / num_episodes

    return stats
