__copyright__ = "Copyright (c) Microsoft Corporation and Mila - Quebec AI Institute"
__license__ = "MIT"

from pprint import pprint

import gym
import segar

env = gym.make("Segar-tilesx2-hard-rgb-v0", num_levels=1, seed=420)
while True:
    env.reset()
    env.render()
    done = False
    first_step = True
    while not done:
        action = env.action_space.sample()
        obs, _, done, misc = env.step(action)
        if first_step:
            first_step = False
            pprint(misc)
        env.render()

print("this will never print")
