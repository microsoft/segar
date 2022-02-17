import time

import gym
import numpy as np
from tqdm import tqdm

import segar

SAMPLES = 1000
# env = gym.make("Segar-empty-easy-rgb-v0")
env = gym.make("Segar-objectsx2-medium-rgb-v0")

timers_step = []
timers_res = []

counter = 0

with tqdm(total=SAMPLES) as pbar:
    while True:
        start = time.perf_counter()
        _ = env.reset()
        timers_res.append(time.perf_counter() - start)
        while True:
            action = env.action_space.sample()
            start = time.perf_counter()
            _, _, done, _ = env.step(action)
            timers_step.append(time.perf_counter() - start)
            counter += 1
            pbar.update(1)
            if done or counter >= SAMPLES:
                break
        if counter >= SAMPLES:
            break

mean_step = np.mean(timers_step)
mean_res = np.mean(timers_res)

print(
    f"Avg step time: {np.around(mean_step,3)}s, "
    f"aka {np.around(1/mean_step,0)}Hz with {len(timers_step)} samples"
)
print(
    f"Avg reset time: {np.around(mean_res,3)}s, "
    f"aka {np.around(1/mean_res,0)}Hz with {len(timers_res)} samples"
)

# Segar-empty-easy-rgb-v0
# Avg step time: 0.011s, aka 91.0Hz with 5000 samples
# Avg reset time: 0.003s, aka 303.0Hz with 101 samples

# Segar-objectsx2-medium-rgb-v0
# Avg step time: 0.024s, aka 41.0Hz with 5000 samples
# Avg reset time: 0.006s, aka 154.0Hz with 101 samples
