# !pip install causal_world
import timeit
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import psutil
import tracemalloc
import linecache

def display_top(snapshot, key_type='lineno', limit=3):
    snapshot = snapshot.filter_traces((
        tracemalloc.Filter(False, "<frozen importlib._bootstrap>"),
        tracemalloc.Filter(False, "<unknown>"),
    ))
    top_stats = snapshot.statistics(key_type)

    print("Top %s lines" % limit)
    for index, stat in enumerate(top_stats[:limit], 1):
        frame = stat.traceback[0]
        # replace "/path/to/module/file.py" with "module/file.py"
        filename = os.sep.join(frame.filename.split(os.sep)[-2:])
        print("#%s: %s:%s: %.1f KiB"
              % (index, filename, frame.lineno, stat.size / 1024))
        line = linecache.getline(frame.filename, frame.lineno).strip()
        if line:
            print('    %s' % line)

    other = top_stats[limit:]
    if other:
        size = sum(stat.size for stat in other)
        print("%s other: %.1f KiB" % (len(other), size / 1024))
    total = sum(stat.size for stat in top_stats)
    print("Total allocated size: %.1f KiB" % (total / 1024))


def get_process_memory():
    process = psutil.Process(os.getpid())
    return process.memory_percent()

def track(func):
    def wrapper(*args, **kwargs):
        tracemalloc.start()
        # mem_before = get_process_memory()
        result = func(*args, **kwargs)
        # mem_after = get_process_memory()
        # print("Memory before: %d Mb, after: %d Mb, consumed: %d Mb" % (
        #     mem_before, mem_after, mem_after - mem_before))
        # return result
        snapshot = tracemalloc.take_snapshot()
        display_top(snapshot)
    return wrapper

n_samples = 1

"""
1. CausalWorld
"""

from causal_world.envs import CausalWorld
from causal_world.task_generators import generate_task

@track
def env_creation():
    task = generate_task(task_generator_id='general')
    env = CausalWorld(task=task, enable_visualization=True)
    env.close()

@track
def env_step():
    env.reset()
    for _ in range(100):
        obs, reward, done, info = env.step(env.action_space.sample())
    # env.close()

CW_1_times = [timeit.timeit(env_creation, number=1) for _ in range(n_samples)]
task = generate_task(task_generator_id='general')
env = CausalWorld(task=task, enable_visualization=True)
CW_2_times = [timeit.timeit(env_step, number=1) for _ in range(n_samples)]
env.close()

CW_1_times = pd.DataFrame({'CausalWorld creation time':np.array(CW_1_times)})
CW_2_times = pd.DataFrame({'CausalWorld execution time':np.array(CW_2_times)})

print('CausalWorld:')
print('Creation time: %.3fs ± %.3fs'% (CW_1_times['CausalWorld creation time'].mean(), CW_1_times['CausalWorld creation time'].std()))
print('Execution time: %.3fs ± %.3fs'% (CW_2_times['CausalWorld execution time'].mean(), CW_2_times['CausalWorld execution time'].std()))

"""
2. SEGAR
"""
from segar.envs.env import SEGAREnv

@track
def env_creation():
    env = SEGAREnv("empty-easy-rgb",
                         num_envs=1,
                         num_levels=1,
                         framestack=1,
                         resolution=64,
                         max_steps=100,
                         _async=False,
                         deterministic_visuals=False,
                         seed=123)
    env.close()

@track
def env_step():
    env.reset()
    for _ in range(100):
        obs, reward, done, info = env.step(env.action_space.sample())

SEGAR_1_times = [timeit.timeit(env_creation, number=1) for _ in range(n_samples)]
env = SEGAREnv("empty-easy-rgb",
                         num_envs=1,
                         num_levels=1,
                         framestack=1,
                         resolution=64,
                         max_steps=100,
                         _async=False,
                         deterministic_visuals=False,
                         seed=123)
SEGAR_2_times = [timeit.timeit(env_step, number=1) for _ in range(n_samples)]
env.close()
SEGAR_1_times = pd.DataFrame({'SEGAR creation time':np.array(SEGAR_1_times)})
SEGAR_2_times = pd.DataFrame({'SEGAR execution time':np.array(SEGAR_2_times)})

print('SEGAR:')
print('Creation time: %.3fs ± %.3fs'% (SEGAR_1_times['SEGAR creation time'].mean(), SEGAR_1_times['SEGAR creation time'].std()))
print('Execution time: %.3fs ± %.3fs'% (SEGAR_2_times['SEGAR execution time'].mean(), SEGAR_2_times['SEGAR execution time'].std()))

import ipdb;ipdb.set_trace()