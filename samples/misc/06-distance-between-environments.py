import contextlib
from datetime import datetime

import gym
from tqdm import tqdm
import os, sys

import segar
from segar.mdps.metrics import task_set_init_dist

RUN = int(sys.argv[1])
print(f"=== Running script on slice {RUN}")


all_envs = [  # 0-20 (21 items total)
    "Segar-empty-easy-rgb-v0",
    "Segar-empty-medium-rgb-v0",
    "Segar-empty-hard-rgb-v0",
    "Segar-objectsx1-easy-rgb-v0",
    "Segar-objectsx2-easy-rgb-v0",
    "Segar-objectsx3-easy-rgb-v0",
    "Segar-objectsx1-medium-rgb-v0",
    "Segar-objectsx2-medium-rgb-v0",
    "Segar-objectsx3-medium-rgb-v0",
    "Segar-objectsx1-hard-rgb-v0",
    "Segar-objectsx2-hard-rgb-v0",
    "Segar-objectsx3-hard-rgb-v0",
    "Segar-tilesx1-easy-rgb-v0",
    "Segar-tilesx2-easy-rgb-v0",
    "Segar-tilesx3-easy-rgb-v0",
    "Segar-tilesx1-medium-rgb-v0",
    "Segar-tilesx2-medium-rgb-v0",
    "Segar-tilesx3-medium-rgb-v0",
    "Segar-tilesx1-hard-rgb-v0",
    "Segar-tilesx2-hard-rgb-v0",
    "Segar-tilesx3-hard-rgb-v0",
]


def get_task_list(env):
    return env.unwrapped.env.envs[0].task_list


def dump_sim(env):
    dt = datetime.today().strftime("%Y%m%d-%H%M%S")
    env.unwrapped.env.envs[0].sim.save(f"sim-coredump-{dt}.pkl")


csv_lines = [",".join([" "] + all_envs)]

for envA_name in tqdm(all_envs[RUN : RUN + 1]):
    output = [envA_name]
    for envB_name in tqdm(all_envs):

        try:
            with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
                envA = gym.make(envA_name)
        except ValueError:
            print("it's in envA creation", envA_name)
            dump_sim(envA)
            envA = gym.make(envA_name)

        try:
            with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
                envB = gym.make(envB_name)
        except ValueError:
            print("it's in envB creation", envB_name)
            dump_sim(envB)
            envB = gym.make(envB_name)

        w2 = task_set_init_dist(get_task_list(envA), get_task_list(envB))
        output.append(w2)
    csv_lines.append(",".join([str(x) for x in output]))

with open(f"env-w2-distances-r{RUN}.csv", "w") as outfile:
    outfile.write("\n".join(csv_lines))
