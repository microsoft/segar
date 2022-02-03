import pickle
import time
from typing import Callable
import warnings
from gym import register

_SIM = None


def get_sim():
    """Gets the sim, if set.
    :return: the simulator.
    """
    if _SIM is None:
        raise RuntimeError('Simulator not set yet')

    return _SIM


def set_sim(sim):
    global _SIM
    if _SIM is not None:
        warnings.warn('Overwriting sim. This can have unexpected '
                      'consequences if using old sim objects somewhere.')
    _SIM = sim


def timeit(fn: Callable):
    def timed_fn(*args, **kwargs):
        t0 = time.time()
        out = fn(*args, **kwargs)
        t1 = time.time()

        if _SIM is not None:
            if len(args) > 0:
                key = f'{args[0].__class__.__name__}.{fn.__name__}_time'
            else:
                key = f'{fn.__name__}_time'
            _SIM.update_results(key, t1 - t0)
        return out

    return timed_fn


def load_sim_from_file(path):
    """Loads a simulator from file.
    This uses pickle, so the simulator loaded will have its associated code
    intact. This can make the loaded sim incompatible with newer code.
    :param path: Path to pickle file.
    :return: Simulator.
    """
    with open(path, 'rb') as f:
        sim = pickle.load(f)
    set_sim(sim)
    return sim

task_names = ["empty", "objects", "tiles"]
difficulties = ["easy", "medium", "hard"]
observations = ["rgb"]

for task in task_names:
    for difficulty in difficulties:
        for observation in observations:
            for n_entities in [1, 2, 3]:
                if task == 'empty':
                    if n_entities == 1:
                        env_name = f"segar-{task}-{difficulty}-{observation}-v0"
                        register(
                            id=env_name,  # FIXME
                            entry_point="segar.envs:SEGARSingleEnv",
                            kwargs={"env_name": f"{task}-{difficulty}-{observation}"},  # FIXME
                            max_episode_steps=100
                        )
                    else:
                        continue
                else:
                    env_name = f"segar-{task}x{n_entities}-{difficulty}-{observation}-v0"
                    register(
                        id=env_name,  # FIXME
                        entry_point="segar.envs:SEGARSingleEnv",
                        kwargs={"env_name": f"{task}x{n_entities}-{difficulty}-{observation}"},  # FIXME
                        max_episode_steps=100
                    )

### Currently available:
# segar-empty-easy-rgb-v0
# segar-empty-medium-rgb-v0
# segar-empty-hard-rgb-v0
# segar-objects-easy-rgb-v0
# segar-objects-medium-rgb-v0
# segar-objects-hard-rgb-v0
# segar-tiles-easy-rgb-v0
# segar-tiles-medium-rgb-v0
# segar-tiles-hard-rgb-v0