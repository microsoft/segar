
import pickle
import time
from typing import Callable
import warnings

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
