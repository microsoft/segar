__author__ = "R Devon Hjelm, Bogdan Mazoure, Florian Golemo"
__copyright__ = "Copyright (c) Microsoft Corporation and Mila - Quebec AI " \
                "Institute"
__license__ = "MIT"

from gym import Env
import jax.numpy as jnp
from flax.training.train_state import TrainState
from jax.random import PRNGKey
from samples.jax.algo import select_action
import numpy as np


def rollouts(env: Env,
             train_state: TrainState,
             key: PRNGKey,
             n_rollouts: int = 10,
             sample: bool = True):
    state = env.reset()
    returns = []
    states = []
    zs = []
    actions = []
    factors = []
    task_ids = []
    while n_rollouts:
        states.append(state)
        action, _, _, z, key = select_action(train_state,
                                             state.astype(jnp.float32) / 255.,
                                             None,
                                             key,
                                             sample=sample)
        zs.append(z)
        actions.append(action)
        state, _, _, infos = env.step(action)
        for info in infos:
            maybe_epinfo = info.get('returns')
            latent_features = info.get('latent_features')
            task_id = info.get('task_id')
            factors.append(latent_features)
            task_ids.append(task_id)
            if maybe_epinfo:
                returns.append(maybe_epinfo)
                n_rollouts -= 1
    return returns, (states, zs, actions, factors, task_ids)
