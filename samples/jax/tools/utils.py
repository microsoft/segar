import os
import sys
from typing import Callable
import glob

import flax
import jax
import jax.numpy as jnp
import numpy as np
import optax
from absl import app, flags
from flax.training import checkpoints
from flax.training.train_state import TrainState
from jax.random import PRNGKey
from samples.jax.algo import select_action

def rollouts(env, train_state, key, n_rollouts=10):
    state = env.reset()
    returns = []
    states = []
    actions = []
    factors = []
    task_ids = []
    while n_rollouts:
        states.append(state)
        action, _, _, key = select_action(train_state,
                                               state.astype(jnp.float32) /
                                               255.,
                                               None,
                                               key,
                                               sample=True)
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
    return returns, (states, actions, factors)
