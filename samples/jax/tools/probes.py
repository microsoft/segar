from absl import app, flags
from typing import Callable
import flax
from models import MLP, TwinHeadModel
from algo import select_action, extract_latent_factors
import numpy as np
from segar.envs.env import SEGAREnv
import jax.numpy as jnp
from jax.random import PRNGKey
import os
import jax
import optax
from flax.training.train_state import TrainState
from flax.training import checkpoints

def mse_loss(params: flax.core.frozen_dict.FrozenDict, apply_fn: Callable,
            X: jnp.ndarray, y: jnp.ndarray):
    y_hat = apply_fn(params, X)
    predictor_loss = jnp.sum((y - y_hat)**2, 1).mean()
    return predictor_loss, (y - y_hat)**2