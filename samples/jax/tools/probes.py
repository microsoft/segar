from typing import Callable
import flax
import jax.numpy as jnp


def mse_loss(params: flax.core.frozen_dict.FrozenDict, apply_fn: Callable,
             X: jnp.ndarray, y: jnp.ndarray):
    y_hat = apply_fn(params, X)
    predictor_loss = jnp.sum((y - y_hat)**2, 1).mean()
    return predictor_loss, (y - y_hat)**2
