from typing import Tuple, Dict, Any

import flax
import jax
import jax.numpy as jnp

from jaxrl.datasets import Batch
from jax.random import PRNGKey
from jaxrl.networks.common import Model


"""Adapted from JAXRL: https://github.com/ikostrikov/jaxrl"""


def update(key: PRNGKey, actor: Model, critic: Model, temp: Model,
           batch: Batch) -> Tuple[Model, Dict[str, float]]:

    def actor_loss_fn(actor_params: flax.core.FrozenDict[str, Any]) -> Tuple[jnp.ndarray, Dict[str, float]]:
        dist = actor.apply_fn({'params': actor_params}, batch.observations)
        actions = dist.sample(seed=key)
        log_probs = dist.log_prob(actions)
        q1, q2 = critic(batch.observations, actions)
        q = jnp.minimum(q1, q2)
        actor_loss = (log_probs * temp() - q).mean()
        return actor_loss, {
            'actor_loss': actor_loss,
            'entropy': -log_probs.mean()
        }

    new_actor, info = actor.apply_gradient(actor_loss_fn)

    return new_actor, info
