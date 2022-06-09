from typing import Tuple, Dict

import jax.numpy as jnp
from flax import linen as nn

from jaxrl.networks.common import Model

"""Adapted from JAXRL: https://github.com/ikostrikov/jaxrl"""

class Temperature(nn.Module):
    initial_temperature: float = 1.0

    @nn.compact
    def __call__(self) -> jnp.ndarray:
        log_temp = self.param('log_temp',
                              init_fn=lambda key: jnp.full(
                                  (), jnp.log(self.initial_temperature)))
        return jnp.exp(log_temp)


def update(temp: Model, entropy: float,
           target_entropy: float) -> Tuple[Model, Dict[str, float]]:

    def temperature_loss_fn(temp_params):
        temperature = temp.apply_fn({'params': temp_params})
        temp_loss = temperature * (entropy - target_entropy).mean()
        return temp_loss, {'temperature': temperature, 'temp_loss': temp_loss}

    new_temp, info = temp.apply_gradient(temperature_loss_fn)

    return new_temp, info
