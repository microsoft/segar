from typing import Optional, Sequence

import flax.linen as nn
import jax.numpy as jnp

from tensorflow_probability.substrates import jax as tfp

tfd = tfp.distributions
tfb = tfp.bijectors

LOG_STD_MIN = -3.0
LOG_STD_MAX = 2.0

"""
These regulate default inits for conv, pre-linear and pre-ReLU layers
"""

def default_conv_init(scale: Optional[float] = jnp.sqrt(2)):
    return nn.initializers.xavier_uniform()


def default_mlp_init(scale: Optional[float] = 0.01):
    return nn.initializers.xavier_uniform()


def default_logits_init(scale: Optional[float] = 0.01):
    return nn.initializers.xavier_uniform()


class MLP(nn.Module):
    dims: Sequence[int]
    batch_norm: bool = False

    @nn.compact
    def __call__(self, x):
        if self.batch_norm:
            x = nn.LayerNorm()(x)
        for i, dim in enumerate(self.dims):
            x = nn.Dense(dim, kernel_init=default_mlp_init(),
                         name='/%d' % i)(x)
            if self.batch_norm:
                x = nn.LayerNorm()(x)
            if i < len(self.dims) - 1:
                x = nn.relu(x)
        return x


class ResidualBlock(nn.Module):
    """Residual block."""
    num_channels: int
    prefix: str

    @nn.compact
    def __call__(self, x):
        # Conv branch
        y = nn.relu(x)
        y = nn.Conv(self.num_channels,
                    kernel_size=[3, 3],
                    strides=(1, 1),
                    padding='SAME',
                    kernel_init=default_conv_init(),
                    name=self.prefix + '/conv2d_1')(y)
        y = nn.relu(y)
        y = nn.Conv(self.num_channels,
                    kernel_size=[3, 3],
                    strides=(1, 1),
                    padding='SAME',
                    kernel_init=default_conv_init(),
                    name=self.prefix + '/conv2d_2')(y)

        return y + x


class Impala(nn.Module):
    """IMPALA architecture."""
    prefix: str

    @nn.compact
    def __call__(self, x):
        out = x
        for i, (num_channels, num_blocks) in enumerate([(15, 2), (32, 2),
                                                        (32, 2)]):
            conv = nn.Conv(num_channels,
                           kernel_size=[3, 3],
                           strides=(1, 1),
                           padding='SAME',
                           kernel_init=default_conv_init(),
                           name=self.prefix + '/conv2d_%d' % i)
            out = conv(out)

            out = nn.max_pool(out,
                              window_shape=(3, 3),
                              strides=(2, 2),
                              padding='SAME')
            for j in range(num_blocks):
                block = ResidualBlock(num_channels,
                                      prefix='residual_{}_{}'.format(i, j))
                out = block(out)

        out = out.reshape(out.shape[0], -1)
        out = nn.relu(out)
        out = nn.Dense(256,
                       kernel_init=default_mlp_init(),
                       name=self.prefix + '/representation')(out)
        out = nn.relu(out)
        return out


class TwinHeadModel(nn.Module):
    """Critic+Actor for PPO."""
    action_dim: int
    action_scale: float = 1
    add_latent_factors: bool = False
    prefix_critic: str = "critic"
    prefix_actor: str = "policy"
    activation: str = 'tanh'

    def setup(self):
        if self.activation == 'relu':
            self.activation_fn = nn.relu
        elif self.activation == 'tanh':
            self.activation_fn = nn.tanh

        self.encoder = Impala(prefix='shared_encoder')
        self.factor_ln = nn.LayerNorm()
        self.factor_encoder = nn.Dense(256,
                                       kernel_init=default_mlp_init(),
                                       name=self.prefix_critic + '/factor')
        self.obs_joint = nn.Dense(256,
                                  kernel_init=default_mlp_init(),
                                  name=self.prefix_critic + '/obs_joint')
        self.v_1 = nn.Dense(256,
                            kernel_init=default_mlp_init(),
                            name=self.prefix_critic + '/v_1')
        self.v_2 = nn.Dense(1,
                            kernel_init=default_mlp_init(),
                            name=self.prefix_critic + '/v_2')
        self.z_pi = nn.Dense(256, kernel_init=default_mlp_init(), name="z_pi")
        self.means_1 = nn.Dense(256,
                                kernel_init=default_mlp_init(),
                                name="mu_1")
        self.means_2 = nn.Dense(self.action_dim,
                                kernel_init=default_mlp_init(),
                                name="mu_2")
        self.log_stds_1 = nn.Dense(256,
                                   kernel_init=default_mlp_init(),
                                   name="log_std_1")
        self.log_stds_2 = nn.Dense(self.action_dim,
                                   kernel_init=default_mlp_init(),
                                   name="log_std_2")

    @nn.compact
    def __call__(self, x, latent_factors):
        """
        Classical PPO with IMPALA encoder and optionally with augmented latent
        representation using true factor embeddings.

        Policy is a Tanh(DiagGaussian(mu(o), log_sigma(o)))
        """
        z = self.encoder(x)
        if self.add_latent_factors:
            z_factors = self.factor_ln(latent_factors)
            z_factors = self.activation_fn(self.factor_encoder(z_factors))
            z = jnp.concatenate([z, z_factors], axis=-1)
            z = self.activation_fn(self.obs_joint(z))
        # Linear critic
        v = self.v_1(z)
        v = self.activation_fn(v)
        v = self.v_2(v)
        # Common policy trunk

        # mu(z)
        means = self.means_1(z)
        means = self.activation_fn(means)
        means = self.means_2(means)

        # log_std(z)
        log_stds = self.log_stds_1(z)
        log_stds = self.activation_fn(log_stds)
        log_stds = self.log_stds_2(log_stds)

        log_stds = jnp.clip(log_stds, LOG_STD_MIN, LOG_STD_MAX)

        base_dist = tfd.MultivariateNormalDiag(loc=means,
                                               scale_diag=jnp.exp(log_stds))

        pi = tfd.TransformedDistribution(
            distribution=base_dist,
            bijector=tfb.Chain(
                [tfb.Scale(scale=self.action_scale),
                 tfb.Tanh()]))

        return v, pi

    def encode(self, x):
        return self.encoder(x)
