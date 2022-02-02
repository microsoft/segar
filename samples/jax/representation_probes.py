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

FLAGS = flags.FLAGS

flags.DEFINE_string("env_name", "emptyx0-easy-rgb", "Env name")
flags.DEFINE_integer("train_steps", 1_000, "Number of train frames.")
flags.DEFINE_string("model_dir", ".", "PPO weights directory")


def loss_fn(params: flax.core.frozen_dict.FrozenDict, apply_fn: Callable,
            X: jnp.ndarray, y: jnp.ndarray):
    y_hat = apply_fn(params, X)
    predictor_loss = jnp.sum((y - y_hat)**2, 1).mean()
    return predictor_loss, (y - y_hat)**2


def main(argv):
    """
    Load the pre-trained PPO model
    """
    seed = np.random.randint(100000000)
    np.random.seed(seed)
    rng = PRNGKey(seed)
    rng, key = jax.random.split(rng)
    dummy_env = SEGAREnv(FLAGS.env_name,
                   num_envs=1,
                   num_levels=1,
                   framestack=1,
                   resolution=64)
    n_action = dummy_env.action_space.shape[-1]
    model_ppo = TwinHeadModel(action_dim=n_action,
                              prefix_critic='vfunction',
                              prefix_actor="policy",
                              action_scale=1.)

    state = dummy_env.reset().astype(jnp.float32) / 255.

    tx = optax.chain(optax.clip_by_global_norm(2), optax.adam(3e-4, eps=1e-5))
    params_model = model_ppo.init(key, state)
    train_state_ppo = TrainState.create(apply_fn=model_ppo.apply,
                                        params=params_model,
                                        tx=tx)

    model_dir = os.path.join(FLAGS.model_dir, 'model_weights')
    loaded_state = checkpoints.restore_checkpoint('./%s' % model_dir,
                                                  target=train_state_ppo)
    """
    Probe 1. Compute dependence between observation features and factors
    """

    env = SEGAREnv(FLAGS.env_name,
                   num_envs=64,
                   num_levels=1,
                   framestack=1,
                   resolution=64)

    predictor = MLP(dims=[256, 20 * 5])
    tx_predictor = optax.chain(optax.clip_by_global_norm(2),
                               optax.adam(3e-4, eps=1e-5))

    z_obs = loaded_state.apply_fn(loaded_state.params,
                                  state,
                                  method=model_ppo.encode)
    params_predictor = predictor.init(key, z_obs)
    train_state_predictor = TrainState.create(apply_fn=predictor.apply,
                                              params=params_predictor,
                                              tx=tx_predictor)

    batch_obs = []
    batch_latents = []
    for i in range(FLAGS.train_steps):
        action, log_pi, value, new_key = select_action(
            loaded_state,
            state.astype(jnp.float32) / 255.,
            latent_factors=None,
            rng=rng,
            sample=True)
        state, reward, done, info = env.step(action)
        latent_features = extract_latent_factors(info)
        batch_obs.append(state)
        batch_latents.append(latent_features)

        if (len(batch_obs) * len(batch_obs[0])) == FLAGS.batch_size:
            # Representation learning part
            rng, key = jax.random.split(rng)
            X = loaded_state.apply_fn(loaded_state.params,
                                      jnp.stack(batch_obs).reshape(
                                          -1, *batch_obs[0].shape[1:]),
                                      method=model_ppo.encode)
            y = jnp.stack(batch_latents).reshape(-1,
                                                 *batch_latents[0].shape[1:])

            grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
            (total_loss,
             square_res), grads = grad_fn(train_state_predictor.params,
                                          train_state_predictor.apply_fn,
                                          X=X,
                                          y=y)
            train_state_predictor = train_state_predictor.apply_gradients(
                grads=grads)

            per_factor_error = square_res.reshape(-1, 20, 5).sum(2).mean(0)
            print('Error')
            print(per_factor_error)
            batch_obs = []
            batch_latents = []

    return 0


if __name__ == '__main__':
    app.run(main)