from collections import defaultdict
from functools import partial
from typing import Any, Callable, Tuple

import flax
import jax
import jax.numpy as jnp
import numpy as np
from flax.training.train_state import TrainState
from jax.random import PRNGKey
"""
Inspired by code from Flax:
https://github.com/google/flax/blob/main/examples/ppo/ppo_lib.py
"""


def mse_loss(params: flax.core.frozen_dict.FrozenDict, apply_fn: Callable,
             X: jnp.ndarray, y: jnp.ndarray):
    X = jax.lax.stop_gradient(
        X)  # prevent latent factors to backprop into representation
    y_hat = apply_fn(params, X)
    # mimic batchnorm1d
    y = y / jnp.stack([jnp.linalg.norm(y, axis=1)] * y.shape[1]).transpose()
    predictor_loss = jnp.sum((y - y_hat)**2, 1).mean()
    return predictor_loss, (y - y_hat)**2


def compute_grad_norm(grads):
    if hasattr(grads, "items"):
        acc = 0.
        n = 0
        for k, v in grads.items():
            acc += compute_grad_norm(v)
            n += 1
        acc /= n
    else:
        acc = jnp.linalg.norm(grads)
    return acc


def extract_latent_factors(infos):
    latent_features = []
    for i, info in enumerate(infos):
        latent_features.append(info['latent_features'].reshape(-1))
    latent_features = jnp.stack(latent_features)
    return latent_features


def compute_grad_norm(tree):
    if hasattr(tree, "items"):
        acc = 0.
        for k, v in tree.items():
            acc += compute_grad_norm(v)
        return acc
    return jnp.linalg.norm(tree)


def loss_actor_and_critic(params_model: flax.core.frozen_dict.FrozenDict,
                          apply_fn: Callable[..., Any], state: jnp.ndarray,
                          target: jnp.ndarray, value_old: jnp.ndarray,
                          log_pi_old: jnp.ndarray, gae: jnp.ndarray,
                          action: jnp.ndarray, latent_factors: jnp.ndarray,
                          clip_eps: float, critic_coeff: float,
                          entropy_coeff: float, key: PRNGKey) -> jnp.ndarray:
    state = state.astype(jnp.float32) / 255.

    value_pred, pi_dist = apply_fn(params_model, state, latent_factors)
    value_pred = value_pred[:, 0]

    log_prob = pi_dist.log_prob(action)

    value_pred_clipped = value_old + (value_pred - value_old).clip(
        -clip_eps, clip_eps)
    value_losses = jnp.square(value_pred - target)
    value_losses_clipped = jnp.square(value_pred_clipped - target)
    value_loss = 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()

    ratio = jnp.exp(log_prob - log_pi_old)
    gae = (gae - gae.mean()) / (gae.std() + 1e-8)
    loss_actor1 = ratio * gae
    loss_actor2 = jnp.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * gae
    loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
    loss_actor = loss_actor.mean()

    # Approximate using 100 samples
    samples = pi_dist.sample(100, seed=key)
    ent = -pi_dist.log_prob(samples).mean(0).mean()

    total_loss = loss_actor + critic_coeff * value_loss - entropy_coeff * ent

    return total_loss, (value_loss, loss_actor, ent, value_pred.mean(),
                        target.mean(), gae.mean(), log_prob.mean())


@partial(jax.jit, static_argnames=("sample"))
def select_action(
    train_state: TrainState,
    state: jnp.ndarray,
    latent_factors: jnp.ndarray,
    rng: PRNGKey,
    sample: bool = False
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, PRNGKey]:
    value, pi_dist = train_state.apply_fn(train_state.params, state,
                                          latent_factors)

    rng, key = jax.random.split(rng)
    if sample:
        action = pi_dist.sample(seed=key)
    else:
        # Can't analytically compute bijector mean
        action = jnp.tanh(pi_dist.distribution.mean())

    log_prob = pi_dist.log_prob(action)
    return action, log_prob, value[:, 0], rng


def get_transition(
    train_state: TrainState,
    env,
    state,
    latent_factors,
    batch,
    rng: PRNGKey,
):
    action, log_pi, value, new_key = select_action(train_state,
                                                   state.astype(jnp.float32) /
                                                   255.,
                                                   latent_factors,
                                                   rng,
                                                   sample=True)
    next_state, reward, done, info = env.step(action)
    task_ids = np.array([x['task_id'] for x in info])
    latent_factors = extract_latent_factors(info)
    batch.append(state, action, reward, done, np.array(log_pi),
                 np.array(value), task_ids, latent_factors)
    return train_state, next_state, latent_factors, batch, new_key, reward, done, info


@partial(jax.jit)
def flatten_dims(x):
    return x.swapaxes(0, 1).reshape(x.shape[0] * x.shape[1], *x.shape[2:])


@partial(jax.jit,
         static_argnames=("num_envs", "n_steps", "n_minibatch", "epoch_ppo",
                          "clip_eps", "entropy_coeff", "critic_coeff"))
def update_ppo(train_state: TrainState, batch: Tuple, num_envs: int,
               n_steps: int, n_minibatch: int, epoch_ppo: int, clip_eps: float,
               entropy_coeff: float, critic_coeff: float, rng: PRNGKey):

    state, action, reward, log_pi_old, value, target, gae, task_ids, latent_factors = batch

    size_batch = num_envs * n_steps
    assert size_batch % n_minibatch == 0
    size_minibatch = size_batch // n_minibatch

    idxes = jnp.arange(num_envs * n_steps)
    idxes_policy = []
    for _ in range(epoch_ppo):
        rng, key = jax.random.split(rng)
        idxes = jax.random.permutation(rng, idxes)
        idxes_policy.append(idxes)
    idxes_policy = jnp.array(idxes_policy).reshape(-1, size_minibatch)
    key, rng2 = jax.random.split(rng)

    avg_metrics_dict = defaultdict(int)

    state = flatten_dims(state)
    action = flatten_dims(action)

    log_pi_old = flatten_dims(log_pi_old)
    value = flatten_dims(value)
    target = flatten_dims(target)
    gae = flatten_dims(gae)
    if latent_factors is not None:
        latent_factors = flatten_dims(latent_factors)

    def scan_policy(train_state, idx):
        key, rng = jax.random.split(rng2)
        if latent_factors is not None:
            latent_factors_ = latent_factors[idx]
        else:
            latent_factors_ = latent_factors
        grad_fn = jax.value_and_grad(loss_actor_and_critic, has_aux=True)
        total_loss, grads = grad_fn(train_state.params,
                                    train_state.apply_fn,
                                    state=state[idx],
                                    target=target[idx],
                                    value_old=value[idx],
                                    log_pi_old=log_pi_old[idx],
                                    gae=gae[idx],
                                    action=action[idx],
                                    latent_factors=latent_factors_,
                                    clip_eps=clip_eps,
                                    critic_coeff=critic_coeff,
                                    entropy_coeff=entropy_coeff,
                                    key=key)
        grads = jax.tree_util.tree_map(jnp.nan_to_num, grads)
        train_state = train_state.apply_gradients(grads=grads)
        return train_state, total_loss

    train_state, total_loss = jax.lax.scan(scan_policy, train_state,
                                           idxes_policy)
    total_loss, (value_loss, loss_actor, ent, value_pred, target_val, gae_val,
                 log_prob) = total_loss

    avg_metrics_dict['total_loss'] += total_loss.mean()
    avg_metrics_dict['value_loss'] += value_loss.mean()
    avg_metrics_dict['loss_actor'] += loss_actor.mean()
    avg_metrics_dict['ent'] += ent.mean()
    avg_metrics_dict['value_pred'] += value_pred.mean()
    avg_metrics_dict['target_val'] += target_val.mean()
    avg_metrics_dict['gae_val'] += gae_val.mean()
    avg_metrics_dict['log_prob'] += log_prob.mean()
    avg_metrics_dict['action_norm'] += jnp.linalg.norm(action, axis=1).mean()

    return avg_metrics_dict, train_state, rng