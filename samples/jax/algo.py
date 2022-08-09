__author__ = "R Devon Hjelm, Bogdan Mazoure, Florian Golemo"
__copyright__ = "Copyright (c) Microsoft Corporation and Mila - Quebec AI " \
                "Institute"
__license__ = "MIT"

from collections import defaultdict
from functools import partial
from typing import Any, Callable, Tuple

import flax
import jax
import jax.numpy as jnp
import numpy as np
from flax.training.train_state import TrainState
from jax.random import PRNGKey
import flax.linen as nn

import dm_pix as pix
"""
Inspired by code from Flax:
https://github.com/google/flax/blob/main/examples/ppo/ppo_lib.py
"""
"""
Helper methods
"""


@partial(jax.jit)
def flatten_dims(x: jnp.ndarray):
    return x.swapaxes(0, 1).reshape(x.shape[0] * x.shape[1], *x.shape[2:])


def random_crop(obs, rng, n_augs=1):
    augs = []
    # obs_ = jnp.pad(obs,[[0, 0], [2, 2], [2, 2], [0, 0]],'edge')
    obs_shape = obs.shape
    crop_shape = (obs.shape[0], int(obs.shape[1]*0.8),int(obs.shape[2]*0.8), obs.shape[3])
    for _ in range(n_augs):
        rng, key = jax.random.split(rng)
        augs.append(
            jax.image.resize(pix.random_crop(key=key,
                            image=obs,
                            crop_sizes=crop_shape), shape=obs_shape, method="linear"))
    return augs


def compute_grad_norm(grads: flax.core.frozen_dict.FrozenDict) -> jnp.ndarray:
    if hasattr(grads, "items"):
        acc = 0.
        n = 0
        for _, v in grads.items():
            acc += compute_grad_norm(v)
            n += 1
        acc /= n
    else:
        acc = jnp.linalg.norm(grads)
    return acc


def extract_latent_factors(infos: dict):
    latent_features = []
    for _, info in enumerate(infos):
        latent_features.append(info['latent_features'].reshape(-1))
    latent_features = jnp.stack(latent_features)
    return latent_features


def state_update(online_state, target_state, tau: float = 1.):
    """ Update key weights as tau * online + (1-tau) * target
    """
    new_weights = target_update(online_state.params, target_state.params, tau)
    
    target_state = target_state.replace(params=new_weights)
    return target_state


def target_update(online, target, tau: float):
    new_target_params = jax.tree_multimap(
        lambda p, tp: p * tau + tp * (1 - tau), online, target)

    return new_target_params


"""
Losses:
- MSE
- PPO
- CURL
"""

def l2_normalize(A, axis=-1, eps=1e-4):
    return A * jax.lax.rsqrt((A * A).sum(axis=axis, keepdims=True) + eps)


def cos_loss(p, z):
    # z = jax.lax.stop_gradient(z)
    p = l2_normalize(p, axis=1)
    z = l2_normalize(z, axis=1)
    # dist = 2 - 2 * jnp.sum(p * z, axis=1)
    return jnp.mean(jnp.sum((p-z)**2,axis=-1),axis=0)


def mse_loss(params: flax.core.frozen_dict.FrozenDict, apply_fn: Callable,
             X: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    X = jax.lax.stop_gradient(
        X)  # prevent latent factors to backprop into representation
    y_hat = apply_fn(params, X)
    # mimic batchnorm1d
    y = y / jnp.stack([jnp.linalg.norm(y, axis=1)] * y.shape[1]).transpose()
    predictor_loss = jnp.sum((y - y_hat)**2, 1).mean()
    return predictor_loss, (y - y_hat)**2


def loss_actor_and_critic(params_model: flax.core.frozen_dict.FrozenDict,
                          apply_fn: Callable[..., Any], state: jnp.ndarray,
                          target: jnp.ndarray, value_old: jnp.ndarray,
                          log_pi_old: jnp.ndarray, gae: jnp.ndarray,
                          action: jnp.ndarray, latent_factors: jnp.ndarray,
                          clip_eps: float, critic_coeff: float,
                          entropy_coeff: float, key: PRNGKey) -> jnp.ndarray:
    """
    Jointly train actor and critic with entropy reg.
    """
    state = state.astype(jnp.float32) / 255.

    value_pred, pi_dist, z = apply_fn(params_model, state, latent_factors)
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


def loss_curl(params_model: flax.core.frozen_dict.FrozenDict,
              apply_fn: Callable[..., Any],
              params_target: flax.core.frozen_dict.FrozenDict,
              apply_fn_target: Callable[..., Any], state_1: jnp.ndarray,
              state_2: jnp.ndarray, key: PRNGKey) -> jnp.ndarray:
    state_1 = state_1.astype(jnp.float32) / 255.
    state_2 = state_2.astype(jnp.float32) / 255.

    _, _, (z1, bilinear) = apply_fn(params_model, state_1, None)
    _, _, (z2, _) = apply_fn_target(params_target, state_2, None)
    z2 = jax.lax.stop_gradient(z2)

    # CURL loss
    logits = jnp.einsum("ai,bj,ij->ab", z1, z2, bilinear)
    # logits = logits - jnp.max(logits, axis=1)
    logits = nn.log_softmax(logits, axis=1)

    n_classes = logits.shape[0]
    # one_hot_labels = jax.nn.one_hot(jnp.arange(n_classes), num_classes=n_classes)
    one_hot_labels = jnp.eye(n_classes)
    
    curl_loss = -jnp.mean(jnp.sum(one_hot_labels * logits, axis=-1))

    return curl_loss, (curl_loss)


def loss_spr(params_model: flax.core.frozen_dict.FrozenDict,
              apply_fn: Callable[..., Any],
              params_target: flax.core.frozen_dict.FrozenDict,
              apply_fn_target: Callable[..., Any], state_t: jnp.ndarray,
              state_tp1: jnp.ndarray, action_t: jnp.ndarray, key: PRNGKey) -> jnp.ndarray:
    state_t = state_t.astype(jnp.float32) / 255.
    state_tp1 = state_tp1.astype(jnp.float32) / 255.

    _, _, (zt, (z_tilde_t, z_hat_t)) =  apply_fn(params_model, state_t, None, action_t)
    _, _, (zt, (z_tilde_tp1, z_hat_tp1)) =  apply_fn_target(params_target, state_tp1, None, None)

    spr_loss = cos_loss(z_hat_t, z_tilde_tp1)

    return spr_loss, (spr_loss)


@partial(jax.jit, static_argnames=("sample"))
def select_action(
    train_state: TrainState,
    state: jnp.ndarray,
    latent_factors: jnp.ndarray,
    rng: PRNGKey,
    sample: bool = False
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, PRNGKey]:
    """
    Select action either deterministically (mean) or sample
    """
    value, pi_dist, z = train_state.apply_fn(train_state.params, state,
                                             latent_factors)
    rng, key = jax.random.split(rng)
    if sample:
        action = pi_dist.sample(seed=key)
    else:
        # Can't analytically compute bijector mean
        action = jnp.tanh(pi_dist.distribution.mean())

    log_prob = pi_dist.log_prob(action)
    return action, log_prob, value[:, 0], z, rng


def get_transition(
    train_state: TrainState,
    env,
    state,
    latent_factors,
    batch,
    rng: PRNGKey,
) -> Tuple[TrainState, jnp.ndarray, jnp.ndarray, Tuple, PRNGKey, jnp.ndarray,
           jnp.ndarray, dict]:
    """
    Picks the next action a_t~pi(s_t) and adds the resulting transition to the
    replay buffer.
    """

    action, log_pi, value, _, new_key = select_action(
        train_state,
        state.astype(jnp.float32) / 255.,
        latent_factors,
        rng,
        sample=True)
    next_state, reward, done, info = env.step(action)
    task_ids = np.array([x['task_id'] for x in info])
    latent_factors = extract_latent_factors(info)
    batch.append(state, action, reward, done, np.array(log_pi),
                 np.array(value), task_ids, latent_factors)
    return (train_state, next_state, latent_factors, batch, new_key, reward,
            done, info)


@partial(jax.jit,
         static_argnames=("num_envs", "n_steps", "n_minibatch", "epoch_ppo",
                          "clip_eps", "entropy_coeff", "critic_coeff"))
def update_ppo(train_state: TrainState, batch: Tuple, num_envs: int,
               n_steps: int, n_minibatch: int, epoch_ppo: int, clip_eps: float,
               entropy_coeff: float, critic_coeff: float,
               rng: PRNGKey) -> Tuple[dict, TrainState, PRNGKey]:
    """
    Randomize PPO batch (n_envs x n_steps) into M minibatches, and optimize for
    E epochs, as per classical PPO implementation.
    """
    (state, action, reward, log_pi_old, value, target, gae, task_ids,
     latent_factors) = batch

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


@partial(jax.jit,
         static_argnames=("num_envs", "n_steps", "n_minibatch", "epoch_ppo",
                          "clip_eps", "entropy_coeff", "critic_coeff"))
def update_curl(train_state: TrainState, train_state_target: TrainState,
                batch: Tuple, num_envs: int, n_steps: int, n_minibatch: int,
                epoch_ppo: int, clip_eps: float, entropy_coeff: float,
                critic_coeff: float,
                rng: PRNGKey) -> Tuple[dict, TrainState, PRNGKey]:
    """
    CURL aux loss, performs data augmentation inside this function.
    """
    (state, action, reward, log_pi_old, value, target, gae, task_ids,
     latent_factors) = batch

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

    state_1, state_2 = random_crop(state, rng, n_augs=2)

    if latent_factors is not None:
        latent_factors = flatten_dims(latent_factors)

    def scan(train_state, idx):
        key, rng = jax.random.split(rng2)
        if latent_factors is not None:
            latent_factors_ = latent_factors[idx]
        else:
            latent_factors_ = latent_factors
        grad_fn = jax.value_and_grad(loss_curl, has_aux=True)
        total_loss, grads = grad_fn(train_state.params,
                                    train_state.apply_fn,
                                    train_state_target.params,
                                    train_state_target.apply_fn,
                                    state_1=state_1[idx],
                                    state_2=state_2[idx],
                                    key=key)
        grads = jax.tree_util.tree_map(jnp.nan_to_num, grads)
        train_state = train_state.apply_gradients(grads=grads)
        return train_state, total_loss

    train_state, total_loss = jax.lax.scan(scan, train_state, idxes_policy)
    total_loss, (curl_loss) = total_loss

    avg_metrics_dict['curl_loss'] += curl_loss.mean()

    return avg_metrics_dict, train_state, rng


@partial(jax.jit,
         static_argnames=("num_envs", "n_steps", "n_minibatch", "epoch_ppo",
                          "clip_eps", "entropy_coeff", "critic_coeff"))
def update_spr(train_state: TrainState, train_state_target: TrainState,
                batch: Tuple, num_envs: int, n_steps: int, n_minibatch: int,
                epoch_ppo: int, clip_eps: float, entropy_coeff: float,
                critic_coeff: float,
                rng: PRNGKey) -> Tuple[dict, TrainState, PRNGKey]:
    """
    SPR aux loss, performs data augmentation inside this function.
    """
    (state, action, reward, log_pi_old, value, target, gae, task_ids,
     latent_factors) = batch

    size_batch = num_envs * n_steps
    assert size_batch % n_minibatch == 0
    size_minibatch = size_batch // n_minibatch

    idxes = jnp.arange(size_batch)

    avg_metrics_dict = defaultdict(int)

    if latent_factors is not None:
        latent_factors = flatten_dims(latent_factors)

    for _ in range(epoch_ppo):
        rng, key = jax.random.split(rng)
        for mb in range(n_minibatch):
            idx = idxes[mb * size_minibatch:(mb+1)*size_minibatch]

            states_t = flatten_dims(state[:-1, idx])
            actions_t = flatten_dims(action[:-1, idx])
            states_tp1 = flatten_dims(state[1:, idx])
            # actions_tp1 = flatten_dims(action[1:, idx])

            states_t = random_crop(states_t, key, n_augs=1)[0]
            _, key = jax.random.split(key)
            states_tp1 = random_crop(states_tp1, key, n_augs=1)[0]
            _, key = jax.random.split(key)

            if latent_factors is not None:
                latent_factors_ = latent_factors[idx]
            else:
                latent_factors_ = latent_factors

            grad_fn = jax.value_and_grad(loss_spr, has_aux=True)
            total_loss, grads = grad_fn(train_state.params,
                                        train_state.apply_fn,
                                        train_state_target.params,
                                        train_state_target.apply_fn,
                                        state_t=states_t,
                                        state_tp1=states_tp1,
                                        action_t=actions_t,
                                        key=key)
            grads = jax.tree_util.tree_map(jnp.nan_to_num, grads)
            train_state = train_state.apply_gradients(grads=grads)

            total_loss, (spr_loss) = total_loss

            avg_metrics_dict['spr_loss'] += spr_loss.mean() / (n_minibatch * epoch_ppo)

    return avg_metrics_dict, train_state, rng