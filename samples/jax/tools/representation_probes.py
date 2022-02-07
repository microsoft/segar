import glob
import os

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
import pandas as pd
import seaborn as sns
from absl import app, flags
from flax.training import checkpoints
from flax.training.train_state import TrainState
from jax.random import PRNGKey
from samples.jax.models import TwinHeadModel
from segar.envs.env import SEGAREnv
from segar.mdps.metrics import task_set_init_dist

from utils import rollouts

FLAGS = flags.FLAGS

flags.DEFINE_integer("train_steps", 1_000, "Number of train frames.")
flags.DEFINE_integer("n_rollouts", 100, "Number of per-env rollouts.")
flags.DEFINE_string("model_dir", "../data", "PPO weights directory")


def main(argv):
    """
    Load the pre-trained PPO model
    """
    seed = np.random.randint(100000000)
    np.random.seed(seed)
    rng = PRNGKey(seed)
    rng, key = jax.random.split(rng)
    dummy_env = SEGAREnv("empty-easy-rgb",
                         num_envs=1,
                         num_levels=1,
                         framestack=1,
                         resolution=64,
                         seed=123)

    n_action = dummy_env.action_space[0].shape[-1]
    model_ppo = TwinHeadModel(action_dim=n_action,
                              prefix_critic='vfunction',
                              prefix_actor="policy",
                              action_scale=1.)

    state = dummy_env.reset().astype(jnp.float32) / 255.

    tx = optax.chain(optax.clip_by_global_norm(2), optax.adam(3e-4, eps=1e-5))
    params_model = model_ppo.init(key, state, None)
    train_state_ppo = TrainState.create(apply_fn=model_ppo.apply,
                                        params=params_model,
                                        tx=tx)
    """
    Probe 1. Compute 2-Wasserstein between task samples
    """
    returns_df = []
    w2_df = []
    num_test_levels = 500
    for task in ['empty', 'tiles', 'objects']:
        for difficulty in ['easy', 'medium', 'hard']:
            for num_levels in [1, 10, 50]:
                prefix = "checkpoint_%s_%s_%d" % (task, difficulty, num_levels)
                if task == 'empty':
                    env_name = "%s-%s-rgb" % (task, difficulty)
                else:
                    env_name = "%sx1-%s-rgb" % (task, difficulty)
                loaded_state = checkpoints.restore_checkpoint(
                    FLAGS.model_dir, prefix=prefix, target=train_state_ppo)
                ckpt_path = glob.glob(
                    os.path.join(FLAGS.model_dir, prefix + '*'))
                if not len(ckpt_path):
                    print(prefix)
                    continue
                ckpt_path = ckpt_path[0]
                seed = int(ckpt_path.split('_')[-1])

                env_train = SEGAREnv(env_name,
                                     num_envs=1,
                                     num_levels=num_levels,
                                     framestack=1,
                                     resolution=64,
                                     seed=seed)
                env_test = SEGAREnv(env_name,
                                    num_envs=1,
                                    num_levels=num_test_levels,
                                    framestack=1,
                                    resolution=64,
                                    seed=seed + 1)

                returns_train, (states_train, actions_train,
                                factors_train) = rollouts(
                                    env_train,
                                    loaded_state,
                                    rng,
                                    n_rollouts=FLAGS.n_rollouts)
                returns_test, (states_test, actions_test,
                               factors_test) = rollouts(
                                   env_test,
                                   loaded_state,
                                   rng,
                                   n_rollouts=FLAGS.n_rollouts)
                summary = np.mean(returns_test) - np.mean(returns_train)
                returns_df.append(summary)
                w2_distance = task_set_init_dist(
                    env_train.env.envs[0].task_list,
                    env_test.env.envs[0].task_list)
                w2_df.append(w2_distance)
                print('W2(train levels, test levels)=%.5f' % w2_distance)
    data = pd.DataFrame({
        r'$\eta_{test}-\eta_{train}$': returns_df,
        r'$W_2(\mathbb{P}_{test},\mathbb{P}_{train})$': w2_df
    })
    sns.scatterplot(x=r'$W_2(\mathbb{P}_{test},\mathbb{P}_{train})$',
                    y=r'$\eta_{test}-\eta_{train}$',
                    data=data)
    plt.show()
    # predictor = MLP(dims=[256, 20 * 5])
    # tx_predictor = optax.chain(optax.clip_by_global_norm(2),
    #                            optax.adam(3e-4, eps=1e-5))

    # z_obs = loaded_state.apply_fn(loaded_state.params,
    #                               state,
    #                               method=model_ppo.encode)
    # params_predictor = predictor.init(key, z_obs)
    # train_state_predictor = TrainState.create(apply_fn=predictor.apply,
    #                                           params=params_predictor,
    #                                           tx=tx_predictor)

    # batch_obs = []
    # batch_latents = []
    # for i in range(FLAGS.train_steps):
    #     action, log_pi, value, new_key = select_action(
    #         loaded_state,
    #         state.astype(jnp.float32) / 255.,
    #         latent_factors=None,
    #         rng=rng,
    #         sample=True)
    #     state, reward, done, info = env.step(action)
    #     latent_features = extract_latent_factors(info)
    #     batch_obs.append(state)
    #     batch_latents.append(latent_features)

    #     if (len(batch_obs) * len(batch_obs[0])) == FLAGS.batch_size:
    #         # Representation learning part
    #         rng, key = jax.random.split(rng)
    #         X = loaded_state.apply_fn(loaded_state.params,
    #                                   jnp.stack(batch_obs).reshape(
    #                                       -1, *batch_obs[0].shape[1:]),
    #                                   method=model_ppo.encode)
    #         y = jnp.stack(batch_latents).reshape(-1,
    #                                              *batch_latents[0].shape[1:])

    #         grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    #         (total_loss,
    #          square_res), grads = grad_fn(train_state_predictor.params,
    #                                       train_state_predictor.apply_fn,
    #                                       X=X,
    #                                       y=y)
    #         train_state_predictor = train_state_predictor.apply_gradients(
    #             grads=grads)

    #         per_factor_error = square_res.reshape(-1, 20, 5).sum(2).mean(0)
    #         print('Error')
    #         print(per_factor_error)
    #         batch_obs = []
    #         batch_latents = []

    # return 0


if __name__ == '__main__':
    app.run(main)
