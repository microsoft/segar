__author__ = "Bogdan Mazoure"
__copyright__ = "Copyright (c) Microsoft Corporation and Mila - Quebec AI " \
                "Institute"
__license__ = "MIT"

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
from segar.repl.metrics import MINE

from segar.repl.models import SimpleMLP
import torch

from utils import rollouts
import pickle

FLAGS = flags.FLAGS

flags.DEFINE_integer("n_rollouts", 10, "Number of per-env rollouts.")
flags.DEFINE_string("model_dir", "../data", "PPO weights directory")
flags.DEFINE_integer("num_envs", 1, "Number of rollout environments")
flags.DEFINE_boolean("sample", False, "Use a=E[pi(s)] or a~pi(s)?")


def main(argv):
    num_envs = FLAGS.num_envs
    probe_wasserstein = True
    probe_mine = True
    MAX_STEPS = 100
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
                         max_steps=MAX_STEPS,
                         _async=False,
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
    if probe_wasserstein:
        try:
            w2_df = pickle.load(open('../plots/01_Wasserstein.pkl', "rb"))
        except:
            w2_df = []
            num_test_levels = 500
            ctr = 0
            for task in ['empty', 'tiles', 'objects']:
                for difficulty in ['easy', 'medium', 'hard']:
                    for num_levels in [1, 10, 50, 100, 200]:
                        if task == 'empty':
                            env_name = "%s-%s-rgb" % (task, difficulty)
                            prefix = "checkpoint_%s_%s_%d" % (task, difficulty,
                                                              num_levels)
                        else:
                            env_name = "%sx1-%s-rgb" % (task, difficulty)
                            prefix = "checkpoint_%sx1_%s_%d" % (
                                task, difficulty, num_levels)
                        loaded_state = checkpoints.restore_checkpoint(
                            FLAGS.model_dir,
                            prefix=prefix,
                            target=train_state_ppo)
                        ckpt_path = glob.glob(
                            os.path.join(FLAGS.model_dir, prefix + '*'))
                        if not len(ckpt_path):
                            print(prefix)
                            continue
                        ckpt_path = ckpt_path[0]
                        seed = int(ckpt_path.split('_')[-1])

                        try:
                            env_train = SEGAREnv(env_name,
                                                 num_envs=num_envs,
                                                 num_levels=num_levels,
                                                 framestack=1,
                                                 resolution=64,
                                                 max_steps=MAX_STEPS,
                                                 _async=False,
                                                 seed=seed)
                            env_test = SEGAREnv(env_name,
                                                num_envs=num_envs,
                                                num_levels=num_test_levels,
                                                framestack=1,
                                                resolution=64,
                                                max_steps=MAX_STEPS,
                                                _async=False,
                                                seed=seed + 1)
                            returns_train, (states_train, zs_train,
                                            actions_train, factors_train,
                                            task_ids_train) = rollouts(
                                                env_train,
                                                loaded_state,
                                                rng,
                                                n_rollouts=FLAGS.n_rollouts,
                                                sample=FLAGS.sample)
                            returns_test, (states_test, zs_test, actions_test,
                                           factors_test,
                                           task_ids_test) = rollouts(
                                               env_test,
                                               loaded_state,
                                               rng,
                                               n_rollouts=FLAGS.n_rollouts,
                                               sample=FLAGS.sample)
                            w2_distance = task_set_init_dist(
                                env_test.env.envs[0].task_list,
                                env_train.env.envs[0].task_list)
                            w2_df.append(
                                pd.DataFrame({
                                    'returns':
                                    np.concatenate(
                                        [returns_train, returns_test], axis=0),
                                    'set':
                                    ['train'
                                     for _ in range(FLAGS.n_rollouts)] +
                                    ['test' for _ in range(FLAGS.n_rollouts)],
                                    # r'$\eta_{test}-\eta_{train}$': [summary],
                                    r'$W_2(\mathbb{P}_{test},\mathbb{P}_{train})$':
                                    w2_distance,
                                    'task':
                                    task,
                                    'difficulty':
                                    difficulty,
                                    'num_levels':
                                    num_levels
                                }))
                            print('W2(train levels, test levels)=%.5f' %
                                  w2_distance)
                            ctr += 1
                        except Exception as e:
                            print('Exception encountered in simulation:')
                            print(e)

            w2_df = pd.concat(w2_df)
            pickle.dump(w2_df, open('../plots/01_Wasserstein.pkl', "wb"))

        w2_df = w2_df.reset_index(drop=True)
        g = sns.FacetGrid(w2_df[w2_df['task'] == 'empty'],
                          hue="set",
                          col="num_levels",
                          row="difficulty",
                          sharex=False)
        g.map(plt.hist, "returns", alpha=.6)
        plt.legend()
        plt.savefig('../plots/03_returns.png')
        plt.clf()

        x = w2_df.groupby([
            'task', 'difficulty', 'num_levels',
            r'$W_2(\mathbb{P}_{test},\mathbb{P}_{train})$'
        ]).apply(lambda x: x[x['set'] == 'test']['returns'].mean() - x[x[
            'set'] == 'train']['returns'].mean()).reset_index()
        columns = list(x.columns)
        columns[-1] = r'$\eta_{test}-\eta_{train}$'
        x.columns = columns

        # Conditional plot per task type /difficulty
        sns.lmplot(
            x=r'$W_2(\mathbb{P}_{test},\mathbb{P}_{train})$',
            y=r'$\eta_{test}-\eta_{train}$',
            # hue='num_levels',
            row='task',
            col='difficulty',
            data=x)
        plt.savefig('../plots/01_Wasserstein.png')
        plt.clf()

        # Average plot with all tasks combined
        sns.lmplot(x=r'$W_2(\mathbb{P}_{test},\mathbb{P}_{train})$',
                   y=r'$\eta_{test}-\eta_{train}$',
                   data=x)
        plt.savefig('../plots/01_Wasserstein_joint.png')

    if probe_mine:
        try:
            mi_train_df, mi_train_df = pickle.load(
                open('../plots/02_MINE.pkl', "rb"))
        except:
            num_test_levels = 500
            mi_train_df = []
            mi_test_df = []
            ctr = 0
            for task in ['empty', 'tiles', 'objects']:
                for difficulty in ['easy', 'medium', 'hard']:
                    for num_levels in [1, 10, 50, 100, 200]:
                        if task == 'empty':
                            env_name = "%s-%s-rgb" % (task, difficulty)
                            prefix = "checkpoint_%s_%s_%d" % (task, difficulty,
                                                              num_levels)
                        else:
                            env_name = "%sx1-%s-rgb" % (task, difficulty)
                            prefix = "checkpoint_%sx1_%s_%d" % (
                                task, difficulty, num_levels)
                        loaded_state = checkpoints.restore_checkpoint(
                            FLAGS.model_dir,
                            prefix=prefix,
                            target=train_state_ppo)
                        ckpt_path = glob.glob(
                            os.path.join(FLAGS.model_dir, prefix + '*'))
                        if not len(ckpt_path):
                            print(prefix)
                            continue
                        ckpt_path = ckpt_path[0]
                        seed = int(ckpt_path.split('_')[-1])

                        env_train = SEGAREnv(env_name,
                                             num_envs=num_envs,
                                             num_levels=num_levels,
                                             framestack=1,
                                             resolution=64,
                                             max_steps=MAX_STEPS,
                                             _async=False,
                                             seed=seed)
                        env_test = SEGAREnv(env_name,
                                            num_envs=num_envs,
                                            num_levels=num_test_levels,
                                            framestack=1,
                                            resolution=64,
                                            max_steps=MAX_STEPS,
                                            _async=False,
                                            seed=seed + 1)

                        returns_train, (states_train, zs_train, actions_train,
                                        factors_train,
                                        task_ids_train) = rollouts(
                                            env_train,
                                            loaded_state,
                                            rng,
                                            n_rollouts=FLAGS.n_rollouts)
                        returns_test, (states_test, zs_test, actions_test,
                                       factors_test, task_ids_test) = rollouts(
                                           env_test,
                                           loaded_state,
                                           rng,
                                           n_rollouts=FLAGS.n_rollouts)

                        mine_net = SimpleMLP(n_input=256 + 100, n_out=1)
                        opt = torch.optim.Adam(mine_net.parameters(), lr=3e-4)
                        X_train = np.array(zs_train).reshape(
                            -1, zs_train[0].shape[-1])
                        Z_train = np.array(factors_train).reshape(
                            len(factors_train), -1)
                        X_test = np.array(zs_test)[:, 0]
                        Z_test = np.array(factors_test).reshape(
                            len(factors_test), -1)

                        mi_lb_train, mi_lb_test = MINE(mine_net, opt, X_train,
                                                       Z_train, X_test, Z_test)
                        mi_train_df.append(
                            pd.DataFrame({
                                'MI':
                                mi_lb_train['mi/train'],
                                'epoch':
                                np.arange(len(mi_lb_train['mi/train'])),
                                'task':
                                task,
                                'difficulty':
                                difficulty,
                                'num_levels':
                                num_levels
                            }))
                        mi_test_df.append(
                            pd.DataFrame({
                                'MI':
                                mi_lb_test['mi/test'],
                                'epoch':
                                np.arange(len(mi_lb_train['mi/test'])),
                                'task':
                                task,
                                'difficulty':
                                difficulty,
                                'num_levels':
                                num_levels
                            }))
                        ctr += 1

            mi_train_df = pd.concat(mi_train_df)
            mi_test_df = pd.concat(mi_test_df)
            pickle.dump((mi_train_df, mi_test_df),
                        open('../plots/02_MINE.pkl', "wb"))

        sns.lineplot(x='epoch',
                     y='MI',
                     hue='num_levels',
                     row='task',
                     col='difficulty',
                     data=mi_train_df.reset_index(drop=True))
        sns.savefig('../plots/02_MINE.png')

    return 0


if __name__ == '__main__':
    app.run(main)
