__author__ = "Bogdan Mazoure"
__copyright__ = "Copyright (c) Microsoft Corporation and Mila - Quebec AI " \
                "Institute"
__license__ = "MIT"

import glob
import os
import itertools

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

import palettable

FLAGS = flags.FLAGS

flags.DEFINE_integer("n_rollouts", 10, "Number of per-env rollouts.")
flags.DEFINE_string("model_dir", "../data", "PPO weights directory")
flags.DEFINE_integer("num_envs", 1, "Number of rollout environments")
flags.DEFINE_boolean("sample", False, "Use a=E[pi(s)] or a~pi(s)?")
flags.DEFINE_list("probes", ['wasserstein', 'mine', 'ks'], "List of probes")


def main(argv):
    num_envs = FLAGS.num_envs
    probe_wasserstein = 'wasserstein' in FLAGS.probes
    probe_mine = 'mine' in FLAGS.probes
    probe_ks = 'ks' in FLAGS.probes
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
                         deterministic_visuals=False,
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
        if os.path.isfile('../plots/01_Wasserstein.pkl'):
            w2_df = pickle.load(open('../plots/01_Wasserstein.pkl', "rb"))
        else:
            w2_df = []
            num_test_levels = 500
            ctr = 0
            for task in ['empty', 'tiles', 'objects']:
                for difficulty in ['easy', 'medium', 'hard']:
                    for num_levels in [1, 10, 50, 100, 200]:
                        if task == 'empty':
                            env_name = "%s-%s-rgb" % (task, difficulty)
                            prefix = "checkpoint_%s_%s_%d_" % (
                                task, difficulty, num_levels)
                        else:
                            env_name = "%sx1-%s-rgb" % (task, difficulty)
                            prefix = "checkpoint_%sx1_%s_%d_" % (
                                task, difficulty, num_levels)
                        ckpt_path = glob.glob(
                            os.path.join(FLAGS.model_dir, prefix + '*'))
                        if not len(ckpt_path):
                            print(prefix)
                            continue
                        for ckpt in ckpt_path:
                            seed = int(ckpt.split('_')[-1])
                            prefix = prefix+str(seed)
                            loaded_state = checkpoints.restore_checkpoint(
                                FLAGS.model_dir,
                                prefix=prefix,
                                target=train_state_ppo)
                            
                            try:
                                env_train = SEGAREnv(env_name,
                                                    num_envs=num_envs,
                                                    num_levels=num_levels,
                                                    framestack=1,
                                                    resolution=64,
                                                    max_steps=MAX_STEPS,
                                                    _async=False,
                                                    deterministic_visuals=False,
                                                    seed=seed)
                                env_test = SEGAREnv(env_name,
                                                    num_envs=num_envs,
                                                    num_levels=num_test_levels,
                                                    framestack=1,
                                                    resolution=64,
                                                    max_steps=MAX_STEPS,
                                                    _async=False,
                                                    deterministic_visuals=False,
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
        columns[1] = 'Difficulty'
        x.columns = columns

        palette = palettable.cartocolors.qualitative.Pastel_10.mpl_colors

        # Conditional plot per task type /difficulty
        tasks = ['Empty', 'Objects', 'Tiles']
        with sns.plotting_context("notebook", font_scale=1.5):
            g = sns.lmplot(
                x=r'$W_2(\mathbb{P}_{test},\mathbb{P}_{train})$',
                y=r'$\eta_{test}-\eta_{train}$',
                hue='Difficulty',
                # row='task',
                col='task',
                # col_order=['easy','medium','hard'],
                palette=palette,
                sharey=False,
                ci=75,
                data=x)
            for ax, task in zip(g.axes.flatten(), tasks):
                ax.set_title(task)
            # sns.move_legend(g, "lower center", bbox_to_anchor=(.5, 1), ncol=3)
            # plt.tight_layout()
            plt.savefig('../plots/01_Wasserstein.png')
            plt.clf()

        # Average plot with all tasks combined
        with sns.plotting_context("notebook", font_scale=1.25):
            g = sns.lmplot(x=r'$W_2(\mathbb{P}_{test},\mathbb{P}_{train})$',
                           y=r'$\eta_{test}-\eta_{train}$',
                           line_kws={'color': 'deeppink'},
                           scatter_kws={'color': 'deeppink'},
                           data=x)
            plt.savefig('../plots/01_Wasserstein_joint.png')

    """
    Probe 2. Compute MINE lower-bound on MI between agent's state representation
    and latent factors.
    """
    if probe_mine:
        if os.path.isfile('../plots/02_MINE.pkl'):
            mi_train_df, mi_test_df = pickle.load(
                open('../plots/02_MINE.pkl', "rb"))
        else:
            num_test_levels = 500
            mi_train_df = []
            mi_test_df = []
            ctr = 0
            for task in ['empty', 'tiles', 'objects']:
                for difficulty in ['easy', 'medium', 'hard']:
                    for num_levels in [1, 10, 50, 100, 200]:
                        if task == 'empty':
                            env_name = "%s-%s-rgb" % (task, difficulty)
                            prefix = "checkpoint_%s_%s_%d_" % (
                                task, difficulty, num_levels)
                        else:
                            env_name = "%sx1-%s-rgb" % (task, difficulty)
                            prefix = "checkpoint_%sx1_%s_%d_" % (
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
                                                 deterministic_visuals=False,
                                                 seed=seed)
                            env_test = SEGAREnv(env_name,
                                                num_envs=num_envs,
                                                num_levels=num_test_levels,
                                                framestack=1,
                                                resolution=64,
                                                max_steps=MAX_STEPS,
                                                _async=False,
                                                deterministic_visuals=False,
                                                seed=seed + 1)

                            returns_train, (states_train, zs_train,
                                            actions_train, factors_train,
                                            task_ids_train) = rollouts(
                                                env_train,
                                                loaded_state,
                                                rng,
                                                n_rollouts=FLAGS.n_rollouts)
                            returns_test, (states_test, zs_test, actions_test,
                                           factors_test,
                                           task_ids_test) = rollouts(
                                               env_test,
                                               loaded_state,
                                               rng,
                                               n_rollouts=FLAGS.n_rollouts)

                            mine_net = SimpleMLP(n_input=256 + 100, n_out=1)
                            opt = torch.optim.Adam(mine_net.parameters(),
                                                   lr=3e-4)
                            X_train = np.array(zs_train).reshape(
                                -1, zs_train[0].shape[-1])
                            Z_train = np.array(factors_train).reshape(
                                len(factors_train), -1)
                            X_test = np.array(zs_test)[:, 0]
                            Z_test = np.array(factors_test).reshape(
                                len(factors_test), -1)

                            mi_lb_train, mi_lb_test = MINE(
                                mine_net, opt, X_train, Z_train, X_test,
                                Z_test)

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
                                    np.arange(len(mi_lb_test['mi/test'])),
                                    'task':
                                    task,
                                    'difficulty':
                                    difficulty,
                                    'num_levels':
                                    num_levels
                                }))
                            ctr += 1
                        except Exception as e:
                            print('Exception encountered in simulation:')
                            print(e)

            mi_train_df = pd.concat(mi_train_df)
            mi_test_df = pd.concat(mi_test_df)
            pickle.dump((mi_train_df, mi_test_df),
                        open('../plots/02_MINE.pkl', "wb"))

        palette = palettable.scientific.sequential.Acton_20.mpl_colormap
        sns.relplot(x='epoch',
                    y='MI',
                    hue='num_levels',
                    row='task',
                    col='difficulty',
                    col_order=['easy', 'medium', 'hard'],
                    kind='line',
                    palette=palette,
                    data=mi_train_df.reset_index(drop=True))
        plt.savefig('../plots/02_MINE.png')

    """
    Probe 3. Compute Kolmogorov-Smirnov statistic between pairs of latent
    factors' true CDFs.
    """
    if probe_ks:
        if os.path.isfile('../plots/04_KS.pkl'):
            ks_df = pickle.load(open('../plots/04_KS.pkl', "rb"))
        else:
            ks_df = []
            num_test_levels = 500
            ctr = 0
            for task in ['empty', 'tiles', 'objects']:
                for difficulty in ['easy', 'medium', 'hard']:
                    for num_levels in [1, 10, 50, 100, 200]:
                        if task == 'empty':
                            env_name = "%s-%s-rgb" % (task, difficulty)
                            prefix = "checkpoint_%s_%s_%d_" % (
                                task, difficulty, num_levels)
                        else:
                            env_name = "%sx1-%s-rgb" % (task, difficulty)
                            prefix = "checkpoint_%sx1_%s_%d_" % (
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
                                                 deterministic_visuals=False,
                                                 seed=seed)
                            env_test = SEGAREnv(env_name,
                                                num_envs=num_envs,
                                                num_levels=num_test_levels,
                                                framestack=1,
                                                resolution=64,
                                                max_steps=MAX_STEPS,
                                                _async=False,
                                                deterministic_visuals=False,
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
                            n, m = len(env_test.env.envs[0].task_list), len(env_train.env.envs[0].task_list)
                            ks_distance = np.zeros((n,m))
                            # Pick up to 100 levels bc otherwise becomes too slow
                            for i, test_task in enumerate(env_test.env.envs[0].task_list[:100]):
                                for j, train_task in enumerate(env_train.env.envs[0].task_list[:100]):
                                    factors_test=test_task._initialization.get_dists_from_init()
                                    factors_train=train_task._initialization.get_dists_from_init()
                                    thing_factor_sets = []

                                    def factor_cdf(factor_list):
                                        support = []
                                        p = []
                                        for thing, factors in factor_list.items():
                                            for factor, value in factors.items():
                                                if value.scipy_dist is not None:
                                                    p.append(value.scipy_dist.cdf)
                                                    support.append(value.scipy_dist.support())
                                                    
                                        def cdf(X):
                                            acc = 1.
                                            for F, x in zip(p,X):
                                                acc *= F(x)
                                            return acc

                                        return cdf, support
                                    LOW = -100.
                                    HIGH = 100.
                                    n_atoms = 10
                                    cdf_test, support_test= factor_cdf(factors_test)
                                    cdf_train, support_train = factor_cdf(factors_train)
                                    support = []
                                    for s_test, s_train in zip(support_test, support_train):
                                        lo = max(max(s_test[0], s_train[0]), LOW)
                                        hi =  min(min(s_test[1], s_train[1]), HIGH)
                                        support.append(np.linspace(lo, hi, n_atoms))
                                    support = list(itertools.product(*support))
                                    x_max = 0.
                                    delta_max = 0.
                                    for x in support:
                                        delta = np.abs(cdf_test(x)-cdf_train(x))
                                        if delta > delta_max:
                                            delta_max = delta
                                            x_max = x
                                    ks_distance[i,j] = delta_max
                            
                            ks_distance = np.mean(ks_distance)
                            ks_df.append(
                                pd.DataFrame({
                                    'returns':
                                    np.concatenate(
                                        [returns_train, returns_test], axis=0),
                                    'set':
                                    ['train'
                                     for _ in range(FLAGS.n_rollouts)] +
                                    ['test' for _ in range(FLAGS.n_rollouts)],
                                    # r'$\eta_{test}-\eta_{train}$': [summary],
                                    r'$\sup_z||\mathbb{P}_{test}-\mathbb{P}_{train}||$':
                                    ks_distance,
                                    'task':
                                    task,
                                    'difficulty':
                                    difficulty,
                                    'num_levels':
                                    num_levels
                                }))
                            print('L_inf(train levels, test levels)=%.5f' %
                                  ks_distance)
                            ctr += 1
                        except Exception as e:
                            print('Exception encountered in simulation:')
                            print(e)
            
            ks_df = pd.concat(ks_df)
            pickle.dump(ks_df, open('../plots/04_KS.pkl', "wb"))

        ks_df = ks_df.reset_index(drop=True)

        x = ks_df.groupby([
            'task', 'difficulty', 'num_levels',
            r'$\sup_z||\mathbb{P}_{test}-\mathbb{P}_{train}||$'
        ]).apply(lambda x: x[x['set'] == 'test']['returns'].mean() - x[x[
            'set'] == 'train']['returns'].mean()).reset_index()
        columns = list(x.columns)
        columns[-1] = r'$\eta_{test}-\eta_{train}$'
        columns[1] = 'Difficulty'
        x.columns = columns

        palette = palettable.cartocolors.qualitative.Pastel_10.mpl_colors

        # # Conditional plot per task type /difficulty
        # tasks = ['Empty', 'Objects', 'Tiles']
        # with sns.plotting_context("notebook", font_scale=1.5):
        #     g = sns.lmplot(
        #         x=r'$W_2(\mathbb{P}_{test},\mathbb{P}_{train})$',
        #         y=r'$\eta_{test}-\eta_{train}$',
        #         hue='Difficulty',
        #         # row='task',
        #         col='task',
        #         # col_order=['easy','medium','hard'],
        #         palette=palette,
        #         sharey=False,
        #         ci=75,
        #         data=x)
        #     for ax, task in zip(g.axes.flatten(), tasks):
        #         ax.set_title(task)
        #     # sns.move_legend(g, "lower center", bbox_to_anchor=(.5, 1), ncol=3)
        #     # plt.tight_layout()
        #     plt.savefig('../plots/01_Wasserstein.png')
        #     plt.clf()

        # # Average plot with all tasks combined
        # with sns.plotting_context("notebook", font_scale=1.25):
        #     g = sns.lmplot(x=r'$W_2(\mathbb{P}_{test},\mathbb{P}_{train})$',
        #                    y=r'$\eta_{test}-\eta_{train}$',
        #                    line_kws={'color': 'deeppink'},
        #                    scatter_kws={'color': 'deeppink'},
        #                    data=x)
        #     plt.savefig('../plots/01_Wasserstein_joint.png')
    return 0


if __name__ == '__main__':
    app.run(main)
