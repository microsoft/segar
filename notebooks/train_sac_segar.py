"""Inspired by https://github.com/ikostrikov/jaxrl/blob/main/examples/train_pixels.py"""

import os
import importlib
import subprocess

import random

import numpy as np
import jax
import tqdm
from absl import app, flags
from ml_collections import config_flags
from flax.training import checkpoints
from tensorboardX import SummaryWriter

import wandb
from jaxrl.agents.drq.drq_learner import DrQLearner
from jaxrl.agents.sac.sac_learner import SACLearner
from jaxrl.datasets.replay_buffer import ReplayBuffer
from jaxrl.evaluation import evaluate
from segar.envs.env import SEGAREnv

from collections import deque


FLAGS = flags.FLAGS
# Task
flags.DEFINE_string("env_name", "tilesx1-medium-rgb", "Env name")
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_integer("num_envs", 1, "Num of parallel envs.")
flags.DEFINE_integer("num_train_levels", 10, "Num of training levels envs.")
flags.DEFINE_integer("num_test_levels", 500, "Num of test levels envs.")
flags.DEFINE_integer("train_steps", 1_000_000, "Number of train frames.")
flags.DEFINE_integer("framestack", 1, "Number of frames to stack")
flags.DEFINE_integer("resolution", 64, "Resolution of pixel observations")
################ 
# Logging
flags.DEFINE_integer("checkpoint_interval", 10, "Checkpoint frequency")
flags.DEFINE_string("run_id", "jax_run",
                    "Run ID. Change that to change W&B name")
flags.DEFINE_string("wandb_mode", "disabled",
                    "W&B logging (disabled, online, offline)")
flags.DEFINE_string("wandb_key", None, "W&B key")
flags.DEFINE_string("wandb_entity", "dummy_username",
                    "W&B entity (username or team name)")
flags.DEFINE_string("wandb_project", "dummy_project", "W&B project name")
########
flags.DEFINE_string('save_dir', './supp_exps_returns_agent/', 'Tensorboard logging dir.')
flags.DEFINE_integer('eval_episodes', 10,
                     'Number of episodes used for evaluation.')
flags.DEFINE_integer('log_interval', 1000, 'Logging interval.')
flags.DEFINE_integer('eval_interval', 5000, 'Eval interval.') 
flags.DEFINE_integer('batch_size', 256, 'Mini batch size.') 
flags.DEFINE_integer('max_steps', 100, 'Number of environment steps.')
flags.DEFINE_integer('start_training', int(1e3),
                     'Number of environment steps to start training.')
flags.DEFINE_integer(
    'action_repeat', None,
    'Action repeat, if None, uses 2 or PlaNet default values.')
flags.DEFINE_boolean('tqdm', True, 'Use tqdm progress bar.')

PLANET_ACTION_REPEAT = {
    'cartpole-swingup': 8,
    'reacher-easy': 4,
    'cheetah-run': 4,
    'finger-spin': 2,
    'ball_in_cup-catch': 4,
    'walker-walk': 2
}

kwargs = {
    'algo' : 'sac',
    'actor_lr' : 3e-4,
    'critic_lr' : 3e-4,
    'temp_lr' : 3e-4,
    'hidden_dims' : (256, 256),
    'discount' : 0.99,
    'tau' : 0.005,
    'target_update_period' : 1,
    'init_temperature' : 0.1,
    'target_entropy' : None,
    'replay_buffer_size': 100000
}

def main(_):
#     kwargs = dict(FLAGS.config)
    # Setting all rnadom seeds
    if FLAGS.seed == -1:
        seed = np.random.randint(100000000)
    else:
        seed = FLAGS.seed
    np.random.seed(seed)

    algo = kwargs.pop('algo')
    replay_buffer_size = kwargs.pop('replay_buffer_size')

    # If W&B is to be used - set job and group names
    if FLAGS.wandb_key is not None:
        os.environ["WANDB_API_KEY"] = FLAGS.wandb_key
    
    group_name = "%s" % (algo)
    run_name = "%s_nenvs_%d_ltrain_%d_ltest_%d" % (FLAGS.env_name, FLAGS.num_envs, FLAGS.num_train_levels, FLAGS.num_test_levels)

    wandb.init(
        project=FLAGS.wandb_project,
        entity=FLAGS.wandb_entity,
        config=FLAGS,
        group=group_name,
        name=run_name,
        sync_tensorboard=False,
        mode=FLAGS.wandb_mode,
        dir=FLAGS.save_dir,
    )

    summary_writer = SummaryWriter(
        os.path.join(FLAGS.save_dir, group_name, run_name, str(FLAGS.seed)))

    if FLAGS.action_repeat is not None:
        action_repeat = FLAGS.action_repeat
    else:
        action_repeat = 1
        # action_repeat = PLANET_ACTION_REPEAT.get(FLAGS.env_name, 2)

    env_args = dict(
        max_steps=FLAGS.max_steps,
        save_path=os.path.join(FLAGS.save_dir, group_name, run_name, str(FLAGS.seed))
    )

    env = SEGAREnv(
        FLAGS.env_name,
        num_envs=FLAGS.num_envs,
        num_levels=FLAGS.num_train_levels,
        framestack=FLAGS.framestack,
        _async=False,
        seed=FLAGS.seed,
        env_args=env_args
    )
    eval_env = SEGAREnv(
        FLAGS.env_name,
        num_envs=1,
        num_levels=FLAGS.num_test_levels,
        framestack=FLAGS.framestack,
        _async=False,
        seed=FLAGS.seed + 42,
        env_args=env_args
    )

    np.random.seed(FLAGS.seed)
    random.seed(FLAGS.seed)
    if algo == 'sac':
        agent = SACLearner(FLAGS.seed,
                           env.observation_space.sample(),
                           jax.numpy.array(env.action_space.sample())[None, :], **kwargs)
    else:
        agent = DrQLearner(FLAGS.seed,
                        env.observation_space.sample(),
                        jax.numpy.array(env.action_space.sample())[None, :], **kwargs)

    replay_buffer = ReplayBuffer(
        env.observation_space, env.action_space, replay_buffer_size or FLAGS.max_steps // action_repeat)

    eval_returns = []
    observation, done = env.reset(), False

    ####
    returns_train_buf = deque(maxlen=10)
    success_train_buf = deque(maxlen=10)
    #####

    for i in tqdm.tqdm(range(1, FLAGS.train_steps // action_repeat + 1),
                       smoothing=0.1,
                       disable=not FLAGS.tqdm):
        if i < FLAGS.start_training:
            action = env.action_space.sample()[None, :]
        else:
            action = tuple(agent.sample_actions(observation)[None, :])

        next_observation, reward, done, info = env.step(action)
       
        mask = 0
        for e in range(len(info)):
            maybe_success = info[e].get("success")
            if maybe_success:
                success_train_buf.append(maybe_success)
            maybe_epinfo = info[e].get("returns")
            if maybe_epinfo:
                returns_train_buf.append(maybe_epinfo)
                wandb.log({f'training/all_returns': maybe_epinfo}, step=i)

            maybe_timelimit = info[e].get("TimeLimit.truncated")
            if maybe_timelimit:
                mask = 1

        replay_buffer.insert(jax.numpy.array(observation),
                             jax.numpy.array(action)[None, :], reward, mask, float(done[0]),
                             jax.numpy.array(next_observation))
        observation = next_observation

        if i >= FLAGS.start_training:
            batch = replay_buffer.sample(FLAGS.batch_size)
            update_info = agent.update(batch)

            if i % FLAGS.log_interval == 0:
                for k, v in update_info.items():
                    summary_writer.add_scalar(f'training/{k}', v, i)
                    wandb.log({f'training/{k}': v}, step=i)
                summary_writer.flush()

        if i % FLAGS.eval_interval == 0:
            eval_stats = evaluate(agent, eval_env, FLAGS.eval_episodes)

            for k, v in eval_stats.items():
                summary_writer.add_scalar(f'evaluation/average_{k}', v, i)
                wandb.log({f'evaluation/average_{k}': v}, step=i)
            summary_writer.flush()

            eval_returns.append((i, eval_stats['returns']))
            np.savetxt(os.path.join(FLAGS.save_dir, group_name, run_name, f'{FLAGS.seed}.txt'),
                       eval_returns,
                       fmt=['%d', '%.1f'])
        
        if i % FLAGS.checkpoint_interval == 0:
            model_dir = os.path.join(FLAGS.save_dir, group_name, run_name, str(FLAGS.seed))
            checkpoints.save_checkpoint(
                ckpt_dir=model_dir,
                target=(agent.actor, agent.critic, agent.target_critic, agent.temp),
                step=i,
                overwrite=True,
                keep=1,
            )
            
if __name__ == '__main__':
    app.run(main)
