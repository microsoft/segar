import glob
import importlib
import os
import subprocess
from collections import deque
from typing import Any

import numpy as np
import optax
import wandb
from absl import app, flags
from flax.training import checkpoints
from flax.training.train_state import TrainState

import jax
import jax.numpy as jnp


def setup_packages():
    subprocess.call(
        "pip install -e {0}".format(
            os.path.join(os.path.dirname(os.path.realpath(__file__)),
                         os.pardir, os.pardir)),
        shell=True,
    )
    import site

    importlib.reload(site)
    spec = importlib.util.find_spec("segar")
    globals()["segar"] = importlib.util.module_from_spec(spec)


setup_packages()

from segar.envs.env import SEGAREnv

from algo import (extract_latent_factors, get_transition, mse_loss,
                  select_action, update_ppo)
from buffer import Batch
from jax.random import PRNGKey
from models import MLP, TwinHeadModel

try:
    from azureml.core.run import Run
except Exception as e:
    print("Failed to import AzureML")
    print(e)


def safe_mean(x: Any):
    return np.nan if len(x) == 0 else np.mean(x)


FLAGS = flags.FLAGS
# Task
flags.DEFINE_string("env_name", "empty-easy-rgb", "Env name")
flags.DEFINE_integer("seed", 1, "Random seed.")
flags.DEFINE_integer("num_envs", 64, "Num of parallel envs.")
flags.DEFINE_integer("num_train_levels", 10, "Num of training levels envs.")
flags.DEFINE_integer("num_test_levels", 500, "Num of test levels envs.")
flags.DEFINE_integer("train_steps", 1_000_000, "Number of train frames.")
flags.DEFINE_integer("framestack", 1, "Number of frames to stack")
flags.DEFINE_integer("resolution", 64, "Resolution of pixel observations")
# PPO
flags.DEFINE_float("max_grad_norm", 10, "Max grad norm")
flags.DEFINE_float("gamma", 0.999, "Gamma")
flags.DEFINE_integer("n_steps", 30, "GAE n-steps")
flags.DEFINE_integer("n_minibatch", 4, "Number of PPO minibatches")
flags.DEFINE_float("lr", 1e-4, "PPO learning rate")
flags.DEFINE_integer("epoch_ppo", 1, "Number of PPO epochs on a single batch")
flags.DEFINE_float("clip_eps", 0.2, "Clipping range")
flags.DEFINE_float("gae_lambda", 0.95, "GAE lambda")
flags.DEFINE_float("entropy_coeff", 1e-3, "Entropy loss coefficient")
flags.DEFINE_float("critic_coeff", 0.1, "Value loss coefficient")
# Ablations
flags.DEFINE_boolean(
    "probe_latent_factors",
    True,
    "Probe latent factors from the PPO state representation?",
)
flags.DEFINE_boolean("add_latent_factors", False,
                     "Add latent factors to PPO state representation?")
flags.DEFINE_boolean("log_episodes", False, "Log episode samples on W&B?")
# Logging
flags.DEFINE_integer("checkpoint_interval", 10, "Checkpoint frequency")
flags.DEFINE_string("output_dir", ".", "Output dir")
flags.DEFINE_string("run_id", "jax_ppo",
                    "Run ID. Change that to change W&B name")
flags.DEFINE_string("wandb_mode", "disabled",
                    "W&B logging (disabled, online, offline)")
flags.DEFINE_string("wandb_key", None, "W&B key")
flags.DEFINE_string("wandb_entity", "dummy_username",
                    "W&B entity (username or team name)")
flags.DEFINE_string("wandb_project", "dummy_project", "W&B project name")


def main(argv):
    # Setting all rnadom seeds
    if FLAGS.seed == -1:
        seed = np.random.randint(100000000)
    else:
        seed = FLAGS.seed
    np.random.seed(seed)
    key = PRNGKey(seed)

    # If W&B is to be used - set job and group names
    if FLAGS.wandb_key is not None:
        os.environ["WANDB_API_KEY"] = FLAGS.wandb_key
    group_name = "%s_%s_%d" % (FLAGS.run_id, FLAGS.env_name,
                               FLAGS.num_train_levels)
    run_name = "%s_%s_%d_%d" % (
        FLAGS.run_id,
        FLAGS.env_name,
        FLAGS.num_train_levels,
        np.random.randint(100000000),
    )

    run = wandb.init(
        project=FLAGS.wandb_project,
        entity=FLAGS.wandb_entity,
        config=FLAGS,
        group=group_name,
        name=run_name,
        sync_tensorboard=False,
        mode=FLAGS.wandb_mode,
        dir=FLAGS.output_dir,
    )

    # Initialize train and test environemnts
    # Test environments always have the same number of test levels for
    # fair comparison across runs
    MAX_STEPS = 100
    env = SEGAREnv(
        FLAGS.env_name,
        num_envs=FLAGS.num_envs,
        num_levels=FLAGS.num_train_levels,
        framestack=FLAGS.framestack,
        resolution=FLAGS.resolution,
        max_steps=MAX_STEPS,
        _async=False,
        seed=FLAGS.seed,
    )
    env_test = SEGAREnv(
        FLAGS.env_name,
        num_envs=1,
        num_levels=FLAGS.num_test_levels,
        framestack=FLAGS.framestack,
        resolution=FLAGS.resolution,
        max_steps=MAX_STEPS,
        _async=False,
        seed=FLAGS.seed + 1,
    )
    n_action = env.action_space[0].shape[-1]

    # Create PPO model, optimizer and buffer
    model = TwinHeadModel(
        action_dim=n_action,
        prefix_critic="vfunction",
        prefix_actor="policy",
        action_scale=1.0,
        add_latent_factors=FLAGS.add_latent_factors,
    )

    state = env.reset()
    state_test = env_test.reset()
    if FLAGS.add_latent_factors:
        next_state, reward, done, info = env.step(
            env.env.action_space.sample())
        latent_factors = extract_latent_factors(info)
        params_model = model.init(key, state, latent_factors)
    else:
        params_model = model.init(key, state, None)
        latent_factors = None

    tx = optax.chain(optax.clip_by_global_norm(FLAGS.max_grad_norm),
                     optax.adam(FLAGS.lr, eps=1e-5))
    train_state = TrainState.create(apply_fn=model.apply,
                                    params=params_model,
                                    tx=tx)

    # Optionally, create an MLP to probe latent factors
    if FLAGS.probe_latent_factors:
        predictor = MLP(dims=[256, 20 * 5], batch_norm=True)
        tx_predictor = optax.chain(
            optax.clip_by_global_norm(FLAGS.max_grad_norm),
            optax.adam(FLAGS.lr, eps=1e-5),
        )

        z_obs = train_state.apply_fn(train_state.params,
                                     state,
                                     method=model.encode)
        params_predictor = predictor.init(key, z_obs)
        train_state_predictor = TrainState.create(apply_fn=predictor.apply,
                                                  params=params_predictor,
                                                  tx=tx_predictor)

    batch = Batch(
        discount=FLAGS.gamma,
        gae_lambda=FLAGS.gae_lambda,
        n_steps=FLAGS.n_steps + 1,
        num_envs=FLAGS.num_envs,
        n_actions=n_action,
        state_shape=env.observation_space.shape,
        latent_factors=FLAGS.add_latent_factors,
    )

    returns_train_buf = deque(maxlen=10)
    success_train_buf = deque(maxlen=10)
    factor_train_buf = deque(maxlen=10)

    returns_test_buf = deque(maxlen=10)
    success_test_buf = deque(maxlen=10)

    sample_episode_acc = [state[0]]

    for step in range(1, int(FLAGS.train_steps // FLAGS.num_envs + 1)):
        # Pick action according to PPO policy and update state
        action_test, _, _, _, key = select_action(
            train_state,
            state_test.astype(jnp.float32) / 255.0,
            latent_factors,
            key,
            sample=True,
        )
        state_test, _, _, test_infos = env_test.step(action_test)

        (train_state, state, latent_factors, batch, key, reward, done,
         train_infos) = get_transition(train_state, env, state, latent_factors,
                                       batch, key)

        # Save episode returns and success rate
        for info in train_infos:
            maybe_success = info.get("success")
            if maybe_success:
                success_train_buf.append(maybe_success)
            maybe_epinfo = info.get("returns")
            if maybe_epinfo:
                returns_train_buf.append(maybe_epinfo)

        for info in test_infos:
            maybe_success = info.get("success")
            if maybe_success:
                success_test_buf.append(maybe_success)
            maybe_epinfo = info.get("returns")
            if maybe_epinfo:
                returns_test_buf.append(maybe_epinfo)

        sample_episode_acc.append(state[0].copy())
        if done[0]:
            sample_episode_acc = []

        # Train once the batch is full
        if (step * FLAGS.num_envs) % (FLAGS.n_steps + 1) == 0:
            data = batch.get()
            metric_dict, train_state, key = update_ppo(
                train_state,
                data,
                FLAGS.num_envs,
                FLAGS.n_steps,
                FLAGS.n_minibatch,
                FLAGS.epoch_ppo,
                FLAGS.clip_eps,
                FLAGS.entropy_coeff,
                FLAGS.critic_coeff,
                key,
            )

            # Optionally, predict latent factors from observation
            if FLAGS.probe_latent_factors:
                X = train_state.apply_fn(
                    train_state.params,
                    jnp.stack(data[0]).reshape(-1, *data[0][0].shape[1:]),
                    method=model.encode,
                )
                y = jnp.stack(data[-1]).reshape(-1, *data[-1][0].shape[1:])

                grad_fn = jax.value_and_grad(mse_loss, has_aux=True)
                (total_loss, square_res), grads = grad_fn(
                    train_state_predictor.params,
                    train_state_predictor.apply_fn,
                    X=X,
                    y=y,
                )
                train_state_predictor = train_state_predictor.apply_gradients(
                    grads=grads)
                per_factor_error = square_res.reshape(-1, 20,
                                                      5).mean(2).mean(0)
                per_factor_error = per_factor_error.mean()
                factor_train_buf.append(per_factor_error)

            batch.reset()

            renamed_dict = {}
            for k, v in metric_dict.items():
                renamed_dict["metrics/%s" % k] = v
            wandb.log(renamed_dict, step=FLAGS.num_envs * step)

            wandb.log(
                {
                    "returns/eprew_train":
                    safe_mean([x for x in returns_train_buf]),
                    "returns/success_train":
                    safe_mean([x for x in success_train_buf]),
                    "returns/eprew_test":
                    safe_mean([x for x in returns_test_buf]),
                    "returns/success_test":
                    safe_mean([x for x in success_test_buf]),
                    "returns/per_factor_error":
                    safe_mean([x for x in factor_train_buf]),
                },
                step=FLAGS.num_envs * step,
            )
            print("[%d] Returns (train): %f Returns (test): %f" % (
                FLAGS.num_envs * step,
                safe_mean([x for x in returns_train_buf]),
                safe_mean([x for x in returns_test_buf]),
            ))

            # Optionally, log a GIF of the agent's trajectory during training

            # if FLAGS.log_episodes:
            #     sample_episode_acc = np.array(
            #         sample_episode_acc).transpose(0, 3, 1, 2)
            #     # import matplotlib.pyplot as plt
            #     # plt.imshow(sample_episode_acc[0].transpose(1,2,0))
            #     # plt.show()
            #     wandb.log(
            #         {
            #             "video":
            #             wandb.Video(
            #                 sample_episode_acc, fps=4, format="gif")
            #         },
            #         step=FLAGS.num_envs * step)

    # At the end of training, save model locally and on W&B
    model_dir = os.path.join(FLAGS.output_dir, run_name, "model_weights")
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)
    print("Saving model weights")
    checkpoints.save_checkpoint(
        ckpt_dir=model_dir,
        target=train_state,
        step=step * FLAGS.num_envs,
        overwrite=True,
        keep=1,
    )
    if FLAGS.wandb_mode != "disabled":
        artifact = wandb.Artifact(run_name, type="model_checkpoint")
        ckpt = glob.glob(model_dir + "/checkpoint_*")[0]
        artifact.add_file(ckpt)
        run.log_artifact(artifact)

    # Return performance metric for HP tuning
    try:
        run_logger = Run.get_context()
        run_logger.log("test_returns",
                       safe_mean([x for x in returns_train_buf]))
    except Exception as e:
        print("Failed to import AzureML")
        print(e)

    return 0


if __name__ == "__main__":
    app.run(main)
