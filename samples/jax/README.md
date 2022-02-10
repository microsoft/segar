
## Minimal PPO implementation for SEGAR (MiniPPO)

The SEGAR benchmark comes with standard RL baselines, most notably Proximal Policy Optimization (PPO).

SEGAR provides two implementations of PPO: a parallelized PPO built on top of Ray RLLIB (Pytorch), as well as a minmalistic one in Jax (MiniPPO).

## Running MiniPPO on SEGAR

To run MiniPPO on a sample RPP task, execute the following command:

```
python train_ppo.py --env_name="emptyx0-easy-rgb" --train_steps=1_000_000 \
--wandb_key=<YOUR_W&B_KEY>\
--wandb_entity=<YOUR_W&B_USERNAME_OR_TEAM>\
--wandb_project=<YOUR_W&B_PROJECT_NAME>\
--wandb_mode=online
```

The implementation is based on Procgen PPO code [here](https://github.com/bmazoure/ppo_jax).

The results are directly logged into a [Weights & Biases](https://wandb.ai/) dashboard (either online or offline), but the code is easily adaptable to use either tensorboard logging or a simple CSV file.

## SEGAR tasks
We have tested MiniPPO on the following task configuration:
```
# Empty rooms:

emptyx0-easy-rgb
emptyx0-medium-rgb
emptyx0-hard-rgb

# Rooms with objects:

objectsx<K>-easy-rgb
objectsx<K>-medium-rgb
objectsx<K>-hard-rgb

# Rooms with tiles:

tilesx<K>-easy-rgb
tilesx<K>-medium-rgb
tilesx<K>-hard-rgb
``` 
where `<K>` can be replaced by `1,2,3,..` to control for increasing number of entities.

It is important to restrict the action_range parameter to a small enough interval to prevent the agent from applying a force too large (similar to MuJoCo). To enforce this on PPO side, the diagonal Gaussian policy is clipped with a Tanh transform.

All tasks have the option to be parallelized using either `gym.AsyncVecEnv` or `gym.SyncVecEnv` by passing `--num_envs=<N>` for `<N>>1`.
This can considerably speed-up learning and is suggested (default is 64 parallel envs).

