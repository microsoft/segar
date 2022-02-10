from flax.training.train_state import TrainState
from flax.training import checkpoints
import jax.numpy as jnp
import jax
import optax
import numpy as np

from models import TwinHeadModel


class NumpyRepresentation:
    """
    A simple interface between pre-trained PPO policies (in Jax) and any other 
    framework, e.g. PyTorch, Tensorflow, etc. Converts all vectors to NumPy.
    """
    def __init__(self,
                 n_action: int,
                 resolution: int = 64,
                 model_dir: str = './model_weights'):
        self.model_ppo = TwinHeadModel(action_dim=n_action,
                                       prefix_critic='vfunction',
                                       prefix_actor="policy",
                                       action_scale=1.)
        key = jax.random.PRNGKey(123)
        state = jnp.ones(shape=(1, resolution, resolution, 3),
                         dtype=jnp.float32)

        tx = optax.chain(optax.clip_by_global_norm(2),
                         optax.adam(3e-4, eps=1e-5))
        params_model = self.model_ppo.init(key, state)
        train_state_ppo = TrainState.create(apply_fn=self.model_ppo.apply,
                                            params=params_model,
                                            tx=tx)

        self.train_state = checkpoints.restore_checkpoint(
            './%s' % model_dir, target=train_state_ppo)

    def __call__(self,
                 x_: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Input:
            x_: n_batch x resolution x resolution x 3, float32 array [0.,1.]
        Output:
            representation: n_batch x n_d
            pi_logits: n_batch x n_actions
            v: n_batch
        """
        x_ = jnp.array(x_)
        v, pi = self.train_state.apply_fn(self.train_state.params, x_)
        z = self.train_state.apply_fn(self.train_state.params,
                                      x_,
                                      method=self.model_ppo.encode)
        return np.array(z), np.array(pi.distribution.loc), np.array(v)


if __name__ == '__main__':
    np_model = NumpyRepresentation(n_action=2)
    obs = jnp.ones(shape=(64, 64, 3))
    z, _, _ = np_model(obs)
    print(z.shape)
