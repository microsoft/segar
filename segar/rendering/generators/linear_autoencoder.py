"""Module for training and loading Linear AE visual features.

The features trained available on the repo were constructed using:


"""
__all__ = ('LinearAEGenerator',)

import glob
import os
import random
from typing import Union

from absl import app, flags
import numpy as np
import torch
import torch.nn as nn
import torch.multiprocessing
from torch.utils.data import Dataset, DataLoader
import wandb

from segar.factors import (Heat, Friction, Charge, Magnetism, Density,
                           StoredEnergy, Mass, Alive)
from segar.logging import set_logger
from segar.rendering.generators.generator import Generator
from segar.repl.boilerplate import (make_optimizer, data_iterator, updater,
                                    Trainer, set_device, get_device)


torch.multiprocessing.set_sharing_strategy('file_system')
logger = set_logger(debug=True)


class RandomUniformFeatures(Dataset):
    def __init__(self, n_features: int):
        self.n_features = n_features

    def __len__(self) -> int:
        return 2 ** self.n_features

    def __getitem__(self, idx: int) -> np.ndarray:
        bits = np.unpackbits(np.array([idx], dtype='>i8').view(np.uint8))[
               -self.n_features:]
        bits = bits.astype(np.float64)
        mag = np.random.uniform(low=-1.0, high=1.0, size=bits.shape)
        return bits * mag


def make_data_loaders(n_features: int, batch_size: int = 64,
                      n_workers: int = 0) -> DataLoader:
    train_dataset = RandomUniformFeatures(n_features)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size,
                              shuffle=True, pin_memory=True, drop_last=True,
                              num_workers=n_workers, sampler=None)
    return train_loader


@data_iterator
def data_iter(device: Union[str, int], inputs: np.ndarray
              ) -> torch.Tensor:
    n_repeat = random.randint(3, 10) * 2
    features = inputs[:, :, None, None].repeat(1, 1, n_repeat, n_repeat)
    features = features.to(device).float()

    return features


class AutoEncoder(nn.Module):
    def __init__(self, n_inputs: int, n_features: int, kernel_size: int = 5,
                 stride: int = 1, l2_regularization: float = 0.01):
        super().__init__()
        self.n_inputs = n_inputs
        self.n_features = n_features
        self.l2_reg = l2_regularization

        self.convt = torch.nn.ConvTranspose2d(
            n_features, n_inputs, kernel_size, stride, 1)
        self.conv = torch.nn.Conv2d(
            n_inputs, n_features, kernel_size, stride, 1)

        self.mse = nn.functional.mse_loss

    def losses(self, x: torch.Tensor, features: torch.Tensor,
               out: torch.Tensor) -> torch.Tensor:

        loss_recon = self.mse(x, out)

        l2_reg = torch.tensor(0.)
        for param in self.parameters():
            l2_reg += torch.norm(param)
        loss_reg = self.l2_reg * l2_reg

        return loss_recon + loss_reg

    def get_errors(self, x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
        err = (x - out) ** 2
        err_feat = err.mean(2).mean(2).mean(0)
        return err_feat

    def make_images(self) -> torch.Tensor:
        inputs = torch.eye(self.n_features).float()[:, :, None, None]
        images = self.encode(inputs.repeat(1, 1, 6, 6))
        return images

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        features = self.convt(x)
        return torch.tanh(features)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.encode(x)

        out = self.conv(features)
        return features, out

    def init_weights(self, init_scale: float = 1.) -> None:
        '''
        Run custom weight init for modules...
        '''
        for layer in [self.convt, self.conv]:
            nn.init.kaiming_uniform_(layer.weight, a=np.sqrt(5))
            layer.weight.data.mul_(init_scale)


@updater
def update(model: torch.nn.Module, inputs: torch.Tensor
           ) -> tuple[torch.Tensor, dict, dict[str, torch.Tensor]]:
    x = inputs
    features, out = model(x)
    loss = model.losses(x, features, out)
    results = dict(loss=loss.item())
    features = dict(vis_features=features)
    return loss, results, features


def test(model: torch.nn.Module, inputs: torch.Tensor) -> dict:
    x = inputs
    features, out = model(x)
    loss = model.losses(x, features, out)
    errors = model.get_errors(x, out)
    results = dict(error=loss)
    factors = (Heat, Friction, Charge, Magnetism, Density, StoredEnergy,
               Mass, Alive)
    for i, factor in enumerate(factors):
        results[factor.__name__] = errors[i + 1].item()

    return results


def visualize(inputs: tuple[torch.tensor, torch.tensor],
              features: dict[str, torch.Tensor]) -> None:
    images = wandb.Image((features['vis_features'] + .5) * 255)
    wandb.log(dict(examples=images))


class LinearAEGenerator(Generator):
    def __init__(self, stride: int = 3, grayscale: bool = False,
                 kernel_size: int = 5, n_rand_factors: int = 3,
                 model_path: str = None, out_path: str = None,
                 dim_x: int = 64, dim_y: int = 64, seed: int = 0,
                 training_epochs: int = 500):

        n_factors = len(self._factors) + n_rand_factors + 1
        super().__init__(dim_x, dim_y, n_factors, grayscale=grayscale,
                         n_rand_factors=n_rand_factors, stride=stride,
                         model_path=model_path)

        model = AutoEncoder(self.n_channels, self.n_factors,
                            kernel_size=kernel_size, stride=stride)
        model.init_weights(1.0)
        self.model = model.to(get_device())
        if self.model_path is not None:
            self.model.load_state_dict(torch.load(self.model_path))
        else:
            self.train(training_epochs)
            if out_path is not None:
                if os.path.isdir(out_path):
                    out_path = os.path.join(out_path, f'weights_{seed}.pt')
                torch.save(self.model.state_dict(), out_path)

    def train(self, training_epochs: int) -> None:
        train_loader = make_data_loaders(self.n_factors)
        opt = make_optimizer(self.model, learning_rate=1e-5)
        # Note we do not have a separate test set as the train set is drawn
        # at runtime from the underlying distribution.
        trainer = Trainer(train_loader, self.model, opt,
                          data_iter, update, test, vis_func=visualize,
                          max_epochs=training_epochs)
        trainer()

    def gen_visual_features(self, factor_vec: torch.tensor,
                            dim_x: int = None, dim_y: int = None
                            ) -> Union[np.ndarray, None]:

        dim_x, dim_y = self.get_rendering_dims(dim_x, dim_y)

        if sum(factor_vec ** 2) == 0:
            return

        inputs = factor_vec[None, :, None, None].repeat(1, 1, dim_x, dim_y)
        vis_features = self.model.encode(inputs).detach().numpy()

        # Remove edges due to padding effects
        vis_features = vis_features[:, :, 1:-1, 1:-1]
        vis_features = vis_features[0].transpose(1, 2, 0)

        # Set to pixel values.  Features have tanh output range.
        vis_features = (255.0 * (vis_features + 1.) / 2.).astype('uint8')
        return vis_features

    @staticmethod
    def get_paths(dir_path: str) -> list[str]:
        model_paths = glob.glob(os.path.join(dir_path, '*.pt'))
        return model_paths


def set_seeds(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


FLAGS = flags.FLAGS


def main(argv):
    logger.info(f'Running with arguments {argv}.')
    set_device('cpu')
    wandb.init(project='segar_ae_features')
    seed = FLAGS.seed
    for n in range(FLAGS.n_models):
        set_seeds(seed)
        LinearAEGenerator(out_path=FLAGS.out_path,
                          model_path=FLAGS.pretrained_weights,
                          seed=seed)
        seed += 1


if __name__ == '__main__':
    flags.DEFINE_string('pretrained_weights', None,
                        'Optional weights to load.',
                        short_name='p')
    flags.DEFINE_string('out_path', None, 'Optional out path for trained '
                                          'weights.', short_name='o')
    flags.DEFINE_integer('seed', 0, 'Random seed.')
    flags.DEFINE_integer('n_models', 1, 'Number of models to generate')
    app.run(main)
