
from typing import Any, Type, Union

from absl import app, flags
import numpy as np
import torch
from torch.utils.data import DataLoader
import wandb

from segar import get_sim
from segar.configs import get_env_config
from segar.logging import set_logger
from segar.factors import Charge, Magnetism, Mass, StoredEnergy, Density, Factor
from segar.mdps import (Initialization, Observation, RGBObservation,
                      StateObservation)
from segar.repl.boilerplate import (set_device, updater, data_iterator,
                                  Trainer, make_optimizer, get_device)
from segar.repl.data_loaders import create_initialization
from segar.repl.models import MLPRegressor
from segar.repl.static_datasets import IIDFromInit
from segar.sim import Simulator

from numpy_representation import NumpyRepresentation


_factors = [Charge, Mass, Magnetism, StoredEnergy, Density]


logger = set_logger(debug=True)
Simulator()



def create_iid_from_jax_model(initializer: Initialization,
                              input_observation: Observation,
                              target_observation: Observation,
                              n_observations: int = 10000,
                              batch_size=100) -> IIDFromInit:
    sim = get_sim()
    features = []
    targets = []
    np_model = NumpyRepresentation(n_action=2)

    for i in range(n_observations // batch_size):
        inps = []
        targs = []
        for b in range(batch_size):
            initializer.sample()
            initializer()
            input_observation.reset()
            inp = input_observation(sim.state) / 255.
            inps.append(inp[None, :, :, :])
            target = target_observation(sim.state)
            targs.append(target[None, :])

        target = np.concatenate(targs)
        inp = np.concatenate(inps)
        feature = np_model(inp)[0]
        feature = feature.copy()
        features += list(feature)
        targets += list(target)
    return IIDFromInit(features, targets)


def make_data_loaders(factors: list[Type[Factor]], batch_size: int = 64,
                      n_workers: int = 8
                      ) -> tuple[DataLoader, DataLoader, dict]:
    vis_config = get_env_config('visual', 'linear_ae', 'baseline')
    input_observation = RGBObservation(config=vis_config, resolution=64)
    target_observation = StateObservation('golfball', factors=factors)

    initialization = create_initialization()
    train_dataset = create_iid_from_jax_model(initialization,
                                              input_observation,
                                              target_observation,
                                              n_observations=5000)
    test_dataset = create_iid_from_jax_model(initialization, input_observation,
                                             target_observation,
                                             n_observations=5000)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size,
                              shuffle=True, pin_memory=True, drop_last=True,
                              num_workers=n_workers, sampler=None)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size,
                             shuffle=True, pin_memory=True, drop_last=True,
                             num_workers=n_workers, sampler=None)

    data_args = dict(feature_size=256)
    return train_loader, test_loader, data_args


def build_model(data_args: dict) -> torch.nn.Module:
    model = MLPRegressor(data_args['feature_size'], len(_factors))
    logger.info(f'Created model: {model}')

    model = model.to(get_device())
    return model


@data_iterator
def data_iter(device: Union[int, str],
              inputs: tuple[torch.Tensor, torch.Tensor]
              ) -> tuple[torch.Tensor, torch.Tensor]:

    images, targets = inputs
    images = images.to(device).float()
    targets = targets.to(device).float()

    return images, targets


@updater
def update(model: torch.nn.Module, inputs: tuple[torch.tensor, torch.tensor]
           ) -> tuple[torch.Tensor, dict[str, Any], None]:
    imgs, targets = inputs
    loss = model(imgs, targets)
    results = dict(loss=loss.item())
    return loss, results, None


def test(model: torch.nn.Module, inputs: torch.tensor) -> dict:
    errors = model.error(*inputs)
    results = dict((f'{k.__name__} error', v.item())
                   for k, v in zip(_factors, errors))
    return results


FLAGS = flags.FLAGS
flags.DEFINE_boolean('cpu', False, 'Whether to use the CPU for running.')
flags.DEFINE_integer('device', 0, 'Device to run on, overridden if cpu==True.')


def main(argv):
    logger.info(f'Running with arguments {argv}.')
    if FLAGS.cpu:
        set_device('cpu')
    else:
        set_device(FLAGS.device)

    logger.info('Making data.')
    train_loader, test_loader, data_args = make_data_loaders(_factors)
    logger.info('Building model.')
    regressor = build_model(data_args)
    opt = make_optimizer(regressor, learning_rate=1e-4)
    wandb.init(project='segar_rep_test')
    logger.info("Training.")
    trainer = Trainer(train_loader, regressor, opt,
                      data_iter, update, test, test_loader=test_loader)
    trainer()


if __name__ == '__main__':
    app.run(main)
