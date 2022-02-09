"""Example module for regressing factors from visual features.

For showing how to use boilerplate for analysis.

"""

from typing import Any, Union

from absl import app, flags
import torch
import wandb

from segar.logging import set_logger
from segar.factors import Charge, Magnetism, Mass, StoredEnergy, Density
from segar.sim import Simulator
from segar.repl.boilerplate import (set_device, updater, data_iterator,
                                    Trainer, make_optimizer, get_device)
from segar.repl.data_loaders import make_data_loaders
from segar.repl.models import ConvnetRegressor


_factors = [Charge, Mass, Magnetism, StoredEnergy, Density]


logger = set_logger(debug=True)
Simulator()


def build_model(data_args: dict) -> torch.nn.Module:
    model = ConvnetRegressor(len(_factors), input_size=data_args['input_size'],
                             nc=3, norm_type='batch', pool_out=True)
    logger.info(f'Created model: {model}')

    model.init_weights(1.0)

    model = model.to(get_device())
    return model


@data_iterator
def data_iter(device: Union[int, str],
              inputs: tuple[torch.Tensor, torch.Tensor]
              ) -> tuple[torch.Tensor, torch.Tensor]:

    images, targets = inputs
    images = 2. * images.to(device).float() / 255. - 1.
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


def visualize(inputs: tuple[torch.tensor, torch.tensor],
              features: None) -> None:
    images = wandb.Image((inputs[0] + .5) * 255)
    wandb.log(dict(examples=images))


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
    wandb.init(project='segar_regression_test')
    logger.info("Training.")
    trainer = Trainer(train_loader, regressor, opt,
                      data_iter, update, test, test_loader=test_loader,
                      vis_func=visualize)
    trainer()


if __name__ == '__main__':
    app.run(main)
