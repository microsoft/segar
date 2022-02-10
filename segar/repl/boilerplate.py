__copyright__ = "Copyright (c) Microsoft Corporation and Mila - Quebec AI " \
                "Institute"
__license__ = "MIT"
"""Boilerplate for training models easily with PyTorch.

"""

import time
from typing import Callable, Optional, Union

import torch
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from tqdm import tqdm
import wandb

from segar.utils import append_dict, average_dict


_DEVICE = 'cpu'


def set_device(dev: Union[str, int]) -> None:
    """Sets the device to train on.

    :param dev: Device ID.
    """
    global _DEVICE
    _DEVICE = dev


def get_device() -> Union[str, int]:
    """Fetches the device.

    :return: Device ID.
    """
    return _DEVICE


def updater(update_function: Callable) -> Callable:
    """Wrapper for an update function on model parameters.

    :param update_function: Update function that returns loss function,
        results, and features give model, input (data), and additional args.
    :return: Wrapped update function.
    """

    def update(model: torch.nn.Module, opt: Optimizer,
               inputs: tuple[torch.Tensor], **kwargs
               ) -> tuple[dict[str, float],
                          Union[dict[str, torch.Tensor], None]]:

        t0 = time.time()

        opt.zero_grad()
        loss, results, features = update_function(model, inputs, **kwargs)

        loss.backward()
        opt.step()

        t1 = time.time()
        update_time = t1 - t0
        results.update(**{'update time': update_time})
        return results, features

    return update


def make_optimizer(model: torch.nn.Module, learning_rate: float) -> Optimizer:
    """Creates an optimizer.

    Simple Adam optimizer.

    :param model: Model to optimize parameters on.
    :param learning_rate: Learning rate.
    :return: Optimizer.
    """
    opt = torch.optim.Adam(model.parameters(), lr=learning_rate,
                           betas=(0.8, 0.999), eps=1e-8)
    return opt


def data_iterator(yielder: Callable) -> Callable:
    """Wrapper for data iteration.

    :param yielder: Function that yields data.
    :return: Wrapped data iterator.
    """
    def data_iter_(loader: DataLoader, max_iters: int = None,
                   clear_memory: bool = False, desc: str = '',
                   pos: int = 0) -> None:

        n_batches = len(loader)
        if max_iters:
            early_stop = True
            max_iters = min(max_iters, n_batches)
        else:
            early_stop = False
            max_iters = n_batches

        bar = tqdm(total=max_iters, desc=desc, position=pos)

        for i, inputs in enumerate(loader):
            if early_stop and (i >= max_iters):
                break

            outs = yielder(_DEVICE, inputs)
            yield outs

            if bar is not None:
                bar.update(1)

            if clear_memory:
                del outs

        if bar is not None:
            bar.close()
        return

    return data_iter_


class Trainer:
    def __init__(self, train_loader: DataLoader, model: torch.nn.Module,
                 opt: Optimizer, data_iter: Callable, update_func: Callable,
                 test_func: Callable,
                 test_loader: Optional[DataLoader] = None,
                 vis_func: Optional[Callable] = None,
                 max_epochs: int = 100):
        """

        :param train_loader: Data loader for train dataset.
        :param model: Model to train on.
        :param opt: Optimizer.
        :param data_iter: Data iterator function that yeilds data.
        :param update_func: Update function for model parameters.
        :param test_func: Testing function that returns results.
        :param test_loader: Optional data loader for test test.
        :param vis_func: Function that generates visuals.
        :param max_epochs: Max number of epochs to train.
        """
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.model = model
        self.optim = opt
        self.data_iter = data_iter
        self.update_func = update_func
        self.test_func = test_func
        self.vis_func = vis_func
        self.epochs = 0
        self.max_epochs = max_epochs
        self.make_train_iter()

    def make_train_iter(self) -> None:
        """Makes a training iterator.

        """
        self.train_iter = self.data_iter(self.train_loader,
                                         desc=f'Training (epoch'
                                              f' {self.epochs})')

    def next_train(self) -> tuple[tuple[torch.Tensor], bool]:
        """Next training data.

        :return: Training data and whether the iterator has reset.
        """
        next_epoch = False
        try:
            inputs = next(self.train_iter)
        except StopIteration:
            inputs = None
            self.epochs += 1
            next_epoch = True
            self.make_train_iter()

        return inputs, next_epoch

    def test(self) -> tuple[dict, Union[dict, None]]:
        """Test the model using the training and test sets.

        :return: Training and test results.
        """
        self.model.eval()

        def tester(loader, msg=''):
            if loader is None:
                return None
            all_results = {}
            test_iter = self.data_iter(loader, desc=msg)
            while True:
                try:
                    inputs = next(test_iter)
                    results = self.test_func(self.model, inputs)
                    append_dict(all_results, results)
                except StopIteration:
                    break
            return average_dict(all_results)

        test_results = tester(self.test_loader, msg='Evaluating test')
        train_results = tester(self.train_loader, msg='Evaluating train')

        self.model.train()
        return train_results, test_results

    def __call__(self):
        """Main loop function.

        """
        self.model.train()
        last_inputs = None
        features = None
        while True:
            inputs, next_epoch = self.next_train()
            if next_epoch:
                train_results, test_results = self.test()
                for k in train_results.keys():
                    if test_results is None:
                        wandb.log({f'{k}/train': train_results[k]},
                                  step=self.epochs)
                    else:
                        wandb.log({f'{k}/train': train_results[k],
                                   f'{k}/test': test_results[k]},
                                  step=self.epochs)
                if self.vis_func is not None:
                    self.vis_func(last_inputs, features)
            else:
                results, features = self.update_func(
                    self.model, self.optim, inputs)
                last_inputs = inputs
            if self.epochs >= self.max_epochs:
                break
