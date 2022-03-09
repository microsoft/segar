__author__ = "Bogdan Mazoure"
__copyright__ = "Copyright (c) Microsoft Corporation and Mila - Quebec AI " \
                "Institute"
__license__ = "MIT"
"""Representation learning metrics (e.g. MINE).

"""

from segar.repl.models import SimpleMLP
from segar.repl.boilerplate import Trainer, data_iterator
from segar.repl.data_loaders import make_numpy_data_loaders
import numpy as np
import math

import torch
import torch.nn.functional as F

def MINE(model, opt, X_train, Z_train, X_test, Z_test, max_epochs=10):
    train, test, _ = make_numpy_data_loaders(
        X_train=np.concatenate([X_train, Z_train], axis=1),
        y_train=np.ones(shape=X_train.shape[0]),
        X_test=np.concatenate([X_test, Z_test], axis=1),
        y_test=np.ones(shape=X_test.shape[0]))
    
    def update_function(model, opt, joint, **kwargs):
        joint, labels = joint
        log_2 = math.log(2.)
        marginal = torch.gather(joint, 0, torch.rand(size=joint.shape).argsort(0))
        joint = model(joint)
        marginal = model(marginal)

        E_pos = log_2 - F.softplus(-joint)
        E_neg = F.softplus(-marginal) + marginal - log_2

        loss = E_neg.mean() - E_pos.mean()
        jsd_mi = 1. + joint.mean() - torch.exp(marginal).mean()
        results = {'jsd_loss': -loss.detach().cpu().item(),
                    'jsd_mi': jsd_mi.detach().cpu().item()}
        features = None

        opt.zero_grad()
        loss.backward()
        opt.step()
        return results, features

    def test_function(model, joint):
        joint, labels = joint
        log_2 = math.log(2.)
        marginal = torch.gather(joint, 0, torch.rand(size=joint.shape).argsort(0))
        joint = model(joint)
        marginal = model(marginal)

        E_pos = log_2 - F.softplus(-joint)
        E_neg = F.softplus(-marginal) + marginal - log_2

        loss = E_neg.mean() - E_pos.mean()
        jsd_mi = 1. + joint.mean() - torch.exp(marginal).mean()
        results = {'jsd_loss': -loss.detach().cpu().item(),
                    'jsd_mi': jsd_mi.detach().cpu().item()}
        return results

    yielder = lambda _DEVICE, inputs: [inp.to(_DEVICE) for inp in inputs]
    
    trainer = Trainer(model=model,
                      opt=opt,
                      train_loader=train,
                      test_loader=test,
                      data_iter=data_iterator(yielder),
                      update_func=update_function,
                      test_func=test_function,
                      max_epochs=max_epochs)
    train_metrics, test_metrics = trainer(log_wandb=False)
    
    return train_metrics, test_metrics
