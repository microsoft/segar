"""Simple encoders.

"""

import logging
import math

import torch
import torch.nn.functional as nnF
import torch.nn as nn


logger = logging.getLogger('segar.repl.models')


def loss_xent(logits, labels, ignore_index=-1):
    fce = nn.functional.cross_entropy
    loss = fce(logits, labels, ignore_index=ignore_index)
    return loss


def loss_regression(features: torch.Tensor, targets: torch.Tensor
                    ) -> torch.Tensor:
    mse = nn.functional.mse_loss
    loss = mse(features, targets)
    return loss


class Detach(nn.Module):
    @staticmethod
    def forward(input_: torch.Tensor) -> torch.Tensor:
        return input_.detach()


class View(nn.Module):
    def __init__(self, *view):
        super(View, self).__init__()
        self.view = view

    def forward(self, input):
        return input.view(*self.view)


class Permute(nn.Module):
    def __init__(self, *perm):
        super(Permute, self).__init__()
        self.perm = perm

    def forward(self, input):
        return input.permute(*self.perm)


class Flatten(nn.Module):
    def __init__(self, args=None):
        super(Flatten, self).__init__()
        self.args = args

    def forward(self, input):
        return input.view(input.size(0), -1)


class Encoder(nn.Module):
    def __init__(self, nc=3, nf=64, input_size=64, norm_type=None,
                 pool_out=False, **encoder_args):
        super(Encoder, self).__init__()
        self.nc = nc
        self.nf = nf
        self.norm_type = norm_type
        self.input_size = input_size
        self.setup_encoder(norm_type, **encoder_args)
        self.pool_out = pool_out

        dummy_batch = torch.zeros((2, nc, input_size, input_size))
        self._config_modules(dummy_batch)

    @property
    def n_features(self):
        if self._n_features is None:
            raise RuntimeError('n_features not set')
        return self._n_features

    @property
    def n_locs(self):
        if self._n_locs is None:
            raise RuntimeError('n_locs not set')
        return self._n_locs

    @property
    def module(self):
        return self

    def init_weights(self, init_scale=1.):
        '''
        Run custom weight init for modules...
        '''
        for layer in self.layer_list:
            if isinstance(layer, ConvBlock):
                layer.init_weights(init_scale)

    def _forward_acts(self, x, detach_at=None, log=False):
        '''
        Return activations from all layers.
        '''
        # run forward pass through all layers
        layer_acts = []
        layer_in = x
        for i, layer in enumerate(self.layer_list):
            if log:
                logger.debug('Input size to layer: {}'.format(layer_in.size()))
            layer_out = layer(layer_in)
            if detach_at == i:
                layer_out = layer_out.detach()
            layer_acts.append(layer_out)
            layer_in = layer_out

        if self.pool_out:
            layer_acts.append(nnF.avg_pool2d(layer_out,
                                             layer_out.size(2), 1, 0))

        # remove input from the returned list of activations
        return layer_acts

    def _config_modules(self, x):
        '''
        Configure the modules for extracting fake rkhs embeddings for infomax.
        '''
        enc_acts = self._forward_acts(x, log=True)
        self.act_sizes = [a.size()[1:] for a in enc_acts]
        logger.info('Activation sizes for network: {}'.format(self.act_sizes))
        self._n_features = self.act_sizes[-1][0]
        self._n_locs = (self.act_sizes[-1][1], self.act_sizes[-1][2])

    def forward(self, x, **kwargs):
        '''
        Compute activations and Fake RKHS embeddings for the batch.
        '''
        # compute activations in all layers for x
        outs = self._forward_acts(x, **kwargs)
        return outs


class ConvBlock(nn.Module):
    def __init__(self, n_in, n_out, n_kern, n_stride, n_pad, norm_type=None,
                 bias=None, leaky=False):
        super(ConvBlock, self).__init__()

        self.norm_type = norm_type
        if bias is None:
            bias = (norm_type != 'batch')

        self.conv = nn.Conv2d(n_in, n_out, n_kern, n_stride, n_pad, bias=bias)

        if norm_type is None:
            self.norm = None
        elif norm_type == 'batch':
            self.norm = nn.BatchNorm2d(n_out)
        elif norm_type == 'layer':
            self.norm = nn.Sequential(
                Permute(0, 2, 3, 1),
                nn.LayerNorm(n_out),
                Permute(0, 3, 1, 2)
            )
        else:
            raise NotImplementedError(norm_type)

        if leaky:
            self.relu = nn.LeakyReLU(0.2, inplace=False)
        else:
            self.relu = nn.ReLU(inplace=False)

    def init_weights(self, init_scale=1.):
        # initialize first conv in res branch
        # -- rescale the default init for nn.Conv2d layers
        nn.init.kaiming_uniform_(self.conv.weight, a=math.sqrt(5))
        self.conv.weight.data.mul_(init_scale)

    def forward(self, x):
        x = self.conv(x)
        if self.norm is not None:
            x = self.norm(x)
        out = self.relu(x)
        return out


class ConvnetEncoder(Encoder):
    def setup_encoder(self, norm_type, n_out=None):
        nf = self.nf
        nc = self.nc
        n_out = n_out or (self.nf * 8)

        layers = []

        if self.input_size == 64:
            layers.append(ConvBlock(nc, nf, 7, 2, 1, norm_type=None,
                                    bias=False))
            layers.append(ConvBlock(nf, nf * 2, 4, 2, 1, norm_type=norm_type))
            layers.append(ConvBlock(nf * 2, nf * 4, 4, 2, 1,
                                    norm_type=norm_type))
            layers.append(ConvBlock(nf * 4, n_out, 4, 1, 0,
                                    norm_type=norm_type))
        else:
            raise NotImplementedError(self.input_size)

        self.layer_list = nn.ModuleList(layers)


def get_num_correct(lgt, lab):
    lgt = lgt.detach()
    lab = lab.cpu()
    max_lgt = torch.max(lgt.data, 1)[1].cpu()
    num_correct = (max_lgt == lab).sum().item()
    return num_correct


def get_accuracy(lgt, lab):
    num_correct = get_num_correct(lgt, lab)
    return num_correct / float(lgt.size(0))


class SimpleMLP(nn.Module):
    def __init__(self, n_input, n_out, n_hidden=512, p=0.1):
        super(SimpleMLP, self).__init__()
        self.n_input = n_input
        self.n_out = n_out
        self.n_hidden = n_hidden
        if n_hidden is None:
            # use linear classifier
            self.block_forward = nn.Sequential(
                Flatten(),
                nn.Dropout(p=p),
                nn.Linear(n_input, n_out, bias=True)
            )
        else:
            # use simple MLP classifier
            self.block_forward = nn.Sequential(
                Flatten(),
                nn.Dropout(p=p),
                nn.Linear(n_input, n_hidden, bias=False),
                nn.BatchNorm1d(n_hidden),
                nn.ReLU(inplace=True),
                nn.Dropout(p=p),
                nn.Linear(n_hidden, n_out, bias=False)
            )

    def forward(self, x):
        outs = self.block_forward(x)
        return outs


class MLPClassifier(SimpleMLP):
    def get_accuracies(self, features, labels):
        accuracies = dict((k, 100. * get_accuracy(lgt, labels))
                          for k, lgt in features.items())
        return accuracies


class MLPRegressor(SimpleMLP):
    def __init__(self, n_input, n_outs, n_hidden=128, p=0.1):
        super().__init__(n_input, n_outs, n_hidden=n_hidden,
                         p=p)
        self.loss_func = torch.nn.MSELoss()

    def error(self, x, targets):
        outputs = super().forward(x)
        return ((outputs - targets) ** 2).mean(0)

    def forward(self, x, targets):
        outputs = super().forward(x)
        loss = self.loss_func(outputs, targets)
        return loss


class ConvnetClassifier(ConvnetEncoder):
    def __init__(self, n_classes, n_hidden_class=512, dropout=0.1, **kwargs):
        super(ConvnetClassifier, self).__init__(**kwargs)

        n_input = self.n_features * self.n_locs[0] * self.n_locs[1]
        self.mlp = MLPClassifier(n_input, n_classes, n_hidden=n_hidden_class,
                                 p=dropout)

    def get_accuracies(self, *args, **kwargs):
        return self.mlp.get_accuracies(*args, **kwargs)

    def forward(self, x, labels):
        features = super(ConvnetClassifier, self).forward(x)[-1]
        lgt = self.mlp(features)
        loss = loss_xent(lgt, labels)
        return loss


class ConvnetRegressor(ConvnetEncoder):
    def __init__(self, n_outs, n_hidden_class=128, dropout=0.1,
                 **encoder_args):
        super().__init__(**encoder_args)
        n_input = self.n_features * self.n_locs[0] * self.n_locs[1]
        self.mlp = SimpleMLP(n_input, n_outs, n_hidden=n_hidden_class,
                             p=dropout)
        self.loss_func = torch.nn.MSELoss()

    def error(self, x, targets):
        features = super().forward(x)[-1]
        outputs = self.mlp(features)
        return ((outputs - targets) ** 2).mean(0)

    def forward(self, x, targets):
        features = super().forward(x)[-1]
        outputs = self.mlp(features)
        loss = self.loss_func(outputs, targets)
        return loss


class LocCondConvnetClassifier(ConvnetClassifier):
    _model_keys = ('nc', 'nf', 'input_size', 'norm_type', 'n_out', 'pool_out',
                   'n_hidden_class', 'dropout', 'n_hidden_loc')

    def __init__(self, n_hidden_loc=512, **kwargs):
        super(LocCondConvnetClassifier, self).__init__(**kwargs)
        n_input = self.n_features * self.n_locs[0] * self.n_locs[1]
        self.loc_encoder = SimpleMLP(2, n_input, n_hidden=n_hidden_loc, p=0.1)
        self.relu = nn.ReLU(inplace=False)
        self.bn = nn.BatchNorm1d(n_input)

    def forward(self, x, locs, labels):
        features = (super(ConvnetClassifier, self)
                    .forward(x)[-1].flatten(start_dim=1))
        loc_features = self.relu(self.bn(self.loc_encoder.forward(locs)))
        features += loc_features
        lgt = self.mlp(features)
        loss = loss_xent(lgt, labels)
        return loss
