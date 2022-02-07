__author__ = "R Devon Hjelm"
__copyright__ = "Copyright (c) Microsoft Corporation and Mila - Quebec AI " \
                "Institute"
__license__ = "MIT"

"""For defining parameterizations for variations in the pixel observation
    space.

"""

import os

from segar.assets import ASSET_DIR
from segar.rendering.generators import (
    InceptionClusterGenerator, LinearAEGenerator)


INCEPTION_DIR = os.path.join(ASSET_DIR, 'inception_linear')
LINEAR_AE_DIR = os.path.join(ASSET_DIR, 'linear_ae')


_inception_paths = dict(
    baseline=os.path.join(INCEPTION_DIR, 'baseline', 'models'),
    close=os.path.join(INCEPTION_DIR, 'close', 'models'),
    far=os.path.join(INCEPTION_DIR, 'far', 'models')
)


_linear_ae_paths = dict(
    baseline=os.path.join(LINEAR_AE_DIR, 'one'),
    five=os.path.join(LINEAR_AE_DIR, 'five'),
    twenty=os.path.join(LINEAR_AE_DIR, 'twenty')
)


_config_sets = {
    'inception_linear': (_inception_paths, InceptionClusterGenerator),
    'linear_ae': (_linear_ae_paths, LinearAEGenerator)
}


def handle_config(config_set: str, dist_name: str = 'baseline'
                  ) -> dict:
    """Handle renderer configuration.

    :param config_set: Which configuration set.
    :param dist_name: Which distribution (name) to draw config from.
    :return: Configuration dictionary.
    """
    if dist_name is None:
        raise KeyError('`dist_name` must be specified for renderer keys.')
    model_paths, cls = _config_sets[config_set]
    model_path = model_paths[dist_name]
    config = dict(model_path=model_path, cls=cls)
    return config
