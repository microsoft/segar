__copyright__ = "Copyright (c) Microsoft Corporation and Mila - Quebec AI " \
                "Institute"
__license__ = "MIT"
"""Tests spawning environments from config dictionaries.

"""

import logging
import traceback

import numpy as np

from segar.mdps.constructors import mdp_constructor
from segar.configs.handler import get_env_config
from segar.tasks.billiards import billiards_default_config
from segar.tasks.puttputt import puttputt_default_config, \
    invisiball_config


_TASKS = ('puttputt', 'invisiball', 'billiards')
_VIS_GEN = 'linear_ae'

logger = logging.getLogger('tests.test_env_config')


def test(task=None, visual_dist='baseline', iterations=100):
    logger.info(f'Testing environment from config with variables task={task}.')

    if task == 'puttputt':
        initialization = 'PuttPuttInitialization'
        init_config = puttputt_default_config
        task_ = 'PuttPutt'

    elif task == 'invisiball':
        initialization = 'PuttPuttInitialization'
        init_config = invisiball_config
        task_ = 'Invisiball'

    elif task == 'billiards':
        initialization = 'BilliardsInitialization'
        init_config = billiards_default_config
        task_ = 'Billiards'

    visual_config = get_env_config(
        'visual', _VIS_GEN, dist_name=visual_dist)

    config = dict(
        initialization_config=dict(
            c=initialization,
            config=init_config
        ),
        observation_config=dict(
            c='RGBObservation',
            config=visual_config,
            resolution=256
        ),
        sim_config=dict(
            c='Simulator'
        ),
        task_config=dict(
            c=task_
        ),
        mdp_config=dict(
            max_steps_per_episode=iterations,
            episodes_per_arena=1,
            sub_steps=1
        )
    )

    try:
        mdp = mdp_constructor(**config)
        mdp.reset()
        for _ in range(iterations):
            mdp.step(np.array([1., 1.]))
    except Exception:
        traceback.print_exc()
        logger.error('[FAILED]')
        raise ValueError(f'Testing environment from config with variables '
                         f'task={task} FAILED.')
    logger.info('[PASSED]')


if __name__ == '__main__':
    for task in _TASKS:
        test(task=task)

    logger.info('Successfully stepped through MDP envs from config.')
