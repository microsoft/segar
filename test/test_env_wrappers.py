
import logging

import numpy as np

from segar.envs import SEGAREnv


logger = logging.getLogger(__name__)


def test_same_env(env1, env2):
    objs = ["golfball", "goal", "global_friction"]

    env1.env.envs[0].reset(0)
    env2.env.envs[0].reset(0)
    for obj in objs:
        factors_level1 = [
            x.value
            for x in env1.env.envs[0].env_list[0].env.env.sim.things[obj].factors.values()
            if type(x.value) == float or type(x.value) == np.ndarray
        ]
        factors_level2 = [
            x.value
            for x in env2.env.envs[0].env_list[0].env.env.sim.things[obj].factors.values()
            if type(x.value) == float or type(x.value) == np.ndarray
        ]
        names = np.array(
            [
                y
                for y, x in env1.env.envs[0].env_list[0].env.env.sim.things[obj].factors.items()
                if type(x.value) == float or type(x.value) == np.ndarray
            ]
        )
        is_different = False
        for f1, f2, name in zip(factors_level1, factors_level2, names):
            if not np.all(f1 == f2):
                logger.info(f"[{obj}] Factor `{name}` is different")
                is_different = True
        if is_different and obj == 'global_friction':
            raise ValueError(f"[{obj}] should be identical, but isn't")
        if not is_different and obj != 'global_friction':
            raise ValueError(f"[{obj}] All factors are identical")


def test_same_process(env1):
    objs = ["golfball", "goal", "global_friction"]
    env1.env.envs[0].reset(0)
    env1.env.envs[1].reset(0)
    for obj in objs:
        factors_level1 = [
            x.value
            for x in env1.env.envs[0].env_list[0].env.env.sim.things[obj].factors.values()
            if type(x.value) == float or type(x.value) == np.ndarray
        ]
        factors_level2 = [
            x.value
            for x in env1.env.envs[1].env_list[0].env.env.sim.things[obj].factors.values()
            if type(x.value) == float or type(x.value) == np.ndarray
        ]
        names = np.array(
            [
                y
                for y, x in env1.env.envs[0].env_list[0].env.env.sim.things[obj].factors.items()
                if type(x.value) == float or type(x.value) == np.ndarray
            ]
        )
        is_different = False
        for f1, f2, name in zip(factors_level1, factors_level2, names):
            if not np.all(f1 == f2):
                raise ValueError(f"[{obj}] Factor `{name}` is different")
        if not is_different:
            logger.info(f"[{obj}] All factors are identical")


def test_same_level(env1):
    objs = ["golfball", "goal", "global_friction"]
    for obj in objs:
        env1.env.envs[0].reset(0)
        factors_level1 = [
            x.value
            for x in env1.env.envs[0].env_list[0].env.env.sim.things[obj].factors.values()
            if type(x.value) == float or type(x.value) == np.ndarray
        ]

        env1.env.envs[0].reset(1)
        factors_level2 = [
            x.value
            for x in env1.env.envs[0].env_list[1].env.env.sim.things[obj].factors.values()
            if type(x.value) == float or type(x.value) == np.ndarray
        ]

        env1.env.envs[0].reset(0)
        factors_level1_2 = [
            x.value
            for x in env1.env.envs[0].env_list[0].env.env.sim.things[obj].factors.values()
            if type(x.value) == float or type(x.value) == np.ndarray
        ]
        names = np.array(
            [
                y
                for y, x in env1.env.envs[0].env_list[1].env.env.sim.things[obj].factors.items()
                if type(x.value) == float or type(x.value) == np.ndarray
            ]
        )

        is_different = is_different_lvl1 = False
        for f1, f2, f1_2, name in zip(factors_level1, factors_level2, factors_level1_2, names):
            if not np.all(f1 == f2):
                logger.info(f"[{obj}] Factor `{name}` is different")
                is_different = True
            if not np.all(f1 == f1_2):
                logger.info(f"[{obj}] Factor `{name}` is different for level 0")
                is_different_lvl1 = True
        if not is_different and obj != "global_friction":
            raise ValueError(f"[{obj}] All factors are identical")
        if is_different_lvl1:
            raise ValueError(f"[{obj}] All factors are not identical for level 1 and they should be")


def test():
    env_args = dict(
        resolution=64,
        max_steps=200
    )

    env1 = SEGAREnv(
        "puttputt-emptyx0-hard-rgb",
        num_envs=2,
        num_levels=5,
        framestack=1,
        seed=123,
        env_args=env_args
    )
    env2 = SEGAREnv(
        "puttputt-emptyx0-hard-rgb",
        num_envs=1,
        num_levels=5,
        framestack=1,
        seed=456,
        env_args=env_args
    )

    logger.info("===Checking whether 2 env objects are identical")
    test_same_env(env1, env2)
    logger.info("===Checking whether 2 levels are identical")
    test_same_level(env1)
    logger.info("===Checking whether 2 processes are identical")
    test_same_process(env1)


if __name__ == "__main__":
    test()
