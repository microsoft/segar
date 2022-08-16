__copyright__ = "Copyright (c) Microsoft Corporation and Mila - Quebec AI Institute"
__license__ = "MIT"
"""Runs all of the tests for CI.

"""

import logging
import unittest

import test_factors
import test_things
import test_rules
from test_env_wrappers import test as test_wrappers
from test_envs import test as test_env
from test_envs_from_config import test as test_env_config
from unit_tests import run_all_tests

# from test_trajectories import test as test_trajectory


factor_test = unittest.TestLoader().loadTestsFromModule(test_factors)
rule_test = unittest.TestLoader().loadTestsFromModule(test_rules)
thing_test = unittest.TestLoader().loadTestsFromModule(test_things)


_TASKS = ("puttputt", "invisiball", "billiards")
_OBSERVATIONS = ("rgb", "objstate", "tilestate", "allobjstate", "alltilestate")
_VIS_GENS = ("linear_ae",)


logger = logging.getLogger("tests")


if __name__ == "__main__":
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    logger.addHandler(ch)
    logger.info("Beginning simulator and MDP tests.")

    test_wrappers()

    # Test all combinations of task configs, observation state types,
    # and visual generative models. This only checks that they run without
    # errors.

    # Test functionality of modules.
    run_all_tests()

    print("Testing Factors, Things, and Rules")
    unittest.TextTestRunner(verbosity=2).run(factor_test)
    unittest.TextTestRunner(verbosity=2).run(thing_test)
    unittest.TextTestRunner(verbosity=2).run(rule_test)
    print("Done testing Factors, Things, and Rules")

    for task in _TASKS:
        for observations in _OBSERVATIONS:
            for vis_gen in _VIS_GENS:
                test_env(
                    task=task, observations=observations, vis_gen=vis_gen, show=False,
                )

    # Test running the mdp from config.
    for task in _TASKS:
        test_env_config(task)

    # Test trajectories are consistent.
    # TODO, using `test_trajectory`

    logger.info("All tests [PASSED]. Simulation and MDP looks good.")
