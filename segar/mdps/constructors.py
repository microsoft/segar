__copyright__ = "Copyright (c) Microsoft Corporation and Mila - Quebec AI Institute"
__license__ = "MIT"
"""Constructors for MDP objects using pickleable dictionaries.

"""


import logging
from typing import Union

from segar.sim import Simulator
from segar.mdps import (
    MDP,
    RGBObservation,
    ObjectStateObservation,
    MultimodalObservation,
)
from segar.tasks import (
    PuttPuttInitialization,
    PuttPutt,
    Invisiball,
    BilliardsInitialization,
    Billiards,
)


logger = logging.getLogger("segar")

obj_types = Union[
    MDP,
    Simulator,
    RGBObservation,
    ObjectStateObservation,
    MultimodalObservation,
    PuttPuttInitialization,
    PuttPutt,
    Invisiball,
    BilliardsInitialization,
    Billiards,
]


def class_constructor(c: str = None, **kwargs) -> obj_types:
    """Constructs a SEGAR MDP object or sub-object using str name and kwargs.

    :param c: Class name.
    :param kwargs: Initialization kwargs.
    :return: Class object.
    """
    try:
        C = eval(c)
    except NameError:
        raise NameError(f"Class `{c}` not defined within scope.")

    logger.debug(f"Constructing object of class `{c}`")

    args = []
    if "subs" in kwargs.keys():
        subs = kwargs.pop("subs")
        for sub in subs:
            logger.debug(f"`{c}` is composed of class `{sub}`")
            c_ = class_constructor(**sub)
            args.append(c_)

    try:
        out = C(*args, **kwargs)
    except Exception as e:
        logger.error(f"Construction of {c} failed.")
        raise e
    logger.debug(f"Construction of `{c}` was successful")
    return out


def mdp_constructor(
    task_config: dict = None,
    initialization_config: dict = None,
    mdp_config: dict = None,
    observation_config: dict = None,
    sim_config: dict = None,
):
    """Constructs the MDP using configs for each sub-component.

    :return: MDP object.
    """

    initialization_config = initialization_config or {}
    mdp_config = mdp_config or {}
    observation_config = observation_config or {}
    task_config = task_config or {}
    sim_config = sim_config or {}

    class_constructor(**sim_config)
    initialization = class_constructor(**initialization_config)
    observation = class_constructor(**observation_config)
    task_config["initialization"] = initialization
    task = class_constructor(**task_config)

    mdp = MDP(observation, task, **mdp_config)

    logger.debug("Construction of mdp successful.")

    return mdp


def from_config(
    task: type, init: type, obs: type, sim: type = Simulator, config: dict = None,
):
    """Construct MDP from types and configuration.

    :param task: Task class.
    :param init: Initialization class.
    :param obs: Observation class.
    :param sim: Simulator class.
    :param config: Configuration dictionary for initializations.
    :return: MDP
    """
    config = config or {}
    sim_cfg = config.pop("sim", {})
    sim(**sim_cfg)

    init_cfg = config.pop("init", {})
    init = init(**init_cfg)

    task_cfg = config.pop("task", {})
    task = task(init, **task_cfg)

    obs_cfg = config.pop("obs", {})
    obs = obs(**obs_cfg)

    return MDP(obs, task, **config)
