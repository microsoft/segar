__copyright__ = "Copyright (c) Microsoft Corporation and Mila - Quebec AI Institute"
__license__ = "MIT"

"""Handler for passing parameterization that define different MDPs.

"""

__all__ = ("get_env_config",)

from .initialization import handle_config as init_handler
from .rendering import handle_config as visual_handler


def get_env_config(config_type: str, config_set: str, dist_name: str = None) -> dict:
    """Function for drawing some built-in configuration dictionaries

    :param config_type: Type of configuration to draw.
    :param config_set: Which set to draw from.
    :param dist_name: Name of the distribution within a set.
    :return: Configuration dictionary.
    """
    if config_type == "visual":
        return visual_handler(config_set, dist_name=dist_name)
    elif config_type == "initialization":
        return init_handler(config_set, dist_name=dist_name)

    raise NotImplementedError(config_type)
