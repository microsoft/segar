"""Configurations for initialization.

"""

from typing import Union

from rpp.sim.location_priors import RandomEdgeLocation, RandomMiddleLocation
from rpp.things import (Charger, Magnet, Bumper, Damper, Object,
                        ThingFactory, SandTile, MagmaTile, Hole, FireTile,
                        Tile)
from rpp.factors import (DiscreteRangeNoise, Position, Shape,
                         RandomConvexHull, Circle, Size, GaussianNoise,
                         GaussianMixtureNoise, Mass, Mobile, Charge,
                         Magnetism, Friction, UniformNoise)
from rpp.rules import Prior
from rpp.tasks.puttputt import puttputt_default_config, \
    puttputt_random_middle_config, invisiball_config
from rpp.tasks.billiards import billiards_default_config


default_config = {
    'numbers': [
        (ThingFactory([Charger, Magnet, Bumper, Damper, Object]),
         DiscreteRangeNoise(2, 3)),
        (ThingFactory({SandTile: 2 / 5., MagmaTile: 1 / 5., Hole: 1 / 5.,
                       FireTile: 1 / 5.}),
         DiscreteRangeNoise(1, 2))],

    'priors': [
        Prior(Position, RandomEdgeLocation(), entity_type=Object),
        Prior(Position, RandomMiddleLocation(), entity_type=Tile),

        Prior(Shape, RandomConvexHull(), entity_type=Tile),
        Prior(Shape, Circle(0.4), entity_type=Hole),
        Prior(Size, GaussianNoise(0.2, 0.01, clip=(0.1, 0.3)),
              entity_type=Object),
        Prior(Size, GaussianNoise(1.0, 0.01, clip=(0.5, 1.5)),
              entity_type=Tile),

        Prior(Mass, 1.0),
        Prior(Mobile, True),
        Prior(Charge, GaussianMixtureNoise(means=[-1., 1.], stds=[0.1, 0.1]),
              entity_type=Charger),
        Prior(Magnetism,
              GaussianMixtureNoise(means=[-1., 1.], stds=[0.1, 0.1]),
              entity_type=Magnet),
        Prior(Friction, UniformNoise(0.2, 1.0), entity_type=SandTile)
    ]
}


def handle_config(config_set: Union[str, None], dist_name: str = 'default'
                  ) -> dict:
    """Handles drawing initialization configuration from built-ins

    :param config_set: Configuration set.
    :param dist_name: Which distribution name.
    :return: The configuration dictionary.
    """
    if config_set is None:
        return default_config
    elif config_set == 'puttputt':
        if dist_name == 'default':
            return puttputt_default_config
        elif dist_name == 'random_middle':
            return puttputt_random_middle_config
    elif config_set == 'invisiball':
        return invisiball_config
    elif config_set == 'billiards':
        return billiards_default_config

    raise NotImplementedError(config_set, dist_name)
