"""SEGAR Objects.

Objects collide with other collidable objects, have mass, charge,
are mobile, etc.

"""

__all__ = ('Object', 'Magnet', 'Charger', 'Bumper', 'Damper', 'Ball')

from segar.factors import (Acceleration, Alive, Charge, Circle, Collides,
                           Density, Done, InfiniteEnergy, Label, Magnetism,
                           Mass, Mobile, Shape, StoredEnergy, Text, Velocity)
from .things import Thing


class Object(Thing, default={Shape: Circle(0.2), Label: 'object', Text: 'O'}):
    """Objects are things that can move and collide. They move over Tiles.

    """

    _factor_types = Thing._factor_types + (
        Collides, Mobile, Velocity, Mass, Charge, Density, Magnetism,
        StoredEnergy, InfiniteEnergy, Alive, Done, Mobile, Acceleration)


class Ball(Object, default={Shape: Circle(0.2),
                            Mobile: True,
                            Label: 'ball',
                            Text: 'B'}):
    pass


class Magnet(Object, default={Shape: Circle(0.2),
                              Magnetism: -1.0,
                              Mobile: False,
                              Label: 'magnet',
                              Text: 'T'}):
    pass


class Charger(Object, default={Charge: -1.0,
                               Mobile: False,
                               Shape: Circle(0.2),
                               Label: 'charger',
                               Text: 'Q'}):
    pass


class Bumper(Object, default={StoredEnergy: 1.0,
                              Mobile: False,
                              Shape: Circle(0.2),
                              Label: 'bumper',
                              Text: 'U'}):
    pass


class Damper(Object, default={StoredEnergy: -0.5,
                              Mobile: False,
                              Shape: Circle(0.2),
                              Label: 'damper',
                              Text: 'D'}):
    pass
