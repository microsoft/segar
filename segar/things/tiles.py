"""Tiles.

"""

__all__ = ('Tile', 'Hole', 'SandTile', 'MagmaTile', 'FireTile')

from segar.factors import (Shape, Square, Circle, Label, Text, Order, Floor,
                         Friction, Heat, Consumes)
from .things import Thing


class Tile(Thing, default={Shape: Square(1.0), Label: 'tile', Text: 'L'}):
    """Tiles are things that objects can move onto.

    """

    _factor_types = Thing._factor_types + (Floor, Friction, Heat)


class Hole(Tile, default={Shape: Circle(1.0), Label: 'hole', Text: 'H',
                          Order: -1}):
    """Hole tile.

    This tile removes appropriately sized objects from the environment.
    Considered at the top of the tile order by default.

    """

    _factor_types = Tile._factor_types + (Consumes,)


class SandTile(Tile, default={Friction: 0.4, Shape: Square(1.0),
                              Label: 'sand', Text: 'S'}):
    """Special sand tile.

    """
    pass


class MagmaTile(Tile, default={Heat: 1.0, Friction: 0.1, Shape: Square(1.0),
                               Label: 'magma', Text: 'M'}):
    """Special magma tile.

    """
    pass


class FireTile(MagmaTile, default={Heat: 0.5, Label: 'fire', Text: 'F'}):
    """A less-intense version of magma. Has no friction by default.

    """
    pass
