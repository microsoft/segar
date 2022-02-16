__copyright__ = "Copyright (c) Microsoft Corporation and Mila - Quebec AI Institute"
__license__ = "MIT"
"""Boundaries and such.

"""

__all__ = ["Wall", "SquareWall"]

from typing import Union

import numpy as np

from segar.factors import Circle, ConvexHullShape
from segar.things import Thing
from segar.parameters import WallDamping


class Wall:
    """ Wall object.

    This is to encapsulate interactions with the wall and to allow the wall
    to have more interesting interactions with objects.

    """

    def __init__(self, damping: Union[float, WallDamping] = None):
        damping = WallDamping(damping)
        self.damping = damping

    def render(self, visual_map: np.ndarray) -> None:
        pass

    def render_affordances(self, affordance_map):
        pass

    def state(self) -> dict:
        return self.__dict__


class SquareWall(Wall):
    def __init__(self, boundaries, damping=None):
        """

        :param boundaries: Wall boundaries.
        """
        super().__init__(damping=damping)
        self.boundaries = boundaries

    def overlaps_top_boundary(self, obj: Thing, thresh: float = 1e-7):
        shape = obj.Shape.value
        x = obj.Position
        b = self.boundaries
        if isinstance(shape, Circle):
            r = shape.radius
            return x[1] + r >= b[1] - thresh
        elif isinstance(shape, ConvexHullShape):
            points = shape.points
            max_y = points[0][1]
            for p in points:
                if p[1] > max_y:
                    max_y = p[1]
            return x[1] + max_y >= b[1] - thresh
        else:
            raise NotImplementedError

    def overlaps_bottom_boundary(self, obj: Thing, thresh: float = 1e-7):
        shape = obj.Shape.value
        x = obj.Position
        b = self.boundaries
        if isinstance(shape, Circle):
            r = shape.radius
            return x[1] - r <= b[0] + thresh
        elif isinstance(shape, ConvexHullShape):
            points = shape.points
            min_y = points[0][1]
            for p in points:
                if p[1] < min_y:
                    min_y = p[1]
            return x[1] + min_y <= b[0] + thresh
        else:
            raise NotImplementedError

    def overlaps_left_boundary(self, obj: Thing, thresh: float = 1e-7):
        shape = obj.Shape.value
        x = obj.Position
        b = self.boundaries
        if isinstance(shape, Circle):
            r = shape.radius
            return x[0] - r <= b[0] + thresh
        elif isinstance(shape, ConvexHullShape):
            points = shape.points
            min_x = points[0][0]
            for p in points:
                if p[0] < min_x:
                    min_x = p[0]
            return x[0] + min_x <= b[0] + thresh
        else:
            raise NotImplementedError

    def overlaps_right_boundary(self, obj: Thing, thresh: float = 1e-7):
        shape = obj.Shape.value
        x = obj.Position
        b = self.boundaries
        if isinstance(shape, Circle):
            r = shape.radius
            return x[0] + r >= b[1] - thresh
        elif isinstance(shape, ConvexHullShape):
            points = shape.points
            max_x = points[0][0]
            for p in points:
                if p[0] > max_x:
                    max_x = p[0]
            return x[0] + max_x >= b[1] - thresh
        else:
            raise NotImplementedError

    @property
    def state(self):
        return self.__dict__
