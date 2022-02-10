from __future__ import annotations

__copyright__ = (
    "Copyright (c) Microsoft Corporation and Mila - Quebec AI Institute"
)
__license__ = "MIT"
"""Shape factors and shape objects.

"""

__all__ = (
    "Shape",
    "Circle",
    "Square",
    "RandomConvexHull",
    "Hexagon",
    "Triangle",
    "ConvexHullShape",
    "BaseShape",
)

from copy import deepcopy
import math
from typing import Any, Union, Optional

import numpy as np
from numpy.linalg import norm
from scipy.spatial.qhull import ConvexHull, Delaunay

from segar.factors.arrays import Position
from segar.factors.factors import Factor
from segar.factors.number_factors import Size


class BaseShape:
    def __init__(self, size: Union[float, Size]):
        if not isinstance(size, Size):
            size = Size(size)
        self._size = size

    @property
    def size(self) -> float:
        return self._size.value

    def set_size(self, size: Union[Size, float]):
        self._size = size

    def area(self) -> float:
        raise NotImplementedError

    def area_overlap(
        self, other_shape: BaseShape, normal_vector: np.ndarray
    ) -> float:
        if isinstance(self, Circle):
            if isinstance(other_shape, Circle):
                d = norm(normal_vector)
                r1 = self.radius
                r2 = other_shape.radius
                if d >= r1 + r2:
                    return 0.0

                elif (r1 <= r2) and d <= (r2 - r1):
                    return self.area()

                elif (r2 < r1) and d <= (r1 - r2):
                    return other_shape.area()

                else:
                    t1 = r1 ** 2 * math.acos(
                        (d ** 2 + r1 ** 2 - r2 ** 2) / (2 * d * r1)
                    )
                    t2 = r2 ** 2 * math.acos(
                        (d ** 2 + r2 ** 2 - r1 ** 2) / (2 * d * r2)
                    )
                    t3 = 0.5 * math.sqrt(
                        (-d + r1 + r2)
                        * (d + r1 - r2)
                        * (d - r1 + r2)
                        * (d + r1 + r2)
                    )
                    return t1 + t2 - t3
            elif isinstance(other_shape, ConvexHullShape):
                return other_shape.overlaps(self, -normal_vector)
            else:
                raise NotImplementedError(self, other_shape)
        elif isinstance(self, ConvexHullShape):
            if isinstance(other_shape, Circle):
                raise NotImplementedError(self, other_shape)
            elif isinstance(other_shape, ConvexHullShape):
                raise NotImplementedError(self, other_shape)
            else:
                raise NotImplementedError(self, other_shape)
        else:
            raise NotImplementedError(self)

    def overlaps(
        self,
        other_shape: BaseShape,
        normal_vector: np.ndarray,
        thresh: float = 1e-7,
    ) -> bool:

        if isinstance(self, Circle):
            if isinstance(other_shape, Circle):
                return norm(normal_vector) <= np.abs(
                    other_shape.radius + self.radius + thresh
                )
            elif isinstance(other_shape, ConvexHullShape):
                return other_shape.overlaps(self, -normal_vector)
            else:
                raise NotImplementedError(self, other_shape)
        elif isinstance(self, ConvexHullShape):
            if isinstance(other_shape, Circle):
                return self._tessellation.find_simplex(normal_vector) >= 0
            elif isinstance(other_shape, ConvexHullShape):
                raise NotImplementedError(self, other_shape)
            else:
                raise NotImplementedError(self, other_shape)
        else:
            raise NotImplementedError(self)

    def fix_overlap(
        self,
        other_shape: BaseShape,
        normal_vector: np.ndarray,
        unit_vector: Optional[np.ndarray] = None,
        thresh: float = 1e-7,
    ):
        """Fixes overlap between two shapes.

        :param other_shape: Shape of the other object.
        :param normal_vector: Normal vector from other thing shape location
            of this thing. This is for putting shapes in common frame of
            reference.
        :param unit_vector: Optional direction to fix overlap.
        :param thresh:
        :return: (np.array) Vector that fixes overlap.
        """
        dist: float = 0.0

        if not self.overlaps(other_shape, normal_vector, thresh=thresh):
            pass

        elif isinstance(self, Circle):
            if isinstance(other_shape, Circle):
                # If direction is not provided, fix parallel to normal vector.
                unit_norm = normal_vector / norm(normal_vector)
                if unit_vector is None:
                    dist = self.radius + other_shape.radius + thresh
                else:
                    # Negative unit_vector because we want to go backwards
                    cosangle = np.dot(unit_vector, -unit_norm)
                    r1 = self.radius
                    r2 = other_shape.radius
                    d = norm(normal_vector)
                    # Magnitude is solution to quadratic equation:
                    # x ** 2 + x * (-2 * d * cosang) + d ** 2 - (r1 + r2) ** 2
                    a = 1
                    b = -2 * d * cosangle
                    c = d ** 2 - (r1 + r2 + thresh) ** 2
                    dist = (-b + math.sqrt(b ** 2 - 4 * a * c)) / (
                        2 * a
                    ) + thresh
                    assert dist > 0.0

            elif isinstance(other_shape, ConvexHullShape):
                return other_shape.fix_overlap(
                    self, -normal_vector, thresh=thresh
                )

        elif isinstance(self, ConvexHullShape):
            raise NotImplementedError(type(self), ConvexHullShape)

        return dist

    @property
    def state(self):
        return dict(size=self._size, cl=self.__class__.__name__)

    def __copy__(self):
        shape = self.__class__(self._size)
        shape.__dict__.update(deepcopy(self.__dict__))
        return shape


class Circle(BaseShape):
    def __init__(self, size: Union[Size, float] = 1.0):
        """

        :param size: Size of the circle. Radius * 2
        """
        super().__init__(size)

    @property
    def size(self) -> Size:
        return self._size

    def area(self) -> float:
        return math.pi * self.radius ** 2

    @property
    def radius(self) -> float:
        if isinstance(self.size, Size):
            size = self.size.value
        else:
            size = self.size
        return size / 2.0

    def __repr__(self):
        return f"{self.__class__.__name__}(r={self.radius})"

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Circle):
            return False

        return self.size == other.size


class ConvexHullShape(BaseShape):
    """Convex hulls defined by a set of points.

    Covers a number of shapes, such as squares, triangles, etc. This is an
    abstract class and must be subclassed.

    """

    def __init__(self, size: Union[Size, float] = 1.0):
        super().__init__(size)
        points = self.get_initialization_point()
        self._tessellation = Delaunay(points)
        self._hull = ConvexHull(points)
        self._vertices = points[self._hull.vertices]

    def area(self) -> float:
        return self._hull.area

    def get_initialization_point(self) -> np.ndarray:
        """Gets the defining points of the hull.

        This must be overridden by subclasses.

        :return: np.array of 2d points.
        """
        raise NotImplementedError

    @property
    def points(self) -> np.ndarray:
        return self._vertices

    @property
    def state(self) -> dict:
        state = dict(vertices=self._vertices)
        state.update(**super().state)
        return state

    def __repr__(self):
        return f"{self.__class__.__name__}(p={self.points})"

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, self.__class__):
            return False

        return (
            np.allclose(self.points, other.points) and self.size == other.size
        )


class RandomConvexHull(ConvexHullShape):
    """Random convex hull determined by a 2d gaussian.

    """

    def get_initialization_point(self) -> np.ndarray:
        x_points = np.random.normal(0, self.size / 2.0, 100)
        y_points = np.random.normal(0, self.size / 2.0, 100)
        return np.array(list(zip(x_points, y_points)))


class Square(ConvexHullShape):
    def get_initialization_point(self) -> np.ndarray:
        x = self.size / 2
        points = [
            (-x, x),
            (x, -x),
            (-x, -x),
            (x, x),
        ]
        return np.array(points)


class Triangle(ConvexHullShape):
    def get_initialization_point(self) -> np.ndarray:
        d = self.size
        x = math.sqrt(3.0) * d / 2
        points = [
            (0, x / 2),
            (-d / 2, -x / 2),
            (d / 2, -x / 2),
        ]
        return np.array(points)


class Hexagon(ConvexHullShape):
    def get_initialization_point(self) -> np.ndarray:
        d = self.size
        x = math.sqrt(3.0) * d / 2
        points = [
            (-d / 2, 0.0),
            (-d / 4, -x / 2),
            (d / 4, -x / 2),
            (d / 2, 0.0),
            (d / 4, x / 2),
            (-d / 4, x / 2),
        ]
        return np.array(points)


class Shape(Factor[BaseShape]):
    """Abstract shape class.

    """

    def __init__(self, shape: BaseShape):
        size = shape.size
        if not isinstance(size, Size):
            size = Size(size)
            shape._size = size
        self._size = size
        super().__init__(shape)

    @property
    def size(self) -> Size:
        return self._size

    def set_size(self, size: Union[Size, float]):
        if isinstance(size, float):
            size = Size(size)
        self._size = size

    def area(self) -> float:
        return self.value.area()

    def area_overlap(
        self, other: Shape, normal_vector: Union[np.ndarray, Position]
    ) -> float:
        if isinstance(normal_vector, Position):
            normal_vector = normal_vector.value
        return self.value.area_overlap(
            other.value, normal_vector=normal_vector
        )

    def overlaps(
        self, other: Shape, normal_vector: Union[np.ndarray, Position]
    ) -> bool:
        if isinstance(normal_vector, Position):
            normal_vector = normal_vector.value
        return self.value.overlaps(other.value, normal_vector=normal_vector)

    def fix_overlap(
        self,
        other: Union[Shape, BaseShape],
        normal_vector: np.ndarray,
        unit_vector: Optional[np.ndarray] = None,
    ) -> float:
        other = self._get_value(other)
        return self.value.fix_overlap(
            other, normal_vector, unit_vector=unit_vector
        )

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Shape):
            return False

        return self.value == other.value

    def __hash__(self):
        return hash(id(self))
