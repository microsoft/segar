"""Noise distributions for locations in the sim.

"""


__all__ = ('RandomMiddleLocation', 'CenterLocation', 'RandomEdgeLocation',
           'RandomUniformLocation', 'RandomBottomLocation',
           'RandomRightLocation', 'RandomTopLocation',
           'RandomTopRightLocation', 'RandomBottomRightLocation',
           'RandomLeftLocation', 'RandomBottomLeftLocation',
           'RandomTopLeftLocation')

import random
from typing import Tuple

import numpy as np

from segar import get_sim
from segar.factors import Noise, Position


def _get_boundaries(min_distance: float = 0.1) -> Tuple[float, float]:
    """Adds a margin to boundaries if distances are enforced.

    """
    sim = get_sim()
    low, high = sim.boundaries
    if min_distance:
        low += min_distance / 2.
        high -= min_distance / 2.
    return low, high


# These methods are for sampling from specific locations around the arena.
class RandomMiddleLocation(Noise[np.ndarray]):
    def sample(self) -> Position:
        low, high = _get_boundaries()
        mid = (high + low) / 2.
        mid1 = (mid + low) / 2.
        mid2 = (high + mid) / 2.
        return np.random.uniform(mid1, mid2, (2,))


class CenterLocation(Noise[np.ndarray]):
    def sample(self) -> Position:
        low, high = _get_boundaries()
        mid = (high + low) / 2.
        return Position(np.array([mid, mid]))


class RandomEdgeLocation(Noise[np.ndarray]):
    def sample(self) -> Position:
        low, high = _get_boundaries()
        mid = (high + low) / 2.
        q = (mid + low) / 2.

        # Bottom left "tile"
        pos = np.random.uniform(low, q, (2,))
        tile_size = (high - low) / 4.

        c = random.randint(0, 11)
        if c == 0:
            pass
        elif c == 1:
            pos[0] += tile_size
        elif c == 2:
            pos[0] += 2 * tile_size
        elif c == 3:
            pos[0] += 3 * tile_size
        elif c == 4:
            pos[1] += tile_size
        elif c == 5:
            pos[0] += 3 * tile_size
            pos[1] += tile_size
        elif c == 6:
            pos[1] += 2 * tile_size
        elif c == 7:
            pos[0] += 3 * tile_size
            pos[1] += 2 * tile_size
        elif c == 8:
            pos[1] += 3 * tile_size
        elif c == 9:
            pos[0] += tile_size
            pos[1] += 3 * tile_size
        elif c == 10:
            pos[0] += 2 * tile_size
            pos[1] += 3 * tile_size
        elif c == 11:
            pos[0] += 3 * tile_size
            pos[1] += 3 * tile_size
        return Position(pos)


class RandomLeftLocation(Noise[np.ndarray]):
    def sample(self) -> Position:
        low, high = _get_boundaries()
        mid = (high + low) / 2.
        mid1 = (mid + low) / 2.
        return Position(np.random.uniform(low, [mid1, high], (2,)))


class RandomRightLocation(Noise[np.ndarray]):
    def sample(self) -> Position:
        low, high = _get_boundaries()
        mid = (high + low) / 2.
        mid2 = (high + mid) / 2.
        return Position(np.random.uniform([mid2, low], high, (2,)))


class RandomTopLocation(Noise[np.ndarray]):
    def sample(self) -> Position:
        low, high = _get_boundaries()
        mid = (high + low) / 2.
        mid2 = (high + mid) / 2.
        return Position(np.random.uniform([low, mid2], high, (2,)))


class RandomBottomLocation(Noise[np.ndarray]):
    def sample(self) -> Position:
        low, high = _get_boundaries()
        mid = (high + low) / 2.
        mid1 = (mid + low) / 2.
        return Position(np.random.uniform(low, [high, mid1], (2,)))


class RandomTopRightLocation(Noise[np.ndarray]):
    def sample(self) -> Position:
        low, high = _get_boundaries()
        mid = (high + low) / 2.
        mid2 = (high + mid) / 2.
        return Position(np.random.uniform(mid2, high, (2,)))


class RandomTopLeftLocation(Noise[np.ndarray]):
    def sample(self) -> Position:
        low, high = _get_boundaries()
        mid = (high + low) / 2.
        mid1 = (mid + low) / 2.
        mid2 = (high + mid) / 2.
        return Position(np.random.uniform([low, mid2], [mid1, high], (2,)))


class RandomBottomRightLocation(Noise[np.ndarray]):
    def sample(self) -> Position:
        low, high = _get_boundaries()
        mid = (high + low) / 2.
        mid1 = (mid + low) / 2.
        mid2 = (high + mid) / 2.
        return Position(np.random.uniform([mid2, low], [high, mid1], (2,)))


class RandomBottomLeftLocation(Noise[np.ndarray]):
    def sample(self) -> Position:
        low, high = _get_boundaries()
        mid = (high + low) / 2.
        mid1 = (mid + low) / 2.
        return Position(np.random.uniform(low, mid1, (2,)))


class RandomUniformLocation(Noise[np.ndarray]):
    def sample(self) -> Position:
        low, high = _get_boundaries()
        return Position(np.random.uniform(low, high, (2,)))
