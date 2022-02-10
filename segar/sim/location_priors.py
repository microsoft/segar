__copyright__ = "Copyright (c) Microsoft Corporation and Mila - Quebec AI " \
                "Institute"
__license__ = "MIT"
"""Noise distributions for locations in the sim.

"""


__all__ = ('RandomMiddleLocation', 'CenterLocation', 'RandomEdgeLocation',
           'RandomUniformLocation', 'RandomBottomLocation',
           'RandomRightLocation', 'RandomTopLocation',
           'RandomTopRightLocation', 'RandomBottomRightLocation',
           'RandomLeftLocation', 'RandomBottomLeftLocation',
           'RandomTopLeftLocation')

import random
from typing import Tuple, Union

import numpy as np
from scipy.stats import uniform

from segar.factors import Noise, Position


def _get_boundaries(min_distance: float = 0.1) -> Tuple[float, float]:
    """Adds a margin to boundaries if distances are enforced.

    """
    low, high = (-1., 1.)
    if min_distance:
        low += min_distance / 2.
        high -= min_distance / 2.
    return low, high


# These methods are for sampling from specific locations around the arena.
class Position2D(Noise[np.ndarray]):
    def __init__(self, lows: np.ndarray, highs: np.ndarray):
        self.lows = lows
        self.highs = highs
        self._dists = [uniform(lows[0], highs[0]), uniform(lows[1], highs[1])]
        super().__init__(params=dict(lows=lows, highs=highs))

    def cdf(self, samples: np.ndarray) -> Union[np.ndarray, None]:
        cdfs = [dist.cdf(samples[i]) for i, dist in enumerate(self._dists)]
        return np.prod(cdfs, axis=1)

    def log_cdf(self, samples: np.ndarray) -> Union[np.ndarray, None]:
        log_cdfs = [dist.logcdf(samples[i])
                    for i, dist in enumerate(self._dists)]
        return sum(log_cdfs)

    def pdf(self, samples: np.ndarray) -> Union[np.ndarray, None]:
        pdfs = [dist.pdf(samples[i]) for i, dist in enumerate(self._dists)]
        return np.prod(pdfs, axis=1)

    def log_pdf(self, samples: np.ndarray) -> Union[np.ndarray, None]:
        log_pdfs = [dist.logpdf(samples[i])
                    for i, dist in enumerate(self._dists)]
        return sum(log_pdfs)

    def sample(self) -> Position:
        return Position(np.random.uniform(self.lows, self.highs))


class RandomMiddleLocation(Position2D):
    def __init__(self):
        low, high = _get_boundaries()
        mid = (high + low) / 2.
        mid1 = (mid + low) / 2.
        mid2 = (high + mid) / 2.
        super().__init__(np.array([mid1, mid1]), np.array([mid2, mid2]))


class CenterLocation(Position2D):
    def __init__(self):
        low, high = _get_boundaries()
        mid = (high + low) / 2.
        super().__init__(np.array([mid, mid]), np.array([mid, mid]))


class RandomEdgeLocation(Noise[np.ndarray]):
    def __init__(self):
        self.low, self.high = _get_boundaries()
        mid = (self.high + self.low) / 2.
        self.q = (mid + self.low) / 2.
        super().__init__(params=dict(low=self.low, high=self.high, q=self.q))

    def cdf(self, samples: np.ndarray) -> Union[np.ndarray, None]:
        xs = samples[0]
        ys = samples[1]

        def _cdf(x):
            if x < self.low:
                return 0.
            elif self.low <= x < self.high - self.q:
                return uniform(self.low, self.q).cdf(x)
            elif self.high - self.q <= x:
                return uniform(self.high - self.q, self.high).cdf(x)
            else:
                raise ValueError

        xcdfs = np.array([map(_cdf, xs)])
        ycdfs = np.array([map(_cdf, ys)])
        return xcdfs * ycdfs

    def log_cdf(self, samples: np.ndarray) -> Union[np.ndarray, None]:
        return np.log(self.cdf(samples))

    def pdf(self, samples: np.ndarray) -> Union[np.ndarray, None]:
        xs = samples[0]
        ys = samples[1]

        def _pdf(x):
            if x < self.low:
                return 0.
            elif self.low <= x < self.high - self.q:
                return uniform(self.low, self.q).pdf(x)
            elif self.high - self.q <= x:
                return uniform(self.high - self.q, self.high).pdf(x)
            else:
                raise ValueError

        xpdfs = np.array([map(_pdf, xs)])
        ypdfs = np.array([map(_pdf, ys)])
        return xpdfs * ypdfs

    def log_pdf(self, samples: np.ndarray) -> Union[np.ndarray, None]:
        return np.log(self.pdf(samples))

    def sample(self) -> Position:
        # Bottom left "tile"
        pos = np.random.uniform(self.low, self.q, (2,))
        tile_size = (self.high - self.low) / 4.

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


class RandomLeftLocation(Position2D):
    def __init__(self):
        low, high = _get_boundaries()
        mid = (high + low) / 2.
        mid1 = (mid + low) / 2.
        super().__init__(np.array([low, low]), np.array([mid1, high]))


class RandomRightLocation(Position2D):
    def __init__(self):
        low, high = _get_boundaries()
        mid = (high + low) / 2.
        mid2 = (high + mid) / 2.
        super().__init__(np.array([mid2, low]), np.array([high, high]))


class RandomTopLocation(Position2D):
    def __init__(self):
        low, high = _get_boundaries()
        mid = (high + low) / 2.
        mid2 = (high + mid) / 2.
        super().__init__(np.array([low, mid2]), np.array([high, high]))


class RandomBottomLocation(Position2D):
    def __init__(self):
        low, high = _get_boundaries()
        mid = (high + low) / 2.
        mid1 = (mid + low) / 2.
        super().__init__(np.array([low, low]), np.array([high, mid1]))


class RandomTopRightLocation(Position2D):
    def __init__(self):
        low, high = _get_boundaries()
        mid = (high + low) / 2.
        mid2 = (high + mid) / 2.
        super().__init__(np.array([mid2, mid2]), np.array([high, high]))


class RandomTopLeftLocation(Position2D):
    def __init__(self):
        low, high = _get_boundaries()
        mid = (high + low) / 2.
        mid1 = (mid + low) / 2.
        mid2 = (high + mid) / 2.
        super().__init__(np.array([low, mid2]), np.array([mid1, high]))


class RandomBottomRightLocation(Position2D):
    def __init__(self):
        low, high = _get_boundaries()
        mid = (high + low) / 2.
        mid1 = (mid + low) / 2.
        mid2 = (high + mid) / 2.
        super().__init__(np.array([mid2, low]), np.array([high, mid1]))


class RandomBottomLeftLocation(Position2D):
    def __init__(self):
        low, high = _get_boundaries()
        mid = (high + low) / 2.
        mid1 = (mid + low) / 2.
        super().__init__(np.array([low, low]), np.array([mid1, mid1]))


class RandomUniformLocation(Position2D):
    def __init__(self):
        low, high = _get_boundaries()
        super().__init__(np.array([low, low]), np.array([high, high]))
