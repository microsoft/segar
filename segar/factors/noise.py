"""For generating random variables from distributions.

"""

from __future__ import annotations

__all__ = ('Noise', 'GaussianNoise', 'GaussianMixtureNoise', 'UniformNoise',
           'DiscreteRangeNoise', 'Choice', 'GaussianNoise2D')

import math
import random
from typing import Any, List, Tuple, TypeVar, Generic

import numpy as np
import numpy.random as nprandom

from segar.factors.factors import Factor
from segar.metrics import wasserstein_distance


T = TypeVar('T')


def w2_gaussian(m1: float, s1: float, m2: float, s2: float):
    return (m1 - m2) ** 2 + s1 + s2 - 2 * math.sqrt(s1 * s2)


def w2_sample(dist1: Noise[T], dist2: Noise[T], n_samples: int = 100000
              ) -> float:
    """For distributions on which the distance hasn't been implemented yet or
        doesn't have closed form, use samples.

    This comes with some variance, which reduces as n_samples becomes
    larger. It is better to have closed form, whenever possible.

    Note: This is slow and should only be used during analysis.

    :param dist1: First distribution.
    :param dist2: Second distribution.
    :param n_samples: Number of samples to draw from each distribution.
    :return: (float) Wasserstein-2 distance between distributions.
    """

    s1: List[T] = []
    s2: List[T] = []
    for _ in range(n_samples):
        s1.append(dist1.sample().value)
        s2.append(dist2.sample().value)

    x1 = np.array(s1).reshape(n_samples, 1)
    x2 = np.array(s2).reshape(n_samples, 1)

    return wasserstein_distance(x1, x2, p=2)


class Noise(Generic[T]):
    _protected_in_place = False

    def sample(self) -> Factor[T]:
        raise NotImplementedError

    def distance(self, other: Noise[T]) -> float:
        return w2_sample(self, other)

    def __repr__(self) -> str:
        return self.__class__.__name__


class GaussianNoise(Noise[float]):
    def __init__(self, mean: float = 0., std: float = 0.1,
                 clip: Tuple[float, float] = None):
        self.mean = mean
        self.std = std
        self.clip = clip
        super().__init__()

    def sample(self) -> Factor[float]:
        s = nprandom.normal(self.mean, self.std)
        if self.clip is not None:
            s = np.clip(s, *self.clip)
        return Factor[float](s)

    def distance(self, other: Noise[float]) -> float:
        if isinstance(other, GaussianNoise):
            return w2_gaussian(self.mean, self.std, other.mean, other.std)
        else:
            return super().distance(other)


class GaussianNoise2D(Noise[np.ndarray]):
    def __init__(self,
                 mean: np.ndarray = None,
                 std: np.ndarray = None,
                 clip: Tuple[float, float] = None):
        self.mean = mean or np.array([0., 0.])
        self.std = std or np.array([0.1, 0.1])
        self.clip = clip
        super().__init__()

    def sample(self) -> Factor[np.ndarray]:
        s = nprandom.normal(self.mean, self.std)
        if self.clip is not None:
            s = np.clip(s, *self.clip)
        return Factor[np.ndarray](s)

    def distance(self, other: Noise[float]) -> float:
        if isinstance(other, GaussianNoise):
            return w2_gaussian(self.mean, self.std, other.mean, other.std)
        else:
            return super().distance(other)


class UniformNoise(Noise[float]):
    def __init__(self, low: float = 0, high: float = 1):
        self.low = low
        self.high = high
        super().__init__()

    def sample(self) -> Factor[float]:
        s = random.uniform(self.low, self.high)
        return Factor[float](s)


class GaussianMixtureNoise(Noise[float]):
    def __init__(self, means: List[float] = (0., 0.),
                 stds: List[float] = (0.1, 0.1),
                 pvals: List[float] = None):
        assert len(means) == len(stds)

        self.pvals = pvals or [1. / len(means)] * len(means)
        self.components = [GaussianNoise(m, s) for m, s in zip(means, stds)]
        super().__init__()

    def sample(self) -> Factor[float]:
        c = random.choices(self.components, weights=self.pvals)[0]
        s = c.sample()
        return s


class DiscreteRangeNoise(Noise[int]):
    def __init__(self, low: int = 0, high: int = 10):
        assert low < high
        self.low = low
        self.high = high
        super().__init__()

    def sample(self) -> Factor[int]:
        s = random.choice(list(range(self.low, self.high + 1)))
        return Factor[int](s)


class Choice(Noise[T]):
    def __init__(self, keys: List[T], p: List[float] = None):

        if p is not None:
            if len(keys) != len(p):
                raise ValueError

        self.keys = keys
        self.pvals = p
        super().__init__()

    def sample(self) -> T:
        s = nprandom.choice(self.keys, p=self.pvals)
        return s

    def distance(self, other: Any):
        raise NotImplementedError('Choice needs special distance because '
                                  'values are special.')

    @staticmethod
    def _test_init():
        return Choice(['foo', 'bar'], p=[0.1, 0.9])
