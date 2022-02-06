"""For generating random variables from distributions.

"""

from __future__ import annotations

__all__ = ('Noise', 'GaussianNoise', 'GaussianMixtureNoise', 'UniformNoise',
           'DiscreteRangeNoise', 'Choice', 'GaussianNoise2D', 'Deterministic')

import math
import random
from typing import Any, Generic, List, Tuple, TypeVar, Union

import numpy as np
import numpy.random as nprandom
from scipy.stats import norm, multivariate_normal, uniform

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

    def __init__(self,
                 params: dict[str, Any],
                 dist: Union[norm, multivariate_normal, None] = None):
        self._params = params
        self._dist = dist
        super().__init__()

    def sample(self) -> Factor[T]:
        raise NotImplementedError('Not implemented for generic Noise class.')

    def w2_distance(self, other: Noise[T]) -> float:
        return w2_sample(self, other)

    def log_likelihood(self, samples: np.ndarray) -> float:
        return self.log_pdf(samples).sum()

    def empirical_entropy(self, samples: np.ndarray) -> float:
        log_likelihood = self.log_likelihood(samples)
        return -log_likelihood / float(samples.shape[0])

    def cdf(self, samples: np.ndarray) -> Union[np.ndarray, None]:
        if self._dist is None:
            return None
        return self._dist.cdf(samples)

    def log_cdf(self, samples: np.ndarray) -> Union[np.ndarray, None]:
        if self._dist is None:
            return None
        return self._dist.logcdf(samples)

    def pdf(self, samples: np.ndarray) -> Union[np.ndarray, None]:
        if self._dist is None:
            return None
        return self._dist.pdf(samples)

    def log_pdf(self, samples: np.ndarray) -> Union[np.ndarray, None]:
        if self._dist is None:
            return None
        return self._dist.logpdf(samples)

    def dist_info(self) -> dict:
        return dict(
            cls=self.__class__,
            dist=self._dist,
            params=self._params
        )

    @property
    def scipy_dist(self) -> Union[norm, multivariate_normal, None]:
        return self._dist

    def __repr__(self) -> str:
        return self.__class__.__name__


class Deterministic(Noise[T]):
    """Placeholder for deterministic noise.

    """
    def __init__(self, value: T):
        self.value = value
        super().__init__(dist=None, params=dict(value=value))

    def sample(self) -> Factor[T]:
        return Factor[T](self.value)

    def cdf(self, samples: np.ndarray) -> np.ndarray:
        return (samples >= self.value).astype(samples.dtype)

    def log_cdf(self, samples: np.ndarray) -> np.ndarray:
        return np.log(self.cdf(samples))

    def w2_distance(self, other: Noise[T]) -> float:
        return w2_sample(self, other)


class GaussianNoise(Noise[float]):
    def __init__(self, mean: float = 0., std: float = 0.1,
                 clip: Tuple[float, float] = None):
        self.mean = mean
        self.std = std
        self.clip = clip
        super().__init__(dist=norm(self.mean, self.std),
                         params=dict(mean=self.mean, std=self.std))

    def sample(self) -> Factor[float]:
        s = nprandom.normal(self.mean, self.std)
        if self.clip is not None:
            s = np.clip(s, *self.clip)
        return Factor[float](s)

    def w2_distance(self, other: Noise[float]) -> float:
        if isinstance(other, GaussianNoise):
            return w2_gaussian(self.mean, self.std, other.mean, other.std)
        else:
            return super().w2_distance(other)


class GaussianNoise2D(Noise[np.ndarray]):
    def __init__(self,
                 mean: np.ndarray = None,
                 std: np.ndarray = None,
                 cov: np.ndarray = None,
                 clip: Tuple[float, float] = None):
        self.mean: np.ndarray = mean or np.array([0., 0.])
        assert (std is None) or (cov is None)
        if std is None:
            self.cov = cov or np.array([[1.0, 0.0],
                                        [0.0, 1.0]])
        else:
            self.cov: np.ndarray = np.array([[std[0], 0.0],
                                             [0.0, std[1]]])
        self.clip = clip
        dist = multivariate_normal(self.mean, self.cov)
        super().__init__(dist=dist, params=dict(mean=self.mean, cov=self.cov))

    def sample(self) -> Factor[float]:
        s = nprandom.multivariate_normal(self.mean, self.cov)
        if self.clip is not None:
            s = np.clip(s, *self.clip)
        return Factor[np.ndarray](s)

    def w2_distance(self, other: Noise[float]) -> float:
        if isinstance(other, GaussianNoise):
            return w2_gaussian(self.mean, self.std, other.mean, other.std)
        else:
            return super().w2_distance(other)


class UniformNoise(Noise[float]):
    def __init__(self, low: float = 0, high: float = 1):
        self.low = low
        self.high = high
        dist = uniform(low, high)
        super().__init__(dist=dist, params=dict(low=low, high=high))

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
        self._dists = [norm(mean, std) for mean, std in zip(means, stds)]
        super().__init__(params=dict(means=means, stds=stds, pvals=pvals))

    def log_likelihood(self, samples: np.ndarray) -> float:
        return sum([c.log_likelihood(samples) for c in self.components])

    def cdf(self, samples: np.ndarray) -> np.ndarray:
        cdfs = [p * dist.cdf(samples)
                for p, dist in zip(self.pvals, self._dists)]
        return sum(cdfs)

    def log_cdf(self, samples: np.ndarray) -> np.ndarray:
        return np.log(self.cdf(samples))

    def pdf(self, samples: np.ndarray) -> np.ndarray:
        pdfs = [p * dist.pdf(samples)
                for p, dist in zip(self.pvals, self._dists)]
        return sum(pdfs)

    def log_pdf(self, samples: np.ndarray) -> Union[np.ndarray, None]:
        return np.log(self.pdf(samples))

    def sample(self) -> Factor[float]:
        c = random.choices(self.components, weights=self.pvals)[0]
        s = c.sample()
        return s


class DiscreteRangeNoise(Noise[int]):
    def __init__(self, low: int = 0, high: int = 10):
        assert low < high
        self.low = low
        self.high = high
        super().__init__(params=dict(values=list[range(low, high+1)]))

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
        super().__init__(params=dict((k, v) for k, v in zip(keys, p)))

    def sample(self) -> T:
        s = nprandom.choice(self.keys, p=self.pvals)
        return s

    def w2_distance(self, other: Any):
        raise NotImplementedError('Choice needs special distance because '
                                  'values are special.')

    @staticmethod
    def _test_init():
        return Choice(['foo', 'bar'], p=[0.1, 0.9])
