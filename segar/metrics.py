__copyright__ = (
    "Copyright (c) Microsoft Corporation and Mila - Quebec AI Institute"
)
__license__ = "MIT"
"""Metrics used to compare environemnts, sets of factors, etc

"""

import numpy as np
import ot


def wasserstein_distance(
    x1: np.ndarray,
    x2: np.ndarray,
    p: int = 2,
    seed: int = 0,
    n_projections: int = 50,
) -> float:
    """Wasserstein distance.

    Uses euclidean metric.

    :param x1: First set of empirical samples (from P).
    :param x2: Second set of empirical samples (from Q).
    :param p: Order of metric, e.g., 2=W-2
    :param seed: Random seed to use.
    :param n_projections: Number of projections for slice wasserstein
        algorithm.
    :return: Estimated W-p distance.
    """
    n1 = x1.shape[0]
    n2 = x2.shape[0]
    a, b = np.ones((n1,)) / n1, np.ones((n2,)) / n2
    w = ot.sliced_wasserstein_distance(
        x1, x2, a, b, n_projections, seed=seed, p=p
    )
    return w
