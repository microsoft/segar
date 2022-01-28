
import numpy as np
import ot

from rpp.factors.number_factors import NumericFactor
from rpp.factors.bools import BooleanFactor
from rpp.factors.arrays import VectorFactor
from rpp.metrics import wasserstein_distance


def task_set_init_dist(tasks1: list, tasks2: list) -> float:
    """Calculates the distance between initializations of two sets of tasks.

    This uses the w-2 distance, ensuring that sets of factors remain
    concordant within thing and task membership. Cost function is squared
    euclidean. This bring in assumptions about the underlying distribution,
    which may not reflect the true distribution from which factors are sampled.

    Note: if the tasks have not been sampled from, this will return 0.

    :param tasks1: First set of tasks (must have been sampled from).
    :param tasks2: Second set of tasks (must have been sampled from).
    :return: W-2 distance between tasks.
    """

    # First get the initializations from the tasks
    task_things1 = [task.initial_state for task in tasks1]
    task_things2 = [task.initial_state for task in tasks2]

    # Next we need representations that OT can handle. Because things
    # contain different sets of factors, we need to collect the ones that
    # have the same sets.

    thing_factor_sets = []
    for thing_list in task_things1 + task_things2:
        for thing in thing_list:
            factors = [f for f in thing.keys()
                       if issubclass(f, (NumericFactor, BooleanFactor,
                                         VectorFactor))]
            if factors not in thing_factor_sets:
                thing_factor_sets.append(factors)
    # We need to calculate the pairwise distances between all pairs of
    # tasks. This will be our final cost matrix.

    n1 = len(task_things1)
    n2 = len(task_things2)
    pairwise_distances = np.zeros((n1, n2))

    def calculate_distance(things_1, things_2, factor_set) -> float:
        things_1_ = [thing_ for thing_ in things_1 if all(
            thing_.has_factor(factor) for factor in factor_set)]
        things_2_ = [thing_ for thing_ in things_2 if all(
            thing_.has_factor(factor) for factor in factor_set)]

        if len(things_1_) == 0 and len(things_2_) == 0:
            return 0.

        if len(things_1_) == 0 or len(things_2_) == 0:
            return 100.  # Large number for empty sets

        s1 = np.concatenate([thing_.to_numpy(factor_set)[None, :]
                             for thing_ in things_1_])
        s2 = np.concatenate([thing_.to_numpy(factor_set)[None, :]
                             for thing_ in things_2_])

        return wasserstein_distance(s1, s2)

    # Next loop through the types of things
    for i, things1 in enumerate(task_things1):
        for j, things2 in enumerate(task_things2):
            distance = 0.
            for factor_set in thing_factor_sets:
                d = calculate_distance(things1, things2, factor_set)
                # Square distances as these are euclidean.
                distance += d ** 2
            pairwise_distances[i, j] = distance

    # The pairwise square distances are now the cost function for an OT
    # problem.
    w1 = np.ones((n1,)) / float(n1)
    w2 = np.ones((n2,)) / float(n2)
    return np.sqrt(ot.emd2(w1, w2, pairwise_distances, numItermax=100000))
