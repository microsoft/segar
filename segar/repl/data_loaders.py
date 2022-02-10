__copyright__ = (
    "Copyright (c) Microsoft Corporation and Mila - Quebec AI Institute"
)
__license__ = "MIT"
from typing import Optional, Type

from torch.utils.data import DataLoader

from segar.configs.handler import get_env_config
from segar.factors import (
    Charge,
    Magnetism,
    Mass,
    StoredEnergy,
    Density,
    Position,
    Shape,
    Circle,
    Mobile,
    GaussianNoise,
    RandomConvexHull,
    UniformNoise,
    Factor,
    GaussianMixtureNoise,
    Friction,
    Size,
)
from segar.mdps import RGBObservation, StateObservation, Initialization
from segar.rules import Prior
from segar.sim.location_priors import (
    RandomBottomLocation,
    RandomTopLocation,
    RandomMiddleLocation,
)
from segar.things import (
    Charger,
    Magnet,
    Bumper,
    Damper,
    Object,
    SandTile,
    MagmaTile,
    Hole,
    FireTile,
    Tile,
    ThingFactory,
)
from segar.repl.static_datasets.iid_samples import create_iid_from_init
from segar.tasks.puttputt import PuttPuttInitialization, GolfBall, GoalTile

import numpy as np

def create_initialization():
    """Creates a generic initialization that draws from the product of
        marginals over all factors.

    :return: Initialization object.
    """
    config = dict(
        numbers=[
            (
                ThingFactory(
                    [
                        Charger,
                        Magnet,
                        Bumper,
                        Damper,
                        Object,
                        SandTile,
                        MagmaTile,
                        Hole,
                        FireTile,
                    ]
                ),
                0,
            ),
            (GoalTile, 1),
            (GolfBall, 1),
        ],
        priors=[
            Prior(Position, RandomMiddleLocation()),
            Prior(Position, RandomBottomLocation(), entity_type=GolfBall),
            Prior(Position, RandomTopLocation(), entity_type=GoalTile),
            Prior(Shape, RandomConvexHull(0.3), entity_type=Tile),
            Prior(Shape, Circle(0.3), entity_type=GoalTile),
            Prior(
                Size,
                GaussianNoise(0.3, 0.01, clip=(0.1, 0.3)),
                entity_type=Object,
            ),
            Prior(
                Size,
                GaussianNoise(1.0, 0.01, clip=(0.5, 1.5)),
                entity_type=Tile,
            ),
            Prior(Mass, 1.0),
            Prior(Mobile, True),
            Prior(
                Charge,
                GaussianMixtureNoise(means=[-1.0, 1.0], stds=[0.1, 0.1]),
                entity_type=GolfBall,
            ),
            Prior(
                Magnetism,
                GaussianMixtureNoise(means=[-1.0, 1.0], stds=[0.1, 0.1]),
                entity_type=GolfBall,
            ),
            Prior(
                Density,
                GaussianNoise(1.0, 0.1, clip=(0.0, 2.0)),
                entity_type=GolfBall,
            ),
            Prior(Mass, GaussianNoise(2.0, 0.5), entity_type=GolfBall),
            Prior(StoredEnergy, GaussianNoise(0, 2.0), entity_type=GolfBall),
            Prior(Friction, UniformNoise(0.2, 1.0), entity_type=SandTile),
        ],
    )

    initialization = PuttPuttInitialization(config=config)
    return initialization


def make_data_loaders(
    factors: list[Type[Factor]],
    batch_size: int = 64,
    n_workers: int = 8,
    initialization: Optional[Initialization] = None,
) -> tuple[DataLoader, DataLoader, dict]:
    """Makes data loaders for initialization.

    :param factors: Factor types to track as ground truth.
    :param batch_size: Batch size for data loaders.
    :param n_workers: Number of workers for data loaders.
    :param initialization: Optional initialization object to generate ground
        truth from.
    :return:
    """
    vis_config = get_env_config("visual", "linear_ae", "baseline")
    input_observation = RGBObservation(config=vis_config, resolution=64)
    target_observation = StateObservation("golfball", factors=factors)

    initialization = initialization or create_initialization()
    train_dataset = create_iid_from_init(
        initialization,
        input_observation,
        target_observation,
        n_observations=1000,
    )
    test_dataset = create_iid_from_init(
        initialization,
        input_observation,
        target_observation,
        n_observations=1000,
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
        num_workers=n_workers,
        sampler=None,
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
        num_workers=n_workers,
        sampler=None,
    )

    data_args = dict(input_size=input_observation.resolution)
    return train_loader, test_loader, data_args


def make_numpy_data_loaders(
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        batch_size: int = 64,
        n_workers: int = 8) -> tuple[DataLoader, DataLoader, dict]:
    """Makes data loaders for initialization.

    :param X_train: NumPy array for train inputs.
    :param y_train: NumPy array for train labels.
    :param X_test: NumPy array for test inputs.
    :param y_test: NumPy array for test labels.
    :param batch_size: Batch size for data loaders.
    :param n_workers: Number of workers for data loaders.
    :return:
    """

    train_dataset = IIDFromInit(X_train, y_train)
    test_dataset = IIDFromInit(X_test, y_test)

    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              pin_memory=True,
                              drop_last=True,
                              num_workers=n_workers,
                              sampler=None)
    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=batch_size,
                             shuffle=True,
                             pin_memory=True,
                             drop_last=True,
                             num_workers=n_workers,
                             sampler=None)

    return train_loader, test_loader, {}
