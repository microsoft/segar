__copyright__ = "Copyright (c) Microsoft Corporation and Mila - Quebec AI Institute"
__license__ = "MIT"
"""Abstract class for rendering generators.

"""

from typing import Optional, Type, Union

import numpy as np
import torch

from segar.factors import (
    Heat,
    Friction,
    Charge,
    Magnetism,
    Density,
    StoredEnergy,
    Mass,
    Alive,
    Factor,
)


class Generator:
    _factors = (
        Heat,
        Friction,
        Charge,
        Magnetism,
        Density,
        StoredEnergy,
        Mass,
        Alive,
    )

    def __init__(
        self,
        dim_x: int,
        dim_y: int,
        n_factors: int,
        stride: int = 1,
        grayscale: bool = False,
        n_rand_factors: int = 3,
        model_path: Optional[str] = None,
    ):

        n_channels = 1 if grayscale else 3

        self.dim_x = dim_x
        self.dim_y = dim_y
        self.stride = stride
        self.dim_x_with_stride = (dim_x - 1) // self.stride + 1
        self.dim_y_with_stride = (dim_y - 1) // self.stride + 1
        self.n_channels = n_channels
        self.n_rand_factors = n_rand_factors
        self.n_factors = n_factors
        self.model_path = model_path

    def gen_visual_features(
        self, factor_vec: torch.tensor, dim_x: int = None, dim_y: int = None
    ) -> Union[np.ndarray, None]:
        """Generates the visual features.

        Note: Must be implemented in subclasses.

        :param factor_vec: Vector of factors.
        :param dim_x: Size in the first dimension.
        :param dim_y: Size in the second dimension.
        :return: Visual features.
        """
        raise NotImplementedError

    def get_background(self, dim_x: int, dim_y: int) -> np.ndarray:
        """Generate the visual features corresponding to the background.

        Background occupies the first visual feature.

        :param dim_x: Size in the first dimension.
        :param dim_y: Size in the second dimension.
        :return: Visual feature of the background.
        """
        affordance_vec = torch.zeros(self.n_factors)
        affordance_vec[0] = 1.0
        return self.gen_visual_features(affordance_vec, dim_x=dim_x, dim_y=dim_y)

    def get_pattern(self, factor_dict: dict[Type[Factor], Factor]) -> Union[np.ndarray, None]:
        """Generates a pattern for the corresponding set of factors.

        Note: this is the main function exposed to the renderer.

        :param factor_dict: Dictionary of factor types and values.
        :return: Visual feature corresponding to the set of factors.
        """

        factor_list = [0.0]  # First index is background
        for factor_type in self._factors:
            factor = factor_dict.get(factor_type, 0.0)
            if isinstance(factor, Factor):
                factor = factor.value
            factor_list.append(factor)

        factor_array = np.array(factor_list)

        random_factors = np.clip(np.random.normal(0.0, 1.0, size=(self.n_rand_factors,)), -1, 1)
        factor_array = np.append(factor_array, random_factors)

        pattern = self.gen_visual_features(torch.tensor(factor_array).float())
        if pattern is not None:
            pattern = np.zeros((self.dim_y, self.dim_x, self.n_channels), dtype=np.uint8) + pattern

        return pattern

    def get_rendering_dims(
        self, dim_x: Optional[int] = None, dim_y: Optional[int] = None
    ) -> tuple[int, int]:
        """When generating from a convolutional model, need to resize input.

        :param dim_x: Size in the first dimension.
        :param dim_y: Size in the second dimension.
        :return: Resized dimensions according to convolutional stride.
        """
        if dim_x is None:
            dim_x = self.dim_x_with_stride
        else:
            dim_x = (dim_x - 1) // self.stride + 1

        if dim_y is None:
            dim_y = self.dim_y_with_stride
        else:
            dim_y = (dim_y - 1) // self.stride + 1

        return dim_x, dim_y

    @staticmethod
    def get_paths(dir_path: str) -> list[str]:
        """For reloading models from a directory/

        :param dir_path: Path to the directory.
        :return: List of paths to models to load.
        """
        raise NotImplementedError
