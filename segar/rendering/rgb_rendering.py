__copyright__ = "Copyright (c) Microsoft Corporation and Mila - Quebec AI Institute"
__license__ = "MIT"
"""Rendering of RGB images

"""

__all__ = ("RGBRenderer", "RGBTextureRenderer", "register_rule", "register_color")

from typing import Optional, Type, Union, Callable

import numpy as np
import os
import random

from segar.things import Thing, Wall
from segar.factors import Circle, ConvexHullShape, Factor, Mobile, Color

from .generators import Generator
from .rendering import Visual, Renderer, WallVisual, CircleVisual, PolyVisual


def _get_default_color(thing: Thing):
    """Default colors for things.

    :param thing: Thing to determine color from.
    :return: color tuple.
    """

    if thing.has_factor(Color) and thing.Color is not None:
        return thing.Color.value

    tag = thing.Label
    if tag == "damper":
        return 0, 100, 100
    elif tag == "bumper":
        return 100, 100, 0
    elif tag == "sand":
        if thing.Friction <= 0.2:
            return 140, 200, 0
        elif 0.2 < thing.Friction <= 0.4:
            return 200, 200, 0
        elif 0.4 < thing.Friction:
            return 200, 140, 0
    elif tag == "magma":
        return 255, 0, 0
    elif tag == "magnet":
        if thing.Magnetism >= 0.0:
            return 50, 100, 50
        else:
            return 100, 50, 50
    elif tag == "fire":
        return 255, 100, 0
    elif tag == "charger":
        if thing.Charge >= 0.0:
            return 50, 100, 100
        else:
            return 100, 0, 100
    elif tag == "hole":
        return 0, 0, 0
    elif tag == "tile":
        return 0, 50, 50
    elif tag == "object":
        return 100, 100, 100
    elif tag == "ball":
        return 100, 100, 0
    else:
        return None


# The user can add new colors by modifying the rules.
_COLOR_RULES = [_get_default_color]


def register_color(tag, color):
    """Add a custom color rule.

    If a rule doesn't exist for a tag, try to apply the defaults.

    :param tag: Thing tag to apply color to.
    :param color: Color to apply.
    """
    global _COLOR_RULES
    if callable(color):

        def rule(thing):
            return color(thing) if thing.Label == tag else None
    else:
        def rule(thing):
            return color if thing.Label == tag else None

    _COLOR_RULES = [rule] + _COLOR_RULES


def register_rule(rule: Callable):
    """Registers a new rule

    :param rule: a callable rule that takes a Thing and returns None or a color.
    """
    global _COLOR_RULES
    _COLOR_RULES = [rule] + _COLOR_RULES


def get_color(thing):
    """Get color for thing.

    :param thing: Thing to get color for.
    :return: Color tuple if rule exists otherwise None.
    """
    for rule in _COLOR_RULES:
        color = rule(thing)
        if color is not None:
            return color
    return None


class RGBRenderer(Renderer):
    """RGB rendered.

    Default is to solid colors.

    """

    def __init__(self, n_channels: int = 3, img_type: type = np.uint8, annotation: bool = False, **kwargs
                 ) -> None:
        """

        :param n_channels: Number of channels. Default is 3.
        :param img_type: Image type.
        :param kwargs:
        """
        super().__init__(
            n_channels=n_channels, img_type=img_type, annotation=annotation, **kwargs,
        )

    def get_background(self) -> np.ndarray:
        """Makes the background before anything is added.

        """
        return np.zeros((self.dim_y, self.dim_x, self.n_channels), dtype=np.uint8) + 10

    def visual_mapping(self, thing: Union[Thing, Wall]) -> Visual:
        """Maps objects and tiles to their corresponding visuals.

        """
        if isinstance(thing, Wall):
            visual = WallVisual()
        else:
            color = get_color(thing)
            shape = thing.Shape.value

            if color is None:
                raise ValueError(f"Thing {thing} has no color rule.")

            if isinstance(shape, Circle):
                visual_radius = self.absolute_to_pix(shape.radius)

                if thing.has_factor(Mobile):
                    outline = thing.Mobile
                else:
                    outline = True

                visual = CircleVisual(
                    thing,
                    radius=visual_radius,
                    color=color,
                    outline=outline,
                    label=thing.Text,
                    show_label=self.annotation,
                )

            elif isinstance(shape, ConvexHullShape):
                visual_points = np.array([self.coordinates_to_pix(p) for p in shape.points])
                visual = PolyVisual(
                    thing,
                    points=visual_points,
                    color=color,
                    label=thing.Text,
                    show_label=self.annotation,
                )

            else:
                raise NotImplementedError(shape)

        return visual

    def make_floor(self) -> np.ndarray:
        floor = self.get_background()
        self._floor[:] = np.ascontiguousarray(floor, dtype=np.uint8)
        return self._floor

    @property
    def floor(self) -> np.ndarray:
        return self._floor


class RGBTextureRenderer(RGBRenderer):
    """Renderer that uses textures derived from a generative model.

    """

    def __init__(
        self,
        *args,
        grayscale: bool = False,
        viz_generator: Optional[Generator] = None,
        config: dict = None,
        annotation: bool = False,
        n_rand_factors: int = 3,
        **kwargs,
    ):
        """

        :param grayscale: Whether to make the rendering greyscale.
        :param viz_generator: Optional generator, otherwise default to one
            that uses a model trained from an autoencoder.
        :param config: Configuration dictionary for the generative model.
        """
        super().__init__(*args, annotation=annotation, **kwargs)

        if config is None:
            raise ValueError(f"`config` must be provided for " f"{self.__class__.__name__}.")

        try:
            generative_model_path = config["model_path"]
        except KeyError:
            raise KeyError(
                f"{self.__class__.__name__} config must include " f"path(s) to generative model."
            )

        try:
            gclass = config["cls"]
        except KeyError:
            raise KeyError(f"{self.__class__.__name__} config must include " f"generator class.")

        if not issubclass(gclass, Generator):
            raise TypeError(gclass)

        self.viz_generator = None
        if viz_generator is None:
            if os.path.isdir(generative_model_path):
                generative_model_path = gclass.get_paths(generative_model_path)
            else:
                generative_model_path = [generative_model_path]

            if len(generative_model_path) == 0:
                raise ValueError(f"No models found for renderer " f'{config["model_path"]}.')

            self.viz_generators = []

            for model_path in generative_model_path:
                model = gclass(
                    dim_x=self.res,
                    dim_y=self.res,
                    model_path=model_path,
                    grayscale=grayscale,
                    n_rand_factors=n_rand_factors,
                )
                self.viz_generators.append(model)
        else:
            if not isinstance(viz_generator, list):
                viz_generator = [viz_generator]
            self.viz_generators = viz_generator

        self.sample()

    def sample(self) -> None:
        self.viz_generator = random.choice(self.viz_generators)

    def get_pattern(self, thing_factors: dict[Type[Factor], Factor]) -> np.ndarray:
        """Get pattern from arguments.

        """
        passed_factors = thing_factors.copy()
        for factor in self._filter_factors:
            passed_factors.pop(factor, None)

        pattern = self.viz_generator.get_pattern(passed_factors)
        if pattern is None:
            # Use a solid gray visual feature.
            color = 100.0, 100.0, 100.0
            pattern = np.full((self.dim_x, self.dim_y, self.n_channels), color, dtype=np.uint8,)
        return pattern

    def get_background(self) -> np.ndarray:
        return self.viz_generator.get_background(self.dim_x, self.dim_y)

    def visual_mapping(self, thing: Union[Thing, Wall]) -> Visual:
        """Maps objects and tiles to their corresponding visuals.

        """
        if isinstance(thing, Wall):
            visual = WallVisual()
        else:
            shape = thing.Shape.value
            texture = self.get_pattern(thing.factors)
            if isinstance(shape, Circle):
                visual_radius = self.absolute_to_pix(shape.radius)

                if thing.has_factor(Mobile):
                    outline = thing.Mobile
                else:
                    outline = True

                visual = CircleTextureVisual(
                    thing,
                    radius=visual_radius,
                    texture=texture,
                    outline=outline,
                    label=thing.Text,
                    show_label=self.annotation,
                )

            elif isinstance(shape, ConvexHullShape):
                visual_points = np.array([self.coordinates_to_pix(p) for p in shape.points])
                visual = PolyTextureVisual(
                    thing,
                    points=visual_points,
                    texture=texture,
                    label=thing.Text,
                    show_label=self.annotation,
                )

            else:
                raise NotImplementedError(shape)

        return visual


class CircleTextureVisual(CircleVisual):
    """Circle texture visual.

    """

    def __init__(self, thing: Thing, radius: float, texture: np.ndarray, *args, **kwargs):
        size = 2 * radius + 1
        img = texture[0:size, 0:size, :]
        super().__init__(thing, radius, *args, img=img, **kwargs)


class PolyTextureVisual(PolyVisual):
    """Poly texture visual.

    """

    def __init__(
        self, thing: Thing, points: np.ndarray, texture: np.ndarray, *args, **kwargs,
    ):
        min_x = points[:, 0].min()
        min_y = points[:, 1].min()
        width = points[:, 0].max() - min_x
        height = points[:, 1].max() - min_y

        img = texture[0:height, 0:width, :]
        super().__init__(thing, points, *args, img=img, **kwargs)
