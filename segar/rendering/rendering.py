__copyright__ = "Copyright (c) Microsoft Corporation and Mila - Quebec AI Institute"
__license__ = "MIT"
"""For rendering into pixel-based observations.

"""

__all__ = ("Renderer", "Visual", "PolyVisual", "CircleVisual", "WallVisual")

from typing import Any, Optional, Union

import cv2
import numpy as np

from segar import get_sim
from segar.factors import Factor, Floor
from segar.things import Thing
from segar.sim import Simulator
from segar.parameters import (
    Resolution,
    _TEXT_FACE,
    _TEXT_SCALE,
    _TEXT_THICKNESS,
)


class Visual:
    """Visual object for rendering things.

    """

    def __init__(
        self,
        thing: Thing,
        center: tuple[float, float],
        img: np.ndarray,
        alpha: np.ndarray,
        label: Optional[str] = None,
        show_label: bool = False,
    ):
        """

        :param thing: Thing to be rendered.
        :param center: Center of thing.
        :param img: Img to render on. np.array
        :param alpha: Transparency alpha.
        :param label: Optional label to print of thing.
        """
        self.img = img
        self.alpha = alpha
        self.reverse_alpha = 1 - alpha
        self.height = img.shape[0]
        self.width = img.shape[1]
        self.channels = img.shape[2]
        self.center = center
        self.thing = thing
        if label and show_label:
            if isinstance(label, Factor):
                label = label.value
            self.label_img = np.ones_like(img)
            # Add some text (useful for identifying the ball, which the
            # agent can control)
            text_size, _ = cv2.getTextSize(label, _TEXT_FACE, _TEXT_SCALE, _TEXT_THICKNESS)
            text_origin = (
                int((self.width - text_size[0]) / 2),
                int((self.height + text_size[1]) / 2),
            )
            cv2.putText(
                self.label_img,
                label,
                text_origin,
                _TEXT_FACE,
                _TEXT_SCALE,
                np.zeros(self.channels),
                _TEXT_THICKNESS,
                cv2.LINE_AA,
            )
        else:
            self.label_img = None

    def get_position(self) -> np.ndarray:
        """Returns the position of the thing this visual corresponds to.

        :return: Position of thing.
        """
        return self.thing.Position

    def get_obj_size(self) -> float:
        """Returns the size of the thing this visual corresponds to.

        :return: Size of thing.
        """
        return self.thing.Size.value

    def resize(
        self, size: tuple[float, float], center: Optional[tuple[float, float]] = None,
    ) -> None:
        """Resizes the thing according to floor scaling.

        :param size: Tuple size to resize to.
        :param center: Optional center position.
        """
        width, height = size
        if width == self.width and height == self.height and center == self.center:
            return
        if center is not None:
            self.center = center
        self.width = width
        self.height = height
        self.img = cv2.resize(self.img, (width, height))
        if self.label_img is not None:
            self.label_img = cv2.resize(self.label_img, (width, height))
        self.alpha = cv2.resize(self.alpha, (width, height)).reshape((width, height, 1))
        self.reverse_alpha = 1 - self.alpha

    def render(self, position: tuple[float, float], visual_map: np.ndarray) -> None:
        """Renders the visual and affordance maps.

        :param position: The position on the map where the visual should be
            rendered, specified as the center of the visual.
        :param visual_map: (optional) The visual (rgb/grayscale) image to
            render to.
        """
        if not self.thing.Visible:
            return
        x = position[0] - self.center[0]
        y = position[1] - self.center[1]
        map_size = visual_map.shape

        _tx_min = max(x, 0)
        _tx_max = min(x + self.width, map_size[0])
        _sx_min = 0 if x >= 0 else -x
        _sx_max = _sx_min + (_tx_max - _tx_min)

        _ty_min = max(y, 0)
        _ty_max = min(y + self.height, map_size[1])
        _sy_min = 0 if y >= 0 else -y
        _sy_max = _sy_min + (_ty_max - _ty_min)

        # make sure at least some part of the visual is within the map
        if _tx_min >= map_size[0] or _tx_max < 0 or _ty_min >= map_size[1] or _ty_max < 0:
            return

        # update visuals
        visual_map[_ty_min:_ty_max, _tx_min:_tx_max, :] *= self.reverse_alpha[
            _sy_min:_sy_max, _sx_min:_sx_max
        ]
        visual_map[_ty_min:_ty_max, _tx_min:_tx_max, :] += self.img[
            _sy_min:_sy_max, _sx_min:_sx_max
        ]
        if self.label_img is not None:
            visual_map[_ty_min:_ty_max, _tx_min:_tx_max, :] *= self.label_img[
                _sy_min:_sy_max, _sx_min:_sx_max
            ]


class WallVisual:
    """Visual for walls

    TODO: this is an abstract placeholder for future walls with visuals,
    such as maze environments.

    """

    def resize(self, *args, **kwargs):
        pass

    def render(self, *args, **kwargs):
        pass

    def get_obj_size(self) -> float:
        return 0.0

    def get_position(self) -> tuple[float, float]:
        return 0.0, 0.0


class CircleVisual(Visual):
    def __init__(
        self,
        thing: Thing,
        radius: float,
        color: tuple[int, int, int] = None,
        img: np.ndarray = None,
        outline: bool = False,
        label: Optional[str] = None,
        dtype: str = "uint8",
        channels: int = 3,
        show_label: bool = False,
    ):
        """Initializes a circle visual shape.

        :param radius: Circle radius, in pixels.
        :param color: A tuple or array representing a color to draw with.
            This must be present if texture is not specified.
        :param outline: Add an outline to the object.
        :param label: An optional string to overlay onto the texture.
        """
        size = 2 * radius + 1
        if img is None:
            img = np.full((size, size, channels), color, dtype=dtype)
        alpha = np.zeros((size, size, 1), dtype="uint8")
        cv2.circle(alpha, (radius, radius), radius, 1, -1, 8, 0)
        try:
            img = img * alpha
        except ValueError:
            raise ValueError(f"Object may be larger than arena: " f"{alpha.shape} vs {img.shape}")
        if outline:
            cv2.circle(img, (radius, radius), radius, 0, 1, 8, 0)
        super().__init__(
            thing, (radius, radius), img, alpha, label=label, show_label=show_label,
        )

    def get_obj_size(self) -> float:
        return self.thing.Shape.value.radius

    def resize(self, radius: float) -> None:
        size = 2 * radius + 1
        super().resize((size, size), (radius, radius))


class PolyVisual(Visual):
    def __init__(
        self,
        thing: Thing,
        points: np.ndarray,
        img: np.ndarray = None,
        color: tuple[int, int, int] = None,
        label: Optional[str] = None,
        dtype: str = "uint8",
        channels: int = 3,
        show_label: bool = False,
    ) -> None:
        """Initializes a polyline visual shape.

        :param points: the corners of the polyline
        :param img: if specified, a img array. For now, the dimensions
            of the texture must be equal or greater than size.
        :param color: A tuple or array representing a color to fill with. This
            must be present if texture is not specified.
        :param label: An optional string to overlay onto the texture.
        """

        min_x = points[:, 0].min()
        min_y = points[:, 1].min()
        width: int = points[:, 0].max() - min_x
        height: int = points[:, 1].max() - min_y
        points -= np.array([min_x, min_y])[None, :]

        if img is None:
            img = np.full((height, width, channels), color, dtype=dtype)
        alpha = np.zeros((height, width, 1), dtype="uint8")
        cv2.fillPoly(alpha, [points], color=1)
        img = img * alpha
        super().__init__(
            thing, (width // 2, height // 2), img, alpha, label=label, show_label=show_label,
        )

    def render(self, position: tuple[float, float], *args) -> None:
        return super().render(position, *args)


class Renderer:
    """Abstract renderer class.

    """

    def __init__(
        self,
        n_channels: int,
        img_type: type = np.float16,
        res: Union[int, Resolution] = Resolution(),
        dim_x: Optional[int] = None,
        dim_y: Optional[int] = None,
        filter_factors: Optional[list[Factor]] = None,
        annotation: bool = False,
    ):
        """

        :param n_channels: Number of channels for output space.
        :param img_type: Image type.
        :param res: Resolution of the output image.
        :param dim_x: Optional number of x pixels.
        :param dim_y: Optional number of y pixels.
        :param filter_factors: Factors to filter from rendering.
        """
        if isinstance(res, Resolution):
            res = res.value

        self.res = res
        self.dim_x = dim_x or res
        self.dim_y = dim_y or res
        self.n_channels = n_channels
        if annotation:
            dim_y = self.dim_y * 2
        else:
            dim_y = self.dim_y
        self._floor = np.zeros((self.dim_x, self.dim_y, self.n_channels), dtype=img_type)
        self.img = np.zeros((self.dim_x, dim_y, self.n_channels), dtype=img_type)
        self.obj_visuals = []
        self.tile_visuals = []
        self._filter_factors = filter_factors or []
        self.annotation = annotation

    def add_text(
        self,
        txt: str,
        pos: tuple[int, int] = (10, 40),
        size: float = 0.5,
        color: tuple[int, int, int] = (255, 255, 255),
    ) -> None:
        """Adds text to current image observation.

        Results in a change to the image observation.

        :param txt: Text to print.
        :param pos: Pixel coords of the bottom left corner of where the text
            should start (measured from the top left corner of the image).
        :param size: Scale factor of the text (e.g. .5 = small, 1 = normal,
            1.5 = big).
        :param color: Color of the text.
        """

        font = cv2.FONT_HERSHEY_SIMPLEX
        bottomLeftCornerOfText = pos
        fontScale = size
        fontColor = color
        lineType = 2

        cv2.putText(
            self.img, txt, bottomLeftCornerOfText, font, fontScale, fontColor, lineType,
        )

    @property
    def sim(self) -> Simulator:
        return get_sim()

    def sample(self) -> None:
        """For randomization.

        """
        pass

    def set_visuals(self, sim: Optional[Simulator] = None) -> None:
        """ Makes the visualization objects for all of the simulator things.

        :param sim: Simulator to draw objects from to make visuals.

        """
        sim = sim or self.sim
        for thing in sim.things.values():
            if not isinstance(thing, Thing):
                continue
            vis = self.visual_mapping(thing)

            if thing.has_factor(Floor):
                self.tile_visuals.append(vis)
            else:
                self.obj_visuals.append(vis)

    def set_filter(self, filter_factors: list[Factor]) -> None:
        """Sets the filter for this renderer

        :param filter_factors: Factors to filter from rendering.
        """
        self._filter_factors = filter_factors

    def visual_mapping(self, thing: Thing) -> Visual:
        """Creates a Visual object for the corresponding thing.

        Must be overridden.

        :param thing: Simulator Thing to build visualization object from.
        :return: Visual object.
        """
        raise NotImplementedError

    @property
    def floor(self) -> np.ndarray:
        """The floor array.

        """
        return self._floor

    def make_floor(self) -> np.ndarray:
        """Makes the background from the tiles.

        Must be overridden.

        """
        raise NotImplementedError

    def render_tile(self, tile_vis: Visual) -> None:
        """Renders a tile.

        :param tile_vis: Tile Visual object.
        """
        coords = self.coordinates_to_pix(tile_vis.get_position())
        tile_vis.render(coords, self.floor)

    def render_object(self, obj_vis: Visual) -> None:
        """Renders an object.

        :param obj_vis: Object Visual object.
        """
        obj_vis.resize(self.absolute_to_pix(obj_vis.get_obj_size()))
        obj_vis.render(
            self.coordinates_to_pix(obj_vis.get_position()), self.img[:, : self.dim_y],
        )

    def reset(self, sim: Simulator = None) -> None:
        """Resets the renderer.

        Removes rendering objects and creates new one from simulator.

        """

        sim = sim or self.sim

        self.obj_visuals = []
        self.tile_visuals = []
        self.set_visuals(sim)
        self.make_floor()
        self.img *= 0

        # We go in reverse order because the first tiles cover the latter
        # tiles.

        for tile_vis in self.tile_visuals[::-1]:
            self.render_tile(tile_vis)

    def coordinates_to_pix(
        self, coords: Union[np.ndarray, tuple[float, float]]
    ) -> tuple[float, float]:
        """Transforms a pair of global coordinates to pixel coordinates.

        :param coords: Global coordinates (x,y).
        :return: Pixel coordinates (x,y) in OpenCV convention (x starts
            left, y starts at the top).
        """

        x, y = coords

        def rescale(val):
            return int(
                np.around((val - self.sim.boundaries[0]) * self.res / self.sim.arena_size, 0,)
            )

        x_scaled = rescale(x)
        y_scaled = self.res - rescale(y)

        return x_scaled, y_scaled

    def absolute_to_pix(self, val: float) -> int:
        """Converts a single _proportinal_ value in global system to pixels.

        E.g. a ball's size is expressed not as coordinates but as a float
        value -> what's that in pixels?

        :param val: Proportional value in global coordinate system
        :return: Proportional value in pixel coordinate system
        """

        return int(np.around(val * self.res / self.sim.arena_size, 0))

    def show(self, duration: int) -> None:
        """Display the current internal observation image in an OpenCV
            window.

        :param duration: Show the window and pause for how long? In ms.
        """
        # show the visual
        cv2.imshow("frame", self.img[:, :, ::-1])
        cv2.waitKey(duration)

    def add_res_annotation(
        self,
        arr: np.ndarray,
        results: dict[str, Any],
        pos: tuple[int, int] = (10, 40),
        size: float = 0.5,
        color: tuple[int, int, int] = (255, 255, 255),
    ) -> None:

        annotation_arr = np.zeros_like(arr[:, self.dim_y :])

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = size

        for k, v in results.items():
            if isinstance(v, float):
                v = round(v, 2)
            txt = f"{k} = {v}"
            cv2.putText(annotation_arr, txt, pos, font, font_scale, color, 2)
            pos = (pos[0], pos[1] + 20)  # Move cursor

        self.img[:, self.dim_y :] = annotation_arr

    def __call__(self, results: dict[str, Any] = None) -> np.ndarray:
        """Draw current observation (arena background + ball)

        :param results: dictionary of additional results, rendered as string
            in right side of image.

        :return: Rendered image of the environment.
        """

        self.img[:, : self.dim_y] = self.floor[:]

        for obj_vis in self.obj_visuals:
            self.render_object(obj_vis)

        if results is not None and self.annotation:
            self.add_res_annotation(self.img, results)

        return self.img.copy()
