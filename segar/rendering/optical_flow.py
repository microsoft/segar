__copyright__ = "Copyright (c) Microsoft Corporation and Mila - Quebec AI Institute"
__license__ = "MIT"

# TODO fix and update this.
# flake8: noqa

from segar.rendering.rendering import Renderer


class OpticalFloRenderer(Renderer):
    def __init__(self):
        self.optical_flo = OpticalFlo(
            res,
            boundaries,
            _DEFAULT_MAX_VEL,
            self.dt,
            self.coordinates_to_pix,
            self.absolute_to_pix,
            scaling=2,
        )

    def render_optical_flo(self):
        return self.optical_flo.render_optical_flo(self.ball)


class OpticalFlo:
    def __init__(
        self, res, arena_boundaries, max_vel, dt, coordinate2pix_func, absolute2pix_func, scaling=1,
    ):
        self.old_ball_pos = None
        # half_extent = max(arena_boundaries) - min(arena_boundaries) / 2
        # self.half_diagonal = norm((half_extent, half_extent))

        # because it's linear, not diagonal
        self.max_vel = norm((max_vel, max_vel))
        self.img = np.zeros((res, res, 3), np.uint8)
        self.coordinate2pix_func = coordinate2pix_func
        self.absolute2pix_func = absolute2pix_func
        self.dt = dt
        self.scaling = scaling

    def vector2color(self, vector):
        # direction = HSV color hue (H)
        # norm = HSV saturation (S)
        # value is always max

        vector_norm = norm(vector) / self.max_vel / self.dt

        if vector_norm < 0.0000001:
            angle = 0
        else:
            vector_unit = vector / norm(vector)
            dot_product = np.dot(_UP, vector_unit)
            angle = np.arccos(dot_product)
            if vector[0] < 0:
                angle = np.pi + (np.pi - angle)  # to go over 180 to 360
            angle /= 2 * np.pi

        vector_norm *= self.scaling
        vector_norm = np.clip(vector_norm, 0, 1)

        hsv = np.uint8([[[int(angle * 255), int(vector_norm * 255), 255]]])
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        # this is weird but the OpenCV gods demand it
        return tuple([int(x) for x in rgb[0][0]])

    def render_optical_flo(self, ball):
        self.img.fill(255)  # white
        if self.old_ball_pos is None:
            pass
        else:
            diff = ball.pos - self.old_ball_pos
            rgb = self.vector2color(diff)
            cv2.circle(
                self.img,
                self.coordinate2pix_func(ball.pos),
                self.absolute2pix_func(ball.size),
                rgb,
                -1,
            )
        self.old_ball_pos = np.copy(ball.pos)

        return self.img
