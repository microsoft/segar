__copyright__ = "Copyright (c) Microsoft Corporation and Mila - Quebec AI Institute"
__license__ = "MIT"
"""Affordance map rendering

TODO: fix and update this.

"""

import cv2

# flake8: noqa

#  TODO: This needs to be updated


class AffordanceRenderer:
    def __init__(self):
        pass

    def make_floor(self):
        raise NotImplementedError
        # return np.zeros((Affordance._COUNT, self.res, self.res, 1))

    def render(self, obj):
        return obj.render_affordances(self.img)

    def show(self, duration):
        """ Display the current internal observation image in an OpenCV
        window

        :param duration: int, Show the window and pause for how long? In
            milliseconds
        :return: None
        """
        # == being, this works but it slows my machine to a grind. Use
        # at your own discretion
        for layer_idx in range(self.img.shape[0]):
            cv2.imshow(f"affordance {layer_idx}", self.img[layer_idx, :, :, 0])

        cv2.waitKey(duration)
