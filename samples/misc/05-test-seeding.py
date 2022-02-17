__copyright__ = "Copyright (c) Microsoft Corporation and Mila - Quebec AI Institute"
__license__ = "MIT"

import cv2
import gym
import numpy as np
import imageio
import segar  # required to have envs registered

resolution = 64
kwargs = {
    "num_envs": 1,
    "num_levels": 2,
    "framestack": 1,
    "resolution": resolution,
    "max_steps": 200,
    "seed": 123,
}
env1 = gym.make("Segar-tilesx1-hard-rgb-v0", **kwargs)
env2 = gym.make("Segar-tilesx1-hard-rgb-v0", **kwargs)
env3 = gym.make("Segar-tilesx1-hard-rgb-v0", **kwargs)

SCALE_FACTOR = 4
coords = []


cv2.namedWindow("image")

img_buf = np.zeros((resolution, resolution * 3 + 2, 3), np.uint8)

images = []
GIF_frames = 500
while 1:
    obs1 = env1.reset()
    obs2 = env2.reset()
    obs3 = env3.reset()
    done = False
    while 1:
        img_buf[:, :resolution, :] = obs1
        img_buf[:, resolution + 1 : resolution * 2 + 1, :] = obs2
        img_buf[:, (resolution + 1) * 2 :, :] = obs3
        img_buf_rescaled = cv2.resize(img_buf, (0, 0), fx=SCALE_FACTOR, fy=SCALE_FACTOR)
        if GIF_frames:
            images.append(img_buf_rescaled)
            GIF_frames -= 1
        cv2.imshow("image", img_buf_rescaled[:, :, ::-1])
        cv2.waitKey(1)

        action = env1.action_space.sample()

        obs1, rew1, done1, _ = env1.step(action)
        obs2, rew2, done2, _ = env2.step(action)
        obs3, rew3, done3, _ = env3.step(action)

        if done1 or done2 or done3:
            break

        if not GIF_frames:
            imageio.mimsave("3-parallel-envs.gif", images, fps=30)
            print("GIF saved!")
            quit()

# just adding this to make a test push
