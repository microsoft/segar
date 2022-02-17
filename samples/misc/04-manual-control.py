import cv2
import gym
import numpy as np
import segar
from segar.factors import Position

env = gym.make("Segar-tilesx1-medium-rgb-v1")

SCALE_FACTOR = 4
coords = []


def get_coords(event, x, y, flags, param):
    global coords
    if event == cv2.EVENT_LBUTTONDOWN:
        coords.append((x, y))


def get_pos(env):
    return env.env.env.envs[0]._get_current_sim().things["golfball"][Position].value


cv2.namedWindow("image")
cv2.setMouseCallback("image", get_coords)

while 1:
    obs = env.reset()
    done = False
    while 1:

        cv2.imshow("image", cv2.resize(obs, (0, 0), fx=SCALE_FACTOR, fy=SCALE_FACTOR))
        # k = cv2.waitKey(-1) & 0xFF
        cv2.waitKey(10)

        if coords:
            xy = np.array(coords.pop())
            xy = xy / SCALE_FACTOR / obs.shape[0] * 2 - 1
            xy[1] *= -1  # opencv is upside down
            ballpos = get_pos(env)
            diff = xy - ballpos
            obs, rew, done, _ = env.step(diff)
            print(f"action: {np.around(diff, 2)},reward: {rew}, done: {done}")

        if done:
            break
