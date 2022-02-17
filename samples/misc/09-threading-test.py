__copyright__ = "Copyright (c) Microsoft Corporation and Mila - Quebec AI Institute"
__license__ = "MIT"

from multiprocessing import Process
from multiprocessing import Queue as mpQueue
from queue import Queue as thQueue
from threading import Thread
import numpy as np

import gym
import segar
import cv2

""" Open 3 environments in parallel - one in main thread, one in sub thread, one in process"""


def make_env(unique_name, queue_child=None, mpQueue_parent=None, thQueue_parent=None, seed=123):
    """This is either given a MP queue or a threading queue in which case it's writing
    to the queues. Or it's given the handles for BOTH queues in which case it's the main thread,
    reading from the queues, and displaying the window.
    """

    def show_img(obs):
        if queue_child is not None:
            queue_child.put(obs)
        else:
            mp_obs = mpQueue_parent.get()
            th_ob = thQueue_parent.get()
            img[:, :64, :] = obs
            img[:, 65 : 65 + 64, :] = th_ob
            img[:, 130:, :] = mp_obs
            cv2.imshow("viewer", cv2.resize(img, (0, 0), fx=4, fy=4))
            cv2.waitKey(1)

    if mpQueue_parent is not None:
        # then we are the main thread
        img = np.zeros((64, 64 * 3 + 2, 3), np.uint8)

    env = gym.make("Segar-tilesx2-hard-rgb-v0", unique_name=unique_name, seed=seed)
    while True:
        obs = env.reset()
        show_img(obs)
        done = False
        while not done:
            # only step is queue is empty (i.e. if sim has been rendered)
            if queue_child is not None and not queue_child.empty():
                continue
            action = env.action_space.sample()
            try:
                obs, _, done, _ = env.step(action)
            except RuntimeError:
                print("runtime err")
                pass
            show_img(obs)


if __name__ == "__main__":
    th_queue = thQueue()
    mp_queue = mpQueue()

    new_thread = Thread(
        target=make_env, kwargs={"queue_child": th_queue, "unique_name": "thread", "seed": 123}
    )
    new_process = Process(
        target=make_env, kwargs={"queue_child": mp_queue, "unique_name": "proc", "seed": 123}
    )

    new_thread.start()
    new_process.start()
    print("started thread/proc, now doing main call")
    make_env(mpQueue_parent=mp_queue, thQueue_parent=th_queue, unique_name="main", seed=123)
    print("this will never be reached")
