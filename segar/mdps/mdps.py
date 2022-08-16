__copyright__ = "Copyright (c) Microsoft Corporation and Mila - Quebec AI Institute"
__license__ = "MIT"
"""Module for MDPs

MDPs are constructed from states, observation spaces, and tasks. Tasks are
composed of stopping conditions, reward functions, and action spaces.

"""

__all__ = ("MDP",)

import math
import random
from typing import Optional

import gym
import numpy as np

from segar import get_sim, timeit
from segar.rendering.rgb_rendering import RGBRenderer
from segar.sim import Simulator
from segar.utils import append_dict

from .observations import Observation
from .tasks import Task


class MDP(gym.Env):
    """A framework for Markov decision processes.

    Technically a POMDP, as observations are not necessarily the same as the
    underlying states.

    """

    def __init__(
        self,
        observation: Observation,
        task: Task,
        sub_steps: int = 1,
        max_steps_per_episode: int = 100,
        episodes_per_arena: int = 1,
        result_keys: Optional[dict] = None,
        sim: Optional[Simulator] = None,
        reset_renderer_every_call: bool = False,
        stop_on_done: bool = True
    ):
        """

        :param observation: Observation object. Contains the observation
        space and generates observations to be consumed by the MDP.
        :param task: Task object. The task is composed of the reward
        function, action space, and stopping conditions.
        :param sub_steps: This MDP with perform a number of steps through
        the simulator for every step as a Gym environment.
        :param max_steps_per_episode: Number of Gym steps max to step through.
        :param episodes_per_arena: Number of episodes to run through for
        every randomization of the MDP.
        :param result_keys: Results from the full state dictionary to log.
        :param sim: Optional non-global simulator to pass to MDP.
        :param reset_renderer_every_call: Resets the renderer every call. Only use if visual appearances change.
        """

        self._observation = observation
        self._task = task
        # This renderer is for human consumption.
        self._renderer = RGBRenderer(annotation=True, reset_every_call=reset_renderer_every_call)

        self.sub_steps = sub_steps
        self.num_episodes = 0
        self.max_steps_per_episode = max_steps_per_episode
        self.episodes_per_arena = episodes_per_arena
        self.result_keys = result_keys or []

        self.num_steps = 0
        self.num_sub_steps = 0
        self.stop_on_done = stop_on_done
        self.total_reward = 0
        self._sim = None
        self.set_sim(sim)
        self.set_component_sims()
        super().__init__()
        self.reset()

    @property
    def sim(self) -> Simulator:
        if self._sim is None:
            return get_sim()
        return self._sim

    @property
    def observation_space(self):
        return self._observation.observation_space

    @property
    def action_space(self):
        return self._task.action_space

    @property
    def state(self) -> dict:
        return self.sim.state

    @timeit
    def reward(self, state: dict) -> float:
        return self._task.reward(state)

    @timeit
    def observation(self, state: dict) -> np.ndarray:
        return self._observation(state)

    @timeit
    def done(self, state: dict) -> bool:
        if self.num_steps >= self.max_steps_per_episode:
            return True
        return self._task.done(state)

    def apply_action(self, action: np.ndarray) -> None:
        self._task.apply_action(action)

    def demo_action(self) -> np.ndarray:
        """Returns a demo action from the _task.

        :return: Action
        """
        return self._task.demo_action()

    def set_sim(self, sim: Simulator) -> None:
        self._sim = sim

    def set_component_sims(self) -> None:
        self._observation.set_sim(self._sim)
        self._task.set_sim(self._sim)

    @staticmethod
    def seed(seed: int = None) -> None:
        """Sets the seed for this env's random number generator."""
        np.random.seed(seed)
        random.seed(seed)

    def reset(self) -> np.ndarray:
        """Resets the MDP.

        If the maximum episodes per arena have been passed, randomize the
        task and observation space. This randomizes the initialization,
        reward function, and stopping conditions, where applicable and defined.

        :return: First observation.
        """
        # Just to be on the safe side.  Just aligns refs.
        self.set_component_sims()
        reset_arena = self.episodes_per_arena > 0 and (
            self.num_episodes % self.episodes_per_arena == 0
        )

        if reset_arena:
            self.sim.reset()
            self._task.sample()

        self._task.initialize()

        if reset_arena:
            self._observation.sample()
        self._observation.reset()
        self._renderer.reset(self.sim)

        self.num_episodes += 1
        self.num_steps = 0
        self.num_sub_steps = 0
        self.total_reward = 0
        obs = self.observation(self.state)
        return obs

    @timeit
    def render(
        self, mode: str = "human", delay: int = 2, label: str = None, agent_view: bool = False,
    ) -> np.ndarray:
        """Render function for Gym functionality.

        :param mode: 'human' or 'rgb_array'. If 'human',
        show rendering in real-time.
        :param delay: time delay of show.
        :param label: label to add to the image.
        :param agent_view: Show the agent renderer, if exists.

        :return: Rendering of the state.
        """

        if agent_view:
            if hasattr(self._observation, "render") and hasattr(self._observation, "show"):
                if mode == "rgb_array":
                    results = None
                else:
                    results = self._task.results(self.state)
                img = self._observation.render(results=results)
                if hasattr(self._observation, "add_text"):
                    if label is not None:
                        self._observation.add_text(label)
                if mode == "rgb_array":
                    return img
                elif mode == "human":
                    self._observation.show(delay)
                    return img
            else:
                raise AttributeError("Agent's observation space is not " "pixel-based.")

        else:
            results = self._task.results(self.state)
            img = self._renderer(results=results)
            if label is not None:
                self._renderer.add_text(label)
            if mode == "rgb_array":
                return img
            elif mode == "human":
                self._renderer.show(delay)
                return img

    @timeit
    def substep(self) -> tuple[float, bool, dict]:
        """One step through the simulator.

        :return: Reward, done, and results for this step.
        """

        state = self.state

        if not self.done(state) or not self.stop_on_done:
            self.sim.step()

        rew = self.reward(state)
        done = self.done(state)

        step_results = self._task.results(self.state)
        step_results["reward"] = rew

        for k in self.result_keys:
            if k not in state.keys():
                raise KeyError(
                    f"User asked for result {k} but this wasn't "
                    f"found in the states {state.keys()}."
                )
            step_results[k] = state[k]

        self.num_sub_steps += 1
        return rew, done, step_results

    @timeit
    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, dict]:
        """One gym env step

        :param action: Action taken by the agent.
        :return: observations, reward, done, and results
        """
        if np.isinf(action).any() or np.isnan(action).any():
            raise ValueError(f"Got bad action from policy {action}.")
        self.apply_action(action)

        results = dict()
        rew = 0
        done = False
        for _ in range(self.sub_steps):
            rew, done, step_results = self.substep()
            append_dict(results, step_results)
            if done:
                break

        self.total_reward += rew
        self.num_steps += 1
        obs = self.observation(self.state)

        assert math.isfinite(rew)
        return obs, rew, done, results

    def get_custom_metrics(self) -> dict:
        """This method should compute and return any relevant metrics for
        the entire episode.
        """
        return {}

    @property
    def success(self) -> int:
        """Success is task is successful and done.

        """
        return self._task.success and self._task.done

    @property
    def latent_obs(self) -> Observation:
        return self._task.latent_obs
