__copyright__ = "Copyright (c) Microsoft Corporation and Mila - Quebec AI " \
                "Institute"
__license__ = "MIT"
"""Module for observation space components

"""

__all__ = ('Observation', 'ObjectStateObservation', 'TileStateObservation',
           'AllObjectsStateObservation', 'AllTilesStateObservation',
           'AllStateObservation', 'RGBObservation', 'MultimodalObservation',
           'make_stacked_observation', 'StateObservation')

from typing import Any, Optional, Type, Union

from gym.spaces import Box, Tuple
import numpy as np

from segar import get_sim
from segar.factors import (Factor, Charge, Mass, Magnetism, StoredEnergy,
                           Position, Velocity, Acceleration, Done, Alive,
                           Friction, Heat, FACTORS, VectorFactor,
                           BooleanFactor, NumericFactor)
from segar.parameters import Resolution, DoesNotHaveFactor
from segar.rendering.rgb_rendering import RGBTextureRenderer
from segar.sim import Simulator
from segar.types import ThingID


class Observation:
    """Observation object.

    """
    def __init__(self, filter_factors: list[Type[Factor]] = None):
        """

        :param filter_factors: (ObjectFactors or TileFactors) Factors to
            filter out in observations.
        """

        self._sim = None
        self._filter_factors = filter_factors or []
        self._observation_space = None
        self._make_observation_space()

    def set_sim(self, sim: Simulator) -> None:
        self._sim = sim

    def sample(self, *args, **kwargs) -> None:
        """Samples the parameters of the observation space, if stochastic.

        """
        pass

    def reset(self) -> None:
        """Resets the observation space (parameters) to initial state.

        """
        pass

    def _make_observation_space(self):
        raise NotImplementedError

    @property
    def observation_space(self):
        """Protected observation space object.

        """
        if self._observation_space is None:
            raise AttributeError('Observation space for object must be set.')
        return self._observation_space

    @property
    def sim(self) -> Simulator:
        if self._sim is None:
            return get_sim()
        return self._sim

    def __call__(self, state: dict) -> np.ndarray:
        """Generates the observations from the state space.

        :return: Observations.
        """
        raise NotImplementedError


class RGBObservation(Observation):
    """RGB observations.

    Uses a renderer to generate RGB observations from the state space.

    """
    def __init__(self, resolution: Union[Resolution, int] = Resolution(),
                 renderer=None, config: dict = None,
                 filter_factors: list[Type[Factor]] = None,
                 annotation: bool = False):
        """

        :param resolution: Resolution of the pixel space.
        :param renderer: Optional renderer object. Otherwise use built-in RGB
        texture renderer.
        :param config: config to pass to the renderer.
        :param annotation: Whether to annotate the RGB observation.
        """
        if isinstance(resolution, Resolution):
            resolution = resolution.value

        self._renderer = renderer or RGBTextureRenderer(
            3, res=resolution, config=config, annotation=annotation)
        self.img_shape = (resolution, resolution, 3)

        super().__init__(filter_factors=filter_factors)
        self._renderer.set_filter(self._filter_factors)

    def _make_observation_space(self) -> None:
        self._observation_space = Box(
            0, 255,
            shape=self.img_shape,
            dtype=np.uint8
        )

    def render(self, results: dict[str, Any] = None):
        """Renders the state space into an RGB observation.

        """
        return self._renderer(results=results)

    def show(self, *args, **kwargs) -> None:
        """Shows the last rendered image for human interpretation.

        """
        self._renderer.show(*args, **kwargs)

    def sample(self) -> None:
        self._renderer.sample()

    def reset(self) -> None:
        self._renderer.reset(self.sim)

    def add_text(self, *args, **kwargs):
        """Adds text to the pixel observations.

        """
        self._renderer.add_text(*args, **kwargs)

    @property
    def resolution(self) -> int:
        return self._renderer.res

    def __call__(self, state: dict) -> np.ndarray:
        return self._renderer()


class StateObservation(Observation):
    """Single object state observations.

    """

    def __init__(self, unique_id: ThingID,
                 factors: Optional[list[Type[Factor]]] = None,
                 filter_factors: Optional[list[Type[Factor]]] = None):
        """

        :param unique_id: Unique id of the object to return states for.
        :param factors: List of factors to use.
        :param filter_factors: List of Factors to remove from observation
            space.
        """

        if factors is None:
            factors = FACTORS

        self.factors = []
        self.len = 0
        for factor_type in factors:
            if issubclass(factor_type, (NumericFactor, BooleanFactor)):
                self.factors.append(factor_type)
                self.len += 1
            elif issubclass(factor_type, VectorFactor):
                self.factors.append(factor_type)
                self.len += 2

        if filter_factors is not None:
            extra_factors = set(filter_factors) - set(self.factors)
            if len(extra_factors) > 0:
                raise ValueError(f'This observation space cannot filter '
                                 f'factors {extra_factors}. Only'
                                 f' {self.factors} allowed.')

        super().__init__(filter_factors=filter_factors)
        self.unique_id = unique_id

    def _make_observation_space(self) -> None:
        self._observation_space = Box(
            -100, 100,
            shape=(self.len,),
            dtype=np.float32
        )

    def make_observation_from_state(self, state: dict[Type[Factor], Factor]
                                    ) -> np.ndarray:
        obs = []
        for factor in self.factors:
            if factor in self._filter_factors:
                continue
            # If the factor is not in the state space, then assign special
            # value.
            value = state.get(factor, None)

            if value is not None:
                value = value.value

            if issubclass(factor, NumericFactor):
                if value is None:
                    value = DoesNotHaveFactor().value
                obs.append(value)
            elif issubclass(factor, VectorFactor):
                if value is None:
                    value = np.array([DoesNotHaveFactor().value,
                                      DoesNotHaveFactor().value])
                obs += value.tolist()
            elif issubclass(factor, BooleanFactor):
                if value is None:
                    value = DoesNotHaveFactor().value
                obs.append(int(value))

        return np.array(obs)

    def __call__(self, states: dict):
        thing_states = states['things']
        try:
            state = thing_states[self.unique_id]
        except KeyError:
            raise KeyError(f'{self.unique_id} not found in states, '
                           f'found {list(thing_states.keys())}.')
        return self.make_observation_from_state(state)


class ObjectStateObservation(StateObservation):
    """Single object state observations.

    Uses factors (Charge, Mass, Magnetism, StoredEnergy, Position, Velocity,
        Acceleration, Alive, Done)

    """

    def __init__(self, unique_id: ThingID,
                 filter_factors: Optional[list[Type[Factor]]] = None):
        factors = [Charge, Mass, Magnetism, StoredEnergy, Position, Velocity,
                   Acceleration, Alive, Done]
        super().__init__(unique_id,
                         factors=factors,
                         filter_factors=filter_factors)


class TileStateObservation(StateObservation):
    """Single tile state observations.

    Uses factors (Heat, Friction, Position)

    """

    def __init__(self, unique_id: ThingID,
                 filter_factors: Optional[list[Type[Factor]]] = None):
        factors = [Heat, Friction, Position]
        super().__init__(unique_id,
                         factors=factors,
                         filter_factors=filter_factors)


class AllStateObservation(StateObservation):
    """All state observations.

    Output is a fixed-size tensor that has capacity for only a certain
    number of objects. Tensor is filled in-order as they appear,
    but task-specific ids can be provided to allow for fixed positioning.

    """
    def __init__(self, n_things: int = 20, unique_ids: list[ThingID] = None,
                 factors: Optional[list[Type[Factor]]] = None,
                 required_factors: Optional[list[Type[Factor]]] = None,
                 filter_factors: list[Type[Factor]] = None):
        """

        :param n_things: Max number of objects.
        :param unique_ids: Unique ids for objects to be placed first in the
        observation tensor to allow for fixed positioning.
        :param factors: List of factors to use.
        :param required_factors: List of required factors. Otherwise,
            thing is omitted.
        :param filter_factors: List of Factors to remove from observation
            space.
        """
        self.n_things = n_things
        self.required_factors = required_factors
        self.unique_ids = unique_ids or []
        super().__init__(unique_id=None,
                         factors=factors,
                         filter_factors=filter_factors)

    def __call__(self, states: dict) -> np.ndarray:
        if self.required_factors is not None:
            thing_states = dict((k, s) for k, s in states['things'].items()
                                if all(f in s for f in self.required_factors))
        else:
            thing_states = states['things']

        if len(thing_states.keys()) > self.n_things:
            raise ValueError(f'Too many things in sim '
                             f'{len(thing_states.keys())} for this observation'
                             f' space ({self.n_things} max).')

        observations = np.zeros((self.n_things, self.len)).astype(
            np.float32)

        def update_obs(key_, idx):
            try:
                state = thing_states[key_]
            except KeyError:
                raise KeyError(f'{key_} not found in thing states, '
                               f'found {list(thing_states.keys())}.')

            obs = self.make_observation_from_state(state)
            observations[idx] = obs

        i = 0
        for unique_id in self.unique_ids:
            update_obs(unique_id, i)
            i += 1

        for key in thing_states.keys():
            if key not in self.unique_ids:
                update_obs(key, i)
                i += 1

        return observations


class AllObjectsStateObservation(AllStateObservation):
    """All objects state observations.

    Uses factors (Charge, Mass, Magnetism, StoredEnergy, Position, Velocity,
        Acceleration, Alive, Done)

    """

    def __init__(self, n_things: int = 20, unique_ids: list[ThingID] = None,
                 filter_factors: Optional[list[Type[Factor]]] = None):
        factors = [Charge, Mass, Magnetism, StoredEnergy, Position, Velocity,
                   Acceleration, Alive, Done]
        super().__init__(n_things=n_things,
                         unique_ids=unique_ids,
                         factors=factors,
                         required_factors=factors,
                         filter_factors=filter_factors)


class AllTilesStateObservation(AllStateObservation):
    """All tiles state observations.

    Uses factors (Heat, Friction, Position)

    """

    def __init__(self, n_things: int = 20, unique_ids: list[ThingID] = None,
                 filter_factors: Optional[list[Type[Factor]]] = None):
        factors = [Heat, Friction, Position]
        super().__init__(n_things=n_things,
                         unique_ids=unique_ids,
                         factors=factors,
                         required_factors=factors,
                         filter_factors=filter_factors)


class MultimodalObservation(Observation):
    """Combine observation objects for multimodal observations.

    """

    def __init__(self, *observations: Observation):
        self._observations = list(observations)
        super().__init__()

    def _make_observation_space(self):
        self._observation_space = Tuple(
            obs.observation_space for obs in self._observations
        )

    def add_modality(self, observation_modality: Observation) -> None:
        """Adds another observation modality.

        :param observation_modality: Additional modality to add.
        :return: None
        """
        self._observations.append(observation_modality)
        self._make_observation_space()

    def render(self, *args, **kwargs):
        """In case we're using multimodal with one of the modes being
        pixel-based.

        Uses the first renderable observation.

        :return: img if one modality is pixel-based.
        """

        for observation in self._observations:
            if hasattr(observation, 'render'):
                return observation.render(*args, **kwargs)
        raise AttributeError('No renderable observation available.')

    def show(self, *args, **kwargs):
        """In case we're using multimodal with one of the modes being
        pixel-based.

        Uses the first showable observation.

        :return: None
        """
        for observation in self._observations:
            if hasattr(observation, 'show'):
                return observation.show(*args, **kwargs)
        raise AttributeError('No showable observation available.')

    def add_text(self, *args, **kwargs):
        """In case we're using multimodal with one of the modes being
        pixel-based.

        Uses the first observation where add_text works.

        :return: None
        """
        for observation in self._observations:
            if hasattr(observation, 'add_text'):
                return observation.add_text(*args, **kwargs)
        raise AttributeError('No showable observation available.')

    def sample(self) -> None:
        for observation in self._observations:
            observation.sample()

    def reset(self) -> None:
        for observation in self._observations:
            observation.reset()

    def __call__(self, state_dict: dict) -> tuple[np.ndarray, ...]:
        return tuple(obs(state_dict) for obs in self._observations)


def make_stacked_observation(observation_class: Type[Observation],
                             n_stack: int):
    """Makes a stacked version of `observation_class`.

    Note: this stacks according to the sim state buffer. This buffer must be
        as large as n_stack.

    :param observation_class: (Observation) observation class to stack.
    :param n_stack: (int) stack size.
    :return: (StackedC) a subclass of `observation_class` with stacking.
    """
    assert issubclass(observation_class, Observation), observation_class

    if isinstance(observation_class, MultimodalObservation):
        raise NotImplementedError('Stacking not supported (yet) of multimodal '
                                  'observations, create a multimodal '
                                  'observation from stacked observations '
                                  'instead.')

    class StackedC(observation_class):
        def __init__(self, *args, **kwargs):
            self.n_stack = n_stack
            super().__init__(*args, **kwargs)

        def _make_observation_space(self):
            super()._make_observation_space()
            assert type(self._observation_space) == Box

            high = self._observation_space.high.max()
            low = self._observation_space.low.min()
            shape = (self.n_stack, *self._observation_space.shape)
            dtype = self._observation_space.dtype
            self._observation_space = Box(low, high, shape=shape, dtype=dtype)

        def __call__(self, *args, **kwargs):
            states = self.sim.pull_buffer(self.n_stack)
            obses = []
            for state in states:
                obs = super().__call__(state)
                obses.append(obs)

            return np.stack(obses, axis=0)

    return StackedC
