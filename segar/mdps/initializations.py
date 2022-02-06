"""Module for initialization components

Initializations set up the environment, including ensuring appropriate
objects and tiles for task are present.

"""

__all__ = ('Initialization', 'ArenaInitialization')

from copy import deepcopy
from typing import Optional, Type, Union

from segar import get_sim
from segar.factors import Noise, Position, Deterministic, ID, Factor
from segar.rules import Prior, Transition
from segar.sim import Simulator
from segar.things import Entity, ThingFactory


class Initialization:
    """Abstract initialization class.

    """
    def __init__(self):
        self._sim = None

    def _read_config(self, config: dict = None) -> None:
        """Handler for user-friendly configurations.

        Must be implemented to be useful.

        :param config: Dictionary of configurations.
        """

        raise NotImplementedError('Configuration handler not implemented.')

    @property
    def sim(self):
        if self._sim is None:
            return get_sim()
        return self._sim

    def set_sim(self, sim: Simulator) -> None:
        self._sim = sim

    @property
    def initial_state(self) -> list[Entity]:
        raise NotImplementedError('`initial_state` not implemented.')

    def sample(self) -> None:
        """Samples the arena.

        This is where all randomization of the arena parameters should be put.

        """
        pass

    def get_dists_from_init(self) -> dict[ID, dict[Type[Factor], Noise]]:
        raise NotImplementedError

    def __call__(self, init_things: Optional[list[Entity]] = None):
        """Sets the simulator according to the initialization.

        :param init_things: Optional set of initial things to force
            initialization.
        """
        raise NotImplementedError


class ArenaInitialization(Initialization):
    """General class for handling arena initializations.

    """

    def __init__(self, config: dict = None, enforce_distances: bool = True,
                 min_distance: float = 0.1):
        """

        :param config: Configuration file.
        :param enforce_distances: Whether to enforce that thing centers are
            at least min_distance apart. This includes walls.
        :param min_distance: Minimum allowable distance between things.
        """

        super().__init__()

        self._enforce_distances = enforce_distances
        self._min_distance = min_distance

        # These are read in by the config
        self._config = dict()
        self._numbers: list[Union[Type[Entity], ThingFactory],
                            Union[int, Noise]] = []
        self._priors: list[Prior] = []
        self._positions: list[Union[Type[Entity], ThingFactory],
                              list[list[float, float]]]
        self._things: list[Entity] = []
        self._inits: list[Transition] = []

        self._read_config(**config)

    def _read_config(self,
                     priors: list[Prior] = None,
                     numbers: tuple[Union[Type[Entity], ThingFactory],
                                    Union[int, Noise]] = None,
                     positions: list[Union[Type[Entity], ThingFactory],
                                     list[list[float, float]]] = None) -> None:
        """Reads the configurations, which should have information about
            priors, numbers of things, and positions.

        :param priors: list of Prior objects for initialization.
        :param numbers: Numbers of objects, as (N, type)
        :param positions: Positions of objects, good for fixed /
            deterministic positions.
        """
        if numbers is None:
            raise ValueError('Initialization must specify numbers of things.')

        priors = priors or []
        numbers = numbers or []
        positions = positions or []

        self._priors = deepcopy(priors)
        self._numbers = deepcopy(numbers)
        self._positions = deepcopy(positions)

    @property
    def boundaries(self) -> tuple[float, float]:
        return self.sim.boundaries

    def set_sim(self, sim):
        super().set_sim(sim)
        for prior in self._priors:
            prior.set_sim(self.sim)

    def sample(self, max_iterations: int = 100) -> list[Entity]:
        """Samples things and their parameters.

        This method resamples if things overlap under some tolerance threshold.

        :param max_iterations: Max number of iterations to try resampling
            positions until they are valid.
        :return: List of initial things (initial state).
        """

        for prior in self._priors:
            prior.set_sim(self.sim)

        def bad_positions(pos1: Position, pos2: Position):
            return (self._enforce_distances and
                    (pos1 - pos2).norm() <= self._min_distance)

        def check_all_positions(plist: list[Position], fail_on_check=False):
            for i, pos1 in enumerate(plist):
                for pos2 in plist[i + 1:]:
                    if bad_positions(pos1, pos2):
                        if fail_on_check:
                            raise ValueError(
                                f'Deterministic positions {pos1} and '
                                f'{pos2} too close with current '
                                f'distance enforcement ('
                                f'{(pos1 - pos2).norm()} '
                                f'<= {self._min_distance}).')
                        return False
            return True

        positions_ok = True
        things = []
        inits = []
        positions = []

        # Here we sample from the number of things, followed by their
        # factors. If the positions sampled lead to things being too close,
        # resample.
        for i in range(max_iterations):
            self.sim.reset()
            things = []
            for cls, positions in self._positions:
                for pos in positions:
                    things.append(cls({Position: pos}))

            for factor_type, number in self._numbers:
                if isinstance(number, Noise):
                    number = number.sample().value

                for _ in range(number):
                    if isinstance(factor_type, ThingFactory):
                        cls = factor_type()
                    else:
                        cls = factor_type
                    things.append(cls())

            inits = self.sim.get_rule_outcomes(
                self._priors, things_to_apply_on=things)
            for init in inits:
                init()

            positions = [thing[Position] for thing in things
                         if Position in thing]
            positions_ok = check_all_positions(positions)
            if positions_ok:
                break

        if not positions_ok:
            raise RuntimeError(f'Could not ensure min distance in things '
                               f'within {max_iterations} iterations. Last '
                               f'positions attempted were {positions}.')

        self._inits = inits
        self._things = things
        self.sim.shuffle_order(things)
        return self._things

    def get_dists_from_init(self) -> dict[ID, dict[Type[Factor], Noise]]:
        dists = {}
        for thing in self._things:
            thing_dists = {}
            for factor_type, factor in thing.factors.items():
                for init in self._inits:
                    if factor is init.factor:
                        prior: Prior = init.rule
                        thing_dists[factor_type] = prior.distribution

                if factor_type not in thing_dists:
                    thing_dists[factor_type] = Deterministic(factor.value)
            dists[thing[ID]] = thing_dists

        return dists

    def set_arena(self, init_things: Optional[list[Entity]] = None) -> None:
        """Adds the tiles from the arena.

        :param init_things: Optional list of things to initialize arena with.
        """
        self.sim.reset()

        if init_things is None:
            for thing in self._things:
                self.sim.adopt(thing.copy())
        else:
            for thing in init_things:
                self.sim.adopt(thing.copy())

    @property
    def initial_state(self) -> list[Entity]:
        """Provides the initial state in terms of a list of entities.

        Note: if initialization was not sampled from, this will be empty.

        :return: List of initial entities.
        """
        return self._things

    def __call__(self, init_things: Optional[list[Entity]] = None) -> None:
        self.set_arena(init_things=init_things)
        self.sim.fix_overlaps()
