from __future__ import annotations

__copyright__ = "Copyright (c) Microsoft Corporation and Mila - Quebec AI Institute"
__license__ = "MIT"
"""Simulator

"""

__all__ = ("Simulator",)

from collections import deque
from copy import deepcopy
import logging
import pickle
import random
import time
from typing import Any, Dict, List, Optional, Tuple, Type, Union
import warnings

import numpy as np

from segar import set_sim, timeit
from segar.factors import (
    Factor,
    Floor,
    Friction,
    Charge,
    ID,
    InfiniteEnergy,
    Label,
    Magnetism,
    Mass,
    Order,
    Position,
    Shape,
    StoredEnergy,
    Text,
    Velocity,
    FACTORS,
    FACTOR_DEFAULTS,
)
from segar.things import (
    Bumper,
    Charger,
    Damper,
    Entity,
    FireTile,
    Hole,
    MagmaTile,
    Magnet,
    Object,
    SandTile,
    Tile,
)
from segar.rules import (
    colliding,
    overlaps,
    DidNotMatch,
    DidNotPass,
    Differential,
    Rule,
    Transition,
    TransitionFunction,
    move,
    lorentz_law,
    apply_friction,
    apply_burn,
    stop_condition,
    kill_condition,
    consume,
    accelerate,
)
from segar.rules.collisions import (
    overlap_time,
    object_collision,
    overlap_time_wall,
    wall_collision,
    overlaps_wall,
    fix_overlap_wall,
    fix_overlap_objects,
)
from segar.types import ThingID, Time
from segar.parameters import (
    Framerate,
    FloorFriction,
    Gravity,
    MinMass,
    MaxVelocity,
    WallDamping,
    MinVelocity,
)

from segar.things.boundaries import Wall, SquareWall

logger = logging.getLogger(__name__)


first_factors = [ft for ft in FACTORS if ft not in (Velocity, Position)]

_DEFAULT_FACTOR_UPDATE_ORDER = (first_factors, [Velocity])
DEFAULT_RULES = [
    move,
    lorentz_law,
    apply_friction,
    apply_burn,
    stop_condition,
    kill_condition,
    consume,
    accelerate,
]
_PRECISION = 1e-6  # For collision checks.

FactorOrder = tuple[list[Type[Factor]], ...]


class Simulator:
    """The simulator.

    """

    def __init__(
        self,
        boundaries: Tuple[float, float] = (-1, 1),
        framerate: int = FACTOR_DEFAULTS[Framerate],
        friction: float = FACTOR_DEFAULTS[FloorFriction],
        wall_damping: float = FACTOR_DEFAULTS[WallDamping],
        gravity: float = FACTOR_DEFAULTS[Gravity],
        min_mass: float = FACTOR_DEFAULTS[MinMass],
        max_velocity: float = FACTOR_DEFAULTS[MaxVelocity],
        min_velocity: float = FACTOR_DEFAULTS[MinVelocity],
        safe_mode: bool = False,
        save_path: str = None,
        state_buffer_length: int = 1,
        rules: Optional[List[Rule]] = None,
        factor_update_order: FactorOrder = _DEFAULT_FACTOR_UPDATE_ORDER,
        local_sim: bool = False,
    ):
        """

        :param boundaries: Boundaries of the walls/
        :param framerate: Framerate of the video.
        :param friction: Base friction of the environment.
        :param wall_damping: Damping factor of walls.
        :param gravity: Gravity constant for friction.
        :param safe_mode: Extra checks to make sure simulator is working
        properly.
        :param save_path: Path to save simulator to.
        :param state_buffer_length: Length of state buffer, e.g.,
            for stacking.
        :param rules: Optional set of rules.
        :param factor_update_order: List of sets of factors to update in
            order when calling the transition function.
        :param local_sim: Whether the sim is local, i.e., will not replace
            or set global one.
        """

        super().__init__()

        # Arena boundaries and sizes
        self.boundaries = boundaries
        self.arena_size = boundaries[1] - boundaries[0]

        # Simulator objects and tiles
        self._walls = dict()
        self._things: dict[ThingID, Entity] = dict()
        self._sorted_things = None
        self._thing_histories = dict()

        # Physics parameters
        self.parameters = {
            Gravity: Gravity(gravity),
            WallDamping: WallDamping(wall_damping),
            Friction: Friction(friction),
            Time: Time(1.0 / framerate),
            MinMass: MinMass(min_mass),
            MaxVelocity: MaxVelocity(max_velocity),
            MinVelocity: MinVelocity(min_velocity),
        }
        self._rules = []
        self.set_rules(rules or DEFAULT_RULES)
        self._valid_ep_rules = None
        self._factor_update_order = factor_update_order

        # Sim parameters
        self._save_path = save_path
        self._state_buffer = deque(maxlen=state_buffer_length)
        self._results = {}
        self.scaling_factor = np.sqrt(2 * (boundaries[1] - boundaries[0]) ** 2)
        self.safe_mode = safe_mode
        self.timer = 0
        self.time = time.time()
        if not local_sim:
            set_sim(self)
        self.reset()

    # Results and simulator benchmarking.
    def update_results(self, key: str, value: float) -> None:
        """Updates results being collected by the sim.

        :param key: Key for this result.
        :param value: Value for this result.
        """
        if "time" in key:
            key_ = key.replace("time", "count")
        else:
            key_ = key + "_count"
        if key in self._results:
            self._results[key].append(value)
            self._results[key_] += 1
        else:
            self._results[key] = [value]
            self._results[key_] = 1

    def summarize_results(self, d: int = 1) -> dict[str, str]:
        """Summarizes results collected.

        :param d: Scaling factor for results.
        :return: Dictionary of results.
        """
        results = {}
        total_time = sum(np.sum(v) / float(d) for v in self._results.values())
        for k, v in self._results.items():
            if "time" in k:
                res = np.sum(v) / (total_time * float(d))
                results[k] = f"{res:.2}"
            else:
                results[k] = f"{np.mean(v) / float(d):.2}"

        return results

    def save(self, out_path: Optional[str] = None) -> None:
        """Saves the simulator.

        Uses pickle, so saved simulator may be out of date with code changes.

        TODO: save with current git hash.

        """
        save_path = out_path or self._save_path
        if save_path is None:
            raise ValueError("No save path given.")
        logger.info(f"Saving simulator to {save_path}.")
        with open(save_path, "wb") as f:
            pickle.dump(self, f)

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        del state["_rules"]
        del state["_valid_ep_rules"]
        warnings.warn(
            "Removing rules while pickling. Default rules will be "
            "reloaded. Rule customization needs to be redone "
            "manually."
        )
        return state

    def __setstate__(self, state: dict):
        warnings.warn(
            "Any custom rule sets were removed while pickling. "
            "Default rules will be reloaded. Rule customization "
            "needs to be redone manually."
        )
        self.__dict__.update(**state)
        self._rules = DEFAULT_RULES
        self._valid_ep_rules = None

    def reset(self) -> None:
        """Resets all of the object, tiles, and walls.

        """

        self._things.clear()
        self._sorted_things = None
        self._thing_histories.clear()
        self._valid_ep_rules = None
        # Todo: This is built-in, but we want this to be controlled by the
        #  initializer.
        self._walls[0] = SquareWall(self.boundaries, damping=self.parameters[WallDamping])
        self.timer = 0
        self.time = time.time()
        self._results.clear()
        # Add an entity for global friction.
        self.adopt(Entity({Friction: self.parameters[Friction], ID: "global_friction"}))

    # Thing management
    @property
    def thing_ids(self) -> list[ThingID]:
        return list(self._things.keys())

    def thing_ids_with_factor(self, *factor_types: Type[Factor]) -> list[ThingID]:
        """Returns the ids of things that have all specified factors.

        :param factor_types: List of factor types.
        :return: List of ids.
        """
        ids = []
        for k, v in self.things.items():
            has_factor = [v.has_factor(factor_type) for factor_type in factor_types]
            if all(has_factor):
                ids.append(k)
        return ids

    def things_with_factor(self, *factor_types: Type[Factor]) -> dict[ThingID, Entity]:
        """Returns a dictionary of things with all specified factors.

        :param factor_types: List of factor types.
        :return: Dictionary of id, thing with specified factors.
        """
        things_ = {}
        for k, thing in self._things.items():
            has_factor = [thing.has_factor(factor_type) for factor_type in factor_types]
            if all(has_factor):
                things_[k] = thing
        return things_

    def things_without_factor(self, *factor_types: Type[Factor]) -> dict[ThingID, Entity]:
        """Returns a dictionary of things without all specified factors.

        :param factor_types: List of factor types.
        :return: Dictionary of id, thing without specified factors.
        """
        things_ = {}
        for k, thing in self._things.items():
            has_factor = [not thing.has_factor(factor_type) for factor_type in factor_types]
            if all(has_factor):
                things_[k] = thing
        return things_

    def get_owner(self, factor: Factor) -> Union[Entity, None]:
        """Returns the thing that owns the queried factor.

        :param factor: Factor to search for.

        """
        for thing in self.things.values():
            if thing.contains(factor):
                return thing
        return None

    def get_paired_factor(self, key: Factor, query: Type[Factor]) -> Union[Factor, None]:
        """Attempts to find the corresponding factor from the type and
            another factor that belongs to the same thing.

        :param key: Key factor to reference factor being drawn out.
        :param query: Type of factor being drawn.

        """
        owner = self.get_owner(key)
        if owner.has_factor(query):
            return owner[query]
        return None

    @property
    def things(self) -> Dict[ThingID, Entity]:
        if self._sorted_things is None:
            self.sort_things()
        return self._sorted_things

    def get_things(self, *thing_ids: ThingID) -> List[Entity]:
        """Returns all things from IDs.

        :param thing_ids: List of IDs
        :return: List of things with IDs.
        """
        return [self.things[tid] for tid in thing_ids]

    @property
    def walls(self) -> Dict[ThingID, Wall]:
        return self._walls

    def sort_things(self) -> None:
        """Sorts the thing dictionary according to the Order factor.

        """
        sim_things = self.things_with_factor(Order)
        unordered_things = self.things_without_factor(Order)
        sorted_things = dict(
            (k, v) for k, v in sorted(sim_things.items(), key=lambda item: item[1].factors[Order])
        )
        for k, v in unordered_things.items():
            sorted_things[k] = v
        self._sorted_things = sorted_things

    def shuffle_order(self, things: Optional[List[Entity]]) -> None:
        """Shuffles the order of the things.

        :param things: Optional list of things to shuffle.
        """
        things = things or list(self._things.values())
        ordered_things = list(thing for thing in things if thing.has_factor(Order))
        orders = list(range(len(ordered_things)))
        random.shuffle(orders)
        for thing, order in zip(ordered_things, orders):
            if thing[Order] == 0:
                thing.set_factor(Order, order, allow_in_place=True)

    def get_new_id(self, thing: Entity) -> None:
        """Generates a new ID for a thing.

        :param thing: Thing to generate ID for.
        :param thing_id: Optional specified thing ID.
        :return: The ID.
        """

        id_thing = thing[ID].value

        if isinstance(id_thing, str) and (id_thing.isdigit() or id_thing[1:].isdigit()):
            id_thing = int(id_thing)

        if isinstance(id_thing, int) and id_thing == -1:
            id_thing = 0

        if isinstance(id_thing, int) and id_thing in self.things.keys() or id_thing is None:
            # Give a unique ID
            ids = [k for k in self.things.keys() if isinstance(k, int)]
            max_id = max(ids)
            id_thing = max_id + 1

        if not isinstance(id_thing, (int, str, ID)):
            raise TypeError(id_thing)
        with thing.in_place():
            thing[ID] = ID(id_thing)

    # Adoption of things into sim.
    def adopt(self, thing: Entity) -> ThingID:
        """Adds a thing to the simulator.

        :param thing: Thing to adopt.
        :param thing_id: Unique id to give the thing.
        :return: Valid unique id.
        """

        self.get_new_id(thing)

        self._sorted_things = None

        if thing[ID] in self._things:
            raise KeyError(
                f"{thing[ID]} cannot be used as `unique_id` " f"because it already is in use."
            )

        self._things[thing[ID].value] = thing
        return thing[ID].value

    def copy(self):
        raise NotImplementedError("Do not copy the sim.")

    # Thing states
    @property
    def thing_states(self) -> Dict[ThingID, Dict[Type[Factor], Factor]]:
        return dict((tid, thing.state) for tid, thing in self.things.items())

    def change_thing_state(
        self, thing_id: Union[ThingID, ID], factor_type: Type[Factor], value: Any,
    ) -> None:
        """Change an thing state.

        :param thing_id: ID (key) of thing.
        :param factor_type: Factor to change.
        :param value: Value to change attribute to.
        """
        thing = self.things[thing_id]
        thing.set_factor(factor_type, value, allow_in_place=True)

    @property
    def state(self):
        state = dict(
            boundaries=self.boundaries,
            parameters=self.parameters,
            save_path=self._save_path,
            scaling_factor=self.scaling_factor,
            safe_mode=self.safe_mode,
            timer=self.timer,
            walls=dict((tid, wall.state) for tid, wall in self.walls.items()),
            things=self.thing_states,
        )
        return deepcopy(state)

    def has_same_state(self, state: dict) -> bool:
        """Compares sim state with another sim's state

        :param state: Dictionary state for other sim.
        :return: Is equal.
        """
        return states_are_equal(self.state, state)

    # Relations
    @staticmethod
    def is_over(thing1: Entity, thing2: Entity) -> bool:
        if not thing2.has_factor(Floor):
            return False
        if thing1.has_factor(Floor):
            return False

        thing1_shape = thing1[Shape]
        thing2_shape = thing2[Shape]

        x1 = thing1[Position]
        x2 = thing2[Position]
        return thing1_shape.overlaps(thing2_shape, normal_vector=(x2 - x1))

    def is_on(self, thing1: Entity, thing2: Entity) -> bool:
        """Checks if one entity is on another entity.

        Positionless entities always return True.

        :param thing1: First entity.
        :param thing2: Second entity.
        :return: True if first entity is on the second.
        """
        if not thing2.has_factor(Position):
            return True
        for thing in self.things.values():
            if thing1 is thing:
                continue
            if self.is_over(thing1, thing):
                if thing is thing2:
                    return True
                else:
                    return False
        return False

    # Rule management
    @property
    def rules(self) -> List[Rule]:
        return self._rules

    def set_rules(self, rules: List[Rule]) -> None:
        """Sets the rules of the simulator.

        We need to copy the rules and reference this sim as there may be
            multiple sims running and rules something references the sim.

        :param rules: List of rules.
        """
        self._rules = [r.copy_for_sim(self) for r in rules]

    def add_rule(self, rule: Rule) -> None:
        """Adds a single rule to the simulator.

        :param rule: Rule to add.
        """
        if rule not in self.rules:
            self.rules.append(rule)

    def remove_rule(self, rule_name: str) -> None:
        """Removes a single rule from the simulator by name.

        :param rule_name: Name of rule (function name).
        """
        idx = None
        for i, rule in enumerate(self.rules):
            if rule.__name__ == rule_name:
                idx = i
        if idx is not None:
            self._rules.pop(idx)
        else:
            raise ValueError(f"No such rule in sim `{rule_name}`.")
        self._valid_ep_rules = None

    @timeit
    def get_all_thing_tuples(
        self, order: int = 1, things: Optional[List[Entity]] = None
    ) -> List[List[Entity]]:
        """For a given number of things, return the list of all tuples of
            things of that size.

        :param order: Order of the tuples (number of things).
        :param things: Optional list of things to draw tuples from.
        :return: List of lists (tuples) of things.
        """
        tuples = []
        things = things or list(self.things.values())
        for _ in range(order):
            tuples_ = []
            if len(tuples) == 0:
                tuples_ = [[thing] for thing in things]
            else:
                for tup in tuples:
                    for thing in things:
                        if thing not in tup:
                            tup_ = tup + [thing]
                            tuples_.append(tup_)
            tuples = tuples_

        return tuples

    @timeit
    def get_rule_outcomes(
        self,
        rules_to_apply: List[TransitionFunction],
        things_to_apply_on: Optional[List[Entity]] = None,
    ) -> List[Transition]:
        """Calculates the outcomes of all of the rules.

        :param rules_to_apply: List of rules to calculate outcomes.
        :param things_to_apply_on: Things to apply rules to.
        :return: List of transitions (outcomes).
        """
        things_to_apply_on = things_to_apply_on or self.things.values()

        rules_and_args = self.get_valid_rules(
            rules_to_apply=rules_to_apply, things_to_apply_on=things_to_apply_on,
        )
        final_outcomes = self.get_final_outcomes(rules_and_args)

        return list(final_outcomes.values())

    @timeit
    def get_final_outcomes(
        self,
        rules_and_args: List[Tuple[TransitionFunction, List]],
        affected_factors: Optional[list[Type[Factor]]] = None,
    ) -> dict[Factor, Transition]:
        """Find the final outcomes of a set of rules and arguments.

        :param rules_and_args: List of tuples containing rules and arguments
            to apply.
        :param affected_factors: Optional list of factors to apply outcomes
            to. If not provided, apply to all.
        :return: Dictionary of transitions (value) for each factor (key).
        """
        # Gather all of the transitions for all of the factors.
        factor_outcomes = {}
        for rule, args in rules_and_args:

            if affected_factors is not None:
                ret_factor_types = rule.signature_info["return_factor_types"]
                # Check if rule contains factors in effected list
                if len(set(ret_factor_types).intersection(set(affected_factors))) == 0:
                    continue
            res = rule(*args)
            if isinstance(res, DidNotMatch):
                raise ValueError(f"Incorrect args {args} provided for rule " f"{rule}.")
            if not (isinstance(res, DidNotPass) or res is None):
                if not isinstance(res, tuple):
                    res = [res]

                for res_ in res:
                    factor = res_.factor
                    if factor in factor_outcomes:
                        factor_outcomes[factor].append(res_)
                    else:
                        factor_outcomes[factor] = [res_]

        # Conflict resolution. One transition per factor.
        final_outcomes = {}
        for factor, outcomes in factor_outcomes.items():
            # Conflict resolution is handled by Transition.__add__
            outcome = sum(outcomes)
            final_outcomes[factor] = outcome

        return final_outcomes

    @timeit
    def get_valid_rules(
        self,
        rules_to_apply: Optional[List[TransitionFunction]] = None,
        things_to_apply_on: Optional[List[Entity]] = None,
    ) -> List[Tuple[TransitionFunction, list]]:
        """Find all of valid rules and their arguments given a list of things.

        Note: this function only needs to be used once per episode.

        :param rules_to_apply: Option set of rules to check.
        :param things_to_apply_on: Optional set of things to apply rules to.
        :return: List of valid rules and arguments.
        """
        rules_to_apply = rules_to_apply or self.rules
        things_to_apply_on = things_to_apply_on or self.things.values()

        valid_rules = []

        all_tuples = {}

        for rule in rules_to_apply:
            n_objects = rule.n_objects
            if n_objects not in all_tuples:
                all_tuples[n_objects] = self.get_all_thing_tuples(
                    n_objects, things=things_to_apply_on
                )

            tuples = all_tuples[n_objects]
            for args in tuples:
                args: List = args[:]
                parameters = rule.parameters
                for param in parameters:
                    args.append(self.parameters[param])
                res = rule(*args)
                if not isinstance(res, DidNotMatch):
                    valid_rules.append((rule, args))

        return valid_rules

    @timeit
    def apply_rules(self, dt: Time = None) -> None:
        """Applies all non-position rules valid to the episode.

        :param dt: Time scale.
        """
        dt = dt or self.parameters[Time]

        if self._valid_ep_rules is None:
            self._valid_ep_rules = self.get_valid_rules()
        for factor_types in self._factor_update_order:
            if Position in factor_types:
                raise ValueError(
                    f"{Position} factor type must be modified " f"only during collision management."
                )
            rule_outcomes = self.get_final_outcomes(
                self._valid_ep_rules, affected_factors=factor_types
            )

            for res in rule_outcomes.values():
                if isinstance(res, Differential):
                    res(dt)
                elif res is not None:
                    res()
                del res

    @timeit
    def get_position_transitions(self) -> dict[Factor, Transition]:
        """Find all position-relevant transitions according to the rules.

        Position rules are separate from other rules because of collisions.

        :return: Dictionary of factors (keys) and their positional transitions.
        """
        if self._valid_ep_rules is None:
            self._valid_ep_rules = self.get_valid_rules()

        position_transitions = self.get_final_outcomes(
            self._valid_ep_rules, affected_factors=[Position]
        )

        return position_transitions

    # API for adding build-in things
    def add_object(
        self,
        position: np.ndarray,
        initial_factors: Dict[Type[Factor], Any],
        unique_id: Optional[ThingID] = None,
        label: Optional[str] = None,
        text: Optional[str] = "O",
    ) -> ThingID:
        """Adds an object to the sim.

        :param position: Position of the object.
        :param initial_factors: Initial factors for the object.
        :param unique_id: Optional unique id for object.
        :param label: Optional label for object.
        :param text: Optional text to be used in rendering.
        :return: object id (key)
        """
        initial_factors = initial_factors or {}
        initial_factors[Position] = position
        if label is not None:
            initial_factors[Label] = label
        if text is not None:
            initial_factors[Text] = text

        o = Object(initial_factors, unique_id=unique_id, sim=self)
        return o[ID].value

    def add_tile(
        self,
        position: np.ndarray,
        initial_factors: Dict[Type[Factor], Any],
        unique_id: Optional[ThingID] = None,
        label: Optional[str] = None,
        text: Optional[str] = "T",
    ) -> ThingID:
        """Adds a tile to the sim.

        :param position: Position of the tile.
        :param initial_factors: Initial factors for the tile.
        :param unique_id: Optional unique id for tile.
        :param label: Optional label for tile.
        :param text: Optional text to be used in rendering.
        :return: tile id (key)
        """
        initial_factors = initial_factors or {}
        initial_factors[Position] = position
        if label is not None:
            initial_factors[Label] = label
        if text is not None:
            initial_factors[Text] = text

        o = Tile(initial_factors, unique_id=unique_id, sim=self)
        return o[ID].value

    def add_sand(
        self,
        position: np.ndarray,
        friction: float = 0.4,
        text: str = "S",
        unique_id: ThingID = None,
        label: str = None,
        initial_factors: Optional[Dict[Type[Factor], Any]] = None,
    ) -> ThingID:
        """Adds sand to simulator.

        :param position: Position of sand.
        :param friction: Friction of the sand.
        :param unique_id: Optional unique id for sand.
        :param label: Optional label for sand.
        :param text: Optional text to be used in rendering.
        :param initial_factors: Optional initial factors for sand.

        :return: sand id (key).
        """
        initial_factors = initial_factors or {}
        if Friction not in initial_factors:
            initial_factors[Friction] = friction
        initial_factors[Position] = position
        if label is not None:
            initial_factors[Label] = label
        if text is not None:
            initial_factors[Text] = text

        o = SandTile(initial_factors, unique_id=unique_id, sim=self)
        return o[ID].value

    def add_magma(
        self,
        position: np.ndarray,
        text: str = None,
        unique_id: ThingID = None,
        label: str = None,
        initial_factors: Optional[Dict[Type[Factor], Any]] = None,
    ) -> ThingID:
        """Adds magma to simulator.

        :param position: Position of magma.
        :param unique_id: Optional unique id for magma.
        :param label: Optional label for magma.
        :param text: Optional text to be used in rendering.
        :param initial_factors: Optional initial factors for magma.

        :return: magma id (key).
        """
        initial_factors = initial_factors or {}
        initial_factors[Position] = position
        if label is not None:
            initial_factors[Label] = label
        if text is not None:
            initial_factors[Text] = text

        o = MagmaTile(initial_factors, unique_id=unique_id, sim=self)
        return o[ID].value

    def add_fire(
        self,
        position: np.ndarray,
        text: str = "F",
        unique_id: ThingID = None,
        label: str = None,
        initial_factors: Optional[Dict[Type[Factor], Any]] = None,
    ) -> ThingID:
        """Adds fire to simulator.

        :param position: Position of fire.
        :param unique_id: Optional unique id for fire.
        :param label: Optional label for fire.
        :param text: Optional text to be used in rendering.
        :param initial_factors: Optional initial factors for fire.

        :return: fire id (key).
        """
        initial_factors = initial_factors or {}
        initial_factors[Position] = position
        if label is not None:
            initial_factors[Label] = label
        if text is not None:
            initial_factors[Text] = text

        o = FireTile(initial_factors, unique_id=unique_id, sim=self)
        return o[ID].value

    def add_hole(
        self,
        position: np.ndarray,
        text: str = "H",
        unique_id: ThingID = None,
        label: str = None,
        initial_factors: Optional[Dict[Type[Factor], Any]] = None,
    ) -> ThingID:
        """Adds hole to simulator.

        :param position: Position of hole.
        :param unique_id: Optional unique id for hole.
        :param label: Optional label for hole.
        :param text: Optional text to be used in rendering.
        :param initial_factors: Optional initial factors for hole.

        :return: hole id (key).
        """
        initial_factors = initial_factors or {}
        initial_factors[Position] = position
        if label is not None:
            initial_factors[Label] = label
        if text is not None:
            initial_factors[Text] = text

        o = Hole(initial_factors, unique_id=unique_id, sim=self)
        return o[ID].value

    def add_ball(
        self,
        position: np.ndarray,
        initial_factors: Dict[Type[Factor], Any] = None,
        unique_id: Optional[ThingID] = None,
        label: Optional[str] = None,
        text: Optional[str] = "B",
    ) -> ThingID:
        """Adds ball to the simulator.

        :param position: Position of the ball.
        :param initial_factors: Initial factors for the ball.
        :param unique_id: Optional unique id for ball.
        :param label: Optional label for ball.
        :param text: Optional text to be used in rendering.
        :return: ball id (key)
        """
        initial_factors = initial_factors or {}
        initial_factors[Position] = position
        if label is not None:
            initial_factors[Label] = label
        if text is not None:
            initial_factors[Text] = text

        o = Object(initial_factors, unique_id=unique_id, sim=self)
        return o[ID].value

    def add_bumper(
        self,
        position: np.ndarray,
        stored_energy: float = 1.0,
        infinite_energy: bool = True,
        text: str = "U",
        unique_id: ThingID = None,
        label: str = None,
        initial_factors: Optional[Dict[Type[Factor], Any]] = None,
    ) -> ThingID:
        """Adds bumper to simulator.

        :param position: Position of bumper.
        :param stored_energy: Amount of energy stored by the bumper.
        :param infinite_energy: Whether the bumper can lose energy.
        :param unique_id: Optional unique id for bumper.
        :param label: Optional label for bumper.
        :param text: Optional text to be used in rendering.
        :param initial_factors: Optional initial factors for bumper.

        :return: bumper id (key).
        """
        initial_factors = initial_factors or {}
        initial_factors[Position] = position
        if label is not None:
            initial_factors[Label] = label
        if text is not None:
            initial_factors[Text] = text
        if StoredEnergy not in initial_factors:
            initial_factors[StoredEnergy] = stored_energy
        if InfiniteEnergy not in initial_factors:
            initial_factors[InfiniteEnergy] = infinite_energy

        o = Bumper(initial_factors, unique_id=unique_id, sim=self)
        return o[ID].value

    def add_damper(
        self,
        position: np.ndarray,
        stored_energy: float = -0.5,
        infinite_energy: bool = True,
        text: str = "D",
        unique_id: ThingID = None,
        label: str = None,
        initial_factors: Optional[Dict[Type[Factor], Any]] = None,
    ) -> ThingID:
        """Adds damper to simulator.

        :param position: Position of damper.
        :param stored_energy: Amount of energy stored by the damper.
        :param infinite_energy: Whether the damper can lose energy.
        :param unique_id: Optional unique id for damper.
        :param label: Optional label for damper.
        :param text: Optional text to be used in rendering.
        :param initial_factors: Optional initial factors for damper.

        :return: damper id (key).
        """
        initial_factors = initial_factors or {}
        initial_factors[Position] = position
        if label is not None:
            initial_factors[Label] = label
        if text is not None:
            initial_factors[Text] = text
        if StoredEnergy not in initial_factors:
            initial_factors[StoredEnergy] = stored_energy
        if InfiniteEnergy not in initial_factors:
            initial_factors[InfiniteEnergy] = infinite_energy

        o = Damper(initial_factors, unique_id=unique_id, sim=self)
        return o[ID].value

    def add_charger(
        self,
        position: np.ndarray,
        charge: float = -1.0,
        text: str = "Q",
        unique_id: ThingID = None,
        label: str = None,
        initial_factors: Optional[Dict[Type[Factor], Any]] = None,
    ) -> ThingID:
        """Adds charger to simulator.

        :param position: Position of charger.
        :param charge: Charge of the charger.
        :param unique_id: Optional unique id for charger.
        :param label: Optional label for charger.
        :param text: Optional text to be used in rendering.
        :param initial_factors: Optional initial factors for charger.

        :return: charger id (key).
        """
        initial_factors = initial_factors or {}
        initial_factors[Position] = position
        if label is not None:
            initial_factors[Label] = label
        if text is not None:
            initial_factors[Text] = text
        if Charge not in initial_factors:
            initial_factors[Charge] = charge

        o = Charger(initial_factors, unique_id=unique_id, sim=self)
        return o[ID].value

    def add_magnet(
        self,
        position: np.ndarray,
        magnetism: float = 1.0,
        text: str = "B",
        unique_id: ThingID = None,
        label: str = None,
        initial_factors: Optional[Dict[Type[Factor], Any]] = None,
    ) -> ThingID:
        """Adds magnet to simulator.

        :param position: Position of magnet.
        :param magnetism: Magnetism of the magnet.
        :param unique_id: Optional unique id for magnet.
        :param label: Optional label for magnet.
        :param text: Optional text to be used in rendering.
        :param initial_factors: Optional initial factors for magnet.

        :return: magnet id (key).
        """
        initial_factors = initial_factors or {}
        initial_factors[Position] = position
        if label is not None:
            initial_factors[Label] = label
        if text is not None:
            initial_factors[Text] = text
        if Magnetism not in initial_factors:
            initial_factors[Magnetism] = magnetism

        o = Magnet(initial_factors, unique_id=unique_id, sim=self)
        return o[ID].value

    # API for thing factors

    def add_force(
        self, thing_id: Union[ThingID, ID], force: Union[Tuple[float, float], np.ndarray],
    ) -> None:
        """ Apply a force to an object that results in a velocity that's
        relative to the object's mass

        Results in a change to the object's velocity.

        :param force: Force vector.
        :param thing_id: Object to apply force to.
        :return: None
        """
        # F = m*v
        if isinstance(thing_id, ID):
            thing_id = thing_id.value
        thing = self.things[thing_id]
        if thing[Mass] == 0:
            return

        try:
            with thing[Velocity].in_place():
                thing[Velocity] += np.array(force / thing[Mass].value)
        except KeyError:
            raise KeyError("Force can only be added to objects with mass " "and velocity.")

    def add_velocity(
        self, thing_id: ThingID, velocity: Union[Tuple[float, float], np.ndarray],
    ) -> None:
        """ Apply a velocity to an object. This is cumulative with the
            object's existing speed.

        (e.g. if the object's current velocity is (1,.5), then adding (-1, 0)
        will result in a velocity of (0,.5)). Results in a change to the
        object's velocity.

        :param velocity: Velocity vector.
        :param thing_id: Thing to apply velocity to.
        :return: None
        """
        thing = self.things[thing_id]
        try:
            with thing.in_place():
                thing[Velocity] += velocity

        except KeyError:
            raise KeyError("Force can only be added to objects with velocity.")

    def all_stopped(self) -> bool:
        min_velocity = self.parameters[MinVelocity]
        for thing in self.things.values():
            if thing.has_factor[Velocity] and thing[Velocity].norm() > min_velocity:
                return False
        return True

    # Extra functionality
    @timeit
    def l2_distance(self, thing1_id: Union[ThingID, ID], thing2_id: Union[ThingID, ID]):
        """L2 distance between things.

        :param thing1_id: ID (key) for thing 1.
        :param thing2_id: ID (key) for thing 2.
        :return: distance.
        """
        t0 = time.time()
        thing1 = self.things[thing1_id]
        thing2 = self.things[thing2_id]
        t1 = time.time()
        self.update_results("get_things_time", t1 - t0)
        pos_norm = (thing1[Position] - thing2[Position]).norm()
        t2 = time.time()
        self.update_results("compute_norm_time", t2 - t1)
        return pos_norm / self.scaling_factor

    def fix_overlaps(self):
        """For initialization. Move objects such that overlaps are gone.

        :return:
        """
        s = 0
        overlap_pair = None
        wall_overlap = False
        object_overlap = False

        while s < 200:
            shaped_things = self.things_with_factor(Shape, Position)
            # To randomize overlap fixes in case of cycles.
            query_things = list(shaped_things.values())[:]
            random.shuffle(query_things)

            wall_overlap = False
            object_overlap = False
            overlap_pair = None

            for thing in query_things:
                if not (wall_overlap or object_overlap):
                    for wall in self._walls.values():
                        if overlaps_wall(thing, wall):
                            fix_overlap_wall(thing, wall)
                            wall_overlap = True
                            overlap_pair = (thing, wall)

                if not (wall_overlap or object_overlap):
                    for other_thing in shaped_things.values():
                        if thing is other_thing:
                            continue
                        if isinstance(thing, Object) and isinstance(other_thing, Object):
                            if overlaps(thing, other_thing):
                                fix_overlap_objects(thing, other_thing)
                                object_overlap = True
                                overlap_pair = (thing, other_thing)
            if not (wall_overlap or object_overlap):
                break
            s += 1

        if wall_overlap:
            raise ValueError(
                f"Overlaps remain: {overlap_pair[0]}(Position="
                f"{overlap_pair[0][Position]}) and {overlap_pair[1]}("
                f"Wall). When this happens, it may be due to things in "
                f"your initialization sizes being too large for the sim. "
                f"Try reducing their sizes."
            )

        if object_overlap:
            raise ValueError(
                f"Overlaps remain: {overlap_pair[0]}(Position="
                f"{overlap_pair[0][Position]}) and "
                f"{overlap_pair[1]}(Position="
                f"{overlap_pair[1][Position]}). When this "
                f"happens, it may be due to things in your "
                f"initialization sizes being too large for the "
                f"sim. Try reducing their sizes."
            )

        return not (wall_overlap or object_overlap)

    def jiggle_all_velocities(self):
        """Jiggles object velocities randomly.

        :return: None
        """
        for thing in self.things.values():
            if Velocity in thing:
                thing[Velocity] += np.random.normal(0, 2.0, size=(2,))

    @timeit
    def _do_collision(
        self, position_transitions: dict[Factor, Transition]
    ) -> tuple[Time, Union[tuple, None]]:
        """Checks the first collision after performing a discrete time step.

        The simulator takes discrete time steps. As such, we detect if there
        were any collisions by checking overlap and finding the overlap that
        takes the most time to undo.

        :param position_transitions: Dictionary of positional transitions.
        :return: Time to reverse, tuple of colliding objects (if any else
            None).
        """

        max_time = 0.0
        colliding_pair = None

        # First get the objects that are moving from their Position
        # transitions.
        moving_objects: dict[Object, Transition] = {}
        for factor, transition in position_transitions.items():
            if isinstance(factor, Position):
                for thing in self.things.values():
                    if thing.contains(factor):
                        if isinstance(thing, Object):
                            moving_objects[thing] = transition
                        else:
                            raise ValueError()
                        break
            else:
                raise ValueError()

        # Check walls
        for obj, transition in moving_objects.items():
            for wall in self._walls.values():
                if overlaps_wall(obj, wall):
                    if isinstance(transition, Differential):
                        reverse_time = overlap_time_wall(obj, wall)
                        if reverse_time > max_time:
                            max_time = reverse_time
                            colliding_pair = (obj, wall)
                    else:
                        raise ValueError(
                            f"Cannot reverse or fix overlaps " f"from rule type {type(transition)}."
                        )

            for thing in self._things.values():
                if isinstance(thing, Object):
                    if thing is obj:
                        continue

                    if colliding(obj, thing):
                        if isinstance(transition, Differential):
                            reverse_time = overlap_time(obj, thing)

                            if reverse_time > max_time:
                                max_time = reverse_time
                                colliding_pair = (obj, thing)
                        else:
                            raise ValueError(
                                f"Cannot reverse or fix overlaps "
                                f"from rule type "
                                f"{type(transition)}."
                            )

        rdt = max_time
        return rdt, colliding_pair

    @timeit
    def _move_objects(self, dt: Optional[Time] = None) -> None:
        """Moves objects and handles collisions.

        Collisions are handled at a different time scale as everything else.

        :param dt: Time scale to move forward.
        """
        dt = dt or self.parameters[Time]

        # Get current change of position transitions.
        position_transitions = self.get_position_transitions()

        # Move forward by dt and check if there are any overlaps
        # between objects and objects or walls. If so, reverse dynamics
        # until the very first collision.

        for transition in position_transitions.values():
            if isinstance(transition, Differential):
                transition(dt)
            else:
                transition()

        # Due to the time scale of overlap checks, it is possible that
        # objects that don't overlap at t and t + dt
        # overlap at t + dt_rem. So we need to double check every time
        # we rewind if we didn't miss anything.
        rdt_total = Time(0.0)
        colliding_pair = None

        while True:
            rdt, colliding_pair_ = self._do_collision(position_transitions)
            if colliding_pair_ is None:
                break
            else:
                colliding_pair = colliding_pair_
                rdt_total += rdt
                # Rewind.
                for factor, transition in position_transitions.items():
                    if isinstance(transition, Differential):
                        transition(-rdt)

        # If there is a collision, apply collision dynamics, and move again
        # using the remainder time.
        if colliding_pair is not None:
            # Get collision transitions and reverse time.
            o1, o2 = colliding_pair

            if isinstance(o2, SquareWall):
                final_collisions = [wall_collision(o1, o2)]
            else:
                final_collisions = object_collision(o1, o2)

            # Apply collision transitions.
            for collision in final_collisions:
                collision()

            if not (-_PRECISION <= rdt_total < dt + _PRECISION):
                raise RuntimeError(
                    f"Collision error, rollback is out of "
                    f"bounds. If the error is small, "
                    f"try decreasing precision of check not("
                    f"{_PRECISION} <= {rdt_total} < {dt} + "
                    f"{_PRECISION})."
                )

            # Due to precision errors.
            rdt_total = max(Time(0.0), rdt_total)
            rdt_total = min(dt, rdt_total)

            self._move_objects(dt=rdt_total)

        if self.safe_mode:
            # Make sure there are no overlaps.
            for i, obj in enumerate(self.things.values()):
                for j, other_object in enumerate(self.things.values()):
                    if j <= i:
                        continue
                    if isinstance(obj, Object) and isinstance(other_object, Object):
                        assert other_object is not obj
                        if colliding(obj, other_object):
                            raise RuntimeError("Still overlapping!", (obj, other_object))

    def _update_histories(self, history_length=10):
        """Keeps a buffer of all of the objects.

        :param history_length: Max length of buffers.
        """

        for k, thing in self.things.items():
            thing_copy = thing.copy()
            if k in self._thing_histories.keys():
                self._thing_histories[k].append(thing_copy)
            else:
                self._thing_histories[k] = [thing_copy]
            self._thing_histories[k] = self._thing_histories[k][-history_length:]

    def rewind(self):
        """Rewinds objects from past histories.

        Used for debugging.

        """
        self._sorted_things = None
        self._valid_ep_rules = None

        for k in self._things.keys():
            try:
                obj = self._thing_histories[k].pop()
            except IndexError:
                raise IndexError("Rewinding too far.")
            self._things[k] = obj

    @timeit
    def _pre_collision_step(self):
        """Calculates dynamics and then steps forward.

        """
        # this is as cryptic as it gets... this calls the 4 methods of every
        # object in order
        self._update_histories()
        self.apply_rules()

    def limit_velocities(self) -> None:
        """Clips velocity norm according to the global max.

        """
        for thing in self.things.values():
            if thing.has_factor(Velocity):
                v = thing[Velocity]
                max_vel = self.parameters[MaxVelocity]
                if v.norm() > max_vel:
                    v_ = v * max_vel.value / v.norm()
                else:
                    v_ = v
                v.set(v_, allow_in_place=True)

    @timeit
    def step(self):
        """ Execute a single simulation step.

        """

        # Update all the factors except Position
        self._pre_collision_step()

        # Limit velocities
        self.limit_velocities()

        # Finally, move and handle collisions.
        self._move_objects()

        self.timer += self.parameters[Time]
        self._state_buffer.append(self.state)

    def pull_buffer(self, n_elements=None):
        """Pulls elements from the buffer.

        :param n_elements: (int) If provided, pulls last `n_elements`.
        :return: list of states.
        """
        if len(self._state_buffer) == 0:
            self._state_buffer.append(self.state)

        if len(self._state_buffer) < self._state_buffer.maxlen:
            while len(self._state_buffer) < self._state_buffer.maxlen:
                self._state_buffer.append(deepcopy(self._state_buffer[-1]))

        if n_elements is not None:
            if n_elements > self._state_buffer.maxlen:
                raise ValueError(
                    "Attempting to pull more elements than " "maximum elements in buffer."
                )
        else:
            n_elements = self._state_buffer.maxlen

        return list(self._state_buffer)[-n_elements:]


def states_are_equal(s1, s2):
    """Checks if two state dictionaries are equal.

    """
    if s1.keys() != s2.keys():
        return False
    for k in s1.keys():
        v1 = s1[k]
        v2 = s2[k]
        if type(v1) != type(v2):
            return False
        elif isinstance(v1, dict):
            if not states_are_equal(v1, v2):
                return False
        elif isinstance(v1, np.ndarray):
            if not np.allclose(v1, v2, rtol=1e-7):
                return False
        elif v1 != v2:
            return False

    return True
