from __future__ import annotations

__all__ = ('Rule', 'inspect_signature', 'match_pattern')

from types import GenericAlias
from typing import (_GenericAlias, Callable, Optional, Type, TypeVar, Union,
                    get_args, get_origin, get_type_hints)

from rpp import get_sim
from rpp.factors import Factor
from rpp.parameters import Parameter
from rpp.things import Entity
from rpp.things.boundaries import Wall


F = TypeVar('F', bound=Factor)
T = TypeVar('T', bound=Type[Factor])


def inspect_signature(rule_fn: Callable) -> Union[None, dict]:
    """Inspects the signature of the rule function.

    If the signature contains a TypeVar, then inspection needs to wait until
    after init of calling object.

    :param rule_fn: Function to inspect.
    :return: Dictionary of signature infos.
    """
    hints = get_type_hints(rule_fn)
    has_entity = False
    has_factors = False
    returns = None

    n_objects = 0

    input_patterns = []
    # Pattern values should be of type Type[Factor], Type[Entity], TypeVar,
    # or GenericAlias
    parameters = []
    for k, pattern in hints.items():
        if k == 'return':
            returns = pattern
        elif isinstance(pattern, (_GenericAlias, GenericAlias)):
            origin = get_origin(pattern)
            args = get_args(pattern)
            for arg in args:
                if not issubclass(arg, Factor):
                    raise ValueError('Unknown pattern type within tuple, '
                                     'must be Factor.')
            if origin == tuple:
                input_patterns.append(args)
                has_entity = True
                n_objects += 1
            else:
                raise TypeError(origin)
        elif isinstance(pattern, TypeVar):
            # This is a monkey patch. Rules with abstract types will
            # match here. The pattern needs to be fixed after __init__
            # has completed and type inference can conclude.
            return None
        else:
            input_patterns.append(pattern)
            if issubclass(pattern, Entity):
                has_entity = True
                n_objects += 1
            elif issubclass(pattern, Parameter):
                parameters.append(pattern)
            elif issubclass(pattern, Factor):
                has_factors = True
                n_objects = 1
            elif issubclass(pattern, Wall):
                pass
            else:
                raise TypeError(f'Pattern {pattern} of type '
                                f'{type(pattern)} not recognized.')

    if (returns is not None
            and isinstance(returns, (_GenericAlias, GenericAlias))):
        origin = get_origin(returns)
        return_factor_types = get_args(returns)
        if origin == tuple:
            args = tuple()
            for a in return_factor_types:
                args += get_args(a)
            return_factor_types = args
    else:
        return_factor_types = None

    sig = dict(
        has_entity=has_entity,
        has_factors=has_factors,
        n_objects=n_objects,
        input_patterns=input_patterns,
        returns=returns,
        return_factor_types=return_factor_types,
        hints=hints,
        parameters=parameters
    )

    if has_entity and has_factors:
        raise TypeError(f'Rule signature ({sig}) cannot have both factors and '
                        'containers.')

    return sig


def match_pattern(rule: Rule,
                  *inputs: Union[tuple, list, Factor, Entity, Parameter],
                  loose_match: bool = False
                  ) -> Union[None, list[Union[Factor, Entity, Parameter]]]:
    """Attempts to match a set of inputs to a pattern provided.

    If the pattern matches, return a list of args, else return None

    :param rule: Rule function to pattern match on.
    :param inputs: Inputs to check pattern matching on rule.
    :param loose_match: Loosely match: if there are extra args, pass but
        remove them from final input.
    :return: This of arguments to apply to rule given input, in-order.
    """

    args = []

    inputs = list(inputs)
    patterns: list[Type[Parameter], Type[Factor], Type[Entity], tuple] = \
        rule.signature_info['input_patterns'].copy()
    factor_type = rule.factor_type
    entity_type = rule.entity_type

    has_right_entity: bool = entity_type is None
    has_right_factor: bool = factor_type is None

    while len(inputs) > 0 and len(patterns) > 0:
        if len(patterns) == 0 and loose_match:
            break

        inp = inputs.pop(0)
        pat = patterns.pop(0)

        if isinstance(pat, (tuple, list)):
            if isinstance(inp, Entity):
                factors = []
                for p in pat:
                    if p in inp:
                        factors.append(inp[p])
                    else:
                        return None
                args.append(tuple(factors))
                if entity_type is not None:
                    has_right_entity |= isinstance(inp, entity_type)
                if factor_type is not None:
                    has_right_factor |= factor_type in inp
            elif isinstance(inp, (tuple, list)):
                if len(pat) != len(inp):
                    return None
                for p, i in zip(pat, inp):
                    if not isinstance(i, p):
                        return None
                args.append(inp)
            else:
                tup = tuple()
                #  Attempt to unpack factors from input
                for p in pat:
                    if len(tup) > 0:
                        inp = inputs.pop(0)
                    else:
                        return None

                    if factor_type is not None:
                        has_right_factor |= isinstance(inp, factor_type)

                    if p.can_cast(inp):
                        tup.append(inp)
                    else:
                        return None
                args.append(tup)
        elif issubclass(pat, Entity):
            if isinstance(inp, pat):
                args.append(inp)
                if entity_type is not None:
                    has_right_entity |= isinstance(inp, entity_type)
                if factor_type is not None:
                    has_right_factor |= factor_type in inp
            else:
                return None
        elif issubclass(pat, Wall):
            if isinstance(inp, pat):
                args.append(inp)
        elif issubclass(pat, Parameter):
            if isinstance(inp, pat):
                args.append(inp)
            elif pat.can_cast(inp):
                args.append(pat(inp))
            else:
                return None

        elif issubclass(pat, Factor):
            if isinstance(inp, Entity):
                if pat in inp:
                    # Unpack the Entity
                    while pat in inp:
                        inp_ = inp[pat]
                        args.append(inp_)
                        if len(patterns) > 0:
                            pat = patterns.pop(0)
                        else:
                            pat = None
                        if factor_type is not None:
                            has_right_factor |= isinstance(inp_, factor_type)

                    if pat is not None:
                        #  If there was one more pattern, it didn't match.
                        #  Put it back.
                        patterns = [pat] + patterns
                else:
                    return None

                if entity_type is not None:
                    has_right_entity |= isinstance(inp, entity_type)

            elif pat.can_cast(inp):
                args.append(inp)

                if factor_type is not None:
                    has_right_factor |= isinstance(inp, factor_type)

            else:
                return None
        else:
            raise ValueError(f'Unknown pattern type {pat}.')

    if (len(inputs) > 0 and not loose_match) or len(patterns) > 0:
        return None

    if not has_right_entity:
        return None

    if not has_right_factor:
        return None

    return args


class Rule:
    """Generic Rule class.

    """
    def __init__(self, rule_fn: Callable,
                 factor_type: Optional[Type[Factor]] = None,
                 entity_type: Optional[Type[Entity]] = None):
        """Wraps a function into a Rule.

        :param rule_fn: Function to wrap.
        :param factor_type: Optional factor type that must be present in
            entity when pattern matching.
        :param entity_type: Optional entity type that must be present in
            pattern matching.
        """
        self._rule_fn = rule_fn
        self.factor_type = factor_type
        self.entity_type = entity_type
        self._signature_info = inspect_signature(self.rule_fn)
        self._sim = None

    @property
    def sim(self):
        if self._sim is None:
            return get_sim()
        return self._sim

    def set_sim(self, sim):
        self._sim = sim

    def copy_for_sim(self, sim):
        rule_copy = type(self)(self._rule_fn)
        rule_copy.__dict__.update(self.__dict__)
        rule_copy._sim = sim
        return rule_copy

    @property
    def rule_fn(self) -> Callable:
        return self._rule_fn

    @property
    def signature_info(self) -> dict:
        """Info on the signature of the wrapped rule function.

        """
        if self._signature_info is None:
            #  Because typing doesn't finish some type inference at init
            self._signature_info = inspect_signature(self.rule_fn)
        return self._signature_info

    @property
    def parameters(self):
        """Extra parameters in rule.

        """
        return self.signature_info['parameters']

    @property
    def n_objects(self) -> int:
        """Number of objects needed to match rule.

        """
        return self.signature_info['n_objects']

    def __call__(self, *args, **kwargs):
        raise NotImplementedError

    def __repr__(self) -> str:
        returns = self.signature_info['returns']
        input_pattern = self.signature_info['input_patterns']
        r = f'{returns} <- {self.rule_fn.__name__}({input_pattern})'
        if self.factor_type:
            r += f' (if has {self.factor_type})'
        if self.entity_type:
            r += f' (if is {self.entity_type})'
        return r

    @property
    def __name__(self):
        return self.rule_fn.__name__
