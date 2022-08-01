from __future__ import annotations
__copyright__ = "Copyright (c) Microsoft Corporation and Mila - Quebec AI Institute"
__license__ = "MIT"

"""Factors of variation and FactorContainers.

"""

__all__ = ('Factor', 'FactorContainer', 'DEFAULTS', 'FACTORS')

from copy import copy
from typing import get_args, Any, Dict, Generic, Tuple, Type, TypeVar, Union

import numpy as np


DEFAULTS = {}
FACTORS = []

T = TypeVar('T')


# Since we're using types as keys, this will help clean things up.
class MetaFactor(type):
    def __repr__(cls):
        return cls.__name__


class Factor(Generic[T], metaclass=MetaFactor):
    _t: Union[type, None] = None
    _value: T
    _default: Union[T, None] = None
    _protected_in_place: bool = True

    def __init_subclass__(cls, /, default=None, **kwargs):
        cls._t = get_args(cls.__orig_bases__[0])[0]
        if default is None:
            if cls._t == bool:
                default = False
            elif cls._t == float:
                default = 0.
            elif cls._t == int:
                default = 0
            elif cls._t == np.array:
                default = np.array([0., 0.])
            elif cls._t == str:
                default = ''

        cls._default = default

        super().__init_subclass__(**kwargs)
        DEFAULTS[cls] = default
        FACTORS.append(cls)

    def __init__(self, value: Union[Any, T, None] = None):
        if value is None:
            value = self._default

        self._allow_in_place = True
        self._value = None
        self.value = value
        self._allow_in_place = False

    def __eq__(self, other: Union[Factor, T]) -> bool:
        return self.value == Factor._get_value(other)

    def __ne__(self, other: Union[Factor, T]) -> bool:
        return self.value != Factor._get_value(other)

    def _test_value(self, value: Any):
        pass

    @property
    def t(self) -> type:
        # typing isn't really designed for runtime inference, but we do that
        # anyways. So this is a bit of a monkey patch.
        if self._t is None:
            try:
                return get_args(self.__orig_class__)[0]
            except AttributeError:
                raise AttributeError('Inferred type of base Factor class '
                                     'cannot be accessed until after '
                                     '__init__.')
        else:
            return self._t

    @property
    def default(self) -> T:
        return self._default

    # protecting the value

    class _AllowInPlace:
        def __init__(self, factor: Factor):
            self.prior_in_place = factor._allow_in_place
            self.factor = factor

        def __enter__(self):
            self.factor._allow_in_place = True

        def __exit__(self, exc_type, exc_val, exc_tb):
            self.factor._allow_in_place = self.prior_in_place

    def in_place(self):
        return self._AllowInPlace(self)

    @property
    def value(self) -> T:
        return self._value

    @value.setter
    def value(self, value):
        if self._allow_in_place or not self._protected_in_place:
            if hasattr(value, 't'):
                alias = value
                if alias.t == self.t:
                    self._value = alias.value
                else:
                    raise ValueError(
                        'Cannot instantiate factor alias directly from a '
                        'different type of factor unless the factor '
                        'value is of the same type.')
            else:
                self._value = self.new_value(value)
        else:
            raise ValueError('Factor in-place operations are protected.')

    @staticmethod
    def _get_value(other: Any) -> Any:
        if isinstance(other, Factor):
            return other.value
        return other

    def __copy__(self) -> Factor:
        new = self.__class__(self.value)
        return new

    def new_value(self, value: Any) -> T:
        if self._t is None:
            t = None
        else:
            t = self.t

        if isinstance(value, Factor):
            value = value.value

        if (t is None) or (value is None) or isinstance(t, TypeVar):
            return value
        elif isinstance(value, t):
            return value
        elif t == np.ndarray:
            return np.array(value)
        else:
            return t(value)  # Is cast-able?

    @classmethod
    def can_cast(cls, value: Any) -> bool:
        if cls._t is None:
            return False
        t: type = cls._t

        if isinstance(value, Factor):
            value = value.value

        if t is None and value is not None:
            return False
        elif isinstance(value, t):
            return True
        elif t == np.ndarray:
            try:
                np.ndarray(value)
                return True
            except (ValueError, TypeError):
                return False
        else:
            v_t = type(value)
            if t == float and v_t in (float, int):
                return True
            else:
                return t == v_t

    def set(self, value: Union[T, Factor], allow_in_place: bool = False
            ) -> None:
        value = self._get_value(value)
        if allow_in_place:
            with self.in_place():
                self.value = value
        else:
            self.value = value

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.value})'

    def __hash__(self):
        """Factors are *unique*, so we want them to hash differently,
            despite their value.

        """
        return hash(id(self))


F = TypeVar('F', bound=Factor)


class FactorContainer(Factor[dict], default={}):
    """Abstract collection of factors.

    """
    _factor_types = None

    def __init_subclass__(cls, /, default=None, **kwargs):
        default = default or {}
        super().__init_subclass__(default=default, **kwargs)

    def __init__(self, value: Union[Dict[Type[Factor], Factor], None] = None):
        if value is not None:
            factors = {}
            for k, v in value.items():
                val = self._get_value(v)
                if val is None and self.default.get(k, None) is not None:
                    v = copy(self.default[k])
                if isinstance(v, k):
                    factors[k] = v
                else:
                    factors[k] = k(v)
            value = factors
        super().__init__(value=value)

    def has_factor(self, factor_type: Type[F]) -> bool:
        return bool(factor_type in self.keys())

    def contains(self, factor: Factor) -> bool:
        for factor_ in self.value.values():
            if factor is factor_:
                return True
        return False

    def get_factors(self, *factor_types: Type[F]) -> Tuple[F]:
        return tuple(self[factor_type] for factor_type in factor_types)

    def set_factor(self, factor_type: Type[F], value: Union[F, F.t],
                   allow_in_place: bool = False) -> None:
        self[factor_type].set(factor_type(value),
                              allow_in_place=allow_in_place)

    def keys(self):
        return self.value.keys()

    def items(self):
        return self.value.items()

    def values(self):
        return self.value.values()

    class _AllowInPlace:
        def __init__(self, fc: FactorContainer):
            self.prior_in_place = dict((k, v._allow_in_place)
                                       for k, v in fc.value.items())
            self.fc = fc

        def __enter__(self):
            for v in self.fc.values():
                v._allow_in_place = True

        def __exit__(self, exc_type, exc_val, exc_tb):
            for k, v in self.fc.items():
                if k in self.prior_in_place:
                    v._allow_in_place = self.prior_in_place[k]
                else:
                    v._allow_in_place = False

    def in_place(self):
        return self._AllowInPlace(self)

    def __getitem__(self, item: Type[F]) -> F:
        return self.value[item]

    def __setitem__(self, key: Type[F], value: Union[F, F.t]) -> None:
        if not (isinstance(value, Factor) and key.t == value.t):
            if key in self.value:
                allow_in_place = self.value[key]._allow_in_place
                if allow_in_place:
                    self.value[key] = key(value)
                    self.value[key]._allow_in_place = allow_in_place
                else:
                    raise ValueError('In-place operations protected.')
            else:
                self.value[key] = key(value)
        else:
            self[key].set(key(value))

    def __contains__(self, item: Type[F]) -> bool:
        return bool(item in self.value.keys())

    def __call__(self, *factors: Type[F]) -> Tuple[F]:
        return tuple(self[factor] for factor in factors)

    def copy(self):
        factors = self.value
        new_factors = {}
        for k, v in factors.items():
            new_factors[k] = copy(v)
        new = self.__class__(new_factors)
        return new

    def __getattr__(self, name: str) -> Factor:
        if name in ('value', '_value'):
            raise ValueError('`value` should be set at __init__. Getting '
                             'here is an error. This may be due to placing '
                             'an SEGAR component inside an object that is '
                             'being pickled.')

        for k, v in self.value.items():
            if str(k) == name:
                return v

        raise AttributeError(name)

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        return state

    def __setstate__(self, state: dict):
        self.__dict__.update(**state)
