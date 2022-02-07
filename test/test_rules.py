__author__ = "R Devon Hjelm"
__copyright__ = "Copyright (c) Microsoft Corporation and Mila: The Quebec " \
                "AI Company"
__license__ = "MIT"
import unittest

import numpy as np

from segar.factors import (Charge, Magnetism, Mass, Floor, Heat, Friction,
                           GaussianNoise, Label, Position, ID)
from segar.mdps.initializations import ArenaInitialization
from segar.rules import (SetFactor, IsEqual, conditional_transition,
                         TransitionFunction, Differential, DidNotMatch,
                         DidNotPass, inspect_signature, IsOn, Contains, Prior)
from segar.sim import Simulator
from segar.parameters import Gravity
from segar.things import Object, Magnet, Tile, Charger, Entity


Simulator()


def _factors_and_parameters(charge: Charge, magnetism: Magnetism,
                            gravity: Gravity) -> SetFactor[Charge]:
    return SetFactor[Charge](charge, gravity)


_factors_and_parameters_rule = TransitionFunction(
    _factors_and_parameters)
_factors_and_parameters_rule_with_entity = TransitionFunction(
    _factors_and_parameters, entity_type=Object)
_factors_and_parameters_rule_with_factor = TransitionFunction(
    _factors_and_parameters, factor_type=Mass
)
_factors_and_parameters_rule_with_condition = conditional_transition(
    relation=IsEqual(Charge, 1.0)
)(_factors_and_parameters)


_factors_and_parameters_sig = {
    'has_entity': False,
    'has_factors': True, 'n_objects': 1,
    'input_patterns': [Charge, Magnetism, Gravity],
    'returns': SetFactor[Charge],
    'hints': {'charge': Charge,
              'magnetism': Magnetism,
              'return': SetFactor[Charge]}}


_factors_and_parameters_ios = [
    ((Charge(0.3), Magnetism(0.5), Gravity(0.7)),
     SetFactor[Charge](Charge(0.3), Gravity(0.7))),
    ((Object({Charge: 0.3, Magnetism: 0.5}), Gravity(0.7)),
     SetFactor[Charge](Charge(0.3), Gravity(0.7)))
]


#  Tuples and parameters


def _tuples_and_parameters(
        o1_factors: tuple[Charge, Magnetism],
        o2_factors: tuple[Floor, Heat, Friction],
        gravity: Gravity) -> Differential[Charge]:
    charge, _ = o1_factors
    return Differential[Charge](charge, gravity)


_tuples_and_parameters_sig = {
    'has_entity': True,
    'has_factors': False, 'n_objects': 2,
    'input_patterns': [(Charge, Magnetism), (Floor, Heat, Friction), Gravity],
    'returns': Differential[Charge],
    'hints': {'o1_factors': tuple[Charge, Magnetism],
              'o2_factors': tuple[Floor, Heat, Friction],
              'gravity': Gravity,
              'return': Differential[Charge]}}


_tuples_and_parameters_rule = TransitionFunction(
    _tuples_and_parameters)
_tuples_and_parameters_with_entity_rule = TransitionFunction(
    _tuples_and_parameters, entity_type=Magnet)
_tuples_and_parameters_with_factor_rule = TransitionFunction(
    _tuples_and_parameters, factor_type=Mass
)
_tuples_and_parameters_with_condition_rule = conditional_transition(
    relation=IsEqual(Charge, 0.3)
)(_tuples_and_parameters)


_tuples_and_parameters_ios = [
    (((Charge(0.3), Magnetism(0.5)),
      (Floor(), Heat(0.11), Friction(0.13)),
      Gravity(0.7)),
     Differential[Charge](Charge(0.3), Gravity(0.7))),
    ((Object({Charge: 0.3, Magnetism: 0.5}),
      Tile({Heat: 0.11, Friction: 0.13}),
      Gravity(0.7)),
     Differential[Charge](Charge(0.3), Gravity(0.7)))
]


_tuples_and_parameters_with_entity_pass_ios = [
    ((Magnet({Charge: 0.3, Magnetism: 0.5}),
      Tile({Heat: 0.11, Friction: 0.13}),
      Gravity(0.7)),
     Differential[Charge](Charge(0.3), Gravity(0.7)))
]


_tuples_and_parameters_with_entity_fail_ios = [
    ((Charger({Charge: 0.3, Magnetism: 0.5}),
      Tile({Heat: 0.11, Friction: 0.13}),
      Gravity(0.7)),
     Differential[Charge](Charge(0.3), Gravity(0.7)))
]


_tuples_and_parameters_with_factor_pass_ios = [
    ((Entity({Charge: 0.3, Magnetism: 0.5, Mass: 0.17}),
      Tile({Heat: 0.11, Friction: 0.13}),
      Gravity(0.7)),
     Differential[Charge](Charge(0.3), Gravity(0.7)))
]


_tuples_and_parameters_with_factor_fail_ios = [
    ((Entity({Charge: 0.3, Magnetism: 0.5}),
      Tile({Heat: 0.11, Friction: 0.13}),
      Gravity(0.7)),
     Differential[Charge](Charge(0.3), Gravity(0.7)))
]


_tuples_and_parameters_with_condition_pass_ios = [
    ((Entity({Charge: 0.3, Magnetism: 0.5}),
      Tile({Heat: 0.11, Friction: 0.13}),
      Gravity(0.7)),
     Differential[Charge](Charge(0.3), Gravity(0.7)))
]


_tuples_and_parameters_with_condition_fail_ios = [
    ((Entity({Charge: 0.4, Magnetism: 0.5}),
      Tile({Heat: 0.11, Friction: 0.13}),
      Gravity(0.7)),
     Differential[Charge](Charge(0.4), Gravity(0.7)))
]


#  Entities and parameters


def _entities_and_parameters(
        o1: Object,
        o2: Object,
        gravity: Gravity) -> Differential[Charge]:
    charge = o1[Charge]
    return Differential[Charge](charge, gravity)


_entities_and_parameters_sig = {
    'has_entity': True,
    'has_factors': False, 'n_objects': 2,
    'input_patterns': [Object, Object, Gravity],
    'returns': Differential[Charge],
    'hints': {'o1': Object,
              'o2': Object,
              'gravity': Gravity,
              'return': Differential[Charge]}}


_entities_and_parameters_rule = TransitionFunction(
    _entities_and_parameters)


_entities_and_parameters_ios = [
    ((Object({Charge: 0.3, Magnetism: 0.5}),
      Object({Charge: 0.11, Magnetism: 0.13}), Gravity(0.7)),
     Differential[Charge](Charge(0.3), Gravity(0.7)))
]


#  Should fail signature
def _factors_and_entities(
        charge: Charge,
        o2_factors: tuple[Floor, Heat, Friction],
        gravity: Gravity) -> Differential[Charge]:
    pass


class TestRules(unittest.TestCase):
    def test_signatures(self):
        print('Testing rule signatures.')
        rule_names = ('factors_and_parameters', 'tuples_and_parameters',
                      'entities_and_parameters')
        for rule_name in rule_names:
            rule_fn = globals()['_' + rule_name]
            target_info = globals()['_' + rule_name + '_sig']
            signature_info = inspect_signature(rule_fn)
            for k in ('has_entity', 'has_factors', 'input_patterns',
                      'returns'):
                self.assertEqual(signature_info[k], target_info[k],
                                 f'Signature on {rule_name} failed on {k}.')

        bad_rule_names = ('factors_and_entities',)
        for rule_name in bad_rule_names:
            rule_fn = globals()['_' + rule_name]
            with self.assertRaises(TypeError,
                                   msg=f'{rule_name} should fail with '
                                       f'TypeError'):
                inspect_signature(rule_fn)

    def test_rules(self):
        print('Testing rules with inputs that should work.')
        rule_names = ('factors_and_parameters', 'tuples_and_parameters',
                      'entities_and_parameters')
        for rule_name in rule_names:
            rule = globals()['_' + rule_name + '_rule']
            rule_ios = globals()['_' + rule_name + '_ios']
            print(f'Testing {rule_name}')
            for inp, target in rule_ios:
                out = rule(*inp)
                if isinstance(out, DidNotMatch):
                    raise ValueError(f'Input {inp} did not match on'
                                     f' {rule_name}.')
                self.assertEqual(out, target)

    def test_rule_breaking(self):
        print('Testing inputs that should fail with rules.')
        rule_names = ('factors_and_parameters', 'tuples_and_parameters',
                      'entities_and_parameters')
        for rule_name in rule_names:
            for rule_name_ in rule_names:
                if rule_name_ == rule_name:
                    continue
                rule = globals()['_' + rule_name + '_rule']
                rule_ios = globals()['_' + rule_name_ + '_ios']
                print(f'Testing {rule_name} with {rule_name_} IOs')
                for inp, target in rule_ios:
                    out = rule(*inp)
                    self.assertIsInstance(out, DidNotMatch)

    def test_typevar_rules(self):
        print('Testing rules with inferred types')
        rules = (
            IsEqual(Charge, 0.3),
            IsOn(),
            Contains(),
            Prior(Charge, GaussianNoise())
        )
        for rule in rules:
            sig = rule.signature_info
            self.assertIsNotNone(sig)

    def test_rule_with_requirements(self):
        print('Testing rules with required entities or factors')
        rule_names = ('tuples_and_parameters_with_factor',
                      'tuples_and_parameters_with_entity',
                      'tuples_and_parameters_with_condition')
        for rule_name in rule_names:
            rule = globals()['_' + rule_name + '_rule']
            rule_ios_pass = globals()['_' + rule_name + '_pass_ios']
            rule_ios_fail = globals()['_' + rule_name + '_fail_ios']

            for inp, target in rule_ios_pass:
                out = rule(*inp)
                if isinstance(out, DidNotMatch):
                    raise ValueError(f'Input {inp} did not match on'
                                     f' {rule_name}.')
                self.assertEqual(out, target)

            for inp, target in rule_ios_fail:
                out = rule(*inp)
                self.assertIsInstance(out, (DidNotMatch, DidNotPass),
                                      msg=f'Rule {rule_name} with inputs'
                                          f' {inp} was supposed to fail but '
                                          f'did not. Got {out}.')

    def test_priors(self):
        print('Testing priors')
        p1 = (Prior(Charge, 0.3))
        p2 = Prior(Charge, GaussianNoise())
        m = Magnetism(0.1)
        p3 = Prior(Charge, m)

        for prior in (p1, p2, p3):
            c = Charge(0.)
            out = prior(c)
            out()

    def test_rule_priorities(self):
        sim = Simulator(local_sim=True)
        sim.reset()

        class Ball(Object, default={Label: 'ball'}):
            pass

        class Ball2(Object, default={ID: 'golfball', Label: 'ball'}):
            pass

        numbers = [
            (Object, 1),
            (Ball, 1),
            (Ball2, 1),
            (Charger, 1)
        ]

        # Copying for rules is necessary for multi-sim needs.
        priors = [
            Prior(Position, [0.5, 0.5], entity_type=Object),
            Prior(Position, [1., 1.], entity_type=Charger),
            Prior(Position, [1.5, 1.5], relation=IsEqual(Label, 'ball')),
            Prior(Position, [2., 2.], relation=IsEqual(ID, 'golfball'))
        ]
        init = ArenaInitialization(
            config={'numbers': numbers, 'priors': priors})
        init.set_sim(sim)
        if init.sim is not sim:
            raise ValueError('init has wrong sim.')

        for prior in init._priors:
            if prior.sim is not sim:
                raise ValueError('Prior should have same sim.')

        init.sample()
        init.set_arena()

        for thing in sim.things.values():
            if thing.has_factor(Position):
                if thing[ID] == 'golfball':
                    self.assertEqual(thing[Position], np.array([2.0, 2.0]))
                elif thing.has_factor(Label) and thing[Label] == 'ball':
                    self.assertEqual(thing[Position], np.array([1.5, 1.5]))
                elif isinstance(thing, Charger):
                    self.assertEqual(thing[Position], np.array([1.0, 1.0]))
                else:
                    self.assertEqual(thing[Position], np.array([0.5, 0.5]))


def test():
    unittest.main()


if __name__ == '__main__':
    test()
