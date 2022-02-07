__author__ = "R Devon Hjelm"
__copyright__ = "Copyright (c) Microsoft Corporation and Mila - Quebec AI " \
                "Institute"
__license__ = "MIT"
"""To be filled as bugs and mis-implemented features are found.

"""

import logging

import numpy as np

from segar.factors import Charge, Mobile, Hexagon, Size, Square, Shape, \
    Position, Velocity, Alive, Done, Mass, Friction
from segar.types import Time
from segar.parameters import Gravity
from segar.configs.handler import get_env_config
from segar.mdps import RGBObservation, ObjectStateObservation, \
    TileStateObservation, AllObjectsStateObservation, Task, MDP, \
    ArenaInitialization, AllTilesStateObservation, make_stacked_observation
from segar.sim.sim import Simulator
from segar.rules import Prior
from segar.things import Object
from segar import get_sim


logger = logging.getLogger('tests.func')


class MyTask(Task):

    def check_action(self, action):
        assert isinstance(action, np.ndarray) and action.shape == (2,)

    def apply_action(self, action):
        # Just apply force to first thing that has Velocity.
        idx = self.sim.thing_ids_with_factor(Velocity)[0]
        self.sim.add_force(idx, action)

    def demo_action(self):
        return np.random.normal() + np.array((2, 1.5))

    def reward(self, state):
        idx = self.sim.thing_ids_with_factor(Velocity)[0]
        # Just see if the first object is alive.
        return float(state['things'][idx][Alive])

    def done(self, state):
        idxs = self.sim.thing_ids_with_factor(Velocity)
        object_states = dict((idx, state['things'][idx]) for idx in idxs)
        done = False
        for state in object_states.values():
            if not state[Alive] or state[Done]:
                done = True

        return done


def _make_things(sim):
    sim.add_ball(position=np.array([0.05, 0.05]), text='X',
                 unique_id='tennis_ball', initial_factors={Charge: -0.5})
    sim.add_magnet(position=np.array([-0.5, 0.5]), text='M',
                   initial_factors={Mobile: True, Size: 0.3})
    sim.add_sand(position=np.array([0.1, -0.1]),
                 initial_factors={Shape: Hexagon(0.4)},
                 text='S', unique_id='sand_pit')
    sim.add_magma(position=np.array([-0.5, 0.5]),
                  initial_factors={Shape: Square(0.3)},
                  text='G')
    sim.add_charger(position=np.array([0.3, 0.7]), text='C',
                    initial_factors={Charge: 1.0, Mobile: True})


def _test_stacked_observation_one(obs_class=RGBObservation, n_stack=5):
    sim = Simulator(state_buffer_length=10)
    _make_things(sim)
    logger.info(f'Testing stacked {obs_class}.')

    if obs_class == ObjectStateObservation:
        kwargs = dict(unique_id='tennis_ball')
    elif obs_class == TileStateObservation:
        kwargs = dict(unique_id='sand_pit')
    elif obs_class == AllObjectsStateObservation:
        kwargs = dict(unique_ids=['tennis_ball'])
    elif obs_class == AllTilesStateObservation:
        kwargs = dict(unique_ids=['sand_pit'])
    elif obs_class == RGBObservation:
        config = get_env_config('visual', 'linear_ae', 'baseline')
        kwargs = dict(config=config)
    else:
        raise NotImplementedError(obs_class)

    stacked_class = make_stacked_observation(obs_class, n_stack=n_stack)
    observation = obs_class(**kwargs)
    stacked_observation = stacked_class(**kwargs)

    for _ in range(n_stack + 1):
        sim.step()

    stacked_obs_out = stacked_observation()
    obs_out = observation(sim.state)

    try:
        assert stacked_obs_out.shape == (n_stack, *obs_out.shape)
        assert np.allclose(stacked_obs_out[-1], obs_out, atol=1e-7)
    except AssertionError as e:
        logger.error(f'Testing stacked {obs_class} [FAILED].')
        raise e
    logger.info('[PASSED]')


def test_stacked_observation(n_stack=5):
    obs_classes = (RGBObservation, ObjectStateObservation,
                   TileStateObservation, AllObjectsStateObservation,
                   AllTilesStateObservation)

    for obs_class in obs_classes:
        _test_stacked_observation_one(obs_class, n_stack=n_stack)


def _create_mdp(sim=None):
    visual_config = get_env_config('visual', 'linear_ae', dist_name='baseline')
    rgb_obs = RGBObservation(resolution=256, config=visual_config)

    config = dict(
        numbers=[(Object, 1)],
        priors=[Prior(Position, [0., 0.])]
    )

    initialization = ArenaInitialization(config=config)

    action_range = (-100, 100)  # Range of valid action values
    action_shape = (2,)  # 2d vectors
    action_type = np.float32
    baseline_action = np.zeros(2).astype(np.float32)

    task = MyTask(action_range=action_range, action_shape=action_shape,
                  action_type=action_type, baseline_action=baseline_action,
                  initialization=initialization)

    mdp = MDP(rgb_obs, task, sim=sim)
    return mdp


def test_apply_action():
    print('Testing actions apply correctly.')
    sim = get_sim()
    sim.reset()
    mdp = _create_mdp()

    mass = sim.things[0][Mass].value
    gravity = sim.parameters[Gravity]
    mu = sim.things['global_friction'][Friction].value
    dt = sim.parameters[Time]
    for _ in range(100):
        action = np.random.uniform(-2, 2, size=(2,))
        mdp.reset()

        mdp.step(action)
        vel = sim.things[0][Velocity].value
        expected_vel = Velocity(action / mass)

        vel_sign = expected_vel.sign()
        vel_norm = expected_vel.norm()

        f_mag = mu * gravity
        norm_abs_vel = expected_vel.abs() / vel_norm
        da = -vel_sign * f_mag * norm_abs_vel / mass
        expected_vel = (expected_vel + da * dt).value

        if not np.allclose(expected_vel, vel, atol=1e-7):
            raise ValueError(f'Velocity {vel} doesn\'t match expected '
                             f'{expected_vel}.')
    print('[PASSED]')


def test_local_sim():
    print('Testing local sims.')
    sim = Simulator(local_sim=True)
    sim_global = get_sim()
    mdp = _create_mdp(sim)
    if sim is sim_global:
        raise ValueError('Local sim creation failed.')

    if sim is not mdp.sim:
        raise ValueError('MDP sim not set to local sim.')

    if mdp.sim is not mdp._task.sim:
        raise ValueError('MDP sim not task sim.')

    if mdp.sim is not mdp._observation.sim:
        raise ValueError('MDP sim not observation sim.')

    if mdp.sim is not mdp._task._initialization.sim:
        raise ValueError('MDP sim not initialization sim.')
    print('[PASSED]')


def run_all_tests():
    test_funcs = dict((k, f) for k, f
                      in globals().items() if k.startswith('test'))
    for k, f in test_funcs.items():
        logger.info(f'Testing {k}.')
        f()


if __name__ == '__main__':
    run_all_tests()
