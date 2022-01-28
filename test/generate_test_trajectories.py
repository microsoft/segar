import argparse

from segar.sim.sim import Simulator
from segar.mdps.initializations import ArenaInitialization
from segar.tools.sample_trajectories import rollout_sim_only, save_trajectories

from test_configs import configs


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description='Generates trajectories from test configs and saves.')

    parser.add_argument('out_path', type=str,
                        help='Path to output directory.')
    parser.add_argument('--n_envs', default=10, type=int,
                        help='Number of envs per config to generate from.')
    parser.add_argument('--n_steps', default=500, type=int)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    for k, config in configs.items():
        for i in range(args.n_envs):
            sim_path = f'{args.out_path}/test_{k}_sim_{i}.pkl'
            traj_path = f'{args.out_path}/test_{k}_traj_{i}.pkl'
            sim = Simulator(save_path=sim_path)
            sim.jiggle_all_object_velocities()
            sim.save()
            initialization = ArenaInitialization(config=config)
            trajectories, _ = rollout_sim_only(sim, args.n_steps)
            save_trajectories(trajectories, out_path=traj_path)
