import argparse

from segar.sim.sim import load_sim_from_file
from segar.tools.sample_trajectories import rollout_sim_only


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description='Re-runs a simulator forward.')

    parser.add_argument('sim_path', type=str,
                        help='Path to simulator pkl file.')
    parser.add_argument('--n_steps', default=100, type=int)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    sim = load_sim_from_file(args.sim_path)
    trajectories = rollout_sim_only(sim, args.n_steps)
