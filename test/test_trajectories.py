__copyright__ = (
    "Copyright (c) Microsoft Corporation and Mila - Quebec AI Institute"
)
__license__ = "MIT"
"""Module for loading and testing trajectories are consistent with current
code.

"""

import argparse
from glob import glob
import logging
from os import path
import pickle
import traceback

from segar import load_sim_from_file
from segar.tools.sample_trajectories import rollout_sim_only


logger = logging.getLogger("tests.test_trajectories")


def load_trajectory(traj_path: str):
    with open(traj_path, "rb") as f:
        traj = pickle.load(f)
    return traj


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Tests trajectories are consistent when reloading and "
        "running simulator.",
    )

    parser.add_argument(
        "in_path",
        type=str,
        help="Path to simulators and trajectories directory.",
    )
    parser.add_argument("--n_steps", default=500, type=int)
    return parser.parse_args()


def test(sim_paths: list[str]):
    for sim_path in sim_paths:
        logger.info(f"Testing trajectories from {sim_path}.")
        try:
            traj_path = sim_path[:].replace("sim", "traj")
            sim = load_sim_from_file(sim_path)
            trajectories, _ = rollout_sim_only(sim, args.n_steps)
            old_trajectories = load_trajectory(traj_path)
            assert trajectories == old_trajectories, (
                f"Trajectory test " f"failed " f"on {sim_path}."
            )
        except Exception:
            traceback.print_exc()
            logger.error("[FAILED]")
            raise RuntimeError("Failed")

    logger.info("[PASSED]")


if __name__ == "__main__":
    args = parse_args()
    sim_paths = glob(path.join(args.in_path, "*sim*"))
    test(sim_paths)
