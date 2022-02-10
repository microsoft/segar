__copyright__ = (
    "Copyright (c) Microsoft Corporation and Mila - Quebec AI Institute"
)
__license__ = "MIT"
import logging
import traceback

from segar.mdps.mdps import MDP
from segar.mdps.observations import (
    RGBObservation,
    ObjectStateObservation,
    TileStateObservation,
    AllObjectsStateObservation,
    AllTilesStateObservation,
)
from segar.configs.handler import get_env_config
from segar.tasks.billiards import (
    BilliardsInitialization,
    Billiards,
    billiards_default_config,
)
from segar.tasks.puttputt import (
    PuttPuttInitialization,
    Invisiball,
    PuttPutt,
    puttputt_default_config,
    invisiball_config,
)
from segar.tools.sample_trajectories import rollout
from segar.sim.sim import Simulator


_TASKS = ("puttputt", "invisiball", "billiards")
_OBSERVATIONS = ("rgb", "objstate", "tilestate", "allobjstate", "alltilestate")
_VIS_GEN = "linear_ae"

logger = logging.getLogger("tests.test_env")


def test(
    n_envs=10,
    task=None,
    observations=None,
    vis_gen=_VIS_GEN,
    visual_dist="baseline",
    show=False,
    iterations=100,
):

    logger.info(
        f"Testing environment with variables task={task}, "
        f"observations={observations}, visual features={vis_gen}."
    )
    Simulator()

    config = dict(
        max_steps_per_episode=iterations, episodes_per_arena=1, sub_steps=1
    )

    visual_config = get_env_config("visual", vis_gen, dist_name=visual_dist)
    o_render = RGBObservation(resolution=256, config=visual_config)

    if task == "puttputt":
        initialization = PuttPuttInitialization(config=puttputt_default_config)
        t = PuttPutt(initialization)
        unique_obj_id = "golfball"
        unique_tile_id = "goal"
        unique_obj_ids = ["golfball"]
        unique_tile_ids = ["goal"]

    elif task == "invisiball":
        initialization = PuttPuttInitialization(config=invisiball_config)
        t = Invisiball(initialization)
        unique_obj_id = "golfball"
        unique_tile_id = "goal"
        unique_obj_ids = ["golfball"]
        unique_tile_ids = ["goal"]

    elif task == "billiards":
        initialization = BilliardsInitialization(
            config=billiards_default_config
        )
        t = Billiards(initialization)
        unique_obj_id = "cueball"
        unique_tile_id = "0_hole"
        unique_obj_ids = ["cueball"] + [f"{n}_ball" for n in range(1, 9)]
        unique_tile_ids = [f"{n}_hole" for n in range(4)]

    else:
        raise NotImplementedError(task)

    if observations == "rgb":
        o = o_render

    elif observations == "objstate":
        o = ObjectStateObservation(unique_obj_id)

    elif observations == "tilestate":
        o = TileStateObservation(unique_tile_id)

    elif observations == "allobjstate":
        o = AllObjectsStateObservation(unique_ids=unique_obj_ids)

    elif observations == "alltilestate":
        o = AllTilesStateObservation(unique_ids=unique_tile_ids)

    else:
        raise NotImplementedError(observations)

    try:
        mdp = MDP(o, t, **config)

        for i in range(n_envs):
            rollout(
                mdp,
                observation=o_render,
                show_render=show,
                label=task + str(i),
            )
    except Exception:
        traceback.print_exc()
        logger.error("[FAILED]")
        raise ValueError(
            f"Exception raised on task {task} and observations"
            f" {observations}."
        )
    logger.info("[PASSED]")


if __name__ == "__main__":
    for task in _TASKS:
        for observations in _OBSERVATIONS:
            test(
                task=task,
                observations=observations,
                vis_gen=_VIS_GEN,
                show=True,
            )
