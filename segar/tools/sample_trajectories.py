import argparse
from typing import Optional

import numpy as np
import pickle
from PIL import Image

from segar import get_sim, timeit
from segar.configs import get_env_config
from segar.mdps import MDP, RGBObservation
from segar.tasks.billiards import BilliardsInitialization, Billiards
from segar.tasks import PuttPuttInitialization, Invisiball, PuttPutt
from segar.sim import Simulator
from segar.rendering import Renderer


_TASKS = ('puttputt', 'invisiball', 'billiards')
_VIS_GEN = ('linear_ae', 'inception_linear')


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description='Draw trajectories from tasks.')

    parser.add_argument('task', type=str, choices=_TASKS,
                        help='Task to roll out.')

    parser.add_argument(
        '--sim_save_path',
        type=str,
        default=None,
        help='If specified, use this path to save the sim upon failure for '
             'debugging.')

    parser.add_argument('--iterations', type=int, default=500,
                        help='Number of iterations per rollout.')
    parser.add_argument('--n_envs', type=int, default=20,
                        help='Number of rollouts')
    parser.add_argument('--gif_out_path', type=str, default=None,
                        help='Directory where gifs are put.')
    parser.add_argument('--trajectory_out_path', type=str, default=None,
                        help='Directory where trajectories are put.')
    parser.add_argument('--show', action='store_true', default=False,
                        help='Show rendering.')
    parser.add_argument('--visual_gen_model', type=str, choices=_VIS_GEN,
                        default='linear_ae',
                        help='Which generative model to use to generate '
                             'pixel observations.')
    parser.add_argument('--visual_dist', type=str, default='baseline',
                        help='Which configuration set from which to select '
                             'the distribution of visual features.')
    parser.add_argument('--init_dist', type=str, default='default',
                        help='Which configuration set to draw initialization '
                             'from.')

    return parser.parse_args()


def save_gif(imgs: list[np.ndarray], out_path: str = None):
    # PIL doesn't create a full gif if images are redundant, so build in
    # values with a dummy pixel on upper left corner.
    for i in range(len(imgs)):
        imgs[i][0][0] = float(i)
    imgs = [Image.fromarray(img.copy()) for img in imgs]
    imgs[0].save(out_path, save_all=True, append_images=imgs[1:],
                 duration=50, loop=0)


def save_trajectories(trajectories_: list[np.ndarray], out_path: str = None):
    with open(out_path, 'wb') as f:
        pickle.dump(trajectories_, f)


def rollout(mdp: MDP, observation: RGBObservation = None,
            show_render: bool = False, label: Optional[str] = None):
    """For generating full rollouts.

    :param mdp: MDP that controls transitions and actions.
    :param observation: If provided, RGB observation space for
        visualization otherwise use MDP's.
    :param show_render: Whether to show rendering in real-time.
    :param label: The label for the rendering.
    :return: Renderings and trajectories.
    """

    sim_ = get_sim()
    mdp.reset()
    # In case the MDP doesn't do this with the visualization renderer.
    if observation is not None:
        observation.sample()
        observation.reset()

    renderings_ = []
    trajectories_ = []

    @timeit
    def render_observation():
        if observation is None:
            if show_render:
                obs = mdp.render(mode='human', label=label, agent_view=False)
            else:
                obs = mdp.render(mode='rgb_array', label=label,
                                 agent_view=False)
        else:
            obs = observation.render()
            if label is not None:
                observation.add_text(label)
            if show_render:
                observation.show(1)
        renderings_.append(obs.copy())

    render_observation()
    trajectories_.append(sim_.state)
    done = False

    action = mdp.demo_action()
    while not done:
        _, _, done, _ = mdp.step(action)
        action = mdp.demo_action()
        render_observation()
        trajectories_.append(sim_.state)
    return renderings_, trajectories_


def rollout_sim_only(sim: Simulator, n_steps: int = 100,
                     renderer: Renderer = None):
    """Roll out the sim only.

    :param sim: Simulator
    :param renderer: An optional renderer.
    :return: Trajectories.
    """

    trajectories_ = []
    imgs = []
    trajectories_.append(sim.state)
    if renderer is not None:
        renderer.reset(sim)
        imgs.append(renderer().copy())

    for _ in range(n_steps):
        sim.step()
        trajectories_.append(sim.state)
        if renderer is not None:
            imgs.append(renderer().copy())

    return trajectories_, imgs


if __name__ == '__main__':
    args = parse_args()
    sim = Simulator(save_path=args.sim_save_path)

    n_envs = args.n_envs

    config = dict(
        max_steps_per_episode=args.iterations,
        episodes_per_arena=1,
        sub_steps=1
    )

    visual_config = get_env_config(
        'visual', args.visual_gen_model, dist_name=args.visual_dist)
    init_config = get_env_config('initialization', args.task,
                                 dist_name=args.init_dist)

    observations = RGBObservation(resolution=256, config=visual_config,
                                  annotation=args.show)

    if args.task == 'puttputt':
        initialization = PuttPuttInitialization(config=init_config)
        task = PuttPutt(initialization)

    elif args.task == 'invisiball':
        initialization = PuttPuttInitialization(config=init_config)
        task = Invisiball(initialization)

    elif args.task == 'billiards':
        initialization = BilliardsInitialization(config=init_config)
        task = Billiards(initialization)

    else:
        raise NotImplementedError(args.task)

    mdp = MDP(observations, task, **config)

    for i in range(n_envs):
        renderings, trajectories = rollout(
            mdp, show_render=args.show, label=args.task)
        if args.gif_out_path is not None:
            gif_path = f'{args.gif_out_path}/{args.task}_{i}.gif'
            save_gif(renderings, gif_path)
        if args.trajectory_out_path is not None:
            traj_path = f'{args.trajectory_out_path}/{args.task}_{i}.pkl'
            save_trajectories(trajectories, traj_path)
