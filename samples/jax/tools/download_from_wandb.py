__author__ = "R Devon Hjelm, Bogdan Mazoure, Florian Golemo"
__copyright__ = "Copyright (c) Microsoft Corporation and Mila - Quebec AI " \
                "Institute"
__license__ = "MIT"

import os
import json

import wandb
import time
import tqdm

from absl import app, flags

FLAGS = flags.FLAGS
flags.DEFINE_string("wandb_key", None, "W&B key")
flags.DEFINE_string("wandb_entity", "dummy_username",
                    "W&B entity (username or team name)")
flags.DEFINE_string("wandb_project", "dummy_project", "W&B project name")


def main(argv):
    api = wandb.Api(timeout=19)

    runs = api.runs(FLAGS.wandb_entity + "/" + FLAGS.wandb_project)

    for run in tqdm.tqdm(runs):
        params = json.loads(run.json_config)
        if 'large_run' not in params['run_id']['value']:
            continue

        env_name = params['env_name']['value']
        num_levels = params['num_train_levels']['value']
        seed = params['seed']['value']
        things, difficulty, _ = env_name.split('-')

        run_df = run.history(samples=int(1e6))

        columns = run_df.columns
        run_df.columns = [x.split('/')[-1] for x in columns]
        dir_ = os.path.join('../data')
        if not os.path.isdir(dir_):
            os.makedirs(dir_)

        run_df['things'] = things
        run_df['difficulty'] = difficulty
        run_df['num_levels'] = num_levels
        run_df['seed'] = seed

        run_df.to_csv(os.path.join(dir_,
                                   '%s_%s_%s_%d.csv' % (things, difficulty, num_levels, seed)))

        artifact = run.logged_artifacts()
        if len(artifact):
            artifact = artifact[0]
            fn = artifact.file().split(os.path.sep)[-1]
            artifact_dir = artifact.download(dir_)
            os.rename(os.path.join(dir_, fn), os.path.join(
                dir_, 'checkpoint_%s_%s_%s_%d' % (things, difficulty, num_levels, seed)))
        time.sleep(3.)


if __name__ == '__main__':
    app.run(main)
