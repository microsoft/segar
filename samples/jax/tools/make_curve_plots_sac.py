import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter
import matplotlib.ticker as ticker
import matplotlib.patches as mpatches
import seaborn as sns
import sys
import palettable

from segar.envs.env import SEGAREnv
from utils import rollouts_random

def scale(x, pos):
    return '%dk' % (x * 1e-3)
formatter = FuncFormatter(scale)

folder = 'SAC'
metric = 'Returns'
mode = 'txt'
mode = 'wandb'

if mode == 'txt':
    files = glob.glob("../data/%s/drq_faster/**/*.txt" % folder)

    df = []
    for f in files:
        df_ = pd.read_csv(f, sep=" ")
        df_.columns = ['Frames', 'Returns']
        env_name, _, _, _, num_levels, _, _ = f.split('/')[-2].split('_')
        things, difficulty, _ = env_name.split('-')
        df_['Type'] = things
        df_['Difficulty'] = difficulty
        df_['Number of levels'] = num_levels
        df_['Returns'] = df_['Returns'].ewm(10).mean()
        df.append(df_)

    df = pd.concat(df).reset_index(drop=True)
else:
    files = glob.glob("../data/%s/*.csv" % folder)

    df = []
    for f in files:
        algo = f.split('/')[-2]
        df_ = pd.read_csv(f)
        seed = df_['seed'].unique()
        try:
            df_['Returns'] = df_['average_returnss'].ewm(10).mean()
        except:
            continue
        df_['algo'] = algo
        df.append(df_)

    df = pd.concat(df).reset_index(drop=True)

    # df['Returns (test)'] = df['eprew_test'].ewm(10).mean()
    df['Frames'] = df['_step']
    df['Type'] = df['things']
    df['Difficulty'] = df['difficulty']
    df['Number of levels'] = df['num_levels']
    df = df[['Frames','Type','Difficulty','Number of levels','Returns']]
    df = df[~df['Returns'].isna()]
# df['Returns (test)'] = df['eprew_test'].ewm(10).mean()

# del df['eprew_train']
# import ipdb;ipdb.set_trace()
n_rollouts = 1
# Random agent rollouts
random_dict = {}
for task in ['empty', 'tiles', 'objects']:
    d1 = {}
    for difficulty in ['easy', 'medium', 'hard']:
        d2 = {}
        for num_levels in df['Number of levels'].unique().astype(int):
            if task == 'empty':
                env_name = "%s-%s-rgb" % (task, difficulty)
            else:
                env_name = "%sx1-%s-rgb" % (task, difficulty)
            env = SEGAREnv(
                env_name,
                num_envs=1,
                num_levels=num_levels,
                framestack=1,
                resolution=64,
                max_steps=100,
                _async=False,
                deterministic_visuals=False,
                seed=123)
            returns_train = np.mean(rollouts_random(env, n_rollouts=n_rollouts))
            d2[num_levels] = returns_train
        d1[difficulty] = d2
    random_dict[task.capitalize()] = d1

def replace_with_random(x):
    x['Returns'] = random_dict[x['Type'].unique().item()][x['Difficulty'].unique().item()][int(x['Number of levels'].unique().item())]
    return x

df['Type'] = df['Type'].replace({'empty':'Empty','tilesx1':'Tiles','objectsx1':'Objects'})
random_df = df.groupby(['Type', 'Difficulty', 'Number of levels']).apply(replace_with_random).reset_index(drop=True).copy()
df['Algorithm'] = 'SAC'
random_df['Algorithm'] = 'Random agent'
df = pd.concat([df, random_df], axis=0)

def score(x, env_name):
    return x

n_cols = len(df['Number of levels'].unique())
palette = 'Blues' #palettable.cartocolors.qualitative.Pastel_10.mpl_colors[:n_cols]

tasks = ['Empty', 'Objects', 'Tiles']
with sns.plotting_context("notebook", font_scale=1.5):
    fig, ax = plt.subplots()
    g = sns.relplot(
        x='Frames',
        y=metric,
        hue='Number of levels',
        col='Type',
        row='Difficulty',
        hue_order = ['1','10','25','50','100','200'],
        # style_order = ['easy','medium','hard'],
        row_order=['easy','medium','hard'],
        palette=palette,
        # sharey=False,
        # ci=75,
        kind='line',
        data=df[df['Algorithm']=='SAC'],
        ax=ax)
    # ax2 = ax.twinx()
    for row_ax in g.axes:
        for col_ax in row_ax:
            title = col_ax.title.get_text()
            difficulty, type_ = list(map(lambda x:x.split('=')[1].strip(),title.split('|')))
            col_ax.title.set_text('')
            sub_df = random_df[(random_df['Type']==type_) & (random_df['Difficulty']==difficulty)]
            sns.lineplot(x='Frames', y=metric, hue='Number of levels', hue_order = ['1','10','25','50','100','200'], palette='Reds', data=sub_df, ax=col_ax)
            col_ax.legend([],[], frameon=False)

    # g.map(sns.lineplot, x='Frames',
    #     y=metric,
    #     hue='Number of levels',
    #     # col='Type',
    #     # row='Difficulty',
    #     hue_order = ['1','10','25','50','100','200'],
    #     # style_order = ['easy','medium','hard'],
    #     # row_order=['easy','medium','hard'],
    #     palette='Reds',
    #     # sharey=False,
    #     # ci=75,
    #     # kind='line',
    #     data=df[df['Algorithm']=='Random agent'])
    # g2 = sns.relplot(
    #     x='Frames',
    #     y=metric,
    #     hue='Number of levels',
    #     col='Type',
    #     row='Difficulty',
    #     hue_order = ['1','10','25','50','100','200'],
    #     # style_order = ['easy','medium','hard'],
    #     row_order=['easy','medium','hard'],
    #     palette='Reds',
    #     # sharey=False,
    #     # ci=75,
    #     kind='line',
    #     data=df[df['Algorithm']=='Random agent'],
    #     ax=ax2)
    for ax, task in zip(g.axes.flatten(), tasks):
        ax.set_title(task)
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '%dK'%(x/1000) ))
    for i, ax in enumerate(g.axes[:,0]):
        ax.set_ylabel(tasks[i])
        #ax.set(xticks=([0, 4]))
        # ax.set_major_formatter(formatter)
    # sns.move_legend(g, "lower center", bbox_to_anchor=(.5, 1), ncol=3)
    # plt.tight_layout()
    plt.savefig('../plots/05_%s_%s.png' % (folder, metric))
