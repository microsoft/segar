import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter
import matplotlib.ticker as ticker
import matplotlib.patches as mpatches
import seaborn as sns
import palettable

def scale(x, pos):
    return '%dk' % (x * 1e-3)
formatter = FuncFormatter(scale)

folder = 'PPO'
metric = 'Returns'

files = glob.glob("../data/%s/*.csv" % folder)

df = []
for f in files:
    algo = f.split('/')[-2]
    df_ = pd.read_csv(f)
    seed = df_['seed'].unique()
    # if seed != 1:
    #     continue
    df_['algo'] = algo
    df.append(df_)

df = pd.concat(df).reset_index(drop=True)

df['Returns'] = df['eprew_train'].ewm(10).mean()
# df['Returns (test)'] = df['eprew_test'].ewm(10).mean()
df['Frames'] = df['_step']
df['Type'] = df['things']
df['Difficulty'] = df['difficulty']
df['Number of levels'] = df['num_levels']
del df['things'], df['difficulty'], df['num_levels'], df['_step'], df['eprew_train'], df['eprew_test']

def score(x, env_name):
    return x


n_cols = len(df['Number of levels'].unique())

palette = palettable.cartocolors.qualitative.Pastel_10.mpl_colors[:n_cols]

tasks = ['Empty', 'Objects', 'Tiles']
df = df[df['seed']==1]
with sns.plotting_context("notebook", font_scale=1.5):
    g = sns.relplot(
        x='Frames',
        y=metric,
        hue='Number of levels',
        style='Difficulty',
        col='Type',
        # col_order=['easy','medium','hard'],
        palette=palette,
        # sharey=False,
        # ci=75,
        kind='line',
        data=df)
    for ax, task in zip(g.axes.flatten(), tasks):
        ax.set_title(task)
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '%dK'%(x/1000) ))
        #ax.set(xticks=([0, 4]))
        # ax.set_major_formatter(formatter)
    # sns.move_legend(g, "lower center", bbox_to_anchor=(.5, 1), ncol=3)
    # plt.tight_layout()
    plt.savefig('../plots/05_%s_%s.png' % (folder, metric))
