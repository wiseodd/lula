import numpy as np
import pickle
import os, sys, argparse
import matplotlib
import matplotlib.cm as cm
from math import *
import tikzplotlib
import tqdm
import seaborn as sns

sns.set_style('whitegrid')
sns.set_palette('colorblind')


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='MNISTR', choices=['MNISTR', 'MNISTT'])
parser.add_argument('--ood_dset', default='imagenet', choices=['imagenet', 'uniform', 'smooth'])
args = parser.parse_args()

matplotlib.rcParams['figure.figsize'] = (11,8)
matplotlib.rcParams['font.size'] = 30
matplotlib.rcParams['font.family'] = "serif"
matplotlib.rcParams['font.serif'] = 'Times'
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['lines.linewidth'] = 1.0
plt = matplotlib.pyplot


path = f'./results/{args.dataset}/{args.ood_dset}'
_, _, filenames = next(os.walk(path))

method_types = ['MAP', 'DE', 'LA', 'LA-LULA']
x = list(range(0, 181, 15)) if args.dataset == 'MNISTR' else list(range(0, 15, 2))

palette = {
    'MAP': '#0173B2', 'DE': '#CC78BC', 'LA': '#ECE133', 'LA-LULA': '#029E73'
}


def load(str='mmc'):
    return np.load(f'{path}/{metric}s.npy', allow_pickle=True).item()


def plot(vals, name, legend=False):
    plt.figure()

    for method in method_types:
        v = vals[method]
        y = [v[angle][0] for angle in x]
        plt.plot(x, y, lw=3, label=method, alpha=1, c=palette[method])

    ticks = range(0, 181, 30) if args.dataset == 'MNISTR' else range(0, 15, 4)
    plt.xticks(ticks)
    plt.xlim(0, 180 if args.dataset == 'MNISTR' else 14)

    if name != 'loglik':
        plt.ylim(bottom=0)

    if name in ['mmc', 'acc', 'aur']:
        plt.ylim(top=1)

    if legend:
        plt.legend(loc='lower right')

    dir_name = f'figs/{args.dataset}/{args.ood_dset}'
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    tikzplotlib.save(f'{dir_name}/{args.dataset.lower()}_{name}.tex')
    plt.savefig(f'{dir_name}/{args.dataset.lower()}_{name}.pdf', bbox_inches='tight')


metrics = ['acc', 'loglik', 'ece', 'brier', 'mmc']

for metric in tqdm.tqdm(metrics):
    vals = load(metric)
    plot(vals, metric, legend=True)
