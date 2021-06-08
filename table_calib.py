import numpy as np
import pickle
import os, sys, argparse
import scipy.stats as st
import pandas as pd
from collections import defaultdict


parser = argparse.ArgumentParser()
parser.add_argument('--ood_dset', default='imagenet', choices=['imagenet', 'smooth'])
args = parser.parse_args()


method_types = [
    'MAP', 'MAP-Temp', 'DE', 'DE-Temp',
    'LA', 'LA-LULA', 'LA-OOD', 'LA-LULA-OOD', 'MAP-OE', 'LA-LULA-OE'
]
method2str = {
    'MAP': 'MAP', 'MAP-Temp': 'MAP-Temp', 'DE': 'DE', 'DE-Temp': 'DE-Temp',
    'LA': 'LA', 'LA-OOD': 'LLLA', 'LA-LULA': 'LA-LULA', 'LA-LULA-OOD': 'LLLA-LULA', 'MAP-OE': 'OE', 'LA-LULA-OE': 'OE-LULA'
}

datasets = ['MNIST', 'SVHN', 'CIFAR10', 'CIFAR100']
metric2str = {'ece': 'ECE', 'acc': 'Acc.'}

path = f'./results/OOD/{args.ood_dset}/wrn'
_, _, filenames = next(os.walk(path))

TEXTBF = '\\textbf'


def get_dfs(dset, type='ece'):
    def cond(fname, str):
        return f'_{dset.lower()}_' in fname and str in fname

    temps = []
    fnames = [fname for fname in filenames if cond(fname, f'_{type}_')]
    d_all = {k: [] for k in method_types}

    for fname in fnames:
        d = np.load(f'{path}/{fname}', allow_pickle=True).item()

        for k in method_types:
            d_all[k].append(d[k])

    df = pd.DataFrame(d_all)

    return df.mean(), df.sem()


metrics = ['ece', 'acc']
vals = {m: {'ece': [], 'acc': []} for m in method_types}

for dset in datasets:
    for metric in metrics:
        mult = 1 if metric == 'ece' else 100
        df_mean, df_sem = get_dfs(dset, type=metric)

        for method in method_types:
            vals[method][metric].append(f'{df_mean[method]*mult:.1f}$\\pm${df_sem[method]*mult:.1f}')

print()
for i, metric in enumerate(metrics):
    arrow = '$\\downarrow$' if metric == 'ece' else '$\\uparrow$'
    print(f'\\textbf{{{metric2str[metric]}}} {arrow} \\\\')

    for method in method_types:
        if method == 'LA' and len(method_types) > 2:
            print('\\midrule')
        print(f'{method2str[method]} & {" & ".join(vals[method][metric])} \\\\')

    if i < len(metrics)-1:
        print('\n\\midrule\n\\midrule\n')

print()
