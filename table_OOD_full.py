import numpy as np
import pickle
import os, sys, argparse
import scipy.stats as st
import pandas as pd
from collections import defaultdict


parser = argparse.ArgumentParser()
parser.add_argument('--metric', default='fpr95', choices=['mmc', 'fpr95', 'auroc', 'auprc'])
parser.add_argument('--ood_dset', default='imagenet', choices=['imagenet', 'smooth', 'uniform'])
args = parser.parse_args()


method_types = [
    'MAP', 'MAP-Temp', 'DE', 'DE-Temp',
    'LA', 'LA-LULA', 'LA-OOD', 'LA-LULA-OOD', 'MAP-OE', 'LA-LULA-OE'
]
ood_test_sets = {
    'MNIST': ['EMNIST', 'KMNIST', 'FMNIST', 'GrayCIFAR10', 'UniformNoise', 'Noise'],
    'SVHN': ['CIFAR10', 'LSUN', 'CIFAR100', 'FMNIST3D', 'UniformNoise', 'Noise'],
    'CIFAR10': ['SVHN', 'LSUN', 'CIFAR100', 'FMNIST3D', 'UniformNoise', 'Noise'],
    'CIFAR100': ['SVHN', 'LSUN', 'CIFAR10', 'FMNIST3D', 'UniformNoise', 'Noise'],
}
ood_test_names = {
    'MNIST': ['EMNIST', 'KMNIST', 'FMNIST', 'GrayCIFAR10', 'UniformNoise', 'Noise'],
    'SVHN': ['CIFAR10', 'LSUN', 'CIFAR100', 'FMNIST3D', 'UniformNoise', 'Noise'],
    'CIFAR10': ['SVHN', 'LSUN', 'CIFAR100', 'FMNIST3D', 'UniformNoise', 'Noise'],
    'CIFAR100': ['SVHN', 'LSUN', 'CIFAR10', 'FMNIST3D', 'UniformNoise', 'Noise'],
}
datasets = ood_test_sets.keys()
metric2str = {'fpr95': 'FPR@95', 'mmc': 'MMC', 'auroc': 'AUROC', 'auprc': 'AUPRC'}
method2str = {
    'MAP': 'MAP', 'MAP-Temp': 'MAP-Temp', 'DE': 'DE', 'DE-Temp': 'DE-Temp',
    'LA': 'LA', 'LA-OOD': 'LLLA', 'LA-LULA': 'LA-LULA', 'LA-LULA-OOD': 'LLLA-LULA', 'MAP-OE': 'OE', 'LA-LULA-OE': 'OE-LULA'
}

path = f'./results/OOD/{args.ood_dset}/wrn'
_, _, filenames = next(os.walk(path))

TEXTBF = '\\textbf'


def get_dfs(dset, type='mmc', return_dicts=False):
    def cond(fname, str):
        return f'_{dset.lower()}_' in fname and str in fname

    temps = []

    fnames = [fname for fname in filenames if cond(fname, f'_{type}_')]

    for fname in fnames:
        d = np.load(f'{path}/{fname}', allow_pickle=True).item()

        for k in list(d.keys()):
            if not d[k]:  # d[k] is an empty dict
                del d[k]

        if return_dicts:
            temps.append(d)
        else:
            temps.append(pd.DataFrame(d))

    if return_dicts:
        return temps

    df = pd.concat(temps, ignore_index=False)
    df = df[(m for m in method_types)]
    df = df.drop(index=list(set(ood_test_sets[dset])-set(ood_test_names[dset])))
    df_mean = df.groupby(df.index).mean() * 100
    df_std = df.groupby(df.index).sem() * 100

    return df_mean, df_std


def get_str(test_dset, method_type, df_mean, df_std, bold=True):
    try:
        mean = df_mean[method_type][test_dset]
        std = df_std[method_type][test_dset]
    except KeyError:
        mean, std = np.NaN, np.NaN

    mean = round(mean, 1)

    if not np.isnan(mean):
        mean_str = f'\\textbf{{{mean:.1f}}}' if bold else f'{mean:.1f}'
        str = f'{mean_str}'

        if method_type not in ['MAP', 'DE']:
            str += f'$\\pm${std:.1f}'
    else:
        str = '-'

    return str



values = {dset: defaultdict(list) for dset in datasets}

for dset in datasets:
    df_mean, df_std = get_dfs(dset, type=args.metric)

    for test_dset in df_mean.index:
        str = []

        for method_type in method_types:
            str.append(get_str(test_dset, method_type, df_mean, df_std, bold=False ))

        values[dset][test_dset] = str

print()


for i, dset in enumerate(datasets):
    print(f'\\textbf{{{dset}}} & {" & ".join(values[dset][dset])} \\\\')

    for ood_dset in ood_test_names[dset]:
        print(f'{ood_dset} & {" & ".join(values[dset][ood_dset])} \\\\')

    if i < len(datasets)-1:
        print()
        print('\\midrule')
        print()

print()
