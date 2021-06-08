import numpy as np
from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt
import tikzplotlib
import os, sys, argparse
import seaborn as sns

sns.set_style('whitegrid')
sns.set_palette('colorblind')


parser = argparse.ArgumentParser()
parser.add_argument('--ood_dset', default='imagenet', choices=['imagenet', 'uniform', 'smooth'])
args = parser.parse_args()

methods = ['MAP', 'DE', 'LA', 'LA-LULA']
metric2str = {'acc': 'Acc.', 'mmc': 'MMC', 'ece': 'ECE',
              'brier': 'Brier', 'loglik': 'LogLik'}

palette = {
    'MAP': '#0173B2', 'DE': '#CC78BC', 'LA': '#ECE133', 'LA-LULA': '#029E73'
}

path = f'results/CIFAR10C/{args.ood_dset}'


def plot(metric='ece'):
    metric_str = metric2str[metric]
    data = {'Method': [], 'Severity': [], metric_str: []}

    for method in methods:
        vals = np.load(f'{path}/{metric}s.npy', allow_pickle=True).item()

        for distortion in vals[method].keys():
            if distortion == 'clean':
                continue

            for severity in vals[method][distortion].keys():
                data['Method'].append(method)
                data['Severity'].append(int(severity))
                data[metric_str].append(vals[method][distortion][severity][0])


    df = pd.DataFrame(data)

    df_filtered = df[df['Method'].isin(methods)]

    sns.boxplot(
        data=df_filtered, x='Severity', y=metric_str, hue='Method', fliersize=0, width=0.5,
        palette=palette
    )

    dir_name = f'figs/CIFAR10C/{args.ood_dset}'
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    tikzplotlib.save(f'{dir_name}/cifar10c_{metric}.tex')
    plt.savefig(f'{dir_name}/cifar10c_{metric}.pdf', bbox_inches='tight')
    plt.close()


plot(metric='loglik')
plot(metric='ece')
plot(metric='brier')
plot(metric='mmc')
