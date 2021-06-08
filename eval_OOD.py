import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import torch
import numpy as np
from models import models, wrn
from laplace import kfla
import laplace.util as lutil
from util.evaluation import *
import util.dataloaders as dl
from math import *
from tqdm import tqdm, trange
import argparse
import os, sys
from tqdm import tqdm, trange
from collections import defaultdict
from pycalib.calibration_methods import TemperatureScaling

import lula.model


parser = argparse.ArgumentParser()
parser.add_argument('--randseed', type=int, default=9999)
parser.add_argument('--dataset', default='MNIST')
parser.add_argument('--ood_dset', default='imagenet', choices=['best', 'uniform', 'smooth', 'imagenet'])
parser.add_argument('--dont_save', action='store_true', default=False)
args = parser.parse_args()

assert args.dataset in ['MNIST', 'CIFAR10', 'SVHN', 'CIFAR100']

torch.cuda.set_device(0)
np.random.seed(args.randseed)
torch.manual_seed(args.randseed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

num_classes = 100 if args.dataset == 'CIFAR100' else 10
num_channel = 1 if args.dataset == 'MNIST' else 3

train_loader = dl.datasets_dict[args.dataset](train=True, augm_flag=False)
val_loader, test_loader = dl.datasets_dict[args.dataset](train=False, val_size=2000)
targets_val = torch.cat([y for x, y in val_loader], dim=0).numpy()
targets = torch.cat([y for x, y in test_loader], dim=0).numpy()
print(len(train_loader.dataset), len(val_loader.dataset), len(test_loader.dataset))

# For LA-OOD.
ood_val_loader, _ = dl.ImageNet32(dataset=args.dataset, train=False, val_size=2000)

data_shape = [1, 28, 28] if args.dataset == 'MNIST' else [3, 32, 32]

ood_noise_names = ['UniformNoise', 'Noise']
ood_test_names = {
    'MNIST': ['EMNIST', 'KMNIST', 'FMNIST', 'GrayCIFAR10'],
    'CIFAR10': ['SVHN', 'LSUN', 'CIFAR100', 'FMNIST3D'],
    'SVHN': ['CIFAR10', 'LSUN', 'CIFAR100', 'FMNIST3D'],
    'CIFAR100': ['SVHN', 'LSUN', 'CIFAR10', 'FMNIST3D'],
}

ood_names = ood_test_names[args.dataset] + ood_noise_names
ood_test_loaders = {}

for ood_name in ood_test_names[args.dataset]:
    ood_test_loaders[ood_name] = dl.datasets_dict[ood_name](train=False)

for ood_name in ood_noise_names:
    ood_test_loaders[ood_name] = dl.datasets_dict[ood_name](dataset=args.dataset, train=False, size=2000)

method_types = [
    'MAP', 'MAP-Temp', 'MAP-OE', 'DE', 'DE-Temp', 'LA', 'LA-OOD', 'LULA', 'LULA-OOD', 'LULA-OE'
]
method_strs = [
    'MAP', 'MAP-Temp', 'MAP-OE', 'DE', 'DE-Temp', 'LA', 'LA-OOD', 'LA-LULA', 'LA-LULA-OOD', 'LA-LULA-OE'
]

tab_acc = {}
tab_ece = {}
tab_mmc = {mt: {} for mt in method_strs}
tab_auroc = {mt: {} for mt in method_strs}
tab_auprc = {mt: {} for mt in method_strs}
tab_fpr95 = {mt: {} for mt in method_strs}


def load_model(type='MAP'):
    def create_model():
        # OE doesn't play well with WRN on MNIST
        if args.dataset == 'MNIST' and 'OE' in type:
            return models.LeNetMadry().cuda()
        else:
            return wrn.WideResNet(16, 4, num_classes, num_channel).cuda()

    if type in ['DE', 'DE-Temp']:
        K = 5
        model = [create_model() for _ in range(K)]
        state_dicts = torch.load(f'./pretrained_models/{args.dataset}_wrn_de.pt')
        for k in range(K):
            model[k].load_state_dict(state_dicts[k])
            model[k].eval()
    else:
        model = create_model()
        modifier = 'plain' if 'OE' not in type else 'oe'

        if args.dataset == 'MNIST' and 'OE' in type:
            model.load_state_dict(torch.load(f'./pretrained_models/{args.dataset}_{modifier}.pt'))
        else:
            model.load_state_dict(torch.load(f'./pretrained_models/{args.dataset}_wrn_{modifier}.pt'))

        model.eval()

    # Additionally, load these for LULA
    if 'LULA' in type:
        path = 'kfla' + ('/oe' if type == 'LULA-OE' else '')

        if args.dataset == 'MNIST' and type == 'LULA-OE':
            lula_params = torch.load(f'./pretrained_models/{path}/{args.dataset}_lula_{args.ood_dset}.pt')
        else:
            lula_params = torch.load(f'./pretrained_models/{path}/{args.dataset}_wrn_lula_{args.ood_dset}.pt')

        if args.ood_dset == 'best':
            state_dict, n_units, noise = lula_params
            print(f'LULA uses this OOD dataset: {noise}')
        else:
            state_dict, n_units = lula_params

        print(n_units)

        model = lula.model.LULAModel_LastLayer(model, n_units).cuda()
        model.to_gpu()
        model.load_state_dict(state_dict)
        model.disable_grad_mask()
        model.unmask()
        model.eval()

    if type in ['LA', 'LA-OOD', 'LULA', 'LULA-OOD', 'LULA-OE']:
        model = kfla.KFLA(model)
        model.get_hessian(train_loader)

        # For grid search
        interval = torch.logspace(-6, 0, 100)

        if type in ['LA-OOD', 'LULA-OOD']:
            # # Following Kristiadi et al. ICML 2020: https://github.com/wiseodd/last_layer_laplace
            # var0 = model.gridsearch_var0(val_loader, ood_val_loader, interval, n_classes=num_classes, lam=0.25)
            # print(var0)

            # sys.exit(0)

            if type == 'LA-OOD':
                var0s = {
                    'MNIST': 0.3275, 'SVHN': 0.4977, 'CIFAR10': 0.4329, 'CIFAR100': 0.0705
                }
            else: # 'LULA-OOD'
                var0s = {
                    'MNIST': 0.0115, 'SVHN': 0.0705, 'CIFAR10': 0.4329, 'CIFAR100': 0.0705
                }

            var0 = torch.tensor(var0s[args.dataset]).float().cuda()
        else:
            # For standard Laplace and LULA, use the exact prior variance
            var0 = torch.tensor(1/(5e-4*len(train_loader.dataset))).float().cuda()

        model.estimate_variance(var0)

    # Temperature scaling; T=1 means no temp. scaling
    if type == 'MAP-Temp':
        logits = predict_logit(val_loader, model).cpu().numpy()
        T = TemperatureScaling().fit(logits, targets_val).T
    elif type == 'DE-Temp':
        logits = 0

        # Average, then find the optimal temp. https://arxiv.org/abs/2007.08792
        for m in model:
            logits += 1/len(model) * predict_logit(val_loader, m).cpu().numpy()

        T = TemperatureScaling().fit(logits, targets_val).T
    else:
        T = 1

    return model, T


def predict_(test_loader, model, model_name, params=None, T=1):
    assert model_name in method_types

    if model_name in ['LA', 'LA-OOD', 'LULA', 'LULA-OOD', 'LULA-OE']:
        py = lutil.predict(test_loader, model, n_samples=20)
    elif model_name in ['DE', 'DE-Temp']:
        py = predict_ensemble(test_loader, model, T=T)
    else:  # MAP
        py = predict(test_loader, model, T=T)

    return py.cpu().numpy()


def evaluate(model_name):
    assert model_name in method_types

    model, T = load_model(model_name)
    params = None

    if 'LULA' in model_name:
        model_str = f'LA-LULA{model_name[4:]}'
    else:
        model_str = model_name

    py_in = predict_(test_loader, model, model_name, params=params, T=T)
    acc_in = np.mean(np.argmax(py_in, 1) == targets)
    conf_in = get_confidence(py_in)
    mmc = conf_in.mean()
    ece, mce = get_calib(py_in, targets)
    tab_mmc[model_str][args.dataset] = mmc
    tab_auroc[model_str][args.dataset] = None
    tab_auprc[model_str][args.dataset] = None
    tab_fpr95[model_str][args.dataset] = None
    tab_acc[model_str] = acc_in
    tab_ece[model_str] = ece
    print(f'[In, {model_str}] Accuracy: {acc_in:.3f}; ECE: {ece:.3f}; MCE: {mce:.3f}; MMC: {mmc:.3f}')

    for ood_name, ood_test_loader in ood_test_loaders.items():
        py_out = predict_(ood_test_loader, model, model_name, params=params, T=T)
        conf = get_confidence(py_out)
        mmc = conf.mean()
        aur = get_auroc(py_in, py_out)
        aupr = get_aupr(py_in, py_out)
        fpr95, _ = get_fpr95(py_in, py_out)
        tab_mmc[model_str][ood_name] = mmc
        tab_auroc[model_str][ood_name] = aur
        tab_auprc[model_str][ood_name] = aupr
        tab_fpr95[model_str][ood_name] = fpr95
        print(f'[Out-{ood_name}, {model_str}] MMC: {mmc:.3f}; AUROC: {aur:.3f}; AUPRC: {aupr:.3f}; FPR@95: {fpr95:.3f}')


evaluate('MAP')
print()
evaluate('MAP-Temp')
print()
evaluate('MAP-OE')
print()
evaluate('DE')
print()
evaluate('DE-Temp')
print()
evaluate('LA')
print()
evaluate('LA-OOD')
print()
evaluate('LULA')
print()
evaluate('LULA-OOD')
print()
evaluate('LULA-OE')
print()


if not args.dont_save:
    # Save dict
    dir_name = f'results/OOD/{args.ood_dset}'
    suffix = f'{args.dataset.lower()}_{args.randseed}'

    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    np.save(f'{dir_name}/tab_acc_{suffix}', tab_acc)
    np.save(f'{dir_name}/tab_ece_{suffix}', tab_ece)
    np.save(f'{dir_name}/tab_mmc_{suffix}', tab_mmc)
    np.save(f'{dir_name}/tab_auroc_{suffix}', tab_auroc)
    np.save(f'{dir_name}/tab_auprc_{suffix}', tab_auprc)
    np.save(f'{dir_name}/tab_fpr95_{suffix}', tab_fpr95)
