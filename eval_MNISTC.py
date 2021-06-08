import warnings
warnings.filterwarnings('ignore')
import torch
from torchvision.transforms import functional as TF
import torch.utils.data as data_utils
import numpy as np
from models import wrn
from laplace import kfla
import laplace.util as lutil
import util.evaluation as evalutil
import util.dataloaders as dl
import util.misc
from math import *
from tqdm import tqdm, trange
import argparse
import os, sys
from tqdm import tqdm, trange
from collections import defaultdict

import lula


parser = argparse.ArgumentParser()
parser.add_argument('--transform', default='rotation', choices=['rotation', 'translation'])
parser.add_argument('--repeat', type=int, default=1)
parser.add_argument('--ood_dset', default='imagenet', choices=['imagenet', 'uniform', 'smooth'])
args = parser.parse_args()

torch.manual_seed(9999)
np.random.seed(9999)

path = f'./pretrained_models'

train_loader = dl.MNIST(train=True, augm_flag=False)
val_loader, test_loader = dl.MNIST(train=False, val_size=2000)
targets = torch.cat([y for x, y in test_loader], dim=0).numpy()
print(len(train_loader.dataset), len(val_loader.dataset), len(test_loader.dataset))

method_types = ['MAP', 'DE', 'LA', 'LULA']
method_strs = ['MAP', 'DE', 'LA', 'LA-LULA']
rotation_angles = range(15, 181, 15)
translation_pixels = range(2, 15, 2)

dla_statedict = None
dla_ood_statedict = None

tab_acc = defaultdict(lambda: defaultdict(list))
tab_mmc = defaultdict(lambda: defaultdict(list))
tab_ece = defaultdict(lambda: defaultdict(list))
tab_brier = defaultdict(lambda: defaultdict(list))
tab_loglik = defaultdict(lambda: defaultdict(list))


def load_model(type='MAP'):
    def create_model():
        return wrn.WideResNet(16, 4, 10, 1).cuda()

    if type == 'DE':
        K = 5
        model = [create_model() for _ in range(K)]
        state_dicts = torch.load(f'./pretrained_models/MNIST_wrn_de.pt')
        for k in range(K):
            model[k].load_state_dict(state_dicts[k])
            model[k].eval()
    else:
        model = create_model()
        model.load_state_dict(torch.load(f'./pretrained_models/MNIST_wrn_plain.pt'))
        model.eval()

    # Additionally, load these for LULA
    if type == 'LULA':
        lula_params = torch.load(f'./pretrained_models/kfla/MNIST_wrn_lula_{args.ood_dset}.pt')

        if args.ood_dset == 'best':
            state_dict, n_units, noise = lula_params
            print(f'LULA uses this OOD dataset: {noise}')
        else:
            state_dict, n_units = lula_params

        model = lula.model.LULAModel_LastLayer(model, n_units).cuda()
        model.to_gpu()
        model.load_state_dict(state_dict)
        model.disable_grad_mask()
        model.unmask()
        model.eval()

    if type in ['LA', 'LULA']:
        var0 = torch.tensor(1/(5e-4*len(train_loader.dataset))).float().cuda()
        model = kfla.KFLA(model)
        model.get_hessian(train_loader)
        model.estimate_variance(var0)

    return model


def predict_(test_loader, model, model_name, params=None):
    assert model_name in method_types

    if model_name in ['LA', 'LULA']:
        py = lutil.predict(test_loader, model, n_samples=20)
    elif model_name == 'DE':
        py = evalutil.predict_ensemble(test_loader, model)
    elif model_name == 'DKL':
        model_dkl, likelihood = model
        py = evalutil.predict_dkl(test_loader, model_dkl, likelihood, n_samples=20)
    else:  # MAP
        py = evalutil.predict(test_loader, model)

    return py.cpu().numpy()


def evaluate(model_name):
    assert model_name in method_types

    model = load_model(model_name)
    params = None

    if model_name == 'LULA':
        model_str = 'LA-LULA'
    else:
        model_str = model_name

    print(f'Processing for {model_str}')

    # Clean, shift = 0
    py_clean = predict_(test_loader, model, model_name, params=params)

    tab_acc[model_str][0].append(evalutil.get_acc(py_clean, targets))
    tab_mmc[model_str][0].append(evalutil.get_mmc(py_clean))
    tab_ece[model_str][0].append(evalutil.get_calib(py_clean, targets)[0])
    tab_brier[model_str][0].append(evalutil.get_brier(py_clean, targets))
    tab_loglik[model_str][0].append(evalutil.get_loglik(py_clean, targets))

    shifts = rotation_angles if args.transform == 'rotation' else translation_pixels

    for shift in shifts:
        if args.transform == 'rotation':
            # Rotate the test set
            X_shift = torch.cat([TF.rotate(x, shift) for x, _ in test_loader], dim=0)
        else:
            # Translate the test set horizontally
            X_shift = torch.cat([
                TF.affine(x, angle=0, translate=[shift, 0], scale=1, shear=0)
                for x, _ in test_loader
            ], dim=0)

        y_shift = torch.cat([y for _, y in test_loader], dim=0)
        shift_dset = data_utils.TensorDataset(X_shift, y_shift)
        shift_loader = data_utils.DataLoader(shift_dset, batch_size=128, pin_memory=True)

        py_shift = predict_(shift_loader, model, model_name, params=params)
        tab_acc[model_str][shift].append(evalutil.get_acc(py_shift, targets))
        tab_mmc[model_str][shift].append(evalutil.get_mmc(py_shift))
        tab_ece[model_str][shift].append(evalutil.get_calib(py_shift, targets)[0])
        tab_brier[model_str][shift].append(evalutil.get_brier(py_shift, targets))
        tab_loglik[model_str][shift].append(evalutil.get_loglik(py_shift, targets))


for _ in trange(args.repeat):
    evaluate('MAP')
    evaluate('DE')
    evaluate('LA')
    evaluate('LULA')


# Save results
transformed_dset = 'MNISTR' if args.transform == 'rotation' else 'MNISTT'
dir_name = f'results/{transformed_dset}/'
dir_name += f'{args.ood_dset}'

if not os.path.exists(dir_name):
    os.makedirs(dir_name)

# To convert defaultdict to dict
import json

np.save(f'{dir_name}/mmcs', util.misc.ddict2dict(tab_mmc))
np.save(f'{dir_name}/accs', util.misc.ddict2dict(tab_acc))
np.save(f'{dir_name}/eces', util.misc.ddict2dict(tab_ece))
np.save(f'{dir_name}/briers', util.misc.ddict2dict(tab_brier))
np.save(f'{dir_name}/logliks', util.misc.ddict2dict(tab_loglik))
