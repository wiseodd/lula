import warnings
warnings.filterwarnings('ignore')
import torch
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
import reluq


parser = argparse.ArgumentParser()
parser.add_argument('--ood_dset', default='imagenet', choices=['imagenet', 'uniform', 'smooth'])
args = parser.parse_args()

torch.manual_seed(9999)
np.random.seed(9999)

path = f'./pretrained_models'

train_loader = dl.CIFAR10(train=True, augm_flag=False)
val_loader, test_loader = dl.CIFAR10(train=False, val_size=2000)
print(len(train_loader.dataset), len(val_loader.dataset), len(test_loader.dataset))

num_classes = 10
data_shape = [3, 32, 32]

method_types = ['MAP', 'DE', 'LA', 'LULA']
method_strs = ['MAP', 'DE', 'LA', 'LA-LULA']
distortion_types = dl.CorruptedCIFAR10Dataset.distortions
severity_levels = range(1, 6)  # 1 ... 5

tab_acc = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
tab_mmc = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
tab_ece = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
tab_brier = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
tab_loglik = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))


def load_model(type='MAP'):
    def create_model():
        return wrn.WideResNet(16, 4, num_classes).cuda()

    if type == 'DE':
        K = 5
        model = [create_model() for _ in range(K)]
        state_dicts = torch.load(f'./pretrained_models/CIFAR10_wrn_de.pt')
        for k in range(K):
            model[k].load_state_dict(state_dicts[k])
            model[k].eval()
    else:
        model = create_model()
        model.load_state_dict(torch.load(f'./pretrained_models/CIFAR10_wrn_plain.pt'))
        model.eval()

    # Additionally, load these for LULA
    if type == 'LULA':
        lula_params = torch.load(f'./pretrained_models/kfla/CIFAR10_wrn_lula_{args.ood_dset}.pt')

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

    # For all distortions, for all severity
    for d in tqdm(distortion_types, leave=False):
        for s in tqdm(severity_levels, leave=False):
            shift_loader = dl.CorruptedCIFAR10(d, s)
            py_shift = predict_(shift_loader, model, model_name, params=params)
            targets = torch.cat([y for x, y in shift_loader], dim=0).numpy()

            tab_acc[model_str][d][str(s)].append(evalutil.get_acc(py_shift, targets))
            tab_mmc[model_str][d][str(s)].append(evalutil.get_mmc(py_shift))
            tab_ece[model_str][d][str(s)].append(evalutil.get_calib(py_shift, targets)[0])
            tab_brier[model_str][d][str(s)].append(evalutil.get_brier(py_shift, targets))
            tab_loglik[model_str][d][str(s)].append(evalutil.get_loglik(py_shift, targets))


evaluate('MAP')
evaluate('DE')
evaluate('LA')
evaluate('LULA')


# Save results
dir_name = f'results/CIFAR10C/'
dir_name += f'{args.ood_dset}'

if not os.path.exists(dir_name):
    os.makedirs(dir_name)

np.save(f'{dir_name}/mmcs', util.misc.ddict2dict(tab_mmc))
np.save(f'{dir_name}/accs', util.misc.ddict2dict(tab_acc))
np.save(f'{dir_name}/eces', util.misc.ddict2dict(tab_ece))
np.save(f'{dir_name}/briers', util.misc.ddict2dict(tab_brier))
np.save(f'{dir_name}/logliks', util.misc.ddict2dict(tab_loglik))
