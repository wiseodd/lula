import torch
from torch import distributions as dist
from torch import nn, optim
import torch.nn.functional as F
import numpy as np
from math import *
from models.models import LeNetMadry
from models import wrn
from tqdm.auto import tqdm, trange
from util import dataloaders as dl
from laplace import kfla
import laplace.util as lutil
from util.evaluation import timing, predict
import sys, os
import argparse
import traceback

import lula.model
import lula.train

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='CIFAR10', choices=['MNIST', 'CIFAR10', 'SVHN', 'CIFAR100'])
parser.add_argument('--lenet', default=False, action='store_true')
parser.add_argument('--base', default='plain', choices=['plain', 'oe'])
parser.add_argument('--timing', default=False, action='store_true')
args = parser.parse_args()

path = './pretrained_models/kfla'
path += '/oe' if args.base == 'oe' else ''
if not os.path.exists(path):
    os.makedirs(path)


if args.dataset == 'MNIST':
    train_loader = dl.MNIST(train=True, augm_flag=False)
    val_loader, test_loader = dl.MNIST(train=False, augm_flag=False, val_size=2000)
elif args.dataset == 'CIFAR10':
    train_loader = dl.CIFAR10(train=True, augm_flag=False)
    val_loader, test_loader = dl.CIFAR10(train=False, augm_flag=False, val_size=2000)
elif args.dataset == 'SVHN':
    train_loader = dl.SVHN(train=True, augm_flag=False)
    val_loader, test_loader = dl.SVHN(train=False, augm_flag=False, val_size=2000)
elif args.dataset == 'CIFAR100':
    train_loader = dl.CIFAR100(train=True, augm_flag=False)
    val_loader, test_loader = dl.CIFAR100(train=False, augm_flag=False, val_size=2000)


targets = torch.cat([y for x, y in test_loader], dim=0).numpy()
targets_val = torch.cat([y for x, y in val_loader], dim=0).numpy()

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

num_classes = 100 if args.dataset == 'CIFAR100' else 10
num_channel = 1 if args.dataset == 'MNIST' else 3

modifier = '' if args.lenet and args.dataset == 'MNIST' else '_wrn_'


def load_model():
    if args.lenet and args.dataset == 'MNIST':
        model = LeNetMadry()
        model.load_state_dict(torch.load(f'./pretrained_models/{args.dataset}_{args.base}.pt'))
    else:
        model = wrn.WideResNet(16, 4, num_classes, num_channel)
        model.load_state_dict(torch.load(f'./pretrained_models/{args.dataset}_wrn_{args.base}.pt'))

    model.cuda()
    model.eval()

    return model


print()


####################################################################################################
## LULA TRAINING
####################################################################################################

# Prior variance comes from the weight decay used in the MAP training
var0 = torch.tensor(1/(5e-4*len(train_loader.dataset))).float().cuda()
print(var0)

# Grid search
# Smooth Noise already attain maximum entropy in MNIST, so it's not useful
# noise_grid = ['imagenet', 'smooth']
noise_grid = ['imagenet']
n_units_grid = [512] if args.timing else [32, 64, 128, 256, 512, 1024]

lr = 0.1
nll = nn.CrossEntropyLoss(reduction='mean')

best_model = None
best_loss = inf


for noise in noise_grid:
    print(noise)
    print()

    if noise == 'imagenet':
        ood_train_loader = dl.ImageNet32(dataset=args.dataset, train=True)
        ood_val_loader = dl.ImageNet32(dataset=args.dataset, train=False)
    elif noise == 'smooth':
        ood_train_loader = dl.Noise(args.dataset, train=True)
        ood_val_loader = dl.Noise(args.dataset, train=False)
    else:
        ood_train_loader = dl.UniformNoise(args.dataset, train=True, size=len(train_loader.dataset))
        ood_val_loader = dl.UniformNoise(args.dataset, train=False, size=2000)

    best_model_noise = None
    best_loss_noise = inf

    for n_unit in n_units_grid:
        print(n_unit)
        n_lula_units = [n_unit]

        model = load_model()
        model_lula, time_cons = timing(lambda: lula.model.LULAModel_LastLayer(model, n_lula_units).cuda())
        model_lula.to_gpu()
        model_lula.eval()
        model_lula.enable_grad_mask()

        try:
            ood_train_loader.dataset.offset = np.random.randint(len(ood_train_loader.dataset))
            model_lula, time_train = timing(
                lambda: lula.train.train_last_layer(
                    model_lula, nll, val_loader, ood_train_loader, 1/var0, lr=lr,
                    n_iter=10, progressbar=True, mc_samples=10
                )
            )

            if args.timing:
                print(f'Time Construction: {time_cons:.3f}, Time Training: {time_train:.3f}')
                sys.exit(0)

            # Temp
            torch.save(model_lula.state_dict(), f'{path}/{args.dataset}_lula_temp.pt')

            # Grid search criterion (this modifies model_lula, hence the need of the temp above)
            # MMC distance to the optimal for both in- and out-dist val set, under a Laplace
            model_lula.disable_grad_mask()
            model_lula.eval()
            model_lula.unmask()

            # Do a LA over the trained LULA-augmented network
            model_kfla = kfla.KFLA(model_lula)
            model_kfla.get_hessian(train_loader)
            model_kfla.estimate_variance(var0)
            py_in = lutil.predict(val_loader, model_kfla, n_samples=10)
            py_out = lutil.predict(ood_val_loader, model_kfla, n_samples=10, n_data=2000)

            h_in = dist.Categorical(py_in).entropy().mean().cpu().numpy()
            h_out = dist.Categorical(py_out).entropy().mean().cpu().numpy()
            loss = h_in - h_out

            print(f'Loss: {loss:.3f}, H_in: {h_in:.3f}, H_out: {h_out:.3f}')
            print(best_loss_noise)

            # Save the current best
            if loss < best_loss_noise:
                state_dict = torch.load(f'{path}/{args.dataset}_lula_temp.pt')
                torch.save([state_dict, n_lula_units], f'{path}/{args.dataset}{modifier}lula_{noise}.pt')
                best_loss_noise = loss
        except Exception as e:
            print(f'Exception occured: {e}')
            traceback.print_tb(e.__traceback__)
            loss = inf

        print()

    print()

    # Save the current best across noises and n_units
    if best_loss_noise < best_loss:
        state_dict, n_lula_units = torch.load(f'{path}/{args.dataset}{modifier}lula_{noise}.pt')
        torch.save([state_dict, n_lula_units, noise], f'{path}/{args.dataset}{modifier}lula_best.pt')
        best_loss = best_loss_noise

# Cleanup
os.remove(f'{path}/{args.dataset}_lula_temp.pt')


####################################################################################################
## Test the best model
####################################################################################################

model = load_model()

state_dict, n_lula_units, noise = torch.load(f'{path}/{args.dataset}{modifier}lula_best.pt')
model_lula = lula.model.LULAModel_LastLayer(model, n_lula_units).cuda()
model_lula.to_gpu()
model_lula.load_state_dict(state_dict)
model_lula.disable_grad_mask()
model_lula.eval()

# Test
py_in = predict(test_loader, model_lula).cpu().numpy()
acc_in = np.mean(np.argmax(py_in, 1) == targets)*100
mmc = np.max(py_in).mean()
print(f'[In, LULA-{noise}] Accuracy: {acc_in:.3f}; MMC: {mmc:.3f}')
