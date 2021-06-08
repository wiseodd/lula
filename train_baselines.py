import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from sklearn import datasets
from math import *
from models import wrn
from tqdm.auto import tqdm, trange
from util import dataloaders as dl
from util import evaluation as eval_util
import argparse
from torch.cuda import amp


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='CIFAR10', choices=["MNIST", "CIFAR10", "SVHN", "CIFAR100"])
args = parser.parse_args()


np.random.seed(1)
torch.manual_seed(1)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


if args.dataset == 'MNIST':
    train_loader = dl.MNIST(train=True, augm_flag=True)
    val_loader, test_loader = dl.MNIST(train=False, augm_flag=False, val_size=2000)
elif args.dataset == 'CIFAR10':
    train_loader = dl.CIFAR10(train=True, augm_flag=True)
    val_loader, test_loader = dl.CIFAR10(train=False, augm_flag=False,  val_size=2000)
elif args.dataset == 'SVHN':
    train_loader = dl.SVHN(train=True, augm_flag=True)
    val_loader, test_loader = dl.SVHN(train=False, augm_flag=False,  val_size=2000)
elif args.dataset == 'CIFAR100':
    train_loader = dl.CIFAR100(train=True, augm_flag=True)
    val_loader, test_loader = dl.CIFAR100(train=False, augm_flag=False,  val_size=2000)

targets = torch.cat([y for x, y in test_loader], dim=0).numpy()
targets_val = torch.cat([y for x, y in val_loader], dim=0).numpy()


K = 5  # DE's num of components
num_classes = 100 if args.dataset == 'CIFAR100' else 10
num_channel = 1 if args.dataset == 'MNIST' else 3
wd = 5e-4

models_de = [wrn.WideResNet(16, 4, num_classes, num_channel).cuda() for _ in range(K)]
opts_de = [torch.optim.SGD(m.parameters(), lr=0.1, momentum=0.9, weight_decay=wd) for m in models_de]
schs_de = [torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=100*len(train_loader)) for opt in opts_de]

scalers_de = [amp.GradScaler() for _ in range(K)]

pbar = trange(100)

for epoch in pbar:
    train_loss_de = 0
    n = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.cuda(), target.long().cuda()

        # DE
        for k in range(K):
            opts_de[k].zero_grad()

            with amp.autocast():
                output_de = models_de[k](data).squeeze()
                loss_de = F.cross_entropy(output_de, target)

            scalers_de[k].scale(loss_de).backward()
            scalers_de[k].step(opts_de[k])
            scalers_de[k].update()
            schs_de[k].step()

        train_loss_de += loss_de.item()
        n += 1

    train_loss_de /= n

    pred_val_de = eval_util.predict(val_loader, models_de[0]).cpu().numpy()
    acc_val_de = np.mean(np.argmax(pred_val_de, 1) == targets_val)*100

    pbar.set_description(f'[Epoch: {epoch+1}; val_de: {acc_val_de:.1f}]')

torch.save([m.state_dict() for m in models_de], f'./pretrained_models/{args.dataset}_wrn_de.pt')


# Load
state_dicts = torch.load(f'./pretrained_models/{args.dataset}_wrn_de.pt')
for k in range(K):
    models_de[k].load_state_dict(state_dicts[k])
    models_de[k].eval()

print()

# Test
py_de = eval_util.predict_ensemble(test_loader, models_de).cpu().numpy()
acc_de = np.mean(np.argmax(py_de, 1) == targets)*100
mmc_de = np.max(py_de, 1).mean()
print(f'[In, DE] Accuracy: {acc_de:.3f}; MMC: {mmc_de:.3f}')
