import torch
from models import wrn, models
import laplace.util as lutil
from util.evaluation import *
import util.dataloaders as dl
from util.misc import *
from tqdm import tqdm, trange
import numpy as np
import argparse
import pickle
import os
from torch.cuda import amp


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='MNIST', choices=["MNIST", "CIFAR10", "SVHN", "CIFAR100"])
parser.add_argument('--lenet', default=False, action='store_true')
parser.add_argument('--ood_loss', default=False, action='store_true')
parser.add_argument('--randseed', type=int, default=123)
args = parser.parse_args()

np.random.seed(args.randseed)
torch.manual_seed(args.randseed)
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

ood_train_loader = dl.ImageNet32(dataset=args.dataset, train=True)

print(len(train_loader.dataset), len(val_loader.dataset), len(test_loader.dataset))

num_classes = 100 if args.dataset == 'CIFAR100' else 10
num_channel = 1 if args.dataset == 'MNIST' else 3

if args.lenet and args.dataset == 'MNIST':
    model = models.LeNetMadry().cuda()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)
    model_indicator = ''
else:
    model = wrn.WideResNet(16, 4, num_classes, num_channel).cuda()
    opt = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4, nesterov=True)
    model_indicator = '_wrn_'

criterion = torch.nn.CrossEntropyLoss()
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=100*len(train_loader))
scaler = amp.GradScaler()
pbar = trange(100)

for epoch in pbar:
    model.train()
    train_loss = 0
    n = 0

    ood_train_loader.dataset.offset = np.random.randint(len(ood_train_loader.dataset))

    for batch_idx, ((data_in, target), (data_out, _)) in enumerate(zip(train_loader, ood_train_loader)):
        opt.zero_grad()

        data_in, target = data_in.cuda(), target.long().cuda()
        m = len(data_in)
        data_out = data_out[:m].cuda()  # Make sure m_out = m_in
        data = torch.cat([data_in, data_out], dim=0).float()

        with amp.autocast():
            output = model(data).squeeze()
            loss = criterion(output[:m], target)

            if args.ood_loss:
                # Hendrycks et al.
                loss += -0.5*1/num_classes*torch.log_softmax(output[m:], 1).mean()

        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        scheduler.step()

        train_loss += loss.item()
        n += 1

    train_loss /= n
    model.eval()
    pred_val = predict(val_loader, model).cpu().numpy()
    acc_val = np.mean(np.argmax(pred_val, 1) == targets_val)*100

    pbar.set_description(f'[Epoch: {epoch+1}; train_loss: {train_loss:.4f}; val_acc: {acc_val:.1f}]')

modifier = 'plain' if not args.ood_loss else 'oe'
torch.save(model.state_dict(), f'./pretrained_models/{args.dataset}{model_indicator}{modifier}.pt')

model.load_state_dict(torch.load(f'./pretrained_models/{args.dataset}{model_indicator}{modifier}.pt'))
model.eval()

print()

# In-distribution
py_in = predict(test_loader, model).cpu().numpy()
acc_in = np.mean(np.argmax(py_in, 1) == targets)*100
mmc = py_in.max(1).mean()*100
print(f'[In, MAP] Accuracy: {acc_in:.3f}; MMC: {mmc:.3f}')
