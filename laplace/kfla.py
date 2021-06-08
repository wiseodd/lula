import torch
from torch import nn, optim, autograd
from torch.nn import functional as F
from torch.distributions.multivariate_normal import MultivariateNormal
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from util.evaluation import get_auroc
from util.kfac import KFAC
from math import *
from tqdm import tqdm, trange
import numpy as np
import laplace.util as lutil


class KFLA(nn.Module):
    """
    Taken, with modification, from:
    https://github.com/wjmaddox/swa_gaussian/blob/master/swag/posteriors/diag_laplace.py

    In particular, this is KFLA is only for linear layers
    """

    def __init__(self, base_model):
        super().__init__()

        self.net = base_model
        self.params = []
        self.net.apply(lambda module: kfla_parameters(module, self.params))
        self.hessians = None

    def forward(self, x):
        return self.net.forward(x)

    def forward_sample(self, x):
        self.sample()
        return self.net.forward(x)

    def sample(self, scale=1, require_grad=False):
        for module, name in self.params:
            mod_class = module.__class__.__name__

            if mod_class not in ['Linear', 'MaskedLinear']:
                continue

            if name == 'bias':
                w = module.__getattr__(f'{name}_mean')
            else:
                M = module.__getattr__(f'{name}_mean')
                U_half = module.__getattr__(f'{name}_U_half')
                V_half = module.__getattr__(f'{name}_V_half')

                if len(M.shape) == 1:
                    M_ = M.unsqueeze(1)
                elif len(M.shape) > 2:
                    M_ = M.reshape(M.shape[0], np.prod(M.shape[1:]))
                else:
                    M_ = M

                E = torch.randn(*M_.shape, device='cuda')
                w = M_ + scale * U_half @ E @ V_half
                w = w.reshape(*M.shape)

            if require_grad:
                w.requires_grad_()

            module.__setattr__(name, w)

    def estimate_variance(self, var0, invert=True):
        tau = 1/var0

        U, V = self.hessians

        for module, name in self.params:
            mod_class = module.__class__.__name__
            if mod_class not in ['Linear', 'MaskedLinear']:
                continue

            if name == 'bias':
                continue

            U_ = U[(module, name)].clone()
            V_ = V[(module, name)].clone()

            if invert:
                m, n = int(U_.shape[0]), int(V_.shape[0])

                U_ += torch.sqrt(tau)*torch.eye(m, device='cuda')
                V_ += torch.sqrt(tau)*torch.eye(n, device='cuda')

                # For numerical stability
                u = torch.cholesky(U_.cpu() + 1e-6*torch.eye(m))
                v = torch.cholesky(V_.cpu() + 1e-6*torch.eye(n))

                U_ = torch.cholesky(torch.cholesky_inverse(u), upper=False).cuda()
                V_ = torch.cholesky(torch.cholesky_inverse(v), upper=True).cuda()

            module.__getattr__(f'{name}_U_half').copy_(U_)
            module.__getattr__(f'{name}_V_half').copy_(V_)

    def get_hessian(self, train_loader, binary=False):
        criterion = nn.BCEWithLogitsLoss() if binary else nn.CrossEntropyLoss()
        opt = KFAC(self.net)
        U = {}
        V = {}

        # Populate parameters with the means
        self.sample(scale=0, require_grad=True)

        for x, y in tqdm(train_loader):
            x = x.cuda(non_blocking=True)

            self.net.zero_grad()
            out = self(x).squeeze()

            if binary:
                distribution = torch.distributions.Binomial(logits=out)
            else:
                distribution = torch.distributions.Categorical(logits=out)

            y = distribution.sample()
            loss = criterion(out, y)
            loss.backward()
            opt.step()

        with torch.no_grad():
            for group in opt.param_groups:
                if len(group['params']) == 2:
                    weight, bias = group['params']
                else:
                    weight = group['params'][0]
                    bias = None

                module = group['mod']
                state = opt.state[module]

                U_ = state['ggt']
                V_ = state['xxt']

                n_data = len(train_loader.dataset)

                U[(module, 'weight')] = sqrt(n_data)*U_
                V[(module, 'weight')] = sqrt(n_data)*V_

            self.hessians = (U, V)

    def gridsearch_var0(self, val_loader, ood_loader, interval, n_classes=10, lam=1):
        vals, var0s = [], []
        pbar = tqdm(interval)

        for var0 in pbar:
            try:
                self.estimate_variance(var0)

                preds_in, y_in = lutil.predict(val_loader, self, n_samples=5, return_targets=True)
                loss_in = F.nll_loss(torch.log(preds_in + 1e-8), y_in)

                if ood_loader is not None:
                    preds_out = lutil.predict(ood_loader, self, n_samples=5)
                    loss_out = -torch.log(preds_out + 1e-8).mean()
                else:
                    loss_out = 0

                loss = loss_in + lam * loss_out
            except RuntimeError:
                loss = inf

            vals.append(loss)
            var0s.append(var0)

            pbar.set_description(f'var0: {var0:.5f}, L-in: {loss_in:.3f}, L-out: {loss_out:.3f}, L: {loss:.3f}')

        best_var0 = var0s[np.argmin(vals)]

        return best_var0



def kfla_parameters(module, params):
    mod_class = module.__class__.__name__
    if mod_class not in ['Linear', 'MaskedLinear']:
        return

    for name in list(module._parameters.keys()):
        if module._parameters[name] is None:
            continue

        data = module._parameters[name].data
        m, n = int(data.shape[0]), int(np.prod(data.shape[1:]))
        module._parameters.pop(name)
        module.register_buffer(f'{name}_mean', data)
        module.register_buffer(f'{name}_U_half', torch.zeros([m, m], device='cuda'))
        module.register_buffer(f'{name}_V_half', torch.zeros([n, n], device='cuda'))
        module.register_buffer(name, data.new(data.size()).zero_())

        params.append((module, name))
