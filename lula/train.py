import numpy as np
import torch
from torch import distributions as dist
from torch import nn, optim, autograd
from torch.nn import functional as F
from torch.nn.utils.convert_parameters import parameters_to_vector
from tqdm.auto import tqdm, trange
from toolz import itertoolz

from lula.model import *
from lula.util import *


def train_last_layer(lula_model, nll, in_loader, out_loader, prior_prec, l2_penalty=0, lr=1e-1, n_iter=100,
                     fisher_samples=1, alpha=1, beta=1, max_grad_norm=1000, progressbar=True, mc_samples=10):
    # Train only the last-layer
    for m in lula_model.modules():
        if type(m) == MaskedLinear or type(m) == MaskedConv2d:
            for p in m.parameters():
                p.requires_grad = True
        else:
            for p in m.parameters():
                p.requires_grad = False

    opt = optim.Adam(filter(lambda p: p.requires_grad, lula_model.parameters()), lr=lr, weight_decay=0)
    pbar = trange(n_iter) if progressbar else range(n_iter)

    for it in pbar:
        epoch_loss = 0
        n = 0

        for (x_in, _), (x_out, _) in zip(in_loader, out_loader):
            x_in, x_out = x_in.cuda(), x_out.cuda()

            lula_model.disable_grad_mask()

            ll_module = itertoolz.last(lula_model.modules())
            mu_W = ll_module.weight
            mu_b = ll_module.bias

            fisher_diag_W = get_fisher_diag_last_layer_(lula_model, x_in, nll, fisher_samples, bias=False)
            sigma_W = 1/(fisher_diag_W + prior_prec)

            fisher_diag_b = get_fisher_diag_last_layer_(lula_model, x_in, nll, fisher_samples, bias=True)
            sigma_b = 1/(fisher_diag_b + prior_prec)

            lula_model.enable_grad_mask()

            phi_in = lula_model.features(x_in)
            phi_out = lula_model.features(x_out)

            py_in, py_out = 0, 0

            for s in range(mc_samples):
                # Sample from the posterior
                W_s = mu_W + torch.sqrt(sigma_W) * torch.randn(*mu_W.shape, device='cuda')
                b_s = mu_b + torch.sqrt(sigma_b) * torch.randn(*mu_b.shape, device='cuda')

                py_in += 1/mc_samples * torch.softmax(phi_in @ W_s.T + b_s, -1)
                py_out += 1/mc_samples * torch.softmax(phi_out @ W_s.T + b_s, -1)

            loss_in = dist.Categorical(py_in).entropy().mean()
            loss_out = dist.Categorical(py_out).entropy().mean()

            # Min. in-dist uncertainty, max. out-dist uncertainty
            loss = alpha*loss_in - beta*loss_out

            loss.backward()
            torch.nn.utils.clip_grad_norm_(lula_model.parameters(), max_grad_norm)
            opt.step()
            opt.zero_grad()

            epoch_loss += loss.detach().item()
            n += 1

        if progressbar:
            epoch_loss /= n
            weight_norm = parameters_to_vector(lula_model.parameters()).norm(2).detach().item()
            pbar.set_description(f'Loss: {epoch_loss:.3f}; Weight norm: {weight_norm:.3f}')

    return lula_model


def get_fisher_diag_last_layer_(model, x, nll, n_samples, lik_prec=1, bias=False):
    fisher_diag = 0

    for s in range(n_samples):
        output = model(x).squeeze()

        # Obtain the diagonal-Fisher approximation to the Hessian
        if type(nll) == nn.BCEWithLogitsLoss:
            y = torch.distributions.Bernoulli(logits=output).sample()
        elif type(nll) == nn.CrossEntropyLoss:
            y = torch.distributions.Categorical(logits=output).sample()
        else:
            y = torch.distributions.Normal(output, lik_prec).sample()

        loss = nll(output, y)

        ll_params = itertoolz.last(model.modules())
        p = ll_params.bias if bias else ll_params.weight

        grad = autograd.grad([loss], p, retain_graph=True, create_graph=True)[0]
        fisher_diag += x.shape[0] * grad**2

    fisher_diag /= n_samples

    return fisher_diag
