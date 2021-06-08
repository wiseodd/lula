import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class MaskedLinear(nn.Module):

    def __init__(self, base_layer, m_in, m_out):
        """
        The standard nn.Linear layer, but with gradient masking to enforce the LULA construction.
        """
        super(MaskedLinear, self).__init__()

        # Extend the weight matrix
        W_base = base_layer.weight.data.clone()  # (n_out, n_in)
        n_out, n_in = W_base.shape

        W = torch.randn(n_out+m_out, n_in+m_in)
        W[0:n_out, 0:n_in] = W_base.clone()
        W[0:n_out, n_in:] = 0  # Upper-right quadrant

        self.weight = nn.Parameter(W)

        # Extend the bias vector
        if base_layer.bias is not None:
            b_base = base_layer.bias.data.clone()

            b = torch.randn(n_out+m_out)
            b[:n_out] = b_base.clone()

            self.bias = nn.Parameter(b)
        else:
            self.bias = None

        # Apply gradient mask to the weight and bias
        self.mask_w = torch.zeros(n_out+m_out, n_in+m_in)
        self.mask_w[n_out:, :] = 1  # Lower half

        self.mask_b = torch.zeros(n_out+m_out)
        self.mask_b[n_out:] = 1

        self.switch_grad_mask(True)

        # For safekeeping
        self.W_base, self.b_base = W_base, b_base
        self.n_out, self.n_in = n_out, n_in
        self.m_out, self.m_in = m_out, m_in

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)

    def switch_grad_mask(self, on=True):
        if on:
            self.grad_handle_w = self.weight.register_hook(lambda grad: grad.mul_(self.mask_w))
            self.grad_handle_b = self.bias.register_hook(lambda grad: grad.mul_(self.mask_b))
        else:
            self.grad_handle_w.remove()
            self.grad_handle_b.remove()

    def to_gpu(self):
        self.mask_w = self.mask_w.cuda()
        self.mask_b = self.mask_b.cuda()

    def to_unmasked(self):
        lin = nn.Linear(self.n_in+self.m_in, self.n_out+self.m_out)
        lin.weight = self.weight
        lin.bias = self.bias
        return lin

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.n_in+self.m_in, self.n_out+self.m_out, self.bias is not None
        )



class MaskedConv2d(nn.Module):

    def __init__(self, base_layer, m_in, m_out):
        """
        The standard nn.Conv2d layer, but with gradient masking to enforce the LULA construction.
        """
        super(MaskedConv2d, self).__init__()

        self.kernel_size = base_layer.kernel_size
        self.stride = base_layer.stride
        self.padding = base_layer.padding
        self.dilation = base_layer.dilation
        self.groups = base_layer.groups

        # Extend the weight matrix
        W_base = base_layer.weight.data.clone()  # (n_out, n_in, k, k)
        n_out, n_in, k, _ = W_base.shape  # Num of channels

        W = torch.randn(n_out+m_out, n_in+m_in, k, k)
        W[0:n_out, 0:n_in, :, :] = W_base.clone()
        W[0:n_out, n_in:, :, :] = 0  # Upper-right quadrant

        self.weight = nn.Parameter(W)

        # Extend the bias vector
        if base_layer.bias is not None:
            b_base = base_layer.bias.data.clone()

            b = torch.randn(n_out+m_out)
            b[:n_out] = b_base.clone()

            self.bias = nn.Parameter(b)
        else:
            self.bias = None

        # Apply gradient mask to the weight and bias
        self.mask_w = torch.zeros(n_out+m_out, n_in+m_in, k, k)
        self.mask_w[n_out:, :, :, :] = 1  # Lower half

        self.mask_b = torch.zeros(n_out+m_out)
        self.mask_b[n_out:] = 1

        self.switch_grad_mask(True)

        # For safekeeping
        self.W_base, self.b_base = W_base, b_base
        self.n_out, self.n_in = n_out, n_in
        self.m_out, self.m_in = m_out, m_in

    def forward(self, x):
        return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

    def switch_grad_mask(self, on=True):
        if on:
            self.grad_handle_w = self.weight.register_hook(lambda grad: grad.mul_(self.mask_w))
            self.grad_handle_b = self.bias.register_hook(lambda grad: grad.mul_(self.mask_b))
        else:
            self.grad_handle_w.remove()
            self.grad_handle_b.remove()

    def to_gpu(self):
        self.mask_w = self.mask_w.cuda()
        self.mask_b = self.mask_b.cuda()

    def to_unmasked(self):
        conv = nn.Conv2d(self.n_in+self.m_in, self.n_out+self.m_out, self.kernel_size, self.stride, self.padding, self.dilation, self.groups)
        conv.weight = self.weight
        conv.bias = self.bias
        return conv

    def extra_repr(self):
        return 'in_channels={}, out_channels={}, bias={}'.format(
            self.n_in+self.m_in, self.n_out+self.m_out, self.bias is not None
        )
