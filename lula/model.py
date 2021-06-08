import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from tqdm.auto import tqdm, trange

from lula.util import *


class LULAModel(nn.Module):

    def __init__(self, base_model, n_lula_units):
        """
        n_lula_units: Must be list of natural numbers with length equal to the number of hidden layers.
        """

        super(LULAModel, self).__init__()

        # Augment all fc layers
        base_modules = [m for m in base_model.modules()
                        if type(m) != nn.Sequential
                           and type(m) != type(base_model)]

        # Zero unit for both input and output layers
        n_lula_units = [0] + n_lula_units + [0]
        assert len(n_lula_units) == 1 + len([m for m in base_modules if type(m) == nn.Linear or type(m) == nn.Conv2d])

        modules = []
        i = 0
        prev_module = None

        for m in base_modules:
            if type(m) == nn.Linear:
                m_in, m_out = n_lula_units[i], n_lula_units[i+1]
                i += 1
                modules.append(MaskedLinear(m, m_in, m_out))
            elif type(m) == nn.Conv2d:
                m_in, m_out = n_lula_units[i], n_lula_units[i+1]
                i += 1
                modules.append(MaskedConv2d(m, m_in, m_out))
            else:
                modules.append(m)

        self.feature_map = nn.Sequential(*modules[:-1])
        self.clf = modules[-1]

    def forward(self, x):
        x = self.feature_map(x)
        return self.clf(x)

    def enable_grad_mask(self):
        for m in self.modules():
            if type(m) == MaskedLinear or type(m) == MaskedConv2d:
                m.switch_grad_mask(True)

    def disable_grad_mask(self):
        for m in self.modules():
            if type(m) == MaskedLinear or type(m) == MaskedConv2d:
                m.switch_grad_mask(False)

    def to_gpu(self):
        for m in self.modules():
            if type(m) == MaskedLinear or type(m) == MaskedConv2d:
                m.to_gpu()


class LULAModel_LastLayer(nn.Module):

    def __init__(self, base_model, n_lula_units):
        """
        base_model: Must have a method called `features(x)` and module called `clf`
        n_lula_units: Must be list of natural numbers with length equal to the number of   hidden layers.
        """

        super(LULAModel_LastLayer, self).__init__()

        self.base_model = base_model
        self.clf_lula = LULAModel(base_model.clf, n_lula_units)

        # "Delete" the original last-layer
        self.base_model.clf = nn.Identity()

    def forward(self, x):
        x = self.base_model.features(x)
        return self.clf_lula(x)

    def features(self, x):
        x = self.base_model.features(x)
        for m in list(self.clf_lula.children())[:-1]:
            x = m(x)
        return x

    def enable_grad_mask(self):
        self.clf_lula.enable_grad_mask()

    def disable_grad_mask(self):
        self.clf_lula.disable_grad_mask()

    def to_gpu(self):
        self.clf_lula.to_gpu()

    def unmask(self):
        mods = []
        for m in self.clf_lula.children():
            if m.__class__.__name__ in ['MaskedLinear', 'MaskedConv2d']:
                mods.append(m.to_unmasked())
            else:
                mods.append(m)
        self.clf_lula = nn.Sequential(*mods)
