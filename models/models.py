import torch
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class LeNetMadry(nn.Module):

    def __init__(self, feature_extractor=False, use_dropout=False):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 5, 1, padding=2)
        self.conv2 = nn.Conv2d(32, 64, 5, 1, padding=2)
        self.flatten = nn.Flatten()

        self.clf = nn.Sequential(
            nn.Linear(7*7*64, 1024),
            nn.ReLU(),
            nn.Linear(1024, 10)
        )

        self.num_features = 1024
        self.feature_extractor = feature_extractor
        self.use_dropout = use_dropout

    def forward(self, x):
        return self.features(x) if self.feature_extractor else self.clf(self.features_before_clf(x))

    def features(self, x):
        x = self.features_before_clf(x)
        for m in list(self.clf.children())[:-1]:
            x = m(x)
        return x

    def features_before_clf(self, x):
        x = F.relu(self.conv1(x))

        if self.use_dropout:
            x = F.dropout2d(x, 0.2)

        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))

        if self.use_dropout:
            x = F.dropout2d(x)

        x = F.max_pool2d(x, 2, 2)
        x = self.flatten(x)

        if self.use_dropout:
            x = F.dropout(x, 0.2)

        return x
