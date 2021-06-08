import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0, use_dropout=False):
        super(BasicBlock, self).__init__()

        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False) or None

        self.use_dropout = use_dropout

    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))

        if self.equalInOut:
            out = self.relu2(self.bn2(self.conv1(out)))
        else:
            out = self.relu2(self.bn2(self.conv1(x)))

        if self.droprate > 0:
            # Dropout always on training mode if use_dropout = True
            out = F.dropout(out, p=self.droprate, training=self.training or self.use_dropout)

        out = self.conv2(out)

        if not self.equalInOut:
            return torch.add(self.convShortcut(x), out)
        else:
            return torch.add(x, out)


class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0, use_dropout=False):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate, use_dropout)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate, use_dropout):
        layers = []

        for i in range(nb_layers):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate, use_dropout))

        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class WideResNet(nn.Module):

    def __init__(self, depth, widen_factor, num_classes, num_channel=3, dropRate=0.3, feature_extractor=False, use_dropout=False):
        super(WideResNet, self).__init__()

        nChannels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        assert ((depth - 4) % 6 == 0)
        n = (depth - 4) // 6
        block = BasicBlock

        # 1st conv before any network block
        self.conv1 = nn.Conv2d(num_channel, nChannels[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate, use_dropout)
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate, use_dropout)
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate, use_dropout)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.clf = nn.Sequential(
            nn.Linear(nChannels[3], nChannels[3]),
            nn.ReLU(),
            nn.Linear(nChannels[3], num_classes)
        )

        self.num_input_channel = num_channel
        self.nChannels = nChannels[3]
        self.num_features = self.nChannels
        self.feature_extractor = feature_extractor
        self.use_dropout = use_dropout

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        return self.features(x) if self.feature_extractor else self.clf(self.features_before_clf(x))

    def features(self, x):
        x = self.features_before_clf(x)
        for m in list(self.clf.children())[:-1]:
            x = m(x)
        return x

    def features_before_clf(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8 if self.num_input_channel == 3 else 7)
        out = out.view(-1, self.nChannels)
        return out
