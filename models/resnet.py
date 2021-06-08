'''ResNet in PyTorch.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):

    def __init__(self, block, num_blocks, num_classes=10, num_of_channels=3, feature_extractor=False, use_dropout=False):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(num_of_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.clf = nn.Sequential(
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.ReLU(),
            nn.AvgPool2d(4),
            nn.Flatten(),
            nn.Linear(512, num_classes)
        )

        self.num_features = 512
        self.feature_extractor = feature_extractor
        self.use_dropout = use_dropout

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.features(x) if self.feature_extractor else self.clf(self.features_before_clf(x))

    def features(self, x):
        x = self.features_before_clf(x)
        for m in list(self.clf.children())[:-1]:
            x = m(x)
        return x

    def features_before_clf(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)

        if self.use_dropout:
            out = F.dropout2d(out, 0.2)

        out = self.layer2(out)

        if self.use_dropout:
            out = F.dropout2d(out, 0.2)

        out = self.layer3(out)

        if self.use_dropout:
            out = F.dropout2d(out, 0.2)

        out = self.layer4(out)

        if self.use_dropout:
            out = F.dropout2d(out, 0.2)

        return out


def ResNet18(num_of_channels=3, num_classes=10, feature_extractor=False, use_dropout=False):
    if num_classes <= 2:
        num_classes = 1

    return ResNet(BasicBlock, [2,2,2,2],
                  num_of_channels=num_of_channels,
                  num_classes=num_classes,
                  feature_extractor=feature_extractor, use_dropout=use_dropout)

def ResNet18_100(feature_extractor=False):
    return ResNet(BasicBlock, [2,2,2,2], num_classes=100, feature_extractor=feature_extractor)
