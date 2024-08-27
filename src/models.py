import cv2
import torch
from torch import nn
import albumentations as A
from typing import Union, List

from src.utils_pl import final_transforms


class CustomNet0(nn.Module):
    def __init__(self, in_channels=1, activation=nn.ReLU):
        super(CustomNet0, self).__init__()
        self.block1 = self._make_block(in_channels, 16, activation)
        self.block2 = self._make_block(16 + 1, 64, activation)  # +1 cause residual
        self.block3 = self._make_block(64 + 1, 32, activation)  # +1 cause residual
        self.block4 = self._make_block(32 + 1, 16, activation)  # +1 cause residual

        fc = [
            nn.Flatten(1),
            nn.Dropout(p=0.3),
        ]
        fc.extend([
            nn.Linear(16 * 24 * 24, 2),
            nn.Softmax(dim=1),
        ])
        self.fc = nn.Sequential(*fc)

    def _make_block(self, in_channels, out_channels, activation):
        block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            activation(),
        )
        return block

    def forward(self, x):
        block1 = self.block1(x)
        block2 = self.block2(torch.cat((x, block1), 1))  # residual
        block3 = self.block3(torch.cat((x, block2), 1))
        block4 = self.block4(torch.cat((x, block3), 1))
        fc = self.fc(block4)
        return fc


class CustomNet(nn.Module):
    def __init__(self, in_channels=1, activation=nn.ReLU, n_classes=2):
        super(CustomNet, self).__init__()
        self.block1 = self._make_block(in_channels, 16, activation)
        self.block2 = self._make_block(16 + 1, 64, activation)  # +1 cause residual
        self.block3 = self._make_block(64 + 1, 32, activation)  # +1 cause residual
        self.block4 = self._make_block(32 + 1, 16, activation)  # +1 cause residual
        self.maxpool = nn.Sequential(
            nn.MaxPool2d(kernel_size=(3, 3), stride=(3, 3), dilation=(3, 3)),
            nn.Conv2d(in_channels=16, out_channels=8, kernel_size=(3, 3), stride=1, padding=0),
            nn.BatchNorm2d(8),
            activation(),
        )
        self.fc1 = nn.Sequential(
            nn.Flatten(1),
            nn.Dropout(p=0.3),
            nn.Linear(8 * 4 * 4, 16),
            activation(),
        )
        self.fc2 = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(16, n_classes),  # n_classes
            nn.Sigmoid(),
        )

    def _make_block(self, in_channels, out_channels, activation):
        block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            activation(),
        )
        return block

    def forward(self, x):
        block1 = self.block1(x)
        block2 = self.block2(torch.cat((x, block1), 1))  # residual
        block3 = self.block3(torch.cat((x, block2), 1))  # residual
        block4 = self.block4(torch.cat((x, block3), 1))  # residual
        maxpool = self.maxpool(block4)
        fc1 = self.fc1(maxpool)
        fc2 = self.fc2(fc1)
        return fc2


class CustomNet2(nn.Module):
    def __init__(self, in_channels=1, activation=nn.ReLU, n_classes=2):
        super(CustomNet2, self).__init__()
        self.block1 = self._make_block(in_channels, 16, activation)
        self.block2 = self._make_block(16 + 1, 32, activation)  # +1 cause residual
        self.block3 = self._make_block(32 + 1, 64, activation)  # +1 cause residual
        self.block4 = self._make_block(64 + 1, 64, activation)  # +1 cause residual
        self.maxpool = nn.Sequential(
            nn.MaxPool2d(kernel_size=(3, 3), stride=(3, 3), dilation=(3, 3)),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(3, 3), stride=1, padding=0),
            nn.BatchNorm2d(32),
            activation(),
        )
        self.fc1 = nn.Sequential(
            nn.Flatten(1),
            nn.Dropout(p=0.3),
            nn.Linear(32 * 4 * 4, 16),
            activation(),
        )
        self.fc2 = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(16, n_classes),  # n_classes
            nn.Sigmoid(),
        )

    def _make_block(self, in_channels, out_channels, activation):
        block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            activation(),
        )
        return block

    def forward(self, x):
        block1 = self.block1(x)
        block2 = self.block2(torch.cat((x, block1), 1))  # residual
        block3 = self.block3(torch.cat((x, block2), 1))  # residual
        block4 = self.block4(torch.cat((x, block3), 1))  # residual
        maxpool = self.maxpool(block4)
        fc1 = self.fc1(maxpool)
        fc2 = self.fc2(fc1)
        return fc2

if __name__ == '__main__':  # testing
    x = torch.rand((1, 1, 24, 24))
    for model in [CustomNet()]:
        out = model(x)
        print(75, x.shape, out.shape)
