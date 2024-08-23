from collections import OrderedDict

import torch
from torch import nn


class EyeClassifier(nn.Module):
    def __init__(self, activation=nn.GELU()):
        super().__init__()
        self.conv = nn.Sequential(OrderedDict([
            # Conv-1
            ("conv1", nn.Conv2d(in_channels=1, out_channels=128, kernel_size=5, stride=1, padding=2)),
            ("activation1", activation),
            ("maxPool1", nn.MaxPool2d(3, 1)),

            # Conv-2
            ("batchNorm2", nn.BatchNorm2d(128, affine=True)),
            ("conv2", nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)),
            ("activation2", activation),
            ("maxPool2", nn.MaxPool2d(3, 1)),

            # Conv-3
            ("batchNorm3", nn.BatchNorm2d(128, affine=True)),
            ("conv3", nn.Conv2d(in_channels=128, out_channels=96, kernel_size=3, stride=1, padding=1)),
            ("activation3", activation),
            ("maxPool3", nn.MaxPool2d(3, 1)),

            # Conv-4
            ("batchNorm4", nn.BatchNorm2d(96, affine=True)),
            ("conv4", nn.Conv2d(in_channels=96, out_channels=96,
                                kernel_size=3, stride=1, padding=0)),
            ("activation4", activation),
            ("maxPool4", nn.MaxPool2d(3, 1)),
        ]))
        self.global_pool = nn.AdaptiveMaxPool2d(1, return_indices=False)
        self.head = nn.Sequential(OrderedDict([
            # FC-1
            ("dropout1", nn.Dropout(p=0.2)),
            ("fc1", nn.Linear(96, 80)),
            ("activation1", activation),

            # FC-2
            ("dropout2", nn.Dropout(p=0.2)),
            ("fc2", nn.Linear(80, 64)),
            ("activation2", activation),

            # FC-3
            ("dropout3", nn.Dropout(p=0.2)),
            ("fc3", nn.Linear(64, 64)),
            ("activation3", activation),

            # FC-4
            ("fc4", nn.Linear(64, 32)),
            ("activation4", activation),

            # Classifier
            ("classifier", nn.Linear(32, 1))
        ]))

    def forward(self, x):
        x = self.conv(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.head(x)
        x = x.squeeze()
        return x


if __name__ == '__main__':  # testing
    x = torch.rand((1, 3, 62, 62))
    for model in [ONet(), ResNet18()]:
        out = model(x)
        print(x.shape, out.shape)