from typing import Union, List

import cv2
import torch
from torch import nn
import albumentations as A

from src import plt_show_img
from src.utils_pl import final_transforms


class CustomNet(nn.Module):
    def __init__(self, in_channels=1, activation=nn.ReLU, mode: str = "regression"):
        assert mode in ["classification", "regression"]
        super(CustomNet, self).__init__()

        self.block1 = self._make_block(in_channels, 16, activation)
        self.block2 = self._make_block(16 + 1, 64, activation)  # +1 cause residual
        self.block3 = self._make_block(64 + 1, 32, activation)  # +1 cause residual
        self.block4 = self._make_block(32 + 1, 16, activation)  # +1 cause residual

        self.mode = mode
        fc = [
            nn.Flatten(1),
            nn.Dropout(p=0.3),
        ]

        if self.mode == "classification":
            fc.extend([
                nn.Linear(16 * 24 * 24, 2),
                nn.Softmax(dim=1),
            ])
        elif self.mode == "regression":
            fc.extend([
                nn.Linear(16 * 24 * 24, 1),
                nn.Sigmoid(),
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


class OpenEyesClassificator:
    def __init__(self,
                 pretrained: str,
                 device: str = "cuda",
                 augmentation: Union[List, None] = None,
                 mode: str = "regression"):
        assert mode in ["classification", "regression"]
        self.mode = mode
        self.device = device
        self.model = self.load_model(pretrained)
        self.model.eval()

        if augmentation is None:
            augmentation = []
        augmentation.extend(final_transforms)
        self.augmentation = A.Compose(augmentation)

    def load_model(self, path) -> nn.Module:
        state_dict = torch.load(str(path), weights_only=False)['state_dict']
        remove_prefix = 'model.'
        state_dict = {k[len(remove_prefix):] if k.startswith(remove_prefix) else k: v for k, v in state_dict.items()}
        model = CustomNet(mode=self.mode)
        model.load_state_dict(state_dict)
        model.to(self.device)
        return model

    def get_tensor(self, inpIm: str) -> torch.Tensor:  # inference
        image = cv2.imread(str(inpIm), cv2.IMREAD_GRAYSCALE)
        tensor = self.augmentation(image=image)['image']
        return tensor.unsqueeze(0).to(self.device)

    @torch.no_grad()
    def predict(self, inpIm: str) -> float:  # inference
        x = self.get_tensor(inpIm)
        is_open_score = self.model(x)
        if self.mode == "classification":
            is_open_score = is_open_score.squeeze()[1]
        return is_open_score.item()


if __name__ == '__main__':  # testing
    x = torch.rand((1, 1, 24, 24))
    # for model in [OpenEyesClassificator()]:
    #     out = model(x)
    #     print(75, x.shape, out.shape)

    model = OpenEyesClassificator(
        pretrained="/home/iamsvp/PycharmProjects/open-closed_eyes_classifier/logs/eyes_classifier/version_1/checkpoints/epoch=114-val_loss=0.9930.ckpt")
    inpIm = "/home/iamsvp/data/eye/EyesDataset/0/000001.jpg"
    result = model.predict(inpIm)
    plt_show_img(cv2.imread(inpIm, cv2.IMREAD_GRAYSCALE), add_coef=True, title=str(result))
