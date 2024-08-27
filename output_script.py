import cv2
import torch
import argparse
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2 as ToTensor

from src.models import CustomNet2


class OpenEyesClassificator:
    def __init__(self,
                 weights: str,
                 device: str = "cuda",
                 target_class_idx: int = 1,  # open eye
                 model: torch.nn.Module = CustomNet2(),
                 ):
        self.device = device
        self.target_class_idx = target_class_idx
        self.model = self._load_model(model, weights)
        self.model.eval()
        self.augmentation = A.Compose([
            A.Resize(height=24, width=24, always_apply=True),  # constant window size
            A.Normalize(
                mean=0.5479156433505469,
                std=0.13769661157967297,
                max_pixel_value=255.0,
                always_apply=True
            ),
            ToTensor(always_apply=True),
        ])

    def _load_model(self, model: torch.nn.Module, weights_path: str) -> torch.nn.Module:
        state_dict = torch.load(str(weights_path), weights_only=False)['state_dict']
        remove_prefix = 'model.'
        state_dict = {k[len(remove_prefix):] if k.startswith(remove_prefix) else k: v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
        model.to(self.device)
        return model

    def _get_tensor(self, inpIm: str) -> torch.Tensor:  # inference
        image = cv2.imread(str(inpIm), cv2.IMREAD_GRAYSCALE)
        tensor = self.augmentation(image=image)['image']
        return tensor.unsqueeze(0).to(self.device)

    @torch.no_grad()
    def predict(self, inpIm: str) -> float:  # inference
        x = self._get_tensor(inpIm)
        pred = self.model(x)
        is_open_score = pred.squeeze()[self.target_class_idx]
        return is_open_score.item()


def main(args):
    classifier = OpenEyesClassificator(device=args.device, weights=args.weights_path)
    is_open_score = classifier.predict(args.img_path)
    print(f"is_open_score = {is_open_score}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--img_path', type=str, help='',
                        required=True,
                        )
    parser.add_argument('-w', '--weights_path', type=str, help='',
                        default='data/customnet2_relu_final_weights.ckpt'
                        )
    parser.add_argument('-d', '--device', choices=['cpu', 'cuda'], help='',
                        default='cuda',
                        )
    args = parser.parse_args()

    assert Path(args.img_path).exists()

    if args.device == 'cuda' and not torch.cuda.is_available():
        args.device = 'cpu'

    main(args)
