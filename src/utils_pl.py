import cv2
import torch
from pathlib import Path
import albumentations as A
from typing import List, Tuple
import pytorch_lightning as pl
from torch.utils.data import Dataset
from warmup_scheduler import GradualWarmupScheduler
from albumentations.pytorch import ToTensorV2 as ToTensor

'''
DATASET PARAMS: (scalefactor=0.00392156862745098, RGB)
dataset_mean=[0.5479156433505469, 0.5479156433505469, 0.5479156433505469] (mean=0.5479156433505469)
dataset_std=[0.13769661157967297, 0.13769661157967297, 0.13769661157967297] (mean=0.13769661157967297)

SIZE PARAMS: (dataset_len=1477)
min_width=24    min_height=24   widths_mean=24.0        heights_mean=24.0
max_width=24    max_height=24   widths_median=24.0      heights_median=24.0
'''

# ''' ALBUMENTATION
final_transforms = [
    A.Normalize(
        mean=0.5479156433505469,
        std=0.13769661157967297,
        max_pixel_value=255.0,
        always_apply=True
    ),
    ToTensor(always_apply=True),
]


# '''


class CustomDataset(Dataset):
    def __init__(self, imgs: List[str], labels: List[int], augmentation=None) -> None:
        self.imgs = imgs
        self.labels = labels

        if augmentation is None:
            augmentation = []
        augmentation.extend(final_transforms)
        self.augmentation = A.Compose(augmentation)

    def __len__(self) -> int:
        return len(self.imgs)

    def __getitem__(self, item: int) -> Tuple:
        img_path = self.imgs[item]
        img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)  # BGR
        label = torch.Tensor([self.labels[item]]).to(torch.float32)
        # apply augmentations
        img = self.augmentation(image=img)['image']
        return img, label


class Classifier_pl(pl.LightningModule):
    def __init__(
            self,
            model,
            max_epochs=None,
            *args,
            loss_fn=torch.nn.CrossEntropyLoss(),
            start_learning_rate=1e-3,
            checkpoint=None,
            **kwargs):
        super().__init__(*args, **kwargs)
        self.start_learning_rate = start_learning_rate
        self.model = model
        if checkpoint is not None and Path(checkpoint).exists():
            self.model.load_state_dict(torch.load(str(checkpoint)))
        self.loss_fn = loss_fn
        self.save_hyperparameters(ignore=['model', 'loss_fn'])
        self.max_epochs = max_epochs

    def forward(self, img):
        pred = self.model(img)
        return pred

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.start_learning_rate,
            # weight_decay=5e-2,
        )

        warmup_epochs = 5
        verbose = True
        if self.max_epochs:
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.max_epochs - warmup_epochs,
                eta_min=0,
                verbose=verbose)
        else:
            lr_scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=15,
                gamma=0.5,
                verbose=verbose)

        warmup_scheduler = GradualWarmupScheduler(
            optimizer,
            multiplier=1,
            total_epoch=warmup_epochs,
            after_scheduler=lr_scheduler
        )
        return {"optimizer": optimizer, "lr_scheduler": warmup_scheduler, "monitor": 'val_loss'}

    def train_dataloader(self):
        return self.loader['train']

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, stage='train')

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, stage='validation')

    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, stage='test')

    def predict_step(self, batch, batch_idx):  # Inference Dataset only!
        imgs = batch
        preds = self.forward(imgs)
        return preds

    def _shared_step(self, batch, stage):
        imgs, labels = batch
        gts = labels.squeeze(1)
        preds = self.forward(imgs)
        loss = self.loss_fn(preds, gts)
        self.log(f'loss/{stage}', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return {'loss': loss, 'preds': preds}
