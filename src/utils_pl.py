import io
import cv2
import torch
import random
import numpy as np
import seaborn as sns
from pathlib import Path
import albumentations as A
from typing import List, Tuple, Union, Optional
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import torchmetrics
from torch.utils.data import Dataset
from sklearn.metrics import confusion_matrix
from torchmetrics import Metric
from torchmetrics.classification.base import _ClassificationTaskWrapper
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
    A.Resize(height=24, width=24, always_apply=True),  # constant window size
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
    def __init__(self, imgs: List[str], labels: List[int], mode: str = "regression", augmentation=None) -> None:
        assert mode in ['classification', 'regression']
        self.mode = mode
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
        img = self.augmentation(image=img)['image']
        if self.mode == 'classification':
            label = torch.Tensor([self.labels[item]]).to(torch.int64)
        else:
            label = torch.Tensor([self.labels[item]]).to(torch.float32)
        # apply augmentations
        return img, label


class Classifier_pl(pl.LightningModule):
    def __init__(
            self,
            model,
            max_epochs=None,
            *args,
            loss_fn=torch.nn.CrossEntropyLoss(),
            start_learning_rate=1e-3,
            warmup_epochs=3,
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
        self.warmup_epochs = warmup_epochs

    def forward(self, img):
        pred = self.model(img)
        return pred

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.start_learning_rate,
            # weight_decay=1e-3,
        )

        verbose = True
        if self.max_epochs:
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.max_epochs - self.warmup_epochs,
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
            total_epoch=self.warmup_epochs,
            after_scheduler=lr_scheduler
        )
        return {"optimizer": optimizer, "interval": "epoch", "lr_scheduler": warmup_scheduler, "monitor": 'val_loss'}

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
        imgs, gts = batch
        preds = self.forward(imgs)
        if self.model.mode == 'classification':
            gts = gts.squeeze(dim=1)
        loss = self.loss_fn(preds, gts)
        self.log(f'loss/{stage}', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return {'loss': loss, 'preds': preds}


class EERMetric(Metric):
    # https://torchmetrics.readthedocs.io/en/v0.9.0/pages/implement.html
    # https://habr.com/ru/articles/317798/
    # Set to True if the metric is differentiable else set to False
    is_differentiable: Optional[bool] = None

    # Set to True if the metric reaches it optimal value when the metric is maximized.
    # Set to False if it when the metric is minimized.
    higher_is_better: Optional[bool] = False

    # Set to True if the metric during 'update' requires access to the global metric
    # state for its calculations. If not, setting this to False indicates that all
    # batch states are independent and we will optimize the runtime of 'forward'
    full_state_update: bool = True

    def __init__(self) -> None:
        super(EERMetric, self).__init__()
        self.add_state("targets", default=torch.Tensor(), dist_reduce_fx="cat", persistent=True)
        self.add_state("imposters", default=torch.Tensor(), dist_reduce_fx="cat", persistent=True)

    def update(self, preds: torch.Tensor, gts: torch.Tensor):
        target_index = torch.where(gts == 1)[0]  # open eye
        imposter_index = torch.where(gts == 0)[0]  # closed eye
        target = preds[target_index]
        imposter = preds[imposter_index]
        self.targets = torch.cat((self.targets, target), dim=0)
        self.imposters = torch.cat((self.imposters, imposter), dim=0)

    def compute(self):
        min_score = torch.min(torch.min(self.targets), torch.min(self.imposters))
        max_score = torch.max(torch.max(self.targets), torch.max(self.imposters))

        n_tars, n_imps = len(self.targets), len(self.imposters)
        N = 100

        fars, frrs, dists = torch.zeros((N,)), torch.zeros((N,)), torch.zeros((N,))
        mink = torch.tensor(float('inf'))
        eer = 0

        for i, dist in enumerate(torch.linspace(min_score, max_score, N)):
            far = torch.sum(self.imposters > dist) / n_imps
            frr = torch.sum(self.targets <= dist) / n_tars
            fars[i] = far
            frrs[i] = frr
            dists[i] = dist

            k = torch.abs(far - frr)
            if k < mink:
                mink = k
                eer = (far + frr) / 2
        return eer


class MetricSMPCallback(pl.Callback):
    def __init__(self, metrics, activation=None,
                 threshold: Union[float, None] = 0.5,
                 mode: str = 'regression',
                 ) -> None:
        assert mode in ['classification', 'regression']
        self.mode = mode
        self.metrics = metrics
        self.threshold = threshold
        if activation is not None:
            self.activation = activation
        else:
            self.activation = torch.nn.Identity()

    @torch.no_grad()
    def _get_metrics(self, preds, gts, trainer):
        metric_results = {k: dict() for k in self.metrics}
        labels = self.activation(preds)
        if self.mode == 'classification':
            # preds = torch.argmax(labels, dim=1).to(torch.float32).unsqueeze(1)
            preds = labels
        elif self.mode == 'regression':
            # preds = (labels >= self.threshold).to(torch.int8)
            preds = labels
        # print(166, preds, gts)
        # print(167, preds.shape, gts.shape)
        for m_name, metric in self.metrics.items():
            result = metric(preds, gts)
            metric_results[m_name] = round(result.item(), 4)
            # print(170, m_name, metric_results[m_name])
        return metric_results

    @torch.no_grad()
    def _on_shared_batch_end(self, trainer, outputs, batch, batch_idx, stage) -> None:
        imgs, gts = batch
        preds = self.activation(outputs['preds'])
        metrics = self._get_metrics(preds=preds, gts=gts, trainer=trainer)
        for m_name, m_val in metrics.items():
            m_title = f'{m_name}/{stage}'
            trainer.model.log(m_title, m_val, on_step=False, on_epoch=True)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        self._on_shared_batch_end(trainer, outputs, batch, batch_idx, stage='train')

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        self._on_shared_batch_end(trainer, outputs, batch, batch_idx, stage='validation')

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        self._on_shared_batch_end(trainer, outputs, batch, batch_idx, stage='test')
