import cv2
import torch
from pathlib import Path
import albumentations as A
import pytorch_lightning as pl
from torchmetrics import Metric
from torch.utils.data import Dataset
from typing import List, Tuple, Union, Optional
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
        label = self.labels[item]
        img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)  # BGR
        img_t = self.augmentation(image=img)['image']
        label_t = torch.tensor(label).to(torch.int64)
        # apply augmentations
        return img_t, label_t


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
            weight_decay=1e-2,
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

    def _shared_step(self, batch, stage):
        imgs, gts = batch
        preds = self.forward(imgs)
        loss = self.loss_fn(preds, gts)
        self.log(f'loss/{stage}', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return {'loss': loss, 'preds': preds}

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


class EERMetric(Metric):
    # Set to True if the metric is differentiable else set to False
    is_differentiable: Optional[bool] = None

    # Set to True if the metric reaches it optimal value when the metric is maximized.
    # Set to False if it when the metric is minimized.
    higher_is_better: Optional[bool] = False

    # Set to True if the metric during 'update' requires access to the global metric
    # state for its calculations. If not, setting this to False indicates that all
    # batch states are independent and we will optimize the runtime of 'forward'
    full_state_update: bool = True

    def __init__(self, target_class_idx=1) -> None:
        super(EERMetric, self).__init__()
        self.add_state("targets", default=torch.Tensor(), dist_reduce_fx="cat")
        self.add_state("imposters", default=torch.Tensor(), dist_reduce_fx="cat")
        self.target_class_idx = target_class_idx

    @torch.no_grad()
    def update(self, preds: torch.Tensor, gts: torch.Tensor):
        target_index = torch.where(gts == 1)[0]  # open eye
        imposter_index = torch.where(gts == 0)[0]  # closed eye
        target = preds[target_index, self.target_class_idx]
        imposter = preds[imposter_index, self.target_class_idx]
        self.targets = torch.cat((self.targets, target), dim=0)
        self.imposters = torch.cat((self.imposters, imposter), dim=0)

    @torch.no_grad()
    def compute(self):
        if self.imposters.nelement() == 0:
            imposters_min, imposters_max = torch.tensor(float('inf')), torch.tensor(0)
        else:
            imposters_min, imposters_max = torch.min(self.imposters), torch.max(self.imposters)

        if self.targets.nelement() == 0:
            targets_min, targets_max = torch.tensor(float('inf')), torch.tensor(0)
        else:
            targets_min, targets_max = torch.min(self.targets), torch.max(self.targets)
        min_score, max_score = torch.min(imposters_min, targets_min), torch.min(imposters_max, targets_max)
        n_tars, n_imps = len(self.targets), len(self.imposters)
        N = 100

        fars, frrs, dists = torch.zeros((N,)), torch.zeros((N,)), torch.zeros((N,))
        mink = torch.tensor(float('inf'))
        eer = torch.tensor(0)

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
                 threshold: Union[float, None] = None,
                 ) -> None:
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
        # preds = labels
        for m_name, metric in self.metrics.items():
            result = metric(labels, gts)
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
