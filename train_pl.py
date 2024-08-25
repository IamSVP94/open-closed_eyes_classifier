import torch
import numpy as np
import torchmetrics
from pathlib import Path
import albumentations as A
from pytorch_lightning import Trainer
from torch import nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from src import BASE_DIR, AVAIL_GPUS, SEED, NUM_WORKERS, glob_search, CustomDataset, Classifier_pl, MetricSMPCallback, \
    CustomNet

# DATASET
DATASET_DIR = Path('/home/iamsvp/data/eye/EyesDataset/marked/')
imgs = glob_search(DATASET_DIR)
labels = [1 if p.parent.parent.name == 'open' else 0 for p in imgs]  # according to the problem conditions
train_imgs, val_imgs, train_labels, val_labels = train_test_split(imgs, labels, test_size=0.2, random_state=SEED)
print(f'train: {len(train_imgs)}, val: {len(val_imgs)}')

# PARAMS
EXPERIMENT_NAME = f'eyes_classifier'
logdir = BASE_DIR / 'logs/'
logdir.mkdir(parents=True, exist_ok=True)

EPOCHS = 500
start_learning_rate = 1e-3
BATCH_SIZE = 50000 if AVAIL_GPUS else 1
DEVICE = 'cuda' if AVAIL_GPUS else 'cpu'

# AUGMENTATIONS
train_transforms = [
    A.HorizontalFlip(p=0.5),
    A.Sharpen(p=0.2),
    A.CLAHE(p=0.3),
    A.PixelDropout(p=0.2),
    A.RandomBrightnessContrast(p=0.3),
    A.RandomGamma(p=0.3),
    A.Rotate(p=0.5, limit=(-20, 20)),
]

val_transforms = [
    # A.HorizontalFlip(p=0.5),
]

# DATASETS
train = CustomDataset(imgs=train_imgs, labels=train_labels, augmentation=train_transforms)
val = CustomDataset(imgs=val_imgs, labels=val_labels, augmentation=val_transforms)

train_loader = DataLoader(train, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, drop_last=False)
val_loader = DataLoader(val, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, drop_last=False)

''' for checking augmentations
for idx, (imgs_t, labels_t) in enumerate(train_loader):
    if idx > 10:
        exit()
    print(62, idx, imgs_t.shape, labels_t.shape)
    img_n = (((imgs_t[0] * 0.13769661157967297 + 0.5479156433505469)).cpu()).permute(1, 2, 0).numpy()
    label = labels_t[0].cpu().item()
    plt_show_img(img_n, title=label, add_coef=True, mode='cv2')
exit()
# '''

# LOGGER
tb_logger = TensorBoardLogger(save_dir=logdir, name=EXPERIMENT_NAME)
lr_monitor = LearningRateMonitor(logging_interval='epoch')
# METRICS
metrics_callback = MetricSMPCallback(metrics={
    'accuracy': torchmetrics.classification.Accuracy(task="binary").to(device=DEVICE),
    'precision': torchmetrics.classification.Precision(task="binary").to(device=DEVICE),
    'recall': torchmetrics.classification.Recall(task="binary").to(device=DEVICE),
    'F1score': torchmetrics.classification.F1Score(task="binary").to(device=DEVICE),
}, log_img=False, save_img=False, threshold=0.5)
best_loss_saver = ModelCheckpoint(
    mode='min', save_top_k=1, save_last=True, monitor='loss/validation',
    auto_insert_metric_name=False, filename='epoch={epoch:02d}-val_loss={loss/validation:.4f}',
)
best_metric_saver = ModelCheckpoint(
    mode='max', save_top_k=1, save_last=False, monitor='F1score/validation',
    auto_insert_metric_name=False, filename='epoch={epoch:02d}-val_f1score={F1score/validation:.4f}',
)

# MODEL
model = CustomNet(activation=nn.GELU)
# model = EyeClassifier()
model_pl = Classifier_pl(
    model=model,
    # loss_fn=torch.nn.MSELoss(),
    loss_fn=torch.nn.HuberLoss(),
    # loss_fn=torch.nn.L1Loss(),
    start_learning_rate=start_learning_rate,
    max_epochs=EPOCHS
)

# TRAIN
trainer = Trainer(
    max_epochs=EPOCHS, num_sanity_val_steps=0, devices=-1,
    accelerator=DEVICE, logger=tb_logger, log_every_n_steps=1,
    callbacks=[lr_monitor, metrics_callback, best_metric_saver, best_loss_saver],
)

weights = None  # start from checkpoint
trainer.fit(model=model_pl, train_dataloaders=train_loader, val_dataloaders=val_loader, ckpt_path=weights)
