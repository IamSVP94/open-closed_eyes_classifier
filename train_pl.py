import numpy as np
from pathlib import Path
import albumentations as A
import torch
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from torchvision.models import convnext_tiny

from src import BASE_DIR, AVAIL_GPUS, SEED, NUM_WORKERS, glob_search, CustomDataset, plt_show_img, Classifier_pl, \
    EyeClassifier

# PARAMS
EXPERIMENT_NAME = f'eyes_classifier'
logdir = BASE_DIR / 'logs/'
logdir.mkdir(parents=True, exist_ok=True)

input_size = (24, 24)  # h,w
EPOCHS = 300
start_learning_rate = 1e-3
BATCH_SIZE = 5000 if AVAIL_GPUS else 1
DEVICE = 'cuda' if AVAIL_GPUS else 'cpu'

DATASET_DIR = Path('/home/vid/hdd/datasets/EyesDataset/marked/')

imgs = glob_search(DATASET_DIR)
labels = [0 if p.parent.parent.name == 'open' else 1 for p in imgs]
train_imgs, val_imgs, train_labels, val_labels = train_test_split(imgs, labels, test_size=0.2, random_state=SEED)

# AUGMENTATIONS
train_transforms = [
    A.HorizontalFlip(p=0.5),
    # A.Sharpen(p=0.2),
    # A.HueSaturationValue(p=0.3),
    # A.CLAHE(p=0.3),
    # A.PixelDropout(p=0.05),
    # A.RandomBrightnessContrast(p=0.3),
    # A.RandomGamma(p=0.3),
    # A.ToGray(p=0.1),
    # A.ChannelShuffle(p=0.1),
]

val_transforms = [
    A.HorizontalFlip(p=0.5),
]

common_transforms = [
    A.Resize(height=input_size[0], width=input_size[1], always_apply=True),
]

# DATASETS
train = CustomDataset(imgs=train_imgs, labels=train_labels, augmentation=train_transforms + common_transforms)
val = CustomDataset(imgs=val_imgs, labels=val_labels, augmentation=val_transforms + common_transforms)

train_loader = DataLoader(train, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, drop_last=False)
val_loader = DataLoader(val, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, drop_last=False)

''' for checking augmentations
for idx, (imgs_t, labels_t) in enumerate(train_loader):
    print(62, imgs_t.shape)
    if idx > 3:
        exit()
    img_n = (((imgs_t[0] * 0.13769661157967297 + 0.5479156433505469)).cpu()).permute(1, 2, 0).numpy()
    label = labels_t[0].cpu().item()
    print(58, np.mean(img_n), np.std(img_n))
    plt_show_img(img_n, title=label, add_coef=True, mode='plt')
# '''

# LOGGER
tb_logger = TensorBoardLogger(save_dir=logdir, name=EXPERIMENT_NAME)
lr_monitor = LearningRateMonitor(logging_interval='epoch')
best_metric_saver = ModelCheckpoint(
    mode='min', save_top_k=1, save_last=True, monitor='loss/validation',
    auto_insert_metric_name=False, filename='epoch={epoch:02d}-val_loss={loss/validation:.4f}',
)

# MODEL
model = EyeClassifier()
model_pl = Classifier_pl(
    model=model,
    loss_fn=torch.nn.MSELoss(),
    start_learning_rate=start_learning_rate,
    max_epochs=EPOCHS
)

# TRAIN
trainer = Trainer(
    max_epochs=EPOCHS, num_sanity_val_steps=0, devices=-1,
    accelerator=DEVICE, logger=tb_logger, log_every_n_steps=1,
    callbacks=[lr_monitor, best_metric_saver],
)

weights = None  # start from checkpoint
trainer.fit(model=model_pl, train_dataloaders=train_loader, val_dataloaders=val_loader, ckpt_path=weights)
