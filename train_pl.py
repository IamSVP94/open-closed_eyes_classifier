import torch
from torch import nn
from pathlib import Path
import albumentations as A
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from src.models import CustomNet, CustomNet2, CustomNet0
from src.utils_pl import EERMetric
from src import BASE_DIR, AVAIL_GPUS, NUM_WORKERS, glob_search, CustomDataset, Classifier_pl, MetricSMPCallback

# PARAMS
EXPERIMENT_NAME = f'FULL_CustomNet2_RELU_1e-3_weight_decay=1e-2_AdamW'
EPOCHS = 200
start_learning_rate = 1e-3
model = CustomNet2(activation=nn.GELU)
# TODO: try weight decay

# DATASET
DATASET_DIR = Path('/home/iamsvp/data/eye/EyesDataset/together_splitted/')
imgs = glob_search(DATASET_DIR)

train_imgs, train_labels = [], []
val_imgs, val_labels = [], []
for i_path in imgs:
    label = 1 if "open" in i_path.parent.name else 0
    if "train" in i_path.parts:
        train_imgs.append(i_path)
        train_labels.append(label)
    else:
        val_imgs.append(i_path)
        val_labels.append(label)

# AUGMENTATIONS
train_transforms = [
    A.PixelDropout(dropout_prob=0.1, drop_value=None, p=1),
    A.Sharpen(p=0.2),
    A.CLAHE(p=0.3),
    A.RandomBrightnessContrast(p=0.5),
    A.RandomGamma(p=0.5),
    A.Blur(p=0.5),
    A.OneOf([
        A.ElasticTransform(p=0.3),
        A.GridDistortion(p=0.7),
    ], p=0.5),
    A.HorizontalFlip(p=0.5),
    A.Rotate(p=0.9, limit=(-45, 45)),
]

val_transforms = [
    # A.HorizontalFlip(p=0.5),
]

# DATASETS
BATCH_SIZE = int(len(train_imgs) / 2) + 1 if AVAIL_GPUS else 1
DEVICE = 'cuda' if AVAIL_GPUS else 'cpu'

train = CustomDataset(imgs=train_imgs, labels=train_labels, augmentation=train_transforms)
val = CustomDataset(imgs=val_imgs, labels=val_labels, augmentation=val_transforms)

train_loader = DataLoader(train, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=True)
val_loader = DataLoader(val, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=False)

''' for checking augmentations
train_double = CustomDataset(imgs=train_imgs, labels=train_labels, mode=MODE, augmentation=None)
train_double_loader = DataLoader(train_double, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=False)
for idx, ((imgs_t, labels_t), (imgs_t_d, labels_t_d)) in enumerate(zip(train_loader, train_double_loader)):
    if idx > 10:
        exit()
    assert labels_t == labels_t_d
    img = (((imgs_t[0] * 0.13769661157967297 + 0.5479156433505469)).cpu()).permute(1, 2, 0).numpy()
    img_d = (((imgs_t_d[0] * 0.13769661157967297 + 0.5479156433505469)).cpu()).permute(1, 2, 0).numpy()
    concat_img = np.concatenate((img_d, img), axis=1)
    plt_show_img(concat_img, title=labels_t[0].cpu().item(), add_coef=True, mode='cv2')
exit()
# '''

# LOGGER CALLBACK
logdir = BASE_DIR / 'logs/'
logdir.mkdir(parents=True, exist_ok=True)

tb_logger = TensorBoardLogger(save_dir=logdir, name=EXPERIMENT_NAME)
lr_monitor = LearningRateMonitor(logging_interval='epoch')
# METRICS CALLBACK
metrics_callback = MetricSMPCallback(metrics={
    'EERscore': EERMetric().to(device=DEVICE),
})
# WEIGHTS SAVER CALLBACK
best_loss_saver = ModelCheckpoint(
    mode='min', save_top_k=1, save_last=True, monitor='loss/validation',
    auto_insert_metric_name=False, filename='epoch={epoch:02d}-val_loss={loss/validation:.4f}',
)
best_metric_saver = ModelCheckpoint(
    mode='min', save_top_k=1, save_last=False, monitor='EERscore/validation',
    auto_insert_metric_name=False, filename='epoch={epoch:02d}-val_eerscore={EERscore/validation:.4f}',
)

# MODEL
model_pl = Classifier_pl(
    model=model,
    loss_fn=torch.nn.CrossEntropyLoss(),
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
