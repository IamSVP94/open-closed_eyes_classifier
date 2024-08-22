from pathlib import Path
import albumentations as A
from sklearn.model_selection import train_test_split

from src import BASE_DIR, AVAIL_GPUS, glob_search

# PARAMS
EXPERIMENT_NAME = f'eyes_classifier'
logdir = BASE_DIR / 'logs/'
logdir.mkdir(parents=True, exist_ok=True)

input_size = (24, 24)  # h,w
EPOCHS = 200
start_learning_rate = 1e-6
BATCH_SIZE = 290 if AVAIL_GPUS else 1

DATASET_DIR = Path('/home/vid/hdd/datasets/EyesDataset/marked/')

imgs = glob_search(DATASET_DIR)
labels = [0 if p.parent.parent.name == 'open' else 1 for p in imgs]
train_imgs, val_imgs = train_test_split(imgs, stratify=labels, test_size=0.2)

# AUGMENTATIONS
train_transforms = [
    A.HorizontalFlip(p=0.5),
    A.Sharpen(p=0.2),
    A.HueSaturationValue(p=0.3),
    A.CLAHE(p=0.3),
    # A.PixelDropout(p=0.05),
    A.RandomBrightnessContrast(p=0.3),
    A.RandomGamma(p=0.3),
    A.ToGray(p=0.1),
    A.ChannelShuffle(p=0.1),
]
val_transforms = [
    A.HorizontalFlip(p=0.5),
]

common_transforms = [
    A.Resize(height=input_size[0], width=input_size[1], always_apply=True),
    A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), always_apply=True),
]

# DATASETS
train = CustomDataset(csv_file='/home/vid/hdd/datasets/FACES/face_selector_dataset/train.csv',
                      augmentation=train_transforms + common_transforms)
val = CustomDataset(csv_file='/home/vid/hdd/datasets/FACES/face_selector_dataset/test.csv',
                    augmentation=val_transforms + common_transforms)
