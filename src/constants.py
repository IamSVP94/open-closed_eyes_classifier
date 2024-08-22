import torch
import random
from pathlib import Path
import pytorch_lightning as pl

SEED = 2
pl.seed_everything(SEED)  # unified seed
random.seed(SEED)  # unified seed

BASE_DIR = Path(__file__).absolute().parent.parent
AVAIL_GPUS = min(1, torch.cuda.device_count())
NUM_WORKERS = 5  # max 16 for pc in the office
