import shutil
from pathlib import Path

from tqdm import tqdm

from src import glob_search, OpenEyesClassificator
from src.models import CustomNet2

src_dir = "/home/iamsvp/data/eye/EyesDataset/unmarked/"
dst_dir = "/home/iamsvp/data/eye/EyesDataset/unmarked_labeled/"
weights = "/home/iamsvp/PycharmProjects/open-closed_eyes_classifier/logs/CustomNet2_RELU_1e-3_weight_decay=1e-2/version_0/checkpoints/epoch=70-val_eerscore=0.0459.ckpt"

dst_dir = Path(dst_dir)
model = OpenEyesClassificator(model=CustomNet2(), pretrained=weights)
weights = Path(weights)

imgs = glob_search(src_dir, return_pbar=True)
for img_path in imgs:
    result = model.predict(img_path)
    new_path = dst_dir / weights.stem / f"{round(result, 1)}" / img_path.name
    new_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(img_path, new_path)
