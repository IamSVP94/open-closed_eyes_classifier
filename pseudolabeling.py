import shutil
from pathlib import Path

from tqdm import tqdm

from src import glob_search, OpenEyesClassificator

src_dir = "/home/vid/hdd/datasets/EyesDataset/unmarked/"
dst_dir = "/home/vid/hdd/datasets/EyesDataset/unmarked_labeled/"
# weights = "/home/iamsvp/PycharmProjects/open-closed_eyes_classifier/logs/eyes_classifier/version_2/checkpoints/epoch=133-val_f1score=0.9930.ckpt"
weights = "/home/vid/hdd/projects/PycharmProjects/open-closed_eyes_classifier/logs/eyes_classifier/version_88/checkpoints/epoch=131-val_eerscore=0.0271.ckpt"

dst_dir = Path(dst_dir)
model = OpenEyesClassificator(mode="regression", pretrained=weights)
weights = Path(weights)

imgs = glob_search(src_dir, return_pbar=True)
for img_path in imgs:
    result = model.predict(img_path)
    new_path = dst_dir / weights.stem / f"{round(result, 1)}" / img_path.name
    new_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(img_path, new_path)
