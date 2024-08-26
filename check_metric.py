import torch
from torch.utils.data import DataLoader

from src import glob_search, CustomNet, CustomDataset, NUM_WORKERS, EERMetric, AVAIL_GPUS

DEVICE = 'cuda' if AVAIL_GPUS else 'cpu'

# MODEL
model = CustomNet()

weights_path = "/home/iamsvp/PycharmProjects/open-closed_eyes_classifier/logs/eyes_classifier_full/version_0/checkpoints/epoch=110-val_eerscore=0.0231.ckpt"
state_dict = torch.load(str(weights_path), weights_only=False)['state_dict']
remove_prefix = 'model.'
state_dict = {k[len(remove_prefix):] if k.startswith(remove_prefix) else k: v for k, v in state_dict.items()}
model.load_state_dict(state_dict)
model.to(device=DEVICE)

# DATASET
src_dir = "/home/iamsvp/data/eye/EyesDataset/together_splitted/marked/test/"
imgs = glob_search(src_dir)
gts = [1 if "open" in p.parent.name else 0 for p in imgs]

# PREPARING
metric = EERMetric().to(device=DEVICE)
val = CustomDataset(imgs=imgs, labels=gts)
val_loader = DataLoader(val, batch_size=len(imgs), num_workers=NUM_WORKERS, shuffle=False)

# EVAL
model.eval()
with torch.no_grad():
    for idx, (imgs_t, gts) in enumerate(val_loader):
        preds = model(imgs_t.to(device=DEVICE))
        eer = metric(preds, gts.to(device=DEVICE))
        print(f"{eer=}")
