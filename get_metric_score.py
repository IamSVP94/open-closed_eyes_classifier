import argparse
from pathlib import Path
from typing import Union

import torch
from torch.utils.data import DataLoader

from src import glob_search, CustomDataset, NUM_WORKERS, EERMetric, AVAIL_GPUS
from src.models import CustomNet2


def main(args):
    print(f" DEVICE = {args.device}")
    # MODEL
    model = CustomNet2()
    state_dict = torch.load(str(args.weights_path), weights_only=False)['state_dict']
    remove_prefix = 'model.'
    state_dict = {k[len(remove_prefix):] if k.startswith(remove_prefix) else k: v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.to(device=args.device)

    # DATASET
    imgs = glob_search(args.src_dir)
    gts = [1 if "open" in p.parent.name else 0 for p in imgs]

    # PREPARING
    metric = EERMetric().to(device=args.device)
    val = CustomDataset(imgs=imgs, labels=gts)
    val_loader = DataLoader(val, batch_size=len(imgs), num_workers=NUM_WORKERS, shuffle=False)

    # EVAL
    model.eval()
    eer_total = 0
    with torch.no_grad():
        for idx, (imgs_t, gts) in enumerate(val_loader):
            preds = model(imgs_t.to(device=args.device))
            eer = metric(preds, gts.to(device=args.device))
            eer_total += eer.item()

    print(f"eer = {eer_total / (idx + 1)}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--src_dir', type=str, help='',
                        required=True,
                        )
    parser.add_argument('-w', '--weights_path', type=str, help='',
                        default='weights_pl/customnet2_relu_final_weights.ckpt'
                        )
    parser.add_argument('-d', '--device', choices=['cpu', 'cuda'], help='',
                        default='cuda',
                        )
    args = parser.parse_args()

    args.src_dir = Path(args.src_dir).resolve()
    assert args.src_dir.exists()

    if args.device == 'cuda' and not torch.cuda.is_available():
        args.device = 'cpu'

    main(args)
