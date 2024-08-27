import shutil
import argparse
from tqdm import tqdm
from typing import Union
from pathlib import Path
from src import glob_search, SEED
from sklearn.model_selection import train_test_split


def main(args):
    shutil_f = shutil.copy if args.mode == 'copy' else shutil.move

    imgs = glob_search(args.src_dir)
    labels = [1 if p.parent.name == 'open' else 0 for p in imgs]

    train, test = train_test_split(imgs, test_size=args.test_size, shuffle=True, random_state=SEED, stratify=labels)
    for dataset, dataset_name in zip([train, test], ['train', 'test']):
        for img_path in tqdm(dataset):
            img_dir = '/'.join(img_path.parts[len(args.src_dir.parts) - len(img_path.parts):-1])  # for saving hierarchy

            new_img_path = args.dst_dir / dataset_name / img_dir / img_path.name
            new_img_path.parent.mkdir(parents=True, exist_ok=True)  # make destination dir
            shutil_f(str(img_path), str(new_img_path))  # new img
    print(f"Saved in dir '{args.dst_dir}'")


description = ""
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-s', '--src_dir', type=str,
                        # required=True,
                        default="/home/iamsvp/data/eye/EyesDataset/unmarked_labeled/epoch=172-val_eerscore=0.0439",
                        help='(required)')
    parser.add_argument('-d', '--dst_dir', type=str, default=None, help='')
    parser.add_argument('-w', '--test_size', type=Union[int, float], default=0.15, help='')
    parser.add_argument('-m', '--mode', choices=['copy', 'move'], default='copy', help='')
    args = parser.parse_args()

    args.src_dir = Path(args.src_dir).resolve()
    if args.dst_dir:
        args.dst_dir = Path(args.dst_dir).resolve()
    else:
        args.dst_dir = args.src_dir.parent / f'{args.src_dir.name}_splitted'

    print(args.src_dir)
    main(args)
